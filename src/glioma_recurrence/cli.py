"""Command-line interface for the glioma recurrence-risk pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .case import assert_case_geometry, load_case
from .constants import (
    BASELINE_FLAIR,
    BASELINE_T1C,
    BRAIN_MASK,
    DOSE_ON_BASELINE,
    RECURRENCE_RISK,
    case_dir,
)
from .dicom import (
    deidentify_metadata_summary,
    find_instances,
    mr_series_to_volume,
    read_dataset,
    rtdose_to_volume,
    write_ingest_audit,
)
from .evaluation import evaluate_case, summarize_metrics, write_evaluation_report
from .geometry import Volume
from .labels import map_reviewed_mask_to_baseline
from .models import DoseDistanceBandModel, ModelError, VoxelLogisticModel, load_model, save_model
from .nifti import read_volume, write_volume
from .preprocess import brain_mask_from_modalities, resample_flair_and_dose_to_t1c, robust_normalize_mri
from .reports import write_case_qc_report
from .schema import PatientRecord, filter_records, read_manifest


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glioma-risk")
    subparsers = parser.add_subparsers(required=True)

    ingest = subparsers.add_parser("ingest", help="Validate DICOM, convert MRI and RTDOSE to derived NIfTI")
    add_manifest_args(ingest)
    ingest.add_argument("--dicom-root", required=True)
    ingest.set_defaults(func=cmd_ingest)

    preprocess = subparsers.add_parser("preprocess", help="Resample to baseline T1c, normalize MRI, and create brain masks")
    add_manifest_args(preprocess)
    preprocess.add_argument("--prescription-dose-gy", type=float, default=None)
    preprocess.set_defaults(func=cmd_preprocess)

    labels = subparsers.add_parser("make-labels", help="Map human-reviewed recurrence masks to baseline space")
    add_manifest_args(labels)
    labels.add_argument("--assume-baseline-space", action="store_true")
    labels.add_argument("--skip-missing", action="store_true")
    labels.set_defaults(func=cmd_make_labels)

    train = subparsers.add_parser("train", help="Train recurrence-risk model")
    add_manifest_args(train)
    train.add_argument("--model", choices=["dose-distance", "voxel-logistic", "unet"], default="dose-distance")
    train.add_argument("--output", required=True)
    train.add_argument("--prescription-dose-gy", type=float, default=None)
    train.add_argument("--max-voxels-per-case", type=int, default=20000)
    train.add_argument("--epochs", type=int, default=20, help="MONAI U-Net epochs when --model unet is selected")
    train.add_argument("--patch-size", default="96,96,96", help="MONAI U-Net patch size as x,y,z")
    train.add_argument("--include-pseudoprogression", action="store_true")
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate predictions and calibration")
    add_manifest_args(evaluate)
    evaluate.add_argument("--model-path", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--splits", default="validation,test")
    evaluate.add_argument("--write-predictions", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)

    predict = subparsers.add_parser("predict", help="Predict recurrence-risk heatmap for one derived case")
    predict.add_argument("--case-dir", required=True)
    predict.add_argument("--model-path", required=True)
    predict.add_argument("--output-dir", required=True)
    predict.set_defaults(func=cmd_predict)

    return parser


def add_manifest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--derived-root", required=True)


def cmd_ingest(args: argparse.Namespace) -> int:
    records = read_manifest(args.manifest)
    for record in records:
        output_dir = case_dir(args.derived_root, record.patient_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        t1c_paths = find_instances(args.dicom_root, series_uid=record.baseline_t1c_series_uid)
        flair_paths = find_instances(args.dicom_root, series_uid=record.baseline_flair_series_uid)
        rtdose_paths = find_instances(args.dicom_root, sop_instance_uid=record.rtdose_sop_instance_uid)
        if len(rtdose_paths) != 1:
            raise RuntimeError(f"{record.patient_id}: expected one RTDOSE instance, found {len(rtdose_paths)}")

        t1c = mr_series_to_volume(t1c_paths)
        flair = mr_series_to_volume(flair_paths)
        dose = rtdose_to_volume(rtdose_paths[0])
        write_volume(t1c, output_dir / BASELINE_T1C, dtype=np.float32)
        write_volume(flair, output_dir / BASELINE_FLAIR, dtype=np.float32)
        write_volume(dose, output_dir / DOSE_ON_BASELINE, dtype=np.float32)
        summaries = [
            deidentify_metadata_summary(read_dataset(t1c_paths[0]), patient_id=record.patient_id),
            deidentify_metadata_summary(read_dataset(flair_paths[0]), patient_id=record.patient_id),
            deidentify_metadata_summary(read_dataset(rtdose_paths[0]), patient_id=record.patient_id),
        ]
        write_ingest_audit(output_dir / "ingest_audit.json", summaries)
        print(f"ingested {record.patient_id}")
    return 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    records = read_manifest(args.manifest)
    for record in records:
        output_dir = case_dir(args.derived_root, record.patient_id)
        t1c = read_volume(output_dir / BASELINE_T1C)
        flair = read_volume(output_dir / BASELINE_FLAIR)
        dose = read_volume(output_dir / DOSE_ON_BASELINE)
        flair_on_t1c, dose_on_t1c = resample_flair_and_dose_to_t1c(t1c, flair, dose)
        brain_mask = brain_mask_from_modalities(t1c.data, flair_on_t1c.data)
        t1c_norm = Volume(robust_normalize_mri(t1c.data, brain_mask), t1c.affine, t1c.metadata)
        flair_norm = Volume(robust_normalize_mri(flair_on_t1c.data, brain_mask), t1c.affine, flair_on_t1c.metadata)
        write_volume(t1c_norm, output_dir / BASELINE_T1C, dtype=np.float32)
        write_volume(flair_norm, output_dir / BASELINE_FLAIR, dtype=np.float32)
        write_volume(dose_on_t1c, output_dir / DOSE_ON_BASELINE, dtype=np.float32)
        write_volume(Volume(brain_mask, t1c.affine), output_dir / BRAIN_MASK, dtype=np.uint8)
        case = load_case(output_dir, prescription_dose_gy=resolve_prescription(record, args.prescription_dose_gy))
        write_case_qc_report(case, output_dir=output_dir)
        print(f"preprocessed {record.patient_id}")
    return 0


def cmd_make_labels(args: argparse.Namespace) -> int:
    records = read_manifest(args.manifest)
    for record in records:
        if not record.reviewed_recurrence_mask_path and args.skip_missing:
            print(f"skipping {record.patient_id}: no reviewed mask path")
            continue
        output = map_reviewed_mask_to_baseline(
            record,
            derived_root=args.derived_root,
            assume_baseline_space=args.assume_baseline_space,
        )
        case = load_case(case_dir(args.derived_root, record.patient_id), require_label=True)
        write_case_qc_report(case, output_dir=output.parent)
        print(f"mapped label {record.patient_id}: {output}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    records = filter_records(
        read_manifest(args.manifest),
        splits={"train"},
        include_pseudoprogression=args.include_pseudoprogression,
    )
    if not records:
        raise RuntimeError("no training records selected")
    prescription = resolve_dataset_prescription(records, args.prescription_dose_gy)
    cases = [
        _load_training_case(record, args.derived_root, prescription)
        for record in records
    ]
    if args.model == "dose-distance":
        model = DoseDistanceBandModel.fit(
            cases,
            prescription_dose_gy=prescription,
            max_voxels_per_case=args.max_voxels_per_case,
        )
    else:
        if args.model == "voxel-logistic":
            model = VoxelLogisticModel.fit(
                cases,
                prescription_dose_gy=prescription,
                max_voxels_per_case=args.max_voxels_per_case,
            )
        else:
            from .deep import DeepTrainingConfig, train_unet

            train_unet(
                cases,
                output_path=args.output,
                prescription_dose_gy=prescription,
                config=DeepTrainingConfig(max_epochs=args.epochs, patch_size=parse_patch_size(args.patch_size)),
            )
            print(f"trained {args.model}: {args.output}")
            return 0
    save_model(model, args.output)
    metadata = {
        "model": args.model,
        "n_training_cases": len(cases),
        "excluded_pseudoprogression_cases": [
            record.patient_id for record in read_manifest(args.manifest) if record.should_exclude_from_training
        ],
    }
    Path(args.output).with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"trained {args.model}: {args.output}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    model = try_load_simple_model(args.model_path)
    splits = {split.strip() for split in args.splits.split(",") if split.strip()}
    records = filter_records(read_manifest(args.manifest), splits=splits)
    metrics = []
    calibration_labels: list[np.ndarray] = []
    calibration_scores: list[np.ndarray] = []
    for record in records:
        case = load_case(case_dir(args.derived_root, record.patient_id), require_label=True)
        assert_case_geometry(case)
        risk = Volume(predict_with_loaded_model(model, case, args.model_path), case.t1c.affine)
        if args.write_predictions:
            write_volume(risk, Path(args.derived_root) / record.patient_id / RECURRENCE_RISK, dtype=np.float32)
        write_case_qc_report(case, output_dir=Path(args.derived_root) / record.patient_id, risk=risk)
        metrics.append(evaluate_case(case, risk))
        mask = case.brain_mask.data.astype(bool).reshape(-1)
        calibration_labels.append(case.recurrence_mask.data.astype(np.float32).reshape(-1)[mask])
        calibration_scores.append(risk.data.reshape(-1)[mask])
        print(f"evaluated {record.patient_id}")
    summary = summarize_metrics(metrics)
    if calibration_labels:
        from .evaluation import calibration_bins

        summary["calibration"] = calibration_bins(
            np.concatenate(calibration_labels),
            np.concatenate(calibration_scores),
        )
    write_evaluation_report(summary, args.output)
    print(f"wrote evaluation report: {args.output}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    model = try_load_simple_model(args.model_path)
    case = load_case(args.case_dir, require_label=False)
    assert_case_geometry(case)
    risk = Volume(predict_with_loaded_model(model, case, args.model_path), case.t1c.affine)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_volume(risk, output_dir / RECURRENCE_RISK, dtype=np.float32)
    write_case_qc_report(case, output_dir=output_dir, risk=risk)
    print(f"wrote prediction: {output_dir / RECURRENCE_RISK}")
    return 0


def _load_training_case(record: PatientRecord, derived_root: str | Path, prescription: float):
    case = load_case(case_dir(derived_root, record.patient_id), require_label=True, prescription_dose_gy=prescription)
    assert_case_geometry(case)
    return case


def resolve_prescription(record: PatientRecord, override: float | None) -> float:
    if override is not None:
        return override
    if record.prescription_dose_gy is not None:
        return record.prescription_dose_gy
    return 60.0


def resolve_dataset_prescription(records: list[PatientRecord], override: float | None) -> float:
    if override is not None:
        if override <= 0:
            raise ValueError("--prescription-dose-gy must be positive")
        return float(override)
    values = {record.prescription_dose_gy for record in records if record.prescription_dose_gy is not None}
    if len(values) == 1:
        value = float(next(iter(values)))
        if value <= 0:
            raise ValueError("prescription_dose_gy must be positive")
        return value
    if len(values) > 1:
        raise ValueError(
            "multiple prescription_dose_gy values found; pass --prescription-dose-gy for a single V1 model"
        )
    return 60.0


def parse_patch_size(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise ValueError("--patch-size must be three positive integers, e.g. 96,96,96")
    return tuple(parts)


def try_load_simple_model(path: str | Path):
    if str(path).endswith((".pt", ".pth")):
        return None
    try:
        return load_model(path)
    except (ModelError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def predict_with_loaded_model(model, case, model_path: str | Path) -> np.ndarray:
    if model is not None:
        return model.predict_case(case)
    from .deep import predict_unet

    return predict_unet(case, checkpoint_path=model_path)


if __name__ == "__main__":
    raise SystemExit(main())
