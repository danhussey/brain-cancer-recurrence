"""Command-line interface for the MRI-only glioma recurrence-risk pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .case import assert_case_geometry, load_case
from .constants import BASELINE_FLAIR, BASELINE_T1C, BASELINE_TUMOR_MASK, BRAIN_MASK, RECURRENCE_RISK, case_dir
from .evaluation import evaluate_case, summarize_metrics, write_evaluation_report
from .geometry import Volume
from .labels import map_reviewed_mask_to_baseline
from .models import ModelError, TumorDistanceBandModel, VoxelLogisticMRIModel, load_model, save_model
from .nifti import read_volume, write_volume
from .observability import add_observability_args, build_observer
from .preprocess import brain_mask_from_modalities, resample_flair_and_tumor_to_t1c, robust_normalize_mri
from .reports import write_case_qc_report
from .schema import PatientRecord, filter_records, read_manifest


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    observer = build_observer(args)
    args.observer = observer
    observer.start()
    try:
        exit_code = int(args.func(args))
        observer.finish(status="completed" if exit_code == 0 else "failed", exit_code=exit_code)
        return exit_code
    except Exception as exc:
        observer.finish(status="failed", error=str(exc), exit_code=2)
        print(f"error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glioma-risk")
    subparsers = parser.add_subparsers(required=True)

    dicom_audit = subparsers.add_parser("dicom-audit", help="Read DICOM headers and summarize MRI sequence availability")
    dicom_audit.add_argument("--dicom-root", required=True, help="Root directory containing clinical DICOM exports")
    dicom_audit.add_argument("--output", required=True, help="Series-level CSV inventory path")
    dicom_audit.add_argument("--summary-output", required=True, help="Strict JSON summary path")
    dicom_audit.add_argument(
        "--include-patient-id",
        action="store_true",
        help="Write raw PatientID values to the CSV/JSON instead of stable hashed patient keys",
    )
    dicom_audit.add_argument(
        "--include-paths",
        action="store_true",
        help="Write relative example file paths to the CSV. Off by default because folder names may contain identifiers.",
    )
    dicom_audit.add_argument(
        "--patient-id-salt",
        default="glioma-recurrence-risk",
        help="Salt used when hashing PatientID values for local pseudonymous audit keys",
    )
    add_observability_args(dicom_audit)
    dicom_audit.set_defaults(func=cmd_dicom_audit, stage="dicom-audit")

    preprocess = subparsers.add_parser("preprocess", help="Resample baseline MRI/mask to T1c and normalize MRI")
    add_manifest_args(preprocess)
    add_observability_args(preprocess)
    preprocess.set_defaults(func=cmd_preprocess, stage="preprocess")

    labels = subparsers.add_parser("make-labels", help="Map human-reviewed recurrence masks to baseline space")
    add_manifest_args(labels)
    labels.add_argument("--assume-baseline-space", action="store_true")
    labels.add_argument(
        "--registration-mode",
        choices=["simpleitk", "affine", "assume-aligned"],
        default="simpleitk",
        help="How to map reviewed follow-up masks to baseline space",
    )
    labels.add_argument("--skip-missing", action="store_true")
    add_observability_args(labels)
    labels.set_defaults(func=cmd_make_labels, stage="make-labels")

    train = subparsers.add_parser("train", help="Train MRI-only recurrence-risk model")
    add_manifest_args(train)
    train.add_argument("--model", choices=["tumor-distance", "voxel-logistic-mri", "unet"], default="tumor-distance")
    train.add_argument("--output", required=True)
    train.add_argument("--max-voxels-per-case", type=int, default=20000)
    train.add_argument("--epochs", type=int, default=20, help="MONAI U-Net epochs when --model unet is selected")
    train.add_argument("--patch-size", default="96,96,96", help="MONAI U-Net patch size as x,y,z")
    train.add_argument("--include-pseudoprogression", action="store_true")
    add_observability_args(train)
    train.set_defaults(func=cmd_train, stage="train")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate predictions and calibration")
    add_manifest_args(evaluate)
    evaluate.add_argument("--model-path", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--splits", default="validation,test")
    evaluate.add_argument("--write-predictions", action="store_true")
    add_observability_args(evaluate)
    evaluate.set_defaults(func=cmd_evaluate, stage="evaluate")

    predict = subparsers.add_parser("predict", help="Predict recurrence-risk heatmap for one derived case")
    predict.add_argument("--case-dir", required=True)
    predict.add_argument("--model-path", required=True)
    predict.add_argument("--output-dir", required=True)
    add_observability_args(predict)
    predict.set_defaults(func=cmd_predict, stage="predict")

    return parser


def add_manifest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--derived-root", required=True)


def cmd_dicom_audit(args: argparse.Namespace) -> int:
    from .dicom import audit_dicom_tree

    summary = audit_dicom_tree(
        args.dicom_root,
        output_csv=args.output,
        summary_json=args.summary_output,
        include_patient_id=args.include_patient_id,
        include_paths=args.include_paths,
        patient_id_salt=args.patient_id_salt,
    )
    args.observer.event("dicom_audit_summary", **summary)
    args.observer.artifact(args.output, kind="dicom_series_inventory")
    args.observer.artifact(args.summary_output, kind="dicom_audit_summary")
    print(f"wrote DICOM series inventory: {args.output}")
    print(f"wrote DICOM audit summary: {args.summary_output}")
    return 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    records = read_manifest(args.manifest)
    args.observer.event("manifest_loaded", manifest=args.manifest, n_records=len(records))
    for record in records:
        with args.observer.case(record.patient_id, "preprocess"):
            output_dir = case_dir(args.derived_root, record.patient_id)
            t1c = read_volume(output_dir / BASELINE_T1C)
            flair = read_volume(output_dir / BASELINE_FLAIR)
            baseline_tumor = read_volume(output_dir / BASELINE_TUMOR_MASK)
            args.observer.event(
                "case_inputs_loaded",
                patient_id=record.patient_id,
                shape=t1c.shape,
                spacing=t1c.spacing,
            )
            flair_on_t1c, tumor_on_t1c = resample_flair_and_tumor_to_t1c(t1c, flair, baseline_tumor)
            brain_mask = brain_mask_from_modalities(t1c.data, flair_on_t1c.data)
            t1c_norm = Volume(robust_normalize_mri(t1c.data, brain_mask), t1c.affine, t1c.metadata)
            flair_norm = Volume(robust_normalize_mri(flair_on_t1c.data, brain_mask), t1c.affine, flair_on_t1c.metadata)
            write_volume(t1c_norm, output_dir / BASELINE_T1C, dtype=np.float32)
            write_volume(flair_norm, output_dir / BASELINE_FLAIR, dtype=np.float32)
            write_volume(tumor_on_t1c, output_dir / BASELINE_TUMOR_MASK, dtype=np.uint8)
            write_volume(Volume(brain_mask, t1c.affine), output_dir / BRAIN_MASK, dtype=np.uint8)
            case = load_case(output_dir)
            qc_path = write_case_qc_report(case, output_dir=output_dir)
            args.observer.artifact(qc_path, kind="qc_report", patient_id=record.patient_id)
            args.observer.event(
                "case_preprocess_metrics",
                patient_id=record.patient_id,
                brain_voxels=int(np.count_nonzero(brain_mask)),
                baseline_tumor_voxels=int(np.count_nonzero(tumor_on_t1c.data)),
            )
        print(f"preprocessed {record.patient_id}")
    return 0


def cmd_make_labels(args: argparse.Namespace) -> int:
    records = read_manifest(args.manifest)
    args.observer.event("manifest_loaded", manifest=args.manifest, n_records=len(records))
    for record in records:
        if not record.reviewed_recurrence_mask_path and args.skip_missing:
            args.observer.event("case_skipped", patient_id=record.patient_id, reason="missing_reviewed_mask_path")
            print(f"skipping {record.patient_id}: no reviewed mask path")
            continue
        with args.observer.case(record.patient_id, "make-labels"):
            output = map_reviewed_mask_to_baseline(
                record,
                derived_root=args.derived_root,
                assume_baseline_space=args.assume_baseline_space,
                registration_mode=args.registration_mode,
            )
            args.observer.artifact(output, kind="recurrence_mask_on_baseline", patient_id=record.patient_id)
            case = load_case(case_dir(args.derived_root, record.patient_id), require_label=True)
            qc_path = write_case_qc_report(case, output_dir=output.parent)
            args.observer.artifact(qc_path, kind="qc_report", patient_id=record.patient_id)
            args.observer.event(
                "case_label_metrics",
                patient_id=record.patient_id,
                recurrence_voxels=int(np.count_nonzero(case.recurrence_mask.data)),
                registration_mode="assume-aligned" if args.assume_baseline_space else args.registration_mode,
            )
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
    args.observer.event("training_records_selected", n_records=len(records), model=args.model)
    cases = [_load_training_case(record, args.derived_root) for record in records]
    if args.model == "tumor-distance":
        model = TumorDistanceBandModel.fit(cases, max_voxels_per_case=args.max_voxels_per_case)
    elif args.model == "voxel-logistic-mri":
        model = VoxelLogisticMRIModel.fit(cases, max_voxels_per_case=args.max_voxels_per_case)
    else:
        from .deep import DeepTrainingConfig, train_unet

        train_unet(
            cases,
            output_path=args.output,
            config=DeepTrainingConfig(max_epochs=args.epochs, patch_size=parse_patch_size(args.patch_size)),
        )
        args.observer.artifact(args.output, kind="model_checkpoint")
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
    metadata_path = Path(args.output).with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    args.observer.artifact(args.output, kind="model")
    args.observer.artifact(metadata_path, kind="model_metadata")
    args.observer.event(
        "training_completed",
        model=args.model,
        n_training_cases=len(cases),
        excluded_pseudoprogression_cases=metadata["excluded_pseudoprogression_cases"],
    )
    print(f"trained {args.model}: {args.output}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    model = try_load_model(args.model_path)
    splits = {split.strip() for split in args.splits.split(",") if split.strip()}
    all_records = read_manifest(args.manifest)
    records = filter_records(all_records, splits=splits)
    args.observer.event(
        "evaluation_records_selected",
        manifest=args.manifest,
        n_records=len(records),
        splits=sorted(splits),
        model_path=args.model_path,
        model_kind=getattr(model, "kind", "unet"),
    )
    metrics = []
    calibration_labels: list[np.ndarray] = []
    calibration_scores: list[np.ndarray] = []
    for record in records:
        with args.observer.case(record.patient_id, "evaluate"):
            case = load_case(case_dir(args.derived_root, record.patient_id), require_label=True)
            assert_case_geometry(case)
            risk = Volume(predict_with_loaded_model(model, case, args.model_path), case.t1c.affine)
            if args.write_predictions:
                prediction_path = Path(args.derived_root) / record.patient_id / RECURRENCE_RISK
                write_volume(risk, prediction_path, dtype=np.float32)
                args.observer.artifact(prediction_path, kind="prediction", patient_id=record.patient_id)
            qc_path = write_case_qc_report(case, output_dir=Path(args.derived_root) / record.patient_id, risk=risk)
            args.observer.artifact(qc_path, kind="qc_report", patient_id=record.patient_id)
            case_metrics = evaluate_case(case, risk)
            metrics.append(case_metrics)
            args.observer.event("case_evaluation_metrics", **case_metrics.__dict__)
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
    baseline_comparison = tumor_distance_baseline_comparison(
        model=model,
        all_records=all_records,
        evaluation_records=records,
        derived_root=args.derived_root,
        max_voxels_per_case=20000,
    )
    if baseline_comparison is not None:
        baseline_summary = baseline_comparison["tumor_distance"]
        summary["mean_voxel_auprc_delta_vs_tumor_distance"] = numeric_delta(
            summary.get("mean_voxel_auprc"),
            baseline_summary.get("mean_voxel_auprc") if isinstance(baseline_summary, dict) else None,
        )
        summary["mean_brier_score_delta_vs_tumor_distance"] = numeric_delta(
            summary.get("mean_brier_score"),
            baseline_summary.get("mean_brier_score") if isinstance(baseline_summary, dict) else None,
        )
        summary["baseline_comparison"] = baseline_comparison
    args.observer.event("evaluation_summary", **summary_without_cases(summary))
    write_evaluation_report(summary, args.output)
    args.observer.artifact(args.output, kind="evaluation_report")
    print(f"wrote evaluation report: {args.output}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    model = try_load_model(args.model_path)
    case_id = Path(args.case_dir).name
    with args.observer.case(case_id, "predict"):
        case = load_case(args.case_dir, require_label=False)
        assert_case_geometry(case)
        risk = Volume(predict_with_loaded_model(model, case, args.model_path), case.t1c.affine)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = output_dir / RECURRENCE_RISK
        write_volume(risk, prediction_path, dtype=np.float32)
        qc_path = write_case_qc_report(case, output_dir=output_dir, risk=risk)
        args.observer.artifact(prediction_path, kind="prediction", patient_id=case.patient_id)
        args.observer.artifact(qc_path, kind="qc_report", patient_id=case.patient_id)
        args.observer.event(
            "prediction_summary",
            patient_id=case.patient_id,
            model_path=args.model_path,
            risk_min=float(np.nanmin(risk.data)),
            risk_max=float(np.nanmax(risk.data)),
            risk_mean=float(np.nanmean(risk.data)),
        )
    print(f"wrote prediction: {prediction_path}")
    return 0


def _load_training_case(record: PatientRecord, derived_root: str | Path):
    case = load_case(case_dir(derived_root, record.patient_id), require_label=True)
    assert_case_geometry(case)
    return case


def tumor_distance_baseline_comparison(
    *,
    model,
    all_records: list[PatientRecord],
    evaluation_records: list[PatientRecord],
    derived_root: str | Path,
    max_voxels_per_case: int,
) -> dict[str, object] | None:
    if getattr(model, "kind", None) != "voxel-logistic-mri":
        return None
    training_records = filter_records(all_records, splits={"train"})
    if not training_records:
        return None
    training_cases = [_load_training_case(record, derived_root) for record in training_records]
    baseline_model = TumorDistanceBandModel.fit(training_cases, max_voxels_per_case=max_voxels_per_case)
    baseline_metrics = []
    for record in evaluation_records:
        case = load_case(case_dir(derived_root, record.patient_id), require_label=True)
        assert_case_geometry(case)
        baseline_risk = Volume(baseline_model.predict_case(case), case.t1c.affine)
        baseline_metrics.append(evaluate_case(case, baseline_risk))
    return {
        "baseline_model": "tumor-distance",
        "tumor_distance": summarize_metrics(baseline_metrics),
    }


def numeric_delta(value: object, baseline: object) -> float | None:
    if isinstance(value, (float, int)) and isinstance(baseline, (float, int)):
        return float(value) - float(baseline)
    return None


def summary_without_cases(summary: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in summary.items() if key != "cases"}


def parse_patch_size(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise ValueError("--patch-size must be three positive integers, e.g. 96,96,96")
    return tuple(parts)


def try_load_model(path: str | Path):
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
