#!/usr/bin/env python3
"""Prepare a small external CFB-GBM workspace for the pipeline.

This adapter intentionally copies selected cases instead of symlinking them.
The current `preprocess` stage overwrites derived NIfTI files in place, so
symlinking source data would risk mutating the downloaded CFB-GBM files.

By default this script only writes a manifest and copies the minimum baseline
modalities for a selected pilot subset:

- `t0_t1gd` -> `baseline_t1c.nii.gz`
- `t0_flair` -> `baseline_flair.nii.gz`
- `t0_rtdose` -> `dose_gy_on_baseline.nii.gz`
- optional `t0_gtv` -> reviewed mask proxy

The GTV mask is not a recurrence label. Use `--allow-gtv-proxy-labels` only for
pipeline mechanics and QC smoke runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from glioma_recurrence.constants import BASELINE_FLAIR, BASELINE_T1C, DOSE_ON_BASELINE


@dataclass(frozen=True)
class CfbCase:
    patient_id: str
    case_dir: Path
    t1gd: Path
    flair: Path
    rtdose: Path
    gtv: Path | None


@dataclass(frozen=True)
class PreparedCfbDataset:
    manifest: Path
    derived_root: Path
    masks_root: Path
    selected_cases: list[str]
    skipped_cases: dict[str, list[str]]


def discover_cfb_cases(root: str | Path) -> tuple[list[CfbCase], dict[str, list[str]]]:
    cfb_root = Path(root)
    complete: list[CfbCase] = []
    skipped: dict[str, list[str]] = {}
    for patient_dir in sorted(path for path in cfb_root.iterdir() if path.is_dir() and path.name.isdigit()):
        t0 = patient_dir / "t0"
        patient_id = patient_dir.name
        files = _find_t0_files(t0, patient_id)
        missing = [key for key in ("t1gd", "flair", "rtdose") if files.get(key) is None]
        if missing:
            skipped[patient_id] = missing
            continue
        complete.append(
            CfbCase(
                patient_id=patient_id,
                case_dir=patient_dir,
                t1gd=files["t1gd"],
                flair=files["flair"],
                rtdose=files["rtdose"],
                gtv=files.get("gtv"),
            )
        )
    return complete, skipped


def prepare_cfb_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    max_cases: int = 2,
    allow_gtv_proxy_labels: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> PreparedCfbDataset:
    if max_cases < 1:
        raise ValueError("max_cases must be positive")
    cases, skipped = discover_cfb_cases(source_root)
    if allow_gtv_proxy_labels:
        cases = [case for case in cases if case.gtv is not None]
    selected = cases[:max_cases]
    if not selected:
        raise RuntimeError("no CFB-GBM cases matched the requested criteria")

    output = Path(output_root)
    derived_root = output / "derived"
    masks_root = output / "masks"
    reports_root = output / "reports"
    models_root = output / "models"
    if not dry_run:
        for directory in (derived_root, masks_root, reports_root, models_root):
            directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for index, case in enumerate(selected):
        split = _split_for_index(index, len(selected))
        case_output = derived_root / case.patient_id
        mask_path = ""
        if not dry_run:
            case_output.mkdir(parents=True, exist_ok=True)
            _copy_case_file(case.t1gd, case_output / BASELINE_T1C, overwrite=overwrite)
            _copy_case_file(case.flair, case_output / BASELINE_FLAIR, overwrite=overwrite)
            _copy_case_file(case.rtdose, case_output / DOSE_ON_BASELINE, overwrite=overwrite)
            if allow_gtv_proxy_labels and case.gtv is not None:
                mask_output = masks_root / f"{case.patient_id}_t0_gtv_proxy_mask.nii.gz"
                _copy_case_file(case.gtv, mask_output, overwrite=overwrite)
                mask_path = str(mask_output)
        elif allow_gtv_proxy_labels and case.gtv is not None:
            mask_path = str(masks_root / f"{case.patient_id}_t0_gtv_proxy_mask.nii.gz")

        rows.append(
            {
                "patient_id": case.patient_id,
                "baseline_scan_date": "1900-01-01",
                "baseline_t1c_series_uid": f"cfb-gbm-{case.patient_id}-t0-t1gd",
                "baseline_flair_series_uid": f"cfb-gbm-{case.patient_id}-t0-flair",
                "rtdose_sop_instance_uid": f"cfb-gbm-{case.patient_id}-t0-rtdose",
                "recurrence_scan_date": "",
                "recurrence_adjudication": "gtv_proxy_not_recurrence" if allow_gtv_proxy_labels else "unlabeled",
                "reviewed_recurrence_mask_path": mask_path,
                "split": split,
                "radiotherapy_end_date": "",
                "prescription_dose_gy": "60",
            }
        )

    manifest = output / "patients.csv"
    if not dry_run:
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "source_root": str(source_root),
            "manifest": str(manifest),
            "derived_root": str(derived_root),
            "selected_cases": [case.patient_id for case in selected],
            "skipped_case_count": len(skipped),
            "gtv_proxy_labels": allow_gtv_proxy_labels,
            "warning": "GTV proxy labels are not recurrence labels and are only valid for pipeline smoke runs.",
        }
        (output / "cfb_prepare_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    return PreparedCfbDataset(
        manifest=manifest,
        derived_root=derived_root,
        masks_root=masks_root,
        selected_cases=[case.patient_id for case in selected],
        skipped_cases=skipped,
    )


def _find_t0_files(t0: Path, patient_id: str) -> dict[str, Path | None]:
    if not t0.exists():
        return {"t1gd": None, "flair": None, "rtdose": None, "gtv": None}
    return {
        "t1gd": _first_existing(t0, f"{int(patient_id)}_t0_t1gd.nii.gz", f"{patient_id}_t0_t1gd.nii.gz"),
        "flair": _first_existing(t0, f"{int(patient_id)}_t0_flair.nii.gz", f"{patient_id}_t0_flair.nii.gz"),
        "rtdose": _first_existing(t0, f"{int(patient_id)}_t0_rtdose.nii.gz", f"{patient_id}_t0_rtdose.nii.gz"),
        "gtv": _first_existing(t0, f"{int(patient_id)}_t0_gtv.nii.gz", f"{patient_id}_t0_gtv.nii.gz"),
    }


def _first_existing(directory: Path, *names: str) -> Path | None:
    for name in names:
        path = directory / name
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            return path
    return None


def _copy_case_file(source: Path, target: Path, *, overwrite: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists; pass --overwrite to replace it")
    shutil.copy2(source, target)


def _split_for_index(index: int, total: int) -> str:
    if index == 0:
        return "train"
    if index == 1:
        return "validation"
    if total == 3 or index == total - 1:
        return "test"
    return "train"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="Path to the downloaded CFB-GBM directory")
    parser.add_argument("--output-root", required=True, help="External workspace for copied pilot derivatives")
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument("--allow-gtv-proxy-labels", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepared = prepare_cfb_dataset(
        args.source_root,
        args.output_root,
        max_cases=args.max_cases,
        allow_gtv_proxy_labels=args.allow_gtv_proxy_labels,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(f"selected_cases={','.join(prepared.selected_cases)}")
    print(f"manifest={prepared.manifest}")
    print(f"derived_root={prepared.derived_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
