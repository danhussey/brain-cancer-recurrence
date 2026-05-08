#!/usr/bin/env python3
"""Generate a tiny synthetic glioma recurrence-risk dataset.

The generated dataset is for engineering smoke tests only. It creates the
post-ingest derived NIfTI layout plus a `patients.csv` manifest and reviewed
mask files, so the real CLI stages can run from `preprocess` onward without
public data downloads or patient data.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from glioma_recurrence.constants import BASELINE_FLAIR, BASELINE_T1C, BRAIN_MASK, DOSE_ON_BASELINE
from glioma_recurrence.geometry import Volume
from glioma_recurrence.nifti import write_volume


@dataclass(frozen=True)
class SyntheticDatasetPaths:
    root: Path
    manifest: Path
    derived_root: Path
    masks_root: Path
    models_root: Path
    reports_root: Path


def generate_synthetic_dataset(
    output_root: str | Path,
    *,
    n_patients: int = 3,
    shape: tuple[int, int, int] = (16, 16, 16),
    seed: int = 13,
    prescription_dose_gy: float = 60.0,
) -> SyntheticDatasetPaths:
    if n_patients < 2:
        raise ValueError("n_patients must be at least 2 so train and validation splits exist")
    if any(value < 8 for value in shape):
        raise ValueError("all shape dimensions must be at least 8")
    if prescription_dose_gy <= 0:
        raise ValueError("prescription_dose_gy must be positive")

    root = Path(output_root)
    derived_root = root / "derived"
    masks_root = root / "masks"
    models_root = root / "models"
    reports_root = root / "reports"
    for directory in (derived_root, masks_root, models_root, reports_root):
        directory.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    affine = np.diag([1.5, 1.5, 2.0, 1.0]).astype(float)
    rows: list[dict[str, str]] = []
    for index in range(n_patients):
        patient_id = f"SYN{index + 1:03d}"
        split = _split_for_index(index, n_patients)
        case_dir = derived_root / patient_id
        case_dir.mkdir(parents=True, exist_ok=True)

        volumes = _make_case_volumes(
            shape=shape,
            patient_index=index,
            rng=rng,
            prescription_dose_gy=prescription_dose_gy,
        )
        write_volume(Volume(volumes["t1c"], affine), case_dir / BASELINE_T1C, dtype=np.float32)
        write_volume(Volume(volumes["flair"], affine), case_dir / BASELINE_FLAIR, dtype=np.float32)
        write_volume(Volume(volumes["dose"], affine), case_dir / DOSE_ON_BASELINE, dtype=np.float32)
        write_volume(Volume(volumes["brain"], affine), case_dir / BRAIN_MASK, dtype=np.uint8)

        reviewed_mask_path = masks_root / f"{patient_id}_reviewed_mask.nii.gz"
        write_volume(Volume(volumes["label"], affine), reviewed_mask_path, dtype=np.uint8)
        rows.append(
            {
                "patient_id": patient_id,
                "baseline_scan_date": "2024-01-01",
                "baseline_t1c_series_uid": f"synthetic-t1c-{patient_id}",
                "baseline_flair_series_uid": f"synthetic-flair-{patient_id}",
                "rtdose_sop_instance_uid": f"synthetic-rtdose-{patient_id}",
                "recurrence_scan_date": "2024-09-01",
                "recurrence_adjudication": "confirmed",
                "reviewed_recurrence_mask_path": str(reviewed_mask_path),
                "split": split,
                "radiotherapy_end_date": "2024-03-01",
                "prescription_dose_gy": f"{prescription_dose_gy:g}",
            }
        )

    manifest = root / "patients.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "purpose": "engineering smoke test only; no clinical or scientific use",
        "n_patients": n_patients,
        "shape": list(shape),
        "seed": seed,
        "prescription_dose_gy": prescription_dose_gy,
        "manifest": str(manifest),
        "derived_root": str(derived_root),
    }
    (root / "synthetic_dataset.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return SyntheticDatasetPaths(root, manifest, derived_root, masks_root, models_root, reports_root)


def _split_for_index(index: int, n_patients: int) -> str:
    if index == 0:
        return "train"
    if index == 1:
        return "validation"
    if n_patients == 3 or index == n_patients - 1:
        return "test"
    return "train"


def _make_case_volumes(
    *,
    shape: tuple[int, int, int],
    patient_index: int,
    rng: np.random.Generator,
    prescription_dose_gy: float,
) -> dict[str, np.ndarray]:
    grid = np.indices(shape).astype(np.float32)
    center = np.asarray(
        [
            shape[0] * (0.45 + 0.03 * ((patient_index % 3) - 1)),
            shape[1] * (0.50 + 0.02 * ((patient_index % 2) * 2 - 1)),
            shape[2] * 0.50,
        ],
        dtype=np.float32,
    )[:, None, None, None]
    distance = np.sqrt(((grid - center) ** 2).sum(axis=0))
    recurrence_radius = max(2.0, min(shape) * 0.14)
    treatment_radius = max(3.0, min(shape) * 0.28)
    label = (distance <= recurrence_radius).astype(np.uint8)
    brain_center = np.asarray(shape, dtype=np.float32)[:, None, None, None] / 2.0
    brain_distance = np.sqrt((((grid - brain_center) / np.asarray(shape, dtype=np.float32)[:, None, None, None]) ** 2).sum(axis=0))
    brain = (brain_distance <= 0.52).astype(np.uint8)
    dose = prescription_dose_gy * np.exp(-(distance / treatment_radius) ** 2)
    dose *= brain
    edema = np.exp(-(distance / (recurrence_radius * 1.7)) ** 2) * brain
    enhancing = np.exp(-(distance / (recurrence_radius * 0.9)) ** 2) * brain
    t1c = (0.05 * rng.normal(size=shape) + 0.25 * brain + 2.0 * enhancing).astype(np.float32)
    flair = (0.05 * rng.normal(size=shape) + 0.20 * brain + 2.5 * edema).astype(np.float32)
    return {
        "t1c": t1c * brain,
        "flair": flair * brain,
        "dose": dose.astype(np.float32),
        "brain": brain.astype(np.uint8),
        "label": (label & brain).astype(np.uint8),
    }


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be three comma-separated integers, e.g. 16,16,16")
    return tuple(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, help="Directory where synthetic data will be written")
    parser.add_argument("--n-patients", type=int, default=3)
    parser.add_argument("--shape", type=parse_shape, default=(16, 16, 16))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--prescription-dose-gy", type=float, default=60.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = generate_synthetic_dataset(
        args.output_root,
        n_patients=args.n_patients,
        shape=args.shape,
        seed=args.seed,
        prescription_dose_gy=args.prescription_dose_gy,
    )
    print(paths.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

