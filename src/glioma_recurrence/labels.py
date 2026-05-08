"""Reviewed recurrence label ingestion and mapping."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import BASELINE_T1C, RECURRENCE_MASK_ON_BASELINE
from .geometry import Volume, resample_mask_to_reference
from .nifti import read_volume, write_volume
from .schema import PatientRecord


class LabelError(RuntimeError):
    """Raised when label creation would violate review requirements."""


def map_reviewed_mask_to_baseline(
    record: PatientRecord,
    *,
    derived_root: str | Path,
    mask_path: str | Path | None = None,
    assume_baseline_space: bool = False,
) -> Path:
    reviewed_path = Path(mask_path or record.reviewed_recurrence_mask_path)
    if not reviewed_path.exists():
        raise LabelError(
            f"{record.patient_id}: reviewed recurrence mask is required and was not found: {reviewed_path}"
        )
    case_dir = Path(derived_root) / record.patient_id
    baseline = read_volume(case_dir / BASELINE_T1C)
    reviewed = read_volume(reviewed_path)
    if assume_baseline_space:
        mapped = Volume((reviewed.data > 0.5).astype(np.uint8), baseline.affine, reviewed.metadata)
        if reviewed.shape != baseline.shape:
            raise LabelError(
                f"{record.patient_id}: --assume-baseline-space requires mask shape {baseline.shape}; got {reviewed.shape}"
            )
    else:
        mapped = resample_mask_to_reference(reviewed, baseline)
    output = case_dir / RECURRENCE_MASK_ON_BASELINE
    write_volume(mapped, output, dtype=np.uint8)
    return output

