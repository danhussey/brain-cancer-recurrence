"""Case-level derived-data loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import (
    BASELINE_FLAIR,
    BASELINE_T1C,
    BRAIN_MASK,
    DOSE_ON_BASELINE,
    RECURRENCE_MASK_ON_BASELINE,
)
from .geometry import Volume
from .nifti import read_volume


@dataclass(frozen=True)
class CaseData:
    patient_id: str
    t1c: Volume
    flair: Volume
    dose_gy: Volume
    brain_mask: Volume
    recurrence_mask: Volume | None = None
    prescription_dose_gy: float | None = None

    @property
    def mask_bool(self) -> np.ndarray:
        return self.brain_mask.data.astype(bool)

    @property
    def labels_bool(self) -> np.ndarray:
        if self.recurrence_mask is None:
            raise ValueError("case has no recurrence mask")
        return self.recurrence_mask.data.astype(bool)


def load_case(case_dir: str | Path, *, require_label: bool = False, prescription_dose_gy: float | None = None) -> CaseData:
    path = Path(case_dir)
    recurrence_path = path / RECURRENCE_MASK_ON_BASELINE
    recurrence = read_volume(recurrence_path) if recurrence_path.exists() else None
    if require_label and recurrence is None:
        raise FileNotFoundError(f"missing required recurrence label: {recurrence_path}")
    return CaseData(
        patient_id=path.name,
        t1c=read_volume(path / BASELINE_T1C),
        flair=read_volume(path / BASELINE_FLAIR),
        dose_gy=read_volume(path / DOSE_ON_BASELINE),
        brain_mask=read_volume(path / BRAIN_MASK),
        recurrence_mask=recurrence,
        prescription_dose_gy=prescription_dose_gy,
    )


def assert_case_geometry(case: CaseData) -> None:
    reference_shape = case.t1c.shape
    reference_affine = case.t1c.affine
    for name, volume in (
        ("flair", case.flair),
        ("dose_gy", case.dose_gy),
        ("brain_mask", case.brain_mask),
        ("recurrence_mask", case.recurrence_mask),
    ):
        if volume is None:
            continue
        if volume.shape != reference_shape:
            raise ValueError(f"{case.patient_id}: {name} shape {volume.shape} does not match T1c {reference_shape}")
        if not np.allclose(volume.affine, reference_affine, atol=1e-4):
            raise ValueError(f"{case.patient_id}: {name} affine does not match baseline T1c")

