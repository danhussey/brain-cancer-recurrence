from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from glioma_recurrence.constants import BASELINE_T1C, RECURRENCE_MASK_ON_BASELINE
from glioma_recurrence.geometry import Volume
from glioma_recurrence.labels import LabelError, map_reviewed_mask_to_baseline
from glioma_recurrence.nifti import read_volume, write_volume
from glioma_recurrence.schema import PatientRecord


def make_record(mask_path: Path, *, reference_path: str = "") -> PatientRecord:
    return PatientRecord(
        patient_id="P001",
        baseline_scan_date=date(2024, 1, 1),
        baseline_t1c_series_uid="T1",
        baseline_flair_series_uid="F1",
        recurrence_scan_date=date(2024, 8, 1),
        recurrence_adjudication="confirmed",
        reviewed_recurrence_mask_path=str(mask_path),
        split="train",
        reviewed_recurrence_reference_image_path=reference_path,
    )


def test_affine_label_mapping_moves_mask_without_silent_flip(tmp_path: Path):
    derived = tmp_path / "derived"
    case_dir = derived / "P001"
    case_dir.mkdir(parents=True)
    baseline = Volume(np.zeros((7, 7, 7), dtype=np.float32), np.eye(4))
    write_volume(baseline, case_dir / BASELINE_T1C, dtype=np.float32)

    mask = np.zeros((7, 7, 7), dtype=np.uint8)
    mask[2, 2, 2] = 1
    moving_affine = np.eye(4)
    moving_affine[:3, 3] = [1, 0, 0]
    mask_path = tmp_path / "reviewed_mask.nii.gz"
    write_volume(Volume(mask, moving_affine), mask_path, dtype=np.uint8)

    output = map_reviewed_mask_to_baseline(
        make_record(mask_path),
        derived_root=derived,
        registration_mode="affine",
    )

    mapped = read_volume(output)
    assert output.name == RECURRENCE_MASK_ON_BASELINE
    assert int(mapped.data.sum()) == 1
    assert mapped.data[3, 2, 2] == 1


def test_simpleitk_registration_requires_reference_image(tmp_path: Path):
    derived = tmp_path / "derived"
    case_dir = derived / "P001"
    case_dir.mkdir(parents=True)
    write_volume(Volume(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), case_dir / BASELINE_T1C)
    mask_path = tmp_path / "reviewed_mask.nii.gz"
    write_volume(Volume(np.zeros((4, 4, 4), dtype=np.uint8), np.eye(4)), mask_path, dtype=np.uint8)

    with pytest.raises(LabelError, match="reviewed_recurrence_reference_image_path"):
        map_reviewed_mask_to_baseline(make_record(mask_path), derived_root=derived)
