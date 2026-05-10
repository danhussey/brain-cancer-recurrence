from __future__ import annotations

from pathlib import Path

import pytest

from glioma_recurrence.schema import ManifestError, read_manifest


HEADER = (
    "patient_id,baseline_scan_date,baseline_t1c_series_uid,baseline_flair_series_uid,"
    "recurrence_scan_date,recurrence_adjudication,reviewed_recurrence_mask_path,split,"
    "reviewed_recurrence_reference_image_path,source_dataset,baseline_timepoint_id,"
    "recurrence_timepoint_id,radiotherapy_end_date\n"
)


def write_manifest(path: Path, rows: list[str]) -> Path:
    path.write_text(HEADER + "\n".join(rows) + "\n")
    return path


def test_manifest_rejects_patient_level_split_leakage(tmp_path: Path):
    manifest = write_manifest(
        tmp_path / "patients.csv",
        [
            "P001,2024-01-01,T1,F1,2024-06-01,confirmed,/mask.nii.gz,train,/followup.nii.gz,UCSD,t0,t1,",
            "P001,2024-01-01,T2,F2,2024-06-01,confirmed,/mask.nii.gz,test,/followup.nii.gz,UCSD,t0,t1,",
        ],
    )

    with pytest.raises(ManifestError, match="patient-level leakage"):
        read_manifest(manifest)


def test_pseudoprogression_window_is_flagged_unless_confirmed(tmp_path: Path):
    manifest = write_manifest(
        tmp_path / "patients.csv",
        [
            "P001,2024-01-01,T1,F1,2024-02-01,suspected,/mask.nii.gz,train,/followup.nii.gz,UCSD,t0,t1,2024-01-15",
            "P002,2024-01-01,T2,F2,2024-02-01,histologically_confirmed,/mask.nii.gz,train,/followup.nii.gz,UCSD,t0,t1,2024-01-15",
        ],
    )

    records = read_manifest(manifest)

    assert records[0].is_pseudoprogression_window
    assert records[0].should_exclude_from_training
    assert records[1].is_pseudoprogression_window
    assert not records[1].should_exclude_from_training


def test_manifest_reads_optional_ucsd_pair_metadata(tmp_path: Path):
    manifest = write_manifest(
        tmp_path / "patients.csv",
        [
            "P001,2024-01-01,T1,F1,2024-06-01,confirmed,/mask.nii.gz,train,/followup.nii.gz,UCSD,t0,t1,",
        ],
    )

    record = read_manifest(manifest)[0]

    assert record.reviewed_recurrence_reference_image_path == "/followup.nii.gz"
    assert record.source_dataset == "UCSD"
    assert record.baseline_timepoint_id == "t0"
    assert record.recurrence_timepoint_id == "t1"
