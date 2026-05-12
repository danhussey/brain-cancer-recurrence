from __future__ import annotations

import csv
import json
from pathlib import Path

from glioma_recurrence.cli import main as cli_main
from glioma_recurrence.dicom import classify_mri_sequence


def write_fake_dicom(
    path: Path,
    *,
    patient_id: str,
    study_uid: str,
    series_uid: str,
    sop_uid: str,
    series_description: str,
    protocol_name: str,
    instance_number: int,
) -> None:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage

    path.parent.mkdir(parents=True, exist_ok=True)
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.10.999"

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.PatientID = patient_id
    dataset.PatientName = "Example^Patient"
    dataset.PatientBirthDate = "19700101"
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.SOPClassUID = MRImageStorage
    dataset.SOPInstanceUID = sop_uid
    dataset.Modality = "MR"
    dataset.SeriesDescription = series_description
    dataset.ProtocolName = protocol_name
    dataset.StudyDate = "20260101"
    dataset.SeriesDate = "20260101"
    dataset.Manufacturer = "ScannerCo"
    dataset.ManufacturerModelName = "Model X"
    dataset.MagneticFieldStrength = "3"
    dataset.Rows = 64
    dataset.Columns = 64
    dataset.PixelSpacing = [1.0, 1.0]
    dataset.SliceThickness = "1.5"
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
    dataset.InstanceNumber = instance_number
    dataset.save_as(path, enforce_file_format=True)


def test_classify_mri_sequence_handles_core_project_sequences():
    assert classify_mri_sequence("T1 MPRAGE") == "t1"
    assert classify_mri_sequence("T1 post contrast") == "t1c"
    assert classify_mri_sequence("T1gd axial") == "t1c"
    assert classify_mri_sequence("T2 axial") == "t2"
    assert classify_mri_sequence("T2 FLAIR") == "flair"


def test_dicom_audit_cli_writes_pseudonymous_series_inventory(tmp_path: Path):
    dicom_root = tmp_path / "dicom"
    study_uid = "1.2.826.0.1.3680043.10.1000"
    series = {
        "t1": ("1.2.826.0.1.3680043.10.1001", "T1 MPRAGE"),
        "t1c": ("1.2.826.0.1.3680043.10.1002", "T1 post contrast"),
        "t2": ("1.2.826.0.1.3680043.10.1003", "T2 axial"),
        "flair": ("1.2.826.0.1.3680043.10.1004", "T2 FLAIR"),
    }
    for index, (label, (series_uid, description)) in enumerate(series.items(), start=1):
        write_fake_dicom(
            dicom_root / label / "image-1.dcm",
            patient_id="LOCAL-PATIENT-001",
            study_uid=study_uid,
            series_uid=series_uid,
            sop_uid=f"1.2.826.0.1.3680043.10.200{index}",
            series_description=description,
            protocol_name=description,
            instance_number=1,
        )
        write_fake_dicom(
            dicom_root / label / "image-2.dcm",
            patient_id="LOCAL-PATIENT-001",
            study_uid=study_uid,
            series_uid=series_uid,
            sop_uid=f"1.2.826.0.1.3680043.10.300{index}",
            series_description=description,
            protocol_name=description,
            instance_number=2,
        )

    output_csv = tmp_path / "reports" / "dicom-series.csv"
    summary_json = tmp_path / "reports" / "dicom-summary.json"

    assert cli_main(
        [
            "dicom-audit",
            "--dicom-root",
            str(dicom_root),
            "--output",
            str(output_csv),
            "--summary-output",
            str(summary_json),
        ]
    ) == 0

    rows = list(csv.DictReader(output_csv.open()))
    summary = json.loads(summary_json.read_text())

    assert len(rows) == 4
    assert {row["sequence_label"] for row in rows} == {"t1", "t1c", "t2", "flair"}
    assert {int(row["instance_count"]) for row in rows} == {2}
    assert rows[0]["patient_key"].startswith("patient-")
    assert {row["first_file"] for row in rows} == {""}
    assert "LOCAL-PATIENT-001" not in output_csv.read_text()
    assert summary["patient_count"] == 1
    assert summary["studies_with_minimum_t1c_flair"] == 1
    assert summary["studies_with_full_t1_t1c_t2_flair"] == 1
    assert summary["patient_name_present_series"] == 4
    assert summary["patient_birth_date_present_series"] == 4
