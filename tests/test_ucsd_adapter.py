from __future__ import annotations

import csv
import importlib.util
import sys
import zipfile
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from glioma_recurrence.constants import BASELINE_FLAIR, BASELINE_T1C, BASELINE_TUMOR_MASK
from glioma_recurrence.geometry import Volume
from glioma_recurrence.nifti import read_volume, write_volume
from glioma_recurrence.schema import read_manifest


def load_ucsd_adapter():
    script_path = Path.cwd() / "scripts/prepare_ucsd_ptgbm_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_ucsd_ptgbm_dataset", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_ucsd_audit():
    script_path = Path.cwd() / "scripts/audit_ucsd_ptgbm_dataset.py"
    spec = importlib.util.spec_from_file_location("audit_ucsd_ptgbm_dataset", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_fake_ucsd_timepoint(root: Path, subject: str, timepoint: str, *, mask_name: str = "TCT_mask") -> None:
    directory = root / subject / timepoint
    directory.mkdir(parents=True)
    affine = np.eye(4)
    data = np.ones((5, 5, 5), dtype=np.float32)
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[2:4, 2:4, 2:4] = 1
    write_volume(Volume(data, affine), directory / f"{subject}_{timepoint}_t1gd.nii.gz", dtype=np.float32)
    write_volume(Volume(data, affine), directory / f"{subject}_{timepoint}_flair.nii.gz", dtype=np.float32)
    write_volume(Volume(mask, affine), directory / f"{subject}_{timepoint}_{mask_name}.nii.gz", dtype=np.uint8)


def write_fake_workbook(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    sheet_rows = [headers, *rows]
    row_xml = []
    for row_index, values in enumerate(sheet_rows, start=1):
        cells = []
        for column_index, value in enumerate(values):
            ref = f"{chr(ord('A') + column_index)}{row_index}"
            cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>')
        row_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(row_xml)}</sheetData></worksheet>'
    )
    with zipfile.ZipFile(path, "w") as workbook:
        workbook.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        workbook.writestr("xl/workbook.xml", '<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"/>')
        workbook.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def sheet_xml_from_rows(sheet_rows: list[list[str]]) -> str:
    row_xml = []
    for row_index, values in enumerate(sheet_rows, start=1):
        cells = []
        for column_index, value in enumerate(values):
            ref = f"{chr(ord('A') + column_index)}{row_index}"
            cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>')
        row_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(row_xml)}</sheetData></worksheet>'
    )


def write_fake_two_sheet_workbook(path: Path, sheet1_rows: list[list[str]], sheet2_rows: list[list[str]]) -> None:
    with zipfile.ZipFile(path, "w") as workbook:
        workbook.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        workbook.writestr("xl/workbook.xml", '<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"/>')
        workbook.writestr("xl/worksheets/sheet1.xml", sheet_xml_from_rows(sheet1_rows))
        workbook.writestr("xl/worksheets/sheet2.xml", sheet_xml_from_rows(sheet2_rows))


def test_prepare_ucsd_dataset_pairs_longitudinal_recurrence_and_skips_ambiguous_subjects(tmp_path: Path):
    adapter = load_ucsd_adapter()
    source = tmp_path / "UCSD-PTGBM"
    for subject, timepoints in {"S1": ["T0", "T1"], "S2": ["T0"], "S3": ["T0", "T1"]}.items():
        for timepoint in timepoints:
            write_fake_ucsd_timepoint(source, subject, timepoint)
    workbook = tmp_path / "clinical.xlsx"
    write_fake_workbook(
        workbook,
        ["subject_id", "timepoint_id", "scan_date", "progression_status"],
        [
            ["S1", "T0", "2024-01-01", "post treatment baseline"],
            ["S1", "T1", "2024-08-01", "residual/recurrent tumor clinically confirmed"],
            ["S2", "T0", "2024-01-01", "post treatment baseline"],
            ["S3", "T0", "2024-01-01", "post treatment baseline"],
            ["S3", "T1", "2024-08-01", "clinically confirmed pseudoprogression"],
        ],
    )
    output = tmp_path / "prepared"

    prepared = adapter.prepare_ucsd_dataset(source, workbook, output)

    assert prepared.selected_subjects == ["S1"]
    assert prepared.skipped_subjects["S2"] == "fewer than two complete MRI+mask timepoints"
    assert prepared.skipped_subjects["S3"] == "no later residual/recurrent tumor timepoint"
    rows = list(csv.DictReader(prepared.manifest.open()))
    assert set(rows[0]) == {
        "patient_id",
        "baseline_scan_date",
        "baseline_t1c_series_uid",
        "baseline_flair_series_uid",
        "recurrence_scan_date",
        "recurrence_adjudication",
        "reviewed_recurrence_mask_path",
        "split",
        "reviewed_recurrence_reference_image_path",
        "source_dataset",
        "baseline_timepoint_id",
        "recurrence_timepoint_id",
        "radiotherapy_end_date",
    }
    assert rows[0]["recurrence_adjudication"] == "clinically_confirmed"
    assert (prepared.derived_root / rows[0]["patient_id"] / BASELINE_T1C).exists()
    assert (prepared.derived_root / rows[0]["patient_id"] / BASELINE_FLAIR).exists()
    assert (prepared.derived_root / rows[0]["patient_id"] / BASELINE_TUMOR_MASK).exists()
    assert Path(rows[0]["reviewed_recurrence_reference_image_path"]).exists()
    assert read_manifest(prepared.manifest)[0].source_dataset == "UCSD-PTGBM"
    assert int(read_volume(prepared.derived_root / rows[0]["patient_id"] / BASELINE_TUMOR_MASK).data.sum()) > 0


def test_prepare_ucsd_dataset_uses_clinical_ids_and_negative_case_categories(tmp_path: Path):
    adapter = load_ucsd_adapter()
    source = tmp_path / "UCSD-PTGBM"
    for subject, timepoints in {"S1": ["T0", "T1", "T2"], "S2": ["T0", "T1"]}.items():
        for timepoint in timepoints:
            write_fake_ucsd_timepoint(source, subject, timepoint)
    clinical = tmp_path / "clinical.xlsx"
    write_fake_two_sheet_workbook(
        clinical,
        [["Data Collection Name", "Data Descriptor /Metadata Name"], ["ID", "TCIA ID"]],
        [
            ["ID", "BraTS ID", "Patient's Age"],
            ["S1_T0", "B1", "60"],
            ["S1_T1", "B2", "60"],
            ["S1_T2", "B3", "60"],
            ["S2_T0", "B4", "61"],
            ["S2_T1", "B5", "61"],
        ],
    )
    negatives = tmp_path / "negative.xlsx"
    write_fake_workbook(negatives, ["ID", "Dx"], [["S1_T1", "PsP (clinically confirmed)"], ["S2_T1", "NS"]])

    prepared = adapter.prepare_ucsd_dataset(source, clinical, tmp_path / "prepared", negative_cases_table=negatives)

    rows = list(csv.DictReader(prepared.manifest.open()))
    assert prepared.selected_subjects == ["S1"]
    assert prepared.skipped_subjects["S2"] == "no later residual/recurrent tumor timepoint"
    assert rows[0]["baseline_timepoint_id"] == "T0"
    assert rows[0]["recurrence_timepoint_id"] == "T2"
    assert rows[0]["recurrence_adjudication"] == "confirmed"


def test_prepare_ucsd_dataset_can_include_negative_controls_with_empty_labels(tmp_path: Path):
    adapter = load_ucsd_adapter()
    source = tmp_path / "UCSD-PTGBM"
    for subject, timepoints in {"S1": ["T0", "T1"], "S2": ["T0", "T1"]}.items():
        for timepoint in timepoints:
            write_fake_ucsd_timepoint(source, subject, timepoint)
    clinical = tmp_path / "clinical.xlsx"
    write_fake_two_sheet_workbook(
        clinical,
        [["Data Collection Name", "Data Descriptor /Metadata Name"], ["ID", "TCIA ID"]],
        [
            ["ID", "BraTS ID"],
            ["S1_T0", "B1"],
            ["S1_T1", "B2"],
            ["S2_T0", "B3"],
            ["S2_T1", "B4"],
        ],
    )
    negatives = tmp_path / "negative.xlsx"
    write_fake_workbook(negatives, ["ID", "Dx"], [["S2_T1", "Radiation necrosis"]])

    prepared = adapter.prepare_ucsd_dataset(
        source,
        clinical,
        tmp_path / "prepared",
        negative_cases_table=negatives,
        include_negative_controls=True,
    )

    rows = list(csv.DictReader(prepared.manifest.open()))
    assert [row["recurrence_adjudication"] for row in rows] == [
        "confirmed",
        "clinical_negative_radiation_necrosis",
    ]
    negative_row = rows[1]
    negative_mask = read_volume(negative_row["reviewed_recurrence_mask_path"])
    assert int(negative_mask.data.sum()) == 0


def test_prepare_ucsd_dataset_requires_explicit_imaging_only_mode_without_clinical_table(tmp_path: Path):
    adapter = load_ucsd_adapter()
    source = tmp_path / "UCSD-PTGBM"
    write_fake_ucsd_timepoint(source, "S1", "T0")
    write_fake_ucsd_timepoint(source, "S1", "T1")

    try:
        adapter.prepare_ucsd_dataset(source, None, tmp_path / "prepared")
    except RuntimeError as exc:
        assert "--allow-imaging-only-labels" in str(exc)
    else:
        raise AssertionError("expected missing clinical table to require explicit imaging-only mode")


def test_prepare_ucsd_dataset_infers_provisional_pairs_from_filenames(tmp_path: Path):
    adapter = load_ucsd_adapter()
    source = tmp_path / "UCSD-PTGBM"
    write_fake_ucsd_timepoint(source, "S1", "T0")
    write_fake_ucsd_timepoint(source, "S1", "T1")
    write_fake_ucsd_timepoint(source, "S2", "T0")
    output = tmp_path / "prepared"

    prepared = adapter.prepare_ucsd_dataset(source, None, output, allow_imaging_only_labels=True)

    rows = list(csv.DictReader(prepared.manifest.open()))
    assert prepared.selected_subjects == ["S1"]
    assert prepared.skipped_subjects["S2"] == "fewer than two complete MRI+mask timepoints"
    assert rows[0]["recurrence_adjudication"] == "imaging_followup_segmentation_present"
    assert rows[0]["baseline_timepoint_id"] == "T0"
    assert rows[0]["recurrence_timepoint_id"] == "T1"
    assert (output / "ucsd_prepare_summary.json").exists()


def test_prepare_ucsd_dataset_uses_balanced_patient_level_splits(tmp_path: Path):
    adapter = load_ucsd_adapter()
    source = tmp_path / "UCSD-PTGBM"
    for index in range(10):
        subject = f"S{index:02d}"
        write_fake_ucsd_timepoint(source, subject, "T0")
        write_fake_ucsd_timepoint(source, subject, "T1")
    output = tmp_path / "prepared"

    prepared = adapter.prepare_ucsd_dataset(source, None, output, allow_imaging_only_labels=True)

    rows = list(csv.DictReader(prepared.manifest.open()))
    split_counts = Counter(row["split"] for row in rows)
    assert split_counts == {"train": 6, "validation": 2, "test": 2}
    assert len({row["patient_id"] for row in rows}) == 10
    assert adapter.patient_level_split_counts(37, train_fraction=0.70, validation_fraction=0.15) == {
        "train": 25,
        "validation": 6,
        "test": 6,
    }


def test_audit_ucsd_dataset_reports_download_and_pairing_counts(tmp_path: Path):
    audit = load_ucsd_audit()
    source = tmp_path / "UCSD-PTGBM"
    write_fake_ucsd_timepoint(source, "S1", "T0")
    write_fake_ucsd_timepoint(source, "S1", "T1")
    write_fake_ucsd_timepoint(source, "S2", "T0")
    partial = source / "S3" / "T0" / "S3_T0_flair.nii.gz.partial"
    partial.parent.mkdir(parents=True)
    partial.write_text("partial")

    summary = audit.audit_ucsd_dataset(source)

    assert summary["pairing_mode"] == "filename-inferred"
    assert summary["partial_files"] == 1
    assert summary["subjects_seen"] == 2
    assert summary["complete_mri_mask_timepoints"] == 3
    assert summary["subjects_with_2plus_complete_timepoints"] == 1
    assert summary["eligible_pairs"] == 1
    assert summary["eligible_subjects"] == ["S1"]
    assert summary["modality_counts"]["t1c"] == 3
    assert summary["modality_counts"]["flair"] == 3
