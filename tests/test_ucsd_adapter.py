from __future__ import annotations

import csv
import importlib.util
import sys
import zipfile
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
