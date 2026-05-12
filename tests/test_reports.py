from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from glioma_recurrence.case import CaseData
from glioma_recurrence.constants import CASE_QC_HTML, CASE_QC_SUMMARY_JSON, RESEARCH_ONLY_DISCLAIMER
from glioma_recurrence.geometry import Volume
from glioma_recurrence.reports import select_representative_slices, write_case_qc_report


def make_case(*, with_recurrence: bool = True) -> tuple[CaseData, Volume]:
    shape = (12, 10, 8)
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    grid = np.indices(shape).astype(np.float32)
    t1c = grid[0] + 0.25 * grid[1]
    flair = grid[1] + 0.5 * grid[2]
    brain = np.ones(shape, dtype=np.uint8)
    tumor = np.zeros(shape, dtype=np.uint8)
    tumor[3:7, 3:7, 2] = 1
    recurrence = np.zeros(shape, dtype=np.uint8)
    recurrence[6:10, 4:8, 5] = 1
    risk = np.zeros(shape, dtype=np.float32)
    risk[5:11, 4:9, 6] = 0.9
    case = CaseData(
        patient_id="CASE001",
        t1c=Volume(t1c, affine),
        flair=Volume(flair, affine),
        brain_mask=Volume(brain, affine),
        baseline_tumor_mask=Volume(tumor, affine),
        recurrence_mask=Volume(recurrence, affine) if with_recurrence else None,
    )
    return case, Volume(risk, affine)


def test_case_qc_report_writes_interactive_overlay_assets_and_summary(tmp_path: Path):
    case, risk = make_case()

    html_path = write_case_qc_report(case, output_dir=tmp_path, risk=risk)

    assert html_path == tmp_path / CASE_QC_HTML
    assert (tmp_path / CASE_QC_SUMMARY_JSON).exists()
    report = html_path.read_text()
    summary = json.loads((tmp_path / CASE_QC_SUMMARY_JSON).read_text())

    assert "Overlay Controls" in report
    assert 'data-opacity-control="tumor"' in report
    assert 'data-opacity-control="recurrence"' in report
    assert 'data-opacity-control="risk"' in report
    assert "Representative Slices" in report
    assert "T1c with overlays" in report
    assert "FLAIR with overlays" in report
    assert RESEARCH_ONLY_DISCLAIMER in report
    assert summary["patient_id"] == "CASE001"
    assert summary["recurrence_mask_present"] is True
    assert summary["recurrence_voxels"] == 16
    assert summary["risk_present"] is True
    assert summary["risk_stats"]["max"] == 0.9
    assert {item["reason"] for item in summary["selected_slices"]} >= {
        "midline",
        "baseline tumor peak",
        "recurrence peak",
        "risk peak",
    }
    assert list(tmp_path.glob("qc_z*_t1c.png"))
    assert list(tmp_path.glob("qc_z*_flair.png"))
    assert list(tmp_path.glob("qc_z*_tumor.png"))
    assert list(tmp_path.glob("qc_z*_recurrence.png"))
    assert list(tmp_path.glob("qc_z*_risk.png"))


def test_case_qc_report_handles_prediction_without_recurrence_label(tmp_path: Path):
    case, risk = make_case(with_recurrence=False)

    write_case_qc_report(case, output_dir=tmp_path, risk=risk)

    report = (tmp_path / CASE_QC_HTML).read_text()
    summary = json.loads((tmp_path / CASE_QC_SUMMARY_JSON).read_text())
    assert "Recurrence voxels" in report
    assert "not available" in report
    assert summary["recurrence_mask_present"] is False
    assert summary["recurrence_voxels"] is None
    assert summary["risk_present"] is True


def test_representative_slice_selection_is_deterministic_and_unique():
    shape = (8, 8, 8)
    tumor = np.zeros(shape)
    recurrence = np.zeros(shape)
    risk = np.zeros(shape)
    tumor[:, :, 1] = 1
    recurrence[:, :, 4] = 1
    risk[:, :, 7] = 1

    slices = select_representative_slices(shape=shape, baseline_tumor=tumor, recurrence=recurrence, risk=risk)

    assert slices == [
        {"index": 4, "reason": "midline"},
        {"index": 1, "reason": "baseline tumor peak"},
        {"index": 7, "reason": "risk peak"},
    ]
