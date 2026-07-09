from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from glioma_recurrence.case import CaseData
from glioma_recurrence.constants import (
    CASE_QC_HTML,
    CASE_QC_SUMMARY_JSON,
    PREPROCESS_QC_HTML,
    PREPROCESS_QC_SUMMARY_JSON,
    RESEARCH_ONLY_DISCLAIMER,
)
from glioma_recurrence.geometry import Volume
from glioma_recurrence.reports import (
    select_preprocess_slices,
    select_representative_slices,
    select_viewer_slices,
    write_case_qc_report,
    write_preprocess_qc_report,
)


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
    risk[6:10, 4:8, 5] = 1.0
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
    assert "Timepoint Context" in report
    assert "Post-operative, pre-radiotherapy T1c and FLAIR" in report
    assert "prediction-time inputs" in report
    assert "Later follow-up reviewed label mapped back to baseline space" in report
    assert "training and evaluation only" in report
    assert "Overlay Key" in report
    assert "Overlay color key" in report
    assert "Cyan mask; baseline post-op / pre-radiotherapy" in report
    assert "Magenta mask; later follow-up label mapped to baseline" in report
    assert "Blue to orange model output in baseline space" in report
    assert 'data-opacity-control="tumor"' in report
    assert 'data-opacity-control="recurrence"' in report
    assert 'data-opacity-control="risk"' in report
    assert 'value="4" data-initial-slice="4" data-slice-slider' in report
    assert "Axial Slice Browser" in report
    assert 'data-slice-slider' in report
    assert 'data-slice-jump' in report
    assert 'class="summary-help"' in report
    assert "Useful for marginal or distant recurrence review." in report
    assert "Mean risk in recurrence" in report
    assert "Mean risk outside recurrence" in report
    assert "Top 1% risk overlap" in report
    assert "Top 5% risk overlap" in report
    assert "16 recurrence voxels in 16 high-risk voxels; coverage 100.00%; Dice 1.000" in report
    assert "T1c baseline post-op / pre-radiotherapy with overlays" in report
    assert "FLAIR baseline post-op / pre-radiotherapy with overlays" in report
    assert RESEARCH_ONLY_DISCLAIMER in report
    assert summary["patient_id"] == "CASE001"
    assert summary["recurrence_mask_present"] is True
    assert summary["recurrence_voxels"] == 16
    assert summary["recurrence_location"] == {
        "inside_baseline_tumor_voxels": 0,
        "outside_baseline_tumor_voxels": 16,
        "outside_baseline_tumor_fraction": 1.0,
    }
    assert summary["risk_present"] is True
    assert summary["risk_stats"]["max"] == 1.0
    assert summary["prediction_overlap"]["positive_voxels"] == 16
    assert summary["prediction_overlap"]["evaluated_voxels"] == 960
    assert summary["prediction_overlap"]["mean_risk_in_recurrence"] == 1.0
    assert summary["prediction_overlap"]["top_1pct"] == {
        "fraction": 0.01,
        "risk_threshold": 1.0,
        "predicted_voxels": 16,
        "overlap_voxels": 16,
        "recurrence_coverage": 1.0,
        "dice": 1.0,
    }
    assert summary["viewer_slice_count"] == 8
    assert {item["reason"] for item in summary["selected_slices"]} >= {
        "midline",
        "baseline tumor peak",
        "recurrence peak",
        "risk peak",
    }
    assert len(list(tmp_path.glob("qc_z*_t1c.png"))) == 8
    assert len(list(tmp_path.glob("qc_z*_flair.png"))) == 8
    assert len(list(tmp_path.glob("qc_z*_tumor.png"))) == 8
    assert len(list(tmp_path.glob("qc_z*_recurrence.png"))) == 8
    assert len(list(tmp_path.glob("qc_z*_risk.png"))) == 8


def test_case_qc_report_handles_prediction_without_recurrence_label(tmp_path: Path):
    case, risk = make_case(with_recurrence=False)

    write_case_qc_report(case, output_dir=tmp_path, risk=risk)

    report = (tmp_path / CASE_QC_HTML).read_text()
    summary = json.loads((tmp_path / CASE_QC_SUMMARY_JSON).read_text())
    assert "Recurrence voxels" in report
    assert "not available" in report
    assert summary["recurrence_mask_present"] is False
    assert summary["recurrence_voxels"] is None
    assert summary["recurrence_location"] is None
    assert summary["risk_present"] is True
    assert summary["prediction_overlap"] is None


def test_preprocess_qc_report_writes_brain_mask_checkerboard_and_summary(tmp_path: Path):
    case, _ = make_case()
    source_flair = Volume(case.flair.data, np.diag([1.0, 1.0, 3.0, 1.0]))

    html_path = write_preprocess_qc_report(
        case,
        output_dir=tmp_path,
        source_t1c=case.t1c,
        source_flair=source_flair,
        source_baseline_tumor=case.baseline_tumor_mask,
    )

    assert html_path == tmp_path / PREPROCESS_QC_HTML
    assert (tmp_path / PREPROCESS_QC_SUMMARY_JSON).exists()
    report = html_path.read_text()
    summary = json.loads((tmp_path / PREPROCESS_QC_SUMMARY_JSON).read_text())

    assert "Preprocessing QC" in report
    assert "Axial Preprocessing Viewer" in report
    assert "T1c/FLAIR checkerboard" in report
    assert 'data-opacity-control="brain"' in report
    assert 'value="4" data-initial-slice="4" data-slice-slider' in report
    assert "dedicated skull stripping not yet applied" in report
    assert summary["preprocessing_steps"]["bias_correction_status"] == "not applied"
    assert summary["source_geometry"]["flair_matches_t1c_before_preprocess"] is False
    assert "source FLAIR geometry differed from T1c before preprocessing" in summary["quality_flags"]
    assert len(list(tmp_path.glob("preprocess_z*_t1c.png"))) == summary["viewer_slice_count"]
    assert len(list(tmp_path.glob("preprocess_z*_checkerboard.png"))) == summary["viewer_slice_count"]
    assert len(list(tmp_path.glob("preprocess_z*_brain.png"))) == summary["viewer_slice_count"]


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


def test_preprocess_slice_selection_keeps_midline_and_tumor_peak():
    brain = np.zeros((8, 8, 8))
    tumor = np.zeros((8, 8, 8))
    brain[:, :, 2] = 1
    tumor[:, :, 6] = 1

    slices = select_preprocess_slices(shape=(8, 8, 8), brain=brain, baseline_tumor=tumor)

    assert slices == [
        {"index": 4, "reason": "midline"},
        {"index": 2, "reason": "brain mask peak"},
        {"index": 6, "reason": "baseline tumor peak"},
    ]


def test_viewer_slice_selection_samples_large_volumes_and_keeps_key_slices():
    selected = [
        {"index": 3, "reason": "baseline tumor peak"},
        {"index": 98, "reason": "risk peak"},
    ]

    slices = select_viewer_slices(shape=(16, 16, 100), selected_slices=selected, max_slices=10)

    assert len(slices) <= 10
    assert slices[0]["index"] == 0
    assert slices[-1]["index"] == 99
    by_index = {item["index"]: item["reason"] for item in slices}
    assert by_index[3] == "baseline tumor peak"
    assert by_index[98] == "risk peak"
