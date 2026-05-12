"""Stable public constants for the glioma recurrence-risk pipeline."""

from __future__ import annotations

from pathlib import Path

BASELINE_T1C = "baseline_t1c.nii.gz"
BASELINE_FLAIR = "baseline_flair.nii.gz"
BASELINE_TUMOR_MASK = "baseline_tumor_mask.nii.gz"
RECURRENCE_MASK_ON_BASELINE = "recurrence_mask_on_baseline.nii.gz"
BRAIN_MASK = "brain_mask.nii.gz"
RECURRENCE_RISK = "recurrence_risk.nii.gz"
CASE_QC_HTML = "qc_overlay.html"
CASE_QC_SUMMARY_JSON = "qc_summary.json"

DERIVED_FILENAMES = (
    BASELINE_T1C,
    BASELINE_FLAIR,
    BASELINE_TUMOR_MASK,
    RECURRENCE_MASK_ON_BASELINE,
    BRAIN_MASK,
    RECURRENCE_RISK,
)

REQUIRED_MANIFEST_COLUMNS = (
    "patient_id",
    "baseline_scan_date",
    "baseline_t1c_series_uid",
    "baseline_flair_series_uid",
    "recurrence_scan_date",
    "recurrence_adjudication",
    "reviewed_recurrence_mask_path",
    "split",
)

OPTIONAL_MANIFEST_COLUMNS = (
    "reviewed_recurrence_reference_image_path",
    "source_dataset",
    "baseline_timepoint_id",
    "recurrence_timepoint_id",
    "radiotherapy_end_date",
    "baseline_study_instance_uid",
    "baseline_t1_series_uid",
    "baseline_t2_series_uid",
    "input_format",
    "institution_id",
    "scanner_manufacturer",
    "scanner_model",
    "magnetic_field_strength",
    "label_source",
)

ALLOWED_SPLITS = {"train", "validation", "val", "test", "holdout"}

RESEARCH_ONLY_DISCLAIMER = (
    "Research use only. This output is a calibrated recurrence-risk heatmap "
    "in baseline space and is not a clinical dose recommendation."
)


def case_dir(derived_root: str | Path, patient_id: str) -> Path:
    """Return the fixed derived-data directory for one patient."""

    return Path(derived_root) / patient_id
