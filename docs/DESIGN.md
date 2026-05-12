# Design

This is a research pipeline, not a consumer application. Design decisions prioritize legibility, reproducibility, and high-signal QC over decorative UI.

## Experience Principles

- The first screen of any report should identify the patient/case, stage, and research-only limitation.
- QC reports must expose the actual imaging-derived artifacts: T1c, FLAIR, baseline tumor mask, recurrence mask, and prediction.
- QC reports should provide overlay opacity controls, representative slice tabs, and machine-readable summary metadata.
- Visualizations should be dense enough for repeated review and conservative enough for clinical-research audit.
- The pipeline should fail loudly on ambiguous geometry, missing human-reviewed labels, and split leakage.

## Report Design

The current report target is static HTML written by `glioma_recurrence.reports`. It is intentionally dependency-light and viewable from the filesystem. Future report enhancements should keep the fixed mandatory panels and add comparative views only when they improve QC.
