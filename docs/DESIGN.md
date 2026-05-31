# Design

This is a research pipeline, not a consumer application. Design decisions prioritize legibility, reproducibility, and high-signal QC over decorative UI.

## Experience Principles

- Static QC reports identify the patient/case and research-only limitation; stage identity is tracked in CLI observability artifacts.
- QC reports must expose the actual imaging-derived artifacts available for the stage: T1c, FLAIR, baseline tumor mask, recurrence mask, and/or prediction.
- QC reports should provide overlay opacity controls, axial slice browsing, representative quick jumps, and machine-readable summary metadata.
- Case-summary fields should include concise tooltips so collaborators can understand what each QC number represents and why it matters.
- QC summaries should make residual-tumor recurrence separable from marginal or distant recurrence outside the baseline tumor footprint.
- Visualizations should be dense enough for repeated review and conservative enough for clinical-research audit.
- The pipeline should fail loudly on ambiguous geometry, missing human-reviewed labels, and split leakage.

## Report Design

The current report target is static HTML written by `glioma_recurrence.reports`. It is intentionally dependency-light and viewable from the filesystem. Future report enhancements should keep the summary plus available-artifact overlays and add comparative views only when they improve QC.
