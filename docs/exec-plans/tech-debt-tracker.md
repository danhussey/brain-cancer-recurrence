# Technical Debt Tracker

Track known debt as small, agent-actionable items.

| ID | Area | Debt | Status |
| --- | --- | --- | --- |
| TD-001 | UCSD fixtures | Add real-layout UCSD metadata and file-name fixtures for adapter regression tests. | Open |
| TD-002 | Registration | Replace V1 resampling-only preprocessing with explicit registration transform records. | Open |
| TD-003 | Reports | Add slice browsing, overlay opacity controls, and visual regression checks for QC HTML. Slice browsing and opacity controls are done; visual regression remains. | Partial |
| TD-004 | Deep learning | Add `deep` extra integration smoke test behind an optional marker. | Open |
| TD-005 | External validation | Add RHUH-GBM adapter or documented external-validation workflow. | Open |
| TD-006 | UCSD clinical metadata | Locate and ingest UCSD clinical table so filename-inferred labels can be replaced with adjudicated recurrence labels. | Done |
| TD-007 | Evaluation | Add repeated split or cross-validation runner; one deterministic split is not enough for scientific claims. | Open |
| TD-008 | DICOM conversion | Add controlled DICOM-to-NIfTI conversion orchestration with provenance, geometry checks, and de-identification verification. | Open |
| TD-009 | DICOM export | Add standards-aware recurrence-risk export, likely DICOM Parametric Map or DICOM SEG depending final output contract. | Open |
| TD-010 | Weak supervision | Add a patient/timepoint recurrence/no-recurrence training path that does not require voxelwise recurrence masks. | Open |
