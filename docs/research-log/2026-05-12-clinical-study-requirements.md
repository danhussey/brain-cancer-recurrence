# 2026-05-12: Clinical Study Requirements

## Summary

The clinical study direction is now clearer: UCSD remains useful for public engineering validation, but the real project should be prepared for an institutional clinical-data workflow.

## Requirements Captured

- V1 output remains retrospective voxelwise recurrence-risk heatmaps.
- Candidate boost regions and clinical planning workflow outputs are out of scope until prediction performance is demonstrated.
- Baseline MRI should be post-operative and pre-radiotherapy, ideally as close as possible to RT planning.
- Pre-operative MRI is deferred until the basic post-op/pre-RT model works.
- Minimum MRI inputs are T1c and FLAIR.
- T1, T1c, T2, and FLAIR are preferred when using BraTS-style segmentation pipelines.
- Clinical source data is DICOM.
- NIfTI is the internal research format because most registration, segmentation, and model tooling expects it.
- The practical data flow is `DICOM -> NIfTI -> model/evaluation -> NIfTI -> DICOM`.
- Recurrence endpoints are clinician-curated retrospectively with pseudoprogression excluded.
- Spatial recurrence labels should favor auto-segmentation plus expert review over purely expert-drawn manual contours.
- Weak recurrence/no-recurrence supervision is worth exploring separately.
- Initial development cohort target is about 100-150 patients.
- Validation cohort target is about 100-150 patients.
- Initial data will be single-institution, with later validation across a second institution and scanner upgrades.
- Foundation models are a candidate route for scanner/protocol robustness after simple baselines are established.

## Implementation Implication

The repository should keep its NIfTI research core, but expose DICOM-aware intake and future DICOM-aware export boundaries. The first implementation step is read-only DICOM audit and sequence inventory; full DICOM SEG or Parametric Map export should be added only after the output object contract is stable.
