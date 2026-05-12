# Clinical DICOM Requirements V1

## Status

Completed 2026-05-12.

## Goal

Update the repository from public-data-only MRI recurrence prototyping toward the initial clinical study requirements:

- Retrospective recurrence-risk heatmaps only.
- Post-operative, pre-radiotherapy baseline MRI.
- DICOM as clinical source format.
- NIfTI as internal research format.
- T1c and FLAIR minimum; T1/T1c/T2/FLAIR preferred.
- Clinician-curated recurrence endpoint with pseudoprogression excluded.
- Auto-segmentation plus expert review as the preferred spatial-label path.
- Development and validation cohorts of roughly 100-150 patients each.

## Changes

- Added read-only DICOM audit support through `glioma-risk dicom-audit`.
- Added `pydicom` as a base dependency.
- Added DICOM sequence classification for likely `t1`, `t1c`, `t2`, and `flair` series.
- Added pseudonymous patient-key output by default for DICOM inventory CSVs.
- Extended manifest records with optional DICOM/study/scanner/label-source metadata fields.
- Updated architecture, product spec, design doc, README, reliability, security, and quality notes.

## Deliberate Non-Goals

- No DICOM-to-NIfTI conversion orchestration yet.
- No DICOM SEG or Parametric Map export yet.
- No weak-supervision model path yet.
- No boost ROI or clinical planning output.

## Validation

- Fake-DICOM tests cover sequence classification, series grouping, pseudonymous patient keys, PHI-field flags, and full four-sequence availability counts.
- Existing synthetic MRI-only pipeline remains the end-to-end engineering smoke path.
