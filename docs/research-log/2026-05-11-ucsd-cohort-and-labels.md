# 2026-05-11: UCSD Cohort Accounting, Label Semantics, And Data Strategy

## Summary

UCSD-PTGBM has more clinical acquisition records than the downloaded primary NIfTI image package. This is not a failed partial download. TCIA publishes the NIfTI imaging in multiple access points:

- `Images and Segmentations`: 136 subjects, 184 studies, 4,047 NIfTI images.
- `BraTS-GLI 2024 Test Data`: 42 subjects, 59 studies, 1,322 NIfTI images.
- `Clinical Data`: 178 subjects, 243 timepoints.

The counts reconcile exactly: 136 + 42 = 178 subjects, and 184 + 59 = 243 timepoints.

## Local Audit

Local source root: `/Volumes/0437897195U/UCSD-PTGBM`

- Clinical workbook acquisition IDs: 243
- Downloaded NIfTI timepoint folders: 184
- Clinical subjects: 178
- Downloaded NIfTI subjects: 136
- Missing clinical acquisition folders from the local primary NIfTI download: 59
- Missing clinical subjects from the local primary NIfTI download: 42
- Extra downloaded folders not in clinical workbook: 0
- Downloaded timepoint folders with no NIfTI files: 0

Interpretation: the local folder contains the primary `Images and Segmentations` package, not the separate BraTS-GLI 2024 Test Data package.

## BraTS-GLI Test Data

The separate `BraTS-GLI 2024 Test Data` package is not a different clinical-label system. It is a separate UCSD-PTGBM imaging access point containing NIfTI MRI and segmentations for the 42 subjects / 59 studies that are represented in the clinical workbook but absent from the primary `Images and Segmentations` package.

Clinical-table accounting suggests this extra package would add only a modest number of longitudinal subjects:

- Missing longitudinal subjects represented in the clinical table: about 12.
- Missing recurrence-positive longitudinal subjects: about 11.
- Missing longitudinal clinical-negative subjects: about 1.

Interpretation: downloading the BraTS-GLI package is useful for completeness, but it will not turn UCSD into a large recurrence-prediction cohort.

## Timepoint Definition

A UCSD timepoint is one imaging acquisition/session, represented by IDs like `UCSD-PTGBM-0005_01` or `UCSD-PTGBM-0005_02`. It is not a single MRI sequence. One timepoint can contain T1post, FLAIR, diffusion, perfusion, and segmentation files.

Longitudinal recurrence modeling needs at least two complete timepoints for the same subject:

- Earlier timepoint: baseline prediction input.
- Later timepoint: label context for recurrence or clinical negative control.

## Label Semantics

Tumor-looking segmentation at a later timepoint is not automatically recurrence. UCSD also provides negative case categories:

- Pseudoprogression.
- Radiation necrosis.
- Non-specific post-treatment change.

These should not be used as recurrence targets. The adapter now supports `--include-negative-controls`, which keeps such subjects as controls with empty recurrence labels instead of discarding them or mislabeling them as recurrence.

## Confirmed Recurrence Labels

A confirmed recurrence label needs both:

- A patient/timepoint decision that the later abnormality is true recurrent or progressive tumor, not pseudoprogression, radiation necrosis, or non-specific treatment change.
- A voxelwise mask showing the recurrent tumor region used as the model target.

Confirmation can come from pathology, serial MRI progression, RANO-style clinical adjudication, multidisciplinary review, or strong supportive imaging plus clinical context. Pathology is strongest, but retrospective studies often rely on reviewed longitudinal follow-up. Early post-radiotherapy progression calls require special caution because pseudoprogression is common in that window.

Prediction-time models must not use follow-up MRI. Follow-up scans and masks are label-generation and evaluation context only.

## Clinically Corrected Local Cohort

Using the primary NIfTI package plus clinical and negative-case spreadsheets:

- Subjects with at least two complete MRI+mask timepoints: 37
- Recurrence-positive longitudinal pairs: 22
- Clinical negative control pairs: 15
- Clinical-controls split: 25 train, 6 validation, 6 test

Held-out split mix:

- 7 recurrence-positive cases.
- 5 clinical controls.

## Model Result

On the clinical-controls split:

| Model | Mean AUPRC | Brier | Top 1% Coverage | Top 5% Coverage |
| --- | ---: | ---: | ---: | ---: |
| tumor-distance | 0.2641 | 0.0245 | 0.6583 | 0.9072 |
| voxel-logistic-mri | 0.2027 | 0.0786 | 0.2771 | 0.6457 |

The learned logistic baseline does not beat tumor-distance. The project should not treat the learned model as scientifically interesting yet.

## Data Strategy

UCSD now proves the engineering protocol:

- External NIfTI ingestion into the project manifest.
- Patient-level longitudinal pairing.
- Clinical negative-control handling.
- Baseline MRI plus baseline tumor-mask prediction inputs.
- Follow-up recurrence masks as labels only.
- Mandatory patient-level split accounting and baseline comparisons.

UCSD does not provide enough longitudinal confirmed recurrence labels for a strong scientific claim. The next serious step is to obtain a larger institutional cohort, likely from the Kolling / Royal North Shore collaborators, with deidentified longitudinal MRI, stable subject IDs, recurrence adjudication, and reviewed recurrence masks or enough annotation context to generate them.

Minimum useful request:

- T1c and FLAIR at baseline and follow-up.
- Baseline tumor masks if available.
- Follow-up recurrence/progression masks or source annotations.
- Recurrence status and adjudication method.
- Explicit categories for pseudoprogression, radiation necrosis, non-specific change, and true progression.
- Scan/treatment dates or relative day offsets.
- Optional RT dose and GTV/CTV/PTV masks when available.

Target scale: at least 100 longitudinal subjects, ideally 200 or more, with enough confirmed recurrence-positive labels to test whether learned MRI models beat tumor-distance under patient-level validation.

## Sources

- TCIA UCSD-PTGBM collection page: `https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/`
- UCSD-PTGBM Scientific Data descriptor: `https://www.nature.com/articles/s41597-025-06499-z`
