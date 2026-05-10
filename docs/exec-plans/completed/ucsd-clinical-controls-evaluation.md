# UCSD Clinical Controls Evaluation

## Status

Completed 2026-05-11.

## Goal

Use UCSD clinical metadata and negative-case categories to distinguish recurrence-positive follow-up labels from pseudoprogression, radiation necrosis, and non-specific clinical negative controls.

## Work Items

- [x] Parse the UCSD clinical workbook keyed by acquisition ID, such as `UCSD-PTGBM-0005_02`.
- [x] Parse `details_of_negative_cases_TCIA.xlsx`.
- [x] Exclude clinical negative categories from recurrence-positive labels.
- [x] Add explicit `--include-negative-controls` support that retains clinical negatives as empty recurrence-label controls.
- [x] Build a clean external clinical-controls workspace.
- [x] Verify all baseline/follow-up shapes and affines match before using `--assume-baseline-space`.
- [x] Train and evaluate tumor-distance and voxel-logistic MRI baselines.

## Cohort Accounting

- Clinical workbook subjects: 178
- Clinical workbook acquisition timepoints: 243
- Downloaded complete NIfTI subjects: 136
- Downloaded complete NIfTI timepoints: 184
- Subjects with at least two complete MRI+mask timepoints: 37
- Recurrence-positive longitudinal pairs: 22
- Clinical negative control pairs: 15
- Negative control categories: 5 pseudoprogression, 7 non-specific, 3 radiation necrosis

## Workspace

- Root: `/Volumes/0437897195U/UCSD-PTGBM-pipeline-clinical-controls`
- Manifest rows: 37
- Split counts: 25 train, 6 validation, 6 test
- Split label mix: train 15 positive / 10 control; validation 4 positive / 2 control; test 3 positive / 3 control
- QC overlays: 37
- Prediction maps written during evaluation: 12

## Results

### Tumor-Distance Baseline

- Evaluated cases: 12
- Positive held-out cases: 7
- Control held-out cases: 5
- Mean voxel AUPRC: 0.26405943371489127
- Mean Brier score: 0.024462189305590226
- Mean Dice at top 1%: 0.1518248174897449
- Mean Dice at top 5%: 0.045894528847047276
- Mean recurrence coverage at top 1%: 0.6583031472609387
- Mean recurrence coverage at top 5%: 0.9072306635107277

### Voxel-Logistic MRI

- Evaluated cases: 12
- Positive held-out cases: 7
- Control held-out cases: 5
- Mean voxel AUPRC: 0.20272443976751697
- Mean Brier score: 0.07861894920479055
- Mean Dice at top 1%: 0.12137831934712517
- Mean Dice at top 5%: 0.09814558806315886
- Mean recurrence coverage at top 1%: 0.2770936149812107
- Mean recurrence coverage at top 5%: 0.6457270060680379
- Mean voxel AUPRC delta vs tumor-distance: -0.061334993947374294
- Mean Brier score delta vs tumor-distance: 0.054156759899200324

## Interpretation

The clinical metadata does not reduce the usable longitudinal cohort to 22 patients; it identifies 22 recurrence-positive pairs and 15 clinically negative controls among the 37 subjects with at least two complete MRI+mask timepoints. The learned voxel-logistic MRI baseline still does not beat tumor-distance, so the scientific-interest threshold remains unmet. The next project step is repeated split or cross-validation evaluation, followed by model improvements only if they beat the distance baseline.
