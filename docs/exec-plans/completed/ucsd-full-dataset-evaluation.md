# UCSD Full Dataset Evaluation

## Status

Completed 2026-05-10.

Superseded 2026-05-11 by `docs/exec-plans/completed/ucsd-clinical-controls-evaluation.md`, after the UCSD clinical workbook and negative-case categories were added to the external dataset folder.

## Goal

Move from download-completion smoke testing to a more useful full local UCSD engineering run with patient-level train/validation/test splits.

## Work Items

- [x] Confirm `/Volumes/0437897195U/UCSD-PTGBM` has no `.partial` or `.aspera-ckpt` files.
- [x] Confirm no clinical spreadsheet was present in the downloaded image tree at the time of this provisional run.
- [x] Replace the adapter's 35/1/1-style split behavior with deterministic patient-level `70/15/15` assignment.
- [x] Rebuild the external UCSD working set with the improved split.
- [x] Run preprocess and label mapping on the rebuilt working set.
- [x] Train and evaluate tumor-distance and voxel-logistic MRI baselines on the improved split.
- [x] Record final metrics and remaining blockers.

## Decisions

- Clinical-table ingestion remains the preferred path. This run was performed before the clinical workbook was available locally.
- The filename-inferred mode remains explicit and provisional.
- The full local image set yields 37 longitudinal filename-inferred pairs from 136 subjects and 184 timepoints.
- The clean derived workspace is `/Volumes/0437897195U/UCSD-PTGBM-pipeline-split70`.

## Results

- Manifest rows: 37
- Split counts: 25 train, 6 validation, 6 test
- Geometry check: 37/37 baseline and follow-up image/mask affines matched before `--assume-baseline-space` label mapping.
- QC overlays: 37
- Prediction maps written during evaluation: 12

### Tumor-Distance Baseline

- Evaluated cases: 12
- Mean voxel AUPRC: 0.26405943371489127
- Mean Brier score: 0.024462189305590226
- Mean Dice at top 1%: 0.1518248174897449
- Mean Dice at top 5%: 0.045894528847047276
- Mean recurrence coverage at top 1%: 0.6583031472609387
- Mean recurrence coverage at top 5%: 0.9072306635107277

### Voxel-Logistic MRI

- Evaluated cases: 12
- Mean voxel AUPRC: 0.20272443976751697
- Mean Brier score: 0.07861894920479055
- Mean Dice at top 1%: 0.12137831934712517
- Mean Dice at top 5%: 0.09814558806315886
- Mean recurrence coverage at top 1%: 0.2770936149812107
- Mean recurrence coverage at top 5%: 0.6457270060680379
- Mean voxel AUPRC delta vs tumor-distance: -0.061334993947374294
- Mean Brier score delta vs tumor-distance: 0.054156759899200324

## Interpretation

The learned voxel-logistic MRI baseline does not beat tumor-distance on this provisional split. The scientifically interesting bar is therefore not met yet. The next milestone is clinical-table-backed label adjudication and cross-validation or repeated split evaluation before tuning learned models.
