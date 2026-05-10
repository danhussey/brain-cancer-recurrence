# UCSD Partial Dataset Start

## Status

Completed 2026-05-10.

## Goal

Enable safe pipeline bring-up while UCSD-PTGBM images are still downloading and before the clinical workbook is available locally.

## Work Items

- [x] Inspect `/Volumes/0437897195U/UCSD-PTGBM` without copying it into the repository.
- [x] Count complete T1post + FLAIR + tumor-mask timepoints and longitudinal subjects.
- [x] Add explicit `--allow-imaging-only-labels` mode to the UCSD adapter.
- [x] Keep imaging-only labels visibly provisional with `imaging_followup_segmentation_present`.
- [x] Document the provisional workflow in `README.md`.
- [x] Add a read-only UCSD audit command for download completeness checks.
- [x] Build an external derived working set from the partial download.
- [x] Run minimal preprocess / label / train / evaluate smoke commands on the derived working set.

## Decisions

- The adapter still requires the clinical table by default.
- Filename-only pairing is opt-in because it lacks clinical progression adjudication.
- No source dataset files are copied into the repo; derived working data remains on external storage.
- The starter run used `--assume-baseline-space` only after all prepared baseline/follow-up images and masks had matching shape and affine.
- `scripts/audit_ucsd_ptgbm_dataset.py` is read-only and can be rerun while the download is still active.

## Starter Run

- Source root: `/Volumes/0437897195U/UCSD-PTGBM`
- Working root: `/Volumes/0437897195U/UCSD-PTGBM-pipeline`
- Selected subjects: 13
- Split counts: 11 train, 1 validation, 1 test
- Distance baseline mean voxel AUPRC: 0.16484824549588448
- Voxel-logistic MRI mean voxel AUPRC: 0.19179623009661584
- Voxel-logistic MRI mean Brier score was worse than distance baseline in this tiny provisional split, so this is only an engineering smoke result.
