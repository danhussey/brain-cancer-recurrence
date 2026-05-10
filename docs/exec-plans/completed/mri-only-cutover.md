# MRI-Only Cutover Execution Plan

## Goal

Cut the pipeline over to UCSD-style MRI-only recurrence modeling without a backward-compatibility layer for dose-aware workflows.

## Work Items

- [x] Remove RTDOSE, DICOM ingest, CFB pilot, input-mode, and prescription-dose public contracts.
- [x] Make baseline T1c, FLAIR, and baseline tumor mask the required prediction-time inputs.
- [x] Keep follow-up T1c only as label-registration context and recurrence masks as labels.
- [x] Keep `tumor-distance` as the default simple baseline and `voxel-logistic-mri` as the learned baseline.
- [x] Update docs, synthetic data, UCSD adapter tests, and quality/reliability notes.

## Decisions

- UCSD-PTGBM is the primary public-data workflow.
- The stable derived case contract is now MRI-only: no dose map derivative is part of the required layout.
- A learned model is only interesting if it beats the `tumor-distance` baseline under patient-level evaluation.

## Verification

- `uv run --extra dev pytest`: 18 tests passed on 2026-05-10.
- `uv run --extra dev python scripts/validate_knowledge_store.py`: passed on 2026-05-10.
- `git diff --check`: passed on 2026-05-10.
