# Glioma Risk Pipeline V1 Execution Plan

## Goal

Implement a greenfield, agent-legible research pipeline for voxelwise glioma recurrence-risk prediction.

## Work Items

- [x] Add repository knowledge map and design docs.
- [x] Implement manifest, NIfTI IO, affine handling, and CLI stages.
- [x] Implement tumor-distance and voxel-logistic MRI baselines.
- [x] Add optional MONAI/PyTorch U-Net training and inference entry point.
- [x] Add evaluation metrics and QC reports.
- [x] Add unit and smoke tests for geometry, splits, and synthetic overfit.

## Decisions

- Keep MONAI/PyTorch optional so core geometry, baseline, and evaluation tests stay fast.
- Make reviewed recurrence masks mandatory for `make-labels` unless explicitly running draft-only tooling.
- Use baseline T1c space as the canonical planning-space grid.

## Verification

- `uv run --extra dev pytest`: 13 tests passed on 2026-05-09.
