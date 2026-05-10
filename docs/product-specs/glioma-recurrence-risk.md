# Glioma Recurrence Risk Pipeline Spec

## Summary

Build a retrospective research pipeline that predicts future glioma recurrence locations from post-operative MRI. UCSD-PTGBM is the primary public-data workflow.

## V1 Scope

- Inputs: `patients.csv`, baseline T1c/FLAIR MRI, baseline tumor masks, and reviewed recurrence masks.
- Outputs: baseline-space NIfTI derivatives and `recurrence_risk.nii.gz`.
- Models: tumor-distance baseline, voxel-sampled MRI logistic baseline, optional MONAI/PyTorch U-Net.
- Evaluation: voxel AUPRC, Brier score, Dice at fixed predicted volumes, recurrence coverage by top-risk volume, calibration, and baseline comparison.

## Out Of Scope

- Clinical dose recommendations.
- Treatment-plan optimization.
- Prospective clinical decision support.
- Feeding follow-up imaging into prediction-time inputs.
