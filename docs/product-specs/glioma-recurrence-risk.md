# Glioma Recurrence Risk Pipeline Spec

## Summary

Build a retrospective research pipeline that predicts future glioma recurrence locations from post-operative, pre-radiotherapy MRI. UCSD-PTGBM is the public-data engineering workflow; the intended study workflow starts from clinical DICOM and uses NIfTI as the internal research format.

## V1 Scope

- Clinical-data flow: `DICOM -> NIfTI -> model/evaluation -> NIfTI -> future DICOM export`.
- Inputs: `patients.csv`, post-op/pre-RT baseline T1c/FLAIR MRI, baseline tumor masks, clinician-curated recurrence endpoint labels, and reviewed recurrence masks when spatial labels are available.
- Preferred MRI channels: T1, T1c, T2, and FLAIR. Minimum V1 MRI channels: T1c and FLAIR.
- Outputs: baseline-space NIfTI derivatives and `recurrence_risk.nii.gz`; future clinical-system handoff should export DICOM SEG or Parametric Map after the research output contract is stable.
- Data-readiness tooling: read-only DICOM header audit for sequence availability, scanner metadata, and PHI-risk fields.
- Models: tumor-distance baseline, voxel-sampled MRI logistic baseline, optional MONAI/PyTorch U-Net.
- Evaluation: voxel AUPRC, Brier score, Dice at fixed predicted volumes, recurrence coverage by top-risk volume, calibration, baseline comparison, and recurrence stratification inside versus outside the baseline tumor footprint.
- Target scale: about 100-150 development patients and a similarly sized validation cohort, with scanner upgrades and a second institution treated as robustness tests.
- Label strategy: use curated recurrence outcomes with pseudoprogression excluded; explore both weak recurrence/no-recurrence supervision and auto-segmentation plus expert review for spatial labels.

## Out Of Scope

- Clinical dose recommendations.
- Candidate boost-region recommendations.
- Treatment-plan optimization.
- Prospective clinical decision support.
- Feeding follow-up imaging into prediction-time inputs.
- Treating expert-drawn manual contours as the only acceptable spatial-label source.
- Treating high risk at obvious residual tumor as sufficient evidence that the model can improve future treatment targeting.
