# Glioma Recurrence Risk Pipeline

## Scope

Build a retrospective research pipeline that predicts future glioma recurrence locations from post-operative pre-radiotherapy T1c and FLAIR MRI plus RTDOSE. The V1 output is a calibrated voxelwise recurrence-risk heatmap in baseline planning space.

The pipeline is not a clinical dose recommendation, treatment plan optimizer, or prospective decision-support system.

## Inputs

- `patients.csv` manifest with patient IDs, baseline scan dates, MRI series UIDs, RTDOSE SOP Instance UID, recurrence scan date, adjudication, reviewed recurrence-mask path, and split assignment.
- DICOM MRI and RTDOSE files for ingestion.
- Reviewed recurrence masks for label creation.

## Outputs

Derived NIfTI files:

- `baseline_t1c.nii.gz`
- `baseline_flair.nii.gz`
- `dose_gy_on_baseline.nii.gz`
- `recurrence_mask_on_baseline.nii.gz`
- `brain_mask.nii.gz`
- `recurrence_risk.nii.gz`

## Stage Semantics

- `ingest`: de-identify metadata in audit outputs, validate DICOM metadata, convert MRI, and scale RTDOSE into Gy.
- `preprocess`: register or resample FLAIR and RTDOSE to baseline T1c, normalize MRI, and create brain masks.
- `make-labels`: require human-reviewed recurrence masks, then map masks to baseline space.
- `train`: train simple baselines first; optional MONAI/PyTorch 3D U-Net training is available behind the `deep` extra.
- `evaluate`: compute patient-level metrics, calibration, visual QC, and comparison against simple baselines.
- `predict`: produce `recurrence_risk.nii.gz` and a research-only overlay report.

## Safety Constraints

- Follow-up scans are never prediction inputs.
- Recurrence masks are labels only.
- Recurrences inside the early post-RT pseudoprogression window are excluded or flagged unless adjudication indicates clinical or histologic confirmation.
- Every case requires QC overlays for T1c, FLAIR, dose, recurrence mask mapped to baseline, and final prediction.
- The model must beat distance/dose baselines in cross-validation before it is scientifically interesting.

## References

- RANO 2.0 concepts for glioma progression adjudication: https://pmc.ncbi.nlm.nih.gov/articles/PMC10860967/
- DICOM RT Dose object model and patient-coordinate transforms: https://dicom.innolitics.com/ciods/rt-dose
- BraTS-style glioma segmentation review conventions: https://www.med.upenn.edu/cbica/brats2018/tasks.html
- MONAI primitives for medical-imaging deep learning: https://docs.monai.io/en/latest/
- FDA Good Machine Learning Practice principles for any later regulated direction: https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles

