# Glioma Recurrence Risk Pipeline

## Scope

Build a retrospective research pipeline that predicts future glioma recurrence locations from post-operative T1c and FLAIR MRI. The pipeline uses a baseline tumor mask as the tumor-location feature. The V1 output is a calibrated voxelwise recurrence-risk heatmap in baseline space.

The pipeline is not a clinical dose recommendation, treatment plan optimizer, or prospective decision-support system.

## Inputs

- `patients.csv` manifest with patient IDs, baseline scan dates, MRI series UIDs, recurrence scan date, adjudication, reviewed recurrence-mask path, and split assignment.
- Copied NIfTI MRI/masks from dataset adapters.
- Reviewed recurrence masks for label creation.

## Outputs

Derived NIfTI files:

- `baseline_t1c.nii.gz`
- `baseline_flair.nii.gz`
- `baseline_tumor_mask.nii.gz`
- `recurrence_mask_on_baseline.nii.gz`
- `brain_mask.nii.gz`
- `recurrence_risk.nii.gz`

## Stage Semantics

- `preprocess`: register or resample FLAIR and baseline tumor mask to baseline T1c, normalize MRI, and create brain masks.
- `make-labels`: require human-reviewed recurrence masks, then map masks to baseline space by SimpleITK MRI-to-MRI registration unless an explicit affine or assume-aligned fallback is requested.
- `train`: train simple baselines first; MRI-only work must compare learned models against `tumor-distance`. Optional MONAI/PyTorch 3D U-Net training is available behind the `deep` extra.
- `evaluate`: compute patient-level metrics, calibration, visual QC, and comparison against simple baselines.
- `predict`: produce `recurrence_risk.nii.gz` and a research-only overlay report.
- Every CLI stage emits structured observability artifacts unless `--no-observability` is passed.

## Safety Constraints

- Follow-up scans are never prediction inputs.
- Recurrence masks are labels only.
- Follow-up T1c is used only as the moving-image reference for label registration.
- Recurrences inside the early post-RT pseudoprogression window are excluded or flagged unless adjudication indicates clinical or histologic confirmation.
- Every case requires QC overlays for T1c, FLAIR, baseline tumor mask, recurrence mask mapped to baseline, and final prediction.
- The model must beat the `tumor-distance` baseline in cross-validation before it is scientifically interesting.

## References

- RANO 2.0 concepts for glioma progression adjudication: https://pmc.ncbi.nlm.nih.gov/articles/PMC10860967/
- BraTS-style glioma segmentation review conventions: https://www.med.upenn.edu/cbica/brats2018/tasks.html
- MONAI primitives for medical-imaging deep learning: https://docs.monai.io/en/latest/
- FDA Good Machine Learning Practice principles for any later regulated direction: https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles
- UCSD-PTGBM public dataset: https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/
- UCSD-PTGBM data descriptor: https://www.nature.com/articles/s41597-025-06499-z
