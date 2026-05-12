# Glioma Recurrence Risk Pipeline

## Scope

Build a retrospective research pipeline that predicts future glioma recurrence locations from post-operative, pre-radiotherapy MRI. The pipeline uses a baseline tumor mask as the tumor-location feature. The V1 output is a voxelwise recurrence-risk heatmap in baseline space.

Residual tumor at the post-op/pre-RT baseline is expected to be high risk. This is a valid baseline signal, but the harder scientific question is whether models can predict marginal or distant recurrence outside the obvious baseline tumor footprint. Reports and evaluation should preserve that distinction.

The pipeline is not a clinical dose recommendation, boost-region generator, treatment plan optimizer, or prospective decision-support system.

## Inputs

- Clinical DICOM exports are the expected source format for the institutional dataset.
- NIfTI MRI/masks are the internal research format after conversion.
- `patients.csv` manifest with patient IDs, baseline scan dates, MRI series UIDs, recurrence scan date, adjudication, reviewed recurrence-mask path, and split assignment.
- Minimum baseline channels: post-op/pre-RT T1c and FLAIR.
- Preferred baseline channels: T1, T1c, T2, and FLAIR, especially for BraTS-style segmentation pipelines.
- Baseline tumor masks may come from segmentation pipelines, then expert review.
- Recurrence labels may be weak patient/timepoint recurrence labels or spatial recurrence masks. Spatial training and voxelwise evaluation require reviewed recurrence masks.

## Outputs

Derived NIfTI files:

- `baseline_t1c.nii.gz`
- `baseline_flair.nii.gz`
- `baseline_tumor_mask.nii.gz`
- `recurrence_mask_on_baseline.nii.gz`
- `brain_mask.nii.gz`
- `recurrence_risk.nii.gz`

Future DICOM handoff should export voxelwise risk maps as standards-aware DICOM SEG or Parametric Map objects after the research output contract is stable.

## Stage Semantics

- `dicom-audit`: read DICOM headers without pixel data, classify MRI sequences, count T1c/FLAIR and four-sequence availability, summarize scanner metadata, and flag likely PHI-bearing fields.
- `preprocess`: register or resample FLAIR and baseline tumor mask to baseline T1c, normalize MRI, and create brain masks.
- `make-labels`: require human-reviewed recurrence masks, then map masks to baseline space by SimpleITK MRI-to-MRI registration unless an explicit affine or assume-aligned fallback is requested.
- `train`: train simple baselines first; MRI-only work must compare learned models against `tumor-distance`. Optional MONAI/PyTorch 3D U-Net training is available behind the `deep` extra.
- `evaluate`: compute patient-level metrics, calibration, visual QC, recurrence inside/outside baseline tumor summaries, and comparison against simple baselines.
- `predict`: produce `recurrence_risk.nii.gz` and a research-only overlay report.
- Every CLI stage emits structured observability artifacts unless `--no-observability` is passed.

## Safety Constraints

- Follow-up scans are never prediction inputs.
- Recurrence masks are labels only.
- Follow-up T1c is used only as the moving-image reference for label registration.
- Recurrences inside the early post-RT pseudoprogression window are excluded or flagged unless adjudication indicates clinical or histologic confirmation.
- Every case requires QC overlays for T1c, FLAIR, baseline tumor mask, recurrence mask mapped to baseline, and final prediction.
- QC reports should expose recurrence inside versus outside the baseline tumor footprint so obvious residual-tumor recurrence does not get confused with the harder treatment-improvement target.
- The model must beat the `tumor-distance` baseline in cross-validation before it is scientifically interesting.

## Study Defaults

- Development cohort target: 100-150 patients.
- Validation cohort target: 100-150 patients.
- Initial data path: single institution.
- Robustness path: second institution plus scanner upgrades over time.
- Foundation-model motivation: improve robustness to scanner/protocol variation after simple baselines are established.

## References

- RANO 2.0 concepts for glioma progression adjudication: https://pmc.ncbi.nlm.nih.gov/articles/PMC10860967/
- BraTS-style glioma segmentation review conventions: https://www.med.upenn.edu/cbica/brats2018/tasks.html
- MONAI primitives for medical-imaging deep learning: https://docs.monai.io/en/latest/
- FDA Good Machine Learning Practice principles for any later regulated direction: https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles
- UCSD-PTGBM public dataset: https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/
- UCSD-PTGBM data descriptor: https://www.nature.com/articles/s41597-025-06499-z
