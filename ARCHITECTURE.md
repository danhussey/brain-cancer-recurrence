# Architecture

The pipeline is a Python package with a thin CLI. It is structured so that medical-imaging IO is isolated from model and metric logic, making the high-risk geometry paths easy to test.

## Layers

- `schema.py`: manifest contracts, patient records, split validation, recurrence adjudication flags.
- `geometry.py`: affine math, DICOM patient-coordinate transforms, resampling, and mask round-trip helpers.
- `dicom.py`: DICOM discovery, metadata validation, MR series conversion, and RTDOSE scaling.
- `nifti.py`: NIfTI read/write boundary.
- `preprocess.py`: MRI normalization, brain masks, dose channel preparation, and resampling wrappers.
- `labels.py`: reviewed recurrence-mask ingestion and baseline-space mapping.
- `models.py`: simple dose/distance and voxel-logistic baselines.
- `deep.py`: optional MONAI/PyTorch 3D U-Net entry points.
- `evaluation.py`: patient-level metrics, calibration, and baseline comparison helpers.
- `reports.py`: mandatory case-level QC overlays and research-only report text.
- `cli.py`: stage orchestration for `ingest`, `preprocess`, `make-labels`, `train`, `evaluate`, and `predict`.

## Data Contract

The manifest is `patients.csv`. Derived case data lives under `<derived-root>/<patient_id>/` using fixed names:

- `baseline_t1c.nii.gz`
- `baseline_flair.nii.gz`
- `dose_gy_on_baseline.nii.gz`
- `recurrence_mask_on_baseline.nii.gz`
- `brain_mask.nii.gz`
- `recurrence_risk.nii.gz`

## Safety Boundaries

This is a retrospective research pipeline. It outputs calibrated voxelwise recurrence-risk heatmaps in baseline planning space and must not present dose recommendations. The model input contract is baseline T1c, baseline FLAIR, and RT dose. Follow-up scans and reviewed recurrence masks are only used to build labels and evaluate predictions.

