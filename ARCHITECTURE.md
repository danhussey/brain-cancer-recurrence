# Architecture

The pipeline is a Python package with a thin CLI. It is structured so that medical-imaging IO is isolated from model and metric logic, making the high-risk geometry paths easy to test.

## Layers

- `schema.py`: manifest contracts, patient records, split validation, recurrence adjudication flags.
- `geometry.py`: affine math, patient-coordinate transforms, resampling, and mask round-trip helpers.
- `nifti.py`: NIfTI read/write boundary.
- `preprocess.py`: MRI normalization, brain masks, baseline tumor mask preparation, and resampling wrappers.
- `labels.py`: reviewed recurrence-mask ingestion and baseline-space mapping by SimpleITK, affine, or explicitly assumed-aligned registration.
- `models.py`: tumor-distance and voxel-logistic MRI-only baselines.
- `deep.py`: optional MONAI/PyTorch 3D U-Net entry points.
- `evaluation.py`: patient-level metrics, calibration, and baseline comparison helpers.
- `observability.py`: structured JSONL events, run summaries, case timings, and artifact tracking for every CLI stage.
- `reports.py`: mandatory case-level QC overlays and research-only report text.
- `cli.py`: stage orchestration for `preprocess`, `make-labels`, `train`, `evaluate`, and `predict`.

## Data Contract

The manifest is `patients.csv`. Derived case data lives under `<derived-root>/<patient_id>/` using fixed names:

- `baseline_t1c.nii.gz`
- `baseline_flair.nii.gz`
- `baseline_tumor_mask.nii.gz`
- `recurrence_mask_on_baseline.nii.gz`
- `brain_mask.nii.gz`
- `recurrence_risk.nii.gz`

## Safety Boundaries

This is a retrospective research pipeline. It outputs voxelwise recurrence-risk heatmaps in baseline space and must not present dose recommendations. The input contract is baseline T1c, baseline FLAIR, and baseline tumor mask. Follow-up scans and reviewed recurrence masks are only used to build labels and evaluate predictions. Evaluation reports include calibration summaries, but the pipeline does not apply a separate calibration stage to model outputs.
