# Architecture

The pipeline is a Python package with a thin CLI. It is structured so that medical-imaging IO is isolated from model and metric logic, making the high-risk geometry paths easy to test.

## Layers

- `schema.py`: manifest contracts, patient records, split validation, recurrence adjudication flags.
- `dicom.py`: read-only DICOM header audit, MRI sequence classification, and intake readiness summaries.
- `geometry.py`: affine math, patient-coordinate transforms, resampling, and mask round-trip helpers.
- `nifti.py`: NIfTI read/write boundary.
- `preprocess.py`: MRI normalization, brain masks, baseline tumor mask preparation, and resampling wrappers.
- `labels.py`: reviewed recurrence-mask ingestion and baseline-space mapping by SimpleITK, affine, or explicitly assumed-aligned registration.
- `models.py`: tumor-distance and voxel-logistic MRI-only baselines.
- `deep.py`: optional MONAI/PyTorch 3D U-Net entry points.
- `evaluation.py`: patient-level metrics, calibration, and baseline comparison helpers.
- `observability.py`: structured JSONL events, run summaries, case timings, and artifact tracking for every CLI stage.
- `reports.py`: mandatory case-level QC overlays and research-only report text.
- `cli.py`: stage orchestration for `dicom-audit`, `preprocess`, `make-labels`, `train`, `evaluate`, and `predict`.

## Data Contract

Clinical source data is expected to start as DICOM. The research core uses NIfTI derivatives because model, registration, and segmentation tooling operate on NIfTI. The intended clinical-data flow is:

```text
DICOM -> local DICOM audit/series selection -> NIfTI derivatives -> model/evaluation -> NIfTI risk maps -> future DICOM export
```

The current repository implements read-only DICOM audit and NIfTI research derivatives. DICOM export of risk maps is intentionally not implemented yet; it should be added as a standards-aware DICOM SEG or Parametric Map export, not as an ad hoc screenshot.

The manifest is `patients.csv`. Required V1 prediction inputs are post-operative, pre-radiotherapy baseline T1c and FLAIR. T1 and T2 are optional but preferred for BraTS-style or foundation-model paths. Derived case data lives under `<derived-root>/<patient_id>/` using fixed names:

- `baseline_t1c.nii.gz`
- `baseline_flair.nii.gz`
- `baseline_tumor_mask.nii.gz`
- `recurrence_mask_on_baseline.nii.gz`
- `brain_mask.nii.gz`
- `recurrence_risk.nii.gz`

## Safety Boundaries

This is a retrospective research pipeline. It outputs voxelwise recurrence-risk heatmaps in baseline space and must not present dose recommendations or boost-region recommendations. The minimum input contract is post-op/pre-RT baseline T1c, baseline FLAIR, and baseline tumor mask. Follow-up scans and reviewed recurrence masks are only used to build labels and evaluate predictions.
