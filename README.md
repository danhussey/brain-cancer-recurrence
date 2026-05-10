# Glioma Recurrence Risk Pipeline

Research pipeline for predicting future glioma recurrence locations from post-operative MRI. UCSD-PTGBM is the primary public-data workflow.

V1 outputs a calibrated voxelwise recurrence-risk heatmap in the patient baseline space. It is not a clinical dose recommendation, treatment-planning system, or medical device.

## Key Terms

- **Glioma / GBM**: A brain tumor type. GBM means glioblastoma, an aggressive glioma.
- **Patient / case**: One subject in the dataset.
- **Baseline / t0**: The earlier post-treatment MRI timepoint used as prediction input.
- **Follow-up / t1 / t2**: Later imaging timepoints used to determine whether and where recurrence happened.
- **Voxel**: A 3D pixel in an MRI, mask, or risk map.
- **NIfTI**: Common research imaging file format, usually `.nii` or `.nii.gz`.
- **T1c / T1gd**: Contrast-enhanced T1-weighted MRI. This is the main anatomy/tumor channel.
- **FLAIR**: MRI sequence that highlights edema and abnormal fluid-like tissue signal.
- **Baseline tumor mask**: Tumor segmentation at the baseline timepoint. This is a prediction-time location feature.
- **Recurrence mask**: Human-reviewed mask of where tumor recurrence later occurred, mapped back to baseline space.
- **Brain mask**: Binary mask limiting training/evaluation to brain voxels.
- **Registration**: Aligning images from different scans into the same coordinate space.
- **Resampling**: Regridding one image onto another image's voxel grid after alignment.
- **Affine**: Matrix that maps voxel indices to real patient/world coordinates.
- **Risk heatmap**: Voxelwise model output from 0 to 1 estimating recurrence risk.
- **Calibration**: Whether predicted risks match observed recurrence frequencies.
- **AUPRC**: Area under the precision-recall curve; useful when recurrence voxels are rare.
- **Dice**: Spatial overlap score between a predicted region and a label mask.
- **Brier score**: Mean squared error of predicted probabilities; lower is better.
- **Split**: Train, validation, or test assignment at the patient level.
- **Leakage**: Accidental sharing of the same patient across train/validation/test.
- **Pseudoprogression**: Early post-radiotherapy imaging change that can mimic recurrence.
- **QC overlay**: Visual report showing anatomy, masks, and prediction for human review.

## Commands

```sh
glioma-risk preprocess --manifest patients.csv --derived-root derived
glioma-risk make-labels --manifest patients.csv --derived-root derived
glioma-risk train --manifest patients.csv --derived-root derived --model tumor-distance --output models/tumor-distance.json
glioma-risk train --manifest patients.csv --derived-root derived --model voxel-logistic-mri --output models/voxel-logistic-mri.json
glioma-risk evaluate --manifest patients.csv --derived-root derived --model-path models/voxel-logistic-mri.json --output reports/eval.json
glioma-risk predict --case-dir derived/P001 --model-path models/voxel-logistic-mri.json --output-dir derived/P001
```

Use `glioma-risk <stage> --help` for stage-specific options.

The default model is `tumor-distance`, the required simple baseline. `--model voxel-logistic-mri` trains the first learned MRI-only baseline. `--model unet` trains an optional MONAI/PyTorch 3D U-Net checkpoint when the `deep` extra is installed.

## Observability

Every CLI stage writes structured run artifacts by default:

- `events.jsonl`: timestamped stage, case, metric, and artifact events.
- `summary.json`: final status, duration, case status, event counts, command args, and output artifacts.

For manifest stages, artifacts default to `<derived-root>/../observability/<run-id>/`. For `predict`, they default beside the output directory. Override or disable this per command:

```sh
glioma-risk train --manifest patients.csv --derived-root derived --output models/tumor-distance.json --observability-root runs
glioma-risk evaluate --manifest patients.csv --derived-root derived --model-path models/voxel-logistic-mri.json --output reports/eval.json --no-observability
```

## Synthetic Smoke Dataset

Use the synthetic generator for engineering checks when no real dataset is available:

```sh
uv run --extra dev python scripts/generate_synthetic_dataset.py --output-root /private/tmp/glioma-smoke --n-patients 2 --shape 12,12,12
uv run --extra dev python -m glioma_recurrence preprocess --manifest /private/tmp/glioma-smoke/patients.csv --derived-root /private/tmp/glioma-smoke/derived
uv run --extra dev python -m glioma_recurrence make-labels --manifest /private/tmp/glioma-smoke/patients.csv --derived-root /private/tmp/glioma-smoke/derived --assume-baseline-space
uv run --extra dev python -m glioma_recurrence train --manifest /private/tmp/glioma-smoke/patients.csv --derived-root /private/tmp/glioma-smoke/derived --model tumor-distance --output /private/tmp/glioma-smoke/models/tumor-distance.json
uv run --extra dev python -m glioma_recurrence train --manifest /private/tmp/glioma-smoke/patients.csv --derived-root /private/tmp/glioma-smoke/derived --model voxel-logistic-mri --output /private/tmp/glioma-smoke/models/voxel-logistic-mri.json
uv run --extra dev python -m glioma_recurrence evaluate --manifest /private/tmp/glioma-smoke/patients.csv --derived-root /private/tmp/glioma-smoke/derived --model-path /private/tmp/glioma-smoke/models/voxel-logistic-mri.json --output /private/tmp/glioma-smoke/reports/eval.json --splits validation --write-predictions
```

Synthetic data is only for pipeline validation. It is not scientifically meaningful.

## UCSD-PTGBM Workflow

Download UCSD-PTGBM images/segmentations and clinical data from TCIA onto external storage, then prepare a copied working set:

```sh
uv run --extra dev python scripts/prepare_ucsd_ptgbm_dataset.py --source-root /Volumes/External/UCSD-PTGBM --clinical-table /Volumes/External/UCSD-PTGBM/clinical.xlsx --output-root /Volumes/External/UCSD-PTGBM-pipeline --max-subjects 20
```

The adapter selects subjects with at least two complete MRI+mask timepoints, uses the earliest complete post-treatment timepoint as baseline, and uses the earliest later residual/recurrent tumor timepoint as the recurrence label. It copies baseline T1c, baseline FLAIR, `baseline_tumor_mask.nii.gz`, follow-up T1c label-reference images, and reviewed follow-up tumor masks into the working set.

If images arrive before the clinical workbook, make a provisional filename-only working set explicitly:

```sh
uv run --extra dev python scripts/prepare_ucsd_ptgbm_dataset.py --source-root /Volumes/External/UCSD-PTGBM --output-root /Volumes/External/UCSD-PTGBM-pipeline --allow-imaging-only-labels --max-subjects 20
```

This mode is useful for pipeline bring-up only. It pairs the earliest complete image timepoint with the earliest later complete tumor segmentation and marks labels as `imaging_followup_segmentation_present`, without clinical progression adjudication.

Default `make-labels` registration is SimpleITK MRI-to-MRI registration. Use `--registration-mode affine` or `--assume-baseline-space` only when that geometry fallback has been checked.

## Manifest Columns

Required:

- `patient_id`
- `baseline_scan_date`
- `baseline_t1c_series_uid`
- `baseline_flair_series_uid`
- `recurrence_scan_date`
- `recurrence_adjudication`
- `reviewed_recurrence_mask_path`
- `split`

Optional but recommended:

- `reviewed_recurrence_reference_image_path`: follow-up T1c used only for label registration.
- `source_dataset`
- `baseline_timepoint_id`
- `recurrence_timepoint_id`
- `radiotherapy_end_date`

## Derived Files

Each case directory stores:

- `baseline_t1c.nii.gz`
- `baseline_flair.nii.gz`
- `baseline_tumor_mask.nii.gz`
- `recurrence_mask_on_baseline.nii.gz`
- `brain_mask.nii.gz`
- `recurrence_risk.nii.gz`

## Development

```sh
uv run --extra dev pytest
uv run --extra dev python scripts/validate_knowledge_store.py
```
