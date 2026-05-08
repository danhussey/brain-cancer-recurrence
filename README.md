# Glioma Recurrence Risk Pipeline

Research pipeline for predicting future glioma recurrence locations from post-operative pre-radiotherapy MRI plus radiotherapy dose maps.

V1 outputs a calibrated voxelwise recurrence-risk heatmap in the patient baseline planning space. It is not a clinical dose recommendation, treatment-planning system, or medical device.

## Commands

```sh
glioma-risk ingest --manifest patients.csv --dicom-root dicom --derived-root derived
glioma-risk preprocess --manifest patients.csv --derived-root derived --prescription-dose-gy 60
glioma-risk make-labels --manifest patients.csv --derived-root derived
glioma-risk train --manifest patients.csv --derived-root derived --model dose-distance --output models/dose-distance.json
glioma-risk evaluate --manifest patients.csv --derived-root derived --model-path models/dose-distance.json --output reports/eval.json
glioma-risk predict --case-dir derived/P001 --model-path models/dose-distance.json --output-dir derived/P001
```

Use `glioma-risk <stage> --help` for stage-specific options.

The default model is the dose/distance baseline. `--model voxel-logistic` trains a voxel-sampled logistic baseline. `--model unet` trains an optional MONAI/PyTorch 3D U-Net checkpoint when the `deep` extra is installed.

## Manifest Columns

Required:

- `patient_id`
- `baseline_scan_date`
- `baseline_t1c_series_uid`
- `baseline_flair_series_uid`
- `rtdose_sop_instance_uid`
- `recurrence_scan_date`
- `recurrence_adjudication`
- `reviewed_recurrence_mask_path`
- `split`

Optional but recommended:

- `radiotherapy_end_date`
- `prescription_dose_gy`

## Derived Files

Each case directory stores:

- `baseline_t1c.nii.gz`
- `baseline_flair.nii.gz`
- `dose_gy_on_baseline.nii.gz`
- `recurrence_mask_on_baseline.nii.gz`
- `brain_mask.nii.gz`
- `recurrence_risk.nii.gz`

## Development

```sh
uv run --extra dev pytest
```
