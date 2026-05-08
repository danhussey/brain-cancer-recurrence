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

## Synthetic Smoke Dataset

Use the synthetic generator for engineering checks when no real DICOM dataset is available:

```sh
uv run --extra dev python scripts/generate_synthetic_dataset.py --output-root /private/tmp/glioma-smoke --n-patients 2 --shape 12,12,12
uv run --extra dev python -m glioma_recurrence preprocess --manifest /private/tmp/glioma-smoke/patients.csv --derived-root /private/tmp/glioma-smoke/derived --prescription-dose-gy 60
uv run --extra dev python -m glioma_recurrence make-labels --manifest /private/tmp/glioma-smoke/patients.csv --derived-root /private/tmp/glioma-smoke/derived --assume-baseline-space
uv run --extra dev python -m glioma_recurrence train --manifest /private/tmp/glioma-smoke/patients.csv --derived-root /private/tmp/glioma-smoke/derived --model dose-distance --output /private/tmp/glioma-smoke/models/dose-distance.json --prescription-dose-gy 60
uv run --extra dev python -m glioma_recurrence evaluate --manifest /private/tmp/glioma-smoke/patients.csv --derived-root /private/tmp/glioma-smoke/derived --model-path /private/tmp/glioma-smoke/models/dose-distance.json --output /private/tmp/glioma-smoke/reports/eval.json --splits validation --write-predictions
```

Synthetic data is only for pipeline validation. It is not scientifically meaningful.

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
uv run --extra dev python scripts/validate_knowledge_store.py
```
