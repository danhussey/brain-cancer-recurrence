# Glioma Recurrence Risk Pipeline

Research pipeline for predicting future glioma recurrence locations from post-operative pre-radiotherapy MRI plus radiotherapy dose maps.

V1 outputs a calibrated voxelwise recurrence-risk heatmap in the patient baseline planning space. It is not a clinical dose recommendation, treatment-planning system, or medical device.

## Key Terms

- **Glioma / GBM**: A brain tumor type. GBM means glioblastoma, an aggressive glioma.
- **Patient / case**: One subject in the dataset.
- **Baseline / t0**: The pre-radiotherapy planning timepoint used as the prediction input.
- **Follow-up / t1 / t2**: Later imaging timepoints used to determine whether and where recurrence happened.
- **Planning space**: The coordinate grid used for the baseline treatment-planning images.
- **Voxel**: A 3D pixel in an MRI, CT, dose map, mask, or risk map.
- **DICOM**: Common clinical imaging and radiotherapy file format.
- **NIfTI**: Common research imaging file format, usually `.nii` or `.nii.gz`.
- **T1c / T1gd**: Contrast-enhanced T1-weighted MRI. This is the main baseline anatomy/tumor channel.
- **FLAIR**: MRI sequence that highlights edema and abnormal fluid-like tissue signal.
- **CT**: Computed tomography image, commonly used for radiotherapy planning geometry.
- **RTDOSE**: Radiotherapy dose map. Values should represent physical dose in Gy.
- **Gy**: Gray, the physical unit of absorbed radiation dose.
- **Prescription dose**: The intended treatment dose, used here to normalize the dose channel.
- **RTSTRUCT**: Radiotherapy structure set containing clinician-drawn contours.
- **GTV mask**: Gross Tumor Volume mask: the visible tumor/target contour from radiotherapy planning. It is not a future recurrence label.
- **GTV proxy label**: A GTV mask temporarily used to test pipeline mechanics. Do not use it as recurrence ground truth.
- **Recurrence mask**: Human-reviewed mask of where tumor recurrence later occurred, mapped back to baseline space.
- **Brain mask**: Binary mask limiting training/evaluation to brain voxels.
- **Registration**: Aligning images from different scans or modalities into the same coordinate space.
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
- **QC overlay**: Visual report showing anatomy, dose, mask, and prediction for human review.

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

## CFB-GBM External Pilot

When CFB-GBM is stored on an external volume, prepare a small copied pilot workspace on that same volume. Do not symlink source images into `derived/`; `preprocess` writes outputs in place.

```sh
uv run --extra dev python scripts/prepare_cfb_gbm_dataset.py --source-root /Volumes/0437897195U/CFB-GBM --output-root /Volumes/0437897195U/CFB-GBM-pipeline-pilot --max-cases 2 --allow-gtv-proxy-labels
```

`--allow-gtv-proxy-labels` uses baseline GTV masks only as smoke-test proxy labels. They are not recurrence labels and must not be used for scientific recurrence modeling.

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
