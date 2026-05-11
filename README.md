# Glioma Recurrence Risk Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![uv](https://img.shields.io/badge/package%20manager-uv-green)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status: research prototype](https://img.shields.io/badge/status-research%20prototype-orange)](#research-only)
[![Data: UCSD-PTGBM](https://img.shields.io/badge/data-UCSD--PTGBM-lightgrey)](https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/)

Research pipeline for predicting future glioma recurrence locations from post-operative MRI. The current public-data workflow is MRI-only and uses longitudinal baseline/follow-up MRI with tumor segmentations.

V1 outputs a voxelwise recurrence-risk heatmap in the patient baseline space. It is a research prototype, not a clinical dose recommendation, treatment-planning system, or medical device.

## Quickstart

This smoke test needs no medical data. It creates a tiny synthetic dataset, runs the full MRI-only pipeline, writes metrics, and produces a QC HTML overlay.

```sh
git clone https://github.com/danhussey/brain-cancer-recurrence.git
cd brain-cancer-recurrence
uv sync --extra dev

uv run python scripts/generate_synthetic_dataset.py --output-root /tmp/glioma-smoke --n-patients 3 --shape 16,16,16
uv run glioma-risk preprocess --manifest /tmp/glioma-smoke/patients.csv --derived-root /tmp/glioma-smoke/derived
uv run glioma-risk make-labels --manifest /tmp/glioma-smoke/patients.csv --derived-root /tmp/glioma-smoke/derived --assume-baseline-space
uv run glioma-risk train --manifest /tmp/glioma-smoke/patients.csv --derived-root /tmp/glioma-smoke/derived --model tumor-distance --output /tmp/glioma-smoke/models/tumor-distance.json
uv run glioma-risk evaluate --manifest /tmp/glioma-smoke/patients.csv --derived-root /tmp/glioma-smoke/derived --model-path /tmp/glioma-smoke/models/tumor-distance.json --output /tmp/glioma-smoke/reports/eval.json --splits validation,test --write-predictions
uv run glioma-risk predict --case-dir /tmp/glioma-smoke/derived/SYN002 --model-path /tmp/glioma-smoke/models/tumor-distance.json --output-dir /tmp/glioma-smoke/derived/SYN002
```

Open `/tmp/glioma-smoke/derived/SYN002/qc_overlay.html` to inspect the overlay report. The synthetic data is only for checking that the software works; it is not scientifically meaningful.

## What It Does

- Builds longitudinal patient pairs: baseline MRI input, later recurrence mask label.
- Keeps follow-up MRI out of prediction-time model inputs.
- Normalizes baseline T1c and FLAIR MRI and creates brain masks.
- Maps reviewed follow-up recurrence masks back into baseline space.
- Trains simple baseline models before deep learning.
- Evaluates voxel AUPRC, Dice-style overlap, top-risk-volume coverage, Brier score, calibration, and baseline comparisons.
- Writes structured observability artifacts for every CLI run.

## Research Only

This repository is for retrospective research and engineering validation. Do not use it for clinical decision-making, treatment planning, radiotherapy dose design, or patient management.

The current acceptance bar is deliberately conservative: a learned model should beat the `tumor-distance` baseline under patient-level validation before it is treated as scientifically interesting.

## Install Options

Base install includes the default MRI pipeline, NIfTI IO, SimpleITK registration, reports, baselines, and tests:

```sh
uv sync --extra dev
```

Optional MONAI/PyTorch U-Net support:

```sh
uv sync --extra dev --extra deep
```

The default `make-labels` path uses SimpleITK MRI-to-MRI registration. SimpleITK is a base dependency because it is required for normal full runs. Use `--registration-mode affine` or `--assume-baseline-space` only when the geometry fallback has been checked.

## CLI Stages

```sh
glioma-risk preprocess --manifest patients.csv --derived-root derived
glioma-risk make-labels --manifest patients.csv --derived-root derived
glioma-risk train --manifest patients.csv --derived-root derived --model tumor-distance --output models/tumor-distance.json
glioma-risk train --manifest patients.csv --derived-root derived --model voxel-logistic-mri --output models/voxel-logistic-mri.json
glioma-risk evaluate --manifest patients.csv --derived-root derived --model-path models/voxel-logistic-mri.json --output reports/eval.json
glioma-risk predict --case-dir derived/P001 --model-path models/voxel-logistic-mri.json --output-dir derived/P001
```

Use `glioma-risk <stage> --help` for stage-specific options.

Model options:

- `tumor-distance`: required simple recurrence-risk baseline.
- `voxel-logistic-mri`: first learned MRI-only baseline using T1c, FLAIR, baseline tumor mask, and distance features.
- `unet`: optional MONAI/PyTorch 3D U-Net when the `deep` extra is installed.

## UCSD-PTGBM Workflow

Download UCSD-PTGBM images/segmentations and clinical data from TCIA onto external storage, then prepare a copied working set:

```sh
uv run python scripts/prepare_ucsd_ptgbm_dataset.py \
  --source-root /Volumes/External/UCSD-PTGBM \
  --clinical-table /Volumes/External/UCSD-PTGBM/clinical.xlsx \
  --negative-cases-table /Volumes/External/UCSD-PTGBM/details_of_negative_cases_TCIA.xlsx \
  --include-negative-controls \
  --output-root /Volumes/External/UCSD-PTGBM-pipeline
```

The adapter selects subjects with at least two complete MRI+mask timepoints, uses the earliest complete post-treatment timepoint as baseline, and uses the earliest later residual/recurrent tumor timepoint as the recurrence label.

When UCSD negative-case categories are available, `--include-negative-controls` keeps pseudoprogression, radiation-necrosis, and non-specific later timepoints as controls with empty recurrence labels. Their abnormality segmentations are not used as recurrence targets.

Audit download completeness without writing derivatives:

```sh
uv run python scripts/audit_ucsd_ptgbm_dataset.py \
  --source-root /Volumes/External/UCSD-PTGBM \
  --clinical-table /Volumes/External/UCSD-PTGBM/clinical.xlsx \
  --negative-cases-table /Volumes/External/UCSD-PTGBM/details_of_negative_cases_TCIA.xlsx \
  --include-negative-controls \
  --json-output /Volumes/External/UCSD-PTGBM-pipeline/reports/download-audit.json
```

If images arrive before the clinical workbook, a provisional filename-only working set is available for pipeline bring-up:

```sh
uv run python scripts/prepare_ucsd_ptgbm_dataset.py \
  --source-root /Volumes/External/UCSD-PTGBM \
  --output-root /Volumes/External/UCSD-PTGBM-pipeline \
  --allow-imaging-only-labels \
  --max-subjects 20
```

That mode is not appropriate for scientific labels because it has no clinical progression adjudication.

## Observability

Every CLI stage writes structured run artifacts by default:

- `events.jsonl`: timestamped stage, case, metric, and artifact events.
- `summary.json`: final status, duration, case status, event counts, command args, and output artifacts.

For manifest stages, artifacts default to `<derived-root>/../observability/<run-id>/`. For `predict`, they default beside the output directory.

```sh
glioma-risk train --manifest patients.csv --derived-root derived --output models/tumor-distance.json --observability-root runs
glioma-risk evaluate --manifest patients.csv --derived-root derived --model-path models/voxel-logistic-mri.json --output reports/eval.json --no-observability
```

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
- `qc_overlay.html`

## Accessibility

The README is intentionally command-first and plain-language. The glossary below defines common imaging, modeling, and clinical terms so collaborators can inspect the repository without already knowing the project vocabulary.

The repository should not contain patient data, clinical spreadsheets, private credentials, or local derived outputs. Keep real datasets on external storage or institution-approved systems.

## Development

```sh
uv run --extra dev pytest
uv run --extra dev python scripts/validate_knowledge_store.py
```

## Glossary

- **Affine**: Matrix that maps voxel indices to real patient/world coordinates.
- **AUPRC**: Area under the precision-recall curve; useful when recurrence voxels are rare.
- **Baseline / t0**: Earlier post-treatment MRI timepoint used as prediction input.
- **Baseline tumor mask**: Tumor segmentation at the baseline timepoint. This is a prediction-time location feature.
- **Brain mask**: Binary mask limiting training/evaluation to brain voxels.
- **Brier score**: Mean squared error of predicted probabilities; lower is better.
- **Calibration**: Whether predicted risks match observed recurrence frequencies.
- **Confirmed recurrence label**: Reviewed decision that a later abnormality is true recurrent/progressive tumor plus a voxelwise mask of that tumor.
- **Dice**: Spatial overlap score between a predicted region and a label mask.
- **FLAIR**: MRI sequence that highlights edema and abnormal fluid-like tissue signal.
- **Follow-up / t1 / t2**: Later imaging timepoints used to determine whether and where recurrence happened.
- **GBM**: Glioblastoma, an aggressive glioma.
- **Glioma**: Brain tumor type arising from glial cells.
- **Leakage**: Accidental sharing of the same patient across train/validation/test.
- **NIfTI**: Common research imaging file format, usually `.nii` or `.nii.gz`.
- **Patient / case**: One subject in the dataset.
- **Pseudoprogression**: Early post-radiotherapy imaging change that can mimic recurrence.
- **QC overlay**: Visual report showing anatomy, masks, and prediction for human review.
- **Recurrence mask**: Human-reviewed mask of where tumor recurrence later occurred, mapped back to baseline space.
- **Registration**: Aligning images from different scans into the same coordinate space.
- **Resampling**: Regridding one image onto another image's voxel grid after alignment.
- **Risk heatmap**: Voxelwise model output from 0 to 1 estimating recurrence risk.
- **Split**: Train, validation, or test assignment at the patient level.
- **T1c / T1gd**: Contrast-enhanced T1-weighted MRI. This is the main anatomy/tumor channel.
- **Timepoint**: One imaging acquisition/session for one subject, often containing several MRI sequences and masks.
- **Voxel**: A 3D pixel in an MRI, mask, or risk map.
