# Evaluation Lessons

This repository is included in my portfolio as an example of high-stakes evaluation engineering: leakage control, human review, calibration, patient-level validation, data provenance, and reproducible scientific pipelines.

It is not a clinical product and should not be read as a treatment-planning claim. The useful engineering lessons are about how to build a pipeline where the target is hard, the data are longitudinal, and optimistic evaluation mistakes would be misleading.

## General Lessons For Model Evaluation

- **Label leakage:** Follow-up scans define recurrence labels, but they must never become prediction-time model inputs.
- **Patient-level validation:** Train, validation, and test splits are enforced at patient/case level rather than voxel or slice level.
- **Data provenance:** Manifest columns, scanner metadata, label source fields, and observability artifacts make the data path auditable.
- **Human review:** Recurrence labels are expected to be clinician-reviewed, and model outputs are paired with QC overlays for inspection.
- **Calibration:** A risk heatmap should be evaluated as a probabilistic prediction, not only as a segmentation mask.
- **Conservative baselines:** A learned model should beat the tumor-distance baseline before it is scientifically interesting.
- **Quality-control artifacts:** Static HTML overlays, slice browsers, summary JSON, and run logs are first-class outputs, not afterthoughts.
- **Scope boundaries:** Research-only status is explicit; the output is not a dose recommendation, boost-region selection tool, or clinical device.

## Why It Matters Beyond Medicine

The same failure modes appear in broader model-evaluation work: leakage, split contamination, weak baselines, ambiguous labels, missing calibration, and unreviewed artifacts. This project is a concrete biomedical version of those general evaluation problems.
