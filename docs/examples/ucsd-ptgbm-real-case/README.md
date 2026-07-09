# UCSD-PTGBM Real-Data QC Example

This directory contains a static, public-data example generated from the UCSD-PTGBM collection on The Cancer Imaging Archive. The report uses a neutral case ID, `public-ucsd-ptgbm-case`, so the committed HTML does not expose the source case identifier.

Open the reports:

- [Preprocessing QC](public-ucsd-ptgbm-case/preprocess_qc.html)
- [Prediction and label QC](public-ucsd-ptgbm-case/qc_overlay.html)
- [Preprocessing summary JSON](public-ucsd-ptgbm-case/preprocess_qc_summary.json)
- [Prediction summary JSON](public-ucsd-ptgbm-case/qc_summary.json)

Only static HTML, JSON summaries, and PNG assets needed by the reports are committed. NIfTI images, masks, model files, clinical tables, local source paths, and observability logs are excluded.

This example is for research documentation only. It is not a clinical-use output, treatment recommendation, dose recommendation, boost-region selection, or patient-management tool.
