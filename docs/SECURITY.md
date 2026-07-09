# Security And Privacy

This repository handles medical-imaging research data. Treat all real NIfTI images and derived images as sensitive.

## Current Boundaries

- The repository does not include private patient data, source NIfTI volumes, DICOM pixels, clinical spreadsheets, or model checkpoints trained on real patient data.
- The only committed real-data visual example is the neutral-ID public UCSD-PTGBM static QC example under `docs/examples/ucsd-ptgbm-real-case/`.
- `dicom-audit` reads headers without pixel data, writes hashed patient keys by default, and omits source file paths unless explicitly requested.
- `.gitignore` currently covers `derived/`, `models/`, and `reports/` at the repo root.
- Dataset adapters and smoke-test tooling also write `masks/`, `label_refs/`, and `observability/`; those paths are not currently ignored by git.

## Required Practices

- Do not commit NIfTI images, masks, label-reference images, observability artifacts, model checkpoints trained on real patient data, or generated reports containing patient identifiers.
- The committed README example under `docs/examples/ucsd-ptgbm-real-case/` is limited to static HTML, JSON summaries, and PNG report assets generated from public UCSD-PTGBM data with a neutral case ID. Keep source NIfTI files, masks, models, local paths, clinical tables, and source case identifiers out of that example.
- Do not commit DICOM inventories from real clinical exports unless they have been reviewed for identifiers and explicitly approved for sharing.
- Keep de-identification as a dataset-download/adaptation concern and verify it before sharing artifacts.
- Use patient IDs from approved study manifests only.
- Avoid network uploads of data unless covered by institutional approvals.

## Future Controls

- Add data-loss-prevention checks for NIfTI extensions in git.
- Add run manifests with checksums and de-identification status.
- Add access-controlled artifact storage guidance.
- Add DICOM de-identification verification tooling before any external data transfer.
