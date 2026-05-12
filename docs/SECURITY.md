# Security And Privacy

This repository handles medical-imaging research data. Treat all real NIfTI images and derived images as sensitive.

## Current Boundaries

- Derived outputs are ignored by git through `derived/`, `models/`, and `reports/`.
- The repository does not include real patient data.
- `dicom-audit` reads headers without pixel data, writes hashed patient keys by default, and omits source file paths unless explicitly requested.

## Required Practices

- Do not commit NIfTI images, masks, model checkpoints trained on real patient data, or generated reports containing patient identifiers.
- Do not commit DICOM inventories from real clinical exports unless they have been reviewed for identifiers and explicitly approved for sharing.
- Keep de-identification as a dataset-download/adaptation concern and verify it before sharing artifacts.
- Use patient IDs from approved study manifests only.
- Avoid network uploads of data unless covered by institutional approvals.

## Future Controls

- Add data-loss-prevention checks for NIfTI extensions in git.
- Add run manifests with checksums and de-identification status.
- Add access-controlled artifact storage guidance.
- Add DICOM de-identification verification tooling before any external data transfer.
