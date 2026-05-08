# Security And Privacy

This repository handles medical-imaging research data. Treat all real DICOM and derived images as sensitive.

## Current Boundaries

- DICOM ingest writes PHI-light audit summaries rather than copying raw PHI fields into reports.
- Derived outputs are ignored by git through `derived/`, `models/`, and `reports/`.
- The repository does not include real patient data.

## Required Practices

- Do not commit DICOM, NIfTI, masks, model checkpoints trained on real patient data, or generated reports containing patient identifiers.
- Keep de-identification as an ingestion concern and verify it before sharing artifacts.
- Use patient IDs from approved study manifests only.
- Avoid network uploads of data unless covered by institutional approvals.

## Future Controls

- Add data-loss-prevention checks for DICOM/NIfTI extensions in git.
- Add run manifests with checksums and de-identification status.
- Add access-controlled artifact storage guidance.

