# Quality Score

| Area | Grade | Notes |
| --- | --- | --- |
| Manifest and split contracts | B | Required columns and leakage checks are tested. Needs larger fixture coverage. |
| DICOM RTDOSE handling | B | Scaling and affine semantics are unit tested. Needs real-world DICOM fixture tests. |
| Geometry and resampling | B | Affine validation and synthetic round-trip tests exist. Needs registration QA fixtures. |
| Baseline models | B | Dose/distance and voxel-logistic baselines exist with synthetic smoke tests. |
| Deep learning path | C | Optional MONAI U-Net trainer exists. Needs integration tests with `deep` extra and real patch sampling checks. |
| Reporting | C | Mandatory static QC panels exist. Needs richer overlays and visual regression checks. |
| Synthetic smoke harness | B | Deterministic generator and post-ingest pipeline smoke test exist. Needs CI wiring. |
| CFB-GBM pilot adapter | C | External-volume pilot preparation exists. Labels are GTV proxies only until recurrence masks are curated. |
| Agent legibility | B | Core doc map exists. Knowledge-store validation added in V2 scaffold pass. |

## Next Quality Investments

- Add anonymized miniature DICOM/RTDOSE fixtures for the `ingest` stage.
- Add model-card style evaluation summaries.
- Add CI job that runs knowledge-store validation and tests.
- Add doc freshness checks that compare CLI help to README examples.
