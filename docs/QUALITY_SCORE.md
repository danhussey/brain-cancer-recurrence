# Quality Score

| Area | Grade | Notes |
| --- | --- | --- |
| Manifest and split contracts | B | Required columns and leakage checks are tested. Needs larger fixture coverage. |
| Geometry and resampling | B | Affine validation and synthetic round-trip tests exist. Needs registration QA fixtures. |
| Baseline models | B | Tumor-distance and voxel-logistic MRI baselines exist with synthetic smoke tests. |
| Deep learning path | C | Optional MONAI U-Net trainer exists. Needs integration tests with `deep` extra and real patch sampling checks. |
| Reporting | C | Mandatory static QC panels exist. Needs richer overlays and visual regression checks. |
| Observability | B | CLI stages emit JSONL events and summaries with artifacts, timings, and metrics. Needs dashboards once real runs exist. |
| Synthetic smoke harness | B | Deterministic generator and pipeline smoke test exist. Needs CI wiring. |
| UCSD-PTGBM adapter | B | Longitudinal MRI+mask pairing is tested with fake workbook data. Needs validation on the real downloaded layout. |
| Clinical DICOM intake | C | Read-only DICOM header audit exists with fake-DICOM tests. Needs real onsite export validation and DICOM-to-NIfTI conversion orchestration. |
| Agent legibility | B | Core doc map exists. Knowledge-store validation added in V2 scaffold pass. |

## Next Quality Investments

- Add real-layout UCSD fixture metadata for adapter regression tests.
- Validate `dicom-audit` on a de-identified clinical export and record expected scanner/sequence naming patterns.
- Add model-card style evaluation summaries.
- Add CI job that runs knowledge-store validation and tests.
- Add doc freshness checks that compare CLI help to README examples.
