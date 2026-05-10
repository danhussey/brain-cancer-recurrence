# Reliability

Reliability means the pipeline fails before corrupting scientific conclusions.

## Current Reliability Controls

- DICOM RTDOSE requires `DoseUnits == GY` and positive `DoseGridScaling`.
- Affines are validated for shape, finite values, non-singular transforms, and homogeneous bottom row.
- Resampling uses patient-coordinate transforms.
- Patient-level split leakage is rejected at manifest load time.
- Pseudoprogression-window labels are excluded from training unless confirmed.
- Tests cover RTDOSE scaling, orientation, mask round-trip behavior, split leakage, synthetic overfit, and safety wording.
- A deterministic synthetic dataset generator supports end-to-end post-ingest smoke tests without patient data.
- The CFB-GBM adapter copies pilot files rather than symlinking because preprocessing overwrites derived outputs.

## Required Before Real Study Use

- Run ingestion on known-good fixture cases with independently verified dose grids.
- Add visual sign-off workflow for every recurrence-mask mapping.
- Record software versions, command invocations, and input checksums per run.
- Add cross-validation orchestration with immutable split manifests.
