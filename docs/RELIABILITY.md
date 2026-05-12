# Reliability

Reliability means the pipeline fails before corrupting scientific conclusions.

## Current Reliability Controls

- Affines are validated for shape, finite values, non-singular transforms, and homogeneous bottom row.
- Resampling uses patient-coordinate transforms.
- Patient-level split leakage is rejected at manifest load time.
- Pseudoprogression-window labels are excluded from training unless confirmed.
- DICOM intake audit reads headers only, pseudonymizes patient keys by default, and flags likely PHI-bearing fields.
- CLI stages write structured `events.jsonl` and `summary.json` artifacts with case timings, status, and output paths unless disabled.
- Tests cover orientation, mask round-trip behavior, split leakage, synthetic overfit, UCSD adapter pairing, and safety wording.
- A deterministic synthetic dataset generator supports end-to-end smoke tests without patient data.

## Required Before Real Study Use

- Add visual sign-off workflow for every recurrence-mask mapping.
- Record software versions, command invocations, and input checksums per run.
- Add cross-validation orchestration with immutable split manifests.
- Add DICOM-to-NIfTI conversion provenance and round-trip geometry checks before clinical export use.
