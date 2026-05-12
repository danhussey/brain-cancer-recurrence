# Frontend

There is no interactive frontend in V1. The user-facing surface is:

- CLI commands exposed through `glioma-risk`.
- Static HTML QC reports generated per case.
- JSON evaluation reports for downstream analysis.

## Future Frontend Constraints

If an interactive review tool is added later:

- Keep the first viewport focused on actual case artifacts, not marketing or explanatory copy.
- Use compact controls for slice navigation, overlay opacity, risk threshold, and channel toggles.
- Preserve the research-only disclaimer on every exportable report.
- Static QC reports currently support representative slice tabs and overlay opacity controls. Risk threshold controls and visual regression snapshots remain future work.
- Use screenshots or browser-driven checks in tests so agents can verify UI behavior directly.
