# Reporting QC Upgrade

## Status

Completed 2026-05-12.

## Goal

Upgrade case-level QC reports from separate static panels into a reviewable filesystem HTML report with enough controls to inspect anatomy, masks, and predictions together.

## Changes

- Added representative slice selection from midline, baseline tumor peak, recurrence peak, and risk peak.
- Added an axial slice browser with quick jumps to representative slices.
- Added T1c and FLAIR viewer stacks with transparent baseline tumor, recurrence, and risk overlays.
- Added recurrence-inside/outside-baseline-tumor summary fields to make obvious residual-tumor recurrence visible.
- Added browser opacity sliders for each overlay type.
- Added concise case-summary tooltips for reviewer interpretation.
- Added `qc_summary.json` beside each `qc_overlay.html`.
- Added observability artifacts for QC summaries.
- Added focused report tests and smoke-test coverage for summary output.

## Remaining Work

- Add visual regression snapshots for generated reports.
- Add formal QC sign-off status and reviewer notes.
- Add risk-threshold controls after the preferred review workflow is clear.
