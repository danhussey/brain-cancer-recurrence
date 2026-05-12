# 2026-05-12: Residual Tumor Versus Actionable Recurrence

## Finding

The synthetic QC report can make the tumor-distance baseline look conceptually weak because it predicts recurrence near the baseline tumor. That is expected for a synthetic toy case and also clinically plausible: visible residual tumor on post-operative, pre-radiotherapy MRI is a strong recurrence-risk location.

## Interpretation

High risk at residual tumor is not wrong, but it is not enough for the treatment-improvement goal. The useful scientific question is whether a model can predict recurrence beyond the obvious baseline tumor footprint, especially marginal or distant recurrence that might imply altered coverage, dose painting, or future planning hypotheses.

The project should therefore keep two recurrence concepts visible:

- recurrence inside the baseline tumor footprint;
- recurrence outside the baseline tumor footprint.

The second category is the harder and more actionable target. It still remains research-only; it should not be presented as a boost-region recommendation without stronger retrospective validation, external validation, prospective testing, and regulatory controls.

## Repository Change

QC reports now include an axial slice browser and `qc_summary.json` fields that count recurrence voxels inside and outside the baseline tumor mask. This makes it easier to spot models that mostly relearn residual tumor location.
