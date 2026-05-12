# New User Onboarding

## First Read

1. Read `AGENTS.md`.
2. Read `ARCHITECTURE.md`.
3. Read `docs/product-specs/glioma-recurrence-risk.md`.
4. Read `docs/RELIABILITY.md` and `docs/SECURITY.md` before touching real data.

## First Commands

```sh
uv sync --extra dev
uv run --extra dev pytest
uv run --extra dev python scripts/validate_knowledge_store.py
uv run glioma-risk --help
```

## First Dataset

Run the README quickstart first. It creates a synthetic no-data smoke dataset and executes `preprocess`, `make-labels`, `train`, `evaluate`, and `predict` in order.

For real data, start with a de-identified pilot manifest and one case. Review generated QC HTML before trusting any metric. Synthetic outputs validate mechanics only and must not be used for scientific conclusions.

Before converting a clinical DICOM export to NIfTI, run `glioma-risk dicom-audit` on the export root and inspect the sequence availability, scanner metadata, and PHI-field flags.
