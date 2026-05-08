# New User Onboarding

## First Read

1. Read `AGENTS.md`.
2. Read `ARCHITECTURE.md`.
3. Read `docs/product-specs/glioma-recurrence-risk.md`.
4. Read `docs/RELIABILITY.md` and `docs/SECURITY.md` before touching real data.

## First Commands

```sh
uv run --extra dev pytest
uv run --extra dev python scripts/validate_knowledge_store.py
glioma-risk --help
```

## First Dataset

Start with a de-identified pilot manifest and one case. Run `ingest`, `preprocess`, `make-labels`, `train`, `evaluate`, and `predict` in order. Review the generated QC HTML before trusting any metric.

