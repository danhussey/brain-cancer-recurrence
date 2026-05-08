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
uv run --extra dev python scripts/generate_synthetic_dataset.py --output-root /private/tmp/glioma-smoke --n-patients 2 --shape 12,12,12
uv run --extra dev python -m glioma_recurrence --help
```

## First Dataset

Start with a de-identified pilot manifest and one case. Run `ingest`, `preprocess`, `make-labels`, `train`, `evaluate`, and `predict` in order. Review the generated QC HTML before trusting any metric.

If no real data is available, use the synthetic generator to exercise every post-ingest stage. Synthetic outputs validate mechanics only and must not be used for scientific conclusions.
