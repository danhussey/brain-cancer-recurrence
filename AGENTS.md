# Agent Map

This repository is optimized for agent-first development. Keep this file short: it is the table of contents, not the manual.

## Start Here

- Architecture and layering: `ARCHITECTURE.md`
- Product scope: `docs/product-specs/index.md`
- Domain design docs: `docs/design-docs/index.md`
- Research log and durable learnings: `docs/research-log/index.md`
- Current and historical plans: `docs/PLANS.md`
- Quality status: `docs/QUALITY_SCORE.md`
- Reliability posture: `docs/RELIABILITY.md`
- Security and privacy posture: `docs/SECURITY.md`
- External reference cache: `docs/references/`
- Generated artifacts: `docs/generated/`

## Non-Negotiables

- Preserve the research-only framing. Do not add clinical-use, treatment recommendation, or dose-optimization claims.
- Treat `patients.csv` and derived NIfTI filenames as public interfaces.
- Keep follow-up scans out of prediction-time model inputs. Recurrence masks are labels only.
- Keep patient-level boundaries intact. No train/validation/test leakage across duplicate patient IDs.
- Update repository-local docs when implementation changes behavior, safety posture, data contracts, architecture, or command semantics.
- Prefer small, verifiable changes with tests that exercise the user-facing stage or boundary being changed.

## Validation

Run focused checks before handoff:

```sh
uv run --extra dev pytest
uv run --extra dev python scripts/validate_knowledge_store.py
```

For dependency-light code paths, prefer pure functions in `src/glioma_recurrence/` and direct unit tests in `tests/`.
