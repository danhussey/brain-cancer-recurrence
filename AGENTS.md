# Agent Map

This repository is optimized for agent-first development. Keep this file short and use it as the table of contents for deeper, versioned context.

## Start Here

- Product and safety scope: `docs/design-docs/glioma-recurrence-risk-pipeline.md`
- Architecture map: `ARCHITECTURE.md`
- Active execution plans: `docs/exec-plans/active/`
- Completed plans: `docs/exec-plans/completed/`
- External references and cached notes: `docs/references/`

## Working Rules

- Prefer small, verifiable changes with tests that exercise the user-facing stage or data contract being changed.
- Update docs when implementation changes domain behavior, safety posture, data contracts, or command semantics.
- Treat `patients.csv` and derived NIfTI filenames as public interfaces.
- Preserve the research-only framing. Do not add clinical-use, treatment recommendation, or dose-optimization claims.
- Keep follow-up scans out of prediction-time model inputs. Recurrence masks are labels only.
- Keep patient-level boundaries intact. No train/validation/test leakage across duplicate patient IDs.

## Validation

Run the focused test suite before handing off:

```sh
uv run --extra dev pytest
```

For dependency-light code paths, prefer pure functions in `src/glioma_recurrence/` and direct unit tests in `tests/`.
