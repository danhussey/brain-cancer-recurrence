# Observability V1 Execution Plan

## Goal

Make MRI-only pipeline runs legible to agents and researchers without requiring an external observability stack.

## Work Items

- [x] Add structured run observation helpers for stage events, case timings, artifacts, and summaries.
- [x] Wire observability into `preprocess`, `make-labels`, `train`, `evaluate`, and `predict`.
- [x] Add `--observability-root` and `--no-observability` CLI controls on every stage.
- [x] Record evaluation metrics and output artifacts in machine-readable run logs.
- [x] Add a smoke test proving run summaries and events are emitted.

## Decisions

- Use dependency-light JSONL and JSON files instead of a service dependency.
- Default observability output lives beside the derived/output workspace.
- Clinical/research outputs stay separate from run-observability metadata.

## Verification

- `uv run --extra dev pytest`: 18 tests passed on 2026-05-10.
- `uv run --extra dev python scripts/validate_knowledge_store.py`: passed on 2026-05-10.
- `git diff --check`: passed on 2026-05-10.
