"""Structured run observability for pipeline stages."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def utc_run_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class RunObserver:
    stage: str
    root: Path
    command_args: dict[str, object]
    enabled: bool = True
    run_id: str = field(init=False)
    run_dir: Path = field(init=False)
    events_path: Path = field(init=False)
    summary_path: Path = field(init=False)
    started_at: str = field(init=False)
    _start_time: float = field(init=False)
    _event_counts: dict[str, int] = field(default_factory=dict)
    _artifacts: list[dict[str, object]] = field(default_factory=list)
    _case_status: dict[str, str] = field(default_factory=dict)
    _finished: bool = False

    def __post_init__(self) -> None:
        self.run_id = f"{utc_run_stamp()}-{self.stage}-{os.getpid()}"
        self.run_dir = self.root / self.run_id
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.json"

    def start(self) -> None:
        if not self.enabled:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = utc_now_iso()
        self._start_time = time.perf_counter()
        self.event("stage_started", command_args=self.command_args)

    def finish(self, *, status: str, error: str | None = None, exit_code: int | None = None) -> None:
        if not self.enabled or self._finished:
            return
        duration = time.perf_counter() - self._start_time
        self.event("stage_finished", status=status, duration_sec=duration, error=error, exit_code=exit_code)
        payload: dict[str, object] = {
            "run_id": self.run_id,
            "stage": self.stage,
            "status": status,
            "started_at": self.started_at,
            "ended_at": utc_now_iso(),
            "duration_sec": duration,
            "event_counts": dict(sorted(self._event_counts.items())),
            "case_status": dict(sorted(self._case_status.items())),
            "artifacts": self._artifacts,
            "command_args": self.command_args,
        }
        if error is not None:
            payload["error"] = error
        if exit_code is not None:
            payload["exit_code"] = exit_code
        self.summary_path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
        self._finished = True

    def event(self, event: str, **fields: object) -> None:
        if not self.enabled:
            return
        self._event_counts[event] = self._event_counts.get(event, 0) + 1
        payload = {
            "timestamp": utc_now_iso(),
            "run_id": self.run_id,
            "stage": self.stage,
            "event": event,
            **fields,
        }
        with self.events_path.open("a") as handle:
            handle.write(json.dumps(json_safe(payload), sort_keys=True) + "\n")

    @contextmanager
    def case(self, patient_id: str, operation: str) -> Iterator[None]:
        start = time.perf_counter()
        self.event("case_started", patient_id=patient_id, operation=operation)
        try:
            yield
        except Exception as exc:
            duration = time.perf_counter() - start
            self._case_status[patient_id] = "failed"
            self.event(
                "case_failed",
                patient_id=patient_id,
                operation=operation,
                duration_sec=duration,
                error=str(exc),
            )
            raise
        else:
            duration = time.perf_counter() - start
            self._case_status[patient_id] = "completed"
            self.event("case_completed", patient_id=patient_id, operation=operation, duration_sec=duration)

    def artifact(self, path: str | Path, *, kind: str, patient_id: str | None = None) -> None:
        artifact = {"path": str(path), "kind": kind}
        if patient_id is not None:
            artifact["patient_id"] = patient_id
        self._artifacts.append(artifact)
        self.event("artifact_written", **artifact)


class NoOpObserver:
    enabled = False

    def start(self) -> None:
        return None

    def finish(self, *, status: str, error: str | None = None, exit_code: int | None = None) -> None:
        return None

    def event(self, event: str, **fields: object) -> None:
        return None

    @contextmanager
    def case(self, patient_id: str, operation: str) -> Iterator[None]:
        yield

    def artifact(self, path: str | Path, *, kind: str, patient_id: str | None = None) -> None:
        return None


def add_observability_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--observability-root",
        default=None,
        help="Directory for run events and summaries. Defaults beside the derived/output workspace.",
    )
    parser.add_argument("--no-observability", action="store_true", help="Disable run event and summary artifacts")


def build_observer(args: argparse.Namespace) -> RunObserver | NoOpObserver:
    if getattr(args, "no_observability", False):
        return NoOpObserver()
    stage = str(getattr(args, "stage", "unknown"))
    root = resolve_observability_root(args)
    return RunObserver(stage=stage, root=root, command_args=args_payload(args))


def resolve_observability_root(args: argparse.Namespace) -> Path:
    explicit = getattr(args, "observability_root", None)
    if explicit:
        return Path(explicit)
    if hasattr(args, "derived_root"):
        return Path(args.derived_root).parent / "observability"
    if hasattr(args, "output_dir"):
        return Path(args.output_dir).parent / "observability"
    if hasattr(args, "summary_output"):
        return Path(args.summary_output).parent / "observability"
    if hasattr(args, "output"):
        return Path(args.output).parent / "observability"
    return Path("observability")


def args_payload(args: argparse.Namespace) -> dict[str, object]:
    ignored = {"func", "observer"}
    return {key: json_safe(value) for key, value in vars(args).items() if key not in ignored}


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return json_safe(value.item())
    return str(value)
