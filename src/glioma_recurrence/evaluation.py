"""Evaluation metrics for voxelwise recurrence-risk maps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .case import CaseData
from .geometry import Volume


@dataclass(frozen=True)
class CaseMetrics:
    patient_id: str
    voxel_auprc: float
    brier_score: float
    dice_at_top_1pct: float
    dice_at_top_5pct: float
    recurrence_coverage_top_1pct: float
    recurrence_coverage_top_5pct: float
    positive_voxels: int
    evaluated_voxels: int


def average_precision_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or np.count_nonzero(labels) == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(~sorted_labels)
    precision = tp / np.maximum(tp + fp, 1)
    recall_delta = sorted_labels.astype(float) / max(float(np.count_nonzero(labels)), 1.0)
    return float(np.sum(precision * recall_delta))


def brier_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    return float(np.mean((scores - labels) ** 2))


def top_fraction_mask(scores: np.ndarray, fraction: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    count = max(1, int(np.ceil(scores.size * fraction)))
    threshold = np.partition(scores, -count)[-count]
    return scores >= threshold


def dice(labels: np.ndarray, prediction: np.ndarray) -> float:
    labels = np.asarray(labels).astype(bool)
    prediction = np.asarray(prediction).astype(bool)
    denominator = np.count_nonzero(labels) + np.count_nonzero(prediction)
    if denominator == 0:
        return 1.0
    return float(2 * np.count_nonzero(labels & prediction) / denominator)


def recurrence_coverage(labels: np.ndarray, prediction: np.ndarray) -> float:
    labels = np.asarray(labels).astype(bool)
    prediction = np.asarray(prediction).astype(bool)
    positives = np.count_nonzero(labels)
    if positives == 0:
        return float("nan")
    return float(np.count_nonzero(labels & prediction) / positives)


def calibration_bins(labels: np.ndarray, scores: np.ndarray, *, bins: int = 10) -> list[dict[str, float]]:
    labels = np.asarray(labels, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float]] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = (scores >= lower) & (scores <= upper)
        else:
            mask = (scores >= lower) & (scores < upper)
        if np.any(mask):
            rows.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": int(np.count_nonzero(mask)),
                    "mean_predicted": float(scores[mask].mean()),
                    "observed": float(labels[mask].mean()),
                }
            )
        else:
            rows.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": 0,
                    "mean_predicted": float("nan"),
                    "observed": float("nan"),
                }
            )
    return rows


def evaluate_case(case: CaseData, risk: Volume) -> CaseMetrics:
    if case.recurrence_mask is None:
        raise ValueError(f"{case.patient_id}: recurrence mask is required for evaluation")
    mask = case.brain_mask.data.astype(bool).reshape(-1)
    labels = case.recurrence_mask.data.astype(bool).reshape(-1)[mask]
    scores = np.clip(risk.data.reshape(-1)[mask], 0.0, 1.0)
    top_1 = top_fraction_mask(scores, 0.01)
    top_5 = top_fraction_mask(scores, 0.05)
    return CaseMetrics(
        patient_id=case.patient_id,
        voxel_auprc=average_precision_score(labels, scores),
        brier_score=brier_score(labels.astype(np.float32), scores),
        dice_at_top_1pct=dice(labels, top_1),
        dice_at_top_5pct=dice(labels, top_5),
        recurrence_coverage_top_1pct=recurrence_coverage(labels, top_1),
        recurrence_coverage_top_5pct=recurrence_coverage(labels, top_5),
        positive_voxels=int(np.count_nonzero(labels)),
        evaluated_voxels=int(labels.size),
    )


def summarize_metrics(metrics: Iterable[CaseMetrics]) -> dict[str, object]:
    rows = [metric.__dict__ for metric in metrics]
    summary: dict[str, object] = {"cases": rows, "n_cases": len(rows)}
    numeric_keys = [
        "voxel_auprc",
        "brier_score",
        "dice_at_top_1pct",
        "dice_at_top_5pct",
        "recurrence_coverage_top_1pct",
        "recurrence_coverage_top_5pct",
    ]
    for key in numeric_keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        summary[f"mean_{key}"] = float(values.mean()) if values.size else float("nan")
    return summary


def write_evaluation_report(summary: dict[str, object], output: str | Path) -> None:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n")

