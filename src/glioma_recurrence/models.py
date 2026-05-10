"""MRI-only recurrence-risk baselines and serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

from .case import CaseData
from .preprocess import distance_to_mask_mm

ModelKind = Literal["tumor-distance", "voxel-logistic-mri"]


class ModelError(RuntimeError):
    """Raised when a model cannot be trained, loaded, or applied."""


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, -60, 60)
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)


def mri_feature_matrix(case: CaseData) -> np.ndarray:
    baseline_tumor = case.baseline_tumor_mask.data.astype(bool)
    distance = distance_to_mask_mm(baseline_tumor, case.t1c.spacing)
    features = np.stack(
        [
            case.t1c.data.astype(np.float32),
            case.flair.data.astype(np.float32),
            baseline_tumor.astype(np.float32),
            distance,
        ],
        axis=-1,
    )
    return features.reshape(-1, features.shape[-1])


def flattened_brain_indices(case: CaseData) -> np.ndarray:
    return np.flatnonzero(case.brain_mask.data.astype(bool).reshape(-1))


def sample_case_voxels(
    case: CaseData,
    *,
    max_voxels: int,
    rng: np.random.Generator,
    positive_fraction: float = 0.5,
) -> np.ndarray:
    brain = flattened_brain_indices(case)
    if case.recurrence_mask is None:
        raise ModelError(f"{case.patient_id}: recurrence mask is required for training")
    labels = case.recurrence_mask.data.astype(bool).reshape(-1)
    positives = brain[labels[brain]]
    negatives = brain[~labels[brain]]
    if positives.size == 0:
        negative_count = min(max_voxels, negatives.size)
        return rng.choice(negatives, size=negative_count, replace=False)
    positive_count = min(max(1, int(max_voxels * positive_fraction)), positives.size)
    negative_count = min(max_voxels - positive_count, negatives.size)
    selected = [
        rng.choice(positives, size=positive_count, replace=positive_count > positives.size),
        rng.choice(negatives, size=negative_count, replace=False) if negative_count else np.array([], dtype=int),
    ]
    return rng.permutation(np.concatenate(selected).astype(int))


@dataclass
class TumorDistanceBandModel:
    distance_bins_mm: list[float]
    risk_table: list[float]
    global_risk: float
    smoothing: float = 1.0

    kind: ModelKind = "tumor-distance"

    @classmethod
    def fit(
        cls,
        cases: Iterable[CaseData],
        *,
        distance_bins_mm: list[float] | None = None,
        max_voxels_per_case: int = 20000,
        seed: int = 13,
    ) -> "TumorDistanceBandModel":
        distance_bins_mm = distance_bins_mm or [-1_000_000.0, -20.0, -5.0, 0.0, 5.0, 20.0, 50.0, 1_000_000.0]
        counts = np.zeros(len(distance_bins_mm) - 1, dtype=np.float64)
        positives = np.zeros_like(counts)
        rng = np.random.default_rng(seed)
        total_positive = 0.0
        total_count = 0.0
        for case in cases:
            if case.recurrence_mask is None:
                raise ModelError(f"{case.patient_id}: recurrence mask is required for tumor-distance training")
            baseline_tumor = case.baseline_tumor_mask.data.astype(bool)
            indices = sample_case_voxels(case, max_voxels=max_voxels_per_case, rng=rng)
            distance = distance_to_mask_mm(baseline_tumor, case.t1c.spacing).reshape(-1)[indices]
            labels = case.recurrence_mask.data.astype(bool).reshape(-1)[indices]
            buckets = np.digitize(distance, distance_bins_mm, right=False) - 1
            valid = (buckets >= 0) & (buckets < counts.shape[0])
            for bucket, label in zip(buckets[valid], labels[valid]):
                counts[bucket] += 1
                positives[bucket] += float(label)
            total_positive += float(np.count_nonzero(labels))
            total_count += float(labels.size)
        if total_count == 0:
            raise ModelError("no training voxels were sampled")
        smoothing = 1.0
        global_risk = (total_positive + smoothing) / (total_count + 2 * smoothing)
        risk_table = (positives + smoothing * global_risk) / (counts + smoothing)
        risk_table[counts == 0] = global_risk
        return cls(
            distance_bins_mm=[float(value) for value in distance_bins_mm],
            risk_table=[float(value) for value in risk_table],
            global_risk=float(global_risk),
            smoothing=smoothing,
        )

    def predict_case(self, case: CaseData) -> np.ndarray:
        baseline_tumor = case.baseline_tumor_mask.data.astype(bool)
        distance = distance_to_mask_mm(baseline_tumor, case.t1c.spacing)
        buckets = np.digitize(distance, self.distance_bins_mm, right=False) - 1
        table = np.asarray(self.risk_table, dtype=np.float32)
        risk = np.full(case.t1c.shape, self.global_risk, dtype=np.float32)
        valid = (buckets >= 0) & (buckets < table.shape[0]) & case.brain_mask.data.astype(bool)
        risk[valid] = table[buckets[valid]]
        risk[~case.brain_mask.data.astype(bool)] = 0.0
        return np.clip(risk, 0.0, 1.0).astype(np.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "distance_bins_mm": self.distance_bins_mm,
            "risk_table": self.risk_table,
            "global_risk": self.global_risk,
            "smoothing": self.smoothing,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TumorDistanceBandModel":
        return cls(
            distance_bins_mm=[float(value) for value in payload["distance_bins_mm"]],
            risk_table=[float(value) for value in payload["risk_table"]],
            global_risk=float(payload["global_risk"]),
            smoothing=float(payload.get("smoothing", 1.0)),
        )


@dataclass
class VoxelLogisticMRIModel:
    coefficients: list[float]
    intercept: float
    feature_mean: list[float]
    feature_scale: list[float]
    kind: ModelKind = "voxel-logistic-mri"

    @classmethod
    def fit(
        cls,
        cases: Iterable[CaseData],
        *,
        max_voxels_per_case: int = 20000,
        seed: int = 13,
        iterations: int = 500,
        learning_rate: float = 0.2,
        l2: float = 1e-4,
    ) -> "VoxelLogisticMRIModel":
        rng = np.random.default_rng(seed)
        feature_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        for case in cases:
            if case.recurrence_mask is None:
                raise ModelError(f"{case.patient_id}: recurrence mask is required for voxel-logistic-mri training")
            indices = sample_case_voxels(case, max_voxels=max_voxels_per_case, rng=rng)
            feature_parts.append(mri_feature_matrix(case)[indices])
            label_parts.append(case.recurrence_mask.data.astype(np.float32).reshape(-1)[indices])
        if not feature_parts:
            raise ModelError("no training cases were supplied")
        x = np.concatenate(feature_parts, axis=0).astype(np.float64)
        y = np.concatenate(label_parts, axis=0).astype(np.float64)
        if np.unique(y).size < 2:
            raise ModelError("voxel-logistic-mri training requires both positive and negative labels")
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-6] = 1.0
        x = (x - mean) / scale
        weights = np.zeros(x.shape[1], dtype=np.float64)
        intercept = float(np.log((y.mean() + 1e-4) / (1 - y.mean() + 1e-4)))
        positive_weight = float((y.size - y.sum()) / max(y.sum(), 1.0))
        sample_weight = np.where(y > 0.5, positive_weight, 1.0)
        sample_weight = sample_weight / sample_weight.mean()
        for _ in range(iterations):
            prediction = sigmoid(x @ weights + intercept).astype(np.float64)
            error = (prediction - y) * sample_weight
            gradient = (x.T @ error) / x.shape[0] + l2 * weights
            intercept_gradient = float(error.mean())
            weights -= learning_rate * gradient
            intercept -= learning_rate * intercept_gradient
        return cls(
            coefficients=[float(value) for value in weights],
            intercept=float(intercept),
            feature_mean=[float(value) for value in mean],
            feature_scale=[float(value) for value in scale],
        )

    def predict_case(self, case: CaseData) -> np.ndarray:
        features = mri_feature_matrix(case)
        x = (features - np.asarray(self.feature_mean)) / np.asarray(self.feature_scale)
        logits = x @ np.asarray(self.coefficients) + self.intercept
        risk = sigmoid(logits).reshape(case.t1c.shape)
        risk[~case.brain_mask.data.astype(bool)] = 0.0
        return np.clip(risk, 0.0, 1.0).astype(np.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "feature_mean": self.feature_mean,
            "feature_scale": self.feature_scale,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "VoxelLogisticMRIModel":
        return cls(
            coefficients=[float(value) for value in payload["coefficients"]],
            intercept=float(payload["intercept"]),
            feature_mean=[float(value) for value in payload["feature_mean"]],
            feature_scale=[float(value) for value in payload["feature_scale"]],
        )


RiskModel = TumorDistanceBandModel | VoxelLogisticMRIModel


def save_model(model: RiskModel, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(model.to_dict(), indent=2, sort_keys=True) + "\n")


def load_model(path: str | Path) -> RiskModel:
    payload = json.loads(Path(path).read_text())
    kind = payload.get("kind")
    if kind == "tumor-distance":
        return TumorDistanceBandModel.from_dict(payload)
    if kind == "voxel-logistic-mri":
        return VoxelLogisticMRIModel.from_dict(payload)
    raise ModelError(f"unsupported model kind in {path}: {kind!r}")
