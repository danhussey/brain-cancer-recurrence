"""Preprocessing utilities for MRI-only baseline volumes."""

from __future__ import annotations

import numpy as np

from .geometry import Volume, resample_to_reference


def robust_normalize_mri(data: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(data)
    if mask is not None:
        finite &= np.asarray(mask).astype(bool)
    values = data[finite]
    values = values[values != 0]
    if values.size < 10:
        values = data[np.isfinite(data)]
    if values.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    low, high = np.percentile(values, [1, 99])
    clipped = np.clip(data, low, high)
    center = float(np.median(clipped[finite])) if np.any(finite) else float(np.median(clipped))
    spread = float(np.subtract(*np.percentile(clipped[finite] if np.any(finite) else clipped, [75, 25])))
    if spread <= 1e-6:
        spread = float(np.std(clipped[finite] if np.any(finite) else clipped))
    if spread <= 1e-6:
        spread = 1.0
    normalized = (clipped - center) / spread
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized.astype(np.float32)


def brain_mask_from_modalities(t1c: np.ndarray, flair: np.ndarray) -> np.ndarray:
    t1c = np.asarray(t1c)
    flair = np.asarray(flair)
    candidate = (np.isfinite(t1c) & (t1c != 0)) | (np.isfinite(flair) & (flair != 0))
    if not np.any(candidate):
        return np.ones_like(t1c, dtype=np.uint8)
    try:
        from scipy.ndimage import binary_closing, binary_fill_holes

        candidate = binary_fill_holes(binary_closing(candidate, iterations=2))
    except Exception:
        pass
    return candidate.astype(np.uint8)


def resample_flair_and_tumor_to_t1c(
    t1c: Volume,
    flair: Volume,
    baseline_tumor_mask: Volume,
) -> tuple[Volume, Volume]:
    flair_on_t1c = resample_to_reference(flair, t1c, order=1, fill_value=0.0)
    from .geometry import resample_mask_to_reference

    tumor_on_t1c = resample_mask_to_reference(baseline_tumor_mask, t1c)
    return flair_on_t1c, tumor_on_t1c


def distance_to_mask_mm(mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> np.ndarray:
    binary = np.asarray(mask).astype(bool)
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:
        raise RuntimeError("scipy is required for distance-to-mask computation") from exc
    outside_distance = distance_transform_edt(~binary, sampling=spacing_mm)
    inside_distance = -distance_transform_edt(binary, sampling=spacing_mm)
    return np.where(binary, inside_distance, outside_distance).astype(np.float32)
