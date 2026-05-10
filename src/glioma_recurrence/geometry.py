"""Affine and resampling utilities for patient-coordinate medical images."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


class GeometryError(ValueError):
    """Raised when an image geometry operation is invalid or ambiguous."""


@dataclass(frozen=True)
class Volume:
    data: np.ndarray
    affine: np.ndarray
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", np.asarray(self.data))
        object.__setattr__(self, "affine", np.asarray(self.affine, dtype=float))
        validate_affine(self.affine)
        if self.data.ndim != 3:
            raise GeometryError(f"expected 3D volume data; got shape {self.data.shape}")

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.data.shape)

    @property
    def spacing(self) -> tuple[float, float, float]:
        return spacing_from_affine(self.affine)


def validate_affine(affine: np.ndarray) -> None:
    if affine.shape != (4, 4):
        raise GeometryError(f"affine must be 4x4; got {affine.shape}")
    if not np.all(np.isfinite(affine)):
        raise GeometryError("affine contains non-finite values")
    determinant = np.linalg.det(affine[:3, :3])
    if abs(determinant) < 1e-8:
        raise GeometryError("affine has a singular spatial transform")
    if not np.allclose(affine[3], [0, 0, 0, 1]):
        raise GeometryError("affine bottom row must be [0, 0, 0, 1]")


def spacing_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    validate_affine(affine)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if np.any(spacing <= 0):
        raise GeometryError("affine spacing must be positive")
    return tuple(float(value) for value in spacing)


def voxel_to_world(affine: np.ndarray, indices: np.ndarray) -> np.ndarray:
    validate_affine(affine)
    indices = np.asarray(indices, dtype=float)
    homogeneous = np.concatenate([indices, np.ones((*indices.shape[:-1], 1))], axis=-1)
    return homogeneous @ affine.T[..., :3]


def world_to_voxel(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    validate_affine(affine)
    points = np.asarray(points, dtype=float)
    homogeneous = np.concatenate([points, np.ones((*points.shape[:-1], 1))], axis=-1)
    inverse = np.linalg.inv(affine)
    return homogeneous @ inverse.T[..., :3]


def resample_to_reference(
    moving: Volume,
    reference: Volume,
    *,
    order: int = 1,
    fill_value: float = 0.0,
) -> Volume:
    """Resample `moving` onto `reference` using patient-coordinate affines."""

    transform = np.linalg.inv(moving.affine) @ reference.affine
    matrix = transform[:3, :3]
    offset = transform[:3, 3]
    data = _affine_transform(
        moving.data,
        matrix=matrix,
        offset=offset,
        output_shape=reference.shape,
        order=order,
        fill_value=fill_value,
    )
    return Volume(data=data, affine=reference.affine.copy(), metadata=dict(moving.metadata or {}))


def resample_mask_to_reference(mask: Volume, reference: Volume) -> Volume:
    resampled = resample_to_reference(mask, reference, order=0, fill_value=0.0)
    return Volume((resampled.data >= 0.5).astype(np.uint8), resampled.affine, resampled.metadata)


def _affine_transform(
    data: np.ndarray,
    *,
    matrix: np.ndarray,
    offset: np.ndarray,
    output_shape: tuple[int, int, int],
    order: int,
    fill_value: float,
) -> np.ndarray:
    try:
        from scipy.ndimage import affine_transform

        return affine_transform(
            data,
            matrix=matrix,
            offset=offset,
            output_shape=output_shape,
            order=order,
            mode="constant",
            cval=fill_value,
            prefilter=order > 1,
        )
    except Exception:
        if order != 0:
            raise
        return _nearest_affine_transform(data, matrix, offset, output_shape, fill_value)


def _nearest_affine_transform(
    data: np.ndarray,
    matrix: np.ndarray,
    offset: np.ndarray,
    output_shape: tuple[int, int, int],
    fill_value: float,
) -> np.ndarray:
    output = np.full(output_shape, fill_value, dtype=data.dtype)
    for index in np.ndindex(output_shape):
        source = matrix @ np.asarray(index, dtype=float) + offset
        rounded = np.rint(source).astype(int)
        if np.all(rounded >= 0) and np.all(rounded < np.asarray(data.shape)):
            output[index] = data[tuple(rounded)]
    return output


def assert_mask_round_trip(mask: Volume, target: Volume) -> None:
    """Raise if a binary mask does not survive target-space round-trip geometry."""

    mapped = resample_mask_to_reference(mask, target)
    restored = resample_mask_to_reference(mapped, mask)
    if not np.array_equal(mask.data.astype(bool), restored.data.astype(bool)):
        diff = int(np.count_nonzero(mask.data.astype(bool) != restored.data.astype(bool)))
        raise GeometryError(f"mask round-trip changed {diff} voxels; check transforms/orientation")
