from __future__ import annotations

import numpy as np
import pytest

from glioma_recurrence.geometry import GeometryError, Volume, assert_mask_round_trip, voxel_to_world, world_to_voxel


def test_affine_maps_voxel_indices_to_world_and_back():
    affine = np.eye(4)
    affine[:3, :3] = np.diag([2, 3, 4])
    affine[:3, 3] = [10, 20, 30]

    point = voxel_to_world(affine, np.asarray([[1, 2, 3]], dtype=float))[0]
    assert np.allclose(point, [12, 26, 42])
    restored = world_to_voxel(affine, np.asarray([point]))[0]
    assert np.allclose(restored, [1, 2, 3])


def test_mask_round_trip_detects_transform_silent_changes():
    mask = np.zeros((7, 7, 7), dtype=np.uint8)
    mask[2, 3, 4] = 1
    mask[3, 3, 4] = 1
    source = Volume(mask, np.eye(4))
    target_affine = np.eye(4)
    target_affine[:3, 3] = [-1, -1, -1]
    target = Volume(np.zeros((9, 9, 9), dtype=np.uint8), target_affine)

    assert_mask_round_trip(source, target)


def test_invalid_affine_is_rejected():
    with pytest.raises(GeometryError, match="singular"):
        Volume(np.zeros((2, 2, 2)), np.zeros((4, 4)))
