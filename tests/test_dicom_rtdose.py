from __future__ import annotations

import numpy as np
import pytest

from glioma_recurrence.dicom import DicomError, rt_dose_array_gy, rtdose_affine
from glioma_recurrence.geometry import voxel_to_world


class FakeDoseDataset:
    DoseUnits = "GY"
    DoseGridScaling = 0.01
    ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ImagePositionPatient = [10, 20, 30]
    PixelSpacing = [2, 3]
    GridFrameOffsetVector = [0, 4]

    @property
    def pixel_array(self):
        return np.asarray(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
            ],
            dtype=np.uint16,
        )


def test_rtdose_scaling_uses_dose_grid_scaling_and_gy_units():
    dose = rt_dose_array_gy(FakeDoseDataset())

    assert dose.shape == (2, 2, 2)
    assert dose.dtype == np.float32
    assert dose[0, 0, 0] == pytest.approx(0.01)
    assert dose[0, 0, 1] == pytest.approx(0.05)
    assert dose[1, 1, 1] == pytest.approx(0.08)


def test_rtdose_scaling_rejects_relative_units():
    dataset = FakeDoseDataset()
    dataset.DoseUnits = "RELATIVE"

    with pytest.raises(DicomError, match="DoseUnits must be GY"):
        rt_dose_array_gy(dataset)


def test_rtdose_affine_respects_dicom_row_column_axis_semantics():
    affine = rtdose_affine(FakeDoseDataset())

    assert np.allclose(affine[:3, 0], [0, 2, 0])
    assert np.allclose(affine[:3, 1], [3, 0, 0])
    assert np.allclose(affine[:3, 2], [0, 0, 4])
    world = voxel_to_world(affine, np.asarray([[1, 2, 1]], dtype=float))[0]
    assert np.allclose(world, [16, 22, 34])

