from __future__ import annotations

import json

import numpy as np

from glioma_recurrence.case import CaseData
from glioma_recurrence.evaluation import average_precision_score, dice, recurrence_coverage
from glioma_recurrence.geometry import Volume
from glioma_recurrence.models import DoseDistanceBandModel, VoxelLogisticModel


def synthetic_case(patient_id: str) -> CaseData:
    shape = (8, 8, 8)
    affine = np.eye(4)
    t1c = np.zeros(shape, dtype=np.float32)
    flair = np.zeros(shape, dtype=np.float32)
    dose = np.zeros(shape, dtype=np.float32)
    label = np.zeros(shape, dtype=np.uint8)
    label[3:5, 3:5, 3:5] = 1
    dose[label.astype(bool)] = 60.0
    dose[2:6, 2:6, 2:6] = np.maximum(dose[2:6, 2:6, 2:6], 35.0)
    flair[label.astype(bool)] = 4.0
    t1c[label.astype(bool)] = 2.0
    brain = np.ones(shape, dtype=np.uint8)
    return CaseData(
        patient_id=patient_id,
        t1c=Volume(t1c, affine),
        flair=Volume(flair, affine),
        dose_gy=Volume(dose, affine),
        brain_mask=Volume(brain, affine),
        recurrence_mask=Volume(label, affine),
        prescription_dose_gy=60.0,
    )


def test_voxel_logistic_overfits_tiny_synthetic_dataset():
    case = synthetic_case("P001")
    model = VoxelLogisticModel.fit(
        [case],
        prescription_dose_gy=60.0,
        max_voxels_per_case=512,
        iterations=800,
        learning_rate=0.3,
    )

    risk = model.predict_case(case)

    assert float(risk[case.recurrence_mask.data.astype(bool)].mean()) > 0.8
    assert float(risk[~case.recurrence_mask.data.astype(bool)].mean()) < 0.2


def test_dose_distance_baseline_assigns_higher_risk_to_recurrence_region():
    case = synthetic_case("P001")
    model = DoseDistanceBandModel.fit([case], prescription_dose_gy=60.0, max_voxels_per_case=512)

    risk = model.predict_case(case)

    assert float(risk[case.recurrence_mask.data.astype(bool)].mean()) > float(
        risk[~case.recurrence_mask.data.astype(bool)].mean()
    )
    json.dumps(model.to_dict(), allow_nan=False)


def test_metrics_cover_auprc_dice_and_top_volume_coverage():
    labels = np.asarray([1, 1, 0, 0], dtype=bool)
    scores = np.asarray([0.9, 0.8, 0.2, 0.1], dtype=np.float32)

    assert average_precision_score(labels, scores) == 1.0
    assert dice(labels, scores > 0.5) == 1.0
    assert recurrence_coverage(labels, scores > 0.5) == 1.0
