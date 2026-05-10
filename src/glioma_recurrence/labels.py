"""Reviewed recurrence label ingestion and mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from .constants import BASELINE_T1C, RECURRENCE_MASK_ON_BASELINE
from .geometry import Volume, resample_mask_to_reference
from .nifti import read_volume, write_volume
from .schema import PatientRecord

RegistrationMode = Literal["simpleitk", "affine", "assume-aligned"]


class LabelError(RuntimeError):
    """Raised when label creation would violate review requirements."""


def map_reviewed_mask_to_baseline(
    record: PatientRecord,
    *,
    derived_root: str | Path,
    mask_path: str | Path | None = None,
    assume_baseline_space: bool = False,
    registration_mode: RegistrationMode = "simpleitk",
) -> Path:
    if assume_baseline_space:
        registration_mode = "assume-aligned"
    reviewed_path = Path(mask_path or record.reviewed_recurrence_mask_path)
    if not reviewed_path.exists():
        raise LabelError(
            f"{record.patient_id}: reviewed recurrence mask is required and was not found: {reviewed_path}"
        )
    case_dir = Path(derived_root) / record.patient_id
    baseline_path = case_dir / BASELINE_T1C
    baseline = read_volume(baseline_path)
    if registration_mode == "assume-aligned":
        reviewed = read_volume(reviewed_path)
        mapped = Volume((reviewed.data > 0.5).astype(np.uint8), baseline.affine, reviewed.metadata)
        if reviewed.shape != baseline.shape:
            raise LabelError(
                f"{record.patient_id}: --assume-baseline-space requires mask shape {baseline.shape}; got {reviewed.shape}"
            )
    elif registration_mode == "affine":
        reviewed = read_volume(reviewed_path)
        mapped = resample_mask_to_reference(reviewed, baseline)
    elif registration_mode == "simpleitk":
        reference_value = record.reviewed_recurrence_reference_image_path.strip()
        if not reference_value:
            raise LabelError(
                f"{record.patient_id}: SimpleITK registration requires reviewed_recurrence_reference_image_path"
            )
        reference_path = Path(reference_value)
        if not reference_path.exists():
            raise LabelError(
                f"{record.patient_id}: reviewed recurrence reference image was not found: {reference_path}"
            )
        mapped = _register_and_resample_mask_simpleitk(
            fixed_image_path=baseline_path,
            moving_image_path=reference_path,
            moving_mask_path=reviewed_path,
            fixed_reference=baseline,
        )
    else:
        raise LabelError(f"unsupported registration mode: {registration_mode}")
    output = case_dir / RECURRENCE_MASK_ON_BASELINE
    write_volume(mapped, output, dtype=np.uint8)
    return output


def _register_and_resample_mask_simpleitk(
    *,
    fixed_image_path: Path,
    moving_image_path: Path,
    moving_mask_path: Path,
    fixed_reference: Volume,
) -> Volume:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise LabelError(
            "SimpleITK is required for default MRI-to-MRI label registration; "
            "pass --registration-mode affine or --assume-baseline-space only when that fallback is justified"
        ) from exc

    fixed = sitk.Cast(sitk.ReadImage(str(fixed_image_path)), sitk.sitkFloat32)
    moving = sitk.Cast(sitk.ReadImage(str(moving_image_path)), sitk.sitkFloat32)
    moving_mask = sitk.ReadImage(str(moving_mask_path))

    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.05, seed=13)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=100,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInitialTransform(initial_transform, inPlace=False)
    transform = registration.Execute(fixed, moving)

    resampled = sitk.Resample(
        moving_mask,
        fixed,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    data = np.transpose(sitk.GetArrayFromImage(resampled), (2, 1, 0))
    if data.shape != fixed_reference.shape:
        raise LabelError(
            f"SimpleITK output shape {data.shape} did not match baseline shape {fixed_reference.shape}"
        )
    return Volume((data > 0).astype(np.uint8), fixed_reference.affine, {"registration_mode": "simpleitk"})
