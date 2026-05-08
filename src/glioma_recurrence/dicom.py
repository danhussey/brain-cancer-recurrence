"""DICOM ingestion helpers for MRI series and RTDOSE objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .geometry import GeometryError, Volume, dicom_affine

PHI_TAGS_TO_REMOVE = (
    "PatientName",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
)


class DicomError(RuntimeError):
    """Raised when DICOM ingestion cannot safely proceed."""


def read_dataset(path: str | Path):
    try:
        import pydicom
    except ImportError as exc:
        raise DicomError("pydicom is required for DICOM ingestion") from exc
    return pydicom.dcmread(str(path), force=False)


def find_instances(
    dicom_root: str | Path,
    *,
    series_uid: str | None = None,
    sop_instance_uid: str | None = None,
) -> list[Path]:
    root = Path(dicom_root)
    if not root.exists():
        raise DicomError(f"DICOM root does not exist: {root}")
    matches: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            dataset = read_dataset(path)
        except Exception:
            continue
        if series_uid and getattr(dataset, "SeriesInstanceUID", None) != series_uid:
            continue
        if sop_instance_uid and getattr(dataset, "SOPInstanceUID", None) != sop_instance_uid:
            continue
        matches.append(path)
    if not matches:
        filters = ", ".join(
            item for item in (f"series_uid={series_uid}" if series_uid else "", f"sop_uid={sop_instance_uid}" if sop_instance_uid else "") if item
        )
        raise DicomError(f"no DICOM instances found under {root} for {filters}")
    return sorted(matches)


def deidentify_metadata_summary(dataset, *, patient_id: str) -> dict[str, object]:
    """Return a PHI-light metadata summary for audit sidecars."""

    summary = {
        "patient_id": patient_id,
        "modality": str(getattr(dataset, "Modality", "")),
        "series_instance_uid": str(getattr(dataset, "SeriesInstanceUID", "")),
        "sop_instance_uid": str(getattr(dataset, "SOPInstanceUID", "")),
        "study_instance_uid": str(getattr(dataset, "StudyInstanceUID", "")),
        "frame_of_reference_uid": str(getattr(dataset, "FrameOfReferenceUID", "")),
        "rows": int(getattr(dataset, "Rows", 0) or 0),
        "columns": int(getattr(dataset, "Columns", 0) or 0),
    }
    for tag in PHI_TAGS_TO_REMOVE:
        if getattr(dataset, tag, None):
            summary[f"removed_{tag}"] = True
    return summary


def write_ingest_audit(path: str | Path, summaries: Iterable[dict[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(list(summaries), indent=2, sort_keys=True) + "\n")


def validate_mr_series(paths: Iterable[str | Path]) -> None:
    datasets = [read_dataset(path) for path in paths]
    if not datasets:
        raise DicomError("MR series is empty")
    modalities = {str(getattr(dataset, "Modality", "")) for dataset in datasets}
    if modalities != {"MR"}:
        raise DicomError(f"expected MR series; got modalities {sorted(modalities)}")
    frame_uids = {str(getattr(dataset, "FrameOfReferenceUID", "")) for dataset in datasets}
    if len(frame_uids) != 1:
        raise DicomError("MR series contains multiple FrameOfReferenceUID values")
    orientations = {tuple(float(value) for value in getattr(dataset, "ImageOrientationPatient", [])) for dataset in datasets}
    if len(orientations) != 1:
        raise DicomError("MR series contains inconsistent ImageOrientationPatient values")


def mr_series_to_volume(paths: Iterable[str | Path]) -> Volume:
    datasets = [read_dataset(path) for path in paths]
    validate_mr_series(paths)
    first = datasets[0]
    orientation = [float(value) for value in first.ImageOrientationPatient]
    row_direction = np.asarray(orientation[:3], dtype=float)
    column_direction = np.asarray(orientation[3:], dtype=float)
    normal = np.cross(row_direction, column_direction)

    def slice_position(dataset) -> float:
        position = np.asarray([float(value) for value in dataset.ImagePositionPatient], dtype=float)
        return float(position @ normal)

    datasets.sort(key=slice_position)
    positions = [slice_position(dataset) for dataset in datasets]
    if len(positions) > 1:
        slice_spacing = float(np.median(np.diff(positions)))
    else:
        slice_spacing = float(getattr(first, "SliceThickness", 1.0))
    arrays = []
    for dataset in datasets:
        pixel_array = np.asarray(dataset.pixel_array, dtype=np.float32)
        slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
        arrays.append(pixel_array * slope + intercept)
    data = np.stack(arrays, axis=2).astype(np.float32)
    affine = dicom_affine(
        image_orientation_patient=orientation,
        image_position_patient=[float(value) for value in datasets[0].ImagePositionPatient],
        pixel_spacing=[float(value) for value in first.PixelSpacing],
        slice_spacing=slice_spacing,
    )
    return Volume(
        data=data,
        affine=affine,
        metadata={
            "modality": "MR",
            "series_instance_uid": str(getattr(first, "SeriesInstanceUID", "")),
            "frame_of_reference_uid": str(getattr(first, "FrameOfReferenceUID", "")),
        },
    )


def rt_dose_array_gy(dataset) -> np.ndarray:
    units = str(getattr(dataset, "DoseUnits", "")).upper()
    if units != "GY":
        raise DicomError(f"RTDOSE DoseUnits must be GY for physical-dose ingestion; got {units!r}")
    scaling = float(getattr(dataset, "DoseGridScaling", 1.0))
    if scaling <= 0:
        raise DicomError(f"DoseGridScaling must be positive; got {scaling}")
    array = np.asarray(dataset.pixel_array, dtype=np.float32)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    if array.ndim != 3:
        raise DicomError(f"RTDOSE pixel_array must be 2D or 3D; got shape {array.shape}")
    # pydicom exposes RT Dose frames as (frame, row, column); internal volumes use
    # (row, column, slice).
    return np.moveaxis(array * scaling, 0, 2).astype(np.float32)


def rtdose_affine(dataset) -> np.ndarray:
    offsets = [float(value) for value in getattr(dataset, "GridFrameOffsetVector", [0.0])]
    if len(offsets) > 1:
        diffs = np.diff(offsets)
        if not np.allclose(diffs, diffs[0], atol=1e-3):
            raise GeometryError("non-uniform RTDOSE GridFrameOffsetVector is not supported")
        slice_spacing = float(diffs[0])
    else:
        slice_spacing = float(getattr(dataset, "SliceThickness", 1.0) or 1.0)
    return dicom_affine(
        image_orientation_patient=[float(value) for value in dataset.ImageOrientationPatient],
        image_position_patient=[float(value) for value in dataset.ImagePositionPatient],
        pixel_spacing=[float(value) for value in dataset.PixelSpacing],
        slice_spacing=slice_spacing,
    )


def rtdose_to_volume(path: str | Path) -> Volume:
    dataset = read_dataset(path)
    modality = str(getattr(dataset, "Modality", ""))
    if modality != "RTDOSE":
        raise DicomError(f"expected RTDOSE modality; got {modality!r}")
    data = rt_dose_array_gy(dataset)
    affine = rtdose_affine(dataset)
    return Volume(
        data=data,
        affine=affine,
        metadata={
            "modality": "RTDOSE",
            "sop_instance_uid": str(getattr(dataset, "SOPInstanceUID", "")),
            "frame_of_reference_uid": str(getattr(dataset, "FrameOfReferenceUID", "")),
            "dose_units": "Gy",
        },
    )

