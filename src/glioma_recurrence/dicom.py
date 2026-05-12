"""Read-only DICOM intake helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


class DicomAuditError(RuntimeError):
    """Raised when DICOM audit cannot run."""


@dataclass(frozen=True)
class DicomSeriesRecord:
    patient_key: str
    study_instance_uid: str
    series_instance_uid: str
    modality: str
    sequence_label: str
    series_description: str
    protocol_name: str
    study_date: str
    series_date: str
    manufacturer: str
    scanner_model: str
    magnetic_field_strength: str
    rows: str
    columns: str
    pixel_spacing: str
    slice_thickness: str
    image_orientation_patient: str
    image_type: str
    instance_count: int
    patient_name_present: bool
    patient_birth_date_present: bool
    first_file: str


def audit_dicom_tree(
    dicom_root: str | Path,
    *,
    output_csv: str | Path,
    summary_json: str | Path,
    include_patient_id: bool = False,
    include_paths: bool = False,
    patient_id_salt: str = "glioma-recurrence-risk",
) -> dict[str, object]:
    """Scan DICOM headers and write a series-level inventory.

    Pixel data is not loaded. By default the exported patient key is a stable
    hash of PatientID rather than the raw identifier.
    """

    root = Path(dicom_root)
    if not root.exists():
        raise DicomAuditError(f"DICOM root does not exist: {root}")
    series = summarize_dicom_series(
        root,
        include_patient_id=include_patient_id,
        include_paths=include_paths,
        patient_id_salt=patient_id_salt,
    )
    write_dicom_series_csv(series, output_csv)
    summary = summarize_dicom_inventory(root, series)
    summary_path = Path(summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def summarize_dicom_series(
    dicom_root: str | Path,
    *,
    include_patient_id: bool = False,
    include_paths: bool = False,
    patient_id_salt: str = "glioma-recurrence-risk",
) -> list[DicomSeriesRecord]:
    try:
        import pydicom
    except ImportError as exc:
        raise DicomAuditError("pydicom is required for DICOM audit; install project dependencies with uv sync") from exc

    root = Path(dicom_root)
    grouped: dict[tuple[str, str, str], list[tuple[Path, object]]] = defaultdict(list)
    for path in iter_dicom_candidate_paths(root):
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if not getattr(dataset, "SOPInstanceUID", "") and not getattr(dataset, "SeriesInstanceUID", ""):
            continue
        patient_id = text_value(dataset, "PatientID")
        patient_key = patient_id if include_patient_id else stable_patient_key(patient_id, salt=patient_id_salt)
        study_uid = text_value(dataset, "StudyInstanceUID")
        series_uid = text_value(dataset, "SeriesInstanceUID")
        if not study_uid or not series_uid:
            continue
        grouped[(patient_key, study_uid, series_uid)].append((path, dataset))

    records = []
    for (patient_key, study_uid, series_uid), items in sorted(grouped.items()):
        path, dataset = sorted(items, key=lambda item: str(item[0]))[0]
        series_description = text_value(dataset, "SeriesDescription")
        protocol_name = text_value(dataset, "ProtocolName")
        records.append(
            DicomSeriesRecord(
                patient_key=patient_key,
                study_instance_uid=study_uid,
                series_instance_uid=series_uid,
                modality=text_value(dataset, "Modality"),
                sequence_label=classify_mri_sequence(
                    series_description,
                    protocol_name,
                    text_value(dataset, "SequenceName"),
                    text_value(dataset, "ScanningSequence"),
                    text_value(dataset, "ImageType"),
                ),
                series_description=series_description,
                protocol_name=protocol_name,
                study_date=text_value(dataset, "StudyDate"),
                series_date=text_value(dataset, "SeriesDate"),
                manufacturer=text_value(dataset, "Manufacturer"),
                scanner_model=text_value(dataset, "ManufacturerModelName"),
                magnetic_field_strength=text_value(dataset, "MagneticFieldStrength"),
                rows=text_value(dataset, "Rows"),
                columns=text_value(dataset, "Columns"),
                pixel_spacing=multi_value_text(dataset, "PixelSpacing"),
                slice_thickness=text_value(dataset, "SliceThickness"),
                image_orientation_patient=multi_value_text(dataset, "ImageOrientationPatient"),
                image_type=multi_value_text(dataset, "ImageType"),
                instance_count=len(items),
                patient_name_present=bool(text_value(dataset, "PatientName")),
                patient_birth_date_present=bool(text_value(dataset, "PatientBirthDate")),
                first_file=str(path.relative_to(root)) if include_paths else "",
            )
        )
    return records


def summarize_dicom_inventory(root: Path, series: list[DicomSeriesRecord]) -> dict[str, object]:
    studies: dict[tuple[str, str], set[str]] = defaultdict(set)
    sequence_counts = Counter(record.sequence_label for record in series)
    modality_counts = Counter(record.modality for record in series)
    for record in series:
        studies[(record.patient_key, record.study_instance_uid)].add(record.sequence_label)
    minimum_studies = [
        {"patient_key": patient_key, "study_instance_uid": study_uid}
        for (patient_key, study_uid), labels in studies.items()
        if {"t1c", "flair"}.issubset(labels)
    ]
    full_mp_mri_studies = [
        {"patient_key": patient_key, "study_instance_uid": study_uid}
        for (patient_key, study_uid), labels in studies.items()
        if {"t1", "t1c", "t2", "flair"}.issubset(labels)
    ]
    return {
        "dicom_root": str(root),
        "series_count": len(series),
        "patient_count": len({record.patient_key for record in series}),
        "study_count": len(studies),
        "sequence_counts": dict(sorted(sequence_counts.items())),
        "modality_counts": dict(sorted(modality_counts.items())),
        "studies_with_minimum_t1c_flair": len(minimum_studies),
        "studies_with_full_t1_t1c_t2_flair": len(full_mp_mri_studies),
        "patient_name_present_series": sum(record.patient_name_present for record in series),
        "patient_birth_date_present_series": sum(record.patient_birth_date_present for record in series),
        "minimum_t1c_flair_studies": minimum_studies,
        "full_mp_mri_studies": full_mp_mri_studies,
    }


def write_dicom_series_csv(series: Iterable[DicomSeriesRecord], output_csv: str | Path) -> None:
    records = list(series)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(DicomSeriesRecord.__dataclass_fields__)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def iter_dicom_candidate_paths(root: str | Path) -> Iterable[Path]:
    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        yield path


def classify_mri_sequence(*values: str) -> str:
    text = normalize_text(" ".join(value for value in values if value))
    if not text:
        return "unknown"
    tokens = set(text.split())
    if "flair" in text:
        return "flair"
    if "t2" in text:
        return "t2"
    t1_like = "t1" in text or "mprage" in text or "spgr" in text
    contrast_like = bool(
        tokens
        & {
            "post",
            "gd",
            "gad",
            "gadolinium",
            "contrast",
            "ce",
            "cplus",
            "t1ce",
            "t1gd",
            "t1c",
        }
    )
    if t1_like and contrast_like:
        return "t1c"
    if t1_like:
        return "t1"
    return "unknown"


def stable_patient_key(patient_id: str, *, salt: str) -> str:
    source = patient_id.strip() or "missing-patient-id"
    digest = hashlib.sha256(f"{salt}:{source}".encode("utf-8")).hexdigest()
    return f"patient-{digest[:12]}"


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).replace("c ", "cplus ").strip()


def text_value(dataset: object, keyword: str) -> str:
    value = getattr(dataset, keyword, "")
    if value is None:
        return ""
    return str(value)


def multi_value_text(dataset: object, keyword: str) -> str:
    value = getattr(dataset, keyword, "")
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\\".join(str(item) for item in value)
    return str(value)
