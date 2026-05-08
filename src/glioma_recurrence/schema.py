"""Manifest parsing and patient-level split validation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

from .constants import ALLOWED_SPLITS, REQUIRED_MANIFEST_COLUMNS

CONFIRMED_ADJUDICATION_TERMS = {
    "confirmed",
    "clinically_confirmed",
    "histologically_confirmed",
    "pathology_confirmed",
    "rano_confirmed",
}

PSEUDOPROGRESSION_WINDOW_DAYS = 90


class ManifestError(ValueError):
    """Raised when `patients.csv` violates the public data contract."""


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    baseline_scan_date: date
    baseline_t1c_series_uid: str
    baseline_flair_series_uid: str
    rtdose_sop_instance_uid: str
    recurrence_scan_date: date | None
    recurrence_adjudication: str
    reviewed_recurrence_mask_path: str
    split: str
    radiotherapy_end_date: date | None = None
    prescription_dose_gy: float | None = None

    @property
    def normalized_split(self) -> str:
        return "validation" if self.split == "val" else self.split

    @property
    def is_confirmed_recurrence(self) -> bool:
        return normalize_adjudication(self.recurrence_adjudication) in CONFIRMED_ADJUDICATION_TERMS

    @property
    def is_pseudoprogression_window(self) -> bool:
        if self.radiotherapy_end_date is None or self.recurrence_scan_date is None:
            return False
        delta = (self.recurrence_scan_date - self.radiotherapy_end_date).days
        return 0 <= delta <= PSEUDOPROGRESSION_WINDOW_DAYS

    @property
    def should_exclude_from_training(self) -> bool:
        return self.is_pseudoprogression_window and not self.is_confirmed_recurrence


def normalize_adjudication(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def parse_date(value: str, *, column: str) -> date | None:
    value = value.strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise ManifestError(f"{column} must be YYYY-MM-DD; got {value!r}")


def parse_optional_float(value: str, *, column: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ManifestError(f"{column} must be numeric; got {value!r}") from exc


def read_manifest(path: str | Path) -> list[PatientRecord]:
    manifest_path = Path(path)
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ManifestError(f"{manifest_path} is empty")
        missing = sorted(set(REQUIRED_MANIFEST_COLUMNS) - set(reader.fieldnames))
        if missing:
            raise ManifestError(f"{manifest_path} is missing required columns: {', '.join(missing)}")
        records = [_row_to_record(row, row_number=index + 2) for index, row in enumerate(reader)]
    validate_patient_level_splits(records)
    return records


def _row_to_record(row: dict[str, str], *, row_number: int) -> PatientRecord:
    split = row["split"].strip().lower()
    if split not in ALLOWED_SPLITS:
        allowed = ", ".join(sorted(ALLOWED_SPLITS))
        raise ManifestError(f"row {row_number}: split must be one of {allowed}; got {split!r}")
    patient_id = row["patient_id"].strip()
    if not patient_id:
        raise ManifestError(f"row {row_number}: patient_id is required")

    return PatientRecord(
        patient_id=patient_id,
        baseline_scan_date=parse_required_date(row["baseline_scan_date"], "baseline_scan_date", row_number),
        baseline_t1c_series_uid=require_text(row, "baseline_t1c_series_uid", row_number),
        baseline_flair_series_uid=require_text(row, "baseline_flair_series_uid", row_number),
        rtdose_sop_instance_uid=require_text(row, "rtdose_sop_instance_uid", row_number),
        recurrence_scan_date=parse_date(row.get("recurrence_scan_date", ""), column="recurrence_scan_date"),
        recurrence_adjudication=require_text(row, "recurrence_adjudication", row_number),
        reviewed_recurrence_mask_path=row.get("reviewed_recurrence_mask_path", "").strip(),
        split=split,
        radiotherapy_end_date=parse_date(row.get("radiotherapy_end_date", ""), column="radiotherapy_end_date"),
        prescription_dose_gy=parse_optional_float(row.get("prescription_dose_gy", ""), column="prescription_dose_gy"),
    )


def parse_required_date(value: str, column: str, row_number: int) -> date:
    parsed = parse_date(value, column=column)
    if parsed is None:
        raise ManifestError(f"row {row_number}: {column} is required")
    return parsed


def require_text(row: dict[str, str], column: str, row_number: int) -> str:
    value = row[column].strip()
    if not value:
        raise ManifestError(f"row {row_number}: {column} is required")
    return value


def validate_patient_level_splits(records: Iterable[PatientRecord]) -> None:
    patient_to_split: dict[str, str] = {}
    for record in records:
        existing = patient_to_split.get(record.patient_id)
        if existing is None:
            patient_to_split[record.patient_id] = record.normalized_split
            continue
        if existing != record.normalized_split:
            raise ManifestError(
                "patient-level leakage: "
                f"{record.patient_id!r} appears in both {existing!r} and {record.normalized_split!r}"
            )


def filter_records(
    records: Iterable[PatientRecord],
    *,
    splits: set[str] | None = None,
    include_pseudoprogression: bool = False,
) -> list[PatientRecord]:
    selected: list[PatientRecord] = []
    normalized_splits = {"validation" if split == "val" else split for split in splits} if splits else None
    for record in records:
        if normalized_splits and record.normalized_split not in normalized_splits:
            continue
        if record.should_exclude_from_training and not include_pseudoprogression:
            continue
        selected.append(record)
    return selected

