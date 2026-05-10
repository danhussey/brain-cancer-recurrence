#!/usr/bin/env python3
"""Prepare UCSD-PTGBM NIfTI data for MRI-only recurrence modeling.

The adapter expects the user to download and extract UCSD-PTGBM onto external
storage first. It copies selected baseline MRI, baseline tumor mask, follow-up
T1c reference image, and follow-up recurrence/tumor mask into a mutable derived
workspace. It does not symlink downloaded data into `derived/`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree

import numpy as np

from glioma_recurrence.constants import BASELINE_FLAIR, BASELINE_T1C, BASELINE_TUMOR_MASK
from glioma_recurrence.geometry import Volume
from glioma_recurrence.nifti import read_volume, write_volume


SUBJECT_COLUMNS = ("subject_id", "subject", "patient_id", "participant_id", "case_id")
TIMEPOINT_COLUMNS = ("timepoint_id", "timepoint", "study_id", "scan_id", "visit", "session")
COMBINED_TIMEPOINT_COLUMNS = ("id", "acquisition_id", "scan_timepoint_id", "tcia_id")
DATE_COLUMNS = ("scan_date", "study_date", "exam_date", "date", "days_from_surgery")
STATUS_COLUMNS = ("progression_status", "tumor_status", "response_status", "status", "diagnosis", "label")
NEGATIVE_DX_COLUMNS = ("dx", "diagnosis", "negative_case_category", "category")
POSITIVE_STATUS_TERMS = ("residual", "recurrent", "recurrence", "progression", "progressive", "tumor")
EXCLUDED_STATUS_TERMS = (
    "ambiguous",
    "unable",
    "indeterminate",
    "pseudoprogression",
    "radiation necrosis",
    "post treatment change",
    "post-treatment change",
    "treatment related",
    "treatment-related",
    "non-specific",
    "nonspecific",
)
CONFIRMED_STATUS_TERMS = ("pathology", "histolog", "clinically confirmed", "confirmed")
BRATS_TARGET_LABELS = {1, 3, 4}
CLINICAL_POSITIVE_STATUS = "residual/recurrent tumor"
IMAGING_ONLY_RECURRENCE_STATUS = "imaging-only follow-up tumor segmentation present"
IMAGING_ONLY_RECURRENCE_ADJUDICATION = "imaging_followup_segmentation_present"


@dataclass(frozen=True)
class ClinicalTimepoint:
    subject_id: str
    timepoint_id: str
    scan_date: date | None
    status: str
    row_index: int


@dataclass(frozen=True)
class MaskSource:
    paths: tuple[Path, ...]
    label_values: frozenset[int] | None


@dataclass(frozen=True)
class UcsdTimepointFiles:
    t1c: Path
    flair: Path
    mask: MaskSource


@dataclass(frozen=True)
class UcsdPair:
    subject_id: str
    patient_id: str
    baseline: ClinicalTimepoint
    recurrence: ClinicalTimepoint
    baseline_files: UcsdTimepointFiles
    recurrence_files: UcsdTimepointFiles
    recurrence_adjudication: str


@dataclass(frozen=True)
class PreparedUcsdDataset:
    manifest: Path
    derived_root: Path
    masks_root: Path
    label_refs_root: Path
    selected_subjects: list[str]
    skipped_subjects: dict[str, str]


def prepare_ucsd_dataset(
    source_root: str | Path,
    clinical_table: str | Path | None,
    output_root: str | Path,
    *,
    max_subjects: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    allow_imaging_only_labels: bool = False,
    negative_cases_table: str | Path | None = None,
    include_negative_controls: bool = False,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    split_seed: int = 13,
) -> PreparedUcsdDataset:
    source = Path(source_root)
    niftis = sorted(find_nifti_files(source))
    negative_cases = read_negative_case_categories(negative_cases_table) if negative_cases_table is not None else {}
    if clinical_table is None:
        if not allow_imaging_only_labels:
            raise RuntimeError(
                "UCSD clinical table is required unless --allow-imaging-only-labels is passed. "
                "The imaging-only mode is provisional and uses later tumor segmentations as labels without "
                "clinical progression adjudication."
        )
        rows = infer_timepoints_from_filenames(niftis)
    else:
        rows = read_clinical_timepoints(clinical_table, negative_cases=negative_cases)
    pairs, skipped = select_pairs(rows, niftis, max_subjects=max_subjects, include_negative_controls=include_negative_controls)
    if not pairs:
        raise RuntimeError("no UCSD-PTGBM subjects had an eligible baseline and later recurrence label")

    output = Path(output_root)
    derived_root = output / "derived"
    masks_root = output / "masks"
    label_refs_root = output / "label_refs"
    if not dry_run:
        for directory in (derived_root, masks_root, label_refs_root, output / "models", output / "reports"):
            directory.mkdir(parents=True, exist_ok=True)

    split_by_subject = assign_patient_level_splits(
        pairs,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=split_seed,
    )
    manifest_rows: list[dict[str, str]] = []
    for index, pair in enumerate(pairs):
        split = split_by_subject[pair.subject_id]
        case_dir = derived_root / pair.patient_id
        reviewed_mask_path = masks_root / f"{pair.patient_id}_followup_recurrence_mask.nii.gz"
        label_ref_path = label_refs_root / f"{pair.patient_id}_followup_t1c.nii.gz"
        if not dry_run:
            case_dir.mkdir(parents=True, exist_ok=True)
            copy_file(pair.baseline_files.t1c, case_dir / BASELINE_T1C, overwrite=overwrite)
            copy_file(pair.baseline_files.flair, case_dir / BASELINE_FLAIR, overwrite=overwrite)
            write_mask(pair.baseline_files.mask, case_dir / BASELINE_TUMOR_MASK, overwrite=overwrite)
            copy_file(pair.recurrence_files.t1c, label_ref_path, overwrite=overwrite)
            if is_negative_control_adjudication(pair.recurrence_adjudication):
                write_empty_mask_like(pair.recurrence_files.t1c, reviewed_mask_path, overwrite=overwrite)
            else:
                write_mask(pair.recurrence_files.mask, reviewed_mask_path, overwrite=overwrite)

        manifest_rows.append(
            {
                "patient_id": pair.patient_id,
                "baseline_scan_date": date_for_manifest(pair.baseline),
                "baseline_t1c_series_uid": f"ucsd-ptgbm-{pair.subject_id}-{pair.baseline.timepoint_id}-t1c",
                "baseline_flair_series_uid": f"ucsd-ptgbm-{pair.subject_id}-{pair.baseline.timepoint_id}-flair",
                "recurrence_scan_date": date_for_manifest(pair.recurrence),
                "recurrence_adjudication": pair.recurrence_adjudication,
                "reviewed_recurrence_mask_path": str(reviewed_mask_path),
                "split": split,
                "reviewed_recurrence_reference_image_path": str(label_ref_path),
                "source_dataset": "UCSD-PTGBM",
                "baseline_timepoint_id": pair.baseline.timepoint_id,
                "recurrence_timepoint_id": pair.recurrence.timepoint_id,
                "radiotherapy_end_date": "",
            }
        )

    manifest = output / "patients.csv"
    if not dry_run:
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)
        summary = {
            "source_root": str(source),
            "clinical_table": str(clinical_table) if clinical_table is not None else "",
            "negative_cases_table": str(negative_cases_table) if negative_cases_table is not None else "",
            "negative_case_count": len(negative_cases),
            "include_negative_controls": include_negative_controls,
            "selected_positive_subjects": [
                pair.subject_id for pair in pairs if not is_negative_control_adjudication(pair.recurrence_adjudication)
            ],
            "selected_negative_control_subjects": [
                pair.subject_id for pair in pairs if is_negative_control_adjudication(pair.recurrence_adjudication)
            ],
            "allow_imaging_only_labels": allow_imaging_only_labels,
            "manifest": str(manifest),
            "derived_root": str(derived_root),
            "selected_subjects": [pair.subject_id for pair in pairs],
            "skipped_subjects": skipped,
            "split_counts": split_counts(split_by_subject.values()),
            "split_strategy": {
                "train_fraction": train_fraction,
                "validation_fraction": validation_fraction,
                "test_fraction": test_fraction(train_fraction, validation_fraction),
                "split_seed": split_seed,
            },
            "label_policy": (
                "Prefer total cellular tumor masks; otherwise combine enhancing and non-enhancing/core masks. "
                "FLAIR-only edema and resection cavity masks are excluded."
            ),
        }
        (output / "ucsd_prepare_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    return PreparedUcsdDataset(
        manifest=manifest,
        derived_root=derived_root,
        masks_root=masks_root,
        label_refs_root=label_refs_root,
        selected_subjects=[pair.subject_id for pair in pairs],
        skipped_subjects=skipped,
    )


def infer_timepoints_from_filenames(niftis: list[Path]) -> list[ClinicalTimepoint]:
    """Build provisional rows when only UCSD image folders have been downloaded."""

    by_subject: dict[str, set[str]] = {}
    for path in niftis:
        identifiers = infer_subject_timepoint(path)
        if identifiers is None:
            continue
        subject_id, timepoint_id = identifiers
        by_subject.setdefault(subject_id, set()).add(timepoint_id)

    rows: list[ClinicalTimepoint] = []
    for subject_id in sorted(by_subject, key=natural_key):
        for index, timepoint_id in enumerate(sorted(by_subject[subject_id], key=natural_key)):
            rows.append(
                ClinicalTimepoint(
                    subject_id=subject_id,
                    timepoint_id=timepoint_id,
                    scan_date=date(1900, 1, min(index + 1, 28)),
                    status="imaging-only baseline" if index == 0 else IMAGING_ONLY_RECURRENCE_STATUS,
                    row_index=len(rows) + 1,
                )
            )
    return rows


def infer_subject_timepoint(path: Path) -> tuple[str, str] | None:
    text = str(path)
    real_match = re.search(r"(UCSD-PTGBM-\d{4})_(\d+)", text)
    if real_match:
        return real_match.group(1), real_match.group(2)
    parts = path.parts
    if len(parts) < 3:
        return None
    return parts[-3], parts[-2]


def infer_subject_timepoint_from_identifier(identifier: str) -> tuple[str, str] | None:
    value = identifier.strip()
    real_match = re.fullmatch(r"(UCSD-PTGBM-\d{4})_(\d+)", value)
    if real_match:
        return real_match.group(1), real_match.group(2)
    if "_" not in value:
        return None
    subject_id, timepoint_id = value.rsplit("_", 1)
    if not subject_id or not timepoint_id:
        return None
    return subject_id, timepoint_id


def read_clinical_timepoints(
    path: str | Path,
    *,
    negative_cases: dict[str, str] | None = None,
) -> list[ClinicalTimepoint]:
    rows = read_table(path)
    if not rows:
        raise RuntimeError(f"{path} did not contain any clinical rows")
    negative_cases = negative_cases or {}
    subject_column = choose_optional_column(rows[0], SUBJECT_COLUMNS)
    timepoint_column = choose_optional_column(rows[0], TIMEPOINT_COLUMNS)
    combined_column = choose_optional_column(rows[0], COMBINED_TIMEPOINT_COLUMNS)
    if subject_column is None or timepoint_column is None:
        if combined_column is None:
            raise RuntimeError(
                "clinical table is missing subject/timepoint columns; tried "
                f"subject columns {', '.join(SUBJECT_COLUMNS)}, timepoint columns {', '.join(TIMEPOINT_COLUMNS)}, "
                f"or combined columns {', '.join(COMBINED_TIMEPOINT_COLUMNS)}"
            )
    status_column = choose_optional_column(rows[0], STATUS_COLUMNS)
    if status_column is None and combined_column is None:
        raise RuntimeError(
            f"clinical table is missing a progression/status column; tried {', '.join(STATUS_COLUMNS)}"
        )
    date_column = choose_optional_column(rows[0], DATE_COLUMNS)
    timepoints: list[ClinicalTimepoint] = []
    for row_index, row in enumerate(rows, start=2):
        acquisition_id = row.get(combined_column, "").strip() if combined_column else ""
        if subject_column is not None and timepoint_column is not None:
            subject_id = row.get(subject_column, "").strip()
            timepoint_id = row.get(timepoint_column, "").strip()
        else:
            parsed = infer_subject_timepoint_from_identifier(acquisition_id)
            if parsed is None:
                continue
            subject_id, timepoint_id = parsed
        if not subject_id or not timepoint_id:
            continue
        status = row.get(status_column, "").strip() if status_column else ""
        if not status and acquisition_id:
            status = negative_cases.get(acquisition_id, CLINICAL_POSITIVE_STATUS)
        timepoints.append(
            ClinicalTimepoint(
                subject_id=subject_id,
                timepoint_id=timepoint_id,
                scan_date=parse_scan_date(row.get(date_column, "")) if date_column else None,
                status=status,
                row_index=row_index,
            )
        )
    return timepoints


def read_negative_case_categories(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    rows = read_table(path)
    if not rows:
        return {}
    id_column = choose_column(rows[0], COMBINED_TIMEPOINT_COLUMNS, "negative case ID")
    dx_column = choose_optional_column(rows[0], NEGATIVE_DX_COLUMNS)
    categories: dict[str, str] = {}
    for row in rows:
        acquisition_id = row.get(id_column, "").strip()
        if not acquisition_id:
            continue
        categories[acquisition_id] = row.get(dx_column, "").strip() if dx_column else "negative_case"
    return categories


def read_table(path: str | Path) -> list[dict[str, str]]:
    table_path = Path(path)
    suffix = table_path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with table_path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            return [normalize_row(row) for row in reader]
    if suffix == ".xlsx":
        return read_xlsx_table(table_path)
    raise RuntimeError(f"unsupported clinical table format: {table_path}")


def read_xlsx_table(path: Path) -> list[dict[str, str]]:
    tables = read_xlsx_tables(path)
    if not tables:
        return []
    for rows in tables:
        if rows and choose_optional_column(rows[0], COMBINED_TIMEPOINT_COLUMNS) is not None:
            return rows
    return tables[0]


def read_xlsx_tables(path: Path) -> list[list[dict[str, str]]]:
    with zipfile.ZipFile(path) as workbook:
        shared_strings = read_shared_strings(workbook)
        sheet_names = sheet_xml_names(workbook)
        tables = [read_xlsx_sheet(workbook, sheet_name, shared_strings) for sheet_name in sheet_names]
    return [table for table in tables if table]


def read_xlsx_sheet(
    workbook: zipfile.ZipFile,
    sheet_name: str,
    shared_strings: list[str],
) -> list[dict[str, str]]:
    sheet = ElementTree.fromstring(workbook.read(sheet_name))
    rows: list[list[str]] = []
    for row in sheet.findall(".//{*}sheetData/{*}row"):
        cells: dict[int, str] = {}
        for cell in row.findall("{*}c"):
            ref = cell.attrib.get("r", "")
            index = column_index_from_ref(ref)
            cells[index] = xlsx_cell_text(cell, shared_strings)
        if cells:
            max_index = max(cells)
            rows.append([cells.get(index, "") for index in range(max_index + 1)])
    if not rows:
        return []
    headers = [normalize_key(value) for value in rows[0]]
    return [
        {header: row[index].strip() if index < len(row) else "" for index, header in enumerate(headers)}
        for row in rows[1:]
    ]


def read_shared_strings(workbook: zipfile.ZipFile) -> list[str]:
    try:
        root = ElementTree.fromstring(workbook.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    values: list[str] = []
    for item in root.findall("{*}si"):
        values.append("".join(text.text or "" for text in item.findall(".//{*}t")))
    return values


def sheet_xml_names(workbook: zipfile.ZipFile) -> list[str]:
    if "xl/worksheets/sheet1.xml" in workbook.namelist():
        names = [
            name
            for name in workbook.namelist()
            if name.startswith("xl/worksheets/") and name.endswith(".xml")
        ]
        return sorted(names, key=natural_key)
    names = [
        name
        for name in workbook.namelist()
        if name.startswith("xl/worksheets/") and name.endswith(".xml")
    ]
    if not names:
        raise RuntimeError("xlsx workbook did not contain a worksheet")
    return sorted(names, key=natural_key)


def xlsx_cell_text(cell: ElementTree.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//{*}t"))
    value = cell.find("{*}v")
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        index = int(value.text)
        return shared_strings[index] if index < len(shared_strings) else ""
    return value.text


def column_index_from_ref(ref: str) -> int:
    letters = "".join(char for char in ref if char.isalpha()).upper()
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return max(index - 1, 0)


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    return {normalize_key(key): (value or "") for key, value in row.items()}


def normalize_key(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")


def choose_column(row: dict[str, str], candidates: tuple[str, ...], label: str) -> str:
    column = choose_optional_column(row, candidates)
    if column is None:
        raise RuntimeError(f"clinical table is missing a {label} column; tried {', '.join(candidates)}")
    return column


def choose_optional_column(row: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in row:
            return candidate
    return None


def parse_scan_date(value: str) -> date | None:
    value = value.strip()
    if not value:
        return None
    if value.replace(".", "", 1).isdigit():
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    return None


def find_nifti_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and (path.name.endswith(".nii") or path.name.endswith(".nii.gz")) and path.stat().st_size > 0
    ]


def select_pairs(
    rows: list[ClinicalTimepoint],
    niftis: list[Path],
    *,
    max_subjects: int | None,
    include_negative_controls: bool = False,
) -> tuple[list[UcsdPair], dict[str, str]]:
    by_subject: dict[str, list[ClinicalTimepoint]] = {}
    for row in rows:
        by_subject.setdefault(row.subject_id, []).append(row)

    pairs: list[UcsdPair] = []
    skipped: dict[str, str] = {}
    for subject_id in sorted(by_subject):
        timepoints = sorted(by_subject[subject_id], key=timepoint_sort_key)
        usable: list[tuple[ClinicalTimepoint, UcsdTimepointFiles]] = []
        for timepoint in timepoints:
            files = match_timepoint_files(niftis, timepoint)
            if files is not None:
                usable.append((timepoint, files))
        if len(usable) < 2:
            skipped[subject_id] = "fewer than two complete MRI+mask timepoints"
            continue
        selected_pair = choose_pair(
            subject_id,
            usable,
            len(pairs),
            include_negative_controls=include_negative_controls,
        )
        if selected_pair is None:
            skipped[subject_id] = "no later residual/recurrent tumor timepoint"
            continue
        pairs.append(selected_pair)
        if max_subjects is not None and len(pairs) >= max_subjects:
            break
    return pairs, skipped


def choose_pair(
    subject_id: str,
    usable: list[tuple[ClinicalTimepoint, UcsdTimepointFiles]],
    pair_index: int,
    *,
    include_negative_controls: bool = False,
) -> UcsdPair | None:
    first_negative_control: UcsdPair | None = None
    for baseline_index, (baseline, baseline_files) in enumerate(usable[:-1]):
        for recurrence, recurrence_files in usable[baseline_index + 1 :]:
            adjudication = recurrence_adjudication(recurrence.status)
            if adjudication is None:
                negative_adjudication = (
                    negative_control_adjudication(recurrence.status) if include_negative_controls else None
                )
                if negative_adjudication is not None and first_negative_control is None:
                    patient_id = f"UCSD{pair_index + 1:04d}_{safe_id(subject_id)}"
                    first_negative_control = UcsdPair(
                        subject_id=subject_id,
                        patient_id=patient_id,
                        baseline=baseline,
                        recurrence=recurrence,
                        baseline_files=baseline_files,
                        recurrence_files=recurrence_files,
                        recurrence_adjudication=negative_adjudication,
                    )
                continue
            patient_id = f"UCSD{pair_index + 1:04d}_{safe_id(subject_id)}"
            return UcsdPair(
                subject_id=subject_id,
                patient_id=patient_id,
                baseline=baseline,
                recurrence=recurrence,
                baseline_files=baseline_files,
                recurrence_files=recurrence_files,
                recurrence_adjudication=adjudication,
            )
    return first_negative_control


def timepoint_sort_key(timepoint: ClinicalTimepoint) -> tuple[int, object]:
    if timepoint.scan_date is not None:
        return (0, timepoint.scan_date)
    return (1, natural_key(timepoint.timepoint_id))


def recurrence_adjudication(status: str) -> str | None:
    normalized = status.lower()
    if normalize_identifier(normalized) in {
        IMAGING_ONLY_RECURRENCE_ADJUDICATION,
        normalize_identifier(IMAGING_ONLY_RECURRENCE_STATUS),
    }:
        return IMAGING_ONLY_RECURRENCE_ADJUDICATION
    has_positive = any(term in normalized for term in POSITIVE_STATUS_TERMS)
    has_excluded = any(term in normalized for term in EXCLUDED_STATUS_TERMS)
    if not has_positive or has_excluded:
        return None
    if "pathology" in normalized or "histolog" in normalized:
        return "pathology_confirmed"
    if any(term in normalized for term in CONFIRMED_STATUS_TERMS):
        return "clinically_confirmed"
    return "confirmed"


def negative_control_adjudication(status: str) -> str | None:
    normalized = normalize_identifier(status)
    if normalized in {"ns", "non_specific", "nonspecific"}:
        return "clinical_negative_non_specific"
    if "pseudoprogression" in normalized or normalized.startswith("psp") or "_psp_" in f"_{normalized}_":
        return "clinical_negative_pseudoprogression"
    if "radiation_necrosis" in normalized:
        return "clinical_negative_radiation_necrosis"
    return None


def is_negative_control_adjudication(value: str) -> bool:
    return normalize_identifier(value).startswith("clinical_negative_")


def match_timepoint_files(niftis: list[Path], timepoint: ClinicalTimepoint) -> UcsdTimepointFiles | None:
    candidates = [
        path
        for path in niftis
        if contains_identifier(path, timepoint.subject_id) and contains_identifier(path, timepoint.timepoint_id)
    ]
    if not candidates:
        return None
    t1c = choose_first(candidates, is_t1c_file)
    flair = choose_first(candidates, is_flair_file)
    mask = choose_mask_source(candidates)
    if t1c is None or flair is None or mask is None:
        return None
    return UcsdTimepointFiles(t1c=t1c, flair=flair, mask=mask)


def choose_mask_source(paths: list[Path]) -> MaskSource | None:
    preferred = choose_first(paths, is_total_cellular_tumor_mask)
    if preferred is not None:
        return MaskSource(paths=(preferred,), label_values=None)
    compartments = tuple(path for path in paths if is_tumor_core_compartment_mask(path))
    if compartments:
        return MaskSource(paths=compartments, label_values=None)
    segmentation = choose_first(paths, is_brats_like_segmentation)
    if segmentation is not None:
        return MaskSource(paths=(segmentation,), label_values=frozenset(BRATS_TARGET_LABELS))
    return None


def choose_first(paths: list[Path], predicate) -> Path | None:
    matches = [path for path in paths if predicate(path)]
    return sorted(matches, key=lambda path: natural_key(str(path)))[0] if matches else None


def is_t1c_file(path: Path) -> bool:
    text = normalized_path_text(path)
    if is_mask_like(path):
        return False
    return any(token in text for token in ("t1c", "t1ce", "t1gd", "t1_post", "t1post", "postcontrast", "post_contrast"))


def is_flair_file(path: Path) -> bool:
    text = normalized_path_text(path)
    return "flair" in text and not is_mask_like(path)


def is_total_cellular_tumor_mask(path: Path) -> bool:
    text = normalized_path_text(path)
    tokens = path_tokens(path)
    return (
        "tct" in tokens
        or "total_cellular_tumor" in text
        or "totalcellulartumor" in text
        or "cellular_tumor" in text
    )


def is_tumor_core_compartment_mask(path: Path) -> bool:
    text = normalized_path_text(path)
    tokens = path_tokens(path)
    if any(token in tokens for token in ("flair", "edema", "oedema", "cavity", "resection")):
        return False
    return (
        any(token in tokens for token in ("enhancing", "ect", "et", "necrotic", "core", "ncr", "net", "nect"))
        or "nonenhancing" in text
        or "non_enhancing" in text
    )


def is_brats_like_segmentation(path: Path) -> bool:
    text = normalized_path_text(path)
    return any(token in text for token in ("seg", "segmentation", "mask")) and not any(
        token in text for token in ("flair", "edema", "oedema", "cavity", "resection")
    )


def is_mask_like(path: Path) -> bool:
    text = normalized_path_text(path)
    tokens = path_tokens(path)
    return (
        any(token in tokens for token in ("seg", "mask", "tumor", "tumour", "label", "tct", "ect", "nect", "gtv"))
        or "cellular_tumor" in text
    )


def contains_identifier(path: Path, identifier: str) -> bool:
    return normalize_identifier(identifier) in normalized_path_text(path)


def normalized_path_text(path: Path) -> str:
    name = str(path).lower()
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    return re.sub(r"[^a-z0-9]+", "_", name)


def path_tokens(path: Path) -> set[str]:
    return {token for token in normalized_path_text(path).split("_") if token}


def normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def safe_id(value: str) -> str:
    return normalize_identifier(value).upper()


def date_for_manifest(timepoint: ClinicalTimepoint) -> str:
    if timepoint.scan_date is not None:
        return timepoint.scan_date.isoformat()
    numbers = [part for part in re.split(r"(\d+)", timepoint.timepoint_id) if part.isdigit()]
    day = min(max(int(numbers[-1]) if numbers else 1, 1), 28)
    return date(1900, 1, day).isoformat()


def assign_patient_level_splits(
    pairs: list[UcsdPair],
    *,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> dict[str, str]:
    test = test_fraction(train_fraction, validation_fraction)
    if train_fraction <= 0 or validation_fraction < 0 or test < 0:
        raise ValueError("split fractions must be non-negative and train_fraction must be positive")
    if not pairs:
        return {}

    counts = patient_level_split_counts(len(pairs), train_fraction=train_fraction, validation_fraction=validation_fraction)
    ordered = sorted(
        pairs,
        key=lambda pair: (stable_split_key(pair.subject_id, seed), natural_key(pair.subject_id)),
    )
    split_by_subject: dict[str, str] = {}
    train_end = counts["train"]
    validation_end = train_end + counts["validation"]
    for index, pair in enumerate(ordered):
        if index < train_end:
            split = "train"
        elif index < validation_end:
            split = "validation"
        else:
            split = "test"
        split_by_subject[pair.subject_id] = split
    return split_by_subject


def patient_level_split_counts(total: int, *, train_fraction: float, validation_fraction: float) -> dict[str, int]:
    if total < 1:
        return {"train": 0, "validation": 0, "test": 0}
    if total == 1:
        return {"train": 1, "validation": 0, "test": 0}
    if total == 2:
        return {"train": 1, "validation": 1, "test": 0}

    test = test_fraction(train_fraction, validation_fraction)
    validation_count = max(1, int(round(total * validation_fraction)))
    test_count = max(1, int(round(total * test)))
    while validation_count + test_count > total - 1:
        if validation_count >= test_count and validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break
    train_count = total - validation_count - test_count
    return {"train": train_count, "validation": validation_count, "test": test_count}


def test_fraction(train_fraction: float, validation_fraction: float) -> float:
    return 1.0 - train_fraction - validation_fraction


def stable_split_key(subject_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{subject_id}".encode("utf-8")).hexdigest()


def split_counts(splits: Iterable[str]) -> dict[str, int]:
    counts = {"train": 0, "validation": 0, "test": 0}
    for split in splits:
        counts[split] = counts.get(split, 0) + 1
    return counts


def copy_file(source: Path, target: Path, *, overwrite: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists; pass --overwrite to replace it")
    shutil.copy2(source, target)


def write_mask(source: MaskSource, target: Path, *, overwrite: bool) -> None:
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists; pass --overwrite to replace it")
    combined: np.ndarray | None = None
    affine: np.ndarray | None = None
    for path in source.paths:
        volume = read_volume(path)
        data = np.rint(volume.data).astype(np.int16)
        binary = np.isin(data, list(source.label_values)) if source.label_values is not None else data > 0
        if combined is None:
            combined = binary
            affine = volume.affine
        else:
            if combined.shape != binary.shape:
                raise RuntimeError(f"mask component shape mismatch while combining {source.paths}")
            combined |= binary
    if combined is None or affine is None:
        raise RuntimeError("mask source did not contain any readable masks")
    write_volume(Volume(combined.astype(np.uint8), affine), target, dtype=np.uint8)


def write_empty_mask_like(reference: Path, target: Path, *, overwrite: bool) -> None:
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists; pass --overwrite to replace it")
    volume = read_volume(reference)
    write_volume(Volume(np.zeros(volume.shape, dtype=np.uint8), volume.affine), target, dtype=np.uint8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="Extracted UCSD-PTGBM NIfTI root")
    parser.add_argument("--clinical-table", default=None, help="UCSD clinical CSV/TSV/XLSX table")
    parser.add_argument("--negative-cases-table", default=None, help="Optional UCSD negative-case category table")
    parser.add_argument(
        "--include-negative-controls",
        action="store_true",
        help="Include subjects whose later clinical categories are PsP, radiation necrosis, or non-specific as empty-label controls.",
    )
    parser.add_argument("--output-root", required=True, help="External workspace for copied derivatives")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-imaging-only-labels",
        action="store_true",
        help=(
            "Infer longitudinal pairs from image filenames when the clinical table is unavailable. "
            "This is provisional and does not provide clinical progression adjudication."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepared = prepare_ucsd_dataset(
        args.source_root,
        args.clinical_table,
        args.output_root,
        max_subjects=args.max_subjects,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        allow_imaging_only_labels=args.allow_imaging_only_labels,
        negative_cases_table=args.negative_cases_table,
        include_negative_controls=args.include_negative_controls,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
    )
    print(f"selected_subjects={','.join(prepared.selected_subjects)}")
    print(f"manifest={prepared.manifest}")
    print(f"derived_root={prepared.derived_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
