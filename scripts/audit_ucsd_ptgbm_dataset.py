#!/usr/bin/env python3
"""Audit UCSD-PTGBM download completeness without preparing derivatives."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def load_prepare_module() -> ModuleType:
    script_path = Path(__file__).with_name("prepare_ucsd_ptgbm_dataset.py")
    spec = importlib.util.spec_from_file_location("prepare_ucsd_ptgbm_dataset", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def audit_ucsd_dataset(source_root: str | Path, clinical_table: str | Path | None = None) -> dict[str, object]:
    prepare = load_prepare_module()
    source = Path(source_root)
    niftis = sorted(prepare.find_nifti_files(source))
    rows = (
        prepare.read_clinical_timepoints(clinical_table)
        if clinical_table is not None
        else prepare.infer_timepoints_from_filenames(niftis)
    )
    pairs, skipped = prepare.select_pairs(rows, niftis, max_subjects=None)
    complete_timepoints = [
        row for row in rows if prepare.match_timepoint_files(niftis, row) is not None
    ]
    complete_by_subject: dict[str, int] = {}
    for row in complete_timepoints:
        complete_by_subject[row.subject_id] = complete_by_subject.get(row.subject_id, 0) + 1

    return {
        "source_root": str(source),
        "clinical_table": str(clinical_table) if clinical_table is not None else "",
        "pairing_mode": "clinical-table" if clinical_table is not None else "filename-inferred",
        "nifti_files": len(niftis),
        "partial_files": count_files(source, "*.partial"),
        "aspera_checkpoint_files": count_files(source, "*.aspera-ckpt"),
        "subjects_seen": len({row.subject_id for row in rows}),
        "timepoints_seen": len({(row.subject_id, row.timepoint_id) for row in rows}),
        "complete_mri_mask_timepoints": len(complete_timepoints),
        "subjects_with_2plus_complete_timepoints": sum(count >= 2 for count in complete_by_subject.values()),
        "eligible_pairs": len(pairs),
        "eligible_subjects": [pair.subject_id for pair in pairs],
        "skipped_subjects": skipped,
        "modality_counts": {
            "t1c": sum(1 for path in niftis if prepare.is_t1c_file(path)),
            "flair": sum(1 for path in niftis if prepare.is_flair_file(path)),
            "total_cellular_tumor_masks": sum(
                1 for path in niftis if prepare.is_total_cellular_tumor_mask(path)
            ),
            "brats_like_segmentations": sum(1 for path in niftis if prepare.is_brats_like_segmentation(path)),
        },
    }


def count_files(root: Path, pattern: str) -> int:
    return sum(1 for path in root.rglob(pattern) if path.is_file())


def print_human(summary: dict[str, object]) -> None:
    print("UCSD-PTGBM audit")
    for key in (
        "source_root",
        "clinical_table",
        "pairing_mode",
        "nifti_files",
        "partial_files",
        "aspera_checkpoint_files",
        "subjects_seen",
        "timepoints_seen",
        "complete_mri_mask_timepoints",
        "subjects_with_2plus_complete_timepoints",
        "eligible_pairs",
    ):
        print(f"{key}={summary[key]}")
    modality_counts = summary["modality_counts"]
    if isinstance(modality_counts, dict):
        for key, value in modality_counts.items():
            print(f"modality_counts.{key}={value}")
    eligible_subjects = summary["eligible_subjects"]
    if isinstance(eligible_subjects, list):
        preview = ",".join(str(subject) for subject in eligible_subjects[:20])
        print(f"eligible_subjects_preview={preview}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="Extracted UCSD-PTGBM NIfTI root")
    parser.add_argument("--clinical-table", default=None, help="Optional UCSD clinical CSV/TSV/XLSX table")
    parser.add_argument("--json-output", default=None, help="Optional path for strict JSON audit output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = audit_ucsd_dataset(args.source_root, args.clinical_table)
    print_human(summary)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"json_output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
