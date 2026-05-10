from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from glioma_recurrence.cli import main as cli_main


def load_generator():
    script_path = Path.cwd() / "scripts/generate_synthetic_dataset.py"
    spec = importlib.util.spec_from_file_location("generate_synthetic_dataset", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.generate_synthetic_dataset


def test_synthetic_dataset_runs_end_to_end_mri_only_pipeline(tmp_path: Path):
    generate_synthetic_dataset = load_generator()
    paths = generate_synthetic_dataset(tmp_path, n_patients=2, shape=(12, 12, 12), seed=7)
    tumor_distance_path = paths.models_root / "tumor-distance.json"
    voxel_model_path = paths.models_root / "voxel-logistic-mri.json"
    eval_path = paths.reports_root / "eval.json"
    validation_case = paths.derived_root / "SYN002"

    assert cli_main(
        [
            "preprocess",
            "--manifest",
            str(paths.manifest),
            "--derived-root",
            str(paths.derived_root),
        ]
    ) == 0
    assert cli_main(
        [
            "make-labels",
            "--manifest",
            str(paths.manifest),
            "--derived-root",
            str(paths.derived_root),
            "--assume-baseline-space",
        ]
    ) == 0
    assert cli_main(
        [
            "train",
            "--manifest",
            str(paths.manifest),
            "--derived-root",
            str(paths.derived_root),
            "--model",
            "tumor-distance",
            "--output",
            str(tumor_distance_path),
            "--max-voxels-per-case",
            "2048",
        ]
    ) == 0
    assert cli_main(
        [
            "train",
            "--manifest",
            str(paths.manifest),
            "--derived-root",
            str(paths.derived_root),
            "--model",
            "voxel-logistic-mri",
            "--output",
            str(voxel_model_path),
            "--max-voxels-per-case",
            "2048",
        ]
    ) == 0
    assert cli_main(
        [
            "evaluate",
            "--manifest",
            str(paths.manifest),
            "--derived-root",
            str(paths.derived_root),
            "--model-path",
            str(voxel_model_path),
            "--output",
            str(eval_path),
            "--splits",
            "validation",
            "--write-predictions",
        ]
    ) == 0
    assert cli_main(
        [
            "predict",
            "--case-dir",
            str(validation_case),
            "--model-path",
            str(voxel_model_path),
            "--output-dir",
            str(validation_case),
        ]
    ) == 0

    report = json.loads(eval_path.read_text())
    assert report["n_cases"] == 1
    assert "baseline_comparison" in report
    assert report["baseline_comparison"]["baseline_model"] == "tumor-distance"
    assert (validation_case / "recurrence_risk.nii.gz").exists()
    assert (validation_case / "qc_overlay.html").exists()


def test_tumor_distance_is_default_train_model(tmp_path: Path):
    generate_synthetic_dataset = load_generator()
    paths = generate_synthetic_dataset(tmp_path, n_patients=2, shape=(12, 12, 12), seed=11)
    model_path = paths.models_root / "default-model.json"
    observability_root = tmp_path / "observability"

    assert cli_main(
        [
            "preprocess",
            "--manifest",
            str(paths.manifest),
            "--derived-root",
            str(paths.derived_root),
        ]
    ) == 0
    assert cli_main(
        [
            "make-labels",
            "--manifest",
            str(paths.manifest),
            "--derived-root",
            str(paths.derived_root),
            "--assume-baseline-space",
        ]
    ) == 0
    assert cli_main(
        [
            "train",
            "--manifest",
            str(paths.manifest),
            "--derived-root",
            str(paths.derived_root),
            "--output",
            str(model_path),
            "--observability-root",
            str(observability_root),
        ]
    ) == 0

    assert json.loads(model_path.read_text())["kind"] == "tumor-distance"
    summaries = [path for path in observability_root.glob("*/summary.json") if "-train-" in str(path)]
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text())
    assert summary["stage"] == "train"
    assert summary["status"] == "completed"
    assert summary["event_counts"]["training_records_selected"] == 1
    events = (summaries[0].parent / "events.jsonl").read_text()
    assert "artifact_written" in events
