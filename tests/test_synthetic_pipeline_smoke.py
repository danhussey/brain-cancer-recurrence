from __future__ import annotations

import json
import importlib.util
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


def test_synthetic_dataset_runs_end_to_end_pipeline(tmp_path: Path):
    generate_synthetic_dataset = load_generator()
    paths = generate_synthetic_dataset(tmp_path, n_patients=2, shape=(12, 12, 12), seed=7)
    model_path = paths.models_root / "dose-distance.json"
    eval_path = paths.reports_root / "eval.json"
    validation_case = paths.derived_root / "SYN002"

    assert cli_main(
        [
            "preprocess",
            "--manifest",
            str(paths.manifest),
            "--derived-root",
            str(paths.derived_root),
            "--prescription-dose-gy",
            "60",
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
            "dose-distance",
            "--output",
            str(model_path),
            "--prescription-dose-gy",
            "60",
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
            str(model_path),
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
            str(model_path),
            "--output-dir",
            str(validation_case),
        ]
    ) == 0

    report = json.loads(eval_path.read_text())
    assert report["n_cases"] == 1
    assert report["mean_voxel_auprc"] > 0.5
    assert (validation_case / "recurrence_risk.nii.gz").exists()
    assert (validation_case / "qc_overlay.html").exists()
