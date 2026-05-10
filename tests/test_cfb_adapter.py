from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

from glioma_recurrence.geometry import Volume
from glioma_recurrence.nifti import write_volume
from glioma_recurrence.schema import read_manifest


def load_cfb_adapter():
    script_path = Path.cwd() / "scripts/prepare_cfb_gbm_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_cfb_gbm_dataset", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_fake_cfb_case(root: Path, patient_id: str) -> None:
    t0 = root / patient_id / "t0"
    t0.mkdir(parents=True)
    affine = np.eye(4)
    data = np.ones((4, 4, 4), dtype=np.float32)
    numeric = str(int(patient_id))
    for suffix in ("t1gd", "flair", "rtdose", "gtv"):
        dtype = np.uint8 if suffix == "gtv" else np.float32
        write_volume(Volume(data, affine), t0 / f"{numeric}_t0_{suffix}.nii.gz", dtype=dtype)


def test_prepare_cfb_dataset_copies_pilot_cases_and_manifest(tmp_path: Path):
    adapter = load_cfb_adapter()
    source = tmp_path / "CFB-GBM"
    write_fake_cfb_case(source, "054")
    write_fake_cfb_case(source, "057")
    output = tmp_path / "prepared"

    prepared = adapter.prepare_cfb_dataset(
        source,
        output,
        max_cases=2,
        allow_gtv_proxy_labels=True,
    )

    assert prepared.selected_cases == ["054", "057"]
    assert (output / "derived/054/baseline_t1c.nii.gz").exists()
    assert (output / "derived/054/baseline_flair.nii.gz").exists()
    assert (output / "derived/054/dose_gy_on_baseline.nii.gz").exists()
    assert (output / "masks/054_t0_gtv_proxy_mask.nii.gz").exists()
    rows = list(csv.DictReader(prepared.manifest.open()))
    assert rows[0]["split"] == "train"
    assert rows[1]["split"] == "validation"
    assert rows[0]["recurrence_adjudication"] == "gtv_proxy_not_recurrence"
    assert read_manifest(prepared.manifest)[0].baseline_scan_date.isoformat() == "1900-01-01"
