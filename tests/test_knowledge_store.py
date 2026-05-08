from __future__ import annotations

import importlib.util
from pathlib import Path


def load_validator():
    script_path = Path.cwd() / "scripts/validate_knowledge_store.py"
    spec = importlib.util.spec_from_file_location("validate_knowledge_store", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate


def test_repository_knowledge_store_structure_is_present():
    validate = load_validator()
    errors = validate(Path.cwd())

    assert errors == []
