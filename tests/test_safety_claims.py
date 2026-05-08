from __future__ import annotations

from pathlib import Path

from glioma_recurrence.constants import RESEARCH_ONLY_DISCLAIMER


def test_research_only_disclaimer_rejects_clinical_dose_recommendation_claims():
    assert "not a clinical dose recommendation" in RESEARCH_ONLY_DISCLAIMER


def test_readme_preserves_no_clinical_use_claim():
    readme = Path("README.md").read_text().lower()

    assert "not a clinical dose recommendation" in readme
    assert "medical device" in readme
    assert "for clinical use" not in readme

