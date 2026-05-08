#!/usr/bin/env python3
"""Validate the repository knowledge-store scaffold.

This check encodes the agent-first repository structure so future changes do
not silently remove the docs that agents rely on for progressive disclosure.
"""

from __future__ import annotations

import sys
from pathlib import Path

REQUIRED_PATHS = (
    "AGENTS.md",
    "ARCHITECTURE.md",
    "docs/design-docs/index.md",
    "docs/design-docs/core-beliefs.md",
    "docs/exec-plans/active/README.md",
    "docs/exec-plans/completed/glioma-risk-pipeline-v1.md",
    "docs/exec-plans/tech-debt-tracker.md",
    "docs/generated/db-schema.md",
    "docs/product-specs/index.md",
    "docs/product-specs/new-user-onboarding.md",
    "docs/references/design-system-reference-llms.txt",
    "docs/references/nixpacks-llms.txt",
    "docs/references/uv-llms.txt",
    "docs/DESIGN.md",
    "docs/FRONTEND.md",
    "docs/PLANS.md",
    "docs/PRODUCT_SENSE.md",
    "docs/QUALITY_SCORE.md",
    "docs/RELIABILITY.md",
    "docs/SECURITY.md",
)

AGENTS_REQUIRED_LINKS = (
    "ARCHITECTURE.md",
    "docs/product-specs/index.md",
    "docs/design-docs/index.md",
    "docs/PLANS.md",
    "docs/QUALITY_SCORE.md",
    "docs/RELIABILITY.md",
    "docs/SECURITY.md",
    "docs/references/",
    "docs/generated/",
)


def validate(root: Path) -> list[str]:
    errors: list[str] = []
    for relative in REQUIRED_PATHS:
        path = root / relative
        if not path.exists():
            errors.append(f"missing required knowledge-store path: {relative}")
            continue
        if path.is_file() and path.stat().st_size == 0:
            errors.append(f"knowledge-store path is empty: {relative}")

    agents_path = root / "AGENTS.md"
    if agents_path.exists():
        agents = agents_path.read_text()
        for link in AGENTS_REQUIRED_LINKS:
            if link not in agents:
                errors.append(f"AGENTS.md does not link to {link}")

    product_index = root / "docs/product-specs/index.md"
    if product_index.exists():
        product_text = product_index.read_text()
        if "glioma-recurrence-risk.md" not in product_text:
            errors.append("product spec index must link glioma-recurrence-risk.md")

    plans = root / "docs/PLANS.md"
    if plans.exists():
        plans_text = plans.read_text()
        for required in ("docs/exec-plans/active/", "docs/exec-plans/completed/", "tech-debt-tracker.md"):
            if required not in plans_text:
                errors.append(f"docs/PLANS.md does not link to {required}")

    return errors


def main(argv: list[str] | None = None) -> int:
    root = Path(argv[0]) if argv else Path.cwd()
    errors = validate(root)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("knowledge-store structure ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

