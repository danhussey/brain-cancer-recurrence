"""QC overlay reports and research-only report text."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np

from .case import CaseData
from .constants import CASE_QC_HTML, RESEARCH_ONLY_DISCLAIMER
from .geometry import Volume


def write_case_qc_report(
    case: CaseData,
    *,
    output_dir: str | Path,
    risk: Volume | None = None,
) -> Path:
    """Write mandatory QC overlays for one case.

    The HTML always includes the required panels. If matplotlib is available it
    also writes PNG overlays beside the HTML report.
    """

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    panels = [
        ("T1c", case.t1c.data),
        ("FLAIR", case.flair.data),
        ("Dose Gy", case.dose_gy.data),
        (
            "Recurrence Mask On Baseline",
            case.recurrence_mask.data if case.recurrence_mask is not None else np.zeros(case.t1c.shape, dtype=np.float32),
        ),
        ("Recurrence Risk", risk.data if risk is not None else np.zeros(case.t1c.shape, dtype=np.float32)),
    ]
    image_rows = _write_png_panels(target_dir, panels)
    if not image_rows:
        image_rows = [
            f"<li>{html.escape(name)}: shape={tuple(int(v) for v in data.shape)}, "
            f"min={float(np.nanmin(data)):.4g}, max={float(np.nanmax(data)):.4g}</li>"
            for name, data in panels
        ]
    body = "\n".join(image_rows)
    report = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>QC Overlay {html.escape(case.patient_id)}</title></head>
<body>
<h1>QC Overlay {html.escape(case.patient_id)}</h1>
<p>{html.escape(RESEARCH_ONLY_DISCLAIMER)}</p>
<ul>
{body}
</ul>
</body>
</html>
"""
    output = target_dir / CASE_QC_HTML
    output.write_text(report)
    return output


def _write_png_panels(target_dir: Path, panels: list[tuple[str, np.ndarray]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    rows: list[str] = []
    for name, data in panels:
        data = np.asarray(data)
        slice_index = data.shape[2] // 2
        image = data[:, :, slice_index]
        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
        if "Mask" in name or "Risk" in name:
            cmap = "magma"
        else:
            cmap = "gray"
        ax.imshow(np.rot90(image), cmap=cmap)
        ax.set_title(name)
        ax.axis("off")
        filename = name.lower().replace(" ", "_") + ".png"
        fig.tight_layout()
        fig.savefig(target_dir / filename)
        plt.close(fig)
        rows.append(f'<li>{html.escape(name)}<br><img src="{html.escape(filename)}" alt="{html.escape(name)}"></li>')
    return rows

