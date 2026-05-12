"""QC overlay reports and research-only report text."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from .case import CaseData
from .constants import CASE_QC_HTML, CASE_QC_SUMMARY_JSON, RESEARCH_ONLY_DISCLAIMER
from .geometry import Volume


def write_case_qc_report(
    case: CaseData,
    *,
    output_dir: str | Path,
    risk: Volume | None = None,
) -> Path:
    """Write static QC overlays for one case.

    The report is viewable directly from disk. When matplotlib is available, it
    writes anatomy slices plus transparent overlays with browser opacity
    controls. Without matplotlib, it still writes a summary-only HTML report.
    """

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    risk_data = risk.data if risk is not None else np.zeros(case.t1c.shape, dtype=np.float32)
    recurrence_data = (
        case.recurrence_mask.data
        if case.recurrence_mask is not None
        else np.zeros(case.t1c.shape, dtype=np.float32)
    )
    slices = select_representative_slices(
        shape=case.t1c.shape,
        baseline_tumor=case.baseline_tumor_mask.data,
        recurrence=recurrence_data,
        risk=risk_data,
    )
    summary = build_qc_summary(case, risk=risk, recurrence_data=recurrence_data, risk_data=risk_data, slices=slices)
    summary_path = target_dir / CASE_QC_SUMMARY_JSON
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    image_assets = _write_overlay_assets(target_dir, case, recurrence_data=recurrence_data, risk_data=risk_data, slices=slices)
    if image_assets:
        report_body = _interactive_report_body(summary, image_assets)
    else:
        report_body = _summary_only_report_body(summary)

    report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QC Overlay {html.escape(case.patient_id)}</title>
{_report_css()}
</head>
<body>
<main>
{report_body}
</main>
{_report_javascript() if image_assets else ""}
</body>
</html>
"""
    output = target_dir / CASE_QC_HTML
    output.write_text(report)
    return output


def build_qc_summary(
    case: CaseData,
    *,
    risk: Volume | None,
    recurrence_data: np.ndarray,
    risk_data: np.ndarray,
    slices: list[dict[str, object]],
) -> dict[str, object]:
    brain = case.brain_mask.data.astype(bool)
    return {
        "patient_id": case.patient_id,
        "research_only": RESEARCH_ONLY_DISCLAIMER,
        "shape": list(case.t1c.shape),
        "spacing_mm": [round(float(value), 6) for value in case.t1c.spacing],
        "brain_voxels": int(np.count_nonzero(brain)),
        "baseline_tumor_voxels": int(np.count_nonzero(case.baseline_tumor_mask.data)),
        "recurrence_mask_present": case.recurrence_mask is not None,
        "recurrence_voxels": int(np.count_nonzero(recurrence_data)) if case.recurrence_mask is not None else None,
        "risk_present": risk is not None,
        "risk_stats": volume_stats(risk_data, mask=brain) if risk is not None else None,
        "t1c_stats": volume_stats(case.t1c.data, mask=brain),
        "flair_stats": volume_stats(case.flair.data, mask=brain),
        "selected_slices": slices,
    }


def volume_stats(data: np.ndarray, *, mask: np.ndarray | None = None) -> dict[str, float]:
    values = np.asarray(data, dtype=np.float32)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"min": 0.0, "p01": 0.0, "median": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "min": round(float(np.min(values)), 6),
        "p01": round(float(np.percentile(values, 1)), 6),
        "median": round(float(np.median(values)), 6),
        "p99": round(float(np.percentile(values, 99)), 6),
        "max": round(float(np.max(values)), 6),
    }


def select_representative_slices(
    *,
    shape: tuple[int, int, int],
    baseline_tumor: np.ndarray,
    recurrence: np.ndarray,
    risk: np.ndarray,
) -> list[dict[str, object]]:
    candidates: list[tuple[int, str]] = [(shape[2] // 2, "midline")]
    for label, data in (
        ("baseline tumor peak", baseline_tumor),
        ("recurrence peak", recurrence),
        ("risk peak", risk),
    ):
        selected = best_slice_index(data)
        if selected is not None:
            candidates.append((selected, label))

    seen: set[int] = set()
    slices: list[dict[str, object]] = []
    for index, reason in candidates:
        bounded = int(min(max(index, 0), shape[2] - 1))
        if bounded in seen:
            continue
        seen.add(bounded)
        slices.append({"index": bounded, "reason": reason})
        if len(slices) >= 4:
            break
    return slices


def best_slice_index(data: np.ndarray) -> int | None:
    values = np.asarray(data, dtype=np.float32)
    if values.ndim != 3:
        return None
    signal = np.nan_to_num(np.abs(values), nan=0.0, posinf=0.0, neginf=0.0)
    scores = np.sum(signal, axis=(0, 1))
    if not np.any(scores > 0):
        return None
    return int(np.argmax(scores))


def _write_overlay_assets(
    target_dir: Path,
    case: CaseData,
    *,
    recurrence_data: np.ndarray,
    risk_data: np.ndarray,
    slices: list[dict[str, object]],
) -> list[dict[str, object]]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    assets: list[dict[str, object]] = []
    for item in slices:
        slice_index = int(item["index"])
        prefix = f"qc_z{slice_index:03d}"
        t1c_file = f"{prefix}_t1c.png"
        flair_file = f"{prefix}_flair.png"
        tumor_file = f"{prefix}_tumor.png"
        recurrence_file = f"{prefix}_recurrence.png"
        risk_file = f"{prefix}_risk.png"

        plt.imsave(target_dir / t1c_file, normalize_slice(case.t1c.data[:, :, slice_index]), cmap="gray", vmin=0, vmax=1)
        plt.imsave(target_dir / flair_file, normalize_slice(case.flair.data[:, :, slice_index]), cmap="gray", vmin=0, vmax=1)
        plt.imsave(
            target_dir / tumor_file,
            mask_overlay_rgba(case.baseline_tumor_mask.data[:, :, slice_index], color=(0.0, 0.85, 1.0), alpha=0.85),
        )
        plt.imsave(
            target_dir / recurrence_file,
            mask_overlay_rgba(recurrence_data[:, :, slice_index], color=(1.0, 0.15, 0.85), alpha=0.85),
        )
        plt.imsave(target_dir / risk_file, risk_overlay_rgba(risk_data[:, :, slice_index]))

        assets.append(
            {
                "index": slice_index,
                "reason": item["reason"],
                "t1c": t1c_file,
                "flair": flair_file,
                "tumor": tumor_file,
                "recurrence": recurrence_file,
                "risk": risk_file,
            }
        )
    return assets


def normalize_slice(data: np.ndarray) -> np.ndarray:
    image = np.rot90(np.asarray(data, dtype=np.float32))
    values = image[np.isfinite(image)]
    if values.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    low = float(np.percentile(values, 1))
    high = float(np.percentile(values, 99))
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)


def mask_overlay_rgba(data: np.ndarray, *, color: tuple[float, float, float], alpha: float) -> np.ndarray:
    mask = np.rot90(np.asarray(data) > 0)
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    overlay[..., 3] = mask.astype(np.float32) * alpha
    return overlay


def risk_overlay_rgba(data: np.ndarray) -> np.ndarray:
    image = normalize_slice(data)
    overlay = np.zeros((*image.shape, 4), dtype=np.float32)
    overlay[..., 0] = image
    overlay[..., 1] = np.sqrt(image) * 0.35
    overlay[..., 2] = 1.0 - image
    overlay[..., 3] = np.clip(image, 0.0, 1.0) * 0.9
    return overlay


def _interactive_report_body(summary: dict[str, object], assets: list[dict[str, object]]) -> str:
    patient_id = str(summary["patient_id"])
    buttons = "\n".join(_slice_button(asset, active=offset == 0) for offset, asset in enumerate(assets))
    slice_sections = "\n".join(_slice_section(asset, active=offset == 0) for offset, asset in enumerate(assets))
    summary_table = _summary_table(summary)
    return f"""
<header>
  <p class="eyebrow">Research QC Overlay</p>
  <h1>{html.escape(patient_id)}</h1>
  <p class="disclaimer">{html.escape(RESEARCH_ONLY_DISCLAIMER)}</p>
</header>
<section aria-labelledby="summary-heading">
  <h2 id="summary-heading">Case Summary</h2>
  {summary_table}
</section>
<section aria-labelledby="controls-heading">
  <h2 id="controls-heading">Overlay Controls</h2>
  <div class="controls">
    {_range_control("tumor", "Baseline tumor", "0.55")}
    {_range_control("recurrence", "Recurrence mask", "0.65")}
    {_range_control("risk", "Risk heatmap", "0.60")}
  </div>
</section>
<section aria-labelledby="slice-heading">
  <h2 id="slice-heading">Representative Slices</h2>
  <nav class="slice-tabs" aria-label="Representative slices">
    {buttons}
  </nav>
  {slice_sections}
</section>
"""


def _slice_button(asset: dict[str, object], *, active: bool) -> str:
    index = int(asset["index"])
    class_attr = ' class="active"' if active else ""
    reason = html.escape(str(asset["reason"]))
    return f'<button type="button" data-slice-tab="{index}"{class_attr}>z={index} <span>{reason}</span></button>'


def _range_control(key: str, label: str, value: str) -> str:
    return (
        f'<label>{html.escape(label)} '
        f'<input type="range" min="0" max="1" step="0.05" value="{value}" data-opacity-control="{key}">'
        "</label>"
    )


def _slice_section(asset: dict[str, object], *, active: bool) -> str:
    index = int(asset["index"])
    classes = "slice-view active" if active else "slice-view"
    return f"""
  <article class="{classes}" data-slice-view="{index}">
    <h3>Slice z={index}: {html.escape(str(asset["reason"]))}</h3>
    <div class="viewer-grid">
      {_viewer_stack("T1c", str(asset["t1c"]), asset)}
      {_viewer_stack("FLAIR", str(asset["flair"]), asset)}
    </div>
  </article>
"""


def _viewer_stack(label: str, base_file: str, asset: dict[str, object]) -> str:
    escaped_label = html.escape(label)
    return f"""
<figure>
  <figcaption>{escaped_label} with overlays</figcaption>
  <div class="image-stack">
    <img src="{html.escape(base_file)}" alt="{escaped_label} anatomy slice">
    <img src="{html.escape(str(asset["tumor"]))}" alt="Baseline tumor overlay" class="overlay" data-overlay="tumor">
    <img src="{html.escape(str(asset["recurrence"]))}" alt="Recurrence mask overlay" class="overlay" data-overlay="recurrence">
    <img src="{html.escape(str(asset["risk"]))}" alt="Risk heatmap overlay" class="overlay" data-overlay="risk">
  </div>
</figure>
"""


def _summary_table(summary: dict[str, object]) -> str:
    rows = [
        ("Shape", " x ".join(str(value) for value in summary["shape"])),
        ("Spacing mm", " / ".join(str(value) for value in summary["spacing_mm"])),
        ("Brain voxels", str(summary["brain_voxels"])),
        ("Baseline tumor voxels", str(summary["baseline_tumor_voxels"])),
        ("Recurrence mask present", str(summary["recurrence_mask_present"])),
        ("Recurrence voxels", "not available" if summary["recurrence_voxels"] is None else str(summary["recurrence_voxels"])),
        ("Risk map present", str(summary["risk_present"])),
    ]
    body = "\n".join(
        f"<tr><th scope=\"row\">{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
        for label, value in rows
    )
    return f"<table>{body}</table>"


def _summary_only_report_body(summary: dict[str, object]) -> str:
    selected_slices = ", ".join(
        f'z={item["index"]} ({item["reason"]})' for item in summary["selected_slices"] if isinstance(item, dict)
    )
    return f"""
<header>
  <p class="eyebrow">Research QC Overlay</p>
  <h1>{html.escape(str(summary["patient_id"]))}</h1>
  <p class="disclaimer">{html.escape(RESEARCH_ONLY_DISCLAIMER)}</p>
</header>
<section>
  <h2>Case Summary</h2>
  {_summary_table(summary)}
  <p>Matplotlib was unavailable, so image overlays were not rendered. Selected slices: {html.escape(selected_slices)}</p>
</section>
"""


def _report_css() -> str:
    return """
<style>
:root { color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
body { margin: 0; background: #f7f7f5; color: #1b1b1b; }
main { max-width: 1180px; margin: 0 auto; padding: 28px; }
header, section { margin-bottom: 24px; }
.eyebrow { margin: 0 0 4px; font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: #555; }
h1 { margin: 0 0 8px; font-size: 30px; }
h2 { margin: 0 0 12px; font-size: 19px; }
h3 { margin: 0 0 12px; font-size: 16px; }
.disclaimer { max-width: 860px; padding: 10px 12px; background: #fff3cd; border-left: 4px solid #b58100; }
table { border-collapse: collapse; width: 100%; background: #fff; }
th, td { padding: 8px 10px; border-bottom: 1px solid #ddd; text-align: left; font-size: 14px; }
th { width: 220px; color: #333; }
.controls { display: flex; flex-wrap: wrap; gap: 14px; padding: 12px; background: #fff; border: 1px solid #ddd; }
.controls label { display: flex; gap: 8px; align-items: center; white-space: nowrap; }
.slice-tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
.slice-tabs button { border: 1px solid #aaa; background: #fff; padding: 8px 10px; cursor: pointer; }
.slice-tabs button.active { border-color: #111; background: #111; color: #fff; }
.slice-tabs span { display: block; font-size: 11px; opacity: 0.78; }
.slice-view { display: none; }
.slice-view.active { display: block; }
.viewer-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
figure { margin: 0; background: #fff; border: 1px solid #ddd; padding: 10px; }
figcaption { margin-bottom: 8px; font-weight: 600; }
.image-stack { position: relative; width: 100%; background: #000; }
.image-stack img { display: block; width: 100%; height: auto; }
.image-stack .overlay { position: absolute; inset: 0; pointer-events: none; }
</style>
"""


def _report_javascript() -> str:
    return """
<script>
for (const input of document.querySelectorAll('[data-opacity-control]')) {
  const key = input.dataset.opacityControl;
  const apply = () => {
    for (const image of document.querySelectorAll(`[data-overlay="${key}"]`)) {
      image.style.opacity = input.value;
    }
  };
  input.addEventListener('input', apply);
  apply();
}
for (const button of document.querySelectorAll('[data-slice-tab]')) {
  button.addEventListener('click', () => {
    const selected = button.dataset.sliceTab;
    for (const tab of document.querySelectorAll('[data-slice-tab]')) {
      tab.classList.toggle('active', tab === button);
    }
    for (const view of document.querySelectorAll('[data-slice-view]')) {
      view.classList.toggle('active', view.dataset.sliceView === selected);
    }
  });
}
</script>
"""
