"""QC overlay reports and research-only report text."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from .case import CaseData
from .constants import (
    CASE_QC_HTML,
    CASE_QC_SUMMARY_JSON,
    PREPROCESS_QC_HTML,
    PREPROCESS_QC_SUMMARY_JSON,
    RESEARCH_ONLY_DISCLAIMER,
)
from .geometry import Volume

MAX_BROWSER_SLICES = 64

SUMMARY_FIELD_HELP = {
    "Shape": "Voxel grid dimensions in x, y, and z. Useful for catching unexpected crops or wrong image dimensions.",
    "Spacing mm": "Physical voxel spacing. Useful for checking that registration and resampling preserved real-world scale.",
    "Brain voxels": "Number of voxels inside the brain mask. Useful for spotting failed masking or empty anatomy.",
    "Baseline tumor voxels": "Tumor-mask volume at the prediction baseline. Useful because residual tumor is an expected high-risk region.",
    "Recurrence mask present": "Whether this case has a mapped follow-up recurrence label. Useful for knowing if label QC is possible.",
    "Recurrence voxels": "Total recurrence-label volume in baseline space. Useful for checking label size and class imbalance.",
    "Recurrence inside baseline tumor": "Recurrence voxels overlapping the baseline tumor mask. Useful for separating obvious residual-tumor recurrence from harder cases.",
    "Recurrence outside baseline tumor": "Recurrence voxels outside the baseline tumor mask. Useful for marginal or distant recurrence review.",
    "Risk map present": "Whether a model prediction was loaded into the report. Useful for distinguishing preprocessing QC from prediction QC.",
    "Mean risk in recurrence": "Average predicted risk inside the mapped recurrence label. Useful as a quick signal of whether the risk map is elevated where recurrence occurred.",
    "Mean risk outside recurrence": "Average predicted risk in brain voxels outside the recurrence label. Useful for comparing recurrence regions against the surrounding brain.",
    "Top 1% risk overlap": "Overlap between the recurrence label and the highest-risk 1% of evaluated brain voxels, reported as recurrence coverage and Dice.",
    "Top 5% risk overlap": "Overlap between the recurrence label and the highest-risk 5% of evaluated brain voxels, reported as recurrence coverage and Dice.",
    "Viewer slices": "Number of axial slices rendered into the static viewer. Useful for knowing how much of the volume can be inspected here.",
}

PREPROCESS_FIELD_HELP = {
    "Shape": "Baseline T1c voxel grid after preprocessing. All model inputs should share this grid.",
    "Spacing mm": "Physical voxel spacing after preprocessing. Useful for catching unexpected anisotropy or scaling errors.",
    "Brain mask voxels": "Number of voxels retained by the current brain-mask proxy. Useful for spotting failed skull stripping or empty anatomy.",
    "Brain mask fraction": "Fraction of the whole volume inside the brain mask. Very small or very large values usually need review.",
    "Baseline tumor voxels": "Baseline tumor-mask volume after nearest-neighbor resampling to T1c space.",
    "Baseline tumor fraction": "Baseline tumor voxels divided by brain-mask voxels. Useful for catching missing or implausibly large masks.",
    "Skull stripping": "Current skull-stripping status. V1 uses a heuristic brain mask; dedicated tools should be added before serious cohort analysis.",
    "Bias correction": "Current bias-field correction status. This report makes the missing step visible instead of implying it happened.",
    "FLAIR alignment": "How FLAIR was brought onto the baseline T1c grid.",
    "Registration QC": "Visual QC artifact for checking whether T1c and FLAIR anatomy line up after preprocessing.",
    "Viewer slices": "Number of axial slices rendered into the preprocessing viewer.",
}


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
    selected_slices = select_representative_slices(
        shape=case.t1c.shape,
        baseline_tumor=case.baseline_tumor_mask.data,
        recurrence=recurrence_data,
        risk=risk_data,
    )
    viewer_slices = select_viewer_slices(shape=case.t1c.shape, selected_slices=selected_slices)
    summary = build_qc_summary(
        case,
        risk=risk,
        recurrence_data=recurrence_data,
        risk_data=risk_data,
        selected_slices=selected_slices,
        viewer_slices=viewer_slices,
    )
    summary_path = target_dir / CASE_QC_SUMMARY_JSON
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    image_assets = _write_overlay_assets(
        target_dir,
        case,
        recurrence_data=recurrence_data,
        risk_data=risk_data,
        slices=viewer_slices,
    )
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
{_report_javascript(image_assets) if image_assets else ""}
</body>
</html>
"""
    output = target_dir / CASE_QC_HTML
    output.write_text(report)
    return output


def write_preprocess_qc_report(
    case: CaseData,
    *,
    output_dir: str | Path,
    source_t1c: Volume | None = None,
    source_flair: Volume | None = None,
    source_baseline_tumor: Volume | None = None,
    flair_alignment: str = "header-affine resampling to baseline T1c grid",
) -> Path:
    """Write a preprocessing-focused QC report for one baseline case."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    selected_slices = select_preprocess_slices(
        shape=case.t1c.shape,
        brain=case.brain_mask.data,
        baseline_tumor=case.baseline_tumor_mask.data,
    )
    viewer_slices = select_viewer_slices(shape=case.t1c.shape, selected_slices=selected_slices)
    summary = build_preprocess_qc_summary(
        case,
        source_t1c=source_t1c,
        source_flair=source_flair,
        source_baseline_tumor=source_baseline_tumor,
        flair_alignment=flair_alignment,
        selected_slices=selected_slices,
        viewer_slices=viewer_slices,
    )
    summary_path = target_dir / PREPROCESS_QC_SUMMARY_JSON
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    image_assets = _write_preprocess_assets(target_dir, case, slices=viewer_slices)
    if image_assets:
        report_body = _preprocess_interactive_report_body(summary, image_assets)
    else:
        report_body = _preprocess_summary_only_report_body(summary)

    report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Preprocessing QC {html.escape(case.patient_id)}</title>
{_report_css()}
</head>
<body>
<main>
{report_body}
</main>
{_report_javascript(image_assets) if image_assets else ""}
</body>
</html>
"""
    output = target_dir / PREPROCESS_QC_HTML
    output.write_text(report)
    return output


def build_qc_summary(
    case: CaseData,
    *,
    risk: Volume | None,
    recurrence_data: np.ndarray,
    risk_data: np.ndarray,
    selected_slices: list[dict[str, object]],
    viewer_slices: list[dict[str, object]],
) -> dict[str, object]:
    brain = case.brain_mask.data.astype(bool)
    baseline_tumor = case.baseline_tumor_mask.data.astype(bool)
    recurrence_location = recurrence_location_stats(
        recurrence_data,
        baseline_tumor=baseline_tumor,
        brain=brain,
        recurrence_present=case.recurrence_mask is not None,
    )
    prediction_overlap = prediction_overlap_stats(
        risk_data,
        recurrence_data=recurrence_data,
        brain=brain,
        recurrence_present=case.recurrence_mask is not None,
        risk_present=risk is not None,
    )
    return {
        "patient_id": case.patient_id,
        "research_only": RESEARCH_ONLY_DISCLAIMER,
        "shape": list(case.t1c.shape),
        "spacing_mm": [round(float(value), 6) for value in case.t1c.spacing],
        "brain_voxels": int(np.count_nonzero(brain)),
        "baseline_tumor_voxels": int(np.count_nonzero(case.baseline_tumor_mask.data)),
        "recurrence_mask_present": case.recurrence_mask is not None,
        "recurrence_voxels": int(np.count_nonzero(recurrence_data)) if case.recurrence_mask is not None else None,
        "recurrence_location": recurrence_location,
        "risk_present": risk is not None,
        "risk_stats": volume_stats(risk_data, mask=brain) if risk is not None else None,
        "prediction_overlap": prediction_overlap,
        "t1c_stats": volume_stats(case.t1c.data, mask=brain),
        "flair_stats": volume_stats(case.flair.data, mask=brain),
        "selected_slices": selected_slices,
        "viewer_slices": viewer_slices,
        "viewer_slice_count": len(viewer_slices),
    }


def build_preprocess_qc_summary(
    case: CaseData,
    *,
    source_t1c: Volume | None,
    source_flair: Volume | None,
    source_baseline_tumor: Volume | None,
    flair_alignment: str,
    selected_slices: list[dict[str, object]],
    viewer_slices: list[dict[str, object]],
) -> dict[str, object]:
    brain = case.brain_mask.data.astype(bool)
    baseline_tumor = case.baseline_tumor_mask.data.astype(bool)
    total_voxels = int(np.prod(case.t1c.shape))
    brain_voxels = int(np.count_nonzero(brain))
    tumor_voxels = int(np.count_nonzero(baseline_tumor))
    brain_fraction = 0.0 if total_voxels == 0 else brain_voxels / total_voxels
    tumor_fraction = 0.0 if brain_voxels == 0 else tumor_voxels / brain_voxels
    source_geometry = _source_geometry_summary(
        source_t1c=source_t1c,
        source_flair=source_flair,
        source_baseline_tumor=source_baseline_tumor,
    )
    return {
        "patient_id": case.patient_id,
        "research_only": RESEARCH_ONLY_DISCLAIMER,
        "shape": list(case.t1c.shape),
        "spacing_mm": [round(float(value), 6) for value in case.t1c.spacing],
        "brain_mask_voxels": brain_voxels,
        "brain_mask_fraction": round(float(brain_fraction), 6),
        "baseline_tumor_voxels": tumor_voxels,
        "baseline_tumor_fraction_of_brain": round(float(tumor_fraction), 6),
        "t1c_stats": volume_stats(case.t1c.data, mask=brain),
        "flair_stats": volume_stats(case.flair.data, mask=brain),
        "source_geometry": source_geometry,
        "preprocessing_steps": {
            "brain_mask_method": "heuristic non-zero T1c/FLAIR mask with optional binary cleanup",
            "skull_stripping_status": "proxy-only; dedicated skull stripping not yet applied",
            "bias_correction_status": "not applied",
            "flair_alignment": flair_alignment,
            "baseline_tumor_resampling": "nearest-neighbor to baseline T1c grid",
            "visual_registration_qc": "T1c/FLAIR checkerboard plus mask overlays",
        },
        "quality_flags": preprocess_quality_flags(
            brain_fraction=brain_fraction,
            brain_voxels=brain_voxels,
            tumor_voxels=tumor_voxels,
            source_geometry=source_geometry,
        ),
        "selected_slices": selected_slices,
        "viewer_slices": viewer_slices,
        "viewer_slice_count": len(viewer_slices),
    }


def _source_geometry_summary(
    *,
    source_t1c: Volume | None,
    source_flair: Volume | None,
    source_baseline_tumor: Volume | None,
) -> dict[str, object]:
    summary: dict[str, object] = {}
    for key, volume in (
        ("t1c", source_t1c),
        ("flair", source_flair),
        ("baseline_tumor", source_baseline_tumor),
    ):
        if volume is None:
            summary[key] = None
            continue
        summary[key] = {
            "shape": list(volume.shape),
            "spacing_mm": [round(float(value), 6) for value in volume.spacing],
        }

    if source_t1c is not None and source_flair is not None:
        summary["flair_matches_t1c_before_preprocess"] = bool(
            source_flair.shape == source_t1c.shape and np.allclose(source_flair.affine, source_t1c.affine, atol=1e-4)
        )
    if source_t1c is not None and source_baseline_tumor is not None:
        summary["baseline_tumor_matches_t1c_before_preprocess"] = bool(
            source_baseline_tumor.shape == source_t1c.shape
            and np.allclose(source_baseline_tumor.affine, source_t1c.affine, atol=1e-4)
        )
    return summary


def preprocess_quality_flags(
    *,
    brain_fraction: float,
    brain_voxels: int,
    tumor_voxels: int,
    source_geometry: dict[str, object],
) -> list[str]:
    flags: list[str] = []
    if brain_voxels == 0:
        flags.append("empty brain mask")
    elif brain_fraction < 0.01:
        flags.append("brain mask covers less than 1% of the volume")
    elif brain_fraction > 0.95:
        flags.append("brain mask covers more than 95% of the volume")
    if tumor_voxels == 0:
        flags.append("empty baseline tumor mask")
    if source_geometry.get("flair_matches_t1c_before_preprocess") is False:
        flags.append("source FLAIR geometry differed from T1c before preprocessing")
    if source_geometry.get("baseline_tumor_matches_t1c_before_preprocess") is False:
        flags.append("source baseline tumor geometry differed from T1c before preprocessing")
    return flags


def recurrence_location_stats(
    recurrence_data: np.ndarray,
    *,
    baseline_tumor: np.ndarray,
    brain: np.ndarray,
    recurrence_present: bool,
) -> dict[str, float | int] | None:
    if not recurrence_present:
        return None
    recurrence = np.asarray(recurrence_data) > 0
    baseline = np.asarray(baseline_tumor, dtype=bool)
    brain_mask = np.asarray(brain, dtype=bool)
    recurrence_voxels = int(np.count_nonzero(recurrence))
    inside = int(np.count_nonzero(recurrence & baseline))
    outside = int(np.count_nonzero(recurrence & brain_mask & ~baseline))
    outside_fraction = 0.0 if recurrence_voxels == 0 else outside / recurrence_voxels
    return {
        "inside_baseline_tumor_voxels": inside,
        "outside_baseline_tumor_voxels": outside,
        "outside_baseline_tumor_fraction": round(float(outside_fraction), 6),
    }


def prediction_overlap_stats(
    risk_data: np.ndarray,
    *,
    recurrence_data: np.ndarray,
    brain: np.ndarray,
    recurrence_present: bool,
    risk_present: bool,
) -> dict[str, object] | None:
    if not recurrence_present or not risk_present:
        return None

    brain_mask = np.asarray(brain, dtype=bool).reshape(-1)
    if not np.any(brain_mask):
        return None

    labels = np.asarray(recurrence_data).astype(bool).reshape(-1)[brain_mask]
    scores = np.clip(np.asarray(risk_data, dtype=np.float32).reshape(-1)[brain_mask], 0.0, 1.0)
    positives = int(np.count_nonzero(labels))
    negatives = int(labels.size - positives)
    stats: dict[str, object] = {
        "evaluated_voxels": int(labels.size),
        "positive_voxels": positives,
        "mean_risk_in_recurrence": _rounded_mean(scores[labels]) if positives else None,
        "mean_risk_outside_recurrence": _rounded_mean(scores[~labels]) if negatives else None,
    }
    for fraction, key in ((0.01, "top_1pct"), (0.05, "top_5pct")):
        count = max(1, int(np.ceil(scores.size * fraction)))
        threshold = float(np.partition(scores, -count)[-count])
        predicted = scores >= threshold
        predicted_voxels = int(np.count_nonzero(predicted))
        overlap_voxels = int(np.count_nonzero(labels & predicted))
        denominator = positives + predicted_voxels
        stats[key] = {
            "fraction": fraction,
            "risk_threshold": round(threshold, 6),
            "predicted_voxels": predicted_voxels,
            "overlap_voxels": overlap_voxels,
            "recurrence_coverage": None if positives == 0 else round(float(overlap_voxels / positives), 6),
            "dice": None if denominator == 0 else round(float(2 * overlap_voxels / denominator), 6),
        }
    return stats


def _rounded_mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return round(float(np.mean(finite)), 6)


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


def select_preprocess_slices(
    *,
    shape: tuple[int, int, int],
    brain: np.ndarray,
    baseline_tumor: np.ndarray,
) -> list[dict[str, object]]:
    candidates: list[tuple[int, str]] = [(shape[2] // 2, "midline")]
    for label, data in (
        ("brain mask peak", brain),
        ("baseline tumor peak", baseline_tumor),
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
    return slices


def select_viewer_slices(
    *,
    shape: tuple[int, int, int],
    selected_slices: list[dict[str, object]],
    max_slices: int = MAX_BROWSER_SLICES,
) -> list[dict[str, object]]:
    """Select axial slices for the static browser without exploding report size."""

    z_count = int(shape[2])
    if z_count <= 0:
        return []

    selected_by_index: dict[int, str] = {}
    for item in selected_slices:
        try:
            index = int(item["index"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= index < z_count:
            selected_by_index[index] = str(item.get("reason", "selected slice"))

    if z_count <= max_slices:
        indices = list(range(z_count))
        default_reason = "all axial slices"
    else:
        reserved = max(1, max_slices - len(selected_by_index))
        sampled = np.linspace(0, z_count - 1, num=reserved, dtype=int).tolist()
        indices = sorted(set(sampled).union(selected_by_index))
        default_reason = "sampled axial slice"

    return [
        {
            "index": int(index),
            "reason": selected_by_index.get(index, default_reason),
        }
        for index in indices
    ]


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


def _write_preprocess_assets(
    target_dir: Path,
    case: CaseData,
    *,
    slices: list[dict[str, object]],
) -> list[dict[str, object]]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    assets: list[dict[str, object]] = []
    for item in slices:
        slice_index = int(item["index"])
        prefix = f"preprocess_z{slice_index:03d}"
        t1c_file = f"{prefix}_t1c.png"
        flair_file = f"{prefix}_flair.png"
        checkerboard_file = f"{prefix}_checkerboard.png"
        brain_file = f"{prefix}_brain.png"
        tumor_file = f"{prefix}_tumor.png"

        t1c_slice = case.t1c.data[:, :, slice_index]
        flair_slice = case.flair.data[:, :, slice_index]
        plt.imsave(target_dir / t1c_file, normalize_slice(t1c_slice), cmap="gray", vmin=0, vmax=1)
        plt.imsave(target_dir / flair_file, normalize_slice(flair_slice), cmap="gray", vmin=0, vmax=1)
        plt.imsave(target_dir / checkerboard_file, checkerboard_rgb(t1c_slice, flair_slice))
        plt.imsave(
            target_dir / brain_file,
            mask_overlay_rgba(case.brain_mask.data[:, :, slice_index], color=(0.25, 0.95, 0.35), alpha=0.45),
        )
        plt.imsave(
            target_dir / tumor_file,
            mask_overlay_rgba(case.baseline_tumor_mask.data[:, :, slice_index], color=(0.0, 0.85, 1.0), alpha=0.85),
        )

        assets.append(
            {
                "index": slice_index,
                "reason": item["reason"],
                "t1c": t1c_file,
                "flair": flair_file,
                "checkerboard": checkerboard_file,
                "brain": brain_file,
                "tumor": tumor_file,
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


def checkerboard_rgb(t1c: np.ndarray, flair: np.ndarray, *, block_size: int = 24) -> np.ndarray:
    t1c_image = normalize_slice(t1c)
    flair_image = normalize_slice(flair)
    rows, columns = np.indices(t1c_image.shape)
    use_t1c = ((rows // block_size) + (columns // block_size)) % 2 == 0
    checker = np.where(use_t1c, t1c_image, flair_image)
    return np.stack([checker, checker, checker], axis=-1).astype(np.float32)


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
    initial_offset = _initial_slice_offset(assets)
    initial_asset = assets[initial_offset]
    quick_jumps = _slice_jump_buttons(summary, assets)
    summary_table = _summary_table(summary)
    assets_json = _json_script_payload(assets)
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
<section aria-labelledby="timepoint-heading">
  <h2 id="timepoint-heading">Timepoint Context</h2>
  {_timepoint_context()}
</section>
<section aria-labelledby="controls-heading">
  <h2 id="controls-heading">Overlay Controls</h2>
  {_overlay_key()}
  <div class="controls">
    {_range_control("tumor", "Baseline tumor", "0.80")}
    {_range_control("recurrence", "Recurrence mask", "0.85")}
    {_range_control("risk", "Risk heatmap", "0.45")}
  </div>
</section>
<section aria-labelledby="slice-heading">
  <h2 id="slice-heading">Axial Slice Browser</h2>
  <div class="slice-browser">
    <div class="slice-controls">
      <label for="slice-slider">Slice <output id="slice-label">z={int(initial_asset["index"])}: {html.escape(str(initial_asset["reason"]))}</output></label>
      <input id="slice-slider" type="range" min="0" max="{len(assets) - 1}" step="1" value="{initial_offset}" data-initial-slice="{initial_offset}" data-slice-slider>
      <nav class="slice-jumps" aria-label="Representative slice jumps">
        {quick_jumps}
      </nav>
    </div>
    <div class="viewer-grid">
      {_viewer_stack("T1c", "t1c", initial_asset)}
      {_viewer_stack("FLAIR", "flair", initial_asset)}
    </div>
  </div>
  <script type="application/json" id="qc-slice-assets">{assets_json}</script>
</section>
"""


def _preprocess_interactive_report_body(summary: dict[str, object], assets: list[dict[str, object]]) -> str:
    patient_id = str(summary["patient_id"])
    initial_offset = _initial_slice_offset(assets)
    initial_asset = assets[initial_offset]
    quick_jumps = _slice_jump_buttons(summary, assets)
    summary_table = _preprocess_summary_table(summary)
    quality_flags = _quality_flags_list(summary.get("quality_flags", []))
    assets_json = _json_script_payload(assets)
    return f"""
<header>
  <p class="eyebrow">Preprocessing QC</p>
  <h1>{html.escape(patient_id)}</h1>
  <p class="disclaimer">{html.escape(RESEARCH_ONLY_DISCLAIMER)}</p>
</header>
<section aria-labelledby="preprocess-summary-heading">
  <h2 id="preprocess-summary-heading">Preprocessing Summary</h2>
  {summary_table}
  {quality_flags}
</section>
<section aria-labelledby="preprocess-controls-heading">
  <h2 id="preprocess-controls-heading">Mask Controls</h2>
  <div class="controls">
    {_range_control("brain", "Brain mask", "0.35")}
    {_range_control("tumor", "Baseline tumor", "0.65")}
  </div>
</section>
<section aria-labelledby="preprocess-slice-heading">
  <h2 id="preprocess-slice-heading">Axial Preprocessing Viewer</h2>
  <div class="slice-browser">
    <div class="slice-controls">
      <label for="slice-slider">Slice <output id="slice-label">z={int(initial_asset["index"])}: {html.escape(str(initial_asset["reason"]))}</output></label>
      <input id="slice-slider" type="range" min="0" max="{len(assets) - 1}" step="1" value="{initial_offset}" data-initial-slice="{initial_offset}" data-slice-slider>
      <nav class="slice-jumps" aria-label="Representative slice jumps">
        {quick_jumps}
      </nav>
    </div>
    <div class="viewer-grid">
      {_preprocess_viewer_stack("T1c", "t1c", initial_asset)}
      {_preprocess_viewer_stack("FLAIR", "flair", initial_asset)}
      {_preprocess_viewer_stack("T1c/FLAIR checkerboard", "checkerboard", initial_asset)}
    </div>
  </div>
  <script type="application/json" id="qc-slice-assets">{assets_json}</script>
</section>
"""


def _initial_slice_offset(assets: list[dict[str, object]]) -> int:
    for offset, asset in enumerate(assets):
        if asset.get("reason") == "midline":
            return offset
    return len(assets) // 2


def _overlay_key() -> str:
    return """
  <h3 class="overlay-key-heading">Overlay Key</h3>
  <div class="overlay-key" aria-label="Overlay color key">
    <div><span class="overlay-swatch tumor"></span><strong>Baseline tumor</strong><span>Cyan mask; baseline post-op / pre-radiotherapy</span></div>
    <div><span class="overlay-swatch recurrence"></span><strong>Recurrence mask</strong><span>Magenta mask; later follow-up label mapped to baseline</span></div>
    <div><span class="overlay-swatch risk"></span><strong>Risk heatmap</strong><span>Blue to orange model output in baseline space</span></div>
  </div>
"""


def _timepoint_context() -> str:
    return """
  <div class="timepoint-context">
    <div>
      <strong>Baseline MRI</strong>
      <span>Post-operative, pre-radiotherapy T1c and FLAIR. These are prediction-time inputs.</span>
    </div>
    <div>
      <strong>Baseline tumor mask</strong>
      <span>Baseline-space tumor mask used as a prediction-time location feature.</span>
    </div>
    <div>
      <strong>Recurrence mask</strong>
      <span>Later follow-up reviewed label mapped back to baseline space. It is used for training and evaluation only.</span>
    </div>
    <div>
      <strong>Risk heatmap</strong>
      <span>Model output in baseline space, not a treatment or dose recommendation.</span>
    </div>
  </div>
"""


def _slice_jump_buttons(summary: dict[str, object], assets: list[dict[str, object]]) -> str:
    offset_by_index = {int(asset["index"]): offset for offset, asset in enumerate(assets)}
    buttons: list[str] = []
    seen: set[int] = set()
    for item in summary.get("selected_slices", []):
        if not isinstance(item, dict):
            continue
        try:
            index = int(item["index"])
        except (KeyError, TypeError, ValueError):
            continue
        if index in seen or index not in offset_by_index:
            continue
        seen.add(index)
        reason = html.escape(str(item.get("reason", "selected slice")))
        buttons.append(
            f'<button type="button" data-slice-jump="{offset_by_index[index]}">'
            f'z={index} <span>{reason}</span></button>'
        )
    return "\n".join(buttons)


def _json_script_payload(value: object) -> str:
    return json.dumps(value).replace("</", "<\\/")


def _range_control(key: str, label: str, value: str) -> str:
    return (
        f'<label>{html.escape(label)} '
        f'<input type="range" min="0" max="1" step="0.05" value="{value}" data-opacity-control="{key}">'
        "</label>"
    )


def _viewer_stack(label: str, base_key: str, asset: dict[str, object]) -> str:
    escaped_label = html.escape(label)
    return f"""
<figure>
  <figcaption>{escaped_label} baseline post-op / pre-radiotherapy with overlays: cyan baseline tumor, magenta later recurrence label, blue/orange risk</figcaption>
  <div class="image-stack">
    <img src="{html.escape(str(asset[base_key]))}" alt="{escaped_label} anatomy slice" data-image-channel="{html.escape(base_key)}">
    <img src="{html.escape(str(asset["risk"]))}" alt="Risk heatmap overlay" class="overlay" data-image-channel="risk" data-overlay="risk">
    <img src="{html.escape(str(asset["tumor"]))}" alt="Baseline tumor overlay" class="overlay" data-image-channel="tumor" data-overlay="tumor">
    <img src="{html.escape(str(asset["recurrence"]))}" alt="Recurrence mask overlay" class="overlay" data-image-channel="recurrence" data-overlay="recurrence">
  </div>
</figure>
"""


def _preprocess_viewer_stack(label: str, base_key: str, asset: dict[str, object]) -> str:
    escaped_label = html.escape(label)
    return f"""
<figure>
  <figcaption>{escaped_label}</figcaption>
  <div class="image-stack">
    <img src="{html.escape(str(asset[base_key]))}" alt="{escaped_label} slice" data-image-channel="{html.escape(base_key)}">
    <img src="{html.escape(str(asset["brain"]))}" alt="Brain mask overlay" class="overlay" data-image-channel="brain" data-overlay="brain">
    <img src="{html.escape(str(asset["tumor"]))}" alt="Baseline tumor overlay" class="overlay" data-image-channel="tumor" data-overlay="tumor">
  </div>
</figure>
"""


def _summary_table(summary: dict[str, object]) -> str:
    recurrence_location = summary.get("recurrence_location")
    if isinstance(recurrence_location, dict):
        recurrence_inside = str(recurrence_location["inside_baseline_tumor_voxels"])
        recurrence_outside = (
            f'{recurrence_location["outside_baseline_tumor_voxels"]} '
            f'({recurrence_location["outside_baseline_tumor_fraction"]:.2%})'
        )
    else:
        recurrence_inside = "not available"
        recurrence_outside = "not available"
    prediction_overlap = summary.get("prediction_overlap")
    if isinstance(prediction_overlap, dict):
        mean_risk_in_recurrence = _format_optional_float(prediction_overlap.get("mean_risk_in_recurrence"))
        mean_risk_outside_recurrence = _format_optional_float(prediction_overlap.get("mean_risk_outside_recurrence"))
        top_1pct_overlap = _format_prediction_overlap(prediction_overlap.get("top_1pct"))
        top_5pct_overlap = _format_prediction_overlap(prediction_overlap.get("top_5pct"))
    else:
        mean_risk_in_recurrence = "not available"
        mean_risk_outside_recurrence = "not available"
        top_1pct_overlap = "not available"
        top_5pct_overlap = "not available"
    rows = [
        ("Shape", " x ".join(str(value) for value in summary["shape"])),
        ("Spacing mm", " / ".join(str(value) for value in summary["spacing_mm"])),
        ("Brain voxels", str(summary["brain_voxels"])),
        ("Baseline tumor voxels", str(summary["baseline_tumor_voxels"])),
        ("Recurrence mask present", str(summary["recurrence_mask_present"])),
        ("Recurrence voxels", "not available" if summary["recurrence_voxels"] is None else str(summary["recurrence_voxels"])),
        ("Recurrence inside baseline tumor", recurrence_inside),
        ("Recurrence outside baseline tumor", recurrence_outside),
        ("Risk map present", str(summary["risk_present"])),
        ("Mean risk in recurrence", mean_risk_in_recurrence),
        ("Mean risk outside recurrence", mean_risk_outside_recurrence),
        ("Top 1% risk overlap", top_1pct_overlap),
        ("Top 5% risk overlap", top_5pct_overlap),
        ("Viewer slices", str(summary["viewer_slice_count"])),
    ]
    body = "\n".join(
        f"<tr><th scope=\"row\">{_summary_label(label)}</th><td>{html.escape(value)}</td></tr>"
        for label, value in rows
    )
    return f"<table>{body}</table>"


def _format_optional_float(value: object) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.3f}"
    return "not available"


def _format_prediction_overlap(value: object) -> str:
    if not isinstance(value, dict):
        return "not available"
    overlap = value.get("overlap_voxels")
    predicted = value.get("predicted_voxels")
    coverage = value.get("recurrence_coverage")
    dice_value = value.get("dice")
    if not isinstance(overlap, int) or not isinstance(predicted, int):
        return "not available"
    coverage_text = "not available" if not isinstance(coverage, (float, int)) else f"{float(coverage):.2%}"
    dice_text = "not available" if not isinstance(dice_value, (float, int)) else f"{float(dice_value):.3f}"
    return f"{overlap} recurrence voxels in {predicted} high-risk voxels; coverage {coverage_text}; Dice {dice_text}"


def _summary_label(label: str) -> str:
    return _summary_label_with_help(label, SUMMARY_FIELD_HELP)


def _preprocess_summary_table(summary: dict[str, object]) -> str:
    steps = summary.get("preprocessing_steps", {})
    if not isinstance(steps, dict):
        steps = {}
    source_geometry = summary.get("source_geometry", {})
    if not isinstance(source_geometry, dict):
        source_geometry = {}
    source_flair_matches = source_geometry.get("flair_matches_t1c_before_preprocess")
    rows = [
        ("Shape", " x ".join(str(value) for value in summary["shape"])),
        ("Spacing mm", " / ".join(str(value) for value in summary["spacing_mm"])),
        ("Brain mask voxels", str(summary["brain_mask_voxels"])),
        ("Brain mask fraction", f'{float(summary["brain_mask_fraction"]):.2%}'),
        ("Baseline tumor voxels", str(summary["baseline_tumor_voxels"])),
        ("Baseline tumor fraction", f'{float(summary["baseline_tumor_fraction_of_brain"]):.2%} of brain mask'),
        ("Skull stripping", str(steps.get("skull_stripping_status", "not recorded"))),
        ("Bias correction", str(steps.get("bias_correction_status", "not recorded"))),
        ("FLAIR alignment", str(steps.get("flair_alignment", "not recorded"))),
        (
            "Registration QC",
            f'{steps.get("visual_registration_qc", "not recorded")}; source FLAIR matched T1c before preprocessing: {source_flair_matches}',
        ),
        ("Viewer slices", str(summary["viewer_slice_count"])),
    ]
    body = "\n".join(
        f"<tr><th scope=\"row\">{_preprocess_summary_label(label)}</th><td>{html.escape(value)}</td></tr>"
        for label, value in rows
    )
    return f"<table>{body}</table>"


def _preprocess_summary_label(label: str) -> str:
    return _summary_label_with_help(label, PREPROCESS_FIELD_HELP)


def _summary_label_with_help(label: str, help_by_label: dict[str, str]) -> str:
    help_text = help_by_label.get(label)
    escaped_label = html.escape(label)
    if help_text is None:
        return escaped_label
    escaped_help = html.escape(help_text)
    return (
        f'<span class="summary-label-text">{escaped_label}</span>'
        f'<span class="summary-help" tabindex="0" role="img" '
        f'aria-label="{escaped_help}" data-tooltip="{escaped_help}">i</span>'
    )


def _quality_flags_list(flags: object) -> str:
    if not isinstance(flags, list) or not flags:
        return '<p class="quality-flags ok">No automatic preprocessing QC flags.</p>'
    items = "\n".join(f"<li>{html.escape(str(flag))}</li>" for flag in flags)
    return f"""
<div class="quality-flags warn">
  <h3>Automatic QC Flags</h3>
  <ul>{items}</ul>
</div>
"""


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


def _preprocess_summary_only_report_body(summary: dict[str, object]) -> str:
    selected_slices = ", ".join(
        f'z={item["index"]} ({item["reason"]})' for item in summary["selected_slices"] if isinstance(item, dict)
    )
    return f"""
<header>
  <p class="eyebrow">Preprocessing QC</p>
  <h1>{html.escape(str(summary["patient_id"]))}</h1>
  <p class="disclaimer">{html.escape(RESEARCH_ONLY_DISCLAIMER)}</p>
</header>
<section>
  <h2>Preprocessing Summary</h2>
  {_preprocess_summary_table(summary)}
  {_quality_flags_list(summary.get("quality_flags", []))}
  <p>Matplotlib was unavailable, so preprocessing images were not rendered. Selected slices: {html.escape(selected_slices)}</p>
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
.summary-label-text { margin-right: 6px; }
.summary-help {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
  border: 1px solid #777;
  border-radius: 50%;
  color: #333;
  background: #f7f7f5;
  font-size: 11px;
  line-height: 1;
  cursor: help;
}
.summary-help::after {
  content: attr(data-tooltip);
  position: absolute;
  left: 22px;
  top: 50%;
  z-index: 20;
  width: min(300px, calc(100vw - 56px));
  padding: 8px 10px;
  border: 1px solid #333;
  background: #fff;
  color: #111;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.18);
  font-size: 12px;
  font-weight: 400;
  line-height: 1.35;
  opacity: 0;
  pointer-events: none;
  transform: translateY(-50%);
  transition: opacity 120ms ease;
}
.summary-help:hover::after,
.summary-help:focus::after { opacity: 1; }
.summary-help:focus { outline: 2px solid #1a73e8; outline-offset: 2px; }
.timepoint-context {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 10px;
}
.timepoint-context div {
  display: grid;
  gap: 4px;
  padding: 10px 12px;
  border: 1px solid #ddd;
  background: #fff;
}
.timepoint-context strong { font-size: 14px; }
.timepoint-context span { color: #555; font-size: 13px; line-height: 1.35; }
.overlay-key-heading { margin: 0 0 8px; }
.overlay-key {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 10px;
  margin: 0 0 10px;
}
.overlay-key div {
  display: grid;
  grid-template-columns: 22px 1fr;
  gap: 2px 8px;
  align-items: center;
  padding: 8px 10px;
  border: 1px solid #ddd;
  background: #fff;
}
.overlay-key strong { font-size: 14px; }
.overlay-key span:last-child { grid-column: 2; color: #555; font-size: 12px; }
.overlay-swatch {
  grid-row: span 2;
  width: 18px;
  height: 18px;
  border: 1px solid #222;
}
.overlay-swatch.tumor { background: rgb(0, 217, 255); }
.overlay-swatch.recurrence { background: rgb(255, 38, 217); }
.overlay-swatch.risk { background: linear-gradient(90deg, rgb(0, 89, 255), rgb(255, 89, 0)); }
.controls { display: flex; flex-wrap: wrap; gap: 14px; padding: 12px; background: #fff; border: 1px solid #ddd; }
.controls label { display: flex; gap: 8px; align-items: center; white-space: nowrap; }
.quality-flags { margin: 12px 0 0; padding: 10px 12px; border: 1px solid #ddd; background: #fff; }
.quality-flags h3 { margin: 0 0 8px; }
.quality-flags ul { margin: 0; padding-left: 18px; }
.quality-flags.ok { border-color: #8abf8a; background: #f1fbf1; }
.quality-flags.warn { border-color: #c49a3a; background: #fff8e6; }
.slice-browser { background: #fff; border: 1px solid #ddd; padding: 12px; }
.slice-controls { display: grid; gap: 10px; margin-bottom: 14px; }
.slice-controls label { display: flex; justify-content: space-between; gap: 12px; font-weight: 600; }
.slice-controls input[type="range"] { width: 100%; }
.slice-jumps { display: flex; flex-wrap: wrap; gap: 8px; }
.slice-jumps button { border: 1px solid #aaa; background: #fff; padding: 8px 10px; cursor: pointer; }
.slice-jumps button.active { border-color: #111; background: #111; color: #fff; }
.slice-jumps span { display: block; font-size: 11px; opacity: 0.78; }
.viewer-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
figure { margin: 0; background: #fff; border: 1px solid #ddd; padding: 10px; }
figcaption { margin-bottom: 8px; font-weight: 600; }
.image-stack { position: relative; width: 100%; background: #000; }
.image-stack img { display: block; width: 100%; height: auto; }
.image-stack .overlay { position: absolute; inset: 0; pointer-events: none; }
</style>
"""


def _report_javascript(assets: list[dict[str, object]]) -> str:
    return """
<script>
const sliceAssets = JSON.parse(document.getElementById('qc-slice-assets').textContent);

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

function setSlice(offset) {
  const bounded = Math.max(0, Math.min(sliceAssets.length - 1, offset));
  const asset = sliceAssets[bounded];
  const slider = document.querySelector('[data-slice-slider]');
  const label = document.getElementById('slice-label');
  if (slider) {
    slider.value = String(bounded);
  }
  if (label) {
    label.value = `z=${asset.index}: ${asset.reason}`;
    label.textContent = `z=${asset.index}: ${asset.reason}`;
  }
  for (const image of document.querySelectorAll('[data-image-channel]')) {
    const channel = image.dataset.imageChannel;
    if (asset[channel]) {
      image.src = asset[channel];
    }
  }
  for (const button of document.querySelectorAll('[data-slice-jump]')) {
    button.classList.toggle('active', Number(button.dataset.sliceJump) === bounded);
  }
}

const slider = document.querySelector('[data-slice-slider]');
if (slider) {
  slider.addEventListener('input', () => setSlice(Number(slider.value)));
}
for (const button of document.querySelectorAll('[data-slice-jump]')) {
  button.addEventListener('click', () => {
    setSlice(Number(button.dataset.sliceJump));
  });
}
const initialSliceOffset = Number(slider?.dataset.initialSlice || 0);
setSlice(initialSliceOffset);
</script>
"""
