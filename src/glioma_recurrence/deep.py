"""Optional MONAI/PyTorch 3D U-Net training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .case import CaseData
from .preprocess import distance_to_mask_mm


class DeepLearningUnavailable(RuntimeError):
    """Raised when MONAI/PyTorch extras are not installed."""


def require_deep_dependencies():
    try:
        import monai
        import torch
    except ImportError as exc:
        raise DeepLearningUnavailable(
            "Install the deep extra to train the MONAI model: "
            "`uv sync --extra deep` or `pip install .[deep]`."
        ) from exc
    return monai, torch


def build_unet(*, in_channels: int = 4, out_channels: int = 1):
    """Build the V1 MONAI 3D U-Net architecture.

    Inputs are baseline T1c, baseline FLAIR, baseline tumor mask, and distance
    from baseline tumor. Follow-up imaging is deliberately absent.
    """

    monai, _torch = require_deep_dependencies()
    return monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    )


@dataclass(frozen=True)
class DeepTrainingConfig:
    patch_size: tuple[int, int, int] = (96, 96, 96)
    max_epochs: int = 20
    learning_rate: float = 1e-4
    focal_gamma: float = 2.0
    positive_weight: float = 10.0


def train_unet(
    cases: list[CaseData],
    *,
    output_path: str | Path,
    config: DeepTrainingConfig = DeepTrainingConfig(),
    seed: int = 13,
    device: str | None = None,
) -> None:
    """Train a compact MONAI 3D U-Net on cropped case patches.

    This is intentionally a V1 research trainer: it uses one sampled patch per
    case per epoch and writes a checkpoint for later `predict_unet` inference.
    Baselines remain the acceptance gate before treating this model as
    scientifically interesting.
    """

    monai, torch = require_deep_dependencies()
    rng = np.random.default_rng(seed)
    runtime_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_unet(in_channels=4, out_channels=1).to(runtime_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    dice_loss = monai.losses.DiceLoss(sigmoid=True)
    bce_loss = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([config.positive_weight], dtype=torch.float32, device=runtime_device)
    )
    model.train()
    history: list[dict[str, float]] = []
    for epoch in range(config.max_epochs):
        losses: list[float] = []
        for case in cases:
            x_np, y_np = sample_training_patch(
                case,
                patch_size=config.patch_size,
                rng=rng,
            )
            x = torch.from_numpy(x_np[None]).to(runtime_device)
            y = torch.from_numpy(y_np[None]).to(runtime_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = bce_loss(logits, y) + dice_loss(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": float(epoch + 1), "loss": float(np.mean(losses))})

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "patch_size": config.patch_size,
                "max_epochs": config.max_epochs,
                "learning_rate": config.learning_rate,
                "focal_gamma": config.focal_gamma,
                "positive_weight": config.positive_weight,
            },
            "history": history,
            "input_channels": ["baseline_t1c", "baseline_flair", "baseline_tumor_mask", "distance_to_baseline_tumor"],
            "safety": "follow-up scans are labels only and are not prediction inputs",
        },
        target,
    )


def predict_unet(case: CaseData, *, checkpoint_path: str | Path, device: str | None = None) -> np.ndarray:
    monai, torch = require_deep_dependencies()
    runtime_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=runtime_device)
    model = build_unet(in_channels=4, out_channels=1).to(runtime_device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    patch_size = tuple(int(value) for value in checkpoint.get("config", {}).get("patch_size", (96, 96, 96)))
    features = case_input_channels(case)
    x = torch.from_numpy(features[None]).to(runtime_device)
    with torch.no_grad():
        logits = monai.inferers.sliding_window_inference(
            x,
            roi_size=patch_size,
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
        )
        risk = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    risk[~case.brain_mask.data.astype(bool)] = 0.0
    return np.clip(risk, 0.0, 1.0)


def case_input_channels(case: CaseData) -> np.ndarray:
    baseline_tumor = case.baseline_tumor_mask.data.astype(bool)
    distance = distance_to_mask_mm(baseline_tumor, case.t1c.spacing)
    return np.stack(
        [
            case.t1c.data.astype(np.float32),
            case.flair.data.astype(np.float32),
            baseline_tumor.astype(np.float32),
            distance.astype(np.float32),
        ],
        axis=0,
    )


def sample_training_patch(
    case: CaseData,
    *,
    patch_size: tuple[int, int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if case.recurrence_mask is None:
        raise ValueError(f"{case.patient_id}: recurrence mask is required for deep training")
    features = case_input_channels(case)
    label = case.recurrence_mask.data.astype(np.float32)[None]
    center = choose_patch_center(case, rng=rng)
    x = crop_or_pad(features, center=center, patch_size=patch_size)
    y = crop_or_pad(label, center=center, patch_size=patch_size)
    return x.astype(np.float32), y.astype(np.float32)


def choose_patch_center(case: CaseData, *, rng: np.random.Generator) -> tuple[int, int, int]:
    labels = case.recurrence_mask.data.astype(bool) if case.recurrence_mask is not None else None
    baseline_tumor = case.baseline_tumor_mask.data.astype(bool)
    brain = case.brain_mask.data.astype(bool)
    candidates = None
    if labels is not None and np.any(labels) and rng.random() < 0.6:
        candidates = np.argwhere(labels)
    elif np.any(baseline_tumor & brain):
        candidates = np.argwhere(baseline_tumor & brain)
    elif np.any(brain):
        candidates = np.argwhere(brain)
    if candidates is None or candidates.size == 0:
        return tuple(int(value // 2) for value in case.t1c.shape)
    return tuple(int(value) for value in candidates[rng.integers(0, len(candidates))])


def crop_or_pad(array: np.ndarray, *, center: tuple[int, int, int], patch_size: tuple[int, int, int]) -> np.ndarray:
    channels = array.shape[0]
    output = np.zeros((channels, *patch_size), dtype=array.dtype)
    source_slices = []
    target_slices = []
    for axis, (center_value, size, max_size) in enumerate(zip(center, patch_size, array.shape[1:])):
        start = int(center_value - size // 2)
        stop = start + int(size)
        source_start = max(start, 0)
        source_stop = min(stop, max_size)
        target_start = source_start - start
        target_stop = target_start + (source_stop - source_start)
        source_slices.append(slice(source_start, source_stop))
        target_slices.append(slice(target_start, target_stop))
    output[(slice(None), *target_slices)] = array[(slice(None), *source_slices)]
    return output
