"""NIfTI IO boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import Volume


class NiftiError(RuntimeError):
    """Raised when NIfTI IO fails."""


def read_volume(path: str | Path) -> Volume:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise NiftiError("nibabel is required for NIfTI IO") from exc

    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(dtype=np.float32))
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise NiftiError(f"{path} must contain a 3D volume; got shape {data.shape}")
    return Volume(data=data, affine=np.asarray(image.affine, dtype=float), metadata={"path": str(path)})


def write_volume(volume: Volume, path: str | Path, *, dtype: np.dtype | type | None = None) -> None:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise NiftiError("nibabel is required for NIfTI IO") from exc

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = volume.data.astype(dtype) if dtype is not None else volume.data
    image = nib.Nifti1Image(data, volume.affine)
    image.header.set_xyzt_units("mm")
    nib.save(image, str(target))

