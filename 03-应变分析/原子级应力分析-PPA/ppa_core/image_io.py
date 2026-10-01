"""Safe image loading for analysis inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class ImageLoadError(ValueError):
    pass


@dataclass(frozen=True)
class ImageData:
    pixels: np.ndarray
    source_path: str
    frame_index: int | None
    frame_count: int
    source_shape: tuple[int, ...]
    source_dtype: str


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        return image[..., :3].mean(axis=-1)
    raise ImageLoadError(
        f"Expected a 2-D image or RGB/RGBA image, received shape {image.shape}. "
        "For stacks, select a single frame explicitly."
    )


def load_analysis_image(path: str | Path, *, frame_index: int | None = None) -> ImageData:
    """Load one analysis frame without confusing a TIFF stack with RGB channels."""
    source_path = str(Path(path).resolve())
    try:
        # 懒导入 tifffile: 无该依赖的环境 (最小 CI 容器) 只要不用 TIFF 读取就能导入本模块。
        import tifffile
        source = tifffile.imread(source_path)
    except Exception as tif_exc:
        try:
            from PIL import Image
            with Image.open(source_path) as image:
                source = np.asarray(image)
        except Exception as pil_exc:
            raise ImageLoadError(f"Unable to read image: {tif_exc}; fallback reader: {pil_exc}") from pil_exc
    source = np.asarray(source)
    source_shape = tuple(source.shape)
    frame_count = 1
    selected_frame: int | None = None
    if source.ndim == 3 and source.shape[-1] not in (3, 4):
        frame_count = int(source.shape[0])
        if frame_index is None:
            raise ImageLoadError(f"This TIFF contains {frame_count} frames; select a frame before analysis.")
        if not 0 <= frame_index < frame_count:
            raise ImageLoadError(f"Frame index {frame_index} is outside [0, {frame_count - 1}].")
        source = source[frame_index]
        selected_frame = int(frame_index)
    elif source.ndim > 3:
        raise ImageLoadError(f"Unsupported image shape {source_shape}; select or export one 2-D frame first.")
    image = _to_grayscale(source).astype(np.float64, copy=False)
    if image.size == 0 or not np.isfinite(image).all():
        raise ImageLoadError("Image is empty or contains non-finite pixels.")
    lo, hi = np.percentile(image, (1, 99))
    if hi > lo:
        image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    else:
        image = np.zeros_like(image, dtype=np.float64)
    return ImageData(image, source_path, selected_frame, frame_count, source_shape, str(source.dtype))
