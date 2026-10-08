"""Frame-aware image loading with the decoded file's identity snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from .project_store import image_identity

MAX_IMAGE_PIXELS = 100_000_000


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
    identity: dict | None = None


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        return image[..., :3].mean(axis=-1)
    raise ImageLoadError(
        f"Expected a 2-D image or RGB/RGBA image, received shape {image.shape}. "
        "For stacks, select a single frame explicitly."
    )


def load_analysis_image(path: str | Path, *, frame_index: int | None = None,
                        expected_identity: dict | None = None) -> ImageData:
    """Load one analysis frame without confusing a TIFF stack with RGB channels."""
    source_path = str(Path(path).resolve())
    if frame_index is not None and (type(frame_index) is not int or frame_index < 0):
        raise ImageLoadError("Frame index must be a non-negative integer.")
    try:
        before = image_identity(source_path, frame_index=frame_index)
        if expected_identity and any(before[k] != expected_identity.get(k)
                                     for k in ('sha256', 'size_bytes', 'frame_index')):
            raise ImageLoadError("Image does not match the project's saved identity.")
        if Path(source_path).suffix.lower() in ('.tif', '.tiff'):
            import tifffile
            with tifffile.TiffFile(source_path) as tif:
                if len(tif.series) != 1:
                    raise ImageLoadError("Multiple TIFF series: export/select one series explicitly.")
                series = tif.series[0]
                source_shape, source_dtype, axes = tuple(series.shape), str(series.dtype), series.axes
                rgb = axes.endswith('YXS') and source_shape[-1] in (3, 4)
                planar = axes.endswith('SYX') and source_shape[-3] in (3, 4)
                image_dims = 3 if rgb or planar else 2
                leading = source_shape[:-image_dims]
                if len(leading) > 1 or (image_dims == 2 and not axes.endswith('YX')):
                    raise ImageLoadError(f"Unsupported TIFF axes {axes}; export one 2-D frame.")
                frame_count = int(leading[0]) if leading else 1
                selected_frame = frame_index if leading else None
                if leading and frame_index is None:
                    raise ImageLoadError(f"This TIFF contains {frame_count} frames; select a frame before analysis.")
                if frame_index is not None and frame_index >= frame_count:
                    raise ImageLoadError(f"Frame index {frame_index} is outside [0, {frame_count-1}].")
                if int(np.prod(source_shape[-image_dims:])) > MAX_IMAGE_PIXELS*(4 if rgb or planar else 1):
                    raise ImageLoadError("Selected image exceeds the supported pixel limit.")
                if leading:
                    if len(series.pages) != frame_count:
                        raise ImageLoadError("Unsupported packed TIFF stack; export the requested frame.")
                    source = series.pages[frame_index].asarray()
                else:
                    source = series.asarray()
                if planar:
                    source = np.moveaxis(source, -3, -1)
        else:
            from PIL import Image
            with Image.open(source_path) as image:
                frame_count = int(getattr(image, 'n_frames', 1))
                if frame_count > 1 and frame_index is None:
                    raise ImageLoadError(f"This image contains {frame_count} frames; select a frame before analysis.")
                if frame_index is not None and frame_index >= frame_count:
                    raise ImageLoadError(f"Frame index {frame_index} is outside [0, {frame_count-1}].")
                if image.width*image.height > MAX_IMAGE_PIXELS:
                    raise ImageLoadError("Selected image exceeds the supported pixel limit.")
                if frame_index is not None:
                    image.seek(frame_index)
                source = np.asarray(image.convert('RGB') if image.mode == 'P' else image).copy()
                source_shape, source_dtype = tuple(source.shape), str(source.dtype)
                selected_frame = frame_index if frame_count > 1 else None
        after = image_identity(source_path, frame_index=frame_index)
        if before != after:
            raise ImageLoadError("Source image changed while it was being read.")
    except ImageLoadError:
        raise
    except Exception as exc:
        raise ImageLoadError(f"Unable to read image: {exc}") from exc
    image = _to_grayscale(source).astype(np.float64, copy=False)
    if image.size == 0 or image.size > MAX_IMAGE_PIXELS or not np.isfinite(image).all():
        raise ImageLoadError("Image is empty or contains non-finite pixels.")
    lo, hi = np.percentile(image, (1, 99))
    if hi > lo:
        image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    else:
        image = np.zeros_like(image, dtype=np.float64)
    after['frame_index'] = selected_frame
    return ImageData(image, source_path, selected_frame, frame_count, source_shape, source_dtype, after)
