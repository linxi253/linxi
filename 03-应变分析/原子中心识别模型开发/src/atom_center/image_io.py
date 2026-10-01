"""Explicit and reproducible image loading for microscopy data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class ImageReadError(RuntimeError):
    """Raised when a file cannot be converted to one unambiguous 2-D image."""


class AmbiguousImageError(ImageReadError):
    """Raised when a stack is supplied without an explicit frame index."""


@dataclass(frozen=True)
class ImageInfo:
    path: Path
    original_shape: tuple[int, ...]
    axes: str
    series_index: int
    frame_index: int | None
    normalized: bool


@dataclass(frozen=True)
class LoadedImage:
    image: NDArray[np.float32]
    info: ImageInfo


@dataclass(frozen=True)
class ImageSeriesInfo:
    """Shape information for one independently selectable image series."""

    path: Path
    series_index: int
    original_shape: tuple[int, ...]
    axes: str
    frame_count: int
    plane_shape: tuple[int, int]


def normalize_percentile(
    image: NDArray[np.generic],
    *,
    low: float = 1.0,
    high: float = 99.0,
) -> NDArray[np.float32]:
    """Map finite intensities to ``[0, 1]`` using fixed percentiles."""

    if not (0.0 <= low < high <= 100.0):
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"normalization expects a 2-D image, got {array.shape}")
    finite = np.isfinite(array)
    if not finite.any():
        raise ImageReadError("image contains no finite pixels")
    lo, hi = np.percentile(array[finite], [low, high])
    if hi <= lo:
        return np.zeros(array.shape, dtype=np.float32)
    clean = np.where(finite, array, lo)
    normalized = np.clip((clean - lo) / (hi - lo), 0.0, 1.0)
    return normalized.astype(np.float32, copy=False)


def _to_grayscale(image: NDArray[np.generic]) -> NDArray[np.generic]:
    if image.ndim == 2:
        return image
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        raise ImageReadError(f"unsupported color image shape: {image.shape}")
    if image.shape[-1] == 1:
        return image[..., 0]
    rgb = np.asarray(image[..., :3], dtype=np.float64)
    # Standard luminance weights; alpha is deliberately ignored.
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _describe_plane_layout(
    shape: tuple[int, ...],
    axes: str,
) -> tuple[int, tuple[int, int]]:
    """Return ``(frame_count, (height, width))`` without reading pixel data."""

    normalized_axes = axes.upper()
    if len(normalized_axes) != len(shape):
        normalized_axes = ""

    if len(shape) == 2:
        return 1, (int(shape[0]), int(shape[1]))

    if normalized_axes and "Y" in normalized_axes and "X" in normalized_axes:
        y_axis = normalized_axes.index("Y")
        x_axis = normalized_axes.index("X")
        channel_axis = next(
            (
                index
                for index, code in enumerate(normalized_axes)
                if code in {"S", "C"} and shape[index] in (1, 3, 4)
            ),
            None,
        )
        leading_shape = [
            size
            for index, size in enumerate(shape)
            if index not in {y_axis, x_axis, channel_axis}
        ]
        frame_count = (
            int(np.prod(leading_shape, dtype=np.int64)) if leading_shape else 1
        )
        return frame_count, (int(shape[y_axis]), int(shape[x_axis]))

    # Conservative fallback matching ``_select_tiff_plane``.
    if len(shape) == 3 and shape[-1] in (1, 3, 4):
        return 1, (int(shape[0]), int(shape[1]))
    if len(shape) == 3:
        return int(shape[0]), (int(shape[1]), int(shape[2]))
    raise ImageReadError(
        f"cannot infer 2-D planes from shape {shape} and axes {axes!r}"
    )


def inspect_image_series(path: str | Path) -> tuple[ImageSeriesInfo, ...]:
    """Inspect selectable frames without silently collapsing a TIFF stack."""

    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise ImageReadError(f"image file does not exist: {image_path}")

    try:
        if image_path.suffix.lower() in {".tif", ".tiff"}:
            import tifffile

            descriptions: list[ImageSeriesInfo] = []
            with tifffile.TiffFile(image_path) as tif:
                for series_index, series in enumerate(tif.series):
                    shape = tuple(int(value) for value in series.shape)
                    axes = series.axes or ""
                    frame_count, plane_shape = _describe_plane_layout(shape, axes)
                    descriptions.append(
                        ImageSeriesInfo(
                            path=image_path,
                            series_index=series_index,
                            original_shape=shape,
                            axes=axes,
                            frame_count=frame_count,
                            plane_shape=plane_shape,
                        )
                    )
            if not descriptions:
                raise ImageReadError(f"TIFF contains no readable series: {image_path}")
            return tuple(descriptions)

        from PIL import Image

        with Image.open(image_path) as handle:
            width, height = handle.size
            bands = len(handle.getbands())
        shape = (height, width) if bands == 1 else (height, width, bands)
        return (
            ImageSeriesInfo(
                path=image_path,
                series_index=0,
                original_shape=tuple(int(value) for value in shape),
                axes="YX" if bands == 1 else "YXS",
                frame_count=1,
                plane_shape=(int(height), int(width)),
            ),
        )
    except ImageReadError:
        raise
    except Exception as exc:
        raise ImageReadError(f"failed to inspect {image_path}: {exc}") from exc


def _select_tiff_plane(
    array: NDArray[np.generic],
    axes: str,
    frame_index: int | None,
) -> tuple[NDArray[np.generic], int | None]:
    axes = axes.upper()
    if len(axes) != array.ndim:
        axes = ""

    if array.ndim == 2:
        return array, None

    if axes and "Y" in axes and "X" in axes:
        y_axis = axes.index("Y")
        x_axis = axes.index("X")
        channel_axis = next(
            (
                index
                for index, code in enumerate(axes)
                if code in {"S", "C"} and array.shape[index] in (1, 3, 4)
            ),
            None,
        )
        leading_axes = [
            index
            for index in range(array.ndim)
            if index not in {y_axis, x_axis, channel_axis}
        ]
        permutation = leading_axes + [y_axis, x_axis]
        if channel_axis is not None:
            permutation.append(channel_axis)
        ordered = np.transpose(array, permutation)
        frame_shape = ordered.shape[: len(leading_axes)]
        frame_count = int(np.prod(frame_shape, dtype=np.int64)) if frame_shape else 1
        tail_shape = ordered.shape[len(leading_axes) :]
        frames = ordered.reshape((frame_count,) + tail_shape)
        if frame_count > 1 and frame_index is None:
            raise AmbiguousImageError(
                f"TIFF contains {frame_count} frames (axes={axes}, shape={array.shape}); "
                "pass frame_index explicitly"
            )
        selected_index = 0 if frame_index is None else frame_index
        if selected_index < 0 or selected_index >= frame_count:
            raise IndexError(
                f"frame_index {selected_index} outside [0, {frame_count - 1}]"
            )
        return _to_grayscale(frames[selected_index]), (
            selected_index if frame_count > 1 else None
        )

    # Conservative fallback for TIFFs without useful axes metadata.
    if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
        return _to_grayscale(array), None
    if array.ndim == 3:
        if frame_index is None:
            raise AmbiguousImageError(
                f"TIFF contains {array.shape[0]} probable frames with shape {array.shape}; "
                "pass frame_index explicitly"
            )
        if frame_index < 0 or frame_index >= array.shape[0]:
            raise IndexError(
                f"frame_index {frame_index} outside [0, {array.shape[0] - 1}]"
            )
        return array[frame_index], frame_index
    raise ImageReadError(
        f"cannot infer a 2-D plane from TIFF shape {array.shape} and axes {axes!r}"
    )


def load_image(
    path: str | Path,
    *,
    frame_index: int | None = None,
    series_index: int = 0,
    normalize: bool = True,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
    preserve_dtype: bool = False,
) -> LoadedImage:
    """Load one explicitly selected microscopy image.

    A multi-frame TIFF never silently becomes an averaged pseudo-image.  Its
    frame must be selected explicitly with ``frame_index``.
    """

    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise ImageReadError(f"image file does not exist: {image_path}")

    suffix = image_path.suffix.lower()
    try:
        if suffix in {".tif", ".tiff"}:
            import tifffile

            with tifffile.TiffFile(image_path) as tif:
                if series_index < 0 or series_index >= len(tif.series):
                    raise IndexError(
                        f"series_index {series_index} outside [0, {len(tif.series) - 1}]"
                    )
                series = tif.series[series_index]
                raw = series.asarray()
                axes = series.axes or ""
            plane, selected_frame = _select_tiff_plane(raw, axes, frame_index)
        else:
            from PIL import Image

            with Image.open(image_path) as handle:
                raw = np.asarray(handle)
            axes = "YX" if raw.ndim == 2 else "YXS"
            plane = _to_grayscale(raw)
            selected_frame = None
    except (AmbiguousImageError, ImageReadError, IndexError):
        raise
    except Exception as exc:
        raise ImageReadError(f"failed to read {image_path}: {exc}") from exc

    if plane.ndim != 2:
        raise ImageReadError(f"selected image is not 2-D: {plane.shape}")
    if normalize:
        output = normalize_percentile(
            plane, low=percentile_low, high=percentile_high
        )
    else:
        output = np.asarray(plane) if preserve_dtype else np.asarray(plane, dtype=np.float32)

    return LoadedImage(
        image=output,
        info=ImageInfo(
            path=image_path,
            original_shape=tuple(int(value) for value in np.shape(raw)),
            axes=axes,
            series_index=series_index,
            frame_index=selected_frame,
            normalized=normalize,
        ),
    )


def read_image(path: str | Path, **kwargs: object) -> NDArray[np.float32]:
    """Convenience wrapper returning only the 2-D image array."""

    return load_image(path, **kwargs).image
