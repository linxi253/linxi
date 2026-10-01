"""Intensity normalization, CLAHE, and stack-wide range estimation."""

import logging
from collections.abc import Callable

import cv2
import numpy as np

from errors import OperationCancelled

logger = logging.getLogger(__name__)

DEFAULT_LOW_PERCENTILE = 0.5
DEFAULT_HIGH_PERCENTILE = 99.5

# CLAHE 局部直方图均衡的分块网格；同时记录进溯源 sidecar 以保证可复现。
CLAHE_TILE_GRID: tuple[int, int] = (8, 8)


def _validated_range(image: np.ndarray, vmin, vmax) -> tuple[float, float]:
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("图像不能为空")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"不支持的图像类型: {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("图像包含 NaN 或 Inf")

    if vmin is None:
        vmin = float(np.percentile(array, 0.5))
    if vmax is None:
        vmax = float(np.percentile(array, 99.5))
    vmin = float(vmin)
    vmax = float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("归一化范围必须是有限数值")
    if vmax < vmin:
        raise ValueError("归一化最大值不能小于最小值")
    return vmin, vmax


def _normalize(
    image: np.ndarray,
    maximum: int,
    dtype: np.dtype,
    vmin=None,
    vmax=None,
) -> np.ndarray:
    array = np.asarray(image)
    vmin, vmax = _validated_range(array, vmin, vmax)
    if vmax - vmin <= np.finfo(np.float64).eps:
        logger.warning("图像动态范围接近零，输出零值图像")
        return np.zeros(array.shape, dtype=dtype)
    scaled = (array.astype(np.float32, copy=False) - vmin) / (vmax - vmin)
    scaled = np.clip(scaled, 0.0, 1.0) * maximum
    return np.rint(scaled).astype(dtype)


def normalize_to_uint8(
    image: np.ndarray,
    vmin: float = None,
    vmax: float = None,
) -> np.ndarray:
    """Linearly map ``[vmin, vmax]`` to the full uint8 range."""
    return _normalize(image, 255, np.uint8, vmin, vmax)


def normalize_to_uint16(
    image: np.ndarray,
    vmin: float = None,
    vmax: float = None,
) -> np.ndarray:
    """Linearly map ``[vmin, vmax]`` to the full uint16 range."""
    return _normalize(image, 65535, np.uint16, vmin, vmax)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 1.2,
    tile_grid_size: tuple[int, int] = CLAHE_TILE_GRID,
) -> np.ndarray:
    """Apply CLAHE to a two-dimensional uint8 image."""
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("CLAHE 仅支持二维灰度图像")
    if array.dtype != np.uint8:
        array = normalize_to_uint8(array)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    return clahe.apply(array)


def apply_clahe_uint16(
    image: np.ndarray,
    clip_limit: float = 1.2,
    tile_grid_size: tuple[int, int] = CLAHE_TILE_GRID,
) -> np.ndarray:
    """Apply CLAHE to a two-dimensional uint16 image."""
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("CLAHE 仅支持二维灰度图像")
    if array.dtype != np.uint16:
        array = normalize_to_uint16(array)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    return clahe.apply(array)


def uint16_to_display_uint8(image: np.ndarray) -> np.ndarray:
    """Convert full-range uint16 output to uint8 without re-windowing."""
    array = np.asarray(image, dtype=np.uint16)
    return ((array.astype(np.uint32) + 128) // 257).astype(np.uint8)


def _histogram_percentile(histogram: np.ndarray, percentile: float) -> int:
    total = int(histogram.sum())
    if total <= 0:
        raise ValueError("无法从空直方图估计范围")
    rank = percentile / 100.0 * (total - 1)
    cumulative = np.cumsum(histogram, dtype=np.uint64)
    return int(np.searchsorted(cumulative, rank + 1, side="left"))


def estimate_global_range(
    reader,
    num_samples: int = None,
    *,
    low_percentile: float = DEFAULT_LOW_PERCENTILE,
    high_percentile: float = DEFAULT_HIGH_PERCENTILE,
    cancel_event=None,
    progress_callback: Callable[[int, int], None] | None = None,
    max_float_samples: int = 2_000_000,
) -> tuple[float, float]:
    """
    Estimate a consistent stack-wide range.

    uint8 and uint16 inputs use an exact histogram over every frame.  Other
    numeric types use deterministic samples from every frame, bounded by
    ``max_float_samples``.  ``num_samples`` remains accepted for source
    compatibility but is intentionally ignored.
    """
    del num_samples
    if not (0.0 <= low_percentile < high_percentile <= 100.0):
        raise ValueError("百分位范围无效")

    frame_count = int(reader.num_frames)
    if frame_count <= 0:
        raise ValueError("TIFF 不包含可处理帧")
    dtype = np.dtype(reader.dtype)

    if dtype in (np.dtype(np.uint8), np.dtype(np.uint16)):
        bins = 256 if dtype == np.dtype(np.uint8) else 65536
        histogram = np.zeros(bins, dtype=np.uint64)
        for index in range(frame_count):
            if cancel_event is not None and cancel_event.is_set():
                raise OperationCancelled("已取消强度范围分析")
            frame = reader.read_frame(index)
            values = np.asarray(frame, dtype=dtype).ravel()
            histogram += np.bincount(values, minlength=bins).astype(
                np.uint64, copy=False
            )
            if progress_callback is not None:
                progress_callback(index + 1, frame_count)
        vmin = float(_histogram_percentile(histogram, low_percentile))
        vmax = float(_histogram_percentile(histogram, high_percentile))
    else:
        per_frame = max(1, max_float_samples // frame_count)
        samples = []
        for index in range(frame_count):
            if cancel_event is not None and cancel_event.is_set():
                raise OperationCancelled("已取消强度范围分析")
            frame = np.asarray(reader.read_frame(index))
            if not np.isfinite(frame).all():
                raise ValueError(f"帧 {index} 包含 NaN 或 Inf")
            flat = frame.ravel()
            if flat.size > per_frame:
                positions = np.linspace(0, flat.size - 1, per_frame, dtype=np.int64)
                flat = flat[positions]
            samples.append(flat.astype(np.float64, copy=False))
            if progress_callback is not None:
                progress_callback(index + 1, frame_count)
        values = np.concatenate(samples)
        low_value, high_value = np.percentile(
            values, (low_percentile, high_percentile)
        )
        vmin, vmax = float(low_value), float(high_value)

    if vmax <= vmin:
        if dtype == np.dtype(np.uint8):
            vmin, vmax = 0.0, 255.0
        elif dtype == np.dtype(np.uint16):
            vmin, vmax = 0.0, 65535.0
        else:
            half_width = max(abs(vmin) * 1e-6, 0.5)
            vmin, vmax = vmin - half_width, vmax + half_width
        logger.warning(
            "百分位范围退化为单值，已扩展为 [%.4f, %.4f] 以避免全黑输出",
            vmin,
            vmax,
        )

    logger.info(
        "全堆栈强度范围: [%.4f, %.4f]（%.2f%%–%.2f%%）",
        vmin,
        vmax,
        low_percentile,
        high_percentile,
    )
    return vmin, vmax
