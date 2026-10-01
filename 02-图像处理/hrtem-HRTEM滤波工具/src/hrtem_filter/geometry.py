"""Image validation, ROI extraction, padding, and exact reverse cropping."""

from __future__ import annotations

import operator
from dataclasses import dataclass

import numpy as np


class ImageValidationError(ValueError):
    """Raised before a malformed image reaches the FFT implementation."""


@dataclass(frozen=True)
class Roi:
    top: int
    left: int
    bottom: int
    right: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.bottom - self.top, self.right - self.left


@dataclass(frozen=True)
class PaddingInfo:
    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    top: int
    bottom: int
    left: int
    right: int

    def crop(self, image: np.ndarray) -> np.ndarray:
        h, w = self.original_shape
        return image[self.top : self.top + h, self.left : self.left + w]


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def clamp_drag_rect(
    shape: tuple[int, int],
    start: tuple[int, int],
    end: tuple[int, int],
) -> tuple[int, int, int, int]:
    """把拖拽两端钳制到图像内，返回 (left, top, right, bottom) 右开区间。

    起点同样必须钳制：matplotlib 坐标区带有边距，按下点可以落在图像之外
    （负坐标或越界），不钳制会生成越界 ROI，直到预览时才以原始错误暴露。
    """
    height, width = shape
    x0 = max(0, min(width - 1, start[0]))
    y0 = max(0, min(height - 1, start[1]))
    x1 = max(0, min(width - 1, end[0]))
    y1 = max(0, min(height - 1, end[1]))
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    return left, top, right + 1, bottom + 1


def validate_image(image: np.ndarray, *, minimum_size: int = 4) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 2:
        raise ImageValidationError(f"仅支持 2D 灰度图像，当前形状为 {array.shape}")
    if min(array.shape) < minimum_size:
        raise ImageValidationError(
            f"图像尺寸至少为 {minimum_size}×{minimum_size}，当前为 {array.shape[1]}×{array.shape[0]}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ImageValidationError(f"图像数据类型必须为数值类型，当前为 {array.dtype}")
    if not np.isfinite(array).all():
        count = int(np.size(array) - np.isfinite(array).sum())
        raise ImageValidationError(f"图像含有 {count} 个 NaN 或 Inf 像素；请先修复原始数据")
    return array.astype(np.float32, copy=False)


def validate_roi(roi: Roi | tuple[int, int, int, int] | None, shape: tuple[int, int]) -> Roi | None:
    if roi is None:
        return None
    if not isinstance(roi, Roi):
        try:
            roi = Roi(*roi)
        except (TypeError, ValueError) as exc:
            raise ImageValidationError("ROI 必须是 (top, left, bottom, right)") from exc
    try:
        roi = Roi(*(operator.index(v) for v in (roi.top, roi.left, roi.bottom, roi.right)))
    except TypeError as exc:
        raise ImageValidationError("ROI 坐标必须为整数") from exc
    h, w = shape
    if not (0 <= roi.top < roi.bottom <= h and 0 <= roi.left < roi.right <= w):
        raise ImageValidationError(f"ROI {roi} 超出图像边界 {w}×{h}")
    if min(roi.shape) < 4:
        raise ImageValidationError("ROI 最小尺寸为 4×4 px")
    return roi


def extract_and_pad(
    image: np.ndarray, roi: Roi | tuple[int, int, int, int] | None = None
) -> tuple[np.ndarray, PaddingInfo, Roi | None]:
    image = validate_image(image)
    checked_roi = validate_roi(roi, image.shape)
    region = image if checked_roi is None else image[checked_roi.top : checked_roi.bottom, checked_roi.left : checked_roi.right]
    h, w = region.shape
    size = _next_power_of_two(max(h, w))
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    padded = np.pad(region, ((top, bottom), (left, right)), mode="reflect")
    return padded.astype(np.float32, copy=False), PaddingInfo((h, w), (size, size), top, bottom, left, right), checked_roi
