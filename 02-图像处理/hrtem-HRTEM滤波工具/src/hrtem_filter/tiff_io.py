"""Safe, explicit TIFF access for grayscale single images and page stacks."""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from .params import OutputEncoding, SaveOptions


class TiffFormatError(ValueError):
    """Raised when TIFF axes or pages cannot be interpreted without guessing."""


class UnsafeOutputPathError(ValueError):
    """Raised when an operation could overwrite its source TIFF."""


@dataclass(frozen=True)
class TiffMetadata:
    axes: str | None
    description: str | None
    resolution: tuple[float, float] | None
    resolution_unit: Any | None


@dataclass(frozen=True)
class TiffStackInfo:
    path: Path
    frame_shape: tuple[int, int]
    frame_count: int
    dtype: np.dtype
    metadata: TiffMetadata

    @property
    def pixel_count(self) -> int:
        return self.frame_count * self.frame_shape[0] * self.frame_shape[1]


def _tag_resolution(page: tifffile.TiffPage, name: str) -> float | None:
    tag = page.tags.get(name)
    if tag is None:
        return None
    value = tag.value
    try:
        numerator, denominator = value
        return float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _reject_non_grayscale_page(page: tifffile.TiffPage, index: int) -> None:
    """Refuse pages that are not single-sample grayscale.

    TIFF photometric values 0 (WhiteIsZero) and 1 (BlackIsZero) are the two
    grayscale interpretations; palette (3), RGB (2), CMYK (5), YCbCr (6) and
    CIELab (8) would be filtered as if their channel or palette indices were
    intensities, which is never scientifically meaningful.  ``samplesperpixel
    > 1`` (e.g. gray + alpha) is rejected for the same reason.
    """
    photometric = getattr(page, "photometric", None)
    if photometric is not None:
        try:
            photometric = int(photometric)
        except (TypeError, ValueError):
            photometric = None
        if photometric not in (0, 1):
            raise TiffFormatError(
                f"第 {index} 页的 photometric={photometric} 不是灰度（仅接受 0/1）；"
                "对调色板索引或彩色分量做滤波没有科学意义。"
                "请先在 Fiji/DM 中导出为灰度 TIFF。"
            )
    samples = getattr(page, "samplesperpixel", 1)
    try:
        samples = int(samples)
    except (TypeError, ValueError):
        samples = 1
    if samples > 1:
        raise TiffFormatError(
            f"第 {index} 页的 samplesperpixel={samples} 大于 1；仅支持单样本灰度页。"
            "请先在 Fiji/DM 中导出为灰度 TIFF。"
        )


def _metadata_from_page(page: tifffile.TiffPage, axes: str | None) -> TiffMetadata:
    x_resolution = _tag_resolution(page, "XResolution")
    y_resolution = _tag_resolution(page, "YResolution")
    resolution = (x_resolution, y_resolution) if x_resolution and y_resolution else None
    resolution_unit = page.tags.get("ResolutionUnit")
    return TiffMetadata(
        axes=axes,
        description=page.description,
        resolution=resolution,
        resolution_unit=resolution_unit.value if resolution_unit else None,
    )


def inspect_tiff(path: str | Path) -> TiffStackInfo:
    """Inspect a grayscale TIFF without decoding its full stack into memory.

    Page stacks are streamed.  A single page with non-YX axes is rejected
    deliberately: treating ``SYX`` or ``CYX`` as a temporal stack would turn a
    colour/sample axis into false measurements.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"找不到 TIFF 文件: {source}")
    with tifffile.TiffFile(source) as handle:
        if not handle.pages:
            raise TiffFormatError("TIFF 不包含任何页面")
        series = handle.series[0] if handle.series else None
        axes = series.axes if series is not None else None
        first = handle.pages[0]
        if first.ndim != 2:
            shape = series.shape if series is not None else first.shape
            raise TiffFormatError(
                f"仅支持灰度 YX 页面堆栈；检测到第一页形状 {first.shape}、series axes={axes!r}、series shape={shape}。"
                "请先在 Fiji/DM 中导出为灰度多页 TIFF，或使用支持该 axes 的后续扩展。"
            )
        shape = tuple(int(v) for v in first.shape)
        dtype = np.dtype(first.dtype)
        _reject_non_grayscale_page(first, 1)
        for index, page in enumerate(handle.pages[1:], start=2):
            if page.ndim != 2 or tuple(page.shape) != shape or np.dtype(page.dtype) != dtype:
                raise TiffFormatError(
                    f"第 {index} 页与第一页的灰度形状/类型不一致；不能安全地作为单一堆栈处理。"
                )
            _reject_non_grayscale_page(page, index)
        return TiffStackInfo(
            path=source,
            frame_shape=shape,
            frame_count=len(handle.pages),
            dtype=dtype,
            metadata=_metadata_from_page(first, axes),
        )


class TiffFrameSource:
    """Frame reader with two access styles.

    ``read_frame`` re-opens the TIFF per call: safe for sporadic GUI access
    and never holds a Windows file lock between calls.  ``session`` keeps one
    handle open for sequential batch reads, avoiding the O(n) page-table
    parse per frame that made large stacks quadratic.
    """

    def __init__(self, info: TiffStackInfo) -> None:
        self.info = info

    def _read_from(self, handle: tifffile.TiffFile, index: int) -> np.ndarray:
        if not 0 <= index < self.info.frame_count:
            raise IndexError(f"帧索引越界: {index}")
        frame = handle.pages[index].asarray()
        if frame.shape != self.info.frame_shape or frame.ndim != 2:
            raise TiffFormatError(f"第 {index + 1} 帧在读取时改变了形状，已中止处理")
        return frame

    def read_frame(self, index: int) -> np.ndarray:
        with tifffile.TiffFile(self.info.path) as handle:
            return self._read_from(handle, index)

    @contextmanager
    def session(self) -> Iterator[_FrameSession]:
        """Open the file once for a sequential read burst (single-threaded)."""
        with tifffile.TiffFile(self.info.path) as handle:
            yield _FrameSession(self, handle)


class _FrameSession:
    """Sequential reader bound to one open TIFF handle."""

    def __init__(self, source: TiffFrameSource, handle: tifffile.TiffFile) -> None:
        self._source = source
        self._handle = handle

    def read_frame(self, index: int) -> np.ndarray:
        return self._source._read_from(self._handle, index)


def ensure_distinct_paths(input_path: str | Path, output_path: str | Path) -> tuple[Path, Path]:
    source = Path(input_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    if os.path.normcase(str(source)) == os.path.normcase(str(target)):
        raise UnsafeOutputPathError("输出文件不能与输入文件相同；请保留原始 TIFF")
    return source, target


def encode_frame(
    image: np.ndarray,
    *,
    source_dtype: np.dtype,
    options: SaveOptions,
    display_range: tuple[float, float] | None,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if options.encoding == OutputEncoding.FLOAT32:
        return image
    if options.encoding == OutputEncoding.UINT8_DISPLAY:
        if display_range is None:
            raise ValueError("8 位显示导出需要全局显示范围")
        low, high = display_range
        return (np.clip((image - low) / (high - low), 0, 1) * 255).round().astype(np.uint8)
    if options.encoding == OutputEncoding.SOURCE_DTYPE_CLIP:
        if not np.issubdtype(source_dtype, np.integer):
            return image
        limits = np.iinfo(source_dtype)
        return np.clip(np.rint(image), limits.min, limits.max).astype(source_dtype)
    raise ValueError(f"不支持的输出编码: {options.encoding}")


class AtomicTiffWriter:
    """Write a TIFF beside its destination and publish it only after validation."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        frame_shape: tuple[int, int],
        expected_frames: int,
        metadata: TiffMetadata,
        provenance_hint: str,
        output_dtype: np.dtype,
    ) -> None:
        self.output_path = Path(output_path).expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_shape = frame_shape
        self.expected_frames = expected_frames
        self.metadata = metadata
        self.provenance_hint = provenance_hint
        self.output_dtype = np.dtype(output_dtype)
        suffix = ".partial.tif"
        descriptor, name = tempfile.mkstemp(prefix=f".{self.output_path.stem}.", suffix=suffix, dir=self.output_path.parent)
        os.close(descriptor)
        self.temporary_path = Path(name)
        predicted_bytes = expected_frames * frame_shape[0] * frame_shape[1] * self.output_dtype.itemsize
        self._writer = tifffile.TiffWriter(self.temporary_path, bigtiff=predicted_bytes >= 3_500_000_000)
        self._written = 0
        self._closed = False

    def write(self, image: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("TIFF writer 已关闭")
        if image.shape != self.frame_shape:
            raise TiffFormatError(f"输出帧形状 {image.shape} 与预期 {self.frame_shape} 不一致")
        if image.dtype != self.output_dtype:
            raise TiffFormatError(f"输出帧类型 {image.dtype} 与预期 {self.output_dtype} 不一致")
        kwargs: dict[str, Any] = {"photometric": "minisblack", "contiguous": True, "metadata": None}
        # 分辨率是逐页标签：只写第一页会让逐页读取标定的工具（如部分
        # ImageJ 插件）在后续页取不到像素标定。
        if self.metadata.resolution is not None:
            kwargs["resolution"] = self.metadata.resolution
        if self.metadata.resolution_unit is not None:
            kwargs["resolutionunit"] = self.metadata.resolution_unit
        if self._written == 0:
            kwargs["description"] = self.provenance_hint
        self._writer.write(image, **kwargs)
        self._written += 1

    def commit(self) -> Path:
        if self._closed:
            raise RuntimeError("TIFF writer 已关闭")
        try:
            self._writer.close()
            self._closed = True
            with tifffile.TiffFile(self.temporary_path) as check:
                if len(check.pages) != self.expected_frames:
                    raise TiffFormatError(
                        f"临时输出只有 {len(check.pages)} 帧，预期 {self.expected_frames} 帧；不会发布不完整文件"
                    )
                if check.pages[0].shape != self.frame_shape:
                    raise TiffFormatError("临时输出尺寸校验失败")
            # 原子替换前把临时 TIFF 数据真正落到磁盘，避免崩溃后目标文件为空或半页。
            # 注意用 "rb+"：Windows 上 fsync 要求句柄可写，只读句柄会 EBADF。
            with open(self.temporary_path, "rb+") as handle:
                os.fsync(handle.fileno())
            os.replace(self.temporary_path, self.output_path)
            return self.output_path
        except Exception:
            # 任何失败都清理临时文件，并保留原始异常信息供上层展示。
            if not self._closed:
                with contextlib.suppress(Exception):
                    self._writer.close()
                self._closed = True
            with contextlib.suppress(FileNotFoundError):
                self.temporary_path.unlink()
            raise

    def abort(self) -> None:
        if not self._closed:
            self._writer.close()
            self._closed = True
        with contextlib.suppress(FileNotFoundError):
            self.temporary_path.unlink()
