"""Validated TIFF stack input and metadata-consistent streaming output."""

from __future__ import annotations

import contextlib
import ctypes
import logging
import math
import os
import threading
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import tifffile
from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException

from errors import InputValidationError, OperationCancelled, OutputValidationError
from version import APP_SHORT_NAME, APP_VERSION

logger = logging.getLogger(__name__)

CLASSIC_TIFF_SAFE_LIMIT = int(3.7 * 1024**3)
MAX_FRAME_PIXELS = 268_435_456


def paths_refer_to_same_file(first, second) -> bool:
    """Compare existing or prospective paths safely on Windows and POSIX."""
    first_path = os.path.normcase(os.path.realpath(os.fspath(first)))
    second_path = os.path.normcase(os.path.realpath(os.fspath(second)))
    if first_path == second_path:
        return True
    try:
        return os.path.samefile(first_path, second_path)
    except (FileNotFoundError, OSError):
        return False


def file_signature(path) -> tuple[int, int, int, int]:
    """Return identity and change-sensitive stat fields for one file.

    The four-tuple (st_dev, st_ino, st_size, st_mtime_ns) detects both
    in-place modification and same-name replacement; Python fills st_ino
    with the NTFS file ID on Windows.
    """
    stat = os.stat(path)
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def estimate_peak_memory_bytes(frame_shape: tuple[int, int]) -> int:
    """Conservative peak-memory estimate for the current spectral pipeline."""
    pixels = int(frame_shape[0]) * int(frame_shape[1])
    return pixels * 104 + 192 * 1024**2


def available_memory_bytes() -> int | None:
    """Return currently available physical memory when the OS exposes it."""
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
        return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page_size * available_pages)
    except (AttributeError, OSError, ValueError):
        return None


def _rational_to_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2:
        numerator, denominator = value
        if denominator:
            return float(numerator) / float(denominator)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _tag_value(page, name, default=None):
    tag = page.tags.get(name)
    return default if tag is None else tag.value


def _safe_imagej_metadata(metadata) -> dict:
    if not isinstance(metadata, dict):
        return {}
    allowed = {
        "unit",
        "spacing",
        "fps",
        "finterval",
        "loop",
        "min",
        "max",
        "ranges",
        "labels",
        "info",
    }
    scalar_numbers = {"spacing", "fps", "finterval", "min", "max"}
    result = {}
    for key, value in metadata.items():
        if key not in allowed:
            continue
        if key in scalar_numbers:
            number = _rational_to_float(value)
            if number is None or not math.isfinite(number):
                logger.warning("忽略无效 ImageJ 数值字段 %s=%r", key, value)
                continue
            result[key] = number
        elif key == "ranges":
            if not isinstance(value, (list, tuple)):
                logger.warning("忽略无效 ImageJ ranges 字段: %r", value)
                continue
            converted = []
            valid = True
            for item in value:
                number = _rational_to_float(item)
                if number is None or not math.isfinite(number):
                    valid = False
                    break
                converted.append(number)
            if valid:
                result[key] = converted
            else:
                logger.warning("忽略无效 ImageJ ranges 字段: %r", value)
        elif key == "loop":
            try:
                result[key] = int(value)
            except (TypeError, ValueError):
                logger.warning("忽略无效 ImageJ loop 字段: %r", value)
        else:
            result[key] = value
    return result


def _ome_physical_metadata(xml: str | None) -> dict:
    if not xml:
        return {}
    try:
        root = ElementTree.fromstring(xml)
    except (ElementTree.ParseError, DefusedXmlException):
        logger.warning("输入 OME-XML 无法解析，将不复制物理标定字段")
        return {}
    pixels = root.find(".//{*}Pixels")
    if pixels is None:
        return {}
    keys = (
        "PhysicalSizeX",
        "PhysicalSizeXUnit",
        "PhysicalSizeY",
        "PhysicalSizeYUnit",
        "PhysicalSizeZ",
        "PhysicalSizeZUnit",
        "TimeIncrement",
        "TimeIncrementUnit",
    )
    result = {}
    for key in keys:
        if key in pixels.attrib:
            value = pixels.attrib[key]
            if key.endswith("Unit"):
                result[key] = value
            else:
                try:
                    result[key] = float(value)
                except ValueError:
                    logger.warning("忽略无效 OME 标定值 %s=%r", key, value)
    return result


class TiffStackReader:
    """Thread-safe reader for one homogeneous 2D grayscale TIFF series."""

    def __init__(self, file_path, cancel_event=None):
        self.file_path = os.path.abspath(os.fspath(file_path))
        self._tf: tifffile.TiffFile | None = None
        self._pages = ()
        self._lock = threading.RLock()
        self._open(cancel_event)

    def _open(self, cancel_event=None):
        if not os.path.isfile(self.file_path):
            raise InputValidationError(f"文件不存在: {self.file_path}")
        try:
            tif = tifffile.TiffFile(self.file_path)
        except Exception as exc:
            raise InputValidationError(f"不是可读取的 TIFF 文件: {exc}") from exc

        try:
            if len(tif.pages) == 0:
                raise InputValidationError("TIFF 文件不包含任何页面")

            if len(tif.series) == 1:
                series = tif.series[0]
                pages = tuple(series.pages)
                axes = str(series.axes)
                stack_shape = tuple(int(value) for value in series.shape)
            elif tif.is_ome or tif.is_imagej:
                raise InputValidationError(
                    f"当前版本仅支持单系列 TIFF，检测到 {len(tif.series)} 个系列"
                )
            else:
                series = None
                pages = tuple(tif.pages)
                first_shape = tuple(int(value) for value in pages[0].shape)
                axes = "YX" if len(pages) == 1 else "QYX"
                stack_shape = (
                    first_shape if len(pages) == 1 else (len(pages), *first_shape)
                )

            if not pages:
                raise InputValidationError("TIFF 系列不包含可读取页面")
            first_page = pages[0]
            frame_shape = tuple(int(value) for value in first_page.shape)
            dtype = np.dtype(first_page.dtype)
            if len(frame_shape) != 2:
                raise InputValidationError(
                    f"仅支持每页二维灰度图像，首帧 shape={frame_shape}"
                )
            if frame_shape[0] <= 0 or frame_shape[1] <= 0:
                raise InputValidationError(f"无效图像尺寸: {frame_shape}")
            if math.prod(frame_shape) > MAX_FRAME_PIXELS:
                raise InputValidationError(
                    f"单帧像素数过大: {math.prod(frame_shape):,}"
                )
            if dtype.kind not in "uif":
                raise InputValidationError(f"不支持的数据类型: {dtype}")
            if "S" in axes or "C" in axes:
                raise InputValidationError(f"仅支持单通道灰度 TIFF，检测到轴: {axes}")

            for index, page in enumerate(pages):
                # 大堆栈逐页结构校验可能耗时数秒；支持在此阶段响应取消，
                # 让 GUI 的"取消打开"即时生效而不是等校验自然结束。
                if cancel_event is not None and cancel_event.is_set():
                    raise OperationCancelled("已取消打开文件")
                page_shape = tuple(int(value) for value in page.shape)
                page_dtype = np.dtype(page.dtype)
                samples = int(getattr(page, "samplesperpixel", 1) or 1)
                if samples != 1 or len(page_shape) != 2:
                    raise InputValidationError(
                        f"帧 {index} 不是二维单通道灰度图像: "
                        f"shape={page_shape}, samples={samples}"
                    )
                if page_shape != frame_shape or page_dtype != dtype:
                    raise InputValidationError(
                        f"帧 {index} 与首帧不一致: "
                        f"shape={page_shape}, dtype={page_dtype}"
                    )

            leading_frames = 1 if len(stack_shape) == 2 else math.prod(stack_shape[:-2])
            if stack_shape[-2:] != frame_shape or leading_frames != len(pages):
                raise InputValidationError(
                    "TIFF 系列轴/页面结构不受支持: "
                    f"shape={stack_shape}, axes={axes}, pages={len(pages)}"
                )

            self._tf = tif
            self._pages = pages
            self.num_frames = len(pages)
            self.shape = frame_shape
            self.stack_shape = stack_shape
            self.axes = axes
            self.dtype = dtype
            self.is_bigtiff = bool(tif.is_bigtiff)
            self.is_ome = bool(tif.is_ome)
            self.is_imagej = bool(tif.is_imagej)
            self.metadata = self._collect_metadata(first_page)
        except Exception:
            tif.close()
            raise

        logger.info(
            "已验证 TIFF: frames=%s, shape=%s, dtype=%s, axes=%s, ome=%s",
            self.num_frames,
            self.shape,
            self.dtype,
            self.axes,
            self.is_ome,
        )

    def _collect_metadata(self, first_page) -> dict[str, Any]:
        x_resolution = _rational_to_float(_tag_value(first_page, "XResolution"))
        y_resolution = _rational_to_float(_tag_value(first_page, "YResolution"))
        resolution_unit = _tag_value(first_page, "ResolutionUnit")
        if hasattr(resolution_unit, "name"):
            resolution_unit = resolution_unit.name
        elif resolution_unit is not None:
            resolution_unit = str(resolution_unit)

        ome_xml = self._tf.ome_metadata if self.is_ome else None
        return {
            "axes": self.axes,
            "stack_shape": self.stack_shape,
            "frame_shape": self.shape,
            "is_ome": self.is_ome,
            "is_imagej": self.is_imagej,
            "is_bigtiff": self.is_bigtiff,
            "x_resolution": x_resolution,
            "y_resolution": y_resolution,
            "resolution_unit": resolution_unit,
            "input_description": _tag_value(first_page, "ImageDescription"),
            "input_software": _tag_value(first_page, "Software"),
            "ome_physical": _ome_physical_metadata(ome_xml),
            "imagej": _safe_imagej_metadata(getattr(self._tf, "imagej_metadata", None)),
        }

    def read_frame(self, index: int) -> np.ndarray:
        if index < 0 or index >= self.num_frames:
            raise IndexError(f"帧索引超出范围: {index}（共 {self.num_frames} 帧）")
        with self._lock:
            if self._tf is None:
                raise RuntimeError("TIFF 文件已关闭")
            try:
                frame = self._pages[index].asarray()
            except Exception as exc:
                raise InputValidationError(
                    f"读取帧 {index} 失败；可能缺少 imagecodecs 或文件已损坏: {exc}"
                ) from exc
            if frame.shape != self.shape or np.dtype(frame.dtype) != self.dtype:
                raise InputValidationError(
                    f"帧 {index} 解码结果不一致: "
                    f"shape={frame.shape}, dtype={frame.dtype}"
                )
            if frame.dtype.kind == "f" and not np.isfinite(frame).all():
                raise InputValidationError(f"帧 {index} 包含 NaN 或 Inf")
            return frame

    def iter_frames(self) -> Iterator[np.ndarray]:
        for index in range(self.num_frames):
            yield self.read_frame(index)

    def close(self):
        with self._lock:
            if self._tf is not None:
                self._tf.close()
                self._tf = None
                self._pages = ()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()


def choose_bigtiff(stack_shape: tuple[int, ...], dtype: np.dtype) -> bool:
    return estimate_output_bytes(stack_shape, dtype) >= CLASSIC_TIFF_SAFE_LIMIT


def estimate_output_bytes(stack_shape: tuple[int, ...], dtype: np.dtype) -> int:
    """Conservatively estimate an uncompressed output TIFF size."""
    raw_bytes = math.prod(stack_shape) * np.dtype(dtype).itemsize
    return int(raw_bytes * 1.08 + 16 * 1024**2)


class TiffStackWriter:
    """Write one validated stack in a single metadata-consistent series."""

    def __init__(
        self,
        file_path,
        *,
        stack_shape: tuple[int, ...],
        dtype,
        metadata: dict | None = None,
        bigtiff: bool | None = None,
        compression: str | None = None,
    ):
        self.file_path = os.path.abspath(os.fspath(file_path))
        self.stack_shape = tuple(int(value) for value in stack_shape)
        self.dtype = np.dtype(dtype)
        self.metadata = dict(metadata or {})
        self.bigtiff = (
            choose_bigtiff(self.stack_shape, self.dtype)
            if bigtiff is None
            else bool(bigtiff)
        )
        self.compression = compression
        self._writer: tifffile.TiffWriter | None = None
        self._frame_count = 0

    @property
    def expected_frames(self) -> int:
        return 1 if len(self.stack_shape) == 2 else math.prod(self.stack_shape[:-2])

    def open(self):
        # 支持同一 writer 实例复用：重新打开时必须重置帧计数，
        # 否则 write_stack 会把上一轮的计数带入本轮校验。
        self._frame_count = 0
        is_ome = bool(self.metadata.get("is_ome"))
        is_imagej = bool(self.metadata.get("is_imagej")) and not is_ome
        self._writer = tifffile.TiffWriter(
            self.file_path,
            mode="w",
            bigtiff=self.bigtiff,
            ome=bool(is_ome),
            imagej=is_imagej,
            shaped=False if not (is_ome or is_imagej) else None,
        )

    def _write_metadata(self) -> dict | None:
        axes = str(self.metadata.get("axes", ""))
        if bool(self.metadata.get("is_ome")):
            values = {"axes": axes}
            values.update(self.metadata.get("ome_physical", {}))
            return values
        if bool(self.metadata.get("is_imagej")):
            values = {"axes": axes}
            values.update(self.metadata.get("imagej", {}))
            return values
        return None

    def write_stack(self, frames: Iterable[np.ndarray]):
        if self._writer is None:
            raise RuntimeError("写入器未打开")

        is_ome = bool(self.metadata.get("is_ome"))
        is_imagej = bool(self.metadata.get("is_imagej")) and not is_ome

        def checked_frames():
            for index, frame in enumerate(frames):
                array = np.asarray(frame)
                if array.shape != self.stack_shape[-2:]:
                    raise ValueError(f"输出帧 {index} 尺寸错误: {array.shape}")
                if np.dtype(array.dtype) != self.dtype:
                    raise ValueError(f"输出帧 {index} 类型错误: {array.dtype}")
                self._frame_count += 1
                yield array
            if self._frame_count != self.expected_frames:
                raise ValueError(
                    f"输出帧数错误: {self._frame_count}/{self.expected_frames}"
                )

        resolution = None
        x_res = self.metadata.get("x_resolution")
        y_res = self.metadata.get("y_resolution")
        if x_res and y_res:
            resolution = (float(x_res), float(y_res))
        resolution_unit = self.metadata.get("resolution_unit")
        if resolution_unit in (None, "NONE", "RESUNIT.NONE"):
            resolution_unit = None

        self._writer.write(
            data=checked_frames(),
            shape=self.stack_shape,
            dtype=self.dtype,
            photometric="minisblack",
            compression=self.compression,
            resolution=resolution,
            resolutionunit=resolution_unit,
            software=f"{APP_SHORT_NAME} {APP_VERSION}",
            description=(
                self.metadata.get("input_description")
                if not (is_ome or is_imagej)
                else None
            ),
            metadata=self._write_metadata(),
            contiguous=self.compression is None,
        )

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def frame_count(self) -> int:
        return self._frame_count


def validate_output_file(
    file_path,
    *,
    expected_frames: int,
    expected_frame_shape: tuple[int, int],
    expected_stack_shape: tuple[int, ...],
    expected_dtype,
    expected_axes: str | None = None,
    expect_ome: bool = False,
) -> dict:
    """Reopen and verify a completed temporary output before publication."""
    try:
        with tifffile.TiffFile(file_path) as tif:
            if len(tif.pages) != expected_frames:
                raise OutputValidationError(
                    f"输出页数错误: {len(tif.pages)}/{expected_frames}"
                )
            for index, page in enumerate(tif.pages):
                if tuple(page.shape) != tuple(expected_frame_shape):
                    raise OutputValidationError(
                        f"输出帧 {index} 尺寸错误: {page.shape}"
                    )
                if np.dtype(page.dtype) != np.dtype(expected_dtype):
                    raise OutputValidationError(
                        f"输出帧 {index} 类型错误: {page.dtype}"
                    )
                try:
                    decoded = page.asarray()
                except Exception as exc:
                    raise OutputValidationError(
                        f"输出帧 {index} 无法完整解码: {exc}"
                    ) from exc
                if tuple(decoded.shape) != tuple(expected_frame_shape) or np.dtype(
                    decoded.dtype
                ) != np.dtype(expected_dtype):
                    raise OutputValidationError(
                        f"输出帧 {index} 解码结果不一致: "
                        f"shape={decoded.shape}, dtype={decoded.dtype}"
                    )
            if not tif.series:
                raise OutputValidationError("输出不包含 TIFF 系列")
            series = tif.series[0]
            if tuple(series.shape) != tuple(expected_stack_shape):
                raise OutputValidationError(f"输出系列尺寸错误: {series.shape}")
            if np.dtype(series.dtype) != np.dtype(expected_dtype):
                raise OutputValidationError(f"输出系列类型错误: {series.dtype}")
            if expect_ome and not tif.is_ome:
                raise OutputValidationError("输出丢失 OME 元数据")
            if expected_axes and expect_ome and series.axes != expected_axes:
                raise OutputValidationError(
                    f"输出 OME 轴错误: {series.axes}/{expected_axes}"
                )
            return {
                "num_frames": len(tif.pages),
                "shape": tuple(series.shape),
                "frame_shape": tuple(tif.pages[0].shape),
                "dtype": str(series.dtype),
                "axes": series.axes,
                "is_bigtiff": bool(tif.is_bigtiff),
                "is_ome": bool(tif.is_ome),
                "file_size": os.path.getsize(file_path),
            }
    except OutputValidationError:
        raise
    except Exception as exc:
        raise OutputValidationError(f"无法验证输出 TIFF: {exc}") from exc


def get_tiff_info(file_path) -> dict:
    """Return validated structural information for one supported TIFF."""
    with TiffStackReader(file_path) as reader:
        return {
            "num_frames": reader.num_frames,
            "shape": reader.shape,
            "stack_shape": reader.stack_shape,
            "axes": reader.axes,
            "dtype": str(reader.dtype),
            "file_size": os.path.getsize(reader.file_path),
            "is_bigtiff": reader.is_bigtiff,
            "is_ome": reader.is_ome,
            "file_path": reader.file_path,
        }
