# -*- coding: utf-8 -*-
"""可靠的 TIFF 漂移矫正核心（v7 合并版）。

v7 由 v5.2（文件安全、TIFF 结构校验、质量门控、审计报告）与 v6.1
（纯平移估计、可配置边界模式、代码清理）合并而来。
本模块不依赖 Tk。失败或取消永远不会删除既有目标文件。
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import cv2
import numpy as np
import tifffile

logger = logging.getLogger(__name__)

# 版本单一来源：GUI 标题、pyproject、打包资源与审计报告均由此派生，
# 避免 7.1→7.2 这类默认参数变更后无法追溯是哪套参数跑出的结果。
TOOL_NAME = "tiff-drift-correction"
__version__ = "7.3.0"

# 默认不限制堆栈解码后的总大小；大型 TIFF 是否能够加载取决于可用内存。
# ``max_bytes`` 参数仍可供调用方显式设置保护上限。
MAX_STACK_BYTES: Optional[int] = None
MAX_FRAME_PIXELS = 100_000_000
# 特征提取与匹配都会调用 OpenCV；池内并发期间用 _opencv_single_thread() 关闭
# OpenCV 内部多线程，两级并行叠加会过度订阅 CPU 反而变慢。
MAX_WORKERS = max(1, min(8, os.cpu_count() or 4))
BIGTIFF_THRESHOLD = 3_500_000_000
# 矫正生效校验的最低位移（px）：低于该量级的平移与插值噪声同级，
# 不强制要求输出像素必须改变。
VERIFY_MIN_SHIFT_MAGNITUDE = 0.25
# 写出前为目标目录保留的磁盘余量。
DISK_SPACE_MARGIN_BYTES = 64 * 1024 ** 2
# 批处理无人值守：解码体积超过物理内存该比例时按文件级失败处理；
# 界面交互加载另行采用更低的确认阈值（见 drift_correction）。
MEMORY_CRITICAL_RATIO = 0.9
# 描述子达到该规模时改用 FLANN 近似最近邻匹配，小规模保持暴力匹配。
FLANN_MIN_DESCRIPTORS = 512
# 检测与匹配核心默认值与 Fiji "Linear Stack Alignment with SIFT" 对齐：
# initial sigma 1.6、steps/octave 3、ratio 0.92、maxEpsilon 25 px、inlier ratio 0.05。
# 无 ImageJ 对应的参数分两类：
#   - nfeatures/contrast_threshold/edge_threshold 是 OpenCV 特有。mpicbg SIFT
#     无特征数上限、无边缘剔除；nfeatures=0 可取消上限（与 ImageJ 一致），
#     但大堆栈会显著增加内存与匹配耗时，故默认保留 5000 上限。
#   - min_matches/min_inliers/max_pair_residual/min_valid_pair_ratio/
#     max_interpolation_gap 是本工具的质量门控，默认已放宽到与 ImageJ 相当的
#     非约束水平（ImageJ 对匹配失败静默沿用上一帧模型，本工具改为按相邻
#     可靠速度插值并在审计报告中警告）。收紧阈值可恢复严格门控，例如
#     RANSAC 3 px 可重新启用对称双峰冲突拒绝。
DEFAULT_SIFT_PARAMS = {
    "nfeatures": 5000,
    "noctave_layers": 3,
    "contrast_threshold": 0.04,
    "edge_threshold": 10.0,
    "sigma": 1.6,
    "ratio_threshold": 0.92,
    "ransac_threshold": 25.0,
    "min_matches": 3,
    "min_inliers": 3,
    "min_inlier_ratio": 0.05,
    "max_pair_residual": 25.0,
    "min_valid_pair_ratio": 0.05,
    "max_interpolation_gap": 10000,
    "skip_interval": 1,
}


class OperationCancelled(RuntimeError):
    """内部取消信号。调用方应将其显示为取消而不是失败。"""


class DetectionQualityError(ValueError):
    """匹配质量不足，不能安全生成漂移曲线。"""


@dataclass
class PairQuality:
    source_index: int
    target_index: int
    keypoints_source: int
    keypoints_target: int
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    median_residual: float = math.inf
    dx: Optional[float] = None
    dy: Optional[float] = None
    status: str = "not_matched"
    message: str = ""


@dataclass
class DetectionResult:
    shifts_x: np.ndarray
    shifts_y: np.ndarray
    pair_quality: list[PairQuality]
    key_indices: list[int]
    normalization: tuple[float, float]
    warnings: list[str] = field(default_factory=list)
    interpolated_pairs: list[int] = field(default_factory=list)

    @property
    def matched_pairs(self) -> int:
        return sum(item.status == "reliable" for item in self.pair_quality)

    @property
    def total_pairs(self) -> int:
        return len(self.pair_quality)

    @property
    def valid_pair_ratio(self) -> float:
        return self.matched_pairs / self.total_pairs if self.total_pairs else 1.0


# v6 API 兼容别名。v7 统一使用 DetectionResult 的属性访问契约
# （v5.2 已移除旧的元组解包行为）。
DriftResult = DetectionResult


def _cancelled(cancel_event: Optional[threading.Event]) -> bool:
    return bool(cancel_event and cancel_event.is_set())


@contextmanager
def _opencv_single_thread():
    """线程池并发期间关闭 OpenCV 内部多线程，避免两级并行过度订阅 CPU。"""
    previous = cv2.getNumThreads()
    try:
        cv2.setNumThreads(1)
        yield
    finally:
        cv2.setNumThreads(previous)


def _finite_number(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数字") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} 必须是有限数")
    return number


def total_physical_memory() -> Optional[int]:
    """返回本机物理内存总量（字节），无法探测时为 None。

    Windows 走 GlobalMemoryStatusEx；其他平台退回 sysconf。
    """
    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    except Exception:
        pass
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return None


class TiffIO:
    """经校验的二维灰度 TIFF 堆栈读写。"""

    SUPPORTED_DTYPES = {
        np.dtype("uint8"), np.dtype("uint16"), np.dtype("int16"),
        np.dtype("uint32"), np.dtype("int32"), np.dtype("float32"),
        np.dtype("float64"), np.dtype("bool"),
    }

    @staticmethod
    def _contiguous_stack_info(tif: "tifffile.TiffFile") -> Optional[tuple[int, tuple, np.dtype]]:
        """单页连续三维序列（整栈写在一个 IFD）返回 (帧数, 帧形状, dtype)。

        inspect_stack 与 read_stack 必须用同一判定；提取为单一实现，
        避免两处条件演化后静默分叉。
        """
        series = tif.series[0]
        if (len(tif.pages) == 1 and len(series.shape) == 3
                and series.axes.endswith("YX") and not series.axes.endswith("SYX")):
            return int(series.shape[0]), tuple(series.shape[-2:]), np.dtype(series.dtype)
        return None

    @staticmethod
    def inspect_stack(filepath: str, max_bytes: Optional[int] = MAX_STACK_BYTES) -> dict:
        """只读取目录信息并逐页校验，不解码整个堆栈。"""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"文件不存在: {filepath}")
        if path.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError(f"不支持的文件格式: {path.suffix}")
        if max_bytes is not None and max_bytes <= 0:
            raise ValueError("内存上限必须大于 0")

        try:
            with tifffile.TiffFile(path) as tif:
                if not tif.pages:
                    raise ValueError("TIFF 文件不包含任何页")
                if len(tif.series) != 1:
                    raise ValueError(f"TIFF 包含 {len(tif.series)} 个 series；请先选择或拆分为单一序列")
                first = tif.pages[0]
                first_shape = tuple(first.shape)
                first_dtype = np.dtype(first.dtype)
                contiguous = TiffIO._contiguous_stack_info(tif)
                if len(first_shape) != 2:
                    if contiguous is None:
                        raise ValueError(f"仅支持单通道二维帧，第一页形状为 {first_shape}")
                    page_count, frame_shape, dtype = contiguous
                    estimated_bytes = int(np.prod(frame_shape)) * page_count * dtype.itemsize
                    page_shapes_checked = 1
                else:
                    page_count = len(tif.pages)
                    frame_shape = first_shape
                    dtype = first_dtype
                    estimated_bytes = 0
                    page_shapes_checked = 0
                    for index, page in enumerate(tif.pages):
                        shape = tuple(page.shape)
                        page_dtype = np.dtype(page.dtype)
                        if len(shape) != 2:
                            raise ValueError(f"第 {index} 页不是二维灰度图，形状为 {shape}")
                        if shape != frame_shape or page_dtype != dtype:
                            raise ValueError(
                                f"第 {index} 页形状或类型不一致：{shape}/{page_dtype}，"
                                f"期望 {frame_shape}/{dtype}")
                        pixels = int(np.prod(shape))
                        if pixels > MAX_FRAME_PIXELS:
                            raise ValueError(f"第 {index} 页尺寸过大：{shape[1]}x{shape[0]}，超过 100MP 限制")
                        estimated_bytes += pixels * dtype.itemsize
                        page_shapes_checked += 1

                if dtype not in TiffIO.SUPPORTED_DTYPES:
                    raise ValueError(f"不支持的像素类型：{dtype}")
                if int(np.prod(frame_shape)) > MAX_FRAME_PIXELS:
                    raise ValueError(f"图像尺寸过大：{frame_shape[1]}x{frame_shape[0]}，超过 100MP 限制")
                if max_bytes is not None and estimated_bytes > max_bytes:
                    raise MemoryError(
                        f"堆栈解码后约需 {estimated_bytes / 1024 ** 3:.2f} GiB，"
                        f"超过调用方设置的 {max_bytes / 1024 ** 3:.2f} GiB 上限。")

                series = tif.series[0]
                axes = str(series.axes or "")
                # C/Z 一旦出现就要求人工确认：无论是否同时存在 T 轴，按页处理
                # 都可能把通道或 z 层混入时间序列（实测 ImageJ TCYX/TZYX 会被
                # 当成数倍数量的"时间帧"），不能只在缺少 T 轴时提醒。
                confirm_required: list[str] = []
                if "C" in axes or "Z" in axes:
                    detail = ""
                    if tif.is_imagej:
                        imagej = tif.imagej_metadata or {}
                        channels = int(imagej.get("channels", 1) or 1)
                        slices = int(imagej.get("slices", 1) or 1)
                        frames = int(imagej.get("frames", 1) or 1)
                        detail = (f"ImageJ 元数据显示 channels={channels}, slices={slices}, "
                                  f"frames={frames}，共 {page_count} 页；")
                    confirm_required.append(
                        f"源 TIFF 序列轴为 {axes}，包含通道（C）或层（Z）维度；{detail}"
                        "按页处理会把通道/z 层混入时间序列，请先确认这些页确实应按页视为时间帧")
                warnings: list[str] = []
                if tif.is_ome:
                    warnings.append("OME-TIFF 将按单一二维 page 序列处理；复杂 C/Z/T 数据应先拆分")

                xres, yres = first.tags.get("XResolution"), first.tags.get("YResolution")
                resolution = None
                if xres and yres:
                    resolution = (
                        float(xres.value[0] / xres.value[1]),
                        float(yres.value[0] / yres.value[1]),
                    )
                unit_tag = first.tags.get("ResolutionUnit")
                return {
                    "source_path": str(path.resolve()),
                    "size": (frame_shape[1], frame_shape[0]),
                    "shape": frame_shape,
                    "n_frames": page_count,
                    "dtype": np.dtype("uint8") if dtype == np.dtype("bool") else dtype,
                    "source_dtype": dtype,
                    "converted_from_1bit": dtype == np.dtype("bool"),
                    "estimated_bytes": estimated_bytes,
                    "axes": axes,
                    "imagej": bool(tif.is_imagej),
                    "ome": bool(tif.is_ome),
                    "source_description": first.description or "",
                    "resolution": resolution,
                    "resolutionunit": unit_tag.value if unit_tag else None,
                    "warnings": warnings,
                    "confirm_required": confirm_required,
                    "page_shapes_checked": page_shapes_checked,
                }
        except (FileNotFoundError, ValueError, MemoryError):
            raise
        except Exception as exc:
            raise ValueError(f"无法读取 TIFF 元数据：{exc}") from exc

    @staticmethod
    def read_stack(filepath: str, dtype=None, max_bytes: Optional[int] = MAX_STACK_BYTES,
                   progress_callback: Optional[Callable[[float], None]] = None,
                   cancel_event: Optional[threading.Event] = None,
                   meta: Optional[dict] = None):
        """读取经 ``inspect_stack`` 验证的单一二维灰度序列。

        ``meta`` 传入 ``inspect_stack`` 的结果可跳过重复的目录扫描（界面
        加载与批处理都已先检视过文件）；传入的元数据仍会被逐页校验，
        文件在两次调用之间被替换时会显式失败而不是读出错位数据。
        """
        if meta is None:
            meta = TiffIO.inspect_stack(filepath, max_bytes=max_bytes)
        path = Path(filepath)
        output_dtype = np.dtype(dtype) if dtype is not None else np.dtype(meta["dtype"])
        if output_dtype not in TiffIO.SUPPORTED_DTYPES - {np.dtype("bool")}:
            raise ValueError(f"不支持的工作像素类型：{output_dtype}")
        frames: list[np.ndarray] = []
        try:
            with tifffile.TiffFile(path) as tif:
                contiguous = TiffIO._contiguous_stack_info(tif)
                if contiguous is not None:
                    stack = tif.series[0].asarray()
                    iterable = (stack[index] for index in range(stack.shape[0]))
                else:
                    iterable = (page.asarray() for page in tif.pages)
                total = int(meta["n_frames"])
                for index, frame in enumerate(iterable):
                    if _cancelled(cancel_event):
                        raise OperationCancelled("加载已取消")
                    arr = np.asarray(frame)
                    if arr.dtype == np.dtype("bool"):
                        arr = arr.astype(np.uint8) * 255
                    arr = np.ascontiguousarray(arr.astype(output_dtype, copy=False))
                    if arr.shape != tuple(meta["shape"]):
                        raise ValueError(f"第 {index} 页读取后形状异常：{arr.shape}")
                    frames.append(arr)
                    if progress_callback:
                        progress_callback((index + 1) / total)
            if len(frames) != int(meta["n_frames"]):
                raise ValueError(
                    f"实际页数 {len(frames)} 与元数据记录的 {meta['n_frames']} 不一致；"
                    "文件可能在读取途中被替换")
        except OperationCancelled:
            raise
        except Exception as exc:
            raise ValueError(f"读取 TIFF 像素数据失败：{exc}") from exc
        logger.info("已加载 %s 页 TIFF：%s，%s", len(frames), meta["shape"], output_dtype)
        return frames, meta

    @staticmethod
    def _same_file(source: Optional[Path], target: Path) -> bool:
        if source is None:
            return False
        try:
            if source.exists() and target.exists():
                return os.path.samefile(source, target)
        except OSError:
            pass
        return os.path.normcase(str(source.resolve())) == os.path.normcase(str(target.resolve()))

    @staticmethod
    def _imagej_description(meta: dict, count: int) -> Optional[str]:
        if not meta.get("imagej"):
            return None
        return (
            "ImageJ=1.53\n"
            f"images={count}\nframes={count}\nhyperstack=true\n"
            "mode=grayscale\nloop=false\n"
        )

    @staticmethod
    def _verify_output(path: Path, expected_count: int, expected_shape: tuple[int, int],
                       expected_dtype: np.dtype) -> None:
        with tifffile.TiffFile(path) as tif:
            if len(tif.pages) != expected_count:
                raise ValueError(f"输出页数校验失败：{len(tif.pages)}，期望 {expected_count}")
            if len(tif.series) != 1:
                raise ValueError(f"输出 TIFF 不是单一堆栈：检测到 {len(tif.series)} 个 series")
            for index in sorted({0, expected_count // 2, expected_count - 1}):
                page = tif.pages[index]
                if tuple(page.shape) != expected_shape or np.dtype(page.dtype) != expected_dtype:
                    raise ValueError(f"输出第 {index} 页的形状或类型校验失败")

    @staticmethod
    def write_stack(filepath: str, frames: Iterable[np.ndarray], meta: dict,
                    progress_callback: Optional[Callable[[float], None]] = None,
                    cancel_event: Optional[threading.Event] = None,
                    overwrite: bool = False,
                    compression: Optional[str] = None) -> int:
        """原子写出 TIFF；失败仅清除本次临时文件。

        Args:
            compression: ``None``/"none"（默认，未压缩，与历史行为逐位一致）
                或 "deflate"（zlib 无损）。压缩只改变容器编码，不改变像素值；
                压缩时不能使用 contiguous 写入（tifffile 明确拒绝二者同用），
                因此改为逐页写出。其余安全检查（同源拒绝、已有目标拒绝、
                磁盘/大文件容器策略、帧数/形状/类型校验、临时文件清理后发布）
                对两种编码完全一致。
        """
        if compression not in (None, "none", "deflate"):
            raise ValueError(
                f"不支持的 compression: {compression!r}（可选 None/'none'/'deflate'）")
        compress = compression == "deflate"
        target = Path(filepath)
        source_value = meta.get("source_path")
        source = Path(source_value).resolve() if source_value else None
        if TiffIO._same_file(source, target):
            raise ValueError("输出文件不能覆盖输入文件")
        if target.exists() and not overwrite:
            raise FileExistsError(f"输出文件已存在：{target}；请更换名称或明确允许覆盖")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not os.access(target.parent, os.W_OK):
            raise PermissionError(f"输出目录无写入权限：{target.parent}")

        iterator = iter(frames)
        try:
            first = np.asarray(next(iterator))
        except StopIteration as exc:
            raise ValueError("没有可保存的帧") from exc
        if first.ndim != 2:
            raise ValueError(f"输出帧必须为二维灰度图，实际为 {first.shape}")
        expected_dtype = np.dtype(meta.get("dtype", first.dtype))
        if first.dtype != expected_dtype:
            raise ValueError(f"输出类型错误：{first.dtype}，期望 {expected_dtype}")
        expected_shape = tuple(first.shape)
        expected_count = int(meta.get("n_frames", 0))
        if expected_count < 1:
            raise ValueError("输出帧数必须大于 0")
        # first 已是最终写出的帧（含 crop），故该估算同时用于磁盘检查与 BigTIFF 判定。
        estimated_bytes = first.nbytes * expected_count
        free_bytes = shutil.disk_usage(target.parent).free
        if free_bytes < estimated_bytes + DISK_SPACE_MARGIN_BYTES:
            raise OSError(
                "目标磁盘空间不足：至少需要约 "
                f"{(estimated_bytes + DISK_SPACE_MARGIN_BYTES) / 1024 ** 3:.2f} GiB")

        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".part.tif", dir=str(target.parent))
        os.close(fd)
        temporary = Path(tmp_name)
        count = 0
        try:
            description = TiffIO._imagej_description(meta, expected_count)
            with tifffile.TiffWriter(temporary, bigtiff=estimated_bytes >= BIGTIFF_THRESHOLD) as writer:
                def write_one(frame: np.ndarray, index: int) -> None:
                    nonlocal count
                    if _cancelled(cancel_event):
                        raise OperationCancelled("保存已取消")
                    page = np.asarray(frame)
                    if page.ndim != 2 or tuple(page.shape) != expected_shape or page.dtype != expected_dtype:
                        raise ValueError(
                            f"第 {index} 帧形状或类型不一致：{page.shape}/{page.dtype}，"
                            f"期望 {expected_shape}/{expected_dtype}")
                    kwargs = {"photometric": "minisblack", "metadata": None, "contiguous": True}
                    if compress:
                        # 压缩与 contiguous 互斥；改为逐页写，其余参数保持一致
                        kwargs = {"photometric": "minisblack", "metadata": None,
                                  "compression": "deflate"}
                    if index == 0:
                        if description:
                            kwargs["description"] = description
                        if meta.get("resolution"):
                            kwargs["resolution"] = meta["resolution"]
                            if meta.get("resolutionunit") is not None:
                                kwargs["resolutionunit"] = meta["resolutionunit"]
                    writer.write(page, **kwargs)
                    count += 1
                    if progress_callback:
                        progress_callback(count / expected_count)

                write_one(first, 0)
                for index, frame in enumerate(iterator, start=1):
                    write_one(frame, index)
            if count != expected_count:
                raise ValueError(f"输出帧数不一致：实际 {count}，期望 {expected_count}")
            with open(temporary, "r+b") as handle:
                os.fsync(handle.fileno())
            TiffIO._verify_output(temporary, count, expected_shape, expected_dtype)
            os.replace(temporary, target)
            logger.info("已安全保存 %s 页到 %s", count, target)
            return count
        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                logger.warning("无法删除临时输出：%s", temporary)
            raise


def _global_normalization(frames: list[np.ndarray], indices: list[int]) -> tuple[float, float]:
    """针对整段关键帧建立统一的 P1--P99.5 SIFT 映射。"""
    samples: list[np.ndarray] = []
    sampled = indices[::max(1, len(indices) // 8)][:8]
    for index in sampled:
        image = np.asarray(frames[index])
        stride = max(1, int(math.sqrt(image.size / 65_536)))
        values = image[::stride, ::stride].reshape(-1)
        values = values[np.isfinite(values)]
        if values.size:
            samples.append(values.astype(np.float64, copy=False))
    if not samples:
        raise DetectionQualityError("图像没有有限像素，无法进行漂移检测")
    values = np.concatenate(samples)
    low, high = np.percentile(values, (1.0, 99.5))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise DetectionQualityError("图像动态范围不足，无法提取可靠特征")
    return float(low), float(high)


def _to_u8(frame: np.ndarray, low: float, high: float) -> np.ndarray:
    image = np.asarray(frame, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = np.zeros(image.shape, dtype=np.float32)
    scaled[finite] = (image[finite] - low) * (255.0 / (high - low))
    return np.ascontiguousarray(np.clip(scaled, 0, 255).astype(np.uint8))


def validate_sift_params(params: Optional[dict]) -> dict:
    """校验并规范化 SIFT 与质量门控参数，返回可直接使用的数值 dict。

    非法输入抛 ValueError。GUI 与核心共用同一套范围规则，避免两处阈值
    各自漂移（例如 GUI 曾放过 ratio=0.05 而核心要求 [0.1, 1)）。
    未知键一律报错，防止调用方拼写错误后静默使用默认值。
    """
    given = params or {}
    unknown = sorted(set(given) - set(DEFAULT_SIFT_PARAMS))
    if unknown:
        raise ValueError(f"未知的 SIFT 参数：{', '.join(unknown)}")
    values = {**DEFAULT_SIFT_PARAMS, **given}
    nfeatures = int(values["nfeatures"])
    if not 0 <= nfeatures <= 100_000:
        raise ValueError("特征点数必须在 0 到 100000 之间")
    octave = int(values["noctave_layers"])
    if octave < 1:
        raise ValueError("Octave 层数必须 >= 1")
    min_matches = int(values["min_matches"])
    if min_matches < 3:
        raise ValueError("最少匹配点必须 >= 3（估计平移至少需要 3 个点）")
    min_inliers = int(values["min_inliers"])
    if min_inliers < 3:
        raise ValueError("最少内点必须 >= 3（估计平移至少需要 3 个点）")
    ratio = _finite_number(values["ratio_threshold"], "Ratio")
    if not 0.1 <= ratio < 1.0:
        raise ValueError("Ratio 必须在 [0.1, 1) 内")
    contrast = _finite_number(values["contrast_threshold"], "对比度阈值")
    edge = _finite_number(values["edge_threshold"], "边缘阈值")
    sigma = _finite_number(values["sigma"], "Sigma")
    ransac = _finite_number(values["ransac_threshold"], "RANSAC 阈值")
    max_residual = _finite_number(values["max_pair_residual"], "最大残差")
    min_ratio = _finite_number(values["min_inlier_ratio"], "最小内点比例")
    valid_ratio = _finite_number(values["min_valid_pair_ratio"], "最小有效帧对比例")
    if contrast <= 0 or edge <= 0 or sigma <= 0 or ransac <= 0 or max_residual <= 0:
        raise ValueError("SIFT 阈值必须大于 0")
    if not 0 < min_ratio <= 1 or not 0 < valid_ratio <= 1:
        raise ValueError("比例参数必须在 (0, 1] 内")
    max_gap = int(values["max_interpolation_gap"])
    if max_gap < 0:
        raise ValueError("允许插值的连续失败帧对数必须 >= 0")
    return {
        "nfeatures": nfeatures, "noctave_layers": octave,
        "contrast_threshold": contrast, "edge_threshold": edge, "sigma": sigma,
        "ratio_threshold": ratio, "ransac_threshold": ransac,
        "min_matches": min_matches, "min_inliers": min_inliers,
        "min_inlier_ratio": min_ratio, "max_pair_residual": max_residual,
        "min_valid_pair_ratio": valid_ratio, "max_interpolation_gap": max_gap,
    }


def _sift_params(params: Optional[dict]) -> tuple[dict, dict]:
    values = validate_sift_params(params)
    sift = {
        "nfeatures": values["nfeatures"],
        "nOctaveLayers": values["noctave_layers"],
        "contrastThreshold": values["contrast_threshold"],
        "edgeThreshold": values["edge_threshold"],
        "sigma": values["sigma"],
    }
    quality = {
        "ratio": values["ratio_threshold"], "ransac": values["ransac_threshold"],
        "min_matches": values["min_matches"], "min_inliers": values["min_inliers"],
        "min_inlier_ratio": values["min_inlier_ratio"],
        "max_residual": values["max_pair_residual"],
        "min_valid_pair_ratio": values["min_valid_pair_ratio"],
        "max_gap": values["max_interpolation_gap"],
    }
    return sift, quality


def _extract_features(frame: np.ndarray, sift_kwargs: dict, low: float, high: float):
    detector = cv2.SIFT_create(**sift_kwargs)
    keypoints, descriptors = detector.detectAndCompute(_to_u8(frame, low, high), None)
    points = np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32) if keypoints else np.empty((0, 2), np.float32)
    return points, descriptors


def _knn_matches(desc_a: np.ndarray, desc_b: np.ndarray, k: int = 2):
    """大规模描述子改用 FLANN 近似最近邻，小规模保持暴力匹配。

    数千描述子时 KD-tree 比 BFMatcher 的 O(N·M·d) 全量距离快约一个
    量级，对 Lowe ratio 测试的精度足够；FLANN 失败时回退暴力匹配，
    行为总是确定。
    """
    if min(desc_a.shape[0], desc_b.shape[0]) >= FLANN_MIN_DESCRIPTORS:
        try:
            matcher = cv2.FlannBasedMatcher(
                dict(algorithm=1, trees=5), dict(checks=64))
            return matcher.knnMatch(desc_a, desc_b, k=k)
        except cv2.error:
            logger.warning("FLANN 匹配失败，已回退暴力匹配", exc_info=True)
    return cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_a, desc_b, k=k)


def _match_pair(pair: tuple[int, int], features: dict, quality: dict) -> PairQuality:
    source, target = pair
    points_a, desc_a = features[source]
    points_b, desc_b = features[target]
    result = PairQuality(source, target, len(points_a), len(points_b))
    if desc_a is None or desc_b is None or len(desc_a) < quality["min_matches"] or len(desc_b) < quality["min_matches"]:
        result.message = "特征点不足"
        return result
    try:
        matches = _knn_matches(desc_a, desc_b, k=2)
    except cv2.error as exc:
        result.message = f"描述子匹配失败：{exc}"
        return result
    good = [first for pair_matches in matches if len(pair_matches) == 2
            for first, second in [pair_matches] if first.distance < quality["ratio"] * second.distance]
    result.good_matches = len(good)
    if len(good) < quality["min_matches"]:
        result.message = "通过 Lowe ratio 的匹配点不足"
        return result
    # 用 RANSAC 仿射模型只筛选空间一致的匹配内点，最终漂移仍取内点位移
    # 的中位数，因此实际应用的是纯平移。v7 曾改用全体位移的中位数门控，
    # 在跳帧或较大帧间位移时会被错误匹配簇拖垮，导致可靠帧对几乎归零。
    src = points_a[[match.queryIdx for match in good]].reshape(-1, 1, 2)
    dst = points_b[[match.trainIdx for match in good]].reshape(-1, 1, 2)
    try:
        _, inlier_mask = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=quality["ransac"],
            maxIters=3000, confidence=0.995, refineIters=10)
    except cv2.error as exc:
        result.message = f"RANSAC 失败：{exc}"
        return result
    if inlier_mask is None:
        result.message = "RANSAC 未找到内点"
        return result
    inliers = inlier_mask.ravel().astype(bool)
    result.inliers = int(inliers.sum())
    result.inlier_ratio = result.inliers / result.good_matches if result.good_matches else 0.0
    if result.inliers < quality["min_inliers"]:
        result.message = "RANSAC 内点不足"
        return result
    all_deltas = (dst - src).reshape(-1, 2)
    deltas = all_deltas[inliers]
    translation = np.median(deltas, axis=0)
    result.median_residual = float(np.median(np.linalg.norm(deltas - translation, axis=1)))
    # RANSAC 对完全对称的双峰可能任意选中其中一峰。若剩余匹配中还存在
    # 规模近似、方向明显冲突的紧密平移簇，则拒绝给出武断结果。
    rival = all_deltas[~inliers]
    if len(rival) >= quality["min_inliers"]:
        rival_centre = np.median(rival, axis=0)
        rival_count = int((np.linalg.norm(rival - rival_centre, axis=1)
                           <= quality["ransac"]).sum())
        comparable = rival_count >= max(quality["min_inliers"], math.ceil(result.inliers * 0.8))
        separated = np.linalg.norm(rival_centre - translation) > 2.0 * quality["ransac"]
        if comparable and separated:
            result.message = "存在规模相近的冲突平移簇"
            return result
    if result.inlier_ratio < quality["min_inlier_ratio"]:
        result.message = "RANSAC 内点比例不足"
        return result
    if result.median_residual > quality["max_residual"]:
        result.message = "帧对平移残差过大"
        return result
    result.dx, result.dy = float(translation[0]), float(translation[1])
    result.status = "reliable"
    return result


def _bounded_map(items: list, worker: Callable, max_workers: int,
                 cancel_event: Optional[threading.Event],
                 on_done: Optional[Callable[[int], None]] = None):
    """有限队列并行，取消时不会继续向线程池灌入任务。"""
    if not items:
        return []
    results = []
    iterator = iter(items)
    with _opencv_single_thread(), ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="drift") as executor:
        pending = {}
        for _ in range(min(max_workers, len(items))):
            item = next(iterator, None)
            if item is not None:
                pending[executor.submit(worker, item)] = item
        done_count = 0
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                item = pending.pop(future)
                if _cancelled(cancel_event):
                    for queued in pending:
                        queued.cancel()
                    return None
                results.append((item, future.result()))
                done_count += 1
                if on_done:
                    on_done(done_count)
                next_item = next(iterator, None)
                if next_item is not None:
                    pending[executor.submit(worker, next_item)] = next_item
    return results


class DriftDetector:
    """SIFT 平移检测；质量不足时显式失败而不是填充零漂移。"""

    @staticmethod
    def detect_drift(frames: list[np.ndarray], progress_callback=None, cancel_event=None,
                     params=None, skip_interval: int = 1, use_parallel: bool = True):
        n_frames = len(frames)
        if n_frames == 0:
            raise ValueError("TIFF 堆栈为空")
        if n_frames == 1:
            return DetectionResult(np.zeros(1, np.float32), np.zeros(1, np.float32), [], [0], (0.0, 1.0))
        try:
            skip = int(skip_interval)
        except (TypeError, ValueError) as exc:
            raise ValueError("跳帧间隔必须为正整数") from exc
        if skip < 1:
            raise ValueError("跳帧间隔必须大于等于 1")
        sift_kwargs, quality_settings = _sift_params(params)
        key_indices = list(range(0, n_frames, skip))
        if key_indices[-1] != n_frames - 1:
            key_indices.append(n_frames - 1)
        low, high = _global_normalization(frames, key_indices)
        if _cancelled(cancel_event):
            return None

        features: dict[int, tuple[np.ndarray, Optional[np.ndarray]]] = {}
        # 关键帧形状在派发前于主线程确定，worker 只读比较，
        # 消除原先多线程同时给 key_shape 赋值的竞态。
        key_shape = tuple(np.asarray(frames[key_indices[0]]).shape)
        if len(key_shape) != 2:
            raise ValueError(
                f"漂移检测仅支持二维灰度帧，第 {key_indices[0]} 帧形状为 {key_shape}")

        def feature_worker(index: int):
            # 二维帧校验按关键帧惰性验证，报错给出帧序号与形状。
            frame = np.asarray(frames[index])
            shape = tuple(frame.shape)
            if len(shape) != 2:
                raise ValueError(f"漂移检测仅支持二维灰度帧，第 {index} 帧形状为 {shape}")
            if shape != key_shape:
                raise ValueError(f"第 {index} 帧尺寸 {shape} 与关键帧 {key_shape} 不一致")
            return _extract_features(frame, sift_kwargs, low, high)
        def feature_progress(done: int):
            if progress_callback:
                progress_callback(0.40 * done / len(key_indices))
        if use_parallel and len(key_indices) > 3:
            mapped = _bounded_map(key_indices, feature_worker, MAX_WORKERS, cancel_event, feature_progress)
            if mapped is None:
                return None
            features = dict(mapped)
        else:
            for done, index in enumerate(key_indices, 1):
                if _cancelled(cancel_event):
                    return None
                features[index] = feature_worker(index)
                feature_progress(done)

        pairs = list(zip(key_indices, key_indices[1:]))
        def match_worker(pair: tuple[int, int]):
            return _match_pair(pair, features, quality_settings)
        def match_progress(done: int):
            if progress_callback:
                progress_callback(0.40 + 0.50 * done / len(pairs))
        if use_parallel and len(pairs) > 3:
            mapped = _bounded_map(pairs, match_worker, MAX_WORKERS, cancel_event, match_progress)
            if mapped is None:
                return None
            qualities = [dict(mapped)[pair] for pair in pairs]
        else:
            qualities = []
            for done, pair in enumerate(pairs, 1):
                if _cancelled(cancel_event):
                    return None
                qualities.append(match_worker(pair))
                match_progress(done)

        valid = np.array([quality.status == "reliable" for quality in qualities], dtype=bool)
        good_count = int(valid.sum())
        if good_count == 0:
            reasons = sorted({quality.message for quality in qualities if quality.message})
            raise DetectionQualityError("未找到可用帧间匹配；" + "；".join(reasons[:3]))
        if len(pairs) > 1:
            ratio = good_count / len(pairs)
            if ratio < quality_settings["min_valid_pair_ratio"]:
                raise DetectionQualityError(
                    f"可靠帧对仅 {good_count}/{len(pairs)}（{ratio:.0%}），低于允许阈值")
            invalid_indices = np.flatnonzero(~valid)
            if invalid_indices.size:
                if invalid_indices[0] == 0 or invalid_indices[-1] == len(pairs) - 1:
                    raise DetectionQualityError("首尾帧对匹配失败，拒绝进行不可靠外推")
                gaps = np.split(invalid_indices, np.where(np.diff(invalid_indices) != 1)[0] + 1)
                longest = max(len(gap) for gap in gaps)
                if longest > quality_settings["max_gap"]:
                    raise DetectionQualityError(f"连续 {longest} 个帧对匹配失败，超过允许插值长度")

        spans = np.asarray([target - source for source, target in pairs], dtype=np.float64)
        raw_x = np.asarray([quality.dx if quality.status == "reliable" else np.nan for quality in qualities], dtype=np.float64) / spans
        raw_y = np.asarray([quality.dy if quality.status == "reliable" else np.nan for quality in qualities], dtype=np.float64) / spans
        interpolated = np.flatnonzero(~np.isfinite(raw_x) | ~np.isfinite(raw_y)).tolist()
        # 只对质量门控已经判定为失败的帧对做插值。不能再对“可靠”帧对的
        # 速度做全局 MAD 清洗：TEM 堆栈会出现真实的阶跃/非匀速漂移，v7.0.2
        # 曾把这些真实运动当作异常值抹掉。真实 500 帧数据因此从约 193 px
        # 被错误压缩到 1.34 px，输出看起来与原图完全相同。
        if interpolated:
            anchors = np.flatnonzero(np.isfinite(raw_x) & np.isfinite(raw_y))
            raw_x[interpolated] = np.interp(interpolated, anchors, raw_x[anchors])
            raw_y[interpolated] = np.interp(interpolated, anchors, raw_y[anchors])
        increments_x, increments_y = raw_x * spans, raw_y * spans
        key_x = np.r_[0.0, np.cumsum(increments_x)]
        key_y = np.r_[0.0, np.cumsum(increments_y)]
        all_indices = np.arange(n_frames)
        result = DetectionResult(
            np.interp(all_indices, key_indices, key_x).astype(np.float32),
            np.interp(all_indices, key_indices, key_y).astype(np.float32),
            qualities, key_indices, (low, high), interpolated_pairs=interpolated,
        )
        if interpolated:
            result.warnings.append(f"{len(interpolated)} 个内部失败帧对已按相邻可靠速度插值")
        if progress_callback:
            progress_callback(1.0)
        return result


class DriftCorrector:
    """保留原始 dtype 的纯平移矫正。"""

    @staticmethod
    def _restore_dtype(image: np.ndarray, dtype: np.dtype) -> np.ndarray:
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            return np.clip(np.rint(image), info.min, info.max).astype(dtype)
        return image.astype(dtype, copy=False)

    @staticmethod
    def correct_single_frame(frame: np.ndarray, shift_x: float, shift_y: float = 0.0,
                             border_mode: str = "constant",
                             border_value: float = 0.0) -> np.ndarray:
        shift_x = _finite_number(shift_x, "水平偏移")
        shift_y = _finite_number(shift_y, "垂直偏移")
        border_value = _finite_number(border_value, "边界填充值")
        image = np.asarray(frame)
        if image.ndim != 2:
            raise ValueError("仅支持二维灰度帧")
        border_map = {"constant": cv2.BORDER_CONSTANT, "reflect": cv2.BORDER_REFLECT_101,
                      "nearest": cv2.BORDER_REPLICATE}
        if border_mode not in border_map:
            raise ValueError(f"未知边界模式：{border_mode}")
        working = image.astype(np.float64 if image.dtype == np.float64 else np.float32, copy=False)
        matrix = np.array([[1.0, 0.0, -shift_x], [0.0, 1.0, -shift_y]], dtype=np.float64)
        corrected = cv2.warpAffine(
            working, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR,
            borderMode=border_map[border_mode], borderValue=float(border_value))
        return DriftCorrector._restore_dtype(corrected, image.dtype)

    @staticmethod
    def common_valid_crop(shape: tuple[int, int], shifts_x, shifts_y=None) -> tuple[int, int, int, int]:
        sx = np.asarray(shifts_x, dtype=np.float64)
        sy = np.zeros_like(sx) if shifts_y is None else np.asarray(shifts_y, dtype=np.float64)
        if sx.ndim != 1 or sy.shape != sx.shape or not np.isfinite(sx).all() or not np.isfinite(sy).all():
            raise ValueError("偏移数组必须为等长的一维有限数值")
        height, width = shape
        left = max(0, int(math.ceil(float(np.max(-sx)))))
        right = min(width, int(math.floor(float(np.min(width - sx)))))
        top = max(0, int(math.ceil(float(np.max(-sy)))))
        bottom = min(height, int(math.floor(float(np.min(height - sy)))))
        if right <= left or bottom <= top:
            raise ValueError("漂移过大，不存在共同有效区域")
        return left, top, right, bottom

    @staticmethod
    def correct_frames(frames: Iterable[np.ndarray], shifts_x, shifts_y=None,
                       crop: Optional[tuple[int, int, int, int]] = None,
                       border_mode: str = "constant",
                       border_value: float = 0.0) -> Iterator[np.ndarray]:
        """逐帧惰性矫正：frames 可为生成器，峰值内存仅为单帧。

        帧数与偏移量的一致性在已知长度时提前校验，否则边消费边校验并在结尾复核，
        因此流式管线仍不会静默产出长度不符的输出。
        """
        expected = len(shifts_x)
        if shifts_y is not None and len(shifts_y) != expected:
            raise ValueError("帧数与偏移量数量不一致")
        known_length = getattr(frames, "__len__", None)
        if known_length is not None and known_length() != expected:
            raise ValueError("帧数与偏移量数量不一致")

        index = -1
        for index, frame in enumerate(frames):
            if index >= expected:
                raise ValueError("帧数与偏移量数量不一致")
            corrected = DriftCorrector.correct_single_frame(
                frame, shifts_x[index], 0.0 if shifts_y is None else shifts_y[index],
                border_mode, border_value)
            if crop:
                left, top, right, bottom = crop
                corrected = corrected[top:bottom, left:right]
            yield corrected
        if index + 1 != expected:
            raise ValueError("帧数与偏移量数量不一致")

    @staticmethod
    def correct_frames_list(frames, shifts_x, shifts_y=None,
                            crop: Optional[tuple[int, int, int, int]] = None,
                            border_mode: str = "constant",
                            border_value: float = 0.0) -> list[np.ndarray]:
        """v6 API 兼容：立即物化为列表。"""
        return list(DriftCorrector.correct_frames(
            frames, shifts_x, shifts_y, crop=crop,
            border_mode=border_mode, border_value=border_value))


def correct_and_save(frames: list[np.ndarray], shifts_x, shifts_y, meta: dict, output_path: str,
                     progress_callback=None, cancel_event=None, overwrite: bool = False,
                     crop_mode: str = "crop",
                     verify_info: Optional[dict] = None,
                     compression: Optional[str] = None) -> Optional[int]:
    """矫正后原子保存。默认裁剪到所有帧的共同有效区域。

    对位移最大的帧执行像素级防空转校验：若位移明显非零，但所谓矫正帧与
    同区域原始帧逐像素完全相同，则在原子替换目标文件前失败，不能再显示
    “保存成功”却产出未经矫正的数据。

    Args:
        verify_info: 可选出参。保存后回填 ``{"crop", "verify_index",
            "changed_ratio"}`` 供审计报告使用；未执行生效校验时后两项为 None。
        compression: ``None``/"none"（默认，未压缩，与历史行为一致）或
            "deflate"。仅调整编码与 contiguous，不改变校正算法、裁剪、
            生效校验或任何安全检查（见 :meth:`TiffIO.write_stack`）。
    """
    if not frames:
        raise ValueError("没有待保存的帧")
    if len(frames) != len(shifts_x) or (shifts_y is not None and len(frames) != len(shifts_y)):
        raise ValueError("帧数与偏移量数量不一致")
    crop = None
    if crop_mode == "crop":
        crop = DriftCorrector.common_valid_crop(tuple(frames[0].shape), shifts_x, shifts_y)
    elif crop_mode != "keep":
        raise ValueError("crop_mode 只能为 'crop' 或 'keep'")
    output_meta = dict(meta)
    output_meta["n_frames"] = len(frames)
    output_meta["dtype"] = np.asarray(frames[0]).dtype
    if crop:
        left, top, right, bottom = crop
        output_meta["size"] = (right - left, bottom - top)
        output_meta["shape"] = (bottom - top, right - left)
        output_meta["crop"] = {"left": left, "top": top, "right": right, "bottom": bottom}
    if verify_info is not None:
        verify_info.clear()
        verify_info.update({"crop": dict(output_meta["crop"]) if crop else None,
                            "verify_index": None, "changed_ratio": None})

    sx = np.asarray(shifts_x, dtype=np.float64)
    sy = np.zeros_like(sx) if shifts_y is None else np.asarray(shifts_y, dtype=np.float64)
    magnitudes = np.hypot(sx, sy)
    verify_index = int(np.argmax(magnitudes))
    verify_required = float(magnitudes[verify_index]) >= VERIFY_MIN_SHIFT_MAGNITUDE
    verified = False

    def verified_frames() -> Iterator[np.ndarray]:
        nonlocal verified
        corrected_frames = DriftCorrector.correct_frames(
            frames, shifts_x, shifts_y, crop=crop)
        for index, corrected in enumerate(corrected_frames):
            if verify_required and index == verify_index:
                source_region = np.asarray(frames[index])
                if crop:
                    left, top, right, bottom = crop
                    source_region = source_region[top:bottom, left:right]
                if np.array_equal(corrected, source_region):
                    raise ValueError(
                        f"拒绝保存未生效的矫正结果：第 {index + 1} 帧检测位移为 "
                        f"dx={sx[index]:.3f}px, dy={sy[index]:.3f}px，"
                        "但矫正后像素与原始数据完全相同")
                changed_ratio = float(np.count_nonzero(corrected != source_region) / corrected.size)
                logger.info(
                    "矫正生效校验：第 %d 帧，位移=(%.3f, %.3f)，变化像素比例=%.2f%%",
                    index + 1, sx[index], sy[index], changed_ratio * 100.0)
                if verify_info is not None:
                    verify_info["verify_index"] = index
                    verify_info["changed_ratio"] = changed_ratio
                verified = True
            yield corrected

    try:
        written = TiffIO.write_stack(
            output_path,
            verified_frames(),
            output_meta, progress_callback=progress_callback, cancel_event=cancel_event,
            overwrite=overwrite,
            compression=compression,
        )
        if verify_required and not verified:
            raise ValueError("矫正生效校验未执行，拒绝报告保存成功")
        return written
    except OperationCancelled:
        return None


def _report_paths(output_folder: Path) -> tuple[Path, Path]:
    # 时间戳精确到微秒：同一秒内连续两次批处理不得互相覆盖审计报告。
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return output_folder / f"drift_batch_report_{stamp}.json", output_folder / f"drift_batch_report_{stamp}.csv"


def _ambiguous_axis_error(meta: dict) -> Optional[str]:
    """meta 含未确认的 C/Z 轴时返回给批处理记录的错误文本。"""
    confirm = meta.get("confirm_required") or []
    if not confirm:
        return None
    return ("文件序列轴含通道(C)或层(Z)维度，未获按时间序列处理的确认："
            + " ".join(confirm)
            + "。如确认这些页确实应按页视为时间帧（TEM 连续采集常被相机软件"
              "把时间轴标注为 slices/Z），请重新运行批处理并在轴含义确认对话框中"
              "选择“是”；API 调用可设置 allow_ambiguous_axes=True 跳过该检查")


_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@")

# 审计报告 CSV 的固定列优先顺序；未列出的新增字段按字母序追加在末尾，
# 保证多次运行之间列序稳定、人工可比。
_REPORT_COLUMN_PRIORITY = (
    "source_path", "status", "shift_source", "output_path", "written_frames",
    "valid_pair_ratio", "max_abs_shift_x", "max_abs_shift_y",
    "shift_x_range", "shift_y_range", "shift_table", "crop",
    "verify_index", "changed_ratio", "duration_s", "started_at",
    "source_size_bytes", "source_mtime", "warnings", "error",
)


def _csv_safe(value):
    """防 Excel 公式注入：以公式引导字符开头的字符串前置单引号。"""
    if isinstance(value, str) and value.startswith(_CSV_FORMULA_PREFIXES):
        return "'" + value
    return value


def write_shift_table(filepath, shifts_x, shifts_y=None, key_indices=None,
                      interpolated_pairs=None) -> Path:
    """写出逐帧位移表 CSV（utf-8-sig，Excel 可直接打开）。

    列：frame（0 起）、shift_x_px、shift_y_px、key_frame、segment_estimated。
    segment_estimated 标记该帧所在帧对未通过质量门控、位移由相邻可靠速度
    插值得到的区段（含区段右端关键帧），供后续定量分析甄别数据来源。
    """
    sx = np.asarray(shifts_x, dtype=np.float64)
    sy = np.zeros_like(sx) if shifts_y is None else np.asarray(shifts_y, dtype=np.float64)
    if sx.ndim != 1 or sy.shape != sx.shape or sx.size == 0:
        raise ValueError("位移表需要非空且等长的一维位移数组")
    if not np.isfinite(sx).all() or not np.isfinite(sy).all():
        raise ValueError("位移数组包含非有限值，无法写出位移表")
    n = int(sx.size)
    keys = list(key_indices) if key_indices else list(range(n))
    key_flags = np.zeros(n, dtype=bool)
    for key in keys:
        if 0 <= key < n:
            key_flags[key] = True
    estimated = np.zeros(n, dtype=bool)
    for pair in set(interpolated_pairs or ()):
        if 0 <= pair < len(keys) - 1:
            lo, hi = keys[pair], min(keys[pair + 1], n - 1)
            if lo + 1 <= hi:
                estimated[lo + 1:hi + 1] = True
    target = Path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "shift_x_px", "shift_y_px", "key_frame", "segment_estimated"])
        for index in range(n):
            writer.writerow([index, f"{sx[index]:.4f}", f"{sy[index]:.4f}",
                             int(key_flags[index]), int(estimated[index])])
    os.replace(temporary, target)
    return target


def batch_process(file_list, output_folder, params=None, skip_interval=1, precomputed=None,
                  progress_callback=None, cancel_event=None, overwrite=False,
                  crop_mode="crop", allow_ambiguous_axes: bool = False):
    """安全批处理；生成 JSON/CSV 审计报告。

    Args:
        precomputed: ``{规范化源路径: {"frames", "shifts_x", "shifts_y", "meta",
            "detection"(可选 DetectionResult)}}``。命中的文件跳过重新读取与检测；
            其帧数据由调用方持有，本函数不释放。
        allow_ambiguous_axes: 批处理无人值守，默认拒绝序列轴含 C/Z 的文件，
            防止把通道或 z 层当时间帧矫正出看似正常实则错误的结果。

    进度按文件内阶段加权上报（检测 0--0.8、保存 0.8--1），大单文件处理
    期间进度条不再长时间停滞。轴确认与内存保护发生在整栈解码之前，
    inspect 结果直接传给 read_stack 复用，不再重复扫描目录。
    """
    output_root = Path(output_folder)
    output_root.mkdir(parents=True, exist_ok=True)
    source_paths = [str(Path(item).resolve()) for item in file_list]
    errors, records = [], []
    completed = 0
    cancelled = False
    n_files = max(1, len(source_paths))

    def stage_progress(file_index: int, lo: float, hi: float) -> Optional[Callable[[float], None]]:
        if progress_callback is None:
            return None

        def report(ratio: float) -> None:
            clamped = max(0.0, min(1.0, float(ratio)))
            progress_callback(min(0.999, (file_index + lo + (hi - lo) * clamped) / n_files))

        return report

    for index, source_name in enumerate(source_paths):
        if _cancelled(cancel_event):
            cancelled = True
            break
        file_started = time.monotonic()
        record = {"source_path": source_name, "status": "failed", "warnings": [],
                  "started_at": datetime.now().isoformat(timespec="milliseconds")}
        try:
            stat = Path(source_name).stat()
            record["source_size_bytes"] = stat.st_size
            record["source_mtime"] = datetime.fromtimestamp(
                stat.st_mtime).isoformat(timespec="seconds")
        except OSError:
            pass
        frames = None
        owns_frames = False
        try:
            cached = (precomputed or {}).get(source_name)
            if cached is not None:
                frames = cached["frames"]
                shifts_x, shifts_y = cached["shifts_x"], cached["shifts_y"]
                meta = cached["meta"]
                # 命中 precomputed 缓存时仍比对磁盘 TIFF 的结构，避免文件已被
                # 替换/增删页/改形状后继续使用旧帧数据写出错误结果。
                disk_meta = TiffIO.inspect_stack(source_name, max_bytes=MAX_STACK_BYTES)
                cached_n_frames = int(meta.get("n_frames", len(frames)))
                if (disk_meta["n_frames"] != cached_n_frames
                        or disk_meta["n_frames"] != len(frames)
                        or disk_meta["shape"] != tuple(meta.get("shape", ()))
                        or disk_meta["dtype"] != np.dtype(meta.get("dtype", disk_meta["dtype"]))):
                    raise ValueError(
                        f"precomputed 缓存与磁盘 TIFF 结构不一致："
                        f"磁盘 {disk_meta['n_frames']}帧/{disk_meta['shape']}/{disk_meta['dtype']}，"
                        f"缓存 {len(frames)}帧/{tuple(meta.get('shape', ()))}/"
                        f"{meta.get('dtype', '未记录')}")
                record["shift_source"] = "precomputed"
                if not allow_ambiguous_axes:
                    axis_error = _ambiguous_axis_error(disk_meta)
                    if axis_error:
                        raise ValueError(axis_error)
                detection = cached.get("detection")
                if detection is not None:
                    record["valid_pair_ratio"] = detection.valid_pair_ratio
                    record["warnings"].extend(detection.warnings)
                save_progress = stage_progress(index, 0.0, 1.0)
            else:
                # 先只读目录：轴确认与内存保护都发生在整栈解码之前；
                # inspect 结果传入 read_stack 复用，消除重复扫描。
                disk_meta = TiffIO.inspect_stack(source_name, max_bytes=MAX_STACK_BYTES)
                if not allow_ambiguous_axes:
                    axis_error = _ambiguous_axis_error(disk_meta)
                    if axis_error:
                        raise ValueError(axis_error)
                total_mem = total_physical_memory()
                estimated = int(disk_meta.get("estimated_bytes") or 0)
                if total_mem and estimated > MEMORY_CRITICAL_RATIO * total_mem:
                    raise MemoryError(
                        f"堆栈解码后约需 {estimated / 1024 ** 3:.1f} GiB，超过本机物理内存 "
                        f"{total_mem / 1024 ** 3:.1f} GiB 的 {MEMORY_CRITICAL_RATIO:.0%}；"
                        "已跳过该文件。如确需处理，请在界面中单独加载并确认")
                frames, meta = TiffIO.read_stack(
                    source_name, cancel_event=cancel_event, meta=disk_meta)
                owns_frames = True
                detection = DriftDetector.detect_drift(
                    frames, params=params, skip_interval=skip_interval,
                    cancel_event=cancel_event,
                    progress_callback=stage_progress(index, 0.0, 0.8))
                if detection is None:
                    record["status"] = "cancelled"
                    records.append(record)
                    cancelled = True
                    break
                shifts_x, shifts_y = detection.shifts_x, detection.shifts_y
                record["shift_source"] = "detected"
                record["valid_pair_ratio"] = detection.valid_pair_ratio
                record["warnings"].extend(detection.warnings)
                save_progress = stage_progress(index, 0.8, 1.0)
            source = Path(source_name)
            output = output_root / f"{source.stem}_corrected{source.suffix}"
            verify: dict = {}
            saved = correct_and_save(frames, shifts_x, shifts_y, meta, str(output),
                                     progress_callback=save_progress,
                                     cancel_event=cancel_event, overwrite=overwrite,
                                     crop_mode=crop_mode, verify_info=verify)
            if saved is None:
                record["status"] = "cancelled"
                records.append(record)
                cancelled = True
                break
            record.update({"status": "completed", "output_path": str(output), "written_frames": saved,
                           "max_abs_shift_x": float(np.max(np.abs(shifts_x))),
                           "max_abs_shift_y": (float(np.max(np.abs(shifts_y)))
                                               if shifts_y is not None else 0.0),
                           "shift_x_range": [float(np.min(shifts_x)), float(np.max(shifts_x))],
                           "shift_y_range": ([float(np.min(shifts_y)), float(np.max(shifts_y))]
                                             if shifts_y is not None else [0.0, 0.0])})
            for key, value in verify.items():
                if value is not None:
                    record[key] = value
            if detection is not None and detection.pair_quality:
                # 逐帧对质量明细只进 JSON 报告，供事后复核匹配质量分布。
                record["pair_quality"] = [
                    {"source": pq.source_index, "target": pq.target_index, "status": pq.status,
                     "dx": pq.dx, "dy": pq.dy,
                     "keypoints_source": pq.keypoints_source,
                     "keypoints_target": pq.keypoints_target,
                     "good_matches": pq.good_matches, "inliers": pq.inliers,
                     "inlier_ratio": round(pq.inlier_ratio, 4),
                     "median_residual": (round(pq.median_residual, 3)
                                         if math.isfinite(pq.median_residual) else None),
                     "message": pq.message}
                    for pq in detection.pair_quality]
            try:
                table_path = write_shift_table(
                    output.with_name(f"{output.stem}_shifts.csv"), shifts_x, shifts_y,
                    detection.key_indices if detection is not None else None,
                    detection.interpolated_pairs if detection is not None else None)
                record["shift_table"] = str(table_path)
            except (OSError, ValueError) as exc:
                record["warnings"].append(f"位移表写出失败：{exc}")
            completed += 1
        except OperationCancelled:
            record["status"] = "cancelled"
            records.append(record)
            cancelled = True
            break
        except Exception as exc:
            message = str(exc)
            errors.append((Path(source_name).name, message))
            record["error"] = message
            logger.exception("批处理失败：%s", source_name)
        finally:
            record["duration_s"] = round(time.monotonic() - file_started, 3)
            # 仅释放本轮自行读取的帧；界面持有的缓存帧不能在此丢弃。
            if owns_frames:
                frames = None
        records.append(record)
        if progress_callback:
            progress_callback((index + 1) / len(source_paths))

    json_path, csv_path = _report_paths(output_root)
    report = {"tool": TOOL_NAME, "tool_version": __version__,
              "generated_at": datetime.now().isoformat(timespec="seconds"), "completed": completed,
              "total": len(source_paths), "cancelled": cancelled, "errors": errors,
              "records": records,
              "settings": {"params": params, "skip_interval": skip_interval,
                           "crop_mode": crop_mode, "overwrite": overwrite,
                           "allow_ambiguous_axes": allow_ambiguous_axes}}
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    used = {key for record in records for key in record}
    used.discard("pair_quality")  # 逐帧对明细只进 JSON，避免 CSV 单元格膨胀
    columns = ([name for name in _REPORT_COLUMN_PRIORITY if name in used]
               + sorted(used - set(_REPORT_COLUMN_PRIORITY)))
    if not columns:
        columns = ["source_path", "status"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = {}
            for key in columns:
                value = record.get(key, "")
                if key in ("shift_x_range", "shift_y_range") and isinstance(value, list):
                    value = f"{value[0]:.3f},{value[1]:.3f}"
                row[key] = value
            row["warnings"] = " | ".join(record.get("warnings", []))
            writer.writerow({key: _csv_safe(row[key]) for key in columns})
    return {"completed": completed, "total": len(source_paths), "errors": errors,
            "cancelled": cancelled, "records": records,
            "report_json": str(json_path), "report_csv": str(csv_path)}


class WorkerThread(threading.Thread):
    """可取消任务线程；非 daemon，关闭窗口时可安全等待其收尾。"""

    def __init__(self, target_func, args=(), kwargs=None, on_progress=None,
                 on_complete=None, on_error=None, on_finished=None):
        super().__init__(daemon=False, name="drift-worker")
        self.target_func = target_func
        self.args = args
        self.kwargs = dict(kwargs or {})
        self.on_progress = on_progress
        self.on_complete = on_complete
        self.on_error = on_error
        self.on_finished = on_finished
        self.cancel_event = threading.Event()
        self.result = None

    def run(self):
        try:
            self.kwargs.setdefault("cancel_event", self.cancel_event)
            self.kwargs.setdefault("progress_callback", self._progress_wrapper)
            self.result = self.target_func(*self.args, **self.kwargs)
            if self.on_complete:
                self.on_complete(self.result)
        except OperationCancelled:
            if self.on_complete:
                self.on_complete(None)
        except Exception as exc:
            logger.exception("后台任务失败")
            if self.on_error:
                self.on_error(exc)
        finally:
            if self.on_finished:
                self.on_finished(self.cancel_event.is_set())

    def _progress_wrapper(self, ratio):
        if self.on_progress:
            self.on_progress(max(0.0, min(1.0, float(ratio))))

    def cancel(self):
        self.cancel_event.set()
