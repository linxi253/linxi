from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import tifffile

from .models import FrameSpec
from .utils import fsync_file


# 经典 TIFF 的 32 位偏移在 4 GiB 附近不可靠。超过该阈值时改用单 IFD 截断式
# 结构（truncate）：只写首页 IFD、像素数据连续存放，ImageJ 1.53c 按描述中的
# frames 数以虚拟栈打开，任意大小可用；普通多页 IFD 结构则任何标准读取器都可读。
TRUNCATE_THRESHOLD_BYTES = int(3.8 * 1024**3)

# 写入描述串的 ImageJ 版本标记（ImageJ 兼容性声明，非工具版本）。
IMAGEJ_VERSION = "1.53c"


class OutputValidationError(ValueError):
    pass


def estimate_raw_bytes(frame_count: int | None, spec: FrameSpec) -> int | None:
    return frame_count * spec.bytes_per_frame if frame_count is not None else None


def needs_truncate(frame_count: int, spec: FrameSpec) -> bool:
    return frame_count * spec.bytes_per_frame > TRUNCATE_THRESHOLD_BYTES


def _axes(spec: FrameSpec) -> str:
    return "TYX" if spec.channels == 1 else "TYXS"


def _validate_tiff(
    path: Path,
    expected_frames: int,
    spec: FrameSpec,
    expected_truncate: bool,
) -> None:
    with tifffile.TiffFile(path) as tif:
        if tif.is_ome:
            raise OutputValidationError("输出不应为 OME-TIFF")
        if not tif.is_imagej:
            raise OutputValidationError("输出不是 ImageJ 兼容的 TIFF 堆栈")
        if tif.is_bigtiff:
            raise OutputValidationError("输出不应为 BigTIFF")
        ij_metadata = tif.imagej_metadata or {}
        if ij_metadata.get("ImageJ") != IMAGEJ_VERSION:
            raise OutputValidationError(
                f"ImageJ 描述版本错误: {ij_metadata.get('ImageJ')!r}，期望 {IMAGEJ_VERSION!r}"
            )
        page_count = len(tif.pages)
        if expected_truncate:
            if page_count != 1:
                raise OutputValidationError(
                    f"截断式堆栈应只有首页 IFD，实际 {page_count} 页"
                )
        elif expected_frames > 1 and page_count != expected_frames:
            raise OutputValidationError(
                f"TIFF 页数错误: {page_count}，期望 {expected_frames}"
            )
        series = tif.series[0]
        expected_shape = (expected_frames, *spec.shape)
        expected_axes = _axes(spec)
        actual_shape = tuple(series.shape)
        actual_axes = series.axes
        # tifffile intentionally squeezes a singleton leading T dimension when
        # presenting a series. The description still carries frames=1, so
        # normalize that reader view before validating the logical stack dims.
        if (
            expected_frames == 1
            and actual_shape == spec.shape
            and actual_axes == expected_axes[1:]
        ):
            actual_shape = (1, *actual_shape)
            actual_axes = f"T{actual_axes}"
        if actual_shape != expected_shape:
            raise OutputValidationError(f"TIFF 维度错误: {series.shape}，期望 {expected_shape}")
        if actual_axes != expected_axes:
            raise OutputValidationError(f"TIFF 轴错误: {series.axes}，期望 {expected_axes}")
        if np.dtype(series.dtype).name != spec.dtype:
            raise OutputValidationError(f"TIFF 位深错误: {series.dtype}，期望 {spec.dtype}")


class StackWriter:
    """Write one disk-backed ImageJ TIFF stack; classic TIFF, never BigTIFF.

    无压缩连续像素数据：≤3.8 GiB 写完整多页 IFD 链（标准读取器可读），
    超过后切换单 IFD 截断式，ImageJ 以虚拟栈打开。
    """

    def __init__(
        self,
        output_path: Path,
        spec: FrameSpec,
        expected_frames: int,
        interval_s: float | None,
    ):
        self.output_path = output_path
        self.spec = spec
        self.expected_frames = expected_frames
        self.interval_s = interval_s
        self.truncated = needs_truncate(expected_frames, spec)
        self._array: np.memmap | None = None

    @property
    def container_name(self) -> str:
        return "ImageJ-TIFF"

    def _metadata(self) -> dict[str, object]:
        # 'ImageJ' 键被 tifffile 用作描述串中的版本标记，不会出现在键值行里
        metadata: dict[str, object] = {"ImageJ": IMAGEJ_VERSION, "axes": _axes(self.spec)}
        # interval_s <= 0（如两帧 PTS 相同被算出的 0 间隔）不可作为固定帧
        # 间隔写入：既会除零，也不是真实的时间轴信息
        if self.interval_s is not None and self.interval_s > 0:
            metadata.update(
                {"finterval": float(self.interval_s), "fps": 1.0 / float(self.interval_s)}
            )
        return metadata

    def _photometric(self) -> str:
        # Explicit value prevents small grayscale widths of 3 or 4 pixels from
        # being misinterpreted by tifffile as RGB/RGBA sample dimensions.
        return "minisblack" if self.spec.channels == 1 else "rgb"

    def open(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._array = tifffile.memmap(
            self.output_path,
            shape=(self.expected_frames, *self.spec.shape),
            dtype=self.spec.dtype,
            imagej=True,
            bigtiff=False,
            truncate=self.truncated,
            photometric=self._photometric(),
            metadata=self._metadata(),
        )

    def write(self, index: int, frame: np.ndarray) -> None:
        if self._array is None:
            raise RuntimeError("writer 尚未打开")
        if index >= self.expected_frames:
            raise OutputValidationError("实际帧数大于预先计数，任务已停止以防生成错误堆栈")
        if frame.shape != self.spec.shape or frame.dtype.name != self.spec.dtype:
            raise OutputValidationError("解码帧与输出规格不一致")
        self._array[index] = frame

    def close(self, actual_frames: int) -> None:
        if self._array is None:
            raise RuntimeError("writer 尚未打开")
        self._array.flush()
        del self._array
        self._array = None
        if actual_frames != self.expected_frames:
            raise OutputValidationError(
                f"实际帧数 {actual_frames} 与预先计数 {self.expected_frames} 不一致"
            )
        # 提交（os.replace）前把像素数据刷到物理磁盘；否则断电时可能提交
        # 一个校验时完好、重启后损坏的堆栈，且孤儿检测无法发现它
        fsync_file(self.output_path)
        _validate_tiff(
            self.output_path,
            self.expected_frames,
            self.spec,
            self.truncated,
        )

    def abort(self) -> None:
        if self._array is not None:
            # 数据即将被删除，flush 只是尽力而为：磁盘满等导致的二次异常
            # 不能掩盖触发 abort 的原始错误
            try:
                self._array.flush()
            except OSError:
                pass
            del self._array
            self._array = None


def remove_partial(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)
