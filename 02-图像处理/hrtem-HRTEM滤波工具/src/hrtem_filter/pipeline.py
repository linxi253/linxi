"""Single-frame and streaming stack processing with cancellation and atomic output."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import numpy as np

from ._version import __version__
from .core import HRTEMFilter
from .geometry import Roi, validate_roi
from .params import FilterParams, OutputEncoding, SaveOptions
from .provenance import (
    build_provenance,
    compute_file_sha256,
    restore_json_bytes,
    write_json_atomically,
)
from .tiff_io import (
    AtomicTiffWriter,
    TiffFrameSource,
    encode_frame,
    ensure_distinct_paths,
    inspect_tiff,
)


class ProcessingCancelled(RuntimeError):
    """A deliberate cancellation; no final TIFF is published."""


class InputFileChangedError(RuntimeError):
    """Raised when the source TIFF changed between inspection and writing."""


ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class ProcessingReport:
    output_path: Path
    provenance_path: Path
    frame_count: int
    output_dtype: np.dtype
    display_range: tuple[float, float] | None


class StackProcessor:
    """Streams pages, never loads a stack into memory, and never overwrites input."""

    # Below this budget the uint8-display scan pass keeps its filtered frames
    # so the write pass does not have to re-run every FFT.
    SCAN_CACHE_BYTES = 256 * 1024 * 1024

    def __init__(self, filter_processor: HRTEMFilter | None = None) -> None:
        self.filter_processor = filter_processor or HRTEMFilter()

    @staticmethod
    def _check_cancel(cancel_event: Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise ProcessingCancelled("用户已取消处理；未发布正式输出文件")

    @staticmethod
    def _assert_input_unchanged(source_path: Path, signature: tuple[int, int, int, int]) -> None:
        """Re-stat the source and compare st_dev/st_ino/size/mtime_ns."""
        try:
            stat = source_path.stat()
        except FileNotFoundError as exc:
            raise InputFileChangedError(
                f"输入文件在读取与写入之间被删除，已中止处理：{source_path}\n"
                f"处理前签名={signature}"
            ) from exc
        current = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if current != signature:
            raise InputFileChangedError(
                f"输入文件在读取与写入之间发生了变化，已中止处理：{source_path}\n"
                f"处理前签名={signature}，写阶段前签名={current}"
            )

    def _global_display_range(
        self,
        source: TiffFrameSource,
        params: FilterParams,
        roi: Roi | None,
        cancel_event: Event | None,
        progress: ProgressCallback | None,
        output_shape: tuple[int, int],
    ) -> tuple[tuple[float, float], list[np.ndarray] | None]:
        low = np.inf
        high = -np.inf
        total = source.info.frame_count
        frame_bytes = output_shape[0] * output_shape[1] * 4
        cache: list[np.ndarray] | None = [] if frame_bytes * total <= self.SCAN_CACHE_BYTES else None
        with source.session() as reader:
            for index in range(total):
                self._check_cancel(cancel_event)
                result = self.filter_processor.process_image(reader.read_frame(index), params, roi=roi)
                frame = result.primary
                low = min(low, float(np.min(frame)))
                high = max(high, float(np.max(frame)))
                if cache is not None:
                    cache.append(frame)
                if progress:
                    progress("scan", index + 1, total)
        if not np.isfinite(low) or not high > low:
            # Constant output is valid; make a deterministic display TIFF.
            base = low if np.isfinite(low) else 0.0
            return (base, base + 1.0), cache
        return (low, high), cache

    def process_tiff(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        params: FilterParams | None = None,
        save_options: SaveOptions | None = None,
        roi: Roi | tuple[int, int, int, int] | None = None,
        cancel_event: Event | None = None,
        progress: ProgressCallback | None = None,
        entry: str | None = None,
        input_hash: bool = False,
    ) -> ProcessingReport:
        params = (params or FilterParams()).validated()
        save_options = (save_options or SaveOptions()).validated()
        started = time.monotonic()
        source_path, target_path = ensure_distinct_paths(input_path, output_path)
        # 处理前记录输入文件身份，写阶段前复核；输入在中间被替换/追加/截断时
        # 必须中止，避免把新文件页错配到旧文件的处理参数与 provenance。
        input_stat = source_path.stat()
        input_signature = (
            input_stat.st_dev,
            input_stat.st_ino,
            input_stat.st_size,
            input_stat.st_mtime_ns,
        )
        info = inspect_tiff(source_path)
        # 内容哈希是可选的归档级标识：stat 签名可被刻意保全（mtime/size
        # 均可人为保持），sha256 不可伪造，代价是对输入的一次完整读取。
        input_sha256 = compute_file_sha256(source_path) if input_hash else None
        checked_roi = validate_roi(roi, info.frame_shape)
        source = TiffFrameSource(info)
        output_shape = checked_roi.shape if checked_roi else info.frame_shape
        if save_options.encoding == OutputEncoding.FLOAT32:
            output_dtype = np.dtype(np.float32)
        elif save_options.encoding == OutputEncoding.UINT8_DISPLAY:
            output_dtype = np.dtype(np.uint8)
        elif np.issubdtype(info.dtype, np.integer):
            output_dtype = info.dtype
        else:
            output_dtype = np.dtype(np.float32)

        roi_tuple = None if checked_roi is None else (
            checked_roi.top,
            checked_roi.left,
            checked_roi.bottom,
            checked_roi.right,
        )
        hint = f"HRTEM Filter v{__version__}; complete provenance is in the adjacent .processing.json file."
        # 写入器先于任何长计算构造：目标目录不存在或不可写（权限、只读卷）
        # 必须在预扫描开始前失败，否则 uint8 显示预扫描可能白算数分钟。
        writer = AtomicTiffWriter(
            target_path,
            frame_shape=output_shape,
            expected_frames=info.frame_count,
            metadata=info.metadata,
            provenance_hint=hint,
            output_dtype=output_dtype,
        )
        provenance_path = target_path.with_suffix(target_path.suffix + ".processing.json")
        json_existed = provenance_path.exists()
        # 旧 provenance 内容先读入内存：JSON 已被新记录替换后 commit 再失败时，
        # 必须把旧记录原样写回，避免"旧 TIFF 配新参数记录"的错误元数据组合。
        previous_provenance = provenance_path.read_bytes() if json_existed else None
        json_published = False
        display_range = save_options.display_range
        cached_frames: list[np.ndarray] | None = None
        try:
            if save_options.encoding == OutputEncoding.UINT8_DISPLAY and display_range is None:
                display_range, cached_frames = self._global_display_range(
                    source, params, checked_roi, cancel_event, progress, output_shape
                )
            # 写阶段前复核：uint8 显示预扫描可能已耗时数分钟，输入文件必须原样未动。
            self._assert_input_unchanged(source_path, input_signature)
            with source.session() as reader:
                for index in range(info.frame_count):
                    self._check_cancel(cancel_event)
                    if cached_frames is not None:
                        frame = cached_frames[index]
                    else:
                        frame = self.filter_processor.process_image(
                            reader.read_frame(index), params, roi=checked_roi
                        ).primary
                    encoded = encode_frame(
                        frame,
                        source_dtype=info.dtype,
                        options=save_options,
                        display_range=display_range,
                    )
                    writer.write(encoded)
                    if progress:
                        progress("write", index + 1, info.frame_count)
            # 写循环结束后复核：预扫描加写入可能耗时数分钟，期间输入若被
            # 同形状覆盖，无法从读到的数据本身察觉，必须以签名再次确认。
            self._assert_input_unchanged(source_path, input_signature)
            # Provenance 在写循环完成后构建，elapsed 才能覆盖帧处理耗时；
            # 记录先于 TIFF 发布：成功输出必须永远有配对处理记录。
            integrity = {
                "method": "stat",
                "verified_before_write": True,
                "st_dev": input_signature[0],
                "st_ino": input_signature[1],
                "st_size": input_signature[2],
                "st_mtime_ns": input_signature[3],
            }
            if input_sha256 is not None:
                integrity["sha256"] = input_sha256
            provenance = build_provenance(
                input_path=source_path,
                input_shape=info.frame_shape,
                input_dtype=info.dtype,
                frame_count=info.frame_count,
                params=params,
                save_options=save_options,
                roi=roi_tuple,
                display_range=display_range,
                source_metadata={
                    "axes": info.metadata.axes,
                    "description": info.metadata.description,
                    "resolution": list(info.metadata.resolution) if info.metadata.resolution else None,
                    "resolution_unit": str(info.metadata.resolution_unit) if info.metadata.resolution_unit else None,
                },
                input_integrity=integrity,
                output_dtype=output_dtype,
                output_frame_shape=output_shape,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                entry=entry,
                fft_workers=self.filter_processor.workers,
            )
            write_json_atomically(provenance_path, provenance)
            json_published = True
            output = writer.commit()
        except Exception:
            writer.abort()
            if json_existed:
                if json_published:
                    restore_json_bytes(provenance_path, previous_provenance)
                # JSON 尚未被替换（写入失败或未执行到）：旧记录保持原样。
            else:
                with contextlib.suppress(FileNotFoundError):
                    provenance_path.unlink()
            raise
        return ProcessingReport(output, provenance_path, info.frame_count, output_dtype, display_range)
