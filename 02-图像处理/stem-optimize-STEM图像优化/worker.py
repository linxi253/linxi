"""Transactional background processing for validated TIFF stacks."""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import logging
import os
import queue
import shutil
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy
import tifffile

from contrast import (
    CLAHE_TILE_GRID,
    DEFAULT_HIGH_PERCENTILE,
    DEFAULT_LOW_PERCENTILE,
    estimate_global_range,
)
from errors import InputValidationError, OperationCancelled
from filters import clear_filter_caches
from pipeline import DEFAULT_PARAMS, process_frame, validate_params
from tiff_handler import (
    TiffStackReader,
    TiffStackWriter,
    available_memory_bytes,
    estimate_output_bytes,
    estimate_peak_memory_bytes,
    file_signature,
    paths_refer_to_same_file,
    validate_output_file,
)
from version import ALGORITHM_VERSION, APP_NAME, APP_VERSION

logger = logging.getLogger(__name__)


def _sha256(path: str, cancel_event=None) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise OperationCancelled("已取消文件校验")
            chunk = stream.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(final_path: str, suffix: str) -> str:
    directory = os.path.dirname(final_path)
    prefix = f".{os.path.basename(final_path)}."
    descriptor, path = tempfile.mkstemp(dir=directory, prefix=prefix, suffix=suffix)
    os.close(descriptor)
    return path


def _fsync_file(path: str):
    """Flush a completed file to stable storage before publication."""
    with open(path, "r+b") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: str):
    """Flush directory metadata so published renames survive power loss.

    Best-effort: filesystems or platforms without directory fsync support
    simply log at debug level.
    """
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32
            create_file = kernel32.CreateFileW
            create_file.restype = ctypes.c_void_p
            create_file.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.LPVOID,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            ]
            flush_buffers = kernel32.FlushFileBuffers
            flush_buffers.argtypes = [wintypes.HANDLE]
            close_handle = kernel32.CloseHandle
            close_handle.argtypes = [wintypes.HANDLE]

            handle = create_file(
                path,
                0x80000000,  # GENERIC_READ
                0x1 | 0x2 | 0x4,  # FILE_SHARE_READ | WRITE | DELETE
                None,
                3,  # OPEN_EXISTING
                0x02000000,  # FILE_FLAG_BACKUP_SEMANTICS
                None,
            )
            if handle == ctypes.c_void_p(-1).value:
                logger.debug("无法打开目录句柄用于 fsync: %s", path)
                return
            try:
                if not flush_buffers(handle):
                    logger.debug("目录 FlushFileBuffers 失败: %s", path)
            finally:
                close_handle(handle)
        else:
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except OSError:
        logger.debug("目录 fsync 不可用，已跳过: %s", path)


def stale_temporary_files(save_path: str) -> list[str]:
    """Return leftover partial/backup files belonging to one output target."""
    directory = os.path.dirname(save_path)
    prefix = f".{os.path.basename(save_path)}."
    try:
        entries = sorted(os.listdir(directory))
    except OSError:
        return []
    stale = []
    for name in entries:
        if not name.startswith(prefix):
            continue
        if name.endswith((".partial.tif", ".partial.json", ".probe", ".backup")):
            stale.append(os.path.join(directory, name))
    return stale


class ProcessingWorker(threading.Thread):
    """Process a stack and publish it only after complete validation."""

    def __init__(
        self,
        file_path: str,
        save_path: str,
        params: dict[str, Any],
        progress_queue: queue.Queue,
        cancel_event: threading.Event,
        global_range=None,
        range_percentiles=None,
        expected_signature=None,
    ):
        super().__init__(daemon=False, name="ProcessingWorker")
        self.file_path = os.path.abspath(os.fspath(file_path))
        self.save_path = os.path.abspath(os.fspath(save_path))
        self.params = dict(params)
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.global_range = global_range
        self.range_percentiles = range_percentiles
        self.expected_signature = expected_signature
        self._reader = None
        self._writer = None
        self._temporary_output = None
        self._temporary_sidecar = None
        self._completed_frames = 0

    def run(self):
        try:
            self._process()
        except OperationCancelled as exc:
            logger.info("处理已安全取消: %s", exc)
            self._send(
                "cancelled",
                f"处理已取消；原输出未被改动。"
                f"\n已完成 {self._completed_frames} 帧临时计算。",
            )
        except Exception as exc:
            logger.exception("处理失败")
            self._send(
                "error",
                f"处理失败，未发布不完整输出：\n{exc}",
            )
        finally:
            self._close_resources()
            self._remove_temporary_files()
            clear_filter_caches()
            gc.collect()

    def _validate_paths(self):
        if paths_refer_to_same_file(self.file_path, self.save_path):
            raise InputValidationError("输入文件和输出文件不能是同一路径")
        if Path(self.save_path).suffix.lower() not in (".tif", ".tiff"):
            raise InputValidationError("输出文件扩展名必须是 .tif 或 .tiff")
        output_directory = os.path.dirname(self.save_path)
        if not os.path.isdir(output_directory):
            raise InputValidationError(f"输出目录不存在: {output_directory}")
        if os.path.isdir(self.save_path):
            raise InputValidationError(f"输出路径是目录而不是文件: {self.save_path}")
        sidecar_path = self.save_path + ".stem.json"
        if os.path.isdir(sidecar_path):
            raise InputValidationError(f"溯源路径是目录而不是文件: {sidecar_path}")
        # Windows 上 os.access(W_OK) 对目录几乎恒真；用真实创建/删除探测
        # 才能提前发现只读目录，避免事务中途以泛化的权限错误失败。
        try:
            descriptor, probe_path = tempfile.mkstemp(
                dir=output_directory,
                prefix=f".{os.path.basename(self.save_path)}.",
                suffix=".probe",
            )
        except OSError as exc:
            raise InputValidationError(
                f"输出目录不可写: {output_directory}（{exc}）"
            ) from exc
        with contextlib.suppress(OSError):
            os.close(descriptor)
            os.unlink(probe_path)

    def _check_cancelled(self, message="用户已取消处理"):
        if self.cancel_event.is_set():
            raise OperationCancelled(message)

    def _range_percentiles(self) -> tuple[float, float]:
        """Return the validated (low, high) percentiles for range analysis."""
        values = self.range_percentiles or (
            DEFAULT_LOW_PERCENTILE,
            DEFAULT_HIGH_PERCENTILE,
        )
        try:
            low, high = (float(values[0]), float(values[1]))
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("范围百分位必须是 (low, high) 数值二元组") from exc
        if not (np.isfinite(low) and np.isfinite(high)):
            raise ValueError("范围百分位必须是有限数值")
        if not (0.0 <= low < high <= 100.0):
            raise ValueError("范围百分位无效：需要 0 ≤ 低 < 高 ≤ 100")
        return low, high

    def _process(self):
        started_monotonic = time.monotonic()
        started_utc = datetime.now(timezone.utc)
        self._validate_paths()
        # GUI 在确认对话框停留期间输入可能被外部修改；对照打开时采集的
        # 签名可保证导出与用户预览过的数据一致（worker 自身的签名/哈希
        # 只能保证事务内部自洽）。
        if (
            self.expected_signature is not None
            and tuple(self.expected_signature) != file_signature(self.file_path)
        ):
            raise InputValidationError(
                "输入文件自上次预览后已发生变化；请重新打开文件后重试"
            )
        params = validate_params(self.params)
        low_percentile, high_percentile = self._range_percentiles()
        self._check_cancelled()

        initial_signature = file_signature(self.file_path)
        self._reader = TiffStackReader(self.file_path)
        reader = self._reader
        frame_count = reader.num_frames
        if file_signature(self.file_path) != initial_signature:
            raise InputValidationError("输入文件在打开期间发生变化，请重新打开后重试")

        estimate = estimate_peak_memory_bytes(reader.shape)
        available = available_memory_bytes()
        if available is not None and estimate > int(available * 0.85):
            raise InputValidationError(
                "预计单帧处理内存约 "
                f"{estimate / 1024**3:.2f} GiB，当前可用内存约 "
                f"{available / 1024**3:.2f} GiB；为防止系统失去响应，"
                "本次任务已拒绝。"
            )
        if available is not None and estimate > int(available * 0.55):
            self._send(
                "warning",
                f"该图像预计占用较高内存：约 {estimate / 1024**3:.2f} GiB。",
            )

        output_dtype = np.uint8 if params["output_bit_depth"] == 8 else np.uint16
        estimated_output = estimate_output_bytes(reader.stack_shape, output_dtype)
        try:
            free_disk = shutil.disk_usage(os.path.dirname(self.save_path)).free
        except OSError as exc:
            logger.warning("无法查询输出磁盘空间: %s", exc)
        else:
            if estimated_output > int(free_disk * 0.90):
                raise InputValidationError(
                    "预计输出至少需要约 "
                    f"{estimated_output / 1024**3:.2f} GiB，目标磁盘仅剩 "
                    f"{free_disk / 1024**3:.2f} GiB；任务已在写入前拒绝。"
                )
            if estimated_output > int(free_disk * 0.50):
                self._send(
                    "warning",
                    "目标磁盘余量偏低：预计输出约 "
                    f"{estimated_output / 1024**3:.2f} GiB，可用 "
                    f"{free_disk / 1024**3:.2f} GiB。",
                )

        self._send("info", "正在计算处理前输入文件 SHA-256…")
        input_hash = _sha256(self.file_path, self.cancel_event)
        if file_signature(self.file_path) != initial_signature:
            raise InputValidationError(
                "输入文件在初始校验期间发生变化，请重新打开后重试"
            )

        if self.global_range is None:
            self._send("info", f"正在分析全部 {frame_count} 帧的强度范围…")

            def analysis_progress(current, total):
                self._send(
                    "analysis_progress",
                    int(current / total * 100),
                    current,
                    total,
                )

            global_range = estimate_global_range(
                reader,
                low_percentile=low_percentile,
                high_percentile=high_percentile,
                cancel_event=self.cancel_event,
                progress_callback=analysis_progress,
            )
        else:
            global_range = (float(self.global_range[0]), float(self.global_range[1]))
            if not np.isfinite(global_range).all() or global_range[1] < global_range[0]:
                raise ValueError("传入的全局强度范围无效")

        self._temporary_output = _temporary_path(self.save_path, ".partial.tif")
        self._temporary_sidecar = _temporary_path(self.save_path, ".partial.json")

        totals = {"pixels": 0, "clipped_low": 0, "clipped_high": 0}

        def processed_frames():
            for index in range(frame_count):
                self._check_cancelled()
                try:
                    frame = reader.read_frame(index)
                    output, stats = process_frame(frame, params, global_range)
                except OperationCancelled:
                    raise
                except Exception as exc:
                    raise RuntimeError(f"帧 {index} 处理失败: {exc}") from exc
                for key in totals:
                    totals[key] += int(stats[key])
                self._completed_frames = index + 1
                self._send(
                    "progress",
                    int((index + 1) / frame_count * 100),
                    index + 1,
                    frame_count,
                )
                yield output

        self._send(
            "info",
            f"开始处理 {frame_count} 帧；输出 "
            f"{params['output_bit_depth']}-bit；"
            f"范围 [{global_range[0]:.3f}, {global_range[1]:.3f}]",
        )
        self._writer = TiffStackWriter(
            self._temporary_output,
            stack_shape=reader.stack_shape,
            dtype=output_dtype,
            metadata=reader.metadata,
        )
        self._writer.open()
        self._writer.write_stack(processed_frames())
        self._writer.close()
        self._writer = None
        self._check_cancelled()

        output_info = validate_output_file(
            self._temporary_output,
            expected_frames=frame_count,
            expected_frame_shape=reader.shape,
            expected_stack_shape=reader.stack_shape,
            expected_dtype=output_dtype,
            expected_axes=reader.axes,
            expect_ome=reader.is_ome,
        )
        self._send("info", "正在确认输入文件未发生变化…")
        final_input_hash = _sha256(self.file_path, self.cancel_event)
        if (
            final_input_hash != input_hash
            or file_signature(self.file_path) != initial_signature
        ):
            raise InputValidationError(
                "输入文件在处理期间发生变化；临时结果已丢弃，请重新打开后重试"
            )
        output_hash = _sha256(self._temporary_output, self.cancel_event)
        _fsync_file(self._temporary_output)
        clipped_low_fraction = (
            totals["clipped_low"] / totals["pixels"] if totals["pixels"] else 0.0
        )
        clipped_high_fraction = (
            totals["clipped_high"] / totals["pixels"] if totals["pixels"] else 0.0
        )

        provenance = {
            "schema": "stem-enhancer-provenance/v1",
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "application": {
                "name": APP_NAME,
                "version": APP_VERSION,
                "algorithm": ALGORITHM_VERSION,
                "libraries": {
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "opencv": cv2.__version__,
                    "tifffile": tifffile.__version__,
                },
            },
            "input": {
                "name": os.path.basename(self.file_path),
                "size_bytes": initial_signature[2],
                "sha256": input_hash,
                "frame_count": frame_count,
                "frame_shape": list(reader.shape),
                "stack_shape": list(reader.stack_shape),
                "axes": reader.axes,
                "dtype": str(reader.dtype),
                "is_ome": reader.is_ome,
                "is_bigtiff": reader.is_bigtiff,
                "calibration": {
                    "x_resolution": reader.metadata.get("x_resolution"),
                    "y_resolution": reader.metadata.get("y_resolution"),
                    "resolution_unit": reader.metadata.get("resolution_unit"),
                    "ome_physical": reader.metadata.get("ome_physical", {}),
                },
            },
            "processing": {
                "parameters": {key: params[key] for key in DEFAULT_PARAMS},
                "clahe_tile_grid": list(CLAHE_TILE_GRID),
                "started_utc": started_utc.isoformat(),
                "duration_seconds": round(
                    time.monotonic() - started_monotonic, 3
                ),
                "global_range": {
                    "low_percentile": low_percentile,
                    "high_percentile": high_percentile,
                    "vmin": global_range[0],
                    "vmax": global_range[1],
                },
                "clipping": {
                    "low_fraction": clipped_low_fraction,
                    "high_fraction": clipped_high_fraction,
                },
            },
            "output": {
                "name": os.path.basename(self.save_path),
                "sha256": output_hash,
                "frame_count": output_info["num_frames"],
                "frame_shape": list(output_info["frame_shape"]),
                "stack_shape": list(output_info["shape"]),
                "axes": output_info["axes"],
                "dtype": output_info["dtype"],
                "is_ome": output_info["is_ome"],
                "is_bigtiff": output_info["is_bigtiff"],
                "size_bytes": output_info["file_size"],
            },
        }
        with open(
            self._temporary_sidecar, "w", encoding="utf-8", newline="\n"
        ) as stream:
            json.dump(provenance, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        self._check_cancelled()
        sidecar_path = self.save_path + ".stem.json"
        self._publish_output_pair(sidecar_path)

        clipping = clipped_low_fraction + clipped_high_fraction
        if clipping > 0.01:
            self._send(
                "warning",
                f"有 {clipping:.2%} 像素在输出映射时被裁剪；请检查强度范围和溯源文件。",
            )

        summary = (
            "处理完成并通过输出校验。\n"
            f"总帧数: {frame_count}\n"
            f"输出: {save_path_summary(self.save_path)}\n"
            f"位深: {params['output_bit_depth']}-bit\n"
            f"BigTIFF: {'是' if output_info['is_bigtiff'] else '否'}\n"
            f"裁剪比例: 低端 {clipped_low_fraction:.3%}，"
            f"高端 {clipped_high_fraction:.3%}\n"
            f"溯源: {os.path.basename(sidecar_path)}"
        )
        self._send("done", summary)

    def _publish_output_pair(self, sidecar_path: str):
        """Publish the validated TIFF atomically, then the provenance sidecar.

        The TIFF is published with a single atomic rename: any existing output
        is replaced only at the instant the fully validated new file takes its
        place, so a crash can never leave a truncated or missing target file.
        A failed rename leaves the previous output untouched.

        The sidecar is derived metadata published with the same atomic replace
        plus a few retries.  If every attempt fails, the previous sidecar (if
        any) is untouched and the raised error states that the new TIFF is
        already in place, so the mismatch is reported instead of silently
        rolling back a large validated file.
        """
        try:
            os.replace(self._temporary_output, self.save_path)
        except Exception as exc:
            raise RuntimeError(f"发布输出文件失败；旧输出未被改动: {exc}") from exc
        self._temporary_output = None

        last_error = None
        for attempt in range(3):
            try:
                os.replace(self._temporary_sidecar, sidecar_path)
                self._temporary_sidecar = None
                break
            except OSError as exc:
                last_error = exc
                time.sleep(0.2 * (attempt + 1))
        else:
            raise RuntimeError(
                "新输出文件已发布，但溯源 sidecar 发布失败；"
                "已有 sidecar（如存在）描述的是上一版输出，"
                f"请勿作为新输出的依据: {last_error}"
            ) from last_error

        _fsync_directory(os.path.dirname(self.save_path))

    def _close_resources(self):
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                logger.exception("关闭临时 TIFF 写入器失败")
            self._writer = None
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                logger.exception("关闭 TIFF 读取器失败")
            self._reader = None

    def _remove_temporary_files(self):
        for path in (self._temporary_output, self._temporary_sidecar):
            if path and os.path.isfile(path):
                try:
                    os.unlink(path)
                except OSError:
                    logger.exception("无法删除临时文件: %s", path)
        self._temporary_output = None
        self._temporary_sidecar = None

    def _send(self, message_type: str, *args):
        self.progress_queue.put((message_type, *args))


def save_path_summary(path: str) -> str:
    directory = os.path.dirname(path)
    basename = os.path.basename(path)
    if len(directory) > 40:
        directory = "…" + directory[-37:]
    return os.path.join(directory, basename)
