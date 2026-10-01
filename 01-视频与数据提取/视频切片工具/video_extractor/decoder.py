from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .ffmpeg import FFmpegError, _startup_kwargs, _stop_process
from .models import BitDepth, ColorMode, ExtractOptions, FrameSpec, SamplingMode, VideoInfo
from .sampling import ffmpeg_filter


# FFmpeg scale 滤镜的数字常量（in/out_color_matrix、in/out_range）。
# ffprobe 返回的 color_space 是符号名，但 scale 滤镜的选项解析器只认
# 部分名称；使用 libswscale 的数值常量最稳。
COLOR_MATRIX_VALUES = {
    "bt709": 1,
    "bt601": 5,
    "bt470bg": 5,
    "smpte170m": 5,
    "smpte240m": 7,
    "bt2020nc": 9,
    "bt2020c": 9,
    "fcc": 4,
}
COLOR_RANGE_VALUES = {
    "tv": 1,
    "limited": 1,
    "mpeg": 1,
    "pc": 2,
    "jpeg": 2,
    "full": 2,
}

# showinfo 日志行的 pts_time 字段；FFmpeg 用 %.6g 输出，可能出现 1e-05 之类科学计数
_PTS_TIME_PATTERN = re.compile(rb"pts_time:([0-9eE+\-.]+)")

# 解码停滞看门狗：超过该时长没有任何帧/进度输出即终止子进程，
# 防止损坏输入或底层 I/O 停滞把任务永久挂在阻塞的管道读取上
STALL_TIMEOUT_SECONDS = 120.0

# stderr 原始字节只保留尾部用于诊断：损坏输入可能产生海量日志，不能无限累积
STDERR_TAIL_BYTES = 64 * 1024


def grayscale_color_filter(color_space: str | None, color_range: str | None) -> str | None:
    """Build a scale filter that declares the YUV matrix/range before format=gray.

    YUV -> gray 最终取的是 Y' 亮度平面，矩阵只影响 YUV<->RGB 转换；
    显式声明可避免 FFmpeg/swscale 对 HD/SD 源使用默认 BT.601 导致潜在偏差。
    无法识别或缺失时返回 None，由调用方回退到默认 format=gray 行为。
    """
    options: list[str] = []
    matrix = COLOR_MATRIX_VALUES.get(color_space or "")
    if matrix is not None:
        options.append(f"in_color_matrix={matrix}:out_color_matrix={matrix}")
    range_value = COLOR_RANGE_VALUES.get(color_range or "")
    if range_value is not None:
        options.append(f"in_range={range_value}:out_range={range_value}")
    if not options:
        return None
    return "scale=" + ":".join(options)


def _is_yuv_source(pix_fmt: str | None) -> bool:
    """判断源像素格式是否属于 YUV 家族（矩阵/范围声明只对 YUV 输入有意义）。"""
    if not pix_fmt:
        return False
    lowered = pix_fmt.lower()
    return not any(token in lowered for token in ("gray", "rgb", "bgr", "gbr"))


def rgb_color_filter(color_space: str | None, color_range: str | None) -> str | None:
    """Build a scale filter declaring the YUV matrix/range before YUV->RGB conversion.

    YUV -> RGB 的像素值取决于源 YUV 按哪个矩阵/范围解释；不声明时 swscale
    按分辨率启发式选择矩阵，BT.709 高清源会被错误换算产生系统性色度偏差。
    只声明输入侧（输出是 RGB，无矩阵概念）；无可识别元数据时返回 None，
    由调用方回退到输出端 -pix_fmt 的默认转换行为。
    """
    options: list[str] = []
    matrix = COLOR_MATRIX_VALUES.get(color_space or "")
    if matrix is not None:
        options.append(f"in_color_matrix={matrix}")
    range_value = COLOR_RANGE_VALUES.get(color_range or "")
    if range_value is not None:
        options.append(f"in_range={range_value}")
    if not options:
        return None
    return "scale=" + ":".join(options)


def resolve_frame_spec(info: VideoInfo, options: ExtractOptions) -> FrameSpec:
    if options.color_mode is ColorMode.PRESERVE:
        color = info.color_family
    else:
        color = "gray" if options.color_mode is ColorMode.GRAYSCALE else "rgb"
    if options.bit_depth is BitDepth.SOURCE:
        depth = 16 if info.bits_per_sample > 8 else 8
    else:
        depth = int(options.bit_depth.value)
        # 拒绝 8 位源放大为 16 位：位复制不产生新信息，
        # 却让下游误以为数据具有 16 位精度
        if depth == 16 and info.bits_per_sample <= 8:
            raise ValueError(
                f"源视频为 {info.bits_per_sample} 位，选择 16 位输出只会做位复制放大，"
                "不会产生新信息；请将位深设为 source 或 8"
            )
    if color == "rgb" and depth == 16:
        # ImageJ 格式的彩色 TIFF 仅支持 8 位/通道
        raise ValueError(
            "ImageJ 格式的彩色 TIFF 仅支持 8 位；请将位深设为 8，"
            "或改用灰度模式以保留 16 位精度"
        )
    if color == "gray":
        return FrameSpec(
            width=info.width,
            height=info.height,
            channels=1,
            dtype="uint16" if depth == 16 else "uint8",
            ffmpeg_pixel_format="gray16le" if depth == 16 else "gray",
            color_family="gray",
            stream_index=info.stream_index,
            color_space=info.color_space,
            color_range=info.color_range,
            source_pixel_format=info.pixel_format,
        )
    return FrameSpec(
        width=info.width,
        height=info.height,
        channels=3,
        dtype="uint16" if depth == 16 else "uint8",
        ffmpeg_pixel_format="rgb48le" if depth == 16 else "rgb24",
        color_family="rgb",
        stream_index=info.stream_index,
        color_space=info.color_space,
        color_range=info.color_range,
        source_pixel_format=info.pixel_format,
    )


@dataclass(frozen=True)
class FrameCount:
    """count pass 结果：输出帧数 + showinfo 捕获的源帧呈现时间戳（秒，升序）。"""

    frames: int
    pts_seconds: list[float] = field(default_factory=list)


class _BoundedStderr:
    """按行重组 stderr：完整行交给回调（showinfo PTS 解析），原始字节只保留尾部。"""

    def __init__(
        self,
        on_line: Callable[[bytes], None] | None = None,
        tail_bytes: int = STDERR_TAIL_BYTES,
    ) -> None:
        self._on_line = on_line
        self._tail_bytes = tail_bytes
        self._buffer = bytearray()
        self._lines: list[bytes] = []
        self._total = 0

    def feed(self, chunk: bytes) -> None:
        self._buffer.extend(chunk)
        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                break
            line = bytes(self._buffer[:newline])
            del self._buffer[: newline + 1]
            self._lines.append(line)
            self._total += len(line)
            if self._on_line is not None:
                self._on_line(line)
            while self._lines and self._total > self._tail_bytes:
                dropped = self._lines.pop(0)
                self._total -= len(dropped)

    def tail_text(self) -> str:
        return b"".join(self._lines).decode("utf-8", errors="replace").strip()


class _StallWatchdog:
    """监控子进程活跃度；超过 timeout 无进展则终止进程，解除阻塞的管道读取。"""

    def __init__(self, process: subprocess.Popen[bytes], timeout: float = STALL_TIMEOUT_SECONDS):
        self._process = process
        self._timeout = timeout
        self._last = time.monotonic()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.stalled = False
        self._thread = threading.Thread(target=self._loop, name="ffmpeg-stall-watchdog", daemon=True)

    def mark_activity(self) -> None:
        with self._lock:
            self._last = time.monotonic()

    def _loop(self) -> None:
        while not self._stop_event.wait(5.0):
            with self._lock:
                idle = time.monotonic() - self._last
            if idle > self._timeout:
                self.stalled = True
                _stop_process(self._process)
                return

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)


class FrameDecoder:
    """FFmpeg rawvideo decoder with an explicit pixel format and cancellation.

    解码约定：
    - 输出保持编码方向的像素矩阵（-noautorotate），旋转元数据由上层记录并告警；
      这保证 rawvideo 字节流与 probe 报告的 width/height 严格一致。
    - 严格模式（默认）加 -xerror：FFmpeg 对解码错误默认容忍并跳帧且退出码
      仍为 0，会把"缺了若干帧"静默标记为成功；科研提取必须显式失败。
    - 解码链尾部固定 scale=W:H：个别编码流会中途改变分辨率，滤镜图重协商后
      rawvideo 字节流会按新尺寸输出，读取端按旧规格 reshape 即静默转置错位；
      尾部尺寸保险把任何此类帧拉回 probe 声明的规格。
    """

    def __init__(self, ffmpeg: Path):
        self.ffmpeg = ffmpeg

    @staticmethod
    def _decode_filters(options: ExtractOptions, spec: FrameSpec, start_time: float = 0.0) -> list[str]:
        """解码链：fps(可选) + 色彩换算 + 尾部尺寸保险。

        不变量：除 fps 外全部为帧保持滤镜（1 帧进 1 帧出），因此帧数只由
        fps 决定 —— _count_filters 只复刻 fps 即可保证两遍计数一致。
        该不变量由测试 test_decode_filters_are_frame_preserving 保护。
        """
        filters: list[str] = []
        sampling_filter = ffmpeg_filter(options.sampling, start_time)
        if sampling_filter:
            filters.append(sampling_filter)
        if spec.color_family == "gray":
            matrix_filter = grayscale_color_filter(spec.color_space, spec.color_range)
            if matrix_filter:
                filters.append(matrix_filter)
            filters.append("format=gray16le" if spec.dtype == "uint16" else "format=gray")
        elif spec.color_family == "rgb":
            # YUV 源转 RGB 时显式声明源矩阵/范围；RGB 源或元数据缺失时
            # 保持旧行为（输出端 -pix_fmt 自动转换）
            if _is_yuv_source(spec.source_pixel_format):
                matrix_filter = rgb_color_filter(spec.color_space, spec.color_range)
                if matrix_filter:
                    filters.append(matrix_filter)
                    filters.append(f"format={spec.ffmpeg_pixel_format}")
        filters.append(f"scale={spec.width}:{spec.height}")
        return filters

    @staticmethod
    def _count_filters(options: ExtractOptions, start_time: float = 0.0) -> list[str]:
        """计数/PTS 收集链：showinfo（捕获源帧 PTS）+ fps（可选）。

        相比解码链省去全部色彩换算——swscale 转换常占解码管线相当比例的
        CPU，而帧数只取决于 fps 滤镜。showinfo 位于 fps 之前，因此两种
        采样模式下捕获的都是"源帧"时间戳，可用于实测可变帧率检测。
        """
        filters: list[str] = ["showinfo"]
        sampling_filter = ffmpeg_filter(options.sampling, start_time)
        if sampling_filter:
            filters.append(sampling_filter)
        return filters

    def _input_args(
        self,
        path: Path,
        spec: FrameSpec,
        *,
        lenient: bool,
        loglevel: str = "error",
    ) -> list[str]:
        args = [str(self.ffmpeg), "-hide_banner", "-nostdin", "-noautorotate"]
        if not lenient:
            args.append("-xerror")
        args += [
            "-loglevel", loglevel,
            "-i", str(path),
            "-map", f"0:{spec.stream_index}", "-an", "-sn", "-dn",
        ]
        return args

    def _command(
        self,
        path: Path,
        options: ExtractOptions,
        spec: FrameSpec,
        *,
        start_time: float = 0.0,
        lenient: bool = False,
    ) -> list[str]:
        command = self._input_args(path, spec, lenient=lenient)
        filters = self._decode_filters(options, spec, start_time)
        if filters:
            command.extend(["-vf", ",".join(filters)])
        # 统一 passthrough（-vsync 0 已弃用）：滤镜链完全决定帧数。
        # rawvideo 输出端默认按 CFR 同步，会在 fps 滤镜锚定的首帧之前
        # 再补一层重复帧（外部审查 R7 的输出端机制）
        command.extend(["-fps_mode", "passthrough"])
        command.extend(["-f", "rawvideo", "-pix_fmt", spec.ffmpeg_pixel_format, "pipe:1"])
        return command

    def iter_frames(
        self,
        path: Path,
        options: ExtractOptions,
        spec: FrameSpec,
        cancelled: threading.Event,
        on_started: Callable[[subprocess.Popen[bytes]], None] | None = None,
        *,
        start_time: float = 0.0,
        lenient: bool = False,
    ) -> Iterator[np.ndarray]:
        command = self._command(path, options, spec, start_time=start_time, lenient=lenient)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **_startup_kwargs(),
        )
        collector = _BoundedStderr()
        watchdog = _StallWatchdog(process)
        stderr_thread: threading.Thread | None = None
        try:
            if on_started:
                on_started(process)

            def drain_stderr() -> None:
                assert process.stderr is not None
                while chunk := process.stderr.read(65536):
                    collector.feed(chunk)

            stderr_thread = threading.Thread(target=drain_stderr, name="ffmpeg-stderr", daemon=True)
            stderr_thread.start()
            watchdog.start()
            assert process.stdout is not None
            while True:
                if cancelled.is_set():
                    _stop_process(process)
                    return
                payload = process.stdout.read(spec.bytes_per_frame)
                watchdog.mark_activity()
                if not payload:
                    break
                if len(payload) != spec.bytes_per_frame:
                    raise FFmpegError("FFmpeg 输出了不完整的视频帧")
                # 不做 .copy()：数组只被读取后写入 memmap（赋值时拷贝），
                # 额外的逐帧深拷贝是纯浪费
                array = np.frombuffer(payload, dtype=np.dtype(spec.dtype)).reshape(spec.shape)
                yield array
            return_code = process.wait(timeout=30)
            stderr_thread.join(timeout=5)
            if cancelled.is_set():
                return
            if watchdog.stalled:
                raise FFmpegError(
                    f"FFmpeg 解码停滞：{STALL_TIMEOUT_SECONDS:g}s 内没有任何输出进展，已终止"
                )
            if return_code != 0:
                detail = collector.tail_text()
                raise FFmpegError(detail or "FFmpeg 解码失败")
        finally:
            watchdog.stop()
            if process.poll() is None:
                _stop_process(process)
            if stderr_thread is not None:
                stderr_thread.join(timeout=5)

    def count_frames(
        self,
        path: Path,
        options: ExtractOptions,
        spec: FrameSpec,
        cancelled: threading.Event,
        on_started: Callable[[subprocess.Popen[bytes]], None] | None = None,
        on_progress: Callable[[int], None] | None = None,
        *,
        start_time: float = 0.0,
        lenient: bool = False,
    ) -> FrameCount:
        """统计输出帧数并顺带收集源帧 PTS。

        与解码使用同一 fps 滤镜参数（start_time 必须一致），输出到 null
        muxer 并通过 -progress 读取帧计数，避免把 rawvideo 像素再传一遍。
        showinfo 滤镜把每帧 PTS 写到 stderr（info 级日志），按行解析，原始
        日志只在有界缓冲中保留尾部。
        """
        command = self._input_args(path, spec, lenient=lenient, loglevel="info")
        filters = self._count_filters(options, start_time)
        command.extend(["-vf", ",".join(filters)])
        # 与解码遍一致：显式 passthrough，帧数只由滤镜链决定
        command.extend(["-fps_mode", "passthrough"])
        command.extend(["-progress", "pipe:1", "-f", "null", os.devnull])
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **_startup_kwargs(),
        )
        pts_seconds: list[float] = []

        def parse_pts_line(line: bytes) -> None:
            match = _PTS_TIME_PATTERN.search(line)
            if match:
                try:
                    pts_seconds.append(float(match.group(1)))
                except ValueError:
                    pass

        collector = _BoundedStderr(on_line=parse_pts_line)
        watchdog = _StallWatchdog(process)
        stderr_thread: threading.Thread | None = None
        frames = 0
        try:
            if on_started:
                on_started(process)

            def drain_stderr() -> None:
                assert process.stderr is not None
                while chunk := process.stderr.read(65536):
                    collector.feed(chunk)

            stderr_thread = threading.Thread(target=drain_stderr, name="ffmpeg-count-stderr", daemon=True)
            stderr_thread.start()
            watchdog.start()
            assert process.stdout is not None
            for raw_line in process.stdout:
                watchdog.mark_activity()
                if cancelled.is_set():
                    _stop_process(process)
                    return FrameCount(0, [])
                line = raw_line.strip()
                if line.startswith(b"frame="):
                    try:
                        frames = int(line.split(b"=", 1)[1])
                        if on_progress:
                            on_progress(frames)
                    except ValueError:
                        pass
            return_code = process.wait(timeout=30)
            stderr_thread.join(timeout=5)
            if cancelled.is_set():
                return FrameCount(0, [])
            if watchdog.stalled:
                raise FFmpegError(
                    f"FFmpeg 帧统计停滞：{STALL_TIMEOUT_SECONDS:g}s 内没有任何输出进展，已终止"
                )
            if return_code != 0:
                detail = collector.tail_text()
                raise FFmpegError(detail or "FFmpeg 帧计数失败")
            return FrameCount(frames, pts_seconds)
        finally:
            watchdog.stop()
            if process.poll() is None:
                _stop_process(process)
            if stderr_thread is not None:
                stderr_thread.join(timeout=5)
