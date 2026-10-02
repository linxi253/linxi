from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .models import VideoInfo


MINIMUM_SAFE_VERSION = (8, 0, 3)


class FFmpegError(RuntimeError):
    pass


@dataclass(frozen=True)
class FFmpegPaths:
    ffmpeg: Path
    ffprobe: Path
    version: tuple[int, int, int]


def _startup_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kwargs


def _parse_rate_fraction(value: str | None) -> tuple[int, int] | None:
    """Parse an FFmpeg rate string into an exact (numerator, denominator) pair."""
    if not value or value == "0/0":
        return None
    try:
        if "/" in value:
            numerator_text, denominator_text = value.split("/", 1)
        else:
            numerator_text, denominator_text = value, "1"
        numerator = int(numerator_text)
        denominator = int(denominator_text)
        if denominator == 0:
            return None
        return numerator, denominator
    except (ValueError, ZeroDivisionError):
        return None


def parse_rate(value: str | None) -> float | None:
    fraction = _parse_rate_fraction(value)
    if fraction is None:
        return None
    return fraction[0] / fraction[1]


# NTSC/PAL 等常见时间基：ffprobe 对同一恒定帧率视频可能把 avg_frame_rate
# 报成 30000/1001 而把 r_frame_rate 报成 30/1，这类“白名单”差异不表示 VFR。
_RATE_WHITELIST_PAIRS = {
    (30000, 1001): (30, 1),
    (24000, 1001): (24, 1),
    (60000, 1001): (60, 1),
    (48000, 1001): (48, 1),
}


def _whitelisted_rate_pair(fraction: tuple[int, int]) -> tuple[int, int]:
    return _RATE_WHITELIST_PAIRS.get(fraction, fraction)


def is_variable_fps(avg_frame_rate: str | None, nominal_frame_rate: str | None) -> bool:
    """Return True only when avg and nominal frame rates are genuinely different.

    Pure and testable. The previous implementation compared floating point values
    and treated the common NTSC pair 30000/1001 vs 30/1 as VFR.  We compare exact
    rational pairs first and normalize the well-known drop-frame/non-drop-frame
    timebase pairs before falling back to an absolute float tolerance.
    """
    avg_fraction = _parse_rate_fraction(avg_frame_rate)
    nom_fraction = _parse_rate_fraction(nominal_frame_rate)
    if avg_fraction is None or nom_fraction is None:
        return False
    if _whitelisted_rate_pair(avg_fraction) == _whitelisted_rate_pair(nom_fraction):
        return False
    avg = avg_fraction[0] / avg_fraction[1]
    nom = nom_fraction[0] / nom_fraction[1]
    return abs(avg - nom) > 0.001


class FFmpegManager:
    """Resolve one verified FFmpeg/ffprobe pair and use it everywhere."""

    def __init__(self, custom_path: str | Path | None = None, allow_unsafe: bool = False):
        self.custom_path = Path(custom_path).expanduser() if custom_path else None
        self.allow_unsafe = allow_unsafe

    @staticmethod
    def _bundled_candidates() -> list[Path]:
        if getattr(sys, "frozen", False):
            root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        else:
            root = Path(__file__).resolve().parents[1]
        return [
            root / "ffmpeg.exe",
            root / "tools" / "ffmpeg" / "ffmpeg.exe",
            root / "tools" / "ffmpeg" / "ffmpeg",
        ]

    def _candidates(self) -> list[Path]:
        candidates: list[Path] = []
        if self.custom_path:
            candidates.append(self.custom_path)
        candidates.extend(path for path in self._bundled_candidates() if path.is_file())
        system = shutil.which("ffmpeg")
        if system:
            candidates.append(Path(system))
        try:
            import imageio_ffmpeg  # type: ignore

            candidates.append(Path(imageio_ffmpeg.get_ffmpeg_exe()))
        except Exception:
            pass
        result: list[Path] = []
        seen: set[str] = set()
        for path in candidates:
            key = str(path).casefold()
            if key not in seen:
                seen.add(key)
                result.append(path)
        return result

    @staticmethod
    def _version(path: Path) -> tuple[int, int, int]:
        try:
            completed = subprocess.run(
                [str(path), "-version"],
                capture_output=True,
                timeout=8,
                check=False,
                **_startup_kwargs(),
            )
        except OSError as error:
            raise FFmpegError(f"无法启动 FFmpeg: {error}") from error
        except subprocess.TimeoutExpired as error:
            raise FFmpegError(f"FFmpeg 版本检测超时: {path}") from error
        stdout = completed.stdout.decode("utf-8", errors="replace")
        stderr = completed.stderr.decode("utf-8", errors="replace")
        if completed.returncode != 0:
            raise FFmpegError(f"FFmpeg 检测失败: {stderr.strip()}")
        match = re.search(
            r"ff(?:mpeg|probe) version\s+n?(\d+)\.(\d+)(?:\.(\d+))?", stdout, re.I
        )
        if not match:
            raise FFmpegError("无法识别 FFmpeg 版本")
        return (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))

    def resolve(self) -> FFmpegPaths:
        failures: list[str] = []
        for ffmpeg in self._candidates():
            try:
                version = self._version(ffmpeg)
                if version < MINIMUM_SAFE_VERSION and not self.allow_unsafe:
                    failures.append(f"{ffmpeg} 版本 {version} 低于安全下限 {MINIMUM_SAFE_VERSION}")
                    continue
                suffix = ".exe" if ffmpeg.suffix.lower() == ".exe" else ""
                ffprobe = ffmpeg.with_name(f"ffprobe{suffix}")
                if not ffprobe.is_file():
                    failures.append(f"{ffmpeg} 缺少同目录 ffprobe")
                    continue
                # ffprobe 必须与 ffmpeg 版本完全一致，保证“同一验证对”不被混装破坏
                probe_version = self._version(ffprobe)
                if probe_version != version:
                    failures.append(
                        f"{ffprobe} 版本 {probe_version} 与 ffmpeg 版本 {version} 不一致，拒绝混装组合"
                    )
                    continue
                return FFmpegPaths(ffmpeg=ffmpeg.resolve(), ffprobe=ffprobe.resolve(), version=version)
            except FFmpegError as error:
                failures.append(str(error))
        detail = "；".join(failures) or "未找到可执行文件"
        raise FFmpegError(
            "未找到可用的安全 FFmpeg。请在 tools/ffmpeg 放置 FFmpeg 8.0.3+，或在界面指定路径。" + detail
        )


def _parse_duration(value: object) -> float | None:
    if value is None:
        return None
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    return duration if duration > 0 else None


# 视频编码的组件位深只可能是这几档；声明为其他值（如 rgb24 的
# bits_per_sample=24 是打包总位深而非组件位深）时不可信，须回退到
# 像素格式名称推断，避免 8 位源被误判为高位深。
_VALID_COMPONENT_DEPTHS = {8, 9, 10, 12, 14, 16}

# 容器/编码器不报告位深字段时的像素格式名称推断表。
# p0xx/p2xx/p4xx 系列是 10/12/16 位 4:2:0/4:2:2/4:4:4 变体，名称无数字后缀可解析。
_PIX_FMT_EXPLICIT_DEPTH = {
    "p010le": 10, "p010be": 10,
    "p016le": 16, "p016be": 16,
    "p210le": 10, "p210be": 10,
    "p212le": 12, "p212be": 12,
    "p410le": 10, "p410be": 10,
    "p412le": 12, "p412be": 12,
}
# 名称尾部数字后缀（yuv420p10le→10、gray16le→16）；rgb48le 的 48 是
# 三通道合计，等效每通道 16 位。
_PIX_FMT_SUFFIX_DEPTH = {"9": 9, "10": 10, "12": 12, "14": 14, "16": 16, "48": 16}


def _resolve_bits_per_sample(stream: dict) -> tuple[int, str]:
    """返回 (有效组件位深, 来源)。

    来源为 declared（容器/编码器显式声明且取值可信）、
    pix_fmt（字段缺失、为 0 或取值不可信时按像素格式名称推断）、
    default（两者都不可得，按 8 位处理）。
    """
    for key in ("bits_per_raw_sample", "bits_per_sample"):
        raw = stream.get(key)
        if raw in (None, "", "N/A"):
            continue
        try:
            bits = int(raw)
        except (TypeError, ValueError):
            continue
        if bits in _VALID_COMPONENT_DEPTHS:
            return bits, "declared"
    pix_fmt = stream.get("pix_fmt")
    if isinstance(pix_fmt, str) and pix_fmt:
        lowered = pix_fmt.lower()
        explicit = _PIX_FMT_EXPLICIT_DEPTH.get(lowered)
        if explicit is not None:
            return explicit, "pix_fmt"
        match = re.search(r"(\d+)(?:le|be)$", lowered)
        if match and match.group(1) in _PIX_FMT_SUFFIX_DEPTH:
            return _PIX_FMT_SUFFIX_DEPTH[match.group(1)], "pix_fmt"
    return 8, "default"


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    try:
        process.terminate()
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _parse_rotation(stream: dict) -> float | None:
    """从容器的 display matrix 侧数据解析声明旋转角（度）。"""
    side_data = stream.get("side_data_list")
    if not isinstance(side_data, list):
        return None
    for entry in side_data:
        if not isinstance(entry, dict) or "rotation" not in entry:
            continue
        try:
            value = float(entry["rotation"])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def probe_video(
    ffprobe: Path,
    path: Path,
    *,
    on_started: Callable[[subprocess.Popen[bytes]], None] | None = None,
) -> VideoInfo:
    command = [
        str(ffprobe), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
    ]
    try:
        # Keep both pipes in binary mode. ffprobe emits JSON as UTF-8, while
        # text=True would decode with the Windows ANSI code page (often GBK).
        # A non-ASCII filename can then crash subprocess's reader thread and
        # make communicate() return None instead of the JSON payload.
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **_startup_kwargs(),
        )
    except OSError as error:
        raise FFmpegError(f"无法启动 ffprobe: {error}") from error
    try:
        if on_started:
            on_started(process)
    except Exception:
        _stop_process(process)
        raise
    try:
        stdout, stderr = process.communicate(timeout=60)
    except subprocess.TimeoutExpired as error:
        _stop_process(process)
        raise FFmpegError(f"ffprobe 读取超时（60 秒）: {path.name}") from error
    except OSError as error:
        _stop_process(process)
        raise FFmpegError(f"ffprobe 读取失败: {path.name}: {error}") from error
    finally:
        if process.poll() is None:
            _stop_process(process)
    stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
    if process.returncode != 0:
        raise FFmpegError(stderr_text.strip() or f"ffprobe 无法读取 {path.name}")
    if stdout is None or not stdout.strip():
        raise FFmpegError(f"ffprobe 未返回探测数据: {path.name}")
    try:
        stdout_text = stdout.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise FFmpegError(f"ffprobe 返回的数据不是有效 UTF-8: {path.name}") from error
    try:
        payload = json.loads(stdout_text)
    except json.JSONDecodeError as error:
        raise FFmpegError(f"ffprobe 返回的 JSON 无效: {path.name}") from error
    if not isinstance(payload, dict):
        raise FFmpegError(f"ffprobe 返回的 JSON 结构无效: {path.name}")

    # 跳过封面图流（attached_pic）：MKV/MP4 的内嵌封面也是 video 流，
    # 若被选中会把一张静态图当作视频处理。
    stream_items = payload.get("streams", [])
    if not isinstance(stream_items, list):
        raise FFmpegError(f"ffprobe 返回的视频流结构无效: {path.name}")
    # 无条件收集全部可用流（只剔除封面图流），让 stream_count 反映真实
    # 可用视频流总数供多流告警使用；解码仍固定用首选流（第一个非封面流）。
    streams = []
    video_ordinal = 0
    chosen_ordinal: int | None = None
    for item in stream_items:
        if not isinstance(item, dict) or item.get("codec_type") != "video":
            continue
        is_attached = False
        disposition = item.get("disposition")
        if isinstance(disposition, dict) and disposition.get("attached_pic"):
            is_attached = True
        if not is_attached:
            streams.append(item)
            if chosen_ordinal is None:
                chosen_ordinal = video_ordinal
        video_ordinal += 1
    if chosen_ordinal is None or not streams:
        raise FFmpegError(f"未找到可用的视频流: {path.name}")
    stream = streams[0]
    try:
        width, height = int(stream["width"]), int(stream["height"])
    except (KeyError, ValueError, TypeError) as error:
        raise FFmpegError(f"视频分辨率无效: {path.name}") from error
    if width <= 0 or height <= 0:
        # 损坏/畸形容器可能报 0 分辨率；bytes_per_frame 会随之变 0，
        # 下游 rawvideo 读取将永远拿不到帧，尽早拒绝
        raise FFmpegError(f"视频分辨率非正值（{width}x{height}）: {path.name}")
    try:
        stream_index = int(stream.get("index", 0))
    except (TypeError, ValueError):
        stream_index = 0
    average_fps = parse_rate(stream.get("avg_frame_rate"))
    nominal_fps = parse_rate(stream.get("r_frame_rate"))
    duration = _parse_duration(stream.get("duration"))
    format_start_time: float | None = None
    if duration is None:
        format_info = payload.get("format")
        if isinstance(format_info, dict):
            duration = _parse_duration(format_info.get("duration"))
    format_info = payload.get("format")
    if isinstance(format_info, dict):
        try:
            value = float(format_info.get("start_time"))
        except (TypeError, ValueError):
            value = None
        if value is not None and math.isfinite(value) and value >= 0:
            format_start_time = value
    try:
        frame_count = int(stream["nb_frames"]) if stream.get("nb_frames") not in (None, "N/A") else None
    except (TypeError, ValueError):
        frame_count = None
    bits, bits_source = _resolve_bits_per_sample(stream)
    pix_fmt = stream.get("pix_fmt")
    color_family = "gray" if pix_fmt and "gray" in pix_fmt.lower() else "rgb"
    variable = is_variable_fps(stream.get("avg_frame_rate"), stream.get("r_frame_rate"))
    return VideoInfo(
        path=path,
        width=width,
        height=height,
        duration_s=duration,
        average_fps=average_fps,
        nominal_fps=nominal_fps,
        frame_count=frame_count,
        pixel_format=pix_fmt,
        bits_per_sample=bits,
        bits_source=bits_source,
        color_family=color_family,
        codec=stream.get("codec_name"),
        is_variable_fps=variable,
        color_space=stream.get("color_space"),
        color_range=stream.get("color_range"),
        stream_index=stream_index,
        stream_count=len(streams),
        video_stream_ordinal=chosen_ordinal,
        rotation_degrees=_parse_rotation(stream),
        format_start_time=format_start_time,
    )


def probe_first_frame_pts(
    ffprobe: Path,
    path: Path,
    video_stream_ordinal: int = 0,
    *,
    timeout: float = 30.0,
) -> float | None:
    """读取选中视频流第一帧的呈现时间戳（秒）。

    只解码一帧，开销可忽略。用途：
    1. fps 滤镜的 start_time —— 视频流晚于容器起始（如带前置音轨）时，
       start_time=0 会在首帧之前复制填充帧，制造不存在的观测；
    2. CSV 记录流起始时间，保持时间轴可回溯。
    任何失败都返回 None，由调用方回退到 start_time=0 的旧行为。
    """
    command = [
        str(ffprobe), "-v", "error",
        "-select_streams", f"v:{video_stream_ordinal}",
        "-read_intervals", "%+#1",
        "-show_entries", "frame=pts_time",
        "-of", "csv=p=0",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, timeout=timeout, check=False,
            **_startup_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    text = completed.stdout.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    first_line = text.splitlines()[0].strip()
    try:
        value = float(first_line)
    except ValueError:
        return None
    return value if math.isfinite(value) else None
