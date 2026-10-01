from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# 采样参数换算出的目标帧率合理上限：超过它几乎必然是单位错误（如把秒当毫秒填入间隔）
MAX_REASONABLE_FPS = 1_000_000.0


class SamplingMode(str, Enum):
    ALL = "all"
    TARGET_FPS = "target_fps"
    INTERVAL = "interval"


class ColorMode(str, Enum):
    PRESERVE = "preserve"
    GRAYSCALE = "grayscale"
    COLOR = "color"


class BitDepth(str, Enum):
    SOURCE = "source"
    UINT8 = "8"
    UINT16 = "16"


class JobState(str, Enum):
    PROBING = "probing"
    COUNTING = "counting"
    WRITING = "writing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class Sampling:
    mode: SamplingMode = SamplingMode.ALL
    value: float | None = None

    def validate(self) -> None:
        if self.mode is SamplingMode.ALL:
            if self.value is not None:
                raise ValueError("全部帧模式不接受采样值；如需采样请改用 target_fps 或 interval 模式")
            return
        if self.value is None:
            raise ValueError("目标帧率或时间间隔必须为正数")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError("目标帧率或时间间隔必须是数值")
        # NaN 的所有比较均为 False，inf 也大于 0，必须显式排除，
        # 否则会以 round(nan)/fps=fps=nan 等内部错误的形式延迟暴露
        if not math.isfinite(self.value) or self.value <= 0:
            raise ValueError("目标帧率或时间间隔必须是有限正数")
        # 派生值校验：interval 传入 1e-320 这类次正规数时 1/value 会溢出为
        # inf，直接进入 fps 滤镜参数；目标帧率过高同理是单位错误
        target_fps = self.target_fps
        assert target_fps is not None
        if not math.isfinite(target_fps):
            raise ValueError("采样参数换算出的目标帧率溢出（inf），请检查数值范围")
        if target_fps > MAX_REASONABLE_FPS:
            raise ValueError(
                f"目标帧率 {target_fps:g} 超过合理上限 {MAX_REASONABLE_FPS:g}，请检查单位"
            )

    @property
    def target_fps(self) -> float | None:
        if self.mode is SamplingMode.TARGET_FPS:
            return self.value
        if self.mode is SamplingMode.INTERVAL:
            return 1.0 / float(self.value)
        return None


@dataclass(frozen=True)
class ExtractOptions:
    sampling: Sampling = field(default_factory=Sampling)
    color_mode: ColorMode = ColorMode.PRESERVE
    # ImageJ 工作流的常见输出是 8 位；高位深源仍可选 source/16
    bit_depth: BitDepth = BitDepth.UINT8
    # 宽松解码：默认 False（严格模式，-xerror）。严格模式下任何解码错误
    # 都使任务失败，防止损坏帧被 FFmpeg 静默跳过后仍标记为成功；
    # 宽松模式允许抢救性提取，但结果可能缺帧且缺帧位置不可知
    lenient_decode: bool = False

    def validate(self) -> None:
        self.sampling.validate()


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    duration_s: float | None
    average_fps: float | None
    nominal_fps: float | None
    frame_count: int | None
    pixel_format: str | None
    bits_per_sample: int
    color_family: str
    codec: str | None
    is_variable_fps: bool
    # 色彩元数据：灰度换算依赖色彩矩阵，记录以便溯源
    color_space: str | None = None
    color_range: str | None = None
    # 位深来源：declared=容器/编码器显式声明；pix_fmt=字段缺失或不可信时
    # 按像素格式名称推断；default=两者都不可得。供 runner 决定是否发推断告警。
    bits_source: str = "declared"
    # 被选中的视频流全局索引，probe 与解码必须指向同一流
    stream_index: int = 0
    # 文件内可用视频流总数（不含封面图流），供多流告警使用
    stream_count: int = 1
    # 选中流在全部视频流（含封面图流）中的 0 基序号；
    # ffprobe -select_streams v:N 按该序号选择，与 -map 的全局索引是两套编号
    video_stream_ordinal: int = 0
    # 容器声明的显示旋转（度，如 -90/180）。输出保持编码方向（-noautorotate），
    # 非零值时向用户告警；None 表示容器未声明
    rotation_degrees: float | None = None
    # 容器起始时间（format.start_time）。ffmpeg 解码时默认把它从时间轴中减去，
    # ffprobe 的帧 PTS 则保留原值；两者相减才是 fps 滤镜 start_time 应取的值
    format_start_time: float | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass(frozen=True)
class FrameSpec:
    width: int
    height: int
    channels: int
    dtype: str
    ffmpeg_pixel_format: str
    color_family: str
    # 与 VideoInfo.stream_index 一致，供解码器 -map 使用
    stream_index: int = 0
    # 源视频色彩元数据：灰度输出与 YUV→RGB 转换均按此选择色彩矩阵/范围
    color_space: str | None = None
    color_range: str | None = None
    # 源像素格式：RGB 输出据此判断是否为 YUV 家族，决定是否声明色彩矩阵
    source_pixel_format: str | None = None

    @property
    def bytes_per_frame(self) -> int:
        bytes_per_sample = 1 if self.dtype == "uint8" else 2
        return self.width * self.height * self.channels * bytes_per_sample

    @property
    def shape(self) -> tuple[int, ...]:
        if self.channels == 1:
            return (self.height, self.width)
        return (self.height, self.width, self.channels)


@dataclass
class VideoResult:
    input_path: str
    state: JobState
    output_path: str | None = None
    frame_count: int = 0
    error: str | None = None
    traceback: str | None = None
    video_info: dict[str, Any] | None = None
    frame_spec: dict[str, Any] | None = None
    estimated_bytes: int | None = None
    output_bytes: int | None = None
    tiff_container: str | None = None
    # 溯源与告警持久化：告警此前只进瞬时日志（GUI 仅保留 600 行），
    # 任务级 JSON 必须能独立回答"这条数据经历了什么"
    input_sha256: str | None = None
    warnings: list[str] = field(default_factory=list)
    frame_manifest_path: str | None = None
    # 时间轴口径：measured=逐帧实测 PTS；model=按采样栅格推算；estimated=启发式估算
    timeline_source: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["state"] = self.state.value
        return result
