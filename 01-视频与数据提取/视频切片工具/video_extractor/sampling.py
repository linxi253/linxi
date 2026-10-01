from __future__ import annotations

from statistics import median

from .models import Sampling, SamplingMode, VideoInfo


def ffmpeg_filter(sampling: Sampling, start_time: float = 0.0) -> str | None:
    """构建 fps 采样滤镜。

    start_time 必须传入选中流第一帧的真实 PTS：固定 0 会在视频流晚于
    容器起始时（如前置音轨）于首帧前复制填充帧，制造不存在的观测。
    计数与解码两遍必须使用同一 start_time 才能保证帧数一致。
    """
    sampling.validate()
    if sampling.mode is SamplingMode.ALL:
        return None
    assert sampling.target_fps is not None
    # fps filter is timeline/PTS based, unlike the previous integer-step implementation.
    return f"fps=fps={sampling.target_fps:.12g}:round=near:start_time={start_time:.12g}"


def effective_fps(sampling: Sampling, info: VideoInfo) -> float | None:
    if sampling.target_fps is not None:
        return sampling.target_fps
    return info.average_fps or info.nominal_fps


def estimate_frame_count(sampling: Sampling, info: VideoInfo) -> int | None:
    if sampling.mode is SamplingMode.ALL:
        if info.frame_count is not None:
            return info.frame_count
        if info.duration_s is not None and effective_fps(sampling, info):
            return max(1, round(info.duration_s * float(effective_fps(sampling, info))))
        return None
    if info.duration_s is None or sampling.target_fps is None:
        return None
    # The actual value is verified by a count pass before stack allocation.
    return max(1, round(info.duration_s * sampling.target_fps))


def output_time_seconds(index: int, sampling: Sampling, info: VideoInfo) -> float | None:
    fps = effective_fps(sampling, info)
    return index / fps if fps else None


# 帧间隔均匀性判定的容差：绝对 2ms 覆盖 MKV 等毫秒精度容器的舍入抖动
# （30fps 时 33/34ms 交替），相对 2% 覆盖高帧率下的量化噪声；
# 真正的可变帧率空档（如 0.1s 与 0.2s 交替）远超该容差
_UNIFORM_ABS_TOLERANCE_S = 0.002
_UNIFORM_REL_TOLERANCE = 0.02


def frame_intervals(pts_seconds: list[float]) -> list[float]:
    return [b - a for a, b in zip(pts_seconds, pts_seconds[1:])]


def timeline_is_uniform(pts_seconds: list[float]) -> bool:
    """按实测 PTS 判断帧间隔是否均匀（恒定帧率）。

    少于 3 帧无法判定间隔分布，按均匀处理（单帧/两帧没有"间隔序列"）。
    非正间隔（重复帧或解码乱序）一律视为非均匀。
    """
    if len(pts_seconds) < 3:
        return True
    deltas = frame_intervals(pts_seconds)
    if any(delta <= 0 for delta in deltas):
        return False
    median_delta = median(deltas)
    tolerance = max(_UNIFORM_ABS_TOLERANCE_S, _UNIFORM_REL_TOLERANCE * median_delta)
    return max(deltas) - min(deltas) <= tolerance


def median_interval(pts_seconds: list[float]) -> float | None:
    """实测均匀时间轴的固定帧间隔；帧数不足时返回 None。"""
    if len(pts_seconds) < 2:
        return None
    return median(frame_intervals(pts_seconds))
