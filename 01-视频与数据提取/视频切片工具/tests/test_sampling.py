from pathlib import Path

import pytest

from video_extractor.models import Sampling, SamplingMode, VideoInfo
from video_extractor.sampling import ffmpeg_filter, output_time_seconds


def _info() -> VideoInfo:
    return VideoInfo(
        path=Path("sample.mp4"), width=10, height=10, duration_s=10.0,
        average_fps=30.0, nominal_fps=30.0, frame_count=300, pixel_format="yuv420p",
        bits_per_sample=8, color_family="rgb", codec="h264", is_variable_fps=False,
    )


def test_target_fps_uses_ffmpeg_timeline_filter_not_integer_step() -> None:
    sampling = Sampling(SamplingMode.TARGET_FPS, 12.0)
    assert "fps=12" in (ffmpeg_filter(sampling) or "")
    assert output_time_seconds(12, sampling, _info()) == 1.0


def test_interval_converts_to_target_rate() -> None:
    sampling = Sampling(SamplingMode.INTERVAL, 0.2)
    assert sampling.target_fps == 5.0
    assert "fps=5" in (ffmpeg_filter(sampling) or "")


def test_sampling_validate_rejects_non_numeric_with_value_error() -> None:
    sampling = Sampling(SamplingMode.TARGET_FPS, "fast")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="数值"):
        sampling.validate()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_sampling_validate_rejects_non_finite(bad_value: float) -> None:
    """NaN 的比较恒为 False、inf 大于 0，必须显式拒绝而不能延迟到 round/滤镜报错。"""
    with pytest.raises(ValueError, match="有限正数"):
        Sampling(SamplingMode.TARGET_FPS, bad_value).validate()
    with pytest.raises(ValueError, match="有限正数"):
        Sampling(SamplingMode.INTERVAL, bad_value).validate()


def test_ffmpeg_filter_includes_stream_start_time() -> None:
    """start_time 必须进入 fps 滤镜：固定 0 会让晚于容器起始的视频流
    在首帧之前被复制填充帧（外部审查 R7）。"""
    sampling = Sampling(SamplingMode.TARGET_FPS, 12.0)
    text = ffmpeg_filter(sampling, start_time=1.5)
    assert "fps=12" in text
    assert "start_time=1.5" in text


def test_all_mode_rejects_sampling_value() -> None:
    """全部帧模式忽略 --value 曾是静默行为，现在必须显式拒绝。"""
    with pytest.raises(ValueError, match="不接受采样值"):
        Sampling(SamplingMode.ALL, 5.0).validate()


def test_interval_overflowing_to_inf_fps_is_rejected() -> None:
    """1e-320 是有限正数，但倒数溢出为 inf 并直接进入 fps 滤镜参数。"""
    with pytest.raises(ValueError, match="溢出"):
        Sampling(SamplingMode.INTERVAL, 1e-320).validate()


def test_absurd_target_fps_is_rejected() -> None:
    with pytest.raises(ValueError, match="合理上限"):
        Sampling(SamplingMode.TARGET_FPS, 1e9).validate()


def test_timeline_is_uniform_detects_real_vfr() -> None:
    from video_extractor.sampling import timeline_is_uniform

    assert timeline_is_uniform([0.0, 0.1, 0.2, 0.3])
    # 外部审查 R2 场景：两率同为 10fps 但 PTS 不均匀的真 VFR
    assert not timeline_is_uniform([0.0, 0.1, 0.3, 0.4, 0.6, 0.7])
    # MKV 毫秒精度容器的 33/34ms 舍入抖动不算 VFR
    assert timeline_is_uniform([0.0, 0.033, 0.067, 0.1, 0.133])
    # 重复帧（零间隔）不是均匀时间轴
    assert not timeline_is_uniform([0.0, 0.1, 0.1, 0.2])
    # 少于 3 帧没有间隔序列，按均匀处理
    assert timeline_is_uniform([0.0])
    assert timeline_is_uniform([0.0, 0.1])


def test_median_interval_returns_typical_delta() -> None:
    from video_extractor.sampling import median_interval

    assert median_interval([0.0, 0.1, 0.2, 0.3]) == pytest.approx(0.1)
    assert median_interval([0.0]) is None
    assert median_interval([]) is None
