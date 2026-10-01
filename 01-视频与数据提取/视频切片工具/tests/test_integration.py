"""Integration tests: decoder + runner against a real FFmpeg-generated test video."""

from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest
import tifffile

from video_extractor.decoder import FrameDecoder, resolve_frame_spec
from video_extractor.ffmpeg import FFmpegManager, probe_video
from video_extractor.models import (
    BitDepth,
    ColorMode,
    ExtractOptions,
    JobState,
    Sampling,
    SamplingMode,
)
from video_extractor.runner import JobRunner, find_videos


@pytest.fixture(scope="module")
def ffmpeg_paths():
    return FFmpegManager().resolve()


@pytest.fixture(scope="module")
def test_video(tmp_path_factory, ffmpeg_paths) -> Path:
    """Generate a 2-second 30fps 64x48 RGB test video."""
    folder = tmp_path_factory.mktemp("video")
    path = folder / "synthetic.mp4"
    cmd = [
        str(ffmpeg_paths.ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=duration=2:size=64x48:rate=30",
        "-c:v", "mpeg4", "-pix_fmt", "yuv420p",
        str(path),
    ]
    subprocess.run(cmd, check=True, timeout=30)
    assert path.is_file()
    return path


@pytest.fixture(scope="module")
def test_wmv(tmp_path_factory, ffmpeg_paths) -> Path:
    """Generate a WMV input so the legacy Windows container stays covered."""
    folder = tmp_path_factory.mktemp("wmv_video")
    path = folder / "synthetic.wmv"
    cmd = [
        str(ffmpeg_paths.ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=64x48:rate=12",
        "-c:v", "wmv2", str(path),
    ]
    subprocess.run(cmd, check=True, timeout=30)
    assert path.is_file()
    return path


# ─── decoder tests ───


def test_probe_video_returns_valid_info(ffmpeg_paths, test_video: Path) -> None:
    info = probe_video(ffmpeg_paths.ffprobe, test_video)
    assert info.width == 64
    assert info.height == 48
    assert info.average_fps is not None and abs(info.average_fps - 30.0) < 0.5
    assert info.frame_count is not None and info.frame_count >= 55
    assert info.stream_count == 1


def test_probe_video_handles_chinese_input_path(
    ffmpeg_paths, test_video: Path, tmp_path: Path
) -> None:
    input_dir = tmp_path / "中文输入目录"
    input_dir.mkdir()
    localized_video = input_dir / "显微视频样品.mp4"
    shutil.copy2(test_video, localized_video)

    info = probe_video(ffmpeg_paths.ffprobe, localized_video)

    assert info.path == localized_video
    assert info.width == 64
    assert info.height == 48


def test_decoder_iter_frames_all(ffmpeg_paths, test_video: Path) -> None:
    options = ExtractOptions(
        sampling=Sampling(SamplingMode.ALL),
        color_mode=ColorMode.COLOR,
        bit_depth=BitDepth.UINT8,
    )
    info = probe_video(ffmpeg_paths.ffprobe, test_video)
    spec = resolve_frame_spec(info, options)
    assert spec.shape == (48, 64, 3)
    assert spec.dtype == "uint8"

    decoder = FrameDecoder(ffmpeg_paths.ffmpeg)
    cancelled = threading.Event()
    frames = list(decoder.iter_frames(test_video, options, spec, cancelled))
    assert len(frames) >= 55
    assert frames[0].shape == (48, 64, 3)
    assert frames[0].dtype == np.uint8


def test_decoder_target_fps_reduces_frames(ffmpeg_paths, test_video: Path) -> None:
    options = ExtractOptions(
        sampling=Sampling(SamplingMode.TARGET_FPS, 10.0),
        color_mode=ColorMode.GRAYSCALE,
        bit_depth=BitDepth.UINT8,
    )
    info = probe_video(ffmpeg_paths.ffprobe, test_video)
    spec = resolve_frame_spec(info, options)
    assert spec.channels == 1

    decoder = FrameDecoder(ffmpeg_paths.ffmpeg)
    cancelled = threading.Event()
    count = decoder.count_frames(test_video, options, spec, cancelled)
    # 2 seconds at 10 fps ≈ 20 frames (allow tolerance)
    assert 18 <= count.frames <= 22
    # showinfo 位于 fps 之前，捕获的是"源帧"PTS（30fps × 2s = 60 帧），
    # 供实测可变帧率检测；输出帧数则是 fps 滤镜之后的计数
    assert len(count.pts_seconds) >= count.frames
    assert count.pts_seconds[0] == 0.0
    assert all(b > a for a, b in zip(count.pts_seconds, count.pts_seconds[1:]))


def test_decoder_cancellation(ffmpeg_paths, test_video: Path) -> None:
    options = ExtractOptions(
        sampling=Sampling(SamplingMode.ALL),
        color_mode=ColorMode.GRAYSCALE,
        bit_depth=BitDepth.UINT8,
    )
    info = probe_video(ffmpeg_paths.ffprobe, test_video)
    spec = resolve_frame_spec(info, options)
    decoder = FrameDecoder(ffmpeg_paths.ffmpeg)
    cancelled = threading.Event()
    cancelled.set()  # cancel immediately
    frames = list(decoder.iter_frames(test_video, options, spec, cancelled))
    assert len(frames) == 0


# ─── runner integration tests ───


def test_runner_auto_tiff_stack_end_to_end(tmp_path: Path, test_video: Path) -> None:
    input_dir = test_video.parent
    output_dir = tmp_path / "output"
    options = ExtractOptions(
        sampling=Sampling(SamplingMode.TARGET_FPS, 10.0),
        color_mode=ColorMode.GRAYSCALE,
        bit_depth=BitDepth.UINT8,
    )
    runner = JobRunner(options)
    results = runner.run_batch(input_dir, output_dir)
    assert len(results) == 1
    result = results[0]
    assert result.state is JobState.COMPLETED
    assert result.frame_count >= 18
    assert result.output_path is not None
    assert result.tiff_container == "ImageJ-TIFF"
    out_file = Path(result.output_path)
    assert out_file.is_file()
    assert out_file.name.endswith("_stack.tif")
    assert result.output_bytes == out_file.stat().st_size
    assert result.output_bytes > 0
    with tifffile.TiffFile(out_file) as tif:
        assert tif.is_imagej
        assert not tif.is_ome
        assert not tif.is_bigtiff
        assert tif.imagej_metadata["ImageJ"] == "1.53c"
        assert tif.series[0].shape[0] == result.frame_count
    # manifest exists
    manifests = list((output_dir / "manifests").glob("job_*.json"))
    assert len(manifests) == 1


def test_runner_rgb_stack_end_to_end(tmp_path: Path, test_video: Path) -> None:
    input_dir = test_video.parent
    output_dir = tmp_path / "output_rgb"
    options = ExtractOptions(
        sampling=Sampling(SamplingMode.TARGET_FPS, 5.0),
        color_mode=ColorMode.COLOR,
        bit_depth=BitDepth.UINT8,
    )
    runner = JobRunner(options)
    results = runner.run_batch(input_dir, output_dir)
    assert len(results) == 1
    result = results[0]
    assert result.state is JobState.COMPLETED
    assert result.frame_count >= 8
    out_file = Path(result.output_path)
    with tifffile.TiffFile(out_file) as tif:
        assert tif.is_imagej
        assert tif.series[0].shape[0] == result.frame_count
        assert tif.series[0].shape[-1] == 3
        assert tif.pages[0].photometric.name == "RGB"


def test_runner_handles_chinese_input_and_output_paths(
    tmp_path: Path, test_video: Path
) -> None:
    input_dir = tmp_path / "中文输入"
    input_dir.mkdir()
    localized_video = input_dir / "透射电镜样品.mp4"
    shutil.copy2(test_video, localized_video)
    output_dir = tmp_path / "中文输出"
    options = ExtractOptions(
        sampling=Sampling(SamplingMode.TARGET_FPS, 2.0),
        color_mode=ColorMode.GRAYSCALE,
        bit_depth=BitDepth.UINT8,
    )

    results = JobRunner(options).run_batch(input_dir, output_dir)

    assert len(results) == 1
    assert results[0].state is JobState.COMPLETED
    assert results[0].frame_count >= 3
    output_file = Path(results[0].output_path)
    assert output_file.is_file()
    assert list((output_dir / "manifests").glob("job_*.json"))
    with tifffile.TiffFile(output_file) as tif:
        assert tif.is_imagej
        assert tif.series[0].shape[0] == results[0].frame_count


def test_runner_wmv_to_tiff_stack(tmp_path: Path, test_wmv: Path) -> None:
    output_dir = tmp_path / "wmv_output"
    options = ExtractOptions(
        sampling=Sampling(SamplingMode.ALL),
        color_mode=ColorMode.GRAYSCALE,
        bit_depth=BitDepth.UINT8,
    )

    results = JobRunner(options).run_batch(test_wmv.parent, output_dir)

    assert len(results) == 1
    assert results[0].state is JobState.COMPLETED
    assert results[0].frame_count >= 10
    with tifffile.TiffFile(Path(results[0].output_path)) as tif:
        assert tif.is_imagej
        assert tif.series[0].shape[0] == results[0].frame_count


def test_runner_rejects_existing_output(tmp_path: Path, test_video: Path) -> None:
    input_dir = test_video.parent
    output_dir = tmp_path / "output_dup"
    options = ExtractOptions(
        sampling=Sampling(SamplingMode.TARGET_FPS, 5.0),
        color_mode=ColorMode.GRAYSCALE,
        bit_depth=BitDepth.UINT8,
    )
    runner = JobRunner(options)
    results1 = runner.run_batch(input_dir, output_dir)
    assert results1[0].state is JobState.COMPLETED
    # Second run should fail because output exists
    runner2 = JobRunner(options)
    results2 = runner2.run_batch(input_dir, output_dir)
    assert results2[0].state is JobState.FAILED
    assert "拒绝覆盖" in (results2[0].error or "")


def test_find_videos_sorted(test_video: Path) -> None:
    videos = find_videos(test_video.parent)
    assert len(videos) >= 1
    assert all(v.suffix.lower() == ".mp4" for v in videos)
