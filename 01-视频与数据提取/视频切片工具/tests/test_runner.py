from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

import video_extractor.runner as runner_mod

from video_extractor.models import ExtractOptions, JobState, Sampling, SamplingMode, VideoInfo
from video_extractor.runner import (
    JobRunner,
    STAGE_HEARTBEAT_NAME,
    cleanup_stale_stages,
    detect_orphan_outputs,
    find_videos,
    output_paths_for,
)


def test_cleanup_stale_stages_only_removes_old_dirs(tmp_path: Path) -> None:
    partial = tmp_path / ".partial"
    old = partial / "old_job"
    fresh = partial / "fresh_job"
    old.mkdir(parents=True)
    fresh.mkdir()
    old_time = time.time() - 48 * 3600
    os.utime(old, (old_time, old_time))

    assert cleanup_stale_stages(tmp_path, max_age_seconds=24 * 3600) == 1
    assert not old.exists()
    assert fresh.exists()


def test_cleanup_stale_stages_skips_active_heartbeat(tmp_path: Path) -> None:
    partial = tmp_path / ".partial"
    active = partial / "active_job"
    active.mkdir(parents=True)
    old_time = time.time() - 48 * 3600
    (active / STAGE_HEARTBEAT_NAME).write_text("active", encoding="ascii")
    os.utime(active, (old_time, old_time))
    # 心跳文件 mtime 是现在；即使目录已超过 24h 也必须跳过

    assert cleanup_stale_stages(tmp_path, max_age_seconds=24 * 3600, heartbeat_max_age_seconds=120) == 0
    assert active.exists()


def test_cleanup_stale_stages_removes_old_dir_with_stale_heartbeat(tmp_path: Path) -> None:
    partial = tmp_path / ".partial"
    stale = partial / "stale_job"
    stale.mkdir(parents=True)
    old_time = time.time() - 48 * 3600
    heartbeat = stale / STAGE_HEARTBEAT_NAME
    heartbeat.write_text("stale", encoding="ascii")
    os.utime(stale, (old_time, old_time))
    os.utime(heartbeat, (old_time, old_time))

    assert cleanup_stale_stages(tmp_path, max_age_seconds=24 * 3600, heartbeat_max_age_seconds=120) == 1
    assert not stale.exists()


def test_find_videos_excludes_hidden_partial_and_output_dir(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".partial").mkdir()
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / ".hidden" / "b.mp4").write_bytes(b"x")
    (tmp_path / ".partial" / "c.mp4").write_bytes(b"x")
    (output / "d.mp4").write_bytes(b"x")

    found = find_videos(tmp_path, exclude=output)
    assert [item.name for item in found] == ["a.mp4"]


def test_detect_orphan_outputs_finds_final_without_manifest(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    video = input_dir / "a.mp4"
    video.write_bytes(b"x")

    assert detect_orphan_outputs(input_dir, output_dir, [video]) == []

    final, manifest, _parent = output_paths_for(input_dir, output_dir, video)
    assert final.name.endswith("_stack.tif")
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_bytes(b"orphan")

    orphans = detect_orphan_outputs(input_dir, output_dir, [video])
    assert len(orphans) == 1
    assert orphans[0][0] == final
    assert orphans[0][1] == manifest


def test_run_batch_empty_input_still_cleans_stale_partial(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeFFmpegManager:
        def __init__(self, *args, **kwargs):
            pass

        def resolve(self):
            return SimpleNamespace(
                ffmpeg=tmp_path / "ffmpeg", ffprobe=tmp_path / "ffprobe",
                version=(8, 0, 3),
            )

    monkeypatch.setattr(runner_mod, "FFmpegManager", FakeFFmpegManager)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    old = output_dir / ".partial" / "old_job"
    old.mkdir(parents=True)
    old_time = time.time() - 48 * 3600
    os.utime(old, (old_time, old_time))

    runner = JobRunner(ExtractOptions())
    with pytest.raises(ValueError, match="没有支持的视频"):
        runner.run_batch(input_dir, output_dir)
    assert not old.exists()


def test_run_batch_warns_for_vfr_with_target_fps(tmp_path: Path, monkeypatch) -> None:
    """VFR 源按目标帧率采样时 fps 滤镜会在空档复制前帧，必须告警。"""

    class FakeFFmpegManager:
        def __init__(self, *args, **kwargs):
            pass

        def resolve(self):
            return SimpleNamespace(
                ffmpeg=tmp_path / "ffmpeg", ffprobe=tmp_path / "ffprobe",
                version=(8, 0, 3),
            )

    def fake_probe(ffprobe, path, *, on_started=None) -> VideoInfo:
        return VideoInfo(
            path=path, width=64, height=48, duration_s=2.0,
            average_fps=30.0, nominal_fps=30.0, frame_count=60,
            pixel_format="yuv420p", bits_per_sample=8, color_family="rgb",
            codec="h264", is_variable_fps=True,
        )

    monkeypatch.setattr(runner_mod, "FFmpegManager", FakeFFmpegManager)
    monkeypatch.setattr(runner_mod, "probe_video", fake_probe)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    video = input_dir / "a.mp4"
    video.write_bytes(b"x")
    output_dir = tmp_path / "output"
    # 只预置清单文件：孤儿检测只回收"有输出无清单"，缺输出的清单会触发拒绝覆盖
    _final, manifest, parent = output_paths_for(input_dir, output_dir, video)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("csv", encoding="utf-8")

    events: list[dict] = []
    options = ExtractOptions(sampling=Sampling(SamplingMode.TARGET_FPS, 10.0))
    results = JobRunner(options).run_batch(input_dir, output_dir, events.append)

    assert any(
        event.get("kind") == "warning" and "可变帧率" in event.get("message", "")
        for event in events
    )
    assert results[0].state is JobState.FAILED
    assert "拒绝覆盖" in (results[0].error or "")


def test_run_batch_warns_for_multiple_video_streams(tmp_path: Path, monkeypatch) -> None:
    """多视频流容器必须触发告警（此前 stream_count 恒为 1，分支不可达）。"""

    class FakeFFmpegManager:
        def __init__(self, *args, **kwargs):
            pass

        def resolve(self):
            return SimpleNamespace(
                ffmpeg=tmp_path / "ffmpeg", ffprobe=tmp_path / "ffprobe",
                version=(8, 0, 3),
            )

    def fake_probe(ffprobe, path, *, on_started=None) -> VideoInfo:
        return VideoInfo(
            path=path, width=64, height=48, duration_s=2.0,
            average_fps=30.0, nominal_fps=30.0, frame_count=60,
            pixel_format="yuv420p", bits_per_sample=8, color_family="rgb",
            codec="h264", is_variable_fps=False,
            stream_index=1, stream_count=2, video_stream_ordinal=1,
        )

    monkeypatch.setattr(runner_mod, "FFmpegManager", FakeFFmpegManager)
    monkeypatch.setattr(runner_mod, "probe_video", fake_probe)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    video = input_dir / "a.mp4"
    video.write_bytes(b"x")
    output_dir = tmp_path / "output"
    # 预置清单制造"拒绝覆盖"失败，让批次在告警后立即终止，无需真实解码
    _final, manifest, _parent = output_paths_for(input_dir, output_dir, video)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("csv", encoding="utf-8")

    events: list[dict] = []
    results = JobRunner(ExtractOptions()).run_batch(input_dir, output_dir, events.append)

    assert any(
        event.get("kind") == "warning" and "条视频流" in event.get("message", "")
        for event in events
    )
    assert results[0].state is JobState.FAILED


def test_runner_two_frames_same_pts_degrades_without_zero_interval(
    tmp_path: Path, monkeypatch
) -> None:
    """恰好 2 帧且 PTS 相同：median 间隔为 0，必须降级为"无固定帧间隔"，
    而不是在写 TIFF 元数据时 1.0/0 除零使任务失败。"""

    class FakeFFmpegManager:
        def __init__(self, *args, **kwargs):
            pass

        def resolve(self):
            return SimpleNamespace(
                ffmpeg=tmp_path / "ffmpeg", ffprobe=tmp_path / "ffprobe",
                version=(8, 0, 3),
            )

    def fake_probe(ffprobe, path, *, on_started=None) -> VideoInfo:
        return VideoInfo(
            path=path, width=4, height=4, duration_s=0.0,
            average_fps=30.0, nominal_fps=30.0, frame_count=2,
            pixel_format="gray", bits_per_sample=8, color_family="gray",
            codec="rawvideo", is_variable_fps=False,
        )

    class FakeCount:
        frames = 2
        pts_seconds = [0.0, 0.0]

    class FakeDecoder:
        def __init__(self, *args, **kwargs):
            pass

        def count_frames(self, video, options, spec, cancelled, *args, **kwargs):
            return FakeCount()

        def iter_frames(self, video, options, spec, cancelled, *args, **kwargs):
            frame = np.zeros((4, 4), dtype=np.uint8)
            yield frame
            yield frame

    monkeypatch.setattr(runner_mod, "FFmpegManager", FakeFFmpegManager)
    monkeypatch.setattr(runner_mod, "probe_video", fake_probe)
    monkeypatch.setattr(runner_mod, "FrameDecoder", FakeDecoder)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    video = input_dir / "dup_pts.mkv"
    video.write_bytes(b"x")
    output_dir = tmp_path / "output"

    results = JobRunner(ExtractOptions()).run_batch(input_dir, output_dir)

    assert results[0].state is JobState.COMPLETED, results[0].error
    assert results[0].timeline_source == "measured"
    assert any("重复 PTS" in w for w in results[0].warnings)
    with tifffile.TiffFile(Path(results[0].output_path)) as tif:
        metadata = tif.imagej_metadata or {}
    assert "finterval" not in metadata
    assert "fps" not in metadata


def test_run_batch_no_vfr_warning_for_all_mode(tmp_path: Path, monkeypatch) -> None:
    """ALL 模式对 VFR 的处理是时间轴估算标记（另有事件），不再发复制帧告警。"""

    class FakeFFmpegManager:
        def __init__(self, *args, **kwargs):
            pass

        def resolve(self):
            return SimpleNamespace(
                ffmpeg=tmp_path / "ffmpeg", ffprobe=tmp_path / "ffprobe",
                version=(8, 0, 3),
            )

    def fake_probe(ffprobe, path, *, on_started=None) -> VideoInfo:
        return VideoInfo(
            path=path, width=64, height=48, duration_s=2.0,
            average_fps=30.0, nominal_fps=30.0, frame_count=60,
            pixel_format="yuv420p", bits_per_sample=8, color_family="rgb",
            codec="h264", is_variable_fps=True,
        )

    monkeypatch.setattr(runner_mod, "FFmpegManager", FakeFFmpegManager)
    monkeypatch.setattr(runner_mod, "probe_video", fake_probe)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    video = input_dir / "a.mp4"
    video.write_bytes(b"x")
    output_dir = tmp_path / "output"
    _final, manifest, parent = output_paths_for(input_dir, output_dir, video)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("csv", encoding="utf-8")

    events: list[dict] = []
    options = ExtractOptions(sampling=Sampling(SamplingMode.ALL))
    results = JobRunner(options).run_batch(input_dir, output_dir, events.append)

    assert not any(
        event.get("kind") == "warning" and "复制前帧" in event.get("message", "")
        for event in events
    )
    assert results[0].state is JobState.FAILED


def _fake_manager(tmp_path: Path, monkeypatch) -> None:
    class FakeFFmpegManager:
        def __init__(self, *args, **kwargs):
            pass

        def resolve(self):
            return SimpleNamespace(
                ffmpeg=tmp_path / "ffmpeg", ffprobe=tmp_path / "ffprobe",
                version=(8, 1, 2),
            )

    monkeypatch.setattr(runner_mod, "FFmpegManager", FakeFFmpegManager)


def test_quarantine_orphan_moves_to_persistent_dir(tmp_path: Path, monkeypatch) -> None:
    """孤儿隔离区必须是 .orphans（持久），而不是会被 24h 清理的 .partial：
    缺 CSV 不证明上次任务崩溃，隔离的数据绝不能被自动删除。"""
    _fake_manager(tmp_path, monkeypatch)
    output = tmp_path / "out"
    output.mkdir()
    final = output / "a_stack.tif"
    final.write_bytes(b"precious")

    runner = JobRunner(ExtractOptions())
    destination = runner._quarantine_orphan(output, final)

    assert destination.is_file()
    assert destination.parent.parent.name == ".orphans"
    assert not final.exists()
    # 即使隔离目录陈旧超过 24h，cleanup 也不能碰它（只清理 .partial）
    stale = time.time() - 48 * 3600
    os.utime(destination.parent, (stale, stale))
    os.utime(destination.parent.parent, (stale, stale))
    assert cleanup_stale_stages(output) == 0
    assert destination.is_file()


def test_disk_reserve_is_capped_on_large_volumes(tmp_path: Path) -> None:
    """大容量卷上保留量不能是无限的 5%（8 TiB 空闲会保留 400 GiB 误拦任务）。"""
    class FakeUsage:
        free = 8 * 1024**4  # 8 TiB

    monkeypatch_backup = runner_mod.shutil.disk_usage
    runner_mod.shutil.disk_usage = lambda path: FakeUsage()
    try:
        # 300 GiB 需求：按旧公式保留 400 GiB 会拒绝；带上限后保留 32 GiB 应通过
        JobRunner._check_space(tmp_path, 300 * 1024**3)
    finally:
        runner_mod.shutil.disk_usage = monkeypatch_backup


def test_verify_file_identity_detects_mutation(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"0" * 100)
    identity = runner_mod.file_identity(video)
    video.write_bytes(b"1" * 90)  # 大小变化
    with pytest.raises(OSError, match="被修改"):
        runner_mod.verify_file_identity(video, identity, "哈希计算")
