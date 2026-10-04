"""科学性回归测试：外部审查与内部审查实测场景的防复发用例。

场景来源：
- R1 旋转元数据导致输出按错误宽高转置（外部审查实测）
- R7 视频流晚于容器起始时 fps start_time=0 制造重复帧（外部审查实测）
- R10 校验阶段取消仍提交输出（外部审查实测）
- R6 哈希与解码之间输入被替换（外部审查实测）
- R11 abort() 二次异常掩盖原始错误并终止整批（外部审查故障注入）
"""

from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

import pytest
import tifffile

from video_extractor.ffmpeg import probe_video
from video_extractor.models import (
    ExtractOptions,
    JobState,
    Sampling,
    SamplingMode,
)
from video_extractor.runner import JobRunner


# ffmpeg_paths（含 FFmpeg 缺失时 SKIP 兜底）见 tests/conftest.py


@pytest.fixture(scope="module")
def base_video(tmp_path_factory, ffmpeg_paths) -> Path:
    """与 test_integration 相同规格的 1 秒 10fps 64x48 基准视频。"""
    folder = tmp_path_factory.mktemp("reg_base")
    path = folder / "base.mp4"
    cmd = [
        str(ffmpeg_paths.ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=64x48:rate=10",
        "-c:v", "mpeg4", "-pix_fmt", "yuv420p", str(path),
    ]
    subprocess.run(cmd, check=True, timeout=30)
    return path


@pytest.fixture(scope="module")
def delayed_video(tmp_path_factory, ffmpeg_paths) -> Path:
    """音频从 0s、视频延迟到 1s 的 MP4：容器起始 0 而视频流起始 1（R7）。"""
    folder = tmp_path_factory.mktemp("reg_delayed")
    path = folder / "delayed.mp4"
    cmd = [
        str(ffmpeg_paths.ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "anullsrc=r=8000:duration=3",
        "-itsoffset", "1",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=64x48:rate=10",
        "-map", "1:v", "-map", "0:a",
        "-c:v", "mpeg4", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
    ]
    subprocess.run(cmd, check=True, timeout=30)
    return path


def test_delayed_video_stream_start_is_not_padded_with_duplicates(
    delayed_video: Path, tmp_path: Path
) -> None:
    """视频流晚于容器起始：采样不得在首帧之前复制填充帧（10 帧源 → 10 帧输出）。"""
    output_dir = tmp_path / "delayed_out"
    options = ExtractOptions(sampling=Sampling(SamplingMode.TARGET_FPS, 10.0))
    runner = JobRunner(options)
    results = runner.run_batch(delayed_video.parent, output_dir)

    assert results[0].state is JobState.COMPLETED
    assert results[0].frame_count == 10
    assert results[0].timeline_source == "model"

    with Path(results[0].frame_manifest_path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert float(rows[0]["stream_start_time_s"]) == pytest.approx(1.0, abs=1e-3)
    assert float(rows[0]["scheduled_time_s"]) == pytest.approx(0.0)


def test_mkv_with_offset_start_time_is_not_empty(
    base_video: Path, ffmpeg_paths, tmp_path: Path
) -> None:
    """整体后移的 MKV（format.start_time=1）：fps start_time 必须换算为滤镜
    时间轴（1-1=0），否则输出为空。"""
    input_dir = tmp_path / "mkv_in"
    input_dir.mkdir()
    delayed_mkv = input_dir / "delayed.mkv"
    subprocess.run(
        [
            str(ffmpeg_paths.ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
            "-itsoffset", "1", "-i", str(base_video), "-c", "copy", str(delayed_mkv),
        ],
        check=True, timeout=30,
    )
    output_dir = tmp_path / "mkv_offset_out"
    options = ExtractOptions(sampling=Sampling(SamplingMode.TARGET_FPS, 10.0))
    results = JobRunner(options).run_batch(input_dir, output_dir)
    assert results[0].state is JobState.COMPLETED
    assert results[0].frame_count == 10


def test_cancel_during_validating_publishes_nothing(
    base_video: Path, tmp_path: Path
) -> None:
    """校验阶段收到取消：不得提交输出（R10）。"""
    input_dir = tmp_path / "cancel_input"
    input_dir.mkdir()
    shutil.copy2(base_video, input_dir / "v.mp4")
    output_dir = tmp_path / "cancel_out"
    options = ExtractOptions(sampling=Sampling(SamplingMode.TARGET_FPS, 10.0))
    runner = JobRunner(options)

    def callback(event: dict) -> None:
        if event.get("kind") == "state" and event.get("state") == "validating":
            runner.cancel()

    results = runner.run_batch(input_dir, output_dir, callback)
    assert results[0].state is JobState.CANCELLED
    assert not list(output_dir.rglob("*_stack.tif"))
    assert not list(output_dir.rglob("*_frames.csv"))


def test_input_replacement_between_phases_is_detected(
    base_video: Path, tmp_path: Path
) -> None:
    """哈希与解码之间输入被替换：必须显式失败而不是让哈希与输出脱节（R6）。"""
    input_dir = tmp_path / "identity_input"
    input_dir.mkdir()
    target = input_dir / "v.mp4"
    shutil.copy2(base_video, target)
    output_dir = tmp_path / "identity_out"
    options = ExtractOptions(sampling=Sampling(SamplingMode.TARGET_FPS, 10.0))
    runner = JobRunner(options)

    def callback(event: dict) -> None:
        if event.get("kind") == "state" and event.get("state") == "validating":
            target.write_bytes(b"replaced-during-processing")

    results = runner.run_batch(input_dir, output_dir, callback)
    assert results[0].state is JobState.FAILED
    assert "被修改" in (results[0].error or "")
    assert not list(output_dir.rglob("*_stack.tif"))


def test_writer_secondary_failure_preserves_primary_error(
    base_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """abort() 再抛异常不能掩盖原始写入错误，也不能终止整批（R11）。"""
    import video_extractor.runner as runner_mod
    from video_extractor.writers import StackWriter as RealStackWriter

    class ExplodingWriter(RealStackWriter):
        def write(self, index, frame):
            raise OSError("primary simulated write failure")

        def abort(self):
            raise OSError("secondary simulated flush failure")

    monkeypatch.setattr(runner_mod, "StackWriter", ExplodingWriter)
    input_dir = tmp_path / "writer_input"
    input_dir.mkdir()
    shutil.copy2(base_video, input_dir / "v.mp4")
    output_dir = tmp_path / "writer_out"
    options = ExtractOptions(sampling=Sampling(SamplingMode.TARGET_FPS, 10.0))
    runner = JobRunner(options)
    results = runner.run_batch(input_dir, output_dir)

    assert results[0].state is JobState.FAILED
    assert "primary simulated write failure" in (results[0].error or "")
    assert "secondary simulated flush failure" in (results[0].error or "")


def test_job_manifest_records_sha_and_warnings(base_video: Path, tmp_path: Path) -> None:
    """任务 JSON 必须包含 input_sha256 与 warnings（README 溯源承诺，R9）。"""
    import json

    output_dir = tmp_path / "provenance_out"
    options = ExtractOptions()
    results = JobRunner(options).run_batch(base_video.parent, output_dir)
    assert results[0].state is JobState.COMPLETED
    manifests = list((output_dir / "manifests").glob("job_*.json"))
    assert len(manifests) == 1
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    record = payload["results"][0]
    assert record["input_sha256"]
    assert isinstance(record["warnings"], list)
    assert record["frame_manifest_path"].endswith("_frames.csv")
    assert record["timeline_source"] == "measured"


def test_rotated_metadata_output_matches_coded_orientation(
    ffmpeg_paths, base_video: Path, tmp_path: Path
) -> None:
    """带 display matrix 90° 的视频：输出必须等于编码方向基准（R1）。

    FFmpeg 默认 autorotate 会把输出旋转为 48x64 而总字节数不变，
    读取端按 probe 的 64x48 reshape 即静默转置——这是外部审查实测的 P1。
    """
    rotated = base_video.parent / "rotated.mp4"
    subprocess.run(
        [
            str(ffmpeg_paths.ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
            "-display_rotation", "90", "-i", str(base_video), "-c", "copy", str(rotated),
        ],
        check=True, timeout=30,
    )
    info = probe_video(ffmpeg_paths.ffprobe, rotated)
    assert (info.width, info.height) == (64, 48)
    assert info.rotation_degrees == 90

    # 隔离输入目录：只包含旋转视频本身
    input_dir = tmp_path / "rot_in"
    input_dir.mkdir()
    shutil.copy2(rotated, input_dir / "rotated.mp4")
    output_dir = tmp_path / "rot_out"
    options = ExtractOptions()
    runner = JobRunner(options)
    results = runner.run_batch(input_dir, output_dir)
    assert results[0].state is JobState.COMPLETED
    assert any("旋转" in w for w in results[0].warnings)

    ground_truth_cmd = [
        str(ffmpeg_paths.ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-noautorotate", "-i", str(rotated), "-map", "0:0", "-an", "-sn", "-dn",
        "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    ground_truth = subprocess.run(
        ground_truth_cmd, capture_output=True, check=True, timeout=60
    ).stdout
    with tifffile.TiffFile(Path(results[0].output_path)) as tif:
        stack = tif.series[0].asarray()
    assert stack.shape == (results[0].frame_count, 48, 64, 3)
    assert stack.tobytes() == ground_truth


def test_true_vfr_gets_measured_timeline_and_no_fake_interval(
    ffmpeg_paths, tmp_path_factory, tmp_path: Path
) -> None:
    """流级帧率相同但 PTS 不均匀的真 VFR：时间轴必须实测、TIFF 不得写入
    固定 finterval、CSV 保留逐帧真实 PTS（R2）。"""
    folder = tmp_path_factory.mktemp("reg_vfr")
    vfr = folder / "vfr.mkv"
    subprocess.run(
        [
            str(ffmpeg_paths.ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=duration=1.4:size=64x48:rate=10",
            "-vf", "setpts='PTS+gte(N,3)*0.1/TB+gte(N,6)*0.1/TB'",
            "-c:v", "mpeg4", "-pix_fmt", "yuv420p",
            "-video_track_timescale", "1000000", str(vfr),
        ],
        check=True, timeout=30,
    )
    output_dir = tmp_path / "vfr_out"
    options = ExtractOptions()
    results = JobRunner(options).run_batch(vfr.parent, output_dir)

    assert results[0].state is JobState.COMPLETED
    assert results[0].timeline_source == "measured"
    assert any("不均匀" in w for w in results[0].warnings)

    with tifffile.TiffFile(Path(results[0].output_path)) as tif:
        metadata = tif.imagej_metadata or {}
    assert "finterval" not in metadata
    assert "fps" not in metadata

    with Path(results[0].frame_manifest_path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    pts = [float(row["source_pts_s"]) for row in rows]
    deltas = [round(b - a, 3) for a, b in zip(pts, pts[1:])]
    assert 0.2 in deltas and 0.1 in deltas
    assert all(row["timeline_source"] == "measured" for row in rows)


@pytest.fixture(scope="module")
def faststart_base(tmp_path_factory, ffmpeg_paths) -> Path:
    """moov 前置（faststart）的基准视频：中段置零只破坏视频数据包，
    moov 仍可读，才能真实触发"解码错误但退出码为 0"的路径。"""
    folder = tmp_path_factory.mktemp("reg_faststart")
    path = folder / "base_faststart.mp4"
    cmd = [
        str(ffmpeg_paths.ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=duration=2:size=64x48:rate=30",
        "-c:v", "mpeg4", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ]
    subprocess.run(cmd, check=True, timeout=30)
    return path


def _make_corrupt_copy(base_video: Path, destination: Path) -> None:
    """中段数据置零：FFmpeg 解码报错但默认退出码仍为 0（外部审查 R3）。"""
    data = bytearray(base_video.read_bytes())
    start = len(data) // 2
    data[start : start + 2048] = b"\x00" * 2048
    destination.write_bytes(bytes(data))


def test_corrupt_video_fails_strict_decode(
    faststart_base: Path, tmp_path: Path
) -> None:
    """严格模式（默认）：解码错误必须显式失败，而不是静默丢帧标记成功（R3）。"""
    input_dir = tmp_path / "corrupt_strict_in"
    input_dir.mkdir()
    _make_corrupt_copy(faststart_base, input_dir / "corrupt.mp4")

    output_dir = tmp_path / "corrupt_strict_out"
    options = ExtractOptions()
    results = JobRunner(options).run_batch(input_dir, output_dir)
    assert results[0].state is JobState.FAILED
    assert not list(output_dir.rglob("*_stack.tif"))


def test_lenient_decode_carries_salvage_warning(
    faststart_base: Path, tmp_path: Path
) -> None:
    """宽松模式：抢救性提取必须带缺帧风险告警，结果状态如实反映（R3）。"""
    input_dir = tmp_path / "corrupt_lenient_in"
    input_dir.mkdir()
    _make_corrupt_copy(faststart_base, input_dir / "corrupt.mp4")

    output_dir = tmp_path / "corrupt_lenient_out"
    options = ExtractOptions(lenient_decode=True)
    results = JobRunner(options).run_batch(input_dir, output_dir)
    assert any("宽松解码" in w for w in results[0].warnings)
    # 宽松模式下允许成功（丢帧）或失败，但失败时不得静默缺告警
