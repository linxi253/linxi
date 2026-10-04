import csv
from pathlib import Path

import pytest

from video_extractor.manifest import write_frame_manifest
from video_extractor.models import ExtractOptions, VideoInfo


def _info(tmp_path: Path, variable: bool = False) -> VideoInfo:
    return VideoInfo(
        path=tmp_path / "v.mp4", width=4, height=4, duration_s=1.0,
        average_fps=10.0, nominal_fps=10.0, frame_count=10, pixel_format="yuv420p",
        bits_per_sample=8, color_family="rgb", codec="h264", is_variable_fps=variable,
        color_space="bt709", color_range="tv",
    )


def test_manifest_uses_supplied_hash_and_flags_estimated_time(tmp_path: Path) -> None:
    """哈希由调用方传入（不再重复读输入文件）；VFR 估算时间必须被标记。"""
    target = tmp_path / "frames.csv"
    write_frame_manifest(
        target, _info(tmp_path, variable=True), ExtractOptions(),
        tmp_path / "out.ome.tif", 3, "deadbeef" * 8, time_is_estimated=True,
    )
    with target.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[0]["input_sha256"] == "deadbeef" * 8
    assert rows[0]["time_is_estimated"] == "True"
    assert rows[0]["source_color_space"] == "bt709"
    assert rows[0]["source_color_range"] == "tv"
    assert rows[2]["output_index"] == "2"


def test_manifest_records_measured_timeline_columns(tmp_path: Path) -> None:
    """实测时间轴必须落到 CSV：scheduled 按首帧归零、source_pts 保留绝对时间。"""
    target = tmp_path / "frames_measured.csv"
    write_frame_manifest(
        target, _info(tmp_path), ExtractOptions(),
        tmp_path / "out.tif", 3, "ab" * 32,
        scheduled_times=[0.0, 0.1, 0.3],
        source_pts=[1.0, 1.1, 1.3],
        timeline_source="measured",
        stream_start_time_s=1.0,
    )
    with target.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["timeline_source"] for r in rows] == ["measured"] * 3
    assert rows[0]["source_pts_s"] == "1.0"
    assert rows[2]["source_pts_s"] == "1.3"
    assert rows[2]["scheduled_time_s"] == "0.3"
    assert rows[0]["stream_start_time_s"] == "1.0"


def test_manifest_cancellation_interrupts_row_loop(tmp_path: Path) -> None:
    """写清单过程中收到取消必须中断，而不是继续产出与取消语义矛盾的数据。"""
    import threading

    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(InterruptedError):
        write_frame_manifest(
            tmp_path / "frames_cancel.csv", _info(tmp_path), ExtractOptions(),
            tmp_path / "out.tif", 50_000, "cd" * 32, cancelled=cancelled,
        )
