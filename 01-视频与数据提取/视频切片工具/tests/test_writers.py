from pathlib import Path

import numpy as np
import pytest
import tifffile

import video_extractor.writers as writers_mod
from video_extractor.models import FrameSpec
from video_extractor.writers import (
    OutputValidationError,
    StackWriter,
    estimate_raw_bytes,
    needs_truncate,
)


def test_small_stack_uses_classic_imagej_tiff_and_records_time_axis(tmp_path: Path) -> None:
    spec = FrameSpec(5, 4, 1, "uint16", "gray16le", "gray")
    out = tmp_path / "stack.tif"
    writer = StackWriter(out, spec, 3, 0.25)
    writer.open()
    for index in range(3):
        writer.write(index, np.full(spec.shape, index, dtype=np.uint16))
    writer.close(3)

    assert writer.container_name == "ImageJ-TIFF"
    assert not writer.truncated
    with tifffile.TiffFile(out) as tif:
        assert tif.is_imagej
        assert not tif.is_ome
        assert not tif.is_bigtiff
        assert tif.byteorder == "<"  # II 小端头，魔数 42
        assert len(tif.pages) == 3
        assert tif.series[0].axes == "TYX"
        assert tif.series[0].shape == (3, 4, 5)
        ij = tif.imagej_metadata
        assert ij["ImageJ"] == "1.53c"
        assert ij["frames"] == 3
        assert ij["images"] == 3
        assert ij["hyperstack"] is True
        assert ij["finterval"] == pytest.approx(0.25)
        assert ij["fps"] == pytest.approx(4.0)


def test_rgb_stack_uses_rgb_photometric(tmp_path: Path) -> None:
    spec = FrameSpec(5, 4, 3, "uint8", "rgb24", "rgb")
    out = tmp_path / "rgb.tif"
    writer = StackWriter(out, spec, 2, None)
    writer.open()
    for index in range(2):
        writer.write(index, np.zeros(spec.shape, dtype=np.uint8))
    writer.close(2)

    with tifffile.TiffFile(out) as tif:
        assert tif.series[0].axes == "TYXS"
        assert tif.series[0].shape == (2, 4, 5, 3)
        assert tif.pages[0].photometric.name == "RGB"


def test_truncate_threshold_matches_classic_tiff_limit() -> None:
    spec = FrameSpec(4, 4, 1, "uint8", "gray", "gray")
    last_classic = writers_mod.TRUNCATE_THRESHOLD_BYTES // spec.bytes_per_frame
    assert not needs_truncate(last_classic, spec)
    assert needs_truncate(last_classic + 1, spec)


def test_writer_switches_to_truncated_single_ifd_beyond_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """超过经典 TIFF 安全线时切换单 IFD 截断式；tifffile 仍按描述读出完整 T 轴。"""
    monkeypatch.setattr(writers_mod, "TRUNCATE_THRESHOLD_BYTES", 1)
    spec = FrameSpec(2, 2, 1, "uint8", "gray", "gray")
    out = tmp_path / "huge.tif"
    writer = StackWriter(out, spec, 2, None)
    writer.open()
    writer.write(0, np.zeros(spec.shape, dtype=np.uint8))
    writer.write(1, np.ones(spec.shape, dtype=np.uint8))
    writer.close(2)

    assert writer.truncated
    with tifffile.TiffFile(out) as tif:
        assert not tif.is_bigtiff
        assert len(tif.pages) == 1  # 只有首页 IFD，数据连续
        assert tif.series[0].shape == (2, 2, 2)  # 按描述中的 frames 还原 T 轴
        assert tif.imagej_metadata["frames"] == 2


def test_single_frame_stack_accepts_squeezed_reader_view(tmp_path: Path) -> None:
    spec = FrameSpec(5, 4, 1, "uint8", "gray", "gray")
    out = tmp_path / "single.tif"
    writer = StackWriter(out, spec, 1, 10.0)
    writer.open()
    writer.write(0, np.ones(spec.shape, dtype=np.uint8))
    writer.close(1)

    with tifffile.TiffFile(out) as tif:
        assert tif.series[0].shape == spec.shape
        assert tif.series[0].axes == "YX"
        # ImageJ 描述对单帧省略 frames= 键（省略即 1）
        assert tif.imagej_metadata.get("frames", 1) == 1


def test_stack_writer_rejects_frame_count_mismatch(tmp_path: Path) -> None:
    spec = FrameSpec(4, 4, 1, "uint8", "gray", "gray")
    writer = StackWriter(tmp_path / "stack.tif", spec, 2, None)
    writer.open()
    writer.write(0, np.zeros(spec.shape, dtype=np.uint8))
    with pytest.raises(OutputValidationError, match="预先计数"):
        writer.close(1)


def test_estimate_raw_bytes() -> None:
    spec = FrameSpec(4, 4, 1, "uint8", "gray", "gray")
    assert estimate_raw_bytes(3, spec) == 3 * spec.bytes_per_frame
    assert estimate_raw_bytes(None, spec) is None
