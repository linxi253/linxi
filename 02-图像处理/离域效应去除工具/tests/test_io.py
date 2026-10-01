# -*- coding: utf-8 -*-
"""tif_io 的单元测试。"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tif_io  # noqa: E402


def _write(path, arr, **kw):
    tifffile.imwrite(str(path), arr, **kw)
    return str(path)


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------
def test_load_uint8_round_trip(tmp_path):
    arr = (np.arange(64 * 48).reshape(64, 48) % 256).astype(np.uint8)
    src = _write(tmp_path / "a.tif", arr)
    img, info = tif_io.load(src)
    assert img.dtype == np.uint8
    assert img.shape == (64, 48)
    assert info.shape == (64, 48)
    assert np.array_equal(img, arr)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        tif_io.load(str(tmp_path / "nope.tif"))


def test_inspect_reports_single_page(tmp_path):
    src = _write(tmp_path / "one.tif", np.zeros((16, 20), np.uint8))
    info = tif_io.inspect(src)
    assert info.n_pages == 1
    assert not info.is_stack


def test_inspect_and_load_stack_frame(tmp_path):
    stack = np.stack([np.full((16, 20), i, np.uint8) for i in range(5)])
    src = _write(tmp_path / "stack.tif", stack)
    info = tif_io.inspect(src)
    assert info.n_pages == 5
    assert info.is_stack

    img, _ = tif_io.load(src, frame=3)
    assert img.shape == (16, 20)
    assert np.all(img == 3)


def test_load_frame_out_of_range(tmp_path):
    stack = np.stack([np.zeros((8, 8), np.uint8) for _ in range(3)])
    # photometric 必须显式给：tifffile 对 (3|4, H, W) 的 uint8 默认按 planar
    # RGB(A) 写（未来版本还会改默认），显式声明才不受上游默认值影响。
    src = _write(tmp_path / "s.tif", stack, photometric="minisblack")
    with pytest.raises(IndexError):
        tif_io.load(src, frame=99)


def test_rgb_is_converted_to_gray(tmp_path):
    rgb = np.zeros((10, 12, 3), np.uint8)
    rgb[..., 0] = 255                       # 纯红
    src = _write(tmp_path / "rgb.tif", rgb)
    img, info = tif_io.load(src)
    assert img.ndim == 2
    assert img.shape == (10, 12)
    assert img[0, 0] == pytest.approx(0.299 * 255, abs=1)


def test_uint16_preserved(tmp_path):
    arr = (np.arange(32 * 32).reshape(32, 32) * 40).astype(np.uint16)
    src = _write(tmp_path / "u16.tif", arr)
    img, info = tif_io.load(src)
    assert img.dtype == np.uint16
    assert info.bits == 16


def test_summary_mentions_frames_for_stack(tmp_path):
    stack = np.zeros((4, 8, 8), np.uint8)
    src = _write(tmp_path / "s.tif", stack, photometric="minisblack")
    info = tif_io.inspect(src)
    assert "4" in info.summary()


# --------------------------------------------------------------------------
# 输出路径
# --------------------------------------------------------------------------
def test_make_output_path_never_returns_source(tmp_path):
    src = _write(tmp_path / "img.tif", np.zeros((8, 8), np.uint8))
    out = tif_io.make_output_path(src)
    assert os.path.abspath(out) != os.path.abspath(src)
    assert out.endswith("_deloc.tif")


def test_make_output_path_avoids_collision(tmp_path):
    src = _write(tmp_path / "img.tif", np.zeros((8, 8), np.uint8))
    first = tif_io.make_output_path(src)
    open(first, "wb").close()
    second = tif_io.make_output_path(src)
    assert second != first
    assert second.endswith("_deloc_2.tif")


def test_make_output_path_custom_suffix_and_ext(tmp_path):
    src = _write(tmp_path / "img.tif", np.zeros((8, 8), np.uint8))
    assert tif_io.make_output_path(src, "_mask", ".png").endswith("_mask.png")


# --------------------------------------------------------------------------
# 写出
# --------------------------------------------------------------------------
def test_save_keeps_uint8(tmp_path):
    src = _write(tmp_path / "a.tif", np.full((16, 16), 100, np.uint8))
    img, info = tif_io.load(src)
    out = str(tmp_path / "out.tif")
    report = tif_io.save(out, img.astype(np.float32) + 0.4, source=info)
    assert report["dtype"] == "uint8"
    back = tifffile.imread(out)
    assert back.dtype == np.uint8
    assert np.all(back == 100)


def test_save_float32(tmp_path):
    src = _write(tmp_path / "a.tif", np.full((16, 16), 100, np.uint8))
    img, info = tif_io.load(src)
    out = str(tmp_path / "out.tif")
    report = tif_io.save(out, img.astype(np.float32) / 3.0, source=info, as_float32=True)
    assert report["dtype"] == "float32"
    back = tifffile.imread(out)
    assert back.dtype == np.float32
    assert back[0, 0] == pytest.approx(100 / 3, rel=1e-5)


def test_save_reports_clipping(tmp_path):
    src = _write(tmp_path / "a.tif", np.zeros((10, 10), np.uint8))
    _, info = tif_io.load(src)
    arr = np.full((10, 10), 300.0, np.float32)      # 远超 uint8 上限
    report = tif_io.save(str(tmp_path / "o.tif"), arr, source=info)
    assert report["clipped_fraction"] == pytest.approx(1.0)


def test_save_refuses_to_overwrite_source(tmp_path):
    src = _write(tmp_path / "a.tif", np.zeros((8, 8), np.uint8))
    _, info = tif_io.load(src)
    with pytest.raises(ValueError):
        tif_io.save(src, np.zeros((8, 8), np.float32),
                    source=info, guard_path=src)


def test_save_uint16(tmp_path):
    arr = np.full((8, 8), 40000, np.uint16)
    src = _write(tmp_path / "a.tif", arr)
    img, info = tif_io.load(src)
    out = str(tmp_path / "o.tif")
    report = tif_io.save(out, img.astype(np.float32), source=info)
    assert report["dtype"] == "uint16"
    assert tifffile.imread(out).dtype == np.uint16


# --------------------------------------------------------------------------
# 元数据与位深（P1 回归）
# --------------------------------------------------------------------------
def test_imagej_metadata_survives_round_trip(tmp_path):
    """P1 回归：结果必须保住 ImageJ 标定（unit / spacing / resolution）。

    丢了这些，用户拿处理后的图去 Fiji 量晶格间距会得到错的物理尺度。
    """
    arr = (np.arange(32 * 32).reshape(32, 32) % 4096).astype(np.uint16)
    src = _write(tmp_path / "ij.tif", arr, imagej=True,
                 resolution=(4.0, 4.0), resolutionunit=1,
                 metadata={"unit": "nm", "spacing": 0.25})
    _, info = tif_io.load(src)
    assert info.is_imagej
    assert info.resolution == pytest.approx((4.0, 4.0))
    assert info.resolutionunit == 1
    assert info.imagej_metadata.get("unit") == "nm"

    out = str(tmp_path / "out.tif")
    tif_io.save(out, arr.astype(np.float32), source=info)
    _, back = tif_io.load(out)
    assert back.is_imagej, "输出丢了 ImageJ 头"
    assert back.resolution == pytest.approx((4.0, 4.0)), "输出丢了像素标定"
    assert back.resolutionunit == 1
    assert back.imagej_metadata.get("unit") == "nm"
    assert back.imagej_metadata.get("spacing") == pytest.approx(0.25)


def test_imagej_metadata_survives_float32_export(tmp_path):
    """float32 输出同样要保住 ImageJ 标定头（tifffile 支持 float32 + imagej）。"""
    arr = (np.arange(16 * 16).reshape(16, 16) % 4096).astype(np.uint16)
    src = _write(tmp_path / "ij.tif", arr, imagej=True,
                 resolution=(4.0, 4.0), resolutionunit=1,
                 metadata={"unit": "nm", "spacing": 0.25})
    _, info = tif_io.load(src)

    out = str(tmp_path / "o.tif")
    report = tif_io.save(out, arr.astype(np.float32) / 3.0, source=info, as_float32=True)
    assert report["dtype"] == "float32"

    _, back = tif_io.load(out)
    assert back.is_imagej, "float32 输出丢了 ImageJ 头"
    assert back.resolution == pytest.approx((4.0, 4.0))
    assert back.imagej_metadata.get("unit") == "nm"
    assert back.imagej_metadata.get("spacing") == pytest.approx(0.25)


def test_plain_resolution_survives_round_trip(tmp_path):
    src = _write(tmp_path / "r.tif", np.zeros((16, 16), np.uint16),
                 resolution=(2.5, 2.5), resolutionunit=3)
    _, info = tif_io.load(src)
    out = str(tmp_path / "o.tif")
    tif_io.save(out, np.zeros((16, 16), np.float32), source=info)
    _, back = tif_io.load(out)
    assert back.resolution == pytest.approx((2.5, 2.5))
    assert back.resolutionunit == 3


def test_float_source_is_not_downcast(tmp_path):
    """P1 回归：浮点源选「保持原位深」时必须仍是浮点，不能被静默降成 uint16。"""
    arr = (np.random.default_rng(7).random((16, 16)) * 1000).astype(np.float32)
    src = _write(tmp_path / "f.tif", arr)
    _, info = tif_io.load(src)

    plan = tif_io.preview_clip(arr, info, as_float32=False)
    assert plan["dtype"] == "float32"
    assert plan["clipped_fraction"] == 0.0

    out = str(tmp_path / "o.tif")
    report = tif_io.save(out, arr, source=info)
    assert report["dtype"] == "float32"
    back = tifffile.imread(out)
    assert back.dtype == np.float32
    assert np.allclose(back, arr)


def test_int32_source_keeps_int32(tmp_path):
    arr = (np.arange(16 * 16).reshape(16, 16) * 1000).astype(np.int32)
    src = _write(tmp_path / "i32.tif", arr)
    _, info = tif_io.load(src)
    out = str(tmp_path / "o.tif")
    report = tif_io.save(out, arr.astype(np.float32), source=info)
    assert report["dtype"] == "int32"
    assert report["clipped_fraction"] == 0.0


def test_preview_clip_matches_save(tmp_path):
    """提前提示的裁剪比例必须与实际写盘一致，否则提示就是骗人的。"""
    src = _write(tmp_path / "a.tif", np.zeros((10, 10), np.uint8))
    _, info = tif_io.load(src)
    arr = np.full((10, 10), 300.0, np.float32)

    plan = tif_io.preview_clip(arr, info, as_float32=False)
    report = tif_io.save(str(tmp_path / "o.tif"), arr, source=info)
    assert plan["dtype"] == report["dtype"]
    assert plan["clipped_fraction"] == pytest.approx(report["clipped_fraction"])
    assert plan["clipped_fraction"] == pytest.approx(1.0)


def test_preview_clip_reports_dtype_change(tmp_path):
    src = _write(tmp_path / "a.tif", np.zeros((8, 8), np.uint8))
    _, info = tif_io.load(src)
    plan = tif_io.preview_clip(np.zeros((8, 8), np.float32), info, as_float32=True)
    assert plan["dtype"] == "float32"
    assert plan["dtype_changed"] is True
    assert plan["source_dtype"] == "uint8"


def test_save_rejects_nan(tmp_path):
    bad = np.array([[1.0, np.nan]], np.float32)
    with pytest.raises(ValueError):
        tif_io.save(str(tmp_path / "o.tif"), bad)


def test_save_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        tif_io.save(str(tmp_path / "o.tif"), np.zeros((0, 0), np.float32))


def test_make_output_path_rejects_empty_suffix(tmp_path):
    """空后缀会让它返回源文件本身，违反「绝不覆盖输入」的约定。"""
    src = _write(tmp_path / "img.tif", np.zeros((8, 8), np.uint8))
    with pytest.raises(ValueError):
        tif_io.make_output_path(src, suffix="")


# --------------------------------------------------------------------------
# 数值保真与原子性（P1 回归）
# --------------------------------------------------------------------------
def test_save_rounds_to_nearest_integer(tmp_path):
    """astype 截断会让所有整数输出系统性偏低约 0.5 个灰阶。"""
    src = _write(tmp_path / "a.tif", np.zeros((4, 4), np.uint8))
    _, info = tif_io.load(src)
    arr = np.array([[100.4, 100.7], [254.9, 0.4]], np.float32)
    tif_io.save(str(tmp_path / "o.tif"), arr, source=info)
    back = tifffile.imread(str(tmp_path / "o.tif"))
    assert back[0, 0] == 100
    assert back[0, 1] == 101
    assert back[1, 0] == 255        # 254.9 四舍五入触顶，clip 兜住
    assert back[1, 1] == 0


def test_save_replaces_output_and_leaves_no_temp_files(tmp_path):
    """写盘必须落临时文件再原子替换：可覆盖旧结果，且不残留半成品。"""
    src = _write(tmp_path / "a.tif", np.zeros((8, 8), np.uint8))
    _, info = tif_io.load(src)
    out = str(tmp_path / "o.tif")
    tif_io.save(out, np.full((8, 8), 50.0, np.float32), source=info)
    tif_io.save(out, np.full((8, 8), 200.0, np.float32), source=info)
    assert tifffile.imread(out)[0, 0] == 200
    leftovers = [f for f in os.listdir(tmp_path) if f.startswith(".deloc_save_")]
    assert leftovers == []


# --------------------------------------------------------------------------
# 审查项01：源保护必须拦住大小写变体（Windows 文件系统不区分大小写）
# --------------------------------------------------------------------------
def test_save_refuses_source_case_variant(tmp_path):
    """审查项01 复现：SourceCase.tif 与 sourcecase.tif 是同一个文件，
    旧的 abspath 字符串比较放行，源像素被结果覆盖。"""
    src = _write(tmp_path / "SourceCase.tif", np.full((8, 8), 123, np.uint8))
    _, info = tif_io.load(src)
    with pytest.raises(ValueError, match="拒绝覆盖源文件"):
        tif_io.save(str(tmp_path / "sourcecase.tif"),
                    np.zeros((8, 8), np.float32), source=info, guard_path=src)
    assert tifffile.imread(src)[0, 0] == 123        # 源文件完好


def test_save_guard_accepts_different_files(tmp_path):
    src = _write(tmp_path / "a.tif", np.zeros((8, 8), np.uint8))
    _, info = tif_io.load(src)
    out = str(tmp_path / "b.tif")
    tif_io.save(out, np.full((8, 8), 7.0, np.float32), source=info, guard_path=src)
    assert tifffile.imread(out)[0, 0] == 7


def test_preview_clip_rejects_float32_overflow(tmp_path):
    """审查项02 残留：有限的 float64 1e40 转 float32 变成 Inf。

    旧实现在原数组上校验有限性，转 float32 后写出 Inf 且报告 0 裁剪。
    """
    src = _write(tmp_path / "a.tif", np.zeros((4, 4), np.float32))
    _, info = tif_io.load(src)
    arr = np.full((4, 4), 1e40, np.float64)
    with pytest.raises(ValueError, match="可表示范围"):
        tif_io.preview_clip(arr, info, as_float32=True)
    with pytest.raises(ValueError, match="可表示范围"):
        tif_io.save(str(tmp_path / "o.tif"), arr, source=info, as_float32=True)


# --------------------------------------------------------------------------
# 审查项08：TIFF 布局必须按 axes 判定，不能靠维数猜
# --------------------------------------------------------------------------
def test_single_page_rgb_is_not_a_stack(tmp_path):
    """审查项08 复现：单页 RGB (10,12,3) 的 series 恰好三维，曾被当成
    10 帧；选第 1 帧反而报越界。"""
    rgb = np.zeros((10, 12, 3), np.uint8)
    rgb[..., 0] = 255
    path = _write(tmp_path / "rgb.tif", rgb)
    info = tif_io.inspect(path)
    assert info.n_pages == 1, "单页 RGB 不是多帧栈"
    assert info.shape == (10, 12)
    img, _ = tif_io.load(path)
    assert img.shape == (10, 12)
    assert img[0, 0] == pytest.approx(0.299 * 255, abs=1)   # 亮度合并


def test_planar_rgb_keeps_width_dimension(tmp_path):
    """审查项08 复现：平面分离 RGB (3,10,12) 曾被 _as_2d 末维分支切成
    (3,10)，静默丢失宽度方向。"""
    planar = np.zeros((3, 10, 12), np.uint8)
    planar[0] = 255                                   # 纯红平面
    path = str(tmp_path / "planar.tif")
    tifffile.imwrite(path, planar, photometric="rgb", planarconfig="separate")
    info = tif_io.inspect(path)
    assert info.n_pages == 1 and info.shape == (10, 12)
    img, _ = tif_io.load(path)
    assert img.shape == (10, 12), "平面 RGB 必须合并成 (H,W) 而不是切掉宽度"
    assert img[0, 0] == pytest.approx(0.299 * 255, abs=1)


def test_frame_index_is_validated_at_entry(tmp_path):
    stack = np.stack([np.full((8, 8), i, np.uint8) for i in range(3)])
    path = _write(tmp_path / "s.tif", stack, photometric="minisblack")
    for bad in (-1, 3, 99):
        with pytest.raises(IndexError):
            tif_io.load(path, frame=bad)
    img, _ = tif_io.load(path, frame=2)
    assert np.all(img == 2)


# --------------------------------------------------------------------------
# 审查项09：灰度语义（MINISWHITE）必须随结果保留
# --------------------------------------------------------------------------
def test_miniswhite_is_detected_and_preserved(tmp_path):
    """审查项09：MINISWHITE 的像素语义与 MINISBLACK 相反。

    检测曾写成 ``int(...) or 1``：MINISWHITE 的枚举值恰好是 0，是合法
    取值却被 ``or`` 当成缺失替换成 1，导致该标签永远识别不到。
    """
    src = str(tmp_path / "mw.tif")
    tifffile.imwrite(src, np.full((8, 8), 200, np.uint8), photometric="miniswhite")
    info = tif_io.inspect(src)
    assert info.miniswhite is True, "MINISWHITE 的枚举值为 0，不能被 or 吞掉"

    out = str(tmp_path / "o.tif")
    tif_io.save(out, np.full((8, 8), 100.0, np.float32), source=info)
    with tifffile.TiffFile(out) as tif:
        assert tif.pages[0].photometric.name == "MINISWHITE", "输出丢了灰度语义"


def test_minisblack_stays_minisblack(tmp_path):
    """对照：普通灰度图不得被误标成 MINISWHITE。"""
    src = _write(tmp_path / "mb.tif", np.full((8, 8), 200, np.uint8))
    info = tif_io.inspect(src)
    assert info.miniswhite is False
    out = str(tmp_path / "o.tif")
    tif_io.save(out, np.full((8, 8), 100.0, np.float32), source=info)
    with tifffile.TiffFile(out) as tif:
        assert tif.pages[0].photometric.name == "MINISBLACK"
