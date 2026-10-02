# -*- coding: utf-8 -*-
"""contrast_core 核心算法单元测试（不导入 GUI/Tk/matplotlib）。"""

import math
import warnings
from pathlib import Path
import sys

import numpy as np
import pytest

# 使测试可导入同目录上级的 contrast_core 模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contrast_core import (  # noqa: E402
    ALGORITHMS,
    algorithm_names,
    compute_contrast,
    compute_contrast_stack,
    compute_contrast_stack_ex,
    estimate_display_clim,
    format_metadata_lines,
    method_name_of,
    normalize_tiff_array,
    normalize_tiff_array_ex,
    open_tiff_stack,
    open_tiff_stack_ex,
    page_datetimes,
    pixel_size_from_description,
    pixel_size_nm,
    probe_tiff,
    rgb_to_gray,
    clamp_roi_to_image,
)

ALL_KEYS = [key for _, key in ALGORITHMS]


class TestStdOverMean:
    def test_uniform_image_zero_contrast(self):
        data = np.full((10, 10), 100.0)
        assert compute_contrast(data, "std_over_mean") == 0.0

    def test_known_value(self):
        data = np.array([[1.0, 3.0]])  # mean=2, std=1
        assert compute_contrast(data, "std_over_mean") == pytest.approx(0.5)

    def test_zero_mean_returns_nan(self):
        data = np.array([[-1.0, 1.0]])  # mean=0
        assert math.isnan(compute_contrast(data, "std_over_mean"))


class TestMichelson:
    def test_known_value(self):
        data = np.array([[10.0, 30.0]])  # (30-10)/(30+10) = 0.5
        assert compute_contrast(data, "michelson") == pytest.approx(0.5)

    def test_uniform_image_zero_contrast(self):
        data = np.full((5, 5), 42.0)
        assert compute_contrast(data, "michelson") == 0.0

    def test_zero_denominator_returns_nan(self):
        data = np.array([[-5.0, 5.0]])  # max+min=0
        assert math.isnan(compute_contrast(data, "michelson"))

    def test_full_range_contrast_is_one(self):
        data = np.array([[0.0, 255.0]])
        assert compute_contrast(data, "michelson") == pytest.approx(1.0)


class TestVarOverMean:
    def test_known_value(self):
        data = np.array([[1.0, 3.0]])  # var=1, mean=2
        assert compute_contrast(data, "var_over_mean") == pytest.approx(0.5)

    def test_zero_mean_returns_nan(self):
        data = np.array([[-2.0, 2.0]])
        assert math.isnan(compute_contrast(data, "var_over_mean"))


class TestMichelsonRobust:
    """稳健Michelson：1%/99% 分位替代全局极值，抑制单像素离群。"""

    def test_known_value(self):
        data = np.array([[1.0, 3.0]])
        # p1=1.02, p99=2.98 → (2.98-1.02)/(2.98+1.02) = 0.49
        assert compute_contrast(data, "michelson_robust") == pytest.approx(0.49)

    def test_uniform_image_zero_contrast(self):
        data = np.full((7, 7), 33.0)
        assert compute_contrast(data, "michelson_robust") == 0.0

    def test_zero_denominator_returns_nan(self):
        data = np.array([[-5.0, 5.0]])  # p99 + p1 = 0
        assert math.isnan(compute_contrast(data, "michelson_robust"))

    def test_outlier_suppressed_vs_plain_michelson(self):
        # 200 个 10.0 + 1 个 10000 的热像素：极值版≈1，稳健版≈0
        data = np.full(200, 10.0)
        data[0] = 10000.0
        assert compute_contrast(data, "michelson") > 0.99
        assert compute_contrast(data, "michelson_robust") < 0.01


class TestComputeContrastStack:
    """compute_contrast_stack 与逐帧 compute_contrast 数值等价。"""

    def test_equivalence_all_algorithms(self):
        rng = np.random.default_rng(1)
        stack = rng.integers(0, 4096, size=(9, 14, 11)).astype(np.uint16)
        stack[3, 2, 2] = 65535  # 混入一个热像素
        stack[5] = 0            # 混入一帧全零（应得 NaN）
        for key in ALL_KEYS:
            expected = np.array(
                [compute_contrast(stack[i], key) for i in range(stack.shape[0])])
            got = compute_contrast_stack(stack, key)
            np.testing.assert_allclose(got, expected, rtol=1e-12, equal_nan=True)

    def test_output_shape_and_dtype(self):
        stack = np.zeros((4, 6, 6), dtype=np.float32)
        out = compute_contrast_stack(stack, "std_over_mean")
        assert out.shape == (4,)
        assert out.dtype == np.float64

    def test_empty_stack(self):
        out = compute_contrast_stack(np.zeros((0, 5, 5)), "michelson")
        assert out.shape == (0,)

    def test_empty_roi_returns_nan_without_warnings(self):
        stack = np.zeros((3, 0, 0))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = compute_contrast_stack(stack, "std_over_mean")
        assert np.all(np.isnan(out))

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError):
            compute_contrast_stack(np.zeros((2, 4, 4)), "nope")

    def test_complex_input_raises(self):
        with pytest.raises(ValueError):
            compute_contrast_stack(np.ones((2, 3, 3), dtype=np.complex128), "michelson")


class TestAlgorithmRegistry:
    def test_algorithms_registered(self):
        assert len(ALGORITHMS) == 4
        keys = [key for _, key in ALGORITHMS]
        assert keys == ["std_over_mean", "michelson", "var_over_mean",
                        "michelson_robust"]

    def test_unknown_key_raises_value_error(self):
        data = np.array([[1.0, 3.0]])
        with pytest.raises(ValueError):
            compute_contrast(data, "nonexistent")

    def test_integer_input_handled(self):
        data = np.array([[10, 30]], dtype=np.uint8)
        assert compute_contrast(data, "michelson") == pytest.approx(0.5)


class TestInputGuards:
    """compute_contrast 输入守卫：复数拒绝、空数组 NaN 且无警告。"""

    def test_complex_input_raises(self):
        with pytest.raises(ValueError):
            compute_contrast(np.array([[1 + 2j, 3 + 4j]]), "std_over_mean")

    def test_empty_returns_nan(self):
        for key in ALL_KEYS:
            assert math.isnan(compute_contrast(np.array([]), key))

    def test_empty_no_runtime_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for key in ALL_KEYS:
                compute_contrast(np.array([]), key)


class TestNormalizeTiffArray:
    def test_2d_single_gray_becomes_stack(self):
        arr = np.zeros((16, 32), dtype=np.uint8)
        out = normalize_tiff_array(arr)
        assert out.shape == (1, 16, 32)

    def test_3d_gray_stack_unchanged(self):
        arr = np.zeros((5, 16, 32), dtype=np.uint8)
        out = normalize_tiff_array(arr)
        assert out.shape == (5, 16, 32)

    def test_3d_narrow_gray_stack_w4_is_not_color(self):
        # W=4 的灰度堆叠不应被误判为彩色（无 axes 时保守按堆叠处理）
        arr = np.zeros((20, 64, 4), dtype=np.uint8)
        out = normalize_tiff_array(arr)
        assert out.shape == (20, 64, 4)

    def test_3d_narrow_gray_stack_axes_tyx_is_stack(self):
        arr = np.zeros((20, 64, 4), dtype=np.uint8)
        out = normalize_tiff_array(arr, axes="TYX")
        assert out.shape == (20, 64, 4)

    def test_3d_color_single_converted_to_gray_stack(self):
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 255, size=(32, 40, 3), dtype=np.uint8)
        out = normalize_tiff_array(arr, axes="YXC")
        assert out.shape == (1, 32, 40)
        assert out.dtype == np.uint8

    def test_3d_planar_rgb_syx_converted(self):
        # planar RGB（S 在第一轴，如 tifffile 报告 "SYX"）不应被误判为 3 帧灰度
        arr = np.zeros((3, 8, 8), dtype=np.uint8)
        arr[0] = 255  # R 平面全白
        out = normalize_tiff_array(arr, axes="SYX")
        assert out.shape == (1, 8, 8)
        # 0.2989*255 → rint → 76
        assert out[0, 0, 0] == 76

    def test_3d_planar_rgba_cyx_converted(self):
        arr = np.zeros((4, 8, 8), dtype=np.uint8)
        arr[1] = 255  # G 平面全白
        out = normalize_tiff_array(arr, axes="CYX")
        assert out.shape == (1, 8, 8)
        # 0.5870*255 → rint → 150
        assert out[0, 0, 0] == 150

    def test_4d_color_stack_converted_to_gray_stack(self):
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 255, size=(6, 32, 40, 3), dtype=np.uint8)
        out = normalize_tiff_array(arr, axes="TYXC")
        assert out.shape == (6, 32, 40)
        assert out.dtype == np.uint8

    def test_unsupported_5d_raises(self):
        arr = np.zeros((1, 2, 3, 4, 5))
        with pytest.raises(ValueError):
            normalize_tiff_array(arr)

    def test_rgb_to_gray_uint8_white(self):
        # 0.9999*255 = 254.97，四舍五入后纯白应保持 255（此前截断为 254）
        white = np.full((4, 5, 3), 255, dtype=np.uint8)
        assert np.all(rgb_to_gray(white) == 255)

    def test_rgb_to_gray_uint8_black(self):
        black = np.zeros((4, 5, 3), dtype=np.uint8)
        assert np.all(rgb_to_gray(black) == 0)

    def test_rgb_to_gray_float_path(self):
        # 非饱和混合值：旧实现在浮点路径误加 np.rint 会把 0.45→0、0.8→1
        # 二值化；契约是直接加权求和（不取整），必须断言精确数值防回归。
        img = np.full((2, 2, 3), 0.45, dtype=np.float32)
        img[0, 1] = 0.8
        out = rgb_to_gray(img)
        assert out.dtype == np.float32
        # 0.2989+0.5870+0.1140 = 0.9999 → 0.45*0.9999=0.449955、0.8*0.9999=0.79992
        assert np.allclose(out[0, 0], 0.449955, rtol=1e-6, atol=0)
        assert np.allclose(out[0, 1], 0.79992, rtol=1e-6, atol=0)
        assert not np.any((out == 0) | (out == 1))

    def test_rgb_to_gray_negative_float_clipped(self):
        arr = np.full((2, 2, 3), -10.0, dtype=np.float32)
        out = rgb_to_gray(arr)
        assert np.all(out == 0)


class TestClampRoi:
    def test_clamp_inside_sorted(self):
        assert clamp_roi_to_image(1, 2, 10, 20, h=100, w=100) == (1, 2, 10,20)

    def test_clamp_outside(self):
        assert clamp_roi_to_image(-5, -7, 150, 220, h=100, w=100) == (0, 0, 100, 100)

    def test_clamp_negative_sorted(self):
        assert clamp_roi_to_image(30, 40, 5, 10, h=100, w=100) == (5, 10, 30, 40)


class TestOpenTiffStack:
    """open_tiff_stack：memmap 优先，压缩 TIFF 回退内存读取（回归 2026-09-05 P1）。"""

    def _write(self, tmp_path, name, data, **kwargs):
        import tifffile
        kwargs.setdefault("photometric", "minisblack")
        path = str(tmp_path / name)
        tifffile.imwrite(path, data, **kwargs)
        return path

    def test_uncompressed_uses_memmap(self, tmp_path):
        data = np.arange(3 * 8 * 10, dtype=np.uint16).reshape(3, 8, 10)
        path = self._write(tmp_path, "plain.tif", data)
        arr, mode = open_tiff_stack(path)
        assert arr.shape == (3, 8, 10)
        assert mode == "内存映射"
        np.testing.assert_array_equal(np.asarray(arr), data)

    def test_lzw_compressed_falls_back_to_memory(self, tmp_path):
        data = np.arange(4 * 8 * 10, dtype=np.uint16).reshape(4, 8, 10) * 3
        path = self._write(tmp_path, "lzw.tif", data, compression="LZW")
        arr, mode = open_tiff_stack(path)
        assert arr.shape == (4, 8, 10)
        assert "整卷载入" in mode
        np.testing.assert_array_equal(np.asarray(arr), data)

    def test_deflate_compressed_falls_back_to_memory(self, tmp_path):
        data = np.full((2, 6, 6), 77, dtype=np.uint8)
        path = self._write(tmp_path, "deflate.tif", data, compression="DEFLATE")
        arr, mode = open_tiff_stack(path)
        assert arr.shape == (2, 6, 6)
        assert "整卷载入" in mode
        np.testing.assert_array_equal(np.asarray(arr), data)

    def test_single_gray_image_uncompressed(self, tmp_path):
        data = np.zeros((9, 7), dtype=np.uint8)
        path = self._write(tmp_path, "single.tif", data)
        arr, mode = open_tiff_stack(path)
        assert arr.shape == (1, 9, 7)
        assert mode == "内存映射"


class TestValidationOrder:
    """算法 key 校验必须发生在「空输入提前返回」之前（回归 2026-09-15）。"""

    def test_empty_roi_still_validates_key(self):
        with pytest.raises(ValueError):
            compute_contrast(np.array([]), "nope")

    def test_empty_stack_still_validates_key(self):
        with pytest.raises(ValueError):
            compute_contrast_stack(np.zeros((0, 5, 5)), "nope")

    def test_empty_roi_stack_still_validates_key(self):
        with pytest.raises(ValueError):
            compute_contrast_stack_ex(np.zeros((3, 0, 0)), "nope")


class TestStackStats:
    """compute_contrast_stack_ex 的每帧强度统计（导出用）。"""

    def test_stats_match_per_frame_reference(self):
        rng = np.random.default_rng(11)
        stack = rng.integers(0, 4096, size=(6, 9, 8)).astype(np.uint16)
        contrast, mean, sigma, vmin, vmax = compute_contrast_stack_ex(
            stack, "michelson_robust")
        for i in range(stack.shape[0]):
            ref = stack[i].astype(np.float64)
            assert mean[i] == pytest.approx(ref.mean())
            assert sigma[i] == pytest.approx(ref.std())
            assert vmin[i] == pytest.approx(ref.min())
            assert vmax[i] == pytest.approx(ref.max())
            assert contrast[i] == pytest.approx(compute_contrast(stack[i], "michelson_robust"))

    def test_zero_frame_yields_nan_stats(self):
        stack = np.zeros((3, 4, 4), dtype=np.uint16)
        stack[1] = 0
        contrast, mean, sigma, vmin, vmax = compute_contrast_stack_ex(stack, "michelson")
        assert not np.isfinite(contrast[1])
        assert mean[1] == 0.0 and vmax[1] == 0.0

    def test_empty_roi_stats_are_nan_without_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = compute_contrast_stack_ex(np.zeros((2, 0, 0)), "std_over_mean")
        for part in out:
            assert part.shape == (2,)
            assert np.all(np.isnan(part))

    def test_min_max_detect_outlier(self):
        stack = np.full((2, 5, 5), 10.0)
        stack[1, 0, 0] = 65535.0
        _, _, _, _, vmax = compute_contrast_stack_ex(stack, "std_over_mean")
        assert vmax[0] == 10.0 and vmax[1] == 65535.0


class TestClampRounding:
    """ROI 取整：下界 floor、上界 ceil，覆盖所有被矩形触及的像素。"""

    def test_fractional_bounds_cover_touched_pixels(self):
        assert clamp_roi_to_image(10.9, 10.9, 20.1, 20.1, 100, 100) == (10, 10, 21, 21)

    def test_integer_bounds_unchanged(self):
        assert clamp_roi_to_image(1, 2, 10, 20, 100, 100) == (1, 2, 10, 20)

    def test_negative_and_oob_clamped(self):
        assert clamp_roi_to_image(-0.4, -7.2, 150.7, 219.1, 100, 100) == (0, 0, 100, 100)

    def test_reversed_order(self):
        assert clamp_roi_to_image(30, 40, 5, 10, 100, 100) == (5, 10, 30, 40)


class TestForceFrames:
    """planar 彩色单页 → 灰度 与 → 多帧 的两种解析方式。"""

    def test_planar_rgba_default_becomes_one_gray_frame(self):
        arr = np.zeros((4, 8, 8), dtype=np.uint8)
        arr[0] = 255  # R 平面
        out, notes = normalize_tiff_array_ex(arr, axes="SYX")
        assert out.shape == (1, 8, 8)
        assert any("alpha" in n for n in notes)
        assert any("多帧解析" in n for n in notes)

    def test_force_frames_keeps_planes_as_frames(self):
        arr = np.zeros((4, 8, 8), dtype=np.uint8)
        arr[3] = 200
        out, notes = normalize_tiff_array_ex(arr, axes="SYX", force_frames=True)
        assert out.shape == (4, 8, 8)
        assert out[3, 0, 0] == 200
        assert any("多帧解析" in n for n in notes)

    def test_force_frames_ignored_without_color_axis(self):
        arr = np.zeros((5, 8, 8), dtype=np.uint16)
        out, _ = normalize_tiff_array_ex(arr, axes="TYX", force_frames=True)
        assert out.shape == (5, 8, 8)

    def test_planar_rgb_note_when_three_planes(self):
        arr = np.zeros((3, 8, 8), dtype=np.uint8)
        _, notes = normalize_tiff_array_ex(arr, axes="SYX")
        assert any("3 个平面" in n for n in notes)
        assert not any("alpha" in n for n in notes)

    def test_gray_stack_has_no_notes(self):
        arr = np.zeros((6, 8, 8), dtype=np.uint8)
        out, notes = normalize_tiff_array_ex(arr, axes="TYX")
        assert out.shape == (6, 8, 8)
        assert notes == []


class TestFourDAxes:
    """4D 分支优先使用 axes，而不是尺寸启发式。"""

    def test_planar_color_stack_with_axes(self):
        # axes=TCYX 而 T=3=C：此前会被 shape[0] > shape[1] 判据错误拒绝
        arr = np.zeros((3, 3, 8, 8), dtype=np.uint8)
        arr[:, 0] = 255
        out, notes = normalize_tiff_array_ex(arr, axes="TCYX")
        assert out.shape == (3, 8, 8)
        assert any("planar" in n for n in notes)

    def test_chunky_color_stack_with_axes(self):
        arr = np.zeros((5, 8, 8, 3), dtype=np.uint8)
        out, notes = normalize_tiff_array_ex(arr, axes="TYXC")
        assert out.shape == (5, 8, 8)
        assert out.dtype == np.uint8
        assert any("彩色堆叠" in n for n in notes)

    def test_single_channel_4d_with_axes(self):
        arr = np.zeros((4, 8, 8, 1), dtype=np.uint16)
        out, _ = normalize_tiff_array_ex(arr, axes="TYXC")
        assert out.shape == (4, 8, 8)

    def test_tzyx_rejected_with_actionable_message(self):
        arr = np.zeros((5, 3, 64, 80), dtype=np.uint8)
        with pytest.raises(ValueError) as exc:
            normalize_tiff_array(arr, axes="TZYX")
        assert "层析" in str(exc.value)
        assert "T, Y, X" in str(exc.value)

    def test_axes_less_heuristics_kept(self):
        assert normalize_tiff_array(np.zeros((4, 8, 8, 1))).shape == (4, 8, 8)
        assert normalize_tiff_array(np.zeros((4, 8, 8, 3))).shape == (4, 8, 8)
        assert normalize_tiff_array(np.zeros((5, 3, 8, 8))).shape == (5, 8, 8)

    def test_unknown_4d_shape_message_includes_axes(self):
        with pytest.raises(ValueError) as exc:
            normalize_tiff_array(np.zeros((2, 5, 6, 7)), axes="TQYX")
        assert "TQYX" in str(exc.value)


class TestPixelSizeHeuristic:
    def test_common_description_formats(self):
        assert pixel_size_from_description("Pixel size = 0.52 nm") == "0.52 nm/px"
        assert pixel_size_from_description("pixel scale:1.2 um") == "1.2 µm/px"
        assert pixel_size_from_description("0.25nm/pixel") == "0.25 nm/px"

    def test_no_false_positive(self):
        assert pixel_size_from_description("") == ""
        assert pixel_size_from_description("300 kV, spot 3, tilt 12") == ""
        assert pixel_size_from_description("no scale information here") == ""


class TestProbeTiff:
    def _write(self, tmp_path, name, data, **kwargs):
        import tifffile
        kwargs.setdefault("photometric", "minisblack")
        path = str(tmp_path / name)
        tifffile.imwrite(path, data, **kwargs)
        return path

    def test_plain_stack_fields(self, tmp_path):
        data = np.arange(3 * 8 * 10, dtype=np.uint16).reshape(3, 8, 10)
        probe = probe_tiff(self._write(tmp_path, "p.tif", data))
        assert probe.shape == (3, 8, 10)
        assert probe.axes == "QYX"
        assert probe.dtype_str == "uint16"
        assert probe.nbytes == data.nbytes
        assert probe.n_pages == 3
        assert not probe.is_compressed
        assert probe.compression == "无压缩"
        assert probe.file_size > 0 and probe.file_mtime
        assert probe.size_text.endswith("B")  # 480 B

    def test_compressed_flag(self, tmp_path):
        data = np.zeros((2, 6, 6), dtype=np.uint8)
        probe = probe_tiff(self._write(tmp_path, "c.tif", data, compression="LZW"))
        assert probe.is_compressed
        assert probe.compression == "LZW"

    def test_metadata_lines_contain_layout(self, tmp_path):
        data = np.zeros((2, 6, 6), dtype=np.uint8)
        probe = probe_tiff(self._write(tmp_path, "m.tif", data))
        lines = format_metadata_lines(probe)
        assert any("TIFF 布局" in ln for ln in lines)
        assert all(ln.startswith("# ") for ln in lines)

    def test_non_tiff_raises_tiffileerror(self, tmp_path):
        import tifffile
        path = tmp_path / "bad.dat"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
        with pytest.raises(tifffile.TiffFileError):
            probe_tiff(str(path))


class TestOpenTiffStackEx:
    def test_notes_empty_for_gray_stack(self, tmp_path):
        import tifffile
        path = str(tmp_path / "g.tif")
        tifffile.imwrite(path, np.zeros((3, 6, 7), dtype=np.uint8), photometric="minisblack")
        arr, mode, notes = open_tiff_stack_ex(path)
        assert arr.shape == (3, 6, 7)
        assert mode == "内存映射"
        assert notes == []

    def test_notes_describe_color_conversion(self, tmp_path):
        import tifffile
        path = str(tmp_path / "planar.tif")
        tifffile.imwrite(path, np.zeros((4, 6, 7), dtype=np.uint8))
        arr, _, notes = open_tiff_stack_ex(path)
        # tifffile 把 (4,H,W) 写成 planar RGBA，默认按颜色解析
        assert arr.shape == (1, 6, 7)
        assert notes and any("平面" in n for n in notes)

    def test_force_frames_multi_frame(self, tmp_path):
        import tifffile
        path = str(tmp_path / "planar.tif")
        tifffile.imwrite(path, np.arange(4 * 6 * 7, dtype=np.uint16).reshape(4, 6, 7))
        arr, _, notes = open_tiff_stack_ex(path, force_frames=True)
        assert arr.shape == (4, 6, 7)
        assert any("多帧解析" in n for n in notes)


class TestRegistryHelpers:
    def test_algorithm_names_matches_registry(self):
        assert algorithm_names() == [name for name, _ in ALGORITHMS]

    def test_method_name_of_roundtrip(self):
        for name, key in ALGORITHMS:
            assert method_name_of(key) == name
        assert method_name_of("nope") == "nope"


class TestNonFiniteWarnings:
    """含 ±Inf 的数据：衬度按 NaN 传播且不触发 RuntimeWarning（v2.5.0）。"""

    def test_inf_frame_contrast_nan_without_warnings(self):
        stack = np.full((2, 4, 4), 10.0)
        stack[1, 0, 0] = np.inf  # 仅 +Inf：此前 std/divide 触发 invalid 警告
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out, mean, std, vmin, vmax = compute_contrast_stack_ex(stack, "std_over_mean")
        assert np.isfinite(out[0])
        assert np.isnan(out[1])

    def test_inf_single_frame_without_warnings(self):
        arr = np.full((4, 4), 10.0)
        arr[0, 0] = -np.inf
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            value = compute_contrast(arr, "michelson")
        assert np.isnan(value)

    def test_inf_detectable_from_both_extrema(self):
        # 仅 +Inf 帧：vmin 有限而 vmax=Inf —— GUI 依赖两端同时检查才能告警
        stack = np.full((2, 4, 4), 10.0)
        stack[1, 0, 0] = np.inf
        _, _, _, vmin, vmax = compute_contrast_stack_ex(stack, "michelson")
        finite = np.isfinite(vmin) & np.isfinite(vmax)
        assert list(finite) == [True, False]


class TestMemmapFallback:
    """memmap 抛非 ValueError 异常（网络盘/非常规布局）也应回退内存读取。"""

    def test_oserror_falls_back_to_memory(self, tmp_path, monkeypatch):
        import tifffile
        data = np.arange(3 * 6 * 7, dtype=np.uint16).reshape(3, 6, 7)
        path = str(tmp_path / "net.tif")
        tifffile.imwrite(path, data, photometric="minisblack")

        def _raise(_path):
            raise OSError("simulated network share failure")

        monkeypatch.setattr(tifffile, "memmap", _raise)
        arr, mode = open_tiff_stack(path)
        assert arr.shape == (3, 6, 7)
        assert "整卷载入" in mode
        np.testing.assert_array_equal(np.asarray(arr), data)

    def test_valueerror_keeps_precise_message(self, tmp_path):
        import tifffile
        data = np.arange(4 * 6 * 6, dtype=np.uint16).reshape(4, 6, 6) * 5
        path = str(tmp_path / "lzw.tif")
        tifffile.imwrite(path, data, photometric="minisblack", compression="LZW")
        arr, mode = open_tiff_stack(path)
        assert "压缩 TIFF" in mode
        np.testing.assert_array_equal(np.asarray(arr), data)


class TestPageDatetimes:
    """逐页 DateTime 标签读取（CSV 时间列数据源）。"""

    @staticmethod
    def _write_with_datetimes(path, data, stamps):
        import tifffile
        with tifffile.TiffWriter(path) as writer:
            for frame, stamp in zip(data, stamps):
                writer.write(frame, photometric="minisblack",
                             extratags=[(306, 2, 19, stamp, False)])
        return str(path)

    def test_reads_per_page_tags(self, tmp_path):
        data = [np.full((4, 4), v, dtype=np.uint16) for v in (1, 2, 3)]
        stamps = ["2026:09:28 10:00:00", "2026:09:28 10:00:05", "2026:09:28 10:00:09"]
        path = self._write_with_datetimes(tmp_path / "dt.tif", data, stamps)
        assert page_datetimes(path) == stamps

    def test_missing_tags_return_empty(self, tmp_path):
        import tifffile
        path = str(tmp_path / "plain.tif")
        tifffile.imwrite(path, np.zeros((3, 4, 4), dtype=np.uint16),
                         photometric="minisblack")
        assert page_datetimes(path) == []

    def test_unreadable_file_returns_empty(self, tmp_path):
        bad = tmp_path / "bad.dat"
        bad.write_bytes(b"\x00" * 64)
        assert page_datetimes(str(bad)) == []


class TestPixelSizeNm:
    """启发式像素尺寸的 nm 换算（ROI 面积估算用）。"""

    def test_unit_conversions(self):
        assert pixel_size_nm("0.52 nm/px") == pytest.approx(0.52)
        assert pixel_size_nm("1.2 µm/px") == pytest.approx(1200.0)
        assert pixel_size_nm("500 pm/px") == pytest.approx(0.5)
        assert pixel_size_nm("0.5 Å/px") == pytest.approx(0.05)

    def test_from_raw_description(self):
        # 直接喂原始描述串也应可用（与格式化输出同一套模式）
        assert pixel_size_nm("Pixel size = 0.52 nm") == pytest.approx(0.52)

    def test_invalid_returns_none(self):
        assert pixel_size_nm("") is None
        assert pixel_size_nm("no scale information here") is None


class TestEstimateDisplayClim:
    """显示灰度范围采样（v2.5.0 从 GUI 迁入 core 并修复全 NaN 警告）。"""

    def test_all_nan_returns_default_without_warning(self):
        arr = np.full((3, 8, 8), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            clim = estimate_display_clim(arr)
        assert clim == (0.0, 1.0)

    def test_partial_nan_ignored(self):
        arr = np.ones((2, 8, 8))
        arr[0, 0, 0] = np.nan
        arr[1] = 3.0
        vmin, vmax = estimate_display_clim(arr)
        assert (vmin, vmax) == (1.0, 3.0)

    def test_uniform_expands_half_level(self):
        clim = estimate_display_clim(np.full((2, 6, 6), 5.0))
        assert clim[0] == pytest.approx(4.5)
        assert clim[1] == pytest.approx(5.5)

    def test_bounds_within_actual_range(self):
        rng = np.random.default_rng(7)
        arr = rng.uniform(10.0, 90.0, size=(30, 64, 64))
        vmin, vmax = estimate_display_clim(arr)
        assert 10.0 <= vmin <= vmax <= 90.0

    def test_invalid_input_returns_default(self):
        assert estimate_display_clim(None) == (0.0, 1.0)
        assert estimate_display_clim(np.zeros((0, 4, 4))) == (0.0, 1.0)


class TestZAxisNote:
    """3D ZYX（层析切片堆栈）不再静默按帧处理——日志给出提示。"""

    def test_zyx_stack_notes_mention_z(self):
        arr = np.zeros((5, 16, 16), dtype=np.uint16)
        _, notes = normalize_tiff_array_ex(arr, axes="ZYX")
        assert any("Z 轴" in n for n in notes)

    def test_tyx_stack_still_no_notes(self):
        arr = np.zeros((5, 16, 16), dtype=np.uint16)
        _, notes = normalize_tiff_array_ex(arr, axes="TYX")
        assert notes == []
