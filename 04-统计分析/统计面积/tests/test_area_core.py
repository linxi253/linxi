# -*- coding: utf-8 -*-
"""area_core 纯函数单元测试（不导入 GUI/Tk/tifffile）。"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from area_core import (  # noqa: E402
    polygon_area,
    polygon_perimeter,
    clamp_point,
    flip_polygon_y,
    natural_sort_key,
    polygon_self_intersects,
    rebuild_measurements,
    rgb_to_gray_uint8,
    convert_area_to_units,
    count_out_of_bounds_points,
    coerce_positive_float,
    group_series_by_root,
    normalize_tiff_data,
    ruler_ratio,
    sanitize_project_polygons,
    sanitize_ruler_points,
)


class TestNaturalSort:
    def test_numeric_order(self):
        names = ["Frame10.tif", "Frame2.tif", "Frame1.tif"]
        assert sorted(names, key=natural_sort_key) == [
            "Frame1.tif", "Frame2.tif", "Frame10.tif"]

    def test_prefix_and_suffix(self):
        names = ["img_002.tiff", "img_10.tiff", "img_001.tiff"]
        assert sorted(names, key=natural_sort_key) == [
            "img_001.tiff", "img_002.tiff", "img_10.tiff"]

    def test_no_digits_stable(self):
        names = ["b.tif", "a.tif", "c.tif"]
        assert sorted(names, key=natural_sort_key) == ["a.tif", "b.tif", "c.tif"]

    def test_mixed_digit_and_text(self):
        names = ["sample10b.tif", "sample10a.tif", "sample9.tif"]
        assert sorted(names, key=natural_sort_key) == [
            "sample9.tif", "sample10a.tif", "sample10b.tif"]

    def test_full_paths(self):
        paths = ["D:/x/frame10.tif", "D:/x/frame2.tif"]
        assert sorted(paths, key=natural_sort_key) == [
            "D:/x/frame2.tif", "D:/x/frame10.tif"]


class TestFlipPolygonY:
    def test_flip_twice_is_identity(self):
        pts = [(1.0, 2.5), (3.0, 7.0), (0.0, 10.0)]
        once = flip_polygon_y(pts, 10.0)
        assert once == [(1.0, 7.5), (3.0, 3.0), (0.0, 0.0)]
        assert flip_polygon_y(once, 10.0) == pytest.approx(pts)

    def test_empty(self):
        assert flip_polygon_y([], 5.0) == []

    def test_preserves_x(self):
        pts = [(4.0, 1.0)]
        assert flip_polygon_y(pts, 8.0)[0][0] == 4.0


class TestPolygonArea:
    def test_triangle(self):
        assert polygon_area([(0, 0), (4, 0), (0, 3)]) == pytest.approx(6.0)

    def test_rectangle(self):
        assert polygon_area([(0, 0), (10, 0), (10, 5), (0, 5)]) == pytest.approx(50.0)

    def test_too_few_points(self):
        assert polygon_area([(0, 0), (1, 1)]) == 0.0


class TestClampPoint:
    def test_inside(self):
        assert clamp_point(5, 6, 10, 10) == (5.0, 6.0)

    def test_outside(self):
        assert clamp_point(-1, 20, 10, 10) == (0.0, 10.0)


class TestSelfIntersect:
    def test_simple_rectangle_no_intersect(self):
        assert not polygon_self_intersects([(0, 0), (10, 0), (10, 10), (0, 10)])

    def test_bowtie_intersects(self):
        assert polygon_self_intersects([(0, 0), (10, 10), (10, 0), (0, 10)])

    def test_collinear_spike_detected(self):
        # 修复前：折回尖刺因共线不严格相交而漏报
        assert polygon_self_intersects([(0, 0), (10, 0), (5, 0), (5, 10), (0, 10)])

    def test_vertex_on_edge_detected(self):
        assert polygon_self_intersects([(0, 0), (10, 0), (10, 10), (5, 0), (0, 10)])

    def test_collinear_adjacent_vertices_no_false_positive(self):
        # 共线顶点把一条边分成两段，属合法简单多边形
        assert not polygon_self_intersects([(0, 0), (5, 0), (10, 0), (10, 10), (0, 10)])

    def test_segments_collinear_overlap(self):
        from area_core import _segments_intersect
        assert _segments_intersect((0, 0), (10, 0), (5, 0), (15, 0))

    def test_segments_touch_at_endpoint(self):
        from area_core import _segments_intersect
        assert _segments_intersect((0, 0), (10, 0), (10, 0), (10, 5))

    def test_segments_parallel_no_overlap(self):
        from area_core import _segments_intersect
        assert not _segments_intersect((0, 0), (1, 0), (5, 1), (6, 1))

    def test_segments_collinear_disjoint(self):
        from area_core import _segments_intersect
        assert not _segments_intersect((0, 0), (1, 0), (5, 0), (6, 0))


class TestRgbToGray:
    def test_white(self):
        white = np.full((4, 5, 3), 255, dtype=np.uint8)
        assert rgb_to_gray_uint8(white).dtype == np.uint8
        assert np.all(rgb_to_gray_uint8(white) >= 254)

    def test_black(self):
        black = np.zeros((4, 5, 4), dtype=np.uint8)
        assert np.all(rgb_to_gray_uint8(black) == 0)

    def test_uint16_scaled_not_wrapped(self):
        # 修复前：30000 会被 astype(uint8) 按 mod-256 截成 45
        img = np.full((4, 5, 3), 30000, dtype=np.uint16)
        out = rgb_to_gray_uint8(img)
        assert out[0, 0] == pytest.approx(30000 * 255.0 / 65535.0, abs=1)

    def test_uint16_full_range_maps_to_white(self):
        img = np.full((4, 5, 3), 65535, dtype=np.uint16)
        assert np.all(rgb_to_gray_uint8(img) >= 254)

    def test_float_unit_range_scaled(self):
        img = np.full((4, 5, 3), 0.5, dtype=np.float32)
        out = rgb_to_gray_uint8(img)
        assert out[0, 0] == pytest.approx(127, abs=2)

    def test_float_0_255_range_clipped(self):
        img = np.full((4, 5, 3), 200.0, dtype=np.float32)
        assert np.all(rgb_to_gray_uint8(img) >= 198)

    def test_int16_negative_clipped(self):
        img = np.full((4, 5, 3), -100, dtype=np.int16)
        assert np.all(rgb_to_gray_uint8(img) == 0)

    def test_nan_becomes_zero(self):
        img = np.full((4, 5, 3), np.nan, dtype=np.float32)
        assert np.all(rgb_to_gray_uint8(img) == 0)

    def test_alpha_channel_ignored(self):
        rgb = np.full((4, 5, 3), 255, dtype=np.uint8)
        rgba = np.dstack([rgb, np.full((4, 5), 0, dtype=np.uint8)])
        assert np.array_equal(rgb_to_gray_uint8(rgba), rgb_to_gray_uint8(rgb))


class TestRulerRatio:
    def test_normal(self):
        assert ruler_ratio(100, 1000) == pytest.approx(0.1)

    def test_accepts_numeric_strings(self):
        assert ruler_ratio("100", "1000") == pytest.approx(0.1)

    def test_zero_pixel_length(self):
        with pytest.raises(ValueError):
            ruler_ratio(0.0, 1000)

    def test_zero_actual_length(self):
        with pytest.raises(ValueError):
            ruler_ratio(100, -5)

    def test_non_numeric(self):
        with pytest.raises(ValueError):
            ruler_ratio("abc", 1000)


class TestGroupSeriesByRoot:
    def test_copies_share_root(self):
        measurements = {
            0: {1: {'pixel_area': 10.0}},
            1: {2: {'pixel_area': 12.0}},
            2: {3: {'pixel_area': 14.0}},
        }
        roots = {1: 1, 2: 1, 3: 1}  # 复制链共享 root
        out = group_series_by_root(measurements, roots)
        assert out == {1: [(0, 10.0, 1), (1, 12.0, 2), (2, 14.0, 3)]}

    def test_missing_root_defaults_to_self(self):
        measurements = {0: {5: {'pixel_area': 3.0}}}
        out = group_series_by_root(measurements, {})
        assert out == {5: [(0, 3.0, 5)]}

    def test_legacy_numeric_measurement(self):
        measurements = {0: {1: 7.0}, 1: {2: 9.0}}
        roots = {1: 1, 2: 1}
        out = group_series_by_root(measurements, roots)
        assert out == {1: [(0, 7.0, 1), (1, 9.0, 2)]}

    def test_frames_sorted_within_series(self):
        measurements = {
            3: {1: {'pixel_area': 1.0}},
            0: {1: {'pixel_area': 2.0}},
        }
        out = group_series_by_root(measurements, {})
        assert [f for f, _a, _p in out[1]] == [0, 3]

    def test_zero_value_kept(self):
        measurements = {0: {1: {'pixel_area': 0}}}
        out = group_series_by_root(measurements, {})
        assert out[1][0][1] == 0


class TestSanitizeProjectPolygons:
    def test_valid_passthrough(self):
        raw = {"0": {"1": [[0, 0], [10, 0], [10, 5]]}}
        cleaned, dropped = sanitize_project_polygons(raw, total_frames=5)
        assert cleaned == {0: {1: [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0)]}}
        assert dropped == 0

    def test_out_of_range_frame_dropped(self):
        raw = {"7": {"1": [[0, 0], [10, 0], [10, 5]]}}
        cleaned, dropped = sanitize_project_polygons(raw, total_frames=5)
        assert cleaned == {}
        assert dropped == 1

    def test_bad_frame_key_dropped(self):
        raw = {"abc": {"1": [[0, 0], [10, 0], [10, 5]]}}
        cleaned, dropped = sanitize_project_polygons(raw, 5)
        assert cleaned == {} and dropped == 1

    def test_malformed_points_dropped(self):
        raw = {"0": {
            "1": [[0, 0], [10, 0], ["x", "y"]],   # 非数值
            "2": [[0, 0], [10, 0]],               # 少于3点
            "3": [[0, 0], [10, 0], [10, 5]],      # 正常
        }}
        cleaned, dropped = sanitize_project_polygons(raw, 5)
        assert list(cleaned[0].keys()) == [3]
        assert dropped == 2

    def test_nan_points_dropped(self):
        raw = {"0": {"1": [[0, 0], [10, 0], [float("nan"), 5]]}}
        cleaned, dropped = sanitize_project_polygons(raw, 5)
        assert cleaned == {} and dropped == 1

    def test_negative_frame_dropped(self):
        raw = {"-1": {"1": [[0, 0], [10, 0], [10, 5]]}}
        cleaned, dropped = sanitize_project_polygons(raw, 5)
        assert cleaned == {} and dropped == 1


class TestRebuildMeasurements:
    def test_rebuild_matches_direct_computation(self):
        polygons = {0: {1: [(0, 0), (10, 0), (10, 5), (0, 5)]}}
        out = rebuild_measurements(polygons, 2.0, True)
        assert out[0][1]['pixel_area'] == pytest.approx(50.0)
        assert out[0][1]['unit_area'] == pytest.approx(12.5)

    def test_uncalibrated_unit_equals_pixel(self):
        polygons = {0: {1: [(0, 0), (4, 0), (0, 3)]}}
        out = rebuild_measurements(polygons, 1.0, False)
        assert out[0][1]['unit_area'] == pytest.approx(6.0)

    def test_empty(self):
        assert rebuild_measurements({}, 1.0, False) == {}


class TestAreaUnits:
    def test_not_calibrated_returns_pixels(self):
        assert convert_area_to_units(100.0, 2.0, False) == 100.0

    def test_calibrated(self):
        assert convert_area_to_units(100.0, 2.0, True) == pytest.approx(25.0)

    def test_calibrated_zero_ratio(self):
        assert convert_area_to_units(100.0, 0.0, True) == 100.0


class TestPolygonPerimeter:
    def test_square(self):
        assert polygon_perimeter([(0, 0), (10, 0), (10, 5), (0, 5)]) == pytest.approx(30.0)

    def test_triangle(self):
        assert polygon_perimeter([(0, 0), (3, 0), (0, 4)]) == pytest.approx(12.0)

    def test_includes_closing_edge(self):
        # 两个点：周长 = 往返两段 = 2 × 距离（闭合边生效）
        assert polygon_perimeter([(0, 0), (3, 4)]) == pytest.approx(10.0)

    def test_too_few_points(self):
        assert polygon_perimeter([(1, 1)]) == 0.0
        assert polygon_perimeter([]) == 0.0


class TestCountOutOfBoundsPoints:
    def test_all_inside(self):
        polys = {0: {1: [(0, 0), (10, 0), (10, 5)]}}
        assert count_out_of_bounds_points(polys, 100, 50) == 0

    def test_counts_outside(self):
        polys = {0: {1: [(0, 0), (10, 0), (150, 5)],     # x > w
                     2: [(0, 0), (10, 0), (10, 60)]}}     # y > h
        assert count_out_of_bounds_points(polys, 100, 50) == 2

    def test_boundary_is_inside(self):
        polys = {0: {1: [(0, 0), (100, 50), (0, 50)]}}
        assert count_out_of_bounds_points(polys, 100, 50) == 0

    def test_empty(self):
        assert count_out_of_bounds_points({}, 100, 50) == 0


class TestSanitizeRulerPoints:
    def test_valid(self):
        raw = [[10.5, 20.0], [110.0, 25.0]]
        assert sanitize_ruler_points(raw) == [(10.5, 20.0), (110.0, 25.0)]

    def test_clamped_to_bounds(self):
        raw = [[-5, 20], [110, 200]]
        assert sanitize_ruler_points(raw, 100, 50) == [(0.0, 20.0), (100.0, 50.0)]

    def test_wrong_count_rejected(self):
        assert sanitize_ruler_points([[1, 1]]) == []
        assert sanitize_ruler_points([[1, 1], [2, 2], [3, 3]]) == []

    def test_malformed_rejected(self):
        assert sanitize_ruler_points([["a", 1], [2, 2]]) == []
        assert sanitize_ruler_points([[float("nan"), 1], [2, 2]]) == []
        assert sanitize_ruler_points([None, [2, 2]]) == []
        assert sanitize_ruler_points("nope") == []

    def test_numeric_strings_accepted(self):
        assert sanitize_ruler_points([["10", "20"], ["30", "40"]]) == \
            [(10.0, 20.0), (30.0, 40.0)]


class TestCoercePositiveFloat:
    def test_valid(self):
        assert coerce_positive_float(0.5) == 0.5
        assert coerce_positive_float("2.5") == 2.5

    def test_invalid_returns_none(self):
        assert coerce_positive_float(None) is None
        assert coerce_positive_float("abc") is None
        assert coerce_positive_float(0) is None
        assert coerce_positive_float(-1.5) is None
        assert coerce_positive_float(float("nan")) is None
        assert coerce_positive_float(float("inf")) is None
        assert coerce_positive_float([1]) is None


class TestPlanarColorSupport:
    def test_cyx_planar_rgb(self):
        # 修复前：被误判成 3 帧灰度堆叠（R/G/B 平面当 3 帧）
        arr = np.zeros((3, 100, 200), dtype=np.uint8)
        arr[0] = 255  # R 通道全红
        out, note = normalize_tiff_data(arr, axes="CYX")
        assert out.shape == (1, 100, 200)
        assert "平面格式" in note
        # 红色 → 灰度后应为 R 权重 0.2989*255 ≈ 76
        assert out[0, 0, 0] == pytest.approx(76, abs=2)

    def test_cyx_unsupported_channel_count_raises(self):
        arr = np.zeros((5, 100, 200), dtype=np.uint8)
        with pytest.raises(ValueError):
            normalize_tiff_data(arr, axes="CYX")

    def test_tyx_still_stack(self):
        # 帧数为 3 的灰度堆叠不受 planar 分支影响
        arr = np.zeros((3, 100, 200), dtype=np.uint8)
        out, _ = normalize_tiff_data(arr, axes="TYX")
        assert out.shape == (3, 100, 200)


class TestRgbToGrayInfHeuristic:
    def test_unit_range_with_inf_not_misjudged(self):
        # 修复前：+Inf 先被折算成 255 再参与 max 判断，
        # 整张 [0,1] 图被误判为 [0,255]，全部压成近 0
        img = np.full((4, 5, 3), 0.5, dtype=np.float32)
        img[0, 0, 0] = np.inf
        out = rgb_to_gray_uint8(img)
        assert out[0, 1] == pytest.approx(127, abs=2)   # 正常像素按 [0,1] 缩放
        assert out[0, 0] == 255                          # Inf → 白

    def test_0_255_range_with_inf_kept(self):
        img = np.full((4, 5, 3), 200.0, dtype=np.float32)
        img[0, 0, 0] = np.inf
        out = rgb_to_gray_uint8(img)
        assert out[0, 1] >= 198
        assert out[0, 0] == 255


class TestNormalizeTiffData:
    def test_2d(self):
        arr = np.zeros((10, 20), dtype=np.uint8)
        out, note = normalize_tiff_data(arr, axes="YX")
        assert out.shape == (1, 10, 20)

    def test_3d_gray_stack_w3_not_color(self):
        arr = np.zeros((20, 64, 3), dtype=np.uint8)
        out, note = normalize_tiff_data(arr, axes="TYX")
        assert out.shape == (20, 64, 3)

    def test_3d_color_single(self):
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 255, size=(32, 40, 3), dtype=np.uint8)
        out, note = normalize_tiff_data(arr, axes="YXC")
        assert out.shape == (1, 32, 40)
        assert out.dtype == np.uint8

    def test_4d_color_stack(self):
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 255, size=(6, 32, 40, 3), dtype=np.uint8)
        out, note = normalize_tiff_data(arr, axes="TYXC")
        assert out.shape == (6, 32, 40)
        assert out.dtype == np.uint8

    def test_unsupported_5d(self):
        with pytest.raises(ValueError):
            normalize_tiff_data(np.zeros((1, 2, 3, 4, 5)), axes="")
