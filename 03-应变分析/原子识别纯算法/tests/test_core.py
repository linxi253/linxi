from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

from atomic_core import (
    AtomRecord,
    AtomicToolError,
    DetectionParams,
    DetectionRegion,
    IntensityParams,
    MarkerRule,
    calibrate_from_points,
    close_tiff_stack,
    detect_atoms,
    link_frame_points,
    load_tiff_stack,
    make_records,
    measure_integrated_intensities,
    measure_intensities_with_diagnostics,
    normalize_for_display,
    normalize_tiff_array,
    parse_finite_float,
    percentile_ranks,
    recalculate_record_intensities,
    refine_clicked_point,
    refine_point_on_work,
    render_marked_frame,
    rule_for_percentile,
)

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



def gaussian_frame(shape, centers, sigma=1.2, background=10.0, amplitude=100.0):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    image = np.full(shape, background, dtype=np.float64)
    for x, y in centers:
        image += amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    return image


class TiffNormalizationTests(unittest.TestCase):
    def test_single_and_stack_shapes(self):
        single = np.zeros((5, 7), dtype=np.uint16)
        stack = np.zeros((3, 5, 7), dtype=np.uint16)
        self.assertEqual(normalize_tiff_array(single).shape, (1, 5, 7))
        self.assertEqual(normalize_tiff_array(stack, axes="TYX").shape, (3, 5, 7))

    def test_rgb_axes_are_converted_but_narrow_stack_is_not(self):
        rgb = np.zeros((5, 7, 3), dtype=np.uint8)
        narrow_stack = np.zeros((2, 5, 3), dtype=np.uint16)
        self.assertEqual(normalize_tiff_array(rgb, axes="YXS").shape, (1, 5, 7))
        self.assertEqual(normalize_tiff_array(narrow_stack, axes="TYX").shape, (2, 5, 3))

    def test_channel_first_rgb_axes_are_supported(self):
        rgb = np.zeros((3, 5, 7), dtype=np.uint8)
        self.assertEqual(normalize_tiff_array(rgb, axes="SYX").shape, (1, 5, 7))

    def test_load_stack_reports_frames_and_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stack.tif"
            source = np.arange(3 * 8 * 9, dtype=np.uint16).reshape(3, 8, 9)
            tifffile.imwrite(path, source, metadata={"axes": "TYX"}, photometric="minisblack")
            loaded, info = load_tiff_stack(path)
            self.assertEqual(loaded.shape, source.shape)
            self.assertEqual(info.stack_shape, source.shape)
            self.assertEqual(info.dtype, "uint16")
            np.testing.assert_array_equal(loaded, source)
            close_tiff_stack(loaded)


class DetectionTests(unittest.TestCase):
    def test_bright_grid_is_detected_at_subpixel_centers(self):
        centers = [(12.3, 13.2), (31.4, 14.1), (13.0, 34.2), (32.1, 35.0)]
        frame = gaussian_frame((48, 48), centers)
        points = detect_atoms(
            frame,
            DetectionParams(sigma=0.6, min_distance=9, window=7, threshold=None, bright=True, method="com"),
        )
        self.assertEqual(len(points), len(centers))
        for center in centers:
            self.assertLess(np.min(np.linalg.norm(points - np.asarray(center), axis=1)), 0.6)

    def test_dark_atoms_are_supported(self):
        bright = gaussian_frame((40, 40), [(12.2, 11.8), (28.1, 27.9)], background=0, amplitude=1)
        dark = bright.max() - bright
        points = detect_atoms(
            dark,
            DetectionParams(sigma=0.5, min_distance=8, window=7, threshold=None, bright=False, method="com"),
        )
        self.assertEqual(len(points), 2)

    def test_calibration_returns_valid_detection_and_aperture_parameters(self):
        frame = gaussian_frame((64, 64), [(15, 20), (27, 20), (39, 20)], sigma=1.6)
        result = calibrate_from_points(frame, [(15, 20), (27, 20), (39, 20)], bright=True)
        self.assertAlmostEqual(result.nearest_spacing, 12.0, places=5)
        result.detection_params.validated()
        result.intensity_params.validated()
        self.assertGreater(result.match_distance, 0)

    def test_rectangle_region_filters_candidates_before_detection(self):
        centers = [(10, 10), (30, 10), (10, 30), (30, 30)]
        frame = gaussian_frame((44, 44), centers)
        region = DetectionRegion("rectangle", ((0, 0), (20, 20)))
        points = detect_atoms(frame, DetectionParams(threshold=None), region=region)
        self.assertEqual(len(points), 1)
        self.assertLess(np.linalg.norm(points[0] - np.array([10, 10])), 0.6)

    def test_polygon_region_filters_candidates(self):
        centers = [(10, 10), (30, 10), (10, 30), (30, 30)]
        frame = gaussian_frame((44, 44), centers)
        triangle = DetectionRegion("polygon", ((2, 2), (38, 2), (2, 38)))
        points = detect_atoms(frame, DetectionParams(threshold=None), region=triangle)
        self.assertEqual(len(points), 3)
        self.assertFalse(np.any(np.linalg.norm(points - np.array([30, 30]), axis=1) < 1.0))


class IntegratedIntensityTests(unittest.TestCase):
    def test_local_background_subtracted_aperture_sum(self):
        frame = np.full((31, 31), 10.0)
        yy, xx = np.mgrid[0:31, 0:31]
        aperture = np.hypot(xx - 15, yy - 15) <= 2.2
        frame[aperture] = 15.0
        params = IntensityParams(2.2, 3.0, 5.0, True)
        value = measure_integrated_intensities(frame, [(15, 15)], params)[0]
        self.assertAlmostEqual(value, 5.0 * int(aperture.sum()), places=8)

    def test_dark_atom_integral_is_positive_with_dark_polarity(self):
        frame = np.full((31, 31), 20.0)
        yy, xx = np.mgrid[0:31, 0:31]
        aperture = np.hypot(xx - 15, yy - 15) <= 2.0
        frame[aperture] = 12.0
        value = measure_integrated_intensities(
            frame, [(15, 15)], IntensityParams(2.0, 3.0, 5.0, False)
        )[0]
        self.assertAlmostEqual(value, 8.0 * int(aperture.sum()), places=8)

    def test_percentiles_are_independent_rank_positions(self):
        ranks = percentile_ranks([10.0, 20.0, 30.0, 40.0])
        np.testing.assert_allclose(ranks, [12.5, 37.5, 62.5, 87.5])


class StableIdTests(unittest.TestCase):
    def test_ids_follow_positions_across_reordered_and_missing_frames(self):
        frames = [
            np.array([[5.0, 5.0], [20.0, 10.0], [35.0, 25.0]]),
            np.array([[35.2, 24.9], [5.2, 5.1], [20.1, 10.2]]),
            np.array([[5.1, 5.0], [35.1, 25.1]]),
            np.array([[20.2, 10.1], [35.0, 25.2], [5.0, 5.2]]),
        ]
        ids = link_frame_points(frames, max_distance=1.0)
        first_lookup = {tuple(point): int(atom_id) for point, atom_id in zip(frames[0], ids[0])}
        id_a = first_lookup[(5.0, 5.0)]
        id_b = first_lookup[(20.0, 10.0)]
        id_c = first_lookup[(35.0, 25.0)]
        self.assertEqual(int(ids[1][1]), id_a)
        self.assertEqual(int(ids[1][2]), id_b)
        self.assertEqual(int(ids[1][0]), id_c)
        self.assertEqual(int(ids[3][0]), id_b)  # 中间漏一帧后仍恢复原 ID

    def test_out_of_gate_track_does_not_steal_near_match(self):
        frames = [
            np.array([[0.0, 0.0], [0.0, 7.0]]),
            np.array([[0.0, 1.0], [0.0, 2.0]]),
        ]
        ids = link_frame_points(frames, max_distance=5.0)
        self.assertEqual(int(ids[1][0]), int(ids[0][0]))


class MarkerRuleTests(unittest.TestCase):
    def test_only_selected_percentile_range_is_rendered(self):
        frame = np.zeros((40, 40), dtype=np.uint16)
        records = [
            AtomRecord(1, 10.0, 10.0, 1.0, 15.0),
            AtomRecord(2, 30.0, 30.0, 2.0, 75.0),
        ]
        rules = [MarkerRule("10-20", 10, 20, "#ff0000", "circle", True)]
        rendered = render_marked_frame(frame, records, rules, marker_radius=5, show_ids=False)
        self.assertEqual(rendered.shape, (40, 40, 3))
        self.assertGreater(int(rendered[5:16, 5:16, 0].max()), 0)
        self.assertEqual(int(rendered[25:36, 25:36].max()), 0)
        self.assertIsNotNone(rule_for_percentile(15.0, rules))
        self.assertIsNone(rule_for_percentile(75.0, rules))


class ThresholdValidationTests(unittest.TestCase):
    def test_absolute_threshold_outside_normalized_domain_is_rejected(self):
        # 0–1 之间一律按分位数解释；亮原子没有独立于分位数的绝对阈值区间。
        with self.assertRaises(AtomicToolError):
            DetectionParams(threshold=1.5).validated()
        with self.assertRaises(AtomicToolError):
            DetectionParams(threshold=1.0).validated()
        with self.assertRaises(AtomicToolError):
            DetectionParams(threshold=0.0).validated()  # v1.3 起亮原子 0 也拒绝（匹配一切像素）
        with self.assertRaises(AtomicToolError):
            DetectionParams(threshold=-0.5).validated()  # 亮原子负绝对值同样无意义
        # 暗原子：检测图取负后值域 -1–0，仅 -1–0 之间的绝对值合法。
        with self.assertRaises(AtomicToolError):
            DetectionParams(threshold=1.2, bright=False).validated()
        with self.assertRaises(AtomicToolError):
            DetectionParams(threshold=0.0, bright=False).validated()
        with self.assertRaises(AtomicToolError):
            DetectionParams(threshold=-1.0, bright=False).validated()
        # 分位数语义（0–1 之间，与极性无关）与暗原子值域内绝对值仍然合法。
        DetectionParams(threshold=0.6).validated()
        DetectionParams(threshold=0.6, bright=False).validated()
        DetectionParams(threshold=-0.5, bright=False).validated()

    def test_threshold_error_messages_explain_polarity_semantics(self):
        for bright in (True, False):
            with self.assertRaises(AtomicToolError) as context:
                DetectionParams(threshold=1.5, bright=bright).validated()
            message = str(context.exception)
            self.assertIn("分位数", message)
            self.assertIn("亮原子" if bright else "暗原子", message)


class TruncationSymmetryTests(unittest.TestCase):
    """截断标志必须对四条边使用同一像素中心标准，不允许左右不对称。"""

    PARAMS = IntensityParams(2.0, 3.0, 4.0, True)

    def _flag(self, frame, x, y):
        return measure_intensities_with_diagnostics(frame, [(x, y)], self.PARAMS)[0].truncated

    def test_disk_crossing_any_edge_by_less_than_1px_is_flagged(self):
        frame = np.full((40, 40), 10.0)
        # 越过左边界 0.9 px（v1.2 曾因 1px 容差漏标）。
        self.assertTrue(self._flag(frame, 3.1, 20.0))
        # 越过右边界 0.9 px。
        self.assertTrue(self._flag(frame, 35.9, 20.0))
        # 越过上/下边界 0.9 px。
        self.assertTrue(self._flag(frame, 20.0, 3.1))
        self.assertTrue(self._flag(frame, 20.0, 35.9))

    def test_disk_fully_inside_or_exactly_touching_is_not_flagged(self):
        frame = np.full((40, 40), 10.0)
        self.assertFalse(self._flag(frame, 4.9, 20.0))
        self.assertFalse(self._flag(frame, 34.9, 20.0))  # 34.9 + 4 = 38.9 <= 39
        # 恰好碰到最后一个像素中心（x + outer == width - 1）不算截断。
        self.assertFalse(self._flag(frame, 35.0, 20.0))


class NeighborContaminationTests(unittest.TestCase):
    PARAMS = IntensityParams(2.0, 3.0, 4.0, True)

    def test_close_neighbor_is_flagged_with_distance(self):
        frame = np.full((60, 60), 10.0)
        yy, xx = np.mgrid[0:60, 0:60]
        frame += 90.0 * np.exp(-((xx - 30.0) ** 2 + (yy - 30.0) ** 2) / 2.0)
        frame += 50.0 * np.exp(-((xx - 33.5) ** 2 + (yy - 30.0) ** 2) / 2.0)
        samples = measure_intensities_with_diagnostics(
            frame, [(30.0, 30.0), (33.5, 30.0)], self.PARAMS
        )
        # 污染阈值 = 孔径 2.0 + 外径 4.0 = 6.0；间距 3.5 必然命中。
        for sample in samples:
            self.assertTrue(sample.neighbor_contaminated)
            self.assertAlmostEqual(sample.nearest_neighbor, 3.5, places=6)

    def test_distant_or_absent_neighbors_are_clean(self):
        frame = np.full((60, 60), 10.0)
        samples = measure_intensities_with_diagnostics(frame, [(30.0, 30.0)], self.PARAMS)
        self.assertEqual(samples[0].nearest_neighbor, np.inf)
        self.assertFalse(samples[0].neighbor_contaminated)
        samples = measure_intensities_with_diagnostics(
            frame, [(20.0, 30.0), (40.0, 30.0)], self.PARAMS
        )
        for sample in samples:
            self.assertAlmostEqual(sample.nearest_neighbor, 20.0, places=6)
            self.assertFalse(sample.neighbor_contaminated)


class TrackAgingTests(unittest.TestCase):
    def test_track_recovers_within_max_missed_frames(self):
        frames = [np.array([[5.0, 5.0]])]
        frames.extend([np.empty((0, 2)) for _ in range(3)])  # 连续漏检 3 帧
        frames.append(np.array([[5.1, 5.0]]))
        ids = link_frame_points(frames, max_distance=1.0, max_missed=8)
        self.assertEqual(int(ids[-1][0]), int(ids[0][0]))

    def test_stale_track_is_pruned_after_max_missed_frames(self):
        frames = [np.array([[5.0, 5.0]])]
        frames.extend([np.empty((0, 2)) for _ in range(9)])  # 超过 max_missed=8
        frames.append(np.array([[5.1, 5.0]]))
        ids = link_frame_points(frames, max_distance=1.0, max_missed=8)
        self.assertNotEqual(int(ids[-1][0]), int(ids[0][0]))

    def test_max_missed_must_be_positive(self):
        with self.assertRaises(AtomicToolError):
            link_frame_points([np.empty((0, 2))], max_distance=1.0, max_missed=0)


class LocalRefineTests(unittest.TestCase):
    def test_local_roi_refine_matches_full_image_refine(self):
        centers = [(30.7, 41.3), (12.2, 15.9)]
        frame = gaussian_frame((96, 96), centers, sigma=1.4)
        params = DetectionParams(sigma=0.6, min_distance=9, window=7, method="com")
        for x, y in centers:
            click_x, click_y = x + 0.8, y - 0.6
            full = refine_clicked_point(frame, click_x, click_y, params)
            work = normalize_for_display(frame)
            local = refine_point_on_work(work, click_x, click_y, params)
            # 局部 ROI 与整图路径必须给出同一亚像素位置。
            self.assertLess(np.hypot(full[0] - local[0], full[1] - local[1]), 1e-9)
            click_error = np.hypot(click_x - x, click_y - y)
            refine_error = np.hypot(local[0] - x, local[1] - y)
            self.assertLess(
                refine_error,
                click_error,
                "COM 精炼应比原始点击更接近真实峰心（窗口偏离峰心时 COM 存在固有偏置，"
                "此处只要求单调改进，不设绝对容差）",
            )

    def test_parse_finite_float_gives_field_level_chinese_errors(self):
        self.assertEqual(parse_finite_float(" 2.5 ", "积分孔径半径"), 2.5)
        for bad in ("abc", "", "1,5"):
            with self.assertRaises(AtomicToolError) as context:
                parse_finite_float(bad, "积分孔径半径")
            self.assertIn("积分孔径半径", str(context.exception))
        with self.assertRaises(AtomicToolError):
            parse_finite_float("inf", "积分孔径半径")


class IntensityDiagnosticsTests(unittest.TestCase):
    def test_center_atom_reports_pixels_and_background_without_truncation(self):
        frame = np.full((31, 31), 10.0)
        yy, xx = np.mgrid[0:31, 0:31]
        aperture = np.hypot(xx - 15, yy - 15) <= 2.2
        frame[aperture] = 15.0
        sample = measure_intensities_with_diagnostics(
            frame, [(15.0, 15.0)], IntensityParams(2.2, 3.0, 5.0, True)
        )[0]
        self.assertFalse(sample.truncated)
        self.assertEqual(sample.aperture_pixels, int(aperture.sum()))
        self.assertGreater(sample.background_pixels, 0)
        self.assertAlmostEqual(sample.background_level, 10.0, places=8)
        self.assertAlmostEqual(sample.intensity, 5.0 * int(aperture.sum()), places=8)

    def test_edge_atom_is_flagged_truncated_but_still_measured(self):
        frame = np.full((31, 31), 10.0)
        yy, xx = np.mgrid[0:31, 0:31]
        frame[np.hypot(xx - 1, yy - 15) <= 2.0] = 15.0
        sample = measure_intensities_with_diagnostics(
            frame, [(1.0, 15.0)], IntensityParams(2.0, 3.0, 5.0, True)
        )[0]
        self.assertTrue(sample.truncated)
        self.assertTrue(np.isfinite(sample.intensity))
        # 与延伸到图像外的理想圆盘对比：孔径被边界裁掉了一部分像素。
        yy_full, xx_full = np.mgrid[-4:35, -4:35]
        ideal = np.hypot(xx_full - 1, yy_full - 15) <= 2.0
        self.assertLess(sample.aperture_pixels, int(ideal.sum()))

    def test_make_records_carries_qc_fields_and_preserves_source_on_recalc(self):
        frame = np.full((31, 31), 10.0)
        records = make_records(
            frame,
            np.array([[15.0, 15.0], [1.0, 15.0]]),
            [1, 2],
            IntensityParams(2.0, 3.0, 5.0, True),
        )
        self.assertFalse(records[0].truncated)
        self.assertTrue(records[1].truncated)
        self.assertEqual(records[0].source, "auto")
        records[0].source = "manual"
        recalculated = recalculate_record_intensities(frame, records, IntensityParams(2.0, 3.0, 5.0, True))
        self.assertEqual(recalculated[0].source, "manual")
        self.assertEqual(recalculated[1].truncated, True)

    def test_legacy_intensity_function_matches_diagnostics(self):
        frame = np.full((31, 31), 10.0)
        legacy = measure_integrated_intensities(
            frame, [(15.0, 15.0), (1.0, 15.0)], IntensityParams(2.0, 3.0, 5.0, True)
        )
        self.assertEqual(legacy.shape, (2,))


class GaussianFallbackTests(unittest.TestCase):
    def test_gaussian_method_falls_back_to_batched_com_when_fit_fails(self):
        centers = [(12.3, 13.2), (31.4, 14.1)]
        frame = gaussian_frame((48, 48), centers)
        params = DetectionParams(
            sigma=0.6, min_distance=9, window=7, threshold=None, bright=True, method="gaussian"
        )
        with patch("atomic_core.curve_fit", side_effect=RuntimeError("fit failed")):
            points = detect_atoms(frame, params)
        self.assertEqual(len(points), len(centers))
        for center in centers:
            self.assertLess(np.min(np.linalg.norm(points - np.asarray(center), axis=1)), 0.8)


class CalibrationSamplingTests(unittest.TestCase):
    def test_sample_count_counts_usable_rois(self):
        frame = gaussian_frame((64, 64), [(15, 20), (27, 20), (39, 20)], sigma=1.6)
        result = calibrate_from_points(frame, [(15, 20), (27, 20), (39, 20)], bright=True)
        self.assertEqual(result.sample_count, 3)

    def test_corner_clicks_are_all_excluded_and_defaults_remain_valid(self):
        # 两个点击点都贴图像角：ROI 峰值贴角（max_radius<2）全部被剔除，
        # 此时 sample_count=0、峰值未测得，但默认参数仍有效不崩溃。
        centers = [(0.5, 0.5), (2.0, 2.0)]
        frame = gaussian_frame((64, 64), centers, sigma=1.2)
        result = calibrate_from_points(frame, centers, bright=True)
        self.assertEqual(result.sample_count, 0)
        self.assertFalse(np.isfinite(result.peak_min))
        result.detection_params.validated()
        result.intensity_params.validated()
        self.assertGreater(result.match_distance, 0)


if __name__ == "__main__":
    unittest.main()
