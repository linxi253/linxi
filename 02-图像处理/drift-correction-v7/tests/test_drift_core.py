# -*- coding: utf-8 -*-
import csv
import hashlib
import json
import tempfile
import threading
import unittest
from datetime import datetime, timedelta
from unittest import mock
from pathlib import Path

import cv2
import numpy as np
import tifffile

from drift_core import (
    DEFAULT_SIFT_PARAMS,
    DetectionQualityError,
    DetectionResult,
    OperationCancelled,
    PairQuality,
    DriftCorrector,
    DriftDetector,
    DriftResult,
    MAX_STACK_BYTES,
    TiffIO,
    __version__,
    _csv_safe,
    _match_pair,
    _opencv_single_thread,
    _report_paths,
    _sift_params,
    batch_process,
    correct_and_save,
    total_physical_memory,
    write_shift_table,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pages(path: Path, frames: list[np.ndarray]) -> None:
    with tifffile.TiffWriter(path) as writer:
        for frame in frames:
            writer.write(frame, photometric="minisblack", metadata=None)


class DriftCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="drift-core-test-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_write_is_atomic_when_later_frame_is_invalid(self):
        source = self.root / "source.tif"
        _write_pages(source, [np.zeros((16, 16), np.uint8)] * 2)
        target = self.root / "existing.tif"
        tifffile.imwrite(target, np.full((16, 16), 91, np.uint8))
        before = _hash(target)
        meta = {"source_path": str(source), "dtype": np.dtype("uint8"), "n_frames": 2}
        with self.assertRaises(ValueError):
            TiffIO.write_stack(target, [np.zeros((16, 16), np.uint8), np.zeros((8, 8), np.uint8)],
                               meta, overwrite=True)
        self.assertTrue(target.exists())
        self.assertEqual(before, _hash(target))

    def test_refuses_to_overwrite_source(self):
        source = self.root / "source.tif"
        _write_pages(source, [np.zeros((8, 8), np.uint16)] * 2)
        frames, meta = TiffIO.read_stack(source)
        before = _hash(source)
        with self.assertRaises(ValueError):
            correct_and_save(frames, np.zeros(2), np.zeros(2), meta, str(source))
        self.assertEqual(before, _hash(source))

    def test_cancelled_write_keeps_existing_target(self):
        source = self.root / "source.tif"
        _write_pages(source, [np.zeros((8, 8), np.uint8)] * 2)
        frames, meta = TiffIO.read_stack(source)
        target = self.root / "existing.tif"
        tifffile.imwrite(target, np.full((8, 8), 23, np.uint8))
        before = _hash(target)
        cancelled = threading.Event()
        cancelled.set()
        self.assertIsNone(correct_and_save(frames, np.zeros(2), np.zeros(2), meta, str(target),
                                           overwrite=True, cancel_event=cancelled))
        self.assertEqual(before, _hash(target))

    def test_output_is_one_series_and_keeps_uint16(self):
        source = self.root / "source.tif"
        frames = [np.full((24, 20), value, np.uint16) for value in (1, 1000, 65000, 42)]
        _write_pages(source, frames)
        loaded, meta = TiffIO.read_stack(source)
        output = self.root / "corrected.tif"
        self.assertEqual(correct_and_save(loaded, np.zeros(4), np.zeros(4), meta, str(output)), 4)
        with tifffile.TiffFile(output) as tif:
            self.assertEqual(len(tif.pages), 4)
            self.assertEqual(len(tif.series), 1)
            self.assertEqual(tif.series[0].shape, (4, 24, 20))
            self.assertEqual(np.dtype(tif.pages[0].dtype), np.dtype("uint16"))

    def test_mixed_page_shape_is_rejected(self):
        source = self.root / "mixed.tif"
        with tifffile.TiffWriter(source) as writer:
            writer.write(np.zeros((8, 8), np.uint8), metadata=None)
            writer.write(np.zeros((9, 8), np.uint8), metadata=None)
        with self.assertRaises(ValueError):
            TiffIO.inspect_stack(source)

    def test_blank_stack_is_detection_failure_not_zero_drift(self):
        blank = [np.zeros((64, 64), np.float32) for _ in range(4)]
        with self.assertRaises(DetectionQualityError):
            DriftDetector.detect_drift(blank, use_parallel=False)

    def test_float_zero_to_one_stack_detects_translation(self):
        rng = np.random.default_rng(20260729)
        base = rng.random((192, 192), dtype=np.float32)
        for _ in range(36):
            x, y = rng.integers(12, 180, 2)
            cv2.circle(base, (int(x), int(y)), int(rng.integers(2, 7)), float(rng.random()), -1)
        truth_x = np.array([0.0, 2.25, 4.75, 7.0], dtype=np.float32)
        truth_y = np.array([0.0, -1.25, -2.5, -4.0], dtype=np.float32)
        frames = []
        for dx, dy in zip(truth_x, truth_y):
            matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
            frames.append(cv2.warpAffine(base, matrix, (192, 192), borderValue=0))
        result = DriftDetector.detect_drift(frames, use_parallel=False)
        self.assertLess(float(np.max(np.abs(result.shifts_x - truth_x))), 0.15)
        self.assertLess(float(np.max(np.abs(result.shifts_y - truth_y))), 0.15)

    def test_detect_correct_save_end_to_end_reduces_known_drift(self):
        rng = np.random.default_rng(20260829)
        base = rng.integers(0, 256, (192, 192), dtype=np.uint8)
        for _ in range(48):
            x, y = rng.integers(16, 176, 2)
            cv2.circle(base, (int(x), int(y)), int(rng.integers(2, 7)),
                       int(rng.integers(0, 256)), -1)
        truth_x = np.arange(9, dtype=np.float32) * 4.0
        truth_y = np.arange(9, dtype=np.float32) * -2.0
        frames = [
            cv2.warpAffine(
                base, np.array([[1, 0, dx], [0, 1, dy]], np.float32),
                (192, 192), flags=cv2.INTER_LINEAR, borderValue=0)
            for dx, dy in zip(truth_x, truth_y)
        ]
        source = self.root / "known-drift.tif"
        target = self.root / "known-drift-corrected.tif"
        _write_pages(source, frames)
        loaded, meta = TiffIO.read_stack(source)
        result = DriftDetector.detect_drift(loaded, skip_interval=4, use_parallel=False)
        written = correct_and_save(
            loaded, result.shifts_x, result.shifts_y, meta, str(target), crop_mode="crop")
        corrected, corrected_meta = TiffIO.read_stack(target)

        self.assertEqual(written, len(frames))
        self.assertEqual(len(corrected), len(frames))
        self.assertLess(float(np.max(np.abs(result.shifts_x - truth_x))), 0.25)
        self.assertLess(float(np.max(np.abs(result.shifts_y - truth_y))), 0.25)
        left, top, right, bottom = DriftCorrector.common_valid_crop(
            base.shape, result.shifts_x, result.shifts_y)
        reference = base[top:bottom, left:right].astype(np.float32)
        raw_error = float(np.mean([
            np.mean(np.abs(frame[top:bottom, left:right].astype(np.float32) - reference))
            for frame in loaded[1:]
        ]))
        corrected_error = float(np.mean([
            np.mean(np.abs(frame.astype(np.float32) - reference))
            for frame in corrected[1:]
        ]))
        self.assertEqual(corrected_meta["shape"], reference.shape)
        self.assertLess(corrected_error, raw_error * 0.35)

    def test_common_crop_and_integer_dtype(self):
        frames = [np.arange(100, dtype=np.uint16).reshape(10, 10)] * 3
        crop = DriftCorrector.common_valid_crop((10, 10), np.array([0.0, -2.0, -3.0]), np.zeros(3))
        self.assertEqual(crop, (3, 0, 10, 10))
        corrected = list(DriftCorrector.correct_frames(frames, np.array([0.0, -2.0, -3.0]),
                                                        np.zeros(3), crop=crop))
        self.assertTrue(all(frame.dtype == np.uint16 for frame in corrected))
        self.assertTrue(all(frame.shape == (10, 7) for frame in corrected))

    def test_common_valid_crop_all_positive_shifts(self):
        """全正偏移时，有效区是 x < width - max(sx) 的区域。"""
        width, height = 10, 12
        shifts_x = np.array([1.0, 2.0, 3.0])
        shifts_y = np.array([1.0, 1.5, 2.0])
        left, top, right, bottom = DriftCorrector.common_valid_crop(
            (height, width), shifts_x, shifts_y)
        self.assertEqual((left, top, right, bottom), (0, 0, 7, 10))
        # 有效区即所有帧 warpAffine 后都被原图覆盖的区域：
        # 对有效区内的每个输出像素，反推源坐标 x+dx 必须在 [0, width) 内。
        for x in range(left, right):
            for dx in shifts_x:
                self.assertGreaterEqual(x + dx, 0)
                self.assertLess(x + dx, width)
        for y in range(top, bottom):
            for dy in shifts_y:
                self.assertGreaterEqual(y + dy, 0)
                self.assertLess(y + dy, height)

    def test_common_valid_crop_mixed_sign_shifts(self):
        """正负混合偏移时，有效区两侧都要收缩。"""
        width, height = 10, 10
        shifts_x = np.array([-2.0, 3.0])
        shifts_y = np.array([1.0, -1.0])
        left, top, right, bottom = DriftCorrector.common_valid_crop(
            (height, width), shifts_x, shifts_y)
        self.assertEqual((left, top, right, bottom), (2, 1, 7, 9))
        for x in range(left, right):
            for dx in shifts_x:
                self.assertGreaterEqual(x + dx, 0)
                self.assertLess(x + dx, width)
        for y in range(top, bottom):
            for dy in shifts_y:
                self.assertGreaterEqual(y + dy, 0)
                self.assertLess(y + dy, height)

    def test_common_valid_crop_raises_when_no_common_area(self):
        """偏移过大导致有效区为空时必须抛错。"""
        with self.assertRaises(ValueError):
            DriftCorrector.common_valid_crop((10, 10), np.array([-6.0, 5.0]))
        with self.assertRaises(ValueError):
            DriftCorrector.common_valid_crop((10, 10), np.array([0.0]), np.array([-6.0, 5.0]))

    def test_batch_uses_absolute_path_cache_and_writes_audit_report(self):
        source = self.root / "source.tif"
        _write_pages(source, [np.full((12, 12), value, np.uint8) for value in (1, 2)])
        frames, meta = TiffIO.read_stack(source)
        output = self.root / "output"
        key = str(source.resolve())
        result = batch_process([str(source)], str(output), precomputed={
            key: {"frames": frames, "shifts_x": np.zeros(2), "shifts_y": np.zeros(2),
                  "meta": meta},
        })
        self.assertEqual(result["completed"], 1)
        self.assertFalse(result["cancelled"])
        self.assertEqual(result["records"][0]["shift_source"], "precomputed")
        self.assertTrue((output / "source_corrected.tif").is_file())
        self.assertTrue(Path(result["report_json"]).is_file())
        self.assertTrue(Path(result["report_csv"]).is_file())

    def test_batch_precomputed_cache_structure_mismatch_is_rejected(self):
        source = self.root / "source.tif"
        _write_pages(source, [np.full((12, 12), value, np.uint8) for value in (1, 2)])
        frames, meta = TiffIO.read_stack(source)
        key = str(source.resolve())
        bad_meta = dict(meta)
        bad_meta["n_frames"] = 3  # 与磁盘实际 2 帧不一致
        result = batch_process([str(source)], str(self.root / "output"), precomputed={
            key: {"frames": frames, "shifts_x": np.zeros(2), "shifts_y": np.zeros(2),
                  "meta": bad_meta},
        })
        self.assertEqual(result["completed"], 0)
        self.assertEqual(len(result["errors"]), 1)
        self.assertIn("precomputed 缓存与磁盘 TIFF 结构不一致", result["errors"][0][1])

    def test_batch_failure_is_not_reported_as_cancelled(self):
        broken = self.root / "broken.tif"
        broken.write_bytes(b"not a tiff")
        result = batch_process([str(broken)], str(self.root / "output"))
        self.assertEqual(result["completed"], 0)
        self.assertEqual(len(result["errors"]), 1)
        self.assertFalse(result["cancelled"])

    def test_batch_cancellation_is_reported(self):
        source = self.root / "source.tif"
        _write_pages(source, [np.full((12, 12), value, np.uint8) for value in (1, 2)])
        cancelled = threading.Event()
        cancelled.set()
        result = batch_process([str(source)], str(self.root / "output"),
                               cancel_event=cancelled)
        self.assertTrue(result["cancelled"])
        self.assertEqual(result["completed"], 0)

    def test_quality_gate_parameters_are_honoured(self):
        rng = np.random.default_rng(4)
        base = rng.random((128, 128), dtype=np.float32)
        for _ in range(24):
            x, y = rng.integers(10, 118, 2)
            cv2.circle(base, (int(x), int(y)), int(rng.integers(2, 6)), float(rng.random()), -1)
        frames = []
        for dx in (0.0, 1.5, 3.0, 4.5):
            matrix = np.array([[1, 0, dx], [0, 1, 0]], dtype=np.float32)
            frames.append(cv2.warpAffine(base, matrix, (128, 128), borderValue=0))
        # 默认门控下可靠
        DriftDetector.detect_drift(frames, use_parallel=False)
        # 收紧门控必须让检测显式失败，证明参数确实传导到匹配层。
        # min_inlier_ratio 不在此列：无噪合成数据的内点比例恰为 1.0，
        # 而 (0, 1] 内任何阈值都无法使 `ratio < threshold` 成立，其传导由
        # _sift_params 的解析测试覆盖。
        for override in ({"max_pair_residual": 1e-6}, {"min_inliers": 100000}):
            with self.assertRaises(DetectionQualityError):
                DriftDetector.detect_drift(frames, use_parallel=False, params=override)
        self.assertEqual(_sift_params({"min_inlier_ratio": 0.42})[1]["min_inlier_ratio"], 0.42)

    def test_min_matches_above_min_inliers_is_accepted(self):
        """min_inliers < min_matches 是合法配置（内点必然不多于匹配点）。"""
        sift, quality = _sift_params({"min_matches": 8})
        self.assertEqual(quality["min_matches"], 8)
        self.assertEqual(quality["min_inliers"], DEFAULT_SIFT_PARAMS["min_inliers"])
        for bad in ({"min_matches": 2}, {"min_inliers": 2},
                    {"noctave_layers": 0}, {"max_interpolation_gap": -1}):
            with self.assertRaises(ValueError):
                _sift_params(bad)

    def test_correct_frames_is_lazy_and_length_checked(self):
        frames = [np.full((8, 8), 3, np.uint8) for _ in range(4)]
        consumed = []

        def counting():
            for index, frame in enumerate(frames):
                consumed.append(index)
                yield frame

        iterator = DriftCorrector.correct_frames(counting(), np.zeros(4), np.zeros(4))
        next(iterator)
        self.assertEqual(consumed, [0])  # 未预先物化整个堆栈
        self.assertEqual(len(list(DriftCorrector.correct_frames(
            (frame for frame in frames), np.zeros(4), np.zeros(4)))), 4)
        for shift_count in (3, 5):
            with self.assertRaises(ValueError):
                list(DriftCorrector.correct_frames(
                    (frame for frame in frames), np.zeros(shift_count), np.zeros(shift_count)))

    def test_detection_result_requires_attribute_access(self):
        frames = [np.zeros((8, 8), np.uint8)]
        result = DriftDetector.detect_drift(frames, use_parallel=False)
        self.assertEqual(result.shifts_x.shape, (1,))
        with self.assertRaises(TypeError):
            _shifts_x, _shifts_y = result  # 旧的元组解包契约已移除

    def test_drift_result_alias_is_detection_result(self):
        """v6 的 DriftResult 名称保留为兼容别名，契约与 v5.2 一致。"""
        self.assertIs(DriftResult, DetectionResult)

    def test_match_pair_symmetric_clusters_fail_not_choose_arbitrarily(self):
        """规模相同、方向冲突的双峰匹配应显式失败。

        双峰保护依赖紧的 RANSAC 阈值；默认阈值已放宽到 ImageJ 水平（25 px），
        在该宽容度下 RANSAC 会把两簇合并为近似零位移（与 ImageJ 行为一致），
        因此这里显式注入紧阈值以覆盖保护机制本身。
        """
        rng = np.random.default_rng(20260731)
        n = 20
        points_a = np.column_stack([
            rng.uniform(0, 100, n), rng.uniform(0, 100, n),
        ]).astype(np.float32)
        half = n // 2
        delta = np.empty((n, 2), dtype=np.float32)
        delta[:half] = (3.0, -2.0)
        delta[half:] = (-3.0, 2.0)
        points_b = points_a + delta
        desc_a = rng.standard_normal((n, 128)).astype(np.float32)
        desc_b = desc_a.copy()  # 自匹配距离 0，保证 n 对 good matches
        _, quality = _sift_params({"ransac_threshold": 3.0, "max_pair_residual": 3.0})
        result = _match_pair(
            (0, 1), {0: (points_a, desc_a), 1: (points_b, desc_b)}, quality)
        self.assertEqual(result.good_matches, n)
        self.assertNotEqual(result.status, "reliable")
        self.assertIsNone(result.dx)  # 不得给出 (0,0) 的系统偏差结果

    def test_match_pair_ransac_recovers_largest_consistent_cluster(self):
        # 簇间最近距离 23~26 px，只有紧 RANSAC 阈值才能把四簇分开
        # （默认 25 px 会把多簇合并，与 ImageJ 宽容行为一致）。
        rng = np.random.default_rng(20260829)
        cluster_sizes = (12, 8, 8, 8)
        translations = ((12.0, -5.0), (-12.0, 5.0), (0.0, 15.0), (0.0, -15.0))
        n = sum(cluster_sizes)
        points_a = np.column_stack([
            rng.uniform(0, 200, n), rng.uniform(0, 200, n),
        ]).astype(np.float32)
        deltas = np.concatenate([
            np.repeat(np.asarray([translation], np.float32), size, axis=0)
            for size, translation in zip(cluster_sizes, translations)
        ])
        points_b = points_a + deltas
        desc_a = rng.standard_normal((n, 128)).astype(np.float32)
        desc_b = desc_a.copy()
        _, quality = _sift_params({"ransac_threshold": 3.0, "max_pair_residual": 3.0})
        result = _match_pair(
            (0, 1), {0: (points_a, desc_a), 1: (points_b, desc_b)}, quality)
        self.assertEqual(result.status, "reliable")
        self.assertAlmostEqual(result.dx, 12.0, places=3)
        self.assertAlmostEqual(result.dy, -5.0, places=3)

    def test_match_pair_pure_translation_estimates_robust_median(self):
        rng = np.random.default_rng(11)
        n = 24
        points_a = np.column_stack([
            rng.uniform(0, 100, n), rng.uniform(0, 100, n),
        ]).astype(np.float32)
        # 大部分点平移 (3, -2)，少量离群点平移 (30, 30)
        points_b = points_a + np.array([3.0, -2.0], dtype=np.float32)
        points_b[:2] += np.array([27.0, 32.0], dtype=np.float32)
        desc_a = rng.standard_normal((n, 128)).astype(np.float32)
        desc_b = desc_a.copy()
        _, quality = _sift_params({})
        result = _match_pair(
            (0, 1), {0: (points_a, desc_a), 1: (points_b, desc_b)}, quality)
        self.assertEqual(result.status, "reliable")
        self.assertAlmostEqual(result.dx, 3.0, places=3)
        self.assertAlmostEqual(result.dy, -2.0, places=3)

    def test_reliable_nonuniform_motion_is_accumulated_without_velocity_cleaning(self):
        """真实 TEM 阶跃漂移已通过帧对质量门控，不能再被全局统计抹掉。"""
        increments_x = np.array([0.1, 0.0, 18.0, -2.5, 0.2, 7.0], np.float64)
        increments_y = np.array([0.0, 0.2, 35.0, 0.1, -4.0, 9.0], np.float64)
        base = np.arange(64, dtype=np.uint8).reshape(8, 8)
        frames = [base.copy() for _ in range(len(increments_x) + 1)]

        def fake_match(pair, features, quality):
            source, target = pair
            return PairQuality(
                source_index=source, target_index=target,
                keypoints_source=20, keypoints_target=20,
                good_matches=20, inliers=20, inlier_ratio=1.0,
                median_residual=0.0, dx=float(increments_x[source]),
                dy=float(increments_y[source]), status="reliable")

        with mock.patch("drift_core._extract_features", return_value=(None, None)), \
                mock.patch("drift_core._match_pair", side_effect=fake_match):
            result = DriftDetector.detect_drift(frames, use_parallel=False)

        self.assertTrue(np.allclose(result.shifts_x, np.r_[0.0, np.cumsum(increments_x)]))
        self.assertTrue(np.allclose(result.shifts_y, np.r_[0.0, np.cumsum(increments_y)]))

    def test_nonzero_correction_that_changes_no_pixels_is_rejected_atomically(self):
        source = self.root / "source-noop.tif"
        frames = [np.arange(256, dtype=np.uint8).reshape(16, 16) for _ in range(2)]
        _write_pages(source, frames)
        loaded, meta = TiffIO.read_stack(source)
        target = self.root / "existing-noop.tif"
        tifffile.imwrite(target, np.full((8, 8), 73, np.uint8))
        before = _hash(target)

        with mock.patch.object(
                DriftCorrector, "correct_single_frame",
                side_effect=lambda frame, *args, **kwargs: np.asarray(frame)):
            with self.assertRaisesRegex(ValueError, "拒绝保存未生效"):
                correct_and_save(
                    loaded, np.array([0.0, 3.0]), np.zeros(2), meta,
                    str(target), overwrite=True, crop_mode="crop")

        self.assertEqual(before, _hash(target))

    def test_correct_frames_list_keeps_v6_api(self):
        frames = [np.full((8, 8), 7, np.uint8) for _ in range(3)]
        out = DriftCorrector.correct_frames_list(frames, np.zeros(3), np.zeros(3))
        self.assertEqual(len(out), 3)
        self.assertTrue(all(frame.dtype == np.uint8 for frame in out))

    def test_corrector_border_value_is_applied(self):
        frame = np.full((16, 16), 5, np.uint8)
        corrected = DriftCorrector.correct_single_frame(
            frame, 2.0, 0.0, border_value=9)
        self.assertEqual(int(corrected[0, -1]), 9)

    def test_opencv_thread_count_is_restored(self):
        before = cv2.getNumThreads()
        with _opencv_single_thread():
            self.assertEqual(cv2.getNumThreads(), 1)
        self.assertEqual(cv2.getNumThreads(), before)
        with self.assertRaises(RuntimeError):
            with _opencv_single_thread():
                raise RuntimeError("boom")
        self.assertEqual(cv2.getNumThreads(), before)

    def test_stack_has_no_default_total_size_limit(self):
        self.assertIsNone(MAX_STACK_BYTES)

    def test_explicit_memory_limit_is_still_enforced(self):
        source = self.root / "source.tif"
        _write_pages(source, [np.zeros((32, 32), np.uint8)] * 4)
        with self.assertRaises(MemoryError) as caught:
            TiffIO.read_stack(source, max_bytes=16)
        self.assertIn("调用方设置", str(caught.exception))

    # ---- v7.0.4：C/Z 轴确认防线 ----

    def _write_imagej(self, path: Path, data: np.ndarray, axes: str) -> None:
        tifffile.imwrite(path, data, imagej=True,
                         metadata={'axes': axes, 'mode': 'grayscale'})

    def test_imagej_multichannel_requires_axis_confirmation(self):
        source = self.root / "tc.tif"
        self._write_imagej(source, np.zeros((4, 3, 16, 16), np.uint16), 'TCYX')
        meta = TiffIO.inspect_stack(source)
        self.assertEqual(meta["n_frames"], 12)
        self.assertTrue(meta["confirm_required"])
        self.assertIn("channels=3", meta["confirm_required"][0])

    def test_imagej_zstack_requires_axis_confirmation(self):
        source = self.root / "tz.tif"
        self._write_imagej(source, np.zeros((4, 3, 16, 16), np.uint8), 'TZYX')
        meta = TiffIO.inspect_stack(source)
        self.assertTrue(meta["confirm_required"])

    def test_plain_time_series_needs_no_axis_confirmation(self):
        source = self.root / "t.tif"
        self._write_imagej(source, np.zeros((6, 16, 16), np.uint8), 'TYX')
        self.assertEqual(TiffIO.inspect_stack(source)["confirm_required"], [])
        source2 = self.root / "pages.tif"
        _write_pages(source2, [np.zeros((8, 8), np.uint8)] * 3)
        self.assertEqual(TiffIO.inspect_stack(source2)["confirm_required"], [])

    def test_batch_refuses_ambiguous_axes_by_default(self):
        source = self.root / "tc.tif"
        self._write_imagej(source, np.zeros((2, 2, 16, 16), np.uint16), 'TCYX')
        result = batch_process([str(source)], str(self.root / "out"))
        self.assertEqual(result["completed"], 0)
        self.assertIn("未获按时间序列处理的确认", result["errors"][0][1])

    def test_batch_allow_ambiguous_axes_passes_precomputed(self):
        source = self.root / "tc.tif"
        self._write_imagej(source, np.zeros((2, 2, 16, 16), np.uint16), 'TCYX')
        meta = TiffIO.inspect_stack(source)
        frames = [np.full((16, 16), value, np.uint16) for value in (10, 200, 30, 40)]
        output = self.root / "out-allowed"
        result = batch_process([str(source)], str(output), precomputed={
            str(source.resolve()): {
                "frames": frames, "shifts_x": np.zeros(4), "shifts_y": np.zeros(4),
                "meta": dict(meta)},
        }, allow_ambiguous_axes=True)
        self.assertEqual(result["completed"], 1)
        self.assertTrue((output / "tc_corrected.tif").is_file())

    # ---- v7.0.4：审计报告改进 ----

    def test_report_paths_differ_within_same_second(self):
        # Windows 的 datetime.now() 粒度约 15.6ms，直接连调两次可能拿到同一
        # 时刻；注入推进 1 微秒的假时钟，确定性验证微秒后缀生效。
        base = datetime(2026, 9, 6, 1, 2, 3, 1000)
        stamps = iter([base, base + timedelta(microseconds=1)])
        fake_clock = mock.Mock()
        fake_clock.now.side_effect = lambda: next(stamps)
        with mock.patch("drift_core.datetime", fake_clock):
            first = _report_paths(self.root)
            second = _report_paths(self.root)
        self.assertNotEqual(first[0], second[0])
        self.assertNotEqual(first[1], second[1])

    def test_cancelled_inflight_file_is_recorded_as_cancelled(self):
        source = self.root / "source.tif"
        _write_pages(source, [np.zeros((8, 8), np.uint8)] * 2)
        detection = DetectionResult(
            shifts_x=np.zeros(2, np.float32), shifts_y=np.zeros(2, np.float32),
            pair_quality=[], key_indices=[0, 1], normalization=(0.0, 1.0))

        def fake_correct_and_save(*args, **kwargs):
            raise OperationCancelled("保存已取消")

        with mock.patch("drift_core.DriftDetector.detect_drift", return_value=detection), \
                mock.patch("drift_core.correct_and_save", side_effect=fake_correct_and_save):
            result = batch_process([str(source)], str(self.root / "out"))
        self.assertTrue(result["cancelled"])
        self.assertEqual(result["records"][0]["status"], "cancelled")
        with open(result["report_json"], encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["records"][0]["status"], "cancelled")

    def test_verify_info_reports_crop_and_changed_ratio(self):
        frames = [np.arange(64, dtype=np.uint8).reshape(8, 8) + index
                  for index in range(2)]
        meta = {"source_path": str(self.root / "s.tif"),
                "dtype": np.dtype("uint8"), "n_frames": 2}
        target = self.root / "verify-info.tif"
        info: dict = {}
        written = correct_and_save(frames, np.array([0.0, 2.0]), np.zeros(2), meta,
                                   str(target), verify_info=info)
        self.assertEqual(written, 2)
        self.assertEqual(info["verify_index"], 1)
        self.assertGreater(info["changed_ratio"], 0.0)
        self.assertEqual(info["crop"], {"left": 0, "top": 0, "right": 6, "bottom": 8})

    def test_defaults_align_with_imagej(self):
        """默认检测/匹配参数与 Fiji Linear Stack Alignment with SIFT 对齐。"""
        _, quality = _sift_params({})
        self.assertEqual(quality["ratio"], 0.92)          # closest/next closest ratio
        self.assertEqual(quality["ransac"], 25.0)         # maximal alignment error
        self.assertEqual(quality["min_inlier_ratio"], 0.05)  # inlier ratio
        # ImageJ 无对应门控，默认放宽到非约束水平（下限/等于 RANSAC 阈值）
        self.assertEqual(quality["min_matches"], 3)
        self.assertEqual(quality["min_inliers"], 3)
        self.assertEqual(quality["max_residual"], 25.0)
        self.assertEqual(quality["min_valid_pair_ratio"], 0.05)
        self.assertEqual(quality["max_gap"], 10000)

    def test_sift_params_rejects_unknown_keys(self):
        with self.assertRaises(ValueError):
            _sift_params({"nfeature": 100})  # 拼写错误不得静默使用默认值

    # ---- v7.3：单页连续堆栈、位移表、FLANN、插值间隙、报告溯源 ----

    def test_single_page_contiguous_series_roundtrip(self):
        """整栈写在单个 IFD 的连续序列必须能被 inspect/read 正确处理。"""
        data = np.arange(3 * 8 * 6, dtype=np.uint16).reshape(3, 8, 6)
        source = self.root / "contiguous.tif"
        with tifffile.TiffWriter(source) as writer:
            writer.write(data, photometric="minisblack", contiguous=True, metadata=None)
        meta = TiffIO.inspect_stack(source)
        self.assertEqual(meta["n_frames"], 3)
        self.assertEqual(meta["shape"], (8, 6))
        frames, meta2 = TiffIO.read_stack(source, meta=meta)
        self.assertEqual(len(frames), 3)
        self.assertTrue(all(frame.shape == (8, 6) for frame in frames))
        self.assertEqual(frames[0].tolist(), data[0].tolist())
        self.assertEqual(frames[2].tolist(), data[2].tolist())
        self.assertEqual(int(meta2["estimated_bytes"]), data.nbytes)

    def test_read_stack_with_stale_meta_page_count_is_rejected(self):
        """两次调用之间文件被替换时，传入的旧 meta 必须导致显式失败。"""
        source = self.root / "source.tif"
        _write_pages(source, [np.zeros((8, 8), np.uint8)] * 2)
        meta = TiffIO.inspect_stack(source)
        stale = dict(meta)
        stale["n_frames"] = 3
        with self.assertRaises(ValueError):
            TiffIO.read_stack(source, meta=stale)

    def test_write_shift_table_flags_interpolated_segments(self):
        target = self.root / "shifts.csv"
        path = write_shift_table(
            target, np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.zeros(5),
            key_indices=[0, 2, 4], interpolated_pairs=[1])
        self.assertEqual(path, target)
        rows = list(csv.reader(target.read_text(encoding="utf-8-sig").splitlines()))
        self.assertEqual(rows[0][:5], ["frame", "shift_x_px", "shift_y_px",
                                       "key_frame", "segment_estimated"])
        key_flags = {int(r[0]): r[3] for r in rows[1:]}
        self.assertEqual(key_flags, {0: "1", 1: "0", 2: "1", 3: "0", 4: "1"})
        # 帧对 1 覆盖 keys[1]=2 → keys[2]=4：区段内部(3)与右端关键帧(4)标记为估计
        estimated = {int(r[0]): r[4] for r in rows[1:]}
        self.assertEqual(estimated, {0: "0", 1: "0", 2: "0", 3: "1", 4: "1"})
        self.assertEqual(rows[1][1], "0.0000")
        self.assertEqual(rows[4][1], "3.0000")

    def test_match_pair_flann_path_large_descriptor_sets(self):
        """≥512 描述子走 FLANN 近似最近邻，平移恢复精度不得劣化。"""
        rng = np.random.default_rng(20260928)
        n = 1024
        points_a = np.column_stack([
            rng.uniform(0, 400, n), rng.uniform(0, 400, n)]).astype(np.float32)
        points_b = points_a + np.array([5.0, -3.0], dtype=np.float32)
        desc_a = rng.standard_normal((n, 128)).astype(np.float32)
        desc_b = desc_a.copy()  # 自匹配距离 0，保证 n 对 good matches
        _, quality = _sift_params({})
        result = _match_pair((0, 1), {0: (points_a, desc_a), 1: (points_b, desc_b)}, quality)
        self.assertEqual(result.status, "reliable")
        self.assertAlmostEqual(result.dx, 5.0, places=2)
        self.assertAlmostEqual(result.dy, -3.0, places=2)

    def test_interpolation_gap_longer_than_allowed_is_rejected(self):
        """连续失败帧对超过 max_interpolation_gap 时必须显式失败。"""
        # 帧必须有动态范围，否则在归一化阶段就被拒绝，走不到帧对匹配。
        base = np.arange(64, dtype=np.uint8).reshape(8, 8)
        frames = [base.copy() for _ in range(8)]
        statuses = ["reliable", "failed", "failed", "failed",
                    "reliable", "reliable", "reliable"]

        def fake_match(pair, features, quality):
            source, target = pair
            item = PairQuality(source, target, 20, 20)
            if statuses[source] == "reliable":
                item.good_matches, item.inliers = 20, 20
                item.inlier_ratio, item.median_residual = 1.0, 0.0
                item.dx, item.dy, item.status = 1.0, 0.0, "reliable"
            else:
                item.message = "模拟失败"
            return item

        with mock.patch("drift_core._extract_features", return_value=(None, None)), \
                mock.patch("drift_core._match_pair", side_effect=fake_match):
            with self.assertRaisesRegex(DetectionQualityError, "连续"):
                DriftDetector.detect_drift(frames, use_parallel=False,
                                           params={"max_interpolation_gap": 2})

    def test_batch_report_records_version_shift_table_and_pair_quality(self):
        source = self.root / "source.tif"
        _write_pages(source, [np.full((12, 12), value, np.uint16) for value in (1, 2)])
        frames, meta = TiffIO.read_stack(source)
        detection = DetectionResult(
            shifts_x=np.zeros(2, np.float32), shifts_y=np.zeros(2, np.float32),
            pair_quality=[PairQuality(0, 1, 8, 8, good_matches=6, inliers=6,
                                      inlier_ratio=1.0, median_residual=0.1,
                                      dx=0.0, dy=0.0, status="reliable")],
            key_indices=[0, 1], normalization=(0.0, 1.0))
        output = self.root / "out-audit"
        result = batch_process([str(source)], str(output), precomputed={
            str(source.resolve()): {
                "frames": frames, "shifts_x": np.zeros(2), "shifts_y": np.zeros(2),
                "meta": meta, "detection": detection},
        })
        self.assertEqual(result["completed"], 1)
        record = result["records"][0]
        table = Path(record["shift_table"])
        self.assertTrue(table.is_file())
        self.assertEqual(table.name, "source_corrected_shifts.csv")
        rows = list(csv.reader(table.read_text(encoding="utf-8-sig").splitlines()))
        self.assertEqual(len(rows), 3)  # 表头 + 2 帧
        self.assertEqual(record["pair_quality"][0]["status"], "reliable")
        self.assertEqual(record["pair_quality"][0]["inliers"], 6)
        self.assertIn("duration_s", record)
        self.assertIn("started_at", record)
        self.assertIn("source_size_bytes", record)
        self.assertIn("source_mtime", record)
        self.assertEqual(record["shift_x_range"], [0.0, 0.0])
        self.assertEqual(record["shift_y_range"], [0.0, 0.0])
        with open(result["report_json"], encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["tool_version"], __version__)
        self.assertEqual(report["tool"], "tiff-drift-correction")
        # CSV 列序固定：source_path/status 排在最前
        header = Path(result["report_csv"]).read_text(encoding="utf-8-sig").splitlines()[0]
        self.assertEqual(header.split(",")[:2], ["source_path", "status"])

    def test_csv_safe_prefixes_formula_characters(self):
        self.assertEqual(_csv_safe("=1+1"), "'=1+1")
        self.assertEqual(_csv_safe("+SUM(A1)"), "'+SUM(A1)")
        self.assertEqual(_csv_safe("@x"), "'@x")
        self.assertEqual(_csv_safe("正常文本"), "正常文本")
        self.assertEqual(_csv_safe(-1.5), -1.5)  # 数值不转义

    def test_batch_progress_is_monotonic_and_tracks_subfile_stages(self):
        sources = []
        precomputed = {}
        for name in ("a.tif", "b.tif"):
            path = self.root / name
            _write_pages(path, [np.full((12, 12), value, np.uint8) for value in (1, 2)])
            frames, meta = TiffIO.read_stack(path)
            sources.append(str(path))
            precomputed[str(path.resolve())] = {
                "frames": frames, "shifts_x": np.zeros(2), "shifts_y": np.zeros(2),
                "meta": meta}
        seen: list[float] = []
        result = batch_process(sources, str(self.root / "out-progress"),
                               precomputed=precomputed, progress_callback=seen.append)
        self.assertEqual(result["completed"], 2)
        self.assertTrue(seen)
        self.assertEqual(max(seen), 1.0)
        # 单文件内的阶段进度严格小于该文件的完成边界，全程单调不减
        self.assertTrue(all(a <= b + 1e-9 for a, b in zip(seen, seen[1:])))
        self.assertTrue(any(0.0 < value < 0.5 for value in seen))

    def test_total_physical_memory_is_positive_or_none(self):
        memory = total_physical_memory()
        if memory is not None:
            self.assertGreater(memory, 1024 ** 3)


if __name__ == "__main__":
    unittest.main()
