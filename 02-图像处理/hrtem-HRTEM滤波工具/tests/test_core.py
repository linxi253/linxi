from __future__ import annotations

import threading
import unittest

import numpy as np

from hrtem_filter import FilterParams, HRTEMFilter
from hrtem_filter.geometry import ImageValidationError, Roi
from hrtem_filter.params import ParameterError


class CoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.filter = HRTEMFilter()

    def test_rectangular_images_keep_their_original_shape(self) -> None:
        for shape in ((32, 64), (64, 32), (31, 47)):
            with self.subTest(shape=shape):
                image = np.random.default_rng(1).normal(size=shape).astype(np.float32)
                result = self.filter.process_image(image)
                self.assertEqual(result.primary.shape, shape)
                self.assertTrue(np.isfinite(result.primary).all())

    def test_stem_primary_is_the_stem_result(self) -> None:
        image = np.random.default_rng(2).normal(size=(64, 64)).astype(np.float32)
        result = self.filter.process_image(image, FilterParams(stem_filter=True))
        self.assertEqual(result.primary_key, "stem_wiener_filtered")
        self.assertIn("stem_wiener_filtered", result.outputs)
        self.assertNotIn("wiener_filtered", result.outputs)

    def test_roi_is_cropped_back_to_the_exact_selection(self) -> None:
        image = np.random.default_rng(9).normal(size=(64, 80)).astype(np.float32)
        roi = Roi(5, 7, 39, 61)
        result = self.filter.process_image(image, roi=roi)
        self.assertEqual(result.primary.shape, (34, 54))
        self.assertEqual(result.roi, roi)

    def test_delta_zero_is_explicit_butterworth(self) -> None:
        params = FilterParams(delta=0, apply_butterworth=False, stem_filter=True).validated()
        self.assertEqual(params.primary_output, "butterworth")
        self.assertTrue(params.apply_butterworth)
        self.assertFalse(params.stem_filter)

    def test_invalid_stem_width_is_rejected(self) -> None:
        with self.assertRaises(ParameterError):
            FilterParams(crosshair_width=3).validated()

    def test_roi_coordinates_must_be_integers(self) -> None:
        image = np.zeros((32, 32), dtype=np.float32)
        with self.assertRaises(ImageValidationError):
            self.filter.process_image(image, roi=(1.5, 2, 20, 30))

    def test_roi_accepts_numpy_integer_coordinates(self) -> None:
        image = np.zeros((32, 32), dtype=np.float32)
        roi = (np.int64(1), np.int64(2), np.int64(20), np.int64(30))
        result = self.filter.process_image(image, roi=roi)
        self.assertEqual(result.primary.shape, (19, 28))

    def test_nan_input_is_rejected(self) -> None:
        image = np.ones((16, 16), dtype=np.float32)
        image[2, 3] = np.nan
        with self.assertRaises(ImageValidationError):
            self.filter.process_image(image)

    def test_butterworth_zero_radius_is_half_power(self) -> None:
        mask = self.filter.butterworth_filter(64, 4, 10)
        self.assertAlmostEqual(float(mask[32, 42]), 2 ** -0.5, places=5)

    def test_memory_estimate_uses_padded_fft_size(self) -> None:
        self.assertEqual(HRTEMFilter.estimate_peak_bytes((1024, 700)), 1024 * 1024 * 128)
        self.assertEqual(
            HRTEMFilter.estimate_peak_bytes((1024, 700), dtype=np.dtype("float64")),
            1024 * 1024 * 256,
        )
        self.assertEqual(
            HRTEMFilter.estimate_peak_bytes((1024, 700), output_count=3),
            1024 * 1024 * 128 * 3,
        )

    def test_butterworth_rejects_odd_size_but_process_image_handles_rectangles(self) -> None:
        with self.assertRaises(ValueError):
            self.filter.butterworth_filter(63, 4, 10)
        # 奇数矩形图像仍由 process_image 自动填充到偶数二次幂尺寸。
        image = np.random.default_rng(5).normal(size=(31, 47)).astype(np.float32)
        result = self.filter.process_image(image)
        self.assertEqual(result.primary.shape, (31, 47))

    def test_all_outputs_with_butterworth_primary_includes_butterworth(self) -> None:
        """Regression: include_all_outputs + butterworth primary must not KeyError."""
        image = np.random.default_rng(3).normal(size=(64, 64)).astype(np.float32)
        result = self.filter.process_image(
            image, FilterParams(primary_output="butterworth"), include_all_outputs=True
        )
        self.assertEqual(result.primary_key, "butterworth_filtered")
        for key in ("butterworth_filtered", "wiener_filtered", "absf_filtered"):
            self.assertIn(key, result.outputs)

    def test_include_all_outputs_includes_butterworth_for_wiener_primary(self) -> None:
        """Regression: include_all_outputs must cover every family, not just the primary."""
        image = np.random.default_rng(7).normal(size=(64, 64)).astype(np.float32)
        result = self.filter.process_image(image, include_all_outputs=True)
        self.assertEqual(result.primary_key, "wiener_filtered")
        for key in ("butterworth_filtered", "wiener_filtered", "absf_filtered"):
            self.assertIn(key, result.outputs)

    def test_stem_crosshair_applies_to_butterworth_output(self) -> None:
        """Regression: stem_filter must not be silently ignored on the BW path."""
        image = np.random.default_rng(4).normal(size=(64, 64)).astype(np.float32)
        plain = self.filter.process_image(image, FilterParams(primary_output="butterworth"))
        stem = self.filter.process_image(
            image,
            FilterParams(primary_output="butterworth", stem_filter=True),
            include_diagnostics=True,
        )
        self.assertIn("stem_crosshair_mask", stem.diagnostics)
        self.assertFalse(np.allclose(plain.primary, stem.primary))

    def test_stem_butterworth_output_uses_stem_prefixed_key(self) -> None:
        """Regression: crosshair-applied BW output must be named like the other stem outputs."""
        image = np.random.default_rng(8).normal(size=(64, 64)).astype(np.float32)
        result = self.filter.process_image(
            image, FilterParams(primary_output="butterworth", stem_filter=True)
        )
        self.assertEqual(result.primary_key, "stem_butterworth_filtered")
        self.assertEqual(list(result.outputs.keys()), ["stem_butterworth_filtered"])
        self.assertNotIn("butterworth_filtered", result.outputs)

    def test_delta_zero_keeps_plain_butterworth_key(self) -> None:
        """delta=0 强制剥离 STEM，键名也必须回到不带前缀的纯 BW 键。"""
        image = np.random.default_rng(13).normal(size=(64, 64)).astype(np.float32)
        result = self.filter.process_image(
            image, FilterParams(delta=0, stem_filter=True, primary_output="wiener")
        )
        self.assertEqual(result.primary_key, "butterworth_filtered")

    def test_include_all_outputs_with_stem_uses_consistent_prefixes(self) -> None:
        image = np.random.default_rng(14).normal(size=(64, 64)).astype(np.float32)
        result = self.filter.process_image(image, FilterParams(stem_filter=True), include_all_outputs=True)
        for key in ("stem_butterworth_filtered", "stem_wiener_filtered", "stem_absf_filtered"):
            self.assertIn(key, result.outputs)
        self.assertEqual(result.primary_key, "stem_wiener_filtered")

    def test_non_integer_parameters_are_rejected(self) -> None:
        """浮点与布尔会静默改变阈值循环语义，validated() 必须显式拒绝。"""
        with self.assertRaises(ParameterError):
            FilterParams(step=2.5).validated()
        with self.assertRaises(ParameterError):
            FilterParams(cycles=True).validated()
        with self.assertRaises(ParameterError):
            FilterParams(bw_order=4.0).validated()

    def test_crosshair_hole_smaller_than_half_width_is_rejected(self) -> None:
        """孔半径小于线宽一半时十字线会侵入中心束，必须显式拒绝。"""
        with self.assertRaises(ParameterError):
            FilterParams(crosshair_width=10, crosshair_hole_radius=4).validated()

    def test_crosshair_hole_equal_to_half_width_is_accepted(self) -> None:
        params = FilterParams(crosshair_width=10, crosshair_hole_radius=5).validated()
        self.assertEqual(params.crosshair_hole_radius, 5)

    def test_crosshair_half_extent_documents_small_image_degeneracy(self) -> None:
        """十字线有效长度按尺寸比例缩放：小图上默认参数会被中心孔吞掉。

        这是 GUI 在小 ROI 上给出 STEM 退化警告的依据。
        """
        factor = (1.0 / (np.sqrt(2.0) - 1.0)) ** 0.1
        self.assertAlmostEqual(HRTEMFilter.crosshair_half_extent(1024, 0.03), 0.03 * 1024 * factor, places=6)
        self.assertLess(HRTEMFilter.crosshair_half_extent(64, 0.03), 4.0)
        self.assertGreater(HRTEMFilter.crosshair_half_extent(1024, 0.03), 30.0)

    def test_radial_cache_keeps_oversized_grid(self) -> None:
        """超出缓存预算的网格仍保留为唯一缓存项。

        大图（如 8192²）正是 cycles×frames 重复旋转平均最多的场景，
        静默禁用缓存会把网格重算放大到不可接受。
        """
        processor = HRTEMFilter(radial_cache_bytes=1)
        image = np.random.default_rng(21).normal(size=(64, 64)).astype(np.float32)
        first = processor.process_image(image).primary
        self.assertIn((64, 64), processor._radial_cache)
        second = processor.process_image(image).primary
        np.testing.assert_array_equal(first, second)

    def test_fft_workers_setter_validates_range(self) -> None:
        processor = HRTEMFilter(workers=2)
        self.assertEqual(processor.workers, 2)
        with self.assertRaises(ValueError):
            processor.workers = 0
        processor.workers = 4
        self.assertEqual(processor.workers, 4)

    def test_numpy_integer_parameters_are_accepted(self) -> None:
        params = FilterParams(step=np.int64(2), cycles=np.int32(99), bw_order=np.int8(4)).validated()
        self.assertEqual(params.step, 2)

    def test_filter_instance_is_thread_safe_for_shared_shapes(self) -> None:
        """多线程共享同一实例处理同尺寸图像：结果必须与串行完全一致。"""
        image = np.random.default_rng(15).normal(size=(64, 64)).astype(np.float32)
        serial = self.filter.process_image(image).primary
        shared = HRTEMFilter()
        results: list[np.ndarray] = []
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(4):
                    results.append(shared.process_image(image).primary)
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertFalse(errors)
        for result in results:
            np.testing.assert_allclose(result, serial, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
