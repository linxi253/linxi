import unittest

import numpy as np

import filters

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



class FilterTests(unittest.TestCase):
    def setUp(self):
        filters.clear_filter_caches()

    def test_small_images_are_supported(self):
        for shape in ((1, 1), (1, 8), (2, 2), (8, 8)):
            with self.subTest(shape=shape):
                image = np.full(shape, 100, dtype=np.uint16)
                result = filters.adaptive_spectral_filter(image)
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.dtype, np.float32)
                self.assertTrue(np.isfinite(result).all())

    def test_zero_blend_is_identity(self):
        image = np.random.default_rng(1).integers(
            0, 4096, size=(64, 64), dtype=np.uint16
        )
        result = filters.adaptive_spectral_filter(image, blend_ratio=0.0)
        np.testing.assert_array_equal(result, image.astype(np.float32))

    def test_texture_mask_does_not_disable_checkerboard_filtering(self):
        image = ((np.indices((128, 128)).sum(axis=0) % 2) * 1000).astype(np.uint16)
        result = filters.adaptive_spectral_filter(image, blend_ratio=1.0)
        self.assertGreater(float(np.max(np.abs(result - image))), 0.0)

    def test_synthetic_lattice_peak_is_preserved_while_noise_is_reduced(self):
        rng = np.random.default_rng(9)
        _, x = np.indices((256, 256))
        clean = 1000.0 + 200.0 * np.sin(2 * np.pi * x / 16)
        noisy = clean + rng.normal(0.0, 80.0, clean.shape)
        result = filters.adaptive_spectral_filter(noisy, blend_ratio=0.5)

        noisy_mse = float(np.mean((noisy - clean) ** 2))
        result_mse = float(np.mean((result - clean) ** 2))
        self.assertLess(result_mse, noisy_mse)

        spectrum = np.abs(np.fft.rfft2(result - result.mean()))
        spectrum[:, 0] = 0
        peak = np.unravel_index(np.argmax(spectrum), spectrum.shape)
        self.assertEqual(peak, (0, 16))

    def test_texture_mask_extreme_float_does_not_overflow(self):
        # float32 平方在 1e20 量级会溢出为 inf；_texture_mask 必须先稳健缩放。
        image = np.linspace(-1e20, 1e20, 64 * 64, dtype=np.float32).reshape(64, 64)
        mask = filters._texture_mask(image)
        self.assertTrue(np.isfinite(mask).all())
        self.assertEqual(mask.shape, image.shape)
        self.assertGreaterEqual(float(mask.min()), 0.0)
        self.assertLessEqual(float(mask.max()), 1.0)

    def test_extreme_magnitude_float_does_not_overflow_in_fft(self):
        # 256×256、幅值 ~1e18 的有限输入在旧实现中会让 float32 功率谱
        # 平方溢出为 inf；现在必须在 FFT 前内部归一化并返回有限结果。
        rng = np.random.default_rng(4)
        image = (rng.random((256, 256)).astype(np.float32) - 0.5) * 2e18
        result = filters.adaptive_spectral_filter(image, blend_ratio=0.5)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(result.shape, image.shape)

    def test_large_magnitude_filter_is_scale_invariant(self):
        # 归一化只改变数值范围，不改变滤波行为：
        # filter(I * c) ≈ filter(I) * c。
        rng = np.random.default_rng(3)
        base = rng.random((128, 128)).astype(np.float32) * 100.0
        kwargs = {"k_factor": 1.2, "blend_ratio": 0.7, "delta": 3.0}
        unit = filters.adaptive_spectral_filter(base, **kwargs)
        huge = filters.adaptive_spectral_filter(
            base * np.float32(1e18), **kwargs
        )
        self.assertTrue(np.isfinite(huge).all())
        np.testing.assert_allclose(
            huge, unit * 1e18, rtol=1e-3, atol=1e13
        )

    def test_texture_mask_regular_float_range_is_finite(self):
        image = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        mask = filters._texture_mask(image)
        self.assertTrue(np.isfinite(mask).all())
        self.assertGreaterEqual(float(mask.min()), 0.0)
        self.assertLessEqual(float(mask.max()), 1.0)

    def test_radial_cache_is_bounded(self):
        for size in (16, 24, 32, 40):
            filters.get_rotational_average(np.ones((size, size), dtype=np.float32))
        self.assertLessEqual(filters.radial_cache_info().currsize, 2)

    def test_invalid_parameters_are_rejected(self):
        image = np.ones((16, 16), dtype=np.uint16)
        for kwargs in (
            {"k_factor": 0},
            {"blend_ratio": -0.1},
            {"blend_ratio": 1.1},
            {"delta": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                filters.adaptive_spectral_filter(image, **kwargs)


if __name__ == "__main__":
    unittest.main()
