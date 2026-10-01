# -*- coding: utf-8 -*-
"""stress_core 纯函数单元测试（不导入 GUI / tifffile / skimage）。"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stress_core import (  # noqa: E402
    safe_corrcoef,
    safe_cov,
    safe_divide,
    corrcoef_or_nan,
    get_smooth_curve,
    normalize_frame,
    rgb_to_gray,
)


class TestSafeCorrcoef:
    def test_normal(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert safe_corrcoef(x, x) == pytest.approx(1.0)

    def test_constant_returns_zero(self):
        x = np.array([5.0, 5.0, 5.0])
        y = np.array([1.0, 2.0, 3.0])
        assert safe_corrcoef(x, y) == 0.0

    def test_short_returns_zero(self):
        assert safe_corrcoef(np.array([1.0]), np.array([2.0])) == 0.0


class TestCorrcoefOrNan:
    """展示口径：无法定义返回 NaN（而非被误读为"测得零相关"的 0.0）。"""

    def test_normal(self):
        x = np.arange(10.0)
        assert corrcoef_or_nan(x, x + 1.0) == pytest.approx(1.0)

    def test_constant_returns_nan(self):
        x = np.full(5, 3.0)
        y = np.arange(5.0)
        assert np.isnan(corrcoef_or_nan(x, y))

    def test_nan_samples_excluded(self):
        x = np.arange(8.0)
        y = x * 2.0
        x2, y2 = x.copy(), y.copy()
        x2[0] = np.nan
        y2[1] = np.nan
        assert corrcoef_or_nan(x2, y2) == pytest.approx(1.0)

    def test_all_nan_returns_nan(self):
        assert np.isnan(corrcoef_or_nan(np.full(4, np.nan), np.arange(4.0)))

    def test_too_few_finite_returns_nan(self):
        x = np.array([1.0, np.nan, np.nan])
        y = np.array([2.0, 3.0, np.nan])
        assert np.isnan(corrcoef_or_nan(x, y))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            corrcoef_or_nan(np.arange(4.0), np.arange(5.0))


class TestSafeCov:
    def test_normal(self):
        x = np.array([1.0, 2.0, 3.0])
        assert safe_cov(x, x) == pytest.approx(np.cov(x, x)[0, 1])

    def test_constant_returns_zero(self):
        x = np.array([2.0, 2.0, 2.0])
        y = np.array([1.0, 2.0, 3.0])
        assert safe_cov(x, y) == 0.0


class TestSafeDivide:
    def test_zero_denominator(self):
        assert safe_divide(1.0, 0.0, fill=-1.0) == -1.0

    def test_normal(self):
        assert safe_divide(6.0, 3.0) == pytest.approx(2.0)


class TestSmoothCurve:
    def test_short_series_returns_input(self):
        times = np.array([1.0, 2.0, 3.0])
        data = np.array([4.0, 5.0, 6.0])
        t, d = get_smooth_curve(times, data)
        assert t is times or np.array_equal(t, times)
        assert d is data or np.array_equal(d, data)

    def test_long_series_smoothed(self):
        times = np.arange(30.0)
        data = np.sin(times / 5.0) + np.random.default_rng(0).normal(0, 0.01, 30)
        t, d = get_smooth_curve(times, data)
        assert len(d) == len(data)
        assert not np.array_equal(d, data)


class TestNormalizeFrame:
    def test_2d_unchanged(self):
        frame = np.zeros((10, 20), dtype=np.uint8)
        out = normalize_frame(frame)
        assert out.shape == (10, 20)

    def test_rgb_converted(self):
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, size=(10, 20, 3), dtype=np.uint8)
        out = normalize_frame(frame)
        assert out.shape == (10, 20)
        assert out.ndim == 2

    def test_single_channel_3d_unwrapped(self):
        frame = np.zeros((1, 10, 20), dtype=np.uint8)
        out = normalize_frame(frame)
        assert out.shape == (10, 20)

    def test_unsupported_4d_raises(self):
        frame = np.zeros((2, 10, 20, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            normalize_frame(frame)


class TestRgbToGray:
    def test_white_uint8(self):
        white = np.full((4, 5, 3), 255, dtype=np.uint8)
        assert np.all(rgb_to_gray(white) >= 254)

    def test_black_uint8(self):
        black = np.zeros((4, 5, 3), dtype=np.uint8)
        assert np.all(rgb_to_gray(black) == 0)
