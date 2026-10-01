# -*- coding: utf-8 -*-
"""deloc_core 的单元测试。"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deloc_core as dc  # noqa: E402


# --------------------------------------------------------------------------
# 频率网格
# --------------------------------------------------------------------------
def test_radial_frequency_center_and_edges():
    fr = dc.radial_frequency((64, 128))
    assert fr.shape == (64, 128)
    assert fr[32, 64] == 0.0
    # 角落接近但不超过 Nyquist 半径 sqrt(0.5^2+0.5^2)
    assert fr.max() < 0.75


def test_radial_frequency_is_in_cycles_per_pixel():
    """8×8 图：中心为 0，水平/垂直方向 4 像素处正好是 Nyquist 0.5。"""
    fr = dc.radial_frequency((8, 8))
    assert fr[4, 4] == pytest.approx(0.0, abs=1e-6)
    assert fr[4, 0] == pytest.approx(0.5, abs=1e-6)
    assert fr[0, 4] == pytest.approx(0.5, abs=1e-6)
    assert fr[4, 7] == pytest.approx(0.375, abs=1e-6)


# --------------------------------------------------------------------------
# 频带掩膜
# --------------------------------------------------------------------------
def test_band_mask_rejects_bad_range():
    with pytest.raises(ValueError):
        dc.band_mask((16, 16), 0.0, 0.1)
    with pytest.raises(ValueError):
        dc.band_mask((16, 16), 0.2, 0.1)


def test_band_mask_range_and_norm():
    m = dc.band_mask((64, 64), 0.10, 0.20, feather=2.0)
    assert m.shape == (64, 64)
    assert m.min() >= 0.0
    assert m.max() == pytest.approx(1.0, abs=1e-6)
    # 带中心位置应当接近 1（归一化后峰值恰为 1）
    assert m[32, 32 + int(0.15 * 64)] > 0.9
    # 带外应当接近 0（羽化会让带边有少量泄漏）
    assert m[32, 32] < 0.01


# --------------------------------------------------------------------------
# 频带分离
# --------------------------------------------------------------------------
def test_components_reconstruct_image():
    rng = np.random.default_rng(0)
    img = rng.normal(100, 20, (96, 96)).astype(np.float32)
    base, lat = dc.bandpassed_components(img, 0.1, 0.2)
    assert np.allclose(base + lat, img, atol=1e-3)
    assert abs(float(lat.mean())) < 1e-4          # 带通分量均值为零


def test_components_isolate_a_grating():
    """纯余弦光栅应当几乎全部落进 lat。

    v2 的频带分离改用 DCT-II（偶对称基、反射边界）：光栅在图像边界的
    偶延拓相位不连续会把少量能量泄漏到带外——余弦约 7% 振幅、正弦约
    18%（实测 base.std：cos≈1.57 / sin≈3.79）。阈值按这一物理事实设定，
    不再是旧 FFT 周期延拓在整周期光栅下的零泄漏。
    """
    n, period = 128, 8.0
    yy, xx = np.mgrid[0:n, 0:n]
    img = (100 + 30 * np.cos(2 * np.pi * xx / period)).astype(np.float32)
    f0 = 1.0 / period
    base, lat = dc.bandpassed_components(img, f0 * 0.7, f0 * 1.3)
    assert lat.std() > 18.0
    assert base.std() < 2.5          # 实测约 1.57：DCT 偶延拓的边界泄漏


# --------------------------------------------------------------------------
# 组合的不变量
# --------------------------------------------------------------------------
@pytest.fixture()
def small_image():
    rng = np.random.default_rng(1)
    return rng.normal(50, 10, (72, 72)).astype(np.float32)


def test_mask_all_keep_is_identity(small_image):
    ones = np.ones(small_image.shape, np.float32)
    out = dc.clean(small_image, ones, 0.1, 0.2, strength=1.0)
    assert np.allclose(out, small_image, atol=1e-3)


def test_mask_all_cut_removes_lattice(small_image):
    zeros = np.zeros(small_image.shape, np.float32)
    base, _ = dc.bandpassed_components(small_image, 0.1, 0.2)
    out = dc.clean(small_image, zeros, 0.1, 0.2, strength=1.0)
    assert np.allclose(out, base, atol=1e-3)


def test_strength_zero_is_identity(small_image):
    zeros = np.zeros(small_image.shape, np.float32)
    out = dc.clean(small_image, zeros, 0.1, 0.2, strength=0.0)
    assert np.allclose(out, small_image, atol=1e-3)


def test_strength_out_of_range_is_rejected(small_image):
    """v2 参数策略：越界强度直接拒绝而不是静默截断（parameters.finite_number）。

    旧版本把 strength=5 截断成 1、-3 截断成 0；静默截断会掩盖调用方的
    参数错误（审查报告：不要用默认值掩盖无效参数）。
    """
    zeros = np.zeros(small_image.shape, np.float32)
    for bad in (5.0, -3.0, float("nan")):
        with pytest.raises(ValueError):
            dc.clean(small_image, zeros, 0.1, 0.2, strength=bad)
    out = dc.clean(small_image, zeros, 0.1, 0.2, strength=0.0)
    assert np.allclose(out, small_image, atol=1e-3)


def test_combine_checks_shape(small_image):
    base, lat = dc.bandpassed_components(small_image, 0.1, 0.2)
    with pytest.raises(ValueError):
        dc.combine(base, lat, np.ones((5, 5), np.float32), 1.0)


def test_soft_mask_midpoint_gives_half_gain(small_image):
    """掩膜 0.5、强度 1 时，晶格带应当只保留一半。"""
    base, lat = dc.bandpassed_components(small_image, 0.1, 0.2)
    half = np.full(small_image.shape, 0.5, np.float32)
    out = dc.combine(base, lat, half, 1.0)
    assert np.allclose(out, base + 0.5 * lat, atol=1e-3)


# --------------------------------------------------------------------------
# 端到端：掩膜外真的被压下去了吗
# --------------------------------------------------------------------------
def test_lattice_suppressed_outside_mask_and_kept_inside():
    n, period = 200, 10.0
    yy, xx = np.mgrid[0:n, 0:n]
    grating = np.sin(2 * np.pi * xx / period).astype(np.float32)
    img = (128 + 25 * grating)

    # 只在左半边有"晶体"
    mask = np.zeros((n, n), np.float32)
    mask[:, : n // 2] = 1.0

    f0 = 1.0 / period
    out = dc.clean(img, mask, f0 * 0.75, f0 * 1.35, strength=1.0, band_feather=1.0)

    def fringe_amplitude(patch):
        return float(patch.std())

    amp_in_before = fringe_amplitude(img[:, 20:80])
    amp_in_after = fringe_amplitude(out[:, 20:80])
    amp_out_before = fringe_amplitude(img[:, 120:180])
    amp_out_after = fringe_amplitude(out[:, 120:180])

    assert amp_in_after > 0.8 * amp_in_before       # 晶体内部保住
    assert amp_out_after < 0.2 * amp_out_before     # 晶体外部压掉


# --------------------------------------------------------------------------
# 频谱辅助
# --------------------------------------------------------------------------
def test_suggest_band_finds_the_grating():
    n, period = 256, 12.0
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:n, 0:n]
    img = (128 + 40 * np.sin(2 * np.pi * xx / period)
           + rng.normal(0, 8, (n, n))).astype(np.float32)
    fmin, fmax = dc.suggest_band(img)
    f0 = 1.0 / period
    assert fmin < f0 < fmax, (fmin, f0, fmax)


def test_suggest_band_on_reference_size_image_uses_original_pixels():
    """P0 回归：726×478（长边 > 512）也必须选到 1/13.6 的晶格带。

    历史 bug：GUI 先把图隔点降到 512 再算谱，频率轴被压缩 step=2 倍，
    「自动选带」给出 0.096~0.199 的 2 倍频，虚影根本压不掉。
    """
    h, w, period = 478, 726, 13.6
    yy, xx = np.mgrid[0:h, 0:w]
    disk = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) < 200 ** 2
    lattice = (45 * np.cos(2 * np.pi * xx / period)
               + 45 * np.cos(2 * np.pi * yy / period))
    img = np.clip(120 + np.where(disk, lattice, 0.0), 0, 255).astype(np.uint8)

    f0 = 1.0 / period
    fmin, fmax = dc.suggest_band(img)
    assert fmin < f0 < fmax, (fmin, f0, fmax)
    assert fmax < 0.15, (fmin, fmax)      # 0.19 附近是 2 倍频，不允许


def test_spectrum_overview_reports_original_pixel_frequencies():
    """面板坐标必须与原图像素一致：峰值落在 1/period 而不是 2/period。"""
    h, w, period = 478, 726, 13.6
    yy, xx = np.mgrid[0:h, 0:w]
    img = (120 + 45 * np.cos(2 * np.pi * xx / period)
           + 45 * np.cos(2 * np.pi * yy / period)).astype(np.float32)

    spec, freq, prof = dc.spectrum_overview(img)
    assert max(spec.shape) <= 512                # 只降显示维度
    assert max(spec.shape) < max(img.shape)      # 726 > 512，确实降了

    lo = int(np.searchsorted(freq, 0.02))
    hi = int(np.searchsorted(freq, 0.40))
    peak = float(freq[lo + int(np.argmax(prof[lo:hi]))])
    assert peak == pytest.approx(1.0 / period, abs=0.005), peak


def test_spectrum_overview_equals_power_spectrum_when_small():
    """不降采样时谱图必须与 power_spectrum 完全一致（没有额外归一化）。"""
    rng = np.random.default_rng(6)
    img = rng.normal(0, 1, (64, 64)).astype(np.float32)
    spec, freq, prof = dc.spectrum_overview(img)
    assert np.array_equal(spec, dc.power_spectrum(img))
    freq2, prof2 = dc.radial_profile(img)
    assert np.array_equal(freq, freq2)
    assert np.allclose(prof, prof2)


def test_block_max_reduce_keeps_peaks_and_bounds_size():
    """块最大值降采样：孤立亮点不丢、长边不超上限、整除时形状精确。"""
    arr = np.zeros((100, 250), np.float32)
    arr[10, 200] = 1.0
    out = dc._block_max_reduce(arr, 64)
    assert max(out.shape) <= 64
    assert out.max() == 1.0                       # 亮点没被抽掉
    assert dc._block_max_reduce(arr, 500).shape == arr.shape
    even = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    assert dc._block_max_reduce(even, 32).shape == (32, 32)


def test_power_spectrum_normalised():
    rng = np.random.default_rng(4)
    img = rng.normal(0, 1, (64, 64)).astype(np.float32)
    spec = dc.power_spectrum(img)
    assert spec.shape == img.shape
    assert 0.0 <= spec.min() and spec.max() <= 1.0


def test_radial_profile_shape():
    rng = np.random.default_rng(5)
    img = rng.normal(0, 1, (64, 64)).astype(np.float32)
    freq, prof = dc.radial_profile(img, nbins=100)
    assert len(freq) == len(prof) == 100
    assert freq[0] == 0.0
    assert np.all(np.isfinite(prof))


# --------------------------------------------------------------------------
# 预览降采样（P0 回归：裸抽样的摩尔纹）
# --------------------------------------------------------------------------
def test_downsample_mean_is_area_average():
    """均值池化 = 每块的平均：斜坡图池化后应是平移了半块的斜坡。"""
    yy, xx = np.mgrid[0:8, 0:12]
    img = xx.astype(np.float32)
    pooled = dc.downsample_mean(img, 4)
    assert pooled.shape == (2, 3)
    assert np.allclose(pooled[0], [1.5, 5.5, 9.5])


def test_downsample_mean_suppresses_aliasing():
    """周期 2 的棋盘格：裸抽样混叠成常数 0（结构完全丢失），池化保住均值。"""
    yy, xx = np.mgrid[0:64, 0:64]
    checker = ((xx + yy) % 2).astype(np.float32)
    assert np.allclose(checker[::2, ::2], 0.0)             # 裸抽样的失真
    pooled = dc.downsample_mean(checker, 2)
    assert pooled.shape == (32, 32)
    assert np.allclose(pooled, 0.5)                        # 池化：均值正确


def test_downsample_mean_step1_and_constant():
    img = np.full((10, 9), 7.0, np.float32)
    assert dc.downsample_mean(img, 1).shape == (10, 9)
    pooled = dc.downsample_mean(img, 3)
    assert pooled.shape == (3, 3)
    assert np.allclose(pooled, 7.0)


def test_downsample_mean_rejects_nan():
    bad = np.zeros((8, 8), np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        dc.downsample_mean(bad, 2)


# --------------------------------------------------------------------------
# 引擎缓存
# --------------------------------------------------------------------------
def test_engine_caches_components(small_image):
    eng = dc.CleanEngine()
    b1, l1 = eng.components(small_image, 0.1, 0.2)
    b2, l2 = eng.components(small_image, 0.1, 0.2)
    assert b1 is b2 and l1 is l2              # 命中缓存

    b3, _ = eng.components(small_image, 0.1, 0.25)
    assert b3 is not b1                       # 频带变了要重算


def test_engine_invalidate(small_image):
    eng = dc.CleanEngine()
    b1, _ = eng.components(small_image, 0.1, 0.2)
    eng.invalidate()
    b2, _ = eng.components(small_image, 0.1, 0.2)
    assert b2 is not b1
    assert np.allclose(b1, b2)


def test_engine_recomputes_for_a_different_image(small_image):
    """换图必须重算。缓存键曾用 id(img)：id 被 CPython 复用时，
    换图可能直接命中上一张图的缓存分量且无任何报错。"""
    eng = dc.CleanEngine()
    b1, _ = eng.components(small_image, 0.1, 0.2)
    other = small_image + 10.0
    b2, _ = eng.components(other, 0.1, 0.2)
    assert b2 is not b1
    assert not np.allclose(b1, b2)


def test_fft_keeps_single_precision(small_image):
    """scipy.fft 对 float32 输入保持 complex64，大图峰值内存减半。"""
    spec = np.fft.fftshift(__import__("scipy").fft.fft2(small_image))
    assert spec.dtype == np.complex64


def test_engine_lattice_amplitude_nonnegative(small_image):
    eng = dc.CleanEngine()
    amp = eng.lattice_amplitude(small_image, 0.1, 0.2)
    assert amp.shape == small_image.shape
    assert amp.min() >= 0.0


# --------------------------------------------------------------------------
# 输入校验
# --------------------------------------------------------------------------
def test_rejects_non_2d():
    with pytest.raises(ValueError):
        dc.bandpassed_components(np.zeros((3, 4, 5)), 0.1, 0.2)


def test_rejects_non_finite():
    bad = np.zeros((16, 16), np.float32)
    bad[3, 3] = np.nan
    with pytest.raises(ValueError):
        dc.bandpassed_components(bad, 0.1, 0.2)
