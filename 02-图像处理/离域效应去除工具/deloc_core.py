# -*- coding: utf-8 -*-
"""离域效应去除核心算法 —— 纯数值实现，不依赖任何 GUI 框架。

背景
----
液相 HRTEM 照片里，晶体边界往外常拖出一条"白色条纹虚影"：那是晶格条纹
经 CTF 离域（delocalization）后扩散到晶体外的分量。虚影和真晶格**共享同
一段空间频率**，所以靠频率滤波分不开，只能靠空间位置分开。

算法
----
    1) FFT 把图像拆成 低频 base 和 晶格带 lat
    2) 用户手绘的晶体 ROI -> 软掩膜 M（向外膨胀 + 羽化）
    3) out = base + (1 - strength * (1 - M)) * lat

    M = 1（晶体内部）  -> gain = 1，晶格原样保留
    M = 0（晶体外部）  -> gain = 1 - strength
                         strength = 1.0 时晶格带被完全移除
                         strength = 0.7 时只衰减到 30%，更保守

坐标系约定
----------
频率一律用 **cycles/pixel**，取值范围 0 ~ 0.5（Nyquist）。
本工具在**原图像素坐标**下操作，ROI 坐标也是原图像素坐标，
所以预览用的降采样不影响 ROI 的精度。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import fft as sp_fft
from scipy import ndimage, signal, special
from parameters import MAX_PIXELS, Parameters, finite_number

__all__ = [
    "radial_frequency",
    "power_spectrum",
    "radial_profile",
    "spectrum_overview",
    "suggest_band",
    "downsample_mean",
    "bandpassed_components",
    "clean",
    "CleanEngine",
    "DEFAULT_FMIN",
    "DEFAULT_FMAX",
]

#: 立项数据（Ag 液相录屏，晶格周期约 13.6 px）标定出来的默认晶格带。
#: 换数据集时应当用 suggest_band() 或手工重新确定。
DEFAULT_FMIN = 0.058
DEFAULT_FMAX = 0.092


# --------------------------------------------------------------------------
# 频率网格
# --------------------------------------------------------------------------
def radial_frequency(shape: Tuple[int, int]) -> np.ndarray:
    """返回与 fftshift 后频谱同形状的径向频率图（cycles/pixel）。

    对 h x w 的图像，fftshift 后的 bin (Y, X) 对应频率
    ((X - cx) / w, (Y - cy) / h)。用广播而不是 mgrid：后者会物化两个
    int64 全尺寸数组（4096² 时约 268MB），纯浪费。
    """
    h, w = _shape(shape)
    cy, cx = h // 2, w // 2
    yy = (np.arange(h, dtype=np.float32) - cy)[:, None]
    xx = (np.arange(w, dtype=np.float32) - cx)[None, :]
    return np.hypot(xx / np.float32(w), yy / np.float32(h))


def _shape(shape):
    if len(shape) != 2 or any(not isinstance(v, (int, np.integer)) or v <= 0 for v in shape):
        raise ValueError("图像尺寸必须是两个正整数")
    if int(shape[0]) * int(shape[1]) > MAX_PIXELS:
        raise ValueError(f"图像超过资源上限 {MAX_PIXELS:,} 像素，请先裁剪")
    return tuple(int(v) for v in shape)


def _prepare(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 2:
        raise ValueError(f"只支持二维灰度图，收到 shape={arr.shape}")
    if arr.size == 0:
        raise ValueError("输入图像为空")
    _shape(arr.shape)
    if arr.dtype.kind not in "buif" or (arr.dtype.kind in "ui" and arr.dtype.itemsize > 4):
        raise ValueError(f"不支持的数据类型: {arr.dtype}，请使用 ≤32 位整数或实数浮点图")
    wide = (arr.dtype.kind in "ui" and arr.dtype.itemsize == 4) or arr.dtype == np.float64
    out = arr.astype(np.float64 if wide else np.float32, copy=False)
    if not np.all(np.isfinite(out)):
        raise ValueError("输入图像含 NaN/Inf")
    return out


def downsample_mean(img: np.ndarray, step: int) -> np.ndarray:
    """均值（面积平均）降采样，用于**图像预览**。

    单纯 ``arr[::step]`` 是裸抽样：高于新 Nyquist（0.5/step cyc/px）的分量
    会折叠回来，细晶格在预览里呈现为假摩尔纹，用户据此刻的 ROI 就跟着错。
    均值池化把每个 step×step 块平均掉，折叠分量基本被压平。ROI 坐标存的是
    原图像素坐标，边缘不足一个整块的行/列被丢弃（< step 像素，不可见）。
    """
    if step <= 1:
        return _prepare(img)
    arr = _prepare(img)
    h, w = arr.shape
    h2, w2 = (h // step) * step, (w // step) * step
    if h2 == 0 or w2 == 0:
        return arr
    return arr[:h2, :w2].reshape(h2 // step, step, w2 // step, step).mean(axis=(1, 3))


def preview_resample(img: np.ndarray, step: int) -> np.ndarray:
    """FIR antialias preview; samples remain centered at original indices k*step.

    Kept separate from the legacy area-average utility. Ceil-sized output retains
    trailing rows/columns; the caller uses the actual sampled extent.
    """
    arr = _prepare(img)
    if not isinstance(step, (int, np.integer)) or step < 1:
        raise ValueError("降采样比例必须为正整数")
    if step == 1:
        return arr
    kernel = signal.firwin(32 * step + 1, .85 / step, window=("kaiser", 8.6)).astype(arr.dtype)
    out = signal.resample_poly(arr, 1, step, axis=0, window=kernel, padtype="line") if arr.shape[0] > 1 else arr
    return signal.resample_poly(out, 1, step, axis=1, window=kernel, padtype="line") if arr.shape[1] > 1 else out


# --------------------------------------------------------------------------
# 频谱分析（供 GUI 的 FFT 小图与自动选带使用）
#
# 频率语义（重要）
# ----------------
# 本段所有函数返回的频率一律是 **原图像素坐标系** 的 cycles/pixel。
# 显示用的降采样只作用在**谱图数组**上（_block_max_reduce），绝不作用在送入
# FFT 的图上：隔点抽样会把频率轴整体压缩 step 倍，并把 Nyquist 以上的内容
# 折叠回来。历史上 GUI 正是先把图隔点降到 512 再算谱，导致"点频谱选带 /
# 自动选带"选到 step 倍频（726x478 立项数据上是 2 倍），虚影压不掉。
# --------------------------------------------------------------------------
def _windowed_power_spectrum(arr: np.ndarray, window: bool = True) -> np.ndarray:
    """去均值（可选加 Hann 窗）后的功率谱，供本段几个公开函数复用。

    用 scipy.fft 而不是 np.fft：输入是 float32 时它保持 complex64，
    谱内存减半（np.fft 一律升 complex128，4096² 图峰值内存差约 0.5GB）。
    fftshift/ifftshift 只是切片翻转，保留 np.fft 的实现即可。
    """
    work = arr - float(arr.mean())
    if window:
        h, w = work.shape
        work *= np.hanning(h).astype(arr.dtype)[:, None]
        work *= np.hanning(w).astype(arr.dtype)[None, :]
    return np.abs(np.fft.fftshift(sp_fft.fft2(work))) ** 2


def _log_normalized(spec: np.ndarray) -> np.ndarray:
    """log1p 后归一化到 0..1，便于直接显示。"""
    out = np.log1p(spec)
    lo, hi = float(out.min()), float(out.max())
    if hi - lo < 1e-12:
        return np.zeros_like(out, dtype=np.float32)
    return ((out - lo) / (hi - lo)).astype(np.float32)


def _radial_bins(
    spec: np.ndarray, shape: Tuple[int, int], nbins: int
) -> Tuple[np.ndarray, np.ndarray]:
    """把功率谱按径向频率分箱平均。频率标签是原图像素坐标系的 cyc/px。"""
    if not isinstance(nbins, (int, np.integer)) or not 8 <= nbins <= 8192:
        raise ValueError("径向频谱分箱数必须在 8 ~ 8192 之间")
    fr = radial_frequency(shape) * nbins
    idx = np.clip(fr.astype(np.int32), 0, nbins - 1).ravel()

    power = spec.ravel()
    total = np.bincount(idx, minlength=nbins).astype(np.float64)
    summed = np.bincount(idx, weights=power, minlength=nbins).astype(np.float64)
    prof = summed / np.maximum(total, 1.0)

    freq = np.arange(nbins, dtype=np.float64) / float(nbins)
    return freq, prof


def power_spectrum(img: np.ndarray, window: bool = True) -> np.ndarray:
    """返回 fftshift 后的对数功率谱（已归一化到 0..1，便于直接显示）。"""
    arr = _prepare(img)
    return _log_normalized(_windowed_power_spectrum(arr, window))


def radial_profile(
    img: np.ndarray, nbins: int = 400, window: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """径向平均功率谱。返回 (频率数组 cyc/px, 功率数组)。"""
    arr = _prepare(img)
    spec = _windowed_power_spectrum(arr, window)
    return _radial_bins(spec, arr.shape, nbins)


def _block_max_reduce(arr: np.ndarray, max_dim: int) -> np.ndarray:
    """按块取最大值把长边压到 <= max_dim。**只用于显示**，不做频率换算。

    取最大值而不是隔点抽样：晶格峰在谱图里是孤立亮点，隔点抽样可能正好
    把整条峰抽没；块最大值保证降采样后峰仍然可见（用户就是要点它）。
    """
    if max_dim is None or max_dim <= 0:
        return arr
    h, w = arr.shape
    step = int(np.ceil(max(h, w) / float(max_dim)))
    if step <= 1:
        return arr
    ph, pw = (-h) % step, (-w) % step
    if ph or pw:  # 补边用 edge 模式，避免在谱图边缘引入假暗区
        arr = np.pad(arr, ((0, ph), (0, pw)), mode="edge")
    hh, ww = arr.shape
    return arr.reshape(hh // step, step, ww // step, step).max(axis=(1, 3))


def spectrum_overview(
    img: np.ndarray, max_display_dim: int = 512, nbins: int = 400
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """一次 FFT 同时给出「显示用谱图」与「径向谱」，频率均为原图 cyc/px。

    返回 ``(显示用对数功率谱, 频率, 功率)``。谱图只在显示维度上做块最大值
    降采样，所以坐标恒为 ±0.5（Nyquist），GUI 可以按这个坐标直接读频率、
    画频带标记、把点到的半径写回 fmin/fmax。
    """
    arr = _prepare(img)
    spec = _windowed_power_spectrum(arr)
    disp = _block_max_reduce(_log_normalized(spec), int(max_display_dim))
    freq, prof = _radial_bins(spec, arr.shape, int(nbins))
    return disp, freq, prof


def suggest_band(
    img: np.ndarray,
    r_min: float = 0.020,
    r_max: float = 0.400,
    half_width: float = 0.35,
) -> Tuple[float, float]:
    """在径向功率谱里找最强的晶格峰，返回建议的 (fmin, fmax)。

    只搜索 r_min..r_max，避开中心低频（形貌/照明不均）和最高频噪声。
    half_width 是相对峰位的半带宽比例。
    """
    freq, prof = radial_profile(img)
    return suggest_band_from_profile(freq, prof, r_min, r_max, half_width)


def suggest_band_from_profile(freq, prof, r_min=.020, r_max=.400, half_width=.35):
    """Return a candidate only when an interior peak rises above its background.

    This is a spectral candidate, not a physical identification of a lattice.
    """
    Parameters(fmin=r_min, fmax=r_max)
    finite_number(half_width, "相对半带宽", .01, .9)
    lo = int(np.searchsorted(freq, r_min))
    hi = int(np.searchsorted(freq, r_max))
    if hi <= lo + 2:
        raise ValueError("搜索频率范围过窄")

    seg = prof[lo:hi]
    # 轻微平滑，避免把噪声尖峰当峰
    seg = ndimage.uniform_filter1d(seg, size=max(3, len(seg) // 60))
    background = float(np.median(seg))
    maximum = float(np.max(seg))
    if not np.isfinite(maximum) or maximum <= 1e-20:
        raise ValueError("没有检测到可信晶格峰，请在频谱上手动选择")
    peaks, props = signal.find_peaks(seg, prominence=max(background * 4, maximum * .05))
    peaks = peaks[seg[peaks] > max(background * 8, maximum * .1)]
    if not len(peaks):
        raise ValueError("没有显著高于背景的晶格峰，请手动复核频谱")
    peak = int(peaks[np.argmax(seg[peaks])]) + lo
    f_peak = float(freq[peak])

    # 以峰为中心、按固定相对带宽给带
    f_lo = max(r_min, f_peak * (1.0 - half_width))
    f_hi = min(r_max, f_peak * (1.0 + half_width))
    return float(f_lo), float(f_hi)


# --------------------------------------------------------------------------
# 频带分离
# --------------------------------------------------------------------------
def band_mask(
    shape: Tuple[int, int], fmin: float, fmax: float, feather: float = 2.0
) -> np.ndarray:
    """Radial transfer; feather is in 0.001 cycles/pixel, not FFT bins."""
    Parameters(fmin=fmin, fmax=fmax, band_feather=feather)
    fr = radial_frequency(shape)
    return _radial_transfer(fr, fmin, fmax, feather)


def _radial_transfer(fr, fmin, fmax, feather):
    if feather == 0:
        return ((fr >= fmin) & (fr <= fmax)).astype(np.float32)
    sigma = feather * .001
    band = special.ndtr((fr - fmin) / sigma) - special.ndtr((fr - fmax) / sigma)
    normalization = 2 * special.ndtr((fmax - fmin) / (2 * sigma)) - 1
    band = np.clip(band / normalization, 0, 1)
    band[fr == 0] = 0
    return band.astype(np.float32)


def bandpassed_components(
    img: np.ndarray, fmin: float, fmax: float, feather: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Reflective boundary filtering using an orthonormal DCT-II.

    The DCT represents an even extension at both borders, avoiding circular
    coupling of opposite image edges without allocating a padded full image.
    """
    arr = _prepare(img)
    Parameters(fmin=fmin, fmax=fmax, band_feather=feather)
    h, w = arr.shape
    fr = np.hypot(np.arange(h, dtype=np.float32)[:, None] / (2*h),
                  np.arange(w, dtype=np.float32)[None, :] / (2*w))
    band = _radial_transfer(fr, fmin, fmax, feather)
    spec = sp_fft.dctn(arr - arr.mean(dtype=np.float64), type=2, norm="ortho")
    spec *= band
    lat = sp_fft.idctn(spec, type=2, norm="ortho").astype(arr.dtype)
    base = arr - lat
    return base, lat


def clean(
    img: np.ndarray,
    mask: np.ndarray,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    strength: float = 1.0,
    band_feather: float = 2.0,
) -> np.ndarray:
    """一站式接口：频带分离 + 掩膜加权。

    mask 取值 0..1，与 img 同形状，1 表示保留晶格。
    """
    return CleanEngine(band_feather).render(img, mask, fmin, fmax, strength)


def combine(
    base: np.ndarray, lat: np.ndarray, mask: np.ndarray, strength: float
) -> np.ndarray:
    """按掩膜把晶格带加权加回去。掩膜或参数变化时只需重跑这一步。"""
    base, lat, mask = np.asarray(base), np.asarray(lat), np.asarray(mask)
    if mask.shape != base.shape or lat.shape != base.shape:
        raise ValueError(f"掩膜形状 {mask.shape} 与图像 {base.shape} 不一致")
    if not all(np.all(np.isfinite(a)) for a in (base, lat, mask)):
        raise ValueError("分量或掩膜包含 NaN/Inf")
    s = finite_number(strength, "抑制强度", 0, 1)
    if np.any((mask < 0) | (mask > 1)):
        raise ValueError("掩膜值必须在 0 ~ 1 之间")
    m = mask.astype(np.float32)
    gain = 1.0 - s * (1.0 - m)
    return base + gain * lat


# --------------------------------------------------------------------------
# 带缓存的引擎（GUI 用它做实时预览）
# --------------------------------------------------------------------------
class CleanEngine:
    """缓存频带分量，让滑块拖动时的预览保持实时。

    频带参数（fmin/fmax/band_feather）变化才需要重算 FFT；
    掩膜、strength 变化只走 combine()，代价只有一次逐像素乘加。
    """

    def __init__(self, band_feather: float = 2.0) -> None:
        self.band_feather = float(band_feather)
        self._key = None
        self._shape = None
        self._img: np.ndarray | None = None
        self._base: np.ndarray | None = None
        self._lat: np.ndarray | None = None

    def components(
        self, img: np.ndarray, fmin: float, fmax: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        # 缓存键用「持有引用后的身份比较」而不是 id(img)：id 会被 CPython
        # 回收复用，换图后可能命中上一张图的缓存分量且无任何报错。
        # 这里持有 img 的引用，img is self._img 不可能被 id 复用骗过。
        key = (float(fmin), float(fmax), self.band_feather)
        if self._base is None or key != self._key or img is not self._img:
            self._base, self._lat = bandpassed_components(
                img, fmin, fmax, self.band_feather
            )
            self._key = key
            self._img = img
            self._shape = np.shape(img)
        return self._base, self._lat

    def render(
        self, img: np.ndarray, mask: np.ndarray, fmin: float, fmax: float, strength: float
    ) -> np.ndarray:
        base, lat = self.components(img, fmin, fmax)
        arr = _prepare(img)
        # Validate public inputs with the same contract as combine, then subtract
        # from the original. Preserve every protected pixel exactly.
        s = finite_number(strength, "抑制强度", 0, 1)
        m = np.asarray(mask)
        if m.shape != arr.shape or not np.all(np.isfinite(m)) or np.any((m < 0) | (m > 1)):
            raise ValueError("掩膜形状必须匹配图像，且所有值须在 0 ~ 1 之间")
        out = arr - s * (1 - m) * lat
        return np.where((m == 1) | (s == 0), arr, out).astype(arr.dtype)

    def lattice_amplitude(self, img: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
        """晶格带的局部振幅包络，用来判断"哪里有条纹"（诊断用）。"""
        _, lat = self.components(img, fmin, fmax)
        return np.sqrt(ndimage.gaussian_filter(lat * lat, 12)).astype(np.float32)

    def invalidate(self) -> None:
        self._key = None
        self._img = None
        self._base = None
        self._lat = None
