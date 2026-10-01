"""STEM 探针：会聚光学参数、像差相位、探针波函数。

与 TEM（平行光）不同，STEM 用物镜光阑把电子束会聚成探针，探针在倒空间的
分布就是光阑孔径函数乘以像差相位：

    P(k) = A(k − k_tilt) · exp(−i χ(k − k_tilt))
    χ(k) = π λ Δf k² + ½ π Cs λ³ k⁴ + π λ A_ast k² cos 2(θ − φ_ast)

实空间探针 = F⁻¹[P]。扫描时把探针平移到 r_p：倒空间乘相位斜坡
exp(−2πi k·r_p)，等价于实空间平移，且在周期网格上自动满足边界条件。

**归一化约定（全工具统一）**：探针满足 Σ_r |ψ(r)|² = 1（一个电子），
于是多层法传播后环形探测器收到的强度分数就是"入射束流被探测器收集的比例"，
在 0–1 之间且可跨厚度/参数直接比较——这是 HAADF 定量化的基础。

Scherzer 探针离焦与 TEM 的 Scherzer 离焦不是同一个数：
  - 探针最优离焦（探针最锐、衬度最强）：Δf = −√(|Cs| λ)
  - TEM 相位衬度 Scherzer 离焦：      Δf = −√(1.5 |Cs| λ)
本模块提供前者；后者仍由 Microscope.scherzer_defocus() 提供。两者的
"最优探针半角" α_opt = (4λ/|Cs|)^(1/4) 是最小探针尺寸对应的会聚角。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from ._fft import fft2, ifft2
from .constants import angle_from_k, k_from_angle
from .microscope import Microscope


def soft_width_k(angle_mrad: float, width_mrad: float, wavelength_a: float) -> float:
    """把角域上的软化宽度（mrad）换算到倒空间（Å⁻¹，精确 sinθ 关系）。"""
    if width_mrad <= 0:
        return 0.0
    hi = k_from_angle(angle_mrad + 0.5 * width_mrad, wavelength_a)
    lo = k_from_angle(max(angle_mrad - 0.5 * width_mrad, 0.0), wavelength_a)
    return float(0.5 * (hi - lo))


# ----------------------------------------------------------------------
# 光学参数
# ----------------------------------------------------------------------
@dataclass
class StemOptics(Microscope):
    """STEM 光学参数 = TEM 光学参数（电压/Cs/像差） + 探针与探测器。

    继承字段（见 microscope.Microscope）：voltage_kv、cs（Å）、defocus（Å）、
    astigmatism（Å）、astigmatism_azimuth（°）、focal_spread、angular_spread。

    Attributes
    ----------
    probe_semiangle : float
        探针会聚半角 α，mrad（物镜光阑大小）。典型 20–30 mrad。
    probe_soft : float
        光阑边缘软化宽度，mrad（0 = 理想硬边）。
    tilt_x, tilt_y : float
        束倾斜（探针在倒空间的平移），mrad。用于模拟样品/束的倾转，
        会改变通道效应与探测器的不对称收集。
    detector_inner, detector_outer : float
        环形探测器内/外收集半角，mrad。HAADF 典型 60–200 mrad。
    detector_soft : float
        探测器边缘软化宽度，mrad（0 = 理想硬边）。
    """

    probe_semiangle: float = 25.0
    probe_soft: float = 0.0
    tilt_x: float = 0.0
    tilt_y: float = 0.0
    detector_inner: float = 60.0
    detector_outer: float = 200.0
    detector_soft: float = 0.0

    # ------------------------------------------------------------------
    # 派生量
    # ------------------------------------------------------------------
    def probe_k(self) -> float:
        """探针光阑半径（倒空间，Å⁻¹）：k = sinα/λ。"""
        return k_from_angle(self.probe_semiangle, self.wavelength)

    def detector_k(self) -> Tuple[float, float]:
        """探测器内/外半径（倒空间，Å⁻¹）：k = sinθ/λ。"""
        lam = self.wavelength
        return (k_from_angle(max(self.detector_inner, 0.0), lam),
                k_from_angle(self.detector_outer, lam))

    def nyquist_angle_mrad(self, sampling_a: float) -> float:
        """给定采样下可无损收集的最大散射半角（mrad）。

        FFT 网格的奈奎斯特频率 k_max = 1/(2Δx)，对应散射角
        sinθ = λ k_max。超过该角度的强度会被折叠回低频
        （混叠），是 HAADF 模拟最容易被忽视的采样约束。
        """
        return float(angle_from_k(1.0 / (2.0 * float(sampling_a)), self.wavelength))

    def scherzer_probe_defocus(self) -> float:
        """探针最优离焦 Δf = −√(|Cs| λ)（Å，负值 = 欠焦）。"""
        lam = self.wavelength
        cs = self.cs
        if cs == 0.0:
            return 0.0
        return float(-np.sign(cs) * np.sqrt(abs(cs) * lam))

    def optimal_probe_semiangle(self) -> float:
        """最优探针半角 α_opt = (4λ/|Cs|)^(1/4)，mrad。

        Cs = 0（理想校正）时该式发散，返回 60 mrad 作为实用上限提示。
        """
        lam = self.wavelength
        cs = abs(self.cs)
        if cs < 1e-6:
            return 60.0
        return float((4.0 * lam / cs) ** 0.25 * 1e3)

    def probe_size_rayleigh(self) -> float:
        """探针尺寸（Rayleigh 判据 r = 0.61λ/α），Å。"""
        alpha = max(self.probe_semiangle, 1e-6) * 1e-3
        return float(0.61 * self.wavelength / alpha)

    def required_sampling(self, margin: float = 0.95) -> float:
        """为无损收集到探测器外角所需的最大采样间隔（Å/px）。

        由奈奎斯特条件 sinθ_out = λ/(2Δx) 反解：
            Δx = λ / (2 sinθ_out)
        margin < 1 时留出余量（默认 0.95，即采样比理论上限再细 5%）。
        探测器外角越大、电压越低，要求的采样越细——这是 HAADF 模拟
        最主要的计算代价来源。
        """
        s = np.sin(max(self.detector_outer, 1e-6) * 1e-3)
        return float(margin * self.wavelength / (2.0 * s))

    def detector_experimental_angles(self, sampling_a: float) -> Tuple[float, float, list]:
        """把探测器内外角裁剪到可收集范围内，返回 (inner, outer, 警告列表)。"""
        k_max = self.nyquist_angle_mrad(sampling_a)
        notes = []
        inner = float(max(self.detector_inner, 0.0))
        outer = float(self.detector_outer)
        if outer > k_max:
            notes.append(
                f"探测器外角 {outer:g} mrad 超过当前采样的可收集上限 "
                f"{k_max:.1f} mrad，已裁剪（如需更大角度请减小采样 Å/px）"
            )
            outer = k_max
        if inner >= outer:
            notes.append(
                f"探测器内角 {inner:g} mrad ≥ 外角 {outer:g} mrad，改为 (0, {outer:.1f})"
            )
            inner = 0.0
        return inner, outer, notes

    def validate(self) -> None:
        if self.probe_semiangle <= 0:
            raise ValueError("探针会聚半角需 > 0")
        if self.probe_soft < 0 or self.detector_soft < 0:
            raise ValueError("软化宽度不能为负")
        if self.detector_outer <= 0:
            raise ValueError("探测器外角需 > 0")
        if self.detector_inner < 0:
            raise ValueError("探测器内角不能为负")
        if self.voltage_kv <= 0:
            raise ValueError("加速电压需 > 0")
        if not np.isfinite(self.tilt_x) or not np.isfinite(self.tilt_y):
            raise ValueError("束倾斜需为有限数值")

    def summary(self) -> str:
        base = super().summary()
        lines = [
            base,
            f"探针半角      : {self.probe_semiangle:g} mrad "
            f"(软化 {self.probe_soft:g})",
            f"束倾斜        : ({self.tilt_x:g}, {self.tilt_y:g}) mrad",
            f"探测器        : {self.detector_inner:g}–{self.detector_outer:g} mrad "
            f"(软化 {self.detector_soft:g})",
        ]
        return "\n".join(lines)


# ----------------------------------------------------------------------
# 频率网格与像差
# ----------------------------------------------------------------------
def frequency_grid(shape: Tuple[int, int], sampling) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (KX, KY, K2) 频率网格（Å⁻¹，未 fftshift，与 FFT 约定一致）。"""
    ny, nx = shape
    try:
        sx, sy = float(sampling[0]), float(sampling[1])
    except (TypeError, IndexError):
        sx = sy = float(sampling)
    kx = np.fft.fftfreq(nx, d=sx)
    ky = np.fft.fftfreq(ny, d=sy)
    KX, KY = np.meshgrid(kx, ky)
    return KX, KY, KX ** 2 + KY ** 2


def aberration_chi(
    optics: StemOptics,
    KX: np.ndarray,
    KY: np.ndarray,
    K2: np.ndarray,
) -> np.ndarray:
    """像差函数 χ(k)，rad（球差 + 离焦 + 二重像散）。"""
    lam = optics.wavelength
    chi = np.pi * optics.defocus * lam * K2 \
        + 0.5 * np.pi * optics.cs * lam ** 3 * K2 ** 2
    if optics.astigmatism != 0.0:
        theta = np.arctan2(KY, KX)
        phi = np.deg2rad(optics.astigmatism_azimuth)
        chi = chi + np.pi * optics.astigmatism * lam * K2 * np.cos(2.0 * (theta - phi))
    return chi


def _annulus_mask(k: np.ndarray, k_in: float, k_out: float, soft_k: float) -> np.ndarray:
    """环形掩模（soft_k = 0 时为理想硬边）。"""
    if soft_k > 0:
        outer = 0.5 * (1.0 - np.tanh((k - k_out) / soft_k))
        inner = 0.5 * (1.0 + np.tanh((k - k_in) / soft_k)) if k_in > 0 else 1.0
        return outer * inner
    return ((k <= k_out) & (k >= k_in)).astype(float)


# ----------------------------------------------------------------------
# 探针
# ----------------------------------------------------------------------
def probe_ft(
    optics: StemOptics,
    shape: Tuple[int, int],
    sampling,
    normalize: bool = True,
) -> np.ndarray:
    """探针的倒空间表示 P(k) = A(k−k_tilt)·exp(−iχ(k−k_tilt))。

    normalize=True 时按 Σ_r|F⁻¹[P]|² = 1 归一（单位束流探针）。
    """
    KX, KY, _ = frequency_grid(shape, sampling)
    lam = optics.wavelength
    tilt_x = optics.tilt_x * 1e-3 / lam
    tilt_y = optics.tilt_y * 1e-3 / lam
    KX_t = KX - tilt_x
    KY_t = KY - tilt_y
    K2_t = KX_t ** 2 + KY_t ** 2

    k_ap = optics.probe_k()
    soft_k = soft_width_k(optics.probe_semiangle, optics.probe_soft, lam)
    aperture = _annulus_mask(np.sqrt(K2_t), 0.0, k_ap, soft_k)
    chi = aberration_chi(optics, KX_t, KY_t, K2_t)
    P = aperture * np.exp(-1j * chi)
    if normalize:
        n = shape[0] * shape[1]
        norm = float(np.sqrt(np.sum(np.abs(P) ** 2)))
        if norm <= 0.0:
            raise ValueError("探针光阑内没有任何频率分量，请检查会聚角与采样")
        P = P * (np.sqrt(n) / norm)
    return P


def probe_wave(
    optics: StemOptics,
    shape: Tuple[int, int],
    sampling,
    position: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """实空间探针波函数 ψ_p(r)，Σ|ψ|² = 1（position=None 时居中）。

    position : (x, y) Å，探针中心位置。
    """
    P = probe_ft(optics, shape, sampling, normalize=True)
    if position is not None:
        KX, KY, _ = frequency_grid(shape, sampling)
        x0, y0 = float(position[0]), float(position[1])
        P = P * np.exp(-2j * np.pi * (KX * x0 + KY * y0))
    return ifft2(P)


def probe_batch_ft(
    optics: StemOptics,
    shape: Tuple[int, int],
    sampling,
    positions: Sequence[Tuple[float, float]],
) -> np.ndarray:
    """一批探针位置的倒空间表示，形状 (B, ny, nx)。

    只做一次光阑/像差计算，再对每个位置乘相位斜坡。
    """
    P0 = probe_ft(optics, shape, sampling, normalize=True)
    KX, KY, _ = frequency_grid(shape, sampling)
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("positions 需为 (B, 2) 的 (x, y) 数组")
    ramp = KX[None, :, :] * pos[:, 0, None, None] + KY[None, :, :] * pos[:, 1, None, None]
    return P0[None, :, :] * np.exp(-2j * np.pi * ramp)


def probe_intensity_profile(
    optics: StemOptics,
    shape: Tuple[int, int],
    sampling,
    n: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """探针的径向强度剖面（用于界面显示探针形状）。

    在大网格上计算以保证频率分辨率，返回 (r_Å, 归一化强度)。
    """
    P = probe_ft(optics, (n, n), sampling, normalize=True)
    psi = ifft2(P)
    inten = np.abs(psi) ** 2
    idx = np.fft.fftshift(np.arange(n) - n // 2)
    rr = np.sqrt(idx[:, None] ** 2 + idx[None, :] ** 2) * float(sampling)
    shifted = np.fft.fftshift(inten)
    bins = np.arange(0.0, rr.max() + float(sampling), float(sampling))
    which = np.digitize(rr.ravel(), bins) - 1
    total = np.bincount(which, weights=shifted.ravel(), minlength=len(bins))
    count = np.bincount(which, minlength=len(bins))
    radial = total / np.maximum(count, 1)
    center = radial.max() if radial.max() > 0 else 1.0
    return bins, radial / center
