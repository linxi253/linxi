"""
成像与衍射后处理（对应 SimulaTEM 的 Go / Files 功能）。

包含：
  - 衬度传递函数 CTF（含 Cs、离焦、像散、时间/空间相干包络、孔径光阑）
  - HRTEM 相位衬度像：I = |F⁻¹[F[ψ_exit] · CTF]|²
  - 衍射花样：|F[ψ_exit]|²（对应 SimulaTEM 的 diffraction pattern）
  - STEM-ADF 成像（对应 SimulaTEM 的 annular aperture / CBD 功能）
  - 图像保存（PNG / 16-bit TIFF，对应 SimulaTEM 的 .bmp/.tif 输出）

离焦符号约定与 SimulaTEM 一致：正 = 过焦，负 = 欠焦。
像差函数：χ(k) = π λ Δf k² + (π/2) Cs λ³ k⁴ + π λ A k² cos2(θ-φ)
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

from .microscope import Microscope
from .multislice import ExitWave, build_slice_transmissions
from .structure import Structure


# ----------------------------------------------------------------------
# 频率网格与像差
# ----------------------------------------------------------------------
def frequency_grid(
    shape: Tuple[int, int], sampling: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (kx², ky² 组合的 k², kx, ky) 频率网格（单位 Å⁻¹，未 fftshift）。"""
    ny, nx = shape
    sx, sy = sampling
    kx = np.fft.fftfreq(nx, d=sx)
    ky = np.fft.fftfreq(ny, d=sy)
    KX, KY = np.meshgrid(kx, ky)
    return KX**2 + KY**2, KX, KY


def _chi_from_grid(scope: Microscope, k2: np.ndarray, KX: np.ndarray, KY: np.ndarray) -> np.ndarray:
    """由频率网格计算像差函数 χ(k)（rad）。"""
    lam = scope.wavelength
    chi = np.pi * scope.defocus * lam * k2 + 0.5 * np.pi * scope.cs * lam**3 * k2**2
    if scope.astigmatism != 0.0:
        theta = np.arctan2(KY, KX)
        phi = np.deg2rad(scope.astigmatism_azimuth)
        chi += np.pi * scope.astigmatism * lam * k2 * np.cos(2.0 * (theta - phi))
    return chi


def _mask_from_grid(scope: Microscope, k: np.ndarray, soft_width_mrad: float = 0.0) -> np.ndarray:
    """由 |k| 网格计算孔径掩模（支持环形光阑与软边）。"""
    lam = scope.wavelength
    if not scope.aperture_outer:
        return np.ones_like(k)
    k_out = (scope.aperture_outer * 1e-3) / lam
    k_in = (max(scope.aperture_inner, 0.0) * 1e-3) / lam

    if soft_width_mrad > 0:
        dk = (soft_width_mrad * 1e-3) / lam
        outer = 0.5 * (1.0 - np.tanh((k - k_out) / (dk + 1e-12)))
        inner = 0.5 * (1.0 + np.tanh((k - k_in) / (dk + 1e-12))) if k_in > 0 else 1.0
        return outer * inner
    return ((k <= k_out) & (k >= k_in)).astype(float)


def aberration_phase(scope: Microscope, shape, sampling) -> np.ndarray:
    """像差函数 χ(k)，rad。"""
    k2, KX, KY = frequency_grid(shape, sampling)
    return _chi_from_grid(scope, k2, KX, KY)


def aperture_mask(
    scope: Microscope,
    shape,
    sampling,
    soft_width_mrad: float = 0.0,
) -> np.ndarray:
    """孔径光阑掩模（支持环形光阑）。

    SimulaTEM 的孔径以倒易 Å 为单位指定；本工具统一用散射半角 mrad：
    k = sinθ/λ ≈ θ/λ。aperture_inner > 0 时为环形光阑
    （对应 annular_apperture_help.txt 描述的行为）。
    """
    k2, _, _ = frequency_grid(shape, sampling)
    return _mask_from_grid(scope, np.sqrt(k2), soft_width_mrad)


def ctf(scope: Microscope, shape, sampling) -> np.ndarray:
    """衬度传递函数 CTF(k) = E_t · E_s · A(k) · exp(-i χ(k))。

    包络函数：
      时间包络（离焦展宽 δ）：E_t = exp(-π² λ² δ² k⁴ / 2)
      空间包络（光束发散 α）：E_s = exp(-2π² α² k² (Δf + Cs λ² k²)²)
    （包络系数沿用 SimulaTEM 惯例；α 为束发散半角，输入 mrad。）
    """
    lam = scope.wavelength
    # 频率网格只构造一次，χ/包络/掩模共用
    k2, KX, KY = frequency_grid(shape, sampling)
    k = np.sqrt(k2)

    chi = _chi_from_grid(scope, k2, KX, KY)
    envelope = np.ones_like(k2)

    if scope.focal_spread > 0:
        envelope *= np.exp(-0.5 * (np.pi * lam * scope.focal_spread * k2) ** 2)
    if scope.angular_spread > 0:
        alpha = scope.angular_spread * 1e-3  # mrad → rad
        envelope *= np.exp(
            -2.0
            * np.pi**2
            * alpha**2
            * k2
            * (scope.defocus + scope.cs * lam**2 * k2) ** 2
        )

    mask = _mask_from_grid(scope, k)
    return envelope * mask * np.exp(-1j * chi)


# ----------------------------------------------------------------------
# Scherzer / 分辨率辅助
# ----------------------------------------------------------------------
def scherzer_defocus(scope: Microscope) -> float:
    """Scherzer 离焦（Å，负值 = 欠焦）。"""
    return scope.scherzer_defocus()


def point_resolution(scope: Microscope) -> float:
    """Scherzer 点分辨率（Å）。"""
    return scope.point_resolution()


# ----------------------------------------------------------------------
# HRTEM 成像与衍射
# ----------------------------------------------------------------------
def hrtem_image(exit_wave: ExitWave, scope: Microscope) -> np.ndarray:
    """HRTEM 相位衬度像强度 I = |F⁻¹[F[ψ]·CTF]|²。

    对应 SimulaTEM 的 "Single image" 计算。
    """
    ctf_arr = ctf(scope, exit_wave.shape, exit_wave.sampling)
    psi_img = np.fft.ifft2(np.fft.fft2(exit_wave.array) * ctf_arr)
    return np.abs(psi_img) ** 2


def diffraction_pattern(exit_wave: ExitWave, shift: bool = True) -> np.ndarray:
    """衍射花样强度 |F[ψ]|²（fftshift 后中心为透射斑）。

    对应 SimulaTEM 的 diffraction pattern 输出。
    """
    dp = np.abs(np.fft.fft2(exit_wave.array)) ** 2
    return np.fft.fftshift(dp) if shift else dp


def ctf_curve(scope: Microscope, k_max: float = 1.5, n: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """一维 CTF 曲线 -sin χ · E（对应 SimulaTEM 主窗口底部的 CTF 图）。"""
    k = np.linspace(1e-4, k_max, n)
    lam = scope.wavelength
    chi = np.pi * scope.defocus * lam * k**2 + 0.5 * np.pi * scope.cs * lam**3 * k**4
    env = np.ones_like(k)
    if scope.focal_spread > 0:
        env *= np.exp(-0.5 * (np.pi * lam * scope.focal_spread * k**2) ** 2)
    if scope.angular_spread > 0:
        alpha = scope.angular_spread * 1e-3
        env *= np.exp(-2.0 * np.pi**2 * alpha**2 * k**2 * (scope.defocus + scope.cs * lam**2 * k**2) ** 2)
    return k, -np.sin(chi) * env


# ----------------------------------------------------------------------
# STEM-ADF（环形暗场）
# ----------------------------------------------------------------------
def stem_adf(
    structure: Structure,
    scope: Microscope,
    probe_semiangle: float = 20.0,
    inner_angle: float = 60.0,
    outer_angle: float = 200.0,
    scan_pts: int = 64,
    sampling: Optional[float] = 0.05,
    gpts: Optional[int] = None,
    slice_thickness: float = 2.0,
    padding: float = 5.0,
    verbose: bool = True,
    table: str = "peng",
) -> Tuple[np.ndarray, float]:
    """STEM 环形暗场（ADF）成像。

    对应 SimulaTEM 的 annular aperture + 收敛束（CBD）功能：
    用带像差的收敛探针逐点扫描，在衍射平面用环形探测器收集强度。

    Parameters
    ----------
    probe_semiangle : float
        探针会聚半角，mrad。
    inner_angle, outer_angle : float
        ADF 探测器内/外半角，mrad。
    scan_pts : int
        扫描网格点数（scan_pts × scan_pts）。
    table : str
        散射因子表（"peng" 默认 / "gauss3" legacy）。

    Returns
    -------
    image : ndarray (scan_pts, scan_pts)
    scan_step_x, scan_step_y : float
        x/y 方向扫描步长，Å（lx≠ly 时两者不同）。

    Warning
    -------
    本实现逐扫描点做全网格 FFT（O(scan_pts²·n_slices·N²logN)），大扫描
    网格 + 厚样品时非常慢，正式 STEM 定量模拟建议用 abTEM 等专用引擎。
    探针会聚半角超出网格奈奎斯特时直接报错；ADF 探测器外角超出时告警
    （超出部分被截断）。
    """
    transmissions, (sx, sy), (lx, ly, lz) = build_slice_transmissions(
        structure, scope, sampling, gpts, slice_thickness, padding, table=table
    )
    ny, nx = transmissions[0].shape
    lam = scope.wavelength
    n_slices = len(transmissions)
    dz = lz / n_slices

    k2, _, _ = frequency_grid((ny, nx), (sx, sy))
    k = np.sqrt(k2)

    # 网格奈奎斯特频率（Å⁻¹）：探针/探测器角度不得超出，否则高频被折叠
    k_nyq = 0.5 * min(1.0 / sx, 1.0 / sy)
    angle_nyq_mrad = k_nyq * lam * 1e3
    k_probe = (probe_semiangle * 1e-3) / lam
    if k_probe > k_nyq:
        raise ValueError(
            f"探针会聚半角 {probe_semiangle:g} mrad 对应 k = {k_probe:.3f} Å⁻¹，"
            f"超出网格奈奎斯特 {k_nyq:.3f} Å⁻¹（≈{angle_nyq_mrad:.1f} mrad）；"
            "请减小采样间距（更多像素）或减小会聚半角"
        )

    # 探针：会聚孔径 × 像差相位
    probe_ap = (k <= k_probe).astype(float)
    chi = aberration_phase(scope, (ny, nx), (sx, sy))
    probe_ft = probe_ap * np.exp(-1j * chi)
    probe = np.fft.ifft2(probe_ft)
    probe /= np.sqrt(np.sum(np.abs(probe) ** 2))  # 归一化为单位强度

    # ADF 探测器掩模（衍射平面，角度 θ ≈ λk）
    k_in = (inner_angle * 1e-3) / lam
    k_out = (outer_angle * 1e-3) / lam
    if k_out > k_nyq:
        warnings.warn(
            f"ADF 探测器外角 {outer_angle:g} mrad（k = {k_out:.2f} Å⁻¹）超出网格"
            f"奈奎斯特 {k_nyq:.2f} Å⁻¹（≈{angle_nyq_mrad:.1f} mrad），"
            "超出部分将被截断，探测强度偏低",
            stacklevel=2,
        )
    detector = ((k >= k_in) & (k <= k_out)).astype(float)

    # 菲涅尔传播子
    kx = np.fft.fftfreq(nx, d=sx)
    ky = np.fft.fftfreq(ny, d=sy)
    K2 = kx[None, :] ** 2 + ky[:, None] ** 2
    prop = np.exp(-1j * np.pi * lam * dz * K2)

    scan_step_x = lx / scan_pts
    scan_step_y = ly / scan_pts
    image = np.zeros((scan_pts, scan_pts), dtype=float)

    probe_ft0 = np.fft.fft2(probe)
    kx_ramp = 2.0 * np.pi * kx
    ky_ramp = 2.0 * np.pi * ky

    for iy in range(scan_pts):
        y0 = iy * scan_step_y
        for ix in range(scan_pts):
            x0 = ix * scan_step_x
            # 探针平移（傅里叶相位斜坡，自动满足周期边界）
            shift_ft = np.exp(-1j * (kx_ramp[None, :] * x0 + ky_ramp[:, None] * y0))
            psi = np.fft.ifft2(probe_ft0 * shift_ft)
            for t in transmissions:
                psi *= t
                psi = np.fft.ifft2(np.fft.fft2(psi) * prop)
            diff = np.fft.fft2(psi)
            image[iy, ix] = np.sum(np.abs(diff) ** 2 * detector)
        if verbose:
            print(f"  STEM 扫描: 行 {iy + 1}/{scan_pts}")
    return image, scan_step_x, scan_step_y


# ----------------------------------------------------------------------
# 图像保存统一由 hrtem_tool.export 提供（robust 0.05–99.95 归一口径），
# 引擎层不再另设 min-max 归一的 save_* 函数以免口径混淆。
# ----------------------------------------------------------------------
