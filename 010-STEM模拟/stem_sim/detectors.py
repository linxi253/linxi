"""STEM 探测器：HAADF/ADF、BF、ABF 的衍射平面掩模与信号归一化。

探测器作用在**衍射平面（远场）强度**上。多层法给出出射波 ψ_exit(r)，其
傅里叶变换 Ψ(k) = F[ψ_exit] 的模方即衍射平面强度，k 与散射半角满足
sinθ = λk。探测器信号取

    S = (1/N) Σ_k |Ψ(k)|² · D(k)

其中 N 为网格点数。由 Parseval 关系 Σ_k|Ψ|² = N·Σ_r|ψ|²，当探针归一化为
Σ|ψ|² = 1 且多层法保持幺正时，S ∈ [0, 1] 且物理意义明确：**该探针位置下
入射束流被探测器收集的比例**。因此不同厚度、不同参数下的图像强度可以直接
定量比较（例如薄样品区 S 与厚度近似成正比，可用于厚度标定）。

本工具实现的探测器（均为轴对称圆形/环形）：
  - HAADF / ADF：环形，内角典型 2–3 倍探针角（60–200 mrad）
  - BF：中心圆孔，外角 ≤ 探针角
  - ABF：环形亮场，收集亮场盘外圈（内角 > 0，外角 ≤ 探针角）

**未实现**：分段探测器（四象限/环形分段）的对向差分 ABF、DPC（差分相位
衬度）的一阶矩信号。这类信号需要改变探测器几何而非仅掩模，见 README
"已知边界"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constants import angle_from_k, k_from_angle
from .probe import StemOptics, frequency_grid, soft_width_k

# 内置探测器类型：名称 → (内角来源, 外角来源)
BUILTIN = ("haadf", "adf", "bf", "abf")


@dataclass
class RingDetector:
    """轴对称环形探测器。

    Attributes
    ----------
    name : str
        输出名（也用于结果字典的键）。
    inner, outer : float
        内/外收集半角，mrad。
    soft : float
        边缘软化宽度，mrad（0 = 理想硬边）。
    clip_to_nyquist : bool
        True（默认）时把外角裁剪到网格的轴向奈奎斯特角
        θ_N = asin(λ/(2Δx))，超过该角的强度是混叠结果、不可信。
        仅在**诊断用途**（如检验全网格 Parseval 互补性）时才设为 False
        ——此时方形网格的角部（|k| 可达 √2·k_N）也会被计入，
        但这部分强度已经混叠，不能当作物理结果。
    """

    name: str
    inner: float
    outer: float
    soft: float = 0.0
    clip_to_nyquist: bool = True

    def validate(self, optics: StemOptics, sampling_a: float) -> List[str]:
        """校验并返回警告；内/外角会被就地裁剪到物理可收集范围。"""
        notes: List[str] = []
        if self.outer <= 0:
            raise ValueError(f"探测器 {self.name}: 外角需 > 0")
        if self.inner < 0:
            raise ValueError(f"探测器 {self.name}: 内角不能为负")
        if self.soft < 0:
            raise ValueError(f"探测器 {self.name}: 软化宽度不能为负")

        if self.clip_to_nyquist:
            theta_max = float(angle_from_k(1.0 / (2.0 * float(sampling_a)),
                                           optics.wavelength))
            if self.outer > theta_max:
                notes.append(
                    f"{self.name}: 外角 {self.outer:g} mrad 超过采样可收集上限 "
                    f"{theta_max:.1f} mrad，已裁剪（更小的 Å/px 可支持更大角度）"
                )
                self.outer = theta_max
        if self.inner >= self.outer:
            notes.append(
                f"{self.name}: 内角 {self.inner:g} ≥ 外角 {self.outer:g} mrad，"
                f"已改为中心圆孔 (0, {self.outer:.1f})"
            )
            self.inner = 0.0
        return notes

    def mask(self, shape: Tuple[int, int], sampling, optics: StemOptics) -> np.ndarray:
        """衍射平面掩模 D(k)（0–1，浮点以便支持软边与圆孔加权）。"""
        _, _, K2 = frequency_grid(shape, sampling)
        lam = optics.wavelength
        k_in = k_from_angle(self.inner, lam)
        k_out = k_from_angle(self.outer, lam)
        token = np.sqrt(K2)
        if self.soft > 0:
            d_out = soft_width_k(self.outer, self.soft, lam)
            outer = 0.5 * (1.0 - np.tanh((token - k_out) / max(d_out, 1e-12)))
            if self.inner > 0:
                d_in = soft_width_k(self.inner, self.soft, lam)
                inner = 0.5 * (1.0 + np.tanh((token - k_in) / max(d_in, 1e-12)))
            else:
                inner = 1.0
            return outer * inner
        return ((token <= k_out) & (token >= k_in)).astype(float)

    def describe(self) -> str:
        return f"{self.name}: {self.inner:g}–{self.outer:g} mrad"


def make_detectors(
    optics: StemOptics,
    sampling_a: float,
    with_bright_field: bool = True,
) -> Tuple[List[RingDetector], List[str]]:
    """按光学参数构造探测器组（HAADF 为主，可选同时输出 BF/ABF）。

    BF/ABF 由探针角自动定标，无需用户输入额外参数：
      BF  : 0 … probe_semiangle
      ABF : 0.6·probe_semiangle … probe_semiangle（亮场盘外圈）
    后者是 ABF（annular bright field）的核心几何：只收集亮场盘最外圈，
    对轻元素柱的灵敏度最高。

    Returns
    -------
    (detectors, notes)
    """
    dets: List[RingDetector] = [
        RingDetector("ADF", optics.detector_inner, optics.detector_outer, optics.detector_soft)
    ]
    if with_bright_field:
        alpha = optics.probe_semiangle
        dets.append(RingDetector("BF", 0.0, alpha, optics.detector_soft))
        dets.append(RingDetector("ABF", 0.6 * alpha, alpha, optics.detector_soft))
    notes: List[str] = []
    for det in dets:
        notes.extend(det.validate(optics, sampling_a))
    return dets, notes


def detector_signal(psi_k: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """由出射波的倒空间振幅算探测器信号（比例，0–1）。

    psi_k : ndarray (..., ny, nx) 出射波的 FFT（未 fftshift）。
    mask  : ndarray (ny, nx) 探测器掩模。
    返回  ndarray (...)（每行/每个探针一个值）。
    """
    inten = np.abs(psi_k) ** 2
    n = float(psi_k.shape[-1] * psi_k.shape[-2])
    return np.einsum("...ij,ij->...", inten, mask) / n
