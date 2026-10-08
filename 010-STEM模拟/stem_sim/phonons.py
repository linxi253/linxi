"""冻结声子（frozen phonon）模型：热位移的产生与 Debye-Waller 因子。

HAADF 图像的 Z 衬度主要来自**热漫散射（TDS）**：声子把电子散射到高角，
环形探测器收集的正是这部分几乎完全非相干的强度。若只用静态势做弹性
多层法，高角信号严重偏弱，且重原子柱会因动力学衍射出现反常衬度
（薄样品下甚至反号）。因此本工具默认启用冻结声子：

1. 对每个原子按 Debye-Waller 因子给一个随机热位移 u（三个方向独立高斯）；
2. 用位移后的结构算一次完整多层法扫描，得到一张 ADF 图；
3. 对 N 个独立声子组态求强度平均。

这就是标准的冻结声子近似（Loane, Xu & Silcox 1991）。本实现采用**非关联**
高斯位移，即只保留 Debye-Waller 因子、不模拟声子色散关联带来的 Kikuchi
带等结构——这是 HAADF 成像模拟的通行做法（高角 TDS 的角分布对关联不敏感）。

Debye-Waller 因子（Warren, *X-Ray Diffraction*, 式 3.16 的 Debye 模型）：

    B = (6 h² / (m_a k_B Θ_D)) · [ φ(Θ_D/T) / (Θ_D/T) + 1/4 ]
    φ(x) = (1/x) ∫₀ˣ ξ/(e^ξ − 1) dξ
    B = 8π² ⟨u²⟩        （⟨u²⟩ 为一维均方位移，故逐方向 σ_u = √(B/8π²)）

对 Au（Θ_D = 165 K，T = 300 K）该式给出 B = 0.6482 Å²、σ_u = 0.0906 Å
（debye_waller_B("Au", 300) 的实算值，与 README「物理与验证」一节的数值
一致），落在实验区间 B ≈ 0.5–0.7 Å² 之内。不同文献对 Au 的 B 拟合值本身
存在离散，Debye 模型取「弹性」Θ_D 只是一阶近似，故工具允许三种输入：
Θ_D（默认）、直接给 σ_u（Å）、或直接给 B（Å²）；
界面会实时显示最终使用的 σ_u，便于与文献值对齐。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .structure import Structure

# ----------------------------------------------------------------------
# 元素数据：原子量（标准原子量）与 Debye 温度（CRC / Ashcroft-Mermin，
# 由低温热容与弹性常数定出的“弹性” Θ_D，是文献中最通用的一套）
# ----------------------------------------------------------------------
ATOMIC_MASS: Dict[str, float] = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.95, "Tc": 98.0, "Ru": 101.07, "Rh": 102.91,
    "Pd": 106.42, "Ag": 107.87, "Cd": 112.41, "In": 114.82, "Sn": 118.71,
    "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29, "Cs": 132.91,
    "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24,
    "Pm": 145.0, "Sm": 150.36, "Eu": 151.96, "Gd": 157.25, "Tb": 158.93,
    "Dy": 162.50, "Ho": 164.93, "Er": 167.26, "Tm": 168.93, "Yb": 173.05,
    "Lu": 174.97, "Hf": 178.49, "Ta": 180.95, "W": 183.84, "Re": 186.21,
    "Os": 190.23, "Ir": 192.22, "Pt": 195.08, "Au": 196.97, "Hg": 200.59,
    "Tl": 204.38, "Pb": 207.2, "Bi": 208.98, "Po": 209.0, "At": 210.0,
    "Rn": 222.0, "Fr": 223.0, "Ra": 226.0, "Ac": 227.0, "Th": 232.04,
    "Pa": 231.04, "U": 238.03,
}

DEBYE_TEMPERATURE: Dict[str, float] = {
    "H": 110.0, "He": 30.0, "Li": 344.0, "Be": 1440.0, "B": 1250.0,
    "C": 2230.0, "N": 900.0, "O": 700.0, "F": 500.0, "Ne": 75.0,
    "Na": 158.0, "Mg": 400.0, "Al": 428.0, "Si": 645.0, "P": 500.0,
    "S": 400.0, "Cl": 300.0, "Ar": 92.0, "K": 91.0, "Ca": 230.0,
    "Sc": 360.0, "Ti": 420.0, "V": 380.0, "Cr": 630.0, "Mn": 410.0,
    "Fe": 470.0, "Co": 445.0, "Ni": 450.0, "Cu": 343.0, "Zn": 327.0,
    "Ga": 240.0, "Ge": 360.0, "As": 282.0, "Se": 90.0, "Br": 150.0,
    "Kr": 72.0, "Rb": 56.0, "Sr": 147.0, "Y": 280.0, "Zr": 291.0,
    "Nb": 275.0, "Mo": 450.0, "Tc": 400.0, "Ru": 600.0, "Rh": 480.0,
    "Pd": 274.0, "Ag": 225.0, "Cd": 209.0, "In": 108.0, "Sn": 200.0,
    "Sb": 211.0, "Te": 153.0, "I": 106.0, "Xe": 64.0, "Cs": 38.0,
    "Ba": 110.0, "La": 142.0, "Ce": 138.0, "Pr": 138.0, "Nd": 137.0,
    "Sm": 152.0, "Eu": 139.0, "Gd": 153.0, "Tb": 159.0, "Dy": 164.0,
    "Ho": 170.0, "Er": 173.0, "Tm": 180.0, "Yb": 118.0, "Lu": 184.0,
    "Hf": 252.0, "Ta": 240.0, "W": 400.0, "Re": 430.0, "Os": 500.0,
    "Ir": 420.0, "Pt": 240.0, "Au": 165.0, "Hg": 71.9, "Tl": 78.5,
    "Pb": 105.0, "Bi": 119.0, "Th": 163.0, "U": 207.0,
}

DEFAULT_DEBYE = 300.0  # 表中缺失元素时的回退值（并给出警告）

# 物理常数（与 constants.py 一致，此处独立列出避免循环导入）
_H_PLANCK = 6.62607015e-34   # J·s
_K_BOLTZ = 1.380649e-23      # J/K
_AMU = 1.66053906660e-27     # kg


# ----------------------------------------------------------------------
# Debye 积分
# ----------------------------------------------------------------------
def debye_phi(x: float) -> float:
    """φ(x) = (1/x)∫₀ˣ ξ/(e^ξ−1) dξ，即 Debye 函数与 1/x 的乘积。

    小 x 用级数 ξ/(e^ξ−1) = 1 − ξ/2 + ξ²/12 − ξ⁴/720 + …
    （避免 x→0 时 0/0），其余用 Simpson 数值积分。
    """
    x = float(x)
    if x <= 0.0:
        return 1.0
    if x < 1e-3:
        # ∫₀ˣ (1 − ξ/2 + ξ²/12) dξ = x − x²/4 + x³/36
        return 1.0 - x / 4.0 + x * x / 36.0
    n = 2000
    t = np.linspace(0.0, x, n + 1)
    # t=0 处 ξ/(e^ξ−1) → 1，用 np.where 避免 0/0 警告
    safe = np.where(t > 0, t, 1.0)
    integrand = np.where(t > 0, safe / np.expm1(safe), 1.0)
    integral = float(np.trapezoid(integrand, t))
    return integral / x


def debye_waller_B(
    element: str,
    temperature: float = 300.0,
    debye_temperature: Optional[float] = None,
) -> float:
    """Debye 模型的 Debye-Waller 因子 B（Å²）。

    Parameters
    ----------
    element : str
        元素符号。
    temperature : float
        样品温度 K。
    debye_temperature : float, 可选
        Θ_D（K）；None 时取内置表，表中缺失则用默认值并仍照常计算
        （有效性由 `resolve_sigma` 负责提示）。
    """
    sym = element.strip().capitalize()
    theta = debye_temperature if debye_temperature else DEBYE_TEMPERATURE.get(sym, DEFAULT_DEBYE)
    theta = float(theta)
    if theta <= 0:
        raise ValueError(f"{element}: Debye 温度必须为正，得到 {theta}")
    t = max(float(temperature), 1e-6)
    mass = ATOMIC_MASS.get(sym)
    if mass is None:
        raise KeyError(f"缺少元素 {element} 的原子量，无法计算 Debye-Waller 因子")
    m = mass * _AMU
    prefactor = 6.0 * _H_PLANCK**2 / (m * _K_BOLTZ * theta)  # m²
    x = theta / t
    b_m2 = prefactor * (debye_phi(x) / x + 0.25)
    return float(b_m2 * 1e20)  # m² → Å²


def rms_displacement(
    element: str,
    temperature: float = 300.0,
    debye_temperature: Optional[float] = None,
) -> float:
    """由 Debye 模型给出的一维均方位移 σ_u = √(B/8π²)（Å）。"""
    return float(np.sqrt(max(debye_waller_B(element, temperature, debye_temperature), 0.0) / (8.0 * np.pi**2)))


def sigma_from_B(b_a2: float) -> float:
    """B（Å²）→ σ_u（Å）。"""
    return float(np.sqrt(max(float(b_a2), 0.0) / (8.0 * np.pi**2)))


def B_from_sigma(sigma_a: float) -> float:
    """σ_u（Å）→ B（Å²）。"""
    return float(8.0 * np.pi**2 * float(sigma_a) ** 2)


def sigma_table(
    symbols: Sequence[str],
    temperature: float = 300.0,
    debye_temperature: Optional[float] = None,
    override_sigma: Optional[float] = None,
) -> Tuple[Dict[str, float], List[str]]:
    """给出结构中每种元素的 σ_u（Å）与提示信息。

    Parameters
    ----------
    override_sigma : float, 可选
        若非 None，则所有元素统一使用该 σ_u（Å），忽略 Debye 模型。

    Returns
    -------
    (dict 元素→σ_u, list of 警告字符串)
    """
    out: Dict[str, float] = {}
    notes: List[str] = []
    present = sorted({s.strip().capitalize() for s in symbols})
    unknown = [s for s in present if s not in ATOMIC_MASS]
    if unknown:
        raise ValueError(
            f"以下元素没有内置原子量，无法计算热位移：{', '.join(unknown)}"
            "（请检查结构文件的元素符号，或手动指定位移幅度 σ）"
        )
    if override_sigma is not None:
        sigma = float(override_sigma)
        if sigma < 0:
            raise ValueError("位移幅度 σ 不能为负")
        for sym in present:
            out[sym] = sigma
        return out, notes

    missing = [s for s in present if s not in DEBYE_TEMPERATURE]
    if missing and debye_temperature is None:
        notes.append(
            "以下元素无内置 Debye 温度，已用默认 Θ_D = "
            f"{DEFAULT_DEBYE:g} K：{', '.join(missing)}"
        )
    for sym in present:
        out[sym] = rms_displacement(sym, temperature, debye_temperature)
    return out, notes


# ----------------------------------------------------------------------
# 声子组态
# ----------------------------------------------------------------------
@dataclass
class PhononConfig:
    """冻结声子配置。

    Attributes
    ----------
    n_configs : int
        独立声子组态数（HAADF 强度对组态数敏感，建议 ≥ 8）。
    temperature : float
        样品温度 K。
    debye_temperature : float, 可选
        统一覆盖的 Θ_D（K）；None = 用内置元素表。
    sigma_override : float, 可选
        统一覆盖的逐方向位移标准差 σ_u（Å）；None = 用 Debye 模型。
    seed : int
        随机种子（保证复现）。
    """

    n_configs: int = 8
    temperature: float = 300.0
    debye_temperature: Optional[float] = None
    sigma_override: Optional[float] = None
    seed: int = 20260915

    def validate(self) -> None:
        if int(self.n_configs) < 1:
            raise ValueError("声子组态数需 ≥ 1")
        if int(self.n_configs) > 256:
            raise ValueError("声子组态数上限 256（再多请改用系列模拟）")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("样品温度需为正的有限值")
        if self.debye_temperature is not None and self.debye_temperature <= 0:
            raise ValueError("Debye 温度需为正")
        if self.sigma_override is not None and self.sigma_override < 0:
            raise ValueError("位移幅度 σ 不能为负")

    def summary(self) -> str:
        if self.sigma_override is not None:
            return f"σ_u = {self.sigma_override:g} Å（手动），{self.n_configs} 组态"
        theta = f"{self.debye_temperature:g} K" if self.debye_temperature else "内置表"
        return f"T = {self.temperature:g} K，Θ_D = {theta}，{self.n_configs} 组态"


class FrozenPhonons:
    """按配置生成带热位移的结构副本（可复现、按组态独立）。

    Examples
    --------
    >>> configs = FrozenPhonons(crystal, PhononConfig(n_configs=4))
    >>> for displaced in configs:
    ...     ...  # displaced 为 Structure，原子数/晶胞与 crystal 相同
    """

    def __init__(self, structure: Structure, config: PhononConfig):
        config.validate()
        self.structure = structure
        self.config = config
        self.sigma, self.notes = sigma_table(
            structure.symbols,
            temperature=config.temperature,
            debye_temperature=config.debye_temperature,
            override_sigma=config.sigma_override,
        )

    def __len__(self) -> int:
        return int(self.config.n_configs)

    def sigma_for(self, symbol: str) -> float:
        return self.sigma[symbol.strip().capitalize()]

    def displacements(self, index: int) -> np.ndarray:
        """第 index 个组态的位移场，形状 (N, 3)，单位 Å。"""
        # 每个组态独立子种子：并行执行时结果与串行完全一致
        rng = np.random.default_rng((int(self.config.seed) + 7919 * int(index)) % (2**63))
        out = np.zeros((len(self.structure), 3), dtype=float)
        symbols = self.structure.symbols
        by_sym: Dict[str, List[int]] = {}
        for i, s in enumerate(symbols):
            by_sym.setdefault(s.strip().capitalize(), []).append(i)
        for sym, idx in by_sym.items():
            sigma = self.sigma[sym]
            if sigma <= 0:
                continue
            out[idx] = rng.normal(0.0, sigma, size=(len(idx), 3))
        return out

    def __getitem__(self, index: int) -> Structure:
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        disp = self.displacements(index)
        return Structure(
            list(self.structure.symbols),
            self.structure.positions + disp,
            self.structure.cell,
        )
