"""
物理常数与相对论电子光学公式。

对应 SimulaTEM 中隐含的电子光学基础：
  - 相对论电子波长 λ(V)
  - 相对论因子 γ
  - 相互作用参数 σ（Multislice 透射函数 q = exp(i σ V_p) 中的 σ）
"""

from __future__ import annotations

import numpy as np

# 基本物理常数（SI）
M0 = 9.1093837015e-31       # 电子静止质量 kg
E_CHARGE = 1.602176634e-19  # 元电荷 C
H_PLANCK = 6.62607015e-34   # 普朗克常数 J·s
C_LIGHT = 2.99792458e8      # 光速 m/s

# 电子静止能量对应的电压（V）：m0 c^2 / e ≈ 510998.95 V
REST_VOLTAGE = M0 * C_LIGHT**2 / E_CHARGE


def gamma_factor(voltage: float) -> float:
    """相对论因子 γ = 1 + eV/(m0 c²)。

    Parameters
    ----------
    voltage : float
        加速电压，单位 V（伏特）。
    """
    return 1.0 + voltage / REST_VOLTAGE


def electron_wavelength(voltage_kv: float) -> float:
    """相对论电子波长。

    λ = h / sqrt(2 m0 e V (1 + eV / (2 m0 c²)))

    Parameters
    ----------
    voltage_kv : float
        加速电压，单位 kV。

    Returns
    -------
    float
        波长，单位 Å。
    """
    v = float(voltage_kv) * 1e3  # kV -> V
    lam_m = H_PLANCK / np.sqrt(
        2.0 * M0 * E_CHARGE * v * (1.0 + E_CHARGE * v / (2.0 * M0 * C_LIGHT**2))
    )
    return float(lam_m * 1e10)  # m -> Å


def interaction_sigma(voltage_kv: float) -> float:
    """电子-样品相互作用参数 σ = 2π γ m0 e λ / h²。

    Multislice 透射函数为 q(x,y) = exp(i σ V_p(x,y))，
    其中 V_p 为投影静电势（V·Å）。

    Parameters
    ----------
    voltage_kv : float
        加速电压，单位 kV。

    Returns
    -------
    float
        σ，单位 rad/(V·Å)。例如 300 kV 时 σ ≈ 6.53e-4。
    """
    v = float(voltage_kv) * 1e3
    g = gamma_factor(v)
    lam_m = electron_wavelength(voltage_kv) * 1e-10  # Å -> m
    sigma_m = 2.0 * np.pi * g * M0 * E_CHARGE * lam_m / H_PLANCK**2  # rad/(V·m)
    return float(sigma_m * 1e-10)  # rad/(V·Å)
