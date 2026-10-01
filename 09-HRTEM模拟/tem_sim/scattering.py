"""
电子散射因子的高斯参数化与投影势相位核。

默认使用 Peng 标准参数化（Peng, Micron 30(6): 625–648, 1999；数据取自
abTEM 仓库 peng_high.json，98 元素。注意 abTEM 原文件的自变量为 g/2，
此处已按 b_phys = b_json/4 换算为以 |g|（Å⁻¹）为自变量的标准形式）：

    f_e(g) = Σ_{i=1..5} a_i · exp(-b_i g²)         [f_e: Å, g: Å⁻¹]

该表的绝对标度经过双重独立验证：
  - f_e(0)_H = Σa = 0.5288 Å，与氢原子第一性原理值 0.5292 Å（Born 极限
    解析解）精确一致；
  - f_e(1 Å⁻¹)_H = 0.0895 Å，与 Mott–Bethe 公式 0.0889 Å 一致。

历史表 gauss3.txt（提取自 SimulaTEM，6 高斯，原约定 f = Σ a_i exp(-g²/b_i)）
保留为 legacy（table="gauss3"，内部换算为 b_canon = 1/b_legacy）。
2026-09 审查确认其绝对标度相对 Peng 标准系统性偏大 ~1.81–1.92×（均值
1.87），且 g 衰减形状不符（缺少窄 g 宽项），故默认不再用于新模拟，
仅供复现 SimulaTEM/0820 交付口径。

由二维傅里叶变换对 ∫ e^{-b g²} e^{-2πi g·ρ} d²g = (π/b) e^{-π²ρ²/b}
可得原子投影势相位：

    φ_atom(ρ) = γλ · Σ_i (π a_i / b_i) · exp(-π² ρ² / b_i)

离散化时对每个像素做解析积分（误差函数形式），保证 ∫K = γλ·a_i：

    K_i[m,n] = (γλ a_i / 4) · Δerf_x · Δerf_y,
    Δerf_x = erf(π x_{n+½}/√b_i) − erf(π x_{n−½}/√b_i)
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:  # scipy 可选：存在时用精确 erf
    from scipy.special import erf as _erf_impl

    def _erf(x: np.ndarray) -> np.ndarray:
        return _erf_impl(x)

except ImportError:  # Abramowitz-Stegun 7.1.26 近似，最大误差 ~1.5e-7

    def _erf(x: np.ndarray) -> np.ndarray:
        sign = np.sign(x)
        x = np.abs(x)
        t = 1.0 / (1.0 + 0.3275911 * x)
        y = 1.0 - (
            ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592
        ) * t * np.exp(-x * x)
        return sign * y


TABLES = ("peng", "gauss3")

# peng: 规范约定 f = Σ a exp(-b g²)；gauss3: 原始文件为 f = Σ a exp(-g²/b)，
# 由 get_factors 统一换算为规范约定后返回。
_DATA_FILES = {"peng": "peng_high.json", "gauss3": "gauss3.txt"}

# 各表每元素高斯项个数
_N_GAUSS = {"peng": 5, "gauss3": 6}

# _FACTOR_CACHE[table][symbol] = (a, b)，均为规范约定（b 单位 Å²）
_FACTOR_CACHE: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}


def _default_table_path(table: str) -> Path:
    with resources.as_file(
        resources.files(__package__) / "data" / _DATA_FILES[table]
    ) as p:
        return Path(p)


def load_peng(path: str | Path | None = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """读取 Peng 参数化 JSON（abTEM peng_high.json 格式）。

    JSON 每元素为 [[a1..a5], [b1..b5]]，其中 b 以 g/2 为自变量拟合，
    换算为规范约定 f = Σ a·exp(-b g²) 需取 b_phys = b_json / 4。

    Returns
    -------
    dict
        {元素符号: (a[5], b[5])}，a 单位 Å，b 单位 Å²。
    """
    if path is None:
        path = _default_table_path("peng")
    raw = json.loads(Path(path).read_text(encoding="ascii"))
    table: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sym, params in raw.items():
        a = np.asarray(params[0], dtype=float)
        b = np.asarray(params[1], dtype=float) / 4.0
        table[sym.strip().capitalize()] = (a, b)
    if not table:
        raise ValueError(f"未能在 {path} 中解析出散射因子数据")
    return table


def load_gauss3(path: str | Path | None = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """读取 SimulaTEM Gauss3 格式的散射因子表（legacy，原始约定）。

    每行格式：元素符号 + 6 个 a_i + 6 个 b_i（共 12 个数），
    原始约定 f_e(g) = Σ a_i · exp(-g²/b_i)。

    Returns
    -------
    dict
        {元素符号: (a[6], b[6])}，**原始约定**（b 与文件一致）。
        经 get_factors(table="gauss3") 访问时才换算为规范约定。
    """
    if path is None:
        path = _default_table_path("gauss3")
    path = Path(path)
    table: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2 * _N_GAUSS["gauss3"] + 1:
                continue
            sym = parts[0].strip().capitalize()
            values = np.array([float(x) for x in parts[1:]])
            n = _N_GAUSS["gauss3"]
            table[sym] = (values[:n], values[n:])
    if not table:
        raise ValueError(f"未能在 {path} 中解析出散射因子数据")
    return table


def _load_table(table: str, path: str | Path | None = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    raw = load_gauss3(path) if table == "gauss3" else load_peng(path)
    if table == "gauss3":
        # legacy 原始约定 f = Σ a exp(-g²/b) → 规范约定 f = Σ a exp(-b' g²)
        raw = {sym: (a, 1.0 / b) for sym, (a, b) in raw.items()}
    return raw


def get_factors(symbol: str, table: str = "peng") -> Tuple[np.ndarray, np.ndarray]:
    """返回元素的 (a, b) 高斯参数（规范约定 f = Σ a·exp(-b g²)，缓存访问）。"""
    if table not in TABLES:
        raise ValueError(f"未知散射因子表 '{table}'（可选 {TABLES}）")
    cached = _FACTOR_CACHE.get(table)
    if cached is None:
        cached = _FACTOR_CACHE.setdefault(table, _load_table(table))
    sym = symbol.strip().capitalize()
    if sym not in cached:
        raise KeyError(f"散射因子表 {table} 中没有元素 '{symbol}'")
    return cached[sym]


def electron_scattering_factor(
    symbol: str, g: np.ndarray | float, table: str = "peng"
) -> np.ndarray | float:
    """电子散射因子 f_e(g) = Σ a_i exp(-b_i g²)。

    Parameters
    ----------
    symbol : str
        元素符号，如 'Au', 'Fe'。
    g : float 或 ndarray
        空间频率模长 |g|，单位 Å⁻¹。
    table : str
        'peng'（默认，物理标准）或 'gauss3'（SimulaTEM legacy）。

    Returns
    -------
    f_e，单位 Å。
    """
    a, b = get_factors(symbol, table)
    scalar = np.ndim(g) == 0
    g = np.asarray(g, dtype=float)
    g2 = g * g
    fe = np.zeros_like(g2)
    for ai, bi in zip(a, b):
        fe += ai * np.exp(-bi * g2)
    return float(fe) if scalar else fe


def phase_kernels(
    symbol: str,
    sampling,
    gamma_lambda: float,
    cutoff: float = 12.0,
    table: str = "peng",
    phase_floor: Optional[float] = None,
) -> List[np.ndarray]:
    """构造元素的高斯相位核列表（用于 Multislice 切片相位累加）。

    每个核对应一个高斯项，像素值为相位在像素面积上的解析积分：
        K_i[m,n] = (γλ a_i / 4) · Δerf_x · Δerf_y
    该形式对任意采样间隔都保证 ∫K = γλ·a_i（总相位权重精确）。

    截断半径取 r_rel 与 r_abs 中较小者：
      - 相对阈值：exp(-π²r²/b) < exp(-cutoff)，即 r > √(b·cutoff)/π；
      - 绝对阈值（可选）：该项核峰值相位 γλ·π·a/b 经衰减低于 phase_floor
        （rad）即截断，用于压制宽尾项（Peng 表宽项 b 可达 ~90 Å²，
        相对阈值会给出 >400 px 的核）。为宽项损失的相位权重
        ≈ phase_floor·b/π rad·Å²，相对该项总权重 ≤0.1% 量级。

    Parameters
    ----------
    symbol : str
        元素符号。
    sampling : float 或 (sx, sy)
        网格采样间隔 Å/px；矩形像素可分别指定 x/y。
    gamma_lambda : float
        γ·λ（γ 为相对论因子，λ 为波长 Å）。
    cutoff : float
        高斯截断指数阈值（无量纲），越大越精确但越慢。
    table : str
        散射因子表（"peng" / "gauss3"）。
    phase_floor : float, 可选
        绝对相位截断阈值 rad；None 表示按表取默认
        （peng → 1e-6，gauss3 → 不启用，保持 legacy 数值逐位一致）。

    Returns
    -------
    list of ndarray
        各高斯项的 2D 相位核（奇数尺寸，形状 (ny, nx)，中心为峰值）。
    """
    try:
        sx, sy = float(sampling[0]), float(sampling[1])
    except (TypeError, IndexError):
        sx = sy = float(sampling)
    if phase_floor is None:
        phase_floor = 1e-6 if table == "peng" else None
    a, b = get_factors(symbol, table)
    kernels = []
    for ai, bi in zip(a, b):
        # 实空间项 γλ(πa/b)·exp(-π²r²/b) 的截断半径
        r_rel = np.sqrt(cutoff * bi) / np.pi
        if phase_floor is not None and phase_floor > 0:
            peak = gamma_lambda * np.pi * ai / bi
            if peak > phase_floor:
                r_abs = np.sqrt(bi * np.log(peak / phase_floor)) / np.pi
            else:
                r_abs = 0.0
            r_cut = min(r_rel, r_abs)
        else:
            r_cut = r_rel
        half_x = max(1, int(np.ceil(r_cut / sx)))
        half_y = max(1, int(np.ceil(r_cut / sy)))
        # 像素边界坐标（相对原子中心）
        edges_x = (np.arange(-half_x, half_x + 2) - 0.5) * sx
        edges_y = (np.arange(-half_y, half_y + 2) - 0.5) * sy
        beta = np.pi / np.sqrt(bi)
        d_erf_x = _erf(beta * edges_x[1:]) - _erf(beta * edges_x[:-1])
        d_erf_y = _erf(beta * edges_y[1:]) - _erf(beta * edges_y[:-1])
        k = (gamma_lambda * ai / 4.0) * np.outer(d_erf_y, d_erf_x)
        kernels.append(k)
    return kernels


# 暴露元素列表（读取默认表）
try:
    ELEMENTS: Tuple[str, ...] = tuple(sorted(_load_table("peng").keys(), key=len))
except Exception:  # pragma: no cover - 数据文件缺失时才触发
    ELEMENTS = ()
