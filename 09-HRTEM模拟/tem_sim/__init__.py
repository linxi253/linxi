"""
tem_sim — 基于 Python 的轻量级 TEM 模拟器（SimulaTEM 功能重构版）。

核心功能映射自 SimulaTEM v1.3.2（Alfredo Gomez & Luis Beltran del Rio）：
  - Cowley-Moody Multislice 算法
  - 原子散射因子的高斯参数化：默认 Peng 1999 标准表（每元素 5 高斯，
    绝对标度经第一性原理/文献验证）；SimulaTEM gauss3.txt（6 高斯）保留为
    legacy（table="gauss3"，仅供复现旧口径，其标度偏大 ~1.87×，见 README）
  - 显微镜参数：加速电压、球差 Cs、离焦、离焦展宽、光束发散、孔径光阑（含环形）
  - 输出：HRTEM 图像、衍射花样、STEM-ADF 图像

单位约定：
  - 长度：Å；角度：mrad；电压：kV；球差 Cs：Å（1 mm = 1e7 Å）
  - 离焦符号：正 = 过焦（over-focus），负 = 欠焦（under-focus），与 SimulaTEM 一致
"""

from .constants import electron_wavelength, gamma_factor, interaction_sigma
from .microscope import Microscope, load_generic_presets
from .scattering import (
    ELEMENTS,
    electron_scattering_factor,
    load_gauss3,
    load_peng,
)
from .structure import Structure, read_structure, zone_axis_cell, zone_axis_info
from .multislice import multislice, multislice_series, projected_phase, ExitWave
from .imaging import (
    ctf,
    hrtem_image,
    diffraction_pattern,
    stem_adf,
    scherzer_defocus,
    point_resolution,
)

__version__ = "1.2.0"

__all__ = [
    "electron_wavelength",
    "gamma_factor",
    "interaction_sigma",
    "Microscope",
    "load_generic_presets",
    "load_gauss3",
    "load_peng",
    "electron_scattering_factor",
    "ELEMENTS",
    "Structure",
    "read_structure",
    "zone_axis_cell",
    "zone_axis_info",
    "multislice",
    "multislice_series",
    "projected_phase",
    "ExitWave",
    "ctf",
    "hrtem_image",
    "diffraction_pattern",
    "stem_adf",
    "scherzer_defocus",
    "point_resolution",
]
