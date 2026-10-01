"""stem_sim — HAADF-STEM 图像模拟引擎（冻结声子多层法）。

与 09-HRTEM模拟 的 tem_sim 引擎同源：共享同一套电子散射因子（Peng 1999
标准表 + erf 精确像素积分）、晶体带轴重构与显微光学参数定义，因此两个工具
的势场、波长、相对论因子口径完全一致；本包在其上增加了 STEM 专有的
会聚探针、批量探针多层法、热漫散射（冻结声子）与环形探测器。

单位约定（与 tem_sim 一致）：
  - 长度 Å；角度 mrad；电压 kV；球差 Cs Å（1 mm = 1e7 Å）
  - 离焦正 = 过焦、负 = 欠焦
  - 探测器信号为**比例**（0–1），即入射束流被探测器收集的份额

快速上手
--------
>>> from stem_sim import (StemOptics, ScanGeometry, PhononConfig,
...                       read_structure, zone_axis_cell, stem_scan)
>>> st = read_structure("cif/Au_fcc_Fm-3m.cif")
>>> crystal = zone_axis_cell(st, (1, 0, 0), target_xy=40.0, target_thickness=150.0)
>>> optics = StemOptics(voltage_kv=300.0, cs=1.0e4, defocus=-60.0,
...                     probe_semiangle=25.0, detector_inner=50.0,
...                     detector_outer=100.0)
>>> geom = ScanGeometry(fov_x=40.0, fov_y=40.0, nx=64, ny=64)
>>> res = stem_scan(crystal, optics, sampling=0.1, geometry=geom,
...                 phonons=PhononConfig(n_configs=8))
>>> res.images["ADF"].shape
(64, 64)
"""

from ._fft import fft_throughput, get_threads, set_threads
from .constants import (
    electron_wavelength,
    gamma_factor,
    interaction_sigma,
    angle_from_k,
    k_from_angle,
)
from .detectors import RingDetector, make_detectors
from .microscope import Microscope
from .multislice import build_slice_transmissions, multislice, multislice_series
from .phonons import (
    ATOMIC_MASS,
    DEBYE_TEMPERATURE,
    FrozenPhonons,
    PhononConfig,
    debye_waller_B,
    rms_displacement,
    sigma_from_B,
    sigma_table,
)
from .probe import (
    StemOptics,
    aberration_chi,
    probe_batch_ft,
    probe_ft,
    probe_intensity_profile,
    probe_wave,
)
from .scan import (
    CancelledError,
    ScanGeometry,
    StemResult,
    estimate_seconds,
    stem_scan,
)
from .scattering import (
    ELEMENTS,
    electron_scattering_factor,
    load_gauss3,
    load_peng,
)
from .structure import (
    Structure,
    from_ase,
    read_structure,
    zone_axis_cell,
    zone_axis_info,
)

__version__ = "0.1.0"

__all__ = [
    # 常数与换算
    "electron_wavelength",
    "gamma_factor",
    "interaction_sigma",
    "k_from_angle",
    "angle_from_k",
    # 结构
    "Structure",
    "read_structure",
    "from_ase",
    "zone_axis_cell",
    "zone_axis_info",
    # 光学
    "Microscope",
    "StemOptics",
    "aberration_chi",
    "probe_ft",
    "probe_wave",
    "probe_batch_ft",
    "probe_intensity_profile",
    # 声子
    "PhononConfig",
    "FrozenPhonons",
    "debye_waller_B",
    "rms_displacement",
    "sigma_from_B",
    "sigma_table",
    "DEBYE_TEMPERATURE",
    "ATOMIC_MASS",
    # 探测器与扫描
    "RingDetector",
    "make_detectors",
    "ScanGeometry",
    "StemResult",
    "stem_scan",
    "estimate_seconds",
    "CancelledError",
    "build_slice_transmissions",
    "multislice",
    "multislice_series",
    # 散射因子
    "ELEMENTS",
    "electron_scattering_factor",
    "load_peng",
    "load_gauss3",
    # 性能
    "get_threads",
    "set_threads",
    "fft_throughput",
]
