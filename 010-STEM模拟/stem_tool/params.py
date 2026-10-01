"""STEM 模拟参数模型：GUI 表单 ↔ 引擎参数之间的转换层。

单位约定（界面统一用对电镜工作者直观的单位，与 09-HRTEM模拟 保持一致）：
  - 厚度 / 扫描视场 / 离焦：nm（引擎内部 Å）
  - 球差 Cs：mm（引擎内部 Å，1 mm = 1e7 Å，可为负 = 校正器过校）
  - 会聚角 / 探测器角度 / 束倾斜：mrad；电压：kV；采样：Å/px
  - 离焦符号沿用 SimulaTEM 惯例：正 = 过焦，负 = 欠焦

与 HRTEM 工具的关键差别
----------------------
* STEM 用**会聚探针**，探针离焦的最优值 Δf = −√(|Cs|λ) 与 TEM 相位衬度的
  Scherzer 离焦 Δf = −√(1.5|Cs|λ) 不同，界面分别提供。
* 探测器外角决定了所需采样（奈奎斯特）：Δx ≤ λ/(2·sinθ_out)，
  这是 HAADF 模拟最主要的计算代价来源，界面实时显示。
* 部分相干（源尺寸/能量展宽）未实现，见 README「已知边界」；
  「显示模糊」只是经验性的图像模糊，不是物理上的部分相干模型。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from stem_sim import PhononConfig, ScanGeometry, StemOptics
from stem_sim.detectors import RingDetector

# 采样自动模式的上下限（Å/px）：低于下限内存/时间不可控，高于上限
# 连探针都采不好
SAMPLING_MIN = 0.02
SAMPLING_MAX = 0.4

# 质量档：采样 0 = 按探测器外角自动（推荐的物理自洽做法）。
# 耗时对"扫描视场"最敏感（网格点数 ∝ 视场²），故各档只调采样精细度、
# 切片厚度、声子组态数与扫描点数，视场由用户按需求给。
# 参考耗时（Au [100]、t=10 nm、视场 2 nm、300 kV、50–100 mrad、24 核并行）：
#   预览 ≈ 20 s   标准 ≈ 2 min   高质量 ≈ 12 min
QUALITY_PRESETS = {
    "预览": dict(sampling_a=0.0, slice_thickness_a=3.0, n_phonons=2,
                 scan_points=24, parallel="auto"),
    "标准": dict(sampling_a=0.0, slice_thickness_a=2.5, n_phonons=8,
                 scan_points=40, parallel="auto"),
    "高质量": dict(sampling_a=0.0, slice_thickness_a=2.0, n_phonons=16,
                   scan_points=64, parallel="auto"),
}


@dataclass
class StemParams:
    """一次 STEM（HAADF/ADF）模拟的全部可调参数。"""

    # ---- 结构与取向 ----
    cif_path: str = ""
    zone: Tuple[int, int, int] = (1, 0, 0)
    thickness_nm: float = 10.0
    scan_fov_nm: float = 2.0           # 扫描视场（nm，正方形；原子分辨率常用 1–5 nm）
    scan_points: int = 40              # 每边扫描点数
    rotation_deg: float = 0.0          # 面内旋转（绕光轴，逆时针为正）
    mirror: bool = False

    # ---- 电镜与探针 ----
    voltage_kv: float = 300.0
    cs_mm: float = 0.01                # 0.01 mm = 10 µm（球差校正）
    defocus_nm: float = -4.44          # 探针离焦（Cs=0.01 mm 的 Scherzer 探针值）
    probe_semiangle_mrad: float = 25.0
    probe_soft_mrad: float = 0.0
    tilt_x_mrad: float = 0.0
    tilt_y_mrad: float = 0.0
    astigmatism_a: float = 0.0
    astigmatism_azimuth_deg: float = 0.0

    # ---- 探测器 ----
    detector_inner_mrad: float = 50.0
    detector_outer_mrad: float = 100.0
    detector_soft_mrad: float = 0.0
    with_bright_field: bool = True     # 同时输出 BF / ABF
    display_detector: str = "ADF"      # 显示哪个探测器的像

    # ---- 冻结声子 ----
    n_phonons: int = 8
    temperature_k: float = 300.0
    debye_temperature_k: float = 0.0   # 0 = 用内置元素表
    sigma_override_a: float = 0.0      # 0 = 用 Debye 模型

    # ---- 多层法网格 ----
    sampling_a: float = 0.0            # 0 = 按探测器外角自动
    slice_thickness_a: float = 2.5

    # ---- 性能 ----
    parallel: str = "auto"             # auto / off
    threads: int = 0                   # 0 = 引擎默认

    # ---- 输出与显示 ----
    output_sampling_a: float = 0.0     # 0 = 保持原生采样
    polarity: int = 1                  # 1 = 正常（亮=强），-1 = 反转
    display_blur_a: float = 0.0        # 经验性图像模糊（不是部分相干模型）
    contrast_lo_pct: float = 0.5
    contrast_hi_pct: float = 99.5

    # ------------------------------------------------------------------
    def replace(self, **kw) -> "StemParams":
        data = asdict(self)
        for key, value in kw.items():
            if key not in data:
                raise KeyError(key)
            data[key] = value
        return StemParams(**data)

    def apply_quality(self, name: str) -> "StemParams":
        """套用质量档（只改采样/切片/声子数/扫描点数/并行）。"""
        if name not in QUALITY_PRESETS:
            raise KeyError(f"未知质量档 {name!r}（可选 {list(QUALITY_PRESETS)}）")
        return self.replace(**QUALITY_PRESETS[name])

    # ------------------------------------------------------------------
    def validate(self) -> None:
        def _finite(name: str, value) -> float:
            v = float(value)
            if not np.isfinite(v):
                raise ValueError(f"{name} 必须为有限数值")
            return v

        zone = [float(v) for v in self.zone]
        if not all(np.isfinite(v) for v in zone):
            raise ValueError("带轴指数必须为有限整数")
        if not any(int(v) for v in self.zone):
            raise ValueError("带轴不能为 [0 0 0]")

        if _finite("厚度", self.thickness_nm) < 0.1:
            raise ValueError("厚度需 ≥ 0.1 nm")
        if _finite("扫描视场", self.scan_fov_nm) < 0.5:
            raise ValueError("扫描视场需 ≥ 0.5 nm")
        if int(self.scan_points) < 2:
            raise ValueError("扫描点数每边需 ≥ 2")
        if int(self.scan_points) > 1024:
            raise ValueError("扫描点数每边上限 1024（总点数将达百万量级）")

        if _finite("加速电压", self.voltage_kv) < 1.0:
            raise ValueError("加速电压需 ≥ 1 kV")
        if _finite("探针会聚半角", self.probe_semiangle_mrad) <= 0:
            raise ValueError("探针会聚半角需 > 0")
        if _finite("探测器外角", self.detector_outer_mrad) <= 0:
            raise ValueError("探测器外角需 > 0")
        for nm, val in (("探测器内角", self.detector_inner_mrad),
                        ("探针光阑软化", self.probe_soft_mrad),
                        ("探测器软化", self.detector_soft_mrad)):
            if _finite(nm, val) < 0:
                raise ValueError(f"{nm}不能为负")

        if int(self.n_phonons) < 1:
            raise ValueError("声子组态数需 ≥ 1")
        if int(self.n_phonons) > 256:
            raise ValueError("声子组态数上限 256")
        if _finite("样品温度", self.temperature_k) <= 0:
            raise ValueError("样品温度需 > 0 K")
        if _finite("Debye 温度", self.debye_temperature_k) < 0:
            raise ValueError("Debye 温度不能为负（0 = 用内置表）")
        if _finite("位移幅度", self.sigma_override_a) < 0:
            raise ValueError("位移幅度不能为负（0 = 用 Debye 模型）")

        if _finite("采样", self.sampling_a) < 0:
            raise ValueError("采样不能为负（0 = 自动）")
        if 0 < self.sampling_a < SAMPLING_MIN:
            raise ValueError(f"采样过细（< {SAMPLING_MIN} Å/px），计算量与内存不可控")
        if self.sampling_a > SAMPLING_MAX:
            raise ValueError(f"采样过粗（> {SAMPLING_MAX} Å/px），探针无法正确取样")
        if _finite("切片厚度", self.slice_thickness_a) < 0.2:
            raise ValueError("切片厚度需 ≥ 0.2 Å")
        if self.parallel not in ("auto", "off"):
            raise ValueError("并行模式只能是 auto 或 off")
        if int(self.threads) < 0:
            raise ValueError("FFT 线程数不能为负（0 = 默认）")

        if _finite("输出采样", self.output_sampling_a) < 0:
            raise ValueError("输出采样不能为负（0 = 原生）")
        if 0 < self.output_sampling_a < 0.02:
            raise ValueError("输出采样 < 0.02 Å/px，渲染尺寸将超出内存合理范围")
        if _finite("显示模糊", self.display_blur_a) < 0:
            raise ValueError("显示模糊不能为负")
        lo, hi = float(self.contrast_lo_pct), float(self.contrast_hi_pct)
        if not (0.0 <= lo < hi <= 100.0):
            raise ValueError("对比度百分位需满足 0 ≤ lo < hi ≤ 100")

    # ------------------------------------------------------------------
    @property
    def zone_str(self) -> str:
        return "[{} {} {}]".format(*self.zone)

    @property
    def scan_fov_a(self) -> float:
        return float(self.scan_fov_nm) * 10.0

    @property
    def scan_step_a(self) -> float:
        return self.scan_fov_a / max(int(self.scan_points), 1)

    def scope(self) -> StemOptics:
        """转换成引擎 StemOptics（nm→Å，mm→Å）。"""
        return StemOptics(
            voltage_kv=float(self.voltage_kv),
            cs=float(self.cs_mm) * 1.0e7,
            defocus=float(self.defocus_nm) * 10.0,
            astigmatism=float(self.astigmatism_a),
            astigmatism_azimuth=float(self.astigmatism_azimuth_deg),
            probe_semiangle=float(self.probe_semiangle_mrad),
            probe_soft=float(self.probe_soft_mrad),
            tilt_x=float(self.tilt_x_mrad),
            tilt_y=float(self.tilt_y_mrad),
            detector_inner=float(self.detector_inner_mrad),
            detector_outer=float(self.detector_outer_mrad),
            detector_soft=float(self.detector_soft_mrad),
        )

    def effective_sampling(self) -> float:
        """实际使用的目标采样 Å/px（0 = 自动时按探测器外角反解）。"""
        if float(self.sampling_a) > 0:
            return float(self.sampling_a)
        s = self.scope().required_sampling()
        return float(min(max(s, SAMPLING_MIN), SAMPLING_MAX))

    def geometry(self) -> ScanGeometry:
        n = int(self.scan_points)
        return ScanGeometry(self.scan_fov_a, self.scan_fov_a, n, n)

    def phonons(self) -> PhononConfig:
        return PhononConfig(
            n_configs=int(self.n_phonons),
            temperature=float(self.temperature_k),
            debye_temperature=(float(self.debye_temperature_k)
                               if float(self.debye_temperature_k) > 0 else None),
            sigma_override=(float(self.sigma_override_a)
                            if float(self.sigma_override_a) > 0 else None),
        )

    def detectors(self) -> list:
        dets = [RingDetector("ADF", float(self.detector_inner_mrad),
                             float(self.detector_outer_mrad),
                             float(self.detector_soft_mrad))]
        if self.with_bright_field:
            alpha = float(self.probe_semiangle_mrad)
            dets.append(RingDetector("BF", 0.0, alpha, float(self.detector_soft_mrad)))
            dets.append(RingDetector("ABF", 0.6 * alpha, alpha,
                                     float(self.detector_soft_mrad)))
        return dets

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "StemParams":
        valid = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in valid}
        params = cls(**kwargs)
        params.zone = tuple(int(v) for v in params.zone)
        params.scan_points = int(params.scan_points)
        params.n_phonons = int(params.n_phonons)
        params.threads = int(params.threads)
        params.validate()
        return params

    def save_json(self, path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load_json(cls, path) -> "StemParams":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def describe(self, extra: str = "") -> str:
        """参数摘要（图片角标 / TIFF 描述通用）。"""
        lines = [
            f"V = {self.voltage_kv:g} kV   Cs = {self.cs_mm:g} mm   "
            f"df = {self.defocus_nm:+g} nm",
            f"探针 = {self.probe_semiangle_mrad:g} mrad   "
            f"探测器 = {self.detector_inner_mrad:g}–{self.detector_outer_mrad:g} mrad",
            f"t = {self.thickness_nm:g} nm   视场 = {self.scan_fov_nm:g} nm / "
            f"{self.scan_points}² 点",
            f"声子 = {self.n_phonons} 组态 @ {self.temperature_k:g} K",
        ]
        if extra:
            lines.append(extra)
        return "\n".join(lines)
