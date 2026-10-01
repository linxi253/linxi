"""
显微镜参数模型（对应 SimulaTEM 的 Microscope / Illumination 对话框）。

SimulaTEM 参数 → 本模块字段映射：
  Voltage (kV)            → voltage_kv
  Spherical aberration    → cs（本工具统一用 Å；SimulaTEM 输入为 mm，1 mm = 1e7 Å）
  Defocus spread (Å)      → focal_spread
  Beam spread (mrad)      → angular_spread
  Defocus (Å)             → defocus（正 = 过焦，负 = 欠焦，与 SimulaTEM 相同）
  Astigmatism (Å, deg)    → astigmatism, astigmatism_azimuth
  Aperture radius (1/Å)   → aperture_outer / aperture_inner（也支持 mrad 半角）
  CBD convergence (mrad)  → 通过 STEM 探针的 aperture_outer 实现
  自动 Scherzer 离焦       → Microscope.scherzer_defocus()
  最佳孔径（CTF 第一零点）  → Microscope.optimal_aperture()

Generic microscope（generic.txt）格式：
    名称  电压(kV)  Cs(Å)  光束发散(mrad)  离焦展宽(Å)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .constants import electron_wavelength, gamma_factor


@dataclass
class Microscope:
    """TEM 显微镜光学参数。

    Attributes
    ----------
    voltage_kv : float
        加速电压，kV。
    cs : float
        球差系数 Cs，Å（SimulaTEM 中以 mm 输入，1 mm = 1e7 Å）。
        可为负值（球差校正条件，如 Cs = -8000 Å = -800 nm）。
    defocus : float
        离焦量，Å。正 = 过焦，负 = 欠焦（SimulaTEM 约定）。
    astigmatism : float
        二重像散幅度，Å（SimulaTEM 的 Astigmatism amplitude）。
    astigmatism_azimuth : float
        像散方位角，度。
    focal_spread : float
        离焦展宽（色差/能量展宽等效），Å（SimulaTEM 的 Defocus spread）。
    angular_spread : float
        光束发散角（半角），mrad（SimulaTEM 的 Beam spread）。
    aperture_outer : float
        物镜光阑外半径（半角），mrad；0 或 None 表示不施加光阑。
    aperture_inner : float
        环形光阑内半径（半角），mrad；0 表示普通圆孔光阑
        （对应 SimulaTEM 的 annular aperture 功能）。
    """

    voltage_kv: float = 400.0
    cs: float = 1.0e7  # SimulaTEM 默认 Cs = 1 mm
    defocus: float = 0.0
    astigmatism: float = 0.0
    astigmatism_azimuth: float = 0.0
    focal_spread: float = 38.0  # SimulaTEM 默认 38 Å
    angular_spread: float = 0.0
    aperture_outer: Optional[float] = None
    aperture_inner: float = 0.0

    # ------------------------------------------------------------------
    @property
    def wavelength(self) -> float:
        """相对论电子波长，Å。"""
        return electron_wavelength(self.voltage_kv)

    @property
    def gamma(self) -> float:
        """相对论因子 γ。"""
        return gamma_factor(self.voltage_kv * 1e3)

    @property
    def gamma_lambda(self) -> float:
        """γ·λ（投影势 → 相位转换系数），Å。"""
        return self.gamma * self.wavelength

    def scherzer_defocus(self) -> float:
        """Scherzer 离焦（abTEM 相同的约定）。

        Δf_Sch = -sign(Cs) · sqrt(3/2 · |Cs| · λ)
        """
        lam = self.wavelength
        return -np.sign(self.cs) * np.sqrt(1.5 * abs(self.cs) * lam)

    def point_resolution(self) -> float:
        """Scherzer 点分辨率，Å（Williams & Carter 经典公式）。

        d = 0.66 · |Cs|^¼ · λ^¾
        """
        lam = self.wavelength
        return 0.66 * abs(self.cs) ** 0.25 * lam**0.75

    def optimal_aperture(self) -> float:
        """最佳孔径半角（mrad）：CTF 的第一个零点。

        对应 SimulaTEM 的 "optimal aperture"（接收至 CTF 第一个零点的全部空间频率）。
        CTF ∝ -sin χ(k)，第一零点即 sin χ 离开原点后首次过零处
        （Scherzer 条件下即 crossover 频率）。
        """
        lam = self.wavelength

        def chi(k: np.ndarray) -> np.ndarray:
            return np.pi * self.defocus * lam * k**2 + 0.5 * np.pi * self.cs * lam**3 * k**4

        # 搜索上界：点分辨率频率的 4 倍
        k_max = 4.0 / (abs(self.cs) * lam**3) ** 0.25 if self.cs != 0 else 4.0
        ks = np.linspace(1e-4, k_max, 20000)
        s = np.sin(chi(ks))
        # 跳过原点附近 sinχ≈0 的区间
        i0 = int(np.argmax(np.abs(s) > 1e-6))
        crossings = np.where(s[i0 + 1 :] * s[i0:-1] < 0)[0]
        if len(crossings) == 0:
            return float(k_max * lam * 1e3)
        i = i0 + 1 + crossings[0]
        # 线性插值求零点
        k0 = ks[i - 1] - s[i - 1] * (ks[i] - ks[i - 1]) / (s[i] - s[i - 1])
        return float(k0 * lam * 1e3)  # 散射角 rad → mrad

    def summary(self) -> str:
        """打印参数摘要（对应 SimulaTEM 的 log file 内容）。"""
        lines = [
            "=== 显微镜参数（SimulaTEM 约定） ===",
            f"加速电压      : {self.voltage_kv:.1f} kV",
            f"电子波长      : {self.wavelength:.5f} Å",
            f"球差 Cs       : {self.cs:.1f} Å ({self.cs / 1e7:.4f} mm)",
            f"离焦          : {self.defocus:+.1f} Å（Scherzer = {self.scherzer_defocus():+.1f} Å）",
            f"像散          : {self.astigmatism:.1f} Å @ {self.astigmatism_azimuth:.1f}°",
            f"离焦展宽      : {self.focal_spread:.1f} Å",
            f"光束发散      : {self.angular_spread:.3f} mrad",
            f"光阑          : 内 {self.aperture_inner:.2f} / 外 "
            f"{self.aperture_outer if self.aperture_outer else '∞'} mrad",
        ]
        return "\n".join(lines)


def load_generic_presets(path: str | Path) -> List[dict]:
    """读取 SimulaTEM 的 generic.txt（generic microscope 预设）。

    每行：名称 电压(kV) Cs(Å) 光束发散(mrad) 离焦展宽(Å)。

    Returns
    -------
    list of dict
        每个预设可直接 `Microscope(**preset)` 构造。
    """
    presets: List[dict] = []
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            name, v_kv, cs_a, beam_mrad, df_spread = parts
            presets.append(
                dict(
                    name=name,
                    voltage_kv=float(v_kv),
                    cs=float(cs_a),
                    angular_spread=float(beam_mrad),
                    focal_spread=float(df_spread),
                )
            )
    return presets
