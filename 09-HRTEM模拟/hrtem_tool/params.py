"""模拟参数模型：GUI 表单 ↔ 引擎 Microscope/超胞参数 之间的转换层。

单位约定（界面统一使用对电镜工作者直观的单位）：
  - 厚度 / 离焦：nm（引擎内部 Å）
  - 球差 Cs：mm（引擎内部 Å，1 mm = 1e7 Å，可为负 = 校正器过校）
  - 光阑 / 发散：mrad；电压：kV；采样：Å/px
  - 离焦符号沿用 SimulaTEM：正 = 过焦，负 = 欠焦
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional, Tuple

from tem_sim import Microscope


@dataclass
class SimParams:
    """一次 HRTEM 模拟的全部可调参数。"""

    # ---- 结构 ----
    cif_path: str = ""
    zone: Tuple[int, int, int] = (1, 1, 0)
    thickness_nm: float = 5.0          # 目标厚度（实际取 ≥ 该值的最小整数个 lz 周期）
    target_xy_a: float = 55.0          # 横向视场目标尺寸 Å

    # ---- 电镜 ----
    voltage_kv: float = 300.0
    cs_mm: float = 1.0
    defocus_nm: float = 0.0
    aperture_mrad: float = 20.0        # 0 = 不加光阑
    focal_spread_a: float = 30.0
    angular_spread_mrad: float = 0.3
    astigmatism_a: float = 0.0
    astigmatism_azimuth_deg: float = 0.0

    # ---- 多层法 ----
    sampling_a: float = 0.07           # 目标采样 Å/px（gpts_mode=auto 时生效）
    slice_thickness_a: float = 2.0
    gpts_mode: str = "auto"            # auto / 256 / 512 / 1024
    table: str = "peng"                # 散射因子表："peng" 标准 / "gauss3" legacy

    # ---- 取向与输出 ----
    rotation_deg: float = 0.0          # 面内旋转（绕光轴，逆时针为正）
    mirror: bool = False               # 水平镜像
    output_sampling_a: float = 0.0     # 0 = 保持原生采样；>0 = 重采样到该 Å/px
    polarity: int = 1                  # 1 = 正相（白底原子亮斑），-1 = 反转
    display_blur_a: float = 0.0        # 显示/导出用高斯模糊（模拟探测器 MTF，Å）
    display_lo_pct: float = 0.5        # 显示与导出 PNG 的对比度百分位下限
    display_hi_pct: float = 99.5       # 显示与导出 PNG 的对比度百分位上限

    # ------------------------------------------------------------------
    def replace(self, **kw) -> "SimParams":
        data = asdict(self)
        for key, value in kw.items():
            if key not in data:
                raise KeyError(key)
            data[key] = value
        return SimParams(**data)

    def validate(self) -> None:
        """参数范围校验（GUI 表单与 JSON/预设加载共用同一套口径）。"""
        if not all(float(v) == float(v) and abs(float(v)) != float("inf") for v in self.zone):
            raise ValueError("带轴指数必须为有限整数")
        if not any(int(v) for v in self.zone):
            raise ValueError("带轴不能为 [0 0 0]")

        def _pos(name: str, value: float, minimum: float) -> float:
            v = float(value)
            if v != v or v in (float("inf"), float("-inf")):
                raise ValueError(f"{name} 必须为有限数值")
            if v < minimum:
                raise ValueError(f"{name} 需 ≥ {minimum:g}")
            return v

        _pos("厚度", self.thickness_nm, 0.01)
        _pos("横向视场", self.target_xy_a, 1.0)
        _pos("加速电压", self.voltage_kv, 1.0)
        _pos("采样", self.sampling_a, 0.01)
        _pos("切片厚度", self.slice_thickness_a, 0.2)
        _pos("输出采样", self.output_sampling_a, 0.0)
        if 0 < self.output_sampling_a < 0.02:
            raise ValueError("输出采样 < 0.02 Å/px，渲染尺寸将超出内存合理范围")
        if float(self.aperture_mrad) < 0:
            raise ValueError("物镜光阑不能为负")
        if float(self.focal_spread_a) < 0:
            raise ValueError("离焦展宽不能为负")
        if float(self.angular_spread_mrad) < 0:
            raise ValueError("束发散不能为负")
        if float(self.display_blur_a) < 0:
            raise ValueError("探测器模糊不能为负")
        if self.gpts_mode not in ("auto", "256", "512", "1024"):
            raise ValueError(f"gpts_mode 非法：{self.gpts_mode!r}（可选 auto/256/512/1024）")
        if self.table not in ("peng", "gauss3"):
            raise ValueError(f"散射因子表非法：{self.table!r}（可选 peng/gauss3）")
        if not (0.0 <= self.display_lo_pct < self.display_hi_pct <= 100.0):
            raise ValueError(
                f"对比度百分位需满足 0 ≤ 下限 < 上限 ≤ 100"
                f"（当前 {self.display_lo_pct:g} / {self.display_hi_pct:g}）"
            )

    @property
    def zone_str(self) -> str:
        return "[{} {} {}]".format(*self.zone)

    def scope(self) -> Microscope:
        """转换成引擎 Microscope（厚度/离焦 nm→Å，Cs mm→Å）。"""
        return Microscope(
            voltage_kv=self.voltage_kv,
            cs=self.cs_mm * 1.0e7,
            defocus=self.defocus_nm * 10.0,
            astigmatism=self.astigmatism_a,
            astigmatism_azimuth=self.astigmatism_azimuth_deg,
            focal_spread=self.focal_spread_a,
            angular_spread=self.angular_spread_mrad,
            aperture_outer=(self.aperture_mrad if self.aperture_mrad > 0 else None),
        )

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SimParams":
        valid = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in valid}
        params = cls(**kwargs)
        try:
            params.zone = tuple(int(v) for v in params.zone)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"带轴指数必须为 3 个整数：{data.get('zone')!r}") from exc
        params.validate()
        return params

    def save_json(self, path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path) -> "SimParams":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def describe(self, thickness_actual_nm: Optional[float] = None) -> str:
        """参数摘要（图片角标 / TIFF 描述通用）。"""
        t = f"{thickness_actual_nm:.3f}" if thickness_actual_nm else f"{self.thickness_nm:g}"
        lines = [
            f"V = {self.voltage_kv:g} kV   Cs = {self.cs_mm:g} mm   "
            f"df = {self.defocus_nm:+g} nm",
            f"t = {t} nm   aperture = {self.aperture_mrad:g} mrad",
            f"spread = {self.focal_spread_a:g} Å   conv = {self.angular_spread_mrad:g} mrad",
        ]
        if self.astigmatism_a:
            lines.append(f"astig = {self.astigmatism_a:g} Å @ {self.astigmatism_azimuth_deg:g}°")
        if self.table == "gauss3":
            lines.append("table = gauss3 (legacy 口径)")
        return "\n".join(lines)
