"""Validated, serialisable processing parameters."""

from __future__ import annotations

import numbers
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Literal


class ParameterError(ValueError):
    """Raised when a processing option is scientifically or technically invalid."""


class OutputEncoding(str, Enum):
    """Encoding policy for persisted TIFF data."""

    FLOAT32 = "float32"
    UINT8_DISPLAY = "uint8-display"
    SOURCE_DTYPE_CLIP = "source-dtype-clip"


PrimaryOutput = Literal["wiener", "absf", "butterworth"]
RotationMethod = Literal["fast_radial_bin", "dm_compatible"]


@dataclass(frozen=True)
class FilterParams:
    """Parameters shared by GUI, CLI, batch jobs, and the Python API.

    ``delta == 0`` has one explicit meaning: a pure Butterworth output.  This
    mirrors the v4 DM workflow and prevents a misleading no-op export.
    """

    step: int = 2
    delta: float = 5.0
    cycles: int = 99
    bw_order: int = 4
    bw_ro: float = 0.30
    apply_butterworth: bool = True
    primary_output: PrimaryOutput = "wiener"
    stem_filter: bool = False
    crosshair_width: int = 4
    crosshair_hole_radius: int = 4
    crosshair_bw_ro: float = 0.03
    low_freq_percent: float = 0.0
    rotation_method: RotationMethod = "fast_radial_bin"

    def validated(self) -> FilterParams:
        # 整数字段必须真的是整数（numpy 整数也算）；2.5 或 True 这类值会静默
        # 改变阈值循环语义，宁可显式拒绝也不悄悄截断。
        for field in ("step", "cycles", "bw_order", "crosshair_width", "crosshair_hole_radius"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise ParameterError(f"{field} 必须为整数，当前为 {value!r}")
        if not 1 <= self.step <= 10:
            raise ParameterError("Step 必须在 1–10 之间")
        if not 0.0 <= self.delta <= 50.0:
            raise ParameterError("Delta 必须在 0–50% 之间")
        if not 1 <= self.cycles <= 99:
            raise ParameterError("Cycles 必须在 1–99 之间")
        if not 1 <= self.bw_order <= 20:
            raise ParameterError("Butterworth 阶数必须在 1–20 之间")
        if not 0.01 <= self.bw_ro <= 3.0:
            raise ParameterError("BW Ro 必须在 0.01–3.0 之间")
        if self.primary_output not in {"wiener", "absf", "butterworth"}:
            raise ParameterError("主输出只能是 Wiener、ABSF 或 Butterworth")
        if self.rotation_method not in {"fast_radial_bin", "dm_compatible"}:
            raise ParameterError("未知的旋转平均方法")
        if not 0.0 <= self.low_freq_percent <= 20.0:
            raise ParameterError("低频回补必须在 0–20% 之间")
        if not 2 <= self.crosshair_width <= 20 or self.crosshair_width % 2:
            raise ParameterError("STEM 十字线宽必须为 2–20 的偶数")
        if not 2 <= self.crosshair_hole_radius <= 40:
            raise ParameterError("STEM 中心孔半径必须在 2–40 px 之间")
        if not 0.01 <= self.crosshair_bw_ro <= 0.30:
            raise ParameterError("XH Ro 必须在 0.01–0.30 之间")
        # DM 参考实现的约束：孔半径至少覆盖线宽的一半，否则十字线会侵入
        # 中心束（低频信息），中心孔就失去了保护意义。
        if self.crosshair_hole_radius < self.crosshair_width // 2:
            raise ParameterError(
                f"STEM 中心孔半径（{self.crosshair_hole_radius} px）小于十字线宽的一半"
                f"（{self.crosshair_width}//2 = {self.crosshair_width // 2} px），"
                "十字线会侵入中心束；请增大孔半径或减小线宽"
            )

        # The DM v4 UI forces this deterministic configuration.  Returning a
        # normalised immutable instance keeps every entry point consistent.
        if self.delta == 0:
            return replace(
                self,
                primary_output="butterworth",
                apply_butterworth=True,
                stem_filter=False,
            )
        if self.primary_output == "butterworth" and not self.apply_butterworth:
            raise ParameterError("选择 Butterworth 输出时必须启用 Butterworth 滤波")
        return self

    @property
    def primary_key(self) -> str:
        # 统一前缀规则：stem_filter 生效时所有输出（含 butterworth）都带
        # stem_ 前缀；delta == 0 经 validated() 强制剥离 stem，回到纯 BW 键。
        params = self.validated()
        prefix = "stem_" if params.stem_filter else ""
        return f"{prefix}{params.primary_output}_filtered"

    def as_dict(self) -> dict[str, object]:
        return asdict(self.validated())


@dataclass(frozen=True)
class SaveOptions:
    """Explicit output encoding instead of implicit per-frame normalisation."""

    encoding: OutputEncoding = OutputEncoding.FLOAT32
    display_range: tuple[float, float] | None = None

    def validated(self) -> SaveOptions:
        if self.display_range is not None:
            if self.encoding != OutputEncoding.UINT8_DISPLAY:
                raise ParameterError("display_range 仅适用于 uint8-display 编码")
            low, high = self.display_range
            if not high > low:
                raise ParameterError("显示导出范围必须满足 high > low")
        # UINT8_DISPLAY without an explicit range is valid: the pipeline
        # performs a deterministic stack-wide pre-pass to derive it.
        return self
