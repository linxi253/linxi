"""数据模型与配置。

本模块刻意不依赖 Tkinter、ncempy 或 matplotlib，便于单元测试和批处理调用。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

Orientation = Literal["auto", "top", "bottom", "left", "right"]
ProgressCallback = Callable[[str, float | None], None]
CancelCallback = Callable[[], bool]

# bootstrap 重采样次数上限：防止误输入天文数字导致近乎无限的计算。
MAX_BOOTSTRAP_RESAMPLES = 100_000


class AnalysisCancelled(RuntimeError):
    """用户从 GUI 请求协作式取消分析。"""


@dataclass(frozen=True)
class DistanceBin:
    """距样品物理表面的一个 inward-distance 区间。"""

    label: str
    low_nm: float
    high_nm: float | None

    def validate(self) -> None:
        if self.low_nm < 0:
            raise ValueError(f"{self.label}: 距离下限不能小于零。")
        if self.high_nm is not None and self.high_nm <= self.low_nm:
            raise ValueError(f"{self.label}: 距离上限必须大于下限。")


@dataclass(frozen=True)
class ReferenceSpec:
    """一个已知价态参考谱。"""

    label: str
    oxidation_state: int
    path: Path

    def validate(self) -> None:
        if not self.label.strip():
            raise ValueError("参考谱标签不能为空。")
        if not self.path.is_file():
            raise FileNotFoundError(f"未找到参考谱: {self.path}")


def default_distance_bins() -> tuple[DistanceBin, ...]:
    """默认的 0–2.2–4.4–6.6–11 nm 表层到内部剖面。"""

    return (
        DistanceBin("E1", 0.0, 2.2),
        DistanceBin("E2", 2.2, 4.4),
        DistanceBin("E3", 4.4, 6.6),
        DistanceBin("E4", 6.6, 11.0),
        DistanceBin("Bulk", 11.0, None),
    )


def _check_ordered_pair(values: tuple[float, float], name: str) -> None:
    """校验 (低, 高) 二元组形状与顺序；兼容 JSON 往返产生的 list。"""

    if len(values) != 2:
        raise ValueError(f"{name} 必须是 (低, 高) 二元组。")
    if not float(values[0]) < float(values[1]):
        raise ValueError(f"{name} 的下界必须小于上界（当前 {values[0]:g}, {values[1]:g}）。")


@dataclass
class AnalysisConfig:
    """一次分析的完整、可序列化参数集。

    v1 首先提供 Cu L2,3 的预设，但元素、拟合区间、背景区间、参考谱和距离分层
    全部由配置驱动，不与具体某个 DM4 文件绑定。
    """

    input_path: Path
    output_dir: Path
    references: tuple[ReferenceSpec, ...]

    # v1 仅实现 Cu L2,3；这两个字段保留为运行清单元数据，管线不读取。
    element: str = "Cu"
    edge: str = "L2,3"
    survey_dataset: int | None = None
    low_loss_dataset: int | None = None
    high_loss_dataset: int | None = None

    surface_orientation: Orientation = "auto"
    surface_search_depth_pixels: int = 12
    surface_smoothness_penalty: float = 0.005

    fit_min_ev: float = 925.0
    fit_max_ev: float = 970.0
    background_min_ev: float = 850.0
    background_max_ev: float = 925.0
    smoothing_ev: float = 1.5
    savgol_polyorder: int = 3

    deconvolution_regularization: float = 0.003
    zlp_window_ev: tuple[float, float] = (-3.0, 3.0)
    low_loss_baseline_ev: tuple[float, float] = (-30.0, -10.0)

    distance_bins: tuple[DistanceBin, ...] = field(default_factory=default_distance_bins)
    bootstrap_resamples: int = 500
    bootstrap_block_columns: int = 5
    random_seed: int = 20260820

    run_sensitivity: bool = True
    sensitivity_regularizations: tuple[float, ...] = (0.001, 0.003, 0.01)
    sensitivity_background_mins_ev: tuple[float, ...] = (820.0, 850.0, 880.0)
    sensitivity_boundary_offsets_pixels: tuple[int, ...] = (-1, 0, 1)
    injection_simulations: int = 200
    injection_fractions: tuple[float, ...] = (
        0.05,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    )
    injection_residual_block_channels: int = 5
    along_surface_segment_nm: float = 11.0

    qc_min_r2: float = 0.50
    qc_min_snr: float = 3.0
    bic_strong_support: float = 10.0

    def validate(self) -> None:
        # Path("") 与 Path(".") 都指向当前目录；必须先拒绝空输出路径，
        # 否则结果会静默散落到启动目录。
        if str(self.input_path).strip() in {"", "."}:
            raise ValueError("输入 DM3/DM4 文件路径不能为空。")
        if str(self.output_dir).strip() in {"", "."}:
            raise ValueError("输出目录不能为空。")
        for name, value in (
            ("Survey", self.survey_dataset),
            ("低损", self.low_loss_dataset),
            ("高损", self.high_loss_dataset),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name}数据集编号不能为负数（当前 {value}）。")
        if not self.input_path.is_file():
            raise FileNotFoundError(f"未找到输入 DM3/DM4 文件: {self.input_path}")
        if self.input_path.suffix.lower() not in {".dm3", ".dm4"}:
            raise ValueError("v1 仅支持 DM3/DM4 输入。")
        if self.output_dir.suffix.lower() in {".dm3", ".dm4"}:
            raise ValueError("输出位置必须是文件夹，而不能是 DM3/DM4 原始文件。")
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise ValueError(f"输出位置已存在且不是文件夹: {self.output_dir}")
        if len(self.references) < 2:
            raise ValueError("至少需要两个参考谱；Cu L2,3 预设通常使用 Cu0/Cu1/Cu2 三个参考。")
        for reference in self.references:
            reference.validate()
        labels = [reference.label for reference in self.references]
        if len(labels) != len(set(labels)):
            raise ValueError("参考谱标签不能重复。")
        if self.fit_min_ev >= self.fit_max_ev:
            raise ValueError("拟合区间无效。")
        if not (self.background_min_ev < self.background_max_ev <= self.fit_min_ev):
            raise ValueError("背景区间应位于拟合区间之前。")
        if self.smoothing_ev <= 0:
            raise ValueError("平滑宽度必须为正。")
        if self.savgol_polyorder < 1:
            raise ValueError("Savitzky-Golay 多项式阶数至少为 1。")
        if self.surface_smoothness_penalty < 0:
            raise ValueError("表面追踪的平滑惩罚不能为负。")
        _check_ordered_pair(self.zlp_window_ev, "ZLP 窗口")
        _check_ordered_pair(self.low_loss_baseline_ev, "低损基线区间")
        if float(self.low_loss_baseline_ev[1]) > float(self.zlp_window_ev[0]):
            raise ValueError("低损基线区间必须整体低于 ZLP 窗口。")
        if self.deconvolution_regularization <= 0:
            raise ValueError("去卷积正则化参数必须为正。")
        if self.bootstrap_resamples < 50:
            raise ValueError("bootstrap 重采样次数至少为 50。")
        if self.bootstrap_resamples > MAX_BOOTSTRAP_RESAMPLES:
            raise ValueError(f"bootstrap 重采样次数不能超过 {MAX_BOOTSTRAP_RESAMPLES}。")
        if self.bootstrap_block_columns < 1:
            raise ValueError("bootstrap 区块宽度至少为 1。")
        if self.surface_search_depth_pixels < 2:
            raise ValueError("表面搜索深度至少为 2 个像素。")
        if self.surface_orientation not in {"auto", "top", "bottom", "left", "right"}:
            raise ValueError(f"未知样品表面方向: {self.surface_orientation}")
        if self.along_surface_segment_nm <= 0:
            raise ValueError("沿表面分段宽度必须为正。")
        if self.injection_simulations < 0:
            raise ValueError("注入恢复模拟次数不能为负。")
        if self.injection_simulations > 0 and not self.injection_fractions:
            raise ValueError("启用注入恢复时至少需要一个注入分数。")
        if any(not 0.0 < value < 1.0 for value in self.injection_fractions):
            raise ValueError("Cu1 注入分数必须全部位于 0 与 1 之间。")
        if tuple(sorted(set(self.injection_fractions))) != self.injection_fractions:
            raise ValueError("Cu1 注入分数必须严格递增且不能重复。")
        if self.injection_residual_block_channels < 1:
            raise ValueError("注入恢复的残差移动块宽度至少为 1 个通道。")
        if any(value <= 0 for value in self.sensitivity_regularizations):
            raise ValueError("敏感性分析的去卷积正则化参数必须均为正。")
        if any(
            value <= 0 or value >= self.background_max_ev
            for value in self.sensitivity_background_mins_ev
        ):
            raise ValueError(
                f"敏感性分析的背景下限必须为正且低于 background_max_ev={self.background_max_ev:g}。"
            )
        if not 0.0 <= self.qc_min_r2 <= 1.0:
            raise ValueError("质量控制 R² 阈值必须位于 0 与 1 之间。")
        if self.qc_min_snr < 0:
            raise ValueError("质量控制 SNR 阈值不能为负。")
        if self.bic_strong_support <= 0:
            raise ValueError("强模型支持的 ΔBIC 阈值必须为正。")
        if not self.distance_bins:
            raise ValueError("至少需要一个距离分层。")
        bin_labels = [item.label for item in self.distance_bins]
        if len(bin_labels) != len(set(bin_labels)):
            raise ValueError("距离分层标签不能重复。")
        previous_high: float | None = None
        for index, item in enumerate(self.distance_bins):
            item.validate()
            if item.high_nm is None and index != len(self.distance_bins) - 1:
                raise ValueError("无上限的内部距离层必须位于最后。")
            if index > 0 and previous_high is not None and item.low_nm < previous_high:
                raise ValueError("距离分层必须按由表面到内部的顺序排列。")
            previous_high = item.high_nm

    def to_dict(self) -> dict[str, Any]:
        """返回可以直接写入 JSON 的不可变运行清单。"""

        raw = asdict(self)
        return _jsonable(raw)


@dataclass(frozen=True)
class DatasetInfo:
    """DM3/DM4 内的一个可读取数据对象。"""

    index: int
    shape: tuple[int, ...]
    dtype: str
    energy_min_ev: float | None
    energy_max_ev: float | None
    spatial_shape: tuple[int, ...]
    role_hint: str
    name: str = ""


@dataclass(frozen=True)
class LoadedDataset:
    """ncempy 读取后的数据对象。"""

    index: int
    data: np.ndarray
    coords: tuple[np.ndarray, ...]
    units: tuple[str, ...]
    title: str

    @property
    def energy_ev(self) -> np.ndarray:
        if not self.coords:
            raise ValueError("该数据对象没有能量坐标。")
        return self.coords[0]


@dataclass(frozen=True)
class BoundaryResult:
    """边缘追踪结果；coordinates 按 (row, column) 给出。

    ``selection`` 仅在 auto 模式下填充：记录四个候选方向的得分与对比度
    质量，供输出审计自动方向选择是否可靠。
    """

    orientation: str
    boundary: np.ndarray
    score_mean: float
    coordinates: np.ndarray
    selection: tuple[dict[str, Any], ...] | None = None


@dataclass(frozen=True)
class FitResult:
    """一个候选谱模型的非负最小二乘拟合。"""

    name: str
    component_indices: tuple[int, ...]
    coefficients: np.ndarray
    fractions: np.ndarray
    fitted: np.ndarray
    residual: np.ndarray
    rss: float
    bic: float
    r2: float
    nrmse: float


@dataclass
class RegionResult:
    """一个距离层的处理、拟合和不确定度结果。"""

    label: str
    low_nm: float
    high_nm: float | None
    pixel_count: int
    along_count: int
    snr: float
    background_exponent: float
    background_log_r2: float
    fit: FitResult
    fits: dict[str, FitResult]
    bootstrap_fractions: np.ndarray
    bootstrap_delta_bic: np.ndarray
    processed_spectrum: np.ndarray
    raw_column_spectra: np.ndarray


@dataclass
class AnalysisArtifacts:
    """管线返回的数值结果与可视化所需数组。"""

    summary: dict[str, Any]
    region_results: dict[str, RegionResult]
    fit_energy_ev: np.ndarray
    references: np.ndarray
    reference_labels: tuple[str, ...]
    high_energy_ev: np.ndarray
    survey_registered: np.ndarray | None
    boundary: BoundaryResult
    distance_nm: np.ndarray
    thickness_t_over_lambda: np.ndarray
    deconvolved_cube: np.ndarray
    sensitivity_rows: list[dict[str, Any]]
    segment_rows: list[dict[str, Any]]
    injection_rows: list[dict[str, Any]]


def _jsonable(value: Any) -> Any:
    """递归转化 dataclass/numpy/path，供 JSON 和 CSV 导出使用。

    数组与 numpy 标量同样要走非有限值清洗：真实 SI 里坏像素产生的 NaN
    若漏进 summary，json.dumps 会写出 ``NaN`` 字面量——严格 JSON 解析器
    （jq / JS / pandas strict 模式）都会拒绝整个文件。
    """

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


jsonable = _jsonable
