"""参数预设（preset）加载。

预设 JSON 提供 AnalysisConfig 的基线参数；CLI/GUI 的显式参数再在其上覆盖。
预设中的未知键会被拒绝，防止拼写错误静默失效； ``distance_bins_nm`` 的
``[标签, 下限, 上限|null]`` 列表会转换为 :class:`DistanceBin` 元组。

另提供 :func:`config_from_saved`，把工具自身导出的 ``analysis_config.json``
原样重建为 :class:`AnalysisConfig`，实现一键复跑。
"""

from __future__ import annotations

import json
from dataclasses import fields
from importlib.resources import files
from pathlib import Path
from typing import Any

from .models import AnalysisConfig, DistanceBin, ReferenceSpec

# 允许出现在预设 JSON 中的简单字段（键名与 AnalysisConfig 字段一一对应）。
_SIMPLE_FIELDS: tuple[str, ...] = (
    "fit_min_ev",
    "fit_max_ev",
    "background_min_ev",
    "background_max_ev",
    "smoothing_ev",
    "savgol_polyorder",
    "deconvolution_regularization",
    "bootstrap_resamples",
    "bootstrap_block_columns",
    "injection_residual_block_channels",
    "along_surface_segment_nm",
    "qc_min_r2",
    "qc_min_snr",
    "bic_strong_support",
)

# 以列表形式书写、加载后转为元组的字段。
_TUPLE_FIELDS: tuple[str, ...] = (
    "zlp_window_ev",
    "low_loss_baseline_ev",
    "sensitivity_regularizations",
    "sensitivity_background_mins_ev",
    "sensitivity_boundary_offsets_pixels",
    "injection_fractions",
)

# 成对区间键 → (下限字段, 上限字段)，兼容 [低, 高] 的写法。
_RANGE_FIELDS: dict[str, tuple[str, str]] = {
    "fit_range_ev": ("fit_min_ev", "fit_max_ev"),
    "background_range_ev": ("background_min_ev", "background_max_ev"),
}

# 仅作为文档/溯源信息保留、不影响配置的键。
_INFORMATIONAL_KEYS: tuple[str, ...] = (
    "name",
    "version",
    "element",
    "edge",
    "required_reference_states",
    "reference_labels",
    "notes",
)


def default_preset_path() -> Path:
    """内置 Cu L2,3 预设的路径（兼容源码树与 PyInstaller 冻结环境）。"""

    resource = files(__package__).joinpath("presets", "cu_l23.json")
    path = Path(str(resource))
    if not path.is_file():
        raise FileNotFoundError(f"未找到内置预设文件: {path}")
    return path


def _parse_bin_bound(text: str, name: str, part: str) -> float:
    """解析距离分层边界数值；非数字时给出带上下文的错误。"""

    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{name}不是数字: “{part}”。") from exc


def parse_distance_bins_text(text: str) -> tuple[DistanceBin, ...]:
    """解析 CLI 的距离分层文本，如 ``"E1:0:2.2,Bulk:11"``。

    每层为 ``标签:下限[:上限]``；省略上限（或上限为空）表示无上限的内部层，
    必须位于最后（由 AnalysisConfig.validate 校验）。
    """

    bins: list[DistanceBin] = []
    for part in text.split(","):
        pieces = [piece.strip() for piece in part.strip().split(":")]
        if not pieces or not pieces[0]:
            raise ValueError(f"距离分层格式无效: “{part}”；应为 标签:下限[:上限]。")
        if len(pieces) == 2:
            label, low_text = pieces
            high: float | None = None
        elif len(pieces) == 3:
            label, low_text, high_text = pieces
            # 上限解析错误必须带上下文：直接 float() 会抛原生英文错误，
            # 把后面那段友好文案变成不可达代码。
            high = None if high_text == "" else _parse_bin_bound(high_text, "距离分层上限", part)
        else:
            raise ValueError(f"距离分层格式无效: “{part}”；应为 标签:下限[:上限]。")
        low = _parse_bin_bound(low_text, "距离分层下限", part)
        bins.append(DistanceBin(label, low, high))
    if not bins:
        raise ValueError("至少需要一个距离分层。")
    return tuple(bins)


def load_preset(path: Path | None = None) -> dict[str, Any]:
    """读取预设 JSON 并转换为可直接传给 AnalysisConfig 的关键字参数。"""

    if path is None:
        path = default_preset_path()
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"预设文件 {path} 不是有效 JSON：{exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"预设文件 {path} 的顶层必须是 JSON 对象。")  # noqa: TRY004 - 统一用 ValueError 供 GUI 展示
    known = (
        set(_SIMPLE_FIELDS)
        | set(_TUPLE_FIELDS)
        | set(_INFORMATIONAL_KEYS)
        | set(_RANGE_FIELDS)
        | {"distance_bins_nm"}
    )
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"预设文件包含未知键：{', '.join(unknown)}。")

    overrides: dict[str, Any] = {}
    for key in _SIMPLE_FIELDS:
        if key in raw:
            overrides[key] = raw[key]
    for key in _TUPLE_FIELDS:
        if key in raw:
            value = raw[key]
            if not isinstance(value, list):
                raise ValueError(f"预设键 {key} 必须是列表。")
            overrides[key] = tuple(value)
    for key, (low_field, high_field) in _RANGE_FIELDS.items():
        if key in raw:
            value = raw[key]
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"预设键 {key} 必须是 [低, 高] 二元列表。")
            overrides[low_field] = value[0]
            overrides[high_field] = value[1]
    if "distance_bins_nm" in raw:
        entries = raw["distance_bins_nm"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("预设键 distance_bins_nm 必须是非空列表。")
        bins: list[DistanceBin] = []
        for entry in entries:
            if not isinstance(entry, list) or len(entry) not in {2, 3}:
                raise ValueError(
                    f"distance_bins_nm 条目无效: {entry!r}；应为 [标签, 下限, 上限|null]。"
                )
            label = str(entry[0])
            low = float(entry[1])
            high = float(entry[2]) if len(entry) == 3 and entry[2] is not None else None
            bins.append(DistanceBin(label, low, high))
        overrides["distance_bins"] = tuple(bins)
    return overrides


# analysis_config.json 中以列表存储、重建时需转回元组的字段。
_SAVED_TUPLE_FIELDS = (
    "zlp_window_ev",
    "low_loss_baseline_ev",
    "sensitivity_regularizations",
    "sensitivity_background_mins_ev",
    "sensitivity_boundary_offsets_pixels",
    "injection_fractions",
)
_SAVED_EXCLUDED_FIELDS = ("input_path", "output_dir", "references")


def config_from_saved(path: Path) -> AnalysisConfig:
    """从本工具导出的 analysis_config.json 重建完整配置（一键复跑）。

    只做结构转换不做文件存在性校验（由随后的 AnalysisConfig.validate 完成），
    未知键直接拒绝，防止旧格式配置把拼写错误的参数静默丢弃。
    """

    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"复跑配置 {path} 不是有效 JSON：{exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"复跑配置 {path} 的顶层必须是 JSON 对象。")  # noqa: TRY004 - 统一用 ValueError 供 CLI/GUI 展示
    allowed = {field.name for field in fields(AnalysisConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"复跑配置包含未知键：{', '.join(unknown)}。")
    for required in ("input_path", "output_dir", "references"):
        if required not in raw:
            raise ValueError(f"复跑配置缺少必需字段 {required}。")

    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        if name == "references":
            references = []
            for item in value:
                try:
                    references.append(
                        ReferenceSpec(
                            label=str(item["label"]),
                            oxidation_state=int(item["oxidation_state"]),
                            path=Path(str(item["path"])),
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"复跑配置中的参考谱条目无效: {item!r}") from exc
            kwargs[name] = tuple(references)
        elif name == "distance_bins":
            bins = []
            for item in value:
                try:
                    bins.append(
                        DistanceBin(
                            label=str(item["label"]),
                            low_nm=float(item["low_nm"]),
                            high_nm=None if item["high_nm"] is None else float(item["high_nm"]),
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"复跑配置中的距离分层条目无效: {item!r}") from exc
            kwargs[name] = tuple(bins)
        elif name in _SAVED_TUPLE_FIELDS:
            if not isinstance(value, list):
                raise ValueError(f"复跑配置键 {name} 必须是列表。")
            kwargs[name] = tuple(value)
        elif name in ("input_path", "output_dir"):
            kwargs[name] = Path(str(value))
        else:
            kwargs[name] = value
    return AnalysisConfig(**kwargs)
