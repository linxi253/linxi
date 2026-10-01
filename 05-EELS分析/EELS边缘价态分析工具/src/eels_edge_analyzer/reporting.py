"""结构化结果、图表和 Markdown 报告导出。"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .dm4io import HashWatch
from .fitting import MODEL_CU0, MODEL_CU0_CU1, MODEL_CU0_CU2, MODEL_FULL
from .models import AnalysisArtifacts, AnalysisConfig, jsonable

# 模型 → 线色。多个绘图函数共用，避免两处维护后不一致。
_MODEL_LINE_COLORS: dict[str, str] = {
    MODEL_CU0: "#7f7f7f",
    MODEL_CU0_CU1: "#ff7f0e",
    MODEL_CU0_CU2: "#17becf",
    MODEL_FULL: "#2ca02c",
}

# 输出目录中出现任一该名单文件即视为“包含既往分析结果”。
_KNOWN_OUTPUT_NAMES = frozenset(
    {
        "edge_analysis_summary.json",
        "analysis_config.json",
        "edge_profile.csv",
        "model_comparison.csv",
        "processing_sensitivity.csv",
        "edge_along_surface_segments.csv",
        "edge_detection_limit.csv",
        "surface_boundary.csv",
        "global_shift_scan.csv",
        "processed_arrays.npz",
        "edge_analysis_report.md",
        "input_integrity.json",
        "run_log.txt",
        "raw_data_overview.png",
        "registered_surface_boundary.png",
        "edge_component_profile.png",
        "edge_outer_model_fits.png",
        "edge_model_comparison.png",
        "edge_processing_sensitivity.png",
        "edge_along_surface_segments.png",
        "edge_detection_limit.png",
        "reference_spectra.png",
        "edge_region_model_fits.png",
    }
)


def output_directory_state(output_dir: Path) -> str:
    """分类输出目录状态：absent/empty/previous_results/other_content。

    供 CLI/GUI 在长时间计算前提示用户，export_artifacts 据此拒绝静默覆盖。
    """

    if not output_dir.exists():
        return "absent"
    if not output_dir.is_dir():
        return "other_content"
    entries = list(output_dir.iterdir())
    if not entries:
        return "empty"
    if {item.name for item in entries} & _KNOWN_OUTPUT_NAMES:
        return "previous_results"
    return "other_content"


def _ensure_writable(output: Path) -> None:
    """预检查目标位置的写权限，把 PermissionError 提前到写入任何文件之前。"""

    target = output if output.exists() else output.parent
    while not target.exists():  # 多级新建目录时沿祖先找到第一个已存在的目录
        target = target.parent
    if not os.access(target, os.W_OK):
        raise PermissionError(f"没有输出目录的写入权限: {target}")


def _csv_safe(value: Any) -> Any:
    """清洗可能被 Excel 当作公式的字符串单元格（CSV 注入加固）。

    仅处理字符串：数值型的负号不构成公式注入，保持原样。
    """

    if isinstance(value, str) and value[:1] in {"=", "+", "@", "\t", "\r"}:
        return "'" + value
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {key: _csv_safe(jsonable(value)) for key, value in row.items()}
                for row in rows
            ]
        )


def _save_figure(figure: Figure, path: Path) -> None:
    figure.savefig(path, dpi=170, bbox_inches="tight")


def _profile_rows(artifacts: AnalysisArtifacts) -> list[dict[str, Any]]:
    return list(artifacts.summary["profile"])


def _plot_overview(artifacts: AnalysisArtifacts, path: Path) -> None:
    figure = Figure(figsize=(14, 4.4), constrained_layout=True)
    axes = figure.subplots(1, 3)
    survey = artifacts.survey_registered
    if survey is None:
        survey = np.sum(np.maximum(artifacts.deconvolved_cube, 0), axis=0)
    panel = axes[0].imshow(survey, cmap="gray")
    axes[0].plot(
        artifacts.boundary.coordinates[:, 1], artifacts.boundary.coordinates[:, 0], color="cyan", lw=2
    )
    axes[0].set(title="Registered survey / surface", xlabel="SI x (pixel)", ylabel="SI y (pixel)")
    figure.colorbar(panel, ax=axes[0], shrink=0.82, label="intensity")

    panel = axes[1].imshow(artifacts.thickness_t_over_lambda, cmap="magma")
    axes[1].set(title="Relative thickness t/λ", xlabel="SI x (pixel)", ylabel="SI y (pixel)")
    figure.colorbar(panel, ax=axes[1], shrink=0.82)

    fit_mask = (artifacts.high_energy_ev >= artifacts.fit_energy_ev.min()) & (
        artifacts.high_energy_ev <= artifacts.fit_energy_ev.max()
    )
    integrated = np.trapezoid(
        artifacts.deconvolved_cube[fit_mask], artifacts.high_energy_ev[fit_mask], axis=0
    )
    panel = axes[2].imshow(integrated, cmap="viridis")
    axes[2].set(title="Deconvolved core-loss signal", xlabel="SI x (pixel)", ylabel="SI y (pixel)")
    figure.colorbar(panel, ax=axes[2], shrink=0.82)
    _save_figure(figure, path)


_CONTOUR_PALETTE = ("#ffd700", "#ff9500", "#00dd44", "white", "#ff66cc", "#00e5ff")


def _plot_boundary(artifacts: AnalysisArtifacts, config: AnalysisConfig, path: Path) -> None:
    figure = Figure(figsize=(8.5, 6.0), constrained_layout=True)
    axis = figure.subplots()
    image = artifacts.survey_registered
    if image is None:
        image = np.sum(np.maximum(artifacts.deconvolved_cube, 0), axis=0)
    panel = axis.imshow(image, cmap="gray")
    coordinates = artifacts.boundary.coordinates
    axis.plot(coordinates[:, 1], coordinates[:, 0], color="cyan", lw=2.4, label="surface")
    legend_handles = [Line2D([0], [0], color="cyan", lw=2.4, label="surface")]
    # 等值线层级与颜色由配置的距离分层推导，而不是硬编码的默认分层。
    for index, item in enumerate(config.distance_bins):
        if item.high_nm is None:
            continue
        color = _CONTOUR_PALETTE[index % len(_CONTOUR_PALETTE)]
        try:
            axis.contour(artifacts.distance_nm, levels=[item.high_nm], colors=[color], linewidths=1.3)
            legend_handles.append(Line2D([0], [0], color=color, lw=1.3, label=f"{item.high_nm:g} nm"))
        except ValueError:
            continue
    axis.set(
        title=f"Physical surface ({artifacts.boundary.orientation}) and inward distance",
        xlabel="SI x (pixel)",
        ylabel="SI y (pixel)",
    )
    axis.legend(handles=legend_handles, loc="upper right")
    figure.colorbar(panel, ax=axis, shrink=0.86, label="intensity")
    _save_figure(figure, path)


def _plot_profile(artifacts: AnalysisArtifacts, path: Path) -> None:
    rows = _profile_rows(artifacts)
    labels = artifacts.reference_labels
    colors = ("#1f77b4", "#ff7f0e", "#d62728")
    figure = Figure(figsize=(10.0, 5.5), constrained_layout=True)
    axis = figure.subplots()
    x = np.arange(len(rows))
    for index, (label, color) in enumerate(zip(labels, colors, strict=True)):
        median = np.array([row[f"{label}_bootstrap_median"] for row in rows])
        low = np.array([row[f"{label}_bootstrap_p2_5"] for row in rows])
        high = np.array([row[f"{label}_bootstrap_p97_5"] for row in rows])
        axis.errorbar(
            x,
            median,
            yerr=np.vstack((median - low, high - median)),
            marker="o",
            lw=2,
            capsize=4,
            color=color,
            label=label,
        )
    axis.set(
        title="Surface-to-interior oxidation-state spectral profile",
        xlabel="Inward distance bin",
        ylabel="MLLS spectral fraction (95% block-bootstrap CI)",
        xticks=x,
        xticklabels=[row["bin"] for row in rows],
        ylim=(-0.02, 1.05),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    _save_figure(figure, path)


def _plot_outer_fit(artifacts: AnalysisArtifacts, path: Path) -> None:
    edge_label = artifacts.summary["configuration"]["distance_bins"][0]["label"]
    result = artifacts.region_results[edge_label]
    figure = Figure(figsize=(11.5, 5.6), constrained_layout=True)
    axis = figure.subplots()
    axis.plot(
        artifacts.fit_energy_ev,
        result.processed_spectrum,
        color="black",
        lw=2.1,
        label=f"{edge_label} experiment",
    )
    for name, fit in result.fits.items():
        axis.plot(
            artifacts.fit_energy_ev,
            fit.fitted,
            lw=1.8,
            color=_MODEL_LINE_COLORS.get(name, "#9467bd"),
            label=f"{name} (BIC={fit.bic:.1f})",
        )
    axis.set(
        title=f"Outermost {edge_label} model comparison",
        xlabel="Energy loss (eV)",
        ylabel="background-subtracted intensity (a.u.)",
    )
    axis.legend(ncol=2, fontsize=8.5)
    axis.grid(alpha=0.2)
    _save_figure(figure, path)


def _plot_references(artifacts: AnalysisArtifacts, path: Path) -> None:
    """三条单位正面积参考谱同框，供用户核对参考质量与差异。"""

    figure = Figure(figsize=(9.5, 5.0), constrained_layout=True)
    axis = figure.subplots()
    for index, label in enumerate(artifacts.reference_labels):
        axis.plot(
            artifacts.fit_energy_ev,
            artifacts.references[:, index],
            lw=1.9,
            label=label,
        )
    axis.set(
        title="Unit-area reference spectra on the fitting grid",
        xlabel="Energy loss (eV)",
        ylabel="normalized intensity (1/eV)",
    )
    axis.legend()
    axis.grid(alpha=0.25)
    _save_figure(figure, path)


def _plot_region_fits(artifacts: AnalysisArtifacts, path: Path) -> None:
    """每个距离层的实验谱与完整模型拟合对照，便于逐层目检。"""

    rows = _profile_rows(artifacts)
    n_cols = 2 if len(rows) > 3 else 1
    n_rows = math.ceil(len(rows) / n_cols)
    figure = Figure(figsize=(6.4 * n_cols, 3.0 * n_rows), constrained_layout=True)
    axes = figure.subplots(n_rows, n_cols, squeeze=False)
    for slot, row in enumerate(rows):
        axis = axes[slot // n_cols][slot % n_cols]
        result = artifacts.region_results[row["bin"]]
        axis.plot(
            artifacts.fit_energy_ev,
            result.processed_spectrum,
            color="black",
            lw=1.8,
            label="experiment",
        )
        axis.plot(
            artifacts.fit_energy_ev,
            result.fit.fitted,
            color=_MODEL_LINE_COLORS[MODEL_FULL],
            lw=1.6,
            ls="--",
            label=MODEL_FULL,
        )
        axis.set(
            title=f"{row['bin']}  R²={result.fit.r2:.3f}  SNR={result.snr:.1f}",
            xlabel="Energy loss (eV)",
            ylabel="intensity (a.u.)",
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    for slot in range(len(rows), n_rows * n_cols):
        axes[slot // n_cols][slot % n_cols].set_axis_off()
    _save_figure(figure, path)


def _plot_model_comparison(artifacts: AnalysisArtifacts, path: Path) -> None:
    rows = artifacts.summary["model_comparison"]
    bins = [item["bin"] for item in _profile_rows(artifacts)]
    figure = Figure(figsize=(10.0, 5.3), constrained_layout=True)
    axis = figure.subplots()
    x = np.arange(len(bins))
    for model in (MODEL_CU0, MODEL_CU0_CU1, MODEL_CU0_CU2, MODEL_FULL):
        values = []
        for bin_label in bins:
            match = next(item for item in rows if item["bin"] == bin_label and item["model"] == model)
            values.append(match["bic"])
        axis.plot(x, values, marker="o", lw=2, label=model, color=_MODEL_LINE_COLORS[model])
    axis.set(
        title="Competing spectral-model BIC (lower is preferred)",
        xlabel="Inward distance bin",
        ylabel="BIC",
        xticks=x,
        xticklabels=bins,
    )
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    _save_figure(figure, path)


def _plot_sensitivity(artifacts: AnalysisArtifacts, path: Path) -> None:
    rows = [row for row in artifacts.sensitivity_rows if "Cu1_fraction" in row]
    figure = Figure(figsize=(10.5, 5.3), constrained_layout=True)
    axis = figure.subplots()
    if not rows:
        axis.text(0.5, 0.5, "Sensitivity analysis was disabled.", ha="center", va="center")
        axis.set_axis_off()
    else:
        # 颜色区分去卷积正则化，marker 形状区分背景下限，两个维度同时可读。
        regularizations = sorted({float(row["regularization"]) for row in rows})
        backgrounds = sorted({float(row["background_min_ev"]) for row in rows})
        markers = ("o", "s", "^", "D", "v", "P", "X", "*")
        palette = [f"C{index % 10}" for index in range(len(regularizations))]
        for r_index, regularization in enumerate(regularizations):
            for b_index, background in enumerate(backgrounds):
                selected = [
                    row
                    for row in rows
                    if float(row["regularization"]) == regularization
                    and float(row["background_min_ev"]) == background
                ]
                if not selected:
                    continue
                axis.scatter(
                    [row["boundary_offset_pixels"] for row in selected],
                    [row["Cu1_fraction"] for row in selected],
                    s=54,
                    alpha=0.85,
                    color=palette[r_index],
                    marker=markers[b_index % len(markers)],
                )
        reg_handles = [
            Line2D([0], [0], color=palette[index], lw=2, label=f"reg={value:g}")
            for index, value in enumerate(regularizations)
        ]
        bg_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                lw=0,
                marker=markers[index % len(markers)],
                linestyle="None",
                label=f"bg_min={value:g}",
            )
            for index, value in enumerate(backgrounds)
        ]
        axis.axhline(0, color="gray", lw=1)
        axis.set(
            title="E1 Cu1 sensitivity to deconvolution/background/boundary",
            xlabel="Boundary offset into sample (pixel)",
            ylabel="Cu1 MLLS fraction",
        )
        axis.legend(handles=reg_handles + bg_handles, fontsize=8, ncol=2)
        axis.grid(alpha=0.25)
    _save_figure(figure, path)


def _plot_segments(artifacts: AnalysisArtifacts, path: Path) -> None:
    rows = artifacts.segment_rows
    figure = Figure(figsize=(10.5, 5.3), constrained_layout=True)
    axis = figure.subplots()
    if not rows:
        axis.text(0.5, 0.5, "Not enough along-surface spectra for segmentation.", ha="center", va="center")
        axis.set_axis_off()
    else:
        x = [row["center_nm"] for row in rows]
        for label, color in (("Cu0", "#1f77b4"), ("Cu1", "#ff7f0e"), ("Cu2", "#d62728")):
            axis.plot(
                x, [row[f"{label}_fraction"] for row in rows], marker="o", lw=1.7, label=label, color=color
            )
        for row in rows:
            if not row["qc_pass"]:
                axis.axvspan(row["start_nm"], row["end_nm"], color="gray", alpha=0.08)
        axis.set(
            title="Outermost-layer composition along the physical surface",
            xlabel="Along-surface position (nm)",
            ylabel="MLLS fraction",
            ylim=(-0.02, 1.05),
        )
        axis.grid(alpha=0.25)
        axis.legend()
    _save_figure(figure, path)


def _plot_detection(artifacts: AnalysisArtifacts, path: Path) -> None:
    rows = artifacts.injection_rows
    figure = Figure(figsize=(9.4, 5.0), constrained_layout=True)
    axis = figure.subplots()
    if not rows:
        axis.text(0.5, 0.5, "Injection/recovery was disabled.", ha="center", va="center")
        axis.set_axis_off()
    else:
        true = np.array([row["true_cu1_fraction"] for row in rows])
        med = np.array([row["recovered_cu1_median"] for row in rows])
        low = np.array([row["recovered_cu1_p2_5"] for row in rows])
        high = np.array([row["recovered_cu1_p97_5"] for row in rows])
        rate = np.array([row["cu1_selection_rate_delta_bic_ge_threshold"] for row in rows])
        axis.errorbar(
            true, med, yerr=np.vstack((med - low, high - med)), marker="o", capsize=4, label="recovered Cu1"
        )
        axis.plot(true, true, "--", color="gray", label="ideal recovery")
        axis.plot(true, rate, marker="s", color="#d62728", label="strong-model selection rate")
        axis.axhline(0.95, color="#d62728", ls=":", lw=1)
        axis.set(
            title="Block-residual Cu1 injection/recovery",
            xlabel="Injected Cu1 fraction",
            ylabel="fraction / selection rate",
            xlim=(0, max(0.55, true.max() + 0.03)),
            ylim=(-0.03, 1.05),
        )
        axis.grid(alpha=0.25)
        axis.legend()
    _save_figure(figure, path)


def _collect_warnings(summary: dict[str, Any], config: AnalysisConfig) -> list[str]:
    """汇总应同时进入报告与 run log 的告警。"""

    warnings: list[str] = []
    thickness_quality = summary.get("thickness_quality") or {}
    if thickness_quality.get("exceeds_threshold"):
        warnings.append(
            f"样品区中位相对厚度 t/λ = "
            f"{float(thickness_quality['median_t_over_lambda_sample']):.2f}，"
            f"超过 {float(thickness_quality['warning_threshold']):g}；"
            "Fourier-ratio 复散射校正可靠性受限，价态结论应谨慎。"
        )
    registration = summary.get("surface_registration") or {}
    if registration.get("auto_orientation_confident") is False:
        warnings.append(
            "auto 表面方向未检出可信的真空-样品明暗对比（常见于 BF 像或表面不在视场内）；"
            "请核对 registered_surface_boundary.png，必要时手动指定方向重跑。"
        )
    block_width = summary.get("along_surface_block_width") or {}
    below = block_width.get("configured_below_suggestion") or []
    if below:
        warnings.append(
            f"bootstrap 块宽（{block_width.get('configured_columns')} 列）小于按自相关建议的块宽，"
            f"相关区域：{', '.join(below)}；不确定度可能被低估。"
        )
    for row in summary.get("profile", []):
        if row.get("r2", 1.0) < config.qc_min_r2 or row.get("snr", 0.0) < config.qc_min_snr:
            warnings.append(
                f"{row['bin']} 未通过质量控制（R²={row.get('r2', float('nan')):.3f}，"
                f"SNR={row.get('snr', float('nan')):.1f}）。"
            )
    return warnings


def _write_report(artifacts: AnalysisArtifacts, config: AnalysisConfig, path: Path) -> None:
    summary = artifacts.summary
    profile = _profile_rows(artifacts)
    edge = profile[0]
    labels = artifacts.reference_labels
    decision = summary["decision"]
    edge_label = edge["bin"]
    delta = edge["delta_bic_m02_minus_m012"]
    detection_limit = summary["detection_limit_cu1_fraction"]
    warnings = _collect_warnings(summary, config)
    if config.injection_simulations <= 0:
        detection_line = "- Cu1 注入恢复检验未启用。"
    elif detection_limit is None:
        maximum_tested = max(
            (row["true_cu1_fraction"] for row in artifacts.injection_rows),
            default=0.0,
        )
        detection_line = (
            f"- Cu1 注入至 {100 * maximum_tested:.0f}% 时，强模型选择率仍未达到 95%；"
            "本次分析不能给出有限检出限。"
        )
    else:
        detection_line = (
            f"- 按单位正面积和移动块残差注入估计，Cu1 的经验强模型检出限约为 {100 * detection_limit:.0f}%。"
        )
    lines = [
        "# EELS 边缘价态分析报告",
        "",
        "## 结论",
        "",
        f"- 证据等级：{decision['evidence_level']}。",
        f"- 是否支持 Cu0/Cu1 混合谱学状态：{decision['supports_Cu0_Cu1_mixed_spectral_state']}。",
        (
            f"- 最外层 {edge_label} 的 Cu1 拟合权重为 "
            f"{100 * edge[f'{labels[1]}_fraction']:.1f}%，移动块 bootstrap 95% 区间为 "
            f"{100 * edge[f'{labels[1]}_bootstrap_p2_5']:.1f}%–"
            f"{100 * edge[f'{labels[1]}_bootstrap_p97_5']:.1f}%。"
        ),
        (
            f"- 在允许 Cu2 后加入 Cu1 的 ΔBIC (M02−M012) 为 {delta:.2f}；"
            f"正值且大于 {config.bic_strong_support:g} 才代表强模型支持。"
        ),
        detection_line,
        *(f"⚠ {item}" for item in warnings),
        "",
        "## 表层到内部的半定量谱权重",
        "",
        "| 区域 | 距表面 (nm) | " + " | ".join(labels) + " | R² | SNR |",
        "|---|---:|" + "---:|" * len(labels) + "---:|---:|",
    ]
    for row in profile:
        distance = (
            f"{row['distance_low_nm']:.2f}–{row['distance_high_nm']:.2f}"
            if row["distance_high_nm"] is not None
            else f"> {row['distance_low_nm']:.2f}"
        )
        fractions = " | ".join(f"{100 * row[f'{label}_fraction']:.1f}%" for label in labels)
        lines.append(f"| {row['bin']} | {distance} | {fractions} | {row['r2']:.3f} | {row['snr']:.1f} |")
    lines.extend(
        [
            "",
            "## 方法和质量控制",
            "",
            f"- 输入：{config.input_path.name}；原始文件 SHA-256：{summary['input']['sha256_before_and_after_compute']}。",
            f"- 表面方向：{summary['surface_registration']['orientation']}；来源：{summary['surface_registration']['source']}。",
            f"- 高损谱采用配对低损谱的正则化 Fourier-ratio 复散射校正，主正则化参数为 {config.deconvolution_regularization:g}。",
            (
                f"- 每个距离层保留沿表面的单列平均谱，并进行 {config.bootstrap_resamples} 次、"
                f"{config.bootstrap_block_columns} 列宽的移动块 bootstrap。"
            ),
            (
                "- 谱与参考均经 Savitzky-Golay 平滑；平滑使相邻能量通道相关，"
                "BIC 的独立噪声假设因此偏乐观，ΔBIC 应结合注入恢复的经验选择率解读。"
            ),
            (
                "- inward 距离按每列相对边界的垂直像素距离乘以空间步长计算；"
                "表面弯曲明显时，曲率较大处的分层距离会偏大。"
            ),
            (
                "- 相对厚度 t/λ 为诊断量：总强度积分在 "
                f"{float(summary['low_loss'].get('total_integral_max_ev', 0.0)):.0f} eV 或谱末端截断，"
                "高能损失未计入。"
            ),
            (
                "- 注入恢复先将 Cu0+Cu2 基线与参考谱统一到单位正面积，再以 "
                f"{config.injection_residual_block_channels} 通道移动块重采样基线残差；"
                "若边缘区已含真实 Cu1 信号，经验检出限会偏乐观。"
            ),
            f"- 内部全局能量平移：{summary['global_energy_shift_ev']:.2f} eV。",
            "",
            "## 解释边界",
            "",
            (
                "结果为投影谱权重，不能直接等同于原子百分比。即使出现多个价态的谱学权重，"
                "也不能仅凭 EELS 证明晶体学复合相；应结合 O-K/EDS、HRTEM、纳米衍射或 4D-STEM。"
            ),
            "",
            "## 输出文件",
            "",
            "- edge_analysis_summary.json：参数、哈希、质量控制和结论。",
            "- edge_profile.csv、model_comparison.csv：用于作图和二次统计的表格。",
            "- processed_arrays.npz：能量轴、参考矩阵、厚度图、距离图和去卷积立方体。",
            (
                "- PNG 图：边界、谱权重剖面、模型比较、敏感性、沿表面连续性和检出能力；"
                "另含 reference_spectra.png（参考谱对照）与 edge_region_model_fits.png（逐层拟合目检）。"
            ),
            "- run_log.txt：运行时间、依赖版本与告警清单。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_run_log(
    summary: dict[str, Any],
    config: AnalysisConfig,
    paths: dict[str, Path],
    warnings: list[str],
    path: Path,
) -> None:
    application = summary.get("application") or {}
    lines = [
        "EELS Edge Analyzer run log",
        f"timestamp_utc: {application.get('timestamp_utc', 'unknown')}",
        f"application: {application.get('name', 'EELS Edge Analyzer')} {application.get('version', '')}",
        f"environment: {json.dumps(summary.get('environment', {}), ensure_ascii=False)}",
        f"input: {config.input_path}",
        f"sha256: {summary.get('input', {}).get('sha256_before_and_after_compute', 'unknown')}",
        "",
        "warnings:",
        *(["- 无"] if not warnings else [f"- {item}" for item in warnings]),
        "",
        "outputs:",
        *(f"- {key}: {value.name}" for key, value in sorted(paths.items())),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_artifacts(
    artifacts: AnalysisArtifacts,
    config: AnalysisConfig,
    *,
    compress_arrays: bool = True,
    overwrite: bool = False,
) -> dict[str, Path]:
    """把管线结果写入独立输出目录，并在写前后检查原始哈希未变化。

    ``compress_arrays=False`` 时 processed_arrays.npz 不做 zlib 压缩，
    牺牲磁盘空间换取大 SI 下的写盘速度。``overwrite=False`` 时若输出目录
    已包含既往分析结果将拒绝静默覆盖（CLI 对应 --overwrite，GUI 会先弹出
    确认框）。
    """

    output = config.output_dir.resolve()
    state = output_directory_state(output)
    if state == "previous_results" and not overwrite:
        raise RuntimeError(
            f"输出目录 {output} 已包含既往分析结果，拒绝静默覆盖；"
            "请更换输出目录，或使用 --overwrite / overwrite=True 显式允许。"
        )
    _ensure_writable(output)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = config.input_path.resolve()
    expected_hash = artifacts.summary["input"]["sha256_before_and_after_compute"]
    hash_watch = HashWatch(raw_path)
    if hash_watch.current() != expected_hash:
        raise RuntimeError("导出前检测到原始数据哈希变化，拒绝写入结果。")

    profile_rows = _profile_rows(artifacts)
    model_rows = list(artifacts.summary["model_comparison"])
    boundary_rows = [
        {
            "index": int(index),
            "row_pixel": int(point[0]),
            "column_pixel": int(point[1]),
            "orientation": artifacts.boundary.orientation,
        }
        for index, point in enumerate(artifacts.boundary.coordinates)
    ]
    paths = {
        "summary": output / "edge_analysis_summary.json",
        "config": output / "analysis_config.json",
        "profile": output / "edge_profile.csv",
        "models": output / "model_comparison.csv",
        "sensitivity": output / "processing_sensitivity.csv",
        "segments": output / "edge_along_surface_segments.csv",
        "detection": output / "edge_detection_limit.csv",
        "boundary": output / "surface_boundary.csv",
        "shift_scan": output / "global_shift_scan.csv",
        "arrays": output / "processed_arrays.npz",
        "report": output / "edge_analysis_report.md",
    }
    write_json(paths["summary"], artifacts.summary)
    write_json(paths["config"], config.to_dict())
    write_csv(paths["profile"], profile_rows)
    write_csv(paths["models"], model_rows)
    write_csv(paths["sensitivity"], artifacts.sensitivity_rows)
    write_csv(paths["segments"], artifacts.segment_rows)
    write_csv(paths["detection"], artifacts.injection_rows)
    write_csv(paths["boundary"], boundary_rows)
    write_csv(paths["shift_scan"], list(artifacts.summary["global_shift_scan"]))
    arrays_payload = {
        "high_loss_energy_ev": artifacts.high_energy_ev,
        "fit_energy_ev": artifacts.fit_energy_ev,
        "reference_matrix": artifacts.references,
        "thickness_t_over_lambda": artifacts.thickness_t_over_lambda,
        "distance_nm": artifacts.distance_nm,
        "deconvolved_high_loss_cube": artifacts.deconvolved_cube,
        "surface_boundary_coordinates": artifacts.boundary.coordinates,
    }
    if compress_arrays:
        np.savez_compressed(paths["arrays"], **arrays_payload)
    else:
        np.savez(paths["arrays"], **arrays_payload)

    plot_paths = {
        "overview": output / "raw_data_overview.png",
        "boundary_plot": output / "registered_surface_boundary.png",
        "profile_plot": output / "edge_component_profile.png",
        "fit_plot": output / "edge_outer_model_fits.png",
        "model_plot": output / "edge_model_comparison.png",
        "sensitivity_plot": output / "edge_processing_sensitivity.png",
        "segments_plot": output / "edge_along_surface_segments.png",
        "detection_plot": output / "edge_detection_limit.png",
        "references_plot": output / "reference_spectra.png",
        "region_fits_plot": output / "edge_region_model_fits.png",
    }
    _plot_overview(artifacts, plot_paths["overview"])
    _plot_boundary(artifacts, config, plot_paths["boundary_plot"])
    _plot_profile(artifacts, plot_paths["profile_plot"])
    _plot_outer_fit(artifacts, plot_paths["fit_plot"])
    _plot_model_comparison(artifacts, plot_paths["model_plot"])
    _plot_sensitivity(artifacts, plot_paths["sensitivity_plot"])
    _plot_segments(artifacts, plot_paths["segments_plot"])
    _plot_detection(artifacts, plot_paths["detection_plot"])
    _plot_references(artifacts, plot_paths["references_plot"])
    _plot_region_fits(artifacts, plot_paths["region_fits_plot"])
    _write_report(artifacts, config, paths["report"])

    after_export = hash_watch.current()
    if after_export != expected_hash:
        raise RuntimeError("导出后检测到原始数据哈希变化，结果目录需要人工审查。")
    integrity = {
        "input_file": str(raw_path),
        "sha256_before_compute_and_after_export": expected_hash,
        "source_unchanged": True,
        "output_directory": str(output),
    }
    integrity_path = output / "input_integrity.json"
    write_json(integrity_path, integrity)
    run_log_path = output / "run_log.txt"
    _write_run_log(artifacts.summary, config, paths | plot_paths, _collect_warnings(artifacts.summary, config), run_log_path)
    return paths | plot_paths | {"integrity": integrity_path, "run_log": run_log_path}
