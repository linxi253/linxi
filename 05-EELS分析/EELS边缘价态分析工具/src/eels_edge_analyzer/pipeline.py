"""EELS 边缘价态分析的端到端可复现管线。"""

from __future__ import annotations

import math
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from typing import Any

import numpy as np
import scipy

from . import __version__
from .dm4io import (
    HashWatch,
    all_tags,
    infer_dataset_indices,
    inspect_dm4,
    read_dataset,
    register_survey_to_si,
    spatial_steps_nm,
)
from .fitting import (
    MODEL_CU0_CU2,
    MODEL_FULL,
    bootstrap_region,
    fit_all_cu_models,
    fit_to_row,
    injection_recovery,
    paired_bootstrap_difference,
    suggested_block_width,
)
from .models import (
    AnalysisArtifacts,
    AnalysisConfig,
    BoundaryResult,
    CancelCallback,
    ProgressCallback,
    RegionResult,
    jsonable,
    validate_paired_energy_axes,
)
from .processing import (
    auto_orientation_confident,
    check_cancel,
    column_spectra_for_mask,
    fourier_ratio_deconvolution,
    fourier_ratio_deconvolution_multi,
    inward_distance_nm,
    low_loss_thickness,
    masks_for_distance_bins,
    offset_boundary,
    process_region_spectrum,
    signal_to_noise,
    trace_surface,
)
from .references import prepare_references, reference_identifiability

# 全局能量平移扫描的名义范围与步长（eV）。实际范围会被数据覆盖余量收窄。
SHIFT_SCAN_LIMIT_EV = 3.0
SHIFT_SCAN_STEP_EV = 0.05
# 样品区中位 t/λ 超过该值时，Fourier-ratio 校正的可靠性下降，结果标记告警。
THICKNESS_WARNING_T_OVER_LAMBDA = 1.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _environment_info() -> dict[str, Any]:
    """记录影响数值结果的依赖版本，供复现性审计。"""

    try:
        ncempy_version: str | None = _distribution_version("ncempy")
    except PackageNotFoundError:  # pragma: no cover - 取决于安装方式
        ncempy_version = None
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "ncempy": ncempy_version,
        "platform": platform.platform(),
    }


def _emit(callback: ProgressCallback | None, message: str, fraction: float | None = None) -> None:
    if callback is not None:
        callback(message, fraction)


def _quantiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.percentile(values, 2.5, axis=0),
        np.percentile(values, 50.0, axis=0),
        np.percentile(values, 97.5, axis=0),
    )


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 0 else 0.0


def _find_primary_bins(config: AnalysisConfig) -> tuple[str, str]:
    edge = config.distance_bins[0].label
    bulk = next(
        (item.label for item in config.distance_bins if item.high_nm is None), config.distance_bins[-1].label
    )
    return edge, bulk


def _ensure_cu_triplet(config: AnalysisConfig) -> None:
    states = sorted(reference.oxidation_state for reference in config.references)
    if states != [0, 1, 2]:
        raise ValueError(
            "Cu L2,3 v1 需要恰好三个参考谱，价态标记为 0、1、2。"
            "其他元素/任意端元组合将在下一阶段作为插件式模型加入。"
        )


def _fit_energy_axis(high_energy_ev: np.ndarray, config: AnalysisConfig) -> np.ndarray:
    step = float(np.median(np.diff(high_energy_ev)))
    values = np.arange(config.fit_min_ev, config.fit_max_ev + 0.25 * step, step)
    values = values[values <= config.fit_max_ev + 1e-8]
    if values.size < 20 or values.min() < high_energy_ev.min() or values.max() > high_energy_ev.max():
        raise ValueError(f"高损谱不充分覆盖拟合区间 {config.fit_min_ev:g}–{config.fit_max_ev:g} eV。")
    return values.astype(float)


def _validate_ascending_energy(energy_ev: np.ndarray, role: str) -> None:
    """样品谱的能量轴必须严格递增；np.interp 对非升序轴会静默出错。"""

    if energy_ev.size >= 2 and not np.all(np.diff(energy_ev) > 0):
        raise ValueError(
            f"{role}的能量轴不是严格递增（{float(energy_ev.min()):.3f}–{float(energy_ev.max()):.3f} eV）；"
            "v1 只支持能量轴单调递增的 DM3/DM4，请检查数据对象是否选对。"
        )


def _profile_row(
    result: RegionResult,
    labels: tuple[str, ...],
) -> dict[str, Any]:
    low, median, high = _quantiles(result.bootstrap_fractions)
    row: dict[str, Any] = {
        "bin": result.label,
        "distance_low_nm": result.low_nm,
        "distance_high_nm": result.high_nm,
        "pixels": result.pixel_count,
        "along_surface_spectra": result.along_count,
        "snr": result.snr,
        "power_law_exponent": result.background_exponent,
        "power_law_log_r2": result.background_log_r2,
        "r2": result.fit.r2,
        "nrmse": result.fit.nrmse,
        "delta_bic_m02_minus_m012": result.fits[MODEL_CU0_CU2].bic - result.fit.bic,
        "bootstrap_delta_bic_p2_5": float(np.percentile(result.bootstrap_delta_bic, 2.5)),
        "bootstrap_delta_bic_median": float(np.median(result.bootstrap_delta_bic)),
        "bootstrap_delta_bic_p97_5": float(np.percentile(result.bootstrap_delta_bic, 97.5)),
    }
    for index, label in enumerate(labels):
        row[f"{label}_fraction"] = float(result.fit.fractions[index])
        row[f"{label}_bootstrap_p2_5"] = float(low[index])
        row[f"{label}_bootstrap_median"] = float(median[index])
        row[f"{label}_bootstrap_p97_5"] = float(high[index])
    return row


def _region_process_function(
    high_energy_ev: np.ndarray,
    fit_energy_ev: np.ndarray,
    config: AnalysisConfig,
    *,
    background_min_ev: float | None = None,
    shift_ev: float = 0.0,
) -> Any:
    lower = config.background_min_ev if background_min_ev is None else background_min_ev

    def process(columns: np.ndarray) -> np.ndarray:
        return process_region_spectrum(
            columns,
            high_energy_ev,
            fit_energy_ev,
            lower,
            config.background_max_ev,
            config.smoothing_ev,
            config.savgol_polyorder,
            shift_ev,
        )[0]

    return process


def _determine_global_shift_from_columns(
    columns: np.ndarray,
    high_energy_ev: np.ndarray,
    fit_energy_ev: np.ndarray,
    references: np.ndarray,
    config: AnalysisConfig,
) -> tuple[float, list[dict[str, float]], float]:
    """固定参考谱、平移样品谱，采用相对 SSE 选择内部区域全局能量校准。

    扫描范围以数据对拟合区间的覆盖余量收窄（不越过数据端点），返回
    (最优平移, 扫描表, 实际扫描半宽)。平移越界会让 np.interp 静默外推，
    使 SNR 估计失真，因此这里从源头限制。
    """

    margin_low = float(config.fit_min_ev - high_energy_ev.min())
    margin_high = float(high_energy_ev.max() - config.fit_max_ev)
    limit = min(SHIFT_SCAN_LIMIT_EV, margin_low, margin_high)
    if limit < SHIFT_SCAN_STEP_EV:
        raise ValueError(
            "高损谱对拟合区间的覆盖余量不足，无法执行能量平移扫描；"
            "请把拟合区间移到数据范围内部或扩大能量采集范围。"
        )
    rows: list[dict[str, float]] = []
    best_shift: float | None = None
    best_objective = np.inf
    for shift in np.arange(-limit, limit + 0.5 * SHIFT_SCAN_STEP_EV, SHIFT_SCAN_STEP_EV):
        spectrum, _ = process_region_spectrum(
            columns,
            high_energy_ev,
            fit_energy_ev,
            config.background_min_ev,
            config.background_max_ev,
            config.smoothing_ev,
            config.savgol_polyorder,
            float(shift),
        )
        fit = fit_all_cu_models(spectrum, references)[MODEL_FULL]
        relative_sse = fit.rss / max(float(np.sum(spectrum * spectrum)), np.finfo(float).eps)
        rows.append(
            {
                "shift_ev": float(shift),
                "relative_sse": float(relative_sse),
                "rss": fit.rss,
                "bic": fit.bic,
                "r2": fit.r2,
            }
        )
        if relative_sse < best_objective:
            best_objective = relative_sse
            best_shift = float(shift)
    if best_shift is None:  # pragma: no cover - scan always has entries
        raise RuntimeError("无法确定全局能量平移。")
    return best_shift, rows, limit


def _run_sensitivity(
    *,
    config: AnalysisConfig,
    high_loss: np.ndarray,
    low_loss: np.ndarray,
    low_energy_ev: np.ndarray,
    high_energy_ev: np.ndarray,
    fit_energy_ev: np.ndarray,
    references: np.ndarray,
    boundary: BoundaryResult,
    y_step_nm: float,
    x_step_nm: float,
    edge_label: str,
    global_shift_ev: float,
    progress: ProgressCallback | None,
    cancel: CancelCallback | None,
) -> list[dict[str, Any]]:
    """针对 E1 评估去卷积、背景和边界设置的离散敏感性。"""

    if not config.run_sensitivity:
        return []
    rows: list[dict[str, Any]] = []
    regularizations = tuple(
        sorted({*config.sensitivity_regularizations, config.deconvolution_regularization})
    )
    total = max(
        1,
        len(regularizations)
        * len(config.sensitivity_background_mins_ev)
        * len(config.sensitivity_boundary_offsets_pixels),
    )
    completed = 0
    # 敏感性分析只考察 E1 及其边界偏移邻域内的像素；仅对该子集执行去卷积，
    # 避免对整个 SI 反复做昂贵的 FFT。未选中的像素输出为 NaN，不会进入拟合。
    offset_union = np.zeros(high_loss.shape[1:], dtype=bool)
    for offset in config.sensitivity_boundary_offsets_pixels:
        shifted_boundary = offset_boundary(boundary, high_loss.shape[1:], int(offset))
        offset_distance = inward_distance_nm(
            shifted_boundary, high_loss.shape[1:], y_step_nm, x_step_nm
        )
        offset_union |= masks_for_distance_bins(offset_distance, config.distance_bins)[edge_label]
    sensitivity_pixels = np.flatnonzero(offset_union.ravel())
    _emit(
        progress,
        f"敏感性分析：对 {len(regularizations)} 组正则化参数共享 FFT 去卷积",
        None,
    )
    cubes, _ = fourier_ratio_deconvolution_multi(
        high_loss,
        low_loss,
        low_energy_ev,
        regularizations,
        config.low_loss_baseline_ev,
        config.zlp_window_ev,
        high_energy_ev=high_energy_ev,
        cancel=cancel,
        pixel_indices=sensitivity_pixels,
    )
    for regularization in regularizations:
        cube = cubes[float(regularization)]
        for background_min in sorted(set(config.sensitivity_background_mins_ev)):
            for offset in config.sensitivity_boundary_offsets_pixels:
                check_cancel(cancel)
                shifted_boundary = offset_boundary(boundary, high_loss.shape[1:], offset)
                distance = inward_distance_nm(shifted_boundary, high_loss.shape[1:], y_step_nm, x_step_nm)
                mask = masks_for_distance_bins(distance, config.distance_bins)[edge_label]
                columns = column_spectra_for_mask(cube, mask, shifted_boundary.orientation)
                if columns.shape[0] < 2:
                    completed += 1
                    continue
                try:
                    processed = _region_process_function(
                        high_energy_ev,
                        fit_energy_ev,
                        config,
                        background_min_ev=float(background_min),
                        shift_ev=global_shift_ev,
                    )(columns)
                    fits = fit_all_cu_models(processed, references)
                    full = fits[MODEL_FULL]
                    row = {
                        "regularization": float(regularization),
                        "background_min_ev": float(background_min),
                        "background_max_ev": float(config.background_max_ev),
                        "boundary_offset_pixels": int(offset),
                        "pixels": int(mask.sum()),
                        "along_surface_spectra": int(columns.shape[0]),
                        "snr": signal_to_noise(processed),
                        "r2": full.r2,
                        "nrmse": full.nrmse,
                        "delta_bic_m02_minus_m012": fits[MODEL_CU0_CU2].bic - full.bic,
                        "Cu0_fraction": float(full.fractions[0]),
                        "Cu1_fraction": float(full.fractions[1]),
                        "Cu2_fraction": float(full.fractions[2]),
                    }
                    row["qc_pass"] = bool(row["r2"] >= config.qc_min_r2 and row["snr"] >= config.qc_min_snr)
                    rows.append(row)
                except ValueError as exc:
                    rows.append(
                        {
                            "regularization": float(regularization),
                            "background_min_ev": float(background_min),
                            "background_max_ev": float(config.background_max_ev),
                            "boundary_offset_pixels": int(offset),
                            "error": str(exc),
                            "qc_pass": False,
                        }
                    )
                completed += 1
                # 前一阶段已 emit 0.72，从这里继续推进到 0.80，避免进度条倒退。
                _emit(progress, "敏感性分析中", 0.72 + 0.08 * completed / total)
    return rows


def _run_segments(
    edge_columns: np.ndarray,
    *,
    along_step_nm: float,
    process: Any,
    references: np.ndarray,
    config: AnalysisConfig,
) -> list[dict[str, Any]]:
    """将最外层按相互独立的沿表面段重新拟合，检查空间连续性。"""

    width = max(1, round(config.along_surface_segment_nm / max(along_step_nm, 1e-12)))
    rows: list[dict[str, Any]] = []
    for start in range(0, edge_columns.shape[0], width):
        stop = min(edge_columns.shape[0], start + width)
        if stop - start < 2:
            continue
        processed = process(edge_columns[start:stop])
        fits = fit_all_cu_models(processed, references)
        full = fits[MODEL_FULL]
        snr = signal_to_noise(processed)
        delta_bic = fits[MODEL_CU0_CU2].bic - full.bic
        qc = full.r2 >= config.qc_min_r2 and snr >= config.qc_min_snr
        rows.append(
            {
                "start_along_index": int(start),
                "stop_along_index_exclusive": int(stop),
                "start_nm": float(start * along_step_nm),
                "end_nm": float(stop * along_step_nm),
                "center_nm": float((start + stop) * along_step_nm / 2.0),
                "spectra": int(stop - start),
                "Cu0_fraction": float(full.fractions[0]),
                "Cu1_fraction": float(full.fractions[1]),
                "Cu2_fraction": float(full.fractions[2]),
                "r2": full.r2,
                "nrmse": full.nrmse,
                "snr": snr,
                "delta_bic_m02_minus_m012": delta_bic,
                "qc_pass": bool(qc),
            }
        )
    return rows


def run_analysis(
    config: AnalysisConfig,
    *,
    progress: ProgressCallback | None = None,
    cancel: CancelCallback | None = None,
) -> AnalysisArtifacts:
    """运行 v1 完整分析，但不写输出；调用方用 reporting.export_artifacts 导出。

    此接口适合 GUI、CLI 和自动化测试共用。输出目录的创建与文件写入显式放在
    reporting 模块，使原始数据读取和结果写入边界清晰。
    """

    config.validate()
    _ensure_cu_triplet(config)
    _emit(progress, "校验原始数据哈希", 0.02)
    hash_watch = HashWatch(config.input_path)
    hash_before = hash_watch.current()

    _emit(progress, "检查 DM4 数据对象", 0.05)
    items = inspect_dm4(config.input_path)
    inferred_survey, inferred_low, inferred_high = infer_dataset_indices(items)
    survey_index = config.survey_dataset if config.survey_dataset is not None else inferred_survey
    low_index = config.low_loss_dataset if config.low_loss_dataset is not None else inferred_low
    high_index = config.high_loss_dataset if config.high_loss_dataset is not None else inferred_high
    if low_index is None or high_index is None:
        raise ValueError("无法自动识别低损和高损 Spectrum Image，请在 GUI 中指定数据集编号。")

    _emit(progress, "读取配对 Dual-EELS 数据", 0.10)
    low = read_dataset(config.input_path, low_index)
    high = read_dataset(config.input_path, high_index)
    survey = read_dataset(config.input_path, survey_index) if survey_index is not None else None
    if low.data.ndim != 3 or high.data.ndim != 3:
        raise ValueError(
            "v1 需要低损和高损对象均为 (energy, y, x) Spectrum Image。"
            f"低损={low.data.shape}，高损={high.data.shape}。"
        )
    if low.data.shape != high.data.shape:
        raise ValueError("低损和高损对象空间/能量维度不一致；v1 仅支持已配对的 Dual-EELS 数据。")
    if low.energy_ev.size != low.data.shape[0] or high.energy_ev.size != high.data.shape[0]:
        raise ValueError("无法确认 EELS 能量轴位于第一个维度。")
    _validate_ascending_energy(low.energy_ev, "低损对象")
    _validate_ascending_energy(high.energy_ev, "高损对象")
    # R1：形状一致不等于能量栅格一致。低/高损色散不同时，Fourier-ratio 会把
    # 按低损索引构造的核与高损谱在错误频率尺度上相乘，且不报错。
    validate_paired_energy_axes(low.energy_ev, high.energy_ev)
    y_step_nm, x_step_nm, step_diagnostics = spatial_steps_nm(high)
    tags = all_tags(config.input_path)
    registered_survey = register_survey_to_si(survey, tags, high.data.shape[1:])
    if registered_survey is None:
        diagnostic_mask = (high.energy_ev >= config.fit_min_ev) & (high.energy_ev <= config.fit_max_ev)
        registered_survey = np.mean(high.data[diagnostic_mask], axis=0)
        survey_source = "integrated_core_loss_fallback"
    else:
        survey_source = "survey_tag_registration"

    _emit(progress, "定位物理样品边缘", 0.16)
    boundary = trace_surface(
        registered_survey,
        config.surface_orientation,
        config.surface_search_depth_pixels,
        config.surface_smoothness_penalty,
    )
    distance_nm = inward_distance_nm(boundary, high.data.shape[1:], y_step_nm, x_step_nm)
    masks = masks_for_distance_bins(distance_nm, config.distance_bins)

    _emit(progress, "计算低损厚度与 ZLP 诊断", 0.21)
    thickness, thickness_diagnostics = low_loss_thickness(
        low.data,
        low.energy_ev,
        config.low_loss_baseline_ev,
        config.zlp_window_ev,
    )
    # 样品区（任一距离层内）的中位 t/λ：超过阈值时 Fourier-ratio 校正与
    # ELNES 定量都变得不可靠，必须作为显式告警进入决策与报告。
    sample_mask = np.zeros(thickness.shape, dtype=bool)
    for item in config.distance_bins:
        sample_mask |= masks[item.label]
    median_t_over_lambda = (
        float(np.nanmedian(thickness[sample_mask])) if sample_mask.any() else float("nan")
    )
    thickness_quality = {
        "median_t_over_lambda_sample": median_t_over_lambda,
        "warning_threshold": THICKNESS_WARNING_T_OVER_LAMBDA,
        "exceeds_threshold": bool(median_t_over_lambda > THICKNESS_WARNING_T_OVER_LAMBDA),
        "note": "t/λ 基于 log-ratio 法且总积分在 200 eV 或谱末端截断，为半定量诊断量。",
    }

    _emit(progress, "执行配对低损复散射校正", 0.30)
    deconvolved, deconvolution_diagnostics = fourier_ratio_deconvolution(
        high.data,
        low.data,
        low.energy_ev,
        config.deconvolution_regularization,
        config.low_loss_baseline_ev,
        config.zlp_window_ev,
        high_energy_ev=high.energy_ev,
        cancel=cancel,
    )
    check_cancel(cancel)
    fit_energy_ev = _fit_energy_axis(high.energy_ev, config)

    _emit(progress, "准备参考谱并检查可辨识性", 0.42)
    base_references, labels, reference_details = prepare_references(
        config.references,
        fit_energy_ev,
        config.smoothing_ev,
        config.savgol_polyorder,
    )

    columns_by_region: dict[str, np.ndarray] = {}
    for item in config.distance_bins:
        columns = column_spectra_for_mask(deconvolved, masks[item.label], boundary.orientation)
        if columns.shape[0] < 2:
            raise ValueError(f"{item.label} 区域可用的沿表面谱少于 2 条，请调整边缘或距离分层。")
        columns_by_region[item.label] = columns

    total_si_pixels = int(high.data.shape[1] * high.data.shape[2])
    assigned_pixels = int(sum(int(masks[item.label].sum()) for item in config.distance_bins))
    coverage = {
        "total_si_pixels": total_si_pixels,
        "bins": [
            {
                "bin": item.label,
                "low_nm": item.low_nm,
                "high_nm": item.high_nm,
                "pixels": int(masks[item.label].sum()),
            }
            for item in config.distance_bins
        ],
        "unassigned_pixels": total_si_pixels - assigned_pixels,
        "unassigned_fraction": float((total_si_pixels - assigned_pixels) / total_si_pixels)
        if total_si_pixels
        else 0.0,
    }

    edge_label, bulk_label = _find_primary_bins(config)
    global_shift_ev, shift_scan, shift_scan_limit_ev = _determine_global_shift_from_columns(
        columns_by_region[bulk_label],
        high.energy_ev,
        fit_energy_ev,
        base_references,
        config,
    )
    references = base_references
    base_process = _region_process_function(
        high.energy_ev,
        fit_energy_ev,
        config,
        shift_ev=global_shift_ev,
    )

    _emit(progress, "拟合距离剖面并估计不确定度", 0.48)
    rng = np.random.default_rng(config.random_seed)
    region_results: dict[str, RegionResult] = {}
    profile_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for region_index, distance_bin in enumerate(config.distance_bins):
        check_cancel(cancel)
        label = distance_bin.label
        columns = columns_by_region[label]
        processed, details = process_region_spectrum(
            columns,
            high.energy_ev,
            fit_energy_ev,
            config.background_min_ev,
            config.background_max_ev,
            config.smoothing_ev,
            config.savgol_polyorder,
            global_shift_ev,
        )
        fits = fit_all_cu_models(processed, references)
        full = fits[MODEL_FULL]
        boot_fractions, boot_delta_bic, _ = bootstrap_region(
            columns,
            base_process,
            references,
            config.bootstrap_resamples,
            config.bootstrap_block_columns,
            rng,
            cancel=cancel,
        )
        result = RegionResult(
            label=label,
            low_nm=distance_bin.low_nm,
            high_nm=distance_bin.high_nm,
            pixel_count=int(masks[label].sum()),
            along_count=int(columns.shape[0]),
            snr=float(details["snr"]),
            background_exponent=float(details["exponent"]),
            background_log_r2=float(details["log_fit_r2"]),
            fit=full,
            fits=fits,
            bootstrap_fractions=boot_fractions,
            bootstrap_delta_bic=boot_delta_bic,
            processed_spectrum=processed,
            raw_column_spectra=columns,
        )
        region_results[label] = result
        profile_rows.append(_profile_row(result, labels))
        for fit in fits.values():
            row = fit_to_row(fit, labels)
            row["bin"] = label
            model_rows.append(row)
        fraction = 0.48 + 0.22 * (region_index + 1) / len(config.distance_bins)
        _emit(progress, f"已完成 {label} 的拟合与 bootstrap", fraction)

    _emit(progress, "比较边缘与内部并执行检出能力测试", 0.72)
    edge_result = region_results[edge_label]
    bulk_result = region_results[bulk_label]
    paired_difference = paired_bootstrap_difference(
        edge_result.raw_column_spectra,
        bulk_result.raw_column_spectra,
        base_process,
        references,
        config.bootstrap_resamples,
        config.bootstrap_block_columns,
        rng,
        cancel=cancel,
    )
    injection_rows, detection_limit = injection_recovery(
        edge_result.fits[MODEL_CU0_CU2],
        references,
        fit_energy_ev,
        config.injection_simulations,
        rng,
        config.bic_strong_support,
        fractions=config.injection_fractions,
        residual_block_width=config.injection_residual_block_channels,
        cancel=cancel,
    )

    along_step_nm = x_step_nm if boundary.orientation in {"top", "bottom"} else y_step_nm
    segment_rows = _run_segments(
        edge_result.raw_column_spectra,
        along_step_nm=along_step_nm,
        process=base_process,
        references=references,
        config=config,
    )
    sensitivity_rows = _run_sensitivity(
        config=config,
        high_loss=high.data,
        low_loss=low.data,
        low_energy_ev=low.energy_ev,
        high_energy_ev=high.energy_ev,
        fit_energy_ev=fit_energy_ev,
        references=references,
        boundary=boundary,
        y_step_nm=y_step_nm,
        x_step_nm=x_step_nm,
        edge_label=edge_label,
        global_shift_ev=global_shift_ev,
        progress=progress,
        cancel=cancel,
    )

    edge_low, edge_median, _edge_high = _quantiles(edge_result.bootstrap_fractions)
    enrich_low, enrich_median, enrich_high = _quantiles(paired_difference)
    delta_bic = edge_result.fits[MODEL_CU0_CU2].bic - edge_result.fit.bic
    difference = edge_result.processed_spectrum - bulk_result.processed_spectrum
    difference_metrics = {
        "cosine_with_cu1_minus_cu0": _cosine(difference, references[:, 1] - references[:, 0]),
        "cosine_with_cu2_minus_cu0": _cosine(difference, references[:, 2] - references[:, 0]),
    }
    accepted_sensitivity = [
        row
        for row in sensitivity_rows
        if row.get("qc_pass") and row.get("boundary_offset_pixels") in {0, 1} and "Cu1_fraction" in row
    ]
    qc_segments = [row for row in segment_rows if row["qc_pass"]]
    supported_segments = [
        row
        for row in qc_segments
        if row["Cu1_fraction"] > 0 and row["delta_bic_m02_minus_m012"] >= config.bic_strong_support
    ]
    checks = {
        "overall_edge_quality_pass": bool(
            edge_result.fit.r2 >= config.qc_min_r2 and edge_result.snr >= config.qc_min_snr
        ),
        "cu1_bootstrap_ci_excludes_zero": bool(edge_low[1] > 0),
        "cu1_edge_enrichment_ci_excludes_zero": bool(enrich_low[1] > 0),
        "adding_cu1_has_delta_bic_at_least_threshold": bool(delta_bic >= config.bic_strong_support),
        "cu1_exceeds_injection_detection_limit": bool(
            detection_limit is not None and edge_median[1] >= detection_limit
        ),
        "difference_more_cu1_like_than_cu2_like": bool(
            difference_metrics["cosine_with_cu1_minus_cu0"] > difference_metrics["cosine_with_cu2_minus_cu0"]
        ),
        "cu1_positive_in_qc_accepted_sensitivity_cases": bool(accepted_sensitivity)
        and min(float(row["Cu1_fraction"]) for row in accepted_sensitivity) > 0,
        "cu1_supported_in_multiple_segments": bool(
            len(qc_segments) >= 3 and len(supported_segments) >= math.ceil(len(qc_segments) / 2)
        ),
    }
    support_keys = (
        "overall_edge_quality_pass",
        "cu1_bootstrap_ci_excludes_zero",
        "cu1_edge_enrichment_ci_excludes_zero",
        "adding_cu1_has_delta_bic_at_least_threshold",
        "cu1_exceeds_injection_detection_limit",
        "cu1_supported_in_multiple_segments",
    )
    supports_mixed_state = all(checks[key] for key in support_keys)
    if supports_mixed_state:
        evidence_level = "supports_Cu0_Cu1_mixed_spectral_state"
    elif edge_result.fit.fractions[1] > 0 and checks["overall_edge_quality_pass"]:
        evidence_level = "Cu1_like_hint_not_independently_identified"
    else:
        evidence_level = "no_reliable_Cu1_like_signal"

    hash_after_compute = hash_watch.current()
    if hash_before != hash_after_compute:
        raise RuntimeError("分析期间原始 DM3/DM4 的 SHA-256 发生变化，已停止导出。")

    block_width_suggestions = {
        label: int(suggested_block_width(result.raw_column_spectra))
        for label, result in region_results.items()
    }
    paired_method = (
        "shared_block_positions"
        if edge_result.raw_column_spectra.shape[0] == bulk_result.raw_column_spectra.shape[0]
        else "independent_resampling"
    )
    summary: dict[str, Any] = {
        "application": {
            "name": "EELS Edge Analyzer",
            "version": __version__,
            "timestamp_utc": _utc_now(),
        },
        "environment": _environment_info(),
        "input": {
            "file": str(config.input_path),
            "sha256_before_and_after_compute": hash_before,
            "datasets": {
                "survey": survey_index,
                "low_loss": low_index,
                "high_loss": high_index,
                "low_loss_shape": list(low.data.shape),
                "high_loss_shape": list(high.data.shape),
            },
            "spatial_step_nm": {"y": y_step_nm, "x": x_step_nm},
            "spatial_step_details": step_diagnostics,
        },
        "configuration": config.to_dict(),
        "surface_registration": {
            "source": survey_source,
            "orientation": boundary.orientation,
            "boundary_pixels": boundary.boundary.tolist(),
            "score_mean": boundary.score_mean,
            "selection": boundary.selection,
            "auto_orientation_confident": auto_orientation_confident(boundary.selection),
        },
        "low_loss": thickness_diagnostics,
        "thickness_quality": thickness_quality,
        "deconvolution": deconvolution_diagnostics,
        "distance_binning": coverage,
        "reference_details": reference_details,
        "reference_identifiability": reference_identifiability(references),
        "global_energy_shift_ev": global_shift_ev,
        "global_shift_scan_limit_ev": shift_scan_limit_ev,
        "global_shift_scan": shift_scan,
        "profile": profile_rows,
        "model_comparison": model_rows,
        "edge_minus_bulk_paired_bootstrap": {
            f"{label}_p2_5": float(enrich_low[index]) for index, label in enumerate(labels)
        }
        | {f"{label}_median": float(enrich_median[index]) for index, label in enumerate(labels)}
        | {f"{label}_p97_5": float(enrich_high[index]) for index, label in enumerate(labels)},
        "edge_minus_bulk_bootstrap_method": paired_method,
        "difference_spectrum": difference_metrics,
        "detection_limit_cu1_fraction": detection_limit,
        "sensitivity": sensitivity_rows,
        "along_surface_segments": segment_rows,
        "along_surface_block_width": {
            "configured_columns": int(config.bootstrap_block_columns),
            "suggested_from_autocorrelation": block_width_suggestions,
            "configured_below_suggestion": sorted(
                label
                for label, width in block_width_suggestions.items()
                if config.bootstrap_block_columns < width
            ),
        },
        "decision": {
            "evidence_level": evidence_level,
            "supports_Cu0_Cu1_mixed_spectral_state": supports_mixed_state,
            "checks": checks,
            "note": (
                "输出为投影谱权重。EELS 光谱混合不能单独证明晶体学复合相；"
                "相结构需由 HRTEM、纳米衍射、4D-STEM 或其他独立证据确认。"
            ),
        },
        "scientific_status": (
            "semi_quantitative_projected_spectral_fractions; reference similarity, "
            "signal-to-noise, plural scattering, thickness and boundary partial-volume "
            "effects must be considered before drawing chemical-state conclusions"
        ),
    }
    _emit(progress, "核心计算完成", 0.92)
    return AnalysisArtifacts(
        summary=jsonable(summary),
        region_results=region_results,
        fit_energy_ev=fit_energy_ev,
        references=references,
        reference_labels=labels,
        high_energy_ev=high.energy_ev,
        survey_registered=registered_survey,
        boundary=boundary,
        distance_nm=distance_nm,
        thickness_t_over_lambda=thickness,
        deconvolved_cube=deconvolved,
        sensitivity_rows=sensitivity_rows,
        segment_rows=segment_rows,
        injection_rows=injection_rows,
    )
