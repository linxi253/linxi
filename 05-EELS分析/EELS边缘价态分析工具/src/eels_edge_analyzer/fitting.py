"""非负 MLLS、模型选择和空间相关 bootstrap。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import nnls

from .models import AnalysisCancelled, CancelCallback, FitResult
from .processing import normalize_positive_area

# 竞争模型的名称常量。这些字符串同时是 JSON/CSV 输出的键值，不得更改。
MODEL_CU0 = "M0_Cu0"
MODEL_CU0_CU1 = "M01_Cu0_Cu1"
MODEL_CU0_CU2 = "M02_Cu0_Cu2"
MODEL_FULL = "M012_Cu0_Cu1_Cu2"


def fit_model(
    spectrum: np.ndarray,
    references: np.ndarray,
    component_indices: tuple[int, ...],
    name: str,
) -> FitResult:
    """对一个端元子集执行非负最小二乘并计算 BIC/R2/NRMSE。"""

    if not component_indices:
        raise ValueError("至少需要一个端元。")
    y = np.asarray(spectrum, dtype=float)
    design = np.asarray(references[:, component_indices], dtype=float)
    coefficients_subset, _ = nnls(design, y)
    fitted = design @ coefficients_subset
    residual = y - fitted
    rss = float(np.sum(residual**2))
    n = max(1, y.size)
    k = len(component_indices)
    bic = float(n * np.log(max(rss / n, np.finfo(float).tiny)) + k * np.log(n))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - rss / total) if total > np.finfo(float).eps else 0.0
    span = float(np.max(y) - np.min(y))
    nrmse = float(np.sqrt(rss / n) / max(span, np.finfo(float).eps))
    coefficients = np.zeros(references.shape[1], dtype=float)
    coefficients[list(component_indices)] = coefficients_subset
    fractions = coefficients / max(float(coefficients.sum()), np.finfo(float).eps)
    return FitResult(
        name=name,
        component_indices=component_indices,
        coefficients=coefficients,
        fractions=fractions,
        fitted=fitted,
        residual=residual,
        rss=rss,
        bic=bic,
        r2=r2,
        nrmse=nrmse,
    )


def fit_all_cu_models(spectrum: np.ndarray, references: np.ndarray) -> dict[str, FitResult]:
    """比较 Cu0、Cu0+Cu1、Cu0+Cu2、Cu0+Cu1+Cu2 四个竞争模型。"""

    if references.shape[1] != 3:
        raise ValueError(
            f"Cu-L v1 需要按 Cu0、Cu1、Cu2 排序的三个参考谱；当前参考数为 {references.shape[1]}。"
        )
    return {
        MODEL_CU0: fit_model(spectrum, references, (0,), MODEL_CU0),
        MODEL_CU0_CU1: fit_model(spectrum, references, (0, 1), MODEL_CU0_CU1),
        MODEL_CU0_CU2: fit_model(spectrum, references, (0, 2), MODEL_CU0_CU2),
        MODEL_FULL: fit_model(spectrum, references, (0, 1, 2), MODEL_FULL),
    }


def moving_block_indices(
    count: int,
    block_width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """生成一个循环移动块 bootstrap 抽样索引，保留沿表面局部相关性。"""

    if count < 1:
        raise ValueError("没有可用于 bootstrap 的沿表面谱。")
    width = max(1, min(int(block_width), count))
    n_blocks = int(np.ceil(count / width))
    starts = rng.integers(0, count, size=n_blocks)
    pieces = [(start + np.arange(width)) % count for start in starts]
    return np.concatenate(pieces)[:count]


def bootstrap_region(
    column_spectra: np.ndarray,
    process: Callable[[np.ndarray], np.ndarray],
    references: np.ndarray,
    resamples: int,
    block_width: int,
    rng: np.random.Generator,
    *,
    cancel: CancelCallback | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对一个距离层执行移动块 bootstrap，返回组分、ΔBIC 与 R2 分布。"""

    fractions = np.empty((resamples, references.shape[1]), dtype=float)
    delta_bic = np.empty(resamples, dtype=float)
    r2 = np.empty(resamples, dtype=float)
    for iteration in range(resamples):
        if cancel is not None and iteration % 20 == 0 and cancel():
            raise AnalysisCancelled("用户已取消 bootstrap。")
        indices = moving_block_indices(column_spectra.shape[0], block_width, rng)
        processed = process(column_spectra[indices])
        fits = fit_all_cu_models(processed, references)
        full = fits[MODEL_FULL]
        fractions[iteration] = full.fractions
        delta_bic[iteration] = fits[MODEL_CU0_CU2].bic - full.bic
        r2[iteration] = full.r2
    return fractions, delta_bic, r2


def paired_block_indices(
    edge_count: int,
    bulk_count: int,
    block_width: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """为边缘与内部区域生成沿表面块抽样索引。

    两个区域覆盖同一批沿表面扫描线：列数一致时共享同一组块位置（真配对，
    消除沿表面共变分量、收紧差值的置信区间）；列数不一致时退化为相互独立
    的两组抽样。
    """

    if edge_count == bulk_count:
        shared = moving_block_indices(edge_count, block_width, rng)
        return shared, shared.copy()
    return (
        moving_block_indices(edge_count, block_width, rng),
        moving_block_indices(bulk_count, block_width, rng),
    )


def paired_bootstrap_difference(
    edge_columns: np.ndarray,
    bulk_columns: np.ndarray,
    process: Callable[[np.ndarray], np.ndarray],
    references: np.ndarray,
    resamples: int,
    block_width: int,
    rng: np.random.Generator,
    *,
    cancel: CancelCallback | None = None,
) -> np.ndarray:
    """对 E1 与内部的组分差执行配对块 bootstrap（列数一致时共享块位置）。"""

    differences = np.empty((resamples, references.shape[1]), dtype=float)
    for iteration in range(resamples):
        if cancel is not None and iteration % 20 == 0 and cancel():
            raise AnalysisCancelled("用户已取消 bootstrap。")
        edge_indices, bulk_indices = paired_block_indices(
            edge_columns.shape[0], bulk_columns.shape[0], block_width, rng
        )
        edge = process(edge_columns[edge_indices])
        bulk = process(bulk_columns[bulk_indices])
        edge_fit = fit_all_cu_models(edge, references)[MODEL_FULL]
        bulk_fit = fit_all_cu_models(bulk, references)[MODEL_FULL]
        differences[iteration] = edge_fit.fractions - bulk_fit.fractions
    return differences


def suggested_block_width(column_spectra: np.ndarray) -> int:
    """从沿表面积分强度的自相关估计移动块 bootstrap 的建议块宽。

    取归一化自相关首次降到 1/e 以下的滞后作为相关长度。仅供诊断输出：
    实际块宽仍由配置决定，但当配置块宽小于该建议时应在报告中提示
    不确定度可能被低估。
    """

    series = np.sum(np.maximum(np.asarray(column_spectra, dtype=float), 0.0), axis=1)
    series = series - float(np.mean(series))
    variance = float(np.sum(series**2))
    if series.size < 3 or variance <= np.finfo(float).eps:
        return 1
    correlations = np.correlate(series, series, mode="full")[series.size - 1 :] / variance
    threshold = float(np.exp(-1.0))
    for lag in range(1, series.size):
        if correlations[lag] < threshold:
            return max(1, lag)
    return max(1, series.size - 1)


def injection_recovery(
    baseline: FitResult,
    references: np.ndarray,
    energy_ev: np.ndarray,
    simulations: int,
    rng: np.random.Generator,
    bic_threshold: float,
    *,
    fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50),
    residual_block_width: int = 5,
    cancel: CancelCallback | None = None,
) -> tuple[list[dict[str, float]], float | None]:
    """在 Cu0+Cu2 基线残差上注入 Cu1，估计本数据的经验检出能力。

    基线拟合与参考谱先统一到单位正面积，确保 ``target`` 表示真正的谱面积
    分数而不受样品绝对强度影响。残差按同一面积缩放，再用移动块重采样保留
    平滑后相邻能量通道的局部相关性。
    """

    if simulations <= 0:
        return [], None
    if residual_block_width < 1:
        raise ValueError("残差移动块宽度至少为 1。")
    baseline_unit = normalize_positive_area(baseline.fitted, energy_ev)
    baseline_area = float(np.trapezoid(np.maximum(np.asarray(baseline.fitted, dtype=float), 0.0), energy_ev))
    residual_unit = np.asarray(baseline.residual, dtype=float) / baseline_area
    rows: list[dict[str, float]] = []
    detection_limit: float | None = None
    for target in fractions:
        if not 0.0 < target < 1.0:
            raise ValueError("Cu1 注入分数必须位于 0 与 1 之间。")
        recovered: list[float] = []
        support: list[float] = []
        for iteration in range(simulations):
            if cancel is not None and iteration % 20 == 0 and cancel():
                raise AnalysisCancelled("用户已取消注入恢复检验。")
            indices = moving_block_indices(residual_unit.size, residual_block_width, rng)
            random_residual = residual_unit[indices]
            synthetic = (1.0 - target) * baseline_unit + target * references[:, 1] + random_residual
            synthetic = normalize_positive_area(synthetic, energy_ev)
            fits = fit_all_cu_models(synthetic, references)
            full = fits[MODEL_FULL]
            recovered.append(float(full.fractions[1]))
            support.append(float(fits[MODEL_CU0_CU2].bic - full.bic >= bic_threshold))
        selection_rate = float(np.mean(support))
        row = {
            "true_cu1_fraction": float(target),
            "recovered_cu1_p2_5": float(np.percentile(recovered, 2.5)),
            "recovered_cu1_median": float(np.median(recovered)),
            "recovered_cu1_p97_5": float(np.percentile(recovered, 97.5)),
            "cu1_selection_rate_delta_bic_ge_threshold": selection_rate,
        }
        rows.append(row)
        if detection_limit is None and selection_rate >= 0.95:
            detection_limit = float(target)
    return rows, detection_limit


def fit_to_row(fit: FitResult, labels: tuple[str, ...]) -> dict[str, float | str]:
    """将一个模型拟合转换为适合 CSV/JSON 的扁平行。"""

    row: dict[str, float | str] = {
        "model": fit.name,
        "rss": fit.rss,
        "bic": fit.bic,
        "r2": fit.r2,
        "nrmse": fit.nrmse,
    }
    for label, coefficient, fraction in zip(labels, fit.coefficients, fit.fractions, strict=True):
        row[f"{label}_coefficient"] = float(coefficient)
        row[f"{label}_fraction"] = float(fraction)
    return row
