"""Dual-EELS 预处理、表面定位和距离分层。

这里的函数均可脱离 GUI 调用。原始 DM3/DM4 只经 ncempy 读取，所有数组均在内存
中处理，结果由 reporting 模块写入独立输出目录。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

from .models import AnalysisCancelled, BoundaryResult, CancelCallback, DistanceBin

# 区域谱预边基线窗口与边信号窗口的宽度（eV）。隐含假设 fit_min_ev 位于边 onset 之前。
PREEEDGE_BASELINE_WINDOW_EV = 3.0
EDGE_SIGNAL_START_OFFSET_EV = 4.0


def check_cancel(cancel: CancelCallback | None) -> None:
    if cancel is not None and cancel():
        raise AnalysisCancelled("用户已取消分析。")


def next_power_of_two(value: int) -> int:
    return 1 << max(1, int(value - 1).bit_length())


def positive_area(spectrum: np.ndarray, energy_ev: np.ndarray) -> float:
    """积分谱的正值面积，作为 MLLS 的统一归一化方式。"""

    return float(np.trapezoid(np.maximum(spectrum, 0.0), energy_ev))


def normalize_positive_area(spectrum: np.ndarray, energy_ev: np.ndarray) -> np.ndarray:
    area = positive_area(spectrum, energy_ev)
    if not np.isfinite(area) or area <= np.finfo(float).eps:
        raise ValueError("谱的正面积为零，不能归一化。")
    return np.asarray(spectrum, dtype=float) / area


def savgol_by_ev(
    spectrum: np.ndarray,
    energy_ev: np.ndarray,
    width_ev: float,
    polyorder: int,
) -> tuple[np.ndarray, int]:
    """将期望 eV 宽度转换为稳定的奇数 Savitzky-Golay 窗口。"""

    if spectrum.size < polyorder + 3:
        return np.asarray(spectrum, dtype=float), 1
    dispersion = float(np.median(np.abs(np.diff(energy_ev))))
    window = max(polyorder + 2, round(width_ev / max(dispersion, 1e-12)))
    if window % 2 == 0:
        window += 1
    window = min(window, spectrum.size if spectrum.size % 2 else spectrum.size - 1)
    if window <= polyorder:
        return np.asarray(spectrum, dtype=float), 1
    return savgol_filter(spectrum, window_length=window, polyorder=polyorder, mode="interp"), window


def low_loss_thickness(
    low_loss: np.ndarray,
    energy_ev: np.ndarray,
    baseline_range_ev: tuple[float, float],
    zlp_window_ev: tuple[float, float],
) -> tuple[np.ndarray, dict[str, object]]:
    """用 log-ratio 法给出相对 t/lambda 映射，同时检查 ZLP 位置。"""

    baseline_mask = (energy_ev >= baseline_range_ev[0]) & (energy_ev <= baseline_range_ev[1])
    zero_mask = (energy_ev >= zlp_window_ev[0]) & (energy_ev <= zlp_window_ev[1])
    total_max_ev = min(200.0, float(energy_ev.max()))
    total_mask = (energy_ev >= min(-5.0, zlp_window_ev[0])) & (energy_ev <= total_max_ev)
    if not baseline_mask.any() or not zero_mask.any() or not total_mask.any():
        raise ValueError("低损谱不覆盖所需的基线、ZLP 或总强度区间。")
    corrected = np.maximum(
        low_loss - np.median(low_loss[baseline_mask], axis=0, keepdims=True),
        0.0,
    )
    zero = np.trapezoid(corrected[zero_mask], energy_ev[zero_mask], axis=0)
    total = np.trapezoid(corrected[total_mask], energy_ev[total_mask], axis=0)
    thickness = np.maximum(np.log(np.maximum(total, 1e-12) / np.maximum(zero, 1e-12)), 0.0)
    zlp_indices = np.flatnonzero(zero_mask)[np.argmax(corrected[zero_mask], axis=0)]
    zlp_energy = energy_ev[zlp_indices]
    diagnostics = {
        "t_over_lambda_percentiles": np.percentile(thickness, [0, 5, 25, 50, 75, 95, 100]).tolist(),
        "zlp_peak_energy_percentiles_ev": np.percentile(zlp_energy, [0, 5, 25, 50, 75, 95, 100]).tolist(),
        # t/λ 是诊断量：总强度积分在 200 eV 或谱末端截断，高能损失未计入。
        "total_integral_max_ev": total_max_ev,
    }
    return thickness.astype(np.float32), diagnostics


def fourier_ratio_deconvolution_multi(
    high_loss: np.ndarray,
    low_loss: np.ndarray,
    low_energy_ev: np.ndarray,
    regularizations: tuple[float, ...] | list[float],
    baseline_range_ev: tuple[float, float],
    zlp_window_ev: tuple[float, float],
    *,
    chunk_rows: int = 8,
    cancel: CancelCallback | None = None,
    pixel_indices: np.ndarray | None = None,
) -> tuple[dict[float, np.ndarray], dict[str, object]]:
    """对多个正则化参数共享逐像素 FFT 的 Fourier-ratio 复散射校正。

    每个像素的 low/zero/high 三个 FFT 与正则化参数无关，只有 Wiener 比值随其
    变化；敏感性分析需要比较多个正则化取值时，共享 FFT 可将去卷积开销近似
    按正则化个数成比例下降。返回 {正则化值: 立方体} 与共享诊断。

    输入必须是 (energy, y, x) 并且低、高损空间网格及通道数相同。全立方体
    路径按空间行分块；敏感性分析可用 ``pixel_indices`` 只校正选定的像素
    子集（其余位置输出 NaN），避免对大 SI 重复计算。

    实测说明：逐像素循环的 32 KB 工作集常驻缓存，比按列批量的 FFT 实现
    更快，因此这里保留逐像素核心；性能收益来自子集路径而非批量 FFT。
    """

    regs = tuple(sorted({float(value) for value in regularizations}))
    if not regs:
        raise ValueError("至少需要一个去卷积正则化参数。")
    if any(value <= 0 for value in regs):
        raise ValueError("正则化参数必须为正。")
    if high_loss.shape != low_loss.shape:
        raise ValueError(
            "v1 的 Fourier-ratio 校正要求配对低损和高损 SI 具有相同形状；"
            f"得到 high={high_loss.shape}, low={low_loss.shape}。"
        )
    if high_loss.ndim != 3:
        raise ValueError("EELS SI 必须是 (energy, y, x) 三维数组。")
    baseline_mask = (low_energy_ev >= baseline_range_ev[0]) & (low_energy_ev <= baseline_range_ev[1])
    zero_window = (low_energy_ev >= zlp_window_ev[0]) & (low_energy_ev <= zlp_window_ev[1])
    if not baseline_mask.any() or not zero_window.any():
        raise ValueError("低损谱不包含去卷积所需的基线或 ZLP 区间。")
    zero_index = int(np.argmin(np.abs(low_energy_ev)))
    zero_candidates = np.flatnonzero(zero_window)
    n_energy, ny, nx = high_loss.shape
    n_fft = next_power_of_two(max(2 * n_energy, 2048))

    if pixel_indices is None:
        selected = np.arange(ny * nx)
    else:
        selected = np.asarray(pixel_indices, dtype=np.intp).ravel()
        if selected.size and (int(selected.min()) < 0 or int(selected.max()) >= ny * nx):
            raise ValueError("pixel_indices 超出 SI 像素范围。")

    outputs = {value: np.full((n_energy, ny, nx), np.nan, dtype=np.float32) for value in regs}
    out_flats = {value: outputs[value].reshape(n_energy, -1) for value in regs}
    high_flat = high_loss.reshape(n_energy, -1)
    low_flat = low_loss.reshape(n_energy, -1)
    peak_shifts: list[int] = []

    def _process_block(columns: np.ndarray) -> None:
        low_block = np.maximum(
            low_flat[:, columns] - np.median(low_flat[baseline_mask][:, columns], axis=0, keepdims=True),
            0.0,
        ).astype(float)
        for position in range(columns.size):
            low = np.array(low_block[:, position], copy=True)
            peak = int(zero_candidates[np.argmax(low[zero_window])])
            channel_shift = zero_index - peak
            peak_shifts.append(channel_shift)
            if channel_shift:
                # 与 DigitalMicrograph/原始验证流程一致：ZLP 局部对齐后以
                # 循环方式放置在 Fourier 原点附近。
                low = np.roll(low, channel_shift)

            kernel = np.zeros(n_fft, dtype=float)
            kernel[: n_energy - zero_index] = low[zero_index:]
            kernel[n_fft - zero_index :] = low[:zero_index]
            kernel_sum = float(np.sum(kernel))
            if kernel_sum <= 0:
                raise ValueError("低损核的积分非正，无法执行去卷积。")
            kernel /= kernel_sum

            zero = low.copy()
            zero[~zero_window] = 0.0
            zero_kernel = np.zeros(n_fft, dtype=float)
            zero_kernel[: n_energy - zero_index] = zero[zero_index:]
            zero_kernel[n_fft - zero_index :] = zero[:zero_index]
            zero_sum = float(np.sum(zero_kernel))
            if zero_sum <= 0:
                raise ValueError("零损核的积分非正，无法执行去卷积。")
            zero_kernel /= zero_sum

            padded_high = np.zeros(n_fft, dtype=float)
            padded_high[:n_energy] = high_flat[:, columns[position]]
            low_fft = np.fft.rfft(kernel)
            zero_fft = np.fft.rfft(zero_kernel)
            high_fft = np.fft.rfft(padded_high)
            denominator = np.abs(low_fft) ** 2
            for value in regs:
                wiener_ratio = zero_fft * np.conj(low_fft) / (denominator + value)
                out_flats[value][:, columns[position]] = np.fft.irfft(
                    high_fft * wiener_ratio, n_fft
                )[:n_energy]

    step = max(1, chunk_rows) * nx
    for c0 in range(0, selected.size, step):
        check_cancel(cancel)
        _process_block(selected[c0 : c0 + step])

    negative_by_reg = {}
    for value in regs:
        selected_flat = out_flats[value][:, selected]
        negative_by_reg[f"{value:g}"] = float(np.mean(selected_flat < 0)) if selected.size else 0.0
    diagnostics: dict[str, object] = {
        "method": "regularized Fourier-ratio using paired Dual-EELS low-loss kernel",
        "regularizations": list(regs),
        "fft_length": int(n_fft),
        "negative_value_fraction_by_regularization": negative_by_reg,
        "low_loss_baseline_range_ev": list(baseline_range_ev),
        "retained_zero_loss_range_ev": list(zlp_window_ev),
        "selected_pixel_fraction": float(selected.size / (ny * nx)),
        "zlp_integer_channel_shift_counts": {
            str(value): int(peak_shifts.count(value)) for value in sorted(set(peak_shifts))
        },
    }
    return outputs, diagnostics


def fourier_ratio_deconvolution(
    high_loss: np.ndarray,
    low_loss: np.ndarray,
    low_energy_ev: np.ndarray,
    regularization: float,
    baseline_range_ev: tuple[float, float],
    zlp_window_ev: tuple[float, float],
    *,
    chunk_rows: int = 8,
    cancel: CancelCallback | None = None,
    pixel_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """基于匹配的低损谱执行逐像素正则化 Fourier-ratio 复散射校正。

    单正则化便捷封装；实现共享 :func:`fourier_ratio_deconvolution_multi`
    的逐像素核心，诊断键与旧版保持兼容。
    """

    reg = float(regularization)
    cubes, shared = fourier_ratio_deconvolution_multi(
        high_loss,
        low_loss,
        low_energy_ev,
        (reg,),
        baseline_range_ev,
        zlp_window_ev,
        chunk_rows=chunk_rows,
        cancel=cancel,
        pixel_indices=pixel_indices,
    )
    diagnostics = {key: value for key, value in shared.items() if key != "regularizations"}
    diagnostics["regularization"] = reg
    by_reg = dict(diagnostics["negative_value_fraction_by_regularization"])  # type: ignore[arg-type]
    diagnostics["negative_value_fraction"] = by_reg[f"{reg:g}"]
    del diagnostics["negative_value_fraction_by_regularization"]
    return cubes[reg], diagnostics


def _trace_top(
    image: np.ndarray,
    max_depth: int,
    smoothness_penalty: float,
) -> tuple[np.ndarray, float]:
    """在图像顶部附近寻找一条连续的最大梯度边界。"""

    if image.ndim != 2:
        raise ValueError("表面追踪需要二维 survey 或积分信号图。")
    cleaned = np.nan_to_num(image.astype(float), nan=float(np.nanmedian(image)))
    smoothed = gaussian_filter(cleaned, sigma=(1.0, 1.0))
    scale = np.percentile(smoothed, 95, axis=0) - np.percentile(smoothed, 5, axis=0)
    scale = np.maximum(scale, np.finfo(float).eps)
    # 经方向变换后，候选表面都应满足“真空在上、样品在下”；因此只接受
    # dark-to-bright 的正梯度。使用绝对梯度会把扫描裁切边或内部缺陷误选为表面。
    score = np.gradient(smoothed, axis=0) / scale[None, :]
    candidate_rows = np.arange(min(max_depth, image.shape[0] - 2) + 1)
    costs = np.full((candidate_rows.size, image.shape[1]), np.inf)
    back = np.zeros_like(costs, dtype=int)
    costs[:, 0] = -score[candidate_rows, 0]
    for column in range(1, image.shape[1]):
        for candidate_index, row in enumerate(candidate_rows):
            transition = costs[:, column - 1] + smoothness_penalty * np.abs(candidate_rows - row)
            best = int(np.argmin(transition))
            costs[candidate_index, column] = transition[best] - score[row, column]
            back[candidate_index, column] = best
    path = np.zeros(image.shape[1], dtype=int)
    path[-1] = int(np.argmin(costs[:, -1]))
    for column in range(image.shape[1] - 1, 0, -1):
        path[column - 1] = back[path[column], column]
    boundary = candidate_rows[path]
    return boundary.astype(int), float(np.mean(score[boundary, np.arange(boundary.size)]))


def _auto_orientation_quality(
    item: BoundaryResult,
    image: np.ndarray,
    max_depth: int,
    dynamic_range: float,
) -> float:
    """auto 模式下对单个候选边界打分：局部梯度得分 × 归一化的真空-样品对比度。"""

    distance = inward_distance_nm(item, image.shape, 1.0, 1.0)
    outside = image[(distance < 0) & (distance >= -4)]
    inside = image[(distance >= 2) & (distance < max(6, max_depth))]
    if outside.size < 3 or inside.size < 3:
        return -np.inf
    contrast = float(np.nanmedian(inside) - np.nanmedian(outside))
    return item.score_mean * max(contrast, 0.0) / dynamic_range


def trace_surface(
    image: np.ndarray,
    orientation: str,
    max_depth: int,
    smoothness_penalty: float,
) -> BoundaryResult:
    """在 top/bottom/left/right/auto 模式下追踪物理表面。"""

    if orientation == "auto":
        candidates = [
            trace_surface(image, item, max_depth, smoothness_penalty)
            for item in ("top", "bottom", "left", "right")
        ]
        # 自动模式不仅比较局部梯度，还要求候选边界外侧接近真空、内侧出现
        # 持续更高的样品信号。这可抑制扫描裁切边和内部明暗带的误识别。
        dynamic_range = max(
            float(np.nanpercentile(image, 95) - np.nanpercentile(image, 5)),
            np.finfo(float).eps,
        )
        scored = [
            (item, _auto_orientation_quality(item, image, max_depth, dynamic_range))
            for item in candidates
        ]
        best = max(scored, key=lambda pair: pair[1])[0]
        selection = tuple(
            {
                "orientation": item.orientation,
                "score_mean": float(item.score_mean),
                "contrast_quality": float(quality),
            }
            for item, quality in scored
        )
        return BoundaryResult(
            orientation=best.orientation,
            boundary=best.boundary,
            score_mean=best.score_mean,
            coordinates=best.coordinates,
            selection=selection,
        )
    ny, nx = image.shape
    if orientation == "top":
        boundary, score = _trace_top(image, max_depth, smoothness_penalty)
        coordinates = np.column_stack((boundary, np.arange(nx)))
    elif orientation == "bottom":
        mirrored, score = _trace_top(image[::-1, :], max_depth, smoothness_penalty)
        boundary = ny - 1 - mirrored
        coordinates = np.column_stack((boundary, np.arange(nx)))
    elif orientation == "left":
        boundary, score = _trace_top(image.T, max_depth, smoothness_penalty)
        coordinates = np.column_stack((np.arange(ny), boundary))
    elif orientation == "right":
        mirrored, score = _trace_top(image[:, ::-1].T, max_depth, smoothness_penalty)
        boundary = nx - 1 - mirrored
        coordinates = np.column_stack((np.arange(ny), boundary))
    else:
        raise ValueError(f"未知表面方向: {orientation}")
    return BoundaryResult(
        orientation=orientation,
        boundary=np.asarray(boundary, dtype=int),
        score_mean=score,
        coordinates=np.asarray(coordinates, dtype=int),
    )


def auto_orientation_confident(selection: tuple[dict[str, Any], ...] | None) -> bool | None:
    """判断 auto 模式选出的方向是否具有可信的真空-样品明暗对比度。

    返回 None 表示非 auto 模式（用户显式指定方向，无需判断）；False 表示
    最优候选的对比度也不大于零——常见于明暗极性相反（如 BF 像）或表面
    不在视场内的数据，auto 结论不可信。
    """

    if not selection:
        return None
    best = max(float(entry.get("contrast_quality", -np.inf)) for entry in selection)
    return bool(np.isfinite(best) and best > 0.0)


def offset_boundary(boundary: BoundaryResult, shape: tuple[int, int], offset: int) -> BoundaryResult:
    """沿 inward 方向平移边界，用于边界误差敏感性检验。"""

    ny, nx = shape
    maximum = ny - 2 if boundary.orientation in {"top", "bottom"} else nx - 2
    # +offset 的定义是向样品内部移动；top/left 增大索引，bottom/right 减小索引。
    sign = 1 if boundary.orientation in {"top", "left"} else -1
    values = np.clip(boundary.boundary + sign * int(offset), 0, maximum)
    if boundary.orientation in {"top", "bottom"}:
        coordinates = np.column_stack((values, np.arange(nx)))
    else:
        coordinates = np.column_stack((np.arange(ny), values))
    return BoundaryResult(
        boundary.orientation, values.astype(int), boundary.score_mean, coordinates.astype(int)
    )


def inward_distance_nm(
    boundary: BoundaryResult,
    shape: tuple[int, int],
    y_step_nm: float,
    x_step_nm: float,
) -> np.ndarray:
    """计算每个 SI 像素相对于表面的有符号 inward 距离（nm）。"""

    ny, nx = shape
    rows = np.arange(ny)[:, None]
    columns = np.arange(nx)[None, :]
    if boundary.orientation == "top":
        pixels = rows - boundary.boundary[None, :]
        step = y_step_nm
    elif boundary.orientation == "bottom":
        pixels = boundary.boundary[None, :] - rows
        step = y_step_nm
    elif boundary.orientation == "left":
        pixels = columns - boundary.boundary[:, None]
        step = x_step_nm
    elif boundary.orientation == "right":
        pixels = boundary.boundary[:, None] - columns
        step = x_step_nm
    else:  # pragma: no cover - BoundaryResult 由 trace_surface 创建
        raise ValueError(f"未知边缘方向: {boundary.orientation}")
    return pixels.astype(float) * float(step)


def masks_for_distance_bins(
    distance_nm: np.ndarray,
    bins: tuple[DistanceBin, ...],
) -> dict[str, np.ndarray]:
    """为每个距离层生成非重叠掩膜，负距离为真空/边界外。"""

    masks: dict[str, np.ndarray] = {}
    # DM4 空间标尺常以 float32 存储，例如 2 × 1.099999994 nm。若直接与
    # 2.2 nm 比较会把恰好位于层边界的像素误分到前一层，故使用远小于一个
    # 像素的数值容差，并统一让边界归属下一层。
    tolerance = max(1e-6, float(np.nanmax(np.abs(distance_nm))) * 1e-9)
    for item in bins:
        if item.high_nm is None:
            masks[item.label] = distance_nm >= item.low_nm - tolerance
        else:
            masks[item.label] = (distance_nm >= item.low_nm - tolerance) & (
                distance_nm < item.high_nm - tolerance
            )
    return masks


def column_spectra_for_mask(
    cube: np.ndarray,
    mask: np.ndarray,
    orientation: str,
) -> np.ndarray:
    """在每条与表面平行的扫描线中先平均，保留沿表面相关性供块 bootstrap 使用。"""

    if cube.ndim != 3 or mask.shape != cube.shape[1:]:
        raise ValueError("数据立方体与距离掩膜的维度不匹配。")
    _, ny, nx = cube.shape
    count = nx if orientation in {"top", "bottom"} else ny
    values = np.full((count, cube.shape[0]), np.nan, dtype=float)
    for index in range(count):
        selected = (
            cube[:, mask[:, index], index]
            if orientation in {"top", "bottom"}
            else cube[:, index, mask[index, :]]
        )
        if selected.size:
            values[index] = np.mean(selected, axis=1)
    return values[np.all(np.isfinite(values), axis=1)]


def power_law_subtract(
    spectrum: np.ndarray,
    energy_ev: np.ndarray,
    background_min_ev: float,
    background_max_ev: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """对区域平均谱拟合 I=A*E^(-r) 并扣除背景。"""

    selected = (
        (energy_ev >= background_min_ev) & (energy_ev <= background_max_ev) & (energy_ev > 0) & (spectrum > 0)
    )
    if selected.sum() < 6:
        raise ValueError(f"背景区间 {background_min_ev:g}–{background_max_ev:g} eV 中的有效正通道不足。")
    log_energy = np.log(energy_ev[selected])
    log_signal = np.log(spectrum[selected])
    slope, intercept = np.polyfit(log_energy, log_signal, deg=1)
    fitted_log = intercept + slope * log_energy
    residual = log_signal - fitted_log
    total = float(np.sum((log_signal - np.mean(log_signal)) ** 2))
    log_r2 = 1.0 - float(np.sum(residual**2)) / total if total > 0 else 1.0
    background = np.exp(intercept) * np.maximum(energy_ev, 1e-12) ** slope
    return np.asarray(spectrum, dtype=float) - background, {
        "amplitude": float(np.exp(intercept)),
        "exponent": float(-slope),
        "log_fit_r2": float(log_r2),
    }


def signal_to_noise(spectrum: np.ndarray) -> float:
    """用相邻通道差分估计噪声，返回峰谷幅度/噪声的保守 SNR。"""

    if spectrum.size < 4:
        return 0.0
    noise = float(np.std(np.diff(spectrum)) / np.sqrt(2.0))
    amplitude = float(np.percentile(spectrum, 95) - np.percentile(spectrum, 5))
    return amplitude / max(noise, np.finfo(float).eps)


def process_region_spectrum(
    column_spectra: np.ndarray,
    high_energy_ev: np.ndarray,
    fit_energy_ev: np.ndarray,
    background_min_ev: float,
    background_max_ev: float,
    smoothing_ev: float,
    polyorder: int,
    shift_ev: float = 0.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """区域谱：先沿表面平均，再扣背景、平滑、能量平移和预边基线校正。

    样品谱不做正面积归一化；MLLS 会通过系数的总量吸收强度尺度，随后只将系数
    归一化为谱权重。这与参考谱的面积归一化相配套。
    """

    if column_spectra.size == 0:
        raise ValueError("该距离层没有可用像素。")
    # 能量平移后的查询点必须完整落在数据范围内；np.interp 会静默使用端点值
    # 外推，使预边窗口变成常数、噪声被估成 0、SNR 虚高为无穷大。
    queries = fit_energy_ev - shift_ev
    if queries.min() < high_energy_ev.min() - 1e-9 or queries.max() > high_energy_ev.max() + 1e-9:
        raise ValueError(
            f"能量平移 {shift_ev:.2f} eV 使拟合窗口 [{queries.min():.1f}, {queries.max():.1f}] eV "
            f"超出高损谱范围 [{high_energy_ev.min():.1f}, {high_energy_ev.max():.1f}] eV；"
            "请缩小平移扫描或把拟合区间移到数据内部。"
        )
    mean = np.mean(column_spectra, axis=0)
    corrected, background = power_law_subtract(
        mean,
        high_energy_ev,
        background_min_ev,
        background_max_ev,
    )
    smoothed, window = savgol_by_ev(corrected, high_energy_ev, smoothing_ev, polyorder)
    raw_fit = np.interp(queries, high_energy_ev, corrected)
    smooth_fit = np.interp(queries, high_energy_ev, smoothed)
    preedge = (fit_energy_ev >= fit_energy_ev.min()) & (
        fit_energy_ev <= fit_energy_ev.min() + PREEEDGE_BASELINE_WINDOW_EV
    )
    raw_fit = raw_fit - float(np.mean(raw_fit[preedge]))
    smooth_fit = smooth_fit - float(np.mean(smooth_fit[preedge]))
    edge = (fit_energy_ev >= fit_energy_ev.min() + EDGE_SIGNAL_START_OFFSET_EV) & (
        fit_energy_ev <= fit_energy_ev.max()
    )
    noise = float(np.std(raw_fit[preedge], ddof=1)) if preedge.sum() > 1 else 0.0
    signal = float(np.max(smooth_fit[edge])) if edge.any() else float(np.max(smooth_fit))
    snr = signal / noise if noise > 0 else float("inf")
    return smooth_fit, {
        "mean_high_loss": mean,
        "background_subtracted": corrected,
        "smoothed_high_loss": smoothed,
        "raw_fit": raw_fit,
        "smooth_fit": smooth_fit,
        "background": background,
        "savgol_window_points": int(window),
        "snr": float(snr),
        "energy_shift_ev": float(shift_ev),
        **background,
    }
