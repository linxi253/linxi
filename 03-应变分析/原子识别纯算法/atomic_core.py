# -*- coding: utf-8 -*-
"""原子识别与原子柱强度分析的纯算法核心。

本模块不导入 Tkinter 或 Matplotlib，便于单元测试和后续复用。坐标约定与
PPA 一致：点坐标为 ``(x, y)``，其中 x 是图像列、y 是图像行。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import curve_fit, linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.stats import rankdata


class AtomicToolError(ValueError):
    """可直接显示给用户的输入或算法错误。"""


def parse_finite_float(text: str, label: str) -> float:
    """把用户输入解析为有限浮点数；失败时抛出带字段名的中文错误。"""
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise AtomicToolError(f"{label}必须是数字（当前输入：{text!r}）。") from None
    if not np.isfinite(value):
        raise AtomicToolError(f"{label}必须是有限数。")
    return value


@dataclass(frozen=True)
class DetectionParams:
    sigma: float = 0.8
    min_distance: float = 8.0
    window: int = 5
    threshold: float | None = None
    bright: bool = True
    method: str = "com"

    def validated(self) -> "DetectionParams":
        if not np.isfinite(self.sigma) or self.sigma < 0:
            raise AtomicToolError("高斯滤波 sigma 必须是大于等于 0 的有限数。")
        if not np.isfinite(self.min_distance) or self.min_distance <= 0:
            raise AtomicToolError("最小原子间距必须是大于 0 的有限数。")
        if self.window < 3 or self.window % 2 == 0:
            raise AtomicToolError("质心窗口必须是大于等于 3 的奇数。")
        if self.method not in {"com", "gaussian"}:
            raise AtomicToolError("质心方法必须是 com 或 gaussian。")
        if self.threshold is not None and not np.isfinite(self.threshold):
            raise AtomicToolError("强度阈值必须留空或填写有限数。")
        if self.threshold is not None and not 0 < self.threshold < 1:
            # 阈值作用于 0–1 归一化检测图：0–1 之间一律按分位数解释。
            # 绝对阈值只对暗原子（检测图取负后值域 -1–0）有可表达的区间。
            if self.bright:
                raise AtomicToolError(
                    f"强度阈值 {self.threshold:g} 无效（亮原子）。阈值作用于 0–1 归一化检测图，"
                    "0–1 之间的小数一律按分位数解释（如 0.6 = P60），亮原子没有独立于分位数的绝对阈值区间。"
                    "请留空使用自适应 P80，或填写 0–1 之间的分位数（如 0.9）。"
                )
            if not -1 < self.threshold < 0:
                raise AtomicToolError(
                    f"强度阈值 {self.threshold:g} 无效（暗原子）。阈值作用于取负后的 -1–0 检测图："
                    "留空使用自适应 P80；填 0–1 之间的小数按分位数解释（如 0.6 = P60）；"
                    "或填 -1–0 之间的绝对值（如 -0.3）。"
                )
        return self


@dataclass(frozen=True)
class IntensityParams:
    aperture_radius: float = 2.5
    background_inner_radius: float = 3.0
    background_outer_radius: float = 4.0
    bright: bool = True

    def validated(self) -> "IntensityParams":
        radii = (
            self.aperture_radius,
            self.background_inner_radius,
            self.background_outer_radius,
        )
        if not all(np.isfinite(v) for v in radii):
            raise AtomicToolError("积分孔径和背景环半径必须是有限数。")
        if self.aperture_radius <= 0:
            raise AtomicToolError("积分孔径半径必须大于 0。")
        if self.background_inner_radius <= self.aperture_radius:
            raise AtomicToolError("背景环内半径必须大于积分孔径半径。")
        if self.background_outer_radius <= self.background_inner_radius:
            raise AtomicToolError("背景环外半径必须大于内半径。")
        return self


@dataclass(frozen=True)
class DetectionRegion:
    """自动识别候选区域；坐标点统一为图像 ``(x, y)``。"""

    kind: str
    vertices: tuple[tuple[float, float], ...]

    def validated(self) -> "DetectionRegion":
        if self.kind not in {"rectangle", "polygon"}:
            raise AtomicToolError("选区类型必须是 rectangle 或 polygon。")
        minimum = 2 if self.kind == "rectangle" else 3
        if len(self.vertices) < minimum:
            raise AtomicToolError("矩形选区需要两个角点，自由选区至少需要三个顶点。")
        points = np.asarray(self.vertices, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise AtomicToolError("选区坐标无效。")
        if self.kind == "rectangle":
            x0, y0 = points[0]
            x1, y1 = points[1]
            if abs(x1 - x0) < 1 or abs(y1 - y0) < 1:
                raise AtomicToolError("矩形选区宽度和高度必须至少为 1 px。")
        else:
            # 鞋带公式检查退化多边形。
            area2 = float(np.sum(points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1)))
            if abs(area2) < 1.0:
                raise AtomicToolError("自由选区面积过小或顶点共线。")
        return self


@dataclass
class AtomRecord:
    atom_id: int
    x: float
    y: float
    integrated_intensity: float = np.nan
    percentile: float = np.nan
    aperture_pixels: int = 0
    background_pixels: int = 0
    background_level: float = np.nan
    truncated: bool = False
    source: str = "auto"
    nearest_neighbor: float = np.inf
    neighbor_contaminated: bool = False

    def copy(self) -> "AtomRecord":
        return AtomRecord(
            int(self.atom_id),
            float(self.x),
            float(self.y),
            float(self.integrated_intensity),
            float(self.percentile),
            int(self.aperture_pixels),
            int(self.background_pixels),
            float(self.background_level),
            bool(self.truncated),
            str(self.source),
            float(self.nearest_neighbor),
            bool(self.neighbor_contaminated),
        )


@dataclass(frozen=True)
class MarkerRule:
    name: str
    low: float
    high: float
    color: str
    marker: str
    visible: bool = True

    def validated(self) -> "MarkerRule":
        if not 0 <= self.low < self.high <= 100:
            raise AtomicToolError("标记范围必须满足 0 ≤ 下限 < 上限 ≤ 100。")
        if self.marker not in {"circle", "square", "triangle", "diamond", "cross", "plus", "star"}:
            raise AtomicToolError("不支持的标记形状。")
        return self


@dataclass(frozen=True)
class TiffStackInfo:
    path: str
    axes: str
    source_shape: tuple[int, ...]
    stack_shape: tuple[int, int, int]
    dtype: str
    memory_mapped: bool
    source: np.ndarray | None = None


@dataclass(frozen=True)
class IntensitySample:
    """单个原子的积分强度及其 QC 诊断信息。

    ``nearest_neighbor`` 是同帧内到最近其他原子的距离（无邻居时为 inf）。
    ``neighbor_contaminated`` 表示该距离小于 ``孔径半径 + 背景环外半径``，
    即相邻原子的核心信号可能进入本原子的孔径或背景环，强度与背景估计
    都可能被抬高或压低，采信时应人工核查。
    """

    intensity: float
    aperture_pixels: int
    background_pixels: int
    background_level: float
    truncated: bool
    nearest_neighbor: float = np.inf
    neighbor_contaminated: bool = False


@dataclass(frozen=True)
class CalibrationResult:
    sample_count: int
    nearest_spacing: float
    median_fwhm: float
    peak_min: float
    peak_max: float
    detection_params: DetectionParams
    intensity_params: IntensityParams
    match_distance: float
    low_contrast: bool = False


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """将 (..., H, W, RGB/RGBA) 转为灰度并保持原始数值范围。"""
    rgb = np.asarray(rgb)
    if rgb.ndim < 3 or rgb.shape[-1] not in (3, 4):
        raise AtomicToolError(f"RGB 转灰度需要末轴为 3 或 4，实际形状为 {rgb.shape}。")
    source_dtype = rgb.dtype
    channels = rgb[..., :3].astype(np.float64)
    gray = channels @ np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)
    if np.issubdtype(source_dtype, np.integer):
        limits = np.iinfo(source_dtype)
        return np.clip(gray, limits.min, limits.max).astype(source_dtype)
    return gray.astype(source_dtype, copy=False)


def normalize_tiff_array(array: np.ndarray, axes: str | None = None) -> np.ndarray:
    """把常见 TIFF 布局规范为 ``(frame, y, x)`` 灰度堆栈。

    维度处理沿用“原子衬度统计”工具的保守策略：缺少 axes 元数据时，
    三维数组默认按灰度堆栈解释，避免把宽度恰好为 3/4 的窄堆栈误判为 RGB。
    """
    arr = np.asarray(array)
    axes_text = (axes or "").upper().strip()

    if arr.ndim == 2:
        return arr[np.newaxis, ...]

    if arr.ndim == 3:
        if axes_text and axes_text[-1:] in {"C", "S"} and arr.shape[-1] in (3, 4):
            return rgb_to_gray(arr)[np.newaxis, ...]
        if axes_text and axes_text[:1] in {"C", "S"} and arr.shape[0] in (3, 4):
            return rgb_to_gray(np.moveaxis(arr, 0, -1))[np.newaxis, ...]
        return arr

    if arr.ndim == 4:
        color_last = arr.shape[-1] in (3, 4) and (
            not axes_text or axes_text[-1:] in {"C", "S"}
        )
        if color_last:
            return rgb_to_gray(arr)
        if arr.shape[-1] == 1:
            return arr[..., 0]
        color_second = arr.shape[1] in (1, 3, 4) and (
            not axes_text or (len(axes_text) > 1 and axes_text[1] in {"C", "S"})
        )
        if color_second:
            if arr.shape[1] == 1:
                return arr[:, 0, ...]
            moved = np.moveaxis(arr[:, :3, ...], 1, -1)
            return rgb_to_gray(moved)
        raise AtomicToolError(
            f"无法识别四维 TIFF 形状 {arr.shape}；支持 (T,H,W,C) 或 (T,C,H,W)。"
        )

    raise AtomicToolError(
        f"不支持 {arr.ndim} 维 TIFF（形状 {arr.shape}）；仅支持单图或二维图像堆栈。"
    )


def load_tiff_stack(path: str | Path) -> tuple[np.ndarray, TiffStackInfo]:
    """优先内存映射读取 TIFF，失败时回退到普通读取。"""
    import tifffile

    resolved = str(Path(path).resolve())
    with tifffile.TiffFile(resolved) as tif:
        if not tif.series:
            raise AtomicToolError("TIFF 中没有可读取的图像序列。")
        axes = tif.series[0].axes or ""
        source_shape = tuple(int(v) for v in tif.series[0].shape)

    mapped = False
    try:
        source = tifffile.memmap(resolved, series=0)
        mapped = True
    except Exception:
        try:
            source = tifffile.imread(resolved, series=0)
        except Exception as error:
            raise AtomicToolError(f"无法读取 TIFF：{error}") from error

    try:
        stack = normalize_tiff_array(source, axes=axes)
        if stack.ndim != 3 or stack.shape[0] < 1 or stack.shape[1] < 1 or stack.shape[2] < 1:
            raise AtomicToolError(f"规范化后的 TIFF 形状无效：{stack.shape}。")
        if not np.issubdtype(stack.dtype, np.number):
            raise AtomicToolError(f"不支持 TIFF 像素类型 {stack.dtype}。")
    except Exception:
        close_tiff_stack(source)
        raise

    info = TiffStackInfo(
        path=resolved,
        axes=axes,
        source_shape=source_shape,
        stack_shape=tuple(int(v) for v in stack.shape),
        dtype=str(stack.dtype),
        memory_mapped=mapped,
        source=source,
    )
    return stack, info


def close_tiff_stack(stack: np.ndarray | None) -> None:
    """释放 TIFF 内存映射文件句柄；普通 ndarray 调用无副作用。"""
    current = stack
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, np.memmap):
            mmap_object = getattr(current, "_mmap", None)
            if mmap_object is not None:
                try:
                    mmap_object.close()
                except Exception:
                    pass
            return
        current = getattr(current, "base", None)


def normalize_for_display(frame: np.ndarray) -> np.ndarray:
    """以 1%–99% 分位数把单帧规范为 [0, 1]，仅供显示与检测。"""
    image = np.asarray(frame, dtype=np.float64)
    if image.ndim != 2:
        raise AtomicToolError(f"检测帧必须是二维数组，实际形状为 {image.shape}。")
    finite = np.isfinite(image)
    if not finite.any():
        raise AtomicToolError("图像不包含有限像素。")
    valid = image[finite]
    low, high = np.percentile(valid, (1, 99))
    if high <= low:
        result = np.zeros_like(image, dtype=np.float64)
    else:
        result = np.clip((image - low) / (high - low), 0.0, 1.0)
    result[~finite] = 0.0
    return result


def greedy_nms(coords: np.ndarray, min_distance: float) -> np.ndarray:
    """按输入顺序执行基于 KD-tree 的贪心非极大值抑制。"""
    points = np.asarray(coords, dtype=np.float64)
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)
    tree = cKDTree(points)
    neighbors: dict[int, list[int]] = {}
    for left, right in tree.query_pairs(float(min_distance), output_type="ndarray"):
        neighbors.setdefault(int(left), []).append(int(right))
        neighbors.setdefault(int(right), []).append(int(left))
    selected: list[int] = []
    suppressed = np.zeros(len(points), dtype=bool)
    for index in range(len(points)):
        if suppressed[index]:
            continue
        selected.append(index)
        for neighbor in neighbors.get(index, ()):
            suppressed[neighbor] = True
    return points[np.asarray(selected, dtype=int)]


def _refine_centroids_batch(image: np.ndarray, coords_rc: np.ndarray, window: int) -> np.ndarray:
    """PPA 同口径的两次迭代批量质心定位，返回 (x, y)。"""
    count = len(coords_rc)
    if count == 0:
        return np.empty((0, 2), dtype=np.float64)
    half = window // 2
    height, width = image.shape
    padded = np.pad(image, half, mode="reflect")
    ys_grid, xs_grid = np.mgrid[0:window, 0:window]
    centers_r = np.clip(np.rint(coords_rc[:, 0]).astype(int), 0, height - 1)
    centers_c = np.clip(np.rint(coords_rc[:, 1]).astype(int), 0, width - 1)
    result_x = centers_c.astype(np.float64)
    result_y = centers_r.astype(np.float64)

    for _ in range(2):
        windows = sliding_window_view(padded, (window, window))[centers_r, centers_c]
        background = np.percentile(windows.reshape(count, -1), 5, axis=1)
        signal = np.maximum(windows - background[:, None, None], 0.0)
        totals = signal.sum(axis=(1, 2))
        weighted_x = np.einsum("nij,ij->n", signal, xs_grid)
        weighted_y = np.einsum("nij,ij->n", signal, ys_grid)
        offsets_x = np.zeros(count, dtype=np.float64)
        offsets_y = np.zeros(count, dtype=np.float64)
        np.divide(weighted_x, totals, out=offsets_x, where=totals > 0)
        np.divide(weighted_y, totals, out=offsets_y, where=totals > 0)
        result_x = centers_c - half + offsets_x
        result_y = centers_r - half + offsets_y
        result_x[totals <= 0] = centers_c[totals <= 0]
        result_y[totals <= 0] = centers_r[totals <= 0]
        centers_c = np.clip(np.rint(result_x).astype(int), 0, width - 1)
        centers_r = np.clip(np.rint(result_y).astype(int), 0, height - 1)
    return np.column_stack((result_x, result_y))


def _gaussian_refine(image: np.ndarray, x: float, y: float, window: int) -> tuple[float, float, bool]:
    """二维高斯精炼单个位置； fitted=False 表示拟合未成功，由调用方批量回退。"""
    half = window // 2
    height, width = image.shape
    xi, yi = int(round(x)), int(round(y))
    y0, y1 = max(0, yi - half), min(height, yi + half + 1)
    x0, x1 = max(0, xi - half), min(width, xi + half + 1)
    if y1 - y0 < 3 or x1 - x0 < 3:
        return float(x), float(y), False
    roi = image[y0:y1, x0:x1].astype(np.float64)
    roi = np.maximum(roi - np.percentile(roi, 5), 0.0)
    if roi.max(initial=0) <= 0:
        return float(x), float(y), False
    ys, xs = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]

    def model(xy, xo, yo, sx, sy, amplitude, offset):
        xv, yv = xy
        return amplitude * np.exp(
            -((xv - xo) ** 2 / (2 * sx**2) + (yv - yo) ** 2 / (2 * sy**2))
        ) + offset

    peak_r, peak_c = np.unravel_index(np.argmax(roi), roi.shape)
    initial = [float(peak_c), float(peak_r), 1.0, 1.0, float(roi.max()), 0.0]
    bounds = (
        [0, 0, 0.3, 0.3, 0, -np.inf],
        [roi.shape[1] - 1, roi.shape[0] - 1, 5, 5, np.inf, np.inf],
    )
    try:
        fitted, _ = curve_fit(
            model,
            np.vstack((xs.ravel(), ys.ravel())),
            roi.ravel(),
            p0=initial,
            bounds=bounds,
            maxfev=500,
        )
        cx, cy = x0 + fitted[0], y0 + fitted[1]
        if abs(cx - x) <= half and abs(cy - y) <= half:
            return float(cx), float(cy), True
    except Exception:
        pass
    return float(x), float(y), False


def refine_point_on_work(work: np.ndarray, x: float, y: float, params: DetectionParams) -> tuple[float, float]:
    """在已完成归一化与极性处理的检测图上精炼单个位置。

    只裁剪点击位置周围的局部窗口做 COM 精炼，避免每次点击对整幅图
    做反射填充；结果与整图精炼一致（两次 COM 迭代的位移远小于窗口）。
    """
    params = params.validated()
    if params.method == "gaussian":
        fitted_x, fitted_y, fitted = _gaussian_refine(work, x, y, params.window)
        if fitted:
            return fitted_x, fitted_y
    height, width = work.shape
    margin = params.window * 2
    x0 = max(0, int(np.floor(x)) - margin)
    x1 = min(width, int(np.floor(x)) + margin + 1)
    y0 = max(0, int(np.floor(y)) - margin)
    y1 = min(height, int(np.floor(y)) + margin + 1)
    if x1 - x0 < params.window or y1 - y0 < params.window:
        # 图像本身小于精炼窗口时退回整图批量精炼（内部会 pad）。
        refined = _refine_centroids_batch(work, np.array([[y, x]], dtype=np.float64), params.window)
        return float(refined[0, 0]), float(refined[0, 1])
    crop = np.asarray(work[y0:y1, x0:x1])
    refined = _refine_centroids_batch(crop, np.array([[y - y0, x - x0]], dtype=np.float64), params.window)
    return float(refined[0, 0]) + x0, float(refined[0, 1]) + y0


def refine_clicked_point(frame: np.ndarray, x: float, y: float, params: DetectionParams) -> tuple[float, float]:
    """按当前检测极性和精炼方法修正一个人工点击位置。"""
    params = params.validated()
    work = normalize_for_display(frame)
    if not params.bright:
        work = -work
    return refine_point_on_work(work, x, y, params)


def points_in_detection_region(points_xy: np.ndarray, region: DetectionRegion | None) -> np.ndarray:
    """返回各坐标是否位于矩形或自由多边形选区内。"""
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if region is None:
        return np.ones(len(points), dtype=bool)
    region = region.validated()
    vertices = np.asarray(region.vertices, dtype=np.float64)
    if region.kind == "rectangle":
        x0, x1 = sorted((float(vertices[0, 0]), float(vertices[1, 0])))
        y0, y1 = sorted((float(vertices[0, 1]), float(vertices[1, 1])))
        return (
            (points[:, 0] >= x0)
            & (points[:, 0] <= x1)
            & (points[:, 1] >= y0)
            & (points[:, 1] <= y1)
        )

    # 向量化射线法；边界上的候选通过独立线段距离判断纳入。
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    on_boundary = np.zeros(len(points), dtype=bool)
    for index in range(len(vertices)):
        x1, y1 = vertices[index]
        x2, y2 = vertices[(index + 1) % len(vertices)]
        crosses = (y1 > y) != (y2 > y)
        denominator = y2 - y1
        if abs(denominator) > 1e-15:
            x_intersection = (x2 - x1) * (y - y1) / denominator + x1
            inside ^= crosses & (x < x_intersection)
        segment = np.array([x2 - x1, y2 - y1], dtype=np.float64)
        length2 = float(np.dot(segment, segment))
        if length2 > 0:
            t = np.clip(((x - x1) * segment[0] + (y - y1) * segment[1]) / length2, 0.0, 1.0)
            projection_x = x1 + t * segment[0]
            projection_y = y1 + t * segment[1]
            on_boundary |= np.hypot(x - projection_x, y - projection_y) <= 1e-9
    return inside | on_boundary


def detect_atoms(
    frame: np.ndarray,
    params: DetectionParams,
    region: DetectionRegion | None = None,
) -> np.ndarray:
    """执行 PPA 同源的传统原子检测，返回亚像素 ``(x, y)`` 数组。"""
    params = params.validated()
    work = normalize_for_display(frame)
    if not params.bright:
        work = -work
    filtered = gaussian_filter(work, sigma=params.sigma) if params.sigma > 0 else work
    if params.threshold is None:
        threshold_abs = float(np.percentile(filtered, 80))
    elif 0 < params.threshold < 1:
        threshold_abs = float(np.percentile(filtered, params.threshold * 100))
    else:
        threshold_abs = float(params.threshold)

    local_max = maximum_filter(filtered, size=3)
    mask = (filtered >= local_max - 1e-12) & (filtered > threshold_abs)
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return np.empty((0, 2), dtype=np.float64)
    strengths = filtered[mask]
    coords = coords[np.argsort(-strengths, kind="stable")]
    if region is not None:
        keep = points_in_detection_region(coords[:, [1, 0]], region)
        coords = coords[keep]
        if len(coords) == 0:
            return np.empty((0, 2), dtype=np.float64)
    selected = greedy_nms(coords, params.min_distance)
    if len(selected) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if params.method == "com":
        return _refine_centroids_batch(work, selected, params.window)
    refined = np.empty((len(selected), 2), dtype=np.float64)
    failed_indices: list[int] = []
    for index, (row, col) in enumerate(selected):
        fitted_x, fitted_y, fitted = _gaussian_refine(work, float(col), float(row), params.window)
        refined[index] = (fitted_x, fitted_y)
        if not fitted:
            failed_indices.append(index)
    if failed_indices:
        # 拟合失败的点最后一次性批量 COM 精炼，避免逐点对整幅图 pad。
        failed = np.asarray(failed_indices, dtype=int)
        refined[failed] = _refine_centroids_batch(work, selected[failed], params.window)
    return refined


def calibrate_from_points(
    frame: np.ndarray,
    clicked_points: Sequence[tuple[float, float]] | np.ndarray,
    bright: bool = True,
) -> CalibrationResult:
    """根据 2–4 个相邻原子估计 PPA 检测参数和积分孔径。"""
    points = np.asarray(clicked_points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2:
        raise AtomicToolError("校准至少需要 2 个原子点。")
    if not np.isfinite(points).all():
        raise AtomicToolError("校准点包含无效坐标。")
    pairwise = []
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            distance = float(np.linalg.norm(points[left] - points[right]))
            if distance > 0:
                pairwise.append(distance)
    if not pairwise:
        raise AtomicToolError("校准点不能重合。")
    nearest_spacing = float(np.min(pairwise))
    suggested_min_distance = max(2.0, round(nearest_spacing * 0.75, 1))

    image = normalize_for_display(frame)
    signal_image = image if bright else 1.0 - image
    height, width = image.shape
    roi_radius = max(3, int(nearest_spacing * 0.8))
    fwhms: list[float] = []
    peaks: list[float] = []
    backgrounds: list[float] = []

    for x, y in points:
        xi, yi = int(round(x)), int(round(y))
        y0, y1 = max(0, yi - roi_radius), min(height, yi + roi_radius + 1)
        x0, x1 = max(0, xi - roi_radius), min(width, xi + roi_radius + 1)
        roi = signal_image[y0:y1, x0:x1].astype(np.float64, copy=True)
        if roi.size < 9:
            continue
        background = float(np.percentile(roi, 10))
        roi_sub = np.maximum(roi - background, 0.0)
        if roi_sub.max(initial=0) <= 0:
            continue
        peak_r, peak_c = np.unravel_index(np.argmax(roi_sub), roi_sub.shape)
        yy, xx = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
        radii = np.hypot(xx - peak_c, yy - peak_r)
        max_radius = int(
            min(
                peak_c,
                peak_r,
                roi.shape[1] - 1 - peak_c,
                roi.shape[0] - 1 - peak_r,
                roi_radius,
            )
        ) + 1
        if max_radius < 2:
            # ROI 贴边时无法可靠估计 FWHM，峰值与背景也不参与统计。
            continue
        peaks.append(float(roi_sub.max()))
        backgrounds.append(background)
        radial_mean = np.array(
            [roi_sub[(radii >= r) & (radii < r + 1)].mean() for r in range(max_radius)]
        )
        peak_value = float(radial_mean[: max(1, max_radius // 3)].max())
        above = np.flatnonzero(radial_mean > peak_value / 2)
        fwhm = 2.0 * (above[-1] + 0.5) if len(above) else 3.0
        fwhms.append(float(np.clip(fwhm, 1.5, roi_radius * 2.0)))

    median_fwhm = float(np.median(fwhms)) if fwhms else 4.0
    suggested_sigma = float(np.clip(round(median_fwhm / 2.5, 1), 0.1, 5.0))
    window_raw = max(3, int(median_fwhm * 1.5))
    suggested_window = window_raw if window_raw % 2 else window_raw + 1
    suggested_window = min(suggested_window, 11)

    suggested_threshold: float | None = None
    if peaks:
        dimmest_signal = float(np.median(backgrounds) + min(peaks) * 0.6)
        percentile = np.searchsorted(np.sort(signal_image.ravel()), dimmest_signal) / signal_image.size
        suggested_threshold = round(float(np.clip(percentile * 0.85, 0.05, 0.75)), 2)

    noise_span = float(np.percentile(signal_image, 30) - np.percentile(signal_image, 10))
    peak_span = float(np.percentile(signal_image, 98) - np.percentile(signal_image, 10))
    method = "gaussian" if median_fwhm > 5 and peaks and noise_span < peak_span * 0.3 else "com"
    low_contrast = bool(peaks) and noise_span > 0 and min(peaks) < 2.0 * noise_span

    # 圆孔径保持在最近邻距离一半以内，背景环放在孔径与相邻原子之间。
    aperture = float(np.clip(median_fwhm * 0.70, 1.25, max(1.3, nearest_spacing * 0.30)))
    background_inner = max(aperture + 0.5, nearest_spacing * 0.34)
    background_outer = max(background_inner + 0.5, nearest_spacing * 0.48)

    return CalibrationResult(
        # sample_count 是实际参与 FWHM/峰值统计的样本数；贴边等无效 ROI 已被剔除。
        sample_count=len(peaks),
        nearest_spacing=nearest_spacing,
        median_fwhm=median_fwhm,
        peak_min=float(min(peaks)) if peaks else np.nan,
        peak_max=float(max(peaks)) if peaks else np.nan,
        detection_params=DetectionParams(
            sigma=suggested_sigma,
            min_distance=suggested_min_distance,
            window=suggested_window,
            threshold=suggested_threshold,
            bright=bool(bright),
            method=method,
        ),
        intensity_params=IntensityParams(
            aperture_radius=round(aperture, 2),
            background_inner_radius=round(background_inner, 2),
            background_outer_radius=round(background_outer, 2),
            bright=bool(bright),
        ),
        match_distance=round(max(1.0, nearest_spacing * 0.45), 2),
    )


def measure_intensities_with_diagnostics(
    frame: np.ndarray,
    points_xy: np.ndarray | Sequence[tuple[float, float]],
    params: IntensityParams,
) -> list[IntensitySample]:
    """逐原子返回积分强度与 QC 诊断。

    背景取同心圆环内像素的中位数。亮原子积分 ``I-bg``，暗原子积分
    ``bg-I``；不对逐像素差值截零，避免给噪声引入系统性正偏差。

    ``truncated`` 表示理想孔径/背景环被图像边界裁剪或范围内含非有限
    像素：此时强度系统性偏低。当前口径下截断原子仍参与单帧百分位
    排名，调用方应通过该标志在界面与导出中明确提示。
    ``nearest_neighbor``/``neighbor_contaminated`` 标记相邻原子对孔径
    或背景环的潜在污染，判定标准为最近邻距离 < 孔径半径 + 背景环外
    半径。
    """
    params = params.validated()
    image = np.asarray(frame, dtype=np.float64)
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    height, width = image.shape
    outer = float(params.background_outer_radius)
    sign = 1.0 if params.bright else -1.0

    neighbor_distance = np.full(len(points), np.inf, dtype=np.float64)
    finite_points = np.isfinite(points).all(axis=1)
    if int(finite_points.sum()) >= 2:
        tree = cKDTree(points[finite_points])
        distances, _ = tree.query(points[finite_points], k=2)
        neighbor_distance[finite_points] = distances[:, 1]

    contamination_radius = outer + float(params.aperture_radius)
    invalid = IntensitySample(np.nan, 0, 0, np.nan, False)
    samples: list[IntensitySample] = []
    for index, (x, y) in enumerate(points):
        nearest = float(neighbor_distance[index])
        contaminated = bool(nearest < contamination_radius)
        if not np.isfinite(x) or not np.isfinite(y):
            samples.append(replace(invalid, nearest_neighbor=nearest, neighbor_contaminated=contaminated))
            continue
        x0 = max(0, int(np.floor(x - outer)))
        x1 = min(width, int(np.ceil(x + outer)) + 1)
        y0 = max(0, int(np.floor(y - outer)))
        y1 = min(height, int(np.ceil(y + outer)) + 1)
        if x1 <= x0 or y1 <= y0:
            samples.append(replace(invalid, nearest_neighbor=nearest, neighbor_contaminated=contaminated))
            continue
        patch = image[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        distance = np.hypot(xx - x, yy - y)
        aperture = (distance <= params.aperture_radius) & np.isfinite(patch)
        background_ring = (
            (distance >= params.background_inner_radius)
            & (distance <= params.background_outer_radius)
            & np.isfinite(patch)
        )
        if not aperture.any() or background_ring.sum() < 3:
            samples.append(replace(invalid, nearest_neighbor=nearest, neighbor_contaminated=contaminated))
            continue
        background = float(np.median(patch[background_ring]))
        # 理想圆盘（半径 outer，像素中心 0..width-1）越出图像边界即视为截断；
        # 左右上下四条边使用同一像素中心标准。
        box_clipped = (
            x - outer < 0.0
            or y - outer < 0.0
            or x + outer > width - 1.0
            or y + outer > height - 1.0
        )
        disk = distance <= outer
        non_finite_inside = not bool(np.isfinite(patch[disk]).all()) if disk.any() else False
        samples.append(
            IntensitySample(
                intensity=float(np.sum(sign * (patch[aperture] - background))),
                aperture_pixels=int(aperture.sum()),
                background_pixels=int(background_ring.sum()),
                background_level=background,
                truncated=bool(box_clipped or non_finite_inside),
                nearest_neighbor=nearest,
                neighbor_contaminated=contaminated,
            )
        )
    return samples


def measure_integrated_intensities(
    frame: np.ndarray,
    points_xy: np.ndarray | Sequence[tuple[float, float]],
    params: IntensityParams,
) -> np.ndarray:
    """测量各原子的局部背景扣除圆孔径积分强度（仅强度值，诊断见上）。"""
    return np.array(
        [sample.intensity for sample in measure_intensities_with_diagnostics(frame, points_xy, params)],
        dtype=np.float64,
    )


def percentile_ranks(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """计算单帧百分位秩；最弱接近 0%，最强接近 100%。"""
    array = np.asarray(values, dtype=np.float64)
    result = np.full(array.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(array)
    count = int(finite.sum())
    if count == 0:
        return result
    ranks = rankdata(array[finite], method="average")
    result[finite] = (ranks - 0.5) / count * 100.0
    return result


def make_records(
    frame: np.ndarray,
    points_xy: np.ndarray,
    atom_ids: Sequence[int],
    intensity_params: IntensityParams,
) -> list[AtomRecord]:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    ids = np.asarray(atom_ids, dtype=int)
    if len(points) != len(ids):
        raise AtomicToolError("点坐标数量与原子 ID 数量不一致。")
    samples = measure_intensities_with_diagnostics(frame, points, intensity_params)
    intensities = np.array([sample.intensity for sample in samples], dtype=np.float64)
    percentiles = percentile_ranks(intensities)
    return [
        AtomRecord(
            int(atom_id),
            float(x),
            float(y),
            float(value),
            float(percentile),
            aperture_pixels=sample.aperture_pixels,
            background_pixels=sample.background_pixels,
            background_level=sample.background_level,
            truncated=sample.truncated,
            nearest_neighbor=sample.nearest_neighbor,
            neighbor_contaminated=sample.neighbor_contaminated,
        )
        for (x, y), atom_id, value, percentile, sample in zip(points, ids, intensities, percentiles, samples)
    ]


def recalculate_record_intensities(
    frame: np.ndarray,
    records: Sequence[AtomRecord],
    intensity_params: IntensityParams,
) -> list[AtomRecord]:
    if not records:
        return []
    points = np.array([(record.x, record.y) for record in records], dtype=np.float64)
    ids = [record.atom_id for record in records]
    updated = make_records(frame, points, ids, intensity_params)
    for new_record, old_record in zip(updated, records):
        new_record.source = old_record.source
    return updated


def _spatial_order(points: np.ndarray, indices: Iterable[int]) -> list[int]:
    return sorted(indices, key=lambda idx: (float(points[idx, 1]), float(points[idx, 0]), int(idx)))


def _gated_assignment(
    catalog_xy: np.ndarray,
    points_xy: np.ndarray,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """门控一对一最小距离匹配（匈牙利算法），返回接受的 (catalog行, points列)。

    超出 max_distance 的组合以极大代价参与分配但会被二次拒绝，因此
    惩罚代价不可能挤掉任何合法匹配。
    """
    gate_penalty = max_distance * 1_000_000.0
    distances = np.linalg.norm(catalog_xy[:, None, :] - points_xy[None, :, :], axis=2)
    gated_cost = np.where(distances <= max_distance, distances, gate_penalty)
    rows, cols = linear_sum_assignment(gated_cost)
    accepted = distances[rows, cols] <= max_distance
    return rows[accepted], cols[accepted]


def link_frame_points(
    frames_points: Sequence[np.ndarray],
    max_distance: float,
    max_missed: int = 8,
) -> list[np.ndarray]:
    """对已漂移校正堆栈做全局轨迹目录的逐帧一对一 ID 关联。

    轨迹在漏检后 ``max_missed`` 帧内重新出现仍恢复原 ID；连续超过
    ``max_missed`` 帧未匹配的轨迹从目录剔除，防止误检噪声点无限累积
    （陈旧轨迹会让匈牙利分配的代价矩阵逐帧膨胀，并稀释 ID 语义）。
    """
    if not np.isfinite(max_distance) or max_distance <= 0:
        raise AtomicToolError("跨帧 ID 关联距离必须大于 0。")
    if int(max_missed) < 1:
        raise AtomicToolError("轨迹老化帧数必须大于等于 1。")
    linked: list[np.ndarray] = []
    track_positions: dict[int, np.ndarray] = {}
    track_last_seen: dict[int, int] = {}
    next_id = 1

    for frame_index, raw_points in enumerate(frames_points):
        points = np.asarray(raw_points, dtype=np.float64).reshape(-1, 2)
        stale = [
            atom_id
            for atom_id, last_seen in track_last_seen.items()
            if frame_index - last_seen > int(max_missed)
        ]
        for atom_id in stale:
            track_positions.pop(atom_id, None)
            track_last_seen.pop(atom_id, None)
        ids = np.full(len(points), -1, dtype=int)
        if len(points) and track_positions:
            track_ids = np.array(sorted(track_positions), dtype=int)
            catalog = np.vstack([track_positions[int(atom_id)] for atom_id in track_ids])
            rows, cols = _gated_assignment(catalog, points, max_distance)
            for row, col in zip(rows, cols):
                ids[col] = int(track_ids[row])

        for point_index in _spatial_order(points, np.flatnonzero(ids < 0)):
            ids[point_index] = next_id
            next_id += 1
        for atom_id, point in zip(ids, points):
            track_positions[int(atom_id)] = point.copy()
            track_last_seen[int(atom_id)] = frame_index
        linked.append(ids)
    return linked


def build_track_catalog(
    frames_records: Sequence[Sequence[AtomRecord]],
    exclude_frame: int | None = None,
) -> dict[int, np.ndarray]:
    grouped: dict[int, list[tuple[float, float]]] = {}
    for frame_index, records in enumerate(frames_records):
        if exclude_frame is not None and frame_index == exclude_frame:
            continue
        for record in records:
            grouped.setdefault(int(record.atom_id), []).append((record.x, record.y))
    return {
        atom_id: np.median(np.asarray(positions, dtype=np.float64), axis=0)
        for atom_id, positions in grouped.items()
    }


def match_points_to_catalog(
    points_xy: np.ndarray,
    catalog: Mapping[int, np.ndarray],
    max_distance: float,
    next_id_start: int,
) -> np.ndarray:
    """把重检或人工添加的单帧点与其他帧的已有 ID 目录关联。"""
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    result = np.full(len(points), -1, dtype=int)
    if len(points) and catalog:
        catalog_ids = np.array(sorted(catalog), dtype=int)
        catalog_points = np.vstack([np.asarray(catalog[int(atom_id)], dtype=float) for atom_id in catalog_ids])
        rows, cols = _gated_assignment(catalog_points, points, max_distance)
        for row, col in zip(rows, cols):
            result[col] = int(catalog_ids[row])
    next_id = max(1, int(next_id_start))
    for point_index in _spatial_order(points, np.flatnonzero(result < 0)):
        result[point_index] = next_id
        next_id += 1
    return result


def rule_for_percentile(percentile: float, rules: Sequence[MarkerRule]) -> MarkerRule | None:
    """按列表顺序返回第一个命中的可见范围规则；规则须已通过 validated() 校验。"""
    if not np.isfinite(percentile):
        return None
    for rule in rules:
        if not rule.visible:
            continue
        upper_match = percentile <= rule.high if rule.high == 100 else percentile < rule.high
        if percentile >= rule.low and upper_match:
            return rule
    return None


def visible_records(records: Sequence[AtomRecord], rules: Sequence[MarkerRule]) -> list[tuple[AtomRecord, MarkerRule]]:
    for rule in rules:
        rule.validated()
    visible: list[tuple[AtomRecord, MarkerRule]] = []
    for record in records:
        rule = rule_for_percentile(record.percentile, rules)
        if rule is not None:
            visible.append((record, rule))
    return visible


def _regular_polygon(cx: float, cy: float, radius: float, sides: int, angle: float) -> list[tuple[float, float]]:
    return [
        (
            cx + radius * np.cos(angle + 2 * np.pi * i / sides),
            cy + radius * np.sin(angle + 2 * np.pi * i / sides),
        )
        for i in range(sides)
    ]


def render_marked_frame(
    frame: np.ndarray,
    records: Sequence[AtomRecord],
    rules: Sequence[MarkerRule],
    marker_radius: float = 6.0,
    show_ids: bool = False,
) -> np.ndarray:
    """渲染无坐标轴的 8 位 RGB 标记帧，供 PNG/TIFF 导出。"""
    from PIL import Image, ImageColor, ImageDraw, ImageFont

    gray = np.rint(normalize_for_display(frame) * 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    canvas = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    radius = max(2.0, float(marker_radius))
    line_width = max(1, int(round(radius / 3)))
    halo_width = line_width + 2
    font = ImageFont.load_default()

    def draw_shape(record: AtomRecord, marker: str, color: str) -> None:
        x, y = float(record.x), float(record.y)
        try:
            marker_color = ImageColor.getrgb(color)
        except ValueError:
            marker_color = (0, 255, 0)
        halo = (0, 0, 0)
        box = (x - radius, y - radius, x + radius, y + radius)

        if marker == "circle":
            draw.ellipse(box, outline=halo, width=halo_width)
            draw.ellipse(box, outline=marker_color, width=line_width)
        elif marker == "square":
            draw.rectangle(box, outline=halo, width=halo_width)
            draw.rectangle(box, outline=marker_color, width=line_width)
        elif marker == "triangle":
            points = _regular_polygon(x, y, radius * 1.15, 3, -np.pi / 2)
            draw.line(points + [points[0]], fill=halo, width=halo_width, joint="curve")
            draw.line(points + [points[0]], fill=marker_color, width=line_width, joint="curve")
        elif marker == "diamond":
            points = [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)]
            draw.line(points + [points[0]], fill=halo, width=halo_width, joint="curve")
            draw.line(points + [points[0]], fill=marker_color, width=line_width, joint="curve")
        elif marker == "star":
            points: list[tuple[float, float]] = []
            for idx in range(10):
                current_radius = radius if idx % 2 == 0 else radius * 0.42
                angle = -np.pi / 2 + idx * np.pi / 5
                points.append((x + current_radius * np.cos(angle), y + current_radius * np.sin(angle)))
            draw.line(points + [points[0]], fill=halo, width=halo_width, joint="curve")
            draw.line(points + [points[0]], fill=marker_color, width=line_width, joint="curve")
        else:
            segments = []
            if marker in {"cross", "plus"}:
                if marker == "cross":
                    segments = [
                        (x - radius, y - radius, x + radius, y + radius),
                        (x - radius, y + radius, x + radius, y - radius),
                    ]
                else:
                    segments = [
                        (x - radius, y, x + radius, y),
                        (x, y - radius, x, y + radius),
                    ]
            for segment in segments:
                draw.line(segment, fill=halo, width=halo_width)
                draw.line(segment, fill=marker_color, width=line_width)

        if show_ids:
            label = str(record.atom_id)
            position = (x + radius + 2, y - radius - 2)
            draw.text(position, label, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))

    for record, rule in visible_records(records, rules):
        draw_shape(record, rule.marker, rule.color)
    return np.asarray(canvas)
