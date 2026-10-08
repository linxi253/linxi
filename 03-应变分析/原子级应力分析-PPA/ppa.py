"""
原子位移与应变分析工具 (PPA) v3
================================
功能:
  - 自动原子识别 + 亚像素质心定位 (COM / 2D高斯拟合)
  - 手动添加/删除原子标记
  - 参考晶格矢量设置（三点 / 双原子列多点拟合 / 手动输入）
  - 局部 Peak Pairs 与兼容晶格-CST 位移/应变分析
  - 可视化：原子位移幅值染色、应变分布云图、位移矢量场、三角剖分应变
  - 导出：原子坐标CSV、标记图、应变图、位移数据CSV、应变张量CSV

算法:
  原子检测: 高斯滤波 → 局部最大值 → 亚像素质心修正
  PPA: 局部正/负 peak pairs → 局部变形梯度
  Legacy CST: 参考构型 Delaunay 三角剖分 → 线性位移梯度 → 应变张量

依赖: numpy, scipy, matplotlib, tifffile, tkinter
"""
import sys

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, simpledialog
except ModuleNotFoundError:
    # 无 Tk 环境（如最小测试容器）仍可导入 ppa 中的纯数值函数；
    # GUI 入口/实例化会在缺 tk 时失败，这是预期行为。
    tk = None
    ttk = filedialog = messagebox = simpledialog = None
import numpy as np
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# ---- 中文字体配置 (自动检测可用字体) ----
import matplotlib.font_manager as fm
_cjk_candidates = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong', 'SimSun',
                   'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Arial Unicode MS']


def _find_cjk_font():
    available = {f.name for f in fm.fontManager.ttflist}
    for _font in _cjk_candidates:
        if _font in available:
            return _font, available
    return None, available


# 正常路径直接读字体缓存; 仅当一个中文字体都找不到时才重建一次 (缓存损坏/过期自愈)。
# 此前无条件 _load_fontmanager(try_read_cache=False) 会在每次启动时全量扫描
# 系统字体, 使打包 exe 的启动耗时长达数分钟。
_cjk_found, _available_fonts = _find_cjk_font()
if _cjk_found is None:
    # 使用私有 API 重建字体缓存时做防御：matplotlib 升级后该方法可能改名/移除。
    try:
        _rebuild_fonts = getattr(fm, "_load_fontmanager", None)
        if _rebuild_fonts is not None:
            _rebuild_fonts(try_read_cache=False)
            _cjk_found, _available_fonts = _find_cjk_font()
    except Exception:
        pass
if _cjk_found:
    matplotlib.rcParams['font.sans-serif'] = [_cjk_found, 'DejaVu Sans']
else:
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
except Exception:
    FigureCanvasTkAgg = NavigationToolbar2Tk = None
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import EllipseCollection
from matplotlib.colors import to_rgba
import matplotlib.patheffects as pe
from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.interpolate import griddata, CloughTocher2DInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
import csv
import logging
import os
import sys
import queue
import threading
import time
from dataclasses import dataclass
from numpy.lib.stride_tricks import sliding_window_view
from ppa_core import (
    AnalysisError,
    ImageLoadError,
    LATTICE_CONDITION_GUIDANCE,
    ProjectValidationError,
    compute_cst_strain,
    compute_local_peak_pair_strain,
    gaussian_refine_point,
    load_analysis_image,
    load_project as load_versioned_project,
    save_project as save_versioned_project,
    validate_max_condition,
    validate_reference_lattice,
)

LOCAL_PPA_ALGORITHM_ID = "peak-pairs-local-all-points-v2"

logger = logging.getLogger(__name__)

# 导入 butter.py 的滤波功能 (预处理用)。
# 优先使用环境变量 PPA_HRTEM_FILTER_DIR 指定的目录，其次回退到历史兄弟目录；
# 都不可用时降级为禁用，并在预处理面板中给出原因。
_HRTEM_IMPORT_NOTE = ""


def _find_hrtem_filter_dir():
    env_dir = os.environ.get("PPA_HRTEM_FILTER_DIR", "").strip()
    if env_dir:
        return os.path.normpath(env_dir)
    sibling = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', '02-图像处理', 'hrtem-HRTEM滤波工具'))
    return sibling if os.path.isdir(sibling) else None


HRTEMFilter = None
_hrtem_dir = _find_hrtem_filter_dir()
if _hrtem_dir is not None:
    sys.path.insert(0, _hrtem_dir)
    try:
        from butter import HRTEMFilter
    except (ImportError, ModuleNotFoundError) as _hrtem_exc:
        _HRTEM_IMPORT_NOTE = f"butter.py 导入失败: {_hrtem_exc}"
    finally:
        if _hrtem_dir in sys.path:
            sys.path.remove(_hrtem_dir)
else:
    _HRTEM_IMPORT_NOTE = ("未找到 butter.py（预处理扩展）。可设置环境变量 "
                          "PPA_HRTEM_FILTER_DIR 指向包含 butter.py 的目录。")


# ================================================================
#  纯计算辅助函数 (与 GUI 解耦, 可独立测试)
# ================================================================

def greedy_nms(coords, min_distance):
    """按给定顺序贪心非极大值抑制。

    ``coords`` 应已按强度降序排列；每个被选中的候选点会抑制其
    ``min_distance`` 内的所有后续候选。使用 cKDTree 邻域查询，复杂度为
    O(N log N + 邻接边数)，避免大规模候选下 O(N^2) 的全量距离计算。
    """
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, dtype=np.float64)
    if len(coords) == 0 or min_distance <= 0:
        return np.empty((0, 2), dtype=coords.dtype)

    tree = cKDTree(coords)
    pairs = tree.query_pairs(min_distance, output_type="ndarray")
    neighbors: dict[int, list[int]] = {}
    for i, j in pairs:
        neighbors.setdefault(int(i), []).append(int(j))
        neighbors.setdefault(int(j), []).append(int(i))

    selected_indices = []
    suppressed = np.zeros(len(coords), dtype=bool)
    for idx_c in range(len(coords)):
        if suppressed[idx_c]:
            continue
        selected_indices.append(idx_c)
        for nb in neighbors.get(idx_c, ()):
            suppressed[nb] = True
    return coords[selected_indices]


def parse_detection_threshold(text):
    """解析检测阈值输入; 返回 (0, 1) 内的百分位数小数, 留空返回 None。

    分析图像在加载时已归一化并裁剪到 [0, 1] (见 ppa_core.image_io),
    因此 >=1 的"绝对阈值"永远选不出任何像素, 该模式是死功能;
    阈值统一为百分位数语义 (0.8 = 取 80% 分位以上的像素)。
    """
    stripped = str(text).strip()
    if not stripped:
        return None
    try:
        value = float(stripped)
    except ValueError:
        raise ValueError("阈值必须是 0~1 之间的百分位数 (如 0.8), 或留空使用自适应。") from None
    if not np.isfinite(value) or not (0.0 < value < 1.0):
        raise ValueError(
            f"阈值 {stripped} 无效: 图像已归一化到 [0,1], 阈值只接受 0~1 之间的"
            "百分位数 (如 0.8 = 保留强度前 20% 的像素), 留空则自适应 (80%)。")
    return value


@dataclass(frozen=True)
class ChainVectorEstimate:
    """One lattice-step vector fitted from an ordered chain of atoms."""

    vector: np.ndarray
    n_points: int
    rms_fit_residual: float
    spacing_std: float
    angle_rms_deg: float


def fit_lattice_vector_from_chain(chain_points, *, label="原子列", min_points=3):
    """Fit one directed lattice vector from an ordered, consecutive atom chain.

    The atom order supplies integer coordinates ``0, 1, ..., N-1``.  A robust
    two-dimensional Huber line fit estimates position = intercept + index * v,
    so both the vector length and its angle are determined jointly from every
    selected atom.  This avoids the circular-angle problem and, unlike taking
    the arithmetic mean of adjacent differences, does not algebraically reduce
    to using only the two endpoints.

    ``chain_points`` must be clicked consecutively in the desired positive
    direction.  Large backtracking, off-axis jumps, or a likely skipped atom
    are rejected because an incorrect integer step would bias the reference
    spacing and therefore the full strain map.
    """
    pts = np.asarray(chain_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise AnalysisError(f"{label}坐标必须是 N×2 数组。")
    if len(pts) < int(min_points):
        raise AnalysisError(f"{label}至少需要 {int(min_points)} 个连续原子。")
    if not np.isfinite(pts).all():
        raise AnalysisError(f"{label}坐标包含 NaN 或无穷值。")

    steps = np.diff(pts, axis=0)
    step_lengths = np.linalg.norm(steps, axis=1)
    if np.any(step_lengths <= 1e-12):
        raise AnalysisError(f"{label}中存在重复原子。")

    gross = pts[-1] - pts[0]
    gross_length = float(np.linalg.norm(gross))
    if gross_length <= 1e-12:
        raise AnalysisError(f"{label}首尾重合，请按同一方向依次选择。")
    gross_unit = gross / gross_length
    forward = steps @ gross_unit
    transverse = np.linalg.norm(steps - forward[:, None] * gross_unit[None, :], axis=1)
    median_forward = float(np.median(forward))
    if median_forward <= 1e-12 or np.any(forward <= 0.0):
        raise AnalysisError(f"{label}点击顺序发生回退，请沿同一方向依次选择连续原子。")
    if np.any(transverse > 0.60 * median_forward):
        raise AnalysisError(f"{label}中有明显偏离原子列的点，请重新选择同一列。")
    if len(steps) >= 3:
        relative_forward = forward / median_forward
        if np.any(relative_forward < 0.55) or np.any(relative_forward > 1.65):
            raise AnalysisError(f"{label}的相邻间距不连续，可能跳过或选错了原子。")

    # Iteratively reweighted least squares with Huber weights.  The regression
    # is vector-valued: x/y are fitted together and one weight is assigned per
    # atom from its 2-D residual, preserving the physical vector direction.
    index = np.arange(len(pts), dtype=np.float64)
    design = np.column_stack((np.ones(len(pts)), index))
    weights = np.ones(len(pts), dtype=np.float64)
    coefficients = np.zeros((2, 2), dtype=np.float64)
    for _ in range(12):
        root_w = np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(
            design * root_w[:, None], pts * root_w[:, None], rcond=None)
        fitted = design @ coefficients
        residual_norm = np.linalg.norm(pts - fitted, axis=1)
        median_residual = float(np.median(residual_norm))
        mad = float(np.median(np.abs(residual_norm - median_residual)))
        scale = 1.4826 * mad
        if scale <= 1e-12:
            break
        cutoff = 1.5 * scale
        new_weights = np.ones_like(weights)
        large = residual_norm > cutoff
        new_weights[large] = cutoff / residual_norm[large]
        if np.max(np.abs(new_weights - weights)) < 1e-5:
            weights = new_weights
            break
        weights = new_weights

    # Refit once with the converged weights so diagnostics describe the exact
    # vector returned to the GUI.
    root_w = np.sqrt(weights)
    coefficients, *_ = np.linalg.lstsq(
        design * root_w[:, None], pts * root_w[:, None], rcond=None)
    fitted = design @ coefficients
    vector = coefficients[1].astype(np.float64, copy=True)
    vector_length = float(np.linalg.norm(vector))
    if vector_length <= 1e-12:
        raise AnalysisError(f"{label}无法拟合出非零晶格矢量。")

    unit = vector / vector_length
    projected_spacing = steps @ unit
    cross = steps[:, 0] * unit[1] - steps[:, 1] * unit[0]
    step_angles = np.arctan2(np.abs(cross), np.maximum(projected_spacing, 1e-12))
    residual = pts - fitted
    return ChainVectorEstimate(
        vector=vector,
        n_points=len(pts),
        rms_fit_residual=float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))),
        spacing_std=float(np.std(projected_spacing, ddof=1)) if len(steps) > 1 else 0.0,
        angle_rms_deg=float(np.degrees(np.sqrt(np.mean(step_angles * step_angles)))),
    )


def _typical_nearest_neighbor_spacing(points):
    """Return a robust image-wide nearest-neighbor spacing in pixels."""
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        raise AnalysisError("至少需要 2 个已识别原子才能估计原子间距。")
    if not np.isfinite(pts).all():
        raise AnalysisError("原子坐标包含 NaN 或无穷值。")
    distances, _ = cKDTree(pts).query(pts, k=2)
    nearest = np.asarray(distances[:, 1], dtype=np.float64)
    nearest = nearest[np.isfinite(nearest) & (nearest > 1e-9)]
    if len(nearest) == 0:
        raise AnalysisError("已识别原子中只有重复坐标，无法估计原子间距。")
    return float(np.median(nearest))


def _fit_unoriented_line(points, orientation_hint, *, label):
    """Total-least-squares line fit with its sign fixed by the click direction."""
    pts = np.asarray(points, dtype=np.float64)
    center = np.mean(pts, axis=0)
    centered = pts - center
    _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) == 0 or singular_values[0] <= 1e-12:
        raise AnalysisError(f"{label}的手选点无法确定方向。")
    direction = axes[0].astype(np.float64, copy=True)
    if float(np.dot(direction, orientation_hint)) < 0.0:
        direction *= -1.0
    return center, direction


def infer_atom_chain_from_anchors(all_points, anchor_indices, *, label="原子列",
                                  min_anchors=2, corridor_fraction=0.30):
    """Infer every detected atom on a line defined by arbitrary clicked anchors.

    The clicked atoms are *anchors*, not assumed lattice neighbors.  A line is
    fitted through them, then all detected atoms in a narrow corridor between
    the outermost anchors are gathered and ordered.  Consequently clicking
    atoms 1, 5 and 9 yields the full 1..9 chain and one fitted vector per true
    inter-atomic interval rather than a four-period supercell vector.

    The returned indices follow the direction from the first clicked anchor to
    the last clicked anchor.  Horizontal, vertical and oblique chains use the
    same geometry; no axis-specific special case is involved.
    """
    pts = np.asarray(all_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise AnalysisError("全部原子坐标必须是 N×2 数组。")
    if not np.isfinite(pts).all():
        raise AnalysisError("全部原子坐标包含 NaN 或无穷值。")

    try:
        anchors = np.asarray(anchor_indices, dtype=np.int64)
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"{label}的手选原子编号无效。") from error
    if anchors.ndim != 1 or len(anchors) < int(min_anchors):
        raise AnalysisError(f"{label}至少任选 {int(min_anchors)} 个原子来确定直线。")
    if len(np.unique(anchors)) != len(anchors):
        raise AnalysisError(f"{label}中存在重复选择的原子。")
    if np.any(anchors < 0) or np.any(anchors >= len(pts)):
        raise AnalysisError(f"{label}的手选原子编号超出范围。")

    anchor_points = pts[anchors]
    orientation_hint = anchor_points[-1] - anchor_points[0]
    if float(np.linalg.norm(orientation_hint)) <= 1e-12:
        raise AnalysisError(f"{label}首末手选点重合，无法确定矢量正方向。")

    typical_spacing = _typical_nearest_neighbor_spacing(pts)
    corridor = max(0.5, float(corridor_fraction) * typical_spacing)
    endpoint_padding = 0.45 * typical_spacing

    center, direction = _fit_unoriented_line(
        anchor_points, orientation_hint, label=label)
    anchor_delta = anchor_points - center
    anchor_projection = anchor_delta @ direction
    anchor_perpendicular = np.linalg.norm(
        anchor_delta - anchor_projection[:, None] * direction[None, :], axis=1)
    if float(np.max(anchor_perpendicular)) > corridor:
        raise AnalysisError(
            f"{label}的手选点没有落在同一条原子列上（最大偏离 "
            f"{float(np.max(anchor_perpendicular)):.2f} px）；请移除选错的点。")

    # Two passes are enough: anchors define the initial corridor; all atoms in
    # that corridor then refine the line direction before the final selection.
    selected = None
    for _ in range(2):
        delta = pts - center
        projection = delta @ direction
        perpendicular = np.linalg.norm(
            delta - projection[:, None] * direction[None, :], axis=1)
        anchor_projection = (anchor_points - center) @ direction
        lower = float(np.min(anchor_projection) - endpoint_padding)
        upper = float(np.max(anchor_projection) + endpoint_padding)
        selected = np.flatnonzero(
            (projection >= lower) & (projection <= upper) &
            (perpendicular <= corridor))
        if len(selected) < 2:
            break
        center, direction = _fit_unoriented_line(
            pts[selected], orientation_hint, label=label)

    if selected is None or len(selected) < 2:
        raise AnalysisError(f"{label}范围内未找到足够的同行原子。")
    if not set(int(i) for i in anchors).issubset(set(int(i) for i in selected)):
        raise AnalysisError(f"{label}的手选点中有原子明显偏离拟合直线。")

    projection = (pts[selected] - center) @ direction
    ordered = selected[np.argsort(projection, kind="stable")]
    # Validate that the automatically gathered points form a directed chain.
    # The minimum remains two: users may choose any count, and non-adjacent
    # anchors normally expand to many intermediate detected atoms here.
    fit_lattice_vector_from_chain(pts[ordered], label=label, min_points=2)
    return ordered.astype(np.int64, copy=False)


def estimate_reference_vectors_from_chains(a_points, b_points, *, min_points=3,
                                           max_condition=None):
    """Estimate and validate two reference vectors from two ordered atom chains.

    ``max_condition`` 由调用方传入，保证与用户在界面上配置的阈值一致；
    ``None`` 保持历史默认（30.0）。
    """
    a_fit = fit_lattice_vector_from_chain(a_points, label="a 方向原子列", min_points=min_points)
    b_fit = fit_lattice_vector_from_chain(b_points, label="b 方向原子列", min_points=min_points)
    if max_condition is None:
        validate_reference_lattice(a_fit.vector, b_fit.vector)
    else:
        validate_reference_lattice(a_fit.vector, b_fit.vector, max_condition=max_condition)
    return a_fit, b_fit


def refine_lattice_basis_in_region(pts, origin, a_vec, b_vec, region_mask,
                                   min_atoms=10, max_iter=3, max_condition=None):
    """
    仅用「无应变参考区」内的原子精化基矢，不吸收区外的真实应变。

    与全图精化的区别: 全图最小二乘会把区外的均匀应变一起拟合进参考态,
    从而抹掉待测信号; 限制在无应变参考区内拟合则只降低基矢的定位噪声,
    参考态仍是物理上的无应变晶格。基矢相对误差从 ~1% 降至 ~1%/√N。

    区内原子的整数索引由用户给定的初始基矢确定; 参考区应足够小
    (无应变、无错配) 以保证这些索引可靠。

    Parameters
    ----------
    pts : (N, 2) np.ndarray  全部原子坐标
    origin, a_vec, b_vec : np.ndarray  初始晶格原点与基矢
    region_mask : (N,) bool  参考区内原子掩码
    min_atoms : int  精化所需的最少区内原子数
    max_iter : int  索引/基矢交替迭代次数

    Returns
    -------
    origin_r, a_r, b_r : np.ndarray  精化后的原点与基矢 (失败时返回输入值)
    applied : bool  是否成功精化
    n_used : int  实际参与拟合的原子数
    """
    pts = np.asarray(pts, dtype=np.float64)
    if region_mask is None:
        return origin, a_vec, b_vec, False, 0
    region_mask = np.asarray(region_mask, dtype=bool)
    n_used = int(region_mask.sum())
    if n_used < min_atoms:
        return origin, a_vec, b_vec, False, n_used

    try:
        Minv = np.linalg.inv(np.column_stack((a_vec, b_vec)))
    except np.linalg.LinAlgError:
        return origin, a_vec, b_vec, False, n_used

    sub = pts[region_mask]
    o_r = np.asarray(origin, dtype=np.float64).copy()
    a_r = np.asarray(a_vec, dtype=np.float64).copy()
    b_r = np.asarray(b_vec, dtype=np.float64).copy()
    try:
        if max_condition is None:
            indices = assign_unique_lattice_indices(sub, o_r, a_r, b_r).lattice_indices
        else:
            indices = assign_unique_lattice_indices(
                sub, o_r, a_r, b_r, max_condition=max_condition).lattice_indices
    except (AnalysisError, ValueError):
        return origin, a_vec, b_vec, False, n_used

    applied = False
    for _ in range(max_iter):
        design = np.column_stack([np.ones(len(sub)), indices.astype(float)])
        if np.linalg.matrix_rank(design) < 3:
            break  # 区内索引共线 (如单行原子), 无法定出两个基矢
        coef_x, *_ = np.linalg.lstsq(design, sub[:, 0], rcond=None)
        coef_y, *_ = np.linalg.lstsq(design, sub[:, 1], rcond=None)
        o_c = np.array([coef_x[0], coef_y[0]])
        a_c = np.array([coef_x[1], coef_y[1]])
        b_c = np.array([coef_x[2], coef_y[2]])
        M_c = np.column_stack((a_c, b_c))
        if abs(np.linalg.det(M_c)) < 1e-12:
            break  # 精化结果奇异, 保留上一次有效值
        o_r, a_r, b_r = o_c, a_c, b_c
        applied = True
        try:
            if max_condition is None:
                idx_c = assign_unique_lattice_indices(sub, o_r, a_r, b_r).lattice_indices
            else:
                idx_c = assign_unique_lattice_indices(
                    sub, o_r, a_r, b_r, max_condition=max_condition).lattice_indices
        except (AnalysisError, ValueError):
            break
        if np.array_equal(idx_c, indices):
            break  # 索引收敛
        indices = idx_c

    return o_r, a_r, b_r, applied, n_used


@dataclass(frozen=True)
class LatticeAssignment:
    """One-to-one atom-to-lattice assignment; every accepted atom is retained."""

    points: np.ndarray
    lattice_indices: np.ndarray
    lattice_coords: np.ndarray
    source_indices: np.ndarray
    residuals_lattice: np.ndarray
    residuals_px: np.ndarray
    reassigned_mask: np.ndarray
    low_confidence_mask: np.ndarray
    n_conflicts: int


def assign_unique_lattice_indices(points, origin, a_vec, b_vec, *,
                                  max_search_radius=4,
                                  low_confidence_limit=0.45,
                                  max_condition=None):
    """Assign every accepted atom to a unique integer lattice site.

    Independent rounding can map two distinct accepted atoms to the same
    integer site.  This function keeps *all* input rows and solves a sparse
    minimum-cost bipartite matching problem over nearby lattice candidates.
    The result is deterministic, one-to-one, and never mutates ``points``.

    ``low_confidence_limit`` is the maximum allowed residual in either lattice
    coordinate.  A larger residual is reported but never removes or skips the
    atom.

    ``max_condition`` must be threaded in from the caller so the lattice
    condition-number limit used here matches the one the user configured; the
    ``None`` default keeps backward compatibility (30.0 via
    :func:`validate_reference_lattice`).
    """
    raw_points = np.asarray(points, dtype=np.float64)
    if raw_points.ndim != 2 or raw_points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if not np.isfinite(raw_points).all():
        raise ValueError("points contains NaN or infinity")
    origin = np.asarray(origin, dtype=np.float64)
    a_vec = np.asarray(a_vec, dtype=np.float64)
    b_vec = np.asarray(b_vec, dtype=np.float64)
    if origin.shape != (2,) or a_vec.shape != (2,) or b_vec.shape != (2,):
        raise ValueError("origin, a_vec and b_vec must have shape (2,)")
    if max_condition is None:
        validate_reference_lattice(a_vec, b_vec)
    else:
        validate_reference_lattice(a_vec, b_vec, max_condition=max_condition)
    matrix = np.column_stack((a_vec, b_vec))
    coords = (raw_points - origin) @ np.linalg.inv(matrix).T
    naive = np.rint(coords).astype(int)
    n_points = len(raw_points)
    if n_points == 0:
        empty_bool = np.empty(0, dtype=bool)
        empty_float = np.empty(0, dtype=float)
        return LatticeAssignment(
            raw_points.copy(), naive, coords, np.empty(0, dtype=int),
            empty_float, empty_float.copy(), empty_bool, empty_bool.copy(), 0)

    n_conflicts = n_points - len({tuple(row) for row in naive})
    assigned = naive.copy()
    if n_conflicts:
        matched = False
        for radius in range(1, max(1, int(max_search_radius)) + 1):
            offsets = np.array(
                [(dn, dm) for dn in range(-radius, radius + 1)
                          for dm in range(-radius, radius + 1)],
                dtype=int,
            )
            candidate_rows = (naive[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
            candidates = np.unique(candidate_rows, axis=0)
            if len(candidates) < n_points:
                continue
            candidate_lookup = {tuple(row): idx for idx, row in enumerate(candidates)}
            rows = np.repeat(np.arange(n_points), len(offsets))
            cols = np.fromiter(
                (candidate_lookup[tuple(row)] for row in candidate_rows),
                dtype=int,
                count=len(candidate_rows),
            )
            ideal_candidates = (origin[None, :]
                                + candidate_rows[:, 0:1] * a_vec[None, :]
                                + candidate_rows[:, 1:2] * b_vec[None, :])
            repeated_points = np.repeat(raw_points, len(offsets), axis=0)
            costs = np.sum((repeated_points - ideal_candidates) ** 2, axis=1)
            # Sparse matching ignores explicit zero entries.  The tiny positive
            # tie-break also makes exactly symmetric assignments reproducible.
            costs = costs + 1e-12 * (1.0 + cols)
            graph = csr_matrix((costs, (rows, cols)), shape=(n_points, len(candidates)))
            try:
                row_ind, col_ind = min_weight_full_bipartite_matching(graph)
            except ValueError:
                continue
            if len(row_ind) != n_points:
                continue
            order = np.argsort(row_ind)
            assigned = candidates[col_ind[order]]
            matched = True
            break
        if not matched:
            raise AnalysisError(
                "无法为全部原子建立唯一晶格索引；请检查参考矢量，或移除真正重复的检测点。"
            )

    ideal = (origin[None, :]
             + assigned[:, 0:1] * a_vec[None, :]
             + assigned[:, 1:2] * b_vec[None, :])
    residuals_lattice = np.linalg.norm(coords - assigned, axis=1)
    residuals_px = np.linalg.norm(raw_points - ideal, axis=1)
    reassigned_mask = np.any(assigned != naive, axis=1)
    low_confidence_mask = np.max(np.abs(coords - assigned), axis=1) > float(low_confidence_limit)
    return LatticeAssignment(
        points=raw_points.copy(),
        lattice_indices=assigned,
        lattice_coords=coords,
        source_indices=np.arange(n_points, dtype=int),
        residuals_lattice=residuals_lattice,
        residuals_px=residuals_px,
        reassigned_mask=reassigned_mask,
        low_confidence_mask=low_confidence_mask,
        n_conflicts=n_conflicts,
    )


def find_lattice_holes(lattice_indices):
    """
    晶格完备性自检: 检测整数索引网格中的空洞 (疑似漏检原子)。

    三明治判据: 沿 +n 或 +m 方向，若 i 与 i+2d 均存在而 i+d 缺失，
    则 i+d 为空洞。边界位置不触发，避免误报。

    Returns
    -------
    holes : list of (n, m)  空洞的整数晶格索引 (排序后)
    """
    idx_set = {tuple(int(v) for v in idx) for idx in lattice_indices}
    holes = set()
    for n_i, m_i in idx_set:
        for dn, dm in ((1, 0), (0, 1)):
            if (n_i + dn, m_i + dm) not in idx_set and \
               (n_i + 2 * dn, m_i + 2 * dm) in idx_set:
                holes.add((n_i + dn, m_i + dm))
    return sorted(holes)


# 剪切分量与旋转角在 y 反射 (物理 y-up ↔ 显示 y-down) 下必须变号;
# 正应变与等效应变 (不变量) 不受影响。
DISPLAY_SIGN_FLIPPED_KEYS = frozenset({'xy', 'gl_xy', 'rot'})


def strain_value_for_display(key, values):
    """把物理坐标 (y 向上) 约定的应变分量转成图像显示坐标 (y 向下) 约定。

    显示轴是 y 向下的; 直接铺物理约定的 ε_xy/ε_xy(GL)/θ 会让剪切和旋转
    的空间解释与屏幕几何相反, 也与 ppa_stats (导入时翻转) 的图相反。
    存储与 CSV 导出保持物理约定, 仅在渲染层取号。
    """
    if values is not None and key in DISPLAY_SIGN_FLIPPED_KEYS:
        return -values
    return values


def _strain_fields_from_result(result, geometry, ideal_grid_display):
    """把核心 StrainResult 转成 GUI/导出字段字典 (纯函数, 不写实例状态)。

    在后台分析线程中调用: 只构造返回值, 实例字段由 Tk 主线程在
    ``_finish_ppa_analysis`` 中一次性写入, 消除与点表编辑的竞态。
    """
    fields = {
        'result_locations_physical': result.locations.copy(),
        'tri_centroids': result.locations * np.array([1.0, -1.0]),
        'strain_xx': result.small_xx,
        'strain_yy': result.small_yy,
        'strain_xy': result.small_xy,
        'strain_eq': result.equivalent_small,
        'rotation': result.polar_rotation,
        'strain_gl_xx': result.green_xx,
        'strain_gl_yy': result.green_yy,
        'strain_gl_xy': result.green_xy,
        'strain_gl_eq': result.equivalent_green,
        'strain_quality_grades': (None if result.site_quality is None
                                  else np.asarray(result.site_quality).copy()),
        'strain_invalid_reasons': (None if result.invalid_reasons is None
                                   else np.asarray(result.invalid_reasons).copy()),
        'strain_geometry': geometry,
        'tri_edge_mask': None,
        'element_area': None,
    }
    if geometry == "triangles" and result.simplices is not None:
        fields['tri_edge_mask'] = result.edge_mask
        reference_physical = ideal_grid_display * np.array([1.0, -1.0])
        triangles = reference_physical[result.simplices]
        edges_a = triangles[:, 1] - triangles[:, 0]
        edges_b = triangles[:, 2] - triangles[:, 0]
        fields['element_area'] = 0.5 * np.abs(
            edges_a[:, 0] * edges_b[:, 1] - edges_a[:, 1] * edges_b[:, 0])
    return fields


def _interpolate_strain_grids(fields, image_shape, grid_size=200):
    """把逐元素应变插值到规则网格 (纯函数; 输入为字段字典)。

    使用 Clough-Tocher C¹ 分片三次插值; 凸包外科学上无效, 保持 NaN,
    绝不用最近邻值填色。返回 (strain_grid, strain_grid_gl, extent)。
    """
    centroids = fields['tri_centroids']
    if centroids is None:
        return {}, {}, None

    h, w = image_shape
    # 工单74: 云图 extent 以像素边缘为界 (-0.5 ~ N-0.5)。底图 imshow 用默认
    # extent, 像素中心在整数坐标、图像盒为 [-0.5, N-0.5]; 若云图盒取 (0, N-1),
    # 不仅整体内缩半像素, N 个插值单元的中心还与网格节点
    # linspace(0, N-1, grid_size) 最多错开半个网格(边缘处), 云图相对底图错位。
    x_min, x_max = -0.5, w - 0.5
    y_min, y_max = -0.5, h - 0.5
    extent = (x_min, x_max, y_min, y_max)

    gx = np.linspace(x_min, x_max, grid_size)
    gy = np.linspace(y_min, y_max, grid_size)
    GX, GY = np.meshgrid(gx, gy)
    grid_points = np.column_stack((GX.ravel(), GY.ravel()))

    def _ct_interp(values):
        values = np.asarray(values, dtype=float)
        valid = np.isfinite(values) & np.isfinite(centroids).all(axis=1)
        valid_points = centroids[valid]
        valid_values = values[valid]
        if len(valid_points) < 3 or np.linalg.matrix_rank(valid_points - valid_points.mean(axis=0)) < 2:
            return np.full((grid_size, grid_size), np.nan, dtype=float)
        try:
            interp = CloughTocher2DInterpolator(valid_points, valid_values)
            return interp(grid_points).reshape(grid_size, grid_size)
        except Exception:
            # Linear fallback preserves the same convex-hull validity rule.
            grid_vals = griddata(valid_points, valid_values, grid_points,
                                 method='linear')
            return grid_vals.reshape(grid_size, grid_size)

    # 对每种应变分量插值 (剪切/旋转先转为显示坐标约定)
    fields_map = {
        'xx': fields['strain_xx'],
        'yy': fields['strain_yy'],
        'xy': strain_value_for_display('xy', fields['strain_xy']),
        'eq': fields['strain_eq'],
        'rot': strain_value_for_display('rot', fields['rotation']),
    }
    strain_grid = {}
    for key, values in fields_map.items():
        if values is not None:
            strain_grid[key] = _ct_interp(values)

    fields_gl = {
        'gl_xx': fields['strain_gl_xx'],
        'gl_yy': fields['strain_gl_yy'],
        'gl_xy': strain_value_for_display('gl_xy', fields['strain_gl_xy']),
        'gl_eq': fields['strain_gl_eq'],
    }
    strain_grid_gl = {}
    for key, values in fields_gl.items():
        if values is not None:
            strain_grid_gl[key] = _ct_interp(values)
    return strain_grid, strain_grid_gl, extent


# 应变云图 (插值视图) 的可选配色。发散型色图以对称色标渲染, 使 ±应变
# 强度可比; 用户未选择时按分量沿用默认配色 (发散分量 RdBu_r, 等效应变 turbo)。
STRAIN_CLOUD_DIVERGING_CMAPS = frozenset({
    'RdBu_r', 'RdBu', 'coolwarm', 'bwr', 'seismic', 'BrBG', 'BrBG_r',
    'PiYG', 'PiYG_r', 'Spectral', 'Spectral_r',
})
STRAIN_CLOUD_CMAP_CHOICES = [
    "自动 (按分量)",
    "RdBu_r", "coolwarm", "seismic", "bwr", "BrBG_r", "PiYG_r",
    "turbo", "viridis", "inferno", "plasma", "magma", "cividis", "jet",
]
STRAIN_CLOUD_AUTO_CMAP_LABEL = STRAIN_CLOUD_CMAP_CHOICES[0]

# 位移幅值染色的自定义红蓝色带 (用户提供 6 色: 蓝 3 阶 + 红 3 阶)。
# 排列为 深蓝→浅蓝→浅红→深红: 位移小偏蓝 (平静), 位移大偏红 (显著)。
DISPLACEMENT_RB_CMAP = LinearSegmentedColormap.from_list(
    'ppa_red_blue',
    ['#394f7f', '#5d7eaf', '#c6def6', '#fda1ab', '#ea7278', '#f52419'],
)
POINT_CMAP_CHOICES = ["turbo (默认)", "红蓝 (自定义)"]


def _resolve_cmap(cmap):
    """Colormap 解析, 兼容 matplotlib>=3.7 (plt.get_cmap 已弃用)。"""
    if isinstance(cmap, str):
        return matplotlib.colormaps[cmap]
    return cmap


class AtomMarkerApp:
    # ---- 渲染参数 ----
    POINT_MARKER_RADIUS = 7.0   # 原子标记圆半径 (图像像素, 随缩放变化)
    MAX_LABELS_DRAWN = 400      # 编号标签绘制上限 (超出则仅在放大后画视口内的)
    MAX_UNDO_ENTRIES = 500      # 撤销栈上限, 防止超长会话内存无限增长
    # 参考晶格/局部 PPA 共用的条件数上限默认值。同时作为**类级**默认，
    # 使不经 __init__ 构造的实例（测试用 __new__）也能解析到与历史一致的 30.0，
    # 而不是抛 AttributeError。__init__ 会写入同值的实例属性。
    lattice_max_condition = float(LATTICE_CONDITION_GUIDANCE["general"])

    # ---- 检测参数预设 ----
    DETECT_PRESETS = {
        "标准 HRTEM (默认)": {
            'sigma': 0.8, 'min_dist': 8, 'window': 5, 'method': 'com',
            'desc': '通用设置，适用于大多数 HRTEM 原子像'
        },
        "高分辨 / 小原子 (2-4 px)": {
            'sigma': 0.4, 'min_dist': 4, 'window': 3, 'method': 'com',
            'desc': '原子直径 2-4 px，降低模糊和抑制半径避免漏检'
        },
        "超分辨 / 极小原子 (<2 px)": {
            'sigma': 0.2, 'min_dist': 3, 'window': 3, 'method': 'com',
            'desc': '原子极小，几乎不做模糊，用最小抑制半径'
        },
        "低倍 / 大原子 (>6 px)": {
            'sigma': 1.5, 'min_dist': 12, 'window': 7, 'method': 'com',
            'desc': '原子直径 >6 px，强模糊去噪，增大间距避免重复检测'
        },
        "STEM HAADF (高衬度亮点)": {
            'sigma': 0.6, 'min_dist': 6, 'window': 5, 'method': 'gaussian',
            'desc': '原子柱成像清晰，高斯拟合可达最高亚像素精度'
        },
        "高噪声 / 低信噪比": {
            'sigma': 1.2, 'min_dist': 8, 'window': 7, 'method': 'com',
            'desc': '强高斯模糊抑制噪声，加大窗口提高 COM 稳定性'
        },
    }

    def __init__(self, root):
        self.root = root
        self.root.title("原子级位移与应变分析工具 v3.4 – PPA · 应变张量")
        self.root.geometry("1500x900")
        self.root.minsize(1200, 700)

        # ---- 全局样式配置 ----
        self._setup_styles()

        # ---- 数据 ----
        self.image = None          # 原始图像 (float64)
        self.image_path = None
        self.image_frame_index = None
        self.image_metadata = None
        self.points = []           # [(x, y), ...]  原子坐标
        self.undo_stack = []       # 撤销记录: [(操作, 数据), ...]

        # ---- PPA 位移分析结果 ----
        self.reference_vecs = None  # (a_vec, b_vec)
        self.ref_origin = None     # np.array([x,y])  晶格原点
        self.reference_metadata = None  # 参考矢量来源、选点与拟合质量（用于复现）
        self.ideal_grid = None      # (N,2) 匹配的理想格点
        self.matched_actual = None  # (N,2) 对应的实际点
        self.displacements = None   # (N,2) 位移矢量
        self.distortions = None     # (N,)  位移幅值
        self.analysis_point_indices = None  # (N,) 恒为原始标注索引 0..N-1
        self.lattice_indices = None         # (N,2) 全局一对一整数晶格索引
        self.lattice_coords = None          # (N,2) 连续晶格坐标
        self.assignment_residuals = None    # (N,) 晶格单位匹配残差
        self.assignment_residuals_px = None # (N,) 像素单位匹配残差
        self.assignment_reassigned_mask = None   # (N,) 是否因冲突重分配
        self.assignment_low_confidence_mask = None  # (N,) 仅质量警告，不排除
        self.assignment_conflict_count = 0

        # ---- 应变张量分析结果 ----
        self.tri_centroids = None      # (M,2) 结果位置 (三角形质心或原子位点, 显示坐标)
        self.strain_xx = None          # (M,) ε_xx 正应变 x
        self.strain_yy = None          # (M,) ε_yy 正应变 y
        self.strain_xy = None          # (M,) ε_xy 张量剪应变 (工程剪应变 γ_xy = 2·ε_xy)
        self.strain_eq = None          # (M,) von Mises 等效应变
        self.rotation = None           # (M,) 刚体旋转 ω (弧度)
        # Green-Lagrange 有限应变分量
        self.strain_gl_xx = None       # (M,) ε_xx GL
        self.strain_gl_yy = None       # (M,) ε_yy GL
        self.strain_gl_xy = None       # (M,) ε_xy GL
        self.strain_gl_eq = None       # (M,) ε_eq GL (von Mises)
        self.strain_grid_gl = {}       # dict: GL 插值网格
        self.strain_type = "infinitesimal"  # "infinitesimal" | "green_lagrange"
        self.strain_grid = {}          # dict: {'xx': (G,G), 'yy': ...} 插值网格
        self.strain_grid_extent = None # (xmin, xmax, ymin, ymax) 网格范围
        self.tri_edge_mask = None      # (M,) bool 边缘三角形掩码（含凸包顶点）
        self.outlier_mask = None       # (N,) bool 匹配异常原子掩码
        self.lattice_holes = []        # 晶格空洞 (疑似漏检) 理想位置 [(x, y), ...]
        self.strain_geometry = None    # "triangles" (Legacy CST) | "sites" (local PPA)
        self.result_locations_physical = None
        self.element_area = None
        self.strain_quality_grades = None
        self.strain_invalid_reasons = None
        self.analysis_method = "peak_pairs"  # local PPA is the scientific default

        # ---- 交互状态 ----
        self.current_mode = "add"   # "add" | "delete" | "select_ref" | "select_ref_multi"
        self.pick_radius = 12
        self._cbar = None
        self.ref_select_indices = []
        # 双方向参考选择：用户点的是“定义原子列的锚点”，不要求相邻；
        # auto_indices 是程序沿拟合直线自动纳入的全部同行原子。
        self.ref_multi_indices = [[], []]       # [a锚点索引, b锚点索引]
        self.ref_multi_auto_indices = [[], []]  # [a完整原子列, b完整原子列]
        self.ref_multi_stage = 0                # 0=a列, 1=b列
        self.selected_point_idx = None  # 当前选中的点（高亮）
        self._calibrate_pts = []       # 校准模式临时点 [(x, y), ...]
        self._ctrl_pressed = False     # Ctrl 键按下状态（缩放用）
        self._zoom_history = []        # (xlim, ylim) 缩放历史栈
        self._image_shape = None       # 图像尺寸记录（用于防止缩放重置）
        self._img_artist = None        # matplotlib AxesImage（在 display_image 中创建）
        self.detect_roi = None         # 检测区域 (x0, y0, x1, y1) 或 None
        self._roi_mode_active = False  # ROI 选区模式是否激活
        self._drawing_roi = None       # ROI 绘制状态: (start_x, start_y) 正在拖拽中
        self._roi_rect = None          # matplotlib Rectangle patch (拖拽预览)
        # 多边形自由选区
        self.detect_polygon = None         # [(x1,y1), ...] 多边形顶点列表（已完成选区）
        self._polygon_mode_active = False  # 自由选区模式是否激活
        self._polygon_vertices = []        # 绘制中的临时顶点列表 [(x,y), ...]
        # 参考区 (基矢精化): 仅用区内原子拟合基矢, 不吸收区外真实应变
        self.ref_region = None             # (x0, y0, x1, y1) 无应变参考区
        self._ref_region_mode_active = False
        self._drawing_ref_region = None    # 拖拽起点 (x, y)
        self._ref_region_rect = None       # 拖拽预览 Rectangle patch
        self.invalid_sites = None          # (K,2) 局部PPA中缺四邻被跳过的原子位置

        # ---- 检测参数 ----
        self.detect_sigma = 0.8      # 高斯预滤波sigma
        self.detect_min_dist = 8     # 最小原子间距
        self.detect_window = 5       # 质心精炼窗口 (奇数)
        self.centroid_method = "com" # "com" | "gaussian"
        # 工单27: 最近一次高斯精炼的回退统计 (None 或 {n_points,n_fallback,fallback_ratio})
        self.last_gaussian_refine_stats = None
        # 参考晶格与局部 PPA 共用的条件数上限。默认取
        # ppa_core.strain.LATTICE_CONDITION_GUIDANCE["general"]（30.0，与历史行为
        # 一致）；用户按体系放宽/收紧后，**整条调用链**（参考区精化、晶格索引分配、
        # 局部应变、参考保存/加载）都必须使用同一取值，否则放宽会被隐藏的默认值
        # 再次挡下（回归 2026-10-03）。阈值必须是有限正数。
        self.lattice_max_condition = float(LATTICE_CONDITION_GUIDANCE["general"])
        self._suggested_threshold = None  # 校准模式推算的阈值（临时）
        self.von_mises_coeff = 4.0 / 9.0  # von Mises 系数 (2D 平面应变严格值=4/9; 文献经验值=2/3)

        # ---- 预处理 ----
        self.processed_image = None     # 预处理后的图像
        self.preprocess_method = "none" # 当前选择的预处理方法
        self.show_processed = False     # 是否显示预处理图像
        self.use_preprocessed = tk.BooleanVar(value=False)  # 是否用预处理图做检测
        self.preprocess_params = {      # 预处理参数
            'gaussian_sigma': 1.0,
            'median_size': 3,
            'bw_order': 4,
            'bw_cutoff': 0.3,
            'delta': 5.0,
            'cycles': 99,
            'step': 2,
        }
        self._filter_proc = None        # HRTEMFilter 实例 (lazy)

        # ---- 显示参数 ----
        self.arrow_scale = 1.0       # 位移箭头放大倍数
        self.arrow_lw    = 1.5       # 箭头线宽
        self.show_all_arrows = False # 是否全原子显示位移箭头

        # ---- 应变云图样式 (插值云图视图共用) ----
        self.strain_cloud_cmap = None   # None=按分量自动配色; 否则为用户选择的色图名
        self.strain_cloud_alpha = 0.75  # 云图不透明度 0~1 (1=完全遮挡底图)
        self._strain_cloud_artist = None  # 当前云图 imshow artist (透明度滑杆的就地更新)

        # ---- 位移点样式 (位移幅值染色 / 位移矢量场共用) ----
        self.atom_dot_size = 40          # 原子点散点大小 (pt²)
        self.point_cmap_key = 'turbo'    # 'turbo' | 'rb' (自定义红蓝色带)
        self._points_scatter_artist = None  # 当前散点 artist (点大小滑杆的就地更新)

        # ---- 色标范围控制 ----
        self.colorbar_manual = False   # False=自动范围, True=手动范围
        self.colorbar_vmin = tk.DoubleVar(value=0.0)   # 色标下限
        self.colorbar_vmax = tk.DoubleVar(value=1.0)   # 色标上限

        # ---- 后台计算 (queue + root.after; 线程内禁止操作 Tk) ----
        self._worker_queue = queue.Queue()
        self._job_generation = 0
        self._poll_after_id = None
        # 鼠标移动状态栏节流 (monotonic 秒)
        self._last_status_update = 0.0

        self.setup_ui()
        self._bind_shortcuts()

    # ================================================================
    #  样式配置
    # ================================================================
    def _setup_styles(self):
        """配置全局 ttk 样式主题 — 现代扁平化风格"""
        style = ttk.Style()
        # 使用 clam 主题作为基础（跨平台一致）
        style.theme_use('clam')

        # ---- 配色方案 (Material Design 风格) ----
        BG_MAIN = '#f5f5f5'        # 主背景
        BG_PANEL = '#ffffff'       # 面板背景
        BG_HEADER = '#1a237e'      # 标题栏深蓝
        ACCENT = '#1565c0'         # 强调色蓝
        ACCENT_HOVER = '#1976d2'   # 悬停色
        ACCENT_ACTIVE = '#0d47a1'  # 激活色
        SUCCESS = '#2e7d32'        # 成功绿
        WARNING = '#f57c00'        # 警告橙
        DANGER = '#c62828'         # 危险红
        TEXT_PRIMARY = '#212121'   # 主文字
        TEXT_SECONDARY = '#757575' # 次要文字
        BORDER = '#e0e0e0'         # 边框

        self.root.configure(bg=BG_MAIN)

        # ---- 全局字体 ----
        FONT_FAMILY = 'Microsoft YaHei UI' if 'Microsoft YaHei UI' in _available_fonts else 'Segoe UI'
        FONT_BASE = (FONT_FAMILY, 9)
        FONT_SMALL = (FONT_FAMILY, 8)
        FONT_TITLE = (FONT_FAMILY, 10, 'bold')
        FONT_HEADER = (FONT_FAMILY, 11, 'bold')

        # ---- 基础样式 ----
        style.configure('.', font=FONT_BASE, background=BG_MAIN)
        style.configure('TFrame', background=BG_MAIN)
        style.configure('TLabel', background=BG_MAIN, foreground=TEXT_PRIMARY)
        style.configure('TButton', padding=(8, 4))

        # ---- LabelFrame 样式 ----
        style.configure('Card.TLabelframe', background=BG_PANEL, relief='solid', borderwidth=1)
        style.configure('Card.TLabelframe.Label', background=BG_PANEL, foreground=ACCENT,
                        font=FONT_TITLE)
        style.map('Card.TLabelframe', background=[('active', BG_PANEL)])

        # ---- 主要操作按钮 (蓝色强调) ----
        style.configure('Accent.TButton', foreground='white', background=ACCENT,
                        font=FONT_BASE, padding=(10, 6))
        style.map('Accent.TButton',
                  background=[('active', ACCENT_HOVER), ('pressed', ACCENT_ACTIVE)],
                  foreground=[('active', 'white')])

        # ---- 成功按钮 (绿色) ----
        style.configure('Success.TButton', foreground='white', background=SUCCESS,
                        font=FONT_BASE, padding=(10, 6))
        style.map('Success.TButton',
                  background=[('active', '#388e3c'), ('pressed', '#1b5e20')])

        # ---- 警告按钮 (橙色) ----
        style.configure('Warning.TButton', foreground='white', background=WARNING,
                        font=FONT_BASE, padding=(8, 4))
        style.map('Warning.TButton',
                  background=[('active', '#fb8c00'), ('pressed', '#e65100')])

        # ---- 危险按钮 (红色) ----
        style.configure('Danger.TButton', foreground='white', background=DANGER,
                        font=FONT_BASE, padding=(8, 4))
        style.map('Danger.TButton',
                  background=[('active', '#d32f2f'), ('pressed', '#b71c1c')])

        # ---- 次要按钮 (灰色轮廓) ----
        style.configure('Outline.TButton', foreground=TEXT_PRIMARY, background=BG_PANEL,
                        relief='solid', borderwidth=1, padding=(8, 4))
        style.map('Outline.TButton',
                  background=[('active', '#e3f2fd')],
                  relief=[('active', 'solid')])

        # ---- RadioButton ----
        style.configure('TRadiobutton', background=BG_MAIN, foreground=TEXT_PRIMARY)
        style.map('TRadiobutton', background=[('active', '#e3f2fd')])

        # ---- Checkbutton ----
        style.configure('TCheckbutton', background=BG_MAIN, foreground=TEXT_PRIMARY)
        style.map('TCheckbutton', background=[('active', '#e3f2fd')])

        # ---- Entry ----
        style.configure('TEntry', fieldbackground='white', foreground=TEXT_PRIMARY)

        # ---- Combobox ----
        style.configure('TCombobox', fieldbackground='white', foreground=TEXT_PRIMARY)

        # ---- Scale (slider) ----
        style.configure('Horizontal.TScale', background=BG_MAIN, troughcolor=BORDER)

        # ---- Separator ----
        style.configure('TSeparator', background=BORDER)

        # ---- 状态栏 ----
        style.configure('Status.TLabel', background='#263238', foreground='#eceff1',
                        font=FONT_SMALL, padding=(8, 4))

        # ---- 标题栏 ----
        style.configure('Header.TLabel', background=BG_HEADER, foreground='white',
                        font=FONT_HEADER, padding=(12, 8))
        style.configure('HeaderSub.TLabel', background=BG_HEADER, foreground='#b3e5fc',
                        font=FONT_SMALL, padding=(12, 0, 12, 6))

        # ---- 存储配色供其他方法使用 ----
        self._colors = {
            'bg_main': BG_MAIN, 'bg_panel': BG_PANEL, 'accent': ACCENT,
            'success': SUCCESS, 'warning': WARNING, 'danger': DANGER,
            'text_primary': TEXT_PRIMARY, 'text_secondary': TEXT_SECONDARY,
            'border': BORDER, 'font_base': FONT_BASE, 'font_small': FONT_SMALL,
            'font_title': FONT_TITLE,
        }

    # ================================================================
    #  UI 布局
    # ================================================================
    def setup_ui(self):
        # ---- 顶部标题栏 ----
        header_frame = tk.Frame(self.root, bg='#1a237e', height=48)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        tk.Label(header_frame, text="⚛  原子位移与应变分析工具 (PPA)",
                 bg='#1a237e', fg='white', font=('Microsoft YaHei UI', 12, 'bold'),
                 anchor='w', padx=12).pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(header_frame, text="v3.4  |  Peak Pairs Analysis · 全点位移 · 应变张量",
                 bg='#1a237e', fg='#90caf9', font=('Microsoft YaHei UI', 8),
                 anchor='e', padx=12).pack(side=tk.RIGHT, fill=tk.Y)

        main_pw = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pw.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ---- 左侧：图像显示 ----
        left_frame = ttk.Frame(main_pw)
        main_pw.add(left_frame, weight=3)

        self.fig = Figure(figsize=(9, 8), dpi=100, facecolor='#fafafa')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#fafafa')
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, left_frame)
        toolbar.update()

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("pick_event", self.on_pick)

        # Ctrl 键状态追踪
        self.root.bind("<KeyPress-Control_L>", lambda e: setattr(self, '_ctrl_pressed', True))
        self.root.bind("<KeyRelease-Control_L>", lambda e: setattr(self, '_ctrl_pressed', False))
        self.root.bind("<KeyPress-Control_R>", lambda e: setattr(self, '_ctrl_pressed', True))
        self.root.bind("<KeyRelease-Control_R>", lambda e: setattr(self, '_ctrl_pressed', False))

        # ---- 右侧：控制面板 (带滚动条) ----
        right_frame = ttk.Frame(main_pw, width=400)
        main_pw.add(right_frame, weight=1)

        # 创建 Canvas + Scrollbar 实现右侧面板滚动
        self.right_canvas = tk.Canvas(right_frame, highlightthickness=0, width=395,
                                      bg=self._colors['bg_main'])
        right_scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL,
                                        command=self.right_canvas.yview)
        self.right_scroll_frame = ttk.Frame(self.right_canvas)

        self.right_scroll_frame.bind("<Configure>",
            lambda e: self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all")))

        self.right_canvas_window = self.right_canvas.create_window(
            (0, 0), window=self.right_scroll_frame, anchor="nw", tags="right_scroll_frame")

        # 让内部 frame 宽度随 Canvas 变化
        def _configure_scroll_frame_width(event):
            self.right_canvas.itemconfig(self.right_canvas_window, width=event.width)
        self.right_canvas.bind("<Configure>", _configure_scroll_frame_width)

        # 鼠标滚轮滚动
        def _on_mousewheel(event):
            self.right_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.right_canvas.bind("<Enter>", lambda e: self.right_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        self.right_canvas.bind("<Leave>", lambda e: self.right_canvas.unbind_all("<MouseWheel>"))

        self.right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 所有控件放到可滚动的 inner frame 中
        inner = self.right_scroll_frame
        self._build_ctrl_panel(inner)
        self._build_preprocess_panel(inner)
        self._build_point_list(inner)
        self._build_analysis_panel(inner)
        self._build_export_panel(inner)

        # ---- 状态栏 (固定在窗口底部，不随滚动) ----
        self.status = ttk.Label(right_frame,
                                text="就绪  |  [A]添加  [D]删除  [R]参考  [C]校准  [ESC]取消  |  Ctrl+Z撤销",
                                style='Status.TLabel', anchor=tk.W, wraplength=390)
        self.status.pack(fill=tk.X, side=tk.BOTTOM, before=self.right_canvas)

    def _build_ctrl_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="🖼 图像与标记", padding=10, style='Card.TLabelframe')
        frame.pack(fill=tk.X, padx=6, pady=4)

        ttk.Button(frame, text="📂 导入图像", command=self.load_image,
                   style='Accent.TButton').pack(fill=tk.X, pady=(0, 4))

        # 自动检测
        detect_frame = ttk.Frame(frame)
        detect_frame.pack(fill=tk.X, pady=2)
        ttk.Button(detect_frame, text="🔍 自动识别", command=self.auto_detect_points,
                   style='Accent.TButton').pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(detect_frame, text="📐 校准", command=self._start_calibrate_from_button,
                   style='Outline.TButton').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        ttk.Button(detect_frame, text="⚙", command=self._show_detect_params,
                   style='Outline.TButton', width=3).pack(side=tk.RIGHT)

        # 检测区域 & 缩放
        roi_zoom_frame = ttk.Frame(frame)
        roi_zoom_frame.pack(fill=tk.X, pady=(2, 0))
        self.btn_roi = ttk.Button(roi_zoom_frame, text="⬜ 选区",
                                   command=self._toggle_roi_mode, style='Outline.TButton')
        self.btn_roi.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.btn_polygon = ttk.Button(roi_zoom_frame, text="✏️ 自由选区",
                                       command=self._toggle_polygon_mode, style='Outline.TButton')
        self.btn_polygon.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        ttk.Button(roi_zoom_frame, text="🔲 清除",
                   command=self._clear_detect_roi, style='Outline.TButton').pack(
                   side=tk.LEFT, fill=tk.X, expand=True)

        # 清除按钮行
        clear_frame = ttk.Frame(frame)
        clear_frame.pack(fill=tk.X, pady=2)
        ttk.Button(clear_frame, text="🗑 清除所有点",
                   command=self.clear_points, style='Outline.TButton').pack(
                   side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(clear_frame, text="🔄 重置缩放",
                   command=self._reset_zoom, style='Outline.TButton').pack(
                   side=tk.RIGHT, fill=tk.X, expand=True, padx=(3, 0))

        # 鼠标模式
        mode_frame = ttk.LabelFrame(frame, text="鼠标模式", padding=6)
        mode_frame.pack(fill=tk.X, pady=4)
        self.mode_var = tk.StringVar(value="add")
        modes = [("➕ 添加 [A]", "add"), ("➖ 删除 [D]", "delete"),
                 ("🎯 参考 [R]", "select_ref"), ("📐 校准 [C]", "calibrate")]
        for text, val in modes:
            ttk.Radiobutton(mode_frame, text=text, variable=self.mode_var,
                            value=val, command=self.set_mode).pack(side=tk.LEFT, padx=3)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="↩ 撤销 [Ctrl+Z]", command=self.undo_last,
                   style='Outline.TButton').pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(btn_frame, text="🧹 去重", command=self.deduplicate_points,
                   style='Outline.TButton').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)

        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=4)
        ttk.Button(frame, text="⚠ 重置 (回到初始状态)", command=self.reset_all,
                   style='Danger.TButton').pack(fill=tk.X)

    def _build_preprocess_panel(self, parent):
        """图像预处理面板 — 滤波方法选择 & 参数调节 & 即时预览"""
        frame = ttk.LabelFrame(parent, text="🔬 图像预处理", padding=10, style='Card.TLabelframe')
        frame.pack(fill=tk.X, padx=6, pady=4)

        # 方法选择
        method_row = ttk.Frame(frame)
        method_row.pack(fill=tk.X, pady=2)
        ttk.Label(method_row, text="滤波方法:").pack(side=tk.LEFT)
        self.preprocess_method_var = tk.StringVar(value="无")
        # 根据 butter.py 是否可用动态生成方法列表
        if HRTEMFilter is not None:
            methods = ["无", "Gaussian 高斯模糊", "Median 中值滤波",
                       "Butterworth 低通", "Wiener 维纳滤波", "ABSF 滤波"]
        else:
            methods = ["无", "Gaussian 高斯模糊", "Median 中值滤波",
                       "Butterworth 低通 (不可用)", "Wiener 维纳滤波 (不可用)", "ABSF 滤波 (不可用)"]
        method_cb = ttk.Combobox(method_row, textvariable=self.preprocess_method_var,
                                 values=methods, state="readonly", width=22)
        method_cb.pack(side=tk.LEFT, padx=4)
        method_cb.bind("<<ComboboxSelected>>", self._on_preprocess_method_changed)

        # 动态参数区域
        self.preprocess_param_frame = ttk.Frame(frame)
        self.preprocess_param_frame.pack(fill=tk.X, pady=4)
        ttk.Label(self.preprocess_param_frame, text="选择滤波方法后在此调整参数",
                  foreground="gray", font=('', 8)).pack(anchor=tk.W, pady=2)

        # 按钮行
        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=3)
        self.btn_apply_preprocess = ttk.Button(btn_row, text="🔄 应用预览",
                                                command=self.apply_preprocess,
                                                style='Accent.TButton')
        self.btn_apply_preprocess.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.btn_toggle_display = ttk.Button(btn_row, text="👁 显示: 原始",
                                              command=self._toggle_display,
                                              style='Outline.TButton')
        self.btn_toggle_display.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        # 使用预处理图像进行检测
        ttk.Checkbutton(frame, text="使用预处理图像进行原子识别",
                        variable=self.use_preprocessed).pack(anchor=tk.W, pady=(4, 0))

        # 状态
        self.preprocess_status = ttk.Label(frame, text="状态: 未应用预处理",
                                           foreground="gray", font=('', 8),
                                           wraplength=320, justify=tk.LEFT)
        self.preprocess_status.pack(anchor=tk.W, pady=(4, 0))
        if HRTEMFilter is None and _HRTEM_IMPORT_NOTE:
            self.preprocess_status.config(
                text=f"状态: 未应用预处理（Butterworth/Wiener/ABSF 不可用）\n{_HRTEM_IMPORT_NOTE}")

    def _on_preprocess_method_changed(self, event=None):
        """根据选择的预处理方法动态切换参数面板"""
        for w in self.preprocess_param_frame.winfo_children():
            w.destroy()

        method_map = {
            "无": "none", "Gaussian 高斯模糊": "gaussian", "Median 中值滤波": "median",
            "Butterworth 低通": "butterworth", "Wiener 维纳滤波": "wiener", "ABSF 滤波": "absf",
            "Butterworth 低通 (不可用)": "butterworth", "Wiener 维纳滤波 (不可用)": "wiener",
            "ABSF 滤波 (不可用)": "absf",
        }
        method = method_map.get(self.preprocess_method_var.get(), "none")
        self.preprocess_method = method

        if method == "none":
            ttk.Label(self.preprocess_param_frame, text="选择滤波方法后在此调整参数",
                      foreground="gray", font=('', 8)).pack(anchor=tk.W, pady=2)
        elif method == "gaussian":
            row = ttk.Frame(self.preprocess_param_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text="Sigma:", width=10).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=self.preprocess_params['gaussian_sigma'])
            lbl = ttk.Label(row, text=f"{var.get():.1f}", width=4)
            var.trace_add("write", lambda *a, v=var, l=lbl:
                          l.config(text=f"{v.get():.1f}"))
            ttk.Scale(row, from_=0.1, to=10.0, variable=var, orient=tk.HORIZONTAL,
                      length=200, command=lambda v, vv=var:
                      self.preprocess_params.update({'gaussian_sigma': float(v)})).pack(
                      side=tk.LEFT, padx=2)
            lbl.pack(side=tk.LEFT)
            ttk.Label(self.preprocess_param_frame, text="越大越模糊，降低噪声",
                      foreground="gray", font=('', 8)).pack(anchor=tk.W, pady=2)
        elif method == "median":
            row = ttk.Frame(self.preprocess_param_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text="核尺寸:", width=10).pack(side=tk.LEFT)
            var = tk.IntVar(value=self.preprocess_params['median_size'])
            # 使用 trace_add 确保键盘输入和箭头点击都能同步参数
            var.trace_add("write", lambda *a, v=var:
                          self.preprocess_params.update({'median_size': v.get()}))
            ttk.Spinbox(row, from_=3, to=9, increment=2, textvariable=var,
                        width=4).pack(side=tk.LEFT, padx=4)
            ttk.Label(self.preprocess_param_frame, text="奇数 (3/5/7/9)，去除椒盐噪声",
                      foreground="gray", font=('', 8)).pack(anchor=tk.W, pady=2)
        elif method == "butterworth":
            row1 = ttk.Frame(self.preprocess_param_frame)
            row1.pack(fill=tk.X, pady=1)
            ttk.Label(row1, text="阶数:", width=10).pack(side=tk.LEFT)
            var_order = tk.IntVar(value=self.preprocess_params['bw_order'])
            var_order.trace_add("write", lambda *a, v=var_order:
                                self.preprocess_params.update({'bw_order': v.get()}))
            ttk.Spinbox(row1, from_=1, to=20, textvariable=var_order, width=4).pack(
                side=tk.LEFT, padx=4)
            ttk.Label(row1, text="越陡峭", foreground="gray", font=('', 8)).pack(side=tk.LEFT)
            row2 = ttk.Frame(self.preprocess_param_frame)
            row2.pack(fill=tk.X, pady=1)
            ttk.Label(row2, text="截止半径:", width=10).pack(side=tk.LEFT)
            var_cut = tk.DoubleVar(value=self.preprocess_params['bw_cutoff'])
            lbl2 = ttk.Label(row2, text=f"{var_cut.get():.2f}", width=5)
            var_cut.trace_add("write", lambda *a, v=var_cut, l=lbl2:
                              l.config(text=f"{v.get():.2f}"))
            ttk.Scale(row2, from_=0.01, to=1.0, variable=var_cut, orient=tk.HORIZONTAL,
                      length=180, command=lambda v: self.preprocess_params.update(
                          {'bw_cutoff': float(v)})).pack(side=tk.LEFT, padx=2)
            lbl2.pack(side=tk.LEFT)
            ttk.Label(self.preprocess_param_frame, text="(×图像半对角线)  越小越平滑",
                      foreground="gray", font=('', 8)).pack(anchor=tk.W, pady=2)
        elif method in ("wiener", "absf"):
            row1 = ttk.Frame(self.preprocess_param_frame)
            row1.pack(fill=tk.X, pady=1)
            ttk.Label(row1, text="Delta %:", width=10).pack(side=tk.LEFT)
            var_d = tk.DoubleVar(value=self.preprocess_params['delta'])
            lbl_d = ttk.Label(row1, text=f"{var_d.get():.1f}%", width=6)
            var_d.trace_add("write", lambda *a, v=var_d, l=lbl_d:
                            l.config(text=f"{v.get():.1f}%"))
            ttk.Scale(row1, from_=1, to=50, variable=var_d, orient=tk.HORIZONTAL,
                      length=180, command=lambda v: self.preprocess_params.update(
                          {'delta': float(v)})).pack(side=tk.LEFT, padx=2)
            lbl_d.pack(side=tk.LEFT)
            row2 = ttk.Frame(self.preprocess_param_frame)
            row2.pack(fill=tk.X, pady=1)
            ttk.Label(row2, text="截止半径:", width=10).pack(side=tk.LEFT)
            var_cut = tk.DoubleVar(value=self.preprocess_params['bw_cutoff'])
            lbl2 = ttk.Label(row2, text=f"{var_cut.get():.2f}", width=5)
            var_cut.trace_add("write", lambda *a, v=var_cut, l=lbl2:
                              l.config(text=f"{v.get():.2f}"))
            ttk.Scale(row2, from_=0.01, to=1.0, variable=var_cut, orient=tk.HORIZONTAL,
                      length=180, command=lambda v: self.preprocess_params.update(
                          {'bw_cutoff': float(v)})).pack(side=tk.LEFT, padx=2)
            lbl2.pack(side=tk.LEFT)
            # 迭代次数和步长参数
            row3 = ttk.Frame(self.preprocess_param_frame)
            row3.pack(fill=tk.X, pady=1)
            ttk.Label(row3, text="迭代次数:", width=10).pack(side=tk.LEFT)
            var_cyc = tk.IntVar(value=self.preprocess_params['cycles'])
            var_cyc.trace_add("write", lambda *a, v=var_cyc:
                              self.preprocess_params.update({'cycles': v.get()}))
            ttk.Spinbox(row3, from_=1, to=999, textvariable=var_cyc, width=5).pack(
                side=tk.LEFT, padx=4)
            ttk.Label(row3, text="步长:", width=4).pack(side=tk.LEFT, padx=(8, 0))
            var_step = tk.IntVar(value=self.preprocess_params['step'])
            var_step.trace_add("write", lambda *a, v=var_step:
                               self.preprocess_params.update({'step': v.get()}))
            ttk.Spinbox(row3, from_=1, to=20, textvariable=var_step, width=4).pack(
                side=tk.LEFT, padx=4)
            name = "Wiener 维纳滤波" if method == "wiener" else "ABSF 滤波"
            ttk.Label(self.preprocess_param_frame,
                      text=f"{name} — Delta越大去噪越强，截止半径控制低通范围",
                      foreground="gray", font=('', 8)).pack(anchor=tk.W, pady=2)

    def _toggle_display(self):
        """切换显示原图 / 预处理图像"""
        if self.processed_image is None:
            messagebox.showinfo("提示", "请先点击「应用预览」生成预处理图像。")
            return
        self.show_processed = not self.show_processed
        if self.show_processed:
            self.btn_toggle_display.config(text="👁 显示: 预处理")
        else:
            self.btn_toggle_display.config(text="👁 显示: 原始")
        self.refresh_display()

    def _get_display_image(self):
        """返回当前应显示的图像"""
        if self.show_processed and self.processed_image is not None:
            return self.processed_image
        return self.image

    def _get_work_image(self, use_preprocessed=None):
        """返回用于原子检测的图像（可能为预处理图像）。

        use_preprocessed: 已在主线程快照的布尔值。后台 worker 线程必须传入
        快照，不得经 None 分支现场读取 tk 变量（Tk 非线程安全，工单75：
        实测主循环未派发时跨线程 .get() 抛 RuntimeError）；None 仅限
        主线程回调（如校准流程）使用。
        """
        if use_preprocessed is None:
            use_preprocessed = self.use_preprocessed.get()
        if use_preprocessed and self.processed_image is not None:
            return self.processed_image
        return self.image

    def _get_filter_processor(self):
        """Lazy init HRTEMFilter 实例"""
        if HRTEMFilter is None:
            raise ImportError(_HRTEM_IMPORT_NOTE or "butter.py 模块不可用，预处理功能被禁用。")
        if self._filter_proc is None:
            self._filter_proc = HRTEMFilter()
        return self._filter_proc

    def apply_preprocess(self):
        """执行预处理并即时显示结果"""
        if self.image is None:
            messagebox.showinfo("提示", "请先导入图像。")
            return

        method = self.preprocess_method
        if method == "none":
            messagebox.showinfo("提示", "请先选择一种滤波方法。")
            return

        self.status.config(text=f"预处理中 ({method})...")
        self.root.update_idletasks()

        try:
            if method == "gaussian":
                result = self._apply_gaussian(self.image)
            elif method == "median":
                result = self._apply_median(self.image)
            elif method == "butterworth":
                result = self._apply_butterworth(self.image)
            elif method == "wiener":
                result = self._apply_wiener(self.image)
            elif method == "absf":
                result = self._apply_absf(self.image)
            else:
                return

            self.processed_image = result
            self.show_processed = True
            self.btn_toggle_display.config(text="👁 显示: 预处理")
            self.preprocess_status.config(
                text=f"状态: 已应用 {method}  |  "
                     f"范围 [{result.min():.2f}, {result.max():.2f}]",
                foreground="#006600")
            self.refresh_display()
            self.status.config(text=f"✅ 预处理完成 ({method}) — 可切换显示对比效果")

        except Exception as e:
            messagebox.showerror("预处理错误", f"滤波失败:\n{str(e)}")
            self.status.config(text="预处理失败")
            import traceback
            traceback.print_exc()

    def _apply_gaussian(self, img):
        """高斯模糊"""
        sigma = self.preprocess_params['gaussian_sigma']
        from scipy.ndimage import gaussian_filter as gf
        return gf(img, sigma=sigma)

    def _apply_median(self, img):
        """中值滤波"""
        size = self.preprocess_params['median_size']
        from scipy.ndimage import median_filter as mf
        return mf(img, size=size)

    def _apply_butterworth(self, img):
        """Butterworth 低通滤波 (FFT-based, 使用 butter.py)"""
        return self._pad_and_filter(img, {
            'delta': 0,                      # 纯 Butterworth
            'apply_bw_filter': True,
            'apply_wiener': False,
            'apply_absf': False,
            'bw_order': self.preprocess_params['bw_order'],
            'bw_ro': self.preprocess_params['bw_cutoff'],
            'stem_filter': False,
        }, 'butterworth_filtered')

    def _apply_wiener(self, img):
        """Wiener 滤波 (使用 butter.py 的完整 Kilaas 流程)"""
        return self._pad_and_filter(img, {
            'delta': self.preprocess_params['delta'],
            'apply_bw_filter': True,
            'apply_wiener': True,
            'apply_absf': False,
            'bw_order': self.preprocess_params['bw_order'],
            'bw_ro': self.preprocess_params['bw_cutoff'],
            'cycles': self.preprocess_params['cycles'],
            'step': self.preprocess_params['step'],
            'stem_filter': False,
        }, 'wiener_filtered')

    def _apply_absf(self, img):
        """ABSF 滤波 (使用 butter.py 的完整 Kilaas 流程)"""
        return self._pad_and_filter(img, {
            'delta': self.preprocess_params['delta'],
            'apply_bw_filter': True,
            'apply_wiener': False,
            'apply_absf': True,
            'bw_order': self.preprocess_params['bw_order'],
            'bw_ro': self.preprocess_params['bw_cutoff'],
            'cycles': self.preprocess_params['cycles'],
            'step': self.preprocess_params['step'],
            'stem_filter': False,
        }, 'absf_filtered')

    def _pad_and_filter(self, img, params_override, result_key):
        """
        将图像填充到 2^n 正方形 → 滤波 → 裁剪回原始尺寸
        butter.py 的 process_image 内部会将图像裁剪到 min(h,w) 的 2^n 正方形，
        因此先填充到 2^n 正方形再传入，输出后裁剪回原始 ROI 尺寸。
        """
        h, w = img.shape
        max_dim = max(h, w)
        # 找到下一个 2^n
        padded_size = 1
        while padded_size < max_dim:
            padded_size *= 2

        # 镜像填充到正方形
        pad_top = (padded_size - h) // 2
        pad_bottom = padded_size - h - pad_top
        pad_left = (padded_size - w) // 2
        pad_right = padded_size - w - pad_left
        padded = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode='reflect')

        # 设置滤波参数
        proc = self._get_filter_processor()
        for k, v in params_override.items():
            proc.params[k] = v

        # 执行滤波
        results = proc.process_image(padded.astype(np.float32).copy())

        # 获取结果
        if result_key not in results:
            # fallback: 尝试其他可能的键
            for fallback in ['wiener_filtered', 'absf_filtered', 'butterworth_filtered', 'original']:
                if fallback in results:
                    result = results[fallback]
                    break
            else:
                raise ValueError(f"滤波未生成 {result_key}，可用: {list(results.keys())}")
        else:
            result = results[result_key]

        # 裁剪回原始尺寸（居中裁剪）
        y_start = (padded_size - h) // 2
        x_start = (padded_size - w) // 2
        cropped = result[y_start:y_start + h, x_start:x_start + w]

        # 归一化到 0-1
        lo, hi = cropped.min(), cropped.max()
        if hi > lo:
            cropped = (cropped - lo) / (hi - lo)

        return cropped

    def _build_point_list(self, parent):
        frame = ttk.LabelFrame(parent, text="📍 原子点列表", padding=6, style='Card.TLabelframe')
        frame.pack(fill=tk.X, padx=6, pady=4)

        list_toolbar = ttk.Frame(frame)
        list_toolbar.pack(fill=tk.X)
        ttk.Label(list_toolbar, text="点击列表项可定位高亮",
                  foreground=self._colors['text_secondary'],
                  font=self._colors['font_small']).pack(side=tk.LEFT)

        self.point_listbox = tk.Listbox(frame, height=5, font=('Consolas', 9),
                                        bg='white', fg='#212121',
                                        selectbackground='#1565c0', selectforeground='white',
                                        relief='flat', bd=1, highlightthickness=1,
                                        highlightcolor='#1565c0', highlightbackground='#e0e0e0')
        self.point_listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, pady=(2, 0))
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.point_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=(2, 0))
        self.point_listbox.config(yscrollcommand=scrollbar.set)
        self.point_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

    def _build_analysis_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="📊 分析", padding=10, style='Card.TLabelframe')
        frame.pack(fill=tk.X, padx=6, pady=4)

        # 参考矢量
        ttk.Label(frame, text="① 参考晶格矢量（请在无应变参考区选择）", font=self._colors['font_title']).pack(anchor=tk.W)
        ref_row = ttk.Frame(frame)
        ref_row.pack(fill=tk.X, pady=3)
        ttk.Button(ref_row, text="🎯 鼠标选择 [R]", command=self.start_ref_select,
                   style='Accent.TButton').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(ref_row, text="✏ 手动输入", command=self.set_reference_manual,
                   style='Outline.TButton').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        ttk.Button(frame, text="🎯 双方向选点平均（推荐）", command=self.start_ref_multi_select,
                   style='Outline.TButton').pack(fill=tk.X, pady=(0, 3))

        self.ref_status_label = ttk.Label(frame, text="未设置",
                                          foreground=self._colors['text_secondary'],
                                          font=self._colors['font_small'])
        self.ref_status_label.pack(anchor=tk.W, pady=(0, 2))

        # 参考区 (基矢精化) — 仅用区内原子拟合基矢, 不吸收区外真实应变
        region_row = ttk.Frame(frame)
        region_row.pack(fill=tk.X, pady=(0, 2))
        self.btn_ref_region = ttk.Button(region_row, text="🧭 框选参考区",
                                          command=self._toggle_ref_region_mode,
                                          style='Outline.TButton')
        self.btn_ref_region.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(region_row, text="✖ 清除", width=7,
                   command=self._clear_ref_region,
                   style='Outline.TButton').pack(side=tk.LEFT)
        self.ref_region_label = ttk.Label(
            frame, text="未设参考区 — 可用双方向选点平均，或框选 ≥10 个原子的无应变区",
            foreground="gray", font=('', 7), wraplength=360)
        self.ref_region_label.pack(anchor=tk.W, pady=(0, 6))

        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=4)

        # 运行分析
        ttk.Label(frame, text="② 原子位移与应变分析", font=self._colors['font_title']).pack(anchor=tk.W)
        self.analysis_method_var = tk.StringVar(value="局部 Peak Pairs (推荐)")
        method_cb = ttk.Combobox(frame, textvariable=self.analysis_method_var, state="readonly",
                                 values=("局部 Peak Pairs (推荐)", "兼容：晶格-CST（三角形）"),
                                 font=('', 8))
        method_cb.pack(fill=tk.X, pady=(2, 1))
        method_cb.bind("<<ComboboxSelected>>", self._on_analysis_method_changed)
        run_row = ttk.Frame(frame)
        run_row.pack(fill=tk.X, pady=4)
        ttk.Button(run_row, text="▶ 执行 PPA 位移与应变分析", command=self.run_ppa_analysis,
                   style='Success.TButton').pack(side=tk.LEFT, fill=tk.X, expand=True)
        # 长分析可中断: 使在飞后台任务代际失效, 其结果将被丢弃
        ttk.Button(run_row, text="⏹ 取消", command=self.cancel_current_job,
                   style='Warning.TButton', width=6).pack(side=tk.LEFT, padx=(4, 0))

        # von Mises 系数
        vm_row = ttk.Frame(frame)
        vm_row.pack(fill=tk.X, pady=2)
        ttk.Label(vm_row, text="von Mises 系数:", font=('', 8)).pack(side=tk.LEFT)
        self.vm_var = tk.StringVar(value=f"{self.von_mises_coeff:.4f}")
        self.vm_var.trace_add("write", lambda *a: self._update_vm_coeff())
        ttk.Entry(vm_row, textvariable=self.vm_var, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(vm_row, text="(平面应变严格值≈0.444, 经验值 0.667)",
                  foreground="gray", font=('', 7)).pack(side=tk.LEFT)

        # 显示选项 — 使用 Combobox 支持更多视图
        view_frame = ttk.Frame(frame)
        view_frame.pack(fill=tk.X, pady=2)
        ttk.Label(view_frame, text="视图:", font=('', 8)).pack(side=tk.LEFT, padx=(0, 4))
        self.view_var = tk.StringVar(value="colored")  # 内部 view key
        view_choices = [
            "位移幅值染色",
            "位移矢量场",
            "应变 ε_xx (正应变 x)",
            "应变 ε_yy (正应变 y)",
            "应变 ε_xy (剪应变)",
            "等效应变 ε_eq (von Mises)",
            "旋转 ω",
            "应变 ε_xx (GL 有限应变)",
            "应变 ε_yy (GL 有限应变)",
            "应变 ε_xy (GL 有限应变)",
            "等效应变 ε_eq (GL 有限应变)",
            "标记图",
        ]
        self.view_cb = ttk.Combobox(view_frame, values=view_choices,
                                     state="readonly", width=24)
        self.view_cb.pack(side=tk.LEFT)
        self.view_cb.bind("<<ComboboxSelected>>", self._on_view_combobox)
        # 初始化显示文本
        self.view_cb.set("位移幅值染色")

        # 色标范围控制
        cbar_frame = ttk.LabelFrame(frame, text="色标范围 (colorbar range)", padding=4)
        cbar_frame.pack(fill=tk.X, pady=(4, 0))

        cbar_row1 = ttk.Frame(cbar_frame)
        cbar_row1.pack(fill=tk.X, pady=1)
        self.colorbar_auto_var = tk.BooleanVar(value=True)  # True = 自动
        ttk.Radiobutton(cbar_row1, text="自动", variable=self.colorbar_auto_var,
                        value=True, command=self._on_colorbar_mode_changed).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(cbar_row1, text="手动", variable=self.colorbar_auto_var,
                        value=False, command=self._on_colorbar_mode_changed).pack(side=tk.LEFT, padx=8)

        cbar_row2 = ttk.Frame(cbar_frame)
        cbar_row2.pack(fill=tk.X, pady=1)
        ttk.Label(cbar_row2, text="Min:", font=('', 8), width=4).pack(side=tk.LEFT)
        self.cbar_min_entry = ttk.Entry(cbar_row2, textvariable=self.colorbar_vmin, width=10, state=tk.DISABLED)
        self.cbar_min_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(cbar_row2, text="Max:", font=('', 8), width=4).pack(side=tk.LEFT, padx=(4, 0))
        self.cbar_max_entry = ttk.Entry(cbar_row2, textvariable=self.colorbar_vmax, width=10, state=tk.DISABLED)
        self.cbar_max_entry.pack(side=tk.LEFT, padx=2)

        def _apply_colorbar_range():
            """点击应用按钮：刷新当前视图"""
            self._on_colorbar_mode_changed()
            # 如有分析结果，立即刷新视图以应用新范围
            if self.displacements is not None or self.strain_xx is not None:
                self._view_changed(self.view_var.get())
                self.status.config(text="色标范围已应用")

        ttk.Button(cbar_row2, text="应用", command=_apply_colorbar_range, width=5).pack(side=tk.LEFT, padx=(6, 0))

        # 箭头参数调节
        arrow_frame = ttk.Frame(frame)
        arrow_frame.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(arrow_frame, text="箭头放大:", font=('', 8)).pack(side=tk.LEFT)
        self.arrow_scale_var = tk.DoubleVar(value=1.0)
        ttk.Scale(arrow_frame, from_=0.2, to=8, variable=self.arrow_scale_var,
                  orient=tk.HORIZONTAL, length=90,
                  command=lambda v: self._update_arrow_params()).pack(side=tk.LEFT, padx=2)
        self.arrow_scale_label = ttk.Label(arrow_frame, text="1.0×", font=('', 8), width=4)
        self.arrow_scale_label.pack(side=tk.LEFT)

        ttk.Label(arrow_frame, text="线宽:", font=('', 8)).pack(side=tk.LEFT, padx=(6, 0))
        self.arrow_lw_var = tk.DoubleVar(value=1.5)
        ttk.Scale(arrow_frame, from_=0.5, to=5, variable=self.arrow_lw_var,
                  orient=tk.HORIZONTAL, length=60,
                  command=lambda v: self._update_arrow_params()).pack(side=tk.LEFT, padx=2)
        self.arrow_lw_label = ttk.Label(arrow_frame, text="1.5", font=('', 8), width=3)
        self.arrow_lw_label.pack(side=tk.LEFT)

        # 点样式: 原子点大小与配色 (作用于位移幅值染色 / 位移矢量场视图)
        dot_row = ttk.Frame(frame)
        dot_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(dot_row, text="点大小:", font=('', 8)).pack(side=tk.LEFT)
        self.point_size_var = tk.DoubleVar(value=self.atom_dot_size)
        ttk.Scale(dot_row, from_=5, to=120, variable=self.point_size_var,
                  orient=tk.HORIZONTAL, length=90,
                  command=lambda v: self._on_point_size_changed()).pack(side=tk.LEFT, padx=2)
        self.point_size_label = ttk.Label(dot_row, text=str(self.atom_dot_size), font=('', 8), width=3)
        self.point_size_label.pack(side=tk.LEFT)

        ttk.Label(dot_row, text="点配色:", font=('', 8)).pack(side=tk.LEFT, padx=(6, 0))
        self.point_cmap_var = tk.StringVar(value=POINT_CMAP_CHOICES[0])
        self.point_cmap_cb = ttk.Combobox(dot_row, textvariable=self.point_cmap_var,
                                          values=POINT_CMAP_CHOICES,
                                          state="readonly", width=11)
        self.point_cmap_cb.pack(side=tk.LEFT, padx=2)
        self.point_cmap_cb.bind("<<ComboboxSelected>>", lambda e: self._on_point_cmap_changed())

        # 全原子箭头选项
        self.show_all_arrows_var = tk.BooleanVar(value=False)
        def _on_show_all_arrows_toggle():
            self.show_all_arrows = self.show_all_arrows_var.get()
            # 位移/矢量视图时立即刷新
            v = self.view_var.get()
            if v in ("colored", "vector"):
                self._view_changed(v)
        ttk.Checkbutton(frame, text="全原子箭头 (所有原子显示位移矢量)",
                        variable=self.show_all_arrows_var,
                        command=_on_show_all_arrows_toggle).pack(anchor=tk.W, pady=(2, 0))

        # 应变云图样式 — 作用于全部插值应变云图视图 (ε_xx/ε_yy/ε_xy/ε_eq/ω 及 GL 分量)
        cloud_frame = ttk.LabelFrame(frame, text="应变云图样式", padding=4)
        cloud_frame.pack(fill=tk.X, pady=(4, 0))

        cmap_row = ttk.Frame(cloud_frame)
        cmap_row.pack(fill=tk.X, pady=1)
        ttk.Label(cmap_row, text="颜色:", font=('', 8)).pack(side=tk.LEFT)
        self.cloud_cmap_var = tk.StringVar(value=STRAIN_CLOUD_AUTO_CMAP_LABEL)
        self.cloud_cmap_cb = ttk.Combobox(cmap_row, textvariable=self.cloud_cmap_var,
                                          values=STRAIN_CLOUD_CMAP_CHOICES,
                                          state="readonly", width=13)
        self.cloud_cmap_cb.pack(side=tk.LEFT, padx=2)
        self.cloud_cmap_cb.bind("<<ComboboxSelected>>", lambda e: self._on_cloud_style_changed())

        alpha_row = ttk.Frame(cloud_frame)
        alpha_row.pack(fill=tk.X, pady=1)
        ttk.Label(alpha_row, text="不透明度:", font=('', 8)).pack(side=tk.LEFT)
        self.cloud_alpha_var = tk.DoubleVar(value=self.strain_cloud_alpha)
        ttk.Scale(alpha_row, from_=0.0, to=1.0, variable=self.cloud_alpha_var,
                  orient=tk.HORIZONTAL, length=90,
                  command=lambda v: self._on_cloud_alpha_changed()).pack(side=tk.LEFT, padx=2)
        self.cloud_alpha_label = ttk.Label(alpha_row, text=f"{self.strain_cloud_alpha:.2f}", font=('', 8), width=4)
        self.cloud_alpha_label.pack(side=tk.LEFT)

    def _build_export_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="💾 导出结果", padding=10, style='Card.TLabelframe')
        frame.pack(fill=tk.X, padx=6, pady=4)

        # 导出按钮网格
        export_grid = ttk.Frame(frame)
        export_grid.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(export_grid, text="📄 原子坐标 CSV", command=self.save_coordinates,
                   style='Outline.TButton').grid(row=0, column=0, sticky='ew', padx=2, pady=2)
        ttk.Button(export_grid, text="🖼 标记图 PNG", command=self.save_marked_image,
                   style='Outline.TButton').grid(row=0, column=1, sticky='ew', padx=2, pady=2)
        ttk.Button(export_grid, text="📊 当前视图 PNG", command=self.save_current_view,
                   style='Outline.TButton').grid(row=1, column=0, sticky='ew', padx=2, pady=2)
        ttk.Button(export_grid, text="➡ 位移 CSV", command=self.save_displacement_csv,
                   style='Outline.TButton').grid(row=1, column=1, sticky='ew', padx=2, pady=2)
        ttk.Button(export_grid, text="📐 应变张量 CSV", command=self.save_strain_csv,
                   style='Outline.TButton').grid(row=2, column=0, columnspan=2, sticky='ew', padx=2, pady=2)
        export_grid.columnconfigure(0, weight=1)
        export_grid.columnconfigure(1, weight=1)

        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=4)

        # 项目保存/加载
        proj_frame = ttk.Frame(frame)
        proj_frame.pack(fill=tk.X)
        ttk.Button(proj_frame, text="💾 保存项目", command=self.save_project,
                   style='Accent.TButton').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(proj_frame, text="📂 加载项目", command=self.load_project,
                   style='Outline.TButton').pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

    # ================================================================
    #  快捷键
    # ================================================================
    def _bind_shortcuts(self):
        self.root.bind("<KeyPress-a>", lambda e: self._set_mode_shortcut("add"))
        self.root.bind("<KeyPress-d>", lambda e: self._set_mode_shortcut("delete"))
        self.root.bind("<KeyPress-r>", lambda e: self._set_mode_shortcut("select_ref"))
        self.root.bind("<KeyPress-c>", lambda e: self._set_mode_shortcut("calibrate"))
        self.root.bind("<Control-z>", lambda e: self.undo_last())
        self.root.bind("<Escape>", self._on_escape)
        self.root.bind("<Delete>", lambda e: self.clear_points())

    def _set_mode_shortcut(self, mode):
        self.mode_var.set(mode)
        self.set_mode()

    def _on_escape(self, event=None):
        if self.ref_select_indices:
            self.ref_select_indices = []
            self._refresh_overlay()
        if any(getattr(self, 'ref_multi_indices', [[], []])):
            self._reset_ref_multi_selection()
            self._refresh_overlay()
        if self._calibrate_pts:
            self._calibrate_pts.clear()
            self._refresh_overlay()
        if self._polygon_mode_active and self._polygon_vertices:
            self._polygon_vertices = []
            self._polygon_mode_active = False
            self.btn_polygon.config(text="✏️ 自由选区")
            self._refresh_overlay()
        # 取消参考区绘制模式(ESC) — 已完成的参考区保留, 用「✖ 清除」移除
        if self._ref_region_mode_active or self._drawing_ref_region is not None:
            self._ref_region_mode_active = False
            self._drawing_ref_region = None
            self._cleanup_ref_region_preview()
            self.btn_ref_region.config(text="🧭 框选参考区")
            self._refresh_overlay()
        # 取消已完成的自由选区(ESC)
        if self.detect_polygon is not None:
            self.detect_polygon = None
            self._refresh_overlay()
        try:
            self.root.unbind("<Return>")
        except Exception:
            pass
        self.selected_point_idx = None
        self.mode_var.set("add")
        self.set_mode()
        self.status.config(text="已取消当前操作")

    # ================================================================
    #  模式切换
    # ================================================================
    def set_mode(self):
        previous_mode = getattr(self, 'current_mode', None)
        self.current_mode = self.mode_var.get()
        # 清除参考选择状态、校准状态
        if self.current_mode != "select_ref" and self.ref_select_indices:
            self.ref_select_indices = []
            self._refresh_overlay()
        # 双方向锚点模式由独立按钮进入；点击任一鼠标模式单选项即退出。
        if previous_mode == "select_ref_multi" or any(getattr(self, 'ref_multi_indices', [[], []])):
            self._reset_ref_multi_selection()
            self._refresh_overlay()
        if self.current_mode != "calibrate":
            self._calibrate_pts.clear()
            try:
                self.root.unbind("<Return>")
            except Exception:
                pass
        # 直接进入参考选择模式
        if self.current_mode == "select_ref":
            self.start_ref_select(mode_switch=True)
            return
        if self.current_mode == "calibrate":
            self._start_calibrate()
            return
        hints = {"add": "左键添加点", "delete": "右键删除点", "calibrate": "左键点击相邻原子"}
        hint = hints.get(self.current_mode, "")
        self.status.config(foreground='', text=f"模式: {hint}  |  快捷键: [A]添加 [D]删除 [R]参考 [C]校准 [ESC]取消  |  Ctrl+Z撤销")

    def _prepare_reference_selection_mode(self):
        """Deactivate transient draw/calibration modes before reference picking."""
        self._calibrate_pts.clear()
        if self._ref_region_mode_active or self._drawing_ref_region is not None:
            self._ref_region_mode_active = False
            self._drawing_ref_region = None
            self._cleanup_ref_region_preview()
            self.btn_ref_region.config(text="🧭 框选参考区")
        if self._roi_mode_active or self._drawing_roi is not None:
            self._roi_mode_active = False
            self._drawing_roi = None
            self._cleanup_roi_preview()
            self.btn_roi.config(text="⬜ 选区")
        if self._polygon_mode_active:
            self._polygon_mode_active = False
            self._polygon_vertices = []
            self.btn_polygon.config(text="✏️ 自由选区")
        try:
            self.root.unbind("<Return>")
        except tk.TclError:
            pass

    def start_ref_select(self, mode_switch=False):
        if len(self.points) < 3:
            if not mode_switch:
                messagebox.showinfo("提示", "需要至少3个标记点: 原点 + a方向点 + b方向点。")
            self.mode_var.set("add")
            self.set_mode()
            return
        self._prepare_reference_selection_mode()
        self._reset_ref_multi_selection()
        self.ref_select_indices = []
        self.current_mode = "select_ref"
        self.mode_var.set("select_ref")
        self.status.config(foreground='blue',
                           text="🔵 请在无应变参考区依次点击3个点: [1]原点 → [2]a方向 → [3]b方向  |  [ESC]取消")
        self._refresh_overlay()

    def _reset_ref_multi_selection(self):
        """Clear transient anchor/auto-chain selection without changing saved vectors."""
        self.ref_multi_indices = [[], []]
        self.ref_multi_auto_indices = [[], []]
        self.ref_multi_stage = 0
        try:
            self.root.unbind("<Return>")
        except (AttributeError, tk.TclError):
            pass

    def start_ref_multi_select(self):
        """Select arbitrary anchors for two atom lines; intermediate atoms are automatic."""
        if len(self.points) < 3:
            messagebox.showinfo(
                "提示", "双方向选点至少需要 3 个已识别原子；每个方向任选至少 2 个，交点可以共用。")
            return
        self._prepare_reference_selection_mode()
        self.ref_select_indices = []
        self._reset_ref_multi_selection()
        self.current_mode = "select_ref_multi"
        # 复用“参考”单选项的视觉状态，但 click handling 由 current_mode 区分。
        self.mode_var.set("select_ref")
        self.root.bind("<Return>", self._advance_ref_multi_selection)
        self.status.config(
            foreground='#a35b00',
            text="🟠 a方向: 任选同一原子列上的 ≥2 个原子（可间隔、横/竖/斜均可） → "
                 "[Enter]自动纳入中间原子并切换到 b  |  右键撤销  |  [ESC]取消")
        self._refresh_overlay()

    # ================================================================
    #  图像加载与显示
    # ================================================================
    def load_image(self):
        path = filedialog.askopenfilename(title="选择TIFF图像",
                                          filetypes=[("TIFF 文件", "*.tif *.tiff"),
                                                     ("PNG 文件", "*.png"),
                                                     ("所有文件", "*.*")])
        if not path:
            return
        try:
            self._load_image_path(path)
        except ImageLoadError as error:
            # A channels-last RGB image is accepted by the core.  Any remaining
            # 3-D TIFF is a stack and must never be averaged as if it were RGB.
            try:
                import tifffile
                source = tifffile.imread(path)
            except Exception:
                source = None
            if source is not None and source.ndim == 3 and source.shape[-1] not in (3, 4):
                frame = simpledialog.askinteger(
                    "选择 TIFF 帧", f"检测到 {source.shape[0]} 帧 TIFF。\n请输入要分析的帧号（从 0 开始）：",
                    minvalue=0, maxvalue=int(source.shape[0] - 1), parent=self.root)
                if frame is None:
                    return
                try:
                    self._load_image_path(path, frame_index=frame)
                except ImageLoadError as frame_error:
                    messagebox.showerror("错误", f"无法加载图像：\n{frame_error}")
                return
            messagebox.showerror("错误", f"无法加载图像：\n{error}")

    def display_image(self):
        self._remove_colorbar()
        disp_img = self._get_display_image()
        if disp_img is not None:
            h, w = disp_img.shape
            if not hasattr(self, '_img_artist') or self._img_artist is None:
                self.ax.clear()
                self.ax.set_facecolor('#fafafa')
                self._img_artist = self.ax.imshow(disp_img, cmap='gray', origin='upper',
                                                  aspect='equal')
                self._image_shape = (h, w)
                self.ax.set_xlim(0, w)
                self.ax.set_ylim(h, 0)
                self.ax.tick_params(colors='#616161', labelsize=8)
                for spine in self.ax.spines.values():
                    spine.set_color('#bdbdbd')
            else:
                try:
                    self._img_artist.set_data(disp_img)
                    clim_min = np.percentile(disp_img, 1)
                    clim_max = np.percentile(disp_img, 99)
                    if clim_max > clim_min:
                        self._img_artist.set_clim(clim_min, clim_max)
                    # 仅当图像实际尺寸改变时重置视图（加载新图像时触发）
                    if (h, w) != getattr(self, '_image_shape', (0, 0)):
                        self._image_shape = (h, w)
                        self.ax.set_xlim(0, w)
                        self.ax.set_ylim(h, 0)
                except Exception:
                    self.ax.clear()
                    self._img_artist = self.ax.imshow(disp_img, cmap='gray', origin='upper',
                                                      aspect='equal')
                    self._image_shape = (h, w)
                    self.ax.set_xlim(0, w)
                    self.ax.set_ylim(h, 0)
        else:
            self.ax.clear()
            self._img_artist = None
        title_suffix = " [预处理]" if self.show_processed and self.processed_image is not None else ""
        self.ax.set_title(f"原子图像{title_suffix}", fontsize=11)
        self.redraw_points()
        self.canvas.draw_idle()

    def _refresh_overlay(self):
        """轻量刷新：只重绘叠加图层（原子点、选区、校准点等），不重置缩放/不变换视图"""
        # 从 ax 更新底图的数据（如果图像数据已更新，比如预处理切换）
        disp_img = self._get_display_image()
        if disp_img is not None and self._img_artist is not None:
            try:
                self._img_artist.set_data(disp_img)
                clim_min = np.percentile(disp_img, 1)
                clim_max = np.percentile(disp_img, 99)
                if clim_max > clim_min:
                    self._img_artist.set_clim(clim_min, clim_max)
            except Exception:
                pass
        self.redraw_points()
        self.canvas.draw_idle()

    def refresh_display(self):
        """智能刷新 — 应变视图时保留叠加层，普通视图直接更新底图"""
        current_view = self.view_var.get()
        if current_view.startswith("strain_") or current_view == "rotation":
            self._view_changed(current_view)
        else:
            self.display_image()

    # ================================================================
    #  原子点管理
    # ================================================================
    def redraw_points(self):
        # 清除旧标记
        for artist in self.ax.lines + self.ax.collections:
            if hasattr(artist, 'is_point_artist'):
                artist.remove()
        for artist in self.ax.patches:
            if hasattr(artist, 'is_point_artist'):
                artist.remove()
        for text in self.ax.texts:
            if hasattr(text, 'is_point_label'):
                text.remove()

        if not self.points:
            # 即使没有点，也画 ROI 框和校准点
            self._draw_calibrate_and_roi()
            self.update_listbox()
            return

        pts = np.array(self.points)
        n_pts = len(pts)

        # ---- 原子圆圈: 单个 EllipseCollection ----
        # 用 units='xy' 保持数据坐标半径语义 (缩放时圆圈随图放大, 与旧 Circle 一致),
        # 但只创建 1 个 artist 而非 N 个 —— 这是大规模图的主要交互瓶颈所在。
        diameters = np.full(n_pts, self.POINT_MARKER_RADIUS * 2.0)
        edge_colors = np.tile(to_rgba('lime'), (n_pts, 1))
        line_widths = np.full(n_pts, 1.2)
        sel = self.selected_point_idx
        if sel is not None and 0 <= sel < n_pts:
            edge_colors[sel] = to_rgba('cyan')
            line_widths[sel] = 2.0
        circles = EllipseCollection(
            widths=diameters, heights=diameters, angles=np.zeros(n_pts),
            units='xy', offsets=pts, offset_transform=self.ax.transData,
            facecolors='none', edgecolors=edge_colors, linewidths=line_widths)
        circles.is_point_artist = True
        self.ax.add_collection(circles)

        # ---- 编号标签: 视口内限量绘制 ----
        self._draw_point_labels(pts)

        # 高亮参考选择点
        if self.ref_select_indices:
            colors = ['#FFD700', '#FF8C00', '#00CED1']  # 原点黄, a方向橙, b方向青
            for order, sel_idx in enumerate(self.ref_select_indices):
                # 防御: 残留的越界索引 (点表已被修改) 只跳过, 不崩溃
                if not (0 <= sel_idx < len(self.points)):
                    continue
                x, y = self.points[sel_idx]
                c = Circle((x, y), radius=16, fill=False,
                           color=colors[order] if order < len(colors) else '#888888',
                           linewidth=2.5, linestyle='-')
                c.is_point_artist = True
                self.ax.add_patch(c)
                t = self.ax.text(x, y + 19, str(order + 1), color=colors[order],
                                 fontsize=12, ha='center', va='bottom', fontweight='bold',
                                 path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])
                t.is_point_label = True

        # 双方向参考选择：细实线/小圆显示自动纳入的完整原子列，
        # 大圆与 a1/b1 标签只表示用户手选的锚点。
        multi_groups = getattr(self, 'ref_multi_indices', [[], []])
        auto_groups = getattr(self, 'ref_multi_auto_indices', [[], []])
        for group_id, group in enumerate(multi_groups):
            color = '#FF8C00' if group_id == 0 else '#00AFC4'
            prefix = 'a' if group_id == 0 else 'b'
            auto_indices = [idx for idx in auto_groups[group_id]
                            if 0 <= idx < len(self.points)]
            if len(auto_indices) >= 2:
                full_chain = np.asarray(
                    [self.points[idx] for idx in auto_indices], dtype=float)
                line, = self.ax.plot(
                    full_chain[:, 0], full_chain[:, 1], '-', color=color,
                    linewidth=2.0, alpha=0.78, zorder=7)
                line.is_point_artist = True
                auto_markers = self.ax.scatter(
                    full_chain[:, 0], full_chain[:, 1], s=75,
                    facecolors='none', edgecolors=color, linewidths=1.3,
                    alpha=0.75, zorder=7)
                auto_markers.is_point_artist = True

            valid_indices = [idx for idx in group if 0 <= idx < len(self.points)]
            # 自动提取前，用虚线连接锚点以显示用户正在定义的直线。
            if len(valid_indices) >= 2 and len(auto_indices) < 2:
                anchors = np.asarray([self.points[idx] for idx in valid_indices], dtype=float)
                line, = self.ax.plot(anchors[:, 0], anchors[:, 1], '--', color=color,
                                     linewidth=1.5, alpha=0.75, zorder=7)
                line.is_point_artist = True
            for order, sel_idx in enumerate(valid_indices):
                x, y = self.points[sel_idx]
                marker = Circle((x, y), radius=13, fill=False, color=color,
                                linewidth=2.2, linestyle='-', zorder=8)
                marker.is_point_artist = True
                self.ax.add_patch(marker)
                label = self.ax.text(
                    x, y + 16, f"{prefix}{order + 1}", color=color,
                    fontsize=9, ha='center', va='bottom', fontweight='bold', zorder=9,
                    path_effects=[pe.withStroke(linewidth=1.4, foreground='black')])
                label.is_point_label = True

        # 校准点标记 & 检测 ROI (无论有无原子点都绘制)
        self._draw_calibrate_and_roi()

        self.update_listbox()

    def _draw_point_labels(self, pts):
        """
        绘制原子编号 —— 只画当前视口内的，且总数受 MAX_LABELS_DRAWN 限制。

        matplotlib 的 Text 无法批量化，逐个创建在大规模图上成本可观；而全图显示
        数千个编号时它们本就重叠不可读。因此小数据集保持原有行为 (全部绘制)，
        大数据集只在放大后绘制视口内编号，并在角落提示如何显示。
        """
        n_pts = len(pts)
        selected = self.selected_point_idx

        if n_pts <= self.MAX_LABELS_DRAWN:
            visible = np.arange(n_pts)          # 小数据集: 与旧行为完全一致
        else:
            x_lo, x_hi = sorted(self.ax.get_xlim())
            y_lo, y_hi = sorted(self.ax.get_ylim())
            inside = np.flatnonzero((pts[:, 0] >= x_lo) & (pts[:, 0] <= x_hi) &
                                    (pts[:, 1] >= y_lo) & (pts[:, 1] <= y_hi))
            if len(inside) > self.MAX_LABELS_DRAWN:
                # 视口内仍过密 → 不绘制编号, 只在角落给出提示 (成本 O(1))
                hint = self.ax.text(
                    0.01, 0.99,
                    f"编号已隐藏 (视口内 {len(inside)} 个点 > {self.MAX_LABELS_DRAWN})"
                    f"  |  Ctrl+滚轮放大以显示",
                    transform=self.ax.transAxes, color='#ffd54f', fontsize=8,
                    ha='left', va='top',
                    path_effects=[pe.withStroke(linewidth=2, foreground='black')])
                hint.is_point_label = True
                # 选中点的编号始终绘制, 便于列表定位
                visible = np.array([selected]) if (
                    selected is not None and 0 <= selected < n_pts) else np.empty(0, dtype=int)
            else:
                visible = inside

        for i in visible:
            x, y = pts[i]
            color = 'yellow' if i == selected else 'white'
            t = self.ax.text(x, y - 13, str(int(i) + 1), color=color, fontsize=9,
                             ha='center', va='top', fontweight='bold',
                             path_effects=[pe.withStroke(linewidth=2, foreground='black')])
            t.is_point_label = True

    def _draw_calibrate_and_roi(self):
        """绘制校准点和检测 ROI 框（独立于原子点的存在）"""
        if self._calibrate_pts:
            for i, (x, y) in enumerate(self._calibrate_pts):
                sc = self.ax.scatter(x, y, marker='D', c='#8844ff', s=80,
                                     edgecolor='white', linewidth=1.0, zorder=10)
                sc.is_point_artist = True
                t = self.ax.text(x, y - 16, str(i + 1), color='white', fontsize=8,
                                 ha='center', va='top', fontweight='bold',
                                 path_effects=[pe.withStroke(linewidth=1.5, foreground='#440088')])
                t.is_point_label = True

        if self.detect_roi is not None:
            import matplotlib.patches as mpatches
            x0, y0, x1, y1 = self.detect_roi
            rect = mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                fill=False, edgecolor='#00ccff', linewidth=1.8,
                linestyle='-', zorder=18)
            rect.is_point_artist = True
            self.ax.add_patch(rect)

        # 绘制自由选区多边形（已完成）
        if self.detect_polygon is not None:
            import matplotlib.patches as mpatches
            poly = np.array(self.detect_polygon)
            patch = mpatches.Polygon(poly, fill=True, alpha=0.15,
                                     facecolor='#ff8800', edgecolor='#ff8800',
                                     linewidth=1.8, linestyle='-', zorder=18)
            patch.is_point_artist = True
            self.ax.add_patch(patch)
        elif self._polygon_mode_active and len(self._polygon_vertices) >= 2:
            # 绘制中的临时多边形（未闭合）
            import matplotlib.patches as mpatches
            verts = np.array(self._polygon_vertices)
            patch = mpatches.Polygon(verts, fill=False, edgecolor='#ff8800',
                                     linewidth=1.8, linestyle='--', zorder=20)
            patch.is_point_artist = True
            self.ax.add_patch(patch)
            # 顶点标记
            sc = self.ax.scatter(verts[:, 0], verts[:, 1], marker='o',
                                 c='#ff8800', s=40, edgecolor='white',
                                 linewidth=1.0, zorder=21)
            sc.is_point_artist = True
            # 顶点编号
            for i, (x, y) in enumerate(verts):
                t = self.ax.text(x, y - 14, str(i + 1), color='#ff8800',
                                 fontsize=8, ha='center', va='top', fontweight='bold',
                                 path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])
                t.is_point_label = True

        # 晶格空洞 (疑似漏检) 红色 × 标记
        holes = getattr(self, 'lattice_holes', None)
        if holes:
            harr = np.asarray(holes)
            sc = self.ax.scatter(harr[:, 0], harr[:, 1], marker='x', c='red',
                                 s=70, linewidth=1.8, zorder=9)
            sc.is_point_artist = True

        # 参考区 (基矢精化范围) 绿色实线框
        if getattr(self, 'ref_region', None) is not None:
            import matplotlib.patches as mpatches
            rx0, ry0, rx1, ry1 = self.ref_region
            rect = mpatches.Rectangle(
                (rx0, ry0), rx1 - rx0, ry1 - ry0,
                fill=False, edgecolor='#00cc66', linewidth=1.8,
                linestyle='-', zorder=17)
            rect.is_point_artist = True
            self.ax.add_patch(rect)
            t = self.ax.text(rx0, ry0 - 4, "参考区", color='#00cc66', fontsize=8,
                             ha='left', va='bottom', fontweight='bold',
                             path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])
            t.is_point_label = True

        # 局部 PPA 几何不足位点：位移仍有效，应变为 NaN。灰色空心方块
        # 只表示无法可靠求局部应变，不表示该原子未参与分析。
        invalid = getattr(self, 'invalid_sites', None)
        if invalid is not None and len(invalid) > 0:
            iarr = np.asarray(invalid)
            sc = self.ax.scatter(iarr[:, 0], iarr[:, 1], marker='s', s=55,
                                 facecolors='none', edgecolors='#9e9e9e',
                                 linewidth=1.2, zorder=8)
            sc.is_point_artist = True

    def update_listbox(self):
        self.point_listbox.delete(0, tk.END)
        for i, (x, y) in enumerate(self.points):
            self.point_listbox.insert(tk.END, f"{i + 1:4d}: ({x:>8.2f}, {y:>8.2f})")

    def _on_listbox_select(self, event):
        sel = self.point_listbox.curselection()
        if sel:
            self.selected_point_idx = sel[0]
            self.redraw_points()
            self.canvas.draw_idle()

    # ================================================================
    #  鼠标交互
    # ================================================================
    def on_click(self, event):
        if event.inaxes != self.ax or self.image is None:
            return

        if self.current_mode == "select_ref_multi":
            if event.button == 1:
                self._handle_ref_multi_click(event)
            elif event.button == 3:
                self._undo_ref_multi_click()
            return

        if self.current_mode == "select_ref" and event.button == 1:
            self._handle_ref_select_click(event)
            return

        if self.current_mode == "calibrate" and event.button == 1:
            self._handle_calibrate_click(event)
            return

        # 参考区模式下，左键开始拖拽 / 右键取消本次拖拽
        if event.button == 1 and self._ref_region_mode_active:
            self._drawing_ref_region = (event.xdata, event.ydata)
            self.status.config(foreground='#00875a',
                text="🟩 拖拽鼠标 → 松手完成参考区  |  右键取消本次拖拽")
            return
        if event.button == 3 and self._ref_region_mode_active:
            self._drawing_ref_region = None
            self._cleanup_ref_region_preview()
            self._refresh_overlay()
            self.status.config(foreground='#00875a',
                text="🟩 参考区: 左键拖拽绘制  |  右键取消  |  [ESC]退出")
            return

        # ROI 选区模式下，左键开始拖拽
        if event.button == 1 and self._roi_mode_active:
            self._drawing_roi = (event.xdata, event.ydata)
            self.status.config(foreground='#0066cc',
                text="🟦 拖拽鼠标 → 松手完成选区  |  右键取消本次拖拽")
            return

        # ROI 选区模式下，右键取消本次拖拽
        if event.button == 3 and self._roi_mode_active:
            self._drawing_roi = None
            self._cleanup_roi_preview()
            self._refresh_overlay()
            self.status.config(foreground='#0066cc',
                text="🟦 选区: 左键拖拽绘制  |  右键取消  |  点按钮退出")
            return

        # 自由选区模式下，左键添加顶点
        if event.button == 1 and self._polygon_mode_active:
            self._add_polygon_vertex(event.xdata, event.ydata)
            return

        # 自由选区模式下，右键/中键闭合多边形
        if event.button in (2, 3) and self._polygon_mode_active:
            self._finish_polygon()
            return

        if event.button == 1 and self.current_mode == "add":
            x, y = event.xdata, event.ydata
            if self.add_point_with_check(x, y):
                self._clear_analysis_results()
                self._refresh_overlay()
            else:
                self.status.config(
                    foreground='orange',
                    text=f"⚠ 点距过近 (最近距 < {self.detect_min_dist*0.3:.1f} px)，已拒绝")

        elif event.button == 3 and self.current_mode == "delete":
            self._delete_nearest_point(event.xdata, event.ydata)

        elif event.button == 1 and self.current_mode == "delete":
            self._delete_nearest_point(event.xdata, event.ydata)

    def _delete_nearest_point(self, x, y):
        if not self.points:
            return
        pts = np.array(self.points)
        dist = np.sqrt((pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2)
        idx = np.argmin(dist)
        if dist[idx] < self.pick_radius:
            deleted = self.points[idx]
            self._record_undo("delete", (idx, deleted))
            del self.points[idx]
            if self.selected_point_idx == idx:
                self.selected_point_idx = None
            elif self.selected_point_idx is not None and idx < self.selected_point_idx:
                self.selected_point_idx -= 1
            self._clear_analysis_results()
            self._refresh_overlay()

    def on_mouse_move(self, event):
        if event.inaxes and self.image is not None:
            disp_img = self._get_display_image()
            x, y = int(np.floor(event.xdata)), int(np.floor(event.ydata))
            valid_coords = 0 <= x < disp_img.shape[1] and 0 <= y < disp_img.shape[0]

            # ROI 拖拽预览（不更新状态栏，保留模式提示）
            if self._drawing_roi is not None and event.xdata is not None:
                import matplotlib.patches as mpatches
                x0, y0 = self._drawing_roi
                x1, y1 = event.xdata, event.ydata
                if self._roi_rect is None:
                    self._roi_rect = mpatches.Rectangle(
                        (x0, y0), 1, 1,
                        fill=False, edgecolor='#00ccff', linewidth=1.5,
                        linestyle='--', zorder=20)
                    self.ax.add_patch(self._roi_rect)
                self._roi_rect.set_xy((min(x0, x1), min(y0, y1)))
                self._roi_rect.set_width(abs(x1 - x0))
                self._roi_rect.set_height(abs(y1 - y0))
                self.canvas.draw_idle()

            # 参考区拖拽预览 (绿色虚线)
            if self._drawing_ref_region is not None and event.xdata is not None:
                import matplotlib.patches as mpatches
                x0, y0 = self._drawing_ref_region
                x1, y1 = event.xdata, event.ydata
                if self._ref_region_rect is None:
                    self._ref_region_rect = mpatches.Rectangle(
                        (x0, y0), 1, 1,
                        fill=False, edgecolor='#00cc66', linewidth=1.5,
                        linestyle='--', zorder=20)
                    self.ax.add_patch(self._ref_region_rect)
                self._ref_region_rect.set_xy((min(x0, x1), min(y0, y1)))
                self._ref_region_rect.set_width(abs(x1 - x0))
                self._ref_region_rect.set_height(abs(y1 - y0))
                self.canvas.draw_idle()

            if valid_coords:
                val = disp_img[y, x]
                pts_info = ""
                # 性能优化：只在点数较少时计算最近点，避免大量原子时 UI 卡顿
                if self.points and len(self.points) <= 2000:
                    pts_arr = np.array(self.points)
                    dist = np.sqrt((pts_arr[:,0]-event.xdata)**2 + (pts_arr[:,1]-event.ydata)**2)
                    nearest = np.argmin(dist)
                    if dist[nearest] < self.pick_radius:
                        pts_info = f"  |  最近点: #{nearest+1}"
                # 节流: 鼠标移动高频触发, 逐次更新状态栏会造成 Tk 重绘抖动
                now = time.monotonic()
                if now - self._last_status_update >= 0.05:
                    self._last_status_update = now
                    self.status.config(text=f"x={x}  y={y}  I={val:.3f}{pts_info}")

    def on_pick(self, event):
        pass  # reserved

    def on_release(self, event):
        """鼠标释放 — 用于完成 ROI / 参考区绘制"""
        # 参考区拖拽完成 (一次性模式: 画完自动退出)
        if self._drawing_ref_region is not None and event.button == 1:
            self._cleanup_ref_region_preview()
            x0, y0 = self._drawing_ref_region
            x1, y1 = event.xdata, event.ydata
            if x1 is not None and y1 is not None:
                x_min, x_max = min(x0, x1), max(x0, x1)
                y_min, y_max = min(y0, y1), max(y0, y1)
                if x_max - x_min > 3 and y_max - y_min > 3:
                    self.ref_region = (x_min, y_min, x_max, y_max)
                    n_in = self._count_atoms_in_ref_region()
                    self._update_ref_region_label()
                    color = '#006600' if n_in >= 10 else '#cc6600'
                    note = "" if n_in >= 10 else "  ⚠ 不足 10 个, 精化将跳过"
                    self.status.config(
                        foreground=color,
                        text=f"✓ 参考区已设置: 区内 {n_in} 个原子{note}")
                else:
                    self.status.config(text="参考区太小 (<3px)，请重新拖拽")
            self._drawing_ref_region = None
            self._ref_region_mode_active = False
            self.btn_ref_region.config(text="🧭 框选参考区")
            self._refresh_overlay()
            return

        if self._drawing_roi is not None and event.button == 1:
            # 先清除拖拽中的预览矩形
            self._cleanup_roi_preview()

            x0, y0 = self._drawing_roi
            x1, y1 = event.xdata, event.ydata
            if x1 is not None and y1 is not None:
                x_min, x_max = min(x0, x1), max(x0, x1)
                y_min, y_max = min(y0, y1), max(y0, y1)
                if x_max - x_min > 3 and y_max - y_min > 3:
                    h, w = self.image.shape
                    self.detect_roi = (
                        max(0, int(x_min)), max(0, int(y_min)),
                        min(w - 1, int(x_max)), min(h - 1, int(y_max)))
                    self.status.config(
                        foreground='#006600',
                        text=f"✓ 检测区域已设置: [{x_min:.0f},{y_min:.0f}] → [{x_max:.0f},{y_max:.0f}]")
                else:
                    self.status.config(text="选区太小 (<3px)，请重新拖拽")
            self._drawing_roi = None  # 拖拽结束，但 ROI 模式保持激活
            self._refresh_overlay()

    def _on_scroll(self, event):
        """Ctrl + 滚轮缩放图像"""
        if not self._ctrl_pressed or self.image is None or event.inaxes != self.ax:
            return
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        scale = 0.85 if event.button == 'up' else 1.0 / 0.85

        cx, cy = event.xdata, event.ydata
        if cx is None or cy is None:
            return

        new_xlim = (cx - (cx - xlim[0]) * scale, cx + (xlim[1] - cx) * scale)
        new_ylim = (cy - (cy - ylim[0]) * scale, cy + (ylim[1] - cy) * scale)

        # clamp 到显示的图像范围内
        disp_img = self._get_display_image()
        h, w = disp_img.shape
        new_xlim = (max(-10, new_xlim[0]), min(w + 10, new_xlim[1]))
        new_ylim = (
            max(-10.0, min(h + 10.0, new_ylim[0])),
            max(-10.0, min(h + 10.0, new_ylim[1]))
        )

        if len(self._zoom_history) < 20:
            self._zoom_history.append((xlim, ylim))

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        # 大规模图的编号按视口显示, 缩放后需刷新叠加层才能出现/隐藏;
        # 小数据集编号已全部绘制, 只需重画 canvas。
        if len(self.points) > self.MAX_LABELS_DRAWN:
            self._refresh_overlay()
        else:
            self.canvas.draw_idle()

    def _reset_zoom(self):
        """重置缩放，回到全图视图"""
        self._zoom_history.clear()
        self._cleanup_roi_preview()
        # 不改变 current_mode 和 _roi_mode_active, 只重置视图范围
        self._remove_colorbar()
        self.ax.clear()
        self._img_artist = None
        self.display_image()
        self.status.config(text="缩放已重置")

    def _toggle_roi_mode(self):
        """切换 ROI 绘制模式"""
        if self.image is None:
            messagebox.showinfo("提示", "请先导入图像。")
            return
        if self._roi_mode_active:
            # 取消 ROI 模式
            self._roi_mode_active = False
            self._drawing_roi = None
            self._cleanup_roi_preview()
            self.btn_roi.config(text="⬜ 选区")
            self.status.config(text="选区模式已取消")
        else:
            # 进入 ROI 模式
            self._roi_mode_active = True
            self._drawing_roi = None
            self.btn_roi.config(text="🔲 退出选区")
            self.status.config(
                foreground='#0066cc',
                text="🟦 选区: 按住左键拖拽 → 松手完成  |  右键取消  |  再点按钮退出")
        self._refresh_overlay()

    def _cleanup_roi_preview(self):
        """安全清除拖拽预览矩形"""
        if self._roi_rect is not None:
            try:
                self._roi_rect.remove()
            except Exception:
                pass
            self._roi_rect = None

    def _clear_detect_roi(self):
        """清除检测区域（矩形+多边形）"""
        self.detect_roi = None
        self.detect_polygon = None
        self._polygon_mode_active = False
        self._polygon_vertices = []
        self._drawing_roi = None
        self._cleanup_roi_preview()
        self.btn_roi.config(text="⬜ 选区")
        self.btn_polygon.config(text="✏️ 自由选区")
        self._roi_mode_active = False
        self._refresh_overlay()
        self.status.config(text="检测区域已清除 — 将对全图进行识别")

    # ================================================================
    #  参考区 (基矢精化范围)
    # ================================================================
    def _toggle_ref_region_mode(self):
        """切换参考区绘制模式 — 拖拽一个无应变区域用于基矢最小二乘"""
        if self.image is None:
            messagebox.showinfo("提示", "请先导入图像。")
            return
        if self._ref_region_mode_active:
            self._ref_region_mode_active = False
            self._drawing_ref_region = None
            self._cleanup_ref_region_preview()
            self.btn_ref_region.config(text="🧭 框选参考区")
            self.status.config(text="参考区模式已取消")
        else:
            # 与其他拖拽模式互斥, 避免同时激活
            if self._roi_mode_active:
                self._roi_mode_active = False
                self._drawing_roi = None
                self._cleanup_roi_preview()
                self.btn_roi.config(text="⬜ 选区")
            self._ref_region_mode_active = True
            self._drawing_ref_region = None
            self.btn_ref_region.config(text="🔲 退出参考区")
            self.status.config(
                foreground='#00875a',
                text="🟩 参考区: 按住左键拖拽无应变区域 → 松手完成  |  右键取消  |  [ESC]退出")
        self._refresh_overlay()

    def _cleanup_ref_region_preview(self):
        """安全清除参考区拖拽预览矩形"""
        if self._ref_region_rect is not None:
            try:
                self._ref_region_rect.remove()
            except Exception:
                pass
            self._ref_region_rect = None

    def _clear_ref_region(self):
        """Clear the 2-D refinement region and retain the currently set vectors."""
        self.ref_region = None
        self._ref_region_mode_active = False
        self._drawing_ref_region = None
        self._cleanup_ref_region_preview()
        self.btn_ref_region.config(text="🧭 框选参考区")
        self._update_ref_region_label()
        self._refresh_overlay()
        self.status.config(text="参考区已清除 — 将直接使用当前设置的参考矢量")

    def _atoms_in_ref_region_mask(self):
        """返回参考区内原子的布尔掩码 (无参考区或无点时返回 None)"""
        if self.ref_region is None or not self.points:
            return None
        x0, y0, x1, y1 = self.ref_region
        pts = np.asarray(self.points, dtype=float)
        return ((pts[:, 0] >= x0) & (pts[:, 0] <= x1) &
                (pts[:, 1] >= y0) & (pts[:, 1] <= y1))

    def _count_atoms_in_ref_region(self):
        mask = self._atoms_in_ref_region_mask()
        return int(mask.sum()) if mask is not None else 0

    def _update_ref_region_label(self):
        if self.ref_region is None:
            metadata = getattr(self, 'reference_metadata', None) or {}
            method = metadata.get('method')
            if method == 'two-chain-robust-fit':
                a_count = int(metadata.get('a_point_count', 0))
                b_count = int(metadata.get('b_point_count', 0))
                a_anchors = int(metadata.get('a_anchor_count', a_count))
                b_anchors = int(metadata.get('b_anchor_count', b_count))
                text = (f"未设二维参考区 — 双方向选点自动拟合 "
                        f"(a手选{a_anchors}→纳入{a_count}点, "
                        f"b手选{b_anchors}→纳入{b_count}点)；仍可框选无应变区进一步二维精化")
                color = "#006600"
            elif method in {'three-point', 'three-point-by-id'}:
                text = "未设参考区 — 当前基矢由 3 点定义，建议改用双方向选点平均或框选 ≥10 个原子的无应变区"
                color = "#cc6600"
            elif method == 'manual':
                text = "未设参考区 — 当前使用手动晶格参数；可框选 ≥10 个原子的无应变区进行数据驱动精化"
                color = "gray"
            else:
                text = "未设参考区 — 可用双方向选点平均，或框选 ≥10 个原子的无应变区"
                color = "gray"
            self.ref_region_label.config(
                text=text, foreground=color)
            return
        x0, y0, x1, y1 = self.ref_region
        n_in = self._count_atoms_in_ref_region()
        if n_in >= 10:
            self.ref_region_label.config(
                text=f"参考区 [{x0:.0f},{y0:.0f}]→[{x1:.0f},{y1:.0f}]  区内 {n_in} 个原子  "
                     f"→ 分析时用区内原子精化基矢 (误差 ≈ 1%/√{n_in})",
                foreground="#006600")
        else:
            self.ref_region_label.config(
                text=f"参考区 [{x0:.0f},{y0:.0f}]→[{x1:.0f},{y1:.0f}]  区内仅 {n_in} 个原子 "
                     f"(<10)，精化将跳过；请扩大参考区",
                foreground="#cc6600")

    # ================================================================
    #  自由选区 (多边形) 绘制
    # ================================================================
    def _toggle_polygon_mode(self):
        """切换自由选区绘制模式 — 逐点点击多边形"""
        if self.image is None:
            messagebox.showinfo("提示", "请先导入图像。")
            return
        if self._polygon_mode_active:
            # 退出多边形模式
            if len(self._polygon_vertices) >= 3:
                # 有≥3个顶点：自动闭合为选区
                self._finish_polygon()
                return
            # 顶点不足3个：直接取消
            self._polygon_mode_active = False
            self._polygon_vertices = []
            self.btn_polygon.config(text="✏️ 自由选区")
            self.status.config(text="自由选区模式已取消")
        else:
            # 进入多边形模式 — 先清除矩形选区避免混淆
            if self._roi_mode_active:
                self._roi_mode_active = False
                self._drawing_roi = None
                self._cleanup_roi_preview()
                self.btn_roi.config(text="⬜ 选区")
            # 如果有已有矩形 ROI，也清除
            if self.detect_roi is not None:
                self.detect_roi = None
            self.detect_polygon = None
            self._polygon_mode_active = True
            self._polygon_vertices = []
            self.btn_polygon.config(text="🔲 退出自由选区")
            self.status.config(
                foreground='#cc6600',
                text="🟠 自由选区: 左键逐次点击顶点  |  右键或[Enter]闭合  |  [ESC]取消  |  再次点击按钮完成")
            # 绑定 Enter 键闭合
            self.root.bind("<Return>", self._finish_polygon)
        self._refresh_overlay()

    def _add_polygon_vertex(self, x, y):
        """添加一个多边形顶点"""
        if x is None or y is None:
            return
        # 防止重复添加完全相同的点（鼠标抖动过滤）
        if self._polygon_vertices and np.linalg.norm(np.array([x, y]) - np.array(self._polygon_vertices[-1])) < 2:
            return
        self._polygon_vertices.append((x, y))
        n = len(self._polygon_vertices)
        hint = "继续点击添加顶点  |  右键/Enter 闭合" if n >= 3 else "至少 3 个顶点才能闭合"
        self.status.config(
            foreground='#cc6600',
            text=f"🟠 已添加 {n} 个顶点  |  {hint}  |  [ESC]取消")
        self._refresh_overlay()

    def _finish_polygon(self, event=None):
        """闭合多边形 — 连接首尾顶点"""
        if len(self._polygon_vertices) < 3:
            self.status.config(
                foreground='red',
                text="⚠ 至少需要 3 个顶点才能形成多边形选区")
            return
        # 保存多边形
        self.detect_polygon = list(self._polygon_vertices)
        self._polygon_vertices = []
        self._polygon_mode_active = False
        self.btn_polygon.config(text="✏️ 自由选区")
        self.status.config(
            foreground='#006600',
            text=f"✓ 自由选区已设置: {len(self.detect_polygon)} 个顶点  |  可用原子识别限制在此区域")
        # 清除可能存在的矩形选区
        self.detect_roi = None
        self._roi_mode_active = False
        self._drawing_roi = None
        self._cleanup_roi_preview()
        self.btn_roi.config(text="⬜ 选区")
        # 释放 Enter 键绑定
        try:
            self.root.unbind("<Return>")
        except Exception:
            pass
        self._refresh_overlay()

    # ================================================================
    #  参考矢量 — 鼠标选择
    # ================================================================
    def _handle_ref_multi_click(self, event):
        """Append one arbitrary line-defining anchor for the active a/b direction."""
        if not self.points or event.xdata is None or event.ydata is None:
            return
        pts = np.asarray(self.points, dtype=float)
        distance = np.hypot(pts[:, 0] - event.xdata, pts[:, 1] - event.ydata)
        idx = int(np.argmin(distance))
        if distance[idx] > self.pick_radius:
            return

        stage = int(getattr(self, 'ref_multi_stage', 0))
        groups = getattr(self, 'ref_multi_indices', [[], []])
        active = groups[stage]
        if idx in active:
            self.status.config(text=f"⚠ 点 {idx + 1} 已是当前{('a' if stage == 0 else 'b')}方向的锚点")
            return
        active.append(idx)
        auto_groups = getattr(self, 'ref_multi_auto_indices', [[], []])
        auto_groups[stage] = []
        self.ref_multi_auto_indices = auto_groups
        self.redraw_points()
        self.canvas.draw_idle()
        direction = 'a' if stage == 0 else 'b'
        next_action = "自动找齐同行原子并切换到 b" if stage == 0 else "自动找齐同行原子并计算"
        self.status.config(
            foreground='#a35b00' if stage == 0 else '#007c91',
            text=f"{direction}方向已选 {len(active)} 个锚点（不要求相邻）: "
                 f"[{', '.join(str(i + 1) for i in active)}]  |  [Enter]{next_action}  |  右键撤销")

    def _undo_ref_multi_click(self):
        """Undo the last chain click; from an empty b chain, return to a."""
        groups = getattr(self, 'ref_multi_indices', [[], []])
        auto_groups = getattr(self, 'ref_multi_auto_indices', [[], []])
        stage = int(getattr(self, 'ref_multi_stage', 0))
        if groups[stage]:
            groups[stage].pop()
            auto_groups[stage] = []
        elif stage == 1:
            self.ref_multi_stage = 0
            stage = 0
            auto_groups[0] = []
        else:
            return
        self.ref_multi_auto_indices = auto_groups
        self._refresh_overlay()
        direction = 'a' if stage == 0 else 'b'
        self.status.config(
            text=f"{direction}方向当前 {len(groups[stage])} 个锚点  |  左键继续选择  |  "
                 "[Enter]自动纳入直线范围内的同行原子")

    def _advance_ref_multi_selection(self, event=None):
        """Infer the full chain from anchors and advance a→b or confirm."""
        groups = getattr(self, 'ref_multi_indices', [[], []])
        stage = int(getattr(self, 'ref_multi_stage', 0))
        selected = groups[stage]
        direction = 'a' if stage == 0 else 'b'
        try:
            inferred = infer_atom_chain_from_anchors(
                self.points, selected, label=f"{direction} 方向原子列")
            fit_lattice_vector_from_chain(
                np.asarray(self.points, dtype=float)[inferred],
                label=f"{direction} 方向原子列", min_points=2)
            auto_groups = getattr(self, 'ref_multi_auto_indices', [[], []])
            auto_groups[stage] = inferred.tolist()
            self.ref_multi_auto_indices = auto_groups
        except (AnalysisError, IndexError, ValueError) as error:
            messagebox.showerror("多点参考选择无效", str(error))
            return "break"

        if stage == 0:
            self.ref_multi_stage = 1
            self.status.config(
                foreground='#007c91',
                text=f"✓ a方向手选 {len(selected)} 个、自动纳入 {len(inferred)} 个原子；  "
                     "🔷 b方向任选同一原子列上的 ≥2 个原子（可间隔、可斜向） → "
                     "[Enter]自动纳入并计算  |  右键撤销  |  [ESC]取消")
            self._refresh_overlay()
            return "break"

        self._confirm_ref_multi_select()
        return "break"

    @staticmethod
    def _display_vector_angle(vector):
        """Vector angle in the GUI's user-facing convention (right=0°, up=90°)."""
        return float((-np.degrees(np.arctan2(vector[1], vector[0]))) % 360.0)

    def _confirm_ref_multi_select(self):
        groups = getattr(self, 'ref_multi_indices', [[], []])
        try:
            all_points = np.asarray(self.points, dtype=float)
            a_indices = infer_atom_chain_from_anchors(
                all_points, groups[0], label="a 方向原子列")
            b_indices = infer_atom_chain_from_anchors(
                all_points, groups[1], label="b 方向原子列")
            a_points = all_points[a_indices]
            b_points = all_points[b_indices]
            a_fit, b_fit = estimate_reference_vectors_from_chains(
                a_points, b_points, min_points=2,
                max_condition=self.lattice_max_condition)
            self.ref_multi_auto_indices = [a_indices.tolist(), b_indices.tolist()]
        except (AnalysisError, IndexError, ValueError) as error:
            messagebox.showerror("多点参考选择无效", str(error))
            return

        a_vec = a_fit.vector
        b_vec = b_fit.vector
        a_len = float(np.linalg.norm(a_vec))
        b_len = float(np.linalg.norm(b_vec))
        included_angle = float(np.degrees(np.arccos(np.clip(
            np.dot(a_vec, b_vec) / (a_len * b_len), -1.0, 1.0))))
        a_cv = a_fit.spacing_std / a_len if a_len > 0 else np.inf
        b_cv = b_fit.spacing_std / b_len if b_len > 0 else np.inf
        quality_note = ""
        if max(a_cv, b_cv) > 0.03 or max(a_fit.angle_rms_deg, b_fit.angle_rms_deg) > 2.0:
            quality_note = ("\n⚠ 列内间距或方向离散较大。若该区域并非无应变、含缺陷或选点不连续，"
                            "平均参考矢量仍会带入系统偏差。\n")

        msg = (
            "双方向原子列平均结果：\n"
            "（手选点只定义直线；中间未点击的同行原子已自动纳入）\n\n"
            f"  a方向: 手选 {len(groups[0])} 个锚点 → 实际纳入 {a_fit.n_points} 个原子 / "
            f"{a_fit.n_points - 1} 个相邻间隔\n"
            f"    长度={a_len:.4f} px  方向={self._display_vector_angle(a_vec):.2f}°\n"
            f"    间距标准差={a_fit.spacing_std:.4f} px  拟合RMS={a_fit.rms_fit_residual:.4f} px  "
            f"角度RMS={a_fit.angle_rms_deg:.2f}°\n"
            f"  b方向: 手选 {len(groups[1])} 个锚点 → 实际纳入 {b_fit.n_points} 个原子 / "
            f"{b_fit.n_points - 1} 个相邻间隔\n"
            f"    长度={b_len:.4f} px  方向={self._display_vector_angle(b_vec):.2f}°\n"
            f"    间距标准差={b_fit.spacing_std:.4f} px  拟合RMS={b_fit.rms_fit_residual:.4f} px  "
            f"角度RMS={b_fit.angle_rms_deg:.2f}°\n"
            f"  a-b夹角={included_angle:.2f}°\n"
            f"{quality_note}\n"
            "确认使用这组参考晶格矢量？")
        choice = messagebox.askyesnocancel("确认双方向平均参考矢量", msg)
        if choice is None:
            return
        if not choice:
            self.ref_multi_indices = [[], []]
            self.ref_multi_auto_indices = [[], []]
            self.ref_multi_stage = 0
            self.status.config(
                foreground='#a35b00',
                text="🟠 重新选择 a方向: 任选同一原子列上的 ≥2 个原子（可间隔） → [Enter]")
            self._refresh_overlay()
            return

        self.reference_vecs = (a_vec.copy(), b_vec.copy())
        self.ref_origin = all_points[int(groups[0][0])].copy()
        self.reference_metadata = {
            'method': 'two-chain-robust-fit',
            'estimator': 'anchor-line-auto-chain-2d-huber-v2',
            'a_anchor_points': all_points[np.asarray(groups[0], dtype=int)].tolist(),
            'b_anchor_points': all_points[np.asarray(groups[1], dtype=int)].tolist(),
            'a_anchor_count': int(len(groups[0])),
            'b_anchor_count': int(len(groups[1])),
            'a_points': a_points.tolist(),
            'b_points': b_points.tolist(),
            'a_point_count': int(a_fit.n_points),
            'b_point_count': int(b_fit.n_points),
            'a_rms_fit_px': float(a_fit.rms_fit_residual),
            'b_rms_fit_px': float(b_fit.rms_fit_residual),
            'a_spacing_std_px': float(a_fit.spacing_std),
            'b_spacing_std_px': float(b_fit.spacing_std),
            'a_angle_rms_deg': float(a_fit.angle_rms_deg),
            'b_angle_rms_deg': float(b_fit.angle_rms_deg),
        }
        self._clear_analysis_results()
        self._reset_ref_multi_selection()
        self.current_mode = "add"
        self.mode_var.set("add")
        self.redraw_points()
        self.canvas.draw_idle()
        self.status.config(
            foreground='green',
            text=f"✓ 双方向参考矢量已设置: a手选{len(groups[0])}→纳入{a_fit.n_points}点/"
                 f"{a_len:.3f}px  b手选{len(groups[1])}→纳入{b_fit.n_points}点/"
                 f"{b_len:.3f}px  夹角={included_angle:.2f}°")
        self._update_ref_status()
        self._update_ref_region_label()

    def _handle_ref_select_click(self, event):
        if not self.points:
            return
        pts = np.array(self.points)
        dist = np.sqrt((pts[:, 0] - event.xdata) ** 2 + (pts[:, 1] - event.ydata) ** 2)
        idx = np.argmin(dist)
        if dist[idx] > self.pick_radius:
            return
        if idx in self.ref_select_indices:
            self.status.config(text=f"⚠ 点 {idx + 1} 已被选中，请点击另一个点")
            return
        if len(self.ref_select_indices) >= 3:
            # 确认框「取消」后停在三点待确认状态; 新点击视为重新开始选择,
            # 否则第 4 个点会让高亮颜色表越界
            self.ref_select_indices = []
        self.ref_select_indices.append(idx)
        self.redraw_points()
        self.canvas.draw_idle()
        n = len(self.ref_select_indices)
        labels = [str(i + 1) for i in self.ref_select_indices]
        roles = ['原点', 'a矢量', 'b矢量']
        role = roles[n - 1] if n <= 3 else '?'
        self.status.config(text=f"已选择 {n}/3: [{', '.join(labels)}]  当前点={role}")
        if n == 3:
            self._confirm_ref_select()

    def _confirm_ref_select(self):
        i_origin, i_a, i_b = self.ref_select_indices
        origin = np.array(self.points[i_origin])
        a_end  = np.array(self.points[i_a])
        b_end  = np.array(self.points[i_b])
        a_vec = a_end - origin
        b_vec = b_end - origin
        a_len = np.linalg.norm(a_vec)
        b_len = np.linalg.norm(b_vec)
        try:
            validate_reference_lattice(a_vec, b_vec,
                                       max_condition=self.lattice_max_condition)
        except AnalysisError as error:
            messagebox.showerror("参考矢量无效", str(error))
            self.ref_select_indices = []
            self._refresh_overlay()
            return
        angle = np.degrees(np.arccos(np.clip(
            np.dot(a_vec, b_vec) / (a_len * b_len), -1, 1)))

        msg = (f"选中的3个点:\n"
               f"  原点:  #{i_origin + 1}  ({origin[0]:.1f}, {origin[1]:.1f})\n"
               f"  a端点: #{i_a + 1}  ({a_end[0]:.1f}, {a_end[1]:.1f})\n"
               f"  b端点: #{i_b + 1}  ({b_end[0]:.1f}, {b_end[1]:.1f})\n\n"
               f"矢量 a: 长度={a_len:.1f} px\n"
               f"矢量 b: 长度={b_len:.1f} px\n"
               f"夹角:  {angle:.1f}°\n\n"
               f"确认使用此晶格矢量？")

        choice = messagebox.askyesnocancel("确认参考矢量", msg)
        if choice is None:  # 取消 — 保持在选择模式
            return
        if not choice:  # 否 — 重新选择
            self.ref_select_indices = []
            self.redraw_points()
            self.canvas.draw_idle()
            self.status.config(foreground='blue',
                               text="🔵 重新选择: 依次点击3个点  |  [ESC]取消")
            return

        self.reference_vecs = (a_vec.copy(), b_vec.copy())
        self.ref_origin = origin.copy()
        self.reference_metadata = {
            'method': 'three-point',
            'origin_point': origin.tolist(),
            'a_point': a_end.tolist(),
            'b_point': b_end.tolist(),
        }
        self._clear_analysis_results()
        self.ref_select_indices = []
        self.current_mode = "add"
        self.mode_var.set("add")
        self.redraw_points()
        self.canvas.draw_idle()
        self.status.config(foreground='green',
                           text=f"✓ 参考矢量已设置: a=({a_vec[0]:.2f},{a_vec[1]:.2f})  b=({b_vec[0]:.2f},{b_vec[1]:.2f})")
        self._update_ref_status()
        self._update_ref_region_label()

    # ================================================================
    #  参考矢量 — 手动输入
    # ================================================================
    def set_reference_manual(self):
        """手动输入晶格参数（间距+角度）"""
        if len(self.points) < 1:
            messagebox.showinfo("提示", "需要至少1个标记点作为原点。")
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("手动输入晶格参数")
        dlg.geometry("320x330")
        dlg.transient(self.root)
        dlg.resizable(False, False)
        dlg.bind("<Escape>", lambda e: dlg.destroy())
        dlg.focus_set()

        main = ttk.Frame(dlg, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="原点用第几个标记点:").pack(anchor=tk.W)
        origin_var = tk.StringVar(value="1")
        ttk.Entry(main, textvariable=origin_var, width=8).pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(main, text="▸ 晶格矢量 a (水平向右=0°, 垂直向上=90°)",
                  font=('', 9, 'bold')).pack(anchor=tk.W)
        row1 = ttk.Frame(main)
        row1.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(row1, text="长度 (像素):").pack(side=tk.LEFT)
        a_len_var = tk.StringVar(value="")
        ttk.Entry(row1, textvariable=a_len_var, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Label(row1, text="角度 (°):").pack(side=tk.LEFT, padx=(8, 0))
        a_ang_var = tk.StringVar(value="0")
        ttk.Entry(row1, textvariable=a_ang_var, width=8).pack(side=tk.LEFT, padx=4)

        ttk.Label(main, text="▸ 晶格矢量 b",
                  font=('', 9, 'bold')).pack(anchor=tk.W, pady=(10, 0))
        row2 = ttk.Frame(main)
        row2.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(row2, text="长度 (像素):").pack(side=tk.LEFT)
        b_len_var = tk.StringVar(value="")
        ttk.Entry(row2, textvariable=b_len_var, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Label(row2, text="角度 (°):").pack(side=tk.LEFT, padx=(8, 0))
        b_ang_var = tk.StringVar(value="90")
        ttk.Entry(row2, textvariable=b_ang_var, width=8).pack(side=tk.LEFT, padx=4)

        ttk.Label(main, text="提示: 也可以先点「距离与角度」\n"
                             "按钮测量两条边的长度和角度再填入",
                  font=('', 7), foreground='gray').pack(anchor=tk.W, pady=(10, 0))

        def on_ok():
            try:
                oi = int(origin_var.get())
                if oi < 1 or oi > len(self.points):
                    raise ValueError
            except ValueError:
                messagebox.showerror("参数错误", f"原点编号必须在1~{len(self.points)}之间", parent=dlg)
                dlg.focus_set()
                return
            try:
                a_len = float(a_len_var.get())
                a_ang = float(a_ang_var.get())
                b_len = float(b_len_var.get())
                b_ang = float(b_ang_var.get())
                if a_len <= 0 or b_len <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("参数错误", "长度必须>0，角度必须是数字", parent=dlg)
                dlg.focus_set()
                return

            a_rad = np.radians(a_ang)
            b_rad = np.radians(b_ang)
            a_vec = np.array([a_len * np.cos(a_rad), -a_len * np.sin(a_rad)])
            b_vec = np.array([b_len * np.cos(b_rad), -b_len * np.sin(b_rad)])

            # 检查共线性 (2D 叉积 z 分量; 避免 np.cross 对 2D 向量的弃用警告)
            cross = a_vec[0] * b_vec[1] - a_vec[1] * b_vec[0]
            if abs(cross) < 1e-8:
                messagebox.showerror("参数错误", "a、b矢量共线，请调整角度。", parent=dlg)
                dlg.focus_set()
                return

            dlg.destroy()
            origin = np.array(self.points[oi - 1])
            self.reference_vecs = (a_vec, b_vec)
            self.ref_origin = origin
            self.reference_metadata = {
                'method': 'manual',
                'origin_point': origin.tolist(),
                'a_length_px': float(a_len),
                'a_angle_deg': float(a_ang),
                'b_length_px': float(b_len),
                'b_angle_deg': float(b_ang),
            }
            self._clear_analysis_results()
            self.status.config(foreground='green',
                               text=f"✓ 参考矢量已设置: a长={a_len:.1f}px@{a_ang:.0f}°  "
                                    f"b长={b_len:.1f}px@{b_ang:.0f}°")
            self._update_ref_status()
            self._update_ref_region_label()

        btn_row = ttk.Frame(main)
        btn_row.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(btn_row, text="确定", command=on_ok).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_row, text="取消", command=lambda: dlg.destroy()).pack(side=tk.RIGHT, padx=4)
        dlg.protocol("WM_DELETE_WINDOW", lambda: dlg.destroy())
        dlg.wait_window()

    def detect_matching_outliers(self, displacements, points, k=6, n_sigma=3.0):
        """
        通过局部位移一致性检测匹配异常的原子（马氏距离法）。

        对每个原子，将其位移矢量与 K 近邻做对比。若马氏距离超过
        n_sigma 阈值，标记为异常匹配（可能是 wrap-around 错误或缺陷原子）。

        Parameters
        ----------
        displacements : (N, 2) np.ndarray
        points : (N, 2) np.ndarray
        k : int  近邻数量
        n_sigma : float  马氏距离阈值

        Returns
        -------
        outlier_mask : (N,) bool
        """
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        N = len(points)
        outlier_mask = np.zeros(N, dtype=bool)
        k_eff = min(k, N - 1)
        if k_eff < 2:
            return outlier_mask

        for i in range(N):
            _, neighbors = tree.query(points[i], k=k_eff + 1)
            neighbors = neighbors[1:]  # 排除自身
            u_neighbors = displacements[neighbors]  # (k, 2)
            u_mean = u_neighbors.mean(axis=0)
            u_i = displacements[i]
            diff = u_i - u_mean
            cov = np.cov(u_neighbors.T)
            # 退化路径与马氏路径一样使用「每轴标准差」做尺度，保持量纲统一。
            std_u = np.std(u_neighbors, axis=0)
            std_u = np.maximum(std_u, 1e-6)
            if cov.ndim < 2 or np.linalg.matrix_rank(cov) < 2:
                # 协方差奇异，退化为按轴标准化的欧氏距离。
                dist = float(np.linalg.norm(diff / std_u))
            else:
                try:
                    dist = float(np.sqrt(diff @ np.linalg.inv(cov) @ diff))
                except np.linalg.LinAlgError:
                    dist = float(np.linalg.norm(diff / std_u))
            if dist > n_sigma:
                outlier_mask[i] = True

        return outlier_mask

    # ================================================================
    #  校准模式 — 点击相邻原子自动推算检测参数
    # ================================================================
    def _start_calibrate_from_button(self):
        """📐 校准按钮入口"""
        if self.image is None:
            messagebox.showinfo("提示", "请先导入图像。")
            return
        if len(self.points) > 0:
            # 已有标记点：提示用户可直接用已有点校准，或点击新点
            self._start_calibrate()
        else:
            # 没有标记点：直接进入校准模式
            self._start_calibrate()

    def _start_calibrate(self):
        """进入校准模式"""
        self._calibrate_pts.clear()
        self.current_mode = "calibrate"
        self.mode_var.set("calibrate")
        self.root.bind("<Return>", self._finish_calibrate)  # 确保 Enter 键绑定
        self.status.config(foreground='#0066cc',
            text="🔵 校准: 点击 2~4 个相邻原子 → 自动推算间距和尺寸 → 按 [Enter] 确认  |  [ESC] 取消")
        self._refresh_overlay()

    def _handle_calibrate_click(self, event):
        """校准模式点击: 收集采样点"""
        x, y = event.xdata, event.ydata
        self._calibrate_pts.append((x, y))
        n = len(self._calibrate_pts)
        self.status.config(foreground='#0066cc',
            text=f"🔵 已点 {n} 个校准点  |  继续点击或按 [Enter] 确认  |  [ESC] 取消")
        self._refresh_overlay()
        self.root.bind("<Return>", self._finish_calibrate)

    def _finish_calibrate(self, event=None):
        """完成校准: 分析采样点并自动设置检测参数"""
        self.root.unbind("<Return>")
        if len(self._calibrate_pts) < 2:
            messagebox.showinfo("提示", "需要至少 2 个校准点来测量原子间距。")
            self._calibrate_pts.clear()
            self.mode_var.set("add")
            self.set_mode()
            return

        pts = np.array(self._calibrate_pts)
        image = self._get_work_image()
        h_img, w_img = image.shape

        # ---- 1. 原子间距: 所有点对的欧氏距离取中位数 ----
        dists = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dists.append(np.linalg.norm(pts[i] - pts[j]))
        dists = np.array(dists)
        median_spacing = np.median(dists)
        # 晶格最近邻距离 ≈ 最小非零距离
        min_pairwise = dists.min()
        suggested_min_dist = max(2, round(min_pairwise * 0.75, 1))

        # ---- 2. 原子尺寸 & 强度: 在每个校准点附近提取 ROI ----
        spot_fwhms = []
        spot_peaks = []       # 每个点的峰值强度（背景扣除后）
        spot_bgs = []         # 每个点的局部背景
        # 估算一个合适的 ROI 半径
        roi_radius = max(3, int(min_pairwise * 0.8))
        for x, y in pts:
            xi, yi = int(round(x)), int(round(y))
            r0 = max(0, yi - roi_radius)
            r1 = min(h_img, yi + roi_radius + 1)
            c0 = max(0, xi - roi_radius)
            c1 = min(w_img, xi + roi_radius + 1)
            roi = image[r0:r1, c0:c1].copy()
            if roi.size < 9:
                continue

            # 背景扣除
            bg = np.percentile(roi, 10)
            roi_sub = np.maximum(roi - bg, 0)
            if roi_sub.max() <= 0:
                continue

            # 记录强度和背景
            spot_peaks.append(float(roi_sub.max()))
            spot_bgs.append(float(bg))

            # 径向强度剖面: 以 ROI 内峰值像素为圆心 (而非几何中心), 消除点击偏差
            peak_r, peak_c = np.unravel_index(np.argmax(roi_sub), roi_sub.shape)
            cy_c, cx_c = float(peak_r), float(peak_c)
            ys, xs = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
            radii = np.sqrt((xs - cx_c)**2 + (ys - cy_c)**2)
            max_r = int(min(cx_c, cy_c,
                            roi.shape[1] - 1 - cx_c, roi.shape[0] - 1 - cy_c,
                            roi_radius)) + 1
            if max_r < 2:
                continue

            radial_mean = np.array([
                roi_sub[(radii >= r) & (radii < r + 1)].mean()
                for r in range(max_r)
            ])
            # FWHM: 找到半高位置
            peak_val = radial_mean[0:max(1, max_r // 3)].max()
            half_max = peak_val / 2.0
            # 找超过 half_max 的最大半径
            above = np.where(radial_mean > half_max)[0]
            if len(above) > 0:
                fwhm = 2.0 * (above[-1] + 0.5)  # 直径 ≈ 2×半高半径
                fwhm = max(1.5, min(fwhm, roi_radius * 2))
            else:
                fwhm = 3.0  # fallback
            spot_fwhms.append(fwhm)

        if spot_fwhms:
            median_fwhm = np.median(spot_fwhms)
        else:
            median_fwhm = 4.0

        # ---- 3. 推算参数 ----
        # sigma ≈ FWHM / 2.5  (高斯 FWHM ≈ 2.355σ，取稍小值避免过度模糊)
        suggested_sigma = round(median_fwhm / 2.5, 1)
        suggested_sigma = max(0.1, min(suggested_sigma, 5.0))

        # 窗口 = 奇数，约 FWHM × 1.5
        window_raw = int(median_fwhm * 1.5)
        window_raw = max(3, window_raw)
        suggested_window = window_raw if window_raw % 2 == 1 else window_raw + 1
        suggested_window = min(suggested_window, 11)

        # ---- 4. 基于最暗校准点反推强度阈值 ----
        suggested_threshold = None   # 默认自适应
        threshold_desc = "自适应 (80% 分位数)"
        if spot_peaks:
            spot_peaks_arr = np.array(spot_peaks)
            spot_bgs_arr = np.array(spot_bgs)
            # 最暗原子的峰高（扣除背景后）和对应的图像绝对强度
            min_peak_sub = spot_peaks_arr.min()
            max_peak_sub = spot_peaks_arr.max()
            typical_bg = np.median(spot_bgs_arr)
            # 最暗原子在原始图像中的表观强度 ≈ 背景 + 峰高
            dimmest_signal = typical_bg + min_peak_sub * 0.6  # 60% 峰高处仍能检测
            # 找该值在全图中的分位数 → 作为阈值分位数
            pct_at_dimmest = np.searchsorted(
                np.sort(image.ravel()),
                dimmest_signal) / image.size
            # 取比最暗原子再低一点的安全余量
            suggested_pct = max(0.05, min(0.75, pct_at_dimmest * 0.85))
            suggested_threshold = round(suggested_pct, 2)
            threshold_desc = (
                f"{suggested_threshold:.0%} 分位数 "
                f"(最亮点峰值 {max_peak_sub:.2f}, 最暗点峰值 {min_peak_sub:.2f})")

        # 根据 FWHM 和信噪比判断精炼方法
        noise_span = np.percentile(image, 30) - np.percentile(image, 10)
        peak_span = np.percentile(image, 98) - np.percentile(image, 10)
        if median_fwhm > 5 and noise_span < peak_span * 0.3 and spot_peaks:
            suggested_method = "gaussian"
            method_desc = "2D 高斯拟合"
        else:
            suggested_method = "com"
            method_desc = "质心法 (COM)"

        # ---- 5. 弹出确认对话框 ----
        if spot_peaks:
            peak_range_str = f"  原子强度范围: {min(spot_peaks):.2f} ~ {max(spot_peaks):.2f}  (扣除背景)\n\n"
        else:
            peak_range_str = "  未成功测量原子强度（ROI 质量不足）\n\n"

        msg = (
            f"校准分析结果:\n\n"
            f"  采样点: {len(pts)} 个\n"
            f"  最近邻间距: {min_pairwise:.1f} px\n"
            f"  原子直径 (FWHM): {median_fwhm:.1f} px\n"
            f"{peak_range_str}"
            f"  建议参数:\n"
            f"    最小原子间距: {suggested_min_dist} px\n"
            f"    高斯滤波 σ: {suggested_sigma}\n"
            f"    质心窗口: {suggested_window}×{suggested_window} px\n"
            f"    强度阈值: {threshold_desc}\n"
            f"    精炼方法: {method_desc}\n\n"
            f"  是否用这些参数打开检测对话框？\n"
            f"  (阈值会根据最暗校准点自动降低，以捕获暗原子)"
        )

        choice = messagebox.askyesnocancel("校准完成", msg)
        if choice is None:  # 取消
            self._calibrate_pts.clear()
            self._refresh_overlay()
            self.mode_var.set("add")
            self.set_mode()
            return
        if choice:  # 是 — 应用参数并打开检测对话框
            self.detect_min_dist = suggested_min_dist
            self.detect_sigma = suggested_sigma
            self.detect_window = suggested_window
            self.centroid_method = suggested_method
            self._suggested_threshold = suggested_threshold

            self._calibrate_pts.clear()
            self._refresh_overlay()
            self.mode_var.set("add")
            self.set_mode()
            self.auto_detect_points()
        else:  # 否 — 重新校准
            self._calibrate_pts.clear()
            self._refresh_overlay()
            self.mode_var.set("calibrate")
            self.set_mode()

    def _update_ref_status(self):
        if self.reference_vecs and self.ref_origin is not None:
            a, b = self.reference_vecs
            angle = np.degrees(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))
            metadata = getattr(self, 'reference_metadata', None) or {}
            method = metadata.get('method')
            if method == 'two-chain-robust-fit':
                source = (f"  [双方向平均 a={int(metadata.get('a_point_count', 0))},"
                          f" b={int(metadata.get('b_point_count', 0))}点]")
            elif method in {'three-point', 'three-point-by-id'}:
                source = "  [三点]"
            elif method == 'manual':
                source = "  [手动]"
            else:
                source = ""
            self.ref_status_label.config(
                text=f"原点=({self.ref_origin[0]:.1f},{self.ref_origin[1]:.1f})  "
                     f"a=({a[0]:.2f},{a[1]:.2f})  b=({b[0]:.2f},{b[1]:.2f})  ∠={angle:.1f}°{source}",
                foreground="#006600")
        else:
            self.ref_status_label.config(text="未设置", foreground="gray")

    def _reference_lattice_metadata(self):
        """JSON-ready reference definition shared by export sidecars."""
        if (getattr(self, 'reference_vecs', None) is None
                or getattr(self, 'ref_origin', None) is None):
            return None
        a_vec, b_vec = self.reference_vecs
        return {
            'coordinate_space': 'image-display-x-right-y-down',
            'origin': np.asarray(self.ref_origin, dtype=float).tolist(),
            'a_vec': np.asarray(a_vec, dtype=float).tolist(),
            'b_vec': np.asarray(b_vec, dtype=float).tolist(),
            'estimation': getattr(self, 'reference_metadata', None),
            'refinement_region': (list(self.ref_region)
                                  if getattr(self, 'ref_region', None) is not None else None),
        }

    # ================================================================
    #  后台任务: queue + root.after (线程中禁止直接操作 Tk)
    # ================================================================
    def _ensure_worker_polling(self):
        if self._poll_after_id is None:
            self._schedule_worker_poll()

    def _schedule_worker_poll(self):
        try:
            self._poll_after_id = self.root.after(50, self._poll_worker_queue)
        except tk.TclError:
            self._poll_after_id = None

    def _poll_worker_queue(self):
        self._poll_after_id = None
        try:
            while True:
                kind, generation, payload = self._worker_queue.get_nowait()
                if generation != self._job_generation:
                    continue  # 代际失效：旧任务结果直接丢弃
                try:
                    if kind == 'detect_done':
                        self._finish_auto_detect(*payload)
                    elif kind == 'detect_error':
                        self._detect_error(payload)
                    elif kind == 'analysis_done':
                        self._finish_ppa_analysis(payload)
                    elif kind == 'analysis_error':
                        self._analysis_error(payload)
                except Exception as error:
                    # 单条结果处理失败不得杀死轮询循环 —— 否则所有后续
                    # 后台任务的结果永远无法送达, 界面表现为静默卡死。
                    import traceback
                    traceback.print_exc()
                    try:
                        messagebox.showerror("内部错误", f"处理后台结果时出错：\n{error}")
                    except Exception:
                        pass
        except queue.Empty:
            pass
        finally:
            self._schedule_worker_poll()

    def cancel_current_job(self):
        """Invalidate the active background job; its result will be ignored."""
        self._job_generation += 1
        self.status.config(text="已取消后台任务")
        self._ensure_worker_polling()

    def _detect_error(self, message):
        messagebox.showerror("自动检测失败", str(message))
        self.status.config(text="自动检测失败")

    def _analysis_error(self, message):
        messagebox.showerror("PPA 分析失败", str(message))
        self.status.config(text="PPA 分析失败")

    # ================================================================
    #  自动原子检测 ⭐ (核心改进)
    # ================================================================
    def auto_detect_points(self):
        if self.image is None:
            messagebox.showinfo("提示", "请先导入图像。")
            return

        # 对话框
        dlg = tk.Toplevel(self.root)
        dlg.title("自动原子检测参数")
        dlg.geometry("420x500")
        dlg.minsize(380, 460)
        dlg.transient(self.root)
        dlg.focus_set()

        # ---- 预设选择 ----
        ttk.Label(dlg, text="参数预设:", font=('', 9, 'bold')).pack(pady=(10, 0), anchor=tk.W, padx=10)
        preset_frame = ttk.Frame(dlg)
        preset_frame.pack(fill=tk.X, padx=10, pady=2)
        preset_names = list(self.DETECT_PRESETS.keys())
        preset_var = tk.StringVar(value=preset_names[0])
        preset_cb = ttk.Combobox(preset_frame, textvariable=preset_var,
                                  values=preset_names, state="readonly", width=36)
        preset_cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(preset_frame, text="应用", command=lambda: _apply_preset()).pack(side=tk.RIGHT, padx=(4, 0))

        self._preset_desc_label = ttk.Label(dlg, text=self.DETECT_PRESETS[preset_names[0]]['desc'],
                                             foreground="#555555", font=('', 7), wraplength=380)
        self._preset_desc_label.pack(anchor=tk.W, padx=12, pady=(0, 6))

        def _apply_preset():
            name = preset_var.get()
            p = self.DETECT_PRESETS[name]
            sigma_var.set(str(p['sigma']))
            min_dist_var.set(str(p['min_dist']))
            window_var.set(str(p['window']))
            cent_var.set(p['method'])
            self._preset_desc_label.config(text=p['desc'])

        # 鼠标点击预设更新描述
        preset_cb.bind("<<ComboboxSelected>>", lambda e: _apply_preset())

        ttk.Separator(dlg, orient='horizontal').pack(fill=tk.X, padx=10, pady=4)

        # ---- 详细参数 ----
        ttk.Label(dlg, text="▸ 最小原子间距 (像素):").pack(pady=(4, 0), anchor=tk.W, padx=10)
        min_dist_var = tk.StringVar(value=str(self.detect_min_dist))
        ttk.Entry(dlg, textvariable=min_dist_var, width=10).pack(anchor=tk.W, padx=10)

        ttk.Label(dlg, text="▸ 高斯预滤波 σ (越大越平滑):").pack(pady=(8, 0), anchor=tk.W, padx=10)
        sigma_var = tk.StringVar(value=str(self.detect_sigma))
        ttk.Entry(dlg, textvariable=sigma_var, width=10).pack(anchor=tk.W, padx=10)

        ttk.Label(dlg, text="▸ 质心精炼窗口 (奇数像素):").pack(pady=(8, 0), anchor=tk.W, padx=10)
        window_row = ttk.Frame(dlg)
        window_row.pack(fill=tk.X, padx=10)
        window_var = tk.StringVar(value=str(self.detect_window))
        ttk.Entry(window_row, textvariable=window_var, width=6).pack(side=tk.LEFT)
        ttk.Label(window_row, text=" 小原子用 3，标准 5，大原子用 7",
                  foreground="gray", font=('', 7)).pack(side=tk.LEFT, padx=4)

        ttk.Label(dlg, text="▸ 强度阈值 (百分位数 0~1, 如 0.8=保留强度前 20%; 留空=自适应 80%):").pack(pady=(8, 0), anchor=tk.W, padx=10)
        # 校准推算的阈值自动预填
        init_thresh = ""
        if self._suggested_threshold is not None:
            init_thresh = str(self._suggested_threshold)
            self._suggested_threshold = None  # 使用后清除
        thresh_var = tk.StringVar(value=init_thresh)
        ttk.Entry(dlg, textvariable=thresh_var, width=10).pack(anchor=tk.W, padx=10)

        ttk.Label(dlg, text="▸ 质心精炼方法:").pack(pady=(8, 0), anchor=tk.W, padx=10)
        cent_var = tk.StringVar(value=self.centroid_method)
        cent_frame = ttk.Frame(dlg)
        cent_frame.pack(anchor=tk.W, padx=10)
        ttk.Radiobutton(cent_frame, text="质心法 (COM, 快速)", variable=cent_var, value="com").pack(side=tk.LEFT)
        ttk.Radiobutton(cent_frame, text="2D高斯拟合 (精确)", variable=cent_var, value="gaussian").pack(side=tk.LEFT, padx=5)

        bright_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(dlg, text="原子为亮点 (暗点取消勾选)", variable=bright_var).pack(anchor=tk.W, padx=10, pady=(8, 0))

        def on_ok():
            try:
                md = float(min_dist_var.get())
                sg = float(sigma_var.get())
                ww = int(window_var.get())
                if md <= 0 or sg < 0 or ww < 3 or ww % 2 == 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("参数错误",
                    "最小间距>0, sigma>=0, 窗口>=3奇数", parent=dlg)
                dlg.focus_set()
                return
            ts = thresh_var.get()
            try:
                th = parse_detection_threshold(ts)
            except ValueError as error:
                messagebox.showerror("参数错误", str(error), parent=dlg)
                dlg.focus_set()
                return
            dlg.destroy()
            self.detect_min_dist = md
            self.detect_sigma = sg
            self.detect_window = ww
            self.centroid_method = cent_var.get()
            self._detect_peaks(md, sg, ww, th, bright_var.get())

        btn_row = ttk.Frame(dlg)
        btn_row.pack(fill=tk.X, padx=10, pady=12)
        ttk.Button(btn_row, text="开始检测", command=on_ok).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_row, text="取消", command=lambda: dlg.destroy()).pack(side=tk.RIGHT, padx=4)
        dlg.protocol("WM_DELETE_WINDOW", lambda: dlg.destroy())
        dlg.wait_window()
        self._preset_desc_label = None  # 对话框关闭，清理引用

    def _detect_peaks(self, min_distance, sigma, window, threshold, bright):
        """Start atomic-peak detection on a worker thread.

        The heavy Gaussian/NMS/centroid refinement runs off the Tk main thread.
        All Tk calls happen either here (before the thread starts) or in
        ``_finish_auto_detect`` (after the result is polled via ``root.after``).
        """
        self.detect_min_dist = min_distance
        self.detect_sigma = sigma
        self.detect_window = window
        self._job_generation += 1
        generation = self._job_generation
        self.status.config(text="正在后台检测原子点...")
        self.root.update_idletasks()
        self._ensure_worker_polling()
        # 工单75: 在主线程把 tk.BooleanVar 快照成普通 bool 传入 worker,
        # worker 内不得再读 tkinter 变量 (Tk 非线程安全)。
        use_preprocessed = bool(self.use_preprocessed.get())

        def _run():
            try:
                if generation != self._job_generation:
                    return
                new_points, gaussian_fallback_stats = self._detect_peaks_worker(
                    min_distance, sigma, window, threshold, bright,
                    use_preprocessed)
                self._worker_queue.put(
                    ('detect_done', generation,
                     (new_points, gaussian_fallback_stats)))
            except Exception as error:
                self._worker_queue.put(('detect_error', generation, str(error)))

        threading.Thread(target=_run, daemon=True).start()

    def _finish_auto_detect(self, new_points, gaussian_fallback_stats=None):
        """Commit a completed auto-detection on the Tk main thread."""
        if not new_points:
            messagebox.showinfo("提示", "未检测到任何原子点，请调整参数。")
            self.status.config(text="自动检测未找到原子点")
            return

        # 工单27: 记录回退比例供项目元数据存档, 并在界面上提示回退情况
        self.last_gaussian_refine_stats = gaussian_fallback_stats
        fallback_note = ""
        if gaussian_fallback_stats and gaussian_fallback_stats.get('n_fallback'):
            fallback_note = (
                f"\n⚠ {gaussian_fallback_stats['n_fallback']}"
                f"/{gaussian_fallback_stats['n_points']} 个点高斯拟合失败，"
                f"已回退 COM 亚像素质心 "
                f"(比例 {gaussian_fallback_stats['fallback_ratio']:.1%})")

        # 三选一: 「是」追加(非破坏), 「否」替换(破坏性, 明确标注), 「取消」丢弃。
        choice = messagebox.askyesnocancel(
            "检测完成",
            f"检测到 {len(new_points)} 个原子点\n"
            f"质心方法: {'COM' if self.centroid_method == 'com' else '2D Gaussian'}\n"
            f"{fallback_note}\n"
            f"如何处理检测结果？\n"
            f"  [是(Y)]   追加到当前列表 (保留已有 {len(self.points)} 个点)\n"
            f"  [否(N)]   替换现有所有点 (当前 {len(self.points)} 个点将被丢弃)\n"
            f"  [取消]    丢弃本次检测结果")
        if choice is None:
            self.status.config(text=f"已丢弃本次检测结果 ({len(new_points)} 个点)")
            return
        if choice:
            self.points.extend(new_points)
        else:
            self.points = new_points
        self.selected_point_idx = None
        self.ref_select_indices = []  # 替换点表后旧索引会错位
        self._reset_ref_multi_selection()
        self._clear_analysis_results()
        self.refresh_display()
        self.status.config(text=f"自动检测完成，共 {len(self.points)} 个原子点 (亚像素定位)"
                                f"{fallback_note.strip()}")

    def _detect_peaks_worker(self, min_distance, sigma, window, threshold, bright,
                             use_preprocessed):
        # 工单75: use_preprocessed 必填(无默认值), 强制调用方在主线程快照后传入,
        # worker 内经 None 分支现场读 tk 变量会在编译期签名层面就不再可能。
        img = self._get_work_image(use_preprocessed).copy()
        h_img, w_img = img.shape

        # 记住检测 ROI 边界，后续做后过滤（不要用 -inf 污染图像）
        roi_bounds = None
        if self.detect_roi is not None:
            roi_bounds = (
                max(0, int(self.detect_roi[0])), max(0, int(self.detect_roi[1])),
                min(w_img, int(self.detect_roi[2])), min(h_img, int(self.detect_roi[3])),
            )

        if not bright:
            img = -img

        # 1) 高斯预滤波
        if sigma > 0:
            img_filt = gaussian_filter(img, sigma=sigma)
        else:
            img_filt = img

        # 2) 阈值: 图像已归一化到 [0,1], 阈值统一为百分位数语义。
        #    threshold 为 (0,1) 内的小数 (parse_detection_threshold 已校验),
        #    None = 自适应 80% 分位。
        percentile = 80.0 if threshold is None else float(threshold) * 100.0
        threshold_abs = np.percentile(img_filt, percentile)

        # 3) 局部最大值检测 — 用小核 (3×3)，后续 NMS 负责间距控制
        #    用 >= 替代 == 避免浮点舍入导致漏检（加入微小容差 1e-12）
        max_filt = maximum_filter(img_filt, size=3)
        peaks_mask = (img_filt >= max_filt - 1e-12) & (img_filt > threshold_abs)

        coords = np.argwhere(peaks_mask)  # (row, col)
        if len(coords) == 0:
            return [], None

        # 4) 按强度排序
        vals = img_filt[peaks_mask]
        idx_sort = np.argsort(-vals)
        coords = coords[idx_sort]

        # -- 在当前步骤预过滤：只保留选区/多边形内的候选峰（大幅减少 NMS 计算量） --
        keep_mask = np.ones(len(coords), dtype=bool)

        # 矩形 ROI 预过滤
        if roi_bounds is not None:
            rx0, ry0, rx1, ry1 = roi_bounds
            keep_mask &= ((coords[:, 1] >= rx0) & (coords[:, 1] < rx1) &
                          (coords[:, 0] >= ry0) & (coords[:, 0] < ry1))

        # 多边形选区预过滤（先过滤 coords，NMS 只处理选区内点）
        if self.detect_polygon is not None:
            from matplotlib.path import Path
            pverts = np.array(self.detect_polygon)
            pcodes = [Path.MOVETO] + [Path.LINETO] * (len(pverts) - 1) + [Path.CLOSEPOLY]
            poly_path = Path(np.vstack([pverts, pverts[0:1]]), codes=pcodes)
            in_poly = poly_path.contains_points(coords[:, [1, 0]])
            keep_mask &= in_poly

        coords = coords[keep_mask]
        if len(coords) == 0:
            return [], None

        # 5) 非极大值抑制（大规模时用向量化贪心，避免 O(N²) 逐点比较）
        selected = []
        if len(coords) > 500:
            # 大规模候选峰：贪心 NMS + cKDTree 邻域查询（O(N log N + 邻接边数)），
            # 替代原来的逐候选全量距离计算（O(N^2)，上万候选时 UI 会卡死）。
            selected = greedy_nms(coords, min_distance)
        else:
            # 小规模：经典暴力 NMS
            for c in coords:
                if all(np.linalg.norm(c - s) > min_distance for s in selected):
                    selected.append(c)
            selected = np.array(selected)  # (N,2)  (row, col)

        # 6) 亚像素质心修正 ⭐
        gaussian_fallback_stats = None
        if self.centroid_method == "com" and len(selected) > 20:
            # COM 批量矢量化: N>20 时比逐原子调用快 10-50×
            cx_arr, cy_arr = self._refine_centroids_batch(img, selected, window=window)
            new_points = [(float(cx_arr[i]), float(cy_arr[i]))
                          for i in range(len(cx_arr))]
        else:
            # 少数 COM 或高斯拟合: 保留逐原子精修
            # 工单27: 高斯拟合失败的点会静默回退 COM, 必须计数并提示
            self._gaussian_fallback_count = 0
            new_points = []
            for i, c in enumerate(selected):
                cx, cy = self._refine_centroid(img, c[1], c[0], window=window,
                                               point_index=i + 1)
                new_points.append((cx, cy))
            if self.centroid_method == "gaussian":
                n_fallback = self._gaussian_fallback_count
                if n_fallback:
                    logger.warning(
                        "高斯精炼: %d/%d 个点高斯拟合失败，已回退 COM 亚像素质心",
                        n_fallback, len(selected))
                gaussian_fallback_stats = {
                    'n_points': int(len(selected)),
                    'n_fallback': int(n_fallback),
                    'fallback_ratio': (n_fallback / len(selected)) if len(selected) else 0.0,
                }

        return new_points, gaussian_fallback_stats

    def _refine_centroid(self, image, x, y, window=5, point_index=None):
        """
        亚像素质心精炼 (COM 迭代收敛)
        x, y: 整数像素坐标 (col, row)
        window: 局部窗大小 (奇数)
        point_index: 1 起算的点序号, 仅用于高斯回退时的告警定位
        返回: (sub_x, sub_y) 亚像素坐标
        """
        # 高斯拟合 (单次; 失败或越界则回退 COM)
        if self.centroid_method == "gaussian":
            fit = self._gaussian_fit_window(image, x, y, window)
            if fit is not None:
                return fit
            # 工单27: 回退不再静默 —— 计数 + 日志, 汇总比例由调用方上报
            self._gaussian_fallback_count = getattr(
                self, '_gaussian_fallback_count', 0) + 1
            logger.warning(
                "第 %s 个点高斯拟合失败，已回退 COM 亚像素质心",
                point_index if point_index is not None else "?")

        # COM: 迭代 2 次, 每次以前次质心重新居中窗口, 消除窗口偏心偏差
        hw = window // 2
        H, W = image.shape
        cx_f, cy_f = float(x), float(y)
        for _ in range(2):
            xi = int(np.clip(round(cx_f), 0, W - 1))
            yi = int(np.clip(round(cy_f), 0, H - 1))
            y0 = max(0, yi - hw)
            y1 = min(H, yi + hw + 1)
            x0 = max(0, xi - hw)
            x1 = min(W, xi + hw + 1)

            # 如果靠边，调整窗口使其居中
            if y1 - y0 < window:
                if y0 == 0:
                    y1 = min(H, window)
                else:
                    y0 = max(0, H - window)
            if x1 - x0 < window:
                if x0 == 0:
                    x1 = min(W, window)
                else:
                    x0 = max(0, W - window)

            roi = image[y0:y1, x0:x1].copy()
            if roi.size == 0:
                return float(x), float(y)

            # 减去局部背景（取 5% 分位数）
            bg = np.percentile(roi, 5)
            roi = np.maximum(roi - bg, 0)

            total = roi.sum()
            if total == 0:
                return float(x), float(y)
            ys, xs = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
            cx_f = x0 + (xs * roi).sum() / total
            cy_f = y0 + (ys * roi).sum() / total

        return cx_f, cy_f

    def _gaussian_fit_window(self, image, x, y, window):
        """
        2D 高斯拟合亚像素定位 (供 _refine_centroid 调用)

        工单27: 拟合实现收敛到 ppa_core.refine.gaussian_refine_point
        (与 atomic_core 同源口径的规范实现), 本方法仅保留
        "(sub_x, sub_y) 或 None (拟合失败)" 的旧契约。

        初值取 ROI 内峰值位置 (而非固定几何中心), 适配靠边原子;
        校验拟合中心偏离初始位置不超过窗口半径, 防止误收敛到邻近峰。
        """
        cx, cy, fitted = gaussian_refine_point(image, x, y, window)
        return (cx, cy) if fitted else None

    def _refine_centroids_batch(self, image, coords_rc, window=5, iterations=2):
        """
        批量 COM 亚像素定位（预填充矢量化版, 迭代收敛）

        对 N 个候选点同时做质心法亚像素精炼。
        预填充图像 + 固定窗口提取，消除逐原子边界检查;
        每次迭代以前次质心重新居中窗口, 消除窗口偏心偏差。

        Parameters
        ----------
        image : np.ndarray (H, W)
        coords_rc : np.ndarray (N, 2)  整数像素坐标 (row, col)
        window : int  奇数窗口大小
        iterations : int  COM 迭代次数 (默认 2)

        Returns
        -------
        cx : np.ndarray (N,)  亚像素 x (col)
        cy : np.ndarray (N,)  亚像素 y (row)
        """
        hw = window // 2
        H, W = image.shape
        N = len(coords_rc)
        if N == 0:
            return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

        # 预填充图像，坐标统一偏移 hw
        padded = np.pad(image, hw, mode='reflect')

        # 预生成窗口内的网格坐标（仅一次）
        ys_grid, xs_grid = np.mgrid[0:window, 0:window]

        centers_r = coords_rc[:, 0].astype(int)
        centers_c = coords_rc[:, 1].astype(int)
        results_x = centers_c.astype(np.float64)
        results_y = centers_r.astype(np.float64)

        # COM 迭代: 每次以前次质心重新居中窗口, 消除窗口偏心偏差
        for _ in range(iterations):
            # 一次性取全部窗口（高级索引得到 (N, window, window) 拷贝），
            # 用向量化分位数/COM 替代逐原子 np.percentile 循环。
            windows = sliding_window_view(padded, (window, window))[centers_r, centers_c]
            bg = np.percentile(windows.reshape(N, -1), 5, axis=1)
            roi_sub = np.maximum(windows - bg[:, None, None], 0.0)
            totals = roi_sub.sum(axis=(1, 2))
            com_x = np.einsum("nij,ij->n", roi_sub, xs_grid)
            com_y = np.einsum("nij,ij->n", roi_sub, ys_grid)
            # 窗口起点为 centers - hw；total==0 时保持原中心不动（避免除零告警）
            safe_com_x = np.zeros_like(com_x)
            safe_com_y = np.zeros_like(com_y)
            np.divide(com_x, totals, out=safe_com_x, where=totals > 0)
            np.divide(com_y, totals, out=safe_com_y, where=totals > 0)
            new_x = centers_c - hw + safe_com_x
            new_y = centers_r - hw + safe_com_y
            new_x[totals <= 0] = centers_c[totals <= 0]
            new_y[totals <= 0] = centers_r[totals <= 0]

            # 更新窗口中心 (裁剪到图像内, 防止填充区外索引)
            centers_c = np.clip(np.round(new_x).astype(int), 0, W - 1)
            centers_r = np.clip(np.round(new_y).astype(int), 0, H - 1)
            results_x, results_y = new_x, new_y

        return results_x, results_y

    def _show_detect_params(self):
        """快速查看/修改检测参数"""
        method_names = {'com': '质心法(COM)', 'gaussian': '2D高斯拟合'}
        msg = (f"当前检测参数:\n\n"
               f"最小间距: {self.detect_min_dist} px\n"
               f"高斯滤波 σ: {self.detect_sigma}\n"
               f"质心窗口: {self.detect_window}×{self.detect_window} px\n"
               f"质心方法: {method_names.get(self.centroid_method, self.centroid_method)}\n\n"
               f"在「自动识别原子点→参数」中修改")
        messagebox.showinfo("检测参数", msg)

    def _on_analysis_method_changed(self, event=None):
        """Keep the selected algorithm explicit in the result metadata and UI."""
        selected = self.analysis_method_var.get() if hasattr(self, 'analysis_method_var') else ""
        self.analysis_method = "lattice_cst" if selected.startswith("兼容") else "peak_pairs"

    def _reset_after_image_load(self):
        """Reset image-dependent state without creating an undo record for old points."""
        self.points.clear()
        self.undo_stack.clear()
        self.reference_vecs = None
        self.ref_origin = None
        self.reference_metadata = None
        self.ref_select_indices = []
        self._reset_ref_multi_selection()
        self._clear_analysis_results()
        self.detect_roi = None
        self.detect_polygon = None
        self._roi_mode_active = False
        self._polygon_mode_active = False
        self._polygon_vertices = []
        self._drawing_roi = None
        self._cleanup_roi_preview()
        self._zoom_history.clear()
        self.btn_roi.config(text="⬜ 选区")
        self.btn_polygon.config(text="✏️ 自由选区")
        # 参考区随图像失效 (坐标系已变)
        self.ref_region = None
        self._ref_region_mode_active = False
        self._drawing_ref_region = None
        self._cleanup_ref_region_preview()
        self.btn_ref_region.config(text="🧭 框选参考区")
        self._update_ref_region_label()
        self._calibrate_pts.clear()   # 旧图校准点不得叠加到新图
        # 视图回基态: 残留应变视图会让 refresh_display 走"请先执行PPA分析"分支
        self.view_var.set("colored")
        if hasattr(self, 'view_cb'):
            self.view_cb.set("位移幅值染色")
        self.processed_image = None
        self.show_processed = False
        self.use_preprocessed.set(False)
        self.btn_toggle_display.config(text="👁 显示: 原始")
        self.preprocess_status.config(text="状态: 未应用预处理", foreground="gray")

    def _load_image_path(self, path, frame_index=None):
        """Load one image frame and make it the only valid state for the application."""
        loaded = load_analysis_image(path, frame_index=frame_index)
        self.image = loaded.pixels
        self.image_path = loaded.source_path
        self.image_frame_index = loaded.frame_index
        self.image_metadata = {
            'source_shape': list(loaded.source_shape),
            'source_dtype': loaded.source_dtype,
            'frame_count': loaded.frame_count,
        }
        self._reset_after_image_load()
        self.display_image()

    # ================================================================
    #  PPA 位移与应变分析 ⭐ (核心)
    # ================================================================
    def run_ppa_analysis(self):
        """Launch PPA displacement/strain analysis.

        Main thread keeps all Tk dialogs (validation, duplicate-index choice,
        bad-match warning) and starts a worker thread for the heavy parts
        (outlier detection, Delaunay/peak-pair strain, grid interpolation).
        The worker only touches non-Tk result fields and posts a payload to
        ``_worker_queue``; ``_poll_worker_queue`` applies it via
        ``root.after``.
        """
        if self.reference_vecs is None or self.ref_origin is None:
            messagebox.showinfo("提示", "请先设置参考晶格矢量。")
            return
        if len(self.points) < 3:
            messagebox.showinfo("提示", "需要至少3个原子点。")
            return

        self.status.config(text="PPA分析中 (晶格匹配)...")
        self.root.update_idletasks()

        a_vec, b_vec = self.reference_vecs
        origin = self.ref_origin
        pts = np.array(self.points)
        n_pts = len(pts)
        # Never reuse a previous analysis result if this run fails validation.
        self._clear_analysis_results()

        # 阈值先校验：NaN/inf 会让"condition > max_condition"恒为 False 而静默
        # 绕过全部病态基矢检查；0/负值会把一切合法基矢判为病态。两者都必须在
        # 任何计算（含参考区精化）之前给出清晰错误。
        try:
            lattice_max_condition = validate_max_condition(
                self.lattice_max_condition, label="lattice_max_condition")
        except AnalysisError as error:
            messagebox.showerror("阈值无效", f"晶格条件数阈值无效：\n{error}")
            self.status.config(text="错误: 晶格条件数阈值无效")
            return
        self.lattice_max_condition = lattice_max_condition

        try:
            validate_reference_lattice(a_vec, b_vec,
                                       max_condition=lattice_max_condition)
        except (np.linalg.LinAlgError, AnalysisError) as error:
            messagebox.showerror("错误", f"参考晶格无效：\n{error}")
            self.status.config(text="错误: 参考晶格无效")
            return

        # ---- 整数晶格索引匹配 (矢量化) + 可选参考区基矢精化 ----
        # 三点基矢可由双原子列多点拟合直接降噪；若另设二维无应变参考区，
        # 则继续用区内全部原子联合精化原点和两条基矢，不吸收区外真实应变。
        refine_applied = False
        da_pct = db_pct = 0.0
        n_ref_used = 0
        region_mask = self._atoms_in_ref_region_mask()
        r_origin, r_a, r_b = origin, a_vec, b_vec
        if region_mask is not None:
            r_origin, r_a, r_b, refine_applied, n_ref_used = refine_lattice_basis_in_region(
                pts, origin, a_vec, b_vec, region_mask,
                max_condition=lattice_max_condition)
            if refine_applied:
                da_pct = np.linalg.norm(r_a - a_vec) / max(np.linalg.norm(a_vec), 1e-12) * 100
                db_pct = np.linalg.norm(r_b - b_vec) / max(np.linalg.norm(b_vec), 1e-12) * 100
        try:
            M_r_inv = np.linalg.inv(np.column_stack((r_a, r_b)))
        except np.linalg.LinAlgError:
            messagebox.showerror("错误", "精化后的参考晶格奇异，请重设参考矢量或参考区。")
            self.status.config(text="错误: 精化基矢奇异")
            return
        # ---- 全局一对一晶格分配: 每个用户确认点都保留并参与位移分析 ----
        try:
            assignment = assign_unique_lattice_indices(
                pts, r_origin, r_a, r_b, max_condition=lattice_max_condition)
        except (AnalysisError, ValueError) as error:
            messagebox.showerror("晶格索引分配失败", str(error))
            self.status.config(text=f"错误: 晶格索引分配失败 ({error})")
            return
        lattice_coords = assignment.lattice_coords
        lattice_indices = assignment.lattice_indices
        self.analysis_point_indices = assignment.source_indices
        self.lattice_coords = lattice_coords.copy()
        self.lattice_indices = lattice_indices.copy()
        self.assignment_residuals = assignment.residuals_lattice.copy()
        self.assignment_residuals_px = assignment.residuals_px.copy()
        self.assignment_reassigned_mask = assignment.reassigned_mask.copy()
        self.assignment_low_confidence_mask = assignment.low_confidence_mask.copy()
        self.assignment_conflict_count = int(assignment.n_conflicts)

        matched_ideal = (r_origin[None, :]
                         + lattice_indices[:, 0:1] * r_a[None, :]
                         + lattice_indices[:, 1:2] * r_b[None, :])       # (N,2) 理想位置
        displacements = pts - matched_ideal
        distortions = np.linalg.norm(displacements, axis=1)

        # 低置信度只进入最终质量报告；不删除原子，也不停止 PPA。
        bad_mask = assignment.low_confidence_mask
        n_bad = int(bad_mask.sum())

        # 保存匹配结果（快速矢量计算仍在主线程；这些字段是后续重计算输入）。
        self.matched_actual = pts.copy()
        self.ideal_grid = matched_ideal.copy()
        self.displacements = displacements.copy()
        self.distortions = distortions

        # 说明: 位移场线性去趋势在此有意不提供 —— 它会移除本分析要测量的
        # 均匀仿射应变/转动。基矢噪声通过「参考区精化」处理，而非去趋势。

        # ---- 启动后台线程：异常检测 + 应变计算 + 网格插值 ----
        self._job_generation += 1
        generation = self._job_generation
        self.status.config(text="PPA分析中 (后台应变计算)...")
        self.root.update_idletasks()
        self._ensure_worker_polling()

        payload_args = dict(
            generation=generation,
            pts=pts,
            lattice_indices=lattice_indices,
            matched_ideal=matched_ideal,
            displacements=displacements,
            r_origin=r_origin,
            r_a=r_a,
            r_b=r_b,
            n_pts=n_pts,
            refine_applied=refine_applied,
            da_pct=da_pct,
            db_pct=db_pct,
            n_ref_used=n_ref_used,
            n_bad=n_bad,
            n_conflicts=int(assignment.n_conflicts),
            n_reassigned=int(assignment.reassigned_mask.sum()),
            # 启动时刻的参数快照: worker 不回读可变实例状态, 与主线程编辑互不干扰
            von_mises_coeff=float(self.von_mises_coeff),
            analysis_method=str(self.analysis_method),
            image_shape=(int(self.image.shape[0]), int(self.image.shape[1])),
            # 与参考晶格校验、索引分配共用同一阈值快照：worker 里不得再读实例
            # 状态，也不得回落到隐藏默认 30，否则用户放宽的阈值会被静默挡下。
            lattice_max_condition=float(lattice_max_condition),
        )

        def _run():
            try:
                self._ppa_analysis_worker(**payload_args)
            except Exception as error:
                self._worker_queue.put(('analysis_error', generation, str(error)))

        threading.Thread(target=_run, daemon=True).start()

    def _ppa_analysis_worker(self, generation, pts, lattice_indices, matched_ideal,
                             displacements, r_origin, r_a, r_b, n_pts,
                             refine_applied, da_pct, db_pct, n_ref_used, n_bad,
                             n_conflicts, n_reassigned, von_mises_coeff,
                             analysis_method, image_shape,
                             lattice_max_condition=None):
        """Heavy, Tk-free half of ``run_ppa_analysis``.

        只读传入参数, 不读写可变实例状态; 全部结果放入 queue payload, 由
        ``_finish_ppa_analysis`` 在 Tk 主线程一次性写入。这消除了
        "分析进行中编辑点表 → 已清空的字段被旧结果复活" 的竞态
        (配合 ``_clear_analysis_results`` 的代际递增闭环)。
        """
        def _cancelled():
            return generation != self._job_generation

        if _cancelled():
            return

        # 局部位移异常检测只产生质量标记。缺陷信号不得被自动删除。
        outlier_mask = self.detect_matching_outliers(displacements, pts, k=6, n_sigma=3.0)
        n_outliers = int(outlier_mask.sum())
        if _cancelled():
            return

        # ---- 晶格完备性自检: 索引空洞 → 疑似漏检位置 (红色×标记) ----
        hole_idx = find_lattice_holes(lattice_indices)
        lattice_holes = [tuple(r_origin + hn * r_a + hm * r_b) for hn, hm in hole_idx[:200]]
        n_holes = len(hole_idx)

        # ---- 应变张量计算 ⭐ ----
        strain_error = None
        strain_fields = None
        invalid_sites = None
        if _cancelled():
            return
        try:
            if analysis_method == "peak_pairs":
                local_kwargs = {"equivalent_coefficient": von_mises_coeff}
                if lattice_max_condition is not None:
                    # 与 validate_reference_lattice 共用同一阈值：此前局部 PPA
                    # 硬编码 30，用户按 LATTICE_CONDITION_GUIDANCE 放宽参考晶格
                    # 阈值后，局部路径仍按 30 拒绝且无提示。
                    local_kwargs["max_condition"] = lattice_max_condition
                result = compute_local_peak_pair_strain(
                    lattice_indices,
                    matched_ideal * np.array([1.0, -1.0]),
                    pts * np.array([1.0, -1.0]),
                    **local_kwargs)
                strain_fields = _strain_fields_from_result(result, "sites", matched_ideal)
                invalid_mask = ~np.asarray(result.quality_mask, dtype=bool)
                invalid_sites = pts[invalid_mask] if np.any(invalid_mask) else None
            else:
                result = compute_cst_strain(
                    matched_ideal * np.array([1.0, -1.0]),
                    pts * np.array([1.0, -1.0]),
                    equivalent_coefficient=von_mises_coeff)
                strain_fields = _strain_fields_from_result(result, "triangles", matched_ideal)
        except AnalysisError as error:
            # 应变失败不致命, 仍显示畸变染色图与跳过说明。
            strain_fields = None
            strain_error = str(error)

        if _cancelled():
            return
        grid_ok = False
        strain_grids = strain_grids_gl = None
        grid_extent = None
        if strain_fields is not None:
            strain_grids, strain_grids_gl, grid_extent = _interpolate_strain_grids(
                strain_fields, image_shape)
            grid_ok = bool(strain_grids)
            if _cancelled():
                return

        self._worker_queue.put((
            'analysis_done',
            generation,
            dict(
                outlier_mask=outlier_mask,
                lattice_holes=lattice_holes,
                n_outliers=n_outliers,
                n_holes=n_holes,
                n_pts=n_pts,
                n_bad=n_bad,
                n_conflicts=n_conflicts,
                n_reassigned=n_reassigned,
                strain_ok=strain_fields is not None,
                grid_ok=grid_ok,
                strain_error=strain_error,
                refine_applied=refine_applied,
                da_pct=da_pct,
                db_pct=db_pct,
                n_ref_used=n_ref_used,
                strain_fields=strain_fields,
                strain_grids=strain_grids,
                strain_grids_gl=strain_grids_gl,
                grid_extent=grid_extent,
                invalid_sites=invalid_sites,
            ),
        ))

    def _finish_ppa_analysis(self, payload):
        """Apply results and show one consolidated, non-blocking quality report."""
        # ---- 结果字段的唯一写入点 (Tk 主线程): worker 只发 payload ----
        fields = payload.get('strain_fields')
        if fields is not None:
            self.result_locations_physical = fields['result_locations_physical']
            self.tri_centroids = fields['tri_centroids']
            self.strain_xx = fields['strain_xx']
            self.strain_yy = fields['strain_yy']
            self.strain_xy = fields['strain_xy']
            self.strain_eq = fields['strain_eq']
            self.rotation = fields['rotation']
            self.strain_gl_xx = fields['strain_gl_xx']
            self.strain_gl_yy = fields['strain_gl_yy']
            self.strain_gl_xy = fields['strain_gl_xy']
            self.strain_gl_eq = fields['strain_gl_eq']
            self.strain_quality_grades = fields['strain_quality_grades']
            self.strain_invalid_reasons = fields['strain_invalid_reasons']
            self.strain_geometry = fields['strain_geometry']
            self.tri_edge_mask = fields['tri_edge_mask']
            self.element_area = fields['element_area']
            self.strain_grid = payload.get('strain_grids') or {}
            self.strain_grid_gl = payload.get('strain_grids_gl') or {}
            self.strain_grid_extent = payload.get('grid_extent')
        else:
            self.strain_grid = {}
            self.strain_grid_gl = {}
            self.strain_grid_extent = None
        self.invalid_sites = payload.get('invalid_sites')
        self.outlier_mask = payload['outlier_mask']
        self.lattice_holes = payload['lattice_holes']
        n_outliers = payload['n_outliers']
        n_holes = payload['n_holes']
        n_pts = payload['n_pts']
        n_bad = payload['n_bad']
        n_conflicts = payload.get('n_conflicts', 0)
        n_reassigned = payload.get('n_reassigned', 0)
        strain_ok = payload['strain_ok']
        grid_ok = payload['grid_ok']
        strain_error = payload.get('strain_error')
        refine_applied = payload['refine_applied']
        da_pct = payload['da_pct']
        db_pct = payload['db_pct']
        n_ref_used = payload['n_ref_used']

        # 显示结果
        self._view_changed("colored")
        n_elements = int(np.isfinite(self.strain_xx).sum()) if self.strain_xx is not None else 0
        element_label = "原子位点" if self.strain_geometry == "sites" else "有效三角形"
        strain_info = ""
        if strain_ok:
            finite_xx = self.strain_xx[np.isfinite(self.strain_xx)]
            strain_info = (f"  |  {n_elements} 个{element_label}  "
                           f"ε_xx ∈ [{finite_xx.min():.4f}, {finite_xx.max():.4f}]")
        holes_note = f"  |  ⚠ {n_holes} 个疑似漏检(红色×)" if n_holes else ""
        algorithm_label = "局部 Peak Pairs" if self.analysis_method == "peak_pairs" else "兼容晶格-CST"
        self.status.config(text=f"✅ {algorithm_label} 完成  |  {n_pts}/{n_pts} 个确认原子已完成位移分析{strain_info}{holes_note}  |  用下拉菜单切换视图")

        def _finite_range(values):
            finite = np.asarray(values)[np.isfinite(values)]
            return (float(finite.min()), float(finite.max())) if len(finite) else (float('nan'), float('nan'))

        strain_msg = ""
        if strain_ok:
            xx_range = _finite_range(self.strain_xx)
            yy_range = _finite_range(self.strain_yy)
            xy_display = -self.strain_xy
            xy_range = _finite_range(xy_display)
            eq_range = _finite_range(self.strain_eq)
            strain_msg = (f"\n应变张量: {n_elements} 个{element_label} (ε_xy 为显示坐标约定)\n"
                          f"  ε_xx 范围: [{xx_range[0]:.4f}, {xx_range[1]:.4f}]\n"
                          f"  ε_yy 范围: [{yy_range[0]:.4f}, {yy_range[1]:.4f}]\n"
                          f"  ε_xy 范围: [{xy_range[0]:.4f}, {xy_range[1]:.4f}]\n"
                          f"  ε_eq 范围: [{eq_range[0]:.4f}, {eq_range[1]:.4f}]")
            if grid_ok:
                strain_msg += "\n网格插值: 200×200 完成"
            else:
                strain_msg += "\n网格插值: 未生成"
        else:
            strain_msg = f"\n应变计算: 未生成 ({strain_error or '有效几何不足'})"
        extra_notes = []
        if n_conflicts:
            extra_notes.append(
                f"晶格索引: 初始发现 {n_conflicts} 个冲突，已全局一对一重分配 {n_reassigned} 个原子；\n"
                f"  全部 {n_pts} 个确认点均已保留并完成位移分析")
        if n_bad:
            extra_notes.append(
                f"匹配质量: {n_bad}/{n_pts} 个点残差较大，仍已保留；建议复核参考原点和 a、b 矢量")
        if n_outliers:
            extra_notes.append(
                f"局部位移: {n_outliers}/{n_pts} 个异常点（可能是真实缺陷或误识别），未删除")
        reference_metadata = getattr(self, 'reference_metadata', None) or {}
        reference_method = reference_metadata.get('method')
        is_multi_reference = reference_method == 'two-chain-robust-fit'
        if refine_applied:
            extra_notes.append(
                f"参考区基矢精化: 用区内 {n_ref_used} 个原子最小二乘\n"
                f"  Δa={da_pct:.2f}%  Δb={db_pct:.2f}%  (伪应变偏置 ≈ 1%/√{n_ref_used})")
        elif self.ref_region is not None:
            fallback = "双方向原子列平均值" if is_multi_reference else "当前初始参考值"
            extra_notes.append(
                f"⚠ 参考区精化未生效 (区内 {n_ref_used} 个原子 <10 或几何退化)，\n"
                f"  本次使用{fallback}")
        elif is_multi_reference:
            extra_notes.append(
                f"参考矢量: 手选锚点定义方向，自动纳入同行原子后二维稳健拟合 "
                f"(a手选={int(reference_metadata.get('a_anchor_count', reference_metadata.get('a_point_count', 0)))}"
                f"→纳入{int(reference_metadata.get('a_point_count', 0))}点, "
                f"b手选={int(reference_metadata.get('b_anchor_count', reference_metadata.get('b_point_count', 0)))}"
                f"→纳入{int(reference_metadata.get('b_point_count', 0))}点)\n"
                f"  拟合RMS: a={float(reference_metadata.get('a_rms_fit_px', 0.0)):.4f}px, "
                f"b={float(reference_metadata.get('b_rms_fit_px', 0.0)):.4f}px")
        elif reference_method == 'manual':
            extra_notes.append(
                "提示: 当前使用手动输入的参考矢量且未设参考区；其系统误差会传递到整幅应变图")
        else:
            extra_notes.append(
                "提示: 未设参考区，基矢仅由 3 点定义 (~1% 伪应变偏置)。\n"
                "  建议改用「🎯 双方向选点平均」或框选无应变参考区")
        grades = self.strain_quality_grades
        if grades is not None:
            grade_a = int(np.sum(grades == 'A-symmetric'))
            grade_b = int(np.sum(grades == 'B-local-fit'))
            grade_c = int(np.sum(grades == 'C-minimal-fit'))
            extra_notes.append(
                f"局部应变质量: A级对称 {grade_a}，B级局部拟合 {grade_b}，C级最小拟合 {grade_c}")
        if self.invalid_sites is not None:
            extra_notes.append(
                f"{len(self.invalid_sites)} 个原子几何邻域不足，位移结果已保留，应变记为 NaN；\n"
                f"  已用灰色方块标记，不代表原子被排除")
        if n_holes:
            extra_notes.append(
                f"⚠ 晶格完备性: 发现 {n_holes} 个疑似漏检位置 (红色×标记),\n"
                f"  建议手动补点后重新分析")
        notes_str = ("\n\n" + "\n".join(extra_notes)) if extra_notes else ""
        report = (f"{algorithm_label} 完成。\n"
                  f"位移分析: {n_pts}/{n_pts} 个确认原子。{strain_msg}{notes_str}\n\n"
                  f"用下拉菜单切换视图查看结果；质量提示不会删除任何原子。")
        if strain_ok:
            messagebox.showinfo("分析质量报告", report)
        else:
            messagebox.showwarning("分析部分完成", report)

    # ================================================================
    #  视图切换 (核心改进)
    # ================================================================
    def _on_view_combobox(self, event=None):
        """Combobox 选择 → 内部 view key 映射"""
        text_to_key = {
            "位移幅值染色": "colored",
            "位移矢量场": "vector",
            "应变 ε_xx (正应变 x)": "strain_xx",
            "应变 ε_yy (正应变 y)": "strain_yy",
            "应变 ε_xy (剪应变)": "strain_xy",
            "等效应变 ε_eq (von Mises)": "strain_eq",
            "旋转 ω": "rotation",
            "应变 ε_xx (GL 有限应变)": "strain_gl_xx",
            "应变 ε_yy (GL 有限应变)": "strain_gl_yy",
            "应变 ε_xy (GL 有限应变)": "strain_gl_xy",
            "等效应变 ε_eq (GL 有限应变)": "strain_gl_eq",
            "标记图": "markers",
        }
        selected_text = self.view_cb.get()  # 从 Combobox 直接读取显示文本
        key = text_to_key.get(selected_text, "colored")
        self._view_changed(key)

    def _view_changed(self, view):
        # 保存当前视图范围，以免 clear+imshow 后意外缩放
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # 重要: 先移除 colorbar (在 ax.clear() 之前), 否则 colorbar 的 axes 会失效
        self._remove_colorbar()
        self.ax.clear()
        self._img_artist = None  # 清除缓存，避免引用死 artist
        self._strain_cloud_artist = None  # 云图 artist 随 ax.clear() 一起失效
        self._points_scatter_artist = None  # 散点 artist 同理

        disp_img = self._get_display_image()
        if disp_img is not None:
            # 回填 _img_artist: 不然 _refresh_overlay 跳过底图更新,
            # display_image 走重建分支并丢失当前缩放
            self._img_artist = self.ax.imshow(disp_img, cmap='gray', origin='upper', aspect='equal')
            # 恢复视图范围，避免切换选项时意外回到全图
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

        # 同步 Combobox 显示文本
        key_to_text = {
            "colored":   "位移幅值染色",
            "vector":    "位移矢量场",
            "strain_xx": "应变 ε_xx (正应变 x)",
            "strain_yy": "应变 ε_yy (正应变 y)",
            "strain_xy": "应变 ε_xy (剪应变)",
            "strain_eq": "等效应变 ε_eq (von Mises)",
            "rotation":  "旋转 ω",
            "strain_gl_xx": "应变 ε_xx (GL 有限应变)",
            "strain_gl_yy": "应变 ε_yy (GL 有限应变)",
            "strain_gl_xy": "应变 ε_xy (GL 有限应变)",
            "strain_gl_eq": "等效应变 ε_eq (GL 有限应变)",
            "markers":   "标记图",
        }
        if hasattr(self, 'view_cb') and view in key_to_text:
            self.view_var.set(view)
            self.view_cb.set(key_to_text[view])

        if view == "markers":
            self.ax.set_title("原子标记图 — 绿色圆圈为标记点")
            self.redraw_points()
            self.canvas.draw_idle()
            return

        if view == "colored":
            self._show_colored_points_view()
        elif view == "vector":
            self._show_displacement_vectors_view()
        elif view == "strain_xx":
            self._show_strain_map('xx')
        elif view == "strain_yy":
            self._show_strain_map('yy')
        elif view == "strain_xy":
            self._show_strain_map('xy')
        elif view == "strain_eq":
            self._show_strain_map('eq')
        elif view == "rotation":
            self._show_strain_map('rot')
        elif view == "strain_gl_xx":
            self._show_strain_map('gl_xx')
        elif view == "strain_gl_yy":
            self._show_strain_map('gl_yy')
        elif view == "strain_gl_xy":
            self._show_strain_map('gl_xy')
        elif view == "strain_gl_eq":
            self._show_strain_map('gl_eq')

        self.canvas.draw_idle()

    def _show_colored_points_view(self):
        """原子位移幅值染色 — 颜色表示位移大小 (单位: 像素)"""
        if self.distortions is None:
            self.ax.set_title("请先执行PPA分析")
            self.redraw_points()
            return
        pts = self.matched_actual if self.matched_actual is not None else np.array(self.points)
        vmin, vmax = self._get_colorbar_range(self.distortions)
        if vmin == vmax:
            vmin, vmax = self.distortions.min(), self.distortions.max()
            if vmin == vmax:
                vmin, vmax = vmin - 0.001, vmax + 0.001
        # 画点：大小与配色由「点样式」控件决定
        cmap_note = "红蓝色图" if self.point_cmap_key == 'rb' else "turbo色图"
        sc = self.ax.scatter(pts[:, 0], pts[:, 1], c=self.distortions,
                             cmap=self._displacement_cmap(),
                             s=self.atom_dot_size, marker='o', edgecolor='black', linewidth=0.4,
                             vmin=vmin, vmax=vmax, zorder=3)
        self._points_scatter_artist = sc
        self._cbar = self.fig.colorbar(sc, ax=self.ax, fraction=0.046, pad=0.04)
        self._cbar.set_label('Atomic displacement (px)', fontsize=10)
        # 叠加位移矢量箭头（可调节）；批量 quiver 替代逐原子 ax.arrow。
        if self.displacements is not None:
            if self.show_all_arrows:
                step, mag_min = 1, 0.005
            else:
                step, mag_min = max(1, len(pts) // 80), 0.01
            idx = np.arange(0, len(pts), step)
            dxy = self.displacements[idx]
            mags = np.linalg.norm(dxy, axis=1)
            keep = mags >= mag_min
            if np.any(keep):
                self.ax.quiver(
                    pts[idx[keep], 0], pts[idx[keep], 1],
                    dxy[keep, 0] * self.arrow_scale,
                    dxy[keep, 1] * self.arrow_scale,
                    angles='xy', scale_units='xy', scale=1.0,
                    width=0.004, color='white', alpha=0.6,
                    linewidth=self.arrow_lw, zorder=4,
                )
        range_note = f" [范围: {vmin:.3g} ~ {vmax:.3g}]" if not self.colorbar_manual else f" [手动: {vmin:.3g} ~ {vmax:.3g}]"
        self.ax.set_title(f"原子位移幅值染色 + 位移矢量 (箭头{self.arrow_scale}×)  |  {cmap_note}{range_note}")

    def _show_displacement_vectors_view(self):
        """位移矢量场"""
        if self.displacements is None:
            self.ax.set_title("请先执行PPA分析")
            self.redraw_points()
            return

        pts = self.matched_actual if self.matched_actual is not None else np.array(self.points)
        # 用颜色表示位移大小
        mag = np.linalg.norm(self.displacements, axis=1)
        vmin, vmax = self._get_colorbar_range(mag)
        if vmin == vmax:
            vmin, vmax = mag.min(), max(mag.max(), 1e-6)

        # 画位移矢量箭头（可调节）；批量 quiver 替代逐原子 ax.arrow。
        if self.show_all_arrows:
            mag_min = 0.001
        else:
            mag_min = 0.005
        d_mag = np.linalg.norm(self.displacements, axis=1)
        keep = d_mag >= mag_min
        if np.any(keep):
            cmap_obj = _resolve_cmap(self._displacement_cmap())
            colors = cmap_obj((d_mag[keep] - vmin) / (vmax - vmin + 1e-10))
            self.ax.quiver(
                pts[keep, 0], pts[keep, 1],
                self.displacements[keep, 0] * self.arrow_scale,
                self.displacements[keep, 1] * self.arrow_scale,
                angles='xy', scale_units='xy', scale=1.0,
                width=0.004, color=colors, alpha=0.7,
                linewidth=self.arrow_lw, zorder=3,
            )

        # 画点
        sc = self.ax.scatter(pts[:, 0], pts[:, 1], c=mag,
                             cmap=self._displacement_cmap(),
                             s=self.atom_dot_size, marker='o', edgecolor='gray', linewidth=0.3,
                             vmin=vmin, vmax=vmax, zorder=2)
        self._points_scatter_artist = sc
        self._cbar = self.fig.colorbar(sc, ax=self.ax, fraction=0.046, pad=0.04)
        self._cbar.set_label('Displacement (px)', fontsize=10)
        range_note = f" [范围: {vmin:.3g} ~ {vmax:.3g}]" if not self.colorbar_manual else f" [手动: {vmin:.3g} ~ {vmax:.3g}]"
        self.ax.set_title(f"位移矢量场 (箭头{self.arrow_scale}×)  |  颜色=位移幅值{range_note}")

    # ================================================================
    #  应变张量可视化 ⭐
    # ================================================================
    def _show_strain_map(self, strain_key):
        """
        显示应变张量分量的连续彩色云图（基于网格插值）。

        strain_key: 'xx' | 'yy' | 'xy' | 'eq' | 'rot' |
                    'gl_xx' | 'gl_yy' | 'gl_xy' | 'gl_eq'
        """
        # 判断是否为 Green-Lagrange 分量
        is_gl = strain_key.startswith('gl_')
        grid_dict = self.strain_grid_gl if is_gl else self.strain_grid

        if not grid_dict or strain_key not in grid_dict:
            self.ax.set_title("请先执行PPA分析 (应变数据未生成)")
            self.redraw_points()
            return

        strain_data = grid_dict[strain_key]
        if strain_data is None or np.all(np.isnan(strain_data)):
            self.ax.set_title("应变插值失败，请检查数据")
            self.redraw_points()
            return

        extent = self.strain_grid_extent
        if extent is None:
            self.ax.set_title("应变网格未生成，请重新执行PPA分析")
            self.redraw_points()
            return
        # 颜色映射：有正负的分量用 RdBu_r，纯正值用 turbo
        gl_suffix = " (GL)" if is_gl else ""
        labels = {
            'xx':    ('ε_xx (正应变 x)' + gl_suffix, 'RdBu_r', 'ε_xx'),
            'yy':    ('ε_yy (正应变 y)' + gl_suffix, 'RdBu_r', 'ε_yy'),
            'xy':    ('ε_xy (剪应变)' + gl_suffix, 'RdBu_r', 'ε_xy'),
            'eq':    ('ε_eq (von Mises 等效应变)' + gl_suffix, 'turbo', 'ε_eq'),
            'rot':   ('极分解旋转角 (rad)', 'RdBu_r', 'θ'),
            'gl_xx': ('ε_xx (GL 有限应变)', 'RdBu_r', 'ε_xx (GL)'),
            'gl_yy': ('ε_yy (GL 有限应变)', 'RdBu_r', 'ε_yy (GL)'),
            'gl_xy': ('ε_xy (GL 有限应变)', 'RdBu_r', 'ε_xy (GL)'),
            'gl_eq': ('ε_eq (GL von Mises)', 'turbo', 'ε_eq (GL)'),
        }
        title, cmap, clabel = labels.get(strain_key, (f'应变 {strain_key}', 'RdBu_r', f'ε_{strain_key}'))

        # 用户在「应变云图样式」中选择的配色覆盖分量默认值; 未知名称回退默认
        if self.strain_cloud_cmap:
            try:
                _resolve_cmap(self.strain_cloud_cmap)
                cmap = self.strain_cloud_cmap
            except (ValueError, KeyError):
                self.strain_cloud_cmap = None
        # 确定颜色范围（自动/手动）; 发散型色图用对称色标使 ±应变强度可比
        symmetric = (cmap in STRAIN_CLOUD_DIVERGING_CMAPS)
        vmin, vmax = self._get_colorbar_range(strain_data, symmetric=symmetric)

        cbar_mode_note = " [手动]" if self.colorbar_manual else ""
        # 不透明度滑杆实时生效; 越界值夹回 [0, 1]
        alpha = max(0.0, min(1.0, float(self.strain_cloud_alpha)))

        # 叠加半透明彩色云图
        # 注意: extent=(left, right, bottom, top)
        # 网格第一行对应 y=y_min (图像顶部), 最后一行对应 y=y_max (图像底部)
        # 配合 origin='upper': top=ymin, bottom=ymax → 第一行在顶部, 最后行在底部 ✓
        im = self.ax.imshow(strain_data,
                            extent=(extent[0], extent[1], extent[3], extent[2]),
                            origin='upper',
                            cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax,
                            interpolation='bilinear', zorder=3)
        self._strain_cloud_artist = im
        self._cbar = self.fig.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)
        self._cbar.set_label(clabel, fontsize=10)
        self.ax.set_title(title + cbar_mode_note, fontsize=11)

    # ================================================================
    #  辅助
    # ================================================================
    def _update_arrow_params(self):
        self.arrow_scale = round(self.arrow_scale_var.get(), 1)
        self.arrow_lw    = round(self.arrow_lw_var.get(), 1)
        self.arrow_scale_label.config(text=f"{self.arrow_scale}×")
        self.arrow_lw_label.config(text=str(self.arrow_lw))
        # 有分析结果时实时刷新（位移或应变视图均可）
        if self.distortions is not None or self.strain_xx is not None:
            self._view_changed(self.view_var.get())

    def _on_cloud_style_changed(self):
        """应变云图配色变更: 发散型色图需按对称色标重算范围, 走整帧刷新。"""
        label = self.cloud_cmap_var.get()
        if label == STRAIN_CLOUD_AUTO_CMAP_LABEL:
            self.strain_cloud_cmap = None
            self.status.config(text="云图配色: 自动 (按分量)")
        else:
            self.strain_cloud_cmap = label
            self.status.config(text=f"云图配色: {label}")
        if self.strain_xx is not None:
            self._view_changed(self.view_var.get())

    def _on_cloud_alpha_changed(self):
        """应变云图透明度变更: 滑杆连续触发, 仅就地更新 artist, 不重建整帧。"""
        try:
            alpha = round(float(self.cloud_alpha_var.get()), 2)
        except (tk.TclError, TypeError, ValueError):
            return
        self.strain_cloud_alpha = max(0.0, min(1.0, alpha))
        self.cloud_alpha_label.config(text=f"{self.strain_cloud_alpha:.2f}")
        artist = self._strain_cloud_artist
        if artist is not None:
            artist.set_alpha(self.strain_cloud_alpha)
            self.canvas.draw_idle()

    def _displacement_cmap(self):
        """位移染色当前配色: turbo 名称或自定义红蓝 Colormap 对象 (均可直接传 scatter)。"""
        return DISPLACEMENT_RB_CMAP if self.point_cmap_key == 'rb' else 'turbo'

    def _on_point_size_changed(self):
        """原子点大小变更: 就地更新散点 artist 的 sizes, 不重建整帧。"""
        try:
            size = int(round(float(self.point_size_var.get())))
        except (tk.TclError, TypeError, ValueError):
            return
        self.atom_dot_size = max(2, size)
        self.point_size_label.config(text=str(self.atom_dot_size))
        artist = self._points_scatter_artist
        if artist is not None:
            n = len(artist.get_offsets())
            if n:
                artist.set_sizes(np.full(n, self.atom_dot_size, dtype=float))
                self.canvas.draw_idle()

    def _on_point_cmap_changed(self):
        """位移点配色变更: 颜色与色标语义随映射变化, 走整帧刷新。"""
        self.point_cmap_key = 'rb' if '红蓝' in self.point_cmap_var.get() else 'turbo'
        if self.distortions is not None:
            self._view_changed(self.view_var.get())

    def _on_colorbar_mode_changed(self):
        """色标自动/手动模式切换"""
        manual = not self.colorbar_auto_var.get()
        self.colorbar_manual = manual
        if manual:
            self.cbar_min_entry.config(state=tk.NORMAL)
            self.cbar_max_entry.config(state=tk.NORMAL)
        else:
            self.cbar_min_entry.config(state=tk.DISABLED)
            self.cbar_max_entry.config(state=tk.DISABLED)

    def _get_colorbar_range(self, data, symmetric=False):
        """
        获取色标范围，根据自动/手动模式返回 (vmin, vmax)。

        Parameters
        ----------
        data : np.ndarray  — 用于自动模式计算范围的数据
        symmetric : bool   — 是否使用对称范围（适用于 RdBu_r 色图）

        Returns
        -------
        (vmin, vmax) : tuple of float
        """
        if self.colorbar_manual:
            try:
                vmin = float(self.colorbar_vmin.get())
                vmax = float(self.colorbar_vmax.get())
                if vmax > vmin:
                    return vmin, vmax
            except (ValueError, tk.TclError):
                pass  # 输入无效时回退到自动

        # 自动模式
        if symmetric:
            p_low, p_high = np.nanpercentile(data, [2, 98])
            vmax_val = max(abs(p_low), abs(p_high))
            if vmax_val == 0 or np.isnan(vmax_val):
                vmax_val = np.nanmax(np.abs(data))
            if vmax_val == 0 or np.isnan(vmax_val):
                vmax_val = 1e-6
            return -vmax_val, vmax_val
        else:
            vmin, vmax = np.nanpercentile(data, [2, 98])
            if vmin == vmax or np.isnan(vmin):
                vmin, vmax = np.nanmin(data), np.nanmax(data)
                if vmin == vmax or np.isnan(vmin):
                    vmin, vmax = 0, 1
            return vmin, vmax

    def _remove_colorbar(self):
        """安全移除 colorbar — 容错处理 matplotlib 版本兼容性"""
        if self._cbar is not None:
            try:
                cax = self._cbar.ax
                self._cbar.remove()
                if cax is not None:
                    try:
                        self.fig.delaxes(cax)
                    except Exception:
                        pass
            except Exception:
                pass
            finally:
                self._cbar = None

    def _clear_analysis_results(self):
        # 使任何在飞的后台任务代际失效: 点表/参考/图像变化后, 旧结果不得
        # 再写回实例状态 (与 worker "只发 payload、由主线程写入" 配合闭环)。
        self._job_generation += 1
        self.ideal_grid = None
        self.matched_actual = None
        self.displacements = None
        self.distortions = None
        self.analysis_point_indices = None
        self.lattice_indices = None
        self.lattice_coords = None
        self.assignment_residuals = None
        self.assignment_residuals_px = None
        self.assignment_reassigned_mask = None
        self.assignment_low_confidence_mask = None
        self.assignment_conflict_count = 0
        # 清空应变分析结果
        self.tri_centroids = None
        self.strain_xx = None
        self.strain_yy = None
        self.strain_xy = None
        self.strain_eq = None
        self.rotation = None
        self.strain_gl_xx = None
        self.strain_gl_yy = None
        self.strain_gl_xy = None
        self.strain_gl_eq = None
        self.strain_grid = {}
        self.strain_grid_gl = {}
        self.strain_grid_extent = None
        self.tri_edge_mask = None
        self.outlier_mask = None
        self.lattice_holes = []
        self.invalid_sites = None
        self.strain_geometry = None
        self.result_locations_physical = None
        self.element_area = None
        self.strain_quality_grades = None
        self.strain_invalid_reasons = None
        self._suggested_threshold = None

    def reset_all(self):
        """重置所有状态，回到刚打开程序的样子"""
        reply = messagebox.askyesno("确认重置", "将清除图像、所有标记点和分析结果，\n恢复到刚打开程序的状态。\n\n确认重置？")
        if not reply:
            return
        self.image = None
        self.image_path = None
        self.image_frame_index = None
        self.image_metadata = None
        self.points.clear()
        self.undo_stack.clear()
        self.reference_vecs = None
        self.ref_origin = None
        self.reference_metadata = None
        self._calibrate_pts = []   # 旧图的校准点不得画到新图上
        self.displacements = None
        self.distortions = None
        self.ideal_grid = None
        self.matched_actual = None
        self.analysis_point_indices = None
        self.lattice_indices = None
        self.lattice_coords = None
        self.assignment_residuals = None
        self.assignment_residuals_px = None
        self.assignment_reassigned_mask = None
        self.assignment_low_confidence_mask = None
        self.assignment_conflict_count = 0
        # 清空应变分析结果
        self.tri_centroids = None
        self.strain_xx = None
        self.strain_yy = None
        self.strain_xy = None
        self.strain_eq = None
        self.rotation = None
        self.strain_gl_xx = None
        self.strain_gl_yy = None
        self.strain_gl_xy = None
        self.strain_gl_eq = None
        self.strain_grid = {}
        self.strain_grid_gl = {}
        self.strain_grid_extent = None
        self.tri_edge_mask = None
        self.outlier_mask = None
        self.lattice_holes = []
        self.invalid_sites = None
        self.strain_quality_grades = None
        self.strain_invalid_reasons = None
        self.selected_point_idx = None
        self.ref_select_indices = []
        self._reset_ref_multi_selection()
        self.ref_select_mode = False
        self.current_mode = "add"
        self.mode_var.set("add")
        self._cbar = None
        # 重置色标范围控制
        self.colorbar_manual = False
        self.colorbar_vmin.set(0.0)
        self.colorbar_vmax.set(1.0)
        if hasattr(self, 'colorbar_auto_var'):
            self.colorbar_auto_var.set(True)
        if hasattr(self, 'cbar_min_entry'):
            self.cbar_min_entry.config(state=tk.DISABLED)
        if hasattr(self, 'cbar_max_entry'):
            self.cbar_max_entry.config(state=tk.DISABLED)
        self.view_var.set("colored")
        if hasattr(self, 'view_cb'):
            self.view_cb.set("位移幅值染色")
        # 重置预处理状态
        self.processed_image = None
        self.show_processed = False
        self.use_preprocessed.set(False)
        self.preprocess_method = "none"
        self.preprocess_method_var.set("无")
        self.btn_toggle_display.config(text="👁 显示: 原始")
        self.preprocess_status.config(text="状态: 未应用预处理", foreground="gray")
        self._on_preprocess_method_changed()
        # 清除检测区域和放大状态
        self.detect_roi = None
        self.detect_polygon = None
        self._roi_mode_active = False
        self._polygon_mode_active = False
        self._polygon_vertices = []
        self._drawing_roi = None
        self._cleanup_roi_preview()
        self._zoom_history.clear()
        self.btn_roi.config(text="⬜ 选区")
        self.btn_polygon.config(text="✏️ 自由选区")
        # 清除参考区
        self.ref_region = None
        self._ref_region_mode_active = False
        self._drawing_ref_region = None
        self._cleanup_ref_region_preview()
        self.btn_ref_region.config(text="🧭 框选参考区")
        self._update_ref_region_label()
        self._img_artist = None     # 必须清除缓存，否则 display_image 引用死 artist
        self.ax.clear()
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("就绪 — 请导入图像")
        # 强制刷新 Canvas 以确保新 Figure 状态正确渲染
        self.canvas.draw()
        self.update_listbox()
        self._update_ref_status()
        self.ref_status_label.config(text="未设置", foreground="gray")
        self.status.config(foreground='', text="已重置，就绪")

    def _record_undo(self, action, data):
        """记录撤销项并封顶, 防止超长会话内存无限增长。"""
        self.undo_stack.append((action, data))
        overflow = len(self.undo_stack) - self.MAX_UNDO_ENTRIES
        if overflow > 0:
            del self.undo_stack[:overflow]

    def add_point_with_check(self, x, y):
        """
        添加原子点前检查与现有点的最小间距。
        如果新点与现有点间距 < detect_min_dist * 0.3，拒绝添加。

        Returns
        -------
        bool : 是否成功添加
        """
        if self.points:
            pts = np.array(self.points)
            dists = np.sqrt((pts[:, 0] - x)**2 + (pts[:, 1] - y)**2)
            min_dist = dists.min()
            if min_dist < self.detect_min_dist * 0.3:
                return False
        idx = len(self.points)
        self.points.append((x, y))
        self._record_undo("add", (idx, (x, y)))
        return True

    def deduplicate_points(self, tolerance=None):
        """
        删除间距 < tolerance 的重复/过近原子点。
        保留强度更大（先被检测到）的点。

        Parameters
        ----------
        tolerance : float or None
            最小允许间距（像素）。默认 = detect_min_dist * 0.3

        Returns
        -------
        int : 删除的点数
        """
        if len(self.points) < 2:
            return 0
        if tolerance is None:
            tolerance = self.detect_min_dist * 0.3
        pts = np.array(self.points)
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        # 找所有距离 < tolerance 的点对
        pairs = tree.query_pairs(tolerance)
        if not pairs:
            return 0

        # 按索引排序，删除较大的索引（保留先添加的点）
        to_remove = set()
        for i, j in pairs:
            to_remove.add(max(i, j))

        if to_remove:
            removed_count = len(to_remove)
            # 保存撤销信息
            removed_points = [(idx, self.points[idx]) for idx in sorted(to_remove, reverse=True)]
            self._record_undo("dedup", removed_points)
            # 从大到小删除以保持索引正确
            for idx in sorted(to_remove, reverse=True):
                del self.points[idx]
            # 点表变化必须使既有分析结果与选择状态失效，否则视图切换会因
            # 长度不匹配崩溃、旧应变结果会被当作当前点表的结果导出
            self.selected_point_idx = None
            self.ref_select_indices = []
            self._reset_ref_multi_selection()
            self._clear_analysis_results()
            self._refresh_overlay()
            self.status.config(text=f"去重完成: 删除了 {removed_count} 个过近点")
            return removed_count
        return 0

    def clear_points(self):
        if self.points:
            self._record_undo("clear", list(self.points))
        self.points.clear()
        self.selected_point_idx = None
        self.ref_select_indices = []
        self._reset_ref_multi_selection()
        self._clear_analysis_results()
        self._refresh_overlay()
        self.status.config(foreground='', text="已清除所有点  (Ctrl+Z 可恢复)")

    def undo_last(self):
        """撤销上一步操作 (Ctrl+Z) — 支持添加/删除/清除"""
        if not self.undo_stack:
            self.status.config(text="没有可撤销的操作")
            return

        action, data = self.undo_stack.pop()
        if action == "add":
            # 撤销"添加": 删除最后加的点
            idx, (x, y) = data
            if 0 <= idx < len(self.points):
                self.points.pop(idx)
            self.status.config(text=f"已撤销添加点 ({x:.0f},{y:.0f}), 剩余 {len(self.points)} 个")
        elif action == "delete":
            # 撤销"删除": 把点插回原来的位置
            idx, (x, y) = data
            self.points.insert(idx, (x, y))
            self.status.config(text=f"已恢复点 {idx+1} ({x:.0f},{y:.0f})")
        elif action == "clear":
            # 撤销"清除": 恢复所有被清除的点
            self.points = data
            self.status.config(text=f"已恢复 {len(data)} 个被清除的点")
        elif action == "dedup":
            # 撤销"去重": 按索引恢复被删除的点
            for idx, (x, y) in sorted(data):
                self.points.insert(idx, (x, y))
            self.status.config(text=f"已恢复 {len(data)} 个被去重的点")
        self.selected_point_idx = None
        # 撤销会改变点序，残留的参考选择索引可能越界
        self.ref_select_indices = []
        self._reset_ref_multi_selection()
        self._clear_analysis_results()
        self._refresh_overlay()

    def _update_vm_coeff(self):
        """von Mises 系数更新回调"""
        try:
            self.von_mises_coeff = float(self.vm_var.get())
        except ValueError:
            pass  # 忽略非法输入，保持旧值

    # ================================================================
    #  版本化项目存储 (schema v2: 含图像 SHA-256 校验与原子写入)
    # ================================================================
    def save_project(self):
        if self.image is None or not self.image_path:
            messagebox.showinfo("提示", "请先导入图像。")
            return
        file = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON 文件", "*.json")], title="保存项目")
        if not file:
            return
        reference = None
        if self.reference_vecs is not None and self.ref_origin is not None:
            a_vec, b_vec = self.reference_vecs
            reference = {
                'origin': [float(v) for v in self.ref_origin],
                'a_vec': [float(v) for v in a_vec],
                'b_vec': [float(v) for v in b_vec],
                'estimation': getattr(self, 'reference_metadata', None),
            }
        algorithm_id = LOCAL_PPA_ALGORITHM_ID if self.analysis_method == "peak_pairs" else "lattice-cst-legacy"
        assignment_payload = None
        if getattr(self, 'lattice_indices', None) is not None and len(self.lattice_indices) == len(self.points):
            assignment_payload = {
                'lattice_indices': np.asarray(self.lattice_indices, dtype=int).tolist(),
                'lattice_coords': np.asarray(self.lattice_coords, dtype=float).tolist() if getattr(self, 'lattice_coords', None) is not None else None,
                'residuals_lattice': np.asarray(self.assignment_residuals, dtype=float).tolist() if getattr(self, 'assignment_residuals', None) is not None else None,
                'residuals_px': np.asarray(self.assignment_residuals_px, dtype=float).tolist() if getattr(self, 'assignment_residuals_px', None) is not None else None,
                'reassigned': np.asarray(self.assignment_reassigned_mask, dtype=bool).tolist() if getattr(self, 'assignment_reassigned_mask', None) is not None else None,
                'low_confidence': np.asarray(self.assignment_low_confidence_mask, dtype=bool).tolist() if getattr(self, 'assignment_low_confidence_mask', None) is not None else None,
                'initial_conflict_count': int(getattr(self, 'assignment_conflict_count', 0)),
                'strain_quality': np.asarray(self.strain_quality_grades, dtype=str).tolist() if getattr(self, 'strain_quality_grades', None) is not None else None,
                'strain_invalid_reasons': np.asarray(self.strain_invalid_reasons, dtype=str).tolist() if getattr(self, 'strain_invalid_reasons', None) is not None else None,
            }
        payload = {
            'points': [[float(x), float(y)] for x, y in self.points],
            'detect_roi': list(self.detect_roi) if self.detect_roi else None,
            'detect_polygon': [[float(x), float(y)] for x, y in self.detect_polygon] if self.detect_polygon else None,
            'reference': reference,
            'ref_region': [float(v) for v in self.ref_region] if self.ref_region else None,
            'image_metadata': self.image_metadata,
            'detect_params': {
                'sigma': self.detect_sigma, 'min_dist': self.detect_min_dist,
                'window': self.detect_window, 'method': self.centroid_method,
                # 工单27: 高斯精炼回退比例入档, 便于追溯坐标质量
                'gaussian_fallback': getattr(self, 'last_gaussian_refine_stats', None),
            },
            # 预处理设置一并存档, 否则加载后无法复现"用预处理图检测"的流程
            'preprocess': {
                'method': self.preprocess_method,
                'params': dict(self.preprocess_params) if self.preprocess_params else {},
                'use_preprocessed': bool(self.use_preprocessed.get()),
            },
            'analysis': {
                'algorithm_id': algorithm_id,
                'equivalent_strain_coefficient': self.von_mises_coeff,
                'outlier_indices': np.flatnonzero(self.outlier_mask).tolist() if self.outlier_mask is not None else [],
                'lattice_assignment': assignment_payload,
            },
            'colorbar': {'manual': self.colorbar_manual, 'vmin': self.colorbar_vmin.get(), 'vmax': self.colorbar_vmax.get()},
        }
        try:
            save_versioned_project(file, payload, self.image_path, frame_index=self.image_frame_index)
        except Exception as error:
            messagebox.showerror("保存失败", f"无法保存项目：\n{error}")
            return
        self.status.config(text=f"✓ 项目已保存（v2，含图像校验）：{os.path.basename(file)}")

    # 预处理参数键 → (类型转换, 默认值)；加载时只接受这些已知键。
    _PREPROCESS_PARAM_TYPES = {
        'gaussian_sigma': (float, 1.0),
        'median_size': (int, 3),
        'bw_order': (int, 4),
        'bw_cutoff': (float, 0.3),
        'delta': (float, 5.0),
        'cycles': (int, 99),
        'step': (int, 2),
    }

    def _restore_preprocess_settings(self, data):
        """从项目数据恢复预处理设置 (对应 save_project 写出的 'preprocess' 块)。

        此前只保存不加载：重开项目后滤波方法/参数/开关全部回到默认，
        无法复现「用预处理图检测」的流程。非法值一律忽略并保持默认。
        """
        preprocess = data.get('preprocess')
        if not isinstance(preprocess, dict):
            return
        method = preprocess.get('method', 'none')
        # 方法名 → 预处理面板的下拉显示文本 (butter.py 缺失时带 "(不可用)" 后缀)
        label_map = {
            "none": "无",
            "gaussian": "Gaussian 高斯模糊",
            "median": "Median 中值滤波",
        }
        if HRTEMFilter is not None:
            label_map.update({
                "butterworth": "Butterworth 低通",
                "wiener": "Wiener 维纳滤波",
                "absf": "ABSF 滤波",
            })
        else:
            label_map.update({
                "butterworth": "Butterworth 低通 (不可用)",
                "wiener": "Wiener 维纳滤波 (不可用)",
                "absf": "ABSF 滤波 (不可用)",
            })
        label = label_map.get(method)
        if label is not None:
            self.preprocess_method_var.set(label)
            self.preprocess_method = method
        params = preprocess.get('params', {})
        if isinstance(params, dict):
            for key, (cast, default) in self._PREPROCESS_PARAM_TYPES.items():
                value = params.get(key, default)
                try:
                    self.preprocess_params[key] = cast(value)
                except (TypeError, ValueError):
                    self.preprocess_params[key] = default
        use_flag = preprocess.get('use_preprocessed', False)
        self.use_preprocessed.set(bool(use_flag))
        # 用恢复后的参数重建参数面板
        self._on_preprocess_method_changed()

    def load_project(self):
        file = filedialog.askopenfilename(filetypes=[("JSON 文件", "*.json")], title="加载项目")
        if not file:
            return
        try:
            data = load_versioned_project(file)
        except ProjectValidationError as error:
            messagebox.showerror("加载已阻止", f"项目图像无法验证，因此不会恢复坐标：\n{error}")
            return
        except Exception as error:
            messagebox.showerror("加载失败", f"无法读取项目文件：\n{error}")
            return

        if data.get('legacy_unverified'):
            image_path = data.get('image_path')
            if not image_path or not os.path.exists(image_path):
                messagebox.showerror("加载已阻止", "旧项目没有可验证的原图像；不会把坐标叠加到当前图像。")
                return
            if not messagebox.askyesno(
                "旧项目未校验", "这是 v1 历史项目，未保存图像哈希。确认原图正确后才会载入；建议立即另存为 v2。\n\n继续？"):
                return
            frame_index = None
        else:
            image = data['image']
            image_path = image['path']
            frame_index = image.get('frame_index')
        try:
            self._load_image_path(image_path, frame_index=frame_index)
        except ImageLoadError as error:
            messagebox.showerror("加载已阻止", f"图像无法加载；不会恢复坐标：\n{error}")
            return

        points = [(float(x), float(y)) for x, y in data.get('points', [])]
        height, width = self.image.shape
        if any(x < 0 or y < 0 or x >= width or y >= height for x, y in points):
            messagebox.showerror("加载已阻止", "项目坐标超出经过验证的图像范围；不会恢复坐标。")
            return
        self.points = points
        self.undo_stack.clear()
        params = data.get('detect_params', {})
        self.detect_sigma = float(params.get('sigma', self.detect_sigma))
        self.detect_min_dist = float(params.get('min_dist', self.detect_min_dist))
        self.detect_window = int(params.get('window', self.detect_window))
        self.centroid_method = params.get('method', self.centroid_method)
        # 工单27: 恢复高斯精炼回退统计 (旧项目文件无此键 -> 保持 None)
        fallback_stats = params.get('gaussian_fallback')
        if isinstance(fallback_stats, dict) and \
                {'n_points', 'n_fallback', 'fallback_ratio'} <= set(fallback_stats):
            self.last_gaussian_refine_stats = {
                'n_points': int(fallback_stats['n_points']),
                'n_fallback': int(fallback_stats['n_fallback']),
                'fallback_ratio': float(fallback_stats['fallback_ratio']),
            }
        else:
            self.last_gaussian_refine_stats = None
        reference = data.get('reference')
        self.reference_metadata = None
        if reference:
            try:
                # 项目里的参考晶格也必须按当前阈值校验，否则加载时会被隐藏默认
                # 30 拒绝（用户放宽后保存的项目无法恢复）。
                validate_reference_lattice(reference['a_vec'], reference['b_vec'],
                                           max_condition=self.lattice_max_condition)
                self.reference_vecs = (np.asarray(reference['a_vec'], dtype=float), np.asarray(reference['b_vec'], dtype=float))
                self.ref_origin = np.asarray(reference['origin'], dtype=float)
                estimation = reference.get('estimation')
                if isinstance(estimation, dict):
                    self.reference_metadata = dict(estimation)
            except (KeyError, AnalysisError) as error:
                messagebox.showwarning("参考晶格未恢复", f"项目中的参考晶格无效：{error}")
        # ROI / 参考区 / 自由选区必须落在经过校验的图像范围内，否则不恢复
        h_img, w_img = self.image.shape

        def _rect_ok(rect):
            try:
                x0, y0, x1, y1 = (float(v) for v in rect)
            except (TypeError, ValueError):
                return False
            return (0.0 <= x0 < x1 <= w_img) and (0.0 <= y0 < y1 <= h_img)

        skipped_regions = []
        self.detect_roi = None
        self.detect_polygon = None
        self.ref_region = None
        roi = data.get('detect_roi')
        if roi:
            if _rect_ok(roi):
                self.detect_roi = tuple(float(v) for v in roi)
            else:
                skipped_regions.append("检测区域")
        polygon = data.get('detect_polygon')
        if polygon:
            try:
                polygon_pts = [(float(x), float(y)) for x, y in polygon]
                polygon_ok = (
                    len(polygon_pts) >= 3
                    and all(0.0 <= x <= w_img and 0.0 <= y <= h_img for x, y in polygon_pts)
                )
            except (TypeError, ValueError):
                polygon_ok = False
            if polygon_ok:
                self.detect_polygon = polygon_pts
            else:
                skipped_regions.append("自由选区")
        ref_region = data.get('ref_region')
        if ref_region:
            if _rect_ok(ref_region):
                self.ref_region = tuple(float(v) for v in ref_region)
            else:
                skipped_regions.append("参考区")
        if skipped_regions:
            messagebox.showwarning(
                "部分区域未恢复",
                "以下项目超出当前图像范围，已忽略：\n" + "、".join(skipped_regions))
        self._update_ref_region_label()
        analysis = data.get('analysis', {})
        self.von_mises_coeff = float(analysis.get('equivalent_strain_coefficient', data.get('von_mises_coeff', 4.0 / 9.0)))
        self.vm_var.set(f"{self.von_mises_coeff:.4f}")
        algorithm_id = analysis.get('algorithm_id', '')
        if algorithm_id == 'lattice-cst-legacy':
            self.analysis_method = 'lattice_cst'
            self.analysis_method_var.set("兼容：晶格-CST（三角形）")
        else:
            self.analysis_method = 'peak_pairs'
            self.analysis_method_var.set("局部 Peak Pairs (推荐)")
        saved_assignment = analysis.get('lattice_assignment')
        if isinstance(saved_assignment, dict):
            try:
                saved_indices = np.asarray(saved_assignment.get('lattice_indices'), dtype=int)
                if saved_indices.shape != (len(self.points), 2):
                    raise ValueError("晶格索引数量与原子点数量不一致")
                if len({tuple(row) for row in saved_indices}) != len(saved_indices):
                    raise ValueError("项目中的晶格索引不是一对一分配")

                def _saved_vector(key, dtype=float):
                    value = saved_assignment.get(key)
                    if value is None:
                        return None
                    array = np.asarray(value, dtype=dtype)
                    if array.shape != (len(self.points),):
                        raise ValueError(f"{key} 数量与原子点数量不一致")
                    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
                        raise ValueError(f"{key} 包含非有限值")
                    return array

                saved_coords = saved_assignment.get('lattice_coords')
                if saved_coords is not None:
                    saved_coords = np.asarray(saved_coords, dtype=float)
                    if saved_coords.shape != (len(self.points), 2) or not np.isfinite(saved_coords).all():
                        raise ValueError("lattice_coords 无效")
                self.lattice_indices = saved_indices
                self.lattice_coords = saved_coords
                self.analysis_point_indices = np.arange(len(self.points), dtype=int)
                self.assignment_residuals = _saved_vector('residuals_lattice')
                self.assignment_residuals_px = _saved_vector('residuals_px')
                self.assignment_reassigned_mask = _saved_vector('reassigned', bool)
                self.assignment_low_confidence_mask = _saved_vector('low_confidence', bool)
                self.assignment_conflict_count = int(saved_assignment.get('initial_conflict_count', 0))
                self.strain_quality_grades = _saved_vector('strain_quality', str)
                self.strain_invalid_reasons = _saved_vector('strain_invalid_reasons', str)
            except (TypeError, ValueError) as error:
                messagebox.showwarning("分析元数据未恢复", f"项目中的历史晶格分配无效，已忽略：{error}")
        self._update_ref_status()
        # 恢复预处理设置 (save_project 的 'preprocess' 块)
        self._restore_preprocess_settings(data)
        # 恢复色标设置 (v2 嵌套 'colorbar' 键; 兼容 v1 历史项目的扁平键)
        colorbar = data.get('colorbar')
        if isinstance(colorbar, dict):
            cbar_manual = bool(colorbar.get('manual', False))
            cbar_vmin, cbar_vmax = colorbar.get('vmin', 0.0), colorbar.get('vmax', 1.0)
        else:
            cbar_manual = bool(data.get('colorbar_manual', False))
            cbar_vmin, cbar_vmax = data.get('colorbar_vmin', 0.0), data.get('colorbar_vmax', 1.0)
        self.colorbar_auto_var.set(not cbar_manual)
        try:
            self.colorbar_vmin.set(float(cbar_vmin))
            self.colorbar_vmax.set(float(cbar_vmax))
        except (TypeError, ValueError):
            self.colorbar_vmin.set(0.0)
            self.colorbar_vmax.set(1.0)
        self._on_colorbar_mode_changed()
        self.display_image()
        label = "旧版未校验项目" if data.get('legacy_unverified') else "已验证项目"
        self.status.config(text=f"✓ 已加载{label}：{os.path.basename(file)}")

    # ================================================================
    #  导出
    # ================================================================
    def save_coordinates(self):
        if not self.points:
            messagebox.showinfo("提示", "没有可保存的点。")
            return
        file = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV 文件", "*.csv")],
                                            title="保存原子坐标")
        if not file:
            return
        # utf-8-sig 带 BOM: 中文表头在 Excel (中文 Windows 默认 GBK) 下不乱码;
        # ppa_stats 的加载器同样按 utf-8-sig 读取, 双向兼容。
        try:
            with open(file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["编号", "x", "y"])
                for i, (x, y) in enumerate(self.points):
                    writer.writerow([i + 1, f"{x:.4f}", f"{y:.4f}"])
        except OSError as error:
            messagebox.showerror("导出失败", f"无法写入文件 (路径/权限?)：\n{error}")
            return
        self.status.config(text=f"✓ 坐标已保存: {os.path.basename(file)}")

    def save_marked_image(self):
        if self.image is None:
            return
        file = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG 文件", "*.png")],
                                            title="保存标记图像")
        if not file:
            return
        fig = Figure(figsize=(8, 8), dpi=200)
        ax = fig.add_subplot(111)
        ax.imshow(self._get_display_image(), cmap='gray', origin='upper')
        if self.points:
            pts = np.array(self.points)
            ax.scatter(pts[:, 0], pts[:, 1], c='lime', s=25,
                       marker='o', edgecolor='black', linewidth=0.5)
            for i, (x, y) in enumerate(pts):
                ax.text(x, y - 12, str(i + 1), color='white', fontsize=7,
                        ha='center', va='top', fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])
        ax.axis('off')
        fig.tight_layout(pad=0)
        try:
            fig.savefig(file, dpi=200, bbox_inches='tight', pad_inches=0)
        except OSError as error:
            messagebox.showerror("导出失败", f"无法写入文件 (路径/权限?)：\n{error}")
            return
        finally:
            plt.close(fig)
        self.status.config(text=f"✓ 标记图已保存: {os.path.basename(file)}")

    def save_current_view(self):
        """保存当前视图 — 自动命名，保留缩放状态"""
        if self.image is None:
            return

        # ---- 自动生成文件名 ----
        if self.image_path:
            base = os.path.splitext(os.path.basename(self.image_path))[0]
            save_dir = os.path.dirname(self.image_path)
        else:
            base = "未命名"
            save_dir = "."
        key_to_label = {
            "colored": "位移幅值", "vector": "位移矢量",
            "strain_xx": "应变_xx", "strain_yy": "应变_yy",
            "strain_xy": "应变_xy", "strain_eq": "等效应变",
            "rotation": "旋转", "markers": "标记图",
            "strain_gl_xx": "GL_应变_xx", "strain_gl_yy": "GL_应变_yy",
            "strain_gl_xy": "GL_应变_xy", "strain_gl_eq": "GL_等效应变",
        }
        view_key = self.view_var.get()
        suffix = key_to_label.get(view_key, view_key)
        default_name = f"{base}_{suffix}.png"
        default_path = os.path.join(save_dir, default_name)

        file = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG 文件", "*.png")],
            title="保存当前视图")
        if not file:
            return

        # ---- 直接保存当前 figure（保持缩放和所有叠加层） ----
        # 保存完整的当前显示状态，包含缩放、颜色条、图例等
        old_title = self.ax.get_title()
        self.ax.set_title("")  # 去掉标题栏
        try:
            self.fig.savefig(file, dpi=300, bbox_inches='tight', pad_inches=0)
        except OSError as error:
            messagebox.showerror("导出失败", f"无法写入文件 (路径/权限?)：\n{error}")
        finally:
            self.ax.set_title(old_title)
        self.status.config(text=f"✓ 视图已保存: {os.path.basename(file)}")

    def save_displacement_csv(self):
        """保存每个原子的位移数据到CSV"""
        if self.displacements is None or self.matched_actual is None:
            messagebox.showinfo("提示", "请先执行PPA分析。")
            return
        file = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV 文件", "*.csv")],
                                            title="保存位移数据")
        if not file:
            return
        physical_actual = self.matched_actual * np.array([1.0, -1.0])
        physical_ideal = self.ideal_grid * np.array([1.0, -1.0])
        physical_displacements = physical_actual - physical_ideal
        n_row = len(self.displacements)
        if len(self.matched_actual) != n_row or len(self.ideal_grid) != n_row:
            messagebox.showerror("导出失败", "位移结果与确认原子数量不一致，请重新执行 PPA。")
            return
        lattice_indices = (np.asarray(self.lattice_indices, dtype=int)
                           if self.lattice_indices is not None else np.zeros((n_row, 2), dtype=int))
        residuals = (np.asarray(self.assignment_residuals, dtype=float)
                     if self.assignment_residuals is not None else np.full(n_row, np.nan))
        residuals_px = (np.asarray(self.assignment_residuals_px, dtype=float)
                        if self.assignment_residuals_px is not None else np.full(n_row, np.nan))
        reassigned = (np.asarray(self.assignment_reassigned_mask, dtype=bool)
                      if self.assignment_reassigned_mask is not None else np.zeros(n_row, dtype=bool))
        low_confidence = (np.asarray(self.assignment_low_confidence_mask, dtype=bool)
                          if self.assignment_low_confidence_mask is not None else np.zeros(n_row, dtype=bool))
        try:
            with open(file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["编号", "实际x（物理）", "实际y（物理）", "参考x（物理）", "参考y（物理）",
                                 "位移dx（物理）", "位移dy（物理）", "位移幅值",
                                 "晶格n", "晶格m", "索引残差（晶格单位）", "索引残差（px）",
                                 "索引重分配", "低置信度"])
                for i in range(n_row):
                    writer.writerow([i + 1,
                                     f"{physical_actual[i, 0]:.4f}",
                                     f"{physical_actual[i, 1]:.4f}",
                                     f"{physical_ideal[i, 0]:.4f}",
                                     f"{physical_ideal[i, 1]:.4f}",
                                     f"{physical_displacements[i, 0]:.4f}",
                                     f"{physical_displacements[i, 1]:.4f}",
                                     f"{self.distortions[i]:.4f}",
                                     int(lattice_indices[i, 0]), int(lattice_indices[i, 1]),
                                     f"{residuals[i]:.6f}", f"{residuals_px[i]:.6f}",
                                     int(reassigned[i]), int(low_confidence[i])])
        except OSError as error:
            messagebox.showerror("导出失败", f"无法写入文件 (路径/权限?)：\n{error}")
            return
        # 位移 CSV 同样写约定 sidecar: ppa_stats 否则只能靠 y<0 启发式猜测
        # 物理约定, 原点选在视场中部时 (y 有正有负) 会猜错并翻转 ε_xy/θ 符号
        metadata_path = os.path.splitext(file)[0] + ".metadata.json"
        try:
            import json
            with open(metadata_path, 'w', encoding='utf-8') as meta_file:
                json.dump({
                    'data_type': 'displacement',
                    'coordinate_convention': 'physical-cartesian-x-right-y-up; image-display-y-down',
                    'algorithm_id': LOCAL_PPA_ALGORITHM_ID if self.analysis_method == 'peak_pairs' else 'lattice-cst-legacy',
                    'image_path': self.image_path,
                    'image_frame_index': self.image_frame_index,
                    'reference_lattice': self._reference_lattice_metadata(),
                    'row_count': n_row,
                    'all_confirmed_atoms_included': True,
                    'quality_columns': {
                        '索引重分配': '1=由全局一对一匹配调整索引; 0=保留初始取整索引',
                        '低置信度': '1=匹配残差较大但仍保留; 0=正常',
                    },
                }, meta_file, ensure_ascii=False, indent=2)
        except OSError:
            pass
        self.status.config(text=f"✓ 位移数据已保存: {os.path.basename(file)}")

    def save_strain_csv(self):
        """Export strain with an explicit element type and a reproducibility sidecar."""
        if self.tri_centroids is None or self.strain_xx is None:
            messagebox.showinfo("提示", "请先执行PPA分析 (应变数据未生成)。")
            return
        file = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV 文件", "*.csv")],
                                            title="保存应变张量数据")
        if not file:
            return
        # 准备 GL 数据（若可用）
        has_gl = self.strain_gl_xx is not None
        has_edge = self.tri_edge_mask is not None
        has_area = self.element_area is not None
        is_site_result = self.strain_geometry == "sites"

        element_label = "原子位点" if self.strain_geometry == "sites" else "三角形"
        export_locations = self.result_locations_physical if self.result_locations_physical is not None else self.tri_centroids
        try:
            with open(file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                header = [f"{element_label}编号", "位置x", "位置y",
                          "ε_xx", "ε_yy", "ε_xy (张量剪应变)", "ε_eq (von Mises)", "极分解旋转 θ (rad)"]
                if has_gl:
                    header += ["ε_xx (GL)", "ε_yy (GL)", "ε_xy (GL)", "ε_eq (GL von Mises)"]
                if has_edge:
                    header += ["边缘三角形"]
                if has_area:
                    header += ["参考三角形面积"]
                if is_site_result:
                    header += ["原始点编号", "晶格n", "晶格m", "应变有效", "计算质量", "无效原因"]
                writer.writerow(header)

                for t in range(len(self.tri_centroids)):
                    row = [t + 1,
                           f"{export_locations[t, 0]:.4f}",
                           f"{export_locations[t, 1]:.4f}",
                           f"{self.strain_xx[t]:.6f}",
                           f"{self.strain_yy[t]:.6f}",
                           f"{self.strain_xy[t]:.6f}",
                           f"{self.strain_eq[t]:.6f}",
                           f"{self.rotation[t]:.6f}"]
                    if has_gl:
                        row += [f"{self.strain_gl_xx[t]:.6f}",
                                f"{self.strain_gl_yy[t]:.6f}",
                                f"{self.strain_gl_xy[t]:.6f}",
                                f"{self.strain_gl_eq[t]:.6f}"]
                    if has_edge:
                        row += [1 if self.tri_edge_mask[t] else 0]  # 数值标记, 保证 ppa_stats 可解析
                    if has_area:
                        row += [f"{self.element_area[t]:.6f}"]
                    if is_site_result:
                        lattice_n, lattice_m = (self.lattice_indices[t]
                                                if self.lattice_indices is not None else (0, 0))
                        quality = (str(self.strain_quality_grades[t])
                                   if self.strain_quality_grades is not None else "unknown")
                        reason = (str(self.strain_invalid_reasons[t])
                                  if self.strain_invalid_reasons is not None else "")
                        row += [t + 1, int(lattice_n), int(lattice_m),
                                int(np.isfinite(self.strain_xx[t])), quality, reason]
                    writer.writerow(row)
        except OSError as error:
            messagebox.showerror("导出失败", f"无法写入文件 (路径/权限?)：\n{error}")
            return
        metadata_path = os.path.splitext(file)[0] + ".metadata.json"
        try:
            import json
            with open(metadata_path, 'w', encoding='utf-8') as meta_file:
                json.dump({
                    'algorithm_id': LOCAL_PPA_ALGORITHM_ID if self.analysis_method == 'peak_pairs' else 'lattice-cst-legacy',
                    'element_geometry': self.strain_geometry,
                    'coordinate_convention': 'physical-cartesian-x-right-y-up; image-display-y-down',
                    'shear_definition': 'epsilon_xy is tensor shear; engineering_gamma_xy = 2 * epsilon_xy',
                    'equivalent_strain_coefficient': self.von_mises_coeff,
                    'image_path': self.image_path,
                    'image_frame_index': self.image_frame_index,
                    'reference_lattice': self._reference_lattice_metadata(),
                    'row_count': len(self.tri_centroids),
                    'all_confirmed_atoms_included': bool(is_site_result),
                    'strain_quality_grades': {
                        'A-symmetric': '四向对称 Peak Pairs',
                        'B-local-fit': '至少三个邻居的局部拟合',
                        'C-minimal-fit': '两个非共线邻居的最小拟合',
                        'invalid': '几何不足，应变为 NaN，位移仍有效',
                    } if is_site_result else None,
                }, meta_file, ensure_ascii=False, indent=2)
        except OSError:
            pass
        self.status.config(text=f"✓ 应变数据已保存: {os.path.basename(file)}  ({len(self.tri_centroids)} 个{element_label})")


if __name__ == "__main__":
    root = tk.Tk()
    app = AtomMarkerApp(root)
    root.mainloop()
