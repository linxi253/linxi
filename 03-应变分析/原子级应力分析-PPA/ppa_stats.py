"""
PPA 数据统计分析与可视化工具 (PPA Stats) v1.0
============================================
功能:
  - 导入 PPA 导出的 CSV 数据 (原子坐标、位移、应变张量)
  - 位移场统计：幅值分布、方向玫瑰图、各向异性分析
  - 应变张量统计：分量描述、主应变分析、变形模式分类、应变梯度
  - 相关性分析：ε_xx vs ε_yy、相关矩阵、应变-密度关系
  - 空间分布：2D 云图插值、线扫描剖面、径向分布
  - 报告导出：多面板综合图 + 统计数据 CSV + 摘要报告

依赖: numpy, scipy, matplotlib, tifffile, tkinter
"""

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except ModuleNotFoundError:
    # 无 Tk 环境（如最小测试容器）仍可导入纯数值函数。
    tk = None
    ttk = filedialog = messagebox = None
import numpy as np
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
except Exception:
    FigureCanvasTkAgg = NavigationToolbar2Tk = None
from matplotlib.figure import Figure
from scipy.interpolate import griddata
from scipy.stats import describe, skew, kurtosis, linregress, gaussian_kde
from scipy.spatial import cKDTree
import csv
import json
import os
import sys
import datetime


# ================================================================
#  数据容器
# ================================================================

class PPAData:
    """统一存放 PPA 三种 CSV 导入的数据"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.atoms = None         # (N, 2) 原子坐标 x, y
        self.displacements = None # (N, 2) 位移 dx, dy
        self.distortions = None   # (N,)   位移幅值
        self.ideal_pos = None     # (N, 2) 理想格点坐标
        self.actual_pos = None    # (N, 2) 实际原子坐标

        self.tri_centroids = None # (M, 2) 三角形质心
        self.strain_xx = None     # (M,)
        self.strain_yy = None
        self.strain_xy = None
        self.strain_eq = None
        self.rotation = None
        self.strain_gl = {}       # dict of (M,) GL 分量
        self.tri_edge = None      # (M,) bool
        self.element_area = None  # (M,) reference-element areas for weighted CST statistics
        self.tri_vertices = None  # (M, 3) 顶点索引 (从PPA三角形顶点CSV导入)

        self.image = None         # 底图
        self.image_path = None

        self.n_atoms = 0
        self.n_tri = 0
        # 源 CSV 是否为物理坐标 (y 向上) 约定; 导入时已统一翻转为显示坐标
        self.physical_source = False
        self.algorithm_id = None   # sidecar 记录的算法来源 (peak-pairs-local 等)

    def check_algorithm_consistency(self, new_algorithm_id):
        """两类算法 (局部PPA/晶格CST) 的结果不可混合统计; 返回不一致提示或 None。"""
        if new_algorithm_id and self.algorithm_id and new_algorithm_id != self.algorithm_id:
            return (f"数据来源算法不一致: 已导入 {self.algorithm_id}, "
                    f"本次导入 {new_algorithm_id}。\n"
                    "局部 Peak Pairs 与晶格-CST 的结果不可混合统计 "
                    "(元素几何与覆盖范围不同)。")
        return None

    def has_displacement(self):
        return self.displacements is not None

    def has_strain(self):
        return self.strain_xx is not None

    def has_atoms(self):
        return self.atoms is not None

    def has_image(self):
        return self.image is not None


# ================================================================
#  CSV 导入器
# ================================================================

def read_export_metadata(filepath):
    """
    读取 ppa.py 导出的 `<name>.metadata.json` 旁车文件 (若存在)。

    v3 起主程序以物理笛卡尔坐标 (x 右、y 上) 导出，而本工具在图像显示坐标
    (y 下) 中绘图。旁车文件记录了 coordinate_convention，用于确定是否需要翻转 y。

    Returns
    -------
    dict  (无旁车文件时返回 {})
    """
    meta_path = os.path.splitext(str(filepath))[0] + ".metadata.json"
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def detect_physical_convention(y_values, metadata=None):
    """
    判断一组 y 坐标是否为物理约定 (y 向上)。

    优先用旁车 metadata 的 coordinate_convention: 含 'physical' 即物理坐标,
    记录了其他明确约定则按显示坐标处理。无 metadata (或字符串无法识别) 时
    退化为启发式: 图像显示坐标是行号, 恒 ≥ 0, 故出现负 y 即为物理坐标;
    这也覆盖了参考原点选在视场中部 (物理 y 有正有负) 的情形。

    Returns
    -------
    bool
    """
    if metadata:
        convention = str(metadata.get('coordinate_convention', ''))
        if 'physical' in convention:
            return True
        if convention:
            return False
    y = np.asarray(y_values, dtype=float)
    if y.size == 0:
        return False
    return bool(np.any(y < 0))


def convention_detection_note(y_values, metadata):
    """坐标约定判定过程的可追溯说明; 判定可靠 (有 sidecar) 时返回 None。

    无 sidecar 时启发式虽然覆盖了自家导出的全部情形, 但无法排除
    第三方按其他原点导出的物理坐标, 提示用户核对 ε_xy/θ 符号。
    """
    if metadata and str(metadata.get('coordinate_convention', '')):
        return None
    y = np.asarray(y_values, dtype=float)
    if y.size == 0:
        return None
    if np.any(y < 0):
        guessed = "检测到负 y → 判为物理坐标 (y 向上), 已翻转为显示坐标"
    else:
        guessed = "y 恒 ≥ 0 → 判为图像显示坐标 (y 向下), 未做翻转"
    return ("未找到 .metadata.json, 坐标约定按启发式判断: " + guessed +
            "。若判断有误, y 方向与 ε_xy/θ 符号会同时颠倒; "
            "请用主程序重新导出 (自动生成 sidecar) 以获得可靠判定。")


def _flip_y_inplace(arrays):
    """把若干 (N,2) 数组的 y 分量取反 (物理 y-up ↔ 显示 y-down)。"""
    for arr in arrays:
        if arr is not None and len(arr):
            arr[:, 1] *= -1.0


def orient_spatial_axes(ax):
    """空间分布图的 y 轴与原图对齐。

    质心/原子坐标是图像显示坐标 (y = 行号, 向下增大), matplotlib 默认
    y 轴向上; 不反转的话图会与原图/云图叠加版垂直镜像, 误导
    "应变集中在样品哪一侧" 的判读。
    """
    ax.invert_yaxis()


def load_displacement_csv(filepath):
    """导入位移 CSV：编号, 实际x, 实际y, 参考x, 参考y, 位移dx, 位移dy, 位移幅值

    v3 主程序以物理坐标 (y 向上) 导出；本函数统一转换为图像显示坐标 (y 向下)，
    使坐标与底图、与本工具全部绘图保持一致。
    """
    # utf-8-sig 兼容带/不带 BOM 的 CSV (主程序导出带 BOM 以便 Excel 打开);
    # 显式指定编码也避免 GBK 区域设置下默认编码解码中文表头失败
    data = np.loadtxt(filepath, delimiter=',', skiprows=1, dtype=float, encoding='utf-8-sig')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 8:
        raise ValueError("位移 CSV 至少需要 8 列（编号, 实际x, 实际y, 参考x, 参考y, dx, dy, 幅值）。")
    if not np.isfinite(data).all():
        raise ValueError("位移 CSV 包含 NaN 或 Infinity。")
    n = data.shape[0]
    actual_pos = data[:, 1:3].copy()
    ideal_pos = data[:, 3:5].copy()
    displacements = data[:, 5:7].copy()
    metadata = read_export_metadata(filepath)
    physical = detect_physical_convention(actual_pos[:, 1], metadata)
    if physical:
        _flip_y_inplace([actual_pos, ideal_pos, displacements])
    return {
        'n': n,
        'actual_pos': actual_pos,      # 实际x, 实际y (显示坐标)
        'ideal_pos': ideal_pos,        # 参考x, 参考y (显示坐标)
        'displacements': displacements,  # dx, dy (显示坐标)
        'distortions': data[:, 7],     # 幅值 (与约定无关)
        'physical_source': physical,
        'convention_note': convention_detection_note(actual_pos[:, 1], metadata),
        'algorithm_id': metadata.get('algorithm_id') if metadata else None,
    }


def load_strain_csv(filepath):
    """
    导入应变 CSV：三角形编号, 质心x, 质心y, ε_xx, ε_yy, ε_xy, ε_eq, ω, [GL 分量], [边缘标记]
    兼容有/无 GL 和边缘标记的版本
    """
    # utf-8-sig 兼容带/不带 BOM 的 CSV (主程序导出带 BOM 以便 Excel 打开)
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)

    # 解析表头判断有哪些列
    n_header = len(header)
    n_base = 8  # 编号, 质心x, 质心y, ε_xx, ε_yy, ε_xy, ε_eq, ω
    has_gl = False
    has_edge = False
    remaining = n_header - n_base  # 去掉基础8列

    col_map = {
        'centroid_x': 1, 'centroid_y': 2,
        'exx': 3, 'eyy': 4, 'exy': 5, 'eeq': 6, 'rot': 7,
    }

    # 可选列一律按表头名定位: 主程序列序调整不会让本工具静默读错列
    if remaining >= 4 and 'ε_xx (GL)' in header:
        has_gl = True
        col_map['gl_xx'] = header.index('ε_xx (GL)')
        col_map['gl_yy'] = header.index('ε_yy (GL)')
        col_map['gl_xy'] = header.index('ε_xy (GL)')
        col_map['gl_eq'] = header.index('ε_eq (GL von Mises)')
    if remaining >= 1 and '边缘三角形' in header:
        has_edge = True
        col_map['edge'] = header.index('边缘三角形')

    # 兼容旧版本中文布尔列 ("是"/"否") 的 CSV: 先按字符串读取再统一转数值
    # (旧版 ppa.py 导出 "是/否" 时 np.loadtxt(dtype=float) 会直接崩溃)
    raw = np.genfromtxt(filepath, delimiter=',', skip_header=1, dtype=str, encoding='utf-8-sig')
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    raw = np.where(raw == '是', '1', np.where(raw == '否', '0', raw))
    if raw.shape[1] < 8:
        raise ValueError("应变 CSV 至少需要 8 列（编号, 质心x, 质心y, ε_xx, ε_yy, ε_xy, ε_eq, ω）。")

    def _numeric_column(index, label):
        try:
            return raw[:, index].astype(float)
        except (ValueError, IndexError) as error:
            raise ValueError(f"应变 CSV 的 {label} 列不是有效数值。") from error

    coordinates = np.column_stack((
        _numeric_column(col_map['centroid_x'], '位置x'),
        _numeric_column(col_map['centroid_y'], '位置y'),
    ))
    strain_columns = {
        'strain_xx': _numeric_column(col_map['exx'], 'ε_xx'),
        'strain_yy': _numeric_column(col_map['eyy'], 'ε_yy'),
        'strain_xy': _numeric_column(col_map['exy'], 'ε_xy'),
        'strain_eq': _numeric_column(col_map['eeq'], 'ε_eq'),
        'rotation': _numeric_column(col_map['rot'], '旋转'),
    }
    if not np.isfinite(coordinates).all():
        raise ValueError("应变 CSV 的位置列包含 NaN 或 Infinity。")

    valid_mask = None
    if '应变有效' in header:
        valid_raw = _numeric_column(header.index('应变有效'), '应变有效')
        if not np.isfinite(valid_raw).all() or not np.isin(valid_raw, [0.0, 1.0]).all():
            raise ValueError("应变有效列必须为 0 或 1。")
        valid_mask = valid_raw.astype(bool)

    all_strain_values = np.column_stack(tuple(strain_columns.values()))
    if np.isinf(all_strain_values).any():
        raise ValueError("应变 CSV 包含 Infinity。")
    if valid_mask is None:
        if not np.isfinite(all_strain_values).all():
            raise ValueError("应变 CSV 包含 NaN 或 Infinity。")
    else:
        # 新版逐原子导出允许几何不足行使用 NaN，但有效行必须全部有限。
        if not np.isfinite(all_strain_values[valid_mask]).all():
            raise ValueError("应变有效行包含 NaN 或 Infinity。")
        invalid_with_finite_flag = (~valid_mask) & np.isfinite(all_strain_values).all(axis=1)
        if invalid_with_finite_flag.any():
            raise ValueError("标记为无效的应变行应使用 NaN，不能伪造有限应变值。")

    result = {
        'n': raw.shape[0],
        'centroids': coordinates.copy(),
        **strain_columns,
    }
    # v3 以物理坐标 (y 向上) 导出位置列；统一转换为显示坐标以便与底图叠加。
    # 剪应变/旋转在两种约定下符号相反, 同步取反以保持自洽。
    metadata = read_export_metadata(filepath)
    physical = detect_physical_convention(result['centroids'][:, 1], metadata)
    result['physical_source'] = physical
    result['convention_note'] = convention_detection_note(result['centroids'][:, 1], metadata)
    result['algorithm_id'] = metadata.get('algorithm_id') if metadata else None
    if physical:
        result['centroids'][:, 1] *= -1.0
        result['strain_xy'] = -result['strain_xy']
        result['rotation'] = -result['rotation']
    if has_gl:
        gl_xx = _numeric_column(col_map['gl_xx'], 'ε_xx (GL)')
        gl_yy = _numeric_column(col_map['gl_yy'], 'ε_yy (GL)')
        gl_xy = _numeric_column(col_map['gl_xy'], 'ε_xy (GL)')
        gl_eq = _numeric_column(col_map['gl_eq'], 'ε_eq (GL)')
        gl_values = np.column_stack((gl_xx, gl_yy, gl_xy, gl_eq))
        if np.isinf(gl_values).any() or (valid_mask is None and not np.isfinite(gl_values).all()) or \
                (valid_mask is not None and not np.isfinite(gl_values[valid_mask]).all()):
            raise ValueError("应变 CSV 的 GL 应变列包含不允许的 NaN 或 Infinity。")
        result['gl_xx'] = gl_xx
        result['gl_yy'] = gl_yy
        result['gl_xy'] = -gl_xy if physical else gl_xy
        result['gl_eq'] = gl_eq
    if has_edge:
        edge = _numeric_column(col_map['edge'], '边缘三角形')
        if not np.isfinite(edge).all():
            raise ValueError("边缘三角形列包含 NaN 或 Infinity。")
        result['edge_mask'] = edge.astype(bool)
    if '参考三角形面积' in header:
        area = _numeric_column(header.index('参考三角形面积'), '参考三角形面积')
        if not np.isfinite(area).all():
            raise ValueError("参考三角形面积列包含 NaN 或 Infinity。")
        result['element_area'] = area
    if valid_mask is not None:
        result['valid_mask'] = valid_mask
    if '原始点编号' in header:
        result['source_indices'] = _numeric_column(header.index('原始点编号'), '原始点编号').astype(int) - 1
    if '晶格n' in header and '晶格m' in header:
        result['lattice_indices'] = np.column_stack((
            _numeric_column(header.index('晶格n'), '晶格n'),
            _numeric_column(header.index('晶格m'), '晶格m'),
        )).astype(int)
    if '计算质量' in header:
        result['quality_grades'] = raw[:, header.index('计算质量')].copy()
    if '无效原因' in header:
        result['invalid_reasons'] = raw[:, header.index('无效原因')].copy()
    return result


def weighted_mean_std(values, weights=None):
    """Return mean and standard deviation with optional positive area weights."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        valid &= np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return float("nan"), float("nan")
    if weights is None:
        return float(values[valid].mean()), float(values[valid].std())
    mean = float(np.average(values[valid], weights=weights[valid]))
    variance = float(np.average((values[valid] - mean) ** 2, weights=weights[valid]))
    return mean, float(np.sqrt(max(variance, 0.0)))


def load_atoms_csv(filepath):
    """导入原子坐标 CSV：编号, x, y (主程序以显示坐标导出, 仍做物理约定防御)"""
    data = np.loadtxt(filepath, delimiter=',', skiprows=1, dtype=float, encoding='utf-8-sig')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError("原子坐标 CSV 至少需要 3 列（编号, x, y）。")
    atoms = data[:, 1:3].copy()
    if detect_physical_convention(atoms[:, 1], read_export_metadata(filepath)):
        atoms[:, 1] *= -1.0
    return {'n': data.shape[0], 'atoms': atoms}


# ================================================================
#  分析函数
# ================================================================

def compute_principal_strains(exx, eyy, exy):
    """
    计算主应变 (ε₁, ε₂) 和方向角 θ
    ε₁, ε₂ = (ε_xx+ε_yy)/2 ± sqrt(((ε_xx-ε_yy)/2)² + ε_xy²)
    θ = 0.5 * arctan2(2*ε_xy, ε_xx-ε_yy)
    """
    mean_norm = (exx + eyy) / 2.0
    diff = (exx - eyy) / 2.0
    radius = np.sqrt(diff**2 + exy**2)
    e1 = mean_norm + radius
    e2 = mean_norm - radius
    theta = 0.5 * np.arctan2(2 * exy, exx - eyy)
    return e1, e2, theta


def compute_strain_gradient(exx, eyy, exy, centroids):
    """
    Estimate |∇ε| at each element by a local least-squares plane.

    The old implementation reported a neighbour difference without dividing by
    distance, which was not a gradient and changed when mesh density changed.
    Returned units are strain per coordinate unit (normally px⁻¹ unless a
    calibrated coordinate export is supplied).
    """
    centroids = np.asarray(centroids, dtype=float)
    tree = cKDTree(centroids)
    n = len(exx)
    # 秩亏邻域 (边界/共线) 记 NaN 而非 0: 0 是合法梯度值, 会污染色标与统计
    grad_xx = np.full(n, np.nan)
    grad_yy = np.full(n, np.nan)
    grad_xy = np.full(n, np.nan)
    k = min(6, n - 1)
    if k < 2:
        return grad_xx, grad_yy, grad_xy

    _, indices = tree.query(centroids, k=k+1)
    values_mat = np.column_stack((exx, eyy, exy))
    for i in range(n):
        neighbors = indices[i]
        offsets = centroids[neighbors] - centroids[i]
        design = np.column_stack((np.ones(len(neighbors)), offsets))
        if np.linalg.matrix_rank(design) < 3:
            continue
        # 三分量一次多列求解, 等价于原来的三次独立 lstsq
        coefficients, *_ = np.linalg.lstsq(design, values_mat[neighbors], rcond=None)
        slopes = np.hypot(coefficients[1], coefficients[2])
        grad_xx[i], grad_yy[i], grad_xy[i] = slopes

    return grad_xx, grad_yy, grad_xy


def compute_deformation_mode(e1, e2):
    """
    变形模式分类：
    - e1 > 0, e2 < 0: 纯剪切
    - e1 > 0, e2 > 0: 双轴拉伸
    - e1 < 0, e2 < 0: 双轴压缩
    - e1 > 0, |e2| << e1: 单轴拉伸
    """
    mode = np.zeros(len(e1), dtype=int)
    ratio = np.zeros(len(e1))

    for i in range(len(e1)):
        if abs(e1[i]) > 1e-10 or abs(e2[i]) > 1e-10:
            ratio[i] = e2[i] / max(abs(e1[i]), 1e-10)
        else:
            ratio[i] = 0

        if abs(e1[i]) < 1e-8 and abs(e2[i]) < 1e-8:
            mode[i] = 0  # 零应变
        elif e1[i] > 0 and e2[i] < 0:
            # Opposite signs alone are not pure shear: uniaxial tension with
            # Poisson contraction has e2 < 0.  Reserve "shear" for near-zero
            # trace / equal-and-opposite principal strains.
            opposite_ratio = abs(e2[i] / e1[i])
            if opposite_ratio >= 0.8:
                mode[i] = 1  # shear-dominated
            elif opposite_ratio <= 0.5:
                mode[i] = 2  # uniaxial tension with transverse contraction
            else:
                mode[i] = 5  # mixed tension/shear
        elif e1[i] > 0 and e2[i] >= 0:
            # e2 == 0 也属于单轴拉伸主导；此分支同时消除原不可达的
            # “elif e1 > 0 and |e2| < 0.3*e1” 冗余分支。
            if e2[i] / e1[i] > 0.8:
                mode[i] = 3  # 等双轴拉伸
            else:
                mode[i] = 2  # 单轴拉伸主导
        elif e1[i] < 0 and e2[i] < 0:
            mode[i] = 4  # 双轴压缩
        else:
            mode[i] = 5  # 混合

    return mode, ratio


# ================================================================
#  GUI 主程序
# ================================================================

class PPAStatsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PPA 统计分析工具 v1.0")
        self.root.geometry("1400x850")

        self.data = PPAData()
        self.output_dir = None

        self.setup_ui()
        self._bind_shortcuts()

    def setup_ui(self):
        # 主分割
        main_pw = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pw.pack(fill=tk.BOTH, expand=True)

        # 左侧：图像显示
        left_frame = ttk.Frame(main_pw)
        main_pw.add(left_frame, weight=3)

        self.fig = Figure(figsize=(9, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, left_frame)
        toolbar.update()

        # 右侧控制面板
        right_frame = ttk.Frame(main_pw, width=380)
        main_pw.add(right_frame, weight=1)

        self._build_file_panel(right_frame)
        self._build_stats_panel(right_frame)
        self._build_analysis_panel(right_frame)
        self._build_export_panel(right_frame)

        # 状态栏
        self.status = ttk.Label(right_frame, text="就绪 | 请导入 PPA 导出数据",
                                relief=tk.SUNKEN, anchor=tk.W, wraplength=350)
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_file_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="数据导入", padding=6)
        frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(frame, text="📄 导入位移 CSV", command=self._load_displacement).pack(fill=tk.X, pady=1)
        ttk.Button(frame, text="📄 导入应变 CSV", command=self._load_strain).pack(fill=tk.X, pady=1)
        ttk.Button(frame, text="📄 导入原子坐标 CSV", command=self._load_atoms).pack(fill=tk.X, pady=1)
        ttk.Button(frame, text="🖼 加载底图图像", command=self._load_image).pack(fill=tk.X, pady=1)

        self.data_status = ttk.Label(frame, text="未导入数据", foreground="gray", font=('', 8))
        self.data_status.pack(anchor=tk.W, fill=tk.X, pady=(2, 0))

    def _build_stats_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="数据概览", padding=6)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)

        self.stats_text = tk.Text(frame, height=12, font=('Consolas', 9), wrap=tk.WORD,
                                  state=tk.DISABLED, bg='#fafafa')
        self.stats_text.pack(fill=tk.BOTH, expand=True)

    def _build_analysis_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="分析生成", padding=6)
        frame.pack(fill=tk.X, padx=5, pady=2)

        # 位移分析
        ttk.Label(frame, text="位移分析", font=('', 9, 'bold')).pack(anchor=tk.W)
        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, pady=1)
        ttk.Button(row1, text="位移幅值直方图", command=self._plot_displacement_hist).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(row1, text="方向玫瑰图", command=self._plot_rose_diagram).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        # 应变分析
        ttk.Label(frame, text="应变分析", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(4, 0))
        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=1)
        ttk.Button(row2, text="分量直方图", command=self._plot_strain_histograms).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(row2, text="主应变分布", command=self._plot_principal_strain).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        row2b = ttk.Frame(frame)
        row2b.pack(fill=tk.X, pady=1)
        ttk.Button(row2b, text="变形模式分析", command=self._plot_deformation_mode).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(row2b, text="应变梯度", command=self._plot_strain_gradient).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        # 相关性
        ttk.Label(frame, text="相关性分析", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(4, 0))
        row3 = ttk.Frame(frame)
        row3.pack(fill=tk.X, pady=1)
        ttk.Button(row3, text="ε_xx vs ε_yy", command=self._plot_strain_correlation).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(row3, text="相关矩阵", command=self._plot_correlation_matrix).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        # 空间分布
        ttk.Label(frame, text="空间分布", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(4, 0))
        row4 = ttk.Frame(frame)
        row4.pack(fill=tk.X, pady=1)
        ttk.Button(row4, text="应变云图叠加", command=self._show_strain_overlay).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        # 参数区
        params_frame = ttk.Frame(frame)
        params_frame.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(params_frame, text="网格:", font=('', 8)).pack(side=tk.LEFT)
        self.grid_size_var = tk.StringVar(value="200")
        ttk.Spinbox(params_frame, from_=50, to=500, textvariable=self.grid_size_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(params_frame, text=" 图片 DPI:", font=('', 8)).pack(side=tk.LEFT, padx=(6, 0))
        self.dpi_var = tk.StringVar(value="300")
        ttk.Spinbox(params_frame, from_=100, to=600, textvariable=self.dpi_var, width=5).pack(side=tk.LEFT, padx=2)

    def _build_export_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="导出", padding=6)
        frame.pack(fill=tk.X, padx=5, pady=2)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=1)
        ttk.Button(row, text="📊 生成全部图表", command=self._generate_all_plots).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(row, text="📝 导出报告", command=self._export_report).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(frame, text="🗑 清空数据", command=self._reset_data).pack(fill=tk.X, pady=(2, 0))

    def _bind_shortcuts(self):
        self.root.bind("<Escape>", lambda e: self._clear_display())

    def _clear_display(self):
        self.ax.clear()
        self.ax.set_title("PPA 统计分析")
        self.canvas.draw_idle()

    def _update_status(self):
        """刷新数据概览和状态栏"""
        parts = []
        if self.data.n_atoms > 0:
            parts.append(f"{self.data.n_atoms} 个原子")
        if self.data.n_tri > 0:
            parts.append(f"{self.data.n_tri} 个三角形")
        if self.data.has_image():
            h, w = self.data.image.shape
            parts.append(f"图像 {w}×{h}")

        if parts:
            summary = " | ".join(parts)
            self.data_status.config(text=f"✓ 已加载: {summary}", foreground="#006600")
        else:
            self.data_status.config(text="未导入数据", foreground="gray")

        # 更新统计文本框
        lines = []
        if self.data.has_displacement():
            d = self.data.distortions
            dx = self.data.displacements[:, 0]
            dy = self.data.displacements[:, 1]
            lines.append("─" * 32)
            lines.append("位移场统计:")
            lines.append(f"  位移幅值: mean={d.mean():.4f}  std={d.std():.4f}")
            lines.append(f"            min={d.min():.5f}  max={d.max():.4f}")
            lines.append(f"  dx:       mean={dx.mean():.4f}  std={dx.std():.4f}")
            lines.append(f"  dy:       mean={dy.mean():.4f}  std={dy.std():.4f}")

        if self.data.has_strain():
            lines.append("─" * 32)
            lines.append("应变张量统计:")
            strain_rows = [("ε_xx", self.data.strain_xx), ("ε_yy", self.data.strain_yy),
                           ("ε_xy", self.data.strain_xy), ("ε_eq", self.data.strain_eq),
                           ("ω   ", self.data.rotation)]
            strain_rows += [(f"ε_{k[-2:]} (GL)", v) for k, v in sorted(self.data.strain_gl.items())
                            if v is not None]
            for name, arr in strain_rows:
                if arr is not None:
                    mean, std = weighted_mean_std(arr, self.data.element_area)
                    weighting = "面积加权" if self.data.element_area is not None else "元素等权"
                    lines.append(f"  {name}: mean={mean:.6f}  std={std:.6f} ({weighting})")
                    lines.append(f"         min={arr.min():.6f}  max={arr.max():.6f}")

            e1, e2, theta = compute_principal_strains(
                self.data.strain_xx, self.data.strain_yy, self.data.strain_xy)
            # 主应变与相邻行保持同一加权口径, 避免面板内数字不可比
            e1_mean, e1_std = weighted_mean_std(e1, self.data.element_area)
            e2_mean, e2_std = weighted_mean_std(e2, self.data.element_area)
            lines.append(f"  ε₁:   mean={e1_mean:.6f}  std={e1_std:.6f}")
            lines.append(f"  ε₂:   mean={e2_mean:.6f}  std={e2_std:.6f}")

        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, "\n".join(lines) if lines else "请导入数据")
        self.stats_text.config(state=tk.DISABLED)

    # ================================================================
    #  文件导入
    # ================================================================

    def _load_displacement(self):
        path = filedialog.askopenfilename(title="导入位移 CSV",
                                          filetypes=[("CSV 文件", "*.csv")])
        if not path:
            return
        try:
            result = load_displacement_csv(path)
            self.data.actual_pos = result['actual_pos']
            self.data.ideal_pos = result['ideal_pos']
            self.data.displacements = result['displacements']
            self.data.distortions = result['distortions']
            self.data.n_atoms = result['n']
            self.data.physical_source = result.get('physical_source', False)

            if self.data.actual_pos is not None:
                self.data.atoms = self.data.actual_pos.copy()

            self.status.config(text=f"✓ 位移数据已导入: {result['n']} 个原子  |  {os.path.basename(path)}",
                               foreground='#006600')
            self._update_status()
            algo_note = self.data.check_algorithm_consistency(result.get('algorithm_id'))
            if algo_note:
                messagebox.showwarning("算法来源不一致", algo_note)
            elif result.get('algorithm_id'):
                self.data.algorithm_id = result['algorithm_id']
            if result.get('convention_note'):
                messagebox.showwarning("坐标约定判定", result['convention_note'])
        except Exception as e:
            messagebox.showerror("导入失败", f"无法解析 CSV 文件:\n{e}")

    def _load_strain(self):
        path = filedialog.askopenfilename(title="导入应变 CSV",
                                          filetypes=[("CSV 文件", "*.csv")])
        if not path:
            return
        try:
            result = load_strain_csv(path)
            original_count = result['n']
            excluded_edge_count = 0
            excluded_invalid_count = 0
            keep = np.ones(original_count, dtype=bool)
            if 'valid_mask' in result:
                keep &= result['valid_mask']
                excluded_invalid_count = int((~result['valid_mask']).sum())
            if 'edge_mask' in result:
                keep &= ~result['edge_mask']
                excluded_edge_count = int(result['edge_mask'].sum())
            if not keep.any():
                raise ValueError("没有具有有限应变且通过质量筛选的元素可统计。")
            if not keep.all():
                for key, value in list(result.items()):
                    if isinstance(value, np.ndarray) and len(value) == original_count:
                        result[key] = value[keep]
                result['n'] = int(keep.sum())
            self.data.tri_centroids = result['centroids']
            self.data.strain_xx = result['strain_xx']
            self.data.strain_yy = result['strain_yy']
            self.data.strain_xy = result['strain_xy']
            self.data.strain_eq = result['strain_eq']
            self.data.rotation = result['rotation']
            self.data.n_tri = result['n']
            self.data.physical_source = result.get('physical_source', False)

            for k in ['gl_xx', 'gl_yy', 'gl_xy', 'gl_eq']:
                if k in result:
                    self.data.strain_gl[k] = result[k]
            if 'edge_mask' in result:
                self.data.tri_edge = result['edge_mask']
            self.data.element_area = result.get('element_area')

            quality_parts = []
            if excluded_invalid_count:
                quality_parts.append(f"{excluded_invalid_count} 个几何不足原子仅保留在原 CSV")
            if excluded_edge_count:
                quality_parts.append(f"排除 {excluded_edge_count} 个边缘元素")
            quality_note = f"（{'；'.join(quality_parts)}）" if quality_parts else ""
            self.status.config(text=f"✓ 应变数据已导入: {result['n']} 个有效元素{quality_note}  |  {os.path.basename(path)}",
                               foreground='#006600')
            self._update_status()
            algo_note = self.data.check_algorithm_consistency(result.get('algorithm_id'))
            if algo_note:
                messagebox.showwarning("算法来源不一致", algo_note)
            elif result.get('algorithm_id'):
                self.data.algorithm_id = result['algorithm_id']
            if result.get('convention_note'):
                messagebox.showwarning("坐标约定判定", result['convention_note'])
        except Exception as e:
            messagebox.showerror("导入失败", f"无法解析 CSV 文件:\n{e}")

    def _load_atoms(self):
        path = filedialog.askopenfilename(title="导入原子坐标 CSV",
                                          filetypes=[("CSV 文件", "*.csv")])
        if not path:
            return
        try:
            result = load_atoms_csv(path)
            self.data.atoms = result['atoms']
            self.data.n_atoms = result['n']
            self.status.config(text=f"✓ 原子坐标已导入: {result['n']} 个原子  |  {os.path.basename(path)}",
                               foreground='#006600')
            self._update_status()
        except Exception as e:
            messagebox.showerror("导入失败", f"无法解析 CSV 文件:\n{e}")

    def _load_image(self):
        path = filedialog.askopenfilename(title="选择底图图像",
                                          filetypes=[("TIFF/PNG", "*.tif *.tiff *.png"),
                                                     ("所有文件", "*.*")])
        if not path:
            return
        try:
            img = plt.imread(path)
            if img.ndim == 3:
                img = img[..., :3].mean(axis=2)
            # 归一化
            lo, hi = np.percentile(img, [1, 99])
            if hi > lo:
                img = np.clip((img - lo) / (hi - lo), 0, 1)
            self.data.image = img.astype(np.float64)
            self.data.image_path = path
            self.status.config(text=f"✓ 底图已加载: {os.path.basename(path)} ({img.shape[1]}×{img.shape[0]})",
                               foreground='#006600')
            self._show_image()
            self._update_status()
        except Exception as e:
            messagebox.showerror("导入失败", f"无法加载图像:\n{e}")

    def _show_image(self):
        """在左侧画布显示底图"""
        if not self.data.has_image():
            return
        self.ax.clear()
        self.ax.imshow(self.data.image, cmap='gray', origin='upper', aspect='equal')
        self.ax.set_title("底图 | ↗ 点击分析按钮生成图表")
        if self.data.has_atoms():
            self.ax.scatter(self.data.atoms[:, 0], self.data.atoms[:, 1],
                            s=5, c='lime', marker='o', alpha=0.5, zorder=2)
        self.canvas.draw_idle()

    def _reset_data(self):
        reply = messagebox.askyesno("确认", "清除所有已导入的数据？")
        if not reply:
            return
        self.data.reset()
        self.ax.clear()
        self.ax.set_title("PPA 统计分析")
        self.canvas.draw_idle()
        self._update_status()
        self.status.config(text="数据已清空")

    # ================================================================
    #  图表生成 — 在当前画布显示
    # ================================================================

    def _show_message(self, msg):
        self.ax.clear()
        self.ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=14,
                     transform=self.ax.transAxes)
        self.ax.set_title(msg)
        self.canvas.draw_idle()

    def _plot_displacement_hist(self):
        if not self.data.has_displacement():
            messagebox.showinfo("提示", "请先导入位移 CSV")
            return
        self.ax.clear()
        d = self.data.distortions
        self.ax.hist(d, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
        # 叠加高斯拟合
        mu, sigma = d.mean(), d.std()
        if sigma > np.finfo(float).eps:
            x = np.linspace(d.min(), d.max(), 200)
            gauss = np.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
            self.ax.plot(x, gauss, 'r-', lw=2, label=f'Gaussian fit\nμ={mu:.4f}, σ={sigma:.4f}')
        else:
            self.ax.text(0.02, 0.95, '常量数据：不拟合高斯分布', transform=self.ax.transAxes,
                         va='top', fontsize=8)
        self.ax.axvline(mu, color='red', ls='--', alpha=0.6)
        self.ax.axvline(np.median(d), color='green', ls='--', alpha=0.6, label=f'median={np.median(d):.4f}')
        self.ax.set_xlabel('Displacement (px)')
        self.ax.set_ylabel('Probability density')
        self.ax.set_title('Atomic Displacement Distribution')
        self.ax.legend(fontsize=8)
        self.canvas.draw_idle()

    def _plot_rose_diagram(self):
        if not self.data.has_displacement():
            messagebox.showinfo("提示", "请先导入位移 CSV")
            return
        dx, dy = self.data.displacements[:, 0], self.data.displacements[:, 1]
        angles = np.arctan2(dy, dx)  # 弧度
        mags = self.data.distortions

        # 只使用非零位移
        nonzero = mags > 1e-6
        angles = angles[nonzero]
        mags = mags[nonzero]

        if len(angles) < 5:
            self._show_message("非零位移太少，无法画玫瑰图")
            return

        self.fig.clear()
        ax_polar = self.fig.add_subplot(111, projection='polar')
        self.ax = ax_polar
        n_bins = 36
        bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        theta = (bins[:-1] + bins[1:]) / 2

        # 加权角度直方图
        counts, _ = np.histogram(angles, bins=bins, weights=mags)
        width = 2 * np.pi / n_bins

        bars = ax_polar.bar(theta, counts, width=width, bottom=0,
                            color='steelblue', alpha=0.7, edgecolor='white')
        # dy 是显示坐标 (向下为正), 极坐标默认逆时针; 顺时针布局使 90° 指向
        # 屏幕下方, 方向读数与图像/样品几何一致
        ax_polar.set_theta_direction(-1)
        ax_polar.set_title('Displacement direction (0°→right, 90°→down; weighted by magnitude)',
                           fontsize=10, pad=15)
        ax_polar.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'], fontsize=7)

        # 主方向统计
        from scipy.stats import circmean
        mean_angle = circmean(angles, high=np.pi, low=-np.pi)
        mean_deg = np.degrees(mean_angle) % 360
        ax_polar.plot([mean_angle, mean_angle], [0, counts.max() * 0.8],
                      'r-', lw=2, label=f'Mean dir: {mean_deg:.0f}°')

        # 各向异性指数 (northrup ratio)
        r_sum = counts.sum()
        if r_sum > 0:
            r_vec = (counts * np.exp(1j * theta)).sum() / r_sum
            anisotropy = abs(r_vec)  # 0 = isotropic, 1 = fully aligned
            ax_polar.legend(fontsize=7, loc='upper right')
            ax_polar.set_title(f'Displacement direction (anisotropy={anisotropy:.3f})', fontsize=10, pad=15)

        self.canvas.draw_idle()

    def _plot_strain_histograms(self):
        if not self.data.has_strain():
            messagebox.showinfo("提示", "请先导入应变 CSV")
            return
        self.fig.clear()
        fields = [
            ('ε_xx', self.data.strain_xx, 'RdBu_r'),
            ('ε_yy', self.data.strain_yy, 'RdBu_r'),
            ('ε_xy', self.data.strain_xy, 'RdBu_r'),
            ('ε_eq', self.data.strain_eq, 'viridis'),
            ('ω', self.data.rotation, 'RdBu_r'),
        ]
        gl_names = {'gl_xx': 'ε_xx (GL)', 'gl_yy': 'ε_yy (GL)',
                    'gl_xy': 'ε_xy (GL)', 'gl_eq': 'ε_eq (GL)'}
        gl_fields = [(gl_names[k], v) for k, v in sorted(self.data.strain_gl.items())
                     if v is not None]
        if gl_fields:
            # GL 分量与同一批直方图一并展示, 不再是导入后无处可看的死数据
            fields += gl_fields
            axes = self.fig.subplots(3, 3)
        else:
            axes = self.fig.subplots(2, 3)
        axes = axes.flatten()
        self.ax = axes[0]
        stats_slot = len(fields)
        for slot, field in enumerate(fields):
            name, arr = field[0], field[1]
            ax_i = axes[slot]
            if arr is None:
                ax_i.text(0.5, 0.5, 'No data', ha='center', va='center')
                continue
            ax_i.hist(arr, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
            mu, sigma = arr.mean(), arr.std()
            if sigma > np.finfo(float).eps:
                x = np.linspace(arr.min(), arr.max(), 200)
                g = np.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
                ax_i.plot(x, g, 'r-', lw=1.5)
            ax_i.axvline(mu, color='r', ls='--', alpha=0.5)
            ax_i.set_xlabel(name)
            ax_i.set_ylabel('Density')
            ax_i.set_title(f'{name}: μ={mu:.5f}, σ={sigma:.5f}', fontsize=9)

        # 剩余子图: 统计参数汇总
        for slot in range(stats_slot, len(axes)):
            axes[slot].axis('off')
        stats_lines = []
        for name, arr, *_ in fields:
            if arr is None:
                continue
            s = describe(arr)
            stats_lines.append(
                f"{name}:  μ={arr.mean():.6f}  σ={arr.std():.6f}\n"
                f"  min={arr.min():.6f}  max={arr.max():.6f}\n"
                f"  skew={skew(arr):.3f}  kurt={kurtosis(arr):.3f}"
            )
        if stats_slot < len(axes):
            stats_ax = axes[stats_slot]
            stats_ax.text(0.05, 0.95, '\n\n'.join(stats_lines),
                          transform=stats_ax.transAxes, va='top', fontsize=8,
                          fontfamily='monospace')
            stats_ax.set_title('Statistics', fontsize=10)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _plot_principal_strain(self):
        if not self.data.has_strain():
            messagebox.showinfo("提示", "请先导入应变 CSV")
            return
        e1, e2, theta = compute_principal_strains(
            self.data.strain_xx, self.data.strain_yy, self.data.strain_xy)

        self.fig.clear()
        axes = self.fig.subplots(1, 3)
        self.ax = axes[0]

        sc1 = axes[0].scatter(self.data.tri_centroids[:, 0], self.data.tri_centroids[:, 1],
                              c=e1, cmap='RdBu_r', s=20, alpha=0.8,
                              vmin=-max(abs(e1)), vmax=max(abs(e1)))
        axes[0].set_title('ε₁ (max principal strain)')
        axes[0].set_aspect('equal')
        plt.colorbar(sc1, ax=axes[0], fraction=0.046)

        sc2 = axes[1].scatter(self.data.tri_centroids[:, 0], self.data.tri_centroids[:, 1],
                              c=e2, cmap='RdBu_r', s=20, alpha=0.8,
                              vmin=-max(abs(e2)), vmax=max(abs(e2)))
        axes[1].set_title('ε₂ (min principal strain)')
        axes[1].set_aspect('equal')
        plt.colorbar(sc2, ax=axes[1], fraction=0.046)

        # 方向场 (用短线段表示主应变方向，长度正比于幅值)
        centroids = self.data.tri_centroids
        step = max(1, len(centroids) // 60)
        for i in range(0, len(centroids), step):
            cx, cy = centroids[i]
            ang = theta[i]
            mag = abs(e1[i] - e2[i]) * 30  # 缩放可见
            dx = mag * np.cos(ang)
            dy = mag * np.sin(ang)
            axes[2].arrow(cx - dx / 2, cy - dy / 2, dx, dy,
                          head_width=0, head_length=0, alpha=0.6,
                          color='red' if e1[i] > 0 else 'blue', lw=0.5)
        axes[2].set_title('Principal direction (red=tension, blue=compression)')
        axes[2].set_aspect('equal')
        # 三张空间图统一 y 轴向下, 与原图及云图叠加版方向一致
        for ax_i in axes:
            orient_spatial_axes(ax_i)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _plot_deformation_mode(self):
        if not self.data.has_strain():
            messagebox.showinfo("提示", "请先导入应变 CSV")
            return
        e1, e2, _ = compute_principal_strains(
            self.data.strain_xx, self.data.strain_yy, self.data.strain_xy)
        modes, ratio = compute_deformation_mode(e1, e2)

        self.fig.clear()
        axes = self.fig.subplots(1, 2)
        self.ax = axes[0]

        # 左图：变形模式空间分布
        mode_colors = ['gray', 'red', 'orange', 'green', 'blue', 'purple']
        mode_labels = ['Zero', 'Shear', 'Uni. tension', 'Biaxial', 'Compression', 'Mixed']
        mode_cmap = plt.matplotlib.colors.ListedColormap(mode_colors)
        sc = axes[0].scatter(self.data.tri_centroids[:, 0], self.data.tri_centroids[:, 1],
                             c=modes, cmap=mode_cmap, s=20, alpha=0.8, vmin=0, vmax=5)
        cbar = plt.colorbar(sc, ax=axes[0], ticks=range(6), fraction=0.046)
        cbar.ax.set_yticklabels(mode_labels, fontsize=7)
        axes[0].set_title('Deformation mode')
        axes[0].set_aspect('equal')
        orient_spatial_axes(axes[0])

        # 右图：ε₁ vs ε₂ 散点图
        axes[1].scatter(e1, e2, c=self.data.strain_eq, cmap='turbo', s=10, alpha=0.6)
        axes[1].axhline(y=0, color='gray', lw=0.5)
        axes[1].axvline(x=0, color='gray', lw=0.5)
        axes[1].set_xlabel('ε₁')
        axes[1].set_ylabel('ε₂')
        axes[1].set_title('ε₁ vs ε₂ (color = ε_eq)')
        axes[1].axis('equal')

        # 统计每种模式占比
        total = len(modes)
        counts_text = "\n".join(
            [f"{mode_labels[m]}: {sum(modes==m)} ({sum(modes==m)/total*100:.1f}%)"
             for m in range(6) if sum(modes==m) > 0])
        axes[1].text(0.05, 0.95, counts_text, transform=axes[1].transAxes,
                     va='top', fontsize=8, fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.3))

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _plot_strain_gradient(self):
        if not self.data.has_strain():
            messagebox.showinfo("提示", "请先导入应变 CSV")
            return
        gxx, gyy, gxy = compute_strain_gradient(
            self.data.strain_xx, self.data.strain_yy, self.data.strain_xy,
            self.data.tri_centroids)

        self.fig.clear()
        axes = self.fig.subplots(1, 3)
        self.ax = axes[0]

        for axis, (data, name) in zip(axes, zip(
            [gxx, gyy, gxy], ['|∇ε_xx|', '|∇ε_yy|', '|∇ε_xy|'])):
            vmin, vmax = np.nanpercentile(data, [5, 95])  # 边界秩亏点是 NaN
            sc = axis.scatter(self.data.tri_centroids[:, 0], self.data.tri_centroids[:, 1],
                              c=data, cmap='inferno', s=20, alpha=0.8,
                              vmin=vmin, vmax=vmax)
            axis.set_title(name)
            axis.set_aspect('equal')
            orient_spatial_axes(axis)
            self.fig.colorbar(sc, ax=axis, fraction=0.046)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _plot_strain_correlation(self):
        if not self.data.has_strain():
            messagebox.showinfo("提示", "请先导入应变 CSV")
            return
        exx, eyy = self.data.strain_xx, self.data.strain_yy
        exy = self.data.strain_xy

        self.fig.clear()
        axes = self.fig.subplots(1, 2)
        self.ax = axes[0]

        # ε_xx vs ε_yy
        hb = axes[0].hexbin(exx, eyy, gridsize=40, cmap='inferno', mincnt=1)
        plt.colorbar(hb, ax=axes[0], label='Count')
        axes[0].axhline(y=0, color='gray', lw=0.5)
        axes[0].axvline(x=0, color='gray', lw=0.5)
        if len(exx) > 3:
            slope, intercept, r_val, p_val, _ = linregress(exx, eyy)
            x_line = np.linspace(exx.min(), exx.max(), 100)
            axes[0].plot(x_line, slope * x_line + intercept, 'r-', lw=1.5,
                         label=f'r={r_val:.3f}, p={p_val:.2e}')
            axes[0].legend(fontsize=8)
        axes[0].set_xlabel('ε_xx')
        axes[0].set_ylabel('ε_yy')
        axes[0].set_title('ε_xx vs ε_yy (hexbin density)')

        # ε_eq vs ε_xy
        hb2 = axes[1].hexbin(self.data.strain_eq, exy, gridsize=40, cmap='inferno', mincnt=1)
        plt.colorbar(hb2, ax=axes[1], label='Count')
        axes[1].axhline(y=0, color='gray', lw=0.5)
        axes[1].set_xlabel('ε_eq')
        axes[1].set_ylabel('ε_xy')
        axes[1].set_title('ε_eq vs ε_xy')

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _plot_correlation_matrix(self):
        if not self.data.has_strain():
            messagebox.showinfo("提示", "请先导入应变 CSV")
            return

        names = ['ε_xx', 'ε_yy', 'ε_xy', 'ε_eq', 'ω']
        arrays = [self.data.strain_xx, self.data.strain_yy,
                  self.data.strain_xy, self.data.strain_eq, self.data.rotation]
        valid = [(n, a) for n, a in zip(names, arrays) if a is not None]
        names = [v[0] for v in valid]
        data_mat = np.column_stack([v[1] for v in valid])
        corr = np.corrcoef(data_mat.T)

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self.ax = ax

        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=10)
        ax.set_yticklabels(names, fontsize=10)

        # 数值标注
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f'{corr[i, j]:.3f}',
                        ha='center', va='center', fontsize=8,
                        color='black' if abs(corr[i, j]) < 0.6 else 'white')

        ax.set_title('Strain Component Correlation Matrix', fontsize=12)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _show_strain_overlay(self):
        if not self.data.has_strain():
            messagebox.showinfo("提示", "请先导入应变 CSV")
            return
        grid_size = int(self.grid_size_var.get())

        self.ax.clear()
        if self.data.has_image():
            self.ax.imshow(self.data.image, cmap='gray', origin='upper', aspect='equal')

        centroids = self.data.tri_centroids
        if self.data.has_image():
            h_img = self.data.image.shape[0]
            w_img = self.data.image.shape[1]
            x_lo, x_hi = 0.0, float(w_img)
            y_lo, y_hi = 0.0, float(h_img)
        else:
            # 无底图时按质心实际范围建网格 (ROI 不从原点开始时避免全 NaN 空图)
            h_img = centroids[:, 1].max() + 10
            w_img = centroids[:, 0].max() + 10
            pad = 5.0
            x_lo = max(0.0, float(centroids[:, 0].min()) - pad)
            x_hi = float(centroids[:, 0].max()) + pad
            y_lo = max(0.0, float(centroids[:, 1].min()) - pad)
            y_hi = float(centroids[:, 1].max()) + pad

        # 创建插值网格
        gx = np.linspace(x_lo, x_hi, grid_size)
        gy = np.linspace(y_lo, y_hi, grid_size)
        GX, GY = np.meshgrid(gx, gy)
        grid_pts = np.column_stack((GX.ravel(), GY.ravel()))

        # 选择插值分量（默认 ε_eq）
        interp = griddata(centroids, self.data.strain_eq, grid_pts,
                          method='linear', fill_value=np.nan)
        interp = interp.reshape(grid_size, grid_size)

        vmin, vmax = np.nanpercentile(interp, [2, 98])
        if vmin == vmax or np.isnan(vmin):
            vmin, vmax = np.nanmin(interp), np.nanmax(interp)
        im = self.ax.imshow(interp, extent=(x_lo, x_hi, y_hi, y_lo),
                            origin='upper', cmap='turbo', alpha=0.6,
                            vmin=vmin, vmax=vmax, interpolation='bilinear')
        cbar = self.fig.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)
        cbar.set_label('ε_eq')
        self.ax.set_title('ε_eq overlay on image')
        self.canvas.draw_idle()

    # ================================================================
    #  批量生成 & 导出
    # ================================================================

    def _ensure_output_dir(self):
        if self.output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(os.getcwd(), f"PPA_Stats_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "图"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "数据"), exist_ok=True)
        return self.output_dir

    def _generate_all_plots(self):
        """生成所有图表并保存到输出目录"""
        if not self.data.has_strain() and not self.data.has_displacement():
            messagebox.showinfo("提示", "请先导入至少一个 CSV 数据文件")
            return

        out_dir = self._ensure_output_dir()
        dpi = int(self.dpi_var.get())
        fig_dir = os.path.join(out_dir, "图")
        data_dir = os.path.join(out_dir, "数据")
        grid_size = int(self.grid_size_var.get())

        self.status.config(text="生成图表中...")
        self.root.update_idletasks()

        generated = []

        # --- 位移图 ---
        if self.data.has_displacement():
            d = self.data.distortions
            # 1. 位移幅值直方图
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(d, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
            mu, sigma = d.mean(), d.std()
            if sigma > np.finfo(float).eps:  # 常数数据 (如零位移) 无高斯曲线, 防除零
                x = np.linspace(d.min(), d.max(), 200)
                g = np.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
                ax.plot(x, g, 'r-', lw=2, label=f'μ={mu:.4f}, σ={sigma:.4f}')
            else:
                ax.axvline(mu, color='r', ls='--', alpha=0.6, label=f'常数: {mu:.4f}')
            ax.axvline(mu, color='r', ls='--', alpha=0.6)
            ax.set_xlabel('Displacement (px)')
            ax.set_ylabel('Probability density')
            ax.set_title('Atomic Displacement Distribution')
            ax.legend(fontsize=8)
            plt.tight_layout()
            fig.savefig(os.path.join(fig_dir, "1_位移幅值分布.png"), dpi=dpi)
            plt.close(fig)
            generated.append("1_位移幅值分布.png")

            # 2. 玫瑰图
            dx, dy = self.data.displacements[:, 0], self.data.displacements[:, 1]
            angles = np.arctan2(dy, dx)
            mags = self.data.distortions
            nonzero = mags > 1e-6
            angles, mags = angles[nonzero], mags[nonzero]
            if len(angles) >= 5:
                fig = plt.figure(figsize=(6, 6))
                ax = fig.add_subplot(111, projection='polar')
                n_bins = 36
                bins = np.linspace(-np.pi, np.pi, n_bins + 1)
                theta_c = (bins[:-1] + bins[1:]) / 2
                counts, _ = np.histogram(angles, bins=bins, weights=mags)
                width = 2 * np.pi / n_bins
                ax.bar(theta_c, counts, width=width, bottom=0,
                       color='steelblue', alpha=0.7, edgecolor='white')
                ax.set_theta_direction(-1)
                r_sum = counts.sum()
                if r_sum > 0:
                    r_vec = (counts * np.exp(1j * theta_c)).sum() / r_sum
                    anisotropy = abs(r_vec)
                    ax.set_title(f'Displacement direction, 90°=down (anisotropy={anisotropy:.3f})',
                                 fontsize=10, pad=15)
                plt.tight_layout()
                fig.savefig(os.path.join(fig_dir, "2_位移方向玫瑰图.png"), dpi=dpi)
                plt.close(fig)
                generated.append("2_位移方向玫瑰图.png")

        # --- 应变图 ---
        if self.data.has_strain():
            centroids = self.data.tri_centroids
            # 3. 应变分量直方图 (有 GL 分量时扩展到 3×3)
            all_strain = {'ε_xx': self.data.strain_xx, 'ε_yy': self.data.strain_yy,
                          'ε_xy': self.data.strain_xy, 'ε_eq': self.data.strain_eq,
                          'ω': self.data.rotation}
            gl_names = {'gl_xx': 'ε_xx (GL)', 'gl_yy': 'ε_yy (GL)',
                        'gl_xy': 'ε_xy (GL)', 'gl_eq': 'ε_eq (GL)'}
            all_strain.update({gl_names[k]: v for k, v in sorted(self.data.strain_gl.items())
                               if v is not None})
            fig, axes = plt.subplots(3, 3 if len(all_strain) > 6 else 2, figsize=(12, 10 if len(all_strain) > 6 else 7))
            axes = axes.flatten()
            for i, (name, arr) in enumerate(all_strain.items()):
                if i >= len(axes):
                    break
                if arr is None:
                    axes[i].text(0.5, 0.5, 'No data', ha='center', va='center')
                    continue
                axes[i].hist(arr, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
                mu, sigma = arr.mean(), arr.std()
                if sigma > np.finfo(float).eps:  # 常数数据 (如理想晶格) 无高斯曲线
                    x = np.linspace(arr.min(), arr.max(), 200)
                    g = np.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
                    axes[i].plot(x, g, 'r-', lw=1.5)
                axes[i].axvline(mu, color='r', ls='--', alpha=0.5)
                axes[i].set_xlabel(name)
                axes[i].set_ylabel('Density')
                axes[i].set_title(f'{name}: μ={mu:.5f}, σ={sigma:.5f}')
            for i in range(len(all_strain), len(axes)):
                axes[i].axis('off')
            plt.tight_layout()
            fig.savefig(os.path.join(fig_dir, "3_应变分量直方图.png"), dpi=dpi)
            plt.close(fig)
            generated.append("3_应变分量直方图.png")

            # 4. 相关矩阵
            names = ['ε_xx', 'ε_yy', 'ε_xy', 'ε_eq', 'ω']
            arrays = [self.data.strain_xx, self.data.strain_yy,
                      self.data.strain_xy, self.data.strain_eq, self.data.rotation]
            valid = [(n, a) for n, a in zip(names, arrays) if a is not None]
            names_v = [v[0] for v in valid]
            if len(valid) >= 2:
                mat = np.column_stack([v[1] for v in valid])
                corr = np.corrcoef(mat.T)
                fig, ax = plt.subplots(figsize=(7, 6))
                im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_xticks(range(len(names_v)))
                ax.set_yticks(range(len(names_v)))
                ax.set_xticklabels(names_v, fontsize=10)
                ax.set_yticklabels(names_v, fontsize=10)
                for i in range(len(names_v)):
                    for j in range(len(names_v)):
                        ax.text(j, i, f'{corr[i, j]:.3f}',
                                ha='center', va='center', fontsize=8,
                                color='black' if abs(corr[i, j]) < 0.6 else 'white')
                ax.set_title('Strain Component Correlation Matrix')
                plt.tight_layout()
                fig.savefig(os.path.join(fig_dir, "4_应变分量相关矩阵.png"), dpi=dpi)
                plt.close(fig)
                generated.append("4_应变分量相关矩阵.png")

            # 5. ε_xx vs ε_yy
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            hb = axes[0].hexbin(self.data.strain_xx, self.data.strain_yy,
                                gridsize=40, cmap='inferno', mincnt=1)
            plt.colorbar(hb, ax=axes[0], label='Count')
            axes[0].axhline(y=0, color='gray', lw=0.5)
            axes[0].axvline(x=0, color='gray', lw=0.5)
            if len(self.data.strain_xx) > 3:
                slope, intercept, r_val, p_val, _ = linregress(
                    self.data.strain_xx, self.data.strain_yy)
                x_line = np.linspace(self.data.strain_xx.min(), self.data.strain_xx.max(), 100)
                axes[0].plot(x_line, slope * x_line + intercept, 'r-', lw=1.5,
                             label=f'r={r_val:.3f}, p={p_val:.2e}')
                axes[0].legend(fontsize=8)
            axes[0].set_xlabel('ε_xx')
            axes[0].set_ylabel('ε_yy')
            axes[0].set_title('ε_xx vs ε_yy')

            hb2 = axes[1].hexbin(self.data.strain_eq, self.data.strain_xy,
                                 gridsize=40, cmap='inferno', mincnt=1)
            plt.colorbar(hb2, ax=axes[1], label='Count')
            axes[1].axhline(y=0, color='gray', lw=0.5)
            axes[1].set_xlabel('ε_eq')
            axes[1].set_ylabel('ε_xy')
            axes[1].set_title('ε_eq vs ε_xy')
            plt.tight_layout()
            fig.savefig(os.path.join(fig_dir, "5_εxx_vs_εyy_散点图.png"), dpi=dpi)
            plt.close(fig)
            generated.append("5_εxx_vs_εyy_散点图.png")

            # 6. 主应变分布
            e1, e2, theta = compute_principal_strains(
                self.data.strain_xx, self.data.strain_yy, self.data.strain_xy)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            sc1 = axes[0].scatter(centroids[:, 0], centroids[:, 1], c=e1,
                                  cmap='RdBu_r', s=20, alpha=0.8,
                                  vmin=-max(abs(e1)), vmax=max(abs(e1)))
            axes[0].set_title('ε₁ (max principal)')
            axes[0].set_aspect('equal')
            orient_spatial_axes(axes[0])
            plt.colorbar(sc1, ax=axes[0], fraction=0.046)
            sc2 = axes[1].scatter(centroids[:, 0], centroids[:, 1], c=e2,
                                  cmap='RdBu_r', s=20, alpha=0.8,
                                  vmin=-max(abs(e2)), vmax=max(abs(e2)))
            axes[1].set_title('ε₂ (min principal)')
            axes[1].set_aspect('equal')
            orient_spatial_axes(axes[1])
            plt.colorbar(sc2, ax=axes[1], fraction=0.046)
            plt.tight_layout()
            fig.savefig(os.path.join(fig_dir, "6_主应变分布.png"), dpi=dpi)
            plt.close(fig)
            generated.append("6_主应变分布.png")

            # 7. 变形模式分类
            modes, ratio = compute_deformation_mode(e1, e2)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            mode_colors = ['gray', 'red', 'orange', 'green', 'blue', 'purple']
            mode_labels = ['Zero', 'Shear', 'Uni. tens.', 'Biaxial', 'Compress.', 'Mixed']
            mode_cmap = plt.matplotlib.colors.ListedColormap(mode_colors)
            sc = axes[0].scatter(centroids[:, 0], centroids[:, 1], c=modes,
                                 cmap=mode_cmap, s=20, alpha=0.8, vmin=0, vmax=5)
            cbar = plt.colorbar(sc, ax=axes[0], ticks=range(6), fraction=0.046)
            cbar.ax.set_yticklabels(mode_labels, fontsize=7)
            axes[0].set_title('Deformation mode')
            axes[0].set_aspect('equal')
            orient_spatial_axes(axes[0])
            axes[1].scatter(e1, e2, c=self.data.strain_eq, cmap='turbo', s=10, alpha=0.6)
            axes[1].axhline(y=0, color='gray', lw=0.5)
            axes[1].axvline(x=0, color='gray', lw=0.5)
            axes[1].set_xlabel('ε₁')
            axes[1].set_ylabel('ε₂')
            axes[1].set_title('ε₁ vs ε₂')
            total = len(modes)
            counts_text = "\n".join(
                [f"{mode_labels[m]}: {sum(modes==m)} ({sum(modes==m)/total*100:.1f}%)"
                 for m in range(6) if sum(modes==m) > 0])
            axes[1].text(0.05, 0.95, counts_text, transform=axes[1].transAxes,
                         va='top', fontsize=8, fontfamily='monospace',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.3))
            plt.tight_layout()
            fig.savefig(os.path.join(fig_dir, "7_变形模式分类.png"), dpi=dpi)
            plt.close(fig)
            generated.append("7_变形模式分类.png")

            # 8. 应变梯度
            gxx, gyy, gxy = compute_strain_gradient(
                self.data.strain_xx, self.data.strain_yy, self.data.strain_xy, centroids)
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
            for slot, (g, label) in enumerate(zip(
                [gxx, gyy, gxy], ['|∇ε_xx|', '|∇ε_yy|', '|∇ε_xy|'])):
                ax_i = axes[slot]  # 原 bug: 把循环序号当 axes 用, 第8步导出必崩
                vmin, vmax = np.nanpercentile(g, [5, 95])  # 边界秩亏点是 NaN
                sc = ax_i.scatter(centroids[:, 0], centroids[:, 1], c=g,
                                  cmap='inferno', s=20, alpha=0.8, vmin=vmin, vmax=vmax)
                ax_i.set_title(label)
                ax_i.set_aspect('equal')
                orient_spatial_axes(ax_i)
                plt.colorbar(sc, ax=ax_i, fraction=0.046)
            plt.tight_layout()
            fig.savefig(os.path.join(fig_dir, "8_应变梯度图.png"), dpi=dpi)
            plt.close(fig)
            generated.append("8_应变梯度图.png")

            # 9. 应变云图叠加(底图)
            if True:
                # 无底图时按质心实际范围建网格: ROI 不从原点开始时, 固定从 0 起
                # 会让大部分格点落在凸包外 (全 NaN), 甚至因 NaN 色标崩溃
                if self.data.has_image():
                    h_img = self.data.image.shape[0]
                    w_img = self.data.image.shape[1]
                    x_lo, x_hi = 0.0, float(w_img)
                    y_lo, y_hi = 0.0, float(h_img)
                else:
                    h_img = centroids[:, 1].max() + 10
                    w_img = centroids[:, 0].max() + 10
                    pad = 5.0
                    x_lo = max(0.0, float(centroids[:, 0].min()) - pad)
                    x_hi = float(centroids[:, 0].max()) + pad
                    y_lo = max(0.0, float(centroids[:, 1].min()) - pad)
                    y_hi = float(centroids[:, 1].max()) + pad
                gx = np.linspace(x_lo, x_hi, grid_size)
                gy = np.linspace(y_lo, y_hi, grid_size)
                GX, GY = np.meshgrid(gx, gy)
                grid_pts = np.column_stack((GX.ravel(), GY.ravel()))

                fig, axes = plt.subplots(2, 3, figsize=(15, 9))
                axes = axes.flatten()
                all_keys = [('ε_xx', self.data.strain_xx, 'RdBu_r'),
                            ('ε_yy', self.data.strain_yy, 'RdBu_r'),
                            ('ε_xy', self.data.strain_xy, 'RdBu_r'),
                            ('ε_eq', self.data.strain_eq, 'turbo'),
                            ('ω', self.data.rotation, 'RdBu_r')]
                for ax_i, (name, arr, cm) in enumerate(all_keys):
                    if arr is None:
                        axes[ax_i].text(0.5, 0.5, 'No data', ha='center', va='center')
                        continue
                    interp = griddata(centroids, arr, grid_pts,
                                      method='linear', fill_value=np.nan).reshape(grid_size, grid_size)
                    vmin, vmax = np.nanpercentile(interp, [2, 98])
                    if vmin == vmax or np.isnan(vmin):
                        vmin, vmax = np.nanmin(interp), np.nanmax(interp)
                    if self.data.has_image():
                        axes[ax_i].imshow(self.data.image, cmap='gray',
                                          origin='upper', aspect='equal',
                                          extent=(x_lo, x_hi, y_hi, y_lo))
                    im = axes[ax_i].imshow(interp, extent=(x_lo, x_hi, y_hi, y_lo),
                                           origin='upper', cmap=cm, alpha=0.6,
                                           vmin=vmin, vmax=vmax, interpolation='bilinear')
                    plt.colorbar(im, ax=axes[ax_i], fraction=0.046)
                    axes[ax_i].set_title(name)
                axes[5].axis('off')
                plt.tight_layout()
                fig.savefig(os.path.join(fig_dir, "9_各分量云图.png"), dpi=dpi)
                plt.close(fig)
                generated.append("9_各分量云图.png")

        # --- 导出统计数据 ---
        csv_path = os.path.join(data_dir, "统计数据.csv")
        has_area = self.data.element_area is not None
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(["量", "均值", "标准差", "最小值", "最大值", "中位数", "偏度", "峰度",
                             "加权均值", "加权标准差", "加权口径"])
            if self.data.has_displacement():
                d = self.data.distortions
                dx = self.data.displacements[:, 0]
                dy = self.data.displacements[:, 1]
                writer.writerow(["位移幅值", f"{d.mean():.6f}", f"{d.std():.6f}",
                                 f"{d.min():.6f}", f"{d.max():.6f}", f"{np.median(d):.6f}",
                                 f"{skew(d):.3f}", f"{kurtosis(d):.3f}", "", "", "不适用(逐原子)"])
                writer.writerow(["dx", f"{dx.mean():.6f}", f"{dx.std():.6f}",
                                 f"{dx.min():.6f}", f"{dx.max():.6f}",
                                 f"{np.median(dx):.6f}", "", "", "", "", "不适用(逐原子)"])
                writer.writerow(["dy", f"{dy.mean():.6f}", f"{dy.std():.6f}",
                                 f"{dy.min():.6f}", f"{dy.max():.6f}",
                                 f"{np.median(dy):.6f}", "", "", "", "", "不适用(逐原子)"])
            if self.data.has_strain():
                strain_rows = [("ε_xx", self.data.strain_xx), ("ε_yy", self.data.strain_yy),
                               ("ε_xy", self.data.strain_xy), ("ε_eq", self.data.strain_eq),
                               ("ω", self.data.rotation)]
                gl_names = {'gl_xx': 'ε_xx (GL)', 'gl_yy': 'ε_yy (GL)',
                            'gl_xy': 'ε_xy (GL)', 'gl_eq': 'ε_eq (GL)'}
                strain_rows += [(gl_names[k], v) for k, v in sorted(self.data.strain_gl.items())
                                if v is not None]
                for name, arr in strain_rows:
                    if arr is None:
                        continue
                    w_mean, w_std = weighted_mean_std(arr, self.data.element_area)
                    writer.writerow([name, f"{arr.mean():.6f}", f"{arr.std():.6f}",
                                     f"{arr.min():.6f}", f"{arr.max():.6f}",
                                     f"{np.median(arr):.6f}",
                                     f"{skew(arr):.3f}", f"{kurtosis(arr):.3f}",
                                     f"{w_mean:.6f}" if has_area else "",
                                     f"{w_std:.6f}" if has_area else "",
                                     "面积加权" if has_area else "元素等权(无面积数据)"])
                e1, e2, _ = compute_principal_strains(
                    self.data.strain_xx, self.data.strain_yy, self.data.strain_xy)
                e1_w = weighted_mean_std(e1, self.data.element_area)
                e2_w = weighted_mean_std(e2, self.data.element_area)
                writer.writerow(["ε₁ (主应变)", f"{e1.mean():.6f}", f"{e1.std():.6f}",
                                 f"{e1.min():.6f}", f"{e1.max():.6f}",
                                 f"{np.median(e1):.6f}", f"{skew(e1):.3f}", f"{kurtosis(e1):.3f}",
                                 f"{e1_w[0]:.6f}" if has_area else "",
                                 f"{e1_w[1]:.6f}" if has_area else "",
                                 "面积加权" if has_area else "元素等权(无面积数据)"])
                writer.writerow(["ε₂ (次应变)", f"{e2.mean():.6f}", f"{e2.std():.6f}",
                                 f"{e2.min():.6f}", f"{e2.max():.6f}",
                                 f"{np.median(e2):.6f}", f"{skew(e2):.3f}", f"{kurtosis(e2):.3f}",
                                 f"{e2_w[0]:.6f}" if has_area else "",
                                 f"{e2_w[1]:.6f}" if has_area else "",
                                 "面积加权" if has_area else "元素等权(无面积数据)"])
        generated.append("统计数据.csv")

        # --- 摘要报告 ---
        report_path = os.path.join(out_dir, "报告摘要.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("PPA 统计分析报告\n")
            f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            if self.data.has_displacement():
                d = self.data.distortions
                f.write("【位移场】\n")
                f.write(f"  原子数量: {self.data.n_atoms}\n")
                f.write(f"  位移幅值: μ={d.mean():.6f} ± {d.std():.6f}\n")
                f.write(f"           min={d.min():.6f}, max={d.max():.6f}\n")
                f.write(f"  各向异性: dx μ={self.data.displacements[:,0].mean():.6f}, "
                        f"dy μ={self.data.displacements[:,1].mean():.6f}\n\n")
            if self.data.has_strain():
                f.write("【应变张量】\n")
                f.write(f"  元素数量: {self.data.n_tri}\n")
                f.write(f"  坐标约定: 源CSV为{'物理坐标(y向上); 导入时已翻转为显示坐标(y向下), ε_xy/θ已变号' if self.data.physical_source else '图像显示坐标(y向下)'}\n")
                f.write(f"  统计口径: 面积加权 (μ±σ 后标注[面积加权]); 无标注为元素等权\n")
                strain_rows = [("ε_xx", self.data.strain_xx), ("ε_yy", self.data.strain_yy),
                               ("ε_xy", self.data.strain_xy), ("ε_eq", self.data.strain_eq),
                               ("ω", self.data.rotation)]
                gl_names = {'gl_xx': 'ε_xx (GL)', 'gl_yy': 'ε_yy (GL)',
                            'gl_xy': 'ε_xy (GL)', 'gl_eq': 'ε_eq (GL)'}
                strain_rows += [(gl_names[k], v) for k, v in sorted(self.data.strain_gl.items())
                                if v is not None]
                for name, arr in strain_rows:
                    if arr is not None:
                        w_mean, w_std = weighted_mean_std(arr, self.data.element_area)
                        note = " [面积加权]" if self.data.element_area is not None else ""
                        f.write(f"  {name}: μ={w_mean:.6f} ± {w_std:.6f}{note}  "
                                f"[{arr.min():.6f}, {arr.max():.6f}]\n")
                e1, e2, _ = compute_principal_strains(
                    self.data.strain_xx, self.data.strain_yy, self.data.strain_xy)
                e1_w = weighted_mean_std(e1, self.data.element_area)
                e2_w = weighted_mean_std(e2, self.data.element_area)
                f.write(f"  ε₁: μ={e1_w[0]:.6f} ± {e1_w[1]:.6f}\n")
                f.write(f"  ε₂: μ={e2_w[0]:.6f} ± {e2_w[1]:.6f}\n\n")
                f.write("【变形模式】\n")
                modes, _ = compute_deformation_mode(e1, e2)
                mode_labels = ['零应变', '纯剪切', '单轴拉伸', '等双轴拉伸', '双轴压缩', '混合']
                total = len(modes)
                for m in range(6):
                    count = sum(modes == m)
                    if count > 0:
                        f.write(f"  {mode_labels[m]}: {count} ({count/total*100:.1f}%)\n")
                f.write("\n")
            f.write("=" * 50 + "\n")
            f.write(f"输出文件共 {len(generated)} 个:\n")
            for g in generated:
                f.write(f"  - {g}\n")

        self.status.config(text=f"✅ 图表生成完成: {len(generated)} 个文件 → {out_dir}",
                           foreground='#006600')
        messagebox.showinfo("完成", f"图表已生成到:\n{out_dir}\n\n共 {len(generated)} 个文件")

    def _export_report(self):
        """生成报告（调用 _generate_all_plots）"""
        self._generate_all_plots()


# ================================================================
#  入口
# ================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = PPAStatsApp(root)
    root.mainloop()
