"""
HAADF-STEM 特征区域演化分析工具
================================
对 HAADF-STEM 图像堆栈（TIFF）进行原子柱强度追踪与特征区域演化分析，
输出 Nature/Science 期刊级别的出版图表。

处理流程：
  1. 背景提取（多帧边缘中值）→ 归一化（消除束流波动）
  2. Otsu 自适应阈值分割 → 形态学清洗
  3. 特征提取（面积 + 柱强度）→ 置信区间估计
  4. 图表输出（Fig1/Fig2 各 6 张 + 2 张组合图 + 1 张分割质检图）
     + CSV/JSON 数据 + 统计报告

无效帧口径：Otsu 失败（如常量帧）的帧记为无效，面积/强度/CI/N_eff 均为
NaN、CSV 的 Frame_Valid=0；分割成功但掩膜为空的帧面积=0（真实测量值）、
强度为 NaN（无特征可测）。旧版把这两类帧的强度记为 1.0，会污染相关/
趋势/谱分析等全部下游统计。

用法：
  python 应力面积统计.py --file input.tif --dt 0.2 --title "Fe surface reconstruction"
  python 应力面积统计.py  # 弹出GUI文件选择对话框
"""

import os
import sys
import json
import hashlib
import argparse
import dataclasses
import subprocess

# 修复：Windows GBK 控制台/重定向环境下，✓/emoji 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 必须在 import pyplot 之前设置（非交互式后端，用于纯文件输出）
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import welch
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
import tifffile as tiff
import matplotlib.colors as mcolors
from scipy import stats
import seaborn as sns

from stress_core import (safe_corrcoef, safe_cov, safe_divide, corrcoef_or_nan,
                         get_smooth_curve, normalize_frame)

# 工具版本：单一来源，banner / Statistical_Summary.md / ANALYSIS_REPORT.md 统一引用
__version__ = "2026.09.2"


@dataclasses.dataclass
class AnalysisConfig:
    """分析参数集中定义（此前散落为函数内魔法数）。

    字段默认值即历史行为；CLI --min-size / --dpi 覆盖对应字段，
    其余字段供编程调用时注入。__post_init__ 保证编程注入的非法值
    在构造时立即报错，与 CLI 侧校验同一口径。
    """
    margin_frac: float = 0.05       # 背景边缘带宽占图像短边比例
    n_bg_frames: int = 5            # 背景估计使用的前 N 帧
    min_size: int = 4               # remove_small_objects 最小保留像素数
    closing_structure: int = 3      # 闭运算/腐蚀/膨胀结构元边长
    savgol_window: int = 21         # Savitzky-Golay 最大窗口
    savgol_polyorder: int = 3       # Savitzky-Golay 多项式阶数
    z95: float = 1.96               # 强度 95% CI 的 z 值
    ess_max_lag: int = 24           # 空间有效样本量估计的最大滞后（像素）
    dpi: int = 600                  # 输出图像 dpi

    def __post_init__(self):
        if not (0.0 < self.margin_frac < 0.5):
            raise ValueError(f"margin_frac 应在 (0, 0.5) 开区间，得到 {self.margin_frac}")
        if self.n_bg_frames < 1:
            raise ValueError(f"n_bg_frames 应 >= 1，得到 {self.n_bg_frames}")
        if self.min_size < 1:
            raise ValueError(f"min_size 应 >= 1，得到 {self.min_size}")
        if self.closing_structure < 1:
            raise ValueError(f"closing_structure 应 >= 1，得到 {self.closing_structure}")
        if self.savgol_window < 3 or self.savgol_polyorder < 1:
            raise ValueError(f"savgol_window 应 >= 3 且 polyorder 应 >= 1，得到 "
                             f"{self.savgol_window}/{self.savgol_polyorder}")
        if not (0.0 < self.z95 <= 10.0):
            raise ValueError(f"z95 应在 (0, 10]，得到 {self.z95}")
        if self.ess_max_lag < 1:
            raise ValueError(f"ess_max_lag 应 >= 1，得到 {self.ess_max_lag}")
        if not (0 < self.dpi <= 10000):
            raise ValueError(f"dpi 应在 [1, 10000]，得到 {self.dpi}")

# 处理失败时的统一返回值，与成功路径的
# (times, areas, intensities, intensity_ci, area_ci, thresholds, n_eff, total_frames)
# 元数一致，调用方可稳定解包。（历史教训：新增返回值时早期失败 return 未同步，
# 曾退化为短元组导致调用方解包崩溃；每次扩展元组必须同步 _PROCESS_FAILURE 与测试。）
_PROCESS_FAILURE = (None, None, None, None, None, None, None, 0)

# 全局 warnings 过滤已移除（linux 移植）：需要压制警告的计算区域改用局部
# np.errstate，避免屏蔽用户脚本中的其他诊断信息。
# safe_corrcoef / safe_cov / safe_divide / get_smooth_curve / normalize_frame
# 等纯函数已抽取到 stress_core.py（linux 移植），配套单元测试见 tests/。


# ========== 风格设置 ==========

def setup_nature_style(dpi=600):
    """设置Nature/Science期刊级别风格（全局只需调用一次）

    dpi 同时写入 savefig.dpi，所有 _save_figure 统一从 rcParams 取值，
    不再在各保存点散落 dpi=600 字面量。
    """
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 9,
        'font.weight': 'normal',
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'axes.labelweight': 'bold',
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#222222',
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.color': '#222222',
        'ytick.color': '#222222',
        'legend.fontsize': 8,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#DDDDDD',
        'figure.dpi': 300,
        'figure.figsize': (7.2, 4.8),
        'figure.autolayout': False,
        'savefig.dpi': dpi,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
        'lines.markeredgewidth': 0.5,
        'grid.color': '#E0E0E0',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.6,
    })
    print("✓ Applied Nature/Science publication style")


def get_color_palette():
    """创建Nature风格的色彩方案（仅返回实际使用的主色盘）"""
    return {
        'blue': '#1F77B4',
        'orange': '#FF7F0E',
        'green': '#2CA02C',
        'red': '#D62728',
        'purple': '#9467BD',
        'brown': '#8C564B',
        'pink': '#E377C2',
        'gray': '#7F7F7F',
        'yellow': '#BCBD22',
        'cyan': '#17BECF',
    }


# ========== 分析辅助 ==========

def compute_power_spectrum(signal, dt):
    """Welch 功率谱密度估计（Hann 窗 + 线性去趋势）。

    相比早期实现直接画 |FFT| 幅度谱：线性去趋势抑制演化趋势主导的
    低频泄漏，分段平均降低方差，结果为可比的功率谱密度（PSD）而非幅度。

    Parameters
    ----------
    signal : array-like, 1D
        时间序列
    dt : float
        采样间隔（秒），必须为正的有限数

    Returns
    -------
    freqs, psd : np.ndarray
        频率轴（Hz）与单边功率谱密度
    """
    signal = np.asarray(signal, dtype=np.float64)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"dt 必须为正的有限数（秒），得到 {dt}")
    nperseg = min(len(signal), 256)
    freqs, psd = welch(signal, fs=1.0 / dt, window='hann',
                       nperseg=nperseg, detrend='linear')
    return freqs, psd


def rolling_correlation(areas, intensities, window_size):
    """滑动窗口 Pearson 相关；常量窗口返回 NaN，绘图时自然断线。

    早期实现使用 safe_corrcoef 的填充值 0.0，常量窗口会被误读为
    "测得零相关"；NaN 才表达"窗口内无法计算"。含非有限值（无效帧）的
    窗口同样返回 NaN。
    """
    areas = np.asarray(areas, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    corr = []
    for i in range(len(areas) - window_size + 1):
        x, y = areas[i:i + window_size], intensities[i:i + window_size]
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            corr.append(np.nan)
        elif np.std(x) < 1e-12 or np.std(y) < 1e-12:
            corr.append(np.nan)
        else:
            corr.append(safe_corrcoef(x, y))
    return np.array(corr)


# ========== NaN 感知统计辅助 ==========
# 无效帧（Otsu 失败）与空掩膜帧会引入 NaN：所有面向报告/图表的统计一律
# 经由这些辅助函数，只使用有限样本，避免 NaN 传播成 "nan" 字面量或静默
# 污染（如 percentile/linregress/nanmean 全 NaN 告警）。

def _fmt(value, spec=".3f", fallback="n/a"):
    """有限数按格式化字符串输出，NaN/inf 输出占位说明。"""
    try:
        if np.isfinite(value):
            return format(float(value), spec)
    except TypeError:
        pass
    return fallback


def _finite_values(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _first_last_change(values):
    """首末有效帧的相对变化（%）；有效帧不足或首值为零时返回 None。"""
    v = _finite_values(values)
    if v.size < 2 or v[0] == 0:
        return None
    return float((v[-1] - v[0]) / v[0] * 100.0)


def _linregress_safe(times, values):
    """有限样本对的线性回归；样本不足返回 None（调用方自行降级展示）。"""
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(times) & np.isfinite(values)
    if np.count_nonzero(mask) < 2:
        return None
    return stats.linregress(times[mask], values[mask])


# ========== 文件选择与输出目录 ==========

def get_file_and_params():
    """
    获取文件路径和分析参数：CLI → GUI → 手动输入 三级降级

    Returns
    -------
    file_path : str
        TIFF文件路径
    time_interval : float
        帧时间间隔（秒）
    title : str
        样品描述（用于图表标题）
    shared_threshold : bool
        是否使用全序列共享 Otsu 阈值
    config : AnalysisConfig
        分析参数（CLI --min-size/--dpi 覆盖对应字段）
    output_override : str or None
        --output 指定的输出目录；None 表示在输入文件旁创建时间戳目录
    """
    parser = argparse.ArgumentParser(description="HAADF-STEM Feature Evolution Analysis")
    parser.add_argument("--file", type=str, help="TIFF文件路径")
    parser.add_argument("--dt", type=float, default=0.2, help="帧时间间隔（秒），默认0.2")
    parser.add_argument("--title", type=str, default="Fe surface reconstruction",
                        help="样品描述（用于图表标题），默认 'Fe surface reconstruction'")
    parser.add_argument("--shared-threshold", action="store_true",
                        help="全序列使用统一 Otsu 阈值（逐帧阈值的中位数），"
                             "消除阈值随帧漂移混入面积序列的噪声；默认逐帧独立 Otsu")
    parser.add_argument("--min-size", type=int, default=4, dest="min_size",
                        help="形态学清洗保留的最小对象面积（像素），默认4")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录（默认在输入文件旁创建带时间戳的 "
                             "NatureStyle_Analysis_* 目录）")
    parser.add_argument("--dpi", type=int, default=600,
                        help="输出图像 dpi，默认600（上限 10000）")
    args = parser.parse_args()

    # CLI 参数校验：显式给出的参数非法时立即报错退出（exit code 2），
    # 而不是静默落入 GUI / 手动输入，或带着无效参数跑到一半崩溃。
    # （--dt 0 曾使时间轴全零、Fig2A 的 rfftfreq 抛 ZeroDivisionError）
    if not np.isfinite(args.dt) or args.dt <= 0:
        parser.error(f"--dt 必须为正的有限数字（秒），得到 {args.dt}")
    if args.min_size < 1:
        parser.error(f"--min-size 必须为正整数，得到 {args.min_size}")
    if not (0 < args.dpi <= 10000):
        parser.error(f"--dpi 必须为 1–10000 的整数，得到 {args.dpi}")
    if args.file is not None:
        expanded = os.path.expanduser(args.file)
        if not os.path.exists(expanded):
            parser.error(f"--file 指定的文件不存在: {args.file}")
        args.file = expanded
    if args.output is not None:
        args.output = os.path.expanduser(args.output)

    config = AnalysisConfig(min_size=args.min_size, dpi=args.dpi)
    output_override = args.output

    if args.file:
        return args.file, args.dt, args.title, args.shared_threshold, config, output_override

    # 尝试GUI文件选择 + 参数询问
    try:
        import tkinter as tk
        from tkinter import filedialog, simpledialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(
            title="Select HAADF-STEM Image Stack (TIFF)",
            initialdir=os.path.expanduser("~"),
            filetypes=[("TIFF files", "*.tif;*.tiff"), ("All files", "*.*")]
        )
        if not file_path:
            root.destroy()
            print("   ✗ 未选择文件，已退出。")
            return "", args.dt, args.title, args.shared_threshold, config, output_override

        # GUI 模式显式询问 dt：此前静默用默认 0.2，用户拿到的整条时间轴
        # 可能都是错的且无任何提示。取消输入视为放弃本次分析。
        dt_value = simpledialog.askfloat(
            "帧时间间隔", "每帧时间间隔（秒）：",
            initialvalue=args.dt, minvalue=1e-9, parent=root)
        if dt_value is None or not np.isfinite(dt_value) or dt_value <= 0:
            root.destroy()
            print("   ✗ 未提供有效的帧时间间隔，已退出。")
            return "", args.dt, args.title, args.shared_threshold, config, output_override

        title_value = simpledialog.askstring(
            "样品描述", "样品描述（用于图表标题，留空使用默认）：",
            initialvalue=args.title, parent=root)
        root.destroy()  # 修复：销毁tkinter根窗口，避免残留进程
        title = title_value if title_value else args.title
        return file_path, float(dt_value), title, args.shared_threshold, config, output_override
    except Exception as e:
        print(f"   ⚠ GUI file dialog failed: {e}")

    # 降级为手动输入（异常循环重试，避免 float(input()) 直接抛异常退出；linux 移植）
    while True:
        file_path = input("Please enter the full path to the TIFF file: ").strip()
        if file_path:
            break
        print("   ⚠ 文件路径不能为空，请重新输入。")

    dt = args.dt
    while True:
        dt_input = input("Time interval per frame in seconds [0.2]: ").strip()
        if not dt_input:
            break
        try:
            parsed_dt = float(dt_input)
            # isfinite 必须与 CLI 校验同口径：'nan'/'inf' 通过 "<=0" 检查
            # 后会使时间轴全 NaN/inf（曾为校验缺口）
            if not np.isfinite(parsed_dt) or parsed_dt <= 0:
                print("   ⚠ 时间间隔必须为正的有限数（秒），请重新输入。")
                continue
            dt = parsed_dt
            break
        except ValueError:
            print(f"   ⚠ 无法解析 '{dt_input}' 为数字，请重新输入（例如 0.2）。")

    while True:
        title_input = input("Sample description for figure titles [Fe surface reconstruction]: ").strip()
        title = title_input if title_input else args.title
        break

    return file_path, dt, title, args.shared_threshold, config, output_override


def get_output_directory(tif_path, override=None):
    """获取输出目录：--output 指定优先；否则在输入文件旁创建时间戳目录。"""
    if override:
        return os.path.abspath(override)
    from datetime import datetime
    tif_dir = os.path.dirname(tif_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(tif_dir, f"NatureStyle_Analysis_{timestamp}")
    return output_dir


def prepare_output_directory(output_dir):
    """创建输出目录；失败时返回错误消息（供 main 转成退出码 1），成功返回 None。

    保护点（此前直接在 try 块外 makedirs，非法路径抛裸 traceback）：
    - 路径已存在但不是目录；
    - 父目录无写权限（提前给出可读信息，而非依赖深层 OSError）；
    - 目录已存在且非空：静默覆盖会丢失上一次结果（时间戳目录为秒级精度，
      同秒重跑也会撞名），明确拒绝并提示换目录。
    """
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        return f"输出路径已存在且不是目录: {output_dir}"
    parent = os.path.dirname(os.path.abspath(output_dir)) or "."
    if os.path.exists(parent) and not os.access(parent, os.W_OK):
        return f"输出目录所在位置没有写入权限: {parent}"
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        return (f"输出目录已存在且非空，为避免覆盖既有结果已终止: {output_dir}\n"
                "     请用 --output 指定新目录，或先移走旧结果后重试。")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        return f"无法创建输出目录 {output_dir}: {e}"
    return None


def sha256_file(path, chunk_bytes=1 << 20):
    """输入文件的 SHA-256（分块读取，供元数据记录保证可追溯）。"""
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk_bytes), b''):
            digest.update(block)
    return digest.hexdigest()


def open_output_directory(output_dir):
    """用系统默认文件管理器打开输出目录（linux 移植）。

    Windows 使用 os.startfile；macOS 使用 open；Linux/其他优先 xdg-open。
    打开失败时打印清晰提示，不静默吞异常。
    """
    if not os.path.isdir(output_dir):
        print(f"   ⚠ 输出目录不存在，无法打开: {output_dir}")
        return

    try:
        if os.name == 'nt':
            os.startfile(output_dir)  # type: ignore[attr-defined]
            return
        opener = 'open' if sys.platform == 'darwin' else 'xdg-open'
        subprocess.Popen([opener, output_dir])
    except Exception as e:
        print(f"   ⚠ 无法自动打开输出目录: {e}")
        print(f"     请手动打开: {output_dir}")


# ========== 核心图像处理 ==========

def effective_sample_size_2d(values, mask, max_lag=24):
    """
    估计空间相关随机场中均值的有效独立样本数。

    HAADF-STEM 图像相邻像素强度强相关，把 ROI 内全部像素当独立样本
    （σ/√N）会严重低估均值的不确定度。对可分离相关的二维随机场，
    均值方差近似按 VIFx·VIFy 膨胀，其中每轴的方差膨胀因子
    VIF = 1 + 2·Σ_k ρ(k)（k 为滞后，累加到自相关首次过零），
    即积分尺度（effective range）的标准结果。据此把像素数 N 折减为
    有效样本数 N_eff = N / (VIFx·VIFy)。

    Parameters
    ----------
    values : ndarray (H, W)
        该帧归一化强度图
    mask : bool ndarray (H, W)
        特征区域掩膜
    max_lag : int
        自相关估计的最大滞后（像素）

    Returns
    -------
    float
        有效样本数，介于 1 和像素数 N 之间
    """
    n = int(np.count_nonzero(mask))
    if n < 16:
        return float(n)

    residual = np.where(mask, values - np.mean(values[mask]), 0.0)

    def axis_vif(axis):
        # 该轴上方差膨胀因子：1 + 2 * Σ_lag ρ(lag)，ρ 过零即停止累加
        vif = 1.0
        for lag in range(1, min(max_lag, residual.shape[axis] - 1) + 1):
            if axis == 1:
                a, b = residual[:, :-lag], residual[:, lag:]
                valid = mask[:, :-lag] & mask[:, lag:]
            else:
                a, b = residual[:-lag, :], residual[lag:, :]
                valid = mask[:-lag, :] & mask[lag:, :]
            if np.count_nonzero(valid) < 8:
                break
            aa, bb = a[valid], b[valid]
            denom = np.sqrt(np.sum(aa * aa) * np.sum(bb * bb))
            if denom <= 0:
                break
            rho = float(np.sum(aa * bb) / denom)
            if rho <= 0:
                break
            vif += 2.0 * rho
        return vif

    n_eff = n / (axis_vif(1) * axis_vif(0))
    return float(min(max(n_eff, 1.0), n))


def process_stem_stack(tif_path, time_interval=0.2, shared_threshold=False,
                       meta=None, config=None):
    """
    HAADF-STEM 原子柱强度追踪与特征区域提取。

    处理流程：
      1. 逐帧坏像素清洗（NaN/inf 以有限像素中值替换）
      2. 多帧边缘中值估计背景基准 I₀（抗单帧异常）
      3. 逐帧归一化 img/I₀（消除束流波动）
      4. Otsu阈值分割特征区域（默认逐帧独立 Otsu；shared_threshold=True 时
         先扫一遍取全序列逐帧阈值的中位数作为统一阈值，避免阈值随帧漂移
         混入面积时间序列——该漂移是分割噪声而非真实演化）
      5. 形态学清洗（去小物体 + 闭运算）
      6. 提取面积（像素数）和平均柱强度（无量纲比值），并记录每帧实际
         使用的阈值（随 CSV 导出，保证分割口径可复现）

    Parameters
    ----------
    tif_path : str
        TIFF图像堆栈路径，Shape: (T, H, W)，单通道16-bit。
        也接受单张灰度 (H,W)、单张彩色 (H,W,3/4)；后两者会被规范化后
        提示至少需要 2 帧时间序列。逐页/memmap 读取（linux 移植），
        不再一次性 imread 全量载入内存。
    time_interval : float
        帧时间间隔（秒），默认0.2
    shared_threshold : bool
        是否使用全序列共享 Otsu 阈值（逐帧阈值中位数）。默认 False
        保持逐帧独立 Otsu 的历史行为。共享模式会对全堆栈多读一遍。
    meta : dict, optional
        若提供 dict，则填充处理元数据（背景基准、形态学参数、阈值模式、
        坏像素统计、版本号等），供报告/摘要记录处理口径，保证可复现。
    config : AnalysisConfig, optional
        分析参数（margin/min_size/闭运算结构元/有效样本量滞后/z95 等），
        默认 None 时使用 AnalysisConfig() 即历史默认行为。

    Returns
    -------
    times : np.ndarray, shape (T,)
        时间轴（秒）
    feature_areas : np.ndarray, shape (T,)
        每帧特征区域面积（像素数）；Otsu 失败的无效帧为 NaN
        （分割成功的空掩膜帧为真实测量值 0）
    feature_intensities : np.ndarray, shape (T,)
        每帧背景归一化平均柱强度（无量纲比值 I/I₀）；无效帧与空掩膜帧
        为 NaN（旧版记 1.0，会把"无特征"伪装成"测得背景强度"污染统计）
    intensity_confidence : np.ndarray, shape (T,)
        每帧强度的95%置信区间半宽（基于空间自相关折减后的有效样本量，
        而非把 ROI 内像素当独立样本）；强度未测量的帧为 NaN
    area_confidence : np.ndarray, shape (T,)
        每帧面积的不确定度半宽（掩膜 ±1 像素形态学边界带的一半，
        反映分割边界对面积的系统不确定性；**不是**统计置信区间）
    thresholds : np.ndarray, shape (T,)
        每帧实际使用的分割阈值（Otsu 失败的帧为 NaN，即 Frame_Valid=0）
    n_eff : np.ndarray, shape (T,)
        每帧强度 CI 使用的空间有效样本数（随 CSV 导出）；未测量帧为 NaN
    total_frames : int
        总帧数

    失败时返回 ``(None, None, None, None, None, None, None, 0)``。
    """
    cfg = config if config is not None else AnalysisConfig()
    print(f"\n📊 PROCESSING HAADF-STEM IMAGE STACK")
    print(f"   File: {os.path.basename(tif_path)}")

    try:
        tif = tiff.TiffFile(tif_path)
    except Exception as e:
        print(f"   ✗ TIFF read error: {e}")
        return _PROCESS_FAILURE

    try:
        with tif:
            # 先读元数据，判断页数和颜色轴；不再一次性 imread 全量数据（linux 移植）。
            series_list = tif.series
            if not series_list:
                print("   ✗ TIFF 中没有可读取的 series。")
                return _PROCESS_FAILURE
            series = series_list[0]
            axes = getattr(series, "axes", "") or ""
            series_shape = tuple(getattr(series, "shape", ()))

            # 4D 彩色堆叠：明确拒绝并给出清晰错误，避免逐页颜色处理歧义（linux 移植）。
            if len(series_shape) == 4 and ("C" in axes or "S" in axes) and series_shape[-1] in (3, 4):
                print("   ✗ 检测到 4D 彩色堆叠 (T, H, W, C)。")
                print("     本工具当前只支持单通道 (T, H, W) 灰度堆叠；")
                print("     请先转换为灰度 TIFF（或按通道拆分）后再分析。")
                return _PROCESS_FAILURE

            pages = tif.pages
            n_pages = len(pages)

            # 判断每帧如何读取（linux 移植）：
            # - 多页 TIFF：每页一帧，逐页 asarray()
            # - 单页 3D 堆叠（如某些 tifffile 写入的 (T,H,W) 文件）：用 memmap
            #   切片逐帧读取，避免 pages[0].asarray() 一次性读入整个堆叠。
            single_page_3d_stack = False
            memmap_stack = None
            if n_pages == 1 and len(series_shape) == 3:
                first_page_shape = pages[0].shape
                is_3d_color = (("C" in axes or "S" in axes) and first_page_shape[-1] in (3, 4))
                if not is_3d_color and first_page_shape[0] > 1:
                    single_page_3d_stack = True
                    try:
                        memmap_stack = tiff.memmap(tif_path)
                    except Exception as e:
                        print(f"   ⚠ 无法以 memmap 方式读取单页 3D 堆叠，回退为整页读取: {e}")
                        single_page_3d_stack = False

            if single_page_3d_stack:
                total_frames = int(series_shape[0])
            else:
                total_frames = n_pages

            bad_pixel_by_frame = {}

            def read_frame(i):
                # 逐帧读取：多页 TIFF 按页读取；单页 3D 堆叠按 memmap 切片读取（linux 移植）。
                if single_page_3d_stack:
                    frame = normalize_frame(np.asarray(memmap_stack[i]))
                else:
                    frame = normalize_frame(pages[i].asarray())
                # 坏像素清洗（v2026.09）：NaN/inf 以该帧有限像素中值替换。
                # 单个坏像素曾使背景中值变 NaN，进而全帧 Otsu 失败、面积静默归零。
                # 整型帧不含 NaN/inf，零开销直通；坏点占比 >50% 视为输入损坏。
                finite = np.isfinite(frame)
                n_bad = int(finite.size - np.count_nonzero(finite))
                if n_bad:
                    ratio = n_bad / finite.size
                    if ratio > 0.5:
                        raise ValueError(
                            f"第 {i} 帧有 {ratio:.1%} 的像素为 NaN/inf，疑似损坏数据，已终止分析")
                    fill = float(np.median(frame[finite]))
                    frame = frame.astype(np.float64, copy=True)
                    frame[~finite] = fill
                    bad_pixel_by_frame[i] = n_bad
                return frame

            if total_frames < 2:
                print(f"   ✗ Need at least 2 frames for time-series analysis, got {total_frames}.")
                if n_pages >= 1:
                    try:
                        single = read_frame(0)
                        print(f"     单页规范化后形状: {single.shape}（单张 2D/彩色图不能做时间序列分析）")
                    except Exception as e:
                        print(f"     单页规范化失败: {e}")
                return _PROCESS_FAILURE

            # 统一规范化第一页/帧，得到 h/w（修复 3D (H,W,C) 背景估计 h/w 错位；linux 移植）
            try:
                first_frame = read_frame(0)
            except Exception as e:
                print(f"   ✗ 首页规范化失败: {e}")
                return _PROCESS_FAILURE
            if first_frame.ndim != 2:
                print(f"   ✗ 首页规范化后应为 2D，得到 {first_frame.shape}")
                return _PROCESS_FAILURE
            h, w = first_frame.shape

            times = np.arange(total_frames) * time_interval
            print(f"   ✓ Frames: {total_frames}, Shape: ({h}, {w}), "
                  f"Time: {times[-1]:.2f}s, dt={time_interval}s")

            # 1. 多帧背景基准估计（取前N帧四周边缘的中值，抗单帧异常和漂移）
            margin = max(1, int(min(h, w) * cfg.margin_frac))
            n_bg_frames = min(cfg.n_bg_frames, total_frames)
            bg_values = []
            for i in range(n_bg_frames):
                frame = read_frame(i).astype(np.float64)
                bg_roi = np.concatenate([
                    frame[:margin, :].ravel(),
                    frame[-margin:, :].ravel(),
                    frame[:, :margin].ravel(),
                    frame[:, -margin:].ravel()
                ])
                bg_values.append(np.median(bg_roi))
            bg_intensity = np.median(bg_values)

            # 背景有效性校验（坏像素清洗后的最后防线）：无效基准会使归一化、
            # 分割与强度读数全线失真，明确失败好过带着错误口径产出垃圾。
            if not np.isfinite(bg_intensity) or bg_intensity <= 0:
                print(f"   ✗ 背景基准无效: {bg_intensity}（前 {n_bg_frames} 帧边缘中值 ≤0 或非有限）。")
                print("     请检查输入数据（探测器本底设置 / 坏帧）后重试。")
                return _PROCESS_FAILURE

            print(f"   ✓ Background intensity (I₀): {bg_intensity:.2f} "
                  f"(median of {n_bg_frames} frames)")

            # 存储结果
            feature_areas = []
            feature_intensities = []
            intensity_confidence = []  # 强度的95%置信区间
            area_confidence = []       # 面积不确定度（±1px 形态学边界带的一半）
            thresholds = []            # 每帧实际使用的分割阈值
            n_eff_series = []          # 每帧强度 CI 的空间有效样本数

            # 分割质检采样：首/中/末帧保留 8-bit 显示图 + 掩膜，
            # 供 Segmentation_QA.png 目检分割质量（出版审稿常规要求）。
            qa_indices = {0, total_frames // 2, total_frames - 1}
            qa_samples = []

            # 共享阈值模式：先扫一遍取每帧 Otsu 的中位数作为全序列统一阈值。
            # 逐帧独立 Otsu 的阈值漂移会混入面积时间序列（阈值噪声 ≠ 真实演化），
            # 共享阈值消除这一来源；第一遍的逐帧阈值保留用于报告阈值稳定性。
            per_frame_otsu = None
            shared_thresh_value = None
            if shared_threshold:
                otsu_vals = []
                for idx in range(total_frames):
                    img = read_frame(idx).astype(np.float64)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        img_norm = img / bg_intensity
                    try:
                        otsu_vals.append(float(threshold_otsu(img_norm)))
                    except ValueError:
                        otsu_vals.append(np.nan)
                per_frame_otsu = np.array(otsu_vals, dtype=np.float64)
                finite_otsu = per_frame_otsu[np.isfinite(per_frame_otsu)]
                if finite_otsu.size == 0:
                    print("   ⚠ 共享阈值不可用（无有效帧阈值），回退为逐帧 Otsu")
                    shared_threshold = False
                else:
                    shared_thresh_value = float(np.median(finite_otsu))
                    print(f"   ✓ 共享 Otsu 阈值: {shared_thresh_value:.4f} "
                          f"(逐帧中位数; 逐帧阈值 SD={np.std(finite_otsu):.4f}, "
                          f"CV={safe_divide(np.std(finite_otsu), np.mean(finite_otsu), 0.0):.3f})")

            progress_step = max(1, total_frames // 20)  # 每5%打印一次进度

            for idx in range(total_frames):
                # 逐帧读取：多页 TIFF 按页读取；单页 3D 堆叠按 memmap 切片读取（linux 移植）。
                img = read_frame(idx).astype(np.float64)

                if idx % progress_step == 0:
                    pct = idx / total_frames * 100
                    print(f"\r   Processing: {pct:.0f}% ({idx}/{total_frames})", end='', flush=True)

                # 2. 归一化处理（消除束流波动影响）；
                #    局部 np.errstate 替代全局警告过滤（linux 移植）。
                with np.errstate(divide='ignore', invalid='ignore'):
                    img_norm = img / bg_intensity  # 无量纲比值 I/I₀

                    # 3. 阈值分割：共享模式用全序列统一阈值；逐帧模式用 Otsu
                    if shared_threshold:
                        thresh = shared_thresh_value
                    else:
                        try:
                            thresh = threshold_otsu(img_norm)
                        except ValueError:
                            # 若图像为常量（Otsu无法计算），该帧无效：
                            # 全部测量值记 NaN（而非伪 0/1.0），CSV Frame_Valid=0
                            feature_areas.append(np.nan)
                            feature_intensities.append(np.nan)
                            intensity_confidence.append(np.nan)
                            area_confidence.append(np.nan)
                            thresholds.append(np.nan)
                            n_eff_series.append(np.nan)
                            continue
                    thresholds.append(thresh)

                    feature_mask = img_norm > thresh

                    # 4. 形态学清洗（去掉孤立的探测器噪声点）
                    #    掩膜为 2D，structure 固定为 2D，避免 3D/4D 输入导致维度不匹配。
                    feature_mask = remove_small_objects(feature_mask, min_size=cfg.min_size)
                    closing_struct = np.ones((cfg.closing_structure, cfg.closing_structure))
                    feature_mask = ndimage.binary_closing(feature_mask, structure=closing_struct)

                    # 5. 提取特征
                    area = int(np.sum(feature_mask))
                    # 面积不确定性：掩膜 ±1 像素形态学边界带的一半。
                    # 分割阈值对边界像素的归属最敏感，这比计数统计（泊松 √A）更
                    # 能反映面积测量的真实不确定度。（主树统计版，勿回退）
                    area_eroded = int(np.sum(ndimage.binary_erosion(feature_mask, structure=closing_struct)))
                    area_dilated = int(np.sum(ndimage.binary_dilation(feature_mask, structure=closing_struct)))
                    area_ci = 0.5 * (area_dilated - area_eroded)

                    if area > 0:
                        mean_intensity = float(np.mean(img_norm[feature_mask]))
                        # 强度的95%置信区间：z95 * σ/√N_eff。
                        # 相邻像素强度空间相关，用有效样本量折减，避免把 ROI 内
                        # 全部像素当独立样本而严重高估置信度。（主树统计版，勿回退）
                        local_variance = float(np.var(img_norm[feature_mask]))
                        n_eff = effective_sample_size_2d(img_norm, feature_mask,
                                                         max_lag=cfg.ess_max_lag)
                        ci = cfg.z95 * np.sqrt(local_variance / n_eff)
                    else:
                        # 空掩膜：面积 0 是真实测量值，但强度无特征可测 → NaN。
                        # （旧版记 1.0="等于背景"会把伪造值混入相关/趋势/谱统计）
                        mean_intensity = np.nan
                        ci = np.nan
                        n_eff = np.nan

                    if idx in qa_indices:
                        # 1–99 分位拉伸到 8-bit，固定显示口径且内存占用极小
                        p_lo, p_hi = np.percentile(img_norm, (1, 99))
                        if p_hi > p_lo:
                            disp8 = np.clip((img_norm - p_lo) / (p_hi - p_lo) * 255.0,
                                            0, 255).astype(np.uint8)
                        else:
                            disp8 = np.zeros(img_norm.shape, dtype=np.uint8)
                        qa_samples.append((idx, disp8, feature_mask.copy(), float(thresh)))

                feature_areas.append(area)
                feature_intensities.append(mean_intensity)
                intensity_confidence.append(ci)
                area_confidence.append(area_ci)
                n_eff_series.append(n_eff)

            print(f"\r   Processing: 100% ({total_frames}/{total_frames})")

            if bad_pixel_by_frame:
                print(f"   ⚠ 非有限坏像素已清洗: {len(bad_pixel_by_frame)}/{total_frames} 帧，"
                      f"共 {sum(bad_pixel_by_frame.values())} 个像素（有限像素中值填充）")

    except Exception as e:
        print(f"   ✗ Processing error: {e}")
        import traceback
        traceback.print_exc()
        return _PROCESS_FAILURE

    feature_areas = np.array(feature_areas, dtype=np.float64)
    feature_intensities = np.array(feature_intensities, dtype=np.float64)
    intensity_confidence = np.array(intensity_confidence, dtype=np.float64)
    area_confidence = np.array(area_confidence, dtype=np.float64)
    thresholds = np.array(thresholds, dtype=np.float64)
    n_eff_series = np.array(n_eff_series, dtype=np.float64)

    valid_frames = np.isfinite(thresholds)
    n_invalid = int(np.count_nonzero(~valid_frames))
    n_empty = int(np.count_nonzero(valid_frames & (feature_areas == 0)))

    print(f"\n   ✓ Analysis complete")
    finite_areas = _finite_values(feature_areas)
    if finite_areas.size:
        print(f"   • Mean feature area: {np.mean(finite_areas):.0f} ± {np.std(finite_areas):.0f} px")
    else:
        print("   • 无有效面积帧（全部帧分割失败）")
    finite_thresh = thresholds[np.isfinite(thresholds)]
    if finite_thresh.size:
        print(f"   • Segmentation threshold: mode={'shared' if shared_threshold else 'per-frame Otsu'}, "
              f"mean={np.mean(finite_thresh):.4f}, SD={np.std(finite_thresh):.4f}")
    if n_invalid:
        print(f"   ⚠ 无效帧（Otsu 失败，计为 NaN）: {n_invalid}/{total_frames}")
    if n_empty:
        print(f"   ⚠ 空掩膜帧（面积=0，强度记 NaN）: {n_empty}/{total_frames}")
    area_change = _first_last_change(feature_areas)
    if area_change is not None:
        print(f"   • Area trend: {area_change:+.1f}% change")

    if meta is not None:
        meta.update({
            "version": __version__,
            "input_file": os.path.abspath(tif_path),
            "image_shape": (int(h), int(w)),
            "dtype": str(first_frame.dtype),
            "n_frames": int(total_frames),
            "time_interval_s": float(time_interval),
            "bg_intensity": float(bg_intensity),
            "bg_margin_px": int(margin),
            "bg_frames": int(n_bg_frames),
            "min_size": cfg.min_size,
            "closing_structure": f"{cfg.closing_structure}x{cfg.closing_structure}",
            "threshold_mode": ("shared (median of per-frame Otsu)"
                               if shared_threshold else "per-frame Otsu"),
            "shared_threshold_value": shared_thresh_value,
            "bad_pixel_frames": len(bad_pixel_by_frame),
            "bad_pixels_total": int(sum(bad_pixel_by_frame.values())),
            "invalid_frames": n_invalid,
            "empty_mask_frames": n_empty,
            "qa_samples": qa_samples,
        })

    return times, feature_areas, feature_intensities, intensity_confidence, area_confidence, thresholds, n_eff_series, total_frames


# ========== 图表面板渲染（单图与组合图共用，消除两处实现漂移） ==========
#
# 历史教训：单图与组合图各自手写一遍面板逻辑，曾出现组合版 Fig1B 丢趋势线、
# Fig2A 丢强度谱、Fig2F 丢阈值线等行为漂移。现抽取 _panel_* 渲染函数，
# 单图（style='full'）与组合图（style='compact'）共用同一份数据口径。

def _save_figure(fig, path, tight=True, tight_rect=None):
    """统一保存：dpi 由 rcParams('savefig.dpi') 控制（setup_nature_style 设置）。"""
    if tight:
        plt.tight_layout(rect=tight_rect)
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _add_panel_label(ax, label, style='full'):
    """面板编号：单图（full）放轴内左上，组合图（compact）放轴外左上。"""
    if style == 'full':
        ax.text(0.02, 0.98, label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')
    else:
        ax.text(-0.15, 1.02, label, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')


def _style_axes(ax, xlabel, ylabel, title=None, style='full'):
    """统一轴标签/标题字号：full 面向单图出版尺寸，compact 面向组合图小面板。"""
    if style == 'full':
        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        if title:
            ax.set_title(title, fontsize=13, fontweight='bold')
    else:
        ax.set_xlabel(xlabel, fontsize=8, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=8, fontweight='bold')


def _smoothed(times, values, config=None):
    window = config.savgol_window if config is not None else 21
    polyorder = config.savgol_polyorder if config is not None else 3
    if len(times) > 10:
        return get_smooth_curve(times, values, max_window=window, polyorder=polyorder)
    return times, values


def _panel_area_evolution(ax, times, areas, intensities, area_ci, colors,
                          title=None, style='full', config=None):
    """Fig1A：面积演化。CI 边界带 + 平滑轨迹；full 模式附加强度渐变背景、
    抽样散点与色条。无效帧（NaN）自然断线；渐变背景的颜色通道用背景
    水平 1.0 填充无效帧，仅作视觉中性色，不参与统计。"""
    x_smooth, area_smooth = _smoothed(times, areas, config)
    # 渐变背景专用副本：无效帧填 1.0（背景水平 → 中性色）
    disp_int = np.where(np.isfinite(intensities), intensities, 1.0)
    _, intensity_smooth = _smoothed(times, disp_int, config)

    i_min, i_max = float(np.min(disp_int)), float(np.max(disp_int))
    if style == 'full':
        # 背景渐变和 colorbar 必须使用同一 min/max，避免两套归一化不一致
        if i_max > i_min:
            intensity_norm = (intensity_smooth - i_min) / (i_max - i_min)
        else:
            intensity_norm = np.ones_like(intensity_smooth)
        cmap_stress = mcolors.LinearSegmentedColormap.from_list(
            'nature_stress',
            ['#000000', '#330000', '#660000', '#990000', '#CC0000',
             '#FF3333', '#FF6666', '#FF9999', '#FFCCCC', '#FFFFFF']
        )
        # 向量化创建渐变背景（修复：替代逐像素Python循环）
        finite_areas = _finite_values(areas)
        ymax = finite_areas.max() * 1.1 if finite_areas.size else 1.0
        img_extent = [times[0], times[-1], 0, ymax]
        gradient = np.zeros((100, len(x_smooth), 4))
        rgba_cols = cmap_stress(intensity_norm)  # shape: (N, 4)
        gradient[:, :, 0] = rgba_cols[np.newaxis, :, 0]
        gradient[:, :, 1] = rgba_cols[np.newaxis, :, 1]
        gradient[:, :, 2] = rgba_cols[np.newaxis, :, 2]
        gradient[:, :, 3] = 0.3
        ax.imshow(gradient, extent=img_extent, aspect='auto', origin='lower', alpha=0.4)

    # 置信带（掩膜 ±1 像素边界带，来自 process_stem_stack）
    ax.fill_between(times, areas - area_ci, areas + area_ci,
                    alpha=0.2, color=colors['blue'], edgecolor='none',
                    label='Area uncertainty (±1 px boundary band)')
    ax.plot(x_smooth, area_smooth, color=colors['blue'],
            linewidth=3 if style == 'full' else 2.5,
            label='Feature area', zorder=10, alpha=0.9)
    if style == 'full':
        ax.plot(x_smooth, area_smooth, color='white', linewidth=5, alpha=0.3, zorder=9)
        if len(times) > 50:
            stride = len(times) // 20
            ax.scatter(times[::stride], areas[::stride], color=colors['blue'],
                       s=25, alpha=0.6, edgecolors='white', linewidth=0.5, zorder=11)
        ax.legend(loc='best', fontsize=9)
        sm = plt.cm.ScalarMappable(cmap=cmap_stress,
                                   norm=plt.Normalize(vmin=i_min, vmax=i_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.01, shrink=0.95)
        cbar.set_label('Column intensity\n(black=low, white=high)',
                       fontsize=9, fontweight='bold')

    _style_axes(ax, 'Time (s)', 'Feature area (pixels)', title, style)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    _add_panel_label(ax, 'A', style)


def _panel_intensity(ax, times, intensities, confidence, colors,
                     title=None, style='full', config=None):
    """Fig1B：柱强度演化。逐帧 95% CI 带（N_eff 折减）+ 趋势线，
    单图/组合图共用同一数据口径。"""
    x_smooth, intensity_smooth = _smoothed(times, intensities, config)
    ax.fill_between(times, intensities - confidence, intensities + confidence,
                    color=colors['red'], alpha=0.15, edgecolor='none',
                    label='95% CI (N_eff corrected)')
    if style == 'full':
        ax.fill_between(x_smooth, 0, intensity_smooth,
                        color=colors['red'], alpha=0.4, edgecolor='none')
        ax.plot(x_smooth, intensity_smooth, color=colors['red'],
                linewidth=2.5, label='Intensity')
    else:
        ax.plot(times, intensities, color=colors['red'], linewidth=2)

    if len(times) > 2:
        trend = _linregress_safe(times, intensities)
        if trend is not None:
            trend_line = trend.slope * x_smooth + trend.intercept
            ax.plot(x_smooth, trend_line, color=colors['purple'], linestyle='--',
                    linewidth=1.8, alpha=0.8,
                    label=f'Trend (r={_fmt(trend.rvalue)})')

    _style_axes(ax, 'Time (s)', 'Column intensity ($I/I_0$)', title, style)
    if style == 'full':
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend(loc='upper right', fontsize=9)
    _add_panel_label(ax, 'B', style)


def _panel_correlation(ax, areas, intensities, times, colors, title=None, style='full'):
    """Fig1C：面积-强度相关。含退化保护：面积/强度恒定或含非有限值时跳过拟合
    （全零面积曾触发 polyfit LinAlgError）。"""
    sc = ax.scatter(areas, intensities, c=times, cmap='viridis',
                    s=40 if style == 'full' else 20, alpha=0.7,
                    edgecolors='white', linewidth=0.5)
    has_fit = False
    fa = _finite_values(areas)
    if (len(areas) > 2 and fa.size == len(areas)
            and _finite_values(intensities).size == len(intensities)
            and np.ptp(fa) > 0):
        poly = np.polyfit(areas, intensities, 1)
        x_range = np.linspace(np.min(areas), np.max(areas), 100)
        corr_r = corrcoef_or_nan(areas, intensities)
        ax.plot(x_range, np.poly1d(poly)(x_range), color=colors['orange'],
                linewidth=2.5 if style == 'full' else 1.5,
                label=f'Fit (r={_fmt(corr_r)})')
        has_fit = True
    elif len(areas) > 2:
        ax.text(0.5, 0.06, 'Constant/invalid feature values; fit skipped',
                transform=ax.transAxes, ha='center', fontsize=9, color='gray')

    _style_axes(ax, 'Feature area (pixels)', 'Column intensity ($I/I_0$)', title, style)
    ax.grid(True, alpha=0.2)
    if style == 'full':
        if has_fit:  # 退化分支没有带标签 artist，空 legend 会告警
            ax.legend(loc='best', fontsize=9)
        cbar = plt.colorbar(sc, ax=ax, pad=0.01, shrink=0.95)
        cbar.set_label('Time (s)', fontsize=9, fontweight='bold')
    _add_panel_label(ax, 'C', style)


def _panel_normalized(ax, times, areas, intensities, colors, title=None, style='full'):
    """Fig1D：归一化参数平行演化（除零与无效帧保护）。"""
    fa, fi = _finite_values(areas), _finite_values(intensities)
    max_area = fa.max() if fa.size and fa.max() > 0 else 1.0
    max_int = fi.max() if fi.size and fi.max() > 0 else 1.0
    y_area = areas / max_area * 100
    y_int = intensities / max_int * 100

    ax.plot(times, y_area, color=colors['blue'], linewidth=2.5, alpha=0.9, label='Area (normalized)')
    ax.plot(times, y_int, color=colors['red'], linewidth=2.5, alpha=0.9, label='Intensity (normalized)')
    ax.fill_between(times, 0, y_area, color=colors['blue'], alpha=0.15)
    ax.fill_between(times, 0, y_int, color=colors['red'], alpha=0.15)

    _style_axes(ax, 'Time (s)', 'Normalized value (%)', title, style)
    if style == 'full':
        ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.2)
    _add_panel_label(ax, 'D', style)


def _panel_phase(ax, times, areas, intensities, colors, title=None, style='full'):
    """Fig1E：相位图（d(Area)/dt vs d(Intensity)/dt）。"""
    if len(areas) > 5:
        area_derivative = np.gradient(areas, times)
        intensity_derivative = np.gradient(intensities, times)
        sc = ax.scatter(area_derivative[1:-1], intensity_derivative[1:-1],
                        c=times[1:-1], cmap='coolwarm',
                        s=30 if style == 'full' else 15, alpha=0.7,
                        edgecolors='black', linewidth=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        if style == 'full':
            # 箭头索引越界保护
            if len(area_derivative) > 10:
                mid_idx = len(area_derivative) // 2
                end_idx = min(mid_idx + 5, len(area_derivative) - 1)
                ax.annotate('', xy=(area_derivative[end_idx], intensity_derivative[end_idx]),
                            xytext=(area_derivative[mid_idx], intensity_derivative[mid_idx]),
                            arrowprops=dict(arrowstyle='->', color='black', lw=1.5, alpha=0.8))
            cbar = plt.colorbar(sc, ax=ax, pad=0.01, shrink=0.95)
            cbar.set_label('Time (s)', fontsize=9, fontweight='bold')

    _style_axes(ax, 'd(Area)/dt (px/s)', 'd(Intensity)/dt (a.u./s)', title, style)
    ax.grid(True, alpha=0.2)
    _add_panel_label(ax, 'E', style)


def _panel_stat_summary(ax, times, areas, intensities, style='full'):
    """Fig1F：统计摘要文本（单图与组合图共用同一数据口径，NaN 感知）。"""
    fa, fi = _finite_values(areas), _finite_values(intensities)
    if fa.size == 0 or fi.size == 0:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No valid measurements', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
        _add_panel_label(ax, 'F', style)
        return

    corr_coef = corrcoef_or_nan(areas, intensities)
    area_trend = _linregress_safe(times, areas)
    int_trend = _linregress_safe(times, intensities)
    frame_rate = safe_divide(1.0, times[1] - times[0], 0.0)
    area_change = _first_last_change(areas)
    joint = np.isfinite(areas) & np.isfinite(intensities)
    cov_value = safe_cov(areas[joint], intensities[joint]) if np.count_nonzero(joint) > 1 else float('nan')

    if area_trend is not None:
        area_trend_txt = (f"{area_trend.slope:.2f} px/s (R²={area_trend.rvalue ** 2:.3f})"
                          if np.isfinite(area_trend.rvalue) else f"{area_trend.slope:.2f} px/s (R²=n/a)")
    else:
        area_trend_txt = "n/a (insufficient valid frames)"
    if int_trend is not None:
        int_trend_txt = f"{int_trend.slope:.4f} /s (R²={_fmt(int_trend.rvalue ** 2)})"
    else:
        int_trend_txt = "n/a (insufficient valid frames)"

    stats_text = f"""
    QUANTITATIVE ANALYSIS

    Temporal Statistics:
    • Total observation: {times[-1]:.2f} s
    • Frame rate: {frame_rate:.0f} Hz
    • Acquisition: {len(times)} frames ({fa.size} valid)

    Feature Area:
    • Mean: {np.mean(fa):.0f} ± {np.std(fa):.0f} px
    • Range: ({np.min(fa):.0f} - {np.max(fa):.0f}) px
    • Trend: {area_trend_txt}
    • Relative Δ: {_fmt(area_change, '.1f', 'n/a')}%

    Column Intensity:
    • Mean: {np.mean(fi):.3f} ± {np.std(fi):.3f}
    • Trend: {int_trend_txt}

    Interdependence:
    • Correlation (ρ): {_fmt(corr_coef, fallback='undefined (constant/invalid)')}
    • Covariance: {_fmt(cov_value, '.2f')}

    [Area in pixels, Intensity as $I/I_0$ ratio]
    """

    ax.axis('off')
    compact = style != 'full'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=6 if compact else 9, fontfamily='Arial',
            verticalalignment='top',
            linespacing=1.2 if compact else 1.5,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA',
                      edgecolor='#DEE2E6', alpha=0.9))
    _add_panel_label(ax, 'F', style)


def _panel_power_spectrum(ax, times, areas, intensities, colors, title=None, style='full'):
    """Fig2A：Welch 功率谱密度（Hann 窗 + 线性去趋势，仅有效帧）。

    注：剔除无效帧后序列存在时间缺口时，Welch 仍按均匀 dt 采样近似；
    无效帧占比高时谱结果应视为粗略估计。
    """
    valid = np.isfinite(areas) & np.isfinite(intensities)
    has_data = np.count_nonzero(valid) > 10
    if has_data:
        dt = times[1] - times[0]
        freqs, psd_area = compute_power_spectrum(areas[valid], dt)
        _, psd_intensity = compute_power_spectrum(intensities[valid], dt)
        if len(freqs) > 1:
            main_freq = freqs[np.argmax(psd_area[1:]) + 1]
            psd_area_safe = np.maximum(psd_area[1:], 1e-20)
            psd_intensity_safe = np.maximum(psd_intensity[1:], 1e-20)
            lw = 2 if style == 'full' else 1.5
            ax.plot(freqs[1:], psd_area_safe, color=colors['blue'], linewidth=lw, label='Area')
            ax.plot(freqs[1:], psd_intensity_safe, color=colors['red'],
                    linewidth=lw, alpha=0.7, label='Intensity')
            if style == 'full':
                ax.axvline(x=main_freq, color='green', linestyle='--', alpha=0.5, linewidth=1)
                ax.text(main_freq * 1.1, np.max(psd_area_safe) * 0.8,
                        f'{main_freq:.2f} Hz', fontsize=8, color='green')
    _style_axes(ax, 'Frequency (Hz)', 'PSD (a.u.²/Hz)', title, style)
    if has_data:
        ax.set_xscale('log')
        ax.set_yscale('log')
        if style == 'full':
            ax.legend(fontsize=9)
    else:
        # 短序列（<=10 有效帧）时没有绘制任何曲线，画占位提示，不调用 legend。
        # 文本用 ASCII：Arial 不含 CJK 字形，中文会渲染成方块（已验证）。
        ax.text(0.5, 0.5, 'Insufficient data (>10 valid frames required)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10, color='gray')
    ax.grid(True, alpha=0.2)
    _add_panel_label(ax, 'A', style)


def _panel_rolling_corr(ax, times, areas, intensities, colors, title=None, style='full'):
    """Fig2B：滑动窗口相关（常量窗口 NaN 断线）。"""
    window_size = max(3, min(15, len(times) // 4))
    if window_size % 2 == 0:
        window_size += 1
    moving_corr = rolling_correlation(areas, intensities, window_size)
    half_w = window_size // 2
    mid_times = times[half_w:-half_w] if 0 < half_w < len(times) else times[:len(moving_corr)]
    plot_len = min(len(mid_times), len(moving_corr))
    ax.plot(mid_times[:plot_len], moving_corr[:plot_len],
            color=colors['purple'],
            linewidth=2.5 if style == 'full' else 1.5, label='Rolling correlation')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    _style_axes(ax, 'Time (s)', 'Correlation coefficient',
                f'Dynamic correlation (window={window_size})' if style == 'full' else None,
                style)
    if style == 'full':
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=9)
    _add_panel_label(ax, 'B', style)


def _panel_ecdf(ax, areas, colors, title=None, style='full'):
    """Fig2C：面积经验累积分布（full 模式附正态拟合参照，仅有效帧）。"""
    finite_areas = _finite_values(areas)
    if finite_areas.size == 0:
        ax.text(0.5, 0.5, 'No valid frames', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')
        _style_axes(ax, 'Feature area (pixels)', 'Cumulative probability', title, style)
        _add_panel_label(ax, 'C', style)
        return
    sorted_areas = np.sort(finite_areas)
    ecdf = np.arange(1, len(sorted_areas) + 1) / len(sorted_areas)
    ax.plot(sorted_areas, ecdf, color=colors['green'],
            linewidth=2.5 if style == 'full' else 1.5, label='Area ECDF')
    if style == 'full':
        try:
            mu, sigma = stats.norm.fit(finite_areas)
            if sigma > 0:
                x_norm = np.linspace(np.min(finite_areas), np.max(finite_areas), 100)
                ax.plot(x_norm, stats.norm.cdf(x_norm, mu, sigma), '--', color='gray',
                        linewidth=1.5, alpha=0.7, label='Normal fit')
        except Exception:
            pass
        ax.legend(fontsize=9)
    _style_axes(ax, 'Feature area (pixels)', 'Cumulative probability', title, style)
    ax.grid(True, alpha=0.2)
    _add_panel_label(ax, 'C', style)


def _panel_states(ax, times, intensities, colors, title=None):
    """Fig2D：基于强度四分位的三状态时间分类（仅有效帧；分位数是相对
    本序列的，不具绝对物理含义）。"""
    valid = np.isfinite(intensities)
    if np.count_nonzero(valid) < 4:
        ax.text(0.5, 0.5, 'Insufficient valid frames', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')
        _style_axes(ax, 'Time (s)', 'Feature state', title)
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['Low', 'Medium', 'High'])
        _add_panel_label(ax, 'D', 'full')
        return
    vi = intensities[valid]
    p25, p75 = np.percentile(vi, 25), np.percentile(vi, 75)
    high_mask = valid & (intensities > p75)
    med_mask = valid & (intensities >= p25) & (intensities <= p75)
    low_mask = valid & (intensities < p25)
    ax.scatter(times[high_mask], [3] * int(np.sum(high_mask)), color=colors['red'],
               s=30, alpha=0.6, label='High', marker='^')
    ax.scatter(times[med_mask], [2] * int(np.sum(med_mask)), color=colors['orange'],
               s=30, alpha=0.6, label='Medium', marker='o')
    ax.scatter(times[low_mask], [1] * int(np.sum(low_mask)), color=colors['blue'],
               s=30, alpha=0.6, label='Low', marker='v')
    _style_axes(ax, 'Time (s)', 'Feature state', title)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Low', 'Medium', 'High'])
    ax.grid(True, alpha=0.2, axis='x')
    _add_panel_label(ax, 'D', 'full')


def _panel_trajectory(ax, times, areas, intensities, colors, title=None, style='full'):
    """Fig2E：参数空间轨迹。"""
    ax.plot(areas, intensities, color='gray', linewidth=0.8, alpha=0.5)
    sc = ax.scatter(areas, intensities, c=times, cmap='plasma',
                    s=25 if style == 'full' else 10, alpha=0.7,
                    edgecolors='black', linewidth=0.3)
    if style == 'full' and len(areas) > 10:
        q = len(areas) // 4
        ax.annotate('', xy=(areas[q * 3], intensities[q * 3]),
                    xytext=(areas[q * 2], intensities[q * 2]),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    _style_axes(ax, 'Feature area (pixels)', 'Column intensity ($I/I_0$)', title, style)
    ax.grid(True, alpha=0.2)
    if style == 'full':
        cbar = plt.colorbar(sc, ax=ax, pad=0.01, shrink=0.95)
        cbar.set_label('Time (s)', fontsize=9, fontweight='bold')
    _add_panel_label(ax, 'E', style)


def _anomaly_score(areas, intensities):
    """启发式异常评分：面积/强度标准化差量的欧氏距离及其 mean+3SD 阈值。

    仅在首末两帧均有效的帧对上计算（NaN 帧对记 NaN，不参与阈值估计）。
    注意：score 近似 Rayleigh 型分布且未做多重比较校正，mean+3SD 不是
    严格的显著性阈值，仅作筛查用途（长序列会有预期内的假阳性）。
    """
    area_diff, intensity_diff = np.diff(areas), np.diff(intensities)
    valid = np.isfinite(area_diff) & np.isfinite(intensity_diff)
    score = np.full(valid.shape, np.nan)
    if np.count_nonzero(valid) >= 3:
        ad, idn_raw = area_diff[valid], intensity_diff[valid]
        a_std, i_std = np.std(ad), np.std(idn_raw)
        adn = (ad - np.mean(ad)) / (a_std if a_std > 1e-12 else 1.0)
        idn = (idn_raw - np.mean(idn_raw)) / (i_std if i_std > 1e-12 else 1.0)
        score[valid] = np.sqrt(adn ** 2 + idn ** 2)
    finite_score = score[np.isfinite(score)]
    if finite_score.size:
        threshold = float(np.mean(finite_score) + 3 * np.std(finite_score))
    else:
        threshold = float('nan')
    return score, threshold, np.where(score > threshold)[0]


def _panel_change(ax, times, areas, intensities, colors, title=None, style='full'):
    """Fig2F：启发式异常筛查（mean+3SD，非正式显著性检验）。"""
    score, threshold, flagged = _anomaly_score(areas, intensities)
    ax.plot(times[1:], score, color=colors['brown'],
            linewidth=2 if style == 'full' else 1.5, label='Anomaly score (heuristic)')
    if np.isfinite(threshold):
        ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.6, linewidth=1,
                   label=f'mean+3SD heuristic ({threshold:.1f})')
    if len(flagged) > 0:
        ax.scatter(times[flagged + 1], score[flagged], color='red',
                   s=40, zorder=5, label='Flagged points')
    _style_axes(ax, 'Time (s)', 'Change magnitude',
                'Heuristic anomaly screening' if style == 'full' else None, style)
    if style == 'full':
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.2)
    _add_panel_label(ax, 'F', style)


def create_individual_fig1_subfigures(times, areas, intensities, confidence, area_ci, colors,
                                      output_dir, title="Fe surface reconstruction", config=None):
    """创建Fig1的各个子图单独保存（A-F共6张 + 1张组合图）"""
    all_files = []

    # === 子图A：特征面积演化（主图） ===
    print("\n   Creating Fig1-A: Feature Area Evolution...")
    fig_a, ax_a = plt.subplots(figsize=(10, 6))
    _panel_area_evolution(ax_a, times, areas, intensities, area_ci, colors,
                          title=f'Evolution of feature area during {title}',
                          style='full', config=config)
    fig_a_path = os.path.join(output_dir, 'Fig1A_FeatureAreaEvolution.png')
    _save_figure(fig_a, fig_a_path)
    all_files.append(fig_a_path)
    print(f"     ✓ Fig1A saved: {os.path.basename(fig_a_path)}")

    # === 子图B：柱强度演化 ===
    print("   Creating Fig1-B: Column Intensity Evolution...")
    fig_b, ax_b = plt.subplots(figsize=(8, 5))
    _panel_intensity(ax_b, times, intensities, confidence, colors,
                     title='Column intensity evolution', style='full', config=config)
    fig_b_path = os.path.join(output_dir, 'Fig1B_ColumnIntensity.png')
    _save_figure(fig_b, fig_b_path)
    all_files.append(fig_b_path)
    print(f"     ✓ Fig1B saved: {os.path.basename(fig_b_path)}")

    # === 子图C：面积-强度相关性 ===
    print("   Creating Fig1-C: Area-Intensity Correlation...")
    fig_c, ax_c = plt.subplots(figsize=(8, 5))
    _panel_correlation(ax_c, areas, intensities, times, colors,
                       title='Area-intensity correlation', style='full')
    fig_c_path = os.path.join(output_dir, 'Fig1C_AreaIntensityCorrelation.png')
    _save_figure(fig_c, fig_c_path)
    all_files.append(fig_c_path)
    print(f"     ✓ Fig1C saved: {os.path.basename(fig_c_path)}")

    # === 子图D：归一化参数平行演化 ===
    print("   Creating Fig1-D: Parallel Evolution...")
    fig_d, ax_d = plt.subplots(figsize=(8, 5))
    _panel_normalized(ax_d, times, areas, intensities, colors,
                      title='Parallel evolution of feature parameters', style='full')
    # 文件名与内容对齐（旧名 Fig1D_3DPerspective.png 为历史遗留，实为归一化平行演化）
    fig_d_path = os.path.join(output_dir, 'Fig1D_ParallelEvolution.png')
    _save_figure(fig_d, fig_d_path)
    all_files.append(fig_d_path)
    print(f"     ✓ Fig1D saved: {os.path.basename(fig_d_path)}")

    # === 子图E：相位图分析 ===
    print("   Creating Fig1-E: Phase Portrait Analysis...")
    fig_e, ax_e = plt.subplots(figsize=(8, 5))
    _panel_phase(ax_e, times, areas, intensities, colors,
                 title='Phase portrait analysis', style='full')
    fig_e_path = os.path.join(output_dir, 'Fig1E_PhasePortrait.png')
    _save_figure(fig_e, fig_e_path)
    all_files.append(fig_e_path)
    print(f"     ✓ Fig1E saved: {os.path.basename(fig_e_path)}")

    # === 子图F：统计摘要 ===
    print("   Creating Fig1-F: Statistical Summary...")
    fig_f, ax_f = plt.subplots(figsize=(8, 5))
    _panel_stat_summary(ax_f, times, areas, intensities, style='full')
    fig_f_path = os.path.join(output_dir, 'Fig1F_StatisticalSummary.png')
    _save_figure(fig_f, fig_f_path)
    all_files.append(fig_f_path)
    print(f"     ✓ Fig1F saved: {os.path.basename(fig_f_path)}")

    # === 组合图（可选）===
    print("   Creating combined Fig1 (optional)...")
    try:
        combined_path = os.path.join(output_dir, 'Fig1_Combined.png')
        create_combined_fig1(times, areas, intensities, confidence, area_ci, colors,
                             combined_path, title, config=config)
        all_files.append(combined_path)
        print(f"     ✓ Combined Fig1 saved: {os.path.basename(combined_path)}")
    except Exception as e:
        print(f"     ⚠ Combined Fig1 skipped: {str(e)[:80]}")

    return all_files


def create_combined_fig1(times, areas, intensities, confidence, area_ci, colors, output_path,
                         title="Fe surface reconstruction", config=None):
    """创建Fig1的组合图（6子图概览；面板渲染与单图共用 _panel_*）"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1, 0.8],
                          width_ratios=[1, 1, 0.6], hspace=0.2, wspace=0.25)

    _panel_area_evolution(fig.add_subplot(gs[0:2, 0:2]), times, areas, intensities,
                          area_ci, colors, style='compact', config=config)
    _panel_intensity(fig.add_subplot(gs[0, 2]), times, intensities, confidence,
                     colors, style='compact', config=config)
    _panel_correlation(fig.add_subplot(gs[1, 2]), areas, intensities, times,
                       colors, style='compact')
    _panel_normalized(fig.add_subplot(gs[2, 0]), times, areas, intensities,
                      colors, style='compact')
    _panel_phase(fig.add_subplot(gs[2, 1]), times, areas, intensities,
                 colors, style='compact')
    _panel_stat_summary(fig.add_subplot(gs[2, 2]), times, areas, intensities, style='compact')

    fig.suptitle(f'Atomic-scale feature evolution during {title}', fontsize=14, fontweight='bold', y=0.98)
    _save_figure(fig, output_path, tight_rect=[0, 0.02, 1, 0.96])


def create_individual_fig2_subfigures(times, areas, intensities, colors, output_dir, config=None):
    """创建Fig2的各个子图单独保存（A-F共6张 + 1张组合图）"""
    all_files = []

    # === 子图A：功率谱分析 ===
    print("\n   Creating Fig2-A: Power Spectrum Analysis...")
    fig_a, ax_a = plt.subplots(figsize=(8, 5))
    _panel_power_spectrum(ax_a, times, areas, intensities, colors,
                          title='Power spectral density (Welch)', style='full')
    fig_a_path = os.path.join(output_dir, 'Fig2A_PowerSpectrum.png')
    _save_figure(fig_a, fig_a_path)
    all_files.append(fig_a_path)
    print(f"     ✓ Fig2A saved: {os.path.basename(fig_a_path)}")

    # === 子图B：移动窗口相关性 ===
    print("   Creating Fig2-B: Moving Window Analysis...")
    fig_b, ax_b = plt.subplots(figsize=(8, 5))
    _panel_rolling_corr(ax_b, times, areas, intensities, colors, style='full')
    fig_b_path = os.path.join(output_dir, 'Fig2B_MovingWindowCorrelation.png')
    _save_figure(fig_b, fig_b_path)
    all_files.append(fig_b_path)
    print(f"     ✓ Fig2B saved: {os.path.basename(fig_b_path)}")

    # === 子图C：累积分布 ===
    print("   Creating Fig2-C: Cumulative Distribution...")
    fig_c, ax_c = plt.subplots(figsize=(8, 5))
    _panel_ecdf(ax_c, areas, colors, title='Empirical cumulative distribution', style='full')
    fig_c_path = os.path.join(output_dir, 'Fig2C_CumulativeDistribution.png')
    _save_figure(fig_c, fig_c_path)
    all_files.append(fig_c_path)
    print(f"     ✓ Fig2C saved: {os.path.basename(fig_c_path)}")

    # === 子图D：状态转移分析 ===
    print("   Creating Fig2-D: State Transitions...")
    fig_d, ax_d = plt.subplots(figsize=(8, 5))
    _panel_states(ax_d, times, intensities, colors, title='Temporal state transitions')
    fig_d_path = os.path.join(output_dir, 'Fig2D_StateTransitions.png')
    _save_figure(fig_d, fig_d_path)
    all_files.append(fig_d_path)
    print(f"     ✓ Fig2D saved: {os.path.basename(fig_d_path)}")

    # === 子图E：参数空间轨迹 ===
    print("   Creating Fig2-E: Parameter Space Trajectory...")
    fig_e, ax_e = plt.subplots(figsize=(8, 5))
    _panel_trajectory(ax_e, times, areas, intensities, colors,
                      title='Parameter space trajectory', style='full')
    fig_e_path = os.path.join(output_dir, 'Fig2E_ParameterSpaceTrajectory.png')
    _save_figure(fig_e, fig_e_path)
    all_files.append(fig_e_path)
    print(f"     ✓ Fig2E saved: {os.path.basename(fig_e_path)}")

    # === 子图F：启发式异常筛查 ===
    print("   Creating Fig2-F: Heuristic Anomaly Screening...")
    fig_f, ax_f = plt.subplots(figsize=(8, 5))
    _panel_change(ax_f, times, areas, intensities, colors, style='full')
    fig_f_path = os.path.join(output_dir, 'Fig2F_ChangePointDetection.png')
    _save_figure(fig_f, fig_f_path)
    all_files.append(fig_f_path)
    print(f"     ✓ Fig2F saved: {os.path.basename(fig_f_path)}")

    print("   Creating combined Fig2 (optional)...")
    try:
        combined_path = os.path.join(output_dir, 'Fig2_Combined.png')
        create_combined_fig2(times, areas, intensities, colors, combined_path, config=config)
        all_files.append(combined_path)
        print(f"     ✓ Combined Fig2 saved: {os.path.basename(combined_path)}")
    except Exception as e:
        print(f"     ⚠ Combined Fig2 skipped: {str(e)[:80]}")
    return all_files


def create_combined_fig2(times, areas, intensities, colors, output_path, config=None):
    """创建Fig2的组合图（6子图概览；面板渲染与单图共用 _panel_*）"""
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.2, wspace=0.25)
    _panel_power_spectrum(fig.add_subplot(gs[0, 0]), times, areas, intensities,
                          colors, style='compact')
    _panel_rolling_corr(fig.add_subplot(gs[0, 1]), times, areas, intensities,
                        colors, style='compact')
    _panel_ecdf(fig.add_subplot(gs[0, 2]), areas, colors, style='compact')
    # D：时间-强度散点（组合图特有视图，单图 D 为状态分类）
    ax_d = fig.add_subplot(gs[1, 0])
    ax_d.scatter(times, intensities, c=intensities, cmap='RdYlBu_r', s=10, alpha=0.7)
    _style_axes(ax_d, 'Time', 'Intensity', style='compact')
    _add_panel_label(ax_d, 'D', 'compact')
    _panel_trajectory(fig.add_subplot(gs[1, 1]), times, areas, intensities, colors, style='compact')
    _panel_change(fig.add_subplot(gs[1, 2]), times, areas, intensities, colors, style='compact')
    fig.suptitle('Advanced analysis of feature dynamics', fontsize=14, fontweight='bold', y=0.98)
    _save_figure(fig, output_path, tight_rect=[0, 0.02, 1, 0.96])


def save_publication_data(times, areas, intensities, thresholds, area_ci, output_dir,
                          shared_threshold=False, intensity_ci=None, n_eff=None,
                          meta=None):
    """保存出版级别数据（CSV + 统计摘要Markdown）

    CSV 记录每帧实际使用的分割阈值、面积不确定度半宽、逐帧强度 95% CI
    半宽（空间自相关折减有效样本量）、N_eff 本身与帧有效性标记
    （Frame_Valid=0 ⇔ 该帧 Otsu 失败，测量值为 NaN），保证分割口径与
    测量不确定度可复现；摘要报告阈值稳定性与处理参数（meta）。
    统计仅基于有效帧（NaN 剔除）。
    """
    if intensity_ci is None:
        intensity_ci = np.full_like(areas, np.nan)
    if n_eff is None:
        n_eff = np.full_like(areas, np.nan)
    valid = np.isfinite(thresholds)

    data_path = os.path.join(output_dir, 'Publication_Data.csv')
    data = np.column_stack((times, areas, intensities, thresholds, area_ci,
                            intensity_ci, n_eff, valid.astype(int)))
    np.savetxt(data_path, data, delimiter=',',
               header=("Time_s,Feature_Area_px,Column_Intensity_AU,"
                       "Segmentation_Threshold,Area_Uncertainty_px,"
                       "Intensity_CI95_halfwidth,N_eff,Frame_Valid"),
               fmt=['%.6f'] * 7 + ['%d'], comments='# ', encoding='utf-8')

    stats_path = os.path.join(output_dir, 'Statistical_Summary.md')
    correlation = corrcoef_or_nan(areas, intensities)
    fa, fi = _finite_values(areas), _finite_values(intensities)
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("# Statistical Analysis Summary\n\n## Dataset Overview\n")
        f.write(f"- Total frames: {len(times)} (valid: {int(np.count_nonzero(valid))})\n"
                f"- Time span: {times[-1]:.2f} seconds\n")
        f.write(f"- Sampling rate: {safe_divide(1.0, times[1]-times[0], 0.0):.0f} Hz\n\n")
        if meta:
            f.write("## Processing Parameters\n")
            f.write(f"- Tool version: {meta.get('version')}\n")
            f.write(f"- Input: {os.path.basename(meta.get('input_file', ''))}\n")
            if meta.get('input_sha256'):
                f.write(f"- Input SHA-256: {meta.get('input_sha256')}\n")
            f.write(f"- Image shape: {meta.get('image_shape')} (dtype {meta.get('dtype')})\n")
            f.write(f"- Background I0: {meta.get('bg_intensity'):.4f} "
                    f"(median of first {meta.get('bg_frames')} frames, "
                    f"edge margin {meta.get('bg_margin_px')} px)\n")
            f.write(f"- Morphology: remove_small_objects(min_size={meta.get('min_size')}) "
                    f"+ binary_closing({meta.get('closing_structure')})\n")
            f.write(f"- Threshold mode: {meta.get('threshold_mode')}\n")
            if meta.get('bad_pixels_total'):
                f.write(f"- Bad pixels cleaned: {meta.get('bad_pixels_total')} "
                        f"in {meta.get('bad_pixel_frames')} frame(s)\n")
            if meta.get('invalid_frames'):
                f.write(f"- Invalid frames (Otsu failed, excluded from statistics): "
                        f"{meta.get('invalid_frames')}\n")
            if meta.get('empty_mask_frames'):
                f.write(f"- Empty-mask frames (area=0, intensity undefined): "
                        f"{meta.get('empty_mask_frames')}\n")
            f.write("\n")
        f.write("## Feature Area Statistics\n")
        if fa.size:
            f.write(f"- Mean ± SD: {np.mean(fa):.0f} ± {np.std(fa):.0f} px\n")
            f.write(f"- Range: {np.min(fa):.0f} - {np.max(fa):.0f} px\n")
            f.write(f"- Median (IQR): {np.median(fa):.0f} ({np.percentile(fa,25):.0f}-{np.percentile(fa,75):.0f}) px\n")
            if np.ptp(fa) > 1e-12:
                f.write(f"- Skewness: {stats.skew(fa):.3f}\n- Kurtosis: {stats.kurtosis(fa):.3f}\n\n")
            else:
                f.write("- Skewness/Kurtosis: n/a (constant series)\n\n")
        else:
            f.write("- n/a (no valid frames)\n\n")
        f.write("## Column Intensity Statistics\n")
        if fi.size:
            f.write(f"- Mean ± SD: {np.mean(fi):.4f} ± {np.std(fi):.4f} ($I/I_0$)\n")
            f.write(f"- Range: {np.min(fi):.4f} - {np.max(fi):.4f}\n")
            f.write(f"- Median (IQR): {np.median(fi):.4f} ({np.percentile(fi,25):.4f}-{np.percentile(fi,75):.4f})\n\n")
        else:
            f.write("- n/a (no valid frames)\n\n")
        f.write("## Temporal Trends\n")
        a_reg = _linregress_safe(times, areas)
        i_reg = _linregress_safe(times, intensities)
        if a_reg is not None:
            f.write(f"- Area trend: slope = {a_reg.slope:.3f} px/s, R² = {_fmt(a_reg.rvalue ** 2)}, "
                    f"p = {_fmt(a_reg.pvalue, '.4f')}\n")
        if i_reg is not None:
            f.write(f"- Intensity trend: slope = {i_reg.slope:.4f} /s, R² = {_fmt(i_reg.rvalue ** 2)}, "
                    f"p = {_fmt(i_reg.pvalue, '.4f')}\n")
        f.write("- 注: p 值基于独立观测假设；本序列存在时间自相关，"
                "应视为描述性指标而非严格的显著性检验。\n")
        f.write("- 注: 强度 CI（z·σ/√N_eff）为该区域空间均值的代表性区间，"
                "混合了原子柱间真实物理不均匀性与测量噪声；未传播 I₀ 估计误差，"
                "且阈值选择存在向高值的偏置（仅统计 > 阈值的像素）。\n\n")
        f.write("## Segmentation Threshold Statistics\n")
        finite_thresh = thresholds[np.isfinite(thresholds)]
        f.write(f"- Mode: {'shared (median of per-frame Otsu)' if shared_threshold else 'per-frame Otsu'}\n")
        if finite_thresh.size:
            t_sd = float(np.std(finite_thresh))
            t_mean = float(np.mean(finite_thresh))
            f.write(f"- Threshold mean ± SD: {t_mean:.4f} ± {t_sd:.4f}\n")
            f.write(f"- Threshold CV: {safe_divide(t_sd, t_mean, 0.0):.3f}\n")
            if not shared_threshold and t_sd > 0:
                f.write("- 注意：逐帧独立 Otsu 的阈值漂移会混入面积时间序列；"
                        "若阈值 CV 较大，建议改用 --shared-threshold 复算并对比两版结论。\n")
        else:
            f.write("- 无有效阈值（全部帧 Otsu 计算失败）\n")
        f.write("\n## Correlation Analysis\n")
        if np.isfinite(correlation):
            f.write(f"- Pearson r = {correlation:.3f} (valid frames only)\n")
        else:
            f.write("- Pearson r: undefined (constant or insufficient valid series)\n")
        if np.isfinite(correlation):
            if correlation > 0.7: f.write("- Strong positive correlation\n")
            elif correlation > 0.3: f.write("- Moderate positive correlation\n")
            elif correlation > -0.3: f.write("- Weak or no correlation\n")
            elif correlation > -0.7: f.write("- Moderate negative correlation\n")
            else: f.write("- Strong negative correlation\n")
    print(f"   ✓ Publication data saved")
    return [data_path, stats_path]


def save_processing_metadata(meta, output_dir, summary=None):
    """把处理元数据（含输入 SHA-256）导出为机器可读 JSON，供程序化比对
    两次运行的处理口径；qa_samples 含图像数组，不进 JSON。"""
    meta_clean = {k: v for k, v in meta.items() if k != "qa_samples"}
    payload = {
        "meta": meta_clean,
        "summary": summary or {},
    }
    path = os.path.join(output_dir, 'Processing_Metadata.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"   ✓ Processing metadata saved")
    return path


def create_segmentation_qa_figure(meta, times, areas, output_dir):
    """分割质检图：首/中/末帧灰度图 + 红色掩膜轮廓。

    出版审稿常规要求目检分割质量；此前所有输出均为派生量，没有任何一帧
    "原图 + 掩膜" 叠加可对照。无采样（全部帧分割失败）时返回 None。
    """
    samples = meta.get("qa_samples") or []
    if not samples:
        return None
    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    axes = np.atleast_1d(axes)
    for ax, (idx, img, mask, thresh) in zip(axes, samples):
        ax.imshow(img, cmap='gray')
        if mask.any():
            ax.contour(mask, levels=[0.5], colors='#FF3333', linewidths=1.5)
        else:
            ax.text(0.5, 0.5, 'empty mask', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='#FF3333')
        area_txt = f"{areas[idx]:.0f} px" if np.isfinite(areas[idx]) else "n/a"
        ax.set_title(f"Frame {idx} (t={times[idx]:.2f} s)\n"
                     f"area={area_txt}, threshold={thresh:.3f}",
                     fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('Segmentation quality check (red contour = feature mask)',
                 fontsize=12, fontweight='bold')
    path = os.path.join(output_dir, 'Segmentation_QA.png')
    _save_figure(fig, path)
    return path


def get_figure_description(idx, figure_type):
    """获取图表的描述信息（与当前实现保持一致；曾遗留 FFT/突变点等旧表述，
    会直接写入用户的 ANALYSIS_REPORT.md 造成文档自相矛盾）。"""
    if figure_type == 'fig1':
        d = ["Feature area evolution with gradient background", "Column intensity evolution with trend line",
             "Area-intensity correlation analysis", "Parallel evolution of normalized parameters",
             "Phase portrait analysis (derivative space)", "Statistical summary with key metrics"]
    else:
        d = ["Welch power spectral density (detrended, Hann window)", "Moving window correlation analysis",
             "Empirical cumulative distribution", "Temporal state transitions",
             "Parameter space trajectory", "Heuristic anomaly screening (mean+3SD, not a significance test)"]
    return d[idx - 1] if idx <= len(d) else "Analysis figure"


def main():
    """主程序入口。

    Returns
    -------
    int
        进程退出码：0 成功，1 失败（供批处理/流水线判断）。
    """
    print("\n" + "=" * 80)
    print("NATURE/SCIENCE STYLE — HAADF-STEM FEATURE EVOLUTION ANALYSIS")
    print("Atomic-scale column intensity tracking")
    print(f"Version: {__version__}")
    print("=" * 80)

    colors = get_color_palette()

    print("\n📁 FILE SELECTION")
    tif_path, time_interval, title, shared_threshold, config, output_override = get_file_and_params()
    if not tif_path:
        print("   ✗ No file selected. Exiting."); return 1
    if not os.path.exists(tif_path):
        print(f"   ✗ File not found: {tif_path}"); return 1
    # 风格设置放在参数解析之后：--dpi 需写入 savefig.dpi
    setup_nature_style(dpi=config.dpi)
    print(f"   ✓ Selected: {os.path.basename(tif_path)}")
    print(f"   ✓ Time interval: {time_interval}s, Title: {title}")
    print(f"   ✓ Threshold mode: {'shared (median of per-frame Otsu)' if shared_threshold else 'per-frame Otsu'}")
    print(f"   ✓ min_size={config.min_size}, dpi={config.dpi}")

    output_dir = get_output_directory(tif_path, override=output_override)
    # makedirs 连同前置校验一起走 prepare_output_directory：非法路径/无写权限/
    # 非空目录都以干净错误信息 + 退出码 1 结束，而不是裸 traceback 或静默覆盖
    prep_error = prepare_output_directory(output_dir)
    if prep_error is not None:
        print(f"   ✗ {prep_error}")
        return 1
    print(f"   ✓ Output directory: {os.path.basename(output_dir)}")

    try:
        print("\n🔬 DATA PROCESSING")
        meta = {}
        times, areas, intensities, confidence, area_ci, thresholds, n_eff_series, total_frames = \
            process_stem_stack(tif_path, time_interval,
                               shared_threshold=shared_threshold, meta=meta, config=config)
        if times is None:
            print("   ✗ Failed to process images. Exiting."); return 1
        try:
            meta["input_sha256"] = sha256_file(tif_path)
        except OSError as e:
            print(f"   ⚠ 无法计算输入文件 SHA-256: {e}")

        print("\n🎨 CREATING FIGURES")
        print("\n   === FIGURE 1 SUBFIGURES ===")
        fig1_files = create_individual_fig1_subfigures(times, areas, intensities, confidence, area_ci,
                                                       colors, output_dir, title, config=config)
        print("\n   === FIGURE 2 SUBFIGURES ===")
        fig2_files = create_individual_fig2_subfigures(times, areas, intensities, colors,
                                                       output_dir, config=config)

        print("\n   === SUPPLEMENTARY FIGURES ===")
        try:
            fig_s1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(times, areas, color='#1F77B4', linewidth=2.5, label='Feature area')
            ax1.fill_between(times, 0, areas, alpha=0.3, color='#1F77B4')
            ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Feature area (pixels)', fontsize=12, fontweight='bold', color='#1F77B4')
            ax1.tick_params(axis='y', labelcolor='#1F77B4'); ax1.grid(True, alpha=0.3)
            ax2 = ax1.twinx()
            ax2.plot(times, intensities, color='#D62728', linewidth=2.5, alpha=0.8, label='Column intensity')
            ax2.set_ylabel('Column intensity ($I/I_0$)', fontsize=12, fontweight='bold', color='#D62728')
            ax2.tick_params(axis='y', labelcolor='#D62728')
            l1, la1 = ax1.get_legend_handles_labels()
            l2, la2 = ax2.get_legend_handles_labels()
            ax1.legend(l1+l2, la1+la2, loc='upper right', fontsize=10)
            ax1.set_title('Dual-axis visualization of feature evolution', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'Supplementary_DualAxis.png'), bbox_inches='tight', facecolor='white')
            plt.close(fig_s1)
            print(f"     ✓ Supplementary 1 saved")
            # 箱线图拆成双子图：面积（数百 px）与强度（≈1–2）差两个数量级，
            # 画在同一线性轴上强度箱体只占轴高 ~0.05%，完全不可见（已验证）
            fig_s2, (axb_a, axb_i) = plt.subplots(1, 2, figsize=(10, 5))
            try:
                bp_a = axb_a.boxplot([_finite_values(areas)], tick_labels=['Area'],
                                     widths=0.4, patch_artist=True)
                bp_i = axb_i.boxplot([_finite_values(intensities)], tick_labels=['Intensity'],
                                     widths=0.4, patch_artist=True)
            except TypeError:
                # matplotlib < 3.9 的参数名为 labels
                bp_a = axb_a.boxplot([_finite_values(areas)], labels=['Area'],
                                     widths=0.4, patch_artist=True)
                bp_i = axb_i.boxplot([_finite_values(intensities)], labels=['Intensity'],
                                     widths=0.4, patch_artist=True)
            for patch, c in zip(bp_a['boxes'], ['#1F77B4']):
                patch.set_facecolor(c); patch.set_alpha(0.6)
            for patch, c in zip(bp_i['boxes'], ['#D62728']):
                patch.set_facecolor(c); patch.set_alpha(0.6)
            axb_a.set_ylabel('Feature area (pixels)', fontsize=12, fontweight='bold')
            axb_i.set_ylabel('Column intensity ($I/I_0$)', fontsize=12, fontweight='bold')
            fig_s2.suptitle('Distribution characteristics', fontsize=14, fontweight='bold')
            for axb in (axb_a, axb_i):
                axb.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'Supplementary_Boxplot.png'), bbox_inches='tight', facecolor='white')
            plt.close(fig_s2)
            print(f"     ✓ Supplementary 2 saved")
        except Exception as e:
            print(f"     ⚠ Supplementary figures skipped: {str(e)[:80]}")

        print("\n💾 SAVING PUBLICATION DATA")
        data_files = save_publication_data(times, areas, intensities, thresholds, area_ci,
                                           output_dir, shared_threshold=shared_threshold,
                                           intensity_ci=confidence, n_eff=n_eff_series,
                                           meta=meta)

        print("\n🔍 SEGMENTATION QUALITY CHECK")
        qa_path = None
        try:
            qa_path = create_segmentation_qa_figure(meta, times, areas, output_dir)
            if qa_path:
                print(f"     ✓ Segmentation QA saved: {os.path.basename(qa_path)}")
            else:
                print("     ⚠ 无有效分割帧，质检图跳过")
        except Exception as e:
            print(f"     ⚠ Segmentation QA skipped: {str(e)[:80]}")

        print("\n📋 GENERATING ANALYSIS REPORT")
        corr_r = corrcoef_or_nan(areas, intensities)
        area_change = _first_last_change(areas)
        n_invalid = int(np.count_nonzero(~np.isfinite(thresholds)))
        fa = _finite_values(areas)
        summary = {
            "n_frames": int(total_frames),
            "valid_frames": int(total_frames - n_invalid),
            "mean_area_px": float(np.mean(fa)) if fa.size else None,
            "sd_area_px": float(np.std(fa)) if fa.size else None,
            "area_change_percent": area_change,
            "pearson_r": corr_r if np.isfinite(corr_r) else None,
            "invalid_frames": n_invalid,
            "empty_mask_frames": int(meta.get("empty_mask_frames", 0)),
        }
        try:
            save_processing_metadata(meta, output_dir, summary=summary)
        except Exception as e:
            print(f"   ⚠ Processing metadata export failed: {e}")

        report_path = os.path.join(output_dir, 'ANALYSIS_REPORT.md')
        corr_txt = f"{corr_r:.3f}" if np.isfinite(corr_r) else "undefined (constant/invalid series)"
        change_txt = f"{area_change:+.1f}%" if area_change is not None else "n/a"
        if fa.size:
            mean_txt = f"mean {np.mean(fa):.0f} px ± {np.std(fa):.0f}"
        else:
            mean_txt = "no valid area measurements"
        with open(report_path, 'w', encoding='utf-8') as f:
            from datetime import datetime
            f.write(f"# HAADF-STEM Feature Analysis Report: {os.path.basename(tif_path)}\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Executive Summary\nOver {total_frames} frames ({times[-1]:.2f} s), feature area showed "
                    f"a {change_txt} change with {mean_txt}. "
                    f"Area-intensity correlation: r = {corr_txt}.\n\n")
            f.write("## Generated Files\n### Figure 1\n")
            for i, file in enumerate(fig1_files, 1):
                f.write(f"- `{os.path.basename(file)}`: {get_figure_description(i, 'fig1')}\n")
            f.write("\n### Figure 2\n")
            for i, file in enumerate(fig2_files, 1):
                f.write(f"- `{os.path.basename(file)}`: {get_figure_description(i, 'fig2')}\n")
            f.write("\n### Quality & Data\n")
            if qa_path:
                f.write(f"- `{os.path.basename(qa_path)}`: First/middle/last frame with segmentation mask overlay "
                        "(visual check of segmentation quality)\n")
            f.write("- `Publication_Data.csv`: Per-frame time, area, intensity, threshold, "
                    "area uncertainty, intensity CI, N_eff, Frame_Valid\n")
            f.write("- `Statistical_Summary.md`: Processing parameters and statistics\n")
            f.write("- `Processing_Metadata.json`: Machine-readable metadata incl. input SHA-256\n")
            f.write(f"\n## Technical Details\n- Tool version: {__version__}\n"
                    f"- Method: Otsu threshold "
                    f"({'shared (median of per-frame Otsu)' if shared_threshold else 'per-frame'})"
                    f" + multi-frame background normalization"
                    f" (I0 = {meta.get('bg_intensity', float('nan')):.4f})\n"
                    f"- Time interval: {time_interval} s/frame\n- Style: Nature/Science publication\n")
            if n_invalid:
                f.write(f"- 注: {n_invalid} 帧分割失败（Otsu 无法计算），已记为无效帧（NaN / "
                        "Frame_Valid=0）并从统计中剔除。\n")
            f.write("- 注: Fig2F 异常评分（mean+3SD）为启发式筛查而非正式显著性检验；"
                    "趋势 p 值未校正时间自相关。两者均应视为描述性指标。\n")
        print(f"   ✓ Analysis report saved")

        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\n📁 Output: {output_dir}")
        print(f"\n🔑 KEY METRICS:")
        print(f"   • Frames: {total_frames}, Time: {times[-1]:.2f}s")
        if fa.size:
            print(f"   • Mean area: {np.mean(fa):.0f} px, Change: {change_txt}")
        print(f"   • Correlation: r = {corr_txt}")
        print("=" * 80 + "\n")
        open_output_directory(output_dir)  # 跨平台打开输出目录（linux 移植）
        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
