# -*- coding: utf-8 -*-
"""
TIFF 漂移矫正工具 v7 - GUI 层

算法/IO/并发实现已拆分至 drift_core.py，本文件仅包含 tkinter 界面与交互逻辑。

v7 合并摘要:
- 保留 v5.2：后台线程/可取消、文件切换锁定、审计报告、裁剪与覆盖确认、
  参数校验、预览性能缓存、漂移曲线帧标记
- 合并 v6.1：纯平移估计、nfeatures 默认 5000、GUI 无遗留死代码
- v7.0.1：恢复 v5.2 的 RANSAC 空间内点筛选，修复跳帧检测大量失败
- v7.0.3：恢复 v5.2 的可靠帧对累计逻辑，禁止未检测的零位移假保存
- v7.0.4：C/Z 轴一律要求确认、UI 事件泵加固、批处理覆盖与目录防护、
  预览百分位显示、参数校验收敛到核心层
- v7.1.0：以 v7.0.4 修复内容发布为新版本号（v7.0.4 与 v7.1.0 为同一次
  修复发布，正式对外版本号为 7.1.0）
- v7.1.1：批处理启动前统一确认 C/Z 轴文件按时间序列处理，修复"只拒不放"
  的流程死胡同
- v7.2.0：默认检测/匹配参数与 ImageJ Linear Stack Alignment 对齐
  （ratio 0.92、RANSAC 25 px、内点比例 0.05），质量门控默认放宽到
  非约束水平
- v7.3.0：加载前元数据预检与内存确认、逐帧位移表导出、报告版本与逐帧对
  质量明细、FLANN 大规模匹配、预览拖拽平移、插值段曲线标示、界面配置
  持久化
"""

import os
import sys
import glob
import json
import math
import time
import logging
import queue
from pathlib import Path
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# matplotlib 中文字体（漂移曲线坐标轴标签）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from drift_core import (
    TiffIO, DriftDetector, DriftCorrector, WorkerThread,
    correct_and_save, batch_process, validate_sift_params, DEFAULT_SIFT_PARAMS,
    __version__, total_physical_memory, write_shift_table,
)

logger = logging.getLogger(__name__)

# ---- GUI 常量 ----
ZOOM_MIN, ZOOM_MAX, ZOOM_FACTOR = 0.1, 5.0, 1.25
WINDOW_SIZE = "1300x850"
WINDOW_MIN = (1000, 700)
# 标题版本由 drift_core.__version__ 派生，与 pyproject/打包资源/审计报告同源。
TITLE = f"TIFF 漂移矫正工具 v{__version__.rsplit('.', 1)[0]}（ImageJ 参数对齐版）"
# 解码体积超过物理内存该比例时要求人工确认（批处理核心层另有 0.9 硬限）。
MEMORY_WARN_RATIO = 0.4
# 界面配置持久化文件与参数变量名（新增参数自动纳入持久化）
_CONFIG_FILENAME = ".drift_correction_v7.json"
_PARAM_VAR_NAMES = (
    "param_nfeatures", "param_noctave_layers", "param_contrast_threshold",
    "param_edge_threshold", "param_sigma", "param_ratio_threshold",
    "param_ransac_threshold", "param_min_matches", "param_skip_interval",
    "param_min_inliers", "param_min_inlier_ratio", "param_max_pair_residual",
    "param_min_valid_pair_ratio", "param_max_interpolation_gap",
)


def _config_path() -> Path:
    return Path.home() / _CONFIG_FILENAME


def _inspect_stack_quiet(filepath, **_ignored):
    """供 WorkerThread 调用：线程包装器会自动注入进度/取消关键字，而
    inspect_stack 只做目录级扫描、不需要它们。"""
    return TiffIO.inspect_stack(filepath)


def _representative_preview_frame(shifts_x, shifts_y) -> int:
    """选择累计漂移最大的帧，让检测后的矫正预览具有可见差异。"""
    sx = np.asarray(shifts_x, dtype=np.float64)
    sy = np.asarray(shifts_y, dtype=np.float64)
    if sx.ndim != 1 or sy.shape != sx.shape or sx.size == 0:
        raise ValueError("预览偏移数组必须为非空、等长的一维数组")
    if not np.isfinite(sx).all() or not np.isfinite(sy).all():
        raise ValueError("预览偏移数组必须全部为有限数")
    return int(np.argmax(np.hypot(sx, sy)))


def _preview_contrast_range(sampled_frames) -> tuple[float, float]:
    """预览显示范围：与核心检测一致的 P1--P99.5 稳健百分位。

    min/max 会被少量热像素拉爆，uint16 数据的预览会整体发黑；
    内部按与 _global_normalization 相同的规则降采样以限制计算量。
    """
    samples: list[np.ndarray] = []
    for frame in sampled_frames:
        image = np.asarray(frame)
        stride = max(1, int(math.sqrt(image.size / 65_536)))
        values = image[::stride, ::stride].reshape(-1)
        values = values[np.isfinite(values)]
        if values.size:
            samples.append(values.astype(np.float64, copy=False))
    if not samples:
        return 0.0, 1.0
    values = np.concatenate(samples)
    low, high = np.percentile(values, (1.0, 99.5))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        base = float(low) if np.isfinite(low) else 0.0
        return base, base + 1.0
    return float(low), float(high)


def _package_health_check() -> None:
    """供最终 EXE 无界面冒烟测试使用；不读写用户文件。"""
    frame = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
    corrected = DriftCorrector.correct_single_frame(frame, 3.0, -2.0)
    if corrected.shape != frame.shape or corrected.dtype != frame.dtype:
        raise RuntimeError("打包健康检查失败：矫正帧结构异常")
    if np.array_equal(corrected, frame):
        raise RuntimeError("打包健康检查失败：非零平移没有改变像素")


# ============================================================
# GUI 主界面
# ============================================================

class DriftCorrectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title(TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.minsize(*WINDOW_MIN)

        # 状态
        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.file_type = tk.StringVar(value="*.tif")
        self.current_frame_idx = 0
        self.frames = []
        self.shifts = None       # 水平偏移
        self.shifts_y = None     # 垂直偏移
        self.auto_shifts = None
        self.auto_shifts_y = None
        self.detection_result = None
        self.manual_dirty = False
        self.file_list = []
        self.current_file_idx = -1
        self.current_filepath = None
        self.document_id = 0
        self.meta = None
        self.image_markers = []   # 图像上的点标记 [(x, y)]
        self.markers = []         # 漂移曲线上的帧标记 [frame_idx]
        self.active_marker = -1
        self.marker_mode = tk.BooleanVar(value=False)
        self.manual_mode = tk.BooleanVar(value=False)
        self.manual_step = tk.StringVar(value="0.5")
        self.zoom_level = 1.0
        self.worker = None        # 后台工作线程
        self._busy = False        # 后台任务运行标志（锁定文件切换等）
        self._closing = False
        self._close_deadline = 0.0   # 关闭窗口时等待 worker 结束的绝对时限
        self._close_force_prompted = False
        self._task_start_time = 0.0
        self._ui_queue = queue.Queue()

        # 预览渲染缓存（P2 性能）
        self._img_artists = {'prev': None, 'curr': None, 'corr': None}
        # 文本输入控件引用集合：焦点判断优先用引用，而不是只按 winfo_class 字符串兜底。
        self._text_input_widgets = set()
        self._marker_artists = []
        self._corr_cache_key = None
        self._corr_cache_img = None
        self._preview_vmin = 0.0
        self._preview_vmax = 255.0
        # 预览平移（缩放后拖拽查看非中心区域；None 表示回到图像中心）
        self._pan_cx = None
        self._pan_cy = None
        self._pan_state = None

        self._build_ui()
        self._bind_shortcuts()
        self._restore_config()
        self.root.after(30, self._drain_ui_queue)

    # --------------------------------------------------------
    # 快捷键（P0 修复：输入框获得焦点时快捷键失效）
    # --------------------------------------------------------
    def _register_text_input(self, widget):
        """登记文本输入控件；_text_input_focused 优先按引用集合判断焦点。"""
        self._text_input_widgets.add(widget)

    def _text_input_focused(self):
        """焦点在文本输入控件上时，全局快捷键必须让位给文本编辑。"""
        w = self.root.focus_get()
        if w is None:
            return False
        if w in self._text_input_widgets:
            return True
        # 兜底：覆盖未显式登记的 Entry/Text/Combobox 等标准输入控件。
        return isinstance(w, (tk.Entry, ttk.Entry, tk.Text, ttk.Combobox))

    def _bind_shortcuts(self):
        def guard(handler):
            def wrapper(event):
                if self._text_input_focused():
                    return
                handler()
            return wrapper

        self.root.bind('<Left>', guard(lambda: self._on_arrow('left')))
        self.root.bind('<Right>', guard(lambda: self._on_arrow('right')))
        self.root.bind('<Up>', guard(lambda: self._on_arrow('up')))
        self.root.bind('<Down>', guard(lambda: self._on_arrow('down')))
        self.root.bind('<Prior>', guard(self._prev_frame))
        self.root.bind('<Next>', guard(self._next_frame))
        self.root.bind('<Control-s>', guard(self._execute_correction))
        self.root.bind('<Delete>', guard(self._delete_last_image_marker))
        self.root.bind('<Escape>', guard(self._clear_all_markers))
        self.root.bind('<plus>', guard(lambda: self._set_zoom(self.zoom_level * ZOOM_FACTOR)))
        self.root.bind('<equal>', guard(lambda: self._set_zoom(self.zoom_level * ZOOM_FACTOR)))
        self.root.bind('<KP_Add>', guard(lambda: self._set_zoom(self.zoom_level * ZOOM_FACTOR)))
        self.root.bind('<minus>', guard(lambda: self._set_zoom(self.zoom_level / ZOOM_FACTOR)))
        self.root.bind('<KP_Subtract>', guard(lambda: self._set_zoom(self.zoom_level / ZOOM_FACTOR)))
        self.root.bind('<0>', guard(lambda: self._set_zoom(1.0)))
        self.root.bind('<KP_0>', guard(lambda: self._set_zoom(1.0)))

    def _on_arrow(self, direction):
        if self._busy:
            return
        if self.manual_mode.get():
            step = self._get_manual_step()
            if direction == 'left':
                self._manual_adjust(-step, 0.0)
            elif direction == 'right':
                self._manual_adjust(step, 0.0)
            elif direction == 'up':
                self._manual_adjust(0.0, -step)
            elif direction == 'down':
                self._manual_adjust(0.0, step)
        else:
            if direction == 'left':
                self._prev_frame()
            elif direction == 'right':
                self._next_frame()
            elif direction == 'up':
                self._adjust_shift(0.5)
            elif direction == 'down':
                self._adjust_shift(-0.5)

    def _get_manual_step(self):
        try:
            value = abs(float(self.manual_step.get()))
            return value if np.isfinite(value) else 0.5
        except ValueError:
            return 0.5

    def _manual_adjust(self, dx, dy):
        """手动模式下调整当前帧偏移，变化量联动后续帧"""
        if self._busy or not self.frames:
            return
        if self.shifts is None:
            self.shifts = np.zeros(len(self.frames), dtype=np.float32)
        if self.shifts_y is None:
            self.shifts_y = np.zeros(len(self.frames), dtype=np.float32)
        idx = self.current_frame_idx
        self.shifts[idx] += dx
        self.shifts_y[idx] += dy
        if dx != 0.0:
            self.shifts[idx + 1:] += dx
        if dy != 0.0:
            self.shifts_y[idx + 1:] += dy
        if dx != 0.0 or dy != 0.0:
            self.manual_dirty = True
        self.shift_var.set(f"{self.shifts[idx]:.2f}")
        self.status_var.set(
            f"手动调整 帧{idx}: dx={self.shifts[idx]:.2f}px, dy={self.shifts_y[idx]:.2f}px | "
            f"后续帧已联动 | 方向键继续调整, PageUp/Down翻帧")
        self._update_preview()
        self._update_curve()

    # --------------------------------------------------------
    # UI 构建
    # --------------------------------------------------------
    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding=5)
        main_frame.pack(fill=tk.BOTH, expand=True)
        self._build_file_section(main_frame)
        self._build_params_section(main_frame)
        self._build_preview_section(main_frame)
        self._build_control_section(main_frame)
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(3, 0))
        self.status_var = tk.StringVar(
            value="就绪 | ←→翻帧 ↑↓调偏移(联动后续帧) 滚轮/+/-缩放 0重置缩放 Ctrl+S保存 Del删标记")
        ttk.Label(status_frame, textvariable=self.status_var,
                  relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X)

    def _build_file_section(self, parent):
        file_frame = ttk.LabelFrame(parent, text="文件选择 (支持拖拽文件到窗口)", padding=5)
        file_frame.pack(fill=tk.X, pady=(0, 3))
        row1 = ttk.Frame(file_frame)
        row1.pack(fill=tk.X, pady=1)
        ttk.Button(row1, text="选择输入文件夹", command=self._select_input_folder).pack(side=tk.LEFT)
        input_entry = ttk.Entry(row1, textvariable=self.input_folder, width=35)
        input_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self._register_text_input(input_entry)
        ttk.Button(row1, text="选择输出文件夹", command=self._select_output_folder).pack(
            side=tk.LEFT, padx=(15, 0))
        output_entry = ttk.Entry(row1, textvariable=self.output_folder, width=35)
        output_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self._register_text_input(output_entry)
        row2 = ttk.Frame(file_frame)
        row2.pack(fill=tk.X, pady=1)
        ttk.Label(row2, text="类型:").pack(side=tk.LEFT)
        self.file_type_combo = ttk.Combobox(row2, textvariable=self.file_type, width=8,
                                            values=["*.tif", "*.tiff"])
        self.file_type_combo.pack(side=tk.LEFT, padx=3)
        self._register_text_input(self.file_type_combo)
        self.btn_refresh = ttk.Button(row2, text="刷新", command=self._load_file_list)
        self.btn_refresh.pack(side=tk.LEFT, padx=5)
        self.file_listbox = tk.Listbox(row2, height=2, selectmode=tk.SINGLE, width=60)
        self.file_listbox.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.file_listbox.bind("<<ListboxSelect>>", self._on_file_select)
        try:
            from tkinterdnd2 import DND_FILES
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self._on_drop)
        except Exception as e:
            # 未安装 tkinterdnd2，或 root 不是 TkinterDnD.Tk 实例
            logger.info(f"拖拽功能不可用: {e}")

    def _on_drop(self, event):
        if self._busy:
            return  # 后台任务运行期间禁止切换文件
        try:
            paths = self.root.tk.splitlist(event.data)
        except tk.TclError:
            paths = [event.data.strip().strip('{}')]
        for path in paths:
            path = path.strip('{}')
            if os.path.isfile(path) and path.lower().endswith(('.tif', '.tiff')):
                self.input_folder.set(os.path.dirname(path))
                self._load_file_list()
                for i, f in enumerate(self.file_list):
                    if os.path.normcase(os.path.abspath(f)) == os.path.normcase(os.path.abspath(path)):
                        self.file_listbox.selection_set(i)
                        self._load_file(f, i)
                        break
                return
            elif os.path.isdir(path):
                self.input_folder.set(path)
                self._load_file_list()
                return

    def _build_params_section(self, parent):
        params_frame = ttk.LabelFrame(parent, text="SIFT 参数 (检测与匹配默认与ImageJ Linear Stack Alignment一致)", padding=3)
        params_frame.pack(fill=tk.X, pady=(0, 3))
        row1 = ttk.Frame(params_frame)
        row1.pack(fill=tk.X)
        d = DEFAULT_SIFT_PARAMS
        self.param_nfeatures = tk.StringVar(value=str(d['nfeatures']))
        self.param_noctave_layers = tk.StringVar(value=str(d['noctave_layers']))
        self.param_contrast_threshold = tk.StringVar(value=str(d['contrast_threshold']))
        self.param_edge_threshold = tk.StringVar(value=str(d['edge_threshold']))
        self.param_sigma = tk.StringVar(value=str(d['sigma']))
        self.param_ratio_threshold = tk.StringVar(value=str(d['ratio_threshold']))
        self.param_ransac_threshold = tk.StringVar(value=str(d['ransac_threshold']))
        self.param_min_matches = tk.StringVar(value=str(d['min_matches']))
        self.param_skip_interval = tk.StringVar(value=str(d['skip_interval']))
        params = [
            ("特征点:", self.param_nfeatures, 5), ("Octave层:", self.param_noctave_layers, 3),
            ("对比度:", self.param_contrast_threshold, 5), ("边缘:", self.param_edge_threshold, 4),
            ("Sigma:", self.param_sigma, 4), ("Ratio:", self.param_ratio_threshold, 4),
            ("RANSAC px:", self.param_ransac_threshold, 4), ("最少点:", self.param_min_matches, 3),
            ("跳帧:", self.param_skip_interval, 3),
        ]
        for label, var, w in params:
            ttk.Label(row1, text=label).pack(side=tk.LEFT)
            entry = ttk.Entry(row1, textvariable=var, width=w)
            entry.pack(side=tk.LEFT, padx=(1, 6))
            self._register_text_input(entry)
        ttk.Button(row1, text="默认", command=self._reset_params, width=4).pack(side=tk.LEFT, padx=5)

        # 质量门控：决定检测是成功、降级插值还是显式失败，必须可调
        row2 = ttk.Frame(params_frame)
        row2.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(row2, text="质量门控:", foreground="gray").pack(side=tk.LEFT, padx=(0, 4))
        self.param_min_inliers = tk.StringVar(value=str(d['min_inliers']))
        self.param_min_inlier_ratio = tk.StringVar(value=str(d['min_inlier_ratio']))
        self.param_max_pair_residual = tk.StringVar(value=str(d['max_pair_residual']))
        self.param_min_valid_pair_ratio = tk.StringVar(value=str(d['min_valid_pair_ratio']))
        self.param_max_interpolation_gap = tk.StringVar(value=str(d['max_interpolation_gap']))
        quality_params = [
            ("最少内点:", self.param_min_inliers, 4),
            ("内点比例:", self.param_min_inlier_ratio, 5),
            ("最大残差px:", self.param_max_pair_residual, 5),
            ("有效帧对比例:", self.param_min_valid_pair_ratio, 5),
            ("最大插值间隙:", self.param_max_interpolation_gap, 6),
        ]
        for label, var, w in quality_params:
            ttk.Label(row2, text=label).pack(side=tk.LEFT)
            entry = ttk.Entry(row2, textvariable=var, width=w)
            entry.pack(side=tk.LEFT, padx=(1, 6))
            self._register_text_input(entry)
        ttk.Label(row2, text="放宽这些阈值可让困难数据通过检测，但结果可靠性下降",
                  foreground="gray").pack(side=tk.LEFT, padx=8)

    def _reset_params(self):
        d = DEFAULT_SIFT_PARAMS
        self.param_nfeatures.set(str(d['nfeatures']))
        self.param_noctave_layers.set(str(d['noctave_layers']))
        self.param_contrast_threshold.set(str(d['contrast_threshold']))
        self.param_edge_threshold.set(str(d['edge_threshold']))
        self.param_sigma.set(str(d['sigma']))
        self.param_ratio_threshold.set(str(d['ratio_threshold']))
        self.param_ransac_threshold.set(str(d['ransac_threshold']))
        self.param_min_matches.set(str(d['min_matches']))
        self.param_skip_interval.set(str(d['skip_interval']))
        self.param_min_inliers.set(str(d['min_inliers']))
        self.param_min_inlier_ratio.set(str(d['min_inlier_ratio']))
        self.param_max_pair_residual.set(str(d['max_pair_residual']))
        self.param_min_valid_pair_ratio.set(str(d['min_valid_pair_ratio']))
        self.param_max_interpolation_gap.set(str(d['max_interpolation_gap']))

    def _get_sift_params(self):
        """收集界面参数并返回数值 dict。

        本方法只负责"字符串 -> 数值"的解析（失败时提示具体字段）；
        范围校验统一委托 drift_core.validate_sift_params，保证 GUI 与
        核心算法使用同一套阈值规则。
        """
        fields = [
            ('nfeatures', self.param_nfeatures, int),
            ('noctave_layers', self.param_noctave_layers, int),
            ('contrast_threshold', self.param_contrast_threshold, float),
            ('edge_threshold', self.param_edge_threshold, float),
            ('sigma', self.param_sigma, float),
            ('ratio_threshold', self.param_ratio_threshold, float),
            ('ransac_threshold', self.param_ransac_threshold, float),
            ('min_matches', self.param_min_matches, int),
            ('min_inliers', self.param_min_inliers, int),
            ('min_inlier_ratio', self.param_min_inlier_ratio, float),
            ('max_pair_residual', self.param_max_pair_residual, float),
            ('min_valid_pair_ratio', self.param_min_valid_pair_ratio, float),
            ('max_interpolation_gap', self.param_max_interpolation_gap, int),
        ]
        params = {}
        for key, var, cast in fields:
            text = var.get().strip()
            try:
                params[key] = cast(text)
            except ValueError:
                messagebox.showwarning("参数错误", f"参数无效: {key} = {text!r}")
                return None
        try:
            validate_sift_params(params)
        except ValueError as exc:
            messagebox.showwarning("参数错误", f"参数无效:\n{exc}")
            return None
        return params

    def _get_skip_interval(self, for_batch: bool = False):
        try:
            val = int(self.param_skip_interval.get())
        except ValueError:
            messagebox.showwarning("参数错误", "跳帧间隔必须为整数")
            return None
        if val < 1:
            messagebox.showwarning("参数错误", "跳帧间隔必须 >= 1")
            return None
        # 帧数上限只约束当前文件的单次检测；批处理各文件帧数不同，
        # 由核心层按各文件自行退化处理（跳帧超过帧数时为首尾单对匹配）。
        if not for_batch and self.frames and val >= len(self.frames):
            messagebox.showwarning("参数错误", f"跳帧间隔({val})不能大于等于帧数({len(self.frames)})")
            return None
        return val

    def _build_preview_section(self, parent):
        preview_frame = ttk.LabelFrame(parent, text="预览 (左:上一帧 | 中:当前帧 | 右:矫正后)", padding=3)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 3))
        self.fig = Figure(figsize=(12, 3.2), dpi=100)
        self.ax_prev = self.fig.add_subplot(131)
        self.ax_curr = self.fig.add_subplot(132)
        self.ax_corr = self.fig.add_subplot(133)
        for ax in [self.ax_prev, self.ax_curr, self.ax_corr]:
            ax.axis('off')
        self.fig.tight_layout(pad=0.5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=preview_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self._on_image_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_image_motion)
        self.canvas.mpl_connect('button_release_event', self._on_image_release)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        # 漂移曲线（点击可添加/移除帧标记）
        curve_frame = ttk.LabelFrame(preview_frame, text="漂移曲线 (点击添加/移除帧标记)", padding=2)
        curve_frame.pack(fill=tk.X)
        self.fig_curve = Figure(figsize=(12, 1.3), dpi=100)
        self.ax_curve = self.fig_curve.add_subplot(111)
        self.fig_curve.tight_layout(pad=0.5)
        self.canvas_curve = FigureCanvasTkAgg(self.fig_curve, master=curve_frame)
        self.canvas_curve.get_tk_widget().pack(fill=tk.X)
        self.canvas_curve.mpl_connect('button_press_event', self._on_curve_click)
        # 帧导航
        nav_frame = ttk.Frame(preview_frame)
        nav_frame.pack(fill=tk.X, pady=(3, 0))
        ttk.Button(nav_frame, text="<<", command=self._prev_frame, width=3).pack(side=tk.LEFT)
        self.frame_slider = ttk.Scale(nav_frame, from_=0, to=1, orient=tk.HORIZONTAL,
                                      command=self._on_slider_change)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(nav_frame, text=">>", command=self._next_frame, width=3).pack(side=tk.LEFT)
        self.frame_label = ttk.Label(nav_frame, text="0/0", width=10)
        self.frame_label.pack(side=tk.LEFT, padx=5)
        ttk.Separator(nav_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Button(nav_frame, text="-", width=3,
                   command=lambda: self._set_zoom(self.zoom_level / ZOOM_FACTOR)).pack(side=tk.LEFT)
        self.zoom_label = ttk.Label(nav_frame, text="100%", width=6, anchor=tk.CENTER)
        self.zoom_label.pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="+", width=3,
                   command=lambda: self._set_zoom(self.zoom_level * ZOOM_FACTOR)).pack(side=tk.LEFT)
        ttk.Button(nav_frame, text="1:1", width=4,
                   command=lambda: self._set_zoom(1.0)).pack(side=tk.LEFT, padx=2)

    def _build_control_section(self, parent):
        ctrl_frame = ttk.LabelFrame(parent, text="矫正控制", padding=3)
        ctrl_frame.pack(fill=tk.X)
        row1 = ttk.Frame(ctrl_frame)
        row1.pack(fill=tk.X, pady=1)
        self.btn_detect = ttk.Button(row1, text="自动检测", command=self._auto_detect)
        self.btn_detect.pack(side=tk.LEFT, padx=3)
        self.btn_save = ttk.Button(row1, text="矫正保存(Ctrl+S)", command=self._execute_correction)
        self.btn_save.pack(side=tk.LEFT, padx=3)
        self.btn_batch = ttk.Button(row1, text="批量处理", command=self._batch_process)
        self.btn_batch.pack(side=tk.LEFT, padx=3)
        self.btn_cancel = ttk.Button(row1, text="取消", command=self._cancel_worker, state=tk.DISABLED)
        self.btn_cancel.pack(side=tk.LEFT, padx=3)
        ttk.Separator(row1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Label(row1, text="偏移:").pack(side=tk.LEFT)
        self.shift_var = tk.StringVar(value="0.00")
        self.shift_entry = ttk.Entry(row1, textvariable=self.shift_var, width=8)
        self.shift_entry.pack(side=tk.LEFT, padx=3)
        self.shift_entry.bind("<Return>", self._on_shift_edit)
        self._register_text_input(self.shift_entry)
        ttk.Button(row1, text="-1", width=3, command=lambda: self._adjust_shift(-1)).pack(side=tk.LEFT, padx=1)
        ttk.Button(row1, text="+1", width=3, command=lambda: self._adjust_shift(1)).pack(side=tk.LEFT, padx=1)
        ttk.Button(row1, text="-0.5", width=4, command=lambda: self._adjust_shift(-0.5)).pack(side=tk.LEFT, padx=1)
        ttk.Button(row1, text="+0.5", width=4, command=lambda: self._adjust_shift(0.5)).pack(side=tk.LEFT, padx=1)
        ttk.Button(row1, text="重置后续", command=self._reset_shift, width=8).pack(side=tk.LEFT, padx=3)
        ttk.Separator(row1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Checkbutton(row1, text="标记模式", variable=self.marker_mode).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="删除标记", command=self._delete_last_image_marker, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="清除标记", command=self._clear_all_markers, width=6).pack(side=tk.LEFT, padx=2)
        self.shift_info_label = ttk.Label(row1, text="")
        self.shift_info_label.pack(side=tk.LEFT, padx=10)
        # 手动矫正模式行
        row_m = ttk.Frame(ctrl_frame)
        row_m.pack(fill=tk.X, pady=1)
        ttk.Checkbutton(row_m, text="手动矫正模式(方向键调偏移)", variable=self.manual_mode,
                        command=self._on_manual_mode_toggle).pack(side=tk.LEFT, padx=2)
        ttk.Label(row_m, text="步长:").pack(side=tk.LEFT, padx=(10, 0))
        manual_step_entry = ttk.Entry(row_m, textvariable=self.manual_step, width=5)
        manual_step_entry.pack(side=tk.LEFT, padx=3)
        self._register_text_input(manual_step_entry)
        ttk.Label(row_m, text="px | ←→调水平 ↑↓调垂直, PageUp/Down翻帧",
                  foreground="gray").pack(side=tk.LEFT, padx=8)
        # 进度行
        row2 = ttk.Frame(ctrl_frame)
        row2.pack(fill=tk.X, pady=(3, 0))
        self.progress = ttk.Progressbar(row2, mode='determinate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.time_label = ttk.Label(row2, text="", width=30)
        self.time_label.pack(side=tk.LEFT, padx=5)

    # --------------------------------------------------------
    # 文件操作
    # --------------------------------------------------------
    def _select_input_folder(self):
        if self._busy:
            return
        folder = filedialog.askdirectory(title="选择输入文件夹")
        if folder:
            self.input_folder.set(folder)
            self._load_file_list()
            self._save_config()

    def _select_output_folder(self):
        if self._busy:
            return
        folder = filedialog.askdirectory(title="选择输出文件夹")
        if folder:
            self.output_folder.set(folder)
            self._save_config()

    def _clear_document(self):
        """列表刷新、加载失败或重新加载时让旧帧绝不关联到新文件。"""
        self.frames = []
        self.meta = None
        self.current_file_idx = -1
        self.current_filepath = None
        self.current_frame_idx = 0
        self.shifts = None
        self.shifts_y = None
        self.auto_shifts = None
        self.auto_shifts_y = None
        self.detection_result = None
        self.manual_dirty = False
        self.markers = []
        self.image_markers = []
        self.active_marker = -1
        self._corr_cache_key = None
        self._corr_cache_img = None
        self._pan_cx = None
        self._pan_cy = None
        self._pan_state = None

    def _load_file_list(self):
        if self._busy:
            return
        folder = self.input_folder.get()
        if not folder or not os.path.isdir(folder):
            return
        pattern = self.file_type.get()
        self.file_list = sorted(glob.glob(os.path.join(folder, pattern)))
        self.file_listbox.delete(0, tk.END)
        for f in self.file_list:
            self.file_listbox.insert(tk.END, os.path.basename(f))
        self._clear_document()
        self.status_var.set(f"找到 {len(self.file_list)} 个文件")

    def _on_file_select(self, event):
        if self._busy:
            return  # 后台任务运行期间禁止切换文件（防结果错配）
        sel = self.file_listbox.curselection()
        if not sel:
            return
        self._load_file(self.file_list[sel[0]], sel[0])

    def _load_file(self, filepath, file_index):
        if self._busy:
            return
        source_path = os.path.abspath(filepath)
        # 两段式加载：先只读目录元数据（轴确认与大堆栈内存确认都发生在
        # 整栈解码之前），确认通过后再进入像素读取；加载失败或取消时
        # 旧文档保持不变。
        self._start_worker(
            target_func=_inspect_stack_quiet,
            args=(source_path,),
            on_complete=lambda meta: self._on_inspect_complete(meta, source_path, file_index),
            status_text=f"读取元数据: {os.path.basename(source_path)}...",
        )

    def _on_inspect_complete(self, meta, source_path, file_index):
        self._set_busy(False)
        if self._closing:
            return
        if meta is None:
            self.status_var.set("加载已取消；保持当前文档不变")
            return
        confirm_required = meta.get('confirm_required', [])
        warnings = meta.get('warnings', [])
        if (confirm_required or warnings) and not messagebox.askyesno(
                "TIFF 轴需确认",
                "\n".join(confirm_required + warnings) + "\n\n仍按页顺序作为时间序列加载吗？"):
            self.status_var.set("已取消加载：请先确认 TIFF 轴含义；保持当前文档不变")
            return
        estimated = int(meta.get('estimated_bytes') or 0)
        total = total_physical_memory()
        if total and estimated > MEMORY_WARN_RATIO * total:
            if not messagebox.askyesno(
                    "大堆栈内存确认",
                    f"该堆栈解码后约需 {estimated / 1024 ** 3:.1f} GiB 内存，"
                    f"本机物理内存共 {total / 1024 ** 3:.1f} GiB。\n"
                    "继续加载可能造成系统严重卡顿甚至失败。\n\n仍要继续加载吗？"):
                self.status_var.set("已取消加载大堆栈；保持当前文档不变")
                return
        self._start_worker(
            target_func=TiffIO.read_stack,
            args=(source_path,),
            kwargs={'meta': meta},
            on_complete=lambda result: self._on_load_complete(result, source_path, file_index),
            status_text=f"加载: {os.path.basename(source_path)}...",
        )

    def _on_load_complete(self, result, source_path, file_index):
        self._set_busy(False)
        if self._closing:
            return
        if result is None:
            self.status_var.set("加载已取消；保持当前文档不变")
            return
        frames, meta = result
        # 在清空旧文档之前先验证新数据；任何失败都保留旧状态。
        step = max(1, len(frames) // 8)
        sample = frames[::step][:8]
        if not any(np.isfinite(frame).any() for frame in sample):
            messagebox.showerror("加载错误", "图像不包含有限像素；保持当前文档不变")
            return
        preview_vmin, preview_vmax = _preview_contrast_range(sample)
        # 加载成功，才清空旧文档并切换到新文档。
        self._clear_document()
        self.frames, self.meta = frames, meta
        self.current_file_idx = file_index
        self.current_filepath = source_path
        self.document_id += 1
        self.current_frame_idx = 0
        # “尚未检测”必须用 None 表示，不能伪装成全零检测结果。旧逻辑会让
        # 批处理直接复用这组零位移，生成与原图相同却标为 corrected 的文件。
        self.shifts = None
        self.shifts_y = None
        self.auto_shifts = None
        self.auto_shifts_y = None
        self.detection_result = None
        self.manual_dirty = False
        self._img_artists = {'prev': None, 'curr': None, 'corr': None}
        self._marker_artists = []
        self._corr_cache_key = None
        self._preview_vmin = preview_vmin
        self._preview_vmax = preview_vmax
        self.shift_var.set("")
        self.shift_info_label.config(text="尚未检测")
        n = len(frames)
        self.frame_slider.config(to=max(1, n - 1))
        self.frame_slider.set(0)
        self._update_preview()
        self._update_curve()
        estimated = meta.get('estimated_bytes')
        size_note = f", 约{estimated / 1024 ** 3:.1f}GiB" if estimated else ""
        self.status_var.set(
            f"已加载: {os.path.basename(source_path)} ({n}帧, "
            f"{meta['size'][0]}x{meta['size'][1]}, 位深={meta.get('source_dtype')}{size_note})")

    # --------------------------------------------------------
    # 界面配置持久化（上次使用的目录与参数）
    # --------------------------------------------------------
    def _save_config(self):
        """记住最近一次使用的目录与参数；失败只记日志，绝不影响主流程。"""
        try:
            data = {name: getattr(self, name).get()
                    for name in ("input_folder", "output_folder", "file_type", "manual_step")
                    if hasattr(self, name)}
            data["params"] = {name: getattr(self, name).get()
                              for name in _PARAM_VAR_NAMES if hasattr(self, name)}
            path = _config_path()
            temporary = path.with_name(path.name + ".part")
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
            os.replace(temporary, path)
        except Exception as exc:
            logger.warning("界面配置保存失败：%s", exc)

    def _restore_config(self):
        try:
            data = json.loads(_config_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        restored_input = False
        folder = data.get("input_folder")
        if isinstance(folder, str) and folder and os.path.isdir(folder):
            self.input_folder.set(folder)
            restored_input = True
        folder = data.get("output_folder")
        if isinstance(folder, str) and folder and os.path.isdir(folder):
            self.output_folder.set(folder)
        file_type = data.get("file_type")
        if file_type in ("*.tif", "*.tiff"):
            self.file_type.set(file_type)
        params = data.get("params")
        if isinstance(params, dict):
            for name in _PARAM_VAR_NAMES:
                value = params.get(name)
                if isinstance(value, str):
                    getattr(self, name).set(value)
        step = data.get("manual_step")
        if isinstance(step, str) and step:
            self.manual_step.set(step)
        if restored_input:
            self._load_file_list()

    # --------------------------------------------------------
    # 预览与导航
    # --------------------------------------------------------
    def _on_manual_mode_toggle(self):
        self._rebuild_preview_axes()
        if self.frames:
            self._update_preview()
        mode = "手动矫正模式" if self.manual_mode.get() else "普通模式"
        self.status_var.set(f"已切换到{mode}")

    def _rebuild_preview_axes(self):
        self.fig.clear()
        if self.manual_mode.get():
            self.ax_prev = None
            self.ax_curr = self.fig.add_subplot(121)
            self.ax_corr = self.fig.add_subplot(122)
        else:
            self.ax_prev = self.fig.add_subplot(131)
            self.ax_curr = self.fig.add_subplot(132)
            self.ax_corr = self.fig.add_subplot(133)
        for ax in [self.ax_curr, self.ax_corr]:
            ax.axis('off')
        if self.ax_prev is not None:
            self.ax_prev.axis('off')
        # axes 重建后旧 artist 全部失效
        self._img_artists = {'prev': None, 'curr': None, 'corr': None}
        self._marker_artists = []
        self.fig.tight_layout(pad=0.5)
        self.canvas.draw_idle()

    def _show_on_ax(self, ax, key, img, title):
        """P2 性能：复用 AxesImage（set_data），仅结构变化时重建"""
        artist = self._img_artists[key]
        if artist is None or artist.axes is not ax:
            ax.clear()
            ax.axis('off')
            artist = ax.imshow(img, cmap='gray',
                               vmin=self._preview_vmin, vmax=self._preview_vmax)
            self._img_artists[key] = artist
        else:
            artist.set_data(img)
            artist.set_clim(self._preview_vmin, self._preview_vmax)
        ax.set_title(title, fontsize=9)

    def _update_preview(self):
        if not self.frames:
            return
        idx = self.current_frame_idx
        n = len(self.frames)
        manual = self.manual_mode.get()
        shift_x = float(self.shifts[idx]) if self.shifts is not None else 0.0
        shift_y = float(self.shifts_y[idx]) if self.shifts_y is not None else 0.0

        # 矫正帧缓存：仅当帧索引或偏移变化时才重算反向平移。
        cache_key = (idx, shift_x, shift_y, id(self.frames))
        if self._corr_cache_key != cache_key:
            if self.shifts is not None or self.shifts_y is not None:
                self._corr_cache_img = DriftCorrector.correct_single_frame(
                    self.frames[idx], shift_x, shift_y)
            else:
                self._corr_cache_img = self.frames[idx]
            self._corr_cache_key = cache_key

        if not manual and self.ax_prev is not None:
            if idx > 0:
                self._show_on_ax(self.ax_prev, 'prev', self.frames[idx - 1],
                                 f"上一帧 ({idx - 1})")
            else:
                if self._img_artists['prev'] is not None:
                    self.ax_prev.clear()
                    self.ax_prev.axis('off')
                    self._img_artists['prev'] = None
                # 只在空状态尚未画过提示时添加，避免反复停留第 0 帧时文字累积。
                if not self.ax_prev.texts:
                    self.ax_prev.text(0.5, 0.5, "无", ha='center', va='center',
                                      transform=self.ax_prev.transAxes)
                self.ax_prev.set_title("上一帧", fontsize=9)

        self._show_on_ax(self.ax_curr, 'curr', self.frames[idx], f"原始帧 ({idx + 1}/{n})")
        self._show_on_ax(self.ax_corr, 'corr', self._corr_cache_img,
                         f"矫正预览 (补偿漂移 dx={shift_x:+.1f}, dy={shift_y:+.1f}px)")

        self._draw_image_markers()
        self._apply_zoom_to_axes()
        self.canvas.draw_idle()
        self.frame_label.config(text=f"{idx + 1}/{n}")
        if self.shifts is not None:
            self.shift_var.set(f"{self.shifts[idx]:.2f}")
            if self.auto_shifts is not None:
                self.shift_info_label.config(
                    text=f"最大:{np.max(np.abs(self.shifts)):.1f}px | 自动:{self.auto_shifts[idx]:.2f}")

    def _update_curve(self):
        self.ax_curve.clear()
        if self.shifts is not None and len(self.shifts) > 0:
            x = np.arange(len(self.shifts))
            # 未通过质量门控、由插值得到的区段以橙色底纹显式标出：
            # 科学 QC 的关键是能看见“哪些数据段是估计出来的”。
            detection = self.detection_result
            has_spans = False
            if detection is not None and detection.interpolated_pairs:
                keys = detection.key_indices
                for pair_idx in detection.interpolated_pairs:
                    if 0 <= pair_idx < len(keys) - 1:
                        self.ax_curve.axvspan(keys[pair_idx], keys[pair_idx + 1],
                                              color='orange', alpha=0.18, zorder=0)
                        has_spans = True
            self.ax_curve.plot(x, self.shifts, 'b-', linewidth=0.8, label='水平', zorder=2)
            if self.shifts_y is not None and np.any(self.shifts_y != 0):
                self.ax_curve.plot(x, self.shifts_y, 'r-', linewidth=0.8, alpha=0.6,
                                   label='垂直', zorder=2)
            if has_spans:
                self.ax_curve.plot([], [], color='orange', alpha=0.4, linewidth=6,
                                   label='插值段')
            self.ax_curve.axvline(self.current_frame_idx, color='green', linewidth=1, alpha=0.7)
            for i, m_idx in enumerate(self.markers):
                color = 'red' if i == self.active_marker else 'orange'
                self.ax_curve.axvline(m_idx, color=color, linewidth=1.5, linestyle='--')
                if m_idx < len(self.shifts):
                    self.ax_curve.plot(m_idx, self.shifts[m_idx], 'o', color=color, markersize=5)
            if self.ax_curve.get_legend_handles_labels()[1]:
                self.ax_curve.legend(fontsize=7, loc='upper right')
        self.ax_curve.set_xlabel("帧", fontsize=8)
        self.ax_curve.set_ylabel("px", fontsize=8)
        self.ax_curve.tick_params(labelsize=7)
        self.fig_curve.tight_layout(pad=0.5)
        self.canvas_curve.draw_idle()

    def _prev_frame(self):
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.frame_slider.set(self.current_frame_idx)
            self._update_preview()
            self._update_curve()

    def _next_frame(self):
        if self.frames and self.current_frame_idx < len(self.frames) - 1:
            self.current_frame_idx += 1
            self.frame_slider.set(self.current_frame_idx)
            self._update_preview()
            self._update_curve()

    def _on_slider_change(self, value):
        idx = int(float(value))
        if self.frames:
            # 单帧堆栈时滑条范围仍为 0..1，必须钳制，否则索引越界。
            idx = max(0, min(idx, len(self.frames) - 1))
        if idx != self.current_frame_idx and self.frames:
            self.current_frame_idx = idx
            self._update_preview()
            self._update_curve()

    # --------------------------------------------------------
    # 缩放
    # --------------------------------------------------------
    def _set_zoom(self, level):
        self.zoom_level = max(ZOOM_MIN, min(ZOOM_MAX, level))
        if self.zoom_level == 1.0:
            # 1:1 视野即整幅图像，平移偏移随之复位
            self._pan_cx = None
            self._pan_cy = None
        self.zoom_label.config(text=f"{self.zoom_level * 100:.0f}%")
        if self.frames:
            self._update_preview()

    def _on_scroll(self, event):
        if event.inaxes not in (self.ax_prev, self.ax_curr, self.ax_corr):
            return
        factor = ZOOM_FACTOR if event.button == 'up' else 1 / ZOOM_FACTOR
        self._set_zoom(self.zoom_level * factor)

    def _apply_zoom_to_axes(self):
        if not self.frames:
            return
        h, w = self.frames[0].shape[:2]
        cx = w / 2.0 if self._pan_cx is None else self._pan_cx
        cy = h / 2.0 if self._pan_cy is None else self._pan_cy
        half_w = w / (2.0 * self.zoom_level)
        half_h = h / (2.0 * self.zoom_level)
        for ax in (self.ax_prev, self.ax_curr, self.ax_corr):
            if ax is not None:
                ax.set_xlim(cx - half_w, cx + half_w)
                ax.set_ylim(cy + half_h, cy - half_h)

    # --------------------------------------------------------
    # 标记系统
    # --------------------------------------------------------
    def _on_image_press(self, event):
        """左键按下分发：标记模式添加点标记，普通模式启动拖拽平移。"""
        if self.marker_mode.get():
            self._on_image_click(event)
            return
        if not self.frames or getattr(event, 'button', None) != 1:
            return
        ax = event.inaxes
        if ax not in (self.ax_prev, self.ax_curr, self.ax_corr):
            return
        if event.xdata is None or event.ydata is None:
            return
        bbox = ax.bbox
        if bbox.width <= 0 or bbox.height <= 0:
            return
        h, w = self.frames[0].shape[:2]
        # 以按下时的屏幕坐标与视图中心为基准，之后每次移动都相对该基准
        # 换算，避免视图变化后累计误差。
        self._pan_state = (
            event.x, event.y,
            w / 2.0 if self._pan_cx is None else self._pan_cx,
            h / 2.0 if self._pan_cy is None else self._pan_cy,
            (w / self.zoom_level) / bbox.width,
            (h / self.zoom_level) / bbox.height,
        )
        try:
            self.canvas.get_tk_widget().config(cursor='fleur')
        except tk.TclError:
            pass

    def _on_image_motion(self, event):
        state = self._pan_state
        if state is None or not self.frames:
            return
        sx0, sy0, cx0, cy0, scale_x, scale_y = state
        h, w = self.frames[0].shape[:2]
        # 拖拽平移（抓图语义：图像跟随光标移动），中心钳制在图像范围内。
        cx = cx0 - (event.x - sx0) * scale_x
        cy = cy0 + (event.y - sy0) * scale_y
        self._pan_cx = min(max(cx, 0.0), float(w))
        self._pan_cy = min(max(cy, 0.0), float(h))
        self._apply_zoom_to_axes()
        self.canvas.draw_idle()

    def _on_image_release(self, event):
        if self._pan_state is not None:
            self._pan_state = None
            try:
                self.canvas.get_tk_widget().config(cursor='')
            except tk.TclError:
                pass

    def _on_image_click(self, event):
        if not self.marker_mode.get():
            return
        if not self.frames:
            return
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if event.inaxes not in (self.ax_prev, self.ax_curr, self.ax_corr):
            return
        h, w = self.frames[0].shape[:2]
        x = max(0, min(w - 1, int(round(event.xdata))))
        y = max(0, min(h - 1, int(round(event.ydata))))
        self.image_markers.append((x, y))
        self.status_var.set(f"标记 ({x}, {y}) | 共{len(self.image_markers)}个 | Del删除, Esc清除")
        self._update_preview()

    def _on_curve_click(self, event):
        """P3 补全：点击漂移曲线添加/移除帧标记（原 markers 渲染代码此前没有添加入口）"""
        if event.inaxes is not self.ax_curve or event.xdata is None:
            return
        if self.shifts is None or not self.frames:
            return
        idx = int(round(event.xdata))
        idx = max(0, min(len(self.frames) - 1, idx))
        if idx in self.markers:
            pos = self.markers.index(idx)
            self.markers.pop(pos)
            if self.active_marker == pos:
                self.active_marker = -1
            elif self.active_marker > pos:
                self.active_marker -= 1
            self.status_var.set(f"移除帧标记 {idx} | 剩余{len(self.markers)}个")
        else:
            self.markers.append(idx)
            self.markers.sort()
            self.active_marker = self.markers.index(idx)
            self.status_var.set(f"添加帧标记 {idx} | 共{len(self.markers)}个 | 再次点击可移除")
        self._update_curve()

    def _draw_image_markers(self):
        for art in self._marker_artists:
            try:
                art.remove()
            except (ValueError, NotImplementedError):
                pass
        self._marker_artists = []
        if not self.image_markers:
            return
        axes = [self.ax_curr, self.ax_corr]
        if not self.manual_mode.get() and self.ax_prev is not None:
            axes.append(self.ax_prev)
        for ax in axes:
            if ax is None:
                continue
            for i, (mx, my) in enumerate(self.image_markers):
                color = 'lime' if i == len(self.image_markers) - 1 else 'yellow'
                line, = ax.plot(mx, my, '+', color=color, markersize=8, markeredgewidth=1.5)
                txt = ax.annotate(f"{i + 1}", (mx, my), textcoords="offset points",
                                  xytext=(4, 4), fontsize=7, color=color)
                self._marker_artists.extend([line, txt])

    def _delete_last_image_marker(self):
        if self.image_markers:
            removed = self.image_markers.pop()
            self.status_var.set(f"删除标记 {removed} | 剩余{len(self.image_markers)}个")
            self._update_preview()

    def _clear_all_markers(self):
        count = len(self.image_markers) + len(self.markers)
        self.image_markers = []
        self.markers = []
        self.active_marker = -1
        self.status_var.set(f"已清除{count}个标记")
        self._update_preview()
        self._update_curve()

    # --------------------------------------------------------
    # 后台任务通用（P1：检测/矫正/批处理统一后台化 + UI 忙锁定）
    # --------------------------------------------------------
    def _set_busy(self, busy):
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        for w in (self.btn_detect, self.btn_save, self.btn_batch,
                  self.btn_refresh, self.file_listbox):
            w.config(state=state)
        self.btn_cancel.config(state=tk.NORMAL if busy else tk.DISABLED)

    def _schedule_ui(self, callback, *args):
        """后台线程只入队；Tk API 永远由主线程调用。"""
        self._ui_queue.put((callback, args))

    def _drain_ui_queue(self):
        """由 Tk 主线程消费后台回调，避免跨线程 root.after。

        单个回调的异常只记录不传播：事件泵必须在 finally 中重排自身，
        否则一次未预期异常就会让所有后续任务完成/进度/错误回调永久失联，
        界面表现为"永远在忙"。
        """
        try:
            while True:
                callback, args = self._ui_queue.get_nowait()
                if self._closing:
                    continue
                try:
                    callback(*args)
                except Exception:
                    logger.exception("UI 回调执行失败")
        except queue.Empty:
            pass
        finally:
            if not self._closing:
                try:
                    self.root.after(30, self._drain_ui_queue)
                except tk.TclError:
                    pass

    def _start_worker(self, target_func, args=(), kwargs=None,
                      on_complete=None, status_text=""):
        """统一的后台任务启动入口"""
        self._set_busy(True)
        self.status_var.set(status_text)
        self.progress['value'] = 0
        self.time_label.config(text="")
        self._task_start_time = time.time()
        self.worker = WorkerThread(
            target_func=target_func,
            args=args,
            kwargs=kwargs,
            on_progress=lambda r: self._schedule_ui(self._update_task_progress, r),
            on_complete=(lambda result: self._schedule_ui(on_complete, result)) if on_complete else None,
            on_error=lambda err: self._schedule_ui(self._on_task_error, err),
        )
        self.worker.start()

    def _update_task_progress(self, ratio):
        self.progress['value'] = ratio * 100
        elapsed = time.time() - self._task_start_time
        if 0.05 < ratio < 1.0:
            eta = elapsed / ratio * (1 - ratio)
            self.time_label.config(text=f"已用{elapsed:.0f}s / 剩余约{eta:.0f}s")

    def _on_task_error(self, err):
        self._set_busy(False)
        self.progress['value'] = 0
        if self._closing:
            return
        self.status_var.set("任务失败")
        messagebox.showerror("错误", f"后台任务失败:\n{err}")

    def _cancel_worker(self):
        if self.worker and self.worker.is_alive():
            self.worker.cancel()
            self.status_var.set("正在取消...")
            self.btn_cancel.config(state=tk.DISABLED)

    def _on_close(self):
        """保存/检测任务必须先结束，避免 daemon 线程留下半写入文件。"""
        self._save_config()
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("任务进行中", "当前任务尚未结束。取消任务并在清理完成后退出吗？"):
                return
            self._closing = True
            self._close_deadline = time.time() + 60.0
            self._close_force_prompted = False
            self.worker.cancel()
            self.status_var.set("正在取消任务并安全退出（最多等待60秒）...")
            self.root.after(50, self._finish_close)
            return
        self._closing = True
        self.root.destroy()

    def _finish_close(self):
        if self.worker and self.worker.is_alive():
            if time.time() < self._close_deadline:
                self.root.after(50, self._finish_close)
                return
            # 60 秒超时后必须给用户强制退出选项，不能无限轮询挂起。
            if not self._close_force_prompted:
                self._close_force_prompted = True
                if messagebox.askyesno(
                        "强制退出",
                        "任务未在60秒内结束。\n\n强制退出会直接关闭窗口；"
                        "已有输出文件不会被破坏，本次临时文件可能残留。\n是否强制退出？"):
                    self.root.destroy()
                    # worker 是非 daemon 线程，仅 destroy 会让解释器一直等它
                    # 结束（窗口已消失但进程残留）；此刻写入方对临时文件已有
                    # 自清理保证，直接结束进程。
                    os._exit(0)
                self._close_force_prompted = False
                self._close_deadline = time.time() + 60.0
                self.status_var.set("继续等待任务结束...")
            self.root.after(50, self._finish_close)
            return
        self.root.destroy()

    # --------------------------------------------------------
    # 漂移检测（后台线程）
    # --------------------------------------------------------
    def _auto_detect(self):
        if self._busy:
            return
        if not self.frames:
            messagebox.showwarning("提示", "请先加载TIFF文件")
            return
        if len(self.frames) == 1:
            messagebox.showwarning("提示", "单帧堆栈不存在帧间漂移，无需检测")
            return
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("提示", "任务正在进行中，请等待完成或取消")
            return
        params = self._get_sift_params()
        if params is None:
            return
        skip = self._get_skip_interval()
        if skip is None:
            return
        document_id = self.document_id
        source_path = self.current_filepath
        self._save_config()

        self._start_worker(
            target_func=DriftDetector.detect_drift,
            args=(self.frames,),
            kwargs={'params': params, 'skip_interval': skip, 'use_parallel': True},
            on_complete=lambda result: self._on_detect_complete(result, document_id, source_path),
            status_text="正在检测漂移...",
        )

    def _on_detect_complete(self, result, document_id=None, source_path=None):
        self._set_busy(False)
        if self._closing:
            return
        if document_id != self.document_id or source_path != self.current_filepath:
            self.status_var.set("已忽略旧文件的检测结果")
            return
        if result is None:
            self.status_var.set("检测已取消")
            self.progress['value'] = 0
            self.time_label.config(text="")
            return

        self.shifts = result.shifts_x
        self.shifts_y = result.shifts_y
        self.auto_shifts = result.shifts_x.copy()
        self.auto_shifts_y = result.shifts_y.copy()
        self.detection_result = result
        self.manual_dirty = False
        # 第 1 帧是配准基准，位移恒为 0。若检测结束仍停在那里，原始图与
        # 矫正预览必然完全相同，会误导用户认为没有执行矫正。
        preview_index = _representative_preview_frame(self.shifts, self.shifts_y)
        self.current_frame_idx = preview_index
        self.frame_slider.set(preview_index)
        self._corr_cache_key = None
        self._pan_cx = None
        self._pan_cy = None
        self.progress['value'] = 100
        elapsed = time.time() - self._task_start_time
        self.time_label.config(text=f"完成! 耗时{elapsed:.1f}s")
        max_s = np.max(np.abs(self.shifts))
        max_sy = np.max(np.abs(self.shifts_y))
        self.status_var.set(
            f"检测完成! 水平最大:{max_s:.1f}px 垂直最大:{max_sy:.1f}px | "
            f"可靠帧对:{result.matched_pairs}/{result.total_pairs} | "
            f"范围X:[{np.min(self.shifts):.1f},{np.max(self.shifts):.1f}] | "
            f"已定位最大漂移帧:{preview_index + 1}/{len(self.frames)}"
            + ((" | " + " | ".join(result.warnings)) if result.warnings else ""))
        self._update_preview()
        self._update_curve()

    # --------------------------------------------------------
    # 矫正与保存（后台线程）
    # --------------------------------------------------------
    def _execute_correction(self):
        if self._busy:
            return
        if not self.frames:
            messagebox.showwarning("提示", "请先加载文件")
            return
        manual_ready = self.manual_mode.get() and self.manual_dirty and self.shifts is not None
        if self.shifts is None or (self.detection_result is None and not manual_ready):
            messagebox.showwarning(
                "提示", "当前没有有效的漂移结果。请先执行自动检测，"
                "或在手动矫正模式下实际调整位移后再保存。")
            return
        output_folder = self.output_folder.get()
        if not output_folder:
            messagebox.showwarning("提示", "请先选择输出文件夹")
            return
        if self.current_file_idx < 0 or not self.current_filepath:
            messagebox.showwarning("提示", "请先选择文件")
            return
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("提示", "任务正在进行中，请等待完成或取消")
            return

        filepath = self.current_filepath
        stem, suffix = os.path.splitext(os.path.basename(filepath))
        output_path = os.path.join(output_folder, f"{stem}_corrected{suffix}")
        overwrite = False
        if os.path.exists(output_path):
            overwrite = messagebox.askyesno(
                "输出已存在", f"{os.path.basename(output_path)} 已存在。\n仅替换该输出文件，继续吗？")
            if not overwrite:
                return
        # 复制偏移数组快照，防止保存期间用户继续手动调整造成数据不一致
        shifts_x = self.shifts.copy()
        shifts_y = self.shifts_y.copy() if self.shifts_y is not None else None
        max_shift_x = float(np.max(np.abs(shifts_x)))
        max_shift_y = float(np.max(np.abs(shifts_y))) if shifts_y is not None else 0.0
        self._save_config()

        self._start_worker(
            target_func=correct_and_save,
            args=(self.frames, shifts_x, shifts_y, self.meta, output_path),
            kwargs={'overwrite': overwrite, 'crop_mode': 'crop'},
            on_complete=lambda res: self._on_save_complete(
                res, output_path, max_shift_x, max_shift_y),
            status_text=f"矫正保存: {os.path.basename(filepath)}...",
        )

    def _on_save_complete(self, written, output_path, max_shift_x=0.0, max_shift_y=0.0):
        self._set_busy(False)
        if written is None:
            self.status_var.set("矫正保存已取消；既有输出保持不变")
            self.progress['value'] = 0
            self.time_label.config(text="")
            return
        self.progress['value'] = 100
        elapsed = time.time() - self._task_start_time
        self.time_label.config(text=f"耗时{elapsed:.1f}s")
        # 同步写出逐帧位移表（含插值区段标记），供后续定量分析导入
        table_note = ""
        try:
            table_path = os.path.splitext(output_path)[0] + "_shifts.csv"
            detection = self.detection_result
            write_shift_table(
                table_path, self.shifts, self.shifts_y,
                detection.key_indices if detection is not None else None,
                detection.interpolated_pairs if detection is not None else None)
            table_note = f" | 位移表: {os.path.basename(table_path)}"
        except (OSError, ValueError) as exc:
            table_note = f" | 位移表写出失败: {exc}"
        self.status_var.set(
            f"矫正已生效并保存: {output_path} ({written}帧) | "
            f"应用最大位移 X={max_shift_x:.1f}px, Y={max_shift_y:.1f}px{table_note}")

    # --------------------------------------------------------
    # 批量处理（后台线程）
    # --------------------------------------------------------
    def _batch_process(self):
        if self._busy:
            return
        if not self.file_list:
            messagebox.showwarning("提示", "请先加载文件列表")
            return
        output_folder = self.output_folder.get()
        if not output_folder:
            messagebox.showwarning("提示", "请先选择输出文件夹")
            return
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("提示", "任务正在进行中，请等待完成或取消")
            return
        params = self._get_sift_params()
        if params is None:
            return
        skip = self._get_skip_interval(for_batch=True)
        if skip is None:
            return
        n = len(self.file_list)
        if not messagebox.askyesno("确认", f"将处理 {n} 个文件，是否继续？"):
            return
        # 输出写进输入目录会让 _corrected.tif 混入输入列表，刷新后存在
        # 二次矫正的风险；允许但必须让用户明确知情。
        input_dir = self.input_folder.get()
        if input_dir and os.path.normcase(os.path.abspath(input_dir)) == os.path.normcase(
                os.path.abspath(output_folder)):
            if not messagebox.askyesno(
                    "输出目录与输入目录相同",
                    "矫正结果会出现在输入文件列表中，刷新后存在被二次矫正的风险。\n"
                    "建议选择独立的输出目录。仍要继续吗？"):
                return
        overwrite = False
        existing = []
        for item in self.file_list:
            stem, suffix = os.path.splitext(os.path.basename(item))
            candidate = os.path.join(output_folder, f"{stem}_corrected{suffix}")
            if os.path.exists(candidate):
                existing.append(os.path.basename(candidate))
        if existing:
            shown = "\n".join(existing[:5])
            if len(existing) > 5:
                shown += f"\n...等共 {len(existing)} 个"
            if not messagebox.askyesno(
                    "输出已存在",
                    f"输出目录已存在 {len(existing)} 个同名矫正结果：\n{shown}\n\n"
                    "覆盖这些文件吗？（选择“否”将取消本次批处理）"):
                return
            overwrite = True
        # 含 C/Z 轴的文件无法从元数据自动区分时间序列与 z-stack/多通道，
        # 批处理前必须获得一次明确的人工确认（只读元数据，预扫描很快）。
        allow_ambiguous_axes = False
        ambiguous = []
        for item in self.file_list:
            try:
                meta = TiffIO.inspect_stack(item)
            except Exception:
                continue  # 读不了的文件交给批处理报告具体错误
            if meta.get("confirm_required"):
                ambiguous.append((os.path.basename(item), meta.get("axes", "?")))
        if ambiguous:
            shown = "; ".join(f"{name}（{axes}）" for name, axes in ambiguous[:3])
            more = f" 等共 {len(ambiguous)} 个" if len(ambiguous) > 3 else ""
            if messagebox.askyesno(
                    "轴含义确认",
                    f"{len(ambiguous)} 个文件的序列轴包含通道(C)或层(Z)维度：\n{shown}{more}\n\n"
                    "文件元数据无法区分时间序列与 z-stack/多通道数据；TEM 连续采集"
                    "常被相机软件把时间轴标注为 slices/Z。\n\n"
                    "选择“是”：这些文件按页顺序作为时间帧处理；\n"
                    "选择“否”：跳过这些文件，其余文件照常处理。"):
                allow_ambiguous_axes = True

        # 当前文件若已检测，复用其结果（帧数据由 GUI 持有，批处理不释放）。
        # key 必须与 batch_process 内的 Path.resolve() 规范化一致，否则缓存不会命中。
        precomputed = {}
        if (self.current_filepath and self.frames and self.shifts is not None
                and self.detection_result is not None):
            precomputed[str(Path(self.current_filepath).resolve())] = {
                'frames': self.frames,
                'shifts_x': self.shifts.copy(),
                'shifts_y': self.shifts_y.copy() if self.shifts_y is not None else None,
                'meta': self.meta,
                'detection': self.detection_result,
            }

        self._start_worker(
            target_func=batch_process,
            args=(self.file_list, output_folder),
            kwargs={'params': params, 'skip_interval': skip, 'precomputed': precomputed,
                    'overwrite': overwrite, 'crop_mode': 'crop',
                    'allow_ambiguous_axes': allow_ambiguous_axes},
            on_complete=lambda res: self._on_batch_complete(res, output_folder),
            status_text=f"批量处理 {n} 个文件...",
        )
        self._save_config()

    def _on_batch_complete(self, result, output_folder):
        self._set_busy(False)
        completed = result['completed']
        total = result['total']
        errors = result['errors']
        elapsed = time.time() - self._task_start_time
        self.time_label.config(text=f"总耗时{elapsed:.0f}s")
        if result['cancelled']:
            self.status_var.set(f"批处理已取消 (完成 {completed}/{total})")
        else:
            self.progress['value'] = 100
            self.status_var.set(f"批量处理完成: {completed}/{total}")
        msg = (f"完成 {completed}/{total} 个文件\n输出: {output_folder}\n"
               f"审计报告: {result.get('report_json', '未生成')}\n"
               f"CSV: {result.get('report_csv', '未生成')}")
        if errors:
            shown = "\n".join(f"  {name}: {err}" for name, err in errors[:10])
            more = f"\n  ...及其余 {len(errors) - 10} 个" if len(errors) > 10 else ""
            msg += f"\n\n失败 {len(errors)} 个:\n{shown}{more}"
            messagebox.showwarning("批处理结果", msg)
        else:
            messagebox.showinfo("批处理结果", msg)

    # --------------------------------------------------------
    # 逐帧微调
    # --------------------------------------------------------
    def _on_shift_edit(self, event=None):
        if self._busy or not self.frames:
            return
        if self.shifts is None:
            if not self.manual_mode.get():
                return
            self.shifts = np.zeros(len(self.frames), dtype=np.float32)
            self.shifts_y = np.zeros(len(self.frames), dtype=np.float32)
        try:
            new_val = float(self.shift_var.get())
            if not np.isfinite(new_val):
                raise ValueError
        except ValueError:
            self.status_var.set(f"无效偏移值: {self.shift_var.get()!r}，请输入数字")
            self.shift_var.set(f"{self.shifts[self.current_frame_idx]:.2f}")
            return
        idx = self.current_frame_idx
        delta = new_val - self.shifts[idx]
        self.shifts[idx] = new_val
        if delta != 0.0:
            self.shifts[idx + 1:] += delta
            self.manual_dirty = True
        self._update_preview()
        self._update_curve()

    def _adjust_shift(self, delta):
        if self._busy or self.shifts is None or not self.frames:
            return
        if delta == 0.0:
            return
        idx = self.current_frame_idx
        self.shifts[idx] += delta
        self.shifts[idx + 1:] += delta
        # 与 _manual_adjust/_on_shift_edit 保持同一语义：任何实际调整都标记
        # 手动修改，manual_dirty 不得因入口不同而漂移。
        self.manual_dirty = True
        self.shift_var.set(f"{self.shifts[idx]:.2f}")
        self._update_preview()
        self._update_curve()

    def _reset_shift(self):
        """把当前帧及其后所有帧的偏移恢复为自动检测值（手动微调是联动的，只还原单帧会断链）。"""
        if self._busy or self.auto_shifts is None or not self.frames:
            return
        idx = self.current_frame_idx
        self.shifts[idx:] = self.auto_shifts[idx:]
        if self.auto_shifts_y is not None and self.shifts_y is not None:
            self.shifts_y[idx:] = self.auto_shifts_y[idx:]
        self.shift_var.set(f"{self.shifts[idx]:.2f}")
        self.status_var.set(f"已将帧{idx}及其后 {len(self.frames) - idx} 帧恢复为自动检测值")
        self._update_preview()
        self._update_curve()


# ============================================================
# 程序入口
# ============================================================

def main():
    # 日志配置放在入口而非模块 import，避免库式导入 drift_correction 时
    # 产生全局副作用。
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    if "--health-check" in sys.argv:
        _package_health_check()
        return
    # Windows 高 DPI 感知（避免界面模糊）
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    # 拖拽支持需要 TkinterDnD.Tk（普通 tk.Tk 没有 drop_target_register）
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except (ImportError, RuntimeError, tk.TclError) as exc:
        logger.warning("拖拽功能不可用，已回退普通 Tk：%s", exc)
        root = tk.Tk()
    app = DriftCorrectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app._on_close)

    def handle_exception(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.error("未捕获异常", exc_info=(exc_type, exc_value, exc_tb))
        try:
            messagebox.showerror("程序错误", f"发生未处理的异常:\n{exc_value}")
        except Exception:
            pass

    sys.excepthook = handle_exception
    # tkinter 事件回调内异常不经过 sys.excepthook，需单独接管
    root.report_callback_exception = handle_exception
    root.mainloop()


if __name__ == "__main__":
    main()
