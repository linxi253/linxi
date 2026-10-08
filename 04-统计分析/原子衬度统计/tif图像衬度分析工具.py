"""
TIF图像衬度分析工具 v2.5.0
功能：对TEM图像堆叠序列进行ROI区域衬度统计分析
版本与算法清单以 contrast_core.__version__ / ALGORITHMS 为单一来源
"""

import csv
import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Tuple

import numpy as np

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
from matplotlib import font_manager
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import tifffile
from tifffile import TiffFileError

import ttkbootstrap as ttkb

from contrast_core import (  # noqa: E402
    ALGORITHMS,
    TiffProbe,
    __version__,
    clamp_roi_to_image,
    compute_contrast_stack_ex,
    estimate_display_clim,
    format_metadata_lines,
    open_tiff_stack_ex,
    page_datetimes,
    pixel_size_nm,
    probe_tiff,
)

# 深色主题配色常量
DARK_BG = '#2b2b2b'
DARK_AXES = '#1e1e1e'
LIGHT_TEXT = '#cccccc'
TITLE_TEXT = '#ffffff'
SPINE_COLOR = '#555555'
MENU_BG = '#2b2b2b'
MENU_FG = '#cccccc'
MENU_ACTIVE_BG = '#444444'
MENU_ACTIVE_FG = '#ffffff'

# ROI 颜色循环
ROI_COLORS = ['#00ffff', '#ff6600', '#66ff66', '#ff66ff', '#ffff00', '#6699ff']

# 缩放限制
ZOOM_MIN_RANGE = 10    # 最小可见像素范围
ZOOM_MAX_SCALE = 5.0   # 最大缩小倍数（相对原始）

# 日志最大行数
LOG_MAX_LINES = 400

# 滑动条防抖延迟(ms)
SLIDER_DEBOUNCE_MS = 80

# 后台分析分块预算：每块像素数上限（float64 约 32MB）与帧数上限，取小者。
# 按像素预算而非固定帧数，避免大 ROI 时单块内存随面积线性膨胀导致 OOM。
ANALYSIS_MAX_ELEMS_PER_CHUNK = 4_000_000
ANALYSIS_MAX_CHUNK_FRAMES = 256

# 结果曲线抽稀目标点数（长序列直接全量绘制会让 matplotlib 卡顿且不可读）
PLOT_MAX_POINTS = 20000

# 稳健 Michelson 的 1%/99% 分位在像素数低于该阈值时无法有效排除
# 单像素离群（n=9 时 p99 仍含 ~92% 权重的最大值），届时给出提示。
ROBUST_MIN_PIXELS = 100

# 压缩 TIFF 整卷载入前的确认阈值
LOAD_CONFIRM_BYTES = 2 * 1024 ** 3

# 优先使用的中文字体（按顺序取第一个系统已安装者）
CJK_FONT_CANDIDATES = (
    'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Source Han Sans SC',
    'WenQuanYi Zen Hei', 'PingFang SC', 'Heiti SC', 'SimSun',
)


def setup_matplotlib_fonts() -> Tuple[str, str]:
    """选择可用的中文字体，返回 (生效字体名, 告警文本)。

    直接使用 ``['Arial', 'SimHei']`` 时 matplotlib 不会逐字形回退到
    SimHei，中文标题/轴标签会渲染成方框（实测 3.10.x 仍如此），因此这里
    显式挑选系统里第一个可用的中文字体并置于首位。
    """
    plt.rcParams['axes.unicode_minus'] = False
    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        available = set()
    for name in CJK_FONT_CANDIDATES:
        if name in available:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            return name, ""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return "", ("未找到可用中文字体，图表中的中文将显示为方框；"
                "请安装 SimHei / 微软雅黑后重启程序")


def style_axes_dark(ax) -> None:
    """将 axes 配置为深色主题样式"""
    ax.set_facecolor(DARK_AXES)
    ax.tick_params(colors=LIGHT_TEXT)
    ax.xaxis.label.set_color(LIGHT_TEXT)
    ax.yaxis.label.set_color(LIGHT_TEXT)
    ax.title.set_color(TITLE_TEXT)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)


def format_number(value) -> str:
    """CSV 数值格式化：非有限值一律写 "NaN"（下游应理解为「无有效衬度」）。"""
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return "NaN"
    if not np.isfinite(fv):
        return "NaN"
    return repr(fv)


class TIFContrastAnalyzer:
    """TIF图像衬度分析工具主类"""

    def __init__(self, root: ttkb.Window) -> None:
        self.root = root
        self.root.title(f"TIF图像衬度分析工具 v{__version__}")
        self._setup_window_size()

        # --- 核心数据 ---
        self.tif_memmap: Optional[np.ndarray] = None
        self.tif_path: Optional[str] = None
        self.total_frames: int = 0
        self.current_frame: int = 0
        self.roi_list: List[Tuple[int, int, int, int]] = []
        self.contrast_data: Optional[List[List[float]]] = None
        # 每帧强度统计：{'mean': [[...]], 'sigma': ..., 'min': ..., 'max': ...}
        self.contrast_stats: Optional[Dict[str, List[List[float]]]] = None
        self._probe: Optional[TiffProbe] = None
        # 逐帧 DateTime 标签（长度等于帧数时启用 CSV 时间列，否则为空）
        self._frame_datetimes: List[str] = []
        # 当前数据实际使用的『多帧解析』取值（重载失败时回滚复选框用）
        self._applied_force_frames: bool = False
        # 多帧解析重载（同一文件换解析方式）时暂存的 ROI 保留请求
        self._reload_preserve: Optional[Dict] = None

        # 图像显示状态：持久 AxesImage 与 ROI artist，切帧只更新数据以保留视图
        self._im_img = None
        self._roi_artists: List = []
        self._stack_clim: Optional[Tuple[float, float]] = None  # 全堆栈显示灰度范围
        self._lock_display_var = tk.BooleanVar(value=True)      # 默认锁定显示范围
        # planar 彩色单页（3/4 个平面）是否按多帧解析
        self._force_frames_var = tk.BooleanVar(value=False)

        # 交互状态
        self.dragging_roi: bool = False
        self.roi_start: Optional[Tuple[float, float]] = None
        self.temp_rect: Optional[Rectangle] = None
        self._blit_bg = None  # blit 背景缓存

        # 后台线程通信（仅在有待处理任务时轮询，避免常驻定时器）
        self._task_queue: queue.Queue = queue.Queue()
        self._analysis_running: bool = False
        self._loading: bool = False
        self._poll_id: Optional[str] = None
        self._cancel_event: threading.Event = threading.Event()
        self._analysis_total_rois: int = 0   # 分析启动时的ROI快照数量
        self._analysis_total_frames: int = 0  # 分析启动时的帧数快照
        # 分析启动时保存的算法快照（分析期间下拉框禁用，完成/摘要只读快照）
        self._analysis_method_idx: int = 0
        self._analysis_method_key: str = ALGORITHMS[0][1]
        self._analysis_method_name: str = ALGORITHMS[0][0]

        # 滑动条防抖
        self._slider_debounce_id: Optional[str] = None
        self._pending_frame: int = 0

        # 记忆上次打开目录
        self._last_dir: str = os.path.expanduser("~")

        # 记录初始视图范围
        self._img_extent: Optional[Tuple[float, float, float, float]] = None

        # 中文字体（影响图内标题/轴标签，必须在创建 Figure 前设置）
        self.font_name, self.font_warning = setup_matplotlib_fonts()

        self._create_ui()
        self._create_menu()

        # 键盘快捷键：方向键在文本输入/列表控件聚焦时不劫持
        self.root.bind("<Left>", lambda e: self._step_frame(-1))
        self.root.bind("<Right>", lambda e: self._step_frame(1))
        # Ctrl+S 快捷导出（_export_data 内部自带 busy/无数据守卫）
        self.root.bind("<Control-s>", lambda e: self._export_data())
        # 分析完成后切换算法：结果不会自动失效（导出仍用启动时快照，
        # 这是正确行为），但需要显式提醒用户重新分析。
        self.contrast_method.bind('<<ComboboxSelected>>', self._on_method_changed)

        # 窗口关闭时清理资源
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._refresh_widget_states()
        if self.font_warning:
            self._log(f"⚠ {self.font_warning}\n")
        self._log(f"就绪 | 图内中文字体: {self.font_name or '未找到（中文可能显示为方框）'}\n")

    def _setup_window_size(self) -> None:
        """自适应窗口大小"""
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        w = min(1400, screen_w - 100)
        h = min(900, screen_h - 100)
        x = (screen_w - w) // 2
        y = (screen_h - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _create_menu(self) -> None:
        """创建菜单栏（使用深色主题常量）"""
        menubar = tk.Menu(self.root, bg=MENU_BG, fg=MENU_FG,
                          activebackground=MENU_ACTIVE_BG, activeforeground=MENU_ACTIVE_FG)

        file_menu = tk.Menu(menubar, tearoff=0, bg=MENU_BG, fg=MENU_FG)
        file_menu.add_command(label="打开TIF文件 (Ctrl+O)", command=self._load_tif)
        file_menu.add_command(label="导出CSV数据 (Ctrl+S)", command=self._export_data)
        file_menu.add_separator()
        file_menu.add_command(label="导入ROI坐标 (JSON)", command=self._import_rois)
        file_menu.add_command(label="导出ROI坐标 (JSON)", command=self._export_rois)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self._on_close)
        menubar.add_cascade(label="文件", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=0, bg=MENU_BG, fg=MENU_FG)
        edit_menu.add_command(label="撤销上一个ROI (Ctrl+Z)", command=self._delete_last_roi)
        edit_menu.add_command(label="清空所有ROI", command=self._clear_roi)
        edit_menu.add_separator()
        edit_menu.add_command(label="复制信息面板内容", command=self._copy_info)
        edit_menu.add_command(label="重置视图 (双击图像)", command=self._reset_view)
        menubar.add_cascade(label="编辑", menu=edit_menu)

        help_menu = tk.Menu(menubar, tearoff=0, bg=MENU_BG, fg=MENU_FG)
        help_menu.add_command(label="快捷键说明", command=self._show_shortcuts)
        help_menu.add_command(label="关于", command=self._show_about)
        menubar.add_cascade(label="帮助", menu=help_menu)

        self.root.config(menu=menubar)
        self.root.bind("<Control-o>", lambda e: self._load_tif())

    def _create_ui(self) -> None:
        """创建主界面。

        工具栏分两行：第一行是文件级操作，第二行是算法/解析选项与进度条。
        两行布局确保进度条在任何窗口宽度下都可见（此前单行工具栏在默认
        1400px 宽度下溢出，进度条被整条裁掉、复选框文字被截断）。
        """
        toolbar = ttk.Frame(self.root, padding=(8, 6, 8, 2))
        toolbar.pack(side=tk.TOP, fill=tk.X)

        row1 = ttk.Frame(toolbar)
        row1.pack(side=tk.TOP, fill=tk.X)

        self.btn_load = ttkb.Button(row1, text="📂 打开TIF堆叠",
                                    command=self._load_tif, bootstyle="primary")
        self.btn_load.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_analyze = ttkb.Button(row1, text="📊 分析衬度",
                                       command=self._analyze_contrast, bootstyle="success")
        self.btn_analyze.pack(side=tk.LEFT, padx=4)

        # 取消按钮：⏹(U+23F9) 在 Tk 默认字体中缺字形会显示为方框，改用 ■
        self.btn_cancel = ttkb.Button(row1, text="■ 取消",
                                      command=self._cancel_analysis, bootstyle="secondary-outline")
        self.btn_cancel.pack(side=tk.LEFT, padx=4)

        self.btn_export = ttkb.Button(row1, text="💾 导出CSV",
                                      command=self._export_data, bootstyle="warning")
        self.btn_export.pack(side=tk.LEFT, padx=4)

        self.btn_clear = ttkb.Button(row1, text="🗑 清空ROI",
                                     command=self._clear_roi, bootstyle="danger-outline")
        self.btn_clear.pack(side=tk.RIGHT, padx=4)

        row2 = ttk.Frame(toolbar)
        row2.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))

        ttk.Label(row2, text="算法:").pack(side=tk.LEFT, padx=(0, 4))
        self.contrast_method = ttk.Combobox(
            row2,
            values=[name for name, _ in ALGORITHMS],
            width=15, state='readonly')
        self.contrast_method.current(0)
        self.contrast_method.pack(side=tk.LEFT)

        ttkb.Checkbutton(row2, text="🔒锁定灰度范围(全堆栈)",
                         variable=self._lock_display_var,
                         command=self._on_toggle_display_lock,
                         bootstyle="info").pack(side=tk.LEFT, padx=(12, 4))

        self.chk_force_frames = ttkb.Checkbutton(
            row2, text="🧩多帧解析(把颜色平面当帧)",
            variable=self._force_frames_var,
            command=self._on_toggle_force_frames,
            bootstyle="info")
        self.chk_force_frames.pack(side=tk.LEFT, padx=(8, 4))

        # 进度条最后打包并 fill+expand：保证它拿到剩余全部宽度，永不被裁掉
        self.progress = ttk.Progressbar(row2, length=140, mode='determinate')
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(12, 0))

        # 2. 主体布局
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # --- 左侧：图像显示 ---
        left_p = ttk.Frame(paned)
        self.fig_img = Figure(figsize=(7, 7), facecolor=DARK_BG)
        self.ax_img = self.fig_img.add_subplot(111)
        style_axes_dark(self.ax_img)
        self.canvas_img = FigureCanvasTkAgg(self.fig_img, master=left_p)
        self.canvas_img.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 状态栏先打包到 BOTTOM，确保它位于最底部（后打包的帧控制条在其上方）
        self.status_var = tk.StringVar(
            value="就绪 | 左键拖拽绘制ROI | 右键删除ROI | 滚轮缩放 | 双击重置视图")
        status_bar = ttk.Label(left_p, textvariable=self.status_var,
                               font=('Consolas', 9), foreground='#888888')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=8)

        # 帧控制区
        controls = ttk.Frame(left_p, padding=(8, 6))
        controls.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Label(controls, text="帧:").pack(side=tk.LEFT, padx=(4, 4))
        self.frame_slider = tk.Scale(controls, from_=1, to=1, orient=tk.HORIZONTAL,
                                     command=self._on_slider_move, showvalue=True,
                                     resolution=1,
                                     bg=DARK_BG, fg=LIGHT_TEXT, troughcolor='#444444',
                                     highlightthickness=0, bd=2)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.frame_slider.config(state=tk.DISABLED)

        # 帧跳转
        jump_frame = ttk.Frame(controls)
        jump_frame.pack(side=tk.RIGHT, padx=6)
        ttk.Label(jump_frame, text="跳转:").pack(side=tk.LEFT, padx=(0, 4))
        self.jump_input = ttk.Entry(jump_frame, width=6)
        self.jump_input.pack(side=tk.LEFT, padx=2)
        self.jump_input.bind("<Return>", lambda e: self._jump_to_frame())
        ttkb.Button(jump_frame, text="GO", command=self._jump_to_frame,
                    width=4, bootstyle="info-outline").pack(side=tk.LEFT, padx=(4, 0))

        # 画布事件绑定
        self.canvas_img.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas_img.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas_img.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas_img.mpl_connect('scroll_event', self._on_scroll)
        self.canvas_img.mpl_connect('button_press_event', self._on_double_click)
        # 窗口尺寸变化后旧的 blit 背景失效，需作废避免拖拽残影。
        # add="+" 必须保留：matplotlib 在同一个 widget 上也绑定了 <Configure>
        # （FigureCanvasTk.resize），Tkinter 默认会用新回调**替换**旧绑定，
        # 一旦覆盖，Figure 就再也不会跟随控件尺寸变化，画面会被裁切。
        self.canvas_img.get_tk_widget().bind("<Configure>", self._on_canvas_resize, add="+")
        # Ctrl+Z 仅作用于图像画布或 ROI 列表；Entry 焦点下保留原生撤销。
        self.canvas_img.get_tk_widget().bind("<Control-z>", lambda e: self._delete_last_roi())

        paned.add(left_p)

        # --- 右侧：结果面板 ---
        right_p = ttk.Frame(paned)
        self.fig_plot = Figure(figsize=(5, 4), facecolor=DARK_BG)
        self.ax_plot = self.fig_plot.add_subplot(111)
        style_axes_dark(self.ax_plot)
        self._set_plot_placeholder("等待分析")
        self.canvas_plot = FigureCanvasTkAgg(self.fig_plot, master=right_p)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ROI 列表面板
        roi_frame = ttk.LabelFrame(right_p, text="ROI 列表 (选中后按Del删除)", padding=4)
        roi_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
        self.roi_listbox = tk.Listbox(roi_frame, height=4, font=('Consolas', 9),
                                      bg=DARK_AXES, fg=LIGHT_TEXT,
                                      selectbackground='#444444', relief=tk.FLAT)
        self.roi_listbox.pack(fill=tk.X)
        # Delete 仅在 Listbox 获得焦点时生效
        self.roi_listbox.bind("<Delete>", lambda e: self._delete_selected_roi())
        self.roi_listbox.bind("<Control-z>", lambda e: self._delete_last_roi())

        # 信息面板（只读：展示日志与统计摘要，防止误编辑）
        info_head = ttk.Frame(right_p)
        info_head.pack(fill=tk.X, padx=5, pady=(5, 0))
        ttk.Label(info_head, text="信息 / 统计（可复制）").pack(side=tk.LEFT)
        self.btn_copy_info = ttkb.Button(info_head, text="📋 复制",
                                         command=self._copy_info,
                                         bootstyle="secondary-outline")
        self.btn_copy_info.pack(side=tk.RIGHT)

        self.info_box = tk.Text(right_p, height=8, font=('Consolas', 10),
                                bg=DARK_AXES, fg=LIGHT_TEXT,
                                insertbackground=LIGHT_TEXT,
                                relief=tk.FLAT, padx=6, pady=6,
                                state=tk.DISABLED)
        self.info_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=(2, 5))

        paned.add(right_p)

    # ==================== 运行状态 ====================

    def _busy(self) -> bool:
        """是否有后台任务进行中（加载或分析）。"""
        return self._loading or self._analysis_running

    def _refresh_widget_states(self) -> None:
        """统一刷新控件可用性（与真实运行态保持一致）。"""
        busy = self._busy()
        has_file = self.tif_memmap is not None
        has_data = self.contrast_data is not None

        self.btn_load.config(state=tk.DISABLED if busy else tk.NORMAL)
        self.btn_analyze.config(state=tk.DISABLED if busy else tk.NORMAL)
        self.btn_cancel.config(state=tk.NORMAL if self._analysis_running else tk.DISABLED)
        self.btn_export.config(state=tk.NORMAL if (has_data and not busy) else tk.DISABLED)
        self.btn_clear.config(state=tk.NORMAL if (self.roi_list and not busy) else tk.DISABLED)
        self.chk_force_frames.config(state=tk.DISABLED if busy else tk.NORMAL)
        self.contrast_method.config(state=tk.DISABLED if self._analysis_running else 'readonly')
        # 帧浏览是只读操作，分析/加载期间仍可用
        self.frame_slider.config(state=tk.NORMAL if has_file else tk.DISABLED)

    def _reject_if_busy(self, what: str = "编辑ROI") -> bool:
        if self._busy():
            stage = "加载" if self._loading else "分析"
            self.status_var.set(f"正在{stage}，暂不可{what}（可先取消分析）")
            return True
        return False

    # ==================== 文件操作 ====================

    def _load_tif(self) -> None:
        """选择并加载TIF文件（头部探测在主线程，像素读取在后台线程）"""
        if self._busy():
            messagebox.showwarning("提示", "正在加载或分析中，请等待完成或取消后再加载新文件")
            return
        path = filedialog.askopenfilename(
            title="选择TIF堆叠文件",
            initialdir=self._last_dir,
            filetypes=[("TIFF文件", "*.tif;*.tiff"), ("所有文件", "*.*")])
        if not path:
            return
        self._start_load(path, ask_confirm=True)

    def _start_load(self, path: str, ask_confirm: bool = True) -> bool:
        """探测头部 -> （必要时确认）-> 启动后台加载；返回是否真正启动。"""
        try:
            probe = probe_tiff(path)
        except TiffFileError as e:
            messagebox.showerror("格式错误", f"不是有效的 TIFF 文件：\n{e}")
            self._reload_preserve = None
            return False
        except FileNotFoundError:
            messagebox.showerror("错误", "文件不存在，请检查路径")
            self._reload_preserve = None
            return False
        except PermissionError:
            messagebox.showerror("错误", "文件被占用或无读取权限，请关闭其他程序后重试")
            self._reload_preserve = None
            return False
        except Exception as e:
            messagebox.showerror("加载失败", f"读取文件头失败: {e}")
            self._reload_preserve = None
            return False

        # 压缩 TIFF 需整卷载入内存，体积过大时先确认（加载前告知，避免静默 OOM）。
        # fail-safe（工单26 修法3）：元数据解析失败时 nbytes 可能为 0，护栏条件
        # `nbytes > 阈值` 会静默退化为 False；因此体积未知即视为超限、必须确认。
        nbytes_known = probe.nbytes > 0
        if ask_confirm and probe.is_compressed and (
                not nbytes_known or probe.nbytes > LOAD_CONFIRM_BYTES):
            size_line = (
                f"预计占用约 {probe.size_text}（{probe.shape_text}, {probe.dtype_str}）。"
                if nbytes_known else
                "堆栈体积无法读取（元数据解析失败），请确认系统内存充足。")
            ok = messagebox.askyesno(
                "确认加载",
                f"该文件为 {probe.compression} 压缩，无法内存映射，需整卷读入内存。\n\n"
                f"{size_line}\n"
                f"继续加载？")
            if not ok:
                self.status_var.set("已取消加载")
                self._reload_preserve = None
                return False

        force_frames = bool(self._force_frames_var.get())
        self._loading = True
        self._refresh_widget_states()
        # 加载阶段没有可度量的进度：进度条切到流动模式，避免残留上一次
        # 分析的读数误导用户（此前会冻结在旧值上）。
        self._progress_set_indeterminate(True)
        self.status_var.set(f"正在加载 {os.path.basename(path)} …（大文件可能需要数十秒）")
        self._log(f"\n{'=' * 46}\n→ 加载: {path}\n"
                  f"  头部: axes={probe.axes or '未知'} 形状={probe.shape_text} "
                  f"类型={probe.dtype_str} 页数={probe.n_pages} "
                  f"压缩={probe.compression} 预计体积={probe.size_text}\n")
        if probe.degraded:
            # 元数据解析失败时显式告知，字段空值/0 不再被误读为真实值（工单26 修法2）
            detail = "; ".join(probe.probe_errors[:3])
            if len(probe.probe_errors) > 3:
                detail += f"（等共 {len(probe.probe_errors)} 项）"
            self._log(f"  ⚠ 元数据解析失败（{len(probe.probe_errors)} 项）："
                      f"上面显示为 未知/0 的字段不可用: {detail}\n")
        threading.Thread(target=self._load_worker,
                         args=(path, probe, force_frames), daemon=True).start()
        self._ensure_polling()
        return True

    def _progress_set_indeterminate(self, on: bool) -> None:
        """加载等不可度量阶段使用流动进度条；结束时恢复确定模式并清零。"""
        try:
            if on:
                self.progress.config(mode='indeterminate')
                self.progress.start(20)
            else:
                self.progress.stop()
                self.progress.config(mode='determinate')
                self.progress['value'] = 0
        except tk.TclError:
            pass

    def _load_worker(self, path: str, probe: TiffProbe, force_frames: bool) -> None:
        """后台加载：读取像素 + 估计显示范围 + 逐页时间标签（都在后台）。"""
        try:
            arr, mode, notes = open_tiff_stack_ex(path, probe=probe,
                                                  force_frames=force_frames)
            if arr.ndim != 3:
                raise ValueError(
                    f"规范化后应为 (T, H, W) 三维灰度堆叠，实际得到 {arr.shape}")
            if arr.shape[0] == 0:
                raise ValueError("文件中没有可用图像帧（帧数为 0）")
            clim = estimate_display_clim(arr)
            datetimes = page_datetimes(path)
            self._task_queue.put(('loaded', (path, arr, mode, notes, clim, probe,
                                             force_frames, datetimes)))
        except Exception as e:  # 含 MemoryError / ValueError / OSError
            self._task_queue.put(('load_error', (type(e).__name__, str(e))))

    def _on_load_complete(self, payload) -> None:
        """加载成功：一次性切换状态（此前的文件状态在此之前保持不变）。"""
        path, arr, mode, notes, clim, probe, force_used, datetimes = payload
        if not self.root.winfo_exists():
            return
        self._loading = False
        self._progress_set_indeterminate(False)

        # 释放旧文件句柄后再接管新数组
        if self.tif_memmap is not None:
            del self.tif_memmap
        self.tif_memmap = arr
        self.tif_path = path
        self._probe = probe
        self._last_dir = os.path.dirname(path)
        self._applied_force_frames = bool(force_used)
        # 页级 DateTime 与帧数一致时才启用（planar 多帧解析等场景页数≠帧数）
        self._frame_datetimes = (list(datetimes)
                                 if datetimes and len(datetimes) == int(arr.shape[0])
                                 else [])

        self.total_frames = int(arr.shape[0])
        self.current_frame = 0
        self._pending_frame = 0
        self.frame_slider.config(from_=1, to=self.total_frames, resolution=1)
        self.frame_slider.set(1)

        # 同一文件仅切换解析方式（多帧解析）时保留 ROI，越界裁剪；
        # 打开新文件则不复用旧 ROI / 旧结果。
        preserve = self._reload_preserve
        self._reload_preserve = None
        kept_rois: List[Tuple[int, int, int, int]] = []
        if preserve is not None:
            for (x1, y1, x2, y2) in preserve["rois"]:
                c = clamp_roi_to_image(x1, y1, x2, y2,
                                       h=int(arr.shape[1]), w=int(arr.shape[2]))
                if (c[2] - c[0]) > 2 and (c[3] - c[1]) > 2:
                    kept_rois.append(c)
        self.roi_list = kept_rois
        self.contrast_data = None
        self.contrast_stats = None
        self.temp_rect = None
        self.dragging_roi = False

        self._stack_clim = clim
        self._reset_image_axes()
        self._update_roi_listbox()
        self._show_frame()
        self._clear_result_panel("等待分析" if not kept_rois else "ROI已保留，可直接分析")

        h, w = arr.shape[1], arr.shape[2]
        self._log(f"✓ 文件: {os.path.basename(path)}\n"
                  f"  帧数: {self.total_frames}\n"
                  f"  尺寸: {h}×{w}\n"
                  f"  类型: {arr.dtype}\n"
                  f"  加载: {mode}\n")
        for note in notes:
            self._log(f"  说明: {note}\n")
        if not notes:
            self._log("  说明: 灰度堆叠，未做颜色转换\n")
        if self._frame_datetimes:
            self._log(f"  说明: 检测到逐页 DateTime 标签，将随 CSV 导出（{len(self._frame_datetimes)} 帧）\n")
        if preserve is not None:
            self._log(f"  说明: 多帧解析切换，已保留 {len(kept_rois)}/{len(preserve['rois'])} 个 ROI（越界部分已裁剪）\n")
        self.status_var.set(
            f"已加载: {os.path.basename(path)} ({self.total_frames}帧, {h}×{w})")
        self._refresh_widget_states()

    def _on_load_error(self, payload) -> None:
        """加载失败：保留原文件与结果，明确告知用户。"""
        kind, msg = payload
        self._loading = False
        self._progress_set_indeterminate(False)
        title, friendly = self._describe_load_error(kind, msg)
        kept = "（已保留当前文件与结果）" if self.tif_memmap is not None else ""
        # 多帧解析重载失败：复选框回到当前数据实际使用的取值，
        # 避免控件状态与真实解析方式静默不一致。
        if self._reload_preserve is not None:
            self._reload_preserve = None
            self._force_frames_var.set(self._applied_force_frames)
            kept = "（已恢复原解析方式，文件与结果未变）"
        self.status_var.set(f"加载失败{kept}")
        self._log(f"✗ 加载失败: {friendly}\n")
        self._refresh_widget_states()
        messagebox.showerror(title, friendly)

    @staticmethod
    def _describe_load_error(kind: str, msg: str) -> Tuple[str, str]:
        """把异常映射为面向用户的标题与说明。"""
        if kind == 'TiffFileError':
            return "格式错误", f"不是有效的 TIFF 文件（文件头无法解析）。\n\n原始信息: {msg}"
        if kind == 'MemoryError':
            return "内存不足", ("文件过大，内存不足。\n\n建议："
                              "① 使用无压缩 TIFF（可内存映射，不整卷载入）；"
                              "② 先裁剪或降采样后再分析。")
        if kind == 'FileNotFoundError':
            return "错误", "文件不存在，请检查路径"
        if kind == 'PermissionError':
            return "错误", "文件被占用或无读取权限，请关闭其他程序后重试"
        if kind == 'ValueError':
            return "格式错误", f"无法解析文件结构: {msg}"
        return "加载失败", f"未知错误（{kind}）: {msg}"

    def _on_toggle_force_frames(self) -> None:
        """切换『多帧解析』：重新解释当前文件（planar 颜色平面按帧读取）。

        像素内容与图像尺寸不变、只有帧语义变化，因此 ROI 予以保留
        （越界自动裁剪）；重载失败则复选框回滚到当前数据实际取值。
        """
        if self._busy():
            return
        if self.tif_path is None:
            self.status_var.set(
                "多帧解析：下一个打开的文件将按该方式解析（planar 颜色平面当作帧）")
            return
        self._reload_preserve = {"rois": list(self.roi_list)}
        self.status_var.set("正在按新解析方式重新加载当前文件…（ROI 将保留）")
        if not self._start_load(self.tif_path, ask_confirm=False):
            self._reload_preserve = None

    # ==================== 导出 ====================

    def _export_data(self) -> None:
        """导出分析结果为CSV：帧号从1开始，文件头带元数据注释块。"""
        if self.contrast_data is None:
            messagebox.showinfo("提示", "请先进行衬度分析")
            return
        if self._busy():
            messagebox.showinfo("提示", "正在加载或分析中，请等待完成后再导出")
            return
        path = filedialog.asksaveasfilename(
            title="导出CSV",
            defaultextension=".csv",
            initialdir=self._last_dir,
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")])
        if not path:
            return
        if not self._warn_if_dir_not_writable(path):
            return
        try:
            n_frames, n_rois = self.write_csv(path)
            self._log(f"✓ 数据已导出: {os.path.basename(path)}"
                      f"（{n_frames} 帧 × {n_rois} ROI，空值以 NaN 表示）\n")
            messagebox.showinfo("导出成功", f"已保存至:\n{path}")
        except PermissionError:
            messagebox.showerror("导出失败",
                                 "文件被占用或无写入权限（若已在 Excel 中打开，请先关闭）")
        except Exception as e:
            messagebox.showerror("导出失败", f"写入文件时出错: {e}")

    @staticmethod
    def _warn_if_dir_not_writable(path: str) -> bool:
        """写入前预检目标目录可写（不可写时直接提示，代替事后的 OS 异常）。"""
        directory = os.path.dirname(os.path.abspath(path)) or "."
        try:
            if not os.access(directory, os.W_OK):
                messagebox.showerror("导出失败", f"目标目录无写入权限:\n{directory}")
                return False
        except OSError:
            pass  # 预检失败不拦截，交给实际写入时的异常处理
        return True

    def write_csv(self, path: str) -> Tuple[int, int]:
        """把当前分析结果写入 CSV，返回 (帧数, ROI 数)。

        与对话框解耦，便于自检/脚本直接调用；异常向上抛出由调用方处理。
        """
        results = self.contrast_data or []
        n_rois = len(results)
        n_frames = len(results[0]) if n_rois else 0
        stats = self.contrast_stats or {}
        stat_keys = [k for k in ("mean", "sigma", "min", "max") if stats.get(k)]
        # 逐帧 DateTime 标签（页级 tag 306）：与帧数一致时作为第二列导出，
        # 让时间序列分析可以直接使用真实采集时刻而非帧序号。
        datetimes = self._frame_datetimes
        use_dt = bool(datetimes) and len(datetimes) == n_frames

        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            f.write(self._build_csv_header(n_frames, n_rois))
            writer = csv.writer(f, lineterminator="\n")
            header = (["Frame"] + (["DateTime"] if use_dt else [])
                      + [f"ROI_{i + 1}" for i in range(n_rois)])
            for key in stat_keys:
                header += [f"ROI_{i + 1}_{key}" for i in range(n_rois)]
            writer.writerow(header)
            for fi in range(n_frames):
                row = ([fi + 1] + ([datetimes[fi]] if use_dt else [])
                       + [format_number(results[i][fi]) for i in range(n_rois)])
                for key in stat_keys:
                    col = stats[key]
                    row += [format_number(col[i][fi])
                            if i < len(col) and fi < len(col[i]) else "NaN"
                            for i in range(n_rois)]
                writer.writerow(row)
            f.write(self._build_csv_trailer(results))
        return n_frames, n_rois

    def _build_csv_header(self, n_frames: int, n_rois: int) -> str:
        """生成 CSV 头部元数据注释块（# 开头，pandas 可用 comment='#' 跳过）。"""
        lines = [
            f"# 软件: TIF图像衬度分析工具 v{__version__}",
            f"# 导出时间: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')}",
            f"# 源文件: {self.tif_path if self.tif_path else '未知'}",
            f"# 算法: {self._analysis_method_name} ({self._analysis_method_key})",
            f"# 帧数: {n_frames}",
        ]
        if self.tif_memmap is not None:
            h, w = self.tif_memmap.shape[1], self.tif_memmap.shape[2]
            lines.append(f"# 图像尺寸: {h}x{w} (像素)")
            lines.append(f"# 像素类型: {self.tif_memmap.dtype}")
        lines += format_metadata_lines(self._probe)
        # 分析完成后 ROI 编辑会使结果失效，因此 contrast_data 有效时
        # roi_list 与分析时的快照一致；不一致时跳过坐标行以防误导。
        if len(self.roi_list) == n_rois:
            # 像素尺寸启发式解析成功时，顺带给出 ROI 物理面积估算
            px_nm = pixel_size_nm(self._probe.pixel_size_text) if self._probe else None
            lines.append("# ROI 坐标 (0基像素, [xmin,ymin]-[xmax,ymax]):")
            for i, (x1, y1, x2, y2) in enumerate(self.roi_list):
                area_part = ""
                if px_nm is not None:
                    area_nm2 = (x2 - x1) * (y2 - y1) * px_nm * px_nm
                    if area_nm2 >= 1e6:
                        area_part = f", 面积≈{area_nm2 / 1e6:.4g} µm²"
                    else:
                        area_part = f", 面积≈{area_nm2:.4g} nm²"
                lines.append(
                    f"#   ROI_{i + 1}: ({x1},{y1})-({x2},{y2}) "
                    f"[{x2 - x1}x{y2 - y1}px{area_part}]")
            if px_nm is not None:
                lines.append("#   （面积按启发式像素尺寸估算，请与原始元数据核对）")
        lines.append("# ROI 取整规则: 下界 floor、上界 ceil（覆盖所有被拖拽矩形触及的像素）；")
        lines.append("#   xmax/ymax 为开区间端点，对应 img[ymin:ymax, xmin:xmax]，"
                     "即实际像素数为 (xmax-xmin)×(ymax-ymin)。")
        lines.append("# 列说明: ROI_i 为衬度值；ROI_i_mean/sigma/min/max 为同一 ROI "
                     "同一帧的原始强度统计（可用于区分结构变化与束流/厚度变化）。")
        lines.append("# 统计口径: σ 为总体标准差（ddof=0），与样本标准差（ddof=1）数值不同；")
        lines.append("#   DateTime 列（若存在）为 TIFF 页级 DateTime 标签原文。")
        lines.append("# 帧号从 1 开始；NaN 表示该帧无有效衬度（如平均强度为 0），不是数值 0。")
        lines.append("# 读取示例: pandas.read_csv(path, comment='#', index_col='Frame')")
        return "\n".join(lines) + "\n"

    def _build_csv_trailer(self, results: List[List[float]]) -> str:
        """生成 CSV 末尾的跨帧统计注释块（同样以 # 开头，不干扰数据解析）。"""
        lines = ["# ── 每 ROI 跨帧统计（仅统计非 NaN 帧）──"]
        for i, vals in enumerate(results):
            arr = np.asarray(vals, dtype=np.float64) if len(vals) else np.array([])
            valid = arr[np.isfinite(arr)]
            if valid.size == 0:
                lines.append(f"# ROI_{i + 1}: 全部无效（NaN）")
                continue
            lines.append(
                f"# ROI_{i + 1}: 有效帧 {valid.size}/{arr.size} "
                f"均值={float(np.mean(valid)):.6g} σ={float(np.std(valid)):.6g} "
                f"范围=[{float(np.min(valid)):.6g}, {float(np.max(valid)):.6g}]")
        return "\n".join(lines) + "\n"

    def _export_rois(self) -> None:
        """导出当前 ROI 坐标为 JSON（便于同一采集条件下复用同一批 ROI）。"""
        if not self.roi_list:
            messagebox.showinfo("提示", "当前没有 ROI 可导出")
            return
        path = filedialog.asksaveasfilename(
            title="导出ROI坐标",
            defaultextension=".json",
            initialdir=self._last_dir,
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")])
        if not path:
            return
        if not self._warn_if_dir_not_writable(path):
            return
        payload = {
            "tool": "TIF图像衬度分析工具",
            "version": __version__,
            "source_file": self.tif_path or "",
            "image_shape": list(self.tif_memmap.shape) if self.tif_memmap is not None else [],
            "roi_format": "[xmin, ymin, xmax, ymax]，0 基像素，xmax/ymax 为开区间端点",
            "rois": [list(r) for r in self.roi_list],
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self._log(f"✓ ROI 坐标已导出: {os.path.basename(path)}"
                      f"（{len(self.roi_list)} 个）\n")
            messagebox.showinfo("导出成功", f"已保存 {len(self.roi_list)} 个 ROI 至:\n{path}")
        except Exception as e:
            messagebox.showerror("导出失败", f"写入文件时出错: {e}")

    def _import_rois(self) -> None:
        """从 JSON 导入 ROI 坐标（替换现有 ROI；越界会被 clamp，过小则跳过）。"""
        if self.tif_memmap is None:
            messagebox.showinfo("提示", "请先打开TIF文件")
            return
        if self._reject_if_busy("导入ROI"):
            return
        path = filedialog.askopenfilename(
            title="导入ROI坐标",
            initialdir=self._last_dir,
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("导入失败", f"无法读取 JSON 文件: {e}")
            return

        rois = data.get("rois") if isinstance(data, dict) else None
        if not isinstance(rois, list):
            messagebox.showerror("导入失败", "JSON 中缺少 rois 数组")
            return

        h, w = int(self.tif_memmap.shape[1]), int(self.tif_memmap.shape[2])
        kept: List[Tuple[int, int, int, int]] = []
        skipped = 0
        for item in rois:
            try:
                if not isinstance(item, (list, tuple)) or len(item) != 4:
                    raise ValueError
                coords = [float(v) for v in item]
                if not all(np.isfinite(coords)):
                    raise ValueError
                x1, y1, x2, y2 = clamp_roi_to_image(*coords, h=h, w=w)
            except (TypeError, ValueError):
                skipped += 1
                continue
            if (x2 - x1) > 2 and (y2 - y1) > 2:
                kept.append((x1, y1, x2, y2))
            else:
                skipped += 1

        if not kept:
            messagebox.showwarning("导入失败", "没有可用的 ROI（全部无效或尺寸 <3px）")
            return

        old = len(self.roi_list)
        self.roi_list = kept
        self.contrast_data = None
        self.contrast_stats = None
        self._update_roi_listbox()
        self._show_frame()
        self._clear_result_panel("ROI已变更，请重新分析")
        self._log(f"✓ 导入 ROI: {len(kept)} 个（替换原有 {old} 个，"
                  f"跳过 {skipped} 条无效/过小记录）\n")
        self.status_var.set(f"已导入 {len(kept)} 个 ROI（原有 ROI 已被替换）")
        self._refresh_widget_states()

    # ==================== 帧浏览 ====================

    def _goto_frame(self, frame_1based: int) -> None:
        """统一帧跳转入口：更新滑动条并显式走防抖渲染。

        部分 Tk 版本中 ``Scale.set()`` 不触发 ``-command``，若只调 set()
        会出现滑动条移动但画面不更新的问题，因此这里显式调用回调。
        若 set() 触发了 command，重复调度只是重置同一防抖定时器，无副作用。
        """
        if self.tif_memmap is None:
            return
        val = max(1, min(int(frame_1based), self.total_frames))
        self.frame_slider.set(val)
        self._on_slider_move(str(val))

    def _step_frame(self, delta: int) -> None:
        """逐帧步进（文本输入框/列表聚焦时不劫持方向键）"""
        if self.tif_memmap is None:
            return
        focus = self.root.focus_get()
        if isinstance(focus, (tk.Entry, ttk.Entry, tk.Text, tk.Listbox,
                              ttk.Combobox, tk.Scale)):
            return
        self._goto_frame(self.frame_slider.get() + delta)

    def _jump_to_frame(self) -> None:
        """跳转至指定帧（接受 "5"/"5.0"/"+5" 等可转 float 的写法）"""
        if self.tif_memmap is None:
            return
        try:
            val = int(round(float(self.jump_input.get().strip())))
        except ValueError:
            self.status_var.set("请输入有效的帧编号（整数）")
            return
        if 1 <= val <= self.total_frames:
            self._goto_frame(val)
        else:
            self.status_var.set(f"请输入 1~{self.total_frames} 之间的帧号")

    def _on_slider_move(self, val) -> None:
        """滑动条回调 — 防抖处理"""
        if self.tif_memmap is None:
            return
        self._pending_frame = int(float(val)) - 1
        if self._slider_debounce_id is not None:
            self.root.after_cancel(self._slider_debounce_id)
        self._slider_debounce_id = self.root.after(
            SLIDER_DEBOUNCE_MS, self._render_pending_frame)

    def _render_pending_frame(self) -> None:
        """防抖后实际渲染（帧号越界时收敛到有效范围，避免 after 回调抛异常）"""
        self._slider_debounce_id = None
        if self.tif_memmap is None:
            return
        last = max(0, self.total_frames - 1)
        self.current_frame = max(0, min(self._pending_frame, last))
        self._show_frame()

    def _show_frame(self) -> None:
        """显示当前帧（仅更新图像数据与ROI，保留缩放/平移视图）"""
        if self.tif_memmap is None:
            return
        self.current_frame = max(0, min(self.current_frame, self.total_frames - 1))
        frame = self.tif_memmap[self.current_frame]
        if self._im_img is None:
            # 首次显示：锁定模式下用全堆栈灰度范围，避免逐帧自动拉伸
            # 掩盖帧间的强度演化；未锁定则交给 matplotlib 逐帧自动拉伸。
            if self._lock_display_var.get() and self._stack_clim is not None:
                self._im_img = self.ax_img.imshow(
                    frame, cmap='gray', interpolation='none',
                    vmin=self._stack_clim[0], vmax=self._stack_clim[1])
            else:
                self._im_img = self.ax_img.imshow(frame, cmap='gray',
                                                  interpolation='none')
            h, w = frame.shape
            self._img_extent = (-0.5, w - 0.5, h - 0.5, -0.5)
        else:
            self._im_img.set_data(frame)
            if not self._lock_display_var.get():
                self._im_img.set_clim(*self._frame_clim(frame))
        self.ax_img.set_title(f"Frame {self.current_frame + 1} / {self.total_frames}",
                              fontsize=10)
        self._draw_all_rois()

        self.canvas_img.draw()
        self._blit_bg = self.canvas_img.copy_from_bbox(self.ax_img.bbox)

    @staticmethod
    def _frame_clim(frame: np.ndarray) -> Tuple[float, float]:
        """逐帧自动拉伸的灰度范围（均匀帧展开半级，避免退化）。"""
        vmin = float(np.min(frame))
        vmax = float(np.max(frame))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return (0.0, 1.0)
        if vmin >= vmax:
            return (vmin - 0.5, vmax + 0.5)
        return (vmin, vmax)

    def _reset_image_axes(self) -> None:
        """载入新文件时复位图像区：清除上一文件的所有 artist 与视图状态。"""
        self.ax_img.clear()
        style_axes_dark(self.ax_img)
        self._im_img = None
        self._roi_artists = []
        self._img_extent = None
        self._blit_bg = None

    def _set_plot_placeholder(self, text: str) -> None:
        """结果区占位提示（无数据时明确说明，而不是残留上一个文件的结果）。"""
        self.ax_plot.clear()
        style_axes_dark(self.ax_plot)
        self.ax_plot.set_title(text, fontsize=9, color='#888888')
        self.ax_plot.set_xticks([])
        self.ax_plot.set_yticks([])

    def _clear_result_panel(self, placeholder: str) -> None:
        """清空结果曲线区（换文件/ROI 变更时调用）。"""
        self._set_plot_placeholder(placeholder)
        self.canvas_plot.draw()

    def _on_toggle_display_lock(self) -> None:
        """切换显示范围锁定：立即按新模式重设当前帧对比度。"""
        if self.tif_memmap is None or self._im_img is None:
            return
        if self._lock_display_var.get():
            if self._stack_clim is not None:
                self._im_img.set_clim(*self._stack_clim)
        else:
            self._im_img.set_clim(*self._frame_clim(self.tif_memmap[self.current_frame]))
        self.canvas_img.draw()
        self._blit_bg = self.canvas_img.copy_from_bbox(self.ax_img.bbox)

    # ==================== ROI 交互 ====================

    def _on_mouse_press(self, event) -> None:
        """鼠标按下：左键开始绘制ROI，右键删除ROI"""
        if event.inaxes != self.ax_img or self.tif_memmap is None:
            return
        # 加载/分析期间禁止编辑ROI，避免面板与结果曲线不一致
        if self._reject_if_busy("编辑ROI"):
            return

        if event.button == 3:  # 右键：删除ROI
            self._delete_roi_at(event.xdata, event.ydata)
            return

        if event.button == 1:  # 左键：开始绘制
            # 双击的第二次 button_press 会带 dblclick=True；不要启动拖拽，
            # 交给 _on_double_click 执行重置视图，避免残留矩形。
            if getattr(event, 'dblclick', False):
                return
            self.dragging_roi = True
            self.roi_start = (event.xdata, event.ydata)
            self.temp_rect = Rectangle(self.roi_start, 0, 0,
                                       linewidth=1.5, edgecolor='#ff4444',
                                       facecolor='none', linestyle='--')
            self.ax_img.add_patch(self.temp_rect)

    def _on_double_click(self, event) -> None:
        """双击重置视图（互斥：先取消可能存在的拖拽）"""
        if event.inaxes != self.ax_img:
            return
        if getattr(event, 'dblclick', False):
            # 双击与拖拽状态机互斥：若已开始拖拽，先清理临时矩形。
            if self.dragging_roi:
                self.dragging_roi = False
                self._remove_temp_rect()
            self._reset_view()

    def _remove_temp_rect(self) -> None:
        """从画布移除临时ROI矩形并清空引用（幂等）。"""
        if self.temp_rect is not None:
            try:
                self.temp_rect.remove()
            except Exception:
                pass
            self.temp_rect = None

    def _reset_view(self) -> None:
        """重置图像视图到初始范围"""
        if self._img_extent is None:
            return
        self.ax_img.set_xlim(self._img_extent[0], self._img_extent[1])
        self.ax_img.set_ylim(self._img_extent[2], self._img_extent[3])
        self.canvas_img.draw()
        self._blit_bg = self.canvas_img.copy_from_bbox(self.ax_img.bbox)
        self.status_var.set("视图已重置")

    def _on_canvas_resize(self, event) -> None:
        """画布尺寸变化：作废 blit 背景缓存（下次拖拽自动回退全量重绘）"""
        self._blit_bg = None

    def _on_mouse_move(self, event) -> None:
        """鼠标移动：使用 blit 技术更新临时ROI矩形"""
        if not self.dragging_roi or event.inaxes != self.ax_img:
            return
        if event.xdata is None or event.ydata is None or self.temp_rect is None:
            return
        w = event.xdata - self.roi_start[0]
        h = event.ydata - self.roi_start[1]
        self.temp_rect.set_width(w)
        self.temp_rect.set_height(h)
        self.status_var.set(f"ROI尺寸: {abs(int(w))}×{abs(int(h))} px")

        # blit 重绘：恢复背景 + 仅重绘临时矩形
        if self._blit_bg is not None:
            self.canvas_img.restore_region(self._blit_bg)
            self.ax_img.draw_artist(self.temp_rect)
            self.canvas_img.blit(self.ax_img.bbox)
        else:
            # 全量重绘后回填缓存，后续移动恢复 blit 路径
            self.canvas_img.draw()
            self._blit_bg = self.canvas_img.copy_from_bbox(self.ax_img.bbox)

    def _on_mouse_release(self, event) -> None:
        """鼠标释放：确认ROI。

        释放在坐标区外（event.xdata 为 None）时，用拖拽期间最后一次
        画布内坐标绘制的矩形几何确认 ROI——等价于把终点 clamp 到图像
        边界，而不是直接丢弃整个拖拽成果。
        """
        if not self.dragging_roi:
            return
        self.dragging_roi = False

        if self.roi_start is None or (self.temp_rect is None
                                      and (event.xdata is None or event.ydata is None)):
            self._remove_temp_rect()
            self.canvas_img.draw()
            return

        x1, y1 = self.roi_start
        if event.xdata is None or event.ydata is None:
            bx, by = self.temp_rect.get_xy()
            x2 = bx + self.temp_rect.get_width()
            y2 = by + self.temp_rect.get_height()
        else:
            x2, y2 = event.xdata, event.ydata
        # 拖拽终点 clamp 集中处理：排序 + floor/ceil + 限制到 [0,w]×[0,h]
        h, w = self.tif_memmap.shape[1], self.tif_memmap.shape[2]
        xmin, ymin, xmax, ymax = clamp_roi_to_image(x1, y1, x2, y2, h, w)

        if (xmax - xmin) > 2 and (ymax - ymin) > 2:
            self.roi_list.append((xmin, ymin, xmax, ymax))
            # 临时矩形画布对象已无用，显式移除（无 ax.clear 兜底）
            self._remove_temp_rect()
            self._after_roi_change()
            self.status_var.set(
                f"已添加 ROI {len(self.roi_list)}: ({xmin},{ymin})-({xmax},{ymax})")
        else:
            self._remove_temp_rect()
            self.canvas_img.draw()
            self.status_var.set("ROI太小（<3px），已忽略")

    def _on_scroll(self, event) -> None:
        """滚轮缩放图像（带边界限制和 None 检查）"""
        if self.tif_memmap is None or event.inaxes != self.ax_img:
            return
        if event.xdata is None or event.ydata is None:
            return

        xlim = self.ax_img.get_xlim()
        ylim = self.ax_img.get_ylim()
        cur_w = xlim[1] - xlim[0]
        cur_h = ylim[1] - ylim[0]

        scale = 0.8 if event.button == 'up' else 1.25
        new_w = cur_w * scale
        new_h = cur_h * scale

        # 限制缩放范围
        if self._img_extent is not None:
            orig_w = self._img_extent[1] - self._img_extent[0]
            orig_h = self._img_extent[2] - self._img_extent[3]
            # 宽、高均不允许缩小到超过原始尺寸的 ZOOM_MAX_SCALE 倍
            if new_w > abs(orig_w) * ZOOM_MAX_SCALE or new_h > abs(orig_h) * ZOOM_MAX_SCALE:
                return
            # 不允许放大到小于 ZOOM_MIN_RANGE 像素
            if new_w < ZOOM_MIN_RANGE or new_h < ZOOM_MIN_RANGE:
                return

        x_center = event.xdata
        y_center = event.ydata
        self.ax_img.set_xlim(x_center - new_w / 2, x_center + new_w / 2)
        self.ax_img.set_ylim(y_center - new_h / 2, y_center + new_h / 2)
        self.canvas_img.draw()
        self._blit_bg = self.canvas_img.copy_from_bbox(self.ax_img.bbox)

    def _draw_all_rois(self) -> None:
        """绘制所有ROI（先移除旧artist；标签位置按边界自适应，避免越界）"""
        for artist in self._roi_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._roi_artists = []
        for i, (x1, y1, x2, y2) in enumerate(self.roi_list):
            color = ROI_COLORS[i % len(ROI_COLORS)]
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=2, edgecolor=color, facecolor='none')
            self.ax_img.add_patch(rect)
            lx, ly, ha, va = self._roi_label_position(x1, y1, x2, y2)
            label = self.ax_img.text(
                lx, ly, f"R{i + 1}",
                color=color, fontsize=9, fontweight='bold',
                horizontalalignment=ha, verticalalignment=va,
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor=DARK_AXES, alpha=0.7, edgecolor='none'))
            self._roi_artists.extend([rect, label])

    def _roi_label_position(self, x1, y1, x2, y2):
        """按 ROI 边界自适应返回标签锚点位置，避免标签落在图像外。

        标签高度按当前缩放级别由显示坐标换算到数据坐标（约 14 显示
        像素），深放大时不再用固定 12 数据像素的粗略近似。
        """
        xlim = self.ax_img.get_xlim()
        ylim = self.ax_img.get_ylim()
        try:
            inv = self.ax_img.transData.inverted()
            (_, dy0) = inv.transform((0.0, 0.0))
            (_, dy1) = inv.transform((0.0, 14.0))
            label_h = abs(dy1 - dy0)
        except Exception:
            label_h = 12  # 换算失败时退回固定近似

        # 优先放在左上角上方；放不下则放到矩形内部左上。
        if y1 - 5 > ylim[0] + label_h:
            ly, va = y1 - 5, 'bottom'
        else:
            ly, va = y1 + label_h, 'top'

        # x 方向优先放在 x1 附近；若右侧没有空间，则靠左边界内。
        if x1 + 2 < xlim[1]:
            lx, ha = x1 + 2, 'left'
        else:
            lx, ha = xlim[1], 'right'

        return lx, ly, ha, va

    def _delete_roi_at(self, x: Optional[float], y: Optional[float]) -> None:
        """删除指定坐标处的ROI（右键）"""
        if x is None or y is None:
            return
        for i in range(len(self.roi_list) - 1, -1, -1):
            x1, y1, x2, y2 = self.roi_list[i]
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.roi_list.pop(i)
                self._after_roi_change()
                self.status_var.set(f"已删除 ROI {i + 1}")
                return
        self.status_var.set("该位置无ROI")

    def _delete_selected_roi(self) -> None:
        """删除列表中选中的ROI"""
        if self._reject_if_busy():
            return
        sel = self.roi_listbox.curselection()
        if sel:
            idx = int(sel[0])
            if 0 <= idx < len(self.roi_list):
                self.roi_list.pop(idx)
                self._after_roi_change()
                self.status_var.set(f"已删除 ROI {idx + 1}")

    def _delete_last_roi(self) -> None:
        """撤销：删除最后一个ROI（仅通过 Ctrl+Z / 菜单触发）"""
        if self._reject_if_busy():
            return
        if self.roi_list:
            self.roi_list.pop()
            self._after_roi_change()
            self.status_var.set("已撤销上一个ROI")

    def _after_roi_change(self) -> None:
        """ROI 变更后的统一收尾：结果失效 + 重绘 + 控件状态。"""
        had_data = self.contrast_data is not None
        self.contrast_data = None
        self.contrast_stats = None
        self._update_roi_listbox()
        self._show_frame()
        self._clear_result_panel("ROI已变更，请重新分析" if had_data else "等待分析")
        self._refresh_widget_states()
        if had_data:
            self.status_var.set("ROI已变更，分析数据已失效，请重新分析")

    def _clear_roi(self) -> None:
        """清空所有ROI"""
        if self._reject_if_busy():
            return
        if not self.roi_list:
            return
        self.roi_list = []
        self.contrast_data = None
        self.contrast_stats = None
        self._update_roi_listbox()
        self._show_frame()
        self._clear_result_panel("等待分析")
        self._info_clear()
        self._refresh_widget_states()
        self.status_var.set("已清空所有ROI")

    def _update_roi_listbox(self) -> None:
        """更新ROI列表面板"""
        self.roi_listbox.delete(0, tk.END)
        for i, (x1, y1, x2, y2) in enumerate(self.roi_list):
            w, h = x2 - x1, y2 - y1
            self.roi_listbox.insert(tk.END,
                                    f"R{i+1}: ({x1},{y1})-({x2},{y2}) [{w}×{h}px]")

    # ==================== 衬度分析（后台线程） ====================

    def _analyze_contrast(self) -> None:
        """启动衬度分析（后台线程）"""
        if self._loading:
            messagebox.showinfo("提示", "正在加载文件，请稍候...")
            return
        if self.tif_memmap is None:
            messagebox.showinfo("提示", "请先打开TIF文件")
            return
        if not self.roi_list:
            messagebox.showinfo("提示", "请先绘制至少一个ROI区域")
            return
        if self._analysis_running:
            messagebox.showinfo("提示", "分析正在进行中，请稍候...")
            return

        # 保存启动时的算法快照：分析期间下拉框禁用，完成/摘要均使用该快照。
        self._save_algorithm_snapshot()

        # 单帧特殊处理：计算衬度值（可导出）并显示强度直方图
        if self.total_frames == 1:
            self._analyze_single_frame()
            return

        self._set_analysis_running(True)
        self._cancel_event.clear()  # 重置取消标志
        # 若上一动作（加载）把进度条留在流动模式，先恢复确定模式
        self._progress_set_indeterminate(False)
        self.progress['value'] = 0
        # 进度按 (ROI, 帧) 组合计，长序列也能连续推进
        n_rois = len(self.roi_list)
        self._analysis_total_rois = n_rois
        self._analysis_total_frames = self.total_frames
        self.progress['maximum'] = max(1, n_rois * self.total_frames)
        self.status_var.set("正在分析...")
        self._log(f"开始分析: {n_rois}个ROI × {self.total_frames}帧"
                  f"（算法: {self._analysis_method_name}）\n")

        # 将 memmap 引用和 ROI 快照作为参数传入，避免 TOCTOU
        thread = threading.Thread(
            target=self._analysis_worker,
            args=(self.tif_memmap, list(self.roi_list), self._analysis_method_key,
                  self.total_frames),
            daemon=True)
        thread.start()
        self._ensure_polling()

    def _save_algorithm_snapshot(self) -> None:
        """保存当前下拉框选中的算法 key/name，供分析期间和完成后使用。"""
        self._analysis_method_idx = self.contrast_method.current()
        if self._analysis_method_idx < 0:
            self._analysis_method_idx = 0
        self._analysis_method_idx %= len(ALGORITHMS)
        self._analysis_method_key = ALGORITHMS[self._analysis_method_idx][1]
        self._analysis_method_name = ALGORITHMS[self._analysis_method_idx][0]

    def _on_method_changed(self, _event=None) -> None:
        """分析完成后切换算法下拉框：提示现有结果仍属旧算法（不自动失效）。"""
        if self._busy() or self.contrast_data is None:
            return
        msg = (f"算法已切换；现有曲线/导出仍为『{self._analysis_method_name}』"
               f"（分析启动时的快照），请重新分析以更新结果")
        self.status_var.set(msg)
        self._log(f"⚠ {msg}\n")

    def _set_analysis_running(self, running: bool) -> None:
        """统一维护分析运行态并刷新控件可用性。"""
        self._analysis_running = running
        self._refresh_widget_states()

    @staticmethod
    def _chunk_frames_for(roi_h: int, roi_w: int) -> int:
        """按像素预算计算每块帧数（保证单块 float64 内存可控）。"""
        pixels = max(1, roi_h * roi_w)
        by_budget = ANALYSIS_MAX_ELEMS_PER_CHUNK // pixels
        return int(max(1, min(ANALYSIS_MAX_CHUNK_FRAMES, by_budget)))

    def _analysis_worker(self, memmap: np.ndarray,
                         roi_list: List[Tuple[int, int, int, int]],
                         method_key: str, total_frames: int) -> None:
        """后台分析工作线程：按（像素预算）分块向量化计算，块间响应取消。"""
        try:
            all_results: List[np.ndarray] = []
            all_means: List[np.ndarray] = []
            all_sigmas: List[np.ndarray] = []
            all_mins: List[np.ndarray] = []
            all_maxs: List[np.ndarray] = []

            for idx, (x1, y1, x2, y2) in enumerate(roi_list):
                if self._cancel_event.is_set():
                    self._task_queue.put(('cancelled', None))
                    return

                roi_h, roi_w = y2 - y1, x2 - x1
                if roi_h <= 0 or roi_w <= 0:
                    raise ValueError(f"ROI {idx + 1} 的尺寸为 0，请重新绘制")
                chunk_frames = self._chunk_frames_for(roi_h, roi_w)

                # 预分配逐帧数组（比逐元素 extend 到 list 省一半以上内存）
                c_vals = np.full(total_frames, np.nan)
                m_vals = np.full(total_frames, np.nan)
                s_vals = np.full(total_frames, np.nan)
                lo_vals = np.full(total_frames, np.nan)
                hi_vals = np.full(total_frames, np.nan)
                n_negative = 0
                n_nonfinite = 0

                for f0 in range(0, total_frames, chunk_frames):
                    if self._cancel_event.is_set():
                        self._task_queue.put(('cancelled', None))
                        return
                    f1 = min(f0 + chunk_frames, total_frames)
                    chunk = memmap[f0:f1, y1:y2, x1:x2]
                    contrast, mean, sigma, vmin, vmax = compute_contrast_stack_ex(
                        chunk, method_key)
                    del chunk
                    c_vals[f0:f1] = contrast
                    m_vals[f0:f1] = mean
                    s_vals[f0:f1] = sigma
                    lo_vals[f0:f1] = vmin
                    hi_vals[f0:f1] = vmax
                    # 同时检查两端：仅含 +Inf 的帧 vmin 有限而 vmax=Inf，
                    # 只看 vmin 会漏报（衬度仍正确为 NaN，但用户无警告）。
                    finite = np.isfinite(vmin) & np.isfinite(vmax)
                    n_nonfinite += int(finite.size - np.count_nonzero(finite))
                    if np.any(finite):
                        n_negative += int(np.count_nonzero(vmin[finite] < 0))
                    self._task_queue.put(('frame_progress', (idx + 1, f1)))

                if method_key == "michelson_robust" and roi_h * roi_w < ROBUST_MIN_PIXELS:
                    self._task_queue.put(('warning', (
                        f"ROI {idx + 1}: 仅 {roi_h * roi_w} 像素，稳健 Michelson 的 "
                        f"1%/99% 分位在 <{ROBUST_MIN_PIXELS} 像素时无法有效排除"
                        f"单像素离群（结果接近极值版），建议扩大 ROI 或换用其他算法\n")))
                if n_negative:
                    self._task_queue.put(('warning', (
                        f"ROI {idx + 1}: {n_negative} 帧含负强度像素；"
                        f"本工具四种衬度定义均假定 I≥0，负值会使 Michelson 类结果 >1 "
                        f"或出现负衬度，请确认数据是否已做背景扣除/归一化\n")))
                if n_nonfinite:
                    self._task_queue.put(('warning', (
                        f"ROI {idx + 1}: {n_nonfinite} 帧检出非有限像素（NaN/Inf），"
                        f"这些帧的衬度按 NaN 记录\n")))

                all_results.append(c_vals)
                all_means.append(m_vals)
                all_sigmas.append(s_vals)
                all_mins.append(lo_vals)
                all_maxs.append(hi_vals)
                self._task_queue.put(('roi_done', idx + 1))

            # 收尾前再响应一次取消：此前最后一个 chunk 结束到 complete 之间
            # 没有检查点，用户在末尾点击取消会看到「正在取消」却照常完成。
            if self._cancel_event.is_set():
                self._task_queue.put(('cancelled', None))
                return
            self._task_queue.put(('complete', (
                all_results,
                {'mean': all_means, 'sigma': all_sigmas,
                 'min': all_mins, 'max': all_maxs})))

        except Exception as e:
            self._task_queue.put(('error', f"{type(e).__name__}: {e}"))

    def _cancel_analysis(self) -> None:
        """取消正在进行的分析"""
        if self._analysis_running:
            self._cancel_event.set()
            self.status_var.set("正在取消分析...")
            self.btn_cancel.config(state=tk.DISABLED)

    def _ensure_polling(self) -> None:
        """仅在存在后台任务时启动轮询（空闲时不再常驻定时器）。"""
        if self._poll_id is None and self.root.winfo_exists():
            self._poll_id = self.root.after(50, self._poll_task_queue)

    def _poll_task_queue(self) -> None:
        """主线程轮询后台任务结果"""
        self._poll_id = None
        try:
            while True:
                msg_type, data = self._task_queue.get_nowait()
                try:
                    self._dispatch_message(msg_type, data)
                except Exception as e:
                    self._analysis_running = False
                    self._loading = False
                    self._log(f"✗ 消息处理异常: {e}\n")
                    self._refresh_widget_states()
        except queue.Empty:
            pass

        if self._busy() and self.root.winfo_exists():
            self._ensure_polling()

    def _dispatch_message(self, msg_type: str, data) -> None:
        """后台消息分发（全部在主线程执行）。"""
        if msg_type == 'roi_done':
            self.progress['value'] = int(data) * self._analysis_total_frames
            self.status_var.set(
                f"分析中... ROI {data}/{self._analysis_total_rois} 完成")
        elif msg_type == 'frame_progress':
            roi_i, frames_done = data
            self.progress['value'] = ((roi_i - 1) * self._analysis_total_frames
                                      + frames_done)
            self.status_var.set(
                f"分析中... ROI {roi_i}/{self._analysis_total_rois} · "
                f"帧 {frames_done}/{self._analysis_total_frames}")
        elif msg_type == 'complete':
            self._on_analysis_complete(data)
        elif msg_type == 'cancelled':
            self._on_analysis_cancelled()
        elif msg_type == 'error':
            self._on_analysis_error(data)
        elif msg_type == 'warning':
            self._log(f"⚠ {data}")
        elif msg_type == 'loaded':
            self._on_load_complete(data)
        elif msg_type == 'load_error':
            self._on_load_error(data)
        else:
            self._log(f"⚠ 未知消息类型: {msg_type}\n")

    def _on_analysis_complete(self, payload) -> None:
        """分析完成回调（主线程）"""
        results, stats = payload
        self._set_analysis_running(False)
        self.contrast_data = results
        self.contrast_stats = stats
        self._plot_contrast_curves(results)
        self._show_statistics(results)
        self.progress['value'] = self.progress['maximum']
        self.status_var.set("分析完成")
        self._log("✓ 分析完成\n")
        self._refresh_widget_states()

    def _on_analysis_cancelled(self) -> None:
        """分析被取消"""
        self._set_analysis_running(False)
        self.progress['value'] = 0
        self.status_var.set("分析已取消（未产生结果）")
        self._log("■ 分析已被用户取消\n")
        self._refresh_widget_states()

    def _on_analysis_error(self, error_msg: str) -> None:
        """分析出错回调"""
        self._set_analysis_running(False)
        self.progress['value'] = 0
        self.status_var.set("分析出错")
        self._log(f"✗ 分析错误: {error_msg}\n")
        self._refresh_widget_states()
        messagebox.showerror("分析错误", f"分析过程中出错:\n{error_msg}")

    def _plot_contrast_curves(self, results: List[List[float]]) -> None:
        """绘制逐帧衬度曲线（x 轴 1 基，与 CSV 帧号一致；超长序列抽稀）。"""
        self.ax_plot.clear()
        style_axes_dark(self.ax_plot)
        n = len(results[0]) if results else 0
        stride = max(1, int(np.ceil(n / PLOT_MAX_POINTS))) if n else 1
        x = np.arange(1, n + 1)[::stride]
        for idx, roi_vals in enumerate(results):
            color = ROI_COLORS[idx % len(ROI_COLORS)]
            y = np.asarray(roi_vals, dtype=np.float64)[::stride]
            self.ax_plot.plot(x, y, label=f"ROI {idx + 1}", color=color, linewidth=1.2)

        title = f"衬度演化 ({self._analysis_method_name})"
        if stride > 1:
            title += f" · 显示抽稀 1/{stride}"
        self.ax_plot.set_title(title, fontsize=10)
        self.ax_plot.set_xlabel("帧序号", fontsize=9)
        self.ax_plot.set_ylabel("衬度值", fontsize=9)
        if results:
            self.ax_plot.legend(facecolor=DARK_AXES, edgecolor=SPINE_COLOR,
                                labelcolor=LIGHT_TEXT, fontsize=8)
        self.canvas_plot.draw()

    def _analyze_single_frame(self) -> None:
        """单帧图像：计算各ROI衬度值（写入结果、可导出）并显示强度直方图。

        与多帧路径共用 ``compute_contrast_stack_ex``（帧数=1），保证衬度
        公式与统计口径完全一致；负强度/非有限/小 ROI 稳健分位三类警告
        也与多帧路径对齐（此前单帧路径完全没有后两类警告）。
        """
        frame = self.tif_memmap[0]
        method_key = self._analysis_method_key
        method_name = self._analysis_method_name

        contrast_vals: List[float] = []
        means: List[float] = []
        sigmas: List[float] = []
        mins: List[float] = []
        maxs: List[float] = []
        for (x1, y1, x2, y2) in self.roi_list:
            roi_data = frame[y1:y2, x1:x2]
            contrast, mean, sigma, vmin, vmax = compute_contrast_stack_ex(
                roi_data[np.newaxis, ...], method_key)
            contrast_vals.append(float(contrast[0]))
            means.append(float(mean[0]))
            sigmas.append(float(sigma[0]))
            mins.append(float(vmin[0]))
            maxs.append(float(vmax[0]))

        # 与多帧路径同构：(帧数=1) × (ROI数)，导出逻辑无需特判
        self.contrast_data = [[v] for v in contrast_vals]
        self.contrast_stats = {'mean': [[v] for v in means],
                               'sigma': [[v] for v in sigmas],
                               'min': [[v] for v in mins],
                               'max': [[v] for v in maxs]}

        self.ax_plot.clear()
        style_axes_dark(self.ax_plot)
        for idx, (x1, y1, x2, y2) in enumerate(self.roi_list):
            roi_data = frame[y1:y2, x1:x2].flatten()
            color = ROI_COLORS[idx % len(ROI_COLORS)]
            self.ax_plot.hist(roi_data, bins=50, alpha=0.5, label=f"ROI {idx+1}",
                              color=color, edgecolor='none')

        self.ax_plot.set_title(f"ROI强度分布 (单帧, {method_name})", fontsize=10)
        self.ax_plot.set_xlabel("像素强度", fontsize=9)
        self.ax_plot.set_ylabel("频次", fontsize=9)
        self.ax_plot.legend(facecolor=DARK_AXES, edgecolor=SPINE_COLOR,
                            labelcolor=LIGHT_TEXT, fontsize=8)
        self.canvas_plot.draw()

        # 统计信息（含按所选算法计算的衬度值与科学性警告）
        self._info_append(f"═══ 单帧统计 ({method_name}) ═══\n\n")
        for idx, (x1, y1, x2, y2) in enumerate(self.roi_list):
            v = contrast_vals[idx]
            v_str = "NaN" if not np.isfinite(v) else f"{v:.4f}"
            self._info_append(
                f"ROI {idx+1}: 衬度={v_str}  "
                f"均值={means[idx]:.2f}  "
                f"σ={sigmas[idx]:.2f}  "
                f"范围=[{mins[idx]:.0f}, {maxs[idx]:.0f}]\n")
            if method_key == "michelson_robust" and (x2 - x1) * (y2 - y1) < ROBUST_MIN_PIXELS:
                self._info_append(
                    f"  ⚠ 该 ROI 仅 {(x2 - x1) * (y2 - y1)} 像素，稳健 Michelson 的 "
                    f"1%/99% 分位在 <{ROBUST_MIN_PIXELS} 像素时无法有效排除单像素离群"
                    f"（结果接近极值版）\n")
            if not (np.isfinite(mins[idx]) and np.isfinite(maxs[idx])):
                self._info_append(
                    "  ⚠ 该 ROI 含非有限像素（NaN/Inf），衬度按 NaN 记录\n")
            elif mins[idx] < 0:
                self._info_append(
                    "  ⚠ 该 ROI 含负强度像素；四种衬度定义均假定 I≥0，"
                    "负值会使 Michelson 类结果 >1 或出现负衬度\n")
        self._info_append("\n提示：单帧模式不产生跨帧曲线；CSV 仍可导出（1 帧 × N ROI）\n")
        self.progress['maximum'] = max(1, len(self.roi_list))
        self.progress['value'] = self.progress['maximum']
        self._log("✓ 单帧分析完成\n")
        self.status_var.set("单帧模式：已计算衬度值并显示强度分布直方图")
        self._refresh_widget_states()

    def _show_statistics(self, results: List[List[float]]) -> None:
        """追加统计摘要（使用分析启动时保存的算法快照；不清空日志历史）"""
        self._info_append("\n──────────────────────\n\n")
        self._info_append(f"═══ 统计摘要 ({self._analysis_method_name}) ═══\n"
                          f"（该摘要同样写入导出 CSV 末尾的 # 注释块）\n\n")

        for idx, roi_vals in enumerate(results):
            arr = np.asarray(roi_vals, dtype=np.float64)
            valid = arr[np.isfinite(arr)]
            if valid.size == 0:
                self._info_append(f"ROI {idx+1}: 全部无效数据\n")
                continue
            self._info_append(
                f"ROI {idx+1}: 有效帧 {valid.size}/{arr.size}  "
                f"均值={np.mean(valid):.4f}  "
                f"σ={np.std(valid):.4f}  "
                f"范围=[{np.min(valid):.4f}, {np.max(valid):.4f}]\n")

    # ==================== 辅助功能 ====================

    def _info_edit(self, fn) -> None:
        """以临时可编辑方式操作只读信息面板（结束自动恢复 DISABLED）。"""
        self.info_box.config(state=tk.NORMAL)
        try:
            fn(self.info_box)
        finally:
            self.info_box.config(state=tk.DISABLED)

    def _info_append(self, text: str) -> None:
        """向信息面板追加文本；超过 LOG_MAX_LINES 行时裁剪头部。"""
        def _op(box: tk.Text) -> None:
            box.insert(tk.END, text)
            line_count = int(box.index('end-1c').split('.')[0])
            if line_count > LOG_MAX_LINES:
                box.delete('1.0', f'{line_count - LOG_MAX_LINES}.0')
            box.see(tk.END)
        self._info_edit(_op)

    def _info_clear(self) -> None:
        """清空信息面板。"""
        self._info_edit(lambda box: box.delete('1.0', tk.END))

    def _log(self, text: str) -> None:
        """写入信息面板（追加式）"""
        self._info_append(text)

    def _copy_info(self) -> None:
        """复制信息面板内容到剪贴板（统计摘要可直接粘贴进报告）。"""
        text = self.info_box.get("1.0", tk.END).strip()
        if not text:
            self.status_var.set("信息面板为空")
            return
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_var.set(f"已复制信息面板内容（{len(text)} 字符）到剪贴板")
        except tk.TclError as e:
            self.status_var.set(f"复制失败: {e}")

    def _on_close(self) -> None:
        """窗口关闭时清理资源"""
        if self._analysis_running:
            self._cancel_event.set()
        if self._poll_id is not None:
            try:
                self.root.after_cancel(self._poll_id)
            except Exception:
                pass
            self._poll_id = None
        if self.tif_memmap is not None:
            del self.tif_memmap
            self.tif_memmap = None
        self.root.destroy()

    def _show_shortcuts(self) -> None:
        """显示快捷键说明"""
        shortcuts = (
            "═══ 快捷键说明 ═══\n\n"
            "← / →        逐帧步进\n"
            "Ctrl+O       打开文件\n"
            "Ctrl+Z       撤销上一个ROI\n\n"
            "═══ 鼠标操作 ═══\n\n"
            "左键拖拽     绘制矩形ROI\n"
            "右键点击     删除该位置ROI\n"
            "滚轮         缩放图像\n"
            "双击         重置视图\n"
            "列表Del      删除选中ROI\n\n"
            "═══ 文件菜单 ═══\n\n"
            "导入/导出ROI坐标 (JSON)：跨文件复用同一批ROI\n"
            "导出CSV：含每帧强度统计与跨帧统计注释块\n"
        )
        messagebox.showinfo("快捷键", shortcuts)

    def _show_about(self) -> None:
        """关于对话框（版本与算法清单取自 contrast_core 单一来源）"""
        algos = "\n".join(f"• {name}" for name, _ in ALGORITHMS)
        font_info = self.font_name or "未找到中文字体（图内中文可能为方框）"
        messagebox.showinfo("关于",
                            f"TIF图像衬度分析工具 v{__version__}\n\n"
                            "用于TEM图像堆叠序列的ROI衬度统计分析\n\n"
                            f"支持算法:\n{algos}\n\n"
                            f"图内字体: {font_info}\n"
                            "AIforTEM 工具集")


def run_selftest(log_path: Optional[str] = None) -> int:
    """打包产物自检：隐藏窗口跑完"加载 → 分析 → 导出"全链路，返回退出码。

    用法: ``TIF图像衬度分析工具vX.Y.Z.exe --selftest [日志路径]``
    日志默认写到系统临时目录下的 contrast_selftest_*/selftest.log，
    不弹任何对话框，适合在目标机器上确认 exe 可用。
    """
    import csv as _csv
    import tempfile
    import traceback

    tmpdir = tempfile.mkdtemp(prefix="contrast_selftest_")
    log_path = log_path or os.path.join(tmpdir, "selftest.log")
    lines: List[str] = [f"selftest @ {time.strftime('%Y-%m-%d %H:%M:%S')}",
                        f"version={__version__} python={sys.version.split()[0]}",
                        f"frozen={getattr(sys, 'frozen', False)}"]
    failures: List[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        lines.append(f"{'PASS' if cond else 'FAIL'}  {name} {detail}")
        if not cond:
            failures.append(name)

    root = None
    try:
        root = ttkb.Window(themename="darkly")
        root.withdraw()
        app = TIFContrastAnalyzer(root)
        lines.append(f"font={app.font_name or '未找到中文字体'}")
        check("中文字体可用", bool(app.font_name), app.font_name)

        arr = np.random.default_rng(0).integers(100, 2000, size=(6, 64, 80)).astype(np.uint16)
        arr[2, 12, 12] = 65535                      # 热像素
        tif_path = os.path.join(tmpdir, "stack.tif")
        tifffile.imwrite(tif_path, arr, photometric="minisblack")

        app._start_load(tif_path)
        _pump(root, lambda: app._busy(), 60)
        check("加载帧数=6", app.total_frames == 6, f"frames={app.total_frames}")
        check("加载: 头部探测", app._probe is not None and app._probe.n_pages == 6)

        app.roi_list = [(8, 8, 40, 40), (30, 20, 60, 50)]
        app._update_roi_listbox()
        app._show_frame()
        app.contrast_method.current(3)              # 稳健 Michelson
        app._analyze_contrast()
        _pump(root, lambda: app._busy(), 120)
        check("分析: 2ROI×6帧", bool(app.contrast_data)
              and len(app.contrast_data) == 2 and len(app.contrast_data[0]) == 6,
              f"shape={[len(r) for r in (app.contrast_data or [])]}")
        check("分析: 逐帧统计", bool(app.contrast_stats)
              and set(app.contrast_stats) >= {"mean", "sigma", "min", "max"})
        check("分析: 进度条走满", app.progress['value'] == app.progress['maximum'])

        csv_path = os.path.join(tmpdir, "out.csv")
        n_frames, n_rois = app.write_csv(csv_path)
        check("导出: 帧数/ROI 数", (n_frames, n_rois) == (6, 2), f"{n_frames}×{n_rois}")
        with open(csv_path, "rb") as fh:
            raw = fh.read().decode("utf-8-sig")
        check("导出: 元数据头", "# 软件: TIF图像衬度分析工具 v" in raw and "# TIFF 布局:" in raw)
        check("导出: 跨帧统计尾部", "# ROI_1: 有效帧 6/6" in raw)
        with open(csv_path, encoding="utf-8-sig", newline="") as fh:
            rows = [r for r in _csv.reader(l for l in fh if not l.startswith("#")) if r]
        # 列数 = Frame + 2 个衬度列 + 2 ROI × 4 个强度统计列
        check("导出: 数据行列数", len(rows) == 7 and len(rows[0]) == 11,
              f"rows={len(rows)} cols={len(rows[0]) if rows else 0}")
        check("导出: 帧号 1 基", rows[1][0] == "1", f"first={rows[1][0]if len(rows) > 1 else '?'}")
    except Exception:
        failures.append("异常")
        lines.append(traceback.format_exc())
    finally:
        try:
            if root is not None:
                root.destroy()
        except Exception:
            pass

    lines.append(f"RESULT: {'OK' if not failures else 'FAILED -> ' + ', '.join(failures)}")
    lines.append(f"workdir: {tmpdir}")
    text = "\n".join(lines) + "\n"
    try:
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(text)
    except OSError:
        pass
    print(text)
    return 1 if failures else 0


def _pump(root, is_busy, timeout: float) -> None:
    """在无 mainloop 情况下泵 Tk 事件直到后台任务结束或超时。"""
    deadline = time.time() + timeout
    while is_busy() and time.time() < deadline:
        root.update()
        time.sleep(0.01)
    root.update()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _idx = sys.argv.index("--selftest")
        _log = sys.argv[_idx + 1] if len(sys.argv) > _idx + 1 else None
        sys.exit(run_selftest(_log))
    root = ttkb.Window(themename="darkly")
    app = TIFContrastAnalyzer(root)
    root.mainloop()
