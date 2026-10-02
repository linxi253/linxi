"""
TIFF堆叠查看与面积测量工具
功能：加载TIFF图像堆叠，手动绘制多边形测量面积，支持标尺校准、数据导出。
修复版本：修正坐标转换系统、JSON加载、性能优化、交互增强。
"""
import sys
import time
import ctypes
import logging
from logging.handlers import RotatingFileHandler
from tkinter import filedialog, messagebox, ttk, simpledialog
import tkinter as tk
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import tifffile
import pandas as pd
import matplotlib

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

matplotlib.use('TkAgg')  # 显式指定后端，避免环境问题
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import to_rgba
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon as MplPolygon
import json
import queue
import threading

from area_core import (
    clamp_point,
    coerce_positive_float,
    convert_area_to_units,
    count_out_of_bounds_points,
    flip_polygon_y,
    group_series_by_root,
    natural_sort_key,
    normalize_tiff_data,
    polygon_area,
    polygon_perimeter,
    polygon_self_intersects,
    rebuild_measurements,
    rgb_to_gray_uint8,
    ruler_ratio,
    sanitize_project_polygons,
    sanitize_ruler_points,
)

logger = logging.getLogger("tiff_measure")

# ---- 常量集中定义（避免魔法数字散落各处） ----
ZOOM_MIN, ZOOM_MAX = 0.1, 100.0
ZOOM_STEP_IN, ZOOM_STEP_OUT = 1.2, 0.8
FRAME_NAV_DEBOUNCE_MS = 50          # 帧导航/缩放防抖间隔
LOAD_POLL_INTERVAL_MS = 100         # 后台加载队列轮询间隔
DISPLAY_ERR_THROTTLE_S = 5.0        # 渲染错误弹窗节流窗口
LOAD_PROGRESS_EVERY_PAGES = 25      # 多页 TIFF 每 N 页报一次进度
BIG_STACK_BYTES = 2 * 1024 ** 3     # 加载前需确认的内存阈值（2 GiB）
LOG_MAX_BYTES = 2 * 1024 * 1024     # 日志单文件上限，超出轮转
LOG_BACKUP_COUNT = 3
# 单位白名单：会拼进导出表头，白名单外的值一律回退 nm（防表头注入）
UNIT_WHITELIST = ("nm", "μm", "um", "mm", "cm", "m")
# 快捷键需要放行的文本类控件类名（焦点在其中时不劫持按键）
TEXT_WIDGET_CLASSES = ("Entry", "TEntry", "Text", "Spinbox", "TSpinbox",
                       "Listbox", "TListbox", "Combobox", "TCombobox")

# matplotlib 中文支持（导出图表时中文不会乱码）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                   'Noto Sans CJK SC', 'WenQuanYi Micro Hei',
                                   'PingFang SC', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# Windows 高DPI支持
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass


def close_tiff_stack(stack) -> None:
    """显式释放 TIFF 内存映射占用的文件句柄（普通 ndarray 无副作用）。

    沿 ``.base`` 链找到 np.memmap 并关闭其底层 ``_mmap``（与
    03-应变分析/原子识别纯算法 atomic_core.close_tiff_stack 同一实现）。
    Windows 下若不关闭，被 memmap 打开的源文件在映射被覆盖前一直
    处于锁定状态（无法移动/删除/覆盖）。
    """
    current = stack
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, np.memmap):
            mmap_obj = getattr(current, "_mmap", None)
            if mmap_obj is not None:
                try:
                    mmap_obj.close()
                except Exception:
                    pass
            return
        current = getattr(current, "base", None)


class TiffStackViewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("TIFF堆叠查看与测量工具 - 带标尺校准")
        self.root.geometry("1400x900")

        # 设置主题
        try:
            self.root.option_add("*Font", "微软雅黑 10")
        except Exception:
            pass

        # 变量初始化
        self.tiff_stack = None
        self.current_frame = 0
        self.total_frames = 0
        self.file_path = None
        self.polygons = {}          # {frame_index: {polygon_id: [(x,y),...]}}
        self.polygon_colors = {}    # {polygon_id: color_str}
        self.polygon_names = {}     # {polygon_id: name_str}
        self.polygon_roots = {}     # {polygon_id: root_id}，复制链共享 root，供面积变化图追踪
        self.measurements = {}      # {frame_index: {polygon_id: {'pixel_area':, 'unit_area':}}}
        self.next_polygon_id = 1
        self.drawing_mode = False
        self.current_polygon_points = []
        self.zoom_factor = 1.0
        self.pan_start = None
        self.pan_offset = [0.0, 0.0]

        # 标尺相关变量
        self.ruler_mode = False
        self.ruler_points = []
        self.pixel_to_unit_ratio = 1.0  # 像素/单位
        self.unit_name = "nm"
        self.ruler_length_pixels = 0
        self.ruler_length_units = 0
        self.ruler_calibrated = False  # 标尺校准状态布尔标志，不再用 ratio != 1.0 判断

        # 后台加载线程通信。
        # 队列只在初始化时创建一次；每次发起新加载时 _load_generation 自增，
        # worker 携带发起时的代次，_poll_load_queue 丢弃过期代次的消息——
        # 旧 worker 无法再把结果写进新会话（双击并发加载竞态）。
        self._load_queue = queue.Queue()
        self._load_generation = 0
        self._load_after_id = None
        self._pending_project = None  # load_project 挂起，待图像加载完成后恢复

        # 临时绘图对象（性能优化）
        self._temp_ruler_line = None
        self._temp_poly_line = None    # 绘制中多边形的临时连线
        self._temp_poly_dots = None    # 绘制中多边形的顶点标记
        self._temp_preview_line = None  # 绘制中"最后顶点→光标"橡皮筋预览线
        self._display_after_id = None  # 帧导航防抖
        self._dirty = False            # 存在未保存修改
        self._last_display_err = 0.0   # 上次渲染错误弹窗时间（节流用）

        # 跨帧对比度锁定（统一 vmin/vmax，帧间强度可比）
        self.lock_contrast_var = tk.BooleanVar(value=False)
        self._contrast_lock = None     # 锁定时为 (vmin, vmax)

        # 颜色列表
        self.color_palette = [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
            '#00FFFF', '#FFA500', '#800080', '#008000', '#000080'
        ]

        self.setup_ui()

        # 窗口关闭处理
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # 定时轮询后台加载队列
        self._poll_load_queue()

    # ==================== UI 构建 ====================

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 顶部工具栏（第一行：文件+帧导航+缩放）
        toolbar1 = ttk.Frame(main_frame)
        toolbar1.pack(fill=tk.X, pady=(0, 5))

        self.open_file_btn = ttk.Button(toolbar1, text="打开TIFF文件", command=self.open_tiff_file)
        self.open_file_btn.pack(side=tk.LEFT, padx=2)
        self.open_folder_btn = ttk.Button(toolbar1, text="打开文件夹", command=self.open_folder)
        self.open_folder_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar1, text="保存项目", command=self.save_project).pack(side=tk.LEFT, padx=2)
        self.load_project_btn = ttk.Button(toolbar1, text="加载项目", command=self.load_project)
        self.load_project_btn.pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar1, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)

        ttk.Label(toolbar1, text="帧:").pack(side=tk.LEFT, padx=2)
        self.frame_var = tk.StringVar(value="1")
        self.frame_spin = ttk.Spinbox(toolbar1, from_=1, to=1, textvariable=self.frame_var,
                                      width=8, command=self.on_frame_changed)
        self.frame_spin.pack(side=tk.LEFT, padx=2)
        # 手输帧号后回车立即生效（Spinbox 的 command 只响应箭头点击）
        self.frame_spin.bind('<Return>', lambda e: self.on_frame_changed())
        self.frame_spin.bind('<KP_Enter>', lambda e: self.on_frame_changed())

        # ★ 修复：帧总数标签改为动态
        self.total_frames_label = ttk.Label(toolbar1, text="/ 1")
        self.total_frames_label.pack(side=tk.LEFT, padx=2)

        ttk.Button(toolbar1, text="◀", width=3, command=self.prev_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar1, text="▶", width=3, command=self.next_frame).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar1, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)

        ttk.Label(toolbar1, text="缩放:").pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar1, text="+", width=3,
                   command=lambda: self.zoom(ZOOM_STEP_IN)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar1, text="-", width=3,
                   command=lambda: self.zoom(ZOOM_STEP_OUT)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar1, text="重置", command=self.reset_view).pack(side=tk.LEFT, padx=2)
        # 锁定对比度：所有帧用统一的 vmin/vmax 显示，帧间强度才可比
        # （默认 imshow 每帧独立 min/max 归一化，会掩盖真实的强度变化）
        ttk.Checkbutton(toolbar1, text="锁定对比度",
                        variable=self.lock_contrast_var,
                        command=self._toggle_contrast_lock).pack(side=tk.LEFT, padx=(12, 2))

        # 第二行：绘制工具+标尺+导出
        toolbar2 = ttk.Frame(main_frame)
        toolbar2.pack(fill=tk.X, pady=(0, 10))

        self.draw_btn = ttk.Button(toolbar2, text="绘制多边形", command=self.toggle_drawing_mode)
        self.draw_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar2, text="完成绘制(Enter)", command=self.finish_polygon).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar2, text="撤销点(Backspace)", command=self.undo_last_point).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar2, text="清除当前帧", command=self.clear_current_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar2, text="清除所有", command=self.clear_all_polygons).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar2, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)

        ttk.Label(toolbar2, text="标尺:").pack(side=tk.LEFT, padx=2)
        self.ruler_btn = ttk.Button(toolbar2, text="设置标尺", command=self.toggle_ruler_mode)
        self.ruler_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar2, text="手动输入", command=self.manual_ruler_input).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar2, text="清除标尺", command=self.clear_ruler).pack(side=tk.LEFT, padx=2)

        self.ruler_info_label = ttk.Label(toolbar2, text="未校准")
        self.ruler_info_label.pack(side=tk.LEFT, padx=10)

        ttk.Separator(toolbar2, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)

        ttk.Button(toolbar2, text="导出数据", command=self.export_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar2, text="导出图表", command=self.export_chart).pack(side=tk.LEFT, padx=2)

        # 主内容区域
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧图像显示区域
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # 嵌入式画布用 OO 接口创建 Figure，不经 pyplot 图窗管理
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 鼠标事件
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_canvas_motion)
        self.canvas.mpl_connect('scroll_event', self.on_canvas_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_canvas_pan_start)
        self.canvas.mpl_connect('motion_notify_event', self.on_canvas_pan)
        self.canvas.mpl_connect('button_release_event', self.on_canvas_pan_end)

        # 右侧控制面板
        right_frame = ttk.Frame(content_frame, width=320)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)

        # 文件信息
        info_frame = ttk.LabelFrame(right_frame, text="文件信息", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        self.info_text = tk.Text(info_frame, height=6, width=32)
        self.info_text.pack(fill=tk.X)
        self.info_text.insert(tk.END, "未加载文件")
        self.info_text.config(state=tk.DISABLED)

        # 标尺信息
        ruler_info_frame = ttk.LabelFrame(right_frame, text="标尺校准", padding="10")
        ruler_info_frame.pack(fill=tk.X, pady=(0, 10))
        self.ruler_text = tk.Text(ruler_info_frame, height=4, width=32)
        self.ruler_text.pack(fill=tk.X)
        self.ruler_text.insert(tk.END, "标尺未校准\n比例: 1 像素 = 1 像素")
        self.ruler_text.config(state=tk.DISABLED)

        # 多边形列表
        poly_frame = ttk.LabelFrame(right_frame, text="多边形管理", padding="10")
        poly_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        poly_list_frame = ttk.Frame(poly_frame)
        poly_list_frame.pack(fill=tk.BOTH, expand=True)

        header_frame = ttk.Frame(poly_list_frame)
        header_frame.pack(fill=tk.X)
        ttk.Label(header_frame, text="名称", width=10).pack(side=tk.LEFT)
        ttk.Label(header_frame, text="面积", width=12).pack(side=tk.LEFT, padx=5)
        ttk.Label(header_frame, text="顶点", width=5).pack(side=tk.LEFT)

        self.poly_listbox = tk.Listbox(poly_list_frame, height=12)
        list_scroll = ttk.Scrollbar(poly_list_frame, orient=tk.VERTICAL,
                                    command=self.poly_listbox.yview)
        self.poly_listbox.config(yscrollcommand=list_scroll.set)
        self.poly_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=(5, 0))
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=(5, 0))

        poly_btn_frame = ttk.Frame(poly_frame)
        poly_btn_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(poly_btn_frame, text="重命名", command=self.rename_polygon).pack(side=tk.LEFT, padx=2)
        ttk.Button(poly_btn_frame, text="删除", command=self.delete_selected_polygon).pack(side=tk.LEFT, padx=2)
        ttk.Button(poly_btn_frame, text="复制到下一帧", command=self.copy_to_next_frame).pack(side=tk.LEFT, padx=2)

        # 测量统计
        stats_frame = ttk.LabelFrame(right_frame, text="测量统计", padding="10")
        stats_frame.pack(fill=tk.X)
        self.stats_text = tk.Text(stats_frame, height=8, width=32)
        self.stats_text.pack(fill=tk.X)
        self.stats_text.insert(tk.END, "无测量数据")
        self.stats_text.config(state=tk.DISABLED)

        # 状态栏
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        self.status_var = tk.StringVar(value="就绪 | 快捷键: ←→切帧, Enter完成, Backspace撤销, Esc取消, 右键撤销点")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)
        # 取消加载按钮：仅在后台加载进行中显示
        self.cancel_load_btn = ttk.Button(status_frame, text="取消加载", command=self.cancel_loading)
        self.coord_var = tk.StringVar(value="坐标: (0, 0)")
        ttk.Label(status_frame, textvariable=self.coord_var).pack(side=tk.RIGHT)

        # 快捷键挂在根窗口上：点击工具栏按钮后焦点离开画布也能用 ←→ 切帧。
        # 焦点在文本类控件（Entry/Spinbox/组合框等）时不劫持按键，让其走
        # 控件默认行为；多边形列表聚焦时放行（列表自身不用这些键，
        # 选中条目后按 Delete 直接删除多边形是核心操作路径）。
        for keysym, handler in (
                ('<Left>', self.prev_frame),
                ('<Right>', self.next_frame),
                ('<Escape>', self.cancel_drawing),
                ('<BackSpace>', self.undo_last_point),
                ('<Return>', self.finish_polygon),
                ('<KP_Enter>', self.finish_polygon),
                ('<Delete>', self.delete_selected_polygon)):
            self.root.bind(keysym,
                           lambda e, fn=handler: self._dispatch_hotkey(fn))

        # 确保画布可以获得键盘焦点（默认焦点）
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.config(takefocus=1)
        canvas_widget.focus_set()

        # 初始化显示
        self.ax.set_title("请打开TIFF文件")
        self.ax.text(0.5, 0.5, "点击工具栏的'打开TIFF文件'或'打开文件夹'加载图像",
                     horizontalalignment='center', verticalalignment='center',
                     transform=self.ax.transAxes, fontsize=12, color='gray')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()

    # ==================== 状态管理 ====================

    def _dispatch_hotkey(self, handler):
        """根窗口快捷键分发：焦点在文本控件时放行默认行为，否则执行快捷键。"""
        focus = self.root.focus_get()
        if focus is not None and focus.winfo_class() in TEXT_WIDGET_CLASSES:
            if focus is not self.poly_listbox:
                return None
        handler()
        return 'break'

    def _confirm_discard_changes(self) -> bool:
        """存在未保存修改时要求确认；返回 False 表示用户取消本次操作。

        打开新文件/文件夹/项目都会整体重置状态，必须先过这道闸。
        """
        if not self._dirty:
            return True
        return messagebox.askyesno(
            "未保存的修改",
            "当前测量数据尚未保存。\n\n继续将放弃这些修改，确定吗？")

    def _toggle_contrast_lock(self):
        """锁定/解锁跨帧对比度：锁定后所有帧共用同一 vmin/vmax 显示。"""
        if self.lock_contrast_var.get():
            if self.tiff_stack is None:
                self.lock_contrast_var.set(False)
                self.status_var.set("请先加载图像")
                return
            frame = self.tiff_stack[self.current_frame]
            vmin, vmax = float(frame.min()), float(frame.max())
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                self._contrast_lock = (vmin, vmax)
                self.status_var.set(f"对比度已锁定: [{vmin:.1f}, {vmax:.1f}]")
            else:
                self.lock_contrast_var.set(False)
                self._contrast_lock = None
                messagebox.showwarning(
                    "对比度锁定",
                    "当前帧值范围无效（含 NaN 或无灰度对比），无法锁定。")
        else:
            self._contrast_lock = None
            self.status_var.set("对比度锁定已解除")
        self.display_current_frame()

    def _reset_state(self):
        """重置所有工作状态（提取公共逻辑）"""
        self.current_frame = 0
        self.polygons = {}
        self.polygon_colors = {}
        self.polygon_names = {}
        self.polygon_roots = {}
        self.measurements = {}
        self.next_polygon_id = 1
        self.drawing_mode = False
        self.current_polygon_points = []
        self.zoom_factor = 1.0
        self.pan_start = None
        self.pan_offset = [0.0, 0.0]
        self.ruler_mode = False
        self.ruler_points = []
        self.pixel_to_unit_ratio = 1.0
        self.unit_name = "nm"
        self.ruler_length_pixels = 0
        self.ruler_length_units = 0
        self.ruler_calibrated = False
        self._temp_ruler_line = None
        self._temp_preview_line = None
        self._contrast_lock = None
        if hasattr(self, 'lock_contrast_var'):
            self.lock_contrast_var.set(False)
        self.draw_btn.config(text="绘制多边形")
        self.ruler_btn.config(text="设置标尺")
        self._dirty = False  # 新文档/新加载，视为已保存状态

    def _update_ui_after_load(self):
        """加载完成后统一更新UI"""
        self.frame_spin.config(from_=1, to=self.total_frames)
        self.frame_var.set("1")
        self.total_frames_label.config(text=f"/ {self.total_frames}")
        self.update_file_info()
        self.update_ruler_info()
        self.display_current_frame()

    # ==================== 文件加载 ====================

    def open_tiff_file(self):
        file_path = filedialog.askopenfilename(
            title="选择TIFF文件",
            filetypes=[("TIFF文件", "*.tif *.tiff"), ("所有文件", "*.*")]
        )
        if file_path:
            if not self._confirm_discard_changes():
                return
            self.load_tiff_file(file_path)

    def open_folder(self):
        folder_path = filedialog.askdirectory(title="选择包含TIFF文件的文件夹")
        if not folder_path:
            return
        if not self._confirm_discard_changes():
            return

        tiff_files = []
        for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
            tiff_files.extend(Path(folder_path).glob(f'*{ext}'))

        if not tiff_files:
            messagebox.showwarning("警告", "文件夹中没有找到TIFF文件")
            return

        tiff_files = sorted(set(tiff_files), key=natural_sort_key)

        if messagebox.askyesno("加载方式", f"找到 {len(tiff_files)} 个TIFF文件。\n\n是否将它们合并为一个堆叠？"):
            self.load_tiff_stack_from_files(tiff_files)
        else:
            self.load_tiff_file(str(tiff_files[0]))

    @staticmethod
    def _estimate_load_bytes(path) -> Optional[int]:
        """从 TIFF 头部估算加载后占用的内存（只读元数据，很快）。

        无法判断、或走 memmap 按需分页路径（不整块占内存）时返回 None。
        """
        try:
            with tifffile.TiffFile(path) as tif:
                if not tif.series:
                    return None
                series = tif.series[0]
                shape = tuple(getattr(series, "shape", ()))
                dtype = getattr(series, "dtype", None)
                if not shape or dtype is None:
                    return None
                axes = getattr(series, "axes", "") or ""
                n_pages = len(tif.pages)
                # 单页 3D 灰度堆叠走 memmap（按需分页），不整块占用内存
                if (n_pages == 1 and len(shape) == 3
                        and "T" in axes and "C" not in axes and "S" not in axes):
                    return None
                return int(np.prod(shape)) * np.dtype(dtype).itemsize
        except Exception:
            return None

    def load_tiff_file(self, file_path):
        """异步加载 TIFF 文件：后台线程读取，主线程只轮询队列。

        注意：这里不预先赋值 self.file_path——只有后台线程读取与维度
        规范化全部成功、_poll_load_queue 在主线程提交状态时才更新，
        保证失败/取消时旧文档（含其路径）完整保留。
        """
        # 大文件防护：正式加载前用元数据估算内存，超过阈值先确认
        est_bytes = self._estimate_load_bytes(file_path)
        if est_bytes is not None and est_bytes > BIG_STACK_BYTES:
            if not messagebox.askyesno(
                    "大文件确认",
                    f"该文件加载后预计占用约 {est_bytes / 1024 ** 3:.1f} GB 内存。\n"
                    "文件过大可能导致加载缓慢甚至内存不足。\n\n仍要继续吗？"):
                self.status_var.set("已取消加载大文件")
                return

        gen = self._load_generation + 1
        self._load_generation = gen
        self._set_load_ui(True)
        self.status_var.set("正在读取 TIFF 元数据...")
        threading.Thread(target=self._load_tiff_worker, args=(file_path, gen),
                         daemon=True).start()

    def _load_tiff_worker(self, file_path, gen):
        """后台线程：先读元数据/分页，再按需读取像素数据。

        gen 为发起本次加载的代次；若中途 _load_generation 变化
        （用户取消或发起新加载），直接静默退出，不再投递消息。
        """
        stale = lambda: self._load_generation != gen
        try:
            with tifffile.TiffFile(file_path) as tif:
                if stale():
                    return

                series_list = tif.series
                if not series_list:
                    raise ValueError("TIFF 中没有可读取的 series")
                series = series_list[0]
                axes = getattr(series, "axes", "") or ""
                series_shape = tuple(getattr(series, "shape", ()))
                pages = tif.pages
                n_pages = len(pages)

                # 单页 3D 灰度堆叠：使用 memmap，避免一次 asarray 读入全量。
                # mode='r' 只读映射——代码从不写栈，可写映射反而会在
                # Windows 上把源文件锁死（无法移动/删除/覆盖）。
                # 其余情况（多页 TIFF、单张彩色、4D 彩色）按页或按需读取。
                if (n_pages == 1 and len(series_shape) == 3
                        and "T" in axes and "C" not in axes and "S" not in axes):
                    try:
                        data = tifffile.memmap(file_path, mode='r')
                    except Exception:
                        data = tif.asarray()
                elif n_pages > 1:
                    # 多页 TIFF：逐页读取，每页前检查取消标志。
                    frames = []
                    for i, page in enumerate(pages):
                        if stale():
                            return
                        frames.append(page.asarray())
                        if len(frames) > 1:
                            shapes = {f.shape for f in frames}
                            if len(shapes) > 1:
                                raise ValueError(f"TIFF 页面尺寸不一致: {shapes}")
                        # 每 N 页报一次进度，长加载不再无反馈
                        if (i + 1) % LOAD_PROGRESS_EVERY_PAGES == 0:
                            self._load_queue.put(('progress', gen, i + 1, n_pages))
                    if len(frames) == 1:
                        data = frames[0]
                    else:
                        data = np.stack(frames)
                else:
                    data = tif.asarray()

            if stale():
                return

            stack, note = normalize_tiff_data(data, axes=axes)

            # 单页 3D 且最后一维为 3/4 的 TIFF 在元数据上可能是彩色，也可能
            # 是 W=3/4 的灰度堆叠；这是 TIFF 格式本身的歧义。给出明确提示。
            if (n_pages == 1 and len(series_shape) == 3
                    and series_shape[-1] in (3, 4)):
                note += ("；如实际为 W=3/4 灰度堆叠，请保存为多页 TIFF 后再加载")

            self._load_queue.put(('ok', gen, stack, note, file_path))

        except Exception as e:
            self._load_queue.put(('error', gen, str(e)))

    def load_tiff_stack_from_files(self, file_paths):
        """异步加载文件夹中的多个 TIFF 文件。"""
        gen = self._load_generation + 1
        self._load_generation = gen
        self._set_load_ui(True)
        self.status_var.set("正在读取文件元数据...")
        threading.Thread(target=self._load_stack_worker, args=(file_paths, gen),
                         daemon=True).start()

    def _load_stack_worker(self, file_paths, gen):
        """后台线程：逐文件、逐页读取并统一为灰度帧。

        单个文件损坏/格式不支持只跳过并记录，不拖垮整个文件夹的加载。
        """
        stale = lambda: self._load_generation != gen
        frames = []
        failed_files = []           # [(文件名, 错误信息)]
        total = len(file_paths)
        try:
            for i, fp in enumerate(file_paths):
                if stale():
                    return
                file_path = str(fp)
                try:
                    with tifffile.TiffFile(file_path) as tif:
                        if not tif.series:
                            raise ValueError("无可读取的 series")
                        series = tif.series[0]
                        axes = getattr(series, "axes", "") or ""
                        series_shape = tuple(getattr(series, "shape", ()))
                        pages = tif.pages
                        n_pages = len(pages)

                        if n_pages == 1 and len(series_shape) == 3 and "T" in axes:
                            # 单页 3D 灰度堆叠：逐帧切出（memmap 只读，
                            # 避免可写映射在 Windows 上锁住源文件）
                            try:
                                stack = tifffile.memmap(file_path, mode='r')
                            except Exception:
                                stack = tif.asarray()
                            for si in range(stack.shape[0]):
                                if stale():
                                    return
                                frame = stack[si]
                                frames.append(self._normalize_folder_frame(frame, axes))
                        elif n_pages > 1:
                            for page in pages:
                                if stale():
                                    return
                                frames.append(self._normalize_folder_frame(page.asarray(), axes))
                        else:
                            data = tif.asarray()
                            stack, _note = normalize_tiff_data(data, axes=axes)
                            for si in range(stack.shape[0]):
                                if stale():
                                    return
                                frames.append(stack[si])
                except Exception as e:
                    logger.warning("文件夹模式跳过无法读取的文件 %s: %s", file_path, e)
                    failed_files.append((Path(file_path).name, str(e)))
                    continue

                self._load_queue.put(('progress', gen, i + 1, total))

            if not frames:
                detail = "没有有效的图像文件"
                if failed_files:
                    detail += f"（{len(failed_files)} 个文件读取失败）"
                self._load_queue.put(('error', gen, detail))
                return

            # 混合 dtype 显式统一，避免 numpy 隐式提升造成意外的大数组
            frames = self._unify_frame_dtype(frames)

            shapes = {f.shape for f in frames}
            if len(shapes) > 1:
                self._load_queue.put(('error', gen, f"图像尺寸不一致，无法合并堆叠: {shapes}"))
                return

            stack = np.stack(frames)
            folder = Path(file_paths[0]).parent
            self._load_queue.put(('stack_ok', gen, stack,
                                  f"文件夹 {folder.name}", str(folder), failed_files))

        except Exception as e:
            self._load_queue.put(('error', gen, str(e)))

    @staticmethod
    def _unify_frame_dtype(frames):
        """多文件合并时统一 dtype。

        - dtype 一致：原样返回；
        - 混合整型：提升到能容纳所有值的整型（如 uint8+uint16→uint16）；
        - 混入浮点：用 float32（图像值域可精确表示），
          避免 numpy 默认把 uint8+float 提升成 float64 导致内存翻倍。
        """
        dtypes = {f.dtype for f in frames}
        if len(dtypes) == 1:
            return frames
        common = np.result_type(*dtypes)
        if common == np.float64 and np.float64 not in dtypes:
            common = np.float32
        logger.info("文件夹模式混合 dtype %s，统一为 %s", dtypes, common)
        return [np.asarray(f, dtype=common) for f in frames]

    @staticmethod
    def _normalize_folder_frame(frame, axes=""):
        """把文件夹模式中的单页图像转成灰度帧。"""
        arr = np.asarray(frame)
        axes = axes or ""
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            # 通道在末轴 (H, W, 3/4)
            if arr.shape[-1] in (3, 4):
                return rgb_to_gray_uint8(arr)
            # 平面彩色 (C, H, W)：axes 以 C/S 开头且通道数为 3/4
            if axes[:1] in ("C", "S") and arr.shape[0] in (3, 4):
                return rgb_to_gray_uint8(np.moveaxis(arr, 0, -1))
            # 其余多通道数据取第 0 通道（罕见，至少不静默丢整帧）
            return arr[..., 0]
        if arr.ndim == 4:
            if arr.shape[-1] in (3, 4):
                return rgb_to_gray_uint8(arr[0])
            if arr.shape[-1] == 1:
                return arr[..., 0]
            raise ValueError(f"不支持的页面形状: {arr.shape}")
        raise ValueError(f"不支持的页面形状: {arr.shape}")

    def _set_load_ui(self, loading: bool):
        """加载期间禁用入口按钮并显示取消按钮，结束时恢复。"""
        state = tk.DISABLED if loading else tk.NORMAL
        for btn in (self.open_file_btn, self.open_folder_btn, self.load_project_btn):
            btn.config(state=state)
        if loading:
            self.cancel_load_btn.pack(side=tk.RIGHT, padx=(0, 10))
        else:
            self.cancel_load_btn.pack_forget()
        self.root.config(cursor="watch" if loading else "")

    def cancel_loading(self):
        """用户取消当前后台加载：代次自增使 worker 在下一个检查点退出。"""
        self._load_generation += 1
        self._pending_project = None
        self._set_load_ui(False)
        self.status_var.set("加载已取消")

    def _poll_load_queue(self):
        """主线程轮询后台加载队列（过期代次的消息直接丢弃）。"""
        try:
            while True:
                msg = self._load_queue.get_nowait()
                tag, gen = msg[0], msg[1]
                if gen != self._load_generation:
                    continue
                if tag == 'progress':
                    _, _, done, total = msg
                    self.status_var.set(f"正在加载: {done}/{total}")
                elif tag == 'ok':
                    _, _, stack, note, file_path = msg
                    # 切换文件：先显式释放旧映射的文件句柄再覆盖引用
                    close_tiff_stack(self.tiff_stack)
                    self.tiff_stack = stack
                    self.total_frames = stack.shape[0]
                    # 读取与维度规范化全部成功后才提交路径（回移主树修复），
                    # 失败/取消路径不会走到这里，旧文档状态完整保留。
                    self.file_path = file_path
                    self._reset_state()
                    self._update_ui_after_load()
                    self._maybe_apply_pending_project()
                    self._set_load_ui(False)
                    logger.info("已加载 %s: %d 帧 %s %s", file_path,
                                self.total_frames, stack.shape[1:], stack.dtype)
                    self.status_var.set(
                        f"已加载: {Path(file_path).name} ({self.total_frames}帧, {note})")
                elif tag == 'stack_ok':
                    _, _, stack, note, folder_path, failed_files = msg
                    # 切换数据源：先显式释放旧映射的文件句柄再覆盖引用
                    close_tiff_stack(self.tiff_stack)
                    self.tiff_stack = stack
                    self.total_frames = stack.shape[0]
                    # 文件夹模式：file_path 记为堆叠来源目录（与主树语义一致），
                    # 避免 update_file_info 对 None 取 Path 抛 TypeError。
                    self.file_path = folder_path
                    self._reset_state()
                    self._update_ui_after_load()
                    self._maybe_apply_pending_project()
                    self._set_load_ui(False)
                    logger.info("已加载文件夹堆叠 %s: %d 帧 %s %s（%d 个文件读取失败）",
                                folder_path, self.total_frames, stack.shape[1:],
                                stack.dtype, len(failed_files))
                    if failed_files:
                        shown = "\n".join(f"- {name}: {err}"
                                          for name, err in failed_files[:10])
                        more = (f"\n…等共 {len(failed_files)} 个失败文件"
                                if len(failed_files) > 10 else "")
                        messagebox.showwarning(
                            "部分文件读取失败",
                            f"以下文件已跳过，其余文件正常加载：\n{shown}{more}")
                    self.status_var.set(f"已加载堆叠: {self.total_frames}帧 ({note})")
                elif tag == 'error':
                    had_pending = self._pending_project is not None
                    self._pending_project = None
                    self._set_load_ui(False)
                    self.status_var.set("加载失败")
                    detail = f"加载文件失败: {msg[2]}"
                    if had_pending:
                        detail += "\n已放弃恢复项目数据。"
                    messagebox.showerror("错误", detail)
        except queue.Empty:
            pass
        except Exception as e:
            self._set_load_ui(False)
            self.status_var.set("加载失败")
            messagebox.showerror("错误", f"加载处理异常: {e}")
        self._load_after_id = self.root.after(LOAD_POLL_INTERVAL_MS,
                                              self._poll_load_queue)

    def on_close(self):
        """窗口关闭：未保存修改需确认；停止后台加载轮询并销毁窗口。"""
        if self._dirty and not messagebox.askyesno(
                "未保存的修改", "当前测量数据尚未保存，确定退出并丢弃吗？"):
            return
        if self._load_after_id is not None:
            try:
                self.root.after_cancel(self._load_after_id)
            except Exception:
                pass
            self._load_after_id = None
        # 代次自增使仍在运行的后台 worker 尽快退出
        self._load_generation += 1
        # 退出路径：显式释放内存映射的文件句柄，解除对源文件的锁定
        close_tiff_stack(self.tiff_stack)
        self.tiff_stack = None
        plt.close('all')
        self.root.destroy()

    # ==================== 信息显示 ====================

    def update_file_info(self):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)

        if self.tiff_stack is not None:
            info = f"文件: {Path(self.file_path).name}\n"
            info += f"总帧数: {self.total_frames}\n"
            info += f"图像尺寸: {self.tiff_stack.shape[2]} x {self.tiff_stack.shape[1]}\n"
            info += f"数据类型: {self.tiff_stack.dtype}\n"
            # 只统计当前帧（大堆叠全量 min/max 开销大）
            frame = self.tiff_stack[self.current_frame]
            info += f"当前帧值范围: {frame.min():.2f} - {frame.max():.2f}"
            self.info_text.insert(tk.END, info)

        self.info_text.config(state=tk.DISABLED)

    def update_ruler_info(self):
        self.ruler_text.config(state=tk.NORMAL)
        self.ruler_text.delete(1.0, tk.END)

        if self.ruler_calibrated:
            info = f"标尺已校准\n"
            info += f"比例: 1 像素 = {1 / self.pixel_to_unit_ratio:.4f} {self.unit_name}\n"
            info += f"或: 1 {self.unit_name} = {self.pixel_to_unit_ratio:.4f} 像素\n"
            info += f"标尺: {self.ruler_length_pixels:.1f}px = {self.ruler_length_units:.1f}{self.unit_name}"
            self.ruler_info_label.config(text=f"1px={1 / self.pixel_to_unit_ratio:.2f}{self.unit_name}")
        else:
            info = "标尺未校准\n比例: 1 像素 = 1 像素"
            self.ruler_info_label.config(text="未校准")

        self.ruler_text.insert(tk.END, info)
        self.ruler_text.config(state=tk.DISABLED)

    # ==================== 核心显示（坐标系统已修复） ====================

    def display_current_frame(self):
        """
        显示当前帧。
        ★ 坐标系统说明（v3：左上原点，y 向下 = 行号，与 ImageJ 等图像工具一致）：
        - imshow 使用 extent=(0,w,h,0)，axes 事件坐标与像素一一对应：
          xdata 范围 [0,w]，ydata 范围 [0,h]（0=第一行）。
        - 存储坐标 = 事件坐标 (x, y)，显示时无需任何翻转。
        - 缩放/平移仅通过 set_xlim/set_ylim 控制视窗；ylim 保持 bottom>top 防反转。
        """
        if self.tiff_stack is None:
            return

        try:
            self.ax.clear()
            # ax.clear() 使所有 artist 失效，清掉引用（下次绘制时重建）
            self._temp_ruler_line = None
            self._temp_poly_line = None
            self._temp_poly_dots = None
            self._temp_preview_line = None
            frame_idx = self.current_frame
            img_data = self.tiff_stack[frame_idx]

            # 确定显示内容
            if len(img_data.shape) == 2:
                display_img = img_data
                cmap = 'gray'
            elif len(img_data.shape) == 3:
                if img_data.shape[2] in (3, 4):
                    display_img = img_data
                    cmap = None
                else:
                    display_img = img_data[:, :, 0]
                    cmap = 'gray'
            else:
                display_img = img_data
                cmap = 'gray'

            h, w = display_img.shape[:2]

            # 显示图像：extent 使 axes 坐标与像素坐标一一对应。
            # 默认 imshow 每帧独立 min/max 归一化，会掩盖帧间真实的
            # 强度变化——锁定对比度后统一用记录的 vmin/vmax 显示。
            imshow_kwargs = dict(cmap=cmap, aspect='equal', extent=(0, w, h, 0))
            if self._contrast_lock is not None:
                imshow_kwargs.update(vmin=self._contrast_lock[0],
                                     vmax=self._contrast_lock[1])
            self.ax.imshow(display_img, **imshow_kwargs)

            # 通过 xlim/ylim 控制缩放和平移
            center_x, center_y = w / 2.0, h / 2.0
            view_w = w / self.zoom_factor
            view_h = h / self.zoom_factor

            left = center_x - view_w / 2 + self.pan_offset[0]
            right = center_x + view_w / 2 + self.pan_offset[0]
            # imshow y 轴向下，bottom 数值上应大于 top；防反转。
            bottom = center_y + view_h / 2 + self.pan_offset[1]
            top = center_y - view_h / 2 + self.pan_offset[1]
            if bottom < top:
                bottom, top = top, bottom

            self.ax.set_xlim(left, right)
            self.ax.set_ylim(bottom, top)

            # ★ 绘制已保存的多边形（存储坐标即显示坐标，无翻转）
            if frame_idx in self.polygons:
                for poly_id, points in self.polygons[frame_idx].items():
                    if points and len(points) >= 3:
                        color = self.polygon_colors.get(poly_id, 'red')
                        # to_rgba 对十六进制/命名色统一处理；color+'20' 只适用于
                        # 十六进制色，项目 JSON 中的命名色会导致 Invalid RGBA
                        polygon = MplPolygon(points, closed=True,
                                             edgecolor=color,
                                             facecolor=to_rgba(color, 0.15),
                                             linewidth=2)
                        self.ax.add_patch(polygon)

                        # 标签（使用质心）
                        cx = sum(p[0] for p in points) / len(points)
                        cy = sum(p[1] for p in points) / len(points)
                        name = self.polygon_names.get(poly_id, str(poly_id))
                        self.ax.text(cx, cy, name, color='white',
                                     fontsize=9, fontweight='bold',
                                     ha='center', va='center',
                                     bbox=dict(boxstyle="round,pad=0.3",
                                               facecolor=color, alpha=0.8))

            # 绘制正在绘制的多边形（临时 artist 在 _refresh 中重建）
            if self.drawing_mode:
                self._refresh_in_progress_artists()

            # 绘制标尺
            if self.ruler_points and len(self.ruler_points) == 2:
                (x1, y1), (x2, y2) = self.ruler_points

                self.ax.plot([x1, x2], [y1, y2], 'y-', linewidth=3, alpha=0.8)
                self.ax.plot([x1, x2], [y1, y2], 'yo', markersize=8, alpha=0.8)

                # 标尺标签
                length_pixels = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if self.ruler_calibrated:
                    length_units = length_pixels / self.pixel_to_unit_ratio
                    label_text = f"{length_pixels:.1f}px = {length_units:.1f}{self.unit_name}"
                else:
                    label_text = f"{length_pixels:.1f}px"

                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                self.ax.text(mid_x, mid_y - 15, label_text,
                             color='yellow', fontsize=10, fontweight='bold',
                             ha='center', va='center',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))

            self.ax.set_title(f"帧 {frame_idx + 1}/{self.total_frames}")
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw()

            self.update_polygon_list()
            self.update_statistics()
            self.update_file_info()

        except Exception as e:
            # 打包成无控制台 exe 后 print 不可见——写日志并对用户节流弹窗
            logger.exception("显示图像时出错")
            now = time.monotonic()
            if now - self._last_display_err > DISPLAY_ERR_THROTTLE_S:
                self._last_display_err = now
                messagebox.showwarning(
                    "显示异常",
                    f"渲染当前帧时出错:\n{e}\n\n详情已写入日志（{DISPLAY_ERR_THROTTLE_S:.0f} 秒内不重复提示）。")

    # ==================== 帧导航 ====================

    def on_frame_changed(self):
        try:
            frame_num = int(self.frame_var.get()) - 1
        except (ValueError, tk.TclError):
            return
        if 0 <= frame_num < self.total_frames:
            self.current_frame = frame_num
            self.display_current_frame()
        else:
            # 越界输入给出反馈并回显当前帧，而不是静默忽略
            valid = max(1, self.total_frames)
            self.status_var.set(f"帧号超出范围，有效范围 1–{valid}")
            self.frame_var.set(str(self.current_frame + 1))

    def prev_frame(self):
        if self.tiff_stack is not None and self.current_frame > 0:
            self.current_frame -= 1
            self.frame_var.set(str(self.current_frame + 1))
            self._debounced_display()

    def next_frame(self):
        if self.tiff_stack is not None and self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.frame_var.set(str(self.current_frame + 1))
            self._debounced_display()

    def _debounced_display(self):
        """★ 帧导航防抖：快速连续切帧时只渲染最后一次"""
        if self._display_after_id:
            self.root.after_cancel(self._display_after_id)
        self._display_after_id = self.root.after(FRAME_NAV_DEBOUNCE_MS,
                                                 self.display_current_frame)

    # ==================== 模式切换 ====================

    def toggle_drawing_mode(self):
        if self.ruler_mode:
            self.toggle_ruler_mode()

        self.drawing_mode = not self.drawing_mode
        if self.drawing_mode:
            self.draw_btn.config(text="取消绘制(Esc)")
            self.current_polygon_points = []
            self.status_var.set("绘制模式: 左键添加点, 右键/Backspace撤销, Enter完成, Esc取消")
        else:
            self.draw_btn.config(text="绘制多边形")
            self.current_polygon_points = []
            self.status_var.set("就绪")
        self.display_current_frame()

    def toggle_ruler_mode(self):
        if self.drawing_mode:
            self.toggle_drawing_mode()

        self.ruler_mode = not self.ruler_mode
        if self.ruler_mode:
            self.ruler_btn.config(text="取消标尺")
            self.ruler_points = []
            self.status_var.set("标尺模式: 点击图像设置标尺起点和终点")
        else:
            self.ruler_btn.config(text="设置标尺")
            self.ruler_points = []
            self.status_var.set("就绪")
        self.display_current_frame()

    def cancel_drawing(self):
        """取消当前绘制（Esc键）"""
        if self.drawing_mode:
            self.current_polygon_points = []
            self.drawing_mode = False
            self.draw_btn.config(text="绘制多边形")
            self.display_current_frame()
            self.status_var.set("已取消绘制")
        elif self.ruler_mode:
            self.ruler_points = []
            self.ruler_mode = False
            self.ruler_btn.config(text="设置标尺")
            self.display_current_frame()
            self.status_var.set("已取消标尺设置")

    # ==================== 鼠标事件（坐标系统已修复） ====================

    def on_canvas_click(self, event):
        """
        画布点击事件。
        ★ v3 坐标系：event.xdata/ydata 即像素坐标（extent=(0,w,h,0)，
        原点左上角，y 向下 = 行号），直接 clamp 后存储，无任何翻转。
        """
        if self.tiff_stack is None:
            return
        if event.inaxes != self.ax:
            return

        img_height = self.tiff_stack.shape[1]
        img_width = self.tiff_stack.shape[2]

        # clamp 到图像范围 [0,w]×[0,h]，避免边界外顶点。
        x, y = clamp_point(event.xdata, event.ydata, img_width, img_height)

        # 右键撤销
        if event.button == 3:
            if self.drawing_mode:
                self.undo_last_point()
            return

        # 仅左键继续
        if event.button != 1:
            return

        # 标尺模式
        if self.ruler_mode:
            if len(self.ruler_points) < 2:
                self.ruler_points.append((x, y))
                self.status_var.set(f"标尺点 {len(self.ruler_points)}: ({int(x)}, {int(y)})")

                if len(self.ruler_points) == 2:
                    (px1, py1), (px2, py2) = self.ruler_points
                    if np.hypot(px2 - px1, py2 - py1) <= 1e-9:
                        # 两点重合会导致比例=0、后续换算除零——要求重画
                        messagebox.showwarning(
                            "标尺无效", "标尺两点重合，长度为 0。\n请重新点击两个不同的点。")
                        self.ruler_points = []
                        self.status_var.set("标尺模式: 请重新点击标尺起点和终点")
                        return
                    self._show_ruler_dialog()
            return

        # 多边形绘制模式
        if self.drawing_mode:
            self.current_polygon_points.append((x, y))
            self.status_var.set(f"绘制中: {len(self.current_polygon_points)}个点 (Enter完成, 右键撤销)")
            self._update_in_progress_overlay()

    def on_canvas_motion(self, event):
        """鼠标移动事件（★ 性能优化：不全量重绘）"""
        if self.tiff_stack is None:
            return

        if event.inaxes == self.ax and event.xdata is not None:
            img_height = self.tiff_stack.shape[1]
            img_width = self.tiff_stack.shape[2]
            cx, cy = clamp_point(event.xdata, event.ydata, img_width, img_height)
            x, y = int(cx), int(cy)
            self.coord_var.set(f"坐标: ({x}, {y})")

            # 标尺模式：绘制临时线（使用 draw_idle 合并刷新）
            if self.ruler_mode and len(self.ruler_points) == 1:
                if self._temp_ruler_line is not None:
                    try:
                        self._temp_ruler_line.remove()
                    except Exception:
                        pass

                x1, y1 = self.ruler_points[0]
                self._temp_ruler_line, = self.ax.plot(
                    [x1, event.xdata], [y1, event.ydata],
                    'y--', linewidth=2, alpha=0.6)
                self.canvas.draw_idle()

            # 绘制模式：从最后顶点到光标画橡皮筋预览线
            if self.drawing_mode and self.current_polygon_points:
                if self._temp_preview_line is not None:
                    try:
                        self._temp_preview_line.remove()
                    except Exception:
                        pass
                lx, ly = self.current_polygon_points[-1]
                self._temp_preview_line, = self.ax.plot(
                    [lx, cx], [ly, cy],
                    'r--', linewidth=1.5, alpha=0.6)
                self.canvas.draw_idle()
        else:
            self.coord_var.set("坐标: (0, 0)")

    def on_canvas_scroll(self, event):
        if event.inaxes == self.ax and event.xdata is not None:
            anchor = (event.xdata, event.ydata)
            if event.button == 'up':
                self.zoom(1.2, anchor=anchor)
            elif event.button == 'down':
                self.zoom(0.8, anchor=anchor)

    def on_canvas_pan_start(self, event):
        """中键开始平移"""
        if event.inaxes == self.ax and event.button == 2:
            if not self.drawing_mode and not self.ruler_mode:
                self.pan_start = (event.xdata, event.ydata)

    def on_canvas_pan(self, event):
        """中键平移（★ 修复：直接加上 axes 坐标差值）"""
        if self.pan_start and event.inaxes == self.ax and event.xdata is not None:
            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]

            # ★ 修复：平移量直接是 axes 坐标差，不需要乘 zoom_factor
            self.pan_offset[0] -= dx
            self.pan_offset[1] -= dy

            self.pan_start = (event.xdata, event.ydata)

            # ★ 性能优化：平移时仅更新视窗范围，用 draw_idle 合并刷新
            h, w = self.tiff_stack[self.current_frame].shape[:2]
            center_x, center_y = w / 2.0, h / 2.0
            view_w = w / self.zoom_factor
            view_h = h / self.zoom_factor
            left = center_x - view_w / 2 + self.pan_offset[0]
            right = center_x + view_w / 2 + self.pan_offset[0]
            bottom = center_y + view_h / 2 + self.pan_offset[1]
            top = center_y - view_h / 2 + self.pan_offset[1]
            if bottom < top:
                bottom, top = top, bottom
            self.ax.set_xlim(left, right)
            self.ax.set_ylim(bottom, top)
            self.canvas.draw_idle()

    def on_canvas_pan_end(self, event):
        """平移结束后完整刷新一次"""
        if self.pan_start is not None:
            self.pan_start = None
            self.display_current_frame()  # 平移结束后完整重绘（含多边形位置更新）

    # ==================== 多边形操作 ====================

    def finish_polygon(self):
        """完成多边形绘制"""
        if not self.drawing_mode:
            return
        if len(self.current_polygon_points) < 3:
            if self.current_polygon_points:
                messagebox.showwarning("警告", "至少需要3个点来创建多边形")
            return

        frame_idx = self.current_frame
        pixel_area = polygon_area(self.current_polygon_points)

        # 自相交/退化检测：鞋带公式对这类形状给出的面积可能不符合几何
        # 直觉（如蝴蝶形取两区域之差而非之和）。需用户明确确认才保存。
        if polygon_self_intersects(self.current_polygon_points):
            keep = messagebox.askyesno(
                "自相交警告",
                "多边形存在自相交边或退化边（折返/压边）。\n"
                "鞋带公式对此类形状计算出的面积可能不是几何直觉上的面积。\n\n"
                "是否仍然保存？（选“否”可继续编辑顶点）")
            if not keep:
                self.status_var.set("已保留当前顶点，请调整后再次完成绘制")
                return

        # 零面积（顶点全部共线/重合）：面积无意义，同样需要确认
        if pixel_area <= 0:
            keep = messagebox.askyesno(
                "退化多边形",
                "该多边形面积为 0（顶点全部共线或重合）。\n\n"
                "是否仍然保存？（选“否”可继续编辑顶点）")
            if not keep:
                self.status_var.set("已保留当前顶点，请调整后再次完成绘制")
                return

        if frame_idx not in self.polygons:
            self.polygons[frame_idx] = {}

        poly_id = self.next_polygon_id
        self.next_polygon_id += 1

        self.polygons[frame_idx][poly_id] = self.current_polygon_points.copy()
        self.polygon_roots[poly_id] = poly_id  # 新多边形是自己的 root

        color_idx = (poly_id - 1) % len(self.color_palette)
        self.polygon_colors[poly_id] = self.color_palette[color_idx]

        unit_area = convert_area_to_units(pixel_area, self.pixel_to_unit_ratio,
                                          self.ruler_calibrated)

        if frame_idx not in self.measurements:
            self.measurements[frame_idx] = {}
        self.measurements[frame_idx][poly_id] = {
            'pixel_area': pixel_area,
            'unit_area': unit_area
        }
        self._dirty = True

        # 重置绘制状态（保持绘制模式以便连续绘制）
        self.current_polygon_points = []

        self.display_current_frame()

        if self.ruler_calibrated:
            self.status_var.set(f"多边形 #{poly_id}: {pixel_area:.1f}px² = {unit_area:.2f}{self.unit_name}² | 继续绘制或按Esc退出")
        else:
            self.status_var.set(f"多边形 #{poly_id}: {pixel_area:.1f}px² | 继续绘制或按Esc退出")

    def undo_last_point(self):
        """撤销最后一个点（Backspace/右键）"""
        if self.drawing_mode and self.current_polygon_points:
            self.current_polygon_points.pop()
            self._update_in_progress_overlay()
            self.status_var.set(f"已撤销，当前 {len(self.current_polygon_points)} 个点")

    def _refresh_in_progress_artists(self):
        """重建“绘制中”多边形的临时连线/顶点标记（不触发重绘）。"""
        for attr in ('_temp_poly_line', '_temp_poly_dots'):
            artist = getattr(self, attr, None)
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass
                setattr(self, attr, None)

        pts = self.current_polygon_points
        if not pts:
            return
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if len(pts) >= 2:
            self._temp_poly_line, = self.ax.plot(xs, ys, 'r-', linewidth=2, alpha=0.8)
        self._temp_poly_dots, = self.ax.plot(xs, ys, 'ro', markersize=6, alpha=0.8)

    def _update_in_progress_overlay(self):
        """绘制中加点/撤点时只更新临时 artist（避免整帧 imshow 重绘）。"""
        self._refresh_in_progress_artists()
        self.canvas.draw_idle()

    def update_polygon_list(self):
        """更新多边形列表（★ 支持自定义名称）"""
        self.poly_listbox.delete(0, tk.END)

        frame_idx = self.current_frame
        if frame_idx in self.polygons and self.polygons[frame_idx]:
            for poly_id in sorted(self.polygons[frame_idx].keys()):
                points = self.polygons[frame_idx][poly_id]
                measurement = self.measurements.get(frame_idx, {}).get(poly_id, {})
                pixel_area = measurement.get('pixel_area', 0)
                unit_area = measurement.get('unit_area', 0)

                name = self.polygon_names.get(poly_id, f"#{poly_id}")

                if self.ruler_calibrated:
                    item_text = f"{name:8s}  {unit_area:9.2f}{self.unit_name}²  {len(points):3d}"
                else:
                    item_text = f"{name:8s}  {pixel_area:9.1f}px²  {len(points):3d}"

                self.poly_listbox.insert(tk.END, item_text)

    def update_statistics(self):
        """更新统计信息"""
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)

        frame_idx = self.current_frame

        if frame_idx in self.measurements and self.measurements[frame_idx]:
            pixel_areas = [m['pixel_area'] for m in self.measurements[frame_idx].values()]
            unit_areas = [m['unit_area'] for m in self.measurements[frame_idx].values()]

            stats = f"当前帧 #{frame_idx + 1}\n"
            stats += f"多边形数量: {len(pixel_areas)}\n\n"

            if self.ruler_calibrated:
                stats += f"--- 实际单位 ({self.unit_name}) ---\n"
                stats += f"总面积: {sum(unit_areas):.2f} {self.unit_name}²\n"
                stats += f"平均面积: {np.mean(unit_areas):.2f} {self.unit_name}²\n"
                stats += f"最小: {min(unit_areas):.2f} / 最大: {max(unit_areas):.2f}\n"
                if len(unit_areas) >= 2:
                    # 样本标准差（ddof=1），与导出汇总一致
                    stats += f"标准差: {np.std(unit_areas, ddof=1):.2f} {self.unit_name}²\n\n"
                else:
                    stats += "\n"
                stats += f"--- 像素 ---\n"
                stats += f"总面积: {sum(pixel_areas):.1f} px²\n"
                if self.pixel_to_unit_ratio:
                    stats += f"比例: 1px = {1 / self.pixel_to_unit_ratio:.4f}{self.unit_name}"
            else:
                stats += f"总面积: {sum(pixel_areas):.1f} px²\n"
                stats += f"平均: {np.mean(pixel_areas):.1f} px²\n"
                stats += f"最小: {min(pixel_areas):.1f} / 最大: {max(pixel_areas):.1f}\n"
                if len(pixel_areas) >= 2:
                    stats += f"标准差: {np.std(pixel_areas, ddof=1):.1f} px²\n"
                stats += f"（标尺未校准）"
        else:
            stats = "当前帧无测量数据"

        self.stats_text.insert(tk.END, stats)
        self.stats_text.config(state=tk.DISABLED)

    def clear_current_frame(self):
        frame_idx = self.current_frame
        if frame_idx not in self.polygons:
            self.status_var.set(f"帧 #{frame_idx + 1} 没有多边形")
            return
        if not messagebox.askyesno(
                "确认", f"确定要清除帧 #{frame_idx + 1} 的所有多边形吗？"):
            return
        del self.polygons[frame_idx]
        if frame_idx in self.measurements:
            del self.measurements[frame_idx]
        self._dirty = True
        self.display_current_frame()
        self.status_var.set(f"已清除帧 #{frame_idx + 1} 的所有多边形")

    def clear_all_polygons(self):
        if messagebox.askyesno("确认", "确定要清除所有帧的多边形吗？"):
            self.polygons = {}
            self.polygon_colors = {}
            self.polygon_names = {}
            self.polygon_roots = {}
            self.measurements = {}
            self.next_polygon_id = 1
            self._dirty = True
            self.display_current_frame()
            self.status_var.set("已清除所有多边形")

    def rename_polygon(self):
        """★ 修复：真正实现重命名功能"""
        selection = self.poly_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "请先选择一个多边形")
            return

        # 从列表文本中解析多边形ID
        item_text = self.poly_listbox.get(selection[0])
        frame_idx = self.current_frame
        if frame_idx not in self.polygons:
            return

        # 找到对应的 poly_id（按排序顺序）
        sorted_ids = sorted(self.polygons[frame_idx].keys())
        idx = selection[0]
        if idx >= len(sorted_ids):
            return
        poly_id = sorted_ids[idx]

        current_name = self.polygon_names.get(poly_id, f"#{poly_id}")
        new_name = simpledialog.askstring("重命名多边形",
                                          f"当前名称: {current_name}\n输入新名称:",
                                          parent=self.root)
        if new_name and new_name.strip():
            self.polygon_names[poly_id] = new_name.strip()
            self._dirty = True
            self.update_polygon_list()
            self.display_current_frame()
            self.status_var.set(f"多边形 #{poly_id} 已重命名为: {new_name.strip()}")

    def delete_selected_polygon(self):
        # 绘制模式下 Delete 键不做删除，避免误删已保存的多边形
        if self.drawing_mode:
            return
        selection = self.poly_listbox.curselection()
        if not selection:
            return

        frame_idx = self.current_frame
        if frame_idx not in self.polygons:
            return

        sorted_ids = sorted(self.polygons[frame_idx].keys())
        idx = selection[0]
        if idx >= len(sorted_ids):
            return
        poly_id = sorted_ids[idx]

        if frame_idx in self.polygons and poly_id in self.polygons[frame_idx]:
            del self.polygons[frame_idx][poly_id]
        if not self.polygons.get(frame_idx):
            # 帧内多边形已删空：整键删除，避免汇总/图表出现"0 个多边形"的幽灵行
            self.polygons.pop(frame_idx, None)
        if frame_idx in self.measurements and poly_id in self.measurements[frame_idx]:
            del self.measurements[frame_idx][poly_id]
        if not self.measurements.get(frame_idx):
            self.measurements.pop(frame_idx, None)
        if poly_id in self.polygon_colors:
            del self.polygon_colors[poly_id]
        if poly_id in self.polygon_names:
            del self.polygon_names[poly_id]
        self.polygon_roots.pop(poly_id, None)
        self._dirty = True

        self.display_current_frame()
        self.status_var.set(f"已删除多边形 #{poly_id}")

    def copy_to_next_frame(self):
        frame_idx = self.current_frame

        if frame_idx not in self.polygons or not self.polygons[frame_idx]:
            messagebox.showwarning("警告", "当前帧没有多边形可复制")
            return
        if frame_idx >= self.total_frames - 1:
            messagebox.showwarning("警告", "已经是最后一帧")
            return

        next_frame = frame_idx + 1
        existing = self.polygons.get(next_frame, {})
        if existing:
            # 防止连点造成重复副本：目标帧已有数据时先确认
            if not messagebox.askyesno(
                    "目标帧已有数据",
                    f"帧 #{next_frame + 1} 已有 {len(existing)} 个多边形。\n"
                    "继续复制将新增副本（不会覆盖已有数据），确定吗？"):
                return
        if next_frame not in self.polygons:
            self.polygons[next_frame] = {}
        if next_frame not in self.measurements:
            self.measurements[next_frame] = {}

        for poly_id, points in self.polygons[frame_idx].items():
            new_poly_id = self.next_polygon_id
            self.next_polygon_id += 1

            self.polygons[next_frame][new_poly_id] = [p for p in points]
            # 复制继承 root，面积变化图据此把各帧副本连成一条追踪线
            self.polygon_roots[new_poly_id] = self.polygon_roots.get(poly_id, poly_id)

            if poly_id in self.polygon_colors:
                self.polygon_colors[new_poly_id] = self.polygon_colors[poly_id]
            if poly_id in self.polygon_names:
                self.polygon_names[new_poly_id] = self.polygon_names[poly_id]

            pixel_area = polygon_area(points)
            unit_area = convert_area_to_units(pixel_area, self.pixel_to_unit_ratio,
                                              self.ruler_calibrated)
            self.measurements[next_frame][new_poly_id] = {
                'pixel_area': pixel_area,
                'unit_area': unit_area
            }

        self.status_var.set(f"已将 {len(self.polygons[frame_idx])} 个多边形复制到帧 #{next_frame + 1}")
        self._dirty = True

    # ==================== 视图控制 ====================

    def zoom(self, factor, anchor=None):
        """缩放视图。

        anchor 为 (x, y) 数据坐标时以该点为锚缩放（滚轮场景下光标下的
        内容保持不动）；anchor 为 None 时以当前视图中心缩放（工具栏按钮）。
        滚轮快速缩放走防抖，只渲染最后一次。
        """
        old_factor = self.zoom_factor
        self.zoom_factor = max(ZOOM_MIN, min(self.zoom_factor * factor, ZOOM_MAX))
        if self.zoom_factor == old_factor:
            return  # 已到缩放边界，视图不变
        if anchor is not None and self.tiff_stack is not None:
            h, w = self.tiff_stack[self.current_frame].shape[:2]
            center_x, center_y = w / 2.0, h / 2.0
            ax_cx = center_x + self.pan_offset[0]  # 当前视图中心（数据坐标）
            ax_cy = center_y + self.pan_offset[1]
            x_c, y_c = anchor
            # 缩放后让 anchor 点保持在原屏幕位置：cx' = xc - (xc - cx)/factor
            self.pan_offset[0] = x_c - (x_c - ax_cx) / factor - center_x
            self.pan_offset[1] = y_c - (y_c - ax_cy) / factor - center_y
        self._debounced_display()

    def reset_view(self):
        self.zoom_factor = 1.0
        self.pan_offset = [0.0, 0.0]
        self.display_current_frame()

    # ==================== 标尺功能 ====================

    def manual_ruler_input(self):
        """手动输入标尺比例"""
        dialog = tk.Toplevel(self.root)
        dialog.title("手动设置标尺比例")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()

        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

        ttk.Label(dialog, text="已知长度（像素）:").pack(pady=(20, 5))
        pixel_var = tk.StringVar(value="100")
        ttk.Entry(dialog, textvariable=pixel_var, width=20).pack()

        ttk.Label(dialog, text="对应实际长度:").pack(pady=(10, 5))
        unit_var = tk.StringVar(value="1000")
        ttk.Entry(dialog, textvariable=unit_var, width=20).pack()

        ttk.Label(dialog, text="单位:").pack(pady=(10, 5))
        unit_name_var = tk.StringVar(value="nm")
        ttk.Combobox(dialog, textvariable=unit_name_var,
                     values=["nm", "μm", "mm", "cm", "m"],
                     width=18, state="readonly").pack()

        example_frame = ttk.LabelFrame(dialog, text="示例", padding="10")
        example_frame.pack(pady=15, padx=20, fill=tk.X)
        ttk.Label(example_frame, text="例如：100像素 = 1000nm\n表示1像素 = 10nm").pack()

        def apply_ruler():
            try:
                ratio = ruler_ratio(pixel_var.get(), unit_var.get())
            except ValueError as e:
                messagebox.showerror("错误", str(e), parent=dialog)
                return

            self.pixel_to_unit_ratio = ratio
            self.unit_name = unit_name_var.get()
            self.ruler_length_pixels = float(pixel_var.get())
            self.ruler_length_units = float(unit_var.get())
            self.ruler_calibrated = True
            self._dirty = True

            # ★ 修复：标尺变更后重新计算所有面积
            self.recalculate_all_measurements()

            self.update_ruler_info()
            self.update_statistics()
            self.display_current_frame()
            dialog.destroy()
            self.status_var.set(
                f"标尺已设置: {self.ruler_length_pixels}px = {self.ruler_length_units}{self.unit_name}")

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=15)
        ttk.Button(btn_frame, text="应用", command=apply_ruler).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy).pack(side=tk.LEFT, padx=10)

    def _show_ruler_dialog(self):
        """标尺两点设置完成后弹出对话框"""
        x1, y1 = self.ruler_points[0]
        x2, y2 = self.ruler_points[1]
        length_pixels = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        dialog = tk.Toplevel(self.root)
        dialog.title("输入标尺实际长度")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()

        dialog.update_idletasks()
        x_pos = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y_pos = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x_pos}+{y_pos}")

        ttk.Label(dialog, text=f"标尺长度: {length_pixels:.2f} 像素\n请输入对应的实际长度:").pack(pady=20)

        input_frame = ttk.Frame(dialog)
        input_frame.pack(pady=10)

        length_var = tk.StringVar(value="1000")
        length_entry = ttk.Entry(input_frame, textvariable=length_var, width=15)
        length_entry.pack(side=tk.LEFT, padx=5)

        unit_var = tk.StringVar(value="nm")
        ttk.Combobox(input_frame, textvariable=unit_var,
                     values=["nm", "μm", "mm", "cm", "m"],
                     width=8, state="readonly").pack(side=tk.LEFT, padx=5)

        def apply_ruler_length():
            try:
                ratio = ruler_ratio(length_pixels, length_var.get())
            except ValueError as e:
                messagebox.showerror("错误", str(e), parent=dialog)
                return

            self.pixel_to_unit_ratio = ratio
            self.unit_name = unit_var.get()
            self.ruler_length_pixels = length_pixels
            self.ruler_length_units = float(length_var.get())
            self.ruler_calibrated = True
            self._dirty = True

            self.ruler_mode = False
            self.ruler_btn.config(text="设置标尺")

            # ★ 修复：标尺变更后重新计算所有面积
            self.recalculate_all_measurements()

            self.update_ruler_info()
            self.update_statistics()
            self.display_current_frame()
            dialog.destroy()
            self.status_var.set(
                f"标尺已校准: {length_pixels:.1f}px = {self.ruler_length_units:.1f}{self.unit_name}")

        def cancel():
            self.ruler_points = []
            self.ruler_mode = False
            self.ruler_btn.config(text="设置标尺")
            self.display_current_frame()
            dialog.destroy()
            self.status_var.set("标尺设置已取消")

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=20)
        ttk.Button(btn_frame, text="确定", command=apply_ruler_length).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="取消", command=cancel).pack(side=tk.LEFT, padx=10)

        length_entry.focus()
        length_entry.select_range(0, tk.END)

    def clear_ruler(self):
        self.ruler_mode = False
        self.ruler_points = []
        self.pixel_to_unit_ratio = 1.0
        self.ruler_calibrated = False
        self.unit_name = "nm"
        self.ruler_length_pixels = 0
        self.ruler_length_units = 0
        self.ruler_btn.config(text="设置标尺")
        self._dirty = True

        # 重新计算面积（回到像素单位）
        self.recalculate_all_measurements()

        self.update_ruler_info()
        self.update_statistics()
        self.display_current_frame()
        self.status_var.set("标尺已清除")

    def recalculate_all_measurements(self):
        """标尺变更后从顶点重建全部测量值（单一数据源）。"""
        self.measurements = rebuild_measurements(
            self.polygons, self.pixel_to_unit_ratio, self.ruler_calibrated)

    # ==================== 数据导出 ====================

    @staticmethod
    def _escape_csv_formula(value):
        """防 CSV 公式注入：以 =+-@ 或制表/回车开头的文本前置单引号。"""
        text = str(value)
        if text[:1] in ('=', '+', '-', '@', '\t', '\r'):
            return "'" + text
        return text

    def _export_summary(self):
        """生成汇总数据（各帧合计 + 全局统计），供导出为第二个文件/工作表。"""
        px_totals, unit_totals = [], []
        rows = []
        for frame_idx in sorted(self.polygons.keys()):
            if not self.polygons[frame_idx]:
                continue  # 跳过空帧（防御：正常路径已不留空帧 dict）
            n = len(self.polygons[frame_idx])
            px_total = sum(
                m['pixel_area'] for m in self.measurements.get(frame_idx, {}).values())
            unit_total = sum(
                m['unit_area'] for m in self.measurements.get(frame_idx, {}).values())
            px_totals.append(px_total)
            unit_totals.append(unit_total)
            row = {'帧号': frame_idx + 1, '多边形数量': n,
                   '总面积_像素²': px_total}
            if self.ruler_calibrated:
                row[f'总面积_{self.unit_name}²'] = unit_total
            rows.append(row)

        flat_px = [m['pixel_area']
                   for fd in self.measurements.values() for m in fd.values()]
        flat_unit = [m['unit_area']
                     for fd in self.measurements.values() for m in fd.values()]
        stats = {'帧号': '全局', '多边形数量': len(flat_px),
                 '总面积_像素²': sum(flat_px)}
        if self.ruler_calibrated:
            stats[f'总面积_{self.unit_name}²'] = sum(unit_totals)
        if flat_px:
            stats['平均面积_像素²'] = float(np.mean(flat_px))
            stats['中位数_像素²'] = float(np.median(flat_px))
            stats['最小_像素²'] = float(np.min(flat_px))
            stats['最大_像素²'] = float(np.max(flat_px))
            if len(flat_px) >= 2:
                # 样本标准差（ddof=1），科研报告惯例
                stats['标准差_像素²'] = float(np.std(flat_px, ddof=1))
        if self.ruler_calibrated and flat_unit:
            # 单位面积统计与像素统计对齐（此前缺最小/最大/中位数）
            stats[f'平均面积_{self.unit_name}²'] = float(np.mean(flat_unit))
            stats[f'中位数_{self.unit_name}²'] = float(np.median(flat_unit))
            stats[f'最小_{self.unit_name}²'] = float(np.min(flat_unit))
            stats[f'最大_{self.unit_name}²'] = float(np.max(flat_unit))
            if len(flat_unit) >= 2:
                stats[f'标准差_{self.unit_name}²'] = float(np.std(flat_unit, ddof=1))
        rows.append(stats)
        return rows

    @staticmethod
    def _force_text_column(worksheet, dataframe, column_name, escape):
        """把以 =/+/-/@ 开头的单元格强制为文本，防止 Excel 当公式执行。

        openpyxl 默认把 '=' 开头的字符串按公式类型写入。强制 data_type='s'
        后 Excel 显示原始文本；API 异常时回退为撇号转义（可见但安全）。
        """
        try:
            col_idx = list(dataframe.columns).index(column_name) + 1
        except ValueError:
            return
        for row in range(2, len(dataframe) + 2):
            try:
                cell = worksheet.cell(row=row, column=col_idx)
                if isinstance(cell.value, str) and cell.value[:1] in ('=', '+', '-', '@'):
                    cell.data_type = 's'
            except Exception:
                logger.warning("xlsx 名称列文本化失败，改用撇号转义", exc_info=True)
                cell = worksheet.cell(row=row, column=col_idx)
                if isinstance(cell.value, str):
                    cell.value = escape(cell.value)
                return

    def export_data(self):
        if not any(self.measurements.values()):
            messagebox.showwarning("警告", "没有测量数据可导出")
            return

        file_path = filedialog.asksaveasfilename(
            title="导出数据",
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("Excel文件", "*.xlsx"), ("所有文件", "*.*")]
        )
        if not file_path:
            return

        try:
            source = str(self.file_path) if self.file_path else ""
            export_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            data = []
            for frame_idx in sorted(self.measurements.keys()):
                for poly_id in sorted(self.measurements[frame_idx].keys()):
                    measurement = self.measurements[frame_idx][poly_id]
                    pixel_area = measurement.get('pixel_area', 0)
                    unit_area = measurement.get('unit_area', 0)

                    points = self.polygons.get(frame_idx, {}).get(poly_id, [])
                    name = self.polygon_names.get(poly_id, f"多边形{poly_id}")
                    perimeter_px = polygon_perimeter(points)

                    row = {
                        '来源文件': source,
                        '导出时间': export_time,
                        '帧号': frame_idx + 1,
                        '多边形ID': poly_id,
                        # 名称保持原值；写 CSV 时再转义，写 xlsx 时强制文本
                        '名称': name,
                        '面积_像素²': pixel_area,
                        '周长_像素': perimeter_px,
                        '顶点数': len(points),
                    }
                    if self.ruler_calibrated:
                        row[f'面积_{self.unit_name}²'] = unit_area
                        row[f'周长_{self.unit_name}'] = (
                            perimeter_px / self.pixel_to_unit_ratio)
                        row['标尺比例'] = f"1px = {1 / self.pixel_to_unit_ratio:.4f}{self.unit_name}"
                    data.append(row)

            df = pd.DataFrame(data)
            df_summary = pd.DataFrame(self._export_summary())
            base = Path(file_path)
            is_xlsx = base.suffix.lower() == '.xlsx'
            written = [str(base)]
            if is_xlsx:
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='测量数据')
                    df_summary.to_excel(writer, index=False, sheet_name='汇总')
                    self._force_text_column(writer.sheets['测量数据'], df,
                                            '名称', self._escape_csv_formula)
            else:
                # CSV：名称列做公式注入转义（' 前置）
                df_csv = df.copy()
                df_csv['名称'] = df_csv['名称'].map(self._escape_csv_formula)
                df_csv.to_csv(file_path, index=False, encoding='utf-8-sig')
                summary_file = base.with_name(base.stem + '_summary.csv')
                df_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
                written.append(str(summary_file))

            # 导出坐标（Y 为左上原点行号，与 ImageJ 一致）
            coord_file = base.with_name(base.stem + '_coordinates.csv')
            coord_data = []
            for frame_idx in sorted(self.polygons.keys()):
                for poly_id in sorted(self.polygons[frame_idx].keys()):
                    name = self.polygon_names.get(poly_id, f"多边形{poly_id}")
                    for i, (x, y) in enumerate(self.polygons[frame_idx][poly_id]):
                        coord_data.append({
                            '帧号': frame_idx + 1,
                            '多边形ID': poly_id,
                            '名称': self._escape_csv_formula(name),
                            '顶点序号': i + 1,
                            'X坐标_像素': x,
                            'Y坐标_像素': y
                        })
            if coord_data:
                pd.DataFrame(coord_data).to_csv(coord_file, index=False,
                                                encoding='utf-8-sig')
                written.append(str(coord_file))

            # 导出标尺信息
            if self.ruler_calibrated:
                ruler_file = base.with_name(base.stem + '_ruler.csv')
                ruler_data = [{
                    '标尺长度_像素': self.ruler_length_pixels,
                    '标尺长度_实际': self.ruler_length_units,
                    '实际单位': self.unit_name,
                    '像素每单位': self.pixel_to_unit_ratio,
                    '单位每像素': 1 / self.pixel_to_unit_ratio,
                }]
                pd.DataFrame(ruler_data).to_csv(ruler_file, index=False,
                                                encoding='utf-8-sig')
                written.append(str(ruler_file))

            self.status_var.set(f"数据已导出: {base.name}")
            logger.info("已导出测量数据: %s", ", ".join(written))
            messagebox.showinfo(
                "成功", "数据已成功导出到:\n" + "\n".join(written))

        except Exception as e:
            logger.exception("导出数据失败")
            messagebox.showerror("错误", f"导出数据失败: {str(e)}")

    def export_chart(self):
        if not self.measurements:
            messagebox.showwarning("警告", "没有测量数据可导出图表")
            return

        file_path = filedialog.asksaveasfilename(
            title="导出图表",
            defaultextension=".png",
            filetypes=[("PNG图像", "*.png"), ("PDF文件", "*.pdf"), ("所有文件", "*.*")]
        )
        if not file_path:
            return

        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            use_unit = self.ruler_calibrated
            area_label = f"面积 ({self.unit_name}²)" if use_unit else "面积 (像素²)"

            # 1. 每个追踪组（复制链共享 root）的面积变化
            ax1 = axes[0, 0]
            series_by_root = group_series_by_root(self.measurements, self.polygon_roots)
            for root_id, series_points in sorted(series_by_root.items()):
                frames = [f + 1 for f, _a, _pid in series_points]
                areas = [convert_area_to_units(a, self.pixel_to_unit_ratio, use_unit)
                         for _f, a, _pid in series_points]
                # 标签/颜色取该组最新一帧副本的名称（复制时名称被继承）
                last_pid = series_points[-1][2]
                name = (self.polygon_names.get(last_pid)
                        or self.polygon_names.get(root_id)
                        or f'#{root_id}')
                color = (self.polygon_colors.get(last_pid)
                         or self.polygon_colors.get(root_id, 'blue'))
                ax1.plot(frames, areas, 'o-', label=name, color=color)
            ax1.set_xlabel('帧号')
            ax1.set_ylabel(area_label)
            ax1.set_title('多边形面积变化')
            handles, labels = ax1.get_legend_handles_labels()
            if labels:
                dedup = dict(zip(labels, handles))
                ax1.legend(dedup.values(), dedup.keys(), loc='best', fontsize=8)
            ax1.grid(True, alpha=0.3)

            # 2. 每帧总面积
            ax2 = axes[0, 1]
            frame_numbers = []
            total_areas = []
            for frame_idx in sorted(self.measurements.keys()):
                total = sum(
                    m['unit_area'] if use_unit else m['pixel_area']
                    for m in self.measurements[frame_idx].values())
                frame_numbers.append(frame_idx + 1)
                total_areas.append(total)

            ax2.bar(frame_numbers, total_areas, color='skyblue', alpha=0.7)
            ax2.set_xlabel('帧号')
            ax2.set_ylabel(f"总{area_label}")
            ax2.set_title('每帧总面积')
            ax2.grid(True, alpha=0.3, axis='y')

            # 3. 每帧多边形数量
            ax3 = axes[1, 0]
            nonempty_frames = [fi for fi in sorted(self.polygons.keys())
                               if self.polygons[fi]]
            fnums = [fi + 1 for fi in nonempty_frames]
            pcounts = [len(self.polygons[fi]) for fi in nonempty_frames]
            ax3.plot(fnums, pcounts, 's-', color='green', linewidth=2, markersize=6)
            ax3.set_xlabel('帧号')
            ax3.set_ylabel('多边形数量')
            ax3.set_title('每帧多边形数量')
            ax3.grid(True, alpha=0.3)

            # 4. 面积分布直方图
            ax4 = axes[1, 1]
            all_areas = [
                m['unit_area'] if use_unit else m['pixel_area']
                for fd in self.measurements.values() for m in fd.values()]

            if all_areas:
                ax4.hist(all_areas, bins=min(20, len(all_areas)), color='orange', alpha=0.7, edgecolor='black')
            ax4.set_xlabel(area_label)
            ax4.set_ylabel('频率')
            ax4.set_title('面积分布直方图')
            ax4.grid(True, alpha=0.3, axis='y')

            # 标题
            if use_unit:
                fig.suptitle(f'测量结果 - 标尺: {self.ruler_length_pixels:.1f}px = {self.ruler_length_units:.1f}{self.unit_name}',
                             fontsize=12, y=0.98)
            else:
                fig.suptitle('测量结果 - 未校准标尺', fontsize=12, y=0.98)

            try:
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
            finally:
                plt.close(fig)  # 保存失败也要关闭图窗，避免 pyplot 泄漏

            self.status_var.set(f"图表已导出: {Path(file_path).name}")
            messagebox.showinfo("成功", f"图表已导出到:\n{file_path}")

        except Exception as e:
            logger.exception("导出图表失败")
            messagebox.showerror("错误", f"导出图表失败: {str(e)}")

    # ==================== 项目保存/加载 ====================

    def save_project(self):
        if self.tiff_stack is None:
            messagebox.showwarning("警告", "没有加载图像，无法保存项目")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存项目",
            defaultextension=".json",
            filetypes=[("JSON项目文件", "*.json"), ("所有文件", "*.*")]
        )
        if not file_path:
            return

        try:
            project_data = {
                # v3: 坐标为左上原点（y 向下 = 行号）；新增 image_height/width。
                # v2 及更早为底部原点存储，加载时自动迁移（见 _apply_project_data）。
                'version': 3,
                'file_path': str(self.file_path),
                'image_height': int(self.tiff_stack.shape[1]),
                'image_width': int(self.tiff_stack.shape[2]),
                'total_frames': self.total_frames,
                'current_frame': self.current_frame,
                'next_polygon_id': self.next_polygon_id,
                'polygons': {},
                'polygon_colors': {str(k): v for k, v in self.polygon_colors.items()},
                'polygon_names': {str(k): v for k, v in self.polygon_names.items()},
                'polygon_roots': {str(k): v for k, v in self.polygon_roots.items()},
                'measurements': {},
                'ruler_points': self.ruler_points,
                'pixel_to_unit_ratio': self.pixel_to_unit_ratio,
                'ruler_calibrated': self.ruler_calibrated,
                'unit_name': self.unit_name,
                'ruler_length_pixels': self.ruler_length_pixels,
                'ruler_length_units': self.ruler_length_units
            }

            for frame_idx, polygons in self.polygons.items():
                project_data['polygons'][str(frame_idx)] = {}
                for poly_id, points in polygons.items():
                    project_data['polygons'][str(frame_idx)][str(poly_id)] = points

            # measurements 仅供旧版本工具读取；本工具加载时一律从顶点重建
            # （顶点是唯一数据源），此字段不参与恢复。
            for frame_idx, measurements in self.measurements.items():
                project_data['measurements'][str(frame_idx)] = {}
                for poly_id, m in measurements.items():
                    project_data['measurements'][str(frame_idx)][str(poly_id)] = m

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(project_data, f, indent=2, ensure_ascii=False)

            self.status_var.set(f"项目已保存: {Path(file_path).name}")
            self._dirty = False  # 已持久化，脏标记复位
            logger.info("项目已保存: %s（%d 帧多边形）",
                        file_path, len(self.polygons))
            messagebox.showinfo("成功", f"项目已保存到:\n{file_path}")

        except Exception as e:
            logger.exception("保存项目失败")
            messagebox.showerror("错误", f"保存项目失败: {str(e)}")

    def load_project(self):
        """★ 修复：正确处理JSON键类型转换。
        图像加载为异步：项目数据先挂起，待 _poll_load_queue 提交图像
        （含 _reset_state）后再恢复——否则恢复的多边形会被异步重置
        冲掉，且 total_frames 尚为旧值导致帧号钳制错误。
        用户取消选择替代图像时立即清空挂起数据，避免残留的
        _pending_project 在之后任意一次加载中被静默套用。"""
        file_path = filedialog.askopenfilename(
            title="加载项目",
            filetypes=[("JSON项目文件", "*.json"), ("所有文件", "*.*")]
        )
        if not file_path:
            return
        if not self._confirm_discard_changes():
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                project_data = json.load(f)
        except Exception as e:
            logger.exception("加载项目失败")
            messagebox.showerror("错误", f"加载项目失败: {str(e)}")
            return

        tiff_path = project_data.get('file_path', '')
        if isinstance(tiff_path, str) and tiff_path.startswith(('\\\\', '//')):
            # UNC 网络路径：连接共享会向对方发起 NTLM 认证握手，先确认
            if not messagebox.askyesno(
                    "网络路径",
                    f"项目指向网络位置:\n{tiff_path}\n\n"
                    "连接网络共享会向对方发送本机认证信息。\n仍要加载吗？"):
                self.status_var.set("已取消加载项目")
                return
        if not (tiff_path and Path(tiff_path).exists()):
            messagebox.showwarning("文件不存在",
                                   f"原始TIFF文件不存在:\n{tiff_path}\n\n请选择新的TIFF文件。")
            replacement = filedialog.askopenfilename(
                title="选择替代的TIFF文件",
                filetypes=[("TIFF文件", "*.tif *.tiff"), ("所有文件", "*.*")]
            )
            if not replacement:
                self._pending_project = None
                self.status_var.set("已取消加载项目")
                return
            tiff_path = replacement

        self._pending_project = project_data
        self.status_var.set("正在加载项目图像，完成后恢复项目数据…")
        self.load_tiff_file(tiff_path)

    def _maybe_apply_pending_project(self):
        """异步图像加载成功提交后，恢复挂起的项目数据。"""
        if self._pending_project is None:
            return
        project_data = self._pending_project
        self._pending_project = None
        try:
            self._apply_project_data(project_data)
        except Exception as e:
            logger.exception("恢复项目数据失败")
            messagebox.showerror("错误", f"恢复项目数据失败: {str(e)}")

    @staticmethod
    def _coerce_int(value, default, lo=None, hi=None):
        """项目文件整数字段安全转换；非法/越界时钳制或取默认值。"""
        try:
            number = int(value)
        except (TypeError, ValueError):
            return default
        if lo is not None and number < lo:
            return lo
        if hi is not None and number > hi:
            return hi
        return number

    def _apply_project_data(self, project_data):
        """在图像已加载、状态已重置后恢复项目的全部数据与 UI。

        所有标量字段（帧号/比例/单位/标尺/颜色等）都经过校验：
        手工编辑或损坏的 JSON 只会降级为默认值并记入告警，
        不会带着非法值进入后续的换算与渲染路径。
        """
        warnings = []

        # 帧号：钳制到 [0, total-1]（修复负值经 numpy 负索引回绕到帧尾）
        self.current_frame = self._coerce_int(
            project_data.get('current_frame', 0), 0,
            lo=0, hi=max(0, self.total_frames - 1))

        # 坐标迁移：v2 及更早以图像底部为 y=0；v3 起为左上原点（行号）。
        try:
            version = int(project_data.get('version', 2))
        except (TypeError, ValueError):
            version = 2
        need_flip = version < 3 and self.tiff_stack is not None
        image_height = self.tiff_stack.shape[1] if need_flip else 0

        # v3 起保存了原始图像尺寸：与当前加载图像比对，不一致则告警
        # （多边形坐标可能整体错位；v2 迁移也只能按当前高度翻转）
        if self.tiff_stack is not None:
            cur_h, cur_w = self.tiff_stack.shape[1], self.tiff_stack.shape[2]
        else:
            cur_h, cur_w = 0, 0
        saved_w = self._coerce_int(project_data.get('image_width'), None)
        saved_h = self._coerce_int(project_data.get('image_height'), None)
        if saved_w and saved_h and (saved_w, saved_h) != (cur_w, cur_h):
            warnings.append(
                f"项目图像尺寸 {saved_w}×{saved_h} 与当前图像 {cur_w}×{cur_h} 不一致，"
                f"多边形坐标可能错位，请核对")
        elif need_flip:
            warnings.append(
                "旧版项目未记录图像尺寸，坐标迁移按当前图像高度翻转，"
                "若原图尺寸不同可能错位")

        # 颜色：键转 int + 值须能被 matplotlib 解析（防 Invalid RGBA 渲染报错）
        self.polygon_colors = {}
        invalid_colors = 0
        for k, v in (project_data.get('polygon_colors') or {}).items():
            try:
                pid = int(k)
            except (TypeError, ValueError):
                continue
            try:
                to_rgba(v)
                self.polygon_colors[pid] = v
            except Exception:
                invalid_colors += 1
        if invalid_colors:
            warnings.append(f"{invalid_colors} 个多边形颜色非法，已改用默认色")

        # 名称：键转 int，值转字符串
        self.polygon_names = {}
        for k, v in (project_data.get('polygon_names') or {}).items():
            try:
                self.polygon_names[int(k)] = str(v)
            except (TypeError, ValueError):
                continue

        # 恢复 lineage（root）；旧版项目无此字段，缺失 id 在多边形恢复后补齐
        self.polygon_roots = {}
        for k, v in (project_data.get('polygon_roots') or {}).items():
            try:
                self.polygon_roots[int(k)] = int(v)
            except (TypeError, ValueError):
                continue

        # 恢复多边形（含校验：越界帧/畸形点对直接丢弃并计数）
        self.polygons, dropped = sanitize_project_polygons(
            project_data.get('polygons', {}), self.total_frames)
        if need_flip:
            for frame_idx, frame_polys in self.polygons.items():
                for poly_id, pts in frame_polys.items():
                    frame_polys[poly_id] = flip_polygon_y(pts, image_height)

        for frame_polys in self.polygons.values():
            for poly_id in frame_polys:
                self.polygon_roots.setdefault(poly_id, poly_id)

        # next_polygon_id 校验：保证新多边形 id 不会与现有 id 冲突（防手改 JSON 覆盖）
        max_existing = max(
            (pid for frame in self.polygons.values() for pid in frame),
            default=0)
        saved_next = self._coerce_int(project_data.get('next_polygon_id'), 1)
        self.next_polygon_id = max(saved_next, max_existing + 1, 1)

        # 标尺两点：畸形数据整体忽略（不会带着垃圾坐标进入渲染）
        raw_ruler_points = project_data.get('ruler_points')
        self.ruler_points = sanitize_ruler_points(
            raw_ruler_points, cur_w, cur_h) if self.tiff_stack is not None else []
        if self.ruler_points and need_flip:
            self.ruler_points = flip_polygon_y(self.ruler_points, image_height)
        if raw_ruler_points and not self.ruler_points:
            warnings.append("标尺两点数据非法，已忽略")

        # 比例：必须是正有限数；已校准但比例非法 → 按未校准处理并告警
        # （字符串/NaN/0 都会在后续 1/ratio 或换算中出问题）
        ratio = coerce_positive_float(project_data.get('pixel_to_unit_ratio'))
        raw_cal = project_data.get('ruler_calibrated')
        if isinstance(raw_cal, bool):
            calibrated = raw_cal
        else:
            calibrated = project_data.get('pixel_to_unit_ratio', 1.0) != 1.0  # 旧版推断
        if calibrated and ratio is None:
            warnings.append("标尺比例字段非法（须为正数），已按未校准处理")
            calibrated = False
        self.ruler_calibrated = calibrated
        self.pixel_to_unit_ratio = ratio if calibrated else 1.0

        # 单位：白名单校验（会拼进导出表头，防注入；白名单外回退 nm）
        unit = project_data.get('unit_name', 'nm')
        if unit not in UNIT_WHITELIST:
            warnings.append(f"单位 {unit!r} 不受支持，已回退为 nm")
            unit = 'nm'
        self.unit_name = unit

        self.ruler_length_pixels = coerce_positive_float(
            project_data.get('ruler_length_pixels')) or 0.0
        self.ruler_length_units = coerce_positive_float(
            project_data.get('ruler_length_units')) or 0.0

        # 测量值从顶点重建（顶点是唯一数据源，避免旧缓存/手改 JSON 造成失真）
        self.measurements = rebuild_measurements(
            self.polygons, self.pixel_to_unit_ratio, self.ruler_calibrated)

        # 顶点越界提示（数据保留，仅提示与当前图像可能不匹配）
        if self.tiff_stack is not None and self.polygons:
            oob = count_out_of_bounds_points(self.polygons, cur_w, cur_h)
            if oob:
                warnings.append(f"{oob} 个顶点落在当前图像范围外（数据已保留，请核对）")

        # 更新UI
        self.frame_var.set(str(self.current_frame + 1))
        self.update_ruler_info()
        self.display_current_frame()

        # 帧数/数据完整性提示
        saved_frames = project_data.get('total_frames')
        if isinstance(saved_frames, int) and saved_frames != self.total_frames:
            warnings.append(
                f"项目帧数({saved_frames})与当前图像帧数({self.total_frames})不一致，"
                f"请确认标尺校准是否仍适用")
        if dropped:
            warnings.append(f"{dropped} 个多边形因帧号越界或数据损坏被忽略")

        if warnings:
            logger.warning("项目加载告警: %s", "; ".join(warnings))

        self.status_var.set(f"项目已加载: {Path(self.file_path).name}")
        summary = f"恢复了 {len(self.polygons)} 帧的多边形数据。"
        if warnings:
            messagebox.showwarning("项目加载",
                                   "项目已加载，部分内容被忽略或需要核对：\n- "
                                   + "\n- ".join(warnings) + f"\n\n{summary}")
        else:
            messagebox.showinfo("成功", f"项目已加载！\n{summary}")


    def run(self):
        self.root.mainloop()


def _setup_logging():
    """日志输出到控制台 + 程序同目录文件（打包 exe 后 print 不可见，靠它留痕）。

    文件句柄带轮转（单文件 2MB × 3 个备份），长期使用不会无限增长。
    """
    try:
        if getattr(sys, 'frozen', False):
            base_dir = Path(sys.executable).parent
        else:
            base_dir = Path(__file__).parent
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(name)s: %(message)s',
            handlers=[
                logging.StreamHandler(),
                RotatingFileHandler(base_dir / '测量工具.log', encoding='utf-8',
                                    maxBytes=LOG_MAX_BYTES,
                                    backupCount=LOG_BACKUP_COUNT),
            ])
    except Exception:
        logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    # 检查依赖
    missing = []
    for mod in ['tifffile', 'pandas', 'matplotlib', 'numpy', 'PIL', 'openpyxl']:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)

    if missing:
        msg = (f"缺少依赖: {', '.join(missing)}\n"
               "请运行: pip install tifffile pandas matplotlib pillow numpy openpyxl")
        # 打包为 console=False 的窗口化 exe 后没有 stdin，
        # input() 会直接抛异常——改用对话框提示，失败再退回 print
        try:
            import tkinter as _tk
            from tkinter import messagebox as _mb
            _root = _tk.Tk()
            _root.withdraw()
            _mb.showerror("缺少依赖", msg)
            _root.destroy()
        except Exception:
            print(msg)
        sys.exit(1)

    _setup_logging()
    app = TiffStackViewer()
    app.run()
