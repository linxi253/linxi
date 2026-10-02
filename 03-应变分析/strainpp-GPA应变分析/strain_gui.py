# -*- coding: utf-8 -*-
"""
Strain++ GPA - 图形化界面

基于 Tkinter + Matplotlib 的应变分析图形界面
使用方法：直接运行 run.py 或双击此文件
"""

import os
import json
import platform
import queue
import sys
import time
import threading
import datetime as _datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# 确保可以导入同目录模块
_curdir = os.path.dirname(os.path.abspath(__file__))
if _curdir not in sys.path:
    sys.path.insert(0, _curdir)

# ttkbootstrap 现代主题
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *

# Matplotlib 嵌入 tkinter
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt

from strainpp_gpa.gpa import GPA
from strainpp_gpa.phase import ComputationCancelled
from strainpp_gpa.dm_reader import read_dm_file, read_dm_file_simple, is_dm_file, read_tiff
from strainpp_gpa.utils import detect_bragg_peaks
from strainpp_gpa.batch import BatchConfig, run_batch
from strainpp_gpa.metadata import (
    dependency_versions as _dependency_versions,
    json_pixel_size as _json_pixel_size,
    project_version as _project_version,
)


# ============================================================
# 全局中文字体配置（含 Linux 回退字体）
# ============================================================
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei',
    'Arial',
]
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Matplotlib 简洁白色配色
# ============================================================
FIG_BG = '#ffffff'         # figure 背景（白色）
AXES_BG = '#f8f9fa'        # axes 背景（极浅灰）
TEXT_COLOR = '#333333'      # 文字颜色
SPINE_COLOR = '#cccccc'    # 边框颜色


def style_axes(ax):
    """简洁白色主题 axes 样式"""
    ax.set_facecolor(AXES_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(0.5)


# 结果标签页的显示顺序与单位（原始图像/功率谱除外，掩膜标签恒为灰度）。
RESULT_FIELDS = [
    'phase1', 'phase2',
    'eps_xx', 'eps_yy', 'eps_xy',
    'e_xx', 'e_yy', 'e_xy', 'e_yx',
    'omega_xy', 'dilatation',
    'u_x', 'u_y',
    'quality_mask',
]

FIELD_UNITS = {
    'phase1': 'rad', 'phase2': 'rad', 'omega_xy': 'rad',
    'u_x': 'nm', 'u_y': 'nm',
}

# 批量处理可选的导出场 (名称 → 显示标签)
from strainpp_gpa.batch import BATCH_FIELDS as _BATCH_FIELDS
from strainpp_gpa.batch import DEFAULT_FIELDS as _DEFAULT_FIELDS


# ============================================================
# 自定义工具栏（带中文提示）
# ============================================================
class ChineseToolbar(NavigationToolbar2Tk):
    """中文版 Matplotlib 工具栏"""
    tooltips = {
        'Home': '重置视图',
        'Back': '上一步视图',
        'Forward': '下一步视图',
        'Pan': '拖动平移',
        'Zoom': '框选放大',
        'Subplots': '子图设置',
        'Save': '保存图片',
    }

    def set_message(self, s):
        pass  # 隐藏底部坐标信息，简化界面


# ============================================================
# 主界面类
# ============================================================
class StrainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Strain++ GPA — 几何相位分析应变测量')
        self.root.geometry('1280x800')
        self.root.minsize(1024, 600)

        # --- 注册 Strain++ 原始配色 (polar & blue-orange) ---
        self._register_strainpp_cmaps()

        # --- 数据状态 ---
        self.image = None          # 原始图像
        self.gpa = GPA()           # GPA 引擎
        self.result = None         # 计算结果
        self.pixel_size = 1.0
        self.image_path = ''
        self._g1_point = None      # (x, y) in FFT coords
        self._g2_point = None
        self._ref_rect = None      # 参考区域 (x1,y1,x2,y2)
        self._reference_selector = None
        self._crop_selector = None
        self._analysis_crop = None  # crop in original-image coordinates
        self._source_image_shape = None
        self._worker_queue = queue.Queue()
        self._job_generation = 0
        # Batch runs get their own message token: they must keep reporting
        # even when the user starts/cancels other jobs (which invalidate the
        # job generation), and their completion must always be delivered so
        # the progress dialog can finish cleanly.
        self._batch_token = 0
        self._last_compute_settings = {}
        self._result_stale = False
        self._stack_source = None   # 连续图像模式：堆栈/序列路径
        self._stack_info = None
        self._batch_cancel = False
        self._batch_dialog = None
        self._field_display_state = {}  # field -> {cmap, vmin, vmax} 所见即所得快照

        # --- 配色方案 ---
        self.cmap_field = '_spp_turbo'  # 原版 Strain++ 默认色环
        self.cmap_power = 'hot'     # 功率谱

        # 配色方案库
        self._colormaps = {
            'Strain++ 原始配色 (原版默认)': [
                ('_spp_turbo',    'Turbo (原版默认)'),
                ('_spp_polar',    'Polar 蓝黑红 (原版)'),
                ('_spp_blor',     'BlOr 蓝橙 (原版)'),
                ('_spp_thermal',  'Thermal 热力 (原版)'),
                ('_spp_greyscale','Greyscale 灰度 (原版)'),
            ],
            '📊 对称场 (发散)': [
                ('RdBu_r',       '红蓝'),
                ('seismic',      '地震波'),
                ('coolwarm',     '冷暖'),
                ('bwr',          '蓝白红'),
                ('PiYG',         '粉绿'),
                ('PRGn',         '紫绿'),
                ('BrBG',         '棕青'),
                ('RdYlBu',       '红黄蓝'),
                ('Spectral',     '光谱'),
            ],
            '🔥 热力/强度': [
                ('hot',          '经典热力'),
                ('inferno',      '地狱火'),
                ('plasma',       '等离子'),
                ('magma',        '岩浆'),
                ('viridis',      '翠绿 (色盲友好)'),
                ('cividis',      '蓝黄 (色盲友好)'),
                ('turbo',        '彩虹增强'),
                ('jet',          '经典彩虹'),
                ('gnuplot',      'Gnuplot'),
            ],
            '🌊 海洋/地质': [
                ('ocean',        '海洋'),
                ('terrain',      '地形'),
                ('gist_earth',   '地球'),
                ('bone',         '骨骼'),
                ('pink',         '粉色'),
                ('copper',       '铜色'),
            ],
            '🎨 艺术/其他': [
                ('twilight',     '暮光'),
                ('twilight_shifted', '暮光偏移'),
                ('hsv',          'HSV 色环'),
                ('nipy_spectral','nipy 光谱'),
                ('gist_rainbow', 'gist 彩虹'),
                ('gist_ncar',    'NCAR'),
                ('gist_stern',   'Stern'),
            ],
        }

        # 当前配色名 (存储 cmap 字符串)
        self._cmap_power_var = tk.StringVar(value='hot')
        self._cmap_field_var = tk.StringVar(value='_spp_turbo')

        # 自定义调节状态
        self._custom_cmap = ''          # 非空时覆盖下拉框选择
        self._custom_ranges = {}        # {name: (vmin, vmax)} 覆盖对称自动范围
        self._custom_reverse = False    # 是否反转色图
        self._per_field_cmap = {}       # {field_name: cmap_string} 单张图独立配色

        # 显示全部像素（原版 Strain++ 风格）：默认勾选，所有像素按原始值
        # 上色，与原版显示一致；取消勾选才套用质量掩膜（灰色=不可靠）。
        # 只影响预览与 PNG 出图，数值导出恒带掩膜。
        self._var_show_all_pixels = tk.BooleanVar(value=True)

        # --- 构建界面 ---
        self._build_menu()
        self._build_toolbar()
        self._build_main_area()
        self._build_statusbar()

        # --- 初始化 Combobox 显示文字 ---
        self._cmap_power_var.set(self._find_cmap_display_name(self.cmap_power, 'power'))
        self._cmap_field_var.set(self._find_cmap_display_name(self.cmap_field, 'field'))

        # --- 初始状态 ---
        self._set_state('idle')
        self.root.after(50, self._poll_worker_queue)

    @staticmethod
    def _register_strainpp_cmaps():
        """Register the exact colormaps of the original Strain++ software.

        Stop tables live in :mod:`strainpp_gpa.cmaps` (shared with the batch
        engine). All maps place zero strain at the 0.5 stop, so background
        noise shows as a colour rather than white.
        """
        from strainpp_gpa.cmaps import register_all
        register_all()

    # ==============================================================
    # 菜单栏
    # ==============================================================
    def _build_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label='打开图像... (Ctrl+O)', command=self._open_image)
        file_menu.add_separator()
        file_menu.add_command(label='导出当前结果...', command=self._export_current)
        file_menu.add_separator()
        file_menu.add_command(label='退出', command=self.root.quit)
        menubar.add_cascade(label='文件', menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label='使用说明', command=self._show_help)
        help_menu.add_command(label='许可协议', command=self._show_license)
        help_menu.add_command(label='关于', command=self._show_about)
        menubar.add_cascade(label='帮助', menu=help_menu)

        self.root.bind('<Control-o>', lambda e: self._open_image())
        self.root.bind('<Control-O>', lambda e: self._open_image())

    # ==============================================================
    # 工具栏
    # ==============================================================
    def _build_toolbar(self):
        # 第一行：文件 | 参数 | 操作按钮
        row1 = ttk.Frame(self.root, padding=(5, 5, 5, 0))
        row1.pack(side=tk.TOP, fill=tk.X)

        self.btn_open = ttkb.Button(row1, text='📂 打开图像', command=self._open_image, bootstyle='primary')
        self.btn_open.pack(side=tk.LEFT, padx=3)

        self.btn_open_stack = ttkb.Button(
            row1, text='🎬 打开连续图像', command=self._open_stack_image,
            bootstyle='primary-outline',
        )
        self.btn_open_stack.pack(side=tk.LEFT, padx=3)

        ttk.Separator(row1, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=8, fill=tk.Y)

        ttk.Label(row1, text='像素(nm):').pack(side=tk.LEFT, padx=2)
        self.var_pixel_size = tk.StringVar(value='1.0')
        ttk.Entry(row1, textvariable=self.var_pixel_size, width=5).pack(side=tk.LEFT, padx=1)

        ttk.Label(row1, text='σ1:').pack(side=tk.LEFT, padx=2)
        self.var_sigma1 = tk.StringVar(value='5.0')
        ttk.Entry(row1, textvariable=self.var_sigma1, width=4).pack(side=tk.LEFT, padx=1)

        ttk.Label(row1, text='σ2:').pack(side=tk.LEFT, padx=2)
        self.var_sigma2 = tk.StringVar(value='5.0')
        ttk.Entry(row1, textvariable=self.var_sigma2, width=4).pack(side=tk.LEFT, padx=1)

        self.var_hann = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1, text='Hann', variable=self.var_hann).pack(side=tk.LEFT, padx=3)

        ttk.Label(row1, text='旋转(°):').pack(side=tk.LEFT, padx=2)
        self.var_rotation = tk.StringVar(value='0.0')
        ttk.Entry(row1, textvariable=self.var_rotation, width=4).pack(side=tk.LEFT, padx=1)

        ttk.Separator(row1, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=8, fill=tk.Y)

        self.btn_detect = ttkb.Button(row1, text='🔍 检测Bragg峰', command=self._auto_detect, bootstyle='info')
        self.btn_detect.pack(side=tk.LEFT, padx=3)

        self.btn_compute = ttkb.Button(row1, text='▶ 计算应变', command=self._compute, bootstyle='success')
        self.btn_compute.pack(side=tk.LEFT, padx=3)

        self.btn_export = ttkb.Button(row1, text='💾 全部导出', command=self._export_all, bootstyle='warning')
        self.btn_export.pack(side=tk.LEFT, padx=3)
        self.btn_cancel = ttkb.Button(
            row1,
            text='■ 取消',
            command=self._cancel_job,
            bootstyle='danger',
        )
        self.btn_batch = ttkb.Button(
            row1,
            text='⚙ 批量处理全部帧',
            command=self._open_batch_dialog,
            bootstyle='warning-outline',
        )

        # 第二行：配色区
        row2 = ttk.Frame(self.root, padding=(5, 0, 5, 5))
        row2.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(row2, text='🎨 场图配色:', font=('', 9)).pack(side=tk.LEFT)
        self._cmb_field = ttk.Combobox(row2, textvariable=self._cmap_field_var,
                                  values=self._get_cmap_list('field'), width=14, state='readonly')
        self._cmb_field.pack(side=tk.LEFT, padx=2)
        self._cmb_field.bind('<<ComboboxSelected>>', self._on_field_cmap_change)

        ttk.Label(row2, text='功率谱配色:', font=('', 9)).pack(side=tk.LEFT, padx=(10, 2))
        self._cmb_power = ttk.Combobox(row2, textvariable=self._cmap_power_var,
                                  values=self._get_cmap_list('power'), width=14, state='readonly')
        self._cmb_power.pack(side=tk.LEFT, padx=2)
        self._cmb_power.bind('<<ComboboxSelected>>', self._on_power_cmap_change)

        self._btn_manual = ttk.Button(row2, text='⚙ 手动调节', command=self._open_cmap_dialog)
        self._btn_manual.pack(side=tk.LEFT, padx=6)

        ttk.Button(row2, text='↺ 重置', command=self._reset_cmap).pack(side=tk.LEFT)

        self._chk_show_all = ttk.Checkbutton(
            row2,
            text='显示全部像素',
            variable=self._var_show_all_pixels,
            command=self._on_show_all_pixels_toggle,
        )
        self._chk_show_all.pack(side=tk.LEFT, padx=(12, 0))

    # ==============================================================
    # 主区域（左侧图像，右侧面板）
    # ==============================================================
    def _build_main_area(self):
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ---- 左侧：图像显示（Notebook 多标签） ----
        left_frame = ttk.Frame(main)
        main.add(left_frame, weight=3)

        self.notebook = ttk.Notebook(left_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # ---- 标签页定义 ----
        # 组织：分析辅助 | 相位 | 应变(对称) | 畸变(非对称) | 旋转/膨胀 | 位移
        self._tab_defs = [
            # (内部名, 标签文字, 类别)
            ('image',       '原始图像',         '📊 分析'),
            ('power',       'FFT 功率谱',     '📊 分析'),
            ('phase1',      '相位 P_g1',      '📡 相位'),
            ('phase2',      '相位 P_g2',      '📡 相位'),
            ('eps_xx',      '应变 ε_xx',      '📐 应变'),
            ('eps_yy',      '应变 ε_yy',      '📐 应变'),
            ('eps_xy',      '应变 ε_xy',      '📐 应变'),
            ('e_xx',        '畸变 e_xx',      '📏 畸变'),
            ('e_yy',        '畸变 e_yy',      '📏 畸变'),
            ('e_xy',        '畸变 e_xy',      '📏 畸变'),
            ('e_yx',        '畸变 e_yx',      '📏 畸变'),
            ('omega_xy',    '旋转 ω_xy',      '🔄 旋转/膨胀'),
            ('dilatation',  '膨胀率 Δ',       '🔄 旋转/膨胀'),
            ('u_x',         '位移 u_x',       '📍 位移'),
            ('u_y',         '位移 u_y',       '📍 位移'),
            ('quality_mask','质量掩膜',        '✅ 质量'),
        ]

        self._tabs = {}
        for name, label, cat in self._tab_defs:
            tab = ttk.Frame(self.notebook)
            self.notebook.add(tab, text=label)
            self._tabs[name] = tab

        # 为每个标签创建画布
        self._canvases = {}
        self._figures = {}
        self._axes = {}
        self._toolbars = {}

        for name, tab_label, _ in self._tab_defs:
            tab_widget = self._tabs[name]  # 真正的 tk Frame

            fig = Figure(figsize=(6, 5), dpi=100, facecolor=FIG_BG)
            ax = fig.add_subplot(111)
            style_axes(ax)
            self._figures[name] = fig
            self._axes[name] = ax

            canvas = FigureCanvasTkAgg(fig, master=tab_widget)
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            canvas.draw()
            self._canvases[name] = canvas

            toolbar = ChineseToolbar(canvas, tab_widget)
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            self._toolbars[name] = toolbar

            # 初始显示提示文字
            ax.text(0.5, 0.5, '请先打开一张图像\n\n菜单栏: 文件 → 打开图像',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=13, color='#aaaaaa')
            ax.set_xticks([])
            ax.set_yticks([])
            canvas.draw()

        # 绑定右键菜单到标签页（单独设置每张图的配色）
        self.notebook.bind('<Button-3>', self._on_tab_right_click)

        # 绑定标签页切换 — 同步范围输入框
        self.notebook.bind('<<NotebookTabChanged>>', lambda e: self._sync_limits_ui())

        # ---- 右侧：信息面板 ----
        right_frame = ttk.Frame(main)
        main.add(right_frame, weight=1)

        # 图像信息
        info_group = ttk.LabelFrame(right_frame, text='图像信息', padding=8)
        info_group.pack(fill=tk.X, padx=3, pady=3)
        self.lbl_info = ttk.Label(info_group, text='未加载图像', justify=tk.LEFT)
        self.lbl_info.pack(fill=tk.X)

        # G 矢量选择
        g_group = ttk.LabelFrame(right_frame, text='G 矢量设置（点击功率谱选点）', padding=8)
        g_group.pack(fill=tk.X, padx=3, pady=3)

        ttk.Label(g_group, text='G1 (x, y):').pack(anchor=tk.W)
        g1_frame = ttk.Frame(g_group)
        g1_frame.pack(fill=tk.X, pady=2)
        self.var_g1x = tk.StringVar(value='')
        self.var_g1y = tk.StringVar(value='')
        self.entry_g1x = ttk.Entry(g1_frame, textvariable=self.var_g1x, width=10)
        self.entry_g1x.pack(side=tk.LEFT, padx=2)
        self.entry_g1y = ttk.Entry(g1_frame, textvariable=self.var_g1y, width=10)
        self.entry_g1y.pack(side=tk.LEFT, padx=2)
        ttk.Button(g1_frame, text='清除', command=lambda: self._clear_g(1)).pack(side=tk.LEFT, padx=3)

        ttk.Label(g_group, text='G2 (x, y):').pack(anchor=tk.W, pady=(8, 0))
        g2_frame = ttk.Frame(g_group)
        g2_frame.pack(fill=tk.X, pady=2)
        self.var_g2x = tk.StringVar(value='')
        self.var_g2y = tk.StringVar(value='')
        self.entry_g2x = ttk.Entry(g2_frame, textvariable=self.var_g2x, width=10)
        self.entry_g2x.pack(side=tk.LEFT, padx=2)
        self.entry_g2y = ttk.Entry(g2_frame, textvariable=self.var_g2y, width=10)
        self.entry_g2y.pack(side=tk.LEFT, padx=2)
        ttk.Button(g2_frame, text='清除', command=lambda: self._clear_g(2)).pack(side=tk.LEFT, padx=3)

        ttk.Label(g_group, text='左键→G1  右键→G2  滚轮缩放',
                  foreground='gray', font=('', 8)).pack(anchor=tk.W, pady=(8, 0))
        for entry in (self.entry_g1x, self.entry_g1y, self.entry_g2x, self.entry_g2y):
            entry.bind('<KeyRelease>', lambda _event: self._on_g_entry_change())
            entry.bind('<FocusOut>', lambda _event: self._on_g_entry_change())

        ref_row = ttk.Frame(g_group)
        ref_row.pack(fill=tk.X, pady=(8, 0))
        self.btn_reference = ttk.Button(
            ref_row,
            text='框选参考区',
            command=self._start_reference_selection,
        )
        self.btn_reference.pack(side=tk.LEFT, padx=2)
        self.btn_crop = ttk.Button(
            ref_row,
            text='裁剪分析区',
            command=self._start_crop_selection,
        )
        self.btn_crop.pack(side=tk.LEFT, padx=2)
        ttk.Button(
            ref_row, text='清除', command=self._clear_reference
        ).pack(side=tk.LEFT, padx=2)
        self.var_reference = tk.StringVar(value='未设置（可选）')
        ttk.Label(
            ref_row,
            textvariable=self.var_reference,
            foreground='gray',
            font=('', 8),
        ).pack(side=tk.LEFT, padx=4)

        # 色值范围调节面板
        limits_group = ttk.LabelFrame(right_frame, text='色值范围调节', padding=8)
        limits_group.pack(fill=tk.X, padx=3, pady=3)

        limits_row1 = ttk.Frame(limits_group)
        limits_row1.pack(fill=tk.X)
        ttk.Label(limits_row1, text='最小值:').pack(side=tk.LEFT, padx=2)
        self._var_limit_min = tk.StringVar(value='')
        self._spin_min = ttk.Spinbox(limits_row1, from_=-1000, to=1000, increment=0.0005,
                                     textvariable=self._var_limit_min, width=9)
        self._spin_min.pack(side=tk.LEFT, padx=2)
        self._var_limit_auto = tk.BooleanVar(value=True)

        limits_row2 = ttk.Frame(limits_group)
        limits_row2.pack(fill=tk.X, pady=2)
        ttk.Label(limits_row2, text='最大值:').pack(side=tk.LEFT, padx=2)
        self._var_limit_max = tk.StringVar(value='')
        self._spin_max = ttk.Spinbox(limits_row2, from_=-1000, to=1000, increment=0.0005,
                                     textvariable=self._var_limit_max, width=9)
        self._spin_max.pack(side=tk.LEFT, padx=2)

        # 绑定事件（只绑定按回车，避免 <FocusOut> 在标签页切换时乱读旧索引）
        for spin in [self._spin_min, self._spin_max]:
            spin.bind('<Return>', lambda e: self._on_limit_change())
            spin.bind('<<Increment>>', lambda e: self._on_limit_change())
            spin.bind('<<Decrement>>', lambda e: self._on_limit_change())

        limits_btn_row = ttk.Frame(limits_group)
        limits_btn_row.pack(fill=tk.X, pady=(4, 0))
        self._btn_limits_auto = ttk.Button(limits_btn_row, text='↺ 自动',
                                           command=self._auto_limits)
        self._btn_limits_auto.pack(side=tk.LEFT, padx=2)

        # 同步标签页切换时更新限制值
        self._notebook_change_bind = None

        # 旋转角快速重算
        rotate_group = ttk.LabelFrame(right_frame, text='旋转角编辑后按回车', padding=8)
        rotate_group.pack(fill=tk.X, padx=3, pady=3)
        rot_row = ttk.Frame(rotate_group)
        rot_row.pack(fill=tk.X)
        ttk.Label(rot_row, text='角度(°):').pack(side=tk.LEFT, padx=2)
        self._var_rotation_recalc = tk.StringVar(value='0.0')
        entry_rot = ttk.Entry(rot_row, textvariable=self._var_rotation_recalc, width=8)
        entry_rot.pack(side=tk.LEFT, padx=2)
        entry_rot.bind('<Return>', lambda e: self._recompute_with_rotation())
        self.btn_rotate_recalc = ttk.Button(
            rot_row, text='↻ 重算', command=self._recompute_with_rotation
        )
        self.btn_rotate_recalc.pack(side=tk.LEFT, padx=3)

        # 结果摘要
        result_group = ttk.LabelFrame(right_frame, text='计算结果', padding=8)
        result_group.pack(fill=tk.X, padx=3, pady=3)
        self.lbl_result = ttk.Label(result_group, text='等待计算...', justify=tk.LEFT,
                                    font=('Consolas', 9))
        self.lbl_result.pack(fill=tk.X)

        # 进度条
        self.progress = ttkb.Progressbar(right_frame, mode='indeterminate', bootstyle='striped-success')

    # ==============================================================
    # 状态栏
    # ==============================================================
    def _build_statusbar(self):
        self.statusbar = ttk.Label(self.root, text='就绪 — 请打开图像开始', relief=tk.SUNKEN,
                                   anchor=tk.W, padding=3)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

    # ==============================================================
    # 状态管理
    # ==============================================================
    def _set_state(self, state):
        """控制按钮的启用/禁用状态"""
        states = {
            'idle':       {'open': True,  'detect': False, 'compute': False, 'export': False},
            'loaded':     {'open': True,  'detect': True,  'compute': False, 'export': False},
            'ready':      {'open': True,  'detect': True,  'compute': True,  'export': False},
            'computed':   {'open': True,  'detect': True,  'compute': True,  'export': True},
            'busy':       {'open': False, 'detect': False, 'compute': False, 'export': False},
        }
        s = states.get(state, states['idle'])
        self.btn_open.config(state=tk.NORMAL if s['open'] else tk.DISABLED)
        self.btn_detect.config(state=tk.NORMAL if s['detect'] else tk.DISABLED)
        self.btn_compute.config(state=tk.NORMAL if s['compute'] else tk.DISABLED)
        self.btn_export.config(state=tk.NORMAL if s['export'] else tk.DISABLED)
        if hasattr(self, 'btn_reference'):
            self.btn_reference.config(
                state=tk.NORMAL if state in ('loaded', 'ready', 'computed') else tk.DISABLED
            )
        if hasattr(self, 'btn_crop'):
            self.btn_crop.config(
                state=tk.NORMAL
                if state in ('loaded', 'ready', 'computed') else tk.DISABLED
            )
        if hasattr(self, 'btn_rotate_recalc'):
            self.btn_rotate_recalc.config(
                state=tk.NORMAL if state == 'computed' else tk.DISABLED
            )

        if state == 'busy':
            self.progress.pack(fill=tk.X, padx=3, pady=2)
            self.progress.start(10)
            self.statusbar.config(text='正在计算中...')
            if not self.btn_cancel.winfo_manager():
                self.btn_cancel.pack(side=tk.LEFT, padx=3)
            # 批量运行时进度走对话框，这里只影响普通计算
        else:
            self.progress.stop()
            self.progress.pack_forget()
            if self.btn_cancel.winfo_manager():
                self.btn_cancel.pack_forget()

        # 批量入口：仅当处于堆栈模式且第一帧计算完成时可用
        if hasattr(self, 'btn_batch'):
            show_batch = (
                state == 'computed'
                and self._stack_source is not None
                and not self.btn_cancel.winfo_manager()
            )
            if show_batch and not self.btn_batch.winfo_manager():
                self.btn_batch.pack(side=tk.LEFT, padx=3)
            elif not show_batch and self.btn_batch.winfo_manager():
                self.btn_batch.pack_forget()

    # ==============================================================
    # 打开图像
    # ==============================================================
    def _cancel_job(self):
        """Invalidate the active worker result; NumPy work exits in the background."""
        self._job_generation += 1
        if getattr(self, '_batch_dialog', None) is not None:
            self._batch_cancel = True
        next_state = 'computed' if self.result is not None else (
            'ready' if self._g1_point and self._g2_point else
            'loaded' if self.image is not None else 'idle'
        )
        self._set_state(next_state)
        self.statusbar.config(text='已取消当前任务')

    def _poll_worker_queue(self):
        """Apply worker results exclusively on Tk's main thread.

        The ``after`` re-registration runs in ``finally`` so no message-
        handling error (e.g. a TclError from a widget that was just
        destroyed) can permanently kill this pump, and a failure inside one
        message only costs that message.
        """
        try:
            while True:
                try:
                    kind, generation, payload = self._worker_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._handle_worker_message(kind, generation, payload)
                except Exception as error:
                    # One broken message (e.g. a just-destroyed widget) must
                    # not take the pump down; the next tick keeps draining.
                    print(
                        f'worker message handling failed ({kind}): '
                        f'{error!r}',
                        file=sys.stderr,
                    )
        finally:
            try:
                self.root.after(50, self._poll_worker_queue)
            except tk.TclError:
                pass  # root already destroyed; the app is shutting down

    def _handle_worker_message(self, kind, generation, payload):
        """Route one worker message; batch kinds use the batch token."""
        if kind in ('batch_progress', 'batch_done', 'batch_error'):
            if generation != self._batch_token:
                return  # stale message from an already-finished batch run
            if kind == 'batch_progress':
                done, total, message = payload
                if getattr(self, '_batch_dialog', None) is not None:
                    self._batch_dialog.on_progress(done, total, message)
            elif kind == 'batch_done':
                records, cancelled, outdir = payload
                dialog = getattr(self, '_batch_dialog', None)
                self._batch_dialog = None
                self._restore_state()
                if dialog is not None:
                    dialog.on_done(records, cancelled, outdir)
            else:
                dialog = getattr(self, '_batch_dialog', None)
                self._batch_dialog = None
                self._restore_state()
                if dialog is not None:
                    dialog.on_error(payload)
                else:
                    messagebox.showerror('批量处理失败', payload)
            return

        if generation != self._job_generation:
            return
        if kind == 'load_done':
            self._finish_image_load(*payload)
        elif kind == 'load_error':
            self._load_error(payload)
        elif kind == 'compute_done':
            self._on_compute_done(*payload)
        elif kind == 'compute_error':
            self._compute_error(payload)
        elif kind == 'detect_done':
            self._finish_auto_detect(*payload)
        elif kind == 'detect_error':
            messagebox.showerror('检测失败', payload)
            self._set_state(
                'ready' if self._g1_point and self._g2_point else 'loaded'
            )
        elif kind == 'export_done':
            saved, outdir = payload
            self._restore_state()
            messagebox.showinfo(
                '导出完成',
                f'已导出 {saved} 个图像/数值文件及 2 个元数据文件到:\n{outdir}',
            )
            self.statusbar.config(
                text=f'导出完成 — {saved + 2} 个文件 → {outdir}'
            )
        elif kind == 'export_error':
            message, traceback_text = payload
            messagebox.showerror('导出失败', message)
            self._restore_state()
            if traceback_text:
                print(traceback_text, file=sys.stderr)

    def _restore_state(self):
        """Return the UI to its normal (non-busy) state after a job ends."""
        self._set_state(
            'computed' if self.result is not None else
            'ready' if self._g1_point and self._g2_point else
            'loaded' if self.image is not None else 'idle'
        )

    def _parse_pixel_size(self):
        """Parse one nm/pixel value or an anisotropic 'y,x' pair."""
        text = self.var_pixel_size.get().strip().replace('，', ',')
        parts = [part.strip() for part in text.split(',') if part.strip()]
        if len(parts) == 1:
            value = float(parts[0])
        elif len(parts) == 2:
            value = (float(parts[0]), float(parts[1]))
        else:
            raise ValueError('像素尺寸应为单个数值，或按“Y,X”输入两个数值')
        values = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError('像素尺寸必须为有限正数')
        return value

    @staticmethod
    def _format_pixel_size(pixel_size):
        values = np.asarray(pixel_size, dtype=float)
        if values.ndim == 0:
            return f'{float(values):g}'
        return f'{values[0]:g},{values[1]:g}'

    def _parse_g_entries(self):
        values = []
        for var, label in (
            (self.var_g1x, 'G1 x'), (self.var_g1y, 'G1 y'),
            (self.var_g2x, 'G2 x'), (self.var_g2y, 'G2 y'),
        ):
            text = var.get().strip()
            if not text:
                raise ValueError('请同时设置 G1 和 G2 的 x、y 坐标（可在功率谱上选点）')
            try:
                values.append(float(text))
            except ValueError:
                raise ValueError(f'{label} 不是有效数值') from None
        if not np.all(np.isfinite(values)):
            raise ValueError('G 矢量必须为有限数值')
        return tuple(values)

    def _on_g_entry_change(self):
        if self.image is None:
            return
        try:
            g1x, g1y, g2x, g2y = self._parse_g_entries()
        except (ValueError, tk.TclError):
            self._set_state('loaded')
            return
        rows, cols = self.image.shape
        cx, cy = cols // 2, rows // 2
        self._g1_point = (g1x + cx, g1y + cy)
        self._g2_point = (g2x + cx, g2y + cy)
        if hasattr(self, '_power_marker_g1'):
            self._power_marker_g1.set_data(
                [self._g1_point[0]], [self._g1_point[1]]
            )
            self._power_marker_g2.set_data(
                [self._g2_point[0]], [self._g2_point[1]]
            )
            self._canvases['power'].draw_idle()
        # The displayed result (if any) corresponds to the previous G-vectors;
        # keep it visible for comparison but require a recompute before export.
        became_stale = self.result is not None and not self._result_stale
        self._result_stale = self.result is not None
        self._set_state('ready')
        if became_stale:
            self.statusbar.config(text='G 矢量已修改 — 重新计算后才能导出')

    def _start_reference_selection(self):
        if self.image is None:
            messagebox.showwarning('提示', '请先打开图像')
            return
        self.notebook.select(self._tabs['image'])
        if self._crop_selector is not None:
            self._crop_selector.set_active(False)
            self._crop_selector.disconnect_events()
            self._crop_selector = None
        if self._reference_selector is not None:
            self._reference_selector.set_active(False)
            self._reference_selector.disconnect_events()
        self._reference_selector = RectangleSelector(
            self._axes['image'],
            self._on_reference_selected,
            useblit=True,
            button=[1],
            minspanx=3,
            minspany=3,
            spancoords='data',
            interactive=True,
        )
        self.statusbar.config(text='请在原始图像上拖动鼠标框选均匀无应变参考区')

    def _start_crop_selection(self):
        """Select an in-memory analysis crop; the source file is untouched."""
        if self.image is None:
            messagebox.showwarning('提示', '请先打开图像')
            return
        self.notebook.select(self._tabs['image'])
        if self._reference_selector is not None:
            self._reference_selector.set_active(False)
            self._reference_selector.disconnect_events()
            self._reference_selector = None
        if self._crop_selector is not None:
            self._crop_selector.set_active(False)
            self._crop_selector.disconnect_events()
        self._crop_selector = RectangleSelector(
            self._axes['image'],
            self._on_crop_selected,
            useblit=True,
            button=[1],
            minspanx=16,
            minspany=16,
            spancoords='data',
            interactive=True,
        )
        self.statusbar.config(
            text='请在原始图像上框选要保留的分析区域（排除标尺、文字和边框）'
        )

    def _on_crop_selected(self, click, release):
        coordinates = (click.xdata, release.xdata, click.ydata, release.ydata)
        if any(value is None for value in coordinates):
            return
        rows, cols = self.image.shape
        x1, x2 = sorted((click.xdata, release.xdata))
        y1, y2 = sorted((click.ydata, release.ydata))
        left = max(0, min(cols - 1, int(np.floor(x1))))
        right = max(left + 1, min(cols, int(np.ceil(x2))))
        top = max(0, min(rows - 1, int(np.floor(y1))))
        bottom = max(top + 1, min(rows, int(np.ceil(y2))))
        if right - left < 16 or bottom - top < 16:
            messagebox.showwarning('裁剪区太小', '分析区至少需要 16×16 像素')
            return
        if not messagebox.askyesno(
            '确认裁剪',
            f'仅在内存中保留 X {left}:{right}, Y {top}:{bottom}？\n'
            '原文件不会修改；G 矢量和已有结果将清空。',
        ):
            return

        base_x = self._analysis_crop[0] if self._analysis_crop else 0
        base_y = self._analysis_crop[1] if self._analysis_crop else 0
        crop_record = (
            base_x + left,
            base_y + top,
            base_x + right,
            base_y + bottom,
        )
        cropped = np.ascontiguousarray(self.image[top:bottom, left:right])
        try:
            pixel_size = self._parse_pixel_size()
        except ValueError as error:
            messagebox.showerror('参数错误', str(error))
            return

        self._job_generation += 1
        generation = self._job_generation
        use_hann = bool(self.var_hann.get())
        filepath = self.image_path
        source_shape = self._source_image_shape
        self._set_state('busy')
        self.statusbar.config(text='正在重建裁剪区域的 FFT...')

        def _run():
            try:
                worker_gpa = GPA()
                worker_gpa.load_image(
                    cropped, pixel_size=pixel_size, use_hann=use_hann
                )
                metadata = {
                    'analysis_crop_xyxy': crop_record,
                    'source_image_shape_yx': source_shape,
                }
                self._worker_queue.put((
                    'load_done',
                    generation,
                    (
                        filepath, cropped, metadata, pixel_size,
                        worker_gpa, '',
                    ),
                ))
            except Exception as error:
                self._worker_queue.put(('load_error', generation, str(error)))

        threading.Thread(target=_run, daemon=True).start()

    def _on_reference_selected(self, click, release):
        rows, cols = self.image.shape
        coordinates = (click.xdata, release.xdata, click.ydata, release.ydata)
        if any(value is None for value in coordinates):
            return
        x1, x2 = sorted((click.xdata, release.xdata))
        y1, y2 = sorted((click.ydata, release.ydata))
        left = max(0, min(cols - 1, int(np.floor(x1))))
        right = max(left + 1, min(cols, int(np.ceil(x2))))
        top = max(0, min(rows - 1, int(np.floor(y1))))
        bottom = max(top + 1, min(rows, int(np.ceil(y2))))
        if right - left < 3 or bottom - top < 3:
            messagebox.showwarning('参考区太小', '参考区至少需要 3×3 像素')
            return
        self._ref_rect = (left, top, right, bottom)
        self.var_reference.set(f'X {left}:{right}, Y {top}:{bottom}')
        self.statusbar.config(text='参考区已设置；计算时将自动修正两个 G 矢量')
        if self._reference_selector is not None:
            self._reference_selector.set_active(False)

    def _clear_reference(self):
        """Remove the optional homogeneous reference rectangle."""
        self._ref_rect = None
        self.var_reference.set('未设置（可选）')
        if self._reference_selector is not None:
            self._reference_selector.set_active(False)
            self._reference_selector.disconnect_events()
            try:
                self._reference_selector.set_visible(False)
            except AttributeError:
                pass
            self._reference_selector = None
            self._canvases['image'].draw_idle()
        self.statusbar.config(text='参考区已清除')

    def _open_image(self):
        filepath = filedialog.askopenfilename(
            title='选择图像文件',
            filetypes=[
                ('所有支持格式', '*.tif *.tiff *.dm3 *.dm4'),
                ('TIFF 图像', '*.tif *.tiff'),
                ('DM3 文件', '*.dm3'),
                ('DM4 文件', '*.dm4'),
                ('所有文件', '*.*'),
            ]
        )
        if not filepath:
            return

        self._job_generation += 1
        generation = self._job_generation
        use_hann = bool(self.var_hann.get())
        self._set_state('busy')
        self.statusbar.config(text=f'正在加载: {os.path.basename(filepath)}...')

        def _run():
            try:
                ext = os.path.splitext(filepath)[1].lower()
                metadata = {}
                warning = ''
                if ext in ('.dm3', '.dm4') or is_dm_file(filepath):
                    # ncempy 是 DM 文件的首选解析器；缺依赖时只影响 DM 读取，
                    # 不应阻止纯 TIFF 用户启动。这里给出清晰提示并使用内置解析器。
                    try:
                        import ncempy.io.dm  # noqa: F401
                    except ImportError:
                        warning = (
                            '未安装 ncempy，已尝试使用内置 DM 解析器读取。\n'
                            '如需最佳 DM3/DM4 兼容性，请执行：\n'
                            'python -m pip install ncempy'
                        )
                    try:
                        image, metadata = read_dm_file(filepath)
                    except Exception as primary_error:
                        image, metadata = read_dm_file_simple(filepath)
                        warning = (
                            (warning + '\n\n') if warning else ''
                        ) + (
                            '标准 DM 解析失败，已使用启发式回退读取。\n'
                            f'原因: {primary_error}\n'
                            '请核对图像方向、尺寸和像素标定。'
                        )
                else:
                    image = read_tiff(filepath)

                image = np.asarray(image)
                if image.ndim != 2:
                    raise ValueError(
                        f'图像必须是2D灰度图，当前为{image.ndim}D数组'
                    )
                if image.shape[0] < 4 or image.shape[1] < 4:
                    raise ValueError(
                        f'图像太小: {image.shape[1]}×{image.shape[0]}，最小需要4×4'
                    )
                if not np.all(np.isfinite(image)):
                    raise ValueError('输入图像含 NaN 或无穷值，请先清理数据')

                pixel_size = metadata.get('pixel_size', 1.0)
                if (ext in ('.dm3', '.dm4') or is_dm_file(filepath)) \
                        and 'pixel_size' not in metadata:
                    # DM 标定不可验证时绝不能静默采用 1 nm/px：位移/长度单位
                    # 会整体错误，必须在继续之前明确告知用户。
                    if 'pixel_size_candidate' in metadata:
                        detail = (
                            '文件中的像素标定无法可靠确认，界面暂用 1 nm/px；'
                            '请根据显微镜元数据手动输入。'
                        )
                    else:
                        detail = (
                            'DM 文件中未找到可验证的像素标定，界面暂用 1 nm/px；'
                            '请根据显微镜采集记录手动输入像素尺寸，'
                            '否则位移和长度的绝对单位是错误的。'
                        )
                    warning = ((warning + '\n\n') if warning else '') + detail
                worker_gpa = GPA()
                import warnings as _warnings
                with _warnings.catch_warnings(record=True) as caught:
                    _warnings.simplefilter('always')
                    worker_gpa.load_image(
                        image, pixel_size=pixel_size, use_hann=use_hann
                    )
                for entry in caught:
                    if issubclass(entry.category, (RuntimeWarning, UserWarning)):
                        warning = (
                            (warning + '\n\n') if warning else ''
                        ) + str(entry.message)
                self._worker_queue.put((
                    'load_done',
                    generation,
                    (filepath, image, metadata, pixel_size, worker_gpa, warning),
                ))
            except Exception as error:
                self._worker_queue.put(('load_error', generation, str(error)))

        threading.Thread(target=_run, daemon=True).start()

    def _open_stack_image(self):
        """打开连续图像：多页堆栈 TIF 或编号帧序列文件夹（二选一）。"""
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label='🎬 多页堆栈 TIF 文件…',
                         command=self._open_stack_file)
        menu.add_command(label='📁 编号帧序列文件夹…',
                         command=self._open_stack_folder)
        try:
            menu.tk_popup(self.root.winfo_pointerx(),
                          self.root.winfo_pointery())
        finally:
            menu.grab_release()

    def _open_stack_file(self):
        filepath = filedialog.askopenfilename(
            title='选择连续图像堆栈',
            filetypes=[
                ('堆栈 TIFF', '*.tif *.tiff'),
                ('所有文件', '*.*'),
            ],
        )
        if filepath:
            self._load_stack_source(filepath)

    def _open_stack_folder(self):
        directory = filedialog.askdirectory(title='选择编号帧序列文件夹')
        if directory:
            self._load_stack_source(directory)

    def _load_stack_source(self, source):
        """后台读取堆栈/序列的第一帧并进入连续图像模式。"""
        self._job_generation += 1
        generation = self._job_generation
        use_hann = bool(self.var_hann.get())
        self._set_state('busy')
        kind = '帧序列文件夹' if os.path.isdir(source) else '堆栈'
        self.statusbar.config(
            text=f'正在读取{kind}: {os.path.basename(source)}...'
        )

        def _run():
            try:
                from strainpp_gpa.stack_reader import open_frame_source

                info, frames = open_frame_source(source)
                try:
                    _, first = next(frames)
                finally:
                    frames.close()  # release the TIFF handle promptly
                metadata = {
                    'stack_source': source,
                    'stack_info': info,
                }
                pixel_size = 1.0
                if info.pixel_size is not None:
                    pixel_size = info.pixel_size
                    metadata['pixel_size'] = pixel_size
                worker_gpa = GPA()
                import warnings as _warnings
                with _warnings.catch_warnings(record=True) as caught:
                    _warnings.simplefilter('always')
                    worker_gpa.load_image(
                        first, pixel_size=pixel_size, use_hann=use_hann
                    )
                warning = ''
                if info.pixel_size is not None:
                    warning = f'已读取 ImageJ 标定: {pixel_size:g} nm/像素'
                for entry in caught:
                    if issubclass(entry.category, (RuntimeWarning, UserWarning)):
                        warning = (
                            (warning + '\n\n') if warning else ''
                        ) + str(entry.message)
                self._worker_queue.put((
                    'load_done',
                    generation,
                    (source, first, metadata, pixel_size, worker_gpa, warning),
                ))
            except Exception as error:
                self._worker_queue.put(('load_error', generation, str(error)))

        threading.Thread(target=_run, daemon=True).start()

    def _finish_image_load(
        self, filepath, image, metadata, pixel_size, worker_gpa, warning
    ):
        """Commit a completed image load on Tk's main thread."""
        self.image = image
        self.image_path = filepath
        self.pixel_size = pixel_size
        self.gpa = worker_gpa
        self.result = None
        self._last_compute_settings = {}
        self._stack_source = metadata.get('stack_source')
        self._stack_info = metadata.get('stack_info')
        self._analysis_crop = metadata.get('analysis_crop_xyxy')
        if metadata.get('source_image_shape_yx') is not None:
            self._source_image_shape = tuple(metadata['source_image_shape_yx'])
        elif self._analysis_crop is None:
            self._source_image_shape = image.shape
        self._g1_point = None
        self._g2_point = None
        self._ref_rect = None
        self.var_reference.set('未设置（可选）')
        self.var_pixel_size.set(self._format_pixel_size(pixel_size))
        self.var_g1x.set('')
        self.var_g1y.set('')
        self.var_g2x.set('')
        self.var_g2y.set('')

        rows, cols = image.shape
        stack_line = ''
        if self._stack_info is not None:
            stack_line = (
                f'\n连续图像: {self._stack_info.n_frames} 帧'
                + (
                    f' @ {self._stack_info.fps:g} fps'
                    if self._stack_info.fps else ''
                )
            )
        self.lbl_info.config(
            text=f'文件: {os.path.basename(filepath)}\n'
                 f'尺寸: {cols} × {rows} 像素\n'
                 f'数值范围: [{np.nanmin(image):.5g}, {np.nanmax(image):.5g}]\n'
                 f'像素尺寸(Y,X): {self._format_pixel_size(pixel_size)} nm'
                 + stack_line
                 + (
                     f'\n原图裁剪: {tuple(self._analysis_crop)}'
                     if self._analysis_crop else ''
                 )
        )

        if self._reference_selector is not None:
            self._reference_selector.set_active(False)
            self._reference_selector.disconnect_events()
            self._reference_selector = None
        if self._crop_selector is not None:
            self._crop_selector.set_active(False)
            self._crop_selector.disconnect_events()
            self._crop_selector = None
        for attr in list(self.__dict__.keys()):
            if attr.startswith('_cbar_'):
                try:
                    getattr(self, attr).remove()
                except (AttributeError, KeyError, ValueError):
                    pass
        for attr in list(self.__dict__.keys()):
            if (
                attr.startswith('_im_')
                or attr.startswith('_cbar_')
                or attr.startswith('_power_')
            ):
                delattr(self, attr)

        self._reset_result_tabs()

        self._show_input_image()
        self._show_power_spectrum()
        self.notebook.select(self._tabs['power'])
        self._set_state('loaded')
        self.statusbar.config(
            text=f'已加载: {os.path.basename(filepath)} — '
                 '请点击功率谱或直接输入两个 G 矢量'
        )
        if warning:
            messagebox.showwarning('读取提示', warning)

    def _reset_result_tabs(self):
        """Reset every result tab to its empty placeholder state."""
        for name, ax in self._axes.items():
            if name in ('image', 'power'):
                continue
            ax.clear()
            ax.text(
                0.5, 0.5, '计算后将显示结果',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=11, color='#aaaaaa',
            )
            ax.set_xticks([])
            ax.set_yticks([])
            self._canvases[name].draw_idle()

    def _load_error(self, message):
        messagebox.showerror('加载失败', f'无法加载图像:\n{message}')
        self._set_state('computed' if self.result is not None else (
            'loaded' if self.image is not None else 'idle'
        ))
        self.statusbar.config(text='图像加载失败')

    def _show_input_image(self):
        """Display the raw input image used for the analysis."""
        ax = self._axes['image']
        ax.clear()
        style_axes(ax)
        image = self._display_data(self.image)
        self._im_image = ax.imshow(
            image, cmap='gray', aspect='equal', origin='upper',
            extent=(0, self.image.shape[1], self.image.shape[0], 0),
        )
        ax.set_title(
            '分析输入图像'
            + ('（已裁剪）' if self._analysis_crop else '')
        )
        ax.set_xlabel('X (像素)')
        ax.set_ylabel('Y (像素)')
        self._canvases['image'].draw_idle()

    # ==============================================================
    # 显示功率谱（支持缩放保持）
    # ==============================================================
    def _show_power_spectrum(self):
        ax = self._axes['power']
        fig = self._figures['power']

        # 确保点击事件只绑一次
        if not hasattr(self, '_power_click_bound'):
            fig.canvas.mpl_connect('button_press_event', self._on_power_click)
            self._power_click_bound = True

        # 保存当前缩放状态
        xlim, ylim = None, None
        old_im = getattr(self, '_im_power', None)
        if old_im is not None:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

        ps = self.gpa.get_power_spectrum()
        M, N = ps.shape
        display_ps = self._display_data(ps)
        finite_ps = ps[np.isfinite(ps)]
        power_vmin, power_vmax = (
            np.quantile(finite_ps, [0.05, 0.999])
            if finite_ps.size else (0.0, 1.0)
        )

        if old_im is None:
            # 首次绘制
            ax.clear()
            style_axes(ax)
            # extent 取 (-0.5, N-0.5, M-0.5, -0.5)：像素中心落在整数数据坐标，
            # event.xdata/ydata 直接等于功率谱数组下标 (col, row)，
            # 点击反算 gx = x - N//2 才与掩膜的峰位 (N//2+gx, M//2+gy) 严格一致
            # （strainpp_gpa/phase.py：forward_fft 用 fftshift，DC 在 (M//2, N//2)）。
            im = ax.imshow(
                display_ps, cmap=self.cmap_power, aspect='equal',
                origin='upper', extent=(-0.5, N - 0.5, M - 0.5, -0.5),
                vmin=power_vmin, vmax=power_vmax,
            )

            ax.set_title('FFT 功率谱 — 左键=G1(红), 右键=G2(绿), 滚轮缩放', fontsize=11)
            ax.set_xlabel('X (像素)')
            ax.set_ylabel('Y (像素)')

            # 标出DC点
            cy, cx = M // 2, N // 2
            ax.plot(cx, cy, 'b+', markersize=15, markeredgewidth=2, label='DC (零点)')

            # 标记 G1 / G2 点
            self._power_marker_g1 = ax.plot([], [], 'ro', markersize=10,
                                             markeredgewidth=2, label='G1')[0]
            self._power_marker_g2 = ax.plot([], [], 'go', markersize=10,
                                             markeredgewidth=2, label='G2')[0]
            self._power_legend = ax.legend(loc='upper right', fontsize=8)

            self._im_power = im
        else:
            # 更新已有图像（保持缩放和标记）
            old_im.set_data(display_ps)
            old_im.set_extent((-0.5, N - 0.5, M - 0.5, -0.5))
            old_im.set_cmap(self.cmap_power)
            old_im.set_clim(power_vmin, power_vmax)

            # 更新标记
            self._power_marker_g1.set_data([], [])
            self._power_marker_g2.set_data([], [])
            if self._g1_point:
                self._power_marker_g1.set_data([self._g1_point[0]], [self._g1_point[1]])
            if self._g2_point:
                self._power_marker_g2.set_data([self._g2_point[0]], [self._g2_point[1]])

        # 恢复缩放
        if xlim and ylim:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        self._canvases['power'].draw_idle()

    # ==============================================================
    # 功率谱点击事件 — 选择 G 矢量
    # ==============================================================
    def _on_power_click(self, event):
        """在功率谱上点击选择 g 矢量（保持缩放）"""
        if event.xdata is None or event.ydata is None:
            return
        if self.gpa.power_spectrum is None:
            return

        x, y = event.xdata, event.ydata
        if not np.isfinite(x) or not np.isfinite(y):
            return
        M, N = self.gpa.shape
        # 基础校验：extent=(-0.5, N-0.5, M-0.5, -0.5) 下像素中心恰在整数数据
        # 坐标 0..N-1 / 0..M-1，event.xdata/ydata 即功率谱数组下标 (col, row)；
        # gx = x - N//2 与掩膜峰位 (N//2+gx, M//2+gy) 严格一致（无半像素偏移）。
        # extent 边缘的半像素区（<0 或 >N-1）不属于任何像素中心，同样拒绝。
        if not (0 <= x <= N - 1 and 0 <= y <= M - 1):
            self.statusbar.config(text=f'选点 ({x:.1f}, {y:.1f}) 超出图像范围，请重新点击')
            return
        gx = x - N // 2
        gy = y - M // 2
        if np.hypot(gx, gy) < 2.0:
            self.statusbar.config(text='所选点距 DC 中心太近（<2 px），建议选择更远的 Bragg 峰')

        if event.button == 1:  # 左键 = G1
            self._g1_point = (x, y)
            self.var_g1x.set(f'{gx:.1f}')
            self.var_g1y.set(f'{gy:.1f}')
            self.statusbar.config(text=f'G1 已设置: ({gx:.1f}, {gy:.1f}) — 右键点击选择 G2')
        elif event.button == 3:  # 右键 = G2
            self._g2_point = (x, y)
            self.var_g2x.set(f'{gx:.1f}')
            self.var_g2y.set(f'{gy:.1f}')
            self.statusbar.config(text=f'G2 已设置: ({gx:.1f}, {gy:.1f}) — 可以点击"计算应变"了')

        # 只更新标记，不重绘整个功率谱（保持缩放）
        if hasattr(self, '_power_marker_g1') and hasattr(self, '_power_marker_g2'):
            self._power_marker_g1.set_data([], [])
            self._power_marker_g2.set_data([], [])
            if self._g1_point:
                self._power_marker_g1.set_data([self._g1_point[0]], [self._g1_point[1]])
            if self._g2_point:
                self._power_marker_g2.set_data([self._g2_point[0]], [self._g2_point[1]])
            self._canvases['power'].draw_idle()

        if self._g1_point and self._g2_point:
            self._result_stale = self.result is not None
            self._set_state('ready')

    def _clear_g(self, which):
        if which == 1:
            self._g1_point = None
            self.var_g1x.set('')
            self.var_g1y.set('')
        else:
            self._g2_point = None
            self.var_g2x.set('')
            self.var_g2y.set('')
        # 只更新标记，保持缩放
        if hasattr(self, '_power_marker_g1') and hasattr(self, '_power_marker_g2'):
            self._power_marker_g1.set_data([], [])
            self._power_marker_g2.set_data([], [])
            self._canvases['power'].draw_idle()
        else:
            self._show_power_spectrum()
        self._set_state('loaded' if self.gpa.power_spectrum is not None else 'idle')
        self.statusbar.config(text='请在功率谱上重新选择 G 矢量')

    # ==============================================================
    # 自动检测 Bragg 峰
    # ==============================================================
    def _auto_detect(self):
        if self.image is None:
            messagebox.showwarning('提示', '请先打开图像')
            return

        try:
            pixel_size = self._parse_pixel_size()
        except ValueError as error:
            messagebox.showerror('参数错误', str(error))
            return

        self._job_generation += 1
        generation = self._job_generation
        image = self.image
        use_hann = bool(self.var_hann.get())
        self._set_state('busy')
        self.statusbar.config(text='正在分析功率谱...')

        def _run():
            try:
                worker_gpa = GPA()
                worker_gpa.load_image(
                    image, pixel_size=pixel_size, use_hann=use_hann
                )
                radius = worker_gpa.estimate_mask_radius()
                peaks = detect_bragg_peaks(
                    worker_gpa.get_power_spectrum(), radius, n_peaks=6
                )
                self._worker_queue.put((
                    'detect_done',
                    generation,
                    (worker_gpa, radius, peaks),
                ))
            except Exception as error:
                self._worker_queue.put(('detect_error', generation, str(error)))

        threading.Thread(target=_run, daemon=True).start()

    def _finish_auto_detect(self, worker_gpa, radius, peaks):
        """Display Bragg candidates after a background detection."""
        self.gpa = worker_gpa
        self.pixel_size = worker_gpa.pixel_size
        # The detection may have used different pixel-size/Hann settings, so
        # any previous result no longer matches the displayed power spectrum.
        if self.result is not None:
            self.result = None
            self._last_compute_settings = {}
            self._result_stale = False
            self._reset_result_tabs()
        self.var_pixel_size.set(self._format_pixel_size(self.pixel_size))
        self._show_power_spectrum()
        if not peaks:
            messagebox.showerror('检测失败', '未找到可靠的 Bragg 峰候选')
            self._set_state(
                'computed' if self.result is not None else
                'ready' if self._g1_point and self._g2_point else 'loaded'
            )
            return
        representative_radius = float(
            np.median([peak[3] for peak in peaks[:6]])
        )
        # 原版 Strain++ 自动 sigma = |G| / 6（processBraggClick: r/(2*3)）。
        sigma_val = max(representative_radius / 6.0, 1.5)
        self.var_sigma1.set(f'{sigma_val:.2f}')
        self.var_sigma2.set(f'{sigma_val:.2f}')

        info = (
            'Bragg峰检测结果\n'
            '─────────────────\n'
            f'最短强峰半径: {radius:.1f} 像素\n'
            f'候选峰族中值半径: {representative_radius:.1f} 像素\n'
            f'保守起始 Gaussian σ: {sigma_val:.2f} 像素\n\n'
            '前6个候选峰:\n'
        )
        for index, (gx, gy, _value, peak_radius) in enumerate(peaks[:6]):
            info += (
                f'  {index + 1}. ({gx:6.1f}, {gy:6.1f}) '
                f'r={peak_radius:.1f}\n'
            )
        self.lbl_result.config(text=info)

        for artist in getattr(self, '_candidate_peak_artists', []):
            try:
                artist.remove()
            except ValueError:
                pass
        self._candidate_peak_artists = []
        rows, cols = worker_gpa.shape
        colors = ['cyan', 'yellow', 'magenta', 'orange', 'lime', 'pink']
        ax = self._axes['power']
        for index, (gx, gy, _value, _radius) in enumerate(peaks[:6]):
            artist = ax.plot(
                gx + cols // 2,
                gy + rows // 2,
                'o',
                color=colors[index],
                markersize=8,
                alpha=0.75,
            )[0]
            self._candidate_peak_artists.append(artist)
        self._canvases['power'].draw_idle()

        self._set_state(
            'computed' if self.result is not None else
            'ready' if self._g1_point and self._g2_point else 'loaded'
        )
        self.statusbar.config(
            text=f'检测完成: Bragg 峰距中心约 {radius:.1f} 像素；'
                 '请选择两个非共线峰'
        )

    # ==============================================================
    # 计算应变
    # ==============================================================
    def _compute(self):
        self._start_compute()

    def _start_compute(self, rotation_override=None):
        """Validate UI inputs and run a complete GPA calculation off-thread."""
        if self.image is None:
            messagebox.showwarning('提示', '请先打开图像')
            return
        try:
            g1x, g1y, g2x, g2y = self._parse_g_entries()
            sigma1 = float(self.var_sigma1.get())
            sigma2 = float(self.var_sigma2.get())
            rotation = (
                float(self.var_rotation.get())
                if rotation_override is None else float(rotation_override)
            )
            pixel_size = self._parse_pixel_size()
        except ValueError as error:
            messagebox.showerror('参数错误', str(error) or '请检查参数')
            return

        if not np.isfinite(sigma1) or sigma1 <= 0 \
                or not np.isfinite(sigma2) or sigma2 <= 0:
            messagebox.showerror('参数错误', '高斯 σ1/σ2 必须为有限正数')
            return
        if not np.isfinite(rotation):
            messagebox.showerror('参数错误', '旋转角必须为有限数值')
            return

        self._job_generation += 1
        generation = self._job_generation
        image = self.image
        use_hann = bool(self.var_hann.get())
        ref_rect = self._ref_rect
        self._set_state('busy')
        self.statusbar.config(text='正在计算应变...')

        def _cancelled():
            return generation != self._job_generation

        def _run():
            try:
                if _cancelled():
                    return
                worker_gpa = GPA()
                worker_gpa.load_image(image, pixel_size=pixel_size, use_hann=use_hann)
                if _cancelled():
                    return
                worker_gpa.set_g1(g1x, g1y, sigma=sigma1)
                if _cancelled():
                    return
                worker_gpa.set_g2(g2x, g2y, sigma=sigma2)
                if _cancelled():
                    return

                refinements = ([], [])
                if ref_rect is not None:
                    left, top, right, bottom = ref_rect
                    mask = np.zeros(image.shape, dtype=bool)
                    mask[top:bottom, left:right] = True
                    refinements = (
                        worker_gpa.phase1.refine_iterative(
                            mask, cancel_check=_cancelled
                        ),
                        worker_gpa.phase2.refine_iterative(
                            mask, cancel_check=_cancelled
                        ),
                    )
                    if not refinements[0] or not refinements[1]:
                        raise ValueError('参考区内可靠像素不足，无法修正 G 矢量')
                    if _cancelled():
                        return

                if rotation != 0.0:
                    worker_gpa.set_rotation(np.radians(rotation))

                if _cancelled():
                    return
                t0 = time.time()
                # mask_results=False：GUI 保留原始值，质量掩膜在显示层
                # 按需套用（显示全部像素开关），统计与数值导出另行掩膜。
                result = worker_gpa.compute(
                    cancel_check=_cancelled, mask_results=False
                )
                elapsed = time.time() - t0
                payload = (
                    worker_gpa,
                    result,
                    elapsed,
                    worker_gpa.phase1.gx,
                    worker_gpa.phase1.gy,
                    worker_gpa.phase2.gx,
                    worker_gpa.phase2.gy,
                    sigma1,
                    sigma2,
                    pixel_size,
                    rotation,
                    use_hann,
                    ref_rect,
                    refinements,
                )
                self._worker_queue.put(('compute_done', generation, payload))
            except ComputationCancelled:
                # Cooperative cancellation: the generation has already been
                # invalidated by _cancel_job, so nothing needs to be queued.
                return
            except Exception as error:
                self._worker_queue.put(('compute_error', generation, str(error)))

        threading.Thread(target=_run, daemon=True).start()

    def _on_compute_done(
        self, worker_gpa, result, elapsed, g1x, g1y, g2x, g2y, sigma1,
        sigma2, pixel_size, rotation, use_hann, ref_rect, refinements,
    ):
        """Called on main thread after computation completes."""
        self.gpa = worker_gpa
        self.result = result
        self.pixel_size = pixel_size
        self._result_stale = False
        self.var_pixel_size.set(self._format_pixel_size(pixel_size))
        self.var_rotation.set(f'{rotation:g}')
        rows, cols = self.image.shape
        self._g1_point = (g1x + cols // 2, g1y + rows // 2)
        self._g2_point = (g2x + cols // 2, g2y + rows // 2)
        self.var_g1x.set(f'{g1x:.6g}')
        self.var_g1y.set(f'{g1y:.6g}')
        self.var_g2x.set(f'{g2x:.6g}')
        self.var_g2y.set(f'{g2y:.6g}')
        self._last_compute_settings = {
            'g1': [g1x, g1y],
            'g2': [g2x, g2y],
            'sigma1_px': sigma1,
            'sigma2_px': sigma2,
            'pixel_size_nm_yx': np.asarray(pixel_size).tolist(),
            'rotation_degrees': rotation,
            'hann_window': use_hann,
            'reference_roi_xyxy': list(ref_rect) if ref_rect else None,
            'analysis_crop_xyxy': (
                list(self._analysis_crop) if self._analysis_crop else None
            ),
            'refinement_steps': [
                [list(step) for step in refinements[0]],
                [list(step) for step in refinements[1]],
            ],
        }
        self._show_power_spectrum()
        self._show_results(result, elapsed, g1x, g1y, g2x, g2y, sigma1, sigma2)

    def _display_field(self, name):
        """Field data for interactive display, honouring 显示全部像素.

        Default (checkbox off) applies the quality mask — NaN outside reliable
        pixels. With the checkbox on, raw values are returned for every pixel
        (original Strain++-style full-colour view). quality_mask itself is
        always returned as float.
        """
        r = self.result
        if r is None:
            return None
        if name == 'quality_mask':
            return r.quality_mask.astype(float)
        data = getattr(r, name, None)
        if data is None:
            return None
        if self._var_show_all_pixels.get():
            return data
        return np.where(r.quality_mask, data, np.nan)

    def _on_show_all_pixels_toggle(self):
        """Redraw every result tab in the selected pixel scope."""
        if self.result is not None:
            self._redraw_all_fields()
        if self._var_show_all_pixels.get():
            self.statusbar.config(
                text='显示全部像素（含不可靠区，原版 Strain++ 风格）— '
                     '数值 TIFF/NPY 导出仍套用质量掩膜'
            )
        else:
            self.statusbar.config(text='显示已套用质量掩膜（NaN = 不可靠像素）')

    def _show_results(self, result, elapsed, g1x, g1y, g2x, g2y, sigma1, sigma2):
        """显示计算结果（主线程）—— 填充所有标签页"""
        for name in RESULT_FIELDS:
            if name not in self._axes:
                continue
            data = self._display_field(name)
            if data is None:
                continue
            self._show_field(
                name, data, self._get_field_title(name),
                'gray' if name == 'quality_mask' else self.cmap_field,
                symmetric=self._field_is_symmetric(name),
                unit=FIELD_UNITS.get(name, ''),
            )

        # --- 文字摘要（统计始终只基于质量掩膜内的可靠像素） ---
        valid_fraction = float(np.mean(result.quality_mask))

        def _stats(name):
            data = np.where(result.quality_mask, getattr(result, name), np.nan)
            return (
                float(np.nanmean(data)),
                float(np.nanstd(data)),
                float(np.nanmin(data)),
            )

        strain_stack = np.stack([
            np.where(result.quality_mask, result.eps_xx, np.nan),
            np.where(result.quality_mask, result.eps_xy, np.nan),
            np.where(result.quality_mask, result.eps_yy, np.nan),
        ])
        finite_strain = strain_stack[np.isfinite(strain_stack)]
        extreme_fraction = (
            float(np.mean(np.abs(finite_strain) > 0.5))
            if finite_strain.size else 0.0
        )
        self._last_compute_settings[
            'extreme_strain_fraction_abs_gt_0_5'
        ] = extreme_fraction
        exx = _stats('eps_xx')
        eyy = _stats('eps_yy')
        exy = _stats('eps_xy')
        omega = _stats('omega_xy')
        dilatation = _stats('dilatation')
        info_text = (
            f'G1: ({g1x:.2f}, {g1y:.2f}) σ1={sigma1:.1f}  '
            f'G2: ({g2x:.2f}, {g2y:.2f}) σ2={sigma2:.1f}\n'
            f'计算耗时: {elapsed:.2f}s  有效像素: {valid_fraction:.1%}\n'
            f'G 矩阵条件数: {result.g_condition_number:.3g}\n'
            + (
                f'⚠ |应变|>50% 像素: {extreme_fraction:.3%}，请检查峰/掩膜/缺陷\n'
                if extreme_fraction > 0 else ''
            )
            + f'{"─"*45}\n'
            f'{"分量(掩膜内)":<12} {"均值":>10} {"标准差":>10} {"最小值":>10}\n'
            f'{"─"*45}\n'
            f'ε_xx:  {exx[0]:>10.5f} {exx[1]:>10.5f} {exx[2]:>10.5f}\n'
            f'ε_yy:  {eyy[0]:>10.5f} {eyy[1]:>10.5f} {eyy[2]:>10.5f}\n'
            f'ε_xy:  {exy[0]:>10.5f} {exy[1]:>10.5f} {exy[2]:>10.5f}\n'
            f'ω_xy:  {omega[0]:>10.5f} {omega[1]:>10.5f} {omega[2]:>10.5f}\n'
            f'Δ:     {dilatation[0]:>10.5f} {dilatation[1]:>10.5f} '
            f'{dilatation[2]:>10.5f}'
        )
        self.lbl_result.config(text=info_text, font=('Consolas', 9))

        self._set_state('computed')
        # 同步旋转角显示（不会触发 trace，因为已经删除了 trace_add）
        self._var_rotation_recalc.set(self.var_rotation.get())
        # 同步范围输入框
        self.root.after(50, self._sync_limits_ui)
        self.statusbar.config(
            text=f'计算完成! 耗时 {elapsed:.2f}s — 有效像素 {valid_fraction:.1%}'
        )

    def _compute_error(self, msg):
        messagebox.showerror('计算失败', msg)
        self._set_state('computed' if self.result is not None else (
            'ready' if self._g1_point and self._g2_point else 'loaded'
        ))
        self.statusbar.config(text='计算出错，请检查参数')

    def _open_batch_dialog(self):
        """以第一帧的计算参数打开批量处理设置对话框。"""
        if self._stack_source is None or self.result is None:
            messagebox.showwarning('提示', '请先在连续图像上完成第一帧计算')
            return
        if self._result_stale:
            messagebox.showwarning('结果已过期', '参数已修改，请重新计算第一帧')
            return
        self._batch_dialog = BatchSettingsDialog(self)

    # ==============================================================
    # 色值范围调节
    # ==============================================================
    def _on_limit_change(self):
        """用户调整范围滑块时更新当前标签页的显示（含 tab 保护）"""
        if self.result is None:
            return
        vmin_str = self._var_limit_min.get().strip()
        vmax_str = self._var_limit_max.get().strip()
        if not vmin_str or not vmax_str:
            self._var_limit_auto.set(True)
            return
        try:
            vmin, vmax = float(vmin_str), float(vmax_str)
            if vmin >= vmax:
                return
        except ValueError:
            return

        # 保存当前 tab 索引（防止任何重绘过程中的 tab 漂移）
        saved_idx = self.notebook.index(self.notebook.select())
        if saved_idx >= len(self._tab_defs):
            return
        name = self._tab_defs[saved_idx][0]
        if name in ('image', 'power'):
            return

        self._custom_ranges[name] = (vmin, vmax)
        self._var_limit_auto.set(False)

        # 只重绘当前这一个场量（不碰其他 canvas，避免 Windows tab 跳动）
        data = self._display_field(name)
        if data is not None:
            # 先用 draw_idle 更新图像
            self._show_field(name, data, self._get_field_title(name),
                             self.cmap_field, symmetric=False)
            # 强制恢复 tab（防御性，防止 Windows canvas draw 干扰 notebook）
            self.notebook.select(saved_idx)

    def _auto_limits(self):
        """恢复当前标签页为自动对称范围（含 tab 保护）"""
        saved_idx = self.notebook.index(self.notebook.select())
        if saved_idx >= len(self._tab_defs):
            return
        name = self._tab_defs[saved_idx][0]
        if name in ('image', 'power'):
            return
        self._custom_ranges.pop(name, None)
        self._var_limit_auto.set(True)

        # 重绘（只画当前这一个）
        data = self._display_field(name)
        if data is not None:
            self._show_field(name, data, self._get_field_title(name),
                             self.cmap_field,
                             symmetric=self._field_is_symmetric(name))
            self.notebook.select(saved_idx)

    def _sync_limits_ui(self):
        """同步范围输入框与当前标签页的取值"""
        idx = self.notebook.index(self.notebook.select())
        if idx >= len(self._tab_defs):
            return
        name = self._tab_defs[idx][0]
        if name in ('image', 'power') or self.result is None:
            self._var_limit_min.set('')
            self._var_limit_max.set('')
            return

        data = self._display_field(name)
        if data is None:
            return

        if name in self._custom_ranges:
            vmin, vmax = self._custom_ranges[name]
            self._var_limit_min.set(f'{vmin:.6f}')
            self._var_limit_max.set(f'{vmax:.6f}')
        else:
            finite = np.asarray(data)[np.isfinite(data)]
            if finite.size == 0:
                self._var_limit_min.set('')
                self._var_limit_max.set('')
                return
            vmin, vmax = self._automatic_limits(
                name, finite, self._field_is_symmetric(name)
            )
            self._var_limit_min.set(f'{vmin:.6f}')
            self._var_limit_max.set(f'{vmax:.6f}')

    # ==============================================================
    # 旋转角实时重算
    # ==============================================================
    def _recompute_with_rotation(self):
        """用新的旋转角重新计算（不改变 g 矢量）"""
        if self.result is None or not self._g1_point or not self._g2_point:
            return
        try:
            rotation = float(self._var_rotation_recalc.get())
            if not np.isfinite(rotation):
                raise ValueError
        except ValueError:
            messagebox.showerror('参数错误', '旋转角必须为有限数值')
            return
        self.var_rotation.set(f'{rotation:g}')
        self.statusbar.config(text=f'旋转角 {rotation}° — 重新计算中...')
        self._start_compute(rotation_override=rotation)

    def _show_field(self, name, data, title, cmap, symmetric=True, unit='', add_contour=False):
        """在指定标签页显示二维场"""
        ax = self._axes[name]
        fig = self._figures[name]
        im_attr = f'_im_{name}'
        cbar_attr = f'_cbar_{name}'

        if data is None:
            ax.clear()
            ax.text(0.5, 0.5, '无数据', transform=ax.transAxes, ha='center', va='center')
            self._canvases[name].draw()
            return

        data_array = np.asarray(data)
        finite = data_array[np.isfinite(data_array)]
        if finite.size == 0:
            ax.clear()
            ax.text(
                0.5, 0.5, '没有可靠的有限数据',
                transform=ax.transAxes, ha='center', va='center',
            )
            self._canvases[name].draw_idle()
            return
        display_data = self._display_data(data_array)
        data_clean = np.ma.masked_invalid(display_data)

        # 应用自定义 cmap（逐层降级：单图独立 > 全局自定义 > 下拉框默认）
        per_field = self._per_field_cmap.get(name, '')
        final_cmap = per_field or self._custom_cmap or cmap
        if per_field:
            try:
                import matplotlib as mpl
                if hasattr(mpl, 'colormaps') and hasattr(mpl.colormaps, 'get_cmap'):
                    mpl.colormaps.get_cmap(per_field)
                else:
                    mpl.cm.get_cmap(per_field)
                final_cmap = per_field
            except Exception:
                final_cmap = self._custom_cmap or cmap
        elif self._custom_cmap:
            try:
                import matplotlib as mpl
                if hasattr(mpl, 'colormaps') and hasattr(mpl.colormaps, 'get_cmap'):
                    mpl.colormaps.get_cmap(self._custom_cmap)
                else:
                    mpl.cm.get_cmap(self._custom_cmap)
                final_cmap = self._custom_cmap
            except Exception:
                pass

        # 反转
        if self._custom_reverse and not final_cmap.endswith('_r'):
            final_cmap = final_cmap + '_r'

        # 应用自定义范围（若有）
        if name in self._custom_ranges:
            vmin, vmax = self._custom_ranges[name]
        else:
            vmin, vmax = self._automatic_limits(name, finite, symmetric)
        if vmin == vmax:
            padding = max(abs(vmin) * 1e-6, 1e-12)
            vmin -= padding
            vmax += padding

        # 记录该场当前的实际显示状态（所见即所得：批量预览据此渲染）
        self._field_display_state[name] = {
            'cmap': final_cmap,
            'vmin': float(vmin),
            'vmax': float(vmax),
        }

        old_im = getattr(self, im_attr, None)

        if old_im is None:
            # 首次创建：ax 在完整空间，img + colorbar 各占一份
            ax.clear()
            style_axes(ax)
            im = ax.imshow(data_clean, cmap=self._masked_cmap(final_cmap),
                           aspect='equal', origin='upper',
                           vmin=vmin, vmax=vmax,
                           extent=(0, data_array.shape[1], data_array.shape[0], 0))
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            # colorbar 深色适配
            cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT_COLOR)
            setattr(self, im_attr, im)
            setattr(self, cbar_attr, cbar)
        else:
            # 后续更新：不重建 ax 和 colorbar，只改色图
            old_im.set_data(data_clean)
            old_im.set_extent((0, data_array.shape[1], data_array.shape[0], 0))
            old_im.set_cmap(self._masked_cmap(final_cmap))
            old_im.set_clim(vmin, vmax)
            old_cbar = getattr(self, cbar_attr, None)
            if old_cbar is not None:
                old_cbar.update_normal(old_im)
            self._canvases[name].draw_idle()
            return

        title_str = f'{title}'
        if unit:
            title_str += f' [{unit}]'
        ax.set_title(title_str, fontsize=11)
        ax.set_xlabel('X (像素)')
        ax.set_ylabel('Y (像素)')

        self._canvases[name].draw()

    @staticmethod
    def _masked_cmap(cmap_name):
        """Colormap copy with NaN mapped to light grey.

        Diverging maps render 0 as white, so NaN must be a distinct colour or
        "no data" and "zero strain" are visually identical.
        """
        try:
            cmap_object = matplotlib.colormaps[cmap_name].copy()
            cmap_object.set_bad('#d9d9d9')
            return cmap_object
        except (KeyError, ValueError, AttributeError, TypeError):
            return cmap_name

    @staticmethod
    def _display_data(data, max_side=1600):
        """Bound GUI preview memory with anti-aliased block means.

        Plain strided decimation aliases high-frequency lattice fringes into
        patterns that do not exist in the data, so each step x step block is
        reduced to the mean of its finite samples; all-NaN blocks stay NaN.
        Full-resolution arrays are returned untouched when no reduction is
        needed.
        """
        array = np.asarray(data)
        if array.ndim != 2:
            return array
        step = max(1, int(np.ceil(max(array.shape) / max_side)))
        if step == 1:
            return array
        rows, cols = array.shape
        rows2 = rows - rows % step
        cols2 = cols - cols % step
        if rows2 < step or cols2 < step:
            return array
        blocks = np.asarray(
            array[:rows2, :cols2], dtype=np.float64
        ).reshape(rows2 // step, step, cols2 // step, step)
        finite = np.isfinite(blocks)
        counts = finite.sum(axis=(1, 3))
        sums = np.where(finite, blocks, 0.0).sum(axis=(1, 3))
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    @staticmethod
    def _automatic_limits(name, finite, symmetric):
        """Robust preview limits; raw exports always retain all values.

        Symmetric fields use a median/MAD estimate (bound = 4 robust sigma)
        so a long tail of unreliable pixels cannot stretch the colour scale
        and wash out the background — matching the fixed-limit saturation
        display of the original Strain++ (default limits ±0.2).
        """
        values = np.asarray(finite, dtype=float)
        if values.size == 0:
            return 0.0, 1.0
        if name == 'quality_mask':
            return 0.0, 1.0
        if symmetric:
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            bound = 4.0 * 1.4826 * mad
            if not np.isfinite(bound) or bound <= 0:
                bound = float(np.max(np.abs(values)))
            return -bound, bound
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        spread = 4.0 * 1.4826 * mad
        if not np.isfinite(spread) or spread <= 0:
            return float(np.min(values)), float(np.max(values))
        return median - spread, median + spread

    # ==============================================================
    # 导出
    # ==============================================================
    def _export_current(self):
        """Export the active field as a preview or full-precision numeric data."""
        if self.result is None and self.gpa.power_spectrum is None:
            messagebox.showwarning('提示', '请先计算应变')
            return

        current_tab = self.notebook.index(self.notebook.select())
        if current_tab >= len(self._tab_defs):
            return

        name = self._tab_defs[current_tab][0]

        if name not in ('image', 'power'):
            # 结果场必须与当前参数一致：G 矢量修改后旧结果不允许导出，
            # 否则会以同名文件静默输出与界面参数不符的数据。
            if self.result is None:
                messagebox.showwarning('提示', '请先计算应变')
                return
            if self._result_stale:
                messagebox.showwarning(
                    '结果已过期',
                    'G 矢量等参数已修改，当前结果不再对应输入框中的设置；'
                    '请重新计算后再导出。',
                )
                return

        # 取数据：PNG 随"显示全部像素"开关；数值恒带质量掩膜（定量语义）。
        if name == 'image':
            data = self.image
        elif name == 'power':
            data = self.gpa.get_power_spectrum()
        else:
            data = self._result_field(name)

        if data is None:
            messagebox.showwarning('提示', '该数据不可用')
            return

        default_name = os.path.splitext(os.path.basename(self.image_path))[0] if self.image_path else 'output'

        filepath = filedialog.asksaveasfilename(
            title='导出当前场',
            initialfile=f'{default_name}_{name}.png',
            defaultextension='.png',
            filetypes=[
                ('PNG 渲染图', '*.png'),
                ('浮点 TIFF 数据', '*.tif'),
                ('NumPy 数据', '*.npy'),
            ],
        )
        if not filepath:
            return

        try:
            extension = os.path.splitext(filepath)[1].lower()
            if extension == '.png':
                png_data = (
                    data if name in ('image', 'power')
                    else self._display_field(name)
                )
                self._save_rendered_png(
                    name, png_data, filepath, self._style_snapshot()
                )
            elif extension in ('.tif', '.tiff', '.npy'):
                self._save_numeric_field(name, data, filepath)
            else:
                raise ValueError('请选择 PNG、TIFF 或 NPY 格式')
            self.statusbar.config(text=f'已保存: {os.path.basename(filepath)}')
        except Exception as e:
            messagebox.showerror('保存失败', str(e))

    @staticmethod
    def _field_data(result, name):
        """Full-precision numeric export data, always quality-masked.

        GUI results keep raw values internally (masking happens at the
        display layer), so numeric exports must apply the quality mask here:
        NaN outside reliable pixels, mask itself as float32.
        """
        if result is None:
            return None
        data = getattr(result, name, None)
        if data is None:
            return None
        if name == 'quality_mask':
            data = data.astype(np.float32)
        else:
            data = np.where(result.quality_mask, data, np.nan)
        return data

    def _result_field(self, name):
        """Current result field for export; phases are quality-masked."""
        return self._field_data(self.result, name)

    def _style_snapshot(self):
        """Snapshot of colormap settings for thread-safe PNG export."""
        return {
            'cmap_field': self.cmap_field,
            'cmap_power': self.cmap_power,
            'custom_cmap': self._custom_cmap,
            'custom_reverse': self._custom_reverse,
            'custom_ranges': dict(self._custom_ranges),
            'per_field_cmap': dict(self._per_field_cmap),
        }

    def _save_rendered_png(self, field_name, data, output_path, style):
        """将数据用当前配色渲染成PNG，与预览完全一致"""
        if data is None:
            raise ValueError(f'{field_name} 没有可导出的数据')
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        source = np.asarray(data)
        finite = source[np.isfinite(source)]
        if finite.size == 0:
            raise ValueError(f'{field_name} 没有有限数据')
        preview = self._display_data(source, max_side=4096)
        data_clean = np.ma.masked_invalid(preview)

        cmap_name = (
            style['cmap_power'] if field_name == 'power' else
            'gray' if field_name in ('image', 'quality_mask') else
            style['cmap_field']
        )
        per_field = style['per_field_cmap'].get(field_name, '')
        final_cmap = per_field or style['custom_cmap'] or cmap_name
        try:
            cmap_object = matplotlib.colormaps[final_cmap].copy()
        except KeyError as error:
            raise ValueError(f'未知 Matplotlib 色图: {final_cmap}') from error
        if style['custom_reverse']:
            cmap_object = cmap_object.reversed()
        # NaN = 浅灰，与发散色图的 0 值白色明确区分（与界面预览一致）。
        cmap_object.set_bad('#d9d9d9')

        if field_name in style['custom_ranges']:
            vmin, vmax = style['custom_ranges'][field_name]
        else:
            vmin, vmax = self._automatic_limits(
                field_name,
                finite,
                self._field_is_symmetric(field_name),
            )
        if vmin == vmax:
            padding = max(abs(vmin) * 1e-6, 1e-12)
            vmin -= padding
            vmax += padding

        height, width = preview.shape
        dpi = 100
        fig = Figure(
            figsize=(max(width / dpi, 2), max(height / dpi, 2)),
            dpi=dpi,
        )
        # Transparent background for PNG export. ``transparent=True`` is not
        # accepted by FigureCanvasAgg in recent matplotlib, so set the canvas
        # alpha directly instead.
        fig.patch.set_alpha(0.0)
        ax = fig.add_subplot(111)
        image_artist = ax.imshow(
            data_clean, cmap=cmap_object, aspect='equal',
            vmin=vmin, vmax=vmax,
        )
        fig.colorbar(image_artist, ax=ax, fraction=0.046, pad=0.04)
        ax.set_axis_off()
        fig.tight_layout(pad=0.1)
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(
            output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05,
        )
        fig.clear()

    @staticmethod
    def _save_numeric_field(field_name, data, output_path):
        """Save numeric arrays without replacing NaN quality-mask values."""
        array = np.asarray(data)
        extension = os.path.splitext(output_path)[1].lower()
        if extension == '.npy':
            np.save(output_path, array, allow_pickle=False)
            return
        if extension in ('.tif', '.tiff'):
            import tifffile
            tifffile.imwrite(
                output_path,
                array.astype(np.float32, copy=False),
                metadata={
                    'field': field_name,
                    'invalid_pixels': 'NaN',
                    'axes': 'YX',
                },
            )
            return
        raise ValueError(f'不支持的数值格式: {extension}')

    def _export_all(self):
        """Export previews, full-resolution arrays, and reproducibility metadata."""
        if self.result is None:
            messagebox.showwarning('提示', '请先计算应变')
            return

        outdir = filedialog.askdirectory(title='选择导出文件夹')
        if not outdir:
            return

        basename = os.path.splitext(os.path.basename(self.image_path))[0] or 'output'

        fields = [
            ('eps_xx',     '应变 ε_xx'),
            ('eps_yy',     '应变 ε_yy'),
            ('eps_xy',     '应变 ε_xy'),
            ('omega_xy',   '旋转 ω_xy'),
            ('dilatation', '膨胀率 Delta'),
            ('e_xx',       '畸变 e_xx'),
            ('e_yy',       '畸变 e_yy'),
            ('e_xy',       '畸变 e_xy'),
            ('e_yx',       '畸变 e_yx'),
            ('u_x',        '位移 u_x'),
            ('u_y',        '位移 u_y'),
            ('phase1',     '相位 P_g1'),
            ('phase2',     '相位 P_g2'),
            ('quality_mask', '可靠像素掩膜'),
        ]

        prospective = []
        for field_name, _label in fields:
            prospective.extend([
                os.path.join(outdir, f'{basename}_{field_name}.png'),
                os.path.join(outdir, f'{basename}_{field_name}.tif'),
            ])
        prospective.extend([
            os.path.join(outdir, f'{basename}_power_spectrum.png'),
            os.path.join(outdir, f'{basename}_power_spectrum.tif'),
            os.path.join(outdir, f'{basename}_info.txt'),
            os.path.join(outdir, f'{basename}_metadata.json'),
        ])
        existing = [path for path in prospective if os.path.exists(path)]
        if existing and not messagebox.askyesno(
            '确认覆盖',
            f'目标目录已有 {len(existing)} 个同名文件，是否全部覆盖？',
        ):
            return

        snapshot = {
            'outdir': outdir,
            'basename': basename,
            'fields': fields,
            'prospective': prospective,
            'result': self.result,
            'image_path': self.image_path,
            'image_shape_yx': self.image.shape,
            'source_image_shape_yx': (
                list(self._source_image_shape)
                if self._source_image_shape else None
            ),
            'analysis_crop_xyxy': (
                list(self._analysis_crop) if self._analysis_crop else None
            ),
            'pixel_size': self.pixel_size,
            'g1': (self.gpa.phase1.gx, self.gpa.phase1.gy),
            'g2': (self.gpa.phase2.gx, self.gpa.phase2.gy),
            'power_spectrum': self.gpa.get_power_spectrum(),
            'settings': dict(self._last_compute_settings),
            'style': self._style_snapshot(),
            'show_all_pixels': bool(self._var_show_all_pixels.get()),
        }

        self._job_generation += 1
        generation = self._job_generation
        self._set_state('busy')
        self.statusbar.config(text='正在导出...')

        def _run():
            try:
                saved = self._export_all_worker(snapshot)
                self._worker_queue.put((
                    'export_done', generation, (saved, outdir),
                ))
            except Exception as error:
                # Capture the traceback here, while the exception context
                # still exists; print_exc() on the Tk thread would report an
                # unrelated (or empty) context.
                import traceback
                self._worker_queue.put((
                    'export_error', generation,
                    (str(error), traceback.format_exc()),
                ))

        threading.Thread(target=_run, daemon=True).start()

    def _export_all_worker(self, snapshot):
        """Perform the heavy PNG/array/metadata export off the Tk main thread."""
        outdir = snapshot['outdir']
        basename = snapshot['basename']
        fields = snapshot['fields']
        prospective = snapshot['prospective']
        result = snapshot['result']
        style = snapshot['style']
        show_all = snapshot['show_all_pixels']
        saved = 0

        def _png_data(name):
            """PNG preview data follows the 显示全部像素 toggle; numeric
            exports always stay quality-masked (see _field_data)."""
            if show_all and name != 'quality_mask':
                return getattr(result, name, None)
            return self._field_data(result, name)

        for field_name, _label in fields:
            numeric = self._field_data(result, field_name)
            png = _png_data(field_name)
            if numeric is None or png is None:
                continue
            png_path = os.path.join(outdir, f'{basename}_{field_name}.png')
            tif_path = os.path.join(outdir, f'{basename}_{field_name}.tif')
            self._save_rendered_png(field_name, png, png_path, style)
            self._save_numeric_field(field_name, numeric, tif_path)
            saved += 2

        ps = snapshot['power_spectrum']
        if ps is not None:
            png_path = os.path.join(outdir, f'{basename}_power_spectrum.png')
            tif_path = os.path.join(outdir, f'{basename}_power_spectrum.tif')
            self._save_rendered_png('power', ps, png_path, style)
            self._save_numeric_field('power', ps, tif_path)
            saved += 2

        infopath = os.path.join(outdir, f'{basename}_info.txt')
        with open(infopath, 'w', encoding='utf-8', newline='\n') as f:
            f.write('Strain++ GPA 分析结果\n')
            f.write('====================\n')
            f.write(f'输入文件: {snapshot["image_path"]}\n')
            image_shape = snapshot['image_shape_yx']
            f.write(f'图像尺寸: {image_shape[1]} x {image_shape[0]}\n')
            f.write(
                '像素尺寸(Y,X): '
                f'{self._format_pixel_size(snapshot["pixel_size"])} nm\n'
            )
            f.write(f'G1: ({snapshot["g1"][0]:.4f}, {snapshot["g1"][1]:.4f})\n')
            f.write(f'G2: ({snapshot["g2"][0]:.4f}, {snapshot["g2"][1]:.4f})\n')
            f.write(f'有效像素比例: {np.mean(result.quality_mask):.6f}\n')
            f.write(f'G 矩阵条件数: {result.g_condition_number:.6g}\n')
            f.write(
                '坐标约定: 数组[y,x]；+x向右，+y向下；\n'
                '          旋转角为数学正方向（+x轴向+y轴）；屏幕上 +y 向下，\n'
                '          正角视觉呈现为顺时针\n'
            )
            f.write(f'配色(场图): {style["cmap_field"]}\n')
            f.write(
                'PNG 显示范围: '
                + ('全部像素（未掩膜）' if show_all else '质量掩膜内')
                + '\n'
            )
            if style['custom_cmap']:
                f.write(f'自定义色图: {style["custom_cmap"]}\n')
            if style['custom_reverse']:
                f.write('色图反转: 是\n')
            f.write('\n--- 统计 ---\n')
            f.write(
                f'{"分量":<12} {"均值":>12} {"标准差":>12} '
                f'{"最小值":>12} {"最大值":>12}\n'
            )
            for field_name, label in fields:
                data = self._field_data(result, field_name)
                if data is None:
                    continue
                d = np.asarray(data)
                f.write(
                    f'{label:<10} {np.nanmean(d):>12.6f} '
                    f'{np.nanstd(d):>12.6f} {np.nanmin(d):>12.6f} '
                    f'{np.nanmax(d):>12.6f}\n'
                )

        metadata_path = os.path.join(outdir, f'{basename}_metadata.json')
        # Schema mirrors strain_analysis.main() so CLI and GUI provenance
        # files can be parsed with the same tooling.
        settings = snapshot['settings']
        refinement_steps = settings.get('refinement_steps')
        if refinement_steps:
            g1_steps = refinement_steps[0] or None
            g2_steps = refinement_steps[1] or None
        else:
            g1_steps = g2_steps = None
        metadata = {
            'software': 'Strain++ GPA',
            'software_version': _project_version(),
            'created_utc': _datetime.datetime.now(
                _datetime.timezone.utc
            ).isoformat(),
            'input': os.path.abspath(snapshot['image_path']),
            'source_image_shape_yx': snapshot['source_image_shape_yx'],
            'image_shape_yx': list(snapshot['image_shape_yx']),
            'analysis_crop_xyxy': snapshot['analysis_crop_xyxy'],
            'pixel_size_nm_yx': _json_pixel_size(snapshot['pixel_size']),
            'g1_fft_pixels': [float(snapshot['g1'][0]), float(snapshot['g1'][1])],
            'g2_fft_pixels': [float(snapshot['g2'][0]), float(snapshot['g2'][1])],
            'sigma1_fft_pixels': settings.get('sigma1_px'),
            'sigma2_fft_pixels': settings.get('sigma2_px'),
            'rotation_degrees': settings.get('rotation_degrees'),
            'hann_window': settings.get('hann_window'),
            'reference_roi_xyxy': settings.get('reference_roi_xyxy'),
            'g1_refinement_steps': g1_steps,
            'g2_refinement_steps': g2_steps,
            'phase_fields_quality_masked': True,
            'coordinate_convention': '+x right, +y down',
            'positive_rotation': (
                'mathematically positive (from +x towards +y); with the '
                'image convention +y down this appears clockwise on screen'
            ),
            'array_indexing': '[y, x]',
            'strain_unit': 'dimensionless; 0.01 = 1%',
            'displacement_unit': 'nm',
            'invalid_pixels': 'NaN',
            'phase_field_semantics': (
                'phase1/phase2 are wrapped geometric phases P_g1/P_g2 '
                'in radians; masked pixels are NaN'
            ),
            'png_show_all_pixels': show_all,
            'numeric_exports_quality_masked': True,
            'valid_fraction': float(np.mean(result.quality_mask)),
            'g_condition_number': float(result.g_condition_number),
            'extreme_strain_fraction_abs_gt_0_5': settings.get(
                'extreme_strain_fraction_abs_gt_0_5'
            ),
            'python': platform.python_version(),
            'dependency_versions': _dependency_versions(),
            'outputs': [
                os.path.basename(path) for path in prospective
            ],
        }
        with open(metadata_path, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            f.write('\n')

        return saved

    # ==============================================================
    # 配色管理
    # ==============================================================
    def _get_cmap_list(self, kind='field'):
        """返回配色名称列表 (用于 Combobox values)"""
        result = []
        for group_name, cmaps in self._colormaps.items():
            # 功率谱不提供为对称应变场设计的配色组
            if kind == 'power' and group_name.startswith(
                ('📊', 'Strain++')
            ):
                continue
            for cmap_name, cmap_label in cmaps:
                result.append(f'{cmap_label} ({cmap_name})')
        return result

    def _find_cmap_display_name(self, cmap_name, kind='field'):
        """根据 cmap 内部名找到显示名"""
        for group_name, cmaps in self._colormaps.items():
            if kind == 'power' and group_name.startswith(
                ('📊', 'Strain++')
            ):
                continue
            for cn, cl in cmaps:
                if cn == cmap_name:
                    return f'{cl} ({cn})'
        return f'?? ({cmap_name})'

    @staticmethod
    def _extract_cmap_name(disp_str):
        """从显示名 "经典热力 (hot)" 提取短名 "hot" """
        if not disp_str:
            return '_spp_turbo'
        idx = disp_str.rfind('(')
        if idx >= 0 and disp_str.endswith(')'):
            return disp_str[idx + 1:-1]
        return disp_str.strip()

    def _on_field_cmap_change(self, event=None):
        """场图配色切换 — 直接读取 Combobox 并应用"""
        new_disp = self._cmb_field.get()
        new_name = self._extract_cmap_name(new_disp)
        if new_name == self.cmap_field:
            return
        self.cmap_field = new_name
        self._cmap_field_var.set(new_disp)  # 保持 UI 同步
        self._custom_cmap = ''  # 下拉框选择覆盖自定义
        if self.result is not None:
            self._redraw_all_fields()
        self.statusbar.config(text=f'场图配色 → {self.cmap_field}')

    def _on_power_cmap_change(self, event=None):
        """功率谱配色切换 — 直接读取 Combobox 并应用"""
        new_disp = self._cmb_power.get()
        new_name = self._extract_cmap_name(new_disp)
        if new_name == self.cmap_power:
            return
        self.cmap_power = new_name
        self._cmap_power_var.set(new_disp)
        if self.gpa.power_spectrum is not None:
            self._show_power_spectrum()
        self.statusbar.config(text=f'功率谱配色 → {self.cmap_power}')

    def _reset_cmap(self):
        """重置配色的自定义设置，回到预设"""
        self._custom_cmap = ''
        self._custom_reverse = False
        self._custom_ranges = {}
        self._per_field_cmap = {}
        # 清除所有标签的 * 标记
        for i, (n, label, _) in enumerate(self._tab_defs):
            clean_label = label.rstrip(' *')
            if self.notebook.tab(i, 'text') != clean_label:
                self.notebook.tab(i, text=clean_label)
        # 恢复下拉框默认值
        self.cmap_field = self._extract_cmap_name(self._cmb_field.get())
        self.cmap_power = self._extract_cmap_name(self._cmb_power.get())
        if self.result is not None:
            self._redraw_all_fields()
        if self.gpa.power_spectrum is not None:
            self._show_power_spectrum()
        self.statusbar.config(text='配色已重置 — 下拉框模式')

    def _redraw_all_fields(self):
        """用当前配色重绘所有场量"""
        if self.result is None:
            return

        for name in RESULT_FIELDS:
            if name not in self._axes:
                continue
            data = self._display_field(name)
            if data is None:
                continue
            self._show_field(
                name, data, self._get_field_title(name),
                'gray' if name == 'quality_mask' else self.cmap_field,
                symmetric=self._field_is_symmetric(name),
                unit=FIELD_UNITS.get(name, ''),
            )

    # ==============================================================
    # 标签页右键菜单 — 单独设置每张图的配色
    # ==============================================================
    def _on_tab_right_click(self, event):
        """右键点击标签页弹出配色菜单"""
        try:
            # 识别被点击的是哪个标签 (ttk.Notebook 的 identify tab 方法)
            tab_index = self.notebook.tk.call(self.notebook._w, 'identify', 'tab',
                                               event.x, event.y)
            if tab_index == '':
                return
            tab_index = int(tab_index)
            if tab_index >= len(self._tab_defs):
                return
            name = self._tab_defs[tab_index][0]
            label = self._tab_defs[tab_index][1]
            # 原图和功率谱使用自己的顺序色图
            if name in ('image', 'power'):
                return
        except Exception:
            return

        # 弹出自定义菜单
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label=f'▼ {label} 单独配色', state='disabled')

        # 添加常用配色
        popular_cmaps = [
            ('_spp_turbo', 'Turbo (原版默认)'), ('_spp_polar', 'Polar (原版)'),
            ('_spp_blor', 'BlOr (原版)'), ('_spp_thermal', 'Thermal (原版)'),
            ('RdBu_r',  '红蓝'), ('coolwarm', '冷暖'),
            ('seismic', '地震'), ('bwr', '蓝白红'),
            ('viridis', '翠绿'), ('plasma', '等离子'),
            ('inferno', '地狱火'),('magma',  '岩浆'),
            ('hot',     '热力'), ('jet',   '彩虹'),
        ]

        # 标记当前使用的配色
        current = self._per_field_cmap.get(name, self.cmap_field)

        for cn, cl in popular_cmaps:
            label_str = f'  ● {cl} ({cn})' if cn == current else f'    {cl} ({cn})'
            menu.add_command(
                label=label_str,
                command=lambda v=cn, n=name: self._set_per_field_cmap(n, v)
            )

        menu.add_separator()
        # 清除该图的独立配色（回到全局默认）
        menu.add_command(label='  清除独立配色（用全局默认）',
                         command=lambda n=name: self._clear_per_field_cmap(n))

        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _set_per_field_cmap(self, name, cmap_name):
        """设置某个场量的独立配色"""
        self._per_field_cmap[name] = cmap_name
        self._update_tab_label(name)
        # 重绘该场量
        if self.result is not None:
            data = self._display_field(name)
            if data is not None:
                self._show_field(name, data, self._get_field_title(name),
                                 self.cmap_field,
                                 symmetric=self._field_is_symmetric(name))
        self.statusbar.config(text=f'{name} 独立配色 → {cmap_name}')

    def _clear_per_field_cmap(self, name):
        """清除某个场量的独立配色，回到全局默认"""
        self._per_field_cmap.pop(name, None)
        self._update_tab_label(name)
        if self.result is not None:
            data = self._display_field(name)
            if data is not None:
                self._show_field(name, data, self._get_field_title(name),
                                 self.cmap_field,
                                 symmetric=self._field_is_symmetric(name))
        self.statusbar.config(text=f'{name} 已清除独立配色')

    def _update_tab_label(self, name):
        """更新标签页文字，有独立配色时加*标记"""
        for i, (n, label, _) in enumerate(self._tab_defs):
            if n == name:
                # 从当前 tab 文字判断是否已有 *，用原标签做加减
                current_text = self.notebook.tab(i, 'text')
                has_star = current_text.endswith(' *')
                has_override = name in self._per_field_cmap
                if has_override and not has_star:
                    self.notebook.tab(i, text=label + ' *')
                elif not has_override and has_star:
                    self.notebook.tab(i, text=label)
                break

    def _get_field_title(self, name):
        """根据内部名获取显示标题"""
        titles = {
            'phase1': '相位 P_g1', 'phase2': '相位 P_g2',
            'eps_xx': '应变 ε_xx', 'eps_yy': '应变 ε_yy', 'eps_xy': '应变 ε_xy = ε_yx',
            'e_xx': '畸变 e_xx', 'e_yy': '畸变 e_yy', 'e_xy': '畸变 e_xy', 'e_yx': '畸变 e_yx',
            'omega_xy': '旋转 ω_xy', 'dilatation': '膨胀率 Δ',
            'u_x': '位移 u_x', 'u_y': '位移 u_y',
            'quality_mask': '可靠像素掩膜（白=有效）',
        }
        return titles.get(name, name)

    @staticmethod
    def _field_is_symmetric(name):
        """Whether automatic limits should be centred around zero."""
        return name not in ('image', 'power', 'quality_mask')

    # ==============================================================
    # 自定义配色调节对话框
    # ==============================================================
    def _open_cmap_dialog(self):
        """打开手动调节配色的对话框"""
        top = tk.Toplevel(self.root)
        top.title('自定义配色调节')
        top.geometry('500x450')
        top.resizable(False, False)

        frame = ttk.Frame(top, padding=15)
        frame.pack(fill=tk.BOTH, expand=True)

        # ===== 色图名称 =====
        ttk.Label(frame, text='色图名称（输入任意 Matplotlib colormap）:',
                  font=('', 10)).pack(anchor=tk.W)
        self._dlg_cmap_var = tk.StringVar(value=self._custom_cmap or self.cmap_field)
        cmap_entry = ttk.Entry(frame, textvariable=self._dlg_cmap_var, width=40)
        cmap_entry.pack(fill=tk.X, pady=2)

        # 常用色图快速选择
        quick_frame = ttk.Frame(frame)
        quick_frame.pack(fill=tk.X, pady=3)
        ttk.Label(quick_frame, text='快速选:', font=('', 8)).pack(side=tk.LEFT)
        popular = ['_spp_turbo', '_spp_polar', '_spp_blor', 'RdBu_r',
                   'coolwarm', 'viridis', 'plasma', 'inferno', 'hot', 'jet']
        for p in popular:
            btn = ttk.Button(quick_frame, text=p, width=9,
                             command=lambda v=p: self._dlg_cmap_var.set(v))
            btn.pack(side=tk.LEFT, padx=1)

        # ===== 反转 =====
        self._dlg_reverse_var = tk.BooleanVar(value=self._custom_reverse)
        ttk.Checkbutton(frame, text='反转色图 (Reverse)', variable=self._dlg_reverse_var,
                        ).pack(anchor=tk.W, pady=(8, 2))

        # ===== 范围调节 =====
        range_group = ttk.LabelFrame(frame, text='色值范围调节（留空 = 自动对称）', padding=10)
        range_group.pack(fill=tk.X, pady=8)

        target = self._tab_defs[
            self.notebook.index(self.notebook.select())
        ][0]
        info_text = ''
        data = (
            self._display_field(target)
            if self.result is not None else None
        )
        if data is not None:
            finite = np.asarray(data)[np.isfinite(data)]
            if finite.size:
                vmin, vmax = self._automatic_limits(
                    target, finite, self._field_is_symmetric(target)
                )
                info_text = (
                    f'当前有限范围: [{finite.min():.5f}, {finite.max():.5f}]\n'
                    f'建议显示范围: [{vmin:.5f}, {vmax:.5f}]'
                )
        ttk.Label(range_group, text=info_text, font=('Consolas', 9),
                  foreground='gray').pack(anchor=tk.W)

        range_row = ttk.Frame(range_group)
        range_row.pack(fill=tk.X, pady=3)

        # 读取当前自定义范围
        cur_min = cur_max = ''
        if target in self._custom_ranges:
            cur_min = f'{self._custom_ranges[target][0]:.6f}'
            cur_max = f'{self._custom_ranges[target][1]:.6f}'

        ttk.Label(range_row, text='最小:').pack(side=tk.LEFT)
        self._dlg_vmin_var = tk.StringVar(value=cur_min)
        ttk.Entry(range_row, textvariable=self._dlg_vmin_var, width=10).pack(side=tk.LEFT, padx=2)

        ttk.Label(range_row, text='最大:').pack(side=tk.LEFT, padx=(8, 0))
        self._dlg_vmax_var = tk.StringVar(value=cur_max)
        ttk.Entry(range_row, textvariable=self._dlg_vmax_var, width=10).pack(side=tk.LEFT, padx=2)

        ttk.Label(range_row, text='  (对应当前标签页)').pack(side=tk.LEFT, padx=5)

        # ===== 预览 & 应用 =====
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=15)

        def _apply_custom():
            """应用自定义设置到所有场图"""
            # 色图名
            name = self._dlg_cmap_var.get().strip()
            if name:
                try:
                    import matplotlib as mpl
                    if hasattr(mpl, 'colormaps') and hasattr(mpl.colormaps, 'get_cmap'):
                        mpl.colormaps.get_cmap(name)
                    else:
                        mpl.cm.get_cmap(name)
                    self._custom_cmap = name
                except Exception:
                    messagebox.showwarning('无效色图',
                        f'"{name}" 不是有效的 Matplotlib colormap\n'
                        f'常见: _spp_turbo (原版), RdBu_r, coolwarm, '
                        f'seismic, viridis, hot, jet')
                    return
            else:
                self._custom_cmap = ''

            # 反转
            self._custom_reverse = self._dlg_reverse_var.get()

            # 范围：只保存当前标签页的
            target_name = self._tab_defs[self.notebook.index(self.notebook.select())][0]
            vmin_str = self._dlg_vmin_var.get().strip()
            vmax_str = self._dlg_vmax_var.get().strip()
            if vmin_str and vmax_str:
                try:
                    self._custom_ranges[target_name] = (float(vmin_str), float(vmax_str))
                except ValueError:
                    messagebox.showwarning('无效范围', '请输入有效数字')
                    return
            else:
                self._custom_ranges.pop(target_name, None)

            # 重绘所有
            if self.result is not None:
                self._redraw_all_fields()

            self.statusbar.config(
                text=f'自定义配色: cmap={name or "下拉"}, '
                     f'reverse={"是" if self._custom_reverse else "否"}, '
                     f'自定义范围字段={len(self._custom_ranges)}'
            )
            top.destroy()

        def _reset_custom():
            """清除所有自定义设置，回到下拉框预设"""
            self._custom_cmap = ''
            self._custom_reverse = False
            self._custom_ranges = {}
            if self.result is not None:
                self._redraw_all_fields()
            self.statusbar.config(text='自定义配色已重置 → 预设模式')
            top.destroy()

        ttk.Button(btn_frame, text='✅ 应用', command=_apply_custom, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text='↺ 重置为预设', command=_reset_custom, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text='✕ 取消', command=top.destroy, width=8).pack(side=tk.RIGHT)

        # 底部提示
        ttk.Label(frame, text='💡 提示: 输入自定义 colormap 名会覆盖下拉框选择；'
                  '重置后可恢复', font=('', 8), foreground='gray').pack(anchor=tk.W)

    # ==============================================================
    # 帮助 & 关于
    # ==============================================================
    def _show_help(self):
        help_text = """Strain++ GPA 使用说明
━━━━━━━━━━━━━━━━━━━━━━

【操作流程】
1. 文件 → 打开图像（支持 TIFF / DM3 / DM4）
   或 "打开连续图像" 导入多页堆栈 TIF / 帧序列文件夹
   （堆栈模式：先在第一帧上完成 G 选择与计算，然后可批量处理全部帧）
2. 若图像含标尺、文字或黑边，先在原图点击「裁剪分析区」排除
3. 在 FFT 功率谱上：
   ● 左键点击 → 选择 G1（红圈）
   ● 右键点击 → 选择 G2（绿圈）
   选择两个不同方向的 Bragg 亮点
4. 点击「自动检测Bragg峰」可帮助定位
5. 调整参数（高斯σ、Hann窗等）；可在原图框选均匀无应变参考区
6. 点击「计算应变」
7. 切换上方标签页查看全部结果：
   📊 功率谱 | 📡 相位P_g1/P_g2
   📐 应变ε_xx/ε_yy/ε_xy
   📏 畸变e_xx/e_yy/e_xy/e_yx
   🔄 旋转ω_xy | 膨胀率Δ
   📍 位移u_x/u_y
8. 文件 → 导出当前结果，或点击「全部导出」
   PNG 是配色预览；浮点 TIFF/NPY 才是定量数据，NaN 表示不可靠像素

【连续图像批量处理】
- "打开连续图像"后，信息面板显示帧数与帧率
- 第一帧照常选 G、设 σ、（建议）框选参考区，点击"计算应变"
- 计算完成后工具栏出现"批量处理全部帧"
- 可勾选导出场（应变/旋转/膨胀/畸变/位移/相位/掩膜）、
  堆栈 TIF 与单帧序列、帧范围与步长
- 建议开启"逐帧精修 G"：样品漂移/转动时 G 自动跟随，
  每帧修正量记录在 batch_statistics.csv（即 G 漂移曲线数据）
- 单帧失败不会中断批处理，失败原因记录在 CSV 中

【选择 G 矢量的技巧】
- 选两个不同方向的亮斑（通常夹角 60°~120°）
- 选最近邻的 Bragg 峰效果最好
- σ 值 = Bragg峰距离 / 6（与原版 Strain++ 自动值一致）；
  越小越平滑，越大纹理越丰富但越嘈杂
- 两个 G 矢量不要太靠近 DC 中心

【参数说明】
- 高斯σ1/σ2: 分别控制 G1/G2 Bragg 峰的提取范围，两个峰可以不同
  越大=越宽=提取更多周围信号
  越小=越紧=仅提取峰中心
- Hann窗: 减少FFT边缘伪影
- 显示全部像素: 默认勾选，所有像素按原始值上色（原版 Strain++ 风格）；
  取消勾选则只显示可靠像素，被剔除区域显示为灰色
- 统计与浮点 TIFF/NPY 导出始终套用质量掩膜，不受该开关影响
- 旋转角: 旋转输出坐标系（度）
- 像素尺寸: 单值表示方形像素；也可按 Y,X 输入两个值
- 参考区: 只选已知均匀、无应变且晶格清晰的区域

【分量含义】
应变(对称) ε = ½(e + eᵀ):
  ε_xx = 沿x方向正应变
  ε_yy = 沿y方向正应变
  ε_xy = ε_yx = 剪应变
畸变(非对称) e = ∇u:
  e_xx = ∂u_x/∂x   e_xy = ∂u_x/∂y
  e_yx = ∂u_y/∂x   e_yy = ∂u_y/∂y
旋转(反对称) ω = ½(e − eᵀ):
  ω_xy = ½(e_xy − e_yx) = 晶格旋转角
膨胀率: Δ = ε_xx + ε_yy = tr(ε)
"""
        top = tk.Toplevel(self.root)
        top.title('使用说明')
        top.geometry('600x550')
        text = tk.Text(top, wrap=tk.WORD, font=('Consolas', 10), padx=10, pady=10)
        text.insert('1.0', help_text)
        text.config(state=tk.DISABLED)
        text.pack(fill=tk.BOTH, expand=True)
        ttk.Button(top, text='关闭', command=top.destroy).pack(pady=5)

    def _show_license(self):
        """Display the complete GPL text shipped with source and executable."""
        base = getattr(sys, '_MEIPASS', os.path.dirname(__file__))
        license_path = os.path.join(base, 'LICENSE')
        if not os.path.isfile(license_path):
            try:
                import importlib.metadata
                distribution = importlib.metadata.distribution('strainpp-gpa')
                for entry in distribution.files or ():
                    if entry.name == 'LICENSE':
                        candidate = distribution.locate_file(entry)
                        if candidate.is_file():
                            license_path = str(candidate)
                            break
            except (
                importlib.metadata.PackageNotFoundError,
                AttributeError,
                OSError,
            ):
                pass
        try:
            with open(license_path, 'r', encoding='utf-8') as stream:
                license_text = stream.read()
        except OSError as error:
            messagebox.showerror('许可协议', f'无法读取 LICENSE:\n{error}')
            return
        top = tk.Toplevel(self.root)
        top.title('GNU GPL v3 许可协议')
        top.geometry('760x600')
        text = tk.Text(
            top, wrap=tk.WORD, font=('Consolas', 9), padx=10, pady=10
        )
        scrollbar = ttk.Scrollbar(top, orient=tk.VERTICAL, command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text.insert('1.0', license_text)
        text.config(state=tk.DISABLED)

    def _show_about(self):
        try:
            from _version import __version__ as app_version
        except Exception:
            app_version = 'unknown'
        about_text = (
            f'Strain++ GPA v{app_version}\n\n'
            '几何相位分析 (Geometric Phase Analysis)\n'
            '用于 HRTEM 图像的纳米级应变测量\n\n'
            '参考论文:\n'
            'Hytch, Snoeck & Kilaas (1998)\n'
            'Ultramicroscopy 74, 131-146\n\n'
            '原始 C++ 软件: JJPPeters/Strainpp\n\n'
            'GNU GPL v3 或更高版本；本软件不提供任何担保。\n'
            '完整条款见“帮助 → 许可协议”。'
        )
        messagebox.showinfo('关于 Strain++ GPA', about_text)


# ============================================================
# 批量处理设置 + 进度对话框
# ============================================================
class BatchSettingsDialog(tk.Toplevel):
    """批量处理参数设置与进度显示。

    参数取自第一帧的实际计算设置（G、σ、标定、Hann、旋转、参考区），
    用户仅需选择导出场、输出形式、帧范围与输出目录。
    """

    def __init__(self, gui):
        super().__init__(gui.root)
        self.gui = gui
        self.title('批量处理全部帧')
        self.geometry('560x640')
        self.resizable(False, True)
        self.protocol('WM_DELETE_WINDOW', self._on_close)

        settings = gui._last_compute_settings
        info = gui._stack_info
        self._n_frames = info.n_frames if info else 1
        self._running = False

        frame = ttk.Frame(self, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        # ---- 参数摘要 ----
        g1 = settings.get('g1', [0, 0])
        g2 = settings.get('g2', [0, 0])
        summary = (
            f'来源: {os.path.basename(gui._stack_source)}  '
            f'共 {self._n_frames} 帧\n'
            f'G1: ({g1[0]:.2f}, {g1[1]:.2f})  σ1={settings.get("sigma1_px")}   '
            f'G2: ({g2[0]:.2f}, {g2[1]:.2f})  σ2={settings.get("sigma2_px")}\n'
            f'像素尺寸: {gui._format_pixel_size(settings.get("pixel_size_nm_yx", 1.0))} nm  '
            f'旋转: {settings.get("rotation_degrees", 0):g}°  '
            f'Hann: {"开" if settings.get("hann_window") else "关"}'
        )
        ttk.Label(frame, text=summary, justify=tk.LEFT,
                  foreground='gray').pack(anchor=tk.W, pady=(0, 8))

        # ---- 导出场选择 ----
        fields_group = ttk.LabelFrame(frame, text='导出场（可多选）', padding=8)
        fields_group.pack(fill=tk.X, pady=3)
        self._field_vars = {}
        order = list(_BATCH_FIELDS.items())
        for index, (name, label) in enumerate(order):
            row, col = divmod(index, 3)
            var = tk.BooleanVar(value=name in _DEFAULT_FIELDS)
            check = ttk.Checkbutton(fields_group, text=label, variable=var)
            check.grid(row=row, column=col, sticky=tk.W, padx=4, pady=1)
            self._field_vars[name] = var

        # ---- 输出形式 ----
        out_group = ttk.LabelFrame(frame, text='输出形式', padding=8)
        out_group.pack(fill=tk.X, pady=3)
        self._var_stack = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            out_group, text='多页堆栈 TIF（每个场一个文件，推荐）',
            variable=self._var_stack,
        ).pack(anchor=tk.W)
        self._var_preview = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            out_group,
            text='彩色预览堆栈（8-bit，可直接查看/汇报；定量仍用上面的 float 堆栈）',
            variable=self._var_preview,
        ).pack(anchor=tk.W)
        seq_row = ttk.Frame(out_group)
        seq_row.pack(anchor=tk.W)
        self._var_sequence = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            seq_row, text='单帧序列文件夹', variable=self._var_sequence,
        ).pack(side=tk.LEFT)
        ttk.Label(seq_row, text='文件夹名:').pack(side=tk.LEFT, padx=(10, 2))
        self._var_seq_dir = tk.StringVar(value='frames')
        ttk.Entry(seq_row, textvariable=self._var_seq_dir,
                  width=12).pack(side=tk.LEFT)

        # ---- 帧范围 ----
        range_group = ttk.LabelFrame(frame, text='帧范围', padding=8)
        range_group.pack(fill=tk.X, pady=3)
        ttk.Label(range_group, text='起始帧:').pack(side=tk.LEFT)
        self._var_start = tk.IntVar(value=0)
        ttk.Spinbox(range_group, from_=0, to=self._n_frames - 1,
                    textvariable=self._var_start,
                    width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(range_group, text='结束帧(含):').pack(side=tk.LEFT, padx=(10, 0))
        self._var_stop = tk.IntVar(value=self._n_frames - 1)
        ttk.Spinbox(range_group, from_=0, to=self._n_frames - 1,
                    textvariable=self._var_stop,
                    width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(range_group, text='步长:').pack(side=tk.LEFT, padx=(10, 0))
        self._var_step = tk.IntVar(value=1)
        ttk.Spinbox(range_group, from_=1, to=max(1, self._n_frames),
                    textvariable=self._var_step,
                    width=4).pack(side=tk.LEFT, padx=2)

        # ---- G 精修 ----
        refine_group = ttk.LabelFrame(frame, text='每帧 G 矢量精修', padding=8)
        refine_group.pack(fill=tk.X, pady=3)
        has_roi = settings.get('reference_roi_xyxy') is not None
        self._var_refine = tk.BooleanVar(value=has_roi)
        refine_check = ttk.Checkbutton(
            refine_group,
            text='使用第一帧的参考区逐帧修正 G（推荐；跟随样品漂移）',
            variable=self._var_refine,
        )
        refine_check.pack(anchor=tk.W)
        if not has_roi:
            refine_check.config(state=tk.DISABLED)
            ttk.Label(
                refine_group,
                text='（未设置参考区——如需逐帧精修，请先在第一帧框选参考区并重算）',
                foreground='gray',
            ).pack(anchor=tk.W)

        # ---- 输出目录 ----
        outdir_group = ttk.LabelFrame(frame, text='输出目录', padding=8)
        outdir_group.pack(fill=tk.X, pady=3)
        stack_source = gui._stack_source or '.'
        if os.path.isdir(stack_source):
            # 序列文件夹：输出放文件夹内，避免与其兄弟文件混在一起
            default_out = os.path.join(stack_source, 'batch_output')
        else:
            default_out = os.path.join(
                os.path.dirname(stack_source), 'batch_output'
            )
        self._var_outdir = tk.StringVar(value=default_out)
        ttk.Entry(outdir_group, textvariable=self._var_outdir).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(outdir_group, text='浏览…',
                   command=self._browse_outdir).pack(side=tk.LEFT, padx=4)

        # ---- 开始 / 进度 ----
        self._btn_start = ttkb.Button(frame, text='▶ 开始批量处理',
                                      command=self._start, bootstyle='success')
        self._btn_start.pack(pady=10)

        self._progress_bar = ttkb.Progressbar(
            frame, mode='determinate', bootstyle='striped-success',
        )
        self._var_progress = tk.StringVar(value='')
        self._lbl_progress = ttk.Label(frame, textvariable=self._var_progress,
                                       justify=tk.LEFT)
        self._btn_cancel_batch = ttkb.Button(
            frame, text='■ 取消批量', command=self._cancel_batch,
            bootstyle='danger',
        )

    def _browse_outdir(self):
        directory = filedialog.askdirectory(title='选择输出目录')
        if directory:
            self._var_outdir.set(directory)

    def _validate(self):
        if not any(var.get() for var in self._field_vars.values()):
            raise ValueError('请至少勾选一个导出场')
        if not self._var_stack.get() and not self._var_sequence.get():
            raise ValueError('请至少选择一种输出形式（堆栈或单帧序列）')
        start, stop, step = (
            self._var_start.get(), self._var_stop.get(), self._var_step.get()
        )
        if not (0 <= start <= stop < self._n_frames):
            raise ValueError(
                f'帧范围无效：需 0 ≤ 起始 ≤ 结束 ≤ {self._n_frames - 1}'
            )
        if step < 1:
            raise ValueError('步长必须 ≥ 1')
        outdir = self._var_outdir.get().strip()
        if not outdir:
            raise ValueError('请选择输出目录')
        return start, stop, step, outdir

    def _start(self):
        try:
            start, stop, step, outdir = self._validate()
        except ValueError as error:
            messagebox.showerror('参数错误', str(error), parent=self)
            return

        settings = self.gui._last_compute_settings
        pixel_values = np.asarray(
            settings.get('pixel_size_nm_yx', [1.0]), dtype=float
        ).ravel()
        # 保留各向异性标定：单值 → 标量，[y, x] → 二元组，与 GPA.load_image
        # 的契约一致（此前仅取 y 分量，各向异性输入的批量结果会与
        # 第一帧 GUI 结果不一致）。
        pixel_size = (
            float(pixel_values[0]) if pixel_values.size == 1
            else (float(pixel_values[0]), float(pixel_values[1]))
        )
        roi = settings.get('reference_roi_xyxy')
        refine = self._var_refine.get() and roi is not None
        # 所见即所得：把界面当前每场的配色与色标范围快照进批量配置
        selected_fields = [name for name, var in self._field_vars.items()
                           if var.get()]
        preview_specs = {
            name: dict(self.gui._field_display_state[name])
            for name in selected_fields
            if name in self.gui._field_display_state
        }
        config = BatchConfig(
            fields=selected_fields,
            export_stack=self._var_stack.get(),
            export_sequence=self._var_sequence.get(),
            sequence_dirname=self._var_seq_dir.get().strip() or 'frames',
            preview_stack=self._var_preview.get(),
            preview_specs=preview_specs,
            preview_masked=not self.gui._var_show_all_pixels.get(),
            refine_g=refine,
            reference_roi=tuple(roi) if refine else None,
            frame_start=start,
            frame_stop=stop + 1,
            frame_step=step,
            pixel_size=pixel_size,
            use_hann=bool(settings.get('hann_window')),
            rotation_degrees=float(settings.get('rotation_degrees', 0.0)),
            sigma1=float(settings.get('sigma1_px', 5.0)),
            sigma2=float(settings.get('sigma2_px', 5.0)),
            g1=tuple(settings.get('g1', (0.0, 0.0))),
            g2=tuple(settings.get('g2', (0.0, 0.0))),
        )
        try:
            config.validate()
        except ValueError as error:
            messagebox.showerror('参数错误', str(error), parent=self)
            return

        self._running = True
        self.gui._batch_cancel = False
        self.gui._batch_dialog = self
        self._btn_start.config(state=tk.DISABLED)
        self._progress_bar.pack(fill=tk.X, pady=(4, 2))
        self._lbl_progress.pack(anchor=tk.W)
        self._btn_cancel_batch.pack(pady=4)
        self.gui.statusbar.config(text='批量处理中...')

        # 批量消息使用独立 token，不受普通计算任务的 generation 失效影响：
        # 批量运行期间用户仍可打开新图或取消其它任务，批量完成消息必须
        # 始终送达，进度对话框才能正常收尾。
        self.gui._batch_token += 1
        token = self.gui._batch_token

        def progress_cb(done, total, message):
            self.gui._worker_queue.put((
                'batch_progress', token, (done, total, message),
            ))

        def cancel_check():
            return self.gui._batch_cancel

        def _run():
            try:
                records, cancelled = run_batch(
                    self.gui._stack_source, config, outdir,
                    progress_cb=progress_cb, cancel_check=cancel_check,
                )
                self.gui._worker_queue.put((
                    'batch_done', token,
                    (records, cancelled, outdir),
                ))
            except Exception as error:
                self.gui._worker_queue.put((
                    'batch_error', token, str(error),
                ))

        threading.Thread(target=_run, daemon=True).start()

    def on_progress(self, done, total, message):
        if not self.winfo_exists():
            return
        self._progress_bar.configure(maximum=total, value=done)
        self._var_progress.set(f'{done}/{total} 帧 — {message}')

    def on_done(self, records, cancelled, outdir):
        self._running = False
        failed = sum(1 for record in records if record.get('status') != 'ok')
        preview_note = ''
        if self._var_preview.get() and self._var_stack.get():
            preview_note = (
                '\n\n查看图像请打开 preview_batch_*.tif（8-bit 彩色，'
                '可直接看）；\nbatch_*.tif 是 float32 定量数据，'
                '在 ImageJ 中需 Auto 对比度'
                '（Ctrl+Shift+C）才能看到纹理。'
            )
        # 对话框可能已被用户提前关闭：弹窗父窗口必须回退到主窗口，
        # 否则对已销毁 widget 的引用会抛 TclError。
        alive = self.winfo_exists()
        parent = self if alive else self.gui.root
        if alive:
            self._var_progress.set(
                f'完成：{len(records)} 帧，{failed} 帧失败'
                + ('（已取消）' if cancelled else '')
            )
            self._btn_cancel_batch.pack_forget()
        self.gui.statusbar.config(
            text=f'批量处理完成 — {len(records)} 帧 → {outdir}'
        )
        messagebox.showinfo(
            '批量处理完成',
            f'已处理 {len(records)} 帧'
            + (f'，{failed} 帧失败' if failed else '')
            + ('（中途取消）' if cancelled else '')
            + f'\n\n结果目录:\n{outdir}\n\n'
            + '每帧统计与 G 漂移记录见 batch_statistics.csv'
            + preview_note,
            parent=parent,
        )

    def on_error(self, message):
        self._running = False
        alive = self.winfo_exists()
        parent = self if alive else self.gui.root
        if alive:
            self._var_progress.set(f'失败: {message}')
        messagebox.showerror('批量处理失败', message, parent=parent)

    def _cancel_batch(self):
        self.gui._batch_cancel = True
        self._var_progress.set('正在取消（当前帧完成后停止）...')

    def _on_close(self):
        if self._running:
            # 明确告知批量的去向，避免"关窗即取消"或"静默后台跑"的歧义。
            cancel = messagebox.askyesno(
                '批量处理进行中',
                '批量任务仍在后台运行。\n\n'
                '是 — 同时取消批处理（当前帧完成后停止）\n'
                '否 — 继续后台运行，完成后在状态栏提示',
                parent=self,
            )
            if cancel:
                self.gui._batch_cancel = True
                self.gui.statusbar.config(text='批量处理正在后台取消…')
            else:
                self.gui.statusbar.config(
                    text='批量处理仍在后台运行，完成后在此提示'
                )
        self.gui._batch_dialog = None
        self.destroy()


# ============================================================
# 入口
# ============================================================
def main():
    root = ttkb.Window(themename='cosmo')

    # 设置窗口图标（如果有的话）
    try:
        ico_path = os.path.join(os.path.dirname(__file__), 'strainpp.ico')
        if os.path.exists(ico_path):
            root.iconbitmap(ico_path)
    except Exception:
        pass

    app = StrainGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
