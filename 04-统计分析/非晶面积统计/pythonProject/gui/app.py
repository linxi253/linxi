# -*- coding: utf-8 -*-
"""
主应用窗口

整合所有模块，提供完整的 GUI 交互界面：
- 文件夹选择与参数配置（配置跨会话持久化）
- 图像预览与切片导航
- 单片分析 / 批量处理
- 结果显示与日志
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import queue
import threading
import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import ttkbootstrap as ttkb

from constants import (
    APP_FULL_TITLE,
    DEFAULT_WINDOW_WIDTH,
    DEFAULT_WINDOW_HEIGHT,
    MIN_WINDOW_WIDTH,
    MIN_WINDOW_HEIGHT,
    RIGHT_PANEL_WIDTH,
    DARK_TEXT_BG,
    LIGHT_TEXT_FG,
    RESIZE_DEBOUNCE_MS,
    DEFAULT_PIXEL_SIZE_NM,
    DEFAULT_PIXEL_CALIBRATED,
    DEFAULT_TIME_INTERVAL,
    PREVIEW_LARGE_FILE_MB,
    SHAPE_FACTOR_CIRCLE,
    SHAPE_FACTOR_SQUARE,
    VERIFY_SF_NEAR_CIRCLE,
    VERIFY_SF_REGULAR,
    COL_FILENAME,
    COL_SLICE,
)
from core.segmentation import segment_image, SegmentationParams, resolve_uint16_shift
from core.measurement import compute_full_measurement, create_annotated_image
from core.analysis import generate_summary_text
from core.logger import AppLogger
from io_utils.tiff_handler import (
    scan_tiff_files,
    scan_uint16_max_shift,
    load_tiff_stack,
    load_tiff_slices,
    get_slice_count,
    get_slice,
    get_tiff_info,
    save_annotated_image,
)
from io_utils.exporter import export_results
from io_utils.app_config import load_config, save_config
from gui.parameter_panel import ParameterPanel
from gui.canvas_viewer import CanvasViewer


class EMImageAnalyzerApp:
    """电镜晶体/非晶区域统计分析工具主应用"""

    def __init__(self, root):
        self.root = root
        self.root.title(APP_FULL_TITLE)
        self.root.geometry(f"{DEFAULT_WINDOW_WIDTH}x{DEFAULT_WINDOW_HEIGHT}")
        self.root.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)

        # ===== 状态变量 =====
        self.input_folder = ""
        self.output_folder = ""
        self.pixel_size = tk.DoubleVar(value=DEFAULT_PIXEL_SIZE_NM)
        self.pixel_calibrated = tk.BooleanVar(value=DEFAULT_PIXEL_CALIBRATED)
        self.time_interval = tk.DoubleVar(value=DEFAULT_TIME_INTERVAL)
        self.processing = False
        self.current_stack = None
        self.current_slice = 0
        self.total_slices = 0
        self.current_filename = None
        self._current_uint16_shift = None   # 当前堆栈的数据集级移位（加载时算一次）
        self._batch_thread = None
        self._closing = False

        # 子线程 -> 主线程 UI 更新队列（子线程只 put，主线程 after 轮询）
        self._ui_queue = queue.Queue()

        # ===== 构建 UI =====
        self._build_ui()

        # ===== 初始化日志 =====
        self.logger = AppLogger(self.root, self.log_text, self.status_var)

        # ===== 窗口事件 =====
        self._resize_timer = None
        self.root.bind('<Configure>', self._on_window_resize)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # 绑定快捷键
        self.root.bind('<Control-plus>', lambda e: self.canvas_viewer.zoom_in())
        self.root.bind('<Control-minus>', lambda e: self.canvas_viewer.zoom_out())
        self.root.bind('<Control-0>', lambda e: self.canvas_viewer.reset_view())

        # ===== 恢复上次会话配置 =====
        self._apply_config(load_config())

        # 主线程轮询批量处理 UI 队列
        self._poll_ui_queue()

    # ==================== UI 构建 ====================

    def _build_ui(self):
        """构建完整 UI 布局"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 标题
        ttk.Label(main_frame, text=APP_FULL_TITLE,
                  font=("Arial", 14, "bold")).pack(pady=(0, 8))

        # 文件夹与参数区域
        self._build_folder_section(main_frame)

        # 主显示区域（左：画布，右：结果面板）
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, pady=8)

        self._build_left_panel(display_frame)
        self._build_right_panel(display_frame)

        # 批量处理结果
        self._build_result_section(main_frame)

        # 日志区域
        self._build_log_section(main_frame)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(self.root, textvariable=self.status_var,
                  relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=3)

    def _build_folder_section(self, parent):
        """文件夹选择与全局参数"""
        folder_frame = ttk.LabelFrame(parent, text="文件夹与参数设置", padding=10)
        folder_frame.pack(fill=tk.X, pady=4)

        # 第一行：文件夹选择
        row1 = ttk.Frame(folder_frame)
        row1.pack(fill=tk.X, pady=3)

        # 输入文件夹
        input_frame = ttk.Frame(row1)
        input_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 15))
        ttk.Label(input_frame, text="输入:").pack(side=tk.LEFT)
        self.input_entry = ttk.Entry(input_frame)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.btn_input_browse = ttkb.Button(input_frame, text="浏览...",
                                            command=self._select_input_folder,
                                            width=7, bootstyle="outline")
        self.btn_input_browse.pack(side=tk.LEFT)

        # 输出文件夹
        output_frame = ttk.Frame(row1)
        output_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 15))
        ttk.Label(output_frame, text="输出:").pack(side=tk.LEFT)
        self.output_entry = ttk.Entry(output_frame)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.btn_output_browse = ttkb.Button(output_frame, text="浏览...",
                                             command=self._select_output_folder,
                                             width=7, bootstyle="outline")
        self.btn_output_browse.pack(side=tk.LEFT)

        # 第二行：参数
        row2 = ttk.Frame(folder_frame)
        row2.pack(fill=tk.X, pady=3)

        # 像素大小（未标定时禁用并固定为假定值 1.0，面积单位按 px² 解读）
        ttk.Label(row2, text="像素大小(nm/px):").pack(side=tk.LEFT, padx=(0, 3))
        self.pixel_size_entry = ttk.Entry(row2, textvariable=self.pixel_size, width=8)
        self.pixel_size_entry.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Checkbutton(row2, text="已标定", variable=self.pixel_calibrated,
                        command=self._on_calibrated_toggle).pack(side=tk.LEFT, padx=(0, 20))
        if not DEFAULT_PIXEL_CALIBRATED:
            self.pixel_size_entry.config(state='disabled')

        # 时间间隔
        ttk.Label(row2, text="时间间隔:").pack(side=tk.LEFT, padx=(0, 3))
        ttk.Entry(row2, textvariable=self.time_interval, width=8).pack(side=tk.LEFT, padx=(0, 20))

        # 文件选择
        ttk.Label(row2, text="文件:").pack(side=tk.LEFT, padx=(0, 3))
        self.file_list_var = tk.StringVar()
        self.file_combo = ttk.Combobox(row2, textvariable=self.file_list_var,
                                       width=22, state='readonly')
        self.file_combo.pack(side=tk.LEFT, padx=(0, 5))
        ttkb.Button(row2, text="刷新", command=self._refresh_file_list,
                    width=5, bootstyle="info-outline").pack(side=tk.LEFT, padx=2)
        ttkb.Button(row2, text="打开", command=self._load_selected_file,
                    width=5, bootstyle="primary").pack(side=tk.LEFT, padx=2)

        # 分割参数面板
        self.param_panel = ParameterPanel(folder_frame, on_params_changed=self._on_params_changed)
        self.param_panel.pack(fill=tk.X, pady=(5, 0))

    def _on_calibrated_toggle(self):
        """标定状态切换：未标定时像素尺寸固定为假定值 1.0（面积按 px² 解读）"""
        if self.pixel_calibrated.get():
            self.pixel_size_entry.config(state='normal')
        else:
            self.pixel_size.set(DEFAULT_PIXEL_SIZE_NM)
            self.pixel_size_entry.config(state='disabled')

    def _build_left_panel(self, parent):
        """左侧面板：切片导航 + 画布"""
        left = ttk.Frame(parent)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        # 切片导航
        nav_frame = ttk.LabelFrame(left, text="切片导航", padding=8)
        nav_frame.pack(fill=tk.X, pady=(0, 5))

        nav_btns = ttk.Frame(nav_frame)
        nav_btns.pack(expand=True)

        ttkb.Button(nav_btns, text="◀◀", command=lambda: self._change_slice(0),
                    width=4, bootstyle="outline").pack(side=tk.LEFT, padx=2)
        ttkb.Button(nav_btns, text="◀", command=lambda: self._change_slice(-1),
                    width=4, bootstyle="outline").pack(side=tk.LEFT, padx=2)

        self.slice_label = ttk.Label(nav_btns, text="切片: 0/0", width=12)
        self.slice_label.pack(side=tk.LEFT, padx=8)

        ttkb.Button(nav_btns, text="▶", command=lambda: self._change_slice(1),
                    width=4, bootstyle="outline").pack(side=tk.LEFT, padx=2)
        ttkb.Button(nav_btns, text="▶▶", command=lambda: self._change_slice('end'),
                    width=4, bootstyle="outline").pack(side=tk.LEFT, padx=2)

        # 操作按钮
        ctrl_frame = ttk.Frame(nav_frame)
        ctrl_frame.pack(pady=(5, 0))
        ttkb.Button(ctrl_frame, text="预览分割", command=self._preview_segmentation,
                    bootstyle="info").grid(row=0, column=0, padx=4)
        ttkb.Button(ctrl_frame, text="分析当前", command=self._analyze_current_slice,
                    bootstyle="success").grid(row=0, column=1, padx=4)
        ttkb.Button(ctrl_frame, text="重置视图", command=self._reset_view,
                    bootstyle="outline").grid(row=0, column=2, padx=4)
        ttkb.Button(ctrl_frame, text="验证计算", command=self._verify_calculations,
                    bootstyle="outline").grid(row=0, column=3, padx=4)

        # 画布
        canvas_frame = ttk.LabelFrame(left, text="图像预览 (滚轮缩放/拖拽平移)", padding=5)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas_viewer = CanvasViewer(canvas_frame)
        self.canvas_viewer.pack(fill=tk.BOTH, expand=True)

    def _build_right_panel(self, parent):
        """右侧面板：分析结果 + 批量控制"""
        right = ttk.Frame(parent, width=RIGHT_PANEL_WIDTH)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        # 分析结果
        result_frame = ttk.LabelFrame(right, text="切片分析结果", padding=8)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self.preview_text = tk.Text(result_frame, wrap=tk.WORD, state='disabled',
                                    height=16, bg=DARK_TEXT_BG, fg=LIGHT_TEXT_FG,
                                    font=('Consolas', 9), relief=tk.FLAT, padx=6, pady=6)
        self.preview_text.pack(fill=tk.BOTH, expand=True)

        # 批量处理
        batch_frame = ttk.LabelFrame(right, text="批量处理", padding=8)
        batch_frame.pack(fill=tk.X)

        btn_row = ttk.Frame(batch_frame)
        btn_row.pack(fill=tk.X, pady=3)
        self.btn_start = ttkb.Button(btn_row, text="开始批量处理",
                                     command=self._start_batch, bootstyle="primary")
        self.btn_start.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self.btn_stop = ttkb.Button(btn_row, text="停止",
                                    command=self._stop_batch, state='disabled', bootstyle="danger")
        self.btn_stop.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)

        self.progress = ttkb.Progressbar(batch_frame, mode='determinate',
                                         bootstyle="striped-primary", maximum=100)
        self.progress.pack(fill=tk.X, pady=3)
        self.progress_label = ttk.Label(batch_frame, text="")
        self.progress_label.pack(fill=tk.X)

    def _build_result_section(self, parent):
        """批量处理结果区域"""
        frame = ttk.LabelFrame(parent, text="批量处理统计结果", padding=8)
        frame.pack(fill=tk.X, pady=4)

        self.result_text = tk.Text(frame, wrap=tk.WORD, height=7,
                                   bg=DARK_TEXT_BG, fg=LIGHT_TEXT_FG,
                                   font=('Consolas', 9), relief=tk.FLAT, padx=6, pady=6)
        self.result_text.pack(fill=tk.BOTH, expand=True)

    def _build_log_section(self, parent):
        """日志区域"""
        frame = ttk.LabelFrame(parent, text="处理日志", padding=8)
        frame.pack(fill=tk.X, pady=4)

        self.log_text = tk.Text(frame, wrap=tk.WORD, height=5,
                                bg=DARK_TEXT_BG, fg=LIGHT_TEXT_FG,
                                font=('Consolas', 9), relief=tk.FLAT, padx=6, pady=6)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=(4, 0))
        ttkb.Button(btn_row, text="清空日志", command=self._clear_log,
                    bootstyle="outline").pack(side=tk.LEFT, padx=4)
        ttkb.Button(btn_row, text="保存日志", command=self._save_log,
                    bootstyle="info-outline").pack(side=tk.LEFT, padx=4)

    # ==================== 配置持久化 ====================

    def _collect_config(self) -> dict:
        """收集当前 UI 状态为可持久化配置（仅主线程调用）"""
        try:
            pixel_size = float(self.pixel_size.get())
        except (tk.TclError, ValueError):
            pixel_size = DEFAULT_PIXEL_SIZE_NM
        try:
            time_interval = float(self.time_interval.get())
        except (tk.TclError, ValueError):
            time_interval = DEFAULT_TIME_INTERVAL

        p = self.param_panel.get_params()
        return {
            'input_folder': self.input_folder,
            'output_folder': self.output_folder,
            'pixel_size': pixel_size,
            'pixel_calibrated': bool(self.pixel_calibrated.get()),
            'time_interval': time_interval,
            'segmentation': {
                'method': p.method,
                'threshold': p.threshold,
                'adaptive_blocksize': p.adaptive_blocksize,
                'adaptive_c': p.adaptive_c,
                'canny_low': p.canny_low,
                'canny_high': p.canny_high,
                'canny_min_area': p.canny_min_area,
                'watershed_kernel': p.watershed_kernel,
                'foreground': p.foreground,
            },
        }

    def _apply_config(self, cfg: dict):
        """恢复持久化配置（非法/缺失项静默跳过，保持默认值）"""
        if not isinstance(cfg, dict):
            return

        input_folder = cfg.get('input_folder')
        if isinstance(input_folder, str) and input_folder and os.path.isdir(input_folder):
            self.input_folder = input_folder
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, input_folder)
            self._refresh_file_list()

        output_folder = cfg.get('output_folder')
        if isinstance(output_folder, str) and output_folder:
            self.output_folder = output_folder
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, output_folder)

        pixel_calibrated = bool(cfg.get('pixel_calibrated', False))
        self.pixel_calibrated.set(pixel_calibrated)
        self._on_calibrated_toggle()
        if pixel_calibrated:
            try:
                val = float(cfg.get('pixel_size', DEFAULT_PIXEL_SIZE_NM))
                if val > 0:
                    self.pixel_size.set(val)
            except (tk.TclError, TypeError, ValueError):
                pass

        try:
            val = float(cfg.get('time_interval', DEFAULT_TIME_INTERVAL))
            if val > 0:
                self.time_interval.set(val)
        except (tk.TclError, TypeError, ValueError):
            pass

        seg = cfg.get('segmentation')
        if isinstance(seg, dict):
            self.param_panel.apply_config(seg)

    def _save_config(self):
        save_config(self._collect_config())

    # ==================== 文件夹操作 ====================

    def _select_input_folder(self):
        folder = filedialog.askdirectory(title="选择包含 TIFF 文件的文件夹")
        if folder:
            self.input_folder = folder
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, folder)
            self.logger.log(f"输入文件夹: {folder}")
            self._refresh_file_list()
            self._save_config()

    def _select_output_folder(self):
        folder = filedialog.askdirectory(title="选择结果输出文件夹")
        if folder:
            self.output_folder = folder
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, folder)
            self.logger.log(f"输出文件夹: {folder}")
            self._save_config()

    def _refresh_file_list(self):
        if not self.input_folder:
            return
        try:
            files = scan_tiff_files(self.input_folder)
            self.file_combo['values'] = files
            if files:
                self.file_combo.current(0)
                self.logger.log(f"找到 {len(files)} 个 TIFF 文件")
            else:
                self.logger.warn("未找到 TIFF 文件")
        except Exception as e:
            self.logger.error(f"刷新文件列表失败: {e}")

    def _load_selected_file(self):
        selected = self.file_combo.get()
        if not selected:
            messagebox.showwarning("提示", "请先选择一个文件")
            return
        if not self.input_folder:
            messagebox.showerror("错误", "请先选择输入文件夹")
            return

        file_path = os.path.join(self.input_folder, selected)
        try:
            # 内存护栏（2026-09-28 审查项）：预览路径整卷载入内存，
            # 大文件先告知体积并请用户确认
            try:
                info = get_tiff_info(file_path)
                size_mb = info['file_size_mb']
                meta_slices = info['num_slices']
            except Exception:
                size_mb, meta_slices = 0.0, None

            if size_mb > PREVIEW_LARGE_FILE_MB:
                if not messagebox.askyesno(
                        "确认", f"{selected} 约 {size_mb / 1024:.1f} GB。\n"
                        f"预览需要把整个文件载入内存，可能长时间无响应甚至失败。\n"
                        f"仍要载入吗？（批量处理不受此限制，按切片流式读取）"):
                    return

            stack = load_tiff_stack(file_path)
            self.current_stack = stack
            self.current_filename = selected
            # 切片数优先用 TIFF 元数据口径（单页彩色/单页 3D 等场景准确），
            # 失败时回退到数组形状启发式
            self.total_slices = meta_slices if meta_slices else get_slice_count(stack)
            self.current_slice = 0
            # 数据集级位移在加载时算一次并缓存：此前每次切片导航都全量
            # 扫描堆栈求 max，大堆栈下每次点击都读几十 GB（2026-09-28 P1）
            self._current_uint16_shift = self._compute_stack_shift()
            self._display_current_slice()
            self.logger.log(f"已加载: {selected}, 切片数: {self.total_slices}, 尺寸: {stack.shape}")
        except Exception as e:
            self.logger.error(f"加载文件失败: {e}")
            messagebox.showerror("错误", f"加载文件失败:\n{e}")

    # ==================== 切片导航 ====================

    def _change_slice(self, direction):
        if self.current_stack is None:
            return
        if direction == 0:
            self.current_slice = 0
        elif direction == 'end':
            self.current_slice = self.total_slices - 1
        elif direction == -1:
            self.current_slice = max(0, self.current_slice - 1)
        elif direction == 1:
            self.current_slice = min(self.total_slices - 1, self.current_slice + 1)
        self._display_current_slice()

    def _display_current_slice(self):
        if self.current_stack is None:
            return
        try:
            img = get_slice(self.current_stack, self.current_slice)
            self.slice_label.config(text=f"切片: {self.current_slice + 1}/{self.total_slices}")
            title = f"{self.current_filename} - 切片 {self.current_slice + 1}/{self.total_slices}"
            # 切片导航保留当前缩放/平移，便于逐帧对比
            self.canvas_viewer.display_image(
                img, title, uint16_shift=self._stack_uint16_shift(), keep_view=True)
        except Exception as e:
            self.logger.error(f"显示切片失败: {e}")

    def _compute_stack_shift(self):
        """对当前堆栈计算数据集级 uint16 移位（加载时调用一次）"""
        stack = self.current_stack
        if stack is None or stack.dtype != np.uint16:
            return None
        return resolve_uint16_shift(float(stack.max()))

    def _stack_uint16_shift(self):
        """预览用：返回缓存的移位数（不再每次全量扫描堆栈）"""
        return self._current_uint16_shift

    # ==================== 分割与分析 ====================

    def _get_current_image(self):
        """获取当前切片图像"""
        if self.current_stack is None:
            return None
        return get_slice(self.current_stack, self.current_slice)

    def _get_safe_pixel_size(self):
        """读取并校验像素尺寸；非法时弹窗提示并返回 None"""
        try:
            val = float(self.pixel_size.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("错误", "像素大小必须是有效数字")
            return None
        if val <= 0:
            messagebox.showerror("错误", "像素大小必须为正数 (nm/px)")
            return None
        if not self.pixel_calibrated.get() and val != DEFAULT_PIXEL_SIZE_NM:
            messagebox.showerror(
                "错误",
                "像素尺寸未标定时只允许使用假定值 1.0 nm/px（面积按 px² 解读）。\n"
                "如需使用实测像素尺寸，请先勾选\"已标定\"。")
            return None
        return val

    def _on_params_changed(self):
        """参数变化回调（仅更新日志，不重建控件）"""
        pass  # 参数面板已自行管理，无需额外操作

    def _segment_and_measure(self, pixel_size):
        """对当前切片执行分割 + 完整测量（预览/分析/验证共用同一条路径）。

        2026-09-28 审查修正：此前三处 handler 各自复制了近乎相同的
        30 行"分割→告警→测量"代码，且"验证计算"漏传 Otsu 阈值，
        口径容易漂移；现统一收敛到本方法。

        Returns:
            (result, cryst_mask, amorph_mask, info)；无图像/分割失败/参数
            非法时返回 None。
        """
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("提示", "请先打开图像文件")
            return None

        try:
            params = self.param_panel.get_params()
            cryst_mask, amorph_mask, info = segment_image(
                img, params, uint16_shift=self._stack_uint16_shift())

            if cryst_mask is None:
                self.logger.error("分割失败")
                return None

            if info.get("invalid_input"):
                self.logger.warn("当前切片含 NaN/Inf，非有限像素已按背景(0)处理")
            if info.get("per_image_normalization"):
                self.logger.warn(
                    f"当前切片为 {img.dtype} 数据，按单图 min/max 归一化，"
                    f"跨切片映射可能不一致（建议转为固定位深后处理）")

            result = compute_full_measurement(
                img, cryst_mask, amorph_mask,
                pixel_size,
                self.current_filename,
                self.current_slice + 1,
                info.get("otsu_threshold"),
                pixel_calibrated=self.pixel_calibrated.get(),
            )
            return result, cryst_mask, amorph_mask, info

        except Exception as e:
            self.logger.error(f"分割/测量失败: {e}\n{traceback.format_exc()}")
            return None

    def _preview_segmentation(self):
        """预览分割效果"""
        pixel_size = self._get_safe_pixel_size()
        if pixel_size is None:
            return

        outcome = self._segment_and_measure(pixel_size)
        if outcome is None:
            return
        result, cryst_mask, amorph_mask, info = outcome

        try:
            # 生成标注图像
            gray = info.get("gray_image")
            if gray is None:
                from core.segmentation import ensure_uint8_gray
                gray = ensure_uint8_gray(self._get_current_image(),
                                         uint16_shift=self._stack_uint16_shift())
            annotated = create_annotated_image(gray, cryst_mask, amorph_mask)
            self.canvas_viewer.display_image(
                annotated, f"分割预览 - {self.current_filename}",
                uint16_shift=self._stack_uint16_shift())

            self._show_result(result)
            self.logger.log(f"预览完成: 晶体比例 {result.crystalline_ratio:.2%}")

        except Exception as e:
            self.logger.error(f"预览分割失败: {e}\n{traceback.format_exc()}")

    def _analyze_current_slice(self):
        """分析当前切片"""
        pixel_size = self._get_safe_pixel_size()
        if pixel_size is None:
            return

        outcome = self._segment_and_measure(pixel_size)
        if outcome is None:
            return
        result = outcome[0]

        self._show_result(result)
        self.logger.log(f"分析完成: 切片 {self.current_slice + 1}")

    def _verify_calculations(self):
        """验证计算结果"""
        pixel_size = self._get_safe_pixel_size()
        if pixel_size is None:
            return

        outcome = self._segment_and_measure(pixel_size)
        if outcome is None:
            return
        result = outcome[0]

        try:
            # 形状因子验证
            text = "═══ 计算验证 ═══\n\n"
            text += f"晶体面积: {result.crystalline_area_nm2:,.2f} nm²\n"
            text += f"晶体周长: {result.crystalline_perimeter_nm:,.2f} nm\n"

            if result.crystalline_perimeter_nm > 0:
                sf = (4 * np.pi * result.crystalline_area_nm2) / (result.crystalline_perimeter_nm ** 2)
                text += f"整体形状因子 (4πA/P²): {sf:.4f}\n\n"
                text += "理论参考:\n"
                text += f"  完美圆形: {SHAPE_FACTOR_CIRCLE:.4f}\n"
                text += f"  正方形:   {SHAPE_FACTOR_SQUARE:.4f}\n\n"

                if sf > VERIFY_SF_NEAR_CIRCLE:
                    text += "→ 形状接近圆形\n"
                elif sf > VERIFY_SF_REGULAR:
                    text += "→ 形状较规则\n"
                else:
                    text += "→ 形状不规则\n"

            text += f"\n区域数量: {result.num_regions}\n"
            text += f"平均形状因子: {result.shape_factor_mean:.4f}\n"

            self._show_text_in_preview(text)
            self.logger.log(f"验证完成: 形状因子 = {result.shape_factor_mean:.4f}")

        except Exception as e:
            self.logger.error(f"验证失败: {e}\n{traceback.format_exc()}")

    def _show_result(self, result):
        """在预览面板显示测量结果"""
        text = "═══ 切片分析结果 ═══\n\n"
        text += f"文件: {result.filename}\n"
        text += f"切片: {result.slice_index}\n"
        text += f"区域数: {result.num_regions}\n"
        if result.otsu_threshold is not None:
            text += f"Otsu阈值: {result.otsu_threshold}\n"
        text += "\n── 面积统计 ──\n"
        text += f"  总像素: {result.total_pixels:,}\n"
        text += f"  晶体像素: {result.crystalline_pixels:,}\n"
        text += f"  非晶像素: {result.amorphous_pixels:,}\n"
        text += f"  晶体比例: {result.crystalline_ratio:.2%}\n"
        text += f"  非晶比例: {result.amorphous_ratio:.2%}\n"
        text += "\n── 尺寸统计 ──\n"
        text += f"  晶体面积: {result.crystalline_area_nm2:,.2f} nm²\n"
        text += f"  非晶面积: {result.amorphous_area_nm2:,.2f} nm²\n"
        text += f"  晶体周长: {result.crystalline_perimeter_nm:,.2f} nm\n"
        text += f"  形状因子: {result.shape_factor_mean:.4f}\n"

        self._show_text_in_preview(text)

    def _show_text_in_preview(self, text: str):
        """更新预览文本框"""
        self.preview_text.config(state='normal')
        self.preview_text.delete('1.0', tk.END)
        self.preview_text.insert('1.0', text)
        self.preview_text.config(state='disabled')

    # ==================== 批量处理 ====================

    def _start_batch(self):
        if self.processing:
            messagebox.showwarning("提示", "处理正在进行中")
            return
        if not self.input_folder or not self.output_folder:
            messagebox.showerror("错误", "请先选择输入和输出文件夹")
            return

        # Tk 变量/控件只能在主线程访问：启动线程前先快照全部参数。
        # 无效输入直接报错（不静默回退默认值），且必须为正数。
        try:
            params = self.param_panel.get_params()
            pixel_size = float(self.pixel_size.get())
            pixel_calibrated = bool(self.pixel_calibrated.get())
            time_interval = float(self.time_interval.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("错误", "像素大小和时间间隔必须是有效数字")
            return

        if pixel_size <= 0:
            messagebox.showerror("错误", "像素大小必须为正数 (nm/px)")
            return
        if time_interval <= 0:
            messagebox.showerror("错误", "时间间隔必须为正数")
            return
        if not pixel_calibrated and pixel_size != DEFAULT_PIXEL_SIZE_NM:
            # 未标定却改过像素尺寸：面积单位无法解释，直接阻断
            messagebox.showerror(
                "错误",
                "像素尺寸未标定时只允许使用假定值 1.0 nm/px（面积按 px² 解读）。\n"
                "如需使用实测像素尺寸，请先勾选\"已标定\"。")
            return

        if not messagebox.askyesno("确认", "开始批量处理所有 TIFF 文件？"):
            return

        os.makedirs(self.output_folder, exist_ok=True)

        # 写权限前置检查（2026-09-28 审查项）：避免跑到最后导出阶段才失败
        try:
            if not os.access(self.output_folder, os.W_OK):
                messagebox.showerror("错误", f"输出文件夹不可写:\n{self.output_folder}")
                return
        except OSError:
            pass  # os.access 在部分文件系统上不可靠，交给实际写入报错

        # 输入/输出目录一并快照（2026-09-28 审查修正：此前 output_folder
        # 在导出时才读取，中途改目录会导致 CSV/Excel 与标注图写到不同位置）
        input_folder = self.input_folder
        output_folder = self.output_folder

        self.processing = True
        self.progress['value'] = 0
        self.progress_label.config(text="准备中...")
        self.btn_start.config(state='disabled')
        self.btn_stop.config(state='normal')
        self._set_batch_widgets_locked(True)

        self._batch_thread = threading.Thread(
            target=self._run_batch,
            args=(input_folder, output_folder, params,
                  pixel_size, time_interval, pixel_calibrated),
            daemon=True,
        )
        self._batch_thread.start()

    def _set_batch_widgets_locked(self, locked: bool):
        """批量处理期间锁定输入/输出目录控件（参数已快照，锁件避免误导）"""
        state = 'disabled' if locked else 'normal'
        for widget in (self.input_entry, self.output_entry,
                       self.btn_input_browse, self.btn_output_browse):
            widget.config(state=state)

    def _stop_batch(self):
        self.processing = False
        self._ui_queue.put(('log', "用户停止处理", 'WARN'))

    def _run_batch(self, input_folder, output_folder, params,
                   pixel_size, time_interval, pixel_calibrated):
        """批量处理主逻辑（在子线程运行，目录/params/pixel_size/time_interval/
        pixel_calibrated 均为启动时快照）。

        子线程只做计算和文件 I/O；所有 Tk 更新通过 _ui_queue 投递，
        由主线程 _poll_ui_queue 轮询执行。
        """
        stopped = False
        try:
            files = scan_tiff_files(input_folder)
            if not files:
                self._ui_queue.put(('error', "未找到 TIFF 文件"))
                return

            # 数据集级 uint16 位深统一：按整个输入目录的最大值确定移位数，
            # 避免逐图判定导致明暗切片映射不一致（回归 2026-09-06 P1）
            uint16_shift, scan_failed = scan_uint16_max_shift(files, input_folder)
            if scan_failed:
                self._ui_queue.put(
                    ('log', f"位深扫描跳过 {len(scan_failed)} 个无法读取的文件: "
                            f"{', '.join(scan_failed[:5])}"
                            + ("..." if len(scan_failed) > 5 else ""), 'WARN'))
            if uint16_shift is not None:
                self._ui_queue.put(
                    ('log', f"uint16 位深统一: 右移 {uint16_shift} 位（按整个输入目录数据量程确定）", 'INFO'))

            # 非 uint8/uint16 输入按单图归一化，跨切片可比性受限：
            # 每次运行只告警一次（逐切片告警会刷屏）
            warned_per_image_norm = False

            # 预读各文件切片数（仅头信息），支持切片粒度进度
            file_slice_counts = {}
            for f in files:
                try:
                    file_slice_counts[f] = int(
                        get_tiff_info(os.path.join(input_folder, f))['num_slices'])
                except Exception:
                    file_slice_counts[f] = None
            known = [c for c in file_slice_counts.values() if c is not None]
            total_slices_all = sum(known) if len(known) == len(files) else None

            all_results = []
            all_region_rows = []
            used_annotated_names = set()
            total_processed = 0
            processed_slices = 0
            total_files = len(files)

            calib_text = "已标定" if pixel_calibrated else "未标定(假定1.0nm/px)"
            self._ui_queue.put(('log', f"开始批量处理 {len(files)} 个文件...", 'INFO'))
            self._ui_queue.put(('log', f"参数: 方法={params.method}, 极性={params.foreground}, "
                                       f"像素={pixel_size}nm/px({calib_text}), 时间间隔={time_interval}", 'INFO'))

            for idx, filename in enumerate(files):
                if not self.processing:
                    stopped = True
                    break

                try:
                    n_slices = file_slice_counts.get(filename)
                    self._ui_queue.put(
                        ('log', f"[{idx + 1}/{total_files}] {filename}"
                                + (f" ({n_slices} 切片)" if n_slices else ""), 'INFO'))

                    file_path = os.path.join(input_folder, filename)
                    stem = Path(filename).stem

                    # 使用 load_tiff_slices 生成器逐页读取，避免全量 imread。
                    for si, img in enumerate(load_tiff_slices(file_path), 1):
                        if not self.processing:
                            stopped = True
                            break

                        cryst_mask, amorph_mask, info = segment_image(
                            img, params, uint16_shift=uint16_shift)
                        if cryst_mask is None:
                            continue
                        if info.get("invalid_input"):
                            self._ui_queue.put(
                                ('log', f"  切片 {si} 含 NaN/Inf，非有限像素已按背景(0)处理", 'WARN'))
                        if info.get("per_image_normalization") and not warned_per_image_norm:
                            warned_per_image_norm = True
                            self._ui_queue.put(
                                ('log', f"输入含非 uint8/uint16 数据（如 {img.dtype}），"
                                        f"按单图 min/max 归一化，跨切片映射可能不一致；"
                                        f"建议转为固定位深后处理（同类告警仅提示一次）", 'WARN'))

                        result = compute_full_measurement(
                            img, cryst_mask, amorph_mask,
                            pixel_size, filename, si,
                            info.get("otsu_threshold"),
                            pixel_calibrated=pixel_calibrated,
                        )
                        all_results.append(result.to_dict())
                        for r in result.regions:
                            row = {COL_FILENAME: filename, COL_SLICE: si}
                            row.update(r)
                            all_region_rows.append(row)
                        total_processed += 1
                        processed_slices += 1

                        # 进度：切片粒度（总切片数未知时回退文件粒度）
                        if total_slices_all:
                            pct = int(processed_slices / total_slices_all * 100)
                        else:
                            pct = int((idx / total_files) * 100)
                        slice_txt = f"{filename} 切片 {si}" + (f"/{n_slices}" if n_slices else "")
                        self._ui_queue.put(('progress', pct, slice_txt))

                        # 保存标注图像（图名用 stem；同运行内同 stem 冲突时追加序号，
                        # 例如输入目录同时含 a.tif 与 a.tiff）
                        gray = info.get("gray_image")
                        if gray is not None:
                            base_name = f"{stem}_slice_{si}"
                            name = base_name
                            k = 2
                            while name in used_annotated_names:
                                name = f"{base_name}_{k}"
                                k += 1
                            used_annotated_names.add(name)
                            annotated = create_annotated_image(gray, cryst_mask, amorph_mask)
                            save_annotated_image(annotated, output_folder, name)

                    if not self.processing:
                        stopped = True
                        break

                    self._ui_queue.put(('log', f"  → {filename} 处理完成", 'INFO'))

                except Exception as e:
                    self._ui_queue.put(
                        ('log', f"  处理 {filename} 失败: {e}\n{traceback.format_exc()}", 'ERROR'))

            # 保存结果
            if all_results:
                df = pd.DataFrame(all_results)
                if stopped:
                    self._ui_queue.put(
                        ('log', f"用户停止：已处理 {total_processed} 个切片，导出部分结果", 'WARN'))
                try:
                    csv_path, excel_path, region_csv_path = export_results(
                        df, output_folder,
                        time_interval=time_interval,
                        pixel_size_nm=pixel_size,
                        seg_params=params,
                        pixel_calibrated=pixel_calibrated,
                        region_rows=all_region_rows,
                    )
                except Exception as e:
                    self._ui_queue.put(('log', f"结果导出失败: {e}", 'ERROR'))
                    self._ui_queue.put(('log', traceback.format_exc(), 'ERROR'))
                    return

                summary = generate_summary_text(df, time_interval)
                self._ui_queue.put(('summary', summary))

                msg = (f"• 切片数: {total_processed}\n"
                       f"• CSV: {os.path.basename(csv_path)}\n"
                       f"• Excel: {os.path.basename(excel_path)}\n"
                       f"• 标注图像: annotated_tif/")
                if region_csv_path:
                    msg += f"\n• 逐区域CSV: {os.path.basename(region_csv_path)} ({len(all_region_rows)} 条)"
                if stopped:
                    self._ui_queue.put(('messagebox', '已停止', f"处理已被用户停止（部分结果已导出）\n\n{msg}"))
                else:
                    self._ui_queue.put(('messagebox', '完成', f"批量处理完成！\n\n{msg}"))
            else:
                self._ui_queue.put(('log', "没有成功处理任何切片", 'WARN'))

        except Exception as e:
            self._ui_queue.put(('log', f"批量处理失败: {e}\n{traceback.format_exc()}", 'ERROR'))
        finally:
            self._ui_queue.put(('finish', stopped))

    def _poll_ui_queue(self):
        """主线程轮询子线程投递的 UI 消息。"""
        try:
            try:
                while True:
                    msg = self._ui_queue.get_nowait()
                    kind = msg[0]

                    if kind == 'progress':
                        _, pct, filename = msg
                        self._update_progress(pct, filename)
                    elif kind == 'log':
                        _, text, level = msg
                        if level == 'ERROR':
                            self.logger.error(text)
                        elif level == 'WARN':
                            self.logger.warn(text)
                        else:
                            self.logger.log(text)
                    elif kind == 'summary':
                        self._show_batch_summary(msg[1])
                    elif kind == 'messagebox':
                        _, title, text = msg
                        messagebox.showinfo(title, text)
                    elif kind == 'error':
                        messagebox.showerror("错误", msg[1])
                    elif kind == 'finish':
                        self._finish_batch(bool(msg[1]) if len(msg) > 1 else False)
            except queue.Empty:
                pass
        except tk.TclError:
            return  # 窗口已销毁，停止轮询
        try:
            self.root.after(100, self._poll_ui_queue)
        except tk.TclError:
            pass

    def _update_progress(self, pct: int, filename: str):
        """更新进度条（主线程调用）"""
        self.progress['value'] = pct
        self.progress_label.config(text=f"{pct}% - {filename}")

    def _finish_batch(self, stopped: bool = False):
        self.processing = False
        self._batch_thread = None
        if stopped:
            # 停止时进度条保持实际进度，不伪造 100%
            self.progress_label.config(text="已停止（部分结果已导出）")
        else:
            self.progress['value'] = 100
            self.progress_label.config(text="完成")
        self.btn_start.config(state='normal')
        self.btn_stop.config(state='disabled')
        self._set_batch_widgets_locked(False)
        self._save_config()
        # 关窗流程等待线程收尾后自行销毁窗口
        if self._closing:
            self._try_destroy_after_batch()

    def _show_batch_summary(self, summary: str):
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert('1.0', summary)

    # ==================== 日志 ====================

    def _clear_log(self):
        self.logger.clear()

    def _save_log(self):
        if not self.output_folder:
            messagebox.showerror("错误", "请先选择输出文件夹")
            return
        try:
            path = self.logger.save(self.output_folder)
            messagebox.showinfo("成功", f"日志已保存:\n{path}")
        except ValueError as e:
            messagebox.showwarning("提示", str(e))
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {e}")

    # ==================== 窗口事件 ====================

    def _on_close(self):
        """窗口关闭：批量处理中先确认，并等待后台线程收尾再销毁窗口。

        2026-09-28 审查修正：此前确认退出后立即 destroy()，daemon 工作线程
        若正在写 Excel/TIFF 会被硬杀，留下半写损坏文件。现改为置停止标志后
        轮询等待线程退出（线程在切片粒度检查标志，等待通常在秒级）。
        """
        if self._closing:
            return  # 已在关窗流程中，忽略重复点击
        if self.processing:
            if not messagebox.askyesno(
                    "确认", "批量处理正在进行，退出将中断处理。\n"
                    "已处理的部分仍会正常导出，程序会等导出完成后关闭。\n"
                    "确定退出吗？"):
                return
            self.processing = False
            self._closing = True
            self.status_var.set("等待后台任务收尾...")
            self._try_destroy_after_batch()
            return
        self._closing = True
        self._save_config()
        self._destroy_window()

    def _try_destroy_after_batch(self):
        """轮询批量线程状态；线程退出后（或超时后）销毁窗口。"""
        thread = self._batch_thread
        if thread is None or not thread.is_alive():
            self._save_config()
            self._destroy_window()
            return
        self.root.after(250, self._try_destroy_after_batch)

    def _destroy_window(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _reset_view(self):
        """重置画布视图"""
        if hasattr(self, 'canvas_viewer'):
            self.canvas_viewer.reset_view()

    def _on_window_resize(self, event):
        """窗口 resize 防抖处理"""
        if event.widget == self.root:
            if self._resize_timer:
                self.root.after_cancel(self._resize_timer)
            self._resize_timer = self.root.after(RESIZE_DEBOUNCE_MS, self._on_resize_done)

    def _on_resize_done(self):
        """防抖结束后刷新画布"""
        self._resize_timer = None
        if hasattr(self, 'canvas_viewer'):
            self.canvas_viewer.reset_view()
