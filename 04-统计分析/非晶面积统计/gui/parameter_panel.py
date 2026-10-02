# -*- coding: utf-8 -*-
"""
参数面板组件

根据选择的分割方法动态显示对应的参数控件：
- 固定阈值：滑块 + 微调按钮
- Otsu：无需参数（自动计算）
- 自适应阈值：块大小 + C 值
- Canny：高低阈值 + 最小面积
- 分水岭：形态学核大小

修复了原版中参数变化时重建所有控件导致闪烁的问题。
"""

import tkinter as tk
from tkinter import ttk
import ttkbootstrap as ttkb

from constants import (
    SEGMENTATION_METHODS,
    METHOD_NAMES_CN,
    DEFAULT_THRESHOLD,
    DEFAULT_ADAPTIVE_BLOCKSIZE,
    DEFAULT_ADAPTIVE_C,
    DEFAULT_CANNY_LOW,
    DEFAULT_CANNY_HIGH,
    DEFAULT_WATERSHED_KERNEL,
    DEFAULT_CANNY_MIN_AREA,
    FOREGROUND_OPTIONS,
)
from core.segmentation import SegmentationParams


class ParameterPanel(ttk.LabelFrame):
    """分割参数面板，根据方法动态切换控件"""

    def __init__(self, parent, on_params_changed=None):
        """
        Args:
            parent: 父容器
            on_params_changed: 参数变化时的回调函数
        """
        super().__init__(parent, text="分割参数设置", padding=10)

        self._on_params_changed = on_params_changed
        self._current_method = None

        # 参数变量（持久化，切换方法时不丢失）
        self.var_method = tk.StringVar(value="threshold")
        self.var_threshold = tk.IntVar(value=DEFAULT_THRESHOLD)
        self.var_adaptive_blocksize = tk.IntVar(value=DEFAULT_ADAPTIVE_BLOCKSIZE)
        self.var_adaptive_c = tk.IntVar(value=DEFAULT_ADAPTIVE_C)
        self.var_canny_low = tk.IntVar(value=DEFAULT_CANNY_LOW)
        self.var_canny_high = tk.IntVar(value=DEFAULT_CANNY_HIGH)
        self.var_canny_min_area = tk.IntVar(value=DEFAULT_CANNY_MIN_AREA)
        self.var_watershed_kernel = tk.IntVar(value=DEFAULT_WATERSHED_KERNEL)
        # 前景极性：显示文本 ↔ 键值 双向映射
        self.var_foreground = tk.StringVar(value=list(FOREGROUND_OPTIONS.values())[0])

        # 参数控件容器
        self._params_container = ttk.Frame(self)
        self._params_container.pack(fill=tk.X, pady=(5, 0))

        # 方法选择行
        method_row = ttk.Frame(self)
        method_row.pack(fill=tk.X, pady=(0, 5), before=self._params_container)

        ttk.Label(method_row, text="分割方法:").pack(side=tk.LEFT, padx=(0, 5))
        self._method_combo = ttk.Combobox(
            method_row,
            textvariable=self.var_method,
            values=[f"{METHOD_NAMES_CN[m]} ({m})" for m in SEGMENTATION_METHODS],
            state="readonly",
            width=22,
        )
        self._method_combo.pack(side=tk.LEFT, padx=(0, 10))
        self._method_combo.current(0)
        self._method_combo.bind('<<ComboboxSelected>>', self._on_method_changed)

        # 前景极性行（对全部分割方法生效）
        polarity_row = ttk.Frame(self)
        polarity_row.pack(fill=tk.X, pady=(0, 5), before=self._params_container)
        ttk.Label(polarity_row, text="前景极性:").pack(side=tk.LEFT, padx=(0, 5))
        self._polarity_combo = ttk.Combobox(
            polarity_row,
            textvariable=self.var_foreground,
            values=list(FOREGROUND_OPTIONS.values()),
            state="readonly",
            width=28,
        )
        self._polarity_combo.pack(side=tk.LEFT, padx=(0, 10))
        self._polarity_combo.bind('<<ComboboxSelected>>', lambda e: self._notify_change())
        ttk.Label(polarity_row, text="明场衍射衬度下晶体常更暗，可反相",
                  foreground="#888888").pack(side=tk.LEFT)

        # 初始显示
        self._build_params_ui("threshold")

    def _on_method_changed(self, event=None):
        """方法选择变化"""
        idx = self._method_combo.current()
        method = SEGMENTATION_METHODS[idx] if 0 <= idx < len(SEGMENTATION_METHODS) else "threshold"
        self._build_params_ui(method)
        self._notify_change()

    def _build_params_ui(self, method: str):
        """根据方法重建参数控件"""
        # 清除旧控件
        for widget in self._params_container.winfo_children():
            widget.destroy()

        self._current_method = method

        if method == "threshold":
            self._build_threshold_ui()
        elif method == "otsu":
            self._build_otsu_ui()
        elif method == "adaptive":
            self._build_adaptive_ui()
        elif method == "canny":
            self._build_canny_ui()
        elif method == "watershed":
            self._build_watershed_ui()

    def _build_threshold_ui(self):
        """固定阈值参数 UI"""
        frame = ttk.Frame(self._params_container)
        frame.pack(fill=tk.X)

        ttk.Label(frame, text="阈值:").pack(side=tk.LEFT, padx=(0, 5))

        # 滑块
        slider = ttk.Scale(frame, from_=0, to=255, orient=tk.HORIZONTAL,
                           variable=self.var_threshold, command=self._on_threshold_slide)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # 数值显示
        self._threshold_label = ttk.Label(frame, text=f"{self.var_threshold.get()}", width=4)
        self._threshold_label.pack(side=tk.LEFT, padx=5)

        # 微调按钮
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(side=tk.LEFT, padx=5)
        for delta in [-10, -5, +5, +10]:
            ttkb.Button(btn_frame, text=f"{delta:+d}", width=4,
                        command=lambda d=delta: self._adjust_threshold(d),
                        bootstyle="outline").pack(side=tk.LEFT, padx=1)

    def _build_otsu_ui(self):
        """Otsu 自动阈值提示"""
        frame = ttk.Frame(self._params_container)
        frame.pack(fill=tk.X)
        ttk.Label(frame, text="Otsu 方法自动计算最佳阈值，无需手动设置参数。",
                  foreground="#888888").pack(side=tk.LEFT)

    def _build_adaptive_ui(self):
        """自适应阈值参数 UI"""
        frame = ttk.Frame(self._params_container)
        frame.pack(fill=tk.X)

        ttk.Label(frame, text="块大小(奇数):").pack(side=tk.LEFT, padx=(0, 3))
        ttk.Entry(frame, textvariable=self.var_adaptive_blocksize, width=6).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(frame, text="C 值:").pack(side=tk.LEFT, padx=(0, 3))
        ttk.Entry(frame, textvariable=self.var_adaptive_c, width=6).pack(side=tk.LEFT)

    def _build_canny_ui(self):
        """Canny 边缘参数 UI"""
        frame = ttk.Frame(self._params_container)
        frame.pack(fill=tk.X)

        ttk.Label(frame, text="低阈值:").pack(side=tk.LEFT, padx=(0, 3))
        ttk.Entry(frame, textvariable=self.var_canny_low, width=6).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(frame, text="高阈值:").pack(side=tk.LEFT, padx=(0, 3))
        ttk.Entry(frame, textvariable=self.var_canny_high, width=6).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(frame, text="最小面积:").pack(side=tk.LEFT, padx=(0, 3))
        ttk.Entry(frame, textvariable=self.var_canny_min_area, width=6).pack(side=tk.LEFT)

    def _build_watershed_ui(self):
        """分水岭参数 UI"""
        frame = ttk.Frame(self._params_container)
        frame.pack(fill=tk.X)

        ttk.Label(frame, text="形态学核大小:").pack(side=tk.LEFT, padx=(0, 3))
        ttk.Entry(frame, textvariable=self.var_watershed_kernel, width=6).pack(side=tk.LEFT)

    def _on_threshold_slide(self, value):
        """滑块拖动回调"""
        if hasattr(self, '_threshold_label'):
            self._threshold_label.config(text=str(int(float(value))))
        self._notify_change()

    def _adjust_threshold(self, delta: int):
        """微调阈值"""
        new_val = max(0, min(255, self.var_threshold.get() + delta))
        self.var_threshold.set(new_val)
        if hasattr(self, '_threshold_label'):
            self._threshold_label.config(text=str(new_val))
        self._notify_change()

    def _notify_change(self):
        """通知外部参数已变化"""
        if self._on_params_changed:
            self._on_params_changed()

    def apply_config(self, cfg: dict):
        """从持久化配置恢复参数（仅接受合法键值，非法项静默保持默认）"""
        method = cfg.get('method')
        if method in SEGMENTATION_METHODS:
            self._method_combo.current(SEGMENTATION_METHODS.index(method))
            self._build_params_ui(method)

        def _set_int(var, key):
            if key in cfg:
                try:
                    var.set(int(cfg[key]))
                except (tk.TclError, TypeError, ValueError):
                    pass

        _set_int(self.var_threshold, 'threshold')
        _set_int(self.var_adaptive_blocksize, 'adaptive_blocksize')
        _set_int(self.var_adaptive_c, 'adaptive_c')
        _set_int(self.var_canny_low, 'canny_low')
        _set_int(self.var_canny_high, 'canny_high')
        _set_int(self.var_canny_min_area, 'canny_min_area')
        _set_int(self.var_watershed_kernel, 'watershed_kernel')

        fg = cfg.get('foreground')
        if fg in FOREGROUND_OPTIONS:
            self.var_foreground.set(FOREGROUND_OPTIONS[fg])

        if hasattr(self, '_threshold_label'):
            self._threshold_label.config(text=str(self.var_threshold.get()))

    def get_params(self) -> SegmentationParams:
        """
        获取当前所有分割参数

        Returns:
            SegmentationParams 数据对象
        """
        idx = self._method_combo.current()
        method = SEGMENTATION_METHODS[idx] if 0 <= idx < len(SEGMENTATION_METHODS) else "threshold"

        # 前景极性：显示文本反查键值
        foreground = next(
            (key for key, text in FOREGROUND_OPTIONS.items() if text == self.var_foreground.get()),
            "bright",
        )

        # 安全读取 IntVar（用户可能输入了非数字内容）
        def safe_int(var, default):
            try:
                return var.get()
            except (tk.TclError, ValueError):
                var.set(default)
                return default

        return SegmentationParams(
            method=method,
            threshold=safe_int(self.var_threshold, DEFAULT_THRESHOLD),
            adaptive_blocksize=safe_int(self.var_adaptive_blocksize, DEFAULT_ADAPTIVE_BLOCKSIZE),
            adaptive_c=safe_int(self.var_adaptive_c, DEFAULT_ADAPTIVE_C),
            canny_low=safe_int(self.var_canny_low, DEFAULT_CANNY_LOW),
            canny_high=safe_int(self.var_canny_high, DEFAULT_CANNY_HIGH),
            canny_min_area=safe_int(self.var_canny_min_area, DEFAULT_CANNY_MIN_AREA),
            watershed_kernel=safe_int(self.var_watershed_kernel, DEFAULT_WATERSHED_KERNEL),
            foreground=foreground,
        )
