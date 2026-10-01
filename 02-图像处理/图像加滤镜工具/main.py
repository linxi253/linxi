# -*- coding: utf-8 -*-
"""TIF 图像滤镜处理工具 GUI。"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk
from scipy.ndimage import gaussian_filter, zoom

from image_filters import FILTER_PARAMS, process_image, validate_params
from version import APP_TITLE
from tif_io import (
    ProcessingCancelled,
    TifDocument,
    display_levels,
    is_tif_file,
    list_tif_files,
    safe_output_path,
    to_display,
)


PREVIEW_MAX_DIM = 1000
PREVIEW_DELAY_MS = 120
REDRAW_DEBOUNCE_MS = 30
PROGRESS_MIN_INTERVAL_S = 0.1
MEMORY_WARN_BYTES = 4 * 1024 ** 3
ERROR_LOG_NAME = 'TIF_FilterTool-errors.log'
GROUPS = [
    ('模糊与锐化', ['gaussian_blur', 'usm']),
    ('白平衡与色彩', ['white_balance', 'temperature', 'tint']),
    ('影调', ['exposure', 'contrast', 'highlights', 'shadows', 'whites', 'blacks']),
    ('质感', ['texture', 'clarity', 'dehaze']),
    ('饱和度', ['vibrance', 'saturation']),
]
PARAM_INFO = {key: (cn, en, mn, mx, dft)
              for key, cn, en, mn, mx, dft, _ in FILTER_PARAMS}
# 仅对彩色图像有意义的滤镜；灰度文件加载后禁用，避免无效操作。
COLOR_PARAM_KEYS = frozenset(
    {'white_balance', 'temperature', 'tint', 'vibrance', 'saturation'})

# UI 分组必须覆盖全部滤镜且无重复/未知键：漏掉的滤镜在界面上不可见，
# 其参数会被静默忽略。import 期显式失败，杜绝人工同步遗漏。
_GROUP_KEYS = [key for _name, keys in GROUPS for key in keys]
if len(_GROUP_KEYS) != len(set(_GROUP_KEYS)) or set(_GROUP_KEYS) != set(PARAM_INFO):
    raise RuntimeError(
        'GROUPS 与 FILTER_PARAMS 不一致；'
        f'重复: {sorted({k for k in _GROUP_KEYS if _GROUP_KEYS.count(k) > 1})}，'
        f'缺失: {sorted(set(PARAM_INFO) - set(_GROUP_KEYS))}，'
        f'未知: {sorted(set(_GROUP_KEYS) - set(PARAM_INFO))}。'
    )
if not COLOR_PARAM_KEYS <= set(PARAM_INFO):
    raise RuntimeError(
        f'COLOR_PARAM_KEYS 引用了未知滤镜: {sorted(COLOR_PARAM_KEYS - set(PARAM_INFO))}。'
    )
FONT_CN = ('Microsoft YaHei UI', 9)
FONT_CN_BOLD = ('Microsoft YaHei UI', 9, 'bold')
FONT_NUM = ('Consolas', 9)


class ParameterSlider(tk.Frame):
    """标签、滑块和数值输入组成的整数滤镜控件。"""

    def __init__(self, parent, key, cn, en, minimum, maximum, default, on_change, status_var):
        super().__init__(parent)
        self.key = key
        self.cn = cn
        self.en = en
        self.minimum = minimum
        self.maximum = maximum
        self.default = default
        self.on_change = on_change
        self.status_var = status_var
        self._updating = False
        self._tip_text: str | None = None
        self._prev_status = '就绪'

        self.label = tk.Label(self, text=cn, width=9, anchor='w', font=FONT_CN, cursor='hand2')
        self.label.grid(row=0, column=0, sticky='w', padx=(2, 2))
        self.label.bind('<Double-Button-1>', lambda _event: self.reset())
        self.label.bind('<Enter>', self._show_tip)
        self.label.bind('<Leave>', self._hide_tip)

        self.var = tk.DoubleVar(value=default)
        self.slider = ttk.Scale(self, from_=minimum, to=maximum, orient='horizontal',
                                variable=self.var, command=self._on_slider)
        self.slider.grid(row=0, column=1, sticky='ew', padx=4)

        self.entry_var = tk.StringVar(value=str(default))
        self.entry = tk.Entry(self, textvariable=self.entry_var, width=6,
                              justify='center', font=FONT_NUM)
        self.entry.grid(row=0, column=2, padx=(2, 2))
        self.entry.bind('<Return>', self._on_entry)
        self.entry.bind('<FocusOut>', self._on_entry)
        self.columnconfigure(1, weight=1)

    def _show_tip(self, _event=None):
        self._tip_text = f'{self.cn} ({self.en})    双击标签可重置'
        self._prev_status = self.status_var.get()
        self.status_var.set(self._tip_text)

    def _hide_tip(self, _event=None):
        if self._tip_text is not None and self.status_var.get() == self._tip_text:
            self.status_var.set(self._prev_status)
        self._tip_text = None

    def _on_slider(self, _value=None):
        if self._updating:
            return
        value = int(round(self.var.get()))
        self._updating = True
        self.entry_var.set(str(value))
        self._updating = False
        self.on_change(self.key, value)

    def _on_entry(self, _event=None):
        try:
            value = int(float(self.entry_var.get()))
        except (TypeError, ValueError, OverflowError):
            self.entry_var.set(str(self.get_value()))
            return
        value = max(self.minimum, min(self.maximum, value))
        self.set_value(value, notify=True)

    def get_value(self) -> int:
        return int(round(self.var.get()))

    def set_value(self, value: int, notify: bool = False):
        value = max(self.minimum, min(self.maximum, int(value)))
        self._updating = True
        self.var.set(value)
        self.entry_var.set(str(value))
        self._updating = False
        if notify:
            self.on_change(self.key, value)

    def set_enabled(self, enabled: bool):
        state = 'normal' if enabled else 'disabled'
        self.slider.configure(state=state)
        self.entry.configure(state=state)

    def reset(self):
        self.set_value(self.default, notify=True)


class FilterApp:
    """具备单任务保护、取消和线程安全 UI 通讯的滤镜应用。"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry('1400x900')
        self.root.minsize(1100, 700)
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

        self.file_list: list[str] = []
        self.file_index = 0
        self.doc: TifDocument | None = None
        self.frame_index = 0
        self.output_dir: str | None = None
        self.sliders: dict[str, ParameterSlider] = {}
        self._toolbar_buttons: list[ttk.Button] = []
        self._busy = False
        self._closing = False
        self._cancel_event: threading.Event | None = None
        self._task_thread: threading.Thread | None = None
        self._events: queue.Queue[tuple] = queue.Queue()
        self._poll_after_id: str | None = None

        self.preview_frame: np.ndarray | None = None
        self.preview_levels: tuple[float, float] = (0.0, 1.0)
        self._original_display: np.ndarray | None = None
        self._processed_display: np.ndarray | None = None
        self._preview_job: str | None = None
        self._preview_generation = 0
        self._preview_busy = False
        self._preview_pending = False
        self._close_scheduled = False
        self._completed_frames = 0
        self._placeholders: dict[tk.Canvas, str] = {}
        self._frame_generation = 0
        self._color_supported = True
        self._redraw_after: dict[tk.Canvas, str | None] = {}
        # True：按原图百分位拉伸显示（适合观察纹理）；False：固定 [0,1]，
        # 与导出数值范围一致（适合判断饱和/裁剪）。
        self._display_auto = True

        self._build_ui()
        self._bind_mousewheel()
        self._bind_shortcuts()
        self._poll_after_id = self.root.after(50, self._poll_events)

    # ----- UI -------------------------------------------------

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass
        self._build_toolbar()
        self._build_main_area()
        self._build_statusbar()

    def _add_toolbar_button(self, parent, text, width, command):
        button = ttk.Button(parent, text=text, width=width, command=command)
        button.pack(side='left', padx=2)
        self._toolbar_buttons.append(button)
        return button

    def _build_toolbar(self):
        bar = ttk.Frame(self.root, padding=6)
        bar.pack(side='top', fill='x')
        self._add_toolbar_button(bar, '打开文件（多选）', 15, self.open_file)
        self._add_toolbar_button(bar, '打开文件夹', 10, self.open_folder)
        self._add_toolbar_button(bar, '输出文件夹', 10, self.choose_output)
        ttk.Separator(bar, orient='vertical').pack(side='left', fill='y', padx=8, pady=2)
        self.btn_save = self._add_toolbar_button(bar, '处理并保存当前文件', 23, self.save_current)
        self._add_toolbar_button(bar, '批量处理文件列表', 18, self.batch_process)
        ttk.Separator(bar, orient='vertical').pack(side='left', fill='y', padx=8, pady=2)
        self._add_toolbar_button(bar, '重置所有参数', 12, self.reset_all)
        self._add_toolbar_button(bar, '载入参数清单', 12, self.load_params)
        self._display_toggle = self._add_toolbar_button(
            bar, '显示范围：自动', 13, self._toggle_display_mode)
        info = ttk.Frame(self.root, padding=(8, 0, 8, 6))
        info.pack(side='top', fill='x')
        self.scope_var = tk.StringVar(value='选择一个或多个 TIFF；堆栈按相同参数逐帧处理，保存为完整 TIFF 堆栈。')
        ttk.Label(info, textvariable=self.scope_var, font=FONT_CN).pack(anchor='w')
        self.output_label = ttk.Label(info, text='输出目录：未设置', font=FONT_CN, foreground='#666')
        self.output_label.pack(anchor='w')

    def _build_main_area(self):
        main = ttk.PanedWindow(self.root, orient='horizontal')
        main.pack(fill='both', expand=True, padx=6, pady=(0, 4))
        left = ttk.Frame(main)
        main.add(left, weight=3)

        original = ttk.LabelFrame(left, text=' 原始图像 ', padding=4)
        original.pack(fill='both', expand=True, pady=(0, 3))
        self.canvas_orig = tk.Canvas(original, bg='#1e1e1e', highlightthickness=0)
        self.canvas_orig.pack(fill='both', expand=True)
        self.canvas_orig.bind(
            '<Configure>', lambda _e: self._on_canvas_configure(self.canvas_orig, self._redraw_original))

        preview = ttk.LabelFrame(left, text=' 预览效果 ', padding=4)
        preview.pack(fill='both', expand=True, pady=(3, 0))
        self.canvas_prev = tk.Canvas(preview, bg='#1e1e1e', highlightthickness=0)
        self.canvas_prev.pack(fill='both', expand=True)
        self.canvas_prev.bind(
            '<Configure>', lambda _e: self._on_canvas_configure(self.canvas_prev, self._redraw_preview))
        self._draw_placeholder(self.canvas_orig, '请打开 TIF 文件\n支持灰度、RGB、RGBA 与堆栈图像')
        self._draw_placeholder(self.canvas_prev, '预览效果将显示在这里')

        right = ttk.Frame(main, width=360)
        main.add(right, weight=1)
        self._build_control_panel(right)

    def _build_control_panel(self, parent):
        header = ttk.Frame(parent)
        header.pack(fill='x', padx=4, pady=(4, 2))
        ttk.Label(header, text='滤镜参数调节', font=FONT_CN_BOLD).pack(side='left')
        ttk.Label(header, text='(滑块左减右增)', font=FONT_CN, foreground='#888').pack(side='left', padx=6)

        container = ttk.Frame(parent)
        container.pack(fill='both', expand=True, padx=2, pady=2)
        self.panel_canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient='vertical', command=self.panel_canvas.yview)
        self.panel_inner = ttk.Frame(self.panel_canvas)
        self.panel_inner.bind('<Configure>', lambda _event: self.panel_canvas.configure(
            scrollregion=self.panel_canvas.bbox('all')))
        self.panel_canvas.create_window((0, 0), window=self.panel_inner, anchor='nw')
        self.panel_canvas.configure(yscrollcommand=scrollbar.set)
        self.panel_canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        self.tip_var = tk.StringVar(value='')
        for group_name, keys in GROUPS:
            self._add_group_header(group_name)
            for key in keys:
                cn, en, minimum, maximum, default = PARAM_INFO[key]
                slider = ParameterSlider(self.panel_inner, key, cn, en, minimum, maximum,
                                         default, self._on_param_change, self.tip_var)
                slider.pack(fill='x', padx=4, pady=2)
                self.sliders[key] = slider
        ttk.Label(self.panel_inner, text='双击参数名可重置该项；←/→ 切换帧，PgUp/PgDn 切换文件',
                  font=FONT_CN, foreground='#888').pack(fill='x', padx=6, pady=(8, 4))

    def _add_group_header(self, name):
        frame = ttk.Frame(self.panel_inner)
        frame.pack(fill='x', padx=4, pady=(10, 4))
        ttk.Separator(frame, orient='horizontal').pack(side='left', fill='x', expand=True, padx=(0, 6))
        ttk.Label(frame, text=name, font=FONT_CN_BOLD, foreground='#4472c4').pack(side='left')
        ttk.Separator(frame, orient='horizontal').pack(side='left', fill='x', expand=True, padx=(6, 0))

    def _bind_mousewheel(self):
        """全局绑定滚轮。

        不能用 <Enter>/<Leave> 按需绑定：滑块是 panel_canvas 内嵌窗口的
        子控件，指针进入子窗口会向父窗口发 <Leave>（NotifyInferior），
        导致参数面板大部分面积滚轮失效。应用只有一个可滚动区域，全局
        绑定行为最简单一致。
        """
        self.root.bind_all('<MouseWheel>', self._on_mousewheel)       # Windows/macOS
        self.root.bind_all('<Button-4>', self._on_scroll_up)          # Linux/X11
        self.root.bind_all('<Button-5>', self._on_scroll_down)        # Linux/X11

    def _on_mousewheel(self, event):
        if event.delta == 0:
            return
        if abs(event.delta) >= 120:
            # Windows：120 的整数倍
            steps = int(round(-event.delta / 120.0))
        else:
            # macOS/高精度触板：±1~5 的小步进
            steps = -1 if event.delta > 0 else 1
        self.panel_canvas.yview_scroll(steps, 'units')

    def _on_scroll_up(self, _event):
        self.panel_canvas.yview_scroll(-1, 'units')

    def _on_scroll_down(self, _event):
        self.panel_canvas.yview_scroll(1, 'units')

    def _on_canvas_configure(self, canvas: tk.Canvas, redraw):
        """Configure 事件防抖：拖动窗口时避免逐像素全图重采样。"""
        after_id = self._redraw_after.get(canvas)
        if after_id is not None:
            try:
                self.root.after_cancel(after_id)
            except tk.TclError:
                pass
        self._redraw_after[canvas] = self.root.after(REDRAW_DEBOUNCE_MS, redraw)

    def _bind_shortcuts(self):
        def navigation(callback):
            def handler(event):
                # 输入框内 ←/→ 用于编辑光标，不切换帧/文件。
                if isinstance(event.widget, (tk.Entry, ttk.Entry)):
                    return None
                callback()
                return None
            return handler

        self.root.bind('<Left>', navigation(self.prev_frame))
        self.root.bind('<Right>', navigation(self.next_frame))
        self.root.bind('<Prior>', navigation(self.prev_file))
        self.root.bind('<Next>', navigation(self.next_file))
        self.root.bind('<Control-o>', lambda _e: self.open_file())
        self.root.bind('<Control-O>', lambda _e: self.open_folder())
        self.root.bind('<Control-s>', lambda _e: self.save_current())
        self.root.bind('<Control-S>', lambda _e: self.save_current())

    def _build_statusbar(self):
        bar = ttk.Frame(self.root, padding=(8, 4))
        bar.pack(side='bottom', fill='x')
        nav = ttk.Frame(bar)
        nav.pack(fill='x')
        self.btn_prev_file = ttk.Button(nav, text='◀ 上一文件', width=10,
                                        command=self.prev_file, state='disabled')
        self.btn_prev_file.pack(side='left', padx=2)
        self.file_label = ttk.Label(nav, text='未加载文件', font=FONT_CN, width=40, anchor='center')
        self.file_label.pack(side='left', padx=4)
        self.btn_next_file = ttk.Button(nav, text='下一文件 ▶', width=10,
                                        command=self.next_file, state='disabled')
        self.btn_next_file.pack(side='left', padx=2)
        ttk.Separator(nav, orient='vertical').pack(side='left', fill='y', padx=8, pady=2)
        self.btn_prev_frame = ttk.Button(nav, text='◀ 上一帧', width=8,
                                         command=self.prev_frame, state='disabled')
        self.btn_prev_frame.pack(side='left', padx=2)
        self.frame_label = ttk.Label(nav, text='', font=FONT_CN, width=14, anchor='center')
        self.frame_label.pack(side='left', padx=2)
        self.btn_next_frame = ttk.Button(nav, text='下一帧 ▶', width=8,
                                         command=self.next_frame, state='disabled')
        self.btn_next_frame.pack(side='left', padx=2)
        self.frame_number = tk.StringVar(value='1')
        self.frame_entry = ttk.Spinbox(nav, from_=1, to=1, width=6,
                                      textvariable=self.frame_number, state='disabled')
        self.frame_entry.pack(side='left', padx=(10, 2))
        self.frame_entry.bind('<Return>', self.jump_to_frame)
        self.btn_jump_frame = ttk.Button(nav, text='跳转帧', width=8,
                                        command=self.jump_to_frame, state='disabled')
        self.btn_jump_frame.pack(side='left', padx=2)
        task = ttk.Frame(bar)
        task.pack(fill='x', pady=(6, 0))
        self.btn_cancel = ttk.Button(task, text='取消处理', width=10,
                                     command=self.cancel_processing, state='disabled')
        self.btn_cancel.pack(side='right', padx=2)
        self.progress = ttk.Progressbar(task, mode='determinate', length=200)
        self.progress.pack(side='right', padx=6)
        self.status_var = tk.StringVar(value='就绪')
        ttk.Label(task, textvariable=self.status_var, font=FONT_CN, foreground='#555').pack(side='left')
        ttk.Label(bar, textvariable=self.tip_var, font=FONT_CN, foreground='#888').pack(anchor='w')

    # ----- Preview --------------------------------------------

    def get_params(self) -> dict[str, int]:
        return {key: slider.get_value() for key, slider in self.sliders.items()}

    def _on_param_change(self, _key, _value):
        if not self._busy:
            self._schedule_preview()

    def _schedule_preview(self):
        self._preview_generation += 1
        if self._preview_job is not None:
            self.root.after_cancel(self._preview_job)
        self._preview_job = self.root.after(PREVIEW_DELAY_MS, self._start_preview)

    def _current_levels(self) -> tuple[float, float]:
        """当前显示映射的黑白场；自动模式用原图百分位，固定模式用 [0,1]。"""
        return self.preview_levels if self._display_auto else (0.0, 1.0)

    def _toggle_display_mode(self):
        self._display_auto = not self._display_auto
        self._display_toggle.configure(
            text='显示范围：自动' if self._display_auto else '显示范围：[0,1]')
        if self.preview_frame is not None:
            self._original_display = to_display(self.preview_frame, self._current_levels())
            self._redraw_original()
            self._schedule_preview()
        self.status_var.set(
            '显示范围：自动（0.5–99.5 百分位拉伸）' if self._display_auto
            else '显示范围：固定 [0,1]（与导出数值范围一致）')

    def _start_preview(self):
        self._preview_job = None
        if self.preview_frame is None or self._busy:
            return
        if self._preview_busy:
            self._preview_pending = True
            return
        generation = self._preview_generation
        frame = self.preview_frame.copy()
        levels = self._current_levels()
        params = self.get_params()
        self._preview_busy = True

        def worker():
            try:
                display = to_display(process_image(frame, params), levels)
                self._events.put(('preview', generation, display, None))
            except Exception as exc:  # 防止 Tk 回调因预览异常而静默失效
                self._events.put(('preview', generation, None, str(exc)))

        threading.Thread(target=worker, daemon=True, name='tif-preview').start()

    def _prepare_preview_frame(self):
        """在后台线程加载并降采样当前帧；结果经事件队列回主线程。

        整帧解码、float 转换和抗混叠降采样对超大帧可达数秒，绝不能
        在 UI 线程执行。加载期间保留旧图，避免画面闪烁；用户切换帧/
        文件会推进 _frame_generation，过期结果直接丢弃。
        """
        if self.doc is None:
            return
        self._frame_generation += 1
        generation = self._frame_generation
        doc, frame_index, auto = self.doc, self.frame_index, self._display_auto
        self.status_var.set(f'正在加载帧 {frame_index + 1}/{doc.num_frames}…')

        def worker():
            try:
                frame = _downscale(doc.get_frame_float(frame_index), PREVIEW_MAX_DIM)
                levels = display_levels(frame)
                original = to_display(frame, levels if auto else (0.0, 1.0))
                self._events.put(('frame_loaded', generation, frame, levels, original, None))
            except Exception as exc:
                self._events.put(('frame_loaded', generation, None, None, None, str(exc)))

        threading.Thread(target=worker, daemon=True, name='tif-frame-loader').start()

    def _on_frame_loaded(self, generation, frame, levels, original, error):
        if generation != self._frame_generation:
            return  # 用户已切换帧或文件，丢弃过期结果
        if error is not None:
            self.preview_frame = None
            self._original_display = None
            self._processed_display = None
            self.canvas_orig.image_ref = None
            self.canvas_prev.image_ref = None
            self._draw_placeholder(self.canvas_orig, '当前帧无法读取')
            self._draw_placeholder(self.canvas_prev, '请尝试其他帧或文件')
            self.status_var.set(
                f'帧读取失败：{os.path.basename(self.doc.path) if self.doc else ""}')
            messagebox.showerror('帧读取失败', error)
            return
        self.preview_frame = frame
        self.preview_levels = levels
        self._original_display = original
        self._processed_display = original
        self._redraw_original()
        self._redraw_preview()
        self._schedule_preview()
        self.status_var.set(
            f'帧 {self.frame_index + 1}/{self.doc.num_frames if self.doc else 1} '
            f'已加载；调整参数实时预览。')

    def _redraw_original(self):
        if self._original_display is not None:
            self._show_on_canvas(self.canvas_orig, self._original_display)
        else:
            self._draw_placeholder(self.canvas_orig,
                                   self._placeholders.get(self.canvas_orig, '请打开 TIF 文件'))

    def _redraw_preview(self):
        if self._processed_display is not None:
            self._show_on_canvas(self.canvas_prev, self._processed_display)
        else:
            self._draw_placeholder(self.canvas_prev,
                                   self._placeholders.get(self.canvas_prev, '预览效果将显示在这里'))

    def reset_all(self):
        for slider in self.sliders.values():
            slider.set_value(slider.default)
        self.status_var.set('所有参数已重置')
        self._schedule_preview()

    # ----- File navigation ------------------------------------

    def open_file(self):
        if self._busy:
            return
        paths = filedialog.askopenfilenames(title='选择 TIFF 图像或堆栈（可多选）',
                                           filetypes=[('TIFF 图像 / 堆栈', '*.tif *.tiff')])
        if not paths:
            return
        if any(not is_tif_file(path) for path in paths):
            messagebox.showwarning('提示', '请选择 .tif 或 .tiff 文件。')
            return
        self.file_list = list(dict.fromkeys(os.path.abspath(path) for path in paths))
        self.file_index = 0
        self._load_current()

    def open_folder(self):
        if self._busy:
            return
        folder = filedialog.askdirectory(title='选择包含 TIFF 图像的文件夹')
        if not folder:
            return
        try:
            files = list_tif_files(folder)
        except OSError as exc:
            messagebox.showerror('打开文件夹失败', f'无法读取该文件夹：\n{folder}\n\n{exc}')
            return
        if not files:
            messagebox.showwarning('提示', '该文件夹中没有找到 TIFF 文件。')
            return
        self.file_list = files
        self.file_index = 0
        self._load_current(show_error=True)
        if self.doc is not None:
            self.status_var.set(f'已加载 {len(files)} 个 TIFF 文件')

    def choose_output(self):
        folder = filedialog.askdirectory(title='选择独立的输出文件夹')
        if not folder:
            return
        folder = os.path.abspath(folder)
        # 选择期就探测写权限：网络共享/只读盘现在报错，而不是等批处理
        # 逐文件失败。
        probe = os.path.join(folder, f'.tif_filtertool_write_probe')
        try:
            with open(probe, 'w', encoding='utf-8'):
                pass
            os.remove(probe)
        except OSError as exc:
            messagebox.showerror(
                '输出文件夹不可写',
                f'所选目录无法写入文件：\n{folder}\n\n{exc}\n请另选输出文件夹。')
            return
        self.output_dir = folder
        self.output_label.configure(text=f'输出目录：{self.output_dir}')

    def _clear_loaded_image(self):
        # 作废在途的帧加载：doc 已置 None，迟到的 frame_loaded 不得再回填。
        self._frame_generation += 1
        self.doc = None
        self.preview_frame = None
        self._original_display = None
        self._processed_display = None
        self.canvas_orig.image_ref = None
        self.canvas_prev.image_ref = None
        self._draw_placeholder(self.canvas_orig, '当前 TIFF 无法安全读取')
        self._draw_placeholder(self.canvas_prev, '请选择其他文件或查看错误提示')

    def _load_current(self, show_error: bool = True):
        if not self.file_list:
            return
        path = self.file_list[self.file_index]
        try:
            self.doc = TifDocument(path)
        except Exception as exc:
            self._clear_loaded_image()
            self._update_nav()
            self._update_file_label(error=True)
            self._update_color_sliders()
            self.status_var.set(f'加载失败：{os.path.basename(path)}')
            if show_error:
                messagebox.showerror('加载失败', f'无法安全处理文件：\n{path}\n\n{exc}')
            return
        self.frame_index = 0
        self._update_nav()
        self._update_file_label()
        self._update_color_sliders()
        status = (f'已加载 {len(self.file_list)} 个文件；当前文件共 {self.doc.num_frames} 帧。')
        if not self._color_supported:
            status += ' 灰度图像，色彩类滑块已禁用。'
        peak = self.doc.peak_bytes()
        if peak >= MEMORY_WARN_BYTES:
            status += (f' 注意：该图像单帧处理内存峰值约 {peak / 1024 ** 3:.1f} GB，'
                       '读取帧时请耐心等待。')
        self.status_var.set(status)
        self._prepare_preview_frame()

    def _update_file_label(self, error: bool = False):
        if not self.file_list:
            self.file_label.configure(text='未加载文件')
            return
        name = os.path.basename(self.file_list[self.file_index])
        prefix = f'[{self.file_index + 1}/{len(self.file_list)}] ' if len(self.file_list) > 1 else ''
        if error or self.doc is None:
            self.file_label.configure(text=f'{prefix}{name}  |  加载失败')
            return
        self.file_label.configure(text=f'{prefix}{name}  |  {self.doc.dtype}  |  {self.doc.shape}')

    def _update_nav(self):
        if self._busy:
            return
        multi_file = len(self.file_list) > 1
        self.btn_prev_file.configure(state='normal' if multi_file else 'disabled')
        self.btn_next_file.configure(state='normal' if multi_file else 'disabled')
        stack = self.doc is not None and self.doc.is_stack
        self.btn_prev_frame.configure(state='normal' if stack else 'disabled')
        self.btn_next_frame.configure(state='normal' if stack else 'disabled')
        self.frame_label.configure(text=(f'帧 {self.frame_index + 1}/{self.doc.num_frames}' if stack else ''))
        self.frame_number.set(str(self.frame_index + 1))
        self.frame_entry.configure(to=self.doc.num_frames if self.doc else 1,
                                   state='normal' if stack else 'disabled')
        self.btn_jump_frame.configure(state='normal' if stack else 'disabled')
        self.btn_save.configure(text='处理当前堆栈全部帧' if stack else '处理并保存当前文件')
        if self.doc is not None:
            self.scope_var.set(
                f'文件列表：{len(self.file_list)} 个 TIFF ｜ 当前：{self.doc.num_frames} 帧 ｜ '
                '预览显示单帧；处理时对全部帧应用当前参数，每个输入保存为一个完整 TIFF。')
        else:
            self.scope_var.set(f'文件列表：{len(self.file_list)} 个 TIFF ｜ 请选择可读取的文件预览。')

    def prev_file(self):
        if not self._busy and self.file_list:
            self.file_index = (self.file_index - 1) % len(self.file_list)
            self._load_current()

    def next_file(self):
        if not self._busy and self.file_list:
            self.file_index = (self.file_index + 1) % len(self.file_list)
            self._load_current()

    def prev_frame(self):
        if not self._busy and self.doc is not None and self.doc.is_stack:
            self.frame_index = (self.frame_index - 1) % self.doc.num_frames
            self._on_frame_change()

    def next_frame(self):
        if not self._busy and self.doc is not None and self.doc.is_stack:
            self.frame_index = (self.frame_index + 1) % self.doc.num_frames
            self._on_frame_change()

    def jump_to_frame(self, _event=None):
        if self._busy or self.doc is None:
            return
        try:
            number = int(self.frame_number.get())
            if not 1 <= number <= self.doc.num_frames:
                raise ValueError
        except ValueError:
            self.frame_number.set(str(self.frame_index + 1))
            self.status_var.set(f'请输入 1 到 {self.doc.num_frames} 之间的帧号。')
            return
        self.frame_index = number - 1
        self._on_frame_change()

    def _on_frame_change(self):
        self._prepare_preview_frame()

    # ----- Save/batch task ------------------------------------

    def _ensure_output_dir(self) -> bool:
        if self.output_dir is None:
            self.choose_output()
        if self.output_dir is None:
            return False
        # 目录不存在是正常的（保存时会创建）；无法创建或不可写才拒绝。
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as exc:
            messagebox.showerror(
                '输出目录不可用',
                f'无法创建输出目录：\n{self.output_dir}\n\n{exc}')
            self.output_dir = None
            self.output_label.configure(text='输出目录：未设置')
            return False
        if not os.access(self.output_dir, os.W_OK):
            messagebox.showerror(
                '输出目录不可用',
                f'输出目录没有写入权限：\n{self.output_dir}\n请重新选择。')
            return False
        return True

    def get_output_path(self, src_path: str) -> str:
        if self.output_dir is None:
            raise RuntimeError('尚未选择输出目录。')
        return safe_output_path(self.output_dir, src_path)

    def load_params(self):
        """从参数清单（*.filter.json）载入滤镜参数，补全复现闭环。"""
        if self._busy:
            return
        path = filedialog.askopenfilename(
            title='选择参数清单',
            filetypes=[('参数清单', '*.filter.json'), ('JSON 文件', '*.json')])
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as exc:
            messagebox.showerror('载入失败', f'无法读取参数文件：\n{path}\n\n{exc}')
            return
        if not isinstance(payload, dict) or not isinstance(payload.get('filters'), dict):
            messagebox.showerror('载入失败',
                                 '该文件不是有效的参数清单（缺少 filters 字段）。')
            return
        schema = payload.get('schema_version')
        if schema != 2:
            messagebox.showwarning(
                '版本提示',
                f'清单 schema_version={schema!r} 与当前版本(2)不同，参数仍会尽量套用。')
        try:
            values = validate_params(payload['filters'])
        except ValueError as exc:
            messagebox.showerror('载入失败', f'清单参数无效：\n{exc}')
            return
        for key, slider in self.sliders.items():
            slider.set_value(int(round(values[key])), notify=False)
        self._schedule_preview()
        source = payload.get('source', {})
        origin = source.get('path') if isinstance(source, dict) else None
        self.status_var.set(
            f'已载入参数（来源：{os.path.basename(str(origin)) if origin else os.path.basename(path)}）；'
            '预览已更新。')

    def save_current(self):
        if self._busy:
            return
        if self.doc is None:
            messagebox.showinfo('提示', '请先打开一个可安全处理的 TIFF 文件。')
            return
        if not self._ensure_output_dir():
            return
        self._start_processing([(self.doc.path, self.get_output_path(self.doc.path))], '正在处理并保存')

    def batch_process(self):
        if self._busy:
            return
        if not self.file_list:
            messagebox.showinfo('提示', '请先打开文件或文件夹。')
            return
        if not self._ensure_output_dir():
            return
        tasks = [(path, self.get_output_path(path)) for path in self.file_list]
        if len(tasks) > 1 and not messagebox.askyesno(
            '批量处理', f'将处理文件列表中的 {len(tasks)} 个 TIFF，每个堆栈的全部帧使用当前滤镜参数，'
                       f'每个输入输出一个完整 TIFF 堆栈。\n输出目录：\n{self.output_dir}\n\n'
                       '已有同名输出会自动改名，源文件不会被覆盖。是否继续？'
        ):
            return
        self._start_processing(tasks, '正在批量处理')

    def _start_processing(self, tasks: list[tuple[str, str]], action: str):
        self._set_busy(True)
        self._cancel_event = threading.Event()
        self._completed_frames = 0
        self.progress.configure(value=0, maximum=1)
        self.status_var.set(f'{action}：正在检查 TIFF 结构…')
        params = self.get_params()

        def worker():
            outputs: list[tuple[str, str | None]] = []
            errors: list[tuple[str, str]] = []
            try:
                prepared: list[tuple[TifDocument, str]] = []
                for file_number, (source, output) in enumerate(tasks, 1):
                    if self._cancel_event is not None and self._cancel_event.is_set():
                        return
                    self._events.put(('scan_progress', file_number, len(tasks)))
                    try:
                        prepared.append((TifDocument(source), output))
                    except Exception as exc:
                        errors.append((source, str(exc)))
                total_frames = sum(doc.num_frames for doc, _ in prepared)
                completed_frames = 0
                self._events.put(('task_ready', total_frames, len(prepared)))
                for file_number, (doc, output) in enumerate(prepared, 1):
                    if self._cancel_event is not None and self._cancel_event.is_set():
                        return

                    # 百万帧级堆栈逐帧上报会淹没事件队列；按时间节流，
                    # 每帧最后一档总是发送，保证进度条收尾精确。
                    last_report = [0.0]

                    def progress(current, total, *, offset=completed_frames, name=doc.name,
                                 file_index=file_number, last=last_report):
                        now = time.monotonic()
                        if current != total and now - last[0] < PROGRESS_MIN_INTERVAL_S:
                            return
                        last[0] = now
                        self._events.put(
                            ('progress', offset + current, total_frames, name, current, total,
                             file_index, len(prepared)))

                    try:
                        # Recheck after earlier tasks: different input folders can contain
                        # the same filename, and an output may have appeared since planning.
                        if os.path.exists(output) or os.path.exists(f'{output}.filter.json'):
                            output = safe_output_path(os.path.dirname(output), doc.path)
                        outputs.append(
                            doc.save_processed(output, params, progress, self._cancel_event))
                        self._events.put(('file_saved', doc.num_frames))
                    except ProcessingCancelled:
                        return
                    except Exception as exc:
                        errors.append((doc.path, str(exc)))
                    completed_frames += doc.num_frames
            except Exception as exc:  # 兜底：任何未预期异常都必须结束 busy 状态
                errors.append(('内部错误', str(exc)))
            finally:
                cancelled = self._cancel_event is not None and self._cancel_event.is_set()
                self._events.put(('complete', outputs, errors, cancelled))

        self._task_thread = threading.Thread(target=worker, daemon=True, name='tif-save-worker')
        self._task_thread.start()

    def cancel_processing(self):
        if self._busy and self._cancel_event is not None:
            self._cancel_event.set()
            self.btn_cancel.configure(state='disabled')
            self.status_var.set('正在取消并清理当前堆栈；已完成的文件会保留…')

    def _set_busy(self, busy: bool):
        self._busy = busy
        self.btn_cancel.configure(state='normal' if busy else 'disabled')
        state = 'disabled' if busy else 'normal'
        for button in self._toolbar_buttons:
            button.configure(state=state)
        for key, slider in self.sliders.items():
            slider.set_enabled(not busy and (key not in COLOR_PARAM_KEYS or self._color_supported))
        if busy:
            self.frame_entry.configure(state='disabled')
            for button in (self.btn_prev_file, self.btn_next_file, self.btn_prev_frame,
                           self.btn_next_file, self.btn_jump_frame):
                button.configure(state='disabled')
        else:
            self._update_nav()

    def _update_color_sliders(self):
        """灰度文件没有色彩信息，禁用色彩类滑块避免用户做无效调节。"""
        self._color_supported = self.doc is None or self.doc.channel_count >= 3
        for key in COLOR_PARAM_KEYS:
            if key in self.sliders:
                self.sliders[key].set_enabled(
                    self._color_supported and not self._busy)

    # ----- Event dispatch / lifecycle -------------------------

    def _poll_events(self):
        self._poll_after_id = None
        try:
            while True:
                event = self._events.get_nowait()
                kind = event[0]
                if kind == 'preview':
                    self._preview_busy = False
                    _kind, generation, display, error = event
                    if error and generation == self._preview_generation:
                        self.status_var.set(f'预览失败：{error}')
                    elif display is not None and generation == self._preview_generation:
                        self._processed_display = display
                        self._redraw_preview()
                    if self._preview_pending:
                        self._preview_pending = False
                        self._schedule_preview()
                elif kind == 'frame_loaded':
                    _kind, generation, frame, levels, original, error = event
                    self._on_frame_loaded(generation, frame, levels, original, error)
                elif kind == 'scan_progress':
                    self.status_var.set(
                        f'正在检查 TIFF 结构 {event[1]}/{event[2]}：读取文件头…')
                elif kind == 'task_ready':
                    self.progress.configure(maximum=max(event[1], 1), value=0)
                    self.status_var.set(f'待处理 {event[2]} 个 TIFF，共 {event[1]} 帧。')
                elif kind == 'progress':
                    _, current, total, name, frame, frame_total, file_index, file_total = event
                    self.progress.configure(maximum=max(total, 1), value=current)
                    if self._cancel_event is None or not self._cancel_event.is_set():
                        self.status_var.set(
                            f'文件 {file_index}/{file_total}：{name} ｜ 帧 {frame}/{frame_total} '
                            f'｜ 总进度 {current}/{total}')
                elif kind == 'file_saved':
                    self._completed_frames += event[1]
                elif kind == 'complete':
                    _, outputs, errors, cancelled = event
                    self._finish_task(outputs, errors, cancelled)
        except queue.Empty:
            pass
        # 关闭中仍需轮询到任务的 complete 事件被消费（_finish_task 会调用
        # close() 销毁窗口），否则工作线程的取消完成事件无人接收，窗口
        # 会永久停在"正在取消"状态。
        if self._closing and not self._busy:
            return
        try:
            if self.root.winfo_exists():
                self._poll_after_id = self.root.after(50, self._poll_events)
        except tk.TclError:
            pass

    def _finish_task(self, outputs, errors, cancelled: bool):
        self._set_busy(False)
        self._cancel_event = None
        self._task_thread = None
        self.progress.configure(value=0, maximum=1)
        if self._closing:
            self.close()
            return
        if cancelled:
            self.status_var.set(f'已取消；已完成 {len(outputs)} 个文件。')
            messagebox.showinfo('已取消', f'任务已取消。已安全完成 {len(outputs)} 个文件；未完成文件未写出。')
            return
        if errors:
            details = '\n'.join(f'{os.path.basename(path)}: {error}' for path, error in errors[:10])
            self.status_var.set(f'完成：成功 {len(outputs)}，失败 {len(errors)}。')
            log_note = self._append_error_log(errors)
            log_note_text = f'\n\n完整错误清单已写入：\n{log_note}' if log_note else ''
            messagebox.showwarning(
                '部分文件处理失败',
                f'成功 {len(outputs)} 个，失败 {len(errors)} 个：\n\n{details}'
                + (f'\n…（共 {len(errors)} 项）' if len(errors) > 10 else '') + log_note_text)
            return
        summary = f'已保存 {len(outputs)} 个 TIFF，共 {self._completed_frames} 帧，以及对应参数清单。'
        self.status_var.set(f'处理完成：{summary}')
        if len(outputs) <= 3:
            messagebox.showinfo('完成', summary + '\n\n' + '\n'.join(path for path, _ in outputs))
        else:
            messagebox.showinfo('完成', summary)

    def _append_error_log(self, errors: list[tuple[str, str]]) -> str | None:
        """把本次任务的全部错误追加到输出目录的日志文件；失败不影响主流程。"""
        if not self.output_dir:
            return None
        log_path = os.path.join(self.output_dir, ERROR_LOG_NAME)
        try:
            with open(log_path, 'a', encoding='utf-8') as handle:
                handle.write(f'\n[{datetime.now().isoformat(timespec="seconds")}] '
                             f'本批次失败 {len(errors)} 项：\n')
                for path, error in errors:
                    handle.write(f'{path}: {error}\n')
        except OSError:
            return None
        return log_path

    def _on_close(self):
        if self._closing:
            return
        if not self._busy:
            self.close()
            return
        if messagebox.askyesno('任务仍在进行', '正在处理 TIFF。是否取消任务，并在安全清理后退出？'):
            self._closing = True
            if self._cancel_event is not None:
                self._cancel_event.set()
            self.status_var.set('正在取消任务并清理临时文件…')

    def close(self):
        """取消所有挂起的 Tk 回调并销毁窗口，避免销毁后残留 after 任务。"""
        if self._close_scheduled:
            return
        self._close_scheduled = True
        self._closing = True
        for after_id in (self._preview_job, self._poll_after_id, *self._redraw_after.values()):
            if after_id is not None:
                try:
                    self.root.after_cancel(after_id)
                except tk.TclError:
                    pass
        self._preview_job = None
        self._poll_after_id = None
        self._redraw_after.clear()
        try:
            # 推迟到 idle 阶段再销毁，避免在回调栈内销毁根窗口时产生
            # ttk ThemeChanged 清理噪音（event generate 指向已销毁的应用）。
            self.root.after_idle(self.root.destroy)
        except tk.TclError:
            pass

    # ----- Canvas ---------------------------------------------

    def _show_on_canvas(self, canvas: tk.Canvas, img_uint8: np.ndarray):
        canvas.update_idletasks()
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 4 or height < 4:
            return
        image = Image.fromarray(img_uint8)
        source_width, source_height = image.size
        scale = min(width / source_width, height / source_height)
        target_size = max(1, int(source_width * scale)), max(1, int(source_height * scale))
        if target_size != image.size:
            image = image.resize(target_size, Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        canvas.delete('all')
        canvas.create_image(width // 2, height // 2, image=photo, anchor='center')
        canvas.image_ref = photo

    def _draw_placeholder(self, canvas: tk.Canvas, text: str):
        self._placeholders[canvas] = text
        canvas.delete('all')
        canvas.update_idletasks()
        canvas.create_text(canvas.winfo_width() // 2, canvas.winfo_height() // 2,
                           text=text, fill='#666', font=('Microsoft YaHei UI', 12), justify='center')


def _downscale(frame: np.ndarray, max_dim: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, max_dim / max(height, width))
    if scale >= 1.0:
        return frame.astype(frame.dtype, copy=True)
    # 高倍缩小前按缩放比高斯预模糊：zoom(order=1) 无抗混叠，直接缩小会让
    # 预览出现导出结果中不存在的摩尔纹（如 TEM 的高频条纹）。
    sigma = (1.0 / scale - 1.0) / 2.0
    blurred = gaussian_filter(
        frame, sigma=(sigma, sigma, 0) if frame.ndim == 3 else sigma, mode='reflect')
    factors = (scale, scale, 1) if frame.ndim == 3 else scale
    return zoom(blurred, factors, order=1, prefilter=False).astype(frame.dtype, copy=False)


def main():
    root = tk.Tk()
    FilterApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
