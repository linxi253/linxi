# -*- coding: utf-8 -*-
"""离域效应去除工具 —— Tk 图形界面。

用法::

    python main.py

界面分三块：左侧画布（看图和画 ROI）、右侧参数面板、底部状态栏。
在画布上圈出**晶体真实边界**，工具会向外膨胀 + 羽化，
把晶体外那圈白色条纹虚影（离域条纹）抑制掉。
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from typing import List, Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import matplotlib

matplotlib.use("TkAgg")
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Polygon as MplPolygon, Rectangle  # noqa: E402

import export_io  # noqa: E402
import tif_io  # noqa: E402
from deloc_core import (  # noqa: E402
    DEFAULT_FMAX,
    DEFAULT_FMIN,
    CleanEngine,
    downsample_mean,
    preview_resample,
    spectrum_overview,
    suggest_band,
)
from roi_model import RoiSet, RoiShape  # noqa: E402
from version import APP_TITLE, WINDOW_TITLE  # noqa: E402

#: 画布上长边最多显示多少像素，避免大图拖动卡顿。
#: ROI 始终存原图像素坐标，降采样不影响精度。
MAX_DISPLAY_DIM = 1100

#: 频带输入改动后等多久再重算 FFT（毫秒）
BAND_DEBOUNCE_MS = 250

COLOR_KEEP = "#ff3040"
COLOR_CUT = "#2f7bff"


class DelocApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1480x900")
        self.root.minsize(1080, 680)

        # --- 数据状态 ---
        self._path: Optional[str] = None
        self._frame = 0
        self.img: Optional[np.ndarray] = None
        self.info: Optional[tif_io.ImageInfo] = None
        self.roi = RoiSet()
        self.engine = CleanEngine()
        self.mask: Optional[np.ndarray] = None
        self.result: Optional[np.ndarray] = None
        self.vmin, self.vmax = 0.0, 255.0
        self._step = 1

        # --- 交互状态 ---
        self.tool = tk.StringVar(value="polygon")
        self.subtract = tk.BooleanVar(value=False)
        self.view = tk.StringVar(value="结果")
        self._points: List[Tuple[float, float]] = []
        self._drawing = False
        self._start: Optional[Tuple[float, float]] = None
        self._cursor: Optional[Tuple[float, float]] = None
        self._pan: Optional[Tuple[float, float]] = None
        self._pan_xlim: Tuple[float, float] = (0.0, 1.0)
        self._pan_ylim: Tuple[float, float] = (0.0, 1.0)
        self._image_artist = None
        self._overlay: List = []
        self._band_artists: List = []
        self._band_job = None
        # --- 结果有效性（审查项03：参数失效/过期后禁止导出旧结果） ---
        # _result_valid 只在一次成功的 _recompute 之后为 True；频带无效、
        # 计算异常、载入新图都会置 False。_result_caption 是与结果像素
        # 同一次计算的参数快照，导出说明一律用它而不是实时控件值。
        self._result_valid = False
        self._result_caption = "尚未计算"
        # 审查项06：栅格化后软掩膜的实际保留覆盖率（0~1）
        self._keep_fraction = 0.0

        # --- 参数状态 ---
        self.var_fmin = tk.StringVar(value=f"{DEFAULT_FMIN:.4f}")
        self.var_fmax = tk.StringVar(value=f"{DEFAULT_FMAX:.4f}")
        self.var_feather = tk.DoubleVar(value=8.0)
        self.var_dilate = tk.DoubleVar(value=6.0)
        self.var_strength = tk.DoubleVar(value=1.0)
        self.var_band_feather = tk.DoubleVar(value=2.0)

        self._build_ui()
        self._bind_keys()
        self._set_status("请先打开一张 TIF，然后在画布上圈出晶体真实边界。")

    # ==================================================================
    # 界面搭建
    # ==================================================================
    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=(8, 6))
        top.pack(fill="x")
        ttk.Button(top, text="打开 TIF…", command=self.open_file).pack(side="left")
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(top, text="保存结果", command=self.save_result).pack(side="left")
        ttk.Button(top, text="保存对比图", command=self.save_compare).pack(side="left")
        ttk.Button(top, text="导出掩膜", command=self.save_mask_png).pack(side="left")
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(top, text="保存 ROI", command=self.save_roi).pack(side="left")
        ttk.Button(top, text="载入 ROI", command=self.load_roi).pack(side="left")
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(top, text="适配窗口", command=self.fit_view).pack(side="left")
        ttk.Button(top, text="1:1", command=self.actual_size).pack(side="left")

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=8, pady=(0, 4))

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)

        bar = ttk.Frame(left)
        bar.pack(fill="x", pady=(0, 4))
        ttk.Label(bar, text="ROI:").pack(side="left")
        for text, val in (("多边形", "polygon"), ("手绘", "freehand"),
                          ("矩形", "rect"), ("椭圆", "ellipse")):
            ttk.Radiobutton(bar, text=text, value=val,
                            variable=self.tool).pack(side="left", padx=1)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Checkbutton(bar, text="挖除(减)", variable=self.subtract).pack(side="left")
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(bar, text="撤销", width=5, command=self.undo).pack(side="left", padx=1)
        ttk.Button(bar, text="重做", width=5, command=self.redo).pack(side="left", padx=1)
        ttk.Button(bar, text="清除", width=5, command=self.clear_roi).pack(side="left", padx=1)

        self.fig = Figure(figsize=(7, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        for name, handler in (
            ("button_press_event", self._on_press),
            ("button_release_event", self._on_release),
            ("motion_notify_event", self._on_motion),
            ("scroll_event", self._on_scroll),
        ):
            self.canvas.mpl_connect(name, handler)

        right = ttk.Frame(body, width=360)
        right.pack(side="right", fill="y", padx=(8, 0))
        right.pack_propagate(False)
        self._build_band_panel(right)
        self._build_param_panel(right)
        self._build_view_panel(right)

        self.status = tk.StringVar(value="")
        self.cursor_pos = tk.StringVar(value="")
        footer = ttk.Frame(self.root, padding=(10, 4))
        footer.pack(fill="x", side="bottom")
        # 光标坐标放独立分栏：原先每次鼠标移动都写主状态栏，把
        # 「自动选带/已保存/频带无效」这类真正重要的提示一闪而过地冲掉。
        ttk.Label(footer, textvariable=self.status, anchor="w",
                  relief="sunken", padding=(6, 3)).pack(side="left", fill="x", expand=True)
        ttk.Label(footer, textvariable=self.cursor_pos, anchor="e", width=28,
                  relief="sunken", padding=(6, 3)).pack(side="left")

    def _build_band_panel(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="晶格频带", padding=6)
        box.pack(fill="x", pady=(0, 6))

        self.fft_fig = Figure(figsize=(3.3, 3.7), dpi=100)
        self.fft_ax = self.fft_fig.add_subplot(211)
        self.prof_ax = self.fft_fig.add_subplot(212)
        self.fft_fig.subplots_adjust(left=0.15, right=0.97, top=0.92,
                                     bottom=0.15, hspace=0.50)
        self.fft_ax.set_aspect("equal")
        for ax in (self.fft_ax, self.prof_ax):
            ax.tick_params(labelsize=6)
        self.fft_ax.set_title("点频谱选带", fontsize=8)
        self.prof_ax.set_title("径向功率谱", fontsize=8)
        self.prof_ax.set_xlabel("cyc/px", fontsize=6)
        self.fft_canvas = FigureCanvasTkAgg(self.fft_fig, master=box)
        self.fft_canvas.get_tk_widget().pack(fill="x")
        self.fft_canvas.mpl_connect("button_press_event", self._on_spectrum_click)

        row = ttk.Frame(box)
        row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="下限").pack(side="left")
        e1 = ttk.Entry(row, textvariable=self.var_fmin, width=8)
        e1.pack(side="left", padx=(2, 8))
        ttk.Label(row, text="上限").pack(side="left")
        e2 = ttk.Entry(row, textvariable=self.var_fmax, width=8)
        e2.pack(side="left", padx=(2, 0))
        for entry in (e1, e2):
            entry.bind("<Return>", lambda _e: self.apply_band())
            entry.bind("<FocusOut>", lambda _e: self.apply_band())
        #: 频带输入框。_typing_in_entry() 靠它让全局快捷键让路（也便于测试）。
        self._band_entries = (e1, e2)

        row2 = ttk.Frame(box)
        row2.pack(fill="x", pady=(4, 0))
        ttk.Button(row2, text="自动选带", command=self.auto_band).pack(side="left")
        # 单位是毫周期/像素（v2 起）：与图像尺寸和变换网格无关，
        # 同一数值在任意分辨率的图上给出相同的相对带宽过渡。
        ttk.Label(row2, text="带宽羽化 (mcyc/px)").pack(side="left", padx=(8, 2))
        ttk.Scale(row2, from_=0.5, to=6.0, variable=self.var_band_feather,
                  orient="horizontal", length=80,
                  command=lambda _v: self.schedule_band()).pack(side="left")

    def _build_param_panel(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="掩膜与强度", padding=6)
        box.pack(fill="x", pady=(0, 6))
        self._slider(box, "向外膨胀 (px)", self.var_dilate, 0, 60,
                     self.on_mask_changed, "把保留区往外扩，盖住晶体外那圈虚影带")
        self._slider(box, "边缘羽化 (px)", self.var_feather, 0, 40,
                     self.on_mask_changed, "过渡柔和度；太小会出现硬边接缝")
        self._slider(box, "抑制强度", self.var_strength, 0.0, 1.0,
                     self.on_strength_changed, "1.0 完全移除虚影；0.7 只衰减，更保守")

    def _slider(self, parent, label, var, lo, hi, cmd, tip) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(3, 6))
        head = ttk.Frame(row)
        head.pack(fill="x")
        ttk.Label(head, text=label).pack(side="left")
        ttk.Label(head, textvariable=var, width=7, anchor="e").pack(side="right")
        ttk.Scale(row, from_=lo, to=hi, variable=var, orient="horizontal",
                  command=lambda _v: cmd()).pack(fill="x")
        ttk.Label(row, text=tip, font=("", 7), foreground="#666666").pack(anchor="w")

    def _build_view_panel(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="预览", padding=6)
        box.pack(fill="x")
        for text in ("结果", "原图", "掩膜"):
            ttk.Radiobutton(box, text=text, value=text, variable=self.view,
                            command=self.refresh_display).pack(anchor="w")

    def _typing_in_entry(self) -> bool:
        """焦点是否落在文本输入框里。

        ttk.Entry 的 bindtags 是 widget / class / toplevel / all，而方向键、
        回车、Ctrl+Z 在 Entry 的类绑定里都没有 break，所以顶层绑定的全局快捷键
        会**同时**触发：在「下限/上限」输入框里按 ←/→ 会切换帧并重建 RoiSet
        （用户的 ROI 和撤销历史无提示地丢掉），按 Ctrl+Z 会撤销 ROI 而不是文字。
        这里统一让路。
        """
        try:
            widget = self.root.focus_get()
        except Exception:  # 某些窗口管理器下 focus_get 会抛异常
            return False
        return isinstance(widget, (tk.Entry, ttk.Entry, ttk.Combobox))

    def _shortcut(self, action):
        """把全局快捷键包一层：输入框获得焦点时不抢按键。"""
        def handler(_event=None):
            if self._typing_in_entry():
                return None
            action()
            return None
        return handler

    def _bind_keys(self) -> None:
        self.root.bind("<Return>", self._shortcut(self.finish_shape))
        self.root.bind("<Escape>", self._shortcut(self.cancel_shape))
        self.root.bind("<Control-z>", self._shortcut(self.undo))
        self.root.bind("<Control-y>", self._shortcut(self.redo))
        self.root.bind("<Control-Z>", self._shortcut(self.redo))
        self.root.bind("<Left>", self._shortcut(lambda: self.step_frame(-1)))
        self.root.bind("<Right>", self._shortcut(lambda: self.step_frame(1)))

    # ==================================================================
    # 状态栏 / 提示
    # ==================================================================
    def _set_status(self, text: str) -> None:
        self.status.set(text)

    def _warn(self, text: str) -> None:
        messagebox.showwarning(APP_TITLE, text)

    def _error(self, text: str) -> None:
        messagebox.showerror(APP_TITLE, text)

    # ==================================================================
    # 打开 / 载入
    # ==================================================================
    def open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="打开 TIF",
            filetypes=[("TIFF 图像", "*.tif *.tiff"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            info = tif_io.inspect(path)
        except Exception as exc:
            self._error(f"读取文件头失败:\n{exc}")
            return

        # 审查项07：有未保存 ROI 时打开新文件必须先确认（与翻帧/关闭同款）
        if len(self.roi) and not messagebox.askyesno(
            APP_TITLE,
            "打开新文件会清空当前 ROI 和撤销历史。\n"
            "需要保留的话请先「保存 ROI」，稍后「载入 ROI」。\n\n仍要继续？",
        ):
            return

        frame = 0
        if info.is_stack:
            n = simpledialog.askinteger(
                APP_TITLE,
                f"该文件有 {info.n_pages} 帧。\n请输入要处理的帧号 "
                f"(0 ~ {info.n_pages - 1})：",
                initialvalue=0, minvalue=0, maxvalue=info.n_pages - 1,
            )
            if n is None:
                return          # 取消：连 _path 都不动，界面维持原状
            frame = int(n)

        # 载入失败必须回滚：否则 _path 指向新文件、img 还是旧图，保存时的
        # 防覆盖 guard 与默认输出文件名会张冠李戴。
        rollback = (self._path, self._frame)
        self._path, self._frame = path, frame
        self._load_current(rollback=rollback)

    def step_frame(self, delta: int) -> None:
        if self.img is None or self.info is None or not self.info.is_stack:
            return
        new = min(max(self._frame + delta, 0), self.info.n_pages - 1)
        if new == self._frame:
            return
        if len(self.roi) and not messagebox.askyesno(
            APP_TITLE,
            "切换帧会清空当前 ROI 和撤销历史。\n"
            "需要保留的话请先「保存 ROI」，换帧后再「载入 ROI」。\n\n仍要切换？",
        ):
            return
        # 审查项18：传回滚点。载入失败时 _load_current 会把帧号退回，
        # 否则 _frame 指向未载入的新帧、img 还是旧帧像素，导出张冠李戴。
        rollback = (self._path, self._frame)
        self._frame = new
        self._load_current(rollback=rollback)

    def _load_current(self, rollback: Optional[Tuple[Optional[str], int]] = None) -> bool:
        try:
            img, info = tif_io.load(self._path, self._frame)
        except Exception as exc:
            if rollback is not None:
                self._path, self._frame = rollback
            self._error(f"载入失败:\n{exc}")
            return False

        self.img = img
        self.info = info
        self.roi = RoiSet()
        self.engine = CleanEngine()
        self.engine.band_feather = float(self.var_band_feather.get())
        self.mask = None
        self.result = None
        self._result_valid = False
        self._result_caption = "尚未计算"
        self._keep_fraction = 0.0
        self._points = []
        self._drawing = False
        # 审查项15：必须把旧图像 artist 从 axes 上摘掉再置 None。只置
        # None 的话旧对象仍挂在 axes 里，反复换图会累积（实测 5 次换图
        # 后 axes 里有 5 张叠画的图像，内存与渲染都翻倍）。
        if self._image_artist is not None:
            try:
                self._image_artist.remove()
            except Exception:
                pass
            self._image_artist = None

        lo, hi = np.percentile(img.astype(np.float32), [0.5, 99.5])
        if hi - lo < 1e-6:
            lo, hi = float(img.min()), float(img.max()) + 1.0
        self.vmin, self.vmax = float(lo), float(hi)
        self._step = max(1, int(np.ceil(max(img.shape) / MAX_DISPLAY_DIM)))

        self._update_spectrum()
        self._clear_overlay()
        self.fit_view()
        self._recompute()

        note = f"  |  第 {self._frame}/{info.n_pages - 1} 帧（← → 切换）" if info.is_stack else ""
        self._set_status(
            f"{os.path.basename(self._path)}  |  {info.summary()}{note}"
            f"  |  请圈出晶体真实边界"
        )
        return True

    # ==================================================================
    # 频谱面板
    # ==================================================================
    def _update_spectrum(self) -> None:
        """刷新频谱面板。

        频谱**必须在全分辨率图上算**：隔点降采样会把频率轴压缩 step 倍并
        引入混叠，而这里显示的横纵坐标（±0.5 = Nyquist）和用户点出来的
        fmin/fmax 都必须是原图像素坐标系的 cyc/px。显示用的降采样由
        deloc_core.spectrum_overview() 在谱图数组上完成。
        """
        if self.img is None:
            return
        try:
            spec, freq, prof = spectrum_overview(self.img)
        except Exception as exc:
            self._error(f"频谱计算失败:\n{exc}")
            return

        self.fft_ax.clear()
        self.fft_ax.set_aspect("equal")
        self.fft_ax.imshow(spec, cmap="inferno", origin="lower",
                           extent=[-0.5, 0.5, -0.5, 0.5])
        self.fft_ax.set_title("点频谱选带", fontsize=8)
        self.fft_ax.tick_params(labelsize=6)

        self.prof_ax.clear()
        n = int(np.searchsorted(freq, 0.42))
        self.prof_ax.semilogy(freq[1:n], np.maximum(prof[1:n], 1e-12),
                              lw=0.9, color="#1f77b4")
        self.prof_ax.set_title("径向功率谱", fontsize=8)
        self.prof_ax.tick_params(labelsize=6)
        self.prof_ax.set_xlabel("cyc/px（原图像素）", fontsize=6)

        self._band_artists = []
        self._draw_band_markers()
        self.fft_canvas.draw_idle()

    def _draw_band_markers(self) -> None:
        for art in self._band_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._band_artists = []
        try:
            fmin, fmax = self.band()
        except ValueError:
            return

        for r in (fmin, fmax):
            c = Circle((0.0, 0.0), r, fill=False, ls="--", lw=1.0, ec="#39ff5a")
            self.fft_ax.add_patch(c)
            self._band_artists.append(c)
            ln = self.prof_ax.axvline(r, color="#39ff5a", ls="--", lw=1.0)
            self._band_artists.append(ln)

    def _on_spectrum_click(self, event) -> None:
        """点频谱图上的晶格峰 -> 自动填 fmin/fmax。

        坐标轴恒为 ±0.5（原图像素坐标系的 Nyquist），所以半径直接就是
        cyc/px，无需任何换算。
        """
        if self.img is None or event.inaxes is not self.fft_ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        r = float(np.hypot(event.xdata, event.ydata))
        if r < 0.004:
            return
        hw = 0.35
        self.var_fmin.set(f"{max(0.004, r * (1 - hw)):.4f}")
        self.var_fmax.set(f"{min(0.5, r * (1 + hw)):.4f}")
        self.apply_band()

    def auto_band(self) -> None:
        if self.img is None:
            self._warn("请先打开一张图。")
            return
        # 全分辨率选带。隔点降采样会把频率轴压缩 step 倍，并把有效带宽砍到
        # 0.5/step，细晶格会混叠成完全不同的频率——这一步不能省。
        self._set_status("正在分析全分辨率频谱…")
        self.root.update_idletasks()
        try:
            fmin, fmax = suggest_band(self.img)
        except Exception as exc:
            self._error(f"自动选带失败:\n{exc}")
            return
        self.var_fmin.set(f"{fmin:.4f}")
        self.var_fmax.set(f"{fmax:.4f}")
        self.apply_band()
        self._set_status(f"自动选带: {fmin:.4f} ~ {fmax:.4f} cyc/px（原图像素）")

    # ==================================================================
    # 参数
    # ==================================================================
    def band(self) -> Tuple[float, float]:
        try:
            fmin = float(self.var_fmin.get())
            fmax = float(self.var_fmax.get())
        except ValueError as exc:
            raise ValueError("频带必须是数字") from exc
        if not (0.0 < fmin < fmax <= 0.5):
            raise ValueError(f"要求 0 < 下限 < 上限 ≤ 0.5，当前 {fmin} ~ {fmax}")
        return fmin, fmax

    def schedule_band(self) -> None:
        """频带改动后延迟重算，避免拖滑块时狂跑 FFT。"""
        if self._band_job is not None:
            try:
                self.root.after_cancel(self._band_job)
            except Exception:
                pass
        self._band_job = self.root.after(BAND_DEBOUNCE_MS, self.apply_band)

    def apply_band(self) -> None:
        if self._band_job is not None:
            try:
                self.root.after_cancel(self._band_job)
            except Exception:
                pass
            self._band_job = None
        if self.img is None:
            return
        try:
            self.band()
        except ValueError as exc:
            # 审查项03：频带无效时旧结果虽在预览里，但必须标记过期、禁止导出
            self._result_valid = False
            self._set_status(f"频带无效: {exc}")
            return
        self.engine.band_feather = float(self.var_band_feather.get())
        self.engine.invalidate()
        self._draw_band_markers()
        self.fft_canvas.draw_idle()
        self._recompute()

    def on_mask_changed(self) -> None:
        if self.img is None or self.mask is None:
            return
        self._recompute()

    def on_strength_changed(self) -> None:
        if self.img is None or self.mask is None:
            return
        self._recompute()

    # ==================================================================
    # ROI 编辑
    # ==================================================================
    def undo(self) -> None:
        if self.roi.undo():
            self._after_roi_edit()

    def redo(self) -> None:
        if self.roi.redo():
            self._after_roi_edit()

    def clear_roi(self) -> None:
        if not len(self.roi):
            return
        self.roi.clear()
        self._after_roi_edit()

    def cancel_shape(self) -> None:
        self._points = []
        self._drawing = False
        self._start = None
        self._cursor = None
        self._redraw_overlay()

    def finish_shape(self) -> None:
        pts = list(self._points)
        self._points = []
        self._drawing = False
        self._start = None
        self._cursor = None
        if self.img is None or len(pts) < 2:
            self._redraw_overlay()
            return
        shape = RoiShape(kind=self.tool.get(), points=pts,
                         subtract=bool(self.subtract.get()))
        if self.roi.add(shape):
            self._after_roi_edit()
        else:
            self._set_status("形状太小，已忽略。")
            self._redraw_overlay()

    def _after_roi_edit(self) -> None:
        self._redraw_overlay()
        self._recompute()
        if not len(self.roi):
            self._set_status("还没有 ROI：此时不做任何抑制。")
        elif not self.roi.has_keep:
            self._set_status("只有挖除区：掩膜全为 0，整幅图的晶格带都会被移除！")
        else:
            # 审查项06：报告实际栅格化出的保留覆盖率，而不是只数形状个数
            note = f"  |  保留覆盖率 {self._keep_fraction:.1%}"
            if self._keep_fraction <= 1e-6:
                note += "（有效保留区为 0，导出会被拦截）"
            self._set_status(self.roi.summary() + note)

    # ==================================================================
    # 画布交互
    # ==================================================================
    def _on_press(self, event) -> None:
        if self.img is None or event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)

        key = (event.key or "").lower()
        if event.button == 2 or (event.button == 1 and key in ("control", "ctrl")):
            self._pan = (x, y)
            self._pan_xlim = self.ax.get_xlim()
            self._pan_ylim = self.ax.get_ylim()
            return
        if event.button != 1:
            return

        tool = self.tool.get()
        if tool == "polygon":
            if event.dblclick:
                self.finish_shape()
                return
            self._points.append((x, y))
            self._redraw_overlay()
        elif tool == "freehand":
            self._drawing = True
            self._points = [(x, y)]
        else:
            self._drawing = True
            self._start = (x, y)
            self._points = [(x, y), (x, y)]

    def _on_motion(self, event) -> None:
        if self.img is None or event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)

        if self._pan is not None:
            dx = x - self._pan[0]
            dy = y - self._pan[1]
            self.ax.set_xlim(self._pan_xlim[0] - dx, self._pan_xlim[1] - dx)
            self.ax.set_ylim(self._pan_ylim[0] - dy, self._pan_ylim[1] - dy)
            self.canvas.draw_idle()
            return

        if self._drawing:
            if self.tool.get() == "freehand":
                self._points.append((x, y))
            elif self._start is not None:
                self._points = [self._start, (x, y)]
            self._redraw_overlay()
        elif self.tool.get() == "polygon" and self._points:
            self._cursor = (x, y)
            self._redraw_overlay()

        h, w = self.img.shape
        self.cursor_pos.set(f"光标 ({int(round(x))}, {int(round(y))})   图像 {w}×{h}")

    def _on_release(self, event) -> None:
        if self._pan is not None:
            self._pan = None
            return
        if not self._drawing:
            return
        self._drawing = False
        if self.tool.get() in ("freehand", "rect", "ellipse"):
            self.finish_shape()

    def _on_scroll(self, event) -> None:
        if self.img is None or event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        def _span(bounds: Tuple[float, float]) -> float:
            # 图像坐标的 Y 轴从大到小（反向），跨度为负。max(跨度, 1e-9)
            # 会把负跨度换成 1e-9，锚点比例爆炸、视图跳到 ±1e12 之外
            # （审查项04）。这里保留符号，只在退化时用带符号的小量。
            value = bounds[1] - bounds[0]
            if abs(value) < 1e-9:
                value = 1e-9 if value >= 0 else -1e-9
            return value

        factor = 1 / 1.25 if event.button == "up" else 1.25
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        wx, wy = _span(xlim) * factor, _span(ylim) * factor
        rx = (event.xdata - xlim[0]) / _span(xlim)
        ry = (event.ydata - ylim[0]) / _span(ylim)
        self.ax.set_xlim(event.xdata - wx * rx, event.xdata + wx * (1 - rx))
        self.ax.set_ylim(event.ydata - wy * ry, event.ydata + wy * (1 - ry))
        self.canvas.draw_idle()

    def fit_view(self) -> None:
        if self.img is None:
            return
        h, w = self.img.shape
        self.ax.set_xlim(-0.5, w - 0.5)
        self.ax.set_ylim(h - 0.5, -0.5)
        self.canvas.draw_idle()

    def actual_size(self) -> None:
        if self.img is None:
            return
        h, w = self.img.shape
        cx, cy = w / 2.0, h / 2.0
        span = MAX_DISPLAY_DIM / 2.0
        self.ax.set_xlim(cx - span, cx + span)
        self.ax.set_ylim(cy + span, cy - span)
        self.canvas.draw_idle()

    # ==================================================================
    # 计算
    # ==================================================================
    def _recompute(self) -> None:
        """重算掩膜与结果。

        ROI 编辑 -> 距离场缓存失效 -> 重算 EDT + 软门限；滑块拖动 ->
        距离场命中缓存 -> 只跑软门限 + 逐像素乘加。频带不变时变换走
        engine 缓存，不重算。

        只有完整成功的一次重算才会把 _result_valid 置回 True，并把参数
        快照进 _result_caption（审查项03：导出说明必须与像素同源）。
        """
        if self.img is None:
            return
        try:
            fmin, fmax = self.band()
        except ValueError:
            self._result_valid = False
            return
        try:
            self.mask = self.roi.rasterize(
                self.img.shape,
                float(self.var_dilate.get()),
                float(self.var_feather.get()),
            )
            self.result = self.engine.render(
                self.img, self.mask, fmin, fmax, float(self.var_strength.get())
            )
        except Exception as exc:
            self._result_valid = False
            self._error(f"处理失败:\n{exc}\n\n{traceback.format_exc(limit=3)}")
            return
        # 审查项06：复用刚栅格化的软掩膜统计有效保留覆盖率，不额外扫描；
        # "加了保留形状但被挖除区完全吃掉 / 整体在图外"在这里现形。
        self._keep_fraction = float(self.mask.mean())
        self._result_caption = self._param_caption()
        self._result_valid = True
        self.refresh_display()

    # ==================================================================
    # 显示
    # ==================================================================
    def _disp(self, arr: np.ndarray) -> np.ndarray:
        """预览降采样（审查项13）。

        FIR 抗混叠重采样（preview_resample）：核为 Kaiser 窗 32·step+1 抽
        头、0.85/step 截止，样本仍落在原索引 k·step 处，预览与 ROI 网格
        严格对齐。均值池化只做 0.5/step 的一阶零点抑制，实测残留混叠；
        裸 ``arr[::step]`` 则完全无抑制——大图（step>=2）上细晶格会折出
        假摩尔纹，用户据此刻的 ROI 会跟着错。只影响显示，不影响结果。
        """
        try:
            return preview_resample(arr, self._step)
        except ValueError:
            # 掩膜等中间数组可能含 NaN（尚未计算的帧）：退回均值池化，
            # 至少保住"无假纹"的下限而不是让预览整帧崩溃。
            return downsample_mean(arr, self._step)

    def refresh_display(self) -> None:
        if self.img is None:
            return
        mode = self.view.get()
        if not len(self.roi):
            # 审查项05：没有 ROI 时掩膜必然全 0、"结果"是删掉晶格的 base，
            # 与状态栏「此时不做任何抑制」相反——一律显示原图，
            # 避免用户在已抑制晶格的图上圈"晶体真实边界"。
            data, vmin, vmax, cmap = self.img, self.vmin, self.vmax, "gray"
        elif mode == "原图":
            data, vmin, vmax, cmap = self.img, self.vmin, self.vmax, "gray"
        elif mode == "掩膜" and self.mask is not None:
            data, vmin, vmax, cmap = self.mask, 0.0, 1.0, "inferno"
        elif self.result is not None:
            data, vmin, vmax, cmap = self.result, self.vmin, self.vmax, "gray"
        else:
            data, vmin, vmax, cmap = self.img, self.vmin, self.vmax, "gray"
        data = self._disp(data)

        h, w = self.img.shape
        extent = (-0.5, w - 0.5, h - 0.5, -0.5)
        if self._image_artist is None:
            self._image_artist = self.ax.imshow(
                data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent,
                interpolation="nearest", aspect="equal",
            )
        else:
            self._image_artist.set_data(data)
            self._image_artist.set_cmap(cmap)
            self._image_artist.set_clim(vmin, vmax)
            self._image_artist.set_extent(extent)
        self.canvas.draw_idle()

    def _clear_overlay(self) -> None:
        for art in self._overlay:
            try:
                art.remove()
            except Exception:
                pass
        self._overlay = []

    def _redraw_overlay(self) -> None:
        self._clear_overlay()
        for shape in self.roi.shapes:
            art = self._shape_artist(shape.points, shape.kind, shape.subtract, False)
            if art is not None:
                self._overlay.append(art)
        if self._points:
            pts = list(self._points)
            if self._cursor is not None and self.tool.get() == "polygon":
                pts.append(self._cursor)
            if len(pts) >= 2:
                art = self._shape_artist(pts, self.tool.get(),
                                         bool(self.subtract.get()), True)
                if art is not None:
                    self._overlay.append(art)
        self.canvas.draw_idle()

    def _shape_artist(self, pts, kind, subtract, pending):
        color = COLOR_CUT if subtract else COLOR_KEEP
        if kind == "rect" and len(pts) >= 2:
            (x0, y0), (x1, y1) = pts[0], pts[-1]
            art = Rectangle((min(x0, x1), min(y0, y1)), abs(x1 - x0), abs(y1 - y0))
        elif kind == "ellipse" and len(pts) >= 2:
            (x0, y0), (x1, y1) = pts[0], pts[-1]
            art = Ellipse(((x0 + x1) / 2.0, (y0 + y1) / 2.0),
                          abs(x1 - x0), abs(y1 - y0))
        else:
            if len(pts) < 3:
                return None
            art = MplPolygon(pts, closed=True)
        art.set_facecolor("none")
        art.set_edgecolor(color)
        art.set_linewidth(1.2)
        art.set_linestyle(":" if pending else ("--" if subtract else "-"))
        self.ax.add_patch(art)
        return art

    # ==================================================================
    # 导出
    # ==================================================================
    def _require_result(self) -> bool:
        if self.img is None or self.result is None:
            self._warn("还没有可保存的结果，请先打开图片并圈出晶体区域。")
            return False
        if not len(self.roi):
            self._warn("还没有圈出晶体区域，结果等于原图。\n请先在画布上画 ROI。")
            return False
        if not self.roi.has_keep:
            self._warn(
                "只画了「挖除区」，没有保留区。\n\n"
                "掩膜会全为 0，整幅图都被当成晶体外，晶格带会被**全部移除**。\n"
                "请再圈出至少一块晶体区域（画的时候不要勾选「挖除(减)」）。"
            )
            return False
        if not self._result_valid:
            # 审查项03：频带无效/参数改动未成功重算时，result 是旧数组
            self._warn(
                "当前显示的是**改动前的旧结果**（参数或 ROI 已变化、尚未成功重算）。\n"
                "请先把频带等参数改回有效值，待预览更新后再导出。"
            )
            return False
        if self._keep_fraction <= 1e-6:
            # 审查项06：形状标志通过但栅格化后有效保留区为 0
            self._warn(
                "保留区栅格化后的有效覆盖率约为 0。\n\n"
                "常见原因：挖除区把保留区完全覆盖，或保留区整体落在图像外。\n"
                "此时掩膜全为 0，整幅图的晶格带都会被移除。"
            )
            return False
        return True

    def save_result(self) -> None:
        if not self._require_result():
            return
        src_dtype = self.info.dtype if self.info is not None else "?"
        as_f32 = messagebox.askyesno(
            APP_TITLE,
            "输出为 float32 TIFF？\n\n"
            "是 = float32（便于后续定量，文件大约 4 倍）\n"
            f"否 = 保持原位深（源数据是 {src_dtype}）",
        )
        # 裁剪提示必须在写盘**之前**：先写再告知，用户只能重来一遍。
        try:
            plan = tif_io.preview_clip(self.result, source=self.info, as_float32=as_f32)
        except ValueError as exc:
            self._error(f"无法按当前位深写出:\n{exc}")
            return
        if plan["clipped_fraction"] > 0.0:
            pct = plan["clipped_fraction"] * 100.0
            if messagebox.askyesno(
                APP_TITLE,
                f"按 {plan['dtype']} 写出会有 {pct:.4f}% 的像素超出量程、被裁剪到端点。\n\n"
                "是 = 改存 float32（不裁剪，推荐）\n"
                "否 = 仍按当前位深保存（超量程像素会被裁剪）",
            ):
                as_f32 = True
                plan = tif_io.preview_clip(self.result, source=self.info, as_float32=True)
        default = tif_io.make_output_path(self._path, "_deloc")
        path = filedialog.asksaveasfilename(
            title="保存结果", defaultextension=".tif",
            initialfile=os.path.basename(default),
            initialdir=os.path.dirname(default),
            filetypes=[("TIFF 图像", "*.tif *.tiff")],
        )
        if not path:
            return
        try:
            report = tif_io.save(path, self.result, source=self.info,
                                 as_float32=as_f32, guard_path=self._path)
        except Exception as exc:
            self._error(f"保存失败:\n{exc}")
            return
        msg = f"已保存:\n{path}\n数据类型: {report['dtype']}"
        if report["dtype_changed"]:
            msg += f"（源数据是 {report['source_dtype']}）"
        if report["clipped_fraction"] > 1e-6:
            msg += f"\n\n注意: 有 {report['clipped_fraction'] * 100:.4f}% 像素被裁剪到量程内"
        messagebox.showinfo(APP_TITLE, msg)
        self._set_status(f"已保存: {path}  ({report['dtype']})")

    def save_mask_png(self) -> None:
        if self.mask is None or not len(self.roi):
            self._warn("还没有掩膜，请先圈出晶体区域。")
            return
        if not self._result_valid:
            self._warn("当前掩膜是改动前的旧结果（尚未成功重算），请先修正参数。")
            return
        default = tif_io.make_output_path(self._path, "_mask", ext=".png")
        path = filedialog.asksaveasfilename(
            title="导出掩膜", defaultextension=".png",
            initialfile=os.path.basename(default),
            initialdir=os.path.dirname(default),
        )
        if not path:
            return
        try:
            from PIL import Image

            # 审查项01：掩膜导出此前没有任何源保护——把输出名指定成源 .tif
            # 会让 Pillow 按扩展名写成 TIFF 覆盖原图。统一走 export_io。
            with export_io.atomic_path(path, self._path) as tmp:
                Image.fromarray(
                    (np.clip(self.mask, 0, 1) * 255).astype(np.uint8)
                ).save(tmp, format="PNG")
        except Exception as exc:
            self._error(f"保存失败:\n{exc}")
            return
        self._set_status(f"掩膜已保存: {path}")

    def save_compare(self) -> None:
        if not self._require_result():
            return
        default = tif_io.make_output_path(self._path, "_compare", ext=".png")
        path = filedialog.asksaveasfilename(
            title="保存对比图", defaultextension=".png",
            initialfile=os.path.basename(default),
            initialdir=os.path.dirname(default),
        )
        if not path:
            return
        try:
            self._write_compare(path)
        except Exception as exc:
            self._error(f"保存失败:\n{exc}")
            return
        self._set_status(f"对比图已保存: {path}")

    def _write_compare(self, path: str) -> None:
        """三栏对比：原图 / 结果 / 掩膜。**同一灰阶**，避免视觉欺骗。

        标题用 _result_caption（与像素同一次计算的参数快照）而不是实时
        控件值——审查项03：否则改完参数不重算就导出，图注会描述错误的
        参数组合。
        """
        import matplotlib.pyplot as plt

        # 审查项01：对比图导出此前无源保护，统一走 export_io 原子提交。
        with export_io.atomic_path(path, self._path) as tmp:
            fig, axes = plt.subplots(1, 3, figsize=(16, 6))
            try:
                panels = [
                    ("原图", self.img, "gray", self.vmin, self.vmax),
                    ("处理结果", self.result, "gray", self.vmin, self.vmax),
                    ("掩膜（亮=保留晶格）", self.mask, "inferno", 0.0, 1.0),
                ]
                for ax, (title, data, cmap, lo, hi) in zip(axes, panels):
                    ax.imshow(data, cmap=cmap, vmin=lo, vmax=hi)
                    ax.set_title(f"{title}   {self._result_caption}", fontsize=10)
                    ax.axis("off")
                fig.tight_layout()
                fig.savefig(tmp, format="PNG", dpi=110)
            finally:
                plt.close(fig)

    def _param_caption(self) -> str:
        try:
            fmin, fmax = self.band()
            band = f"fmin={fmin:.4f} fmax={fmax:.4f}"
        except ValueError:
            band = "频带无效"
        # band_feather 自 v2 起单位是毫周期/像素（0.001 cyc/px），
        # 与图像尺寸无关；旧版按 FFT 数组像素计，非正方形图有方向性响应。
        return (f"{band}  膨胀={self.var_dilate.get():.0f}px  "
                f"羽化={self.var_feather.get():.0f}px  "
                f"强度={self.var_strength.get():.2f}  "
                f"带宽羽化={self.var_band_feather.get():.1f}mcyc/px")

    def save_roi(self) -> None:
        if not len(self.roi):
            self._warn("还没有画任何 ROI。")
            return
        # 审查项03：频带无效时曾经静默写 DEFAULT_FMIN/FMAX 进 JSON——
        # 复现性被破坏且用户毫不知情。无效就拒绝保存，先修参数。
        try:
            fmin, fmax = self.band()
        except ValueError as exc:
            self._warn(f"频带无效，无法保存与当前参数一致的 ROI：\n{exc}\n\n"
                       "请先修正频带再保存。")
            return
        default = tif_io.make_output_path(self._path, "_roi", ext=".json")
        path = filedialog.asksaveasfilename(
            title="保存 ROI", defaultextension=".json",
            initialfile=os.path.basename(default),
            initialdir=os.path.dirname(default),
        )
        if not path:
            return
        try:
            payload = self.roi.to_dict(
                image_shape=list(self.img.shape),
                fmin=fmin, fmax=fmax,
                dilate=float(self.var_dilate.get()),
                feather=float(self.var_feather.get()),
                strength=float(self.var_strength.get()),
                band_feather=float(self.var_band_feather.get()),
            )
            # 审查项01：ROI 导出此前无源保护，把输出名指定成源 .tif 会把
            # 原图写成 JSON。统一走 export_io（原子 + 源保护 + 拒 NaN）。
            export_io.write_json(path, payload, self._path)
        except Exception as exc:
            self._error(f"保存失败:\n{exc}")
            return
        self._set_status(f"ROI 已保存: {path}")

    def load_roi(self) -> None:
        if self.img is None:
            self._warn("请先打开一张图。")
            return
        # 审查项07：载入会直接替换 RoiSet 并丢弃撤销栈，且无法撤销回来。
        # 当前有未保存内容时必须先确认。
        if len(self.roi) and not messagebox.askyesno(
            APP_TITLE,
            "载入 ROI 会替换当前 ROI 和撤销历史（无法撤销恢复）。\n"
            "需要保留的话请先「保存 ROI」。\n\n仍要载入？",
        ):
            return
        path = filedialog.askopenfilename(
            title="载入 ROI", filetypes=[("JSON", "*.json"), ("所有文件", "*.*")]
        )
        if not path:
            return
        try:
            roi, extra = RoiSet.load_json(path)
        except Exception as exc:
            self._error(f"载入失败:\n{exc}")
            return

        notes: List[str] = []
        shape = extra.get("image_shape")
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            try:
                saved = (int(shape[0]), int(shape[1]))
            except (TypeError, ValueError):
                saved = None
            if saved is not None and list(saved) != list(self.img.shape):
                if not messagebox.askyesno(
                    APP_TITLE,
                    f"ROI 是在 {saved[1]}×{saved[0]} 的图上画的，当前图是 "
                    f"{self.img.shape[1]}×{self.img.shape[0]}。\n"
                    f"仍要载入吗？（坐标按原值使用，可能对不上）",
                ):
                    return
        elif shape:
            notes.append(f"image_shape({shape!r})")

        self.roi = roi
        if roi.skipped_on_load:
            notes.append(f"{roi.skipped_on_load} 个损坏形状")
        # ROI 文件里的参数是外部输入，坏值不能让整个载入变成"点了没反应"：
        # 在 console=False 的 exe 里，未捕获异常只会被 Tk 丢进没有 stderr 的虚空。
        for key, var in (("fmin", self.var_fmin), ("fmax", self.var_fmax)):
            if key in extra:
                try:
                    var.set(f"{float(extra[key]):.4f}")
                except (TypeError, ValueError):
                    notes.append(key)
        for key, var in (("dilate", self.var_dilate), ("feather", self.var_feather),
                         ("strength", self.var_strength),
                         ("band_feather", self.var_band_feather)):
            if key in extra:
                try:
                    var.set(float(extra[key]))
                except (TypeError, ValueError):
                    notes.append(key)
        # 载入的参数可能改了带宽羽化：engine 的副本必须跟着同步，
        # 否则下一次 render 还用旧值算变换（缓存键含 band_feather）。
        self.engine.band_feather = float(self.var_band_feather.get())
        self.engine.invalidate()
        self._draw_band_markers()
        self._after_roi_edit()
        tail = f"  |  已忽略无效字段: {', '.join(notes)}" if notes else ""
        self._set_status(f"已载入 ROI: {os.path.basename(path)}  |  {self.roi.summary()}{tail}")

    def on_close(self) -> None:
        if len(self.roi) and not messagebox.askyesno(
            APP_TITLE,
            "有未保存的 ROI，关闭后将丢失（可先「保存 ROI」）。\n仍要退出？",
        ):
            return
        # 撤销尚未触发的防抖任务，避免销毁后回调打到已死的 root 上
        if self._band_job is not None:
            try:
                self.root.after_cancel(self._band_job)
            except Exception:
                pass
            self._band_job = None
        self.root.destroy()


def _error_log_path() -> str:
    """崩溃日志路径：优先放在程序旁边，不可写时退回临时目录。"""
    candidates = []
    try:
        candidates.append(os.path.dirname(os.path.abspath(sys.argv[0])))
    except Exception:
        pass
    candidates.append(tempfile.gettempdir())
    for folder in candidates:
        path = os.path.join(folder, "DelocCleaner_error.log")
        try:
            with open(path, "a", encoding="utf-8"):
                pass
            return path
        except OSError:
            continue
    return os.path.join(tempfile.gettempdir(), "DelocCleaner_error.log")


def _report_callback_exception(exc_type, exc, tb) -> None:
    """Tk 回调里未捕获的异常。

    打包成 console=False 的 exe 后没有 stderr，默认处理只是把 traceback 丢进
    虚空，用户看到的是"点了没反应"。这里改成落盘 + 弹窗。
    """
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    path = _error_log_path()
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    except OSError:
        pass
    try:
        messagebox.showerror(APP_TITLE, f"内部错误:\n{exc}\n\n详细信息已写入:\n{path}")
    except Exception:
        pass


def _enable_dpi_awareness() -> None:
    """声明 DPI 感知，避免 Windows 显示缩放（如 150%）下 Tk 界面整体发虚。

    必须在 tk.Tk() 之前调用。非 Windows 或系统过旧时静默跳过。
    """
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)   # 1 = system DPI aware
    except Exception:
        pass


def main() -> None:
    _enable_dpi_awareness()
    root = tk.Tk()
    root.report_callback_exception = _report_callback_exception
    app = DelocApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
