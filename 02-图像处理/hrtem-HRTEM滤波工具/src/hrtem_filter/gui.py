"""Responsive Tk interface for safe HRTEM/STEM processing."""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from ._version import __version__
from .cancel_state import cancel_status_text, reset_cancel_event_for_preview
from .core import FilterResult, HRTEMFilter
from .geometry import Roi, clamp_drag_rect
from .params import FilterParams, OutputEncoding, ParameterError, SaveOptions
from .pipeline import ProcessingCancelled, StackProcessor
from .tiff_io import TiffFrameSource, TiffStackInfo, inspect_tiff


class HRTEMFilterGUI:
    """A GUI that keeps all Tk operations on the main thread."""

    # 单次操作峰值内存估计超过该值（约 4 GiB）时先向用户确认：真实可用内存
    # 因平台而异，这里做保守提示而不是硬限制。
    LARGE_OPERATION_BYTES = 4 * 1024**3

    def __init__(self, root: tk.Tk | None = None) -> None:
        # Windows 研究工作站通常有 YaHei；回退字体保证其他系统可读。放在实例
        # 化时而不是模块导入时，避免污染导入方的全局 matplotlib 配置。
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        self.root = root or tk.Tk()
        self.root.title(f"HRTEM/STEM 滤波工具 v{__version__} — 定量安全版")
        self.root.geometry("1420x900")
        self.root.minsize(1100, 700)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.info: TiffStackInfo | None = None
        self.frame_source: TiffFrameSource | None = None
        self.current_frame = 0
        self.current_image: np.ndarray | None = None
        self.roi: Roi | None = None
        self.preview_result: FilterResult | None = None
        self._preview_stale = False
        self._drag_start: tuple[int, int] | None = None
        self._live_roi_patch: Rectangle | None = None
        self._motion_cid: int | None = None
        self._worker: threading.Thread | None = None
        self._cancel_event: threading.Event | None = None
        self._messages: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._closing = False
        self._close_deadline = 0.0
        self._close_force_prompted = False
        # One shared numerical core: its radial-grid cache then persists
        # across previews of the same image size.
        self._filter = HRTEMFilter()
        self._build_ui()
        # delta=0 在 validated() 中强制为纯 Butterworth 并剥离 STEM；界面同步
        # 禁用相关控件并提示，避免"勾选了却没生效"的静默行为。放在 _build_ui
        # 之后注册，确保回调可以安全访问 status 控件。
        self.delta_var.trace_add("write", self._sync_delta_zero_state)
        # 任何参数变化都让已显示的预览过期；只置标志不重绘，重绘留到下一次
        # _refresh_display，避免每次按键都触发全图 imshow。
        for var in (
            self.mode_var,
            self.stem_var,
            self.apply_bw_var,
            self.rotation_var,
            self.fft_workers_var,
            *self._param_vars.values(),
        ):
            var.trace_add("write", self._mark_preview_stale)
        self.root.after(80, self._poll_messages)

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(fill=tk.X)
        self.open_button = ttk.Button(toolbar, text="打开 TIFF", command=self.open_file)
        self.open_button.pack(side=tk.LEFT, padx=2)
        self.output_button = ttk.Button(toolbar, text="输出位置", command=self.choose_output)
        self.output_button.pack(side=tk.LEFT, padx=2)
        self.preview_button = ttk.Button(toolbar, text="预览", command=self.preview)
        self.preview_button.pack(side=tk.LEFT, padx=8)
        self.save_button = ttk.Button(toolbar, text="处理并保存", command=self.process_and_save)
        self.save_button.pack(side=tk.LEFT, padx=2)
        self.cancel_button = ttk.Button(toolbar, text="取消", command=self.cancel, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, padx=2)

        ttk.Label(toolbar, text="帧:").pack(side=tk.LEFT, padx=(18, 2))
        self.frame_var = tk.IntVar(value=1)
        self.frame_spin = ttk.Spinbox(toolbar, from_=1, to=1, textvariable=self.frame_var, width=6, command=self._change_frame, state=tk.DISABLED)
        self.frame_spin.pack(side=tk.LEFT)
        self.frame_spin.bind("<Return>", lambda _event: self._change_frame())
        self.frame_total = ttk.Label(toolbar, text="/ 0")
        self.frame_total.pack(side=tk.LEFT, padx=2)

        body = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 4))
        controls = ttk.Frame(body, padding=8, width=380)
        body.add(controls, weight=0)
        preview = ttk.Frame(body, padding=3)
        body.add(preview, weight=1)
        self._build_controls(controls)

        self.figure = Figure(figsize=(9, 7), dpi=100, constrained_layout=True)
        self.ax_original = self.figure.add_subplot(2, 2, 1)
        self.ax_filtered = self.figure.add_subplot(2, 2, 2)
        self.ax_fft = self.figure.add_subplot(2, 2, 3)
        self.ax_mask = self.figure.add_subplot(2, 2, 4)
        self.canvas = FigureCanvasTkAgg(self.figure, master=preview)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)

        footer = ttk.Frame(self.root, padding=6)
        footer.pack(fill=tk.X)
        self.status = tk.StringVar(value="就绪 — 打开一个灰度 TIFF 文件")
        ttk.Label(footer, textvariable=self.status).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.progress = ttk.Progressbar(footer, length=280, mode="determinate")
        self.progress.pack(side=tk.RIGHT)

    def _build_controls(self, parent: ttk.Frame) -> None:
        io = ttk.LabelFrame(parent, text="文件与输出", padding=7)
        io.pack(fill=tk.X, pady=3)
        self.file_label = ttk.Label(io, text="尚未加载", wraplength=340)
        self.file_label.pack(anchor=tk.W)
        self.output_path = tk.StringVar()
        ttk.Entry(io, textvariable=self.output_path).pack(fill=tk.X, pady=(5, 0))

        output = ttk.LabelFrame(parent, text="输出", padding=7)
        output.pack(fill=tk.X, pady=3)
        self.mode_var = tk.StringVar(value="wiener")
        self.stem_var = tk.BooleanVar(value=False)
        self.apply_bw_var = tk.BooleanVar(value=True)
        self.encoding_var = tk.StringVar(value=OutputEncoding.FLOAT32.value)
        self.show_fft_var = tk.BooleanVar(value=True)
        self.use_roi_var = tk.BooleanVar(value=False)
        self.mode_combo = self._row_combo(output, "主输出", self.mode_var, ("wiener", "absf", "butterworth"))
        self.stem_check = ttk.Checkbutton(output, text="应用 STEM 十字掩膜", variable=self.stem_var)
        self.stem_check.pack(anchor=tk.W, pady=2)
        self.apply_bw_check = ttk.Checkbutton(output, text="应用 Butterworth 低通包络", variable=self.apply_bw_var)
        self.apply_bw_check.pack(anchor=tk.W, pady=2)
        self._row_combo(output, "保存类型", self.encoding_var, tuple(item.value for item in OutputEncoding))
        ttk.Checkbutton(output, text="显示 FFT 与掩膜", variable=self.show_fft_var).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(output, text="使用框选 ROI", variable=self.use_roi_var, command=self._refresh_display).pack(anchor=tk.W, pady=2)
        ttk.Button(output, text="清除 ROI", command=self.clear_roi).pack(anchor=tk.W, pady=2)
        ttk.Label(output, text="float32 为默认定量输出；uint8-display 仅供展示", foreground="gray", wraplength=330).pack(anchor=tk.W)

        parameters = ttk.LabelFrame(parent, text="滤波参数", padding=7)
        parameters.pack(fill=tk.X, pady=3)
        self.step_var = tk.StringVar(value="2")
        self.delta_var = tk.StringVar(value="5")
        self.cycles_var = tk.StringVar(value="99")
        self.bw_order_var = tk.StringVar(value="4")
        self.bw_ro_var = tk.StringVar(value="0.30")
        self.low_freq_var = tk.StringVar(value="0")
        self.cross_width_var = tk.StringVar(value="4")
        self.cross_hole_var = tk.StringVar(value="4")
        self.cross_ro_var = tk.StringVar(value="0.03")
        self.rotation_var = tk.StringVar(value="fast_radial_bin")
        for label, value in (
            ("Step", self.step_var), ("Delta %", self.delta_var), ("Cycles", self.cycles_var),
            ("BW 阶数", self.bw_order_var), ("BW Ro", self.bw_ro_var), ("低频回补 %", self.low_freq_var),
            ("十字线宽", self.cross_width_var), ("中心孔半径", self.cross_hole_var), ("XH Ro", self.cross_ro_var),
        ):
            self._row_entry(parameters, label, value)
        self._row_combo(parameters, "旋转平均", self.rotation_var, ("fast_radial_bin", "dm_compatible"))
        self.fft_workers_var = tk.StringVar(value="1")
        self._row_entry(parameters, "FFT 线程", self.fft_workers_var)
        # 参数名 → Tk 变量映射，供 _parse_field 生成中文参数错误消息使用。
        self._param_vars: dict[str, tk.StringVar] = {
            "step": self.step_var, "delta": self.delta_var, "cycles": self.cycles_var,
            "bw_order": self.bw_order_var, "bw_ro": self.bw_ro_var,
            "crosshair_width": self.cross_width_var, "crosshair_hole_radius": self.cross_hole_var,
            "crosshair_bw_ro": self.cross_ro_var, "low_freq_percent": self.low_freq_var,
        }

        help_box = ttk.LabelFrame(parent, text="操作提示", padding=7)
        help_box.pack(fill=tk.X, pady=3)
        ttk.Label(
            help_box,
            text="在左上原图中拖拽鼠标框选 ROI。打开文件后不会自动运行 FFT；请检查 ROI、参数和预计内存后再预览。",
            wraplength=330,
        ).pack(anchor=tk.W)

    @staticmethod
    def _row_entry(parent: ttk.Frame, label: str, value: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text=label, width=14).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=value, width=12).pack(side=tk.LEFT)

    @staticmethod
    def _row_combo(parent: ttk.Frame, label: str, value: tk.StringVar, values: tuple[str, ...]) -> ttk.Combobox:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text=label, width=14).pack(side=tk.LEFT)
        combo = ttk.Combobox(row, textvariable=value, values=values, state="readonly", width=18)
        combo.pack(side=tk.LEFT)
        return combo

    def _set_processing(self, active: bool) -> None:
        state = tk.DISABLED if active else tk.NORMAL
        for widget in (self.open_button, self.output_button, self.preview_button, self.save_button):
            widget.configure(state=state)
        frame_state = tk.DISABLED if active or self.info is None or self.info.frame_count <= 1 else tk.NORMAL
        self.frame_spin.configure(state=frame_state)
        self.cancel_button.configure(state=tk.NORMAL if active else tk.DISABLED)

    def _reset_loaded_state(self) -> None:
        self.info = None
        self.frame_source = None
        self.current_frame = 0
        self.current_image = None
        self.roi = None
        self.preview_result = None
        self.frame_var.set(1)
        self.frame_spin.configure(from_=1, to=1, state=tk.DISABLED)
        self.frame_total.configure(text="/ 0")

    def open_file(self) -> None:
        path = filedialog.askopenfilename(title="选择灰度 TIFF", filetypes=[("TIFF", "*.tif *.tiff")])
        if not path:
            return
        self._reset_loaded_state()
        try:
            self.info = inspect_tiff(path)
            self.frame_source = TiffFrameSource(self.info)
            self.file_label.configure(text=f"{self.info.path.name}\n{self.info.frame_shape[1]}×{self.info.frame_shape[0]} px · {self.info.frame_count} 帧 · {self.info.dtype}")
            self.output_path.set(str(self.info.path.with_name(self.info.path.stem + "_filtered.tif")))
            self.frame_spin.configure(from_=1, to=self.info.frame_count, state=tk.NORMAL if self.info.frame_count > 1 else tk.DISABLED)
            self.frame_total.configure(text=f"/ {self.info.frame_count}")
            self._load_current_frame()
            fft_size = 1 << (max(self.info.frame_shape) - 1).bit_length()
            estimate_mib = HRTEMFilter.estimate_peak_bytes(self.info.frame_shape) / 1024**2
            self.status.set(
                f"已加载。全图 FFT 为 {fft_size}×{fft_size}；单输出峰值内存估计约 {estimate_mib:.0f} MiB。建议先框选 ROI 后预览。"
            )
        except Exception as exc:
            self._reset_loaded_state()
            messagebox.showerror("加载失败", f"{type(exc).__name__}: {exc}")

    def _load_current_frame(self) -> None:
        if self.frame_source is None:
            return
        try:
            self.current_image = self.frame_source.read_frame(self.current_frame).astype(np.float32)
            self.preview_result = None
            self._refresh_display()
        except Exception as exc:
            self.current_image = None
            messagebox.showerror("读取失败", f"{type(exc).__name__}: {exc}")

    def _change_frame(self) -> None:
        if self.info is None:
            return
        try:
            frame = int(self.frame_var.get()) - 1
        except (tk.TclError, ValueError):
            return
        if 0 <= frame < self.info.frame_count:
            self.current_frame = frame
            self._preview_stale = True
            self._load_current_frame()

    def choose_output(self) -> None:
        path = filedialog.asksaveasfilename(title="保存处理结果", defaultextension=".tif", filetypes=[("TIFF", "*.tif *.tiff")])
        if path:
            self.output_path.set(path)

    def _parse_field(self, attr: str, cast: type, label: str) -> float | int:
        """解析参数输入框；非法值给出中文字段名，而不是英文 traceback 文本。"""
        raw = self._param_vars[attr].get()
        try:
            return cast(raw)
        except (TypeError, ValueError):
            kind = "整数" if cast is int else "数值"
            raise ParameterError(f"{label} 必须为{kind}，当前为 {raw!r}") from None

    def _params(self) -> FilterParams:
        return FilterParams(
            step=self._parse_field("step", int, "Step"),
            delta=self._parse_field("delta", float, "Delta %"),
            cycles=self._parse_field("cycles", int, "Cycles"),
            bw_order=self._parse_field("bw_order", int, "BW 阶数"),
            bw_ro=self._parse_field("bw_ro", float, "BW Ro"),
            apply_butterworth=self.apply_bw_var.get(),
            primary_output=self.mode_var.get(),
            stem_filter=self.stem_var.get(),
            crosshair_width=self._parse_field("crosshair_width", int, "十字线宽"),
            crosshair_hole_radius=self._parse_field("crosshair_hole_radius", int, "中心孔半径"),
            crosshair_bw_ro=self._parse_field("crosshair_bw_ro", float, "XH Ro"),
            low_freq_percent=self._parse_field("low_freq_percent", float, "低频回补 %"),
            rotation_method=self.rotation_var.get(),
        ).validated()

    def _sync_delta_zero_state(self, *_args: Any) -> None:
        """Delta=0 为纯 Butterworth 模式（validated() 强制固定输出/包络/剥离 STEM），界面同步反映。"""
        try:
            delta_zero = float(self.delta_var.get()) == 0.0
        except (ValueError, tk.TclError):
            return  # 非法输入交给 _params() 的参数错误提示处理
        state = tk.DISABLED if delta_zero else tk.NORMAL
        self.stem_check.configure(state=state)
        self.mode_combo.configure(state=state)
        self.apply_bw_check.configure(state=state)
        if delta_zero:
            self.status.set("Delta=0 为纯 Butterworth 模式；主输出、包络与 STEM 十字掩膜均已强制固定。")

    def _mark_preview_stale(self, *_args: Any) -> None:
        self._preview_stale = True

    def _fft_workers(self) -> int:
        raw = self.fft_workers_var.get()
        try:
            workers = int(raw)
        except (TypeError, ValueError):
            raise ParameterError(f"FFT 线程必须为正整数，当前为 {raw!r}") from None
        if workers < 1:
            raise ParameterError(f"FFT 线程必须至少为 1，当前为 {workers}")
        return workers

    def _crosshair_scale_warning(self, params: FilterParams, shape: tuple[int, int]) -> str | None:
        """小尺寸输入上十字线有效长度不超过中心孔时给出提示。

        十字线长度按图像尺寸比例缩放（XH Ro 是尺寸比例），而中心孔半径是
        绝对像素：小 ROI 预览会显著缩短十字线，甚至被中心孔完全吞掉，
        STEM 掩膜退化为空操作而看起来"功能失效"。
        """
        size = 1 << (max(shape) - 1).bit_length()
        extent = HRTEMFilter.crosshair_half_extent(size, params.crosshair_bw_ro)
        hole = params.crosshair_hole_radius + params.crosshair_width // 2
        if extent <= hole:
            return (
                f"当前处理尺寸约 {size}×{size} px：十字线有效长度仅约 {extent:.0f} px，"
                f"不超过中心孔半径+线宽（约 {hole} px），STEM 十字掩膜几乎不会改变结果。\n\n"
                "请改用全尺寸图像验证 STEM 效果，或增大 XH Ro / 减小中心孔半径。"
            )
        return None

    def _confirm_large_memory(self, shape: tuple[int, int], action: str) -> bool:
        """大操作启动前确认；返回 False 表示用户放弃。"""
        estimate = HRTEMFilter.estimate_peak_bytes(shape)
        if estimate <= self.LARGE_OPERATION_BYTES:
            return True
        return messagebox.askyesno(
            "内存警告",
            f"{action}的峰值内存估计约 {estimate / 1024**2:.0f} MiB，"
            "可能耗尽物理内存或触发交换，导致系统长时间无响应。\n\n"
            "建议先框选较小的 ROI。仍要继续吗？",
        )

    def _selected_roi(self) -> Roi | None:
        return self.roi if self.use_roi_var.get() else None

    def preview(self) -> None:
        if self.current_image is None:
            messagebox.showwarning("提示", "请先打开 TIFF")
            return
        # 预览不支持取消；进入预览前清掉保存流程的取消事件，
        # 避免保存后点击预览时“取消”按钮误报“正在取消”且无效。
        self._cancel_event = reset_cancel_event_for_preview(self._cancel_event)
        try:
            params = self._params()
            self._filter.workers = self._fft_workers()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        image = self.current_image.copy()
        roi = self._selected_roi()
        shape = roi.shape if roi is not None else image.shape
        if params.stem_filter:
            warning = self._crosshair_scale_warning(params, shape)
            if warning is not None:
                messagebox.showwarning("STEM 十字掩膜", warning)
        if not self._confirm_large_memory(shape, "本次预览"):
            return
        include_diagnostics = self.show_fft_var.get()
        self._set_processing(True)
        self.status.set("正在后台计算预览…")

        def worker() -> None:
            try:
                result = self._filter.process_image(image, params, roi=roi, include_diagnostics=include_diagnostics)
                self._messages.put(("preview_done", result))
            except Exception as exc:
                self._messages.put(("error", f"{type(exc).__name__}: {exc}"))

        self._worker = threading.Thread(target=worker, name="hrtem-preview", daemon=False)
        self._worker.start()

    def process_and_save(self) -> None:
        if self.info is None:
            messagebox.showwarning("提示", "请先打开 TIFF")
            return
        if not self.output_path.get().strip():
            messagebox.showwarning("提示", "请选择输出 TIFF")
            return
        try:
            params = self._params()
            options = SaveOptions(OutputEncoding(self.encoding_var.get())).validated()
            self._filter.workers = self._fft_workers()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self._cancel_event = threading.Event()
        source = self.info.path
        target = Path(self.output_path.get())
        roi = self._selected_roi()
        shape = roi.shape if roi is not None else self.info.frame_shape
        if params.stem_filter:
            warning = self._crosshair_scale_warning(params, shape)
            if warning is not None:
                messagebox.showwarning("STEM 十字掩膜", warning)
        if not self._confirm_large_memory(shape, "本次处理"):
            return
        self.progress.configure(value=0, maximum=self.info.frame_count)
        self._set_processing(True)
        self.status.set("正在后台处理并安全写入…")

        def progress(phase: str, current: int, total: int) -> None:
            self._messages.put(("progress", (phase, current, total)))

        def worker() -> None:
            try:
                # 复用共享滤波实例：径向网格缓存跨预览/保存保留，
                # 同尺寸堆栈的第二帧起不再重建缓存。
                report = StackProcessor(filter_processor=self._filter).process_tiff(
                    source,
                    target,
                    params=params,
                    save_options=options,
                    roi=roi,
                    cancel_event=self._cancel_event,
                    progress=progress,
                    entry="gui",
                )
                self._messages.put(("save_done", report))
            except ProcessingCancelled as exc:
                self._messages.put(("cancelled", str(exc)))
            except Exception as exc:
                self._messages.put(("error", f"{type(exc).__name__}: {exc}"))

        self._worker = threading.Thread(target=worker, name="hrtem-save", daemon=False)
        self._worker.start()

    def cancel(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
        self.status.set(cancel_status_text(self._cancel_event))

    def clear_roi(self) -> None:
        self.roi = None
        self._refresh_display()
        self.status.set("ROI 已清除；勾选使用 ROI 时将使用全图")

    def _drag_rect(self, start: tuple[int, int], end_x: int, end_y: int) -> tuple[int, int, int, int]:
        """拖拽两端换算为钳制在图像内的 (left, top, right, bottom) 右开区间。

        起点同样被钳制：坐标区边距内的按下点可能在图像之外。
        """
        return clamp_drag_rect(self.current_image.shape, start, (end_x, end_y))

    def _discard_live_roi_patch(self) -> None:
        if self._motion_cid is not None:
            self.canvas.mpl_disconnect(self._motion_cid)
            self._motion_cid = None
        if self._live_roi_patch is not None:
            self._live_roi_patch.remove()
            self._live_roi_patch = None

    def _on_mouse_press(self, event: Any) -> None:
        if event.inaxes != self.ax_original or self.current_image is None or event.xdata is None or event.ydata is None:
            return
        self._drag_start = (int(event.xdata), int(event.ydata))
        self._motion_cid = self.canvas.mpl_connect("motion_notify_event", self._on_mouse_motion)

    def _on_mouse_motion(self, event: Any) -> None:
        """拖拽过程中的实时虚线框，让框选范围在松开鼠标前就可见。"""
        if self._drag_start is None or self.current_image is None:
            return
        if event.inaxes != self.ax_original or event.xdata is None or event.ydata is None:
            return
        left, top, right, bottom = self._drag_rect(self._drag_start, int(event.xdata), int(event.ydata))
        if self._live_roi_patch is not None:
            self._live_roi_patch.remove()
        self._live_roi_patch = Rectangle(
            (left, top), right - left, bottom - top,
            fill=False, edgecolor="yellow", linewidth=1.5, linestyle="--",
        )
        self.ax_original.add_patch(self._live_roi_patch)
        self.canvas.draw_idle()

    def _on_mouse_release(self, event: Any) -> None:
        if self._drag_start is None or self.current_image is None:
            return
        start = self._drag_start
        self._drag_start = None
        self._discard_live_roi_patch()
        if event.inaxes != self.ax_original or event.xdata is None or event.ydata is None:
            return
        left, top, right, bottom = self._drag_rect(start, int(event.xdata), int(event.ydata))
        if right - left < 4 or bottom - top < 4:
            self.status.set("ROI 太小；最小尺寸为 4×4 px")
            return
        self.roi = Roi(top, left, bottom, right)
        self.use_roi_var.set(True)
        self._preview_stale = True
        estimate_mib = HRTEMFilter.estimate_peak_bytes(self.roi.shape) / 1024**2
        self.status.set(
            f"ROI: {right - left}×{bottom - top} px；峰值内存估计约 {estimate_mib:.0f} MiB；点击预览应用"
        )
        self._refresh_display()

    def _refresh_display(self) -> None:
        for axis in (self.ax_original, self.ax_filtered, self.ax_fft, self.ax_mask):
            axis.clear()
            axis.set_axis_off()
        if self.current_image is not None:
            self.ax_original.imshow(self.current_image, cmap="gray")
            self.ax_original.set_title("原图（拖拽框选 ROI）")
            if self.roi is not None:
                color = "lime" if self.use_roi_var.get() else "gray"
                self.ax_original.add_patch(Rectangle((self.roi.left, self.roi.top), self.roi.right - self.roi.left, self.roi.bottom - self.roi.top, fill=False, edgecolor=color, linewidth=2))
        if self.preview_result is not None:
            self.ax_filtered.imshow(self.preview_result.primary, cmap="gray")
            stale = "（参数已修改，预览可能过期）" if self._preview_stale else ""
            self.ax_filtered.set_title(f"预览：{self.preview_result.primary_key}{stale}")
            if self.show_fft_var.get():
                fft_image = self.preview_result.diagnostics.get("fft")
                if fft_image is not None:
                    self.ax_fft.imshow(np.log1p(np.abs(fft_image)), cmap="magma")
                    self.ax_fft.set_title("原始 FFT（对数幅值）")
                diag = self.preview_result.diagnostics
                mask = diag.get("wiener_mask")
                if mask is None:
                    mask = diag.get("absf_mask")
                if mask is None:
                    mask = diag.get("butterworth_mask")
                if mask is not None:
                    mask = np.asarray(mask)
                    crosshair = diag.get("stem_crosshair_mask")
                    if crosshair is not None:
                        mask = mask * crosshair
                    self.ax_mask.imshow(mask, cmap="viridis", vmin=0, vmax=1)
                    self.ax_mask.set_title("实际应用的掩膜")
        self.canvas.draw_idle()

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, payload = self._messages.get_nowait()
                if kind == "preview_done":
                    self.preview_result = payload
                    self._preview_stale = False
                    self._set_processing(False)
                    self.status.set(f"预览完成：{payload.primary_key}；显示自动缩放，保存值不归一化。")
                    self._refresh_display()
                elif kind == "progress":
                    phase, current, total = payload
                    self.progress.configure(maximum=total, value=current)
                    text = "计算全局显示范围" if phase == "scan" else "写入"
                    self.status.set(f"{text}: {current}/{total}")
                elif kind == "save_done":
                    self._set_processing(False)
                    self.progress.configure(value=payload.frame_count)
                    self.status.set(f"完成：{payload.output_path.name}")
                    messagebox.showinfo("处理完成", f"已保存：\n{payload.output_path}\n\n处理记录：\n{payload.provenance_path}")
                elif kind == "cancelled":
                    self._set_processing(False)
                    self.status.set("已取消；未发布正式输出文件。")
                    messagebox.showinfo("已取消", payload)
                elif kind == "error":
                    self._set_processing(False)
                    self.status.set("处理失败")
                    messagebox.showerror("处理失败", payload)
        except queue.Empty:
            pass
        if not self._closing:
            self.root.after(80, self._poll_messages)

    def _on_close(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            if not self._closing:
                self._closing = True
                self._close_deadline = time.time() + 60.0
                self._close_force_prompted = False
                if self._cancel_event is not None:
                    self._cancel_event.set()
                self.status.set("正在安全取消任务；完成当前帧后关闭窗口（最多等待60秒）…")
                self.root.after(100, self._wait_for_worker)
            return
        self.root.destroy()

    def _wait_for_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            if time.time() < self._close_deadline:
                self.root.after(100, self._wait_for_worker)
                return
            # 60 秒超时后必须给用户强制退出路径，不能无限轮询挂起。
            if not self._close_force_prompted:
                self._close_force_prompted = True
                if messagebox.askyesno(
                        "强制退出",
                        "任务未在60秒内结束。\n\n强制退出会直接关闭窗口；"
                        "正式输出不会被发布，但本次临时文件可能残留。\n是否强制退出？"):
                    self.root.destroy()
                    return
                self._close_force_prompted = False
                self._close_deadline = time.time() + 60.0
                self.status.set("继续等待任务结束…")
            self.root.after(100, self._wait_for_worker)
            return
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    HRTEMFilterGUI().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
