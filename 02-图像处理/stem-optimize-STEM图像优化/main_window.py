"""Tk desktop interface for safe HRTEM/STEM TIFF enhancement."""

import contextlib
import logging
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from contrast import estimate_global_range
from errors import InputValidationError, OperationCancelled
from filters import clear_filter_caches
from param_panel import ParamPanel
from pipeline import process_frame, validate_params
from preview_canvas import PreviewCanvas
from tiff_handler import (
    TiffStackReader,
    available_memory_bytes,
    estimate_peak_memory_bytes,
    file_signature,
    paths_refer_to_same_file,
)
from version import APP_NAME, APP_VERSION
from worker import ProcessingWorker, stale_temporary_files

logger = logging.getLogger(__name__)


class MainWindow:
    """Main application window and background-task state machine."""

    def __init__(self):
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            with contextlib.suppress(Exception):
                ctypes.windll.shcore.SetProcessDpiAwareness(1)

        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.root.geometry("1060x740")
        self.root.minsize(900, 640)

        self.file_path = None
        self.save_path = None
        self._file_signature = None
        self._reader = None
        self._worker = None
        self._preview_thread = None
        self._open_thread = None
        self._progress_queue = queue.Queue()
        self._cancel_event = threading.Event()
        self._preview_cancel_event = threading.Event()
        self._open_cancel_event = threading.Event()
        self._global_range = None
        self._global_range_percentiles = None
        self._file_generation = 0
        self._state = "idle"
        self._closing = False
        self._close_deadline = None
        self._poll_after_id = None

        self._build_ui()
        self._set_state("idle")
        self._poll_queue()

    def _build_ui(self):
        self._build_menubar()
        main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        main_paned.pack(fill="both", expand=True, padx=5, pady=5)

        left = tk.Frame(main_paned, width=350)
        main_paned.add(left, weight=0)
        self._build_left_panel(left)

        right = tk.Frame(main_paned)
        main_paned.add(right, weight=1)
        self._build_right_panel(right)

        self.status_var = tk.StringVar(value="就绪 — 请打开 TIFF 文件")
        tk.Label(
            self.root,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w",
            padx=5,
        ).pack(side="bottom", fill="x")

    def _build_menubar(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        self.file_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="文件", menu=self.file_menu)
        self.file_menu.add_command(
            label="打开…", command=self.on_open, accelerator="Ctrl+O"
        )
        # 记录菜单项索引而非硬编码数字，菜单项增删重排后自动保持正确。
        # index("end") 返回「刚加入项」的下标，不能再减 1（减 1 会得到 -1，
        # entryconfig(-1) 会抛 TclError: bad menu entry index "-1"）。
        self._menu_open_index = self.file_menu.index("end")
        self.file_menu.add_command(
            label="选择输出路径…",
            command=self.on_save_as,
            accelerator="Ctrl+S",
        )
        self._menu_save_index = self.file_menu.index("end")
        self.file_menu.add_separator()
        self.file_menu.add_command(label="退出", command=self.on_quit)

        self.edit_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="编辑", menu=self.edit_menu)
        self.edit_menu.add_command(label="重置参数", command=self.on_reset_params)
        self._menu_reset_index = self.edit_menu.index("end")

        help_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于", command=self.on_about)

        for key in ("<Control-o>", "<Control-O>"):
            self.root.bind(key, lambda _event: self.on_open())
        for key in ("<Control-s>", "<Control-S>"):
            self.root.bind(key, lambda _event: self.on_save_as())

    def _build_left_panel(self, parent):
        info_frame = tk.LabelFrame(parent, text="文件信息", padx=8, pady=5)
        info_frame.pack(fill="x", padx=5, pady=5)
        self.lbl_filename = tk.Label(
            info_frame,
            text="未选择文件",
            anchor="w",
            font=("Arial", 9, "bold"),
            wraplength=315,
        )
        self.lbl_filename.pack(fill="x")
        self.lbl_fileinfo = tk.Label(
            info_frame,
            text="",
            anchor="w",
            justify="left",
            font=("Consolas", 9),
            fg="#333333",
            wraplength=315,
        )
        self.lbl_fileinfo.pack(fill="x")

        self.param_panel = ParamPanel(parent)
        self.param_panel.pack(fill="x", padx=5, pady=5)

        button_frame = tk.LabelFrame(parent, text="操作", padx=8, pady=5)
        button_frame.pack(fill="x", padx=5, pady=5)
        preview_row = tk.Frame(button_frame)
        preview_row.pack(fill="x", pady=2)
        self.btn_preview = tk.Button(
            preview_row, text="预览效果", command=self.on_preview, width=12
        )
        self.btn_preview.pack(side="left", expand=True, fill="x", padx=2)
        tk.Label(preview_row, text="帧:").pack(side="left", padx=(5, 2))
        self.preview_frame_var = tk.IntVar(value=0)
        self.spin_frame = tk.Spinbox(
            preview_row,
            from_=0,
            to=0,
            width=6,
            textvariable=self.preview_frame_var,
        )
        self.spin_frame.pack(side="left")

        process_row = tk.Frame(button_frame)
        process_row.pack(fill="x", pady=2)
        self.btn_process = tk.Button(
            process_row,
            text="开始安全处理",
            command=self.on_process,
            bg="#2e7d32",
            fg="white",
            font=("Arial", 10, "bold"),
            height=2,
        )
        self.btn_process.pack(side="left", expand=True, fill="x", padx=2)
        self.btn_cancel = tk.Button(
            process_row,
            text="取消",
            command=self.on_cancel,
            bg="#c62828",
            fg="white",
            width=8,
        )
        self.btn_cancel.pack(side="left", padx=2)

        progress_frame = tk.LabelFrame(parent, text="任务进度", padx=8, pady=5)
        progress_frame.pack(fill="x", padx=5, pady=5)
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            mode="determinate",
            maximum=100,
        )
        self.progress_bar.pack(fill="x")
        self.lbl_progress = tk.Label(
            progress_frame,
            text="就绪",
            anchor="w",
            font=("Consolas", 9),
        )
        self.lbl_progress.pack(fill="x", pady=(2, 0))

    def _build_right_panel(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        self.original_canvas = PreviewCanvas(self.notebook)
        self.notebook.add(self.original_canvas, text="原始图像")
        self.filtered_canvas = PreviewCanvas(self.notebook)
        self.notebook.add(self.filtered_canvas, text="增强预览")

    def _set_state(self, state: str):
        self._state = state
        loaded = self._reader is not None
        idle = state == "idle"
        self.file_menu.entryconfig(
            self._menu_open_index, state="normal" if idle else "disabled"
        )
        self.file_menu.entryconfig(
            self._menu_save_index,
            state="normal" if idle and loaded else "disabled",
        )
        self.edit_menu.entryconfig(
            self._menu_reset_index, state="normal" if idle else "disabled"
        )
        self.btn_preview.config(state="normal" if idle and loaded else "disabled")
        self.btn_process.config(state="normal" if idle and loaded else "disabled")
        self.btn_cancel.config(
            state="normal"
            if state in ("preview", "processing", "opening")
            else "disabled"
        )
        self.spin_frame.config(state="normal" if idle and loaded else "disabled")
        self.param_panel.set_enabled(idle)

    def _current_signature(self):
        return file_signature(self.file_path)

    def _verify_input_unchanged(self) -> bool:
        try:
            unchanged = self._current_signature() == self._file_signature
        except OSError:
            unchanged = False
        if not unchanged:
            messagebox.showerror(
                "输入已变化",
                "输入文件自打开后已被修改或删除。请重新打开文件，"
                "避免预览与导出使用不同数据。",
            )
        return unchanged

    def on_open(self):
        if self._state != "idle":
            return
        path = filedialog.askopenfilename(
            title="选择二维灰度 HRTEM/STEM TIFF",
            filetypes=[
                ("TIFF 文件", "*.tif *.tiff"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return

        # 全页结构校验和首帧解码可能耗时数秒（压缩大帧），移入后台线程，
        # 期间界面保持响应且可取消。
        self._open_cancel_event.clear()
        self._set_state("opening")
        self.progress_bar["value"] = 0
        self.lbl_progress.config(text="正在打开并校验文件…")
        self.status_var.set(f"正在校验: {os.path.basename(path)}")
        generation = self._file_generation
        self._open_thread = threading.Thread(
            target=self._run_open,
            args=(path, generation),
            daemon=False,
            name="OpenWorker",
        )
        self._open_thread.start()

    def _run_open(self, path, generation):
        reader = None
        try:
            reader = TiffStackReader(
                path, cancel_event=self._open_cancel_event
            )
            if self._open_cancel_event.is_set():
                raise OperationCancelled()
            estimate = estimate_peak_memory_bytes(reader.shape)
            available = available_memory_bytes()
            if available is not None and estimate > int(available * 0.85):
                raise InputValidationError(
                    "预计单帧处理内存约 "
                    f"{estimate / 1024**3:.2f} GiB，当前可用内存约 "
                    f"{available / 1024**3:.2f} GiB；为防止界面或系统失去响应，"
                    "该文件未载入。"
                )
            first_frame = reader.read_frame(0)
            if self._open_cancel_event.is_set():
                raise OperationCancelled()
            signature = file_signature(reader.file_path)
            memory_text = f"预计峰值内存: {estimate / 1024**3:.2f} GiB"
            if available:
                memory_text += f" / 当前可用 {available / 1024**3:.2f} GiB"
            self._progress_queue.put(
                (
                    "open_done",
                    reader,
                    first_frame,
                    signature,
                    memory_text,
                    generation,
                )
            )
            reader = None  # 所有权随消息移交 GUI 线程
        except OperationCancelled:
            self._progress_queue.put(("open_cancelled", generation))
        except Exception as exc:
            logger.exception("打开 TIFF 失败")
            self._progress_queue.put(("open_error", str(exc), generation))
        finally:
            if reader is not None:
                reader.close()

    def on_save_as(self):
        if self._state != "idle":
            return False
        if not self.file_path:
            messagebox.showwarning("提示", "请先打开输入 TIFF")
            return False
        default_name = "enhanced_" + os.path.basename(self.file_path)
        path = filedialog.asksaveasfilename(
            title="选择安全输出路径",
            defaultextension=".tif",
            initialdir=os.path.dirname(self.file_path),
            initialfile=default_name,
            filetypes=[("TIFF 文件", "*.tif *.tiff")],
        )
        if not path:
            return False
        if os.path.splitext(path)[1].lower() not in (".tif", ".tiff"):
            path += ".tif"
        if paths_refer_to_same_file(self.file_path, path):
            messagebox.showerror(
                "路径不安全",
                "输出路径不能与输入文件相同。原始数据不会被覆盖。",
            )
            return False
        self.save_path = os.path.abspath(path)
        self.status_var.set(f"输出路径: {self.save_path}")
        return True

    def _percentiles_for_analysis(self) -> tuple[float, float]:
        """Return the UI percentiles, invalidating a cached range if changed."""
        percentiles = self.param_panel.get_range_percentiles()
        if (
            self._global_range is not None
            and self._global_range_percentiles != percentiles
        ):
            self._global_range = None
            self._global_range_percentiles = None
        return percentiles

    def on_preview(self):
        if self._state != "idle" or self._reader is None:
            return
        if not self._verify_input_unchanged():
            return
        try:
            frame_index = int(self.preview_frame_var.get())
            if not 0 <= frame_index < self._reader.num_frames:
                raise ValueError(f"帧索引必须在 0–{self._reader.num_frames - 1} 之间")
            params = validate_params(self.param_panel.get_params())
            percentiles = self._percentiles_for_analysis()
        except (ValueError, tk.TclError) as exc:
            messagebox.showwarning("参数错误", str(exc))
            return

        self._preview_cancel_event.clear()
        self.progress_bar["value"] = 0
        self.lbl_progress.config(text="准备预览…")
        self.status_var.set(f"正在生成帧 {frame_index} 的一致性预览…")
        self._set_state("preview")
        generation = self._file_generation
        reader = self._reader
        cached_range = self._global_range
        expected_signature = self._file_signature
        self._preview_thread = threading.Thread(
            target=self._run_preview,
            args=(
                reader,
                generation,
                frame_index,
                params,
                percentiles,
                cached_range,
                expected_signature,
            ),
            daemon=False,
            name="PreviewWorker",
        )
        self._preview_thread.start()

    def _run_preview(
        self,
        reader,
        generation,
        frame_index,
        params,
        percentiles,
        cached_range,
        expected_signature,
    ):
        try:
            if cached_range is None:

                def progress(current, total):
                    self._progress_queue.put(
                        (
                            "preview_analysis_progress",
                            int(current / total * 100),
                            current,
                            total,
                            generation,
                        )
                    )

                global_range = estimate_global_range(
                    reader,
                    low_percentile=percentiles[0],
                    high_percentile=percentiles[1],
                    cancel_event=self._preview_cancel_event,
                    progress_callback=progress,
                )
            else:
                global_range = cached_range
            if self._preview_cancel_event.is_set():
                raise OperationCancelled()
            frame = reader.read_frame(frame_index)
            output, stats = process_frame(frame, params, global_range)
            if file_signature(reader.file_path) != expected_signature:
                raise InputValidationError(
                    "输入文件在预览期间发生变化；本次预览已丢弃，请重新打开文件"
                )
            self._progress_queue.put(
                (
                    "preview_done",
                    frame,
                    output,
                    frame_index,
                    global_range,
                    stats,
                    percentiles,
                    generation,
                )
            )
        except OperationCancelled:
            self._progress_queue.put(("preview_cancelled", generation))
        except Exception as exc:
            logger.exception("预览失败")
            self._progress_queue.put(("preview_error", str(exc), generation))
        finally:
            # 与 ProcessingWorker 一致：预览结束后释放 FFT 半径网格缓存，
            # 否则大帧的数百 MiB~GiB 级网格会跨任务常驻。
            clear_filter_caches()

    def on_process(self):
        if self._state != "idle" or self._reader is None:
            return
        if not self._verify_input_unchanged():
            return
        if not self.save_path and not self.on_save_as():
            return
        if paths_refer_to_same_file(self.file_path, self.save_path):
            messagebox.showerror("路径不安全", "输入和输出不能是同一个文件。")
            self.save_path = None
            return
        if os.path.exists(self.save_path) and not messagebox.askyesno(
            "确认替换",
            f"目标文件已存在:\n{self.save_path}\n\n"
            "程序只会在新输出完整写入并通过校验后原子替换它。继续吗？",
        ):
            return
        # 确认对话框可能停留较长时间，期间输入文件可能被修改/删除；
        # 在启动 worker 前再次校验，保证预览与导出使用同一数据。
        if not self._verify_input_unchanged():
            return
        try:
            params = validate_params(self.param_panel.get_params())
            percentiles = self._percentiles_for_analysis()
        except (ValueError, tk.TclError) as exc:
            messagebox.showwarning("参数错误", str(exc))
            return
        if params["apply_clahe"] and not messagebox.askyesno(
            "非线性增强确认",
            "CLAHE 会按局部直方图非线性改变灰度，帧间强度不再可直接"
            "比较。仅建议展示增强。\n\n仍要启用吗？",
        ):
            return

        stale_paths = stale_temporary_files(self.save_path)
        if stale_paths:
            shown = "\n".join(os.path.basename(path) for path in stale_paths[:5])
            if len(stale_paths) > 5:
                shown += f"\n… 等 {len(stale_paths)} 个文件"
            if messagebox.askyesno(
                "发现残留临时文件",
                "目标目录存在上次异常退出留下的临时文件"
                "（其中 .backup 可能是旧版本程序备份的上一版输出）:\n\n"
                f"{shown}\n\n是否删除？",
            ):
                removed = 0
                for stale_path in stale_paths:
                    try:
                        os.unlink(stale_path)
                    except OSError:
                        logger.warning("无法删除残留临时文件: %s", stale_path)
                    else:
                        removed += 1
                if removed:
                    logger.info("已清理 %d 个残留临时文件", removed)

        self._cancel_event.clear()
        self.progress_bar["value"] = 0
        self.lbl_progress.config(text="准备安全事务…")
        self.status_var.set("处理中；正式输出在完整校验前不会被改动…")
        self._set_state("processing")
        self._worker = ProcessingWorker(
            file_path=self.file_path,
            save_path=self.save_path,
            params=params,
            progress_queue=self._progress_queue,
            cancel_event=self._cancel_event,
            global_range=self._global_range,
            range_percentiles=percentiles,
            expected_signature=self._file_signature,
        )
        self._worker.start()

    def on_cancel(self):
        if self._state == "preview":
            self._preview_cancel_event.set()
            self.status_var.set("正在取消预览…")
            self.btn_cancel.config(state="disabled")
        elif self._state == "processing":
            self._cancel_event.set()
            self.status_var.set("正在安全取消；不会发布临时输出…")
            self.btn_cancel.config(state="disabled")
        elif self._state == "opening":
            self._open_cancel_event.set()
            self.status_var.set("正在取消打开…")
            self.btn_cancel.config(state="disabled")

    def on_reset_params(self):
        if self._state == "idle":
            self.param_panel.reset_to_defaults()
            self.status_var.set("参数已重置；CLAHE 默认关闭")

    def on_about(self):
        messagebox.showinfo(
            "关于",
            f"{APP_NAME} v{APP_VERSION}\n\n"
            "自适应旋转背景频域降噪工具。\n"
            "不使用 PSF/CTF，因此不宣称物理反卷积。\n\n"
            "安全特性:\n"
            "• 输入/输出同路径保护\n"
            "• 临时文件 + 完整校验 + 原子替换\n"
            "• OME/ImageJ 元数据重建\n"
            "• 全堆栈强度范围与预览/导出一致\n"
            "• SHA-256 与处理参数溯源 sidecar\n\n"
            "CLAHE 属于非线性展示增强；定量分析必须保留原始数据。",
        )

    def _background_tasks_active(self) -> bool:
        return (
            (self._worker is not None and self._worker.is_alive())
            or (self._preview_thread is not None and self._preview_thread.is_alive())
            or (self._open_thread is not None and self._open_thread.is_alive())
        )

    def on_quit(self):
        active = self._background_tasks_active()
        if active and not self._closing:
            if not messagebox.askyesno(
                "安全退出",
                "任务仍在运行。退出前将先安全取消并清理临时文件，"
                "可能需要等待当前 FFT 完成。继续吗？",
            ):
                return
            self._closing = True
            self._cancel_event.set()
            self._preview_cancel_event.set()
            self._open_cancel_event.set()
            self._set_state("cancelling")
            self.status_var.set("正在安全清理任务，请稍候…")
            self._close_deadline = time.monotonic() + 60.0
            self.root.after(100, self._finish_close)
            return
        if active:
            return
        self._destroy()

    def _finish_close(self):
        active = self._background_tasks_active()
        if active:
            if time.monotonic() >= self._close_deadline:
                force_quit = messagebox.askyesno(
                    "任务仍未结束",
                    "后台任务在 60 秒内未能安全结束（可能卡在慢速磁盘或网络盘）。\n\n"
                    "强制退出会立即终止进程，可能残留未清理的临时文件"
                    "（.partial/.backup）且无法恢复。仍要强制退出吗？",
                )
                if force_quit:
                    self._destroy()
                    os._exit(1)
                self._close_deadline = time.monotonic() + 60.0
            self.root.after(100, self._finish_close)
        else:
            self._destroy()

    def _destroy(self):
        if self._poll_after_id is not None:
            with contextlib.suppress(tk.TclError):
                self.root.after_cancel(self._poll_after_id)
            self._poll_after_id = None
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        self.root.destroy()

    def _poll_queue(self):
        try:
            while True:
                self._handle_message(self._progress_queue.get_nowait())
        except queue.Empty:
            pass
        if self.root.winfo_exists():
            self._poll_after_id = self.root.after(100, self._poll_queue)

    def _handle_message(self, message):
        message_type = message[0]
        if message_type in ("analysis_progress", "progress"):
            _, percent, current, total = message
            self.progress_bar["value"] = percent
            stage = "范围分析" if message_type == "analysis_progress" else "处理"
            self.lbl_progress.config(text=f"{stage}: {current}/{total} ({percent}%)")
        elif message_type == "preview_analysis_progress":
            _, percent, current, total, generation = message
            if generation == self._file_generation:
                self.progress_bar["value"] = percent
                self.lbl_progress.config(
                    text=f"预览范围分析: {current}/{total} ({percent}%)"
                )
        elif message_type == "info":
            self.lbl_progress.config(text=message[1])
        elif message_type == "warning":
            self.status_var.set(f"警告: {message[1]}")
            logger.warning(message[1])
        elif message_type == "error":
            if not self._closing:
                messagebox.showerror("处理失败", message[1])
            self._worker = None
            self._set_state("idle")
            self.status_var.set("处理失败；正式输出未被不完整结果替换")
        elif message_type == "cancelled":
            if not self._closing:
                messagebox.showinfo("已安全取消", message[1])
            self._worker = None
            self._set_state("idle")
            self.status_var.set("已安全取消；原输出未被改动")
        elif message_type == "done":
            if not self._closing:
                messagebox.showinfo("完成", message[1])
            self._worker = None
            self._set_state("idle")
            self.status_var.set("处理完成并通过校验")
        elif message_type == "open_done":
            (
                _,
                reader,
                first_frame,
                signature,
                memory_text,
                generation,
            ) = message
            if generation != self._file_generation or self._closing:
                reader.close()
                return
            if self._reader is not None:
                self._reader.close()
            self._reader = reader
            self.file_path = reader.file_path
            self.save_path = None
            self._file_signature = signature
            self._global_range = None
            self._global_range_percentiles = None
            self._file_generation += 1

            self.lbl_filename.config(text=os.path.basename(self.file_path))
            self.lbl_fileinfo.config(
                text=(
                    f"帧数: {self._reader.num_frames}\n"
                    f"尺寸: {self._reader.shape[0]} × "
                    f"{self._reader.shape[1]}\n"
                    f"堆栈: {self._reader.stack_shape} "
                    f"[{self._reader.axes}]\n"
                    f"类型: {self._reader.dtype}\n"
                    f"OME: {'是' if self._reader.is_ome else '否'} | "
                    f"BigTIFF: {'是' if self._reader.is_bigtiff else '否'}\n"
                    f"{memory_text}\n路径: {self.file_path}"
                )
            )
            self.spin_frame.config(to=self._reader.num_frames - 1)
            self.preview_frame_var.set(0)
            self.original_canvas.update_image(first_frame)
            self.filtered_canvas.clear()
            self.progress_bar["value"] = 0
            self.lbl_progress.config(text="就绪")
            self.status_var.set(
                f"已验证: {os.path.basename(self.file_path)} "
                f"({self._reader.num_frames} 帧)"
            )
            logger.info("已打开并验证: %s", self.file_path)
            self._open_thread = None
            self._set_state("idle")
        elif message_type == "open_cancelled":
            if message[1] == self._file_generation:
                self._open_thread = None
                self._set_state("idle")
                self.status_var.set("已取消打开")
                self.lbl_progress.config(text="就绪")
        elif message_type == "open_error":
            _, text, generation = message
            if generation == self._file_generation:
                if not self._closing:
                    messagebox.showerror("无法打开 TIFF", text)
                self._open_thread = None
                self._set_state("idle")
                self.status_var.set("打开失败")
        elif message_type == "preview_done":
            (
                _,
                frame,
                output,
                frame_index,
                global_range,
                stats,
                percentiles,
                generation,
            ) = message
            if generation != self._file_generation:
                return
            self._global_range = tuple(global_range)
            self._global_range_percentiles = tuple(percentiles)
            self.original_canvas.update_image(frame, display_range=self._global_range)
            output_maximum = float(np.iinfo(output.dtype).max)
            self.filtered_canvas.update_image(
                output,
                display_range=(0.0, output_maximum),
            )
            self.notebook.select(1)
            clipped = (stats["clipped_low"] + stats["clipped_high"]) / max(
                stats["pixels"], 1
            )
            self.progress_bar["value"] = 100
            self.lbl_progress.config(text=f"预览完成；映射裁剪 {clipped:.3%}")
            self.status_var.set(f"帧 {frame_index} 预览已更新；预览与导出共用范围")
            self._preview_thread = None
            self._set_state("idle")
        elif message_type == "preview_cancelled":
            if message[1] == self._file_generation:
                self._preview_thread = None
                self._set_state("idle")
                self.status_var.set("预览已取消")
                self.lbl_progress.config(text="就绪")
        elif message_type == "preview_error":
            _, text, generation = message
            if generation == self._file_generation:
                if not self._closing:
                    messagebox.showerror("预览失败", text)
                self._preview_thread = None
                self._set_state("idle")
                self.status_var.set("预览失败")

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)
        self.root.mainloop()
