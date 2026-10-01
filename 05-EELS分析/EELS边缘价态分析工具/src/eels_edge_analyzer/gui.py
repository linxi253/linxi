"""Tkinter 图形界面。

EELSEdgeAnalyzerApp 接受一个 root/parent，因此既可作为独立窗口运行，也可由
TEM Suite 的 ToolHost 内嵌到标签页。
"""

from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from .dm4io import format_dataset_table, infer_dataset_indices, inspect_dm4
from .models import AnalysisCancelled, AnalysisConfig, ReferenceSpec
from .pipeline import run_analysis
from .presets import load_preset
from .references import discover_cu_references
from .reporting import export_artifacts, output_directory_state


class EELSEdgeAnalyzerApp:
    """Cu-L Dual-EELS 边缘价态分析的单窗口工作流。"""

    def __init__(self, root: tk.Misc) -> None:
        self.root = root
        self.root.title("EELS边缘价态分析工具 v1")
        self._messages: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._processing = False
        self._cancel_requested = False
        self._result_paths: dict[str, Path] | None = None
        self._worker_thread: threading.Thread | None = None
        self._inspecting = False
        self._inspect_again = False
        self._overwrite_allowed = False

        self.input_var = tk.StringVar()
        self.reference_dir_var = tk.StringVar()
        self.cu0_var = tk.StringVar()
        self.cu1_var = tk.StringVar()
        self.cu2_var = tk.StringVar()
        self.preset_var = tk.StringVar()
        self.use_preset_var = tk.BooleanVar(value=True)
        self.output_var = tk.StringVar()
        self.orientation_var = tk.StringVar(value="auto")
        # 用 StringVar 而不是 IntVar：IntVar.get() 在输入框为空或非整数时抛出
        # TclError，会绕过 _start 的统一错误处理；字符串解析能给出正常报错。
        self.bootstrap_var = tk.StringVar(value="500")
        self.sensitivity_var = tk.BooleanVar(value=True)
        self.injection_var = tk.BooleanVar(value=True)
        self.survey_index_var = tk.StringVar()
        self.low_index_var = tk.StringVar()
        self.high_index_var = tk.StringVar()
        self.status_var = tk.StringVar(value="请选择原始 DM3/DM4 文件。")

        self._build_ui()
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        except (AttributeError, tk.TclError):
            pass
        self.root.after(100, self._poll_messages)

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(1, weight=1)

        input_box = ttk.LabelFrame(main, text="1. 输入与参考谱", padding=10)
        input_box.grid(row=0, column=0, columnspan=3, sticky="nsew")
        input_box.columnconfigure(1, weight=1)
        self._path_row(input_box, 0, "原始 DM3/DM4：", self.input_var, self._browse_input, "浏览并检查")
        self._path_row(
            input_box,
            1,
            "Cu0/Cu1/Cu2 参考谱目录：",
            self.reference_dir_var,
            self._browse_reference_dir,
            "自动寻找参考",
        )
        self._path_row(input_box, 2, "结果输出目录：", self.output_var, self._browse_output, "选择输出目录")

        # 单文件参考谱（三个都填写时优先于参考目录自动发现）。
        ref_files = ttk.Frame(input_box)
        ref_files.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        for position, (text, variable) in enumerate(
            (("Cu0：", self.cu0_var), ("Cu1：", self.cu1_var), ("Cu2：", self.cu2_var))
        ):
            ttk.Label(ref_files, text=text).grid(row=0, column=position * 3, sticky="w")
            ttk.Entry(ref_files, textvariable=variable, width=20).grid(
                row=0, column=position * 3 + 1, sticky="ew", padx=3
            )
            ttk.Button(ref_files, text="…", width=3, command=lambda v=variable: self._browse_reference_file(v)).grid(
                row=0, column=position * 3 + 2
            )
            ref_files.columnconfigure(position * 3 + 1, weight=1)
        ttk.Label(
            input_box,
            text="三个参考文件都填写时优先于目录自动发现；格式支持 DM3/DM4/CSV/TXT/DAT/MSA。",
            foreground="gray",
        ).grid(row=4, column=0, columnspan=3, sticky="w")
        self._path_row(
            input_box,
            5,
            "参数预设（留空=内置 Cu L2,3）：",
            self.preset_var,
            self._browse_preset,
            "选择预设",
        )
        ttk.Checkbutton(
            input_box,
            text="使用参数预设（取消后采用程序默认参数，等效 CLI 的 --no-preset）",
            variable=self.use_preset_var,
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(2, 0))

        dataset_box = ttk.LabelFrame(main, text="2. 数据对象与边缘设置", padding=10)
        dataset_box.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        for column in range(6):
            dataset_box.columnconfigure(column, weight=1 if column in {1, 3, 5} else 0)
        ttk.Label(dataset_box, text="Survey编号（留空自动）：").grid(row=0, column=0, sticky="w")
        ttk.Entry(dataset_box, textvariable=self.survey_index_var, width=8).grid(
            row=0, column=1, sticky="w", padx=(3, 12)
        )
        ttk.Label(dataset_box, text="低损编号：").grid(row=0, column=2, sticky="w")
        ttk.Entry(dataset_box, textvariable=self.low_index_var, width=8).grid(
            row=0, column=3, sticky="w", padx=(3, 12)
        )
        ttk.Label(dataset_box, text="高损编号：").grid(row=0, column=4, sticky="w")
        ttk.Entry(dataset_box, textvariable=self.high_index_var, width=8).grid(
            row=0, column=5, sticky="w", padx=(3, 0)
        )
        ttk.Label(dataset_box, text="样品表面方向：").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            dataset_box,
            textvariable=self.orientation_var,
            values=("auto", "top", "bottom", "left", "right"),
            state="readonly",
            width=10,
        ).grid(row=1, column=1, sticky="w", padx=(3, 12), pady=(8, 0))
        ttk.Label(dataset_box, text="Bootstrap次数：").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Spinbox(
            dataset_box, from_=50, to=5000, increment=50, textvariable=self.bootstrap_var, width=9
        ).grid(row=1, column=3, sticky="w", padx=(3, 12), pady=(8, 0))
        ttk.Checkbutton(dataset_box, text="执行参数敏感性分析", variable=self.sensitivity_var).grid(
            row=1, column=4, columnspan=2, sticky="w", pady=(8, 0)
        )
        ttk.Checkbutton(dataset_box, text="执行 Cu1 注入恢复检验", variable=self.injection_var).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )
        ttk.Label(
            dataset_box,
            text="提示：自动边缘应在运行后查看 registered_surface_boundary.png；若方向不对，请改为 top/bottom/left/right 重跑。",
            foreground="gray",
        ).grid(row=2, column=3, columnspan=3, sticky="w", pady=(8, 0))

        detail_box = ttk.LabelFrame(main, text="DM4 检查结果", padding=8)
        detail_box.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        main.rowconfigure(2, weight=1)
        self.inspect_text = tk.Text(detail_box, height=9, wrap=tk.WORD, font=("Consolas", 9))
        self.inspect_text.pack(fill=tk.BOTH, expand=True)
        self.inspect_text.insert("1.0", "尚未检查输入文件。")
        self.inspect_text.configure(state=tk.DISABLED)

        controls = ttk.Frame(main)
        controls.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        self.run_button = ttk.Button(controls, text="开始分析", command=self._start)
        self.run_button.pack(side=tk.LEFT)
        self.stop_button = ttk.Button(controls, text="停止", command=self._stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))
        self.open_button = ttk.Button(
            controls, text="打开结果目录", command=self._open_output, state=tk.DISABLED
        )
        self.open_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(controls, textvariable=self.status_var).pack(side=tk.LEFT, padx=(16, 0))

        self.progress = ttk.Progressbar(main, mode="determinate", maximum=100)
        self.progress.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        log_box = ttk.LabelFrame(main, text="运行日志", padding=8)
        log_box.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        main.rowconfigure(5, weight=1)
        self.log_text = tk.Text(log_box, height=10, wrap=tk.WORD, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _path_row(
        self, parent: ttk.LabelFrame, row: int, label: str, variable: tk.StringVar, browse: Any, action: str
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=variable, width=76).grid(
            row=row, column=1, sticky="ew", padx=5, pady=3
        )
        ttk.Button(parent, text=action, command=browse).grid(row=row, column=2, pady=3)

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(
            title="选择原始 EELS DM3/DM4",
            filetypes=(("DigitalMicrograph", "*.dm4 *.dm3"), ("All files", "*.*")),
        )
        if path:
            self.input_var.set(path)
            default_output = Path(path).with_name(f"{Path(path).stem}_eels_edge_analysis")
            self.output_var.set(str(default_output))
            # 数据对象编号只对当前 DM4 有效。更换文件时必须清除旧编号，
            # 再由新文件的对象检查结果重新填充，避免静默分析错误对象。
            self.survey_index_var.set("")
            self.low_index_var.set("")
            self.high_index_var.set("")
            self._request_inspect()

    def _browse_reference_dir(self) -> None:
        path = filedialog.askdirectory(title="选择 Cu0/Cu1/Cu2 参考谱目录")
        if path:
            self.reference_dir_var.set(path)
            try:
                references = discover_cu_references(Path(path))
                self._log(
                    "已找到参考谱：" + "；".join(f"{item.label}={item.path.name}" for item in references)
                )
            except (OSError, ValueError, RuntimeError) as exc:
                self._log(f"参考谱目录检查失败：{exc}")

    def _browse_reference_file(self, variable: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            title="选择参考谱",
            filetypes=(
                ("DigitalMicrograph / 文本", "*.dm4 *.dm3 *.csv *.txt *.dat *.msa"),
                ("All files", "*.*"),
            ),
        )
        if path:
            variable.set(path)

    def _browse_preset(self) -> None:
        path = filedialog.askopenfilename(
            title="选择参数预设 JSON",
            filetypes=(("JSON", "*.json"), ("All files", "*.*")),
        )
        if path:
            self.preset_var.set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title="选择独立结果输出目录")
        if path:
            self.output_var.set(path)

    def _request_inspect(self) -> None:
        """请求检查当前输入文件；忙时记住待办，完成后再检查最新文件。

        旧实现忙时直接丢弃第二次请求，界面会停留在上一个文件的对象表。
        """
        text = self.input_var.get().strip()
        if not text:
            return
        if self._inspecting:
            self._inspect_again = True
            return
        self._inspecting = True
        self._set_inspection("正在检查输入文件…")
        # 头部解析通常瞬时完成，但异常 DM 文件可能退化到完整读取路径，
        # 因此统一放入后台线程，避免界面冻结。
        threading.Thread(target=self._inspect_worker, args=(Path(text),), daemon=True).start()

    def _resume_pending_inspect(self) -> None:
        if self._inspect_again:
            self._inspect_again = False
            self._request_inspect()

    def _inspect_worker(self, source: Path) -> None:
        try:
            items = inspect_dm4(source)
            survey, low, high = infer_dataset_indices(items)
        except Exception as exc:  # noqa: BLE001 - 后台检查边界需要回传完整错误
            self._messages.put(("inspect_error", str(exc)))
            return
        self._messages.put(("inspect_done", (items, survey, low, high)))

    def _apply_inspection(self, items: list, survey: int | None, low: int | None, high: int | None) -> None:
        self._set_inspection(format_dataset_table(items, survey, low, high))
        if not self.survey_index_var.get() and survey is not None:
            self.survey_index_var.set(str(survey))
        if not self.low_index_var.get() and low is not None:
            self.low_index_var.set(str(low))
        if not self.high_index_var.get() and high is not None:
            self.high_index_var.set(str(high))

    def _set_inspection(self, text: str) -> None:
        self.inspect_text.configure(state=tk.NORMAL)
        self.inspect_text.delete("1.0", tk.END)
        self.inspect_text.insert("1.0", text)
        self.inspect_text.configure(state=tk.DISABLED)

    @staticmethod
    def _parse_int_field(value: str, field_name: str) -> int:
        text = value.strip()
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"{field_name}必须是整数，当前为“{text}”。") from exc

    def _optional_index(self, value: str, field_name: str) -> int | None:
        text = value.strip()
        if not text:
            return None
        return self._parse_int_field(text, field_name)

    def _make_config(self) -> AnalysisConfig:
        input_text = self.input_var.get().strip()
        if not input_text:
            raise ValueError("请先选择原始 DM3/DM4 输入文件。")
        output_text = self.output_var.get().strip()
        if not output_text:
            # Path("") 等价于当前目录；不拦截会把结果静默写进启动目录。
            raise ValueError("请填写或选择结果输出目录（不能为空）。")
        source = Path(input_text)
        output = Path(output_text)
        single_paths = (
            self.cu0_var.get().strip(),
            self.cu1_var.get().strip(),
            self.cu2_var.get().strip(),
        )
        if all(single_paths):
            references = (
                ReferenceSpec("Cu0", 0, Path(single_paths[0])),
                ReferenceSpec("Cu1", 1, Path(single_paths[1])),
                ReferenceSpec("Cu2", 2, Path(single_paths[2])),
            )
        elif not any(single_paths):
            reference_dir = Path(self.reference_dir_var.get().strip())
            references = discover_cu_references(reference_dir)
        else:
            raise ValueError("Cu0/Cu1/Cu2 参考谱需要三个文件全部指定，或全部留空并使用参考谱目录。")
        # 预设提供基线参数；界面上的显式设置（bootstrap、方向、数据对象等）
        # 始终优先于预设。取消勾选时等效 CLI 的 --no-preset。
        preset_text = self.preset_var.get().strip()
        if not self.use_preset_var.get():
            overrides: dict[str, Any] = {}
        else:
            overrides = dict(load_preset(Path(preset_text)) if preset_text else load_preset())
        config_kwargs: dict[str, Any] = dict(overrides)
        config_kwargs.update(
            input_path=source,
            output_dir=output,
            references=references,
            survey_dataset=self._optional_index(self.survey_index_var.get(), "Survey 编号"),
            low_loss_dataset=self._optional_index(self.low_index_var.get(), "低损编号"),
            high_loss_dataset=self._optional_index(self.high_index_var.get(), "高损编号"),
            surface_orientation=self.orientation_var.get(),
            bootstrap_resamples=self._parse_int_field(self.bootstrap_var.get(), "Bootstrap 次数"),
            run_sensitivity=bool(self.sensitivity_var.get()),
            injection_simulations=200 if self.injection_var.get() else 0,
        )
        return AnalysisConfig(**config_kwargs)

    def _start(self) -> None:
        if self._processing:
            return
        try:
            config = self._make_config()
            config.validate()
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("参数不完整", str(exc), parent=self.root)
            return
        # 输出目录非空时先确认，避免静默覆盖既往结果（与 CLI --overwrite 对应）。
        state = output_directory_state(config.output_dir)
        self._overwrite_allowed = state != "previous_results"
        if state in {"previous_results", "other_content"}:
            noun = "既往分析结果" if state == "previous_results" else "其他文件"
            confirmed = messagebox.askyesno(
                "覆盖确认",
                f"输出目录已包含{noun}：\n{config.output_dir}\n\n"
                "继续分析将写入/覆盖其中的结果文件。是否继续？",
                parent=self.root,
            )
            if not confirmed:
                return
            self._overwrite_allowed = True
        self._processing = True
        self._cancel_requested = False
        self._result_paths = None
        self.run_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.open_button.configure(state=tk.DISABLED)
        self.progress.configure(value=0)
        self._log(f"开始分析：{config.input_path.name}")
        worker = threading.Thread(target=self._worker, args=(config,), daemon=True)
        self._worker_thread = worker
        worker.start()

    def _worker(self, config: AnalysisConfig) -> None:
        def progress(message: str, fraction: float | None) -> None:
            self._messages.put(("progress", (message, fraction)))

        try:
            artifacts = run_analysis(
                config,
                progress=progress,
                cancel=lambda: self._cancel_requested,
            )
            paths = export_artifacts(artifacts, config, overwrite=self._overwrite_allowed)
            self._messages.put(("done", paths))
        except AnalysisCancelled:
            self._messages.put(("cancelled", None))
        except Exception as exc:  # noqa: BLE001 - 后台任务边界需要回传完整错误
            self._messages.put(("error", (str(exc), traceback.format_exc())))

    def _stop(self) -> None:
        if self._processing:
            self._cancel_requested = True
            self.status_var.set("正在请求停止；当前计算块完成后将退出。")
            self.stop_button.configure(state=tk.DISABLED)

    def _on_close(self) -> None:
        """关闭独立窗口或 TEM Suite 标签页时请求取消后台分析。"""

        self._cancel_requested = True
        if not isinstance(self.root, (tk.Tk, tk.Toplevel)):
            return
        worker = self._worker_thread
        if self._processing and worker is not None and worker.is_alive():
            # 后台线程可能正处于导出阶段（导出没有取消点）。硬杀线程会
            # 留下截断的结果文件，因此等待其退出后再关闭窗口。
            self.status_var.set("正在等待后台任务结束后关闭…")
            self.root.after(200, self._wait_worker_then_close)
            return
        self.root.destroy()

    def _wait_worker_then_close(self) -> None:
        worker = self._worker_thread
        if worker is not None and worker.is_alive():
            self.root.after(200, self._wait_worker_then_close)
            return
        self.root.destroy()

    def _poll_messages(self) -> None:
        try:
            try:
                while True:
                    kind, payload = self._messages.get_nowait()
                    if kind == "progress":
                        message, fraction = payload
                        self.status_var.set(message)
                        self._log(message)
                        if fraction is not None:
                            self.progress.configure(value=max(0, min(100, 100 * fraction)))
                    elif kind == "inspect_done":
                        self._inspecting = False
                        items, survey, low, high = payload
                        self._apply_inspection(items, survey, low, high)
                        self._resume_pending_inspect()
                    elif kind == "inspect_error":
                        self._inspecting = False
                        self._set_inspection("DM4 检查失败。")
                        messagebox.showerror("DM4 检查失败", str(payload), parent=self.root)
                        self._resume_pending_inspect()
                    elif kind == "done":
                        self._processing = False
                        self._result_paths = payload
                        self.run_button.configure(state=tk.NORMAL)
                        self.stop_button.configure(state=tk.DISABLED)
                        self.open_button.configure(state=tk.NORMAL)
                        self.progress.configure(value=100)
                        self.status_var.set("分析完成。")
                        self._log(f"完成。报告：{payload['report']}")
                        messagebox.showinfo(
                            "分析完成", f"结果已保存至：\n{Path(payload['report']).parent}", parent=self.root
                        )
                    elif kind == "cancelled":
                        self._processing = False
                        self.run_button.configure(state=tk.NORMAL)
                        self.stop_button.configure(state=tk.DISABLED)
                        self.status_var.set("分析已取消；原始数据未被修改。")
                        self._log("分析已取消。")
                    elif kind == "error":
                        self._processing = False
                        self.run_button.configure(state=tk.NORMAL)
                        self.stop_button.configure(state=tk.DISABLED)
                        message, details = payload
                        self.status_var.set("分析失败。")
                        self._log(details)
                        messagebox.showerror("分析失败", message, parent=self.root)
            except queue.Empty:
                pass
            self.root.after(100, self._poll_messages)
        except tk.TclError:
            # 窗口/宿主控件已销毁（例如嵌入式标签页被关闭）时停止轮询。
            return

    def _open_output(self) -> None:
        if not self._result_paths:
            return
        folder = Path(self._result_paths["report"]).parent
        try:
            os.startfile(str(folder))  # type: ignore[attr-defined]
        except AttributeError:
            messagebox.showinfo("结果目录", str(folder), parent=self.root)

    def _log(self, message: str) -> None:
        self.log_text.insert(tk.END, message.rstrip() + "\n")
        self.log_text.see(tk.END)


def main() -> int:
    root = tk.Tk()
    root.minsize(940, 720)
    EELSEdgeAnalyzerApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
