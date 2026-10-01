from __future__ import annotations

import json
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .ffmpeg import FFmpegError, FFmpegManager
from . import __version__
from .models import BitDepth, ColorMode, ExtractOptions, Sampling, SamplingMode
from .runner import JobRunner
from .utils import atomic_write_json

# ─── 中文显示名映射 ───

MODE_LABELS = {
    SamplingMode.ALL.value: "提取每一帧",
    SamplingMode.TARGET_FPS.value: "按目标帧率",
    SamplingMode.INTERVAL.value: "按时间间隔",
}
COLOR_LABELS = {
    ColorMode.PRESERVE.value: "保持原始",
    ColorMode.GRAYSCALE.value: "灰度",
    ColorMode.COLOR.value: "彩色 RGB",
}
DEPTH_LABELS = {
    BitDepth.SOURCE.value: "保持原始",
    BitDepth.UINT8.value: "8 位",
    BitDepth.UINT16.value: "16 位",
}
STATE_LABELS = {
    "counting": "正在统计帧数…",
    "hashing": "正在计算输入哈希…",
    "writing": "正在写入输出…",
    "validating": "正在校验输出…",
}

CONFIG_DIR = Path.home() / ".tem_video_extractor"
CONFIG_FILE = CONFIG_DIR / "config.json"
MAX_LOG_LINES = 600


def _label_map(labels: dict[str, str]) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Return (display_values, display->value, value->display)."""
    values = list(labels.values())
    d2v = {v: k for k, v in labels.items()}
    v2d = dict(labels)
    return values, d2v, v2d


class ExtractorApp:
    """视频转 TIFF 堆栈 GUI。所有工作线程事件通过 Queue 跨线程传递。"""

    def __init__(self) -> None:
        # 高 DPI 屏上 tk 默认位图缩放会发糊；声明进程 DPI 感知（仅 Windows）
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
        self.root = tk.Tk()
        self.root.title(f"TEM 视频转 TIFF 堆栈工具 v{__version__}")
        self.root.geometry("940x680")
        self.root.minsize(860, 600)
        try:
            self.root.option_add("*Font", "微软雅黑 10")
        except tk.TclError:
            pass

        self.events: queue.Queue[dict] = queue.Queue()
        self.runner: JobRunner | None = None
        self.worker: threading.Thread | None = None
        self._detect_thread: threading.Thread | None = None
        self._cancel_requested = threading.Event()
        self._start_time: float = 0.0
        self._video_started_at: float | None = None
        self._close_started_at: float | None = None

        # ─── 变量 ───
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.ffmpeg_var = tk.StringVar()
        self.ffmpeg_status_var = tk.StringVar(value="未检测")

        mode_values, self._mode_d2v, self._mode_v2d = _label_map(MODE_LABELS)
        color_values, self._color_d2v, self._color_v2d = _label_map(COLOR_LABELS)
        depth_values, self._depth_d2v, self._depth_v2d = _label_map(DEPTH_LABELS)

        self.mode_var = tk.StringVar(value=self._mode_v2d[SamplingMode.ALL.value])
        self.value_var = tk.StringVar()
        self.color_var = tk.StringVar(value=self._color_v2d[ColorMode.PRESERVE.value])
        # ImageJ 工作流的常见输出是 8 位；高位深源仍可手动选择
        self.depth_var = tk.StringVar(value=self._depth_v2d[BitDepth.UINT8.value])
        self.lenient_var = tk.BooleanVar(value=False)

        self.status_var = tk.StringVar(value="就绪")
        self.speed_var = tk.StringVar(value="")
        self.progress_var = tk.DoubleVar(value=0)

        # 存储 combo 值列表供 _build 使用
        self._combo_values = {
            "mode": mode_values,
            "color": color_values,
            "depth": depth_values,
        }

        self._load_config()
        self._build()
        self.mode_var.trace_add("write", self._update_entry_states)
        self._update_entry_states()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(100, self._poll_events)

    # ─── 配置持久化 ───

    def _load_config(self) -> None:
        if not CONFIG_FILE.exists():
            return
        try:
            cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return
        if not isinstance(cfg, dict):
            # 根对象不是字典（如 [] / null / 数字）时读取逻辑会直接
            # AttributeError 且发生在界面构建之前——必须备份坏配置并忽略
            try:
                backup = CONFIG_FILE.with_suffix(f".invalid-{int(time.time())}.json")
                CONFIG_FILE.replace(backup)
            except OSError:
                pass
            return

        def restore_str(key: str, variable: tk.StringVar) -> None:
            value = cfg.get(key)
            if isinstance(value, str) and value:
                variable.set(value)

        restore_str("input", self.input_var)
        restore_str("output", self.output_var)
        restore_str("ffmpeg", self.ffmpeg_var)

        def restore_combo(variable: tk.StringVar, value_to_label: dict[str, str], key: str) -> None:
            value = cfg.get(key)
            if isinstance(value, str):
                label = value_to_label.get(value)
                if label:
                    variable.set(label)

        restore_combo(self.mode_var, self._mode_v2d, "mode")
        restore_combo(self.color_var, self._color_v2d, "color")
        restore_combo(self.depth_var, self._depth_v2d, "depth")
        value = cfg.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            self.value_var.set(str(value))
        elif isinstance(value, str) and value:
            self.value_var.set(value)
        lenient = cfg.get("lenient_decode")
        if isinstance(lenient, bool):
            self.lenient_var.set(lenient)

    def _save_config(self) -> None:
        cfg = {
            "input": self.input_var.get(),
            "output": self.output_var.get(),
            "ffmpeg": self.ffmpeg_var.get(),
            "mode": self._mode_d2v[self.mode_var.get()],
            "value": self.value_var.get(),
            "color": self._color_d2v[self.color_var.get()],
            "depth": self._depth_d2v[self.depth_var.get()],
            "lenient_decode": self.lenient_var.get(),
        }
        try:
            atomic_write_json(CONFIG_FILE, cfg)
        except OSError as error:
            messagebox.showwarning("配置保存失败", f"无法写入 {CONFIG_FILE}:\n{error}")

    # ─── UI 构建 ───

    def _build(self) -> None:
        main = ttk.Frame(self.root, padding=14)
        main.pack(fill=tk.BOTH, expand=True)

        # ── 路径与环境 ──
        paths = ttk.LabelFrame(main, text="输入与运行环境", padding=10)
        paths.pack(fill=tk.X)
        self._row(paths, "输入视频目录:", self.input_var, browse=True)
        self._row(paths, "输出目录:", self.output_var, browse=True)

        ff_row = ttk.Frame(paths)
        ff_row.pack(fill=tk.X, pady=3)
        ttk.Label(ff_row, text="FFmpeg 路径:", width=18).pack(side=tk.LEFT)
        ttk.Entry(ff_row, textvariable=self.ffmpeg_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(ff_row, text="浏览", command=lambda: self._choose_file(self.ffmpeg_var)).pack(side=tk.LEFT, padx=2)
        self.detect_button = ttk.Button(ff_row, text="检测", command=self._detect_ffmpeg)
        self.detect_button.pack(side=tk.LEFT, padx=2)
        ttk.Label(ff_row, textvariable=self.ffmpeg_status_var, foreground="blue", width=20).pack(side=tk.LEFT, padx=4)

        # ── 提取设置 ──
        options = ttk.LabelFrame(main, text="提取设置", padding=10)
        options.pack(fill=tk.X, pady=8)
        output_row = ttk.Frame(options)
        output_row.pack(fill=tk.X, pady=3)
        ttk.Label(output_row, text="输出:", width=18).pack(side=tk.LEFT)
        ttk.Label(
            output_row,
            text="单个 ImageJ TIFF 堆栈（ImageJ 1.53c 元数据，超 3.8 GiB 自动切换虚拟栈结构）",
            foreground="#155724",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self._combo(options, "采样模式:", self.mode_var, self._combo_values["mode"])
        self.value_entry = self._row(options, "FPS / 间隔秒:", self.value_var)
        self._combo(options, "颜色:", self.color_var, self._combo_values["color"])
        self._combo(options, "位深:", self.depth_var, self._combo_values["depth"])
        lenient_row = ttk.Frame(options)
        lenient_row.pack(fill=tk.X, pady=3)
        ttk.Checkbutton(
            lenient_row, variable=self.lenient_var,
            text="宽松解码（损坏帧跳过而非任务失败；结果可能缺帧且位置不可知，仅建议抢救性提取时勾选）",
        ).pack(side=tk.LEFT)

        ttk.Label(
            main,
            text="程序会先统计真实帧数、检查磁盘空间，再写入 ImageJ 兼容的 TIFF 堆栈；已有输出仍会拒绝覆盖，避免误删实验数据。",
            foreground="#5c4500", wraplength=880,
        ).pack(anchor=tk.W, pady=4)

        # ── 进度 ──
        bar = ttk.Progressbar(main, variable=self.progress_var, maximum=100)
        bar.pack(fill=tk.X, pady=5)
        info_row = ttk.Frame(main)
        info_row.pack(fill=tk.X)
        ttk.Label(info_row, textvariable=self.status_var).pack(side=tk.LEFT)
        ttk.Label(info_row, textvariable=self.speed_var, foreground="gray").pack(side=tk.RIGHT)

        # ── 按钮 ──
        # 先从底部预留按钮栏，再让日志占据剩余空间；否则高 DPI/小屏幕下
        # Text 的请求高度可能把“开始处理”挤出可见区域。
        buttons = ttk.Frame(main)
        buttons.pack(side=tk.BOTTOM, fill=tk.X, pady=(2, 0))
        self.start_button = ttk.Button(buttons, text="开始处理", command=self._start)
        self.start_button.pack(side=tk.LEFT)
        self.cancel_button = ttk.Button(buttons, text="取消", command=self._cancel, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="清空日志", command=self._clear_log).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="退出", command=self._close).pack(side=tk.RIGHT)

        # ── 日志 ──
        self.log = tk.Text(main, height=10, state=tk.DISABLED, font=("Consolas", 9))
        self.log.pack(fill=tk.BOTH, expand=True, pady=5)

    def _row(self, parent: ttk.Widget, label: str, variable: tk.StringVar, browse: bool = False) -> tk.Entry:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, width=18).pack(side=tk.LEFT)
        entry = ttk.Entry(row, textvariable=variable)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        if browse:
            ttk.Button(row, text="浏览", command=lambda: self._choose_dir(variable, label)).pack(side=tk.LEFT)
        return entry

    def _combo(self, parent: ttk.Widget, label: str, variable: tk.StringVar, values: list[str]) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, width=18).pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=variable, values=values, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _update_entry_states(self, *_args: object) -> None:
        mode = SamplingMode(self._mode_d2v[self.mode_var.get()])
        self.value_entry.config(state=tk.NORMAL if mode is not SamplingMode.ALL else tk.DISABLED)

    def _choose_dir(self, variable: tk.StringVar, label: str) -> None:
        path = filedialog.askdirectory(title=f"选择{label.rstrip(':')}")
        if path:
            variable.set(path)

    def _choose_file(self, variable: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            title="选择 FFmpeg 可执行文件",
            filetypes=[("可执行文件", "*.exe"), ("所有文件", "*.*")],
        )
        if path:
            variable.set(path)

    # ─── FFmpeg 检测 ───

    def _detect_ffmpeg(self) -> None:
        if self._detect_thread and self._detect_thread.is_alive():
            return
        custom = self.ffmpeg_var.get().strip() or None
        self.ffmpeg_status_var.set("检测中…")
        self.detect_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED)

        def run() -> None:
            try:
                paths = FFmpegManager(custom).resolve()
                self.events.put({"kind": "ffmpeg_detected", "version": ".".join(map(str, paths.version)), "path": str(paths.ffmpeg)})
            except FFmpegError as error:
                self.events.put({"kind": "ffmpeg_detect_failed", "error": str(error)})

        self._detect_thread = threading.Thread(target=run, name="ffmpeg-detect", daemon=True)
        self._detect_thread.start()

    # ─── 参数解析 ───

    def _options(self) -> ExtractOptions:
        mode = SamplingMode(self._mode_d2v[self.mode_var.get()])
        value = None
        if mode is not SamplingMode.ALL:
            raw_value = self.value_var.get().strip()
            if not raw_value:
                raise ValueError("请为所选采样模式填写目标 FPS 或时间间隔（秒）")
            try:
                value = float(raw_value)
            except ValueError:
                raise ValueError(f"采样值必须是数字，当前输入: {raw_value}") from None
        return ExtractOptions(
            sampling=Sampling(mode, value),
            color_mode=ColorMode(self._color_d2v[self.color_var.get()]),
            bit_depth=BitDepth(self._depth_d2v[self.depth_var.get()]),
            lenient_decode=self.lenient_var.get(),
        )

    # ─── 处理控制 ───

    def _start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            output_text = self.output_var.get().strip()
            if not self.input_var.get().strip():
                raise ValueError("请选择输入目录")
            if not output_text:
                raise ValueError("请选择输出目录")
            input_root = Path(self.input_var.get()).expanduser()
            output_root = Path(output_text).expanduser()
            if not input_root.is_dir():
                raise ValueError(f"输入目录不存在: {input_root}")
            if output_root.exists() and not output_root.is_dir():
                raise ValueError(f"输出路径已存在但不是目录: {output_root}")
            options = self._options()
            options.validate()
        except Exception as error:
            messagebox.showerror("无法开始", str(error))
            return

        ffmpeg_path = self.ffmpeg_var.get().strip() or None
        self._save_config()
        self._cancel_requested.clear()
        self.progress_var.set(0)
        self.speed_var.set("")
        self._start_time = time.time()
        self.start_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        self._log("任务已启动")

        def run() -> None:
            try:
                # FFmpeg 解析包含多次子进程调用，放在工作线程避免冻结界面
                self.runner = JobRunner(options, ffmpeg_path)
                # 修复取消竞态：runner 创建前用户已点取消时，必须立即同步取消信号
                if self._cancel_requested.is_set():
                    self.runner.cancel()
                results = self.runner.run_batch(input_root, output_root, self.events.put)
                self.events.put({"kind": "batch_finished", "results": [r.as_dict() for r in results]})
            except Exception as error:
                self.events.put({"kind": "fatal", "error": str(error)})

        self.worker = threading.Thread(target=run, name="extractor-worker", daemon=True)
        self.worker.start()

    def _cancel(self) -> None:
        # 始终 set 事件：即使 runner 还没创建，worker 创建后也会立刻调用 runner.cancel()
        self._cancel_requested.set()
        if self.runner:
            self.runner.cancel()
        self.status_var.set("正在安全取消…")
        self.cancel_button.config(state=tk.DISABLED)

    # ─── 事件轮询 ───

    def _poll_events(self) -> None:
        try:
            while True:
                try:
                    event = self.events.get_nowait()
                except queue.Empty:
                    break
                kind = event.get("kind")
                if kind == "batch_started":
                    total = event.get("total", 0)
                    self._job_id = event.get("job_id")
                    self.status_var.set(f"批任务启动：共 {total} 个视频")
                    self._log(f"批任务启动：共 {total} 个视频")
                elif kind == "ffmpeg_detected":
                    self.ffmpeg_status_var.set(f"✓ v{event.get('version', '')}")
                    self._log(f"FFmpeg 可用: {event.get('path', '')} (v{event.get('version', '')})")
                    self.detect_button.config(state=tk.NORMAL)
                    if not (self.worker and self.worker.is_alive()):
                        self.start_button.config(state=tk.NORMAL)
                elif kind == "ffmpeg_detect_failed":
                    self.ffmpeg_status_var.set("✗ 不可用")
                    self._log(f"FFmpeg 检测失败: {event.get('error', '')}")
                    self.detect_button.config(state=tk.NORMAL)
                    if not (self.worker and self.worker.is_alive()):
                        self.start_button.config(state=tk.NORMAL)
                elif kind == "progress":
                    total = event.get("total") or 0
                    current = event.get("current", 0)
                    batch_index = event.get("index") or 1
                    batch_total = event.get("batch_total") or 1
                    # 批级加权进度：多视频时进度条反映整批完成度而非单个视频。
                    # 单个视频内，计数阶段占 30%、写入阶段占 70%
                    if batch_total > 1 and total:
                        overall = ((batch_index - 1) + 0.3 + 0.7 * current / total) * 100 / batch_total
                        self.progress_var.set(min(100, overall))
                    elif total:
                        self.progress_var.set(min(100, (0.3 + 0.7 * current / total) * 100))
                    # 速度/ETA 以当前视频起点计时，跨视频累计会把速度摊薄失真
                    started_at = self._video_started_at or self._start_time
                    elapsed = time.time() - started_at
                    speed = current / elapsed if elapsed > 1 else 0
                    eta = (total - current) / speed if speed > 0 and total else 0
                    eta_str = f"{int(eta // 60)}:{int(eta % 60):02d}" if eta < 3600 else f"{int(eta // 3600)}h{int((eta % 3600) // 60)}m"
                    batch_prefix = f"[{batch_index}/{batch_total}] " if batch_total > 1 else ""
                    self.speed_var.set(f"{batch_prefix}{speed:.1f} 帧/秒 | 剩余: {eta_str}")
                    self.status_var.set(f"{Path(event['path']).name}: {current}/{total or '?'} 帧")
                elif kind == "counting_progress":
                    name = Path(event["path"]).name
                    current = event.get("current", 0)
                    hint = event.get("total_hint")
                    batch_index = event.get("index") or 1
                    batch_total = event.get("batch_total") or 1
                    if hint:
                        # 计数是一整遍解码，可能耗时较长；把估计帧数作为分母
                        # 映射到该视频份额的前 30%，避免界面"看似卡住"
                        fraction = min(1.0, current / hint)
                        overall = ((batch_index - 1) + 0.3 * fraction) * 100 / batch_total
                        self.progress_var.set(min(100, overall))
                    self.status_var.set(f"{name}: 正在统计帧数（{current}）")
                elif kind == "output_plan":
                    size = float(event.get("estimated_bytes", 0)) / 1024**3
                    self._log(
                        f"  输出计划：{event.get('frames', 0)} 帧，约 {size:.2f} GiB，"
                        f"{event.get('container', 'TIFF')} 堆栈"
                    )
                elif kind == "state":
                    name = Path(event["path"]).name
                    label = STATE_LABELS.get(event.get("state", ""))
                    self.status_var.set(f"{name}: {label}" if label else f"处理中：{name}")
                elif kind == "video_started":
                    self._video_started_at = time.time()
                    self.status_var.set(f"处理中：{Path(event['path']).name}")
                    self._log(f"▶ {Path(event['path']).name}")
                elif kind == "video_finished":
                    result = event["result"]
                    state = result["state"]
                    name = Path(result["input_path"]).name
                    error = result.get("error") or ""
                    frames = result.get("frame_count", 0)
                    if state == "completed":
                        output_bytes = result.get("output_bytes")
                        size_text = f"，{output_bytes / 1024**3:.2f} GiB" if output_bytes else ""
                        self._log(f"✓ {name} ({frames} 帧{size_text})")
                        # 成功时把产物路径写进日志：批量大时用户找不到输出在哪
                        if result.get("output_path"):
                            self._log(f"    输出: {result['output_path']}")
                        if result.get("frame_manifest_path"):
                            self._log(f"    逐帧清单: {result['frame_manifest_path']}")
                        for warning in result.get("warnings") or []:
                            self._log(f"    ⚠ {warning}")
                    else:
                        self._log(f"✗ {name}: {error}")
                elif kind == "batch_finished":
                    results = event["results"]
                    completed = sum(r["state"] == "completed" for r in results)
                    elapsed = time.time() - self._start_time
                    self.status_var.set(f"处理结束：成功 {completed}/{len(results)}，耗时 {elapsed:.1f}s")
                    self.speed_var.set("")
                    self.start_button.config(state=tk.NORMAL)
                    self.cancel_button.config(state=tk.DISABLED)
                    self._log(f"批处理完成：{completed}/{len(results)} 成功")
                    job_id = getattr(self, "_job_id", None)
                    if job_id:
                        manifest_dir = Path(self.output_var.get().strip()) / "manifests"
                        self._log(f"    任务清单目录（含告警与溯源记录）: {manifest_dir}")
                elif kind == "warning":
                    self._log("⚠ " + event.get("message", ""))
                elif kind == "info":
                    self._log(event.get("message", ""))
                elif kind == "fatal":
                    self._log("致命错误: " + event["error"])
                    self.status_var.set("任务异常终止")
                    self.speed_var.set("")
                    self.start_button.config(state=tk.NORMAL)
                    self.cancel_button.config(state=tk.DISABLED)
        except Exception as error:
            # 单个事件处理失败不能中断轮询，否则界面会永久冻结
            self._log(f"事件处理异常: {error}")
        finally:
            try:
                self.root.after(100, self._poll_events)
            except tk.TclError:
                pass

    # ─── 日志 ───

    def _log(self, message: str) -> None:
        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, message + "\n")
        self.log.see(tk.END)
        # 限制行数
        line_count = int(self.log.index("end-1c").split(".")[0])
        if line_count > MAX_LOG_LINES:
            self.log.delete("1.0", f"{line_count - MAX_LOG_LINES}.0")
        self.log.config(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self.log.config(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.config(state=tk.DISABLED)

    # ─── 关闭 ───

    def _close(self) -> None:
        self._save_config()
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("确认退出", "任务仍在运行，是否取消后退出？"):
                return
            self._close_started_at = time.time()
            self._cancel()
            self.root.after(200, self._wait_then_close)
            return
        self.root.destroy()

    def _wait_then_close(self) -> None:
        if self.worker and self.worker.is_alive():
            elapsed = time.time() - (self._close_started_at or time.time())
            if elapsed >= 60:
                self._log("等待线程退出超时（60 秒），强制关闭窗口")
                self.root.destroy()
                return
            self.root.after(200, self._wait_then_close)
        else:
            self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
