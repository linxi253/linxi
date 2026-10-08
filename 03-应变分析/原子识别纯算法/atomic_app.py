# -*- coding: utf-8 -*-
"""原子识别、跨帧 ID 关联与原子柱积分强度分析 GUI。"""

from __future__ import annotations

import csv
import json
import logging
import os
import queue
import shutil
import sys
import tempfile
import threading
from dataclasses import asdict, replace
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

import matplotlib

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle

from atomic_core import (
    AtomRecord,
    AtomicToolError,
    CalibrationResult,
    DetectionParams,
    DetectionRegion,
    IntensityParams,
    MarkerRule,
    build_track_catalog,
    calibrate_from_points,
    close_tiff_stack,
    detect_atoms,
    link_frame_points,
    load_tiff_stack,
    make_records,
    match_points_to_catalog,
    normalize_for_display,
    parse_finite_float,
    recalculate_record_intensities,
    refine_point_on_work,
    render_marked_frame,
    rule_for_percentile,
    visible_records,
)


APP_VERSION = "1.3"

# 界面节奏与容量常量：集中命名，避免魔法数字散落各处。
POLL_INTERVAL_MS = 80
REDRAW_DEBOUNCE_MS = 120
UNDO_HISTORY_LIMIT = 100
TRACK_MAX_MISSED = 8
SESSION_FORMAT = "atomic-recognition-session-1"
# Excel/Sheets 公式注入前缀：用户自由文本写入 CSV 前转义。
CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t")


def _csv_safe_text(value: str) -> str:
    """防止 CSV 单元格在 Excel/Sheets 中被解释为公式：给 =/+/-/@/Tab 开头加单引号。"""
    text = str(value)
    if text.startswith(CSV_FORMULA_PREFIXES):
        return "'" + text
    return text


def _intensity_fingerprint(params: IntensityParams | None) -> str:
    """CSV 每行的强度参数指纹，让部分重算后的各行参数差异可直接审计。"""
    if params is None:
        return "未记录"
    return (
        f"aperture={params.aperture_radius:g}"
        f";bg_inner={params.background_inner_radius:g}"
        f";bg_outer={params.background_outer_radius:g}"
        f";bright={int(params.bright)}"
    )


def _json_float(value: float) -> float | None:
    """NaN/inf 在标准 JSON 中不可表示，序列化为 null。"""
    number = float(value)
    return number if np.isfinite(number) else None


def _restore_float(value: float | None, default: float) -> float:
    return default if value is None else float(value)


def _detection_params_from_json(item: dict | None) -> DetectionParams | None:
    if item is None:
        return None
    return DetectionParams(**item).validated()


def _intensity_params_from_json(item: dict | None) -> IntensityParams | None:
    if item is None:
        return None
    return IntensityParams(**item).validated()


def _region_from_json(item: dict | None) -> DetectionRegion | None:
    if item is None:
        return None
    vertices = tuple((float(point[0]), float(point[1])) for point in item["vertices"])
    return DetectionRegion(str(item["kind"]), vertices).validated()


def _rule_from_json(item: dict) -> MarkerRule:
    return MarkerRule(
        name=str(item.get("name", "未命名范围")),
        low=float(item["low"]),
        high=float(item["high"]),
        color=str(item.get("color", "#00ff66")),
        marker=str(item.get("marker", "circle")),
        visible=bool(item.get("visible", True)),
    ).validated()


def _record_from_json(item: dict) -> AtomRecord:
    x, y = item.get("x"), item.get("y")
    if x is None or y is None:
        raise AtomicToolError("会话记录包含无效坐标（x/y 为空）。")
    return AtomRecord(
        atom_id=int(item["atom_id"]),
        x=float(x),
        y=float(y),
        integrated_intensity=_restore_float(item.get("integrated_intensity"), np.nan),
        percentile=_restore_float(item.get("percentile"), np.nan),
        aperture_pixels=int(item.get("aperture_pixels", 0)),
        background_pixels=int(item.get("background_pixels", 0)),
        background_level=_restore_float(item.get("background_level"), np.nan),
        truncated=bool(item.get("truncated", False)),
        source=str(item.get("source", "auto")),
        nearest_neighbor=_restore_float(item.get("nearest_neighbor"), np.inf),
        neighbor_contaminated=bool(item.get("neighbor_contaminated", False)),
    )


_logger = logging.getLogger("atomic_recognition_tool")
LOG_PATH: Path | None = None


def _setup_logging() -> tuple[logging.Logger, Path | None]:
    """配置滚动日志；打包成无控制台 exe 后这是排查问题的唯一线索。"""
    global LOG_PATH
    if LOG_PATH is not None:
        return _logger, LOG_PATH
    log_dir = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "AtomicRecognitionTool"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        target = log_dir / "app.log"
        handler = RotatingFileHandler(target, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    except OSError:
        target = Path(tempfile.gettempdir()) / "atomic_recognition_tool.log"
        try:
            handler = RotatingFileHandler(target, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        except OSError:
            return _logger, None
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    LOG_PATH = target
    return _logger, LOG_PATH


def _log_exception(message: str, exc_info=None) -> None:
    logger, _ = _setup_logging()
    logger.error("%s", message, exc_info=exc_info)


def _configure_chinese_font() -> set[str]:
    candidates = [
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            matplotlib.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
    return available


AVAILABLE_FONTS = _configure_chinese_font()


MARKER_LABELS = {
    "圆圈": "circle",
    "方框": "square",
    "三角形": "triangle",
    "菱形": "diamond",
    "叉号": "cross",
    "加号": "plus",
    "星形": "star",
}
MARKER_LABEL_BY_KEY = {value: key for key, value in MARKER_LABELS.items()}
MPL_MARKERS = {
    "circle": "o",
    "square": "s",
    "triangle": "^",
    "diamond": "D",
    "cross": "x",
    "plus": "+",
    "star": "*",
}


DETECTION_PRESETS = {
    "标准 HRTEM (默认)": DetectionParams(0.8, 8.0, 5, None, True, "com"),
    "高分辨 / 小原子 (2-4 px)": DetectionParams(0.4, 4.0, 3, None, True, "com"),
    "超分辨 / 极小原子 (<2 px)": DetectionParams(0.2, 3.0, 3, None, True, "com"),
    "低倍 / 大原子 (>6 px)": DetectionParams(1.5, 12.0, 7, None, True, "com"),
    "STEM HAADF (高衬度亮点)": DetectionParams(0.6, 6.0, 5, None, True, "gaussian"),
    "高噪声 / 低信噪比": DetectionParams(1.2, 8.0, 7, None, True, "com"),
}


class AtomicRecognitionApp:
    MAX_DRAWN_LABELS = 500

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"原子识别与原子柱强度分析工具 v{APP_VERSION}")
        self.root.geometry("1500x900")
        self.root.minsize(1200, 700)

        self.stack: np.ndarray | None = None
        self.stack_info = None
        self.current_frame = 0
        self.frame_records: list[list[AtomRecord]] = []
        self.frame_processed: list[bool] = []
        self.frame_regions: list[DetectionRegion | None] = []
        self.frame_detection_params: list[DetectionParams | None] = []
        self.frame_intensity_params: list[IntensityParams | None] = []
        self.frame_edited: list[bool] = []
        self.stack_region: DetectionRegion | None = None
        self.selected_atom_id: int | None = None
        # 撤销快照：(帧号, 记录副本, processed, edited, 快照时的强度参数)。
        self.undo_stack: list[tuple[int, list[AtomRecord], bool, bool, IntensityParams | None]] = []

        self.detection_params = DetectionParams()
        self.intensity_params = IntensityParams()
        self.match_distance = 4.0
        self.marker_radius = 7.0
        self.show_ids = False
        self.rules: list[MarkerRule] = [
            MarkerRule("全部原子", 0.0, 100.0, "#00ff66", "circle", True)
        ]

        self.current_mode = "add"
        self._calibration_points: list[tuple[float, float]] = []
        self._rectangle_start: tuple[float, float] | None = None
        self._polygon_vertices: list[tuple[float, float]] = []
        self._region_preview_artist = None
        self.pick_radius = 12.0
        self._image_artist = None
        self._overlay_artists: list[object] = []
        self._label_artists: list[object] = []

        self._worker_queue: queue.Queue = queue.Queue()
        self._cancel_event = threading.Event()
        self._job_generation = 0
        self._busy = False
        self._shutdown_requested = False
        self._frame_refresh_after_id: int | None = None
        self._overlay_refresh_after_id: int | None = None
        # 单帧显示归一化缓存（帧号 -> 归一化数组），换栈/换帧时失效。
        self._display_cache: tuple[int, np.ndarray] | None = None

        self._setup_crash_reporting()
        self._setup_styles()
        self._build_ui()
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._poll_active = True
        self._poll_after_id = self.root.after(POLL_INTERVAL_MS, self._poll_worker_queue)

    # ------------------------------------------------------------------
    # 崩溃与异常日志
    # ------------------------------------------------------------------
    def _setup_crash_reporting(self) -> None:
        self._crash_dialog_shown = False
        self.root.report_callback_exception = self._on_tk_callback_exception

    def _on_tk_callback_exception(self, exc_type, exc_value, exc_tb) -> None:
        _log_exception("Tk 回调未捕获异常", (exc_type, exc_value, exc_tb))
        try:
            self.status.config(text="发生内部错误，已写入日志；程序继续运行")
        except tk.TclError:
            pass
        if not self._crash_dialog_shown:
            self._crash_dialog_shown = True
            try:
                location = str(LOG_PATH) if LOG_PATH else "临时目录"
                messagebox.showerror("内部错误", f"发生未预期的错误，详情已写入日志文件：\n{location}")
            except tk.TclError:
                pass

    # ------------------------------------------------------------------
    # PPA 同风格界面
    # ------------------------------------------------------------------
    def _setup_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        colors = {
            "bg_main": "#f5f5f5",
            "bg_panel": "#ffffff",
            "bg_header": "#1a237e",
            "accent": "#1565c0",
            "accent_hover": "#1976d2",
            "accent_active": "#0d47a1",
            "success": "#2e7d32",
            "warning": "#f57c00",
            "danger": "#c62828",
            "text_primary": "#212121",
            "text_secondary": "#757575",
            "border": "#e0e0e0",
        }
        self.colors = colors
        self.root.configure(bg=colors["bg_main"])
        family = "Microsoft YaHei UI" if "Microsoft YaHei UI" in AVAILABLE_FONTS else "Segoe UI"
        base = (family, 9)
        small = (family, 8)
        title = (family, 10, "bold")

        style.configure(".", font=base, background=colors["bg_main"])
        style.configure("TFrame", background=colors["bg_main"])
        style.configure("TLabel", background=colors["bg_main"], foreground=colors["text_primary"])
        style.configure("TButton", padding=(8, 4))
        style.configure("Card.TLabelframe", background=colors["bg_panel"], relief="solid", borderwidth=1)
        style.configure(
            "Card.TLabelframe.Label",
            background=colors["bg_panel"],
            foreground=colors["accent"],
            font=title,
        )
        style.configure("Accent.TButton", foreground="white", background=colors["accent"], padding=(10, 6))
        style.map(
            "Accent.TButton",
            background=[("active", colors["accent_hover"]), ("pressed", colors["accent_active"])],
            foreground=[("active", "white")],
        )
        style.configure("Success.TButton", foreground="white", background=colors["success"], padding=(10, 6))
        style.map("Success.TButton", background=[("active", "#388e3c"), ("pressed", "#1b5e20")])
        style.configure("Warning.TButton", foreground="white", background=colors["warning"], padding=(8, 4))
        style.map("Warning.TButton", background=[("active", "#fb8c00"), ("pressed", "#e65100")])
        style.configure("Danger.TButton", foreground="white", background=colors["danger"], padding=(8, 4))
        style.map("Danger.TButton", background=[("active", "#d32f2f"), ("pressed", "#b71c1c")])
        style.configure(
            "Outline.TButton",
            foreground=colors["text_primary"],
            background=colors["bg_panel"],
            relief="solid",
            borderwidth=1,
            padding=(8, 4),
        )
        style.map("Outline.TButton", background=[("active", "#e3f2fd")])
        style.configure("Status.TLabel", background="#263238", foreground="#eceff1", font=small, padding=(8, 4))
        style.configure("Treeview", rowheight=22, fieldbackground="white", background="white")
        style.configure("Treeview.Heading", font=(family, 8, "bold"))

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg=self.colors["bg_header"], height=48)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(
            header,
            text="⚛  原子识别与原子柱强度分析工具",
            bg=self.colors["bg_header"],
            fg="white",
            font=("Microsoft YaHei UI", 12, "bold"),
            anchor="w",
            padx=12,
        ).pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(
            header,
            text=f"v{APP_VERSION}  |  TIFF Stack · 局部积分强度 · 跨帧稳定 ID",
            bg=self.colors["bg_header"],
            fg="#90caf9",
            font=("Microsoft YaHei UI", 8),
            anchor="e",
            padx=12,
        ).pack(side=tk.RIGHT, fill=tk.Y)

        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left = ttk.Frame(main)
        main.add(left, weight=3)
        self.figure = Figure(figsize=(9, 8), dpi=100, facecolor="#fafafa")
        self.axes = self.figure.add_subplot(111)
        self.axes.set_facecolor("#fafafa")
        self.canvas = FigureCanvasTkAgg(self.figure, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, left)
        toolbar.update()
        self.canvas.mpl_connect("button_press_event", self._on_image_click)
        self.canvas.mpl_connect("button_release_event", self._on_image_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        right = ttk.Frame(main, width=410)
        main.add(right, weight=1)
        self.right_canvas = tk.Canvas(right, highlightthickness=0, width=405, bg=self.colors["bg_main"])
        scrollbar = ttk.Scrollbar(right, orient=tk.VERTICAL, command=self.right_canvas.yview)
        self.right_canvas.configure(yscrollcommand=scrollbar.set)
        self.right_inner = ttk.Frame(self.right_canvas)
        self.right_window = self.right_canvas.create_window((0, 0), window=self.right_inner, anchor="nw")
        self.right_inner.bind(
            "<Configure>",
            lambda _event: self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all")),
        )
        self.right_canvas.bind(
            "<Configure>",
            lambda event: self.right_canvas.itemconfigure(self.right_window, width=event.width),
        )

        def mousewheel(event):
            self.right_canvas.yview_scroll(int(-event.delta / 120), "units")

        self.right_canvas.bind("<Enter>", lambda _event: self.right_canvas.bind_all("<MouseWheel>", mousewheel))
        self.right_canvas.bind("<Leave>", lambda _event: self.right_canvas.unbind_all("<MouseWheel>"))
        self.right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._build_image_card(self.right_inner)
        self._build_edit_card(self.right_inner)
        self._build_intensity_card(self.right_inner)
        self._build_rule_card(self.right_inner)
        self._build_point_card(self.right_inner)
        self._build_export_card(self.right_inner)

        self.status = ttk.Label(
            right,
            text="就绪  |  导入单张或堆栈 TIFF",
            style="Status.TLabel",
            anchor=tk.W,
            wraplength=400,
        )
        self.status.pack(fill=tk.X, side=tk.BOTTOM, before=self.right_canvas)

    def _card(self, parent, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, padding=10, style="Card.TLabelframe")
        frame.pack(fill=tk.X, padx=6, pady=4)
        return frame

    def _build_image_card(self, parent) -> None:
        frame = self._card(parent, "🖼 图像、帧与自动识别")
        self.btn_import = ttk.Button(frame, text="📂 导入 TIFF / 堆栈 TIFF", command=self.load_tiff, style="Accent.TButton")
        self.btn_import.pack(fill=tk.X, pady=(0, 4))
        self.file_info_var = tk.StringVar(value="尚未导入图像")
        ttk.Label(frame, textvariable=self.file_info_var, foreground=self.colors["text_secondary"], wraplength=370).pack(
            fill=tk.X, pady=(0, 4)
        )

        nav = ttk.Frame(frame)
        nav.pack(fill=tk.X, pady=2)
        ttk.Button(nav, text="◀", width=4, command=lambda: self._change_frame(-1), style="Outline.TButton").pack(side=tk.LEFT)
        self.frame_scale = ttk.Scale(nav, from_=0, to=0, command=self._on_frame_scale)
        self.frame_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(nav, text="▶", width=4, command=lambda: self._change_frame(1), style="Outline.TButton").pack(side=tk.RIGHT)
        self.frame_label_var = tk.StringVar(value="帧 0 / 0")
        ttk.Label(frame, textvariable=self.frame_label_var, anchor=tk.CENTER).pack(fill=tk.X)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=(5, 2))
        self.btn_detect_current = ttk.Button(row, text="🔍 识别当前帧", command=lambda: self.start_detection(False), style="Accent.TButton")
        self.btn_detect_current.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.btn_detect_all = ttk.Button(row, text="🎞 逐帧识别全部", command=lambda: self.start_detection(True), style="Success.TButton")
        self.btn_detect_all.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        ttk.Button(row, text="⚙", width=3, command=self.show_detection_params, style="Outline.TButton").pack(side=tk.RIGHT)

        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=2)
        self.btn_calibrate = ttk.Button(row2, text="📐 校准", command=self.start_calibration, style="Outline.TButton")
        self.btn_calibrate.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.btn_relink = ttk.Button(row2, text="🔗 重新关联 ID", command=self.relink_all_ids, style="Outline.TButton")
        self.btn_relink.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        self.btn_cancel = ttk.Button(row2, text="⏹ 取消任务", command=self.cancel_job, state=tk.DISABLED, style="Warning.TButton")
        self.btn_cancel.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        selection_row = ttk.Frame(frame)
        selection_row.pack(fill=tk.X, pady=(4, 2))
        self.btn_rectangle = ttk.Button(
            selection_row, text="⬜ 矩形选区", command=self.start_rectangle_selection, style="Outline.TButton"
        )
        self.btn_rectangle.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.btn_polygon = ttk.Button(
            selection_row, text="✏ 自由选区", command=self.start_polygon_selection, style="Outline.TButton"
        )
        self.btn_polygon.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        self.btn_clear_region = ttk.Button(
            selection_row, text="清除选区", command=self.clear_active_region, style="Outline.TButton"
        )
        self.btn_clear_region.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        scope_frame = ttk.LabelFrame(frame, text="选区作用范围", padding=5)
        scope_frame.pack(fill=tk.X, pady=(2, 3))
        self.selection_scope_var = tk.StringVar(value="current")
        ttk.Radiobutton(
            scope_frame,
            text="仅限制当前帧识别",
            variable=self.selection_scope_var,
            value="current",
            command=self._on_selection_scope_changed,
        ).pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(
            scope_frame,
            text="逐帧识别全部中应用",
            variable=self.selection_scope_var,
            value="all",
            command=self._on_selection_scope_changed,
        ).pack(side=tk.LEFT, padx=3)
        self.region_summary_var = tk.StringVar(value="当前没有选区")
        ttk.Label(
            frame,
            textvariable=self.region_summary_var,
            foreground=self.colors["text_secondary"],
            wraplength=370,
        ).pack(fill=tk.X, pady=(0, 2))

        self.progress = ttk.Progressbar(frame, mode="determinate", maximum=1)
        self.progress.pack(fill=tk.X, pady=(4, 0))
        self.detect_summary_var = tk.StringVar(value="参数：间距 8 px，σ 0.8，COM，亮原子")
        ttk.Label(frame, textvariable=self.detect_summary_var, foreground=self.colors["text_secondary"], wraplength=370).pack(
            fill=tk.X, pady=(3, 0)
        )

    def _build_edit_card(self, parent) -> None:
        frame = self._card(parent, "📍 当前帧手动校对")
        self.mode_var = tk.StringVar(value="add")
        mode_row = ttk.Frame(frame)
        mode_row.pack(fill=tk.X)
        ttk.Radiobutton(mode_row, text="➕ 添加 [A]", variable=self.mode_var, value="add", command=self._set_mode).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Radiobutton(mode_row, text="➖ 删除 [D]", variable=self.mode_var, value="delete", command=self._set_mode).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Radiobutton(
            mode_row, text="📐 校准 [C]", variable=self.mode_var, value="calibrate", command=self._set_mode
        ).pack(side=tk.LEFT, padx=3)
        button_row = ttk.Frame(frame)
        button_row.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(button_row, text="↩ 撤销 [Ctrl+Z]", command=self.undo, style="Outline.TButton").pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(button_row, text="🗑 清空当前帧", command=self.clear_current_frame, style="Danger.TButton").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0)
        )
        ttk.Label(
            frame,
            text="左键按当前模式操作；右键可直接删除最近原子。手动操作只影响当前帧。",
            foreground=self.colors["text_secondary"],
            wraplength=370,
        ).pack(fill=tk.X, pady=(4, 0))

    def _build_intensity_card(self, parent) -> None:
        frame = self._card(parent, "⚛ 原子柱局部积分强度")
        grid = ttk.Frame(frame)
        grid.pack(fill=tk.X)
        self.aperture_var = tk.StringVar(value="2.5")
        self.bg_inner_var = tk.StringVar(value="3.0")
        self.bg_outer_var = tk.StringVar(value="4.0")
        self.match_distance_var = tk.StringVar(value="4.0")
        fields = [
            ("积分孔径半径 (px)", self.aperture_var),
            ("背景环内半径 (px)", self.bg_inner_var),
            ("背景环外半径 (px)", self.bg_outer_var),
            ("跨帧 ID 距离 (px)", self.match_distance_var),
        ]
        for row, (label, variable) in enumerate(fields):
            ttk.Label(grid, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(grid, textvariable=variable, width=10).grid(row=row, column=1, sticky="e", pady=2, padx=(8, 0))
        grid.columnconfigure(0, weight=1)
        buttons = ttk.Frame(frame)
        buttons.pack(fill=tk.X, pady=(5, 0))
        self.btn_recalc_current = ttk.Button(
            buttons, text="重算当前帧", command=lambda: self.recalculate_intensities(False), style="Outline.TButton"
        )
        self.btn_recalc_current.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.btn_recalc_all = ttk.Button(
            buttons, text="重算全部帧", command=lambda: self.recalculate_intensities(True), style="Outline.TButton"
        )
        self.btn_recalc_all.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))
        self.intensity_summary_var = tk.StringVar(value="当前帧：尚无原子强度")
        ttk.Label(frame, textvariable=self.intensity_summary_var, foreground=self.colors["text_secondary"], wraplength=370).pack(
            fill=tk.X, pady=(4, 0)
        )
        ttk.Label(
            frame,
            text="强度 = 圆孔径内像素相对局部背景环中位数的积分；亮/暗极性与检测参数同步。",
            foreground=self.colors["text_secondary"],
            wraplength=370,
        ).pack(fill=tk.X, pady=(2, 0))

    def _build_rule_card(self, parent) -> None:
        frame = self._card(parent, "🎨 单帧百分位标记规则")
        columns = ("range", "color", "shape", "visible")
        self.rule_tree = ttk.Treeview(frame, columns=columns, show="headings", height=4, selectmode="browse")
        for column, label, width in [
            ("range", "百分位范围", 105),
            ("color", "颜色", 75),
            ("shape", "形状", 65),
            ("visible", "显示", 45),
        ]:
            self.rule_tree.heading(column, text=label)
            self.rule_tree.column(column, width=width, anchor=tk.CENTER, stretch=True)
        self.rule_tree.pack(fill=tk.X)
        self.rule_tree.bind("<Double-1>", lambda _event: self.edit_rule())
        buttons = ttk.Frame(frame)
        buttons.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(buttons, text="＋", width=3, command=self.add_rule, style="Outline.TButton").pack(side=tk.LEFT)
        ttk.Button(buttons, text="编辑", command=self.edit_rule, style="Outline.TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(buttons, text="显示/隐藏", command=self.toggle_rule, style="Outline.TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(buttons, text="删除", command=self.delete_rule, style="Danger.TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(buttons, text="↑", width=3, command=lambda: self.move_rule(-1), style="Outline.TButton").pack(side=tk.RIGHT)
        ttk.Button(buttons, text="↓", width=3, command=lambda: self.move_rule(1), style="Outline.TButton").pack(side=tk.RIGHT, padx=2)

        display_row = ttk.Frame(frame)
        display_row.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(display_row, text="标记大小").pack(side=tk.LEFT)
        self.marker_radius_var = tk.DoubleVar(value=self.marker_radius)
        ttk.Scale(display_row, from_=3, to=15, variable=self.marker_radius_var, command=self._update_marker_radius).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5
        )
        self.marker_radius_label = ttk.Label(display_row, text="7.0 px", width=7)
        self.marker_radius_label.pack(side=tk.LEFT)
        self.show_ids_var = tk.BooleanVar(value=self.show_ids)
        ttk.Checkbutton(frame, text="显示跨帧原子 ID", variable=self.show_ids_var, command=self._toggle_ids).pack(
            anchor=tk.W, pady=(3, 0)
        )
        ttk.Label(
            frame,
            text="百分位在每一帧内独立计算；低百分位代表较弱原子。重叠规则按列表从上到下首个命中生效。",
            foreground=self.colors["text_secondary"],
            wraplength=370,
        ).pack(fill=tk.X, pady=(3, 0))
        self._refresh_rule_tree()

    def _build_point_card(self, parent) -> None:
        frame = self._card(parent, "📋 当前帧原子位置与强度")
        columns = ("id", "x", "y", "intensity", "percentile", "truncated", "neighbor", "nn_distance", "origin")
        self.point_tree = ttk.Treeview(frame, columns=columns, show="headings", height=7, selectmode="browse")
        specifications = [
            ("id", "ID", 40),
            ("x", "x", 50),
            ("y", "y", 50),
            ("intensity", "积分强度", 70),
            ("percentile", "百分位", 48),
            ("truncated", "截断", 36),
            ("neighbor", "邻居", 36),
            ("nn_distance", "最近邻", 46),
            ("origin", "来源", 42),
        ]
        for column, label, width in specifications:
            self.point_tree.heading(column, text=label)
            self.point_tree.column(column, width=width, anchor=tk.CENTER, stretch=True)
        point_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.point_tree.yview)
        self.point_tree.configure(yscrollcommand=point_scroll.set)
        self.point_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        point_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.point_tree.bind("<<TreeviewSelect>>", self._on_point_selected)

    def _build_export_card(self, parent) -> None:
        frame = self._card(parent, "💾 保存与导出")
        self.btn_save_marked = ttk.Button(
            frame, text="🖼 保存当前帧标记图", command=self.save_current_marked_image, style="Accent.TButton"
        )
        self.btn_save_marked.pack(fill=tk.X, pady=2)
        self.btn_export_stack = ttk.Button(
            frame, text="🎞 导出带标记堆栈 TIFF", command=self.export_marked_stack, style="Success.TButton"
        )
        self.btn_export_stack.pack(fill=tk.X, pady=2)
        self.btn_export_csv = ttk.Button(
            frame, text="📄 导出坐标、ID 与强度 CSV", command=self.export_csv, style="Outline.TButton"
        )
        self.btn_export_csv.pack(fill=tk.X, pady=2)
        session_row = ttk.Frame(frame)
        session_row.pack(fill=tk.X, pady=2)
        self.btn_save_session = ttk.Button(
            session_row, text="💾 保存会话", command=self.save_session, style="Outline.TButton"
        )
        self.btn_save_session.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.btn_load_session = ttk.Button(
            session_row, text="📂 打开会话", command=self.open_session, style="Outline.TButton"
        )
        self.btn_load_session.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))
        ttk.Label(
            frame,
            text="会话保存参数、选区、标记规则与全部逐帧原子（含手工修订）；重新打开需能访问原始 TIFF。",
            foreground=self.colors["text_secondary"],
            wraplength=370,
        ).pack(fill=tk.X, pady=(2, 0))

    # ------------------------------------------------------------------
    # 图像加载、浏览与绘制
    # ------------------------------------------------------------------
    def _bind_shortcuts(self) -> None:
        self._shortcut_handlers = {
            "<KeyPress-a>": lambda _event: self._shortcut_mode("add"),
            "<KeyPress-d>": lambda _event: self._shortcut_mode("delete"),
            "<KeyPress-c>": lambda _event: self._shortcut_calibrate(),
            "<Control-z>": lambda _event: self._shortcut_undo(),
            "<Escape>": lambda _event: self._shortcut_escape(),
            "<Return>": lambda _event: self._shortcut_return(),
        }
        for sequence, handler in self._shortcut_handlers.items():
            self.root.bind(sequence, handler)

    def _typing_focused(self) -> bool:
        """焦点在文本输入控件上时，单键快捷键不能劫持输入（ttk.Entry/Spinbox 都继承 tk.Entry）。"""
        focus = self.root.focus_get()
        if focus is None:
            return False
        return isinstance(focus, (tk.Entry, tk.Text)) or isinstance(focus, ttk.Combobox)

    def _shortcut_mode(self, mode: str) -> None:
        if self._typing_focused():
            return
        self.mode_var.set(mode)
        self._set_mode()

    def _shortcut_calibrate(self) -> None:
        if self._typing_focused():
            return
        self.start_calibration()

    def _shortcut_undo(self) -> None:
        if self._typing_focused():
            return
        self.undo()

    def _shortcut_escape(self) -> None:
        if self._typing_focused():
            return
        self.cancel_transient_mode()

    def _shortcut_return(self) -> None:
        if self._typing_focused():
            return
        self._handle_enter()

    def load_tiff(self) -> None:
        if self._busy:
            messagebox.showinfo("任务进行中", "请先等待当前任务完成或取消任务。")
            return
        if any(self.frame_records) and not messagebox.askyesno(
            "导入新图像",
            "导入新 TIFF 会清空当前全部逐帧识别与手工修订结果。请先导出需要的数据。是否继续？",
        ):
            return
        path = filedialog.askopenfilename(
            title="选择 TIFF 或堆栈 TIFF",
            filetypes=[("TIFF 文件", "*.tif *.tiff"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            stack, info = load_tiff_stack(path)
        except Exception as error:
            messagebox.showerror("导入失败", str(error))
            return
        self._install_stack(stack, info)
        self.status.config(text=f"✓ 已导入 {info.stack_shape[0]} 帧 TIFF；可校准后逐帧识别")

    def _reset_frame_state(self, frame_count: int) -> None:
        """统一重置全部逐帧状态；换栈、加载会话与测试都必须走这里。

        历史上多个平行的逐帧列表只在 load_tiff 中同步初始化，任何绕过
        load_tiff 的调用方都会造成长度不一致（曾导致测试 IndexError）。
        """
        self.current_frame = 0
        self.frame_records = [[] for _ in range(frame_count)]
        self.frame_processed = [False for _ in range(frame_count)]
        self.frame_regions = [None for _ in range(frame_count)]
        self.frame_detection_params = [None for _ in range(frame_count)]
        self.frame_intensity_params = [None for _ in range(frame_count)]
        self.frame_edited = [False for _ in range(frame_count)]
        self.stack_region = None
        self.undo_stack.clear()
        self.selected_atom_id = None
        self._calibration_points.clear()
        self._rectangle_start = None
        self._polygon_vertices.clear()
        self.current_mode = "add"
        if hasattr(self, "mode_var"):
            self.mode_var.set("add")
        self._display_cache = None

    def _install_stack(self, stack: np.ndarray, info) -> None:
        """替换当前堆栈并刷新界面；供导入 TIFF 与加载会话共用。"""
        close_tiff_stack(self.stack)
        if self.stack_info is not None and getattr(self.stack_info, "source", None) is not None:
            close_tiff_stack(self.stack_info.source)
        self.stack = stack
        self.stack_info = info
        self._reset_frame_state(int(info.stack_shape[0]))
        self.frame_scale.configure(from_=0, to=max(0, int(info.stack_shape[0]) - 1))
        self.frame_scale.set(0)
        memory_text = "内存映射" if info.memory_mapped else "已载入内存"
        self.file_info_var.set(
            f"{os.path.basename(info.path)}\n"
            f"{info.stack_shape[0]} 帧 · {info.stack_shape[2]}×{info.stack_shape[1]} · {info.dtype} · {memory_text}"
        )
        self._image_artist = None
        self.show_frame(reset_view=True)

    def on_close(self) -> None:
        edited_frames = [index + 1 for index, edited in enumerate(self.frame_edited) if edited]
        if edited_frames and not self._shutdown_requested:
            preview = ", ".join(map(str, edited_frames[:8]))
            if len(edited_frames) > 8:
                preview += "…"
            answer = messagebox.askyesnocancel(
                "退出确认",
                f"有 {len(edited_frames)} 帧包含未保存的手工修订（帧 {preview}）。\n\n"
                "是：保存会话后退出；否：直接退出（修订将丢失）；取消：返回程序。",
            )
            if answer is None:
                return
            if answer and not self.save_session():
                return  # 用户在保存对话框中取消，视为放弃退出
        self._shutdown()

    def _shutdown(self) -> None:
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        self._cancel_event.set()
        self._poll_active = False
        close_tiff_stack(self.stack)
        if self.stack_info is not None and getattr(self.stack_info, "source", None) is not None:
            close_tiff_stack(self.stack_info.source)
        if getattr(self, "_poll_after_id", None) is not None:
            try:
                self.root.after_cancel(self._poll_after_id)
            except tk.TclError:
                pass
            self._poll_after_id = None
        self.root.destroy()

    def _frame_array(self, frame_index: int | None = None) -> np.ndarray:
        if self.stack is None:
            raise AtomicToolError("请先导入 TIFF 图像。")
        index = self.current_frame if frame_index is None else int(frame_index)
        return np.asarray(self.stack[index])

    def _normalized_frame(self, frame_index: int) -> np.ndarray:
        """带单帧缓存的 1%–99% 显示归一化；帧数据不变时可安全复用。"""
        if self._display_cache is not None and self._display_cache[0] == int(frame_index):
            return self._display_cache[1]
        if self.stack is None:
            raise AtomicToolError("请先导入 TIFF 图像。")
        display = normalize_for_display(np.asarray(self.stack[frame_index]))
        self._display_cache = (int(frame_index), display)
        return display

    def _work_frame(self, frame_index: int) -> np.ndarray:
        """检测用图：归一化后按当前极性处理（取负会新建数组，不污染缓存）。"""
        work = self._normalized_frame(frame_index)
        if not self.detection_params.bright:
            return -work
        return work

    def _on_frame_scale(self, value: str) -> None:
        if self.stack is None:
            return
        target = int(round(float(value)))
        target = int(np.clip(target, 0, len(self.frame_records) - 1))
        if target == self.current_frame:
            return
        self._cancel_region_drawing(switch_to_add=True)
        self.current_frame = target
        self.selected_atom_id = None
        self._schedule_frame_refresh()

    def _schedule_frame_refresh(self) -> None:
        """拖动帧滑条时防抖重绘，避免每个中间刻度都全量刷新。"""
        if self._frame_refresh_after_id is not None:
            try:
                self.root.after_cancel(self._frame_refresh_after_id)
            except tk.TclError:
                pass
        self._frame_refresh_after_id = self.root.after(REDRAW_DEBOUNCE_MS, self._debounced_frame_refresh)

    def _debounced_frame_refresh(self) -> None:
        self._frame_refresh_after_id = None
        self.show_frame(reset_view=False)

    def _change_frame(self, step: int) -> None:
        if self.stack is None:
            return
        target = int(np.clip(self.current_frame + step, 0, len(self.frame_records) - 1))
        if target != self.current_frame:
            self._cancel_region_drawing(switch_to_add=True)
        self.current_frame = target
        self.frame_scale.set(target)
        self.selected_atom_id = None
        self.show_frame(reset_view=False)

    def _clear_overlay_artists(self) -> None:
        for artist in self._overlay_artists + self._label_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._overlay_artists.clear()
        self._label_artists.clear()
        self._region_preview_artist = None

    def show_frame(self, reset_view: bool = False) -> None:
        if self.stack is None:
            self.axes.clear()
            self.axes.set_facecolor("#fafafa")
            self.axes.set_title("请导入 TIFF 图像")
            self.canvas.draw_idle()
            return

        display = self._normalized_frame(self.current_frame)
        height, width = display.shape
        old_xlim = self.axes.get_xlim()
        old_ylim = self.axes.get_ylim()
        if self._image_artist is None:
            self.axes.clear()
            self.axes.set_facecolor("#fafafa")
            self._image_artist = self.axes.imshow(display, cmap="gray", origin="upper", aspect="equal", vmin=0, vmax=1)
            reset_view = True
        else:
            self._image_artist.set_data(display)
            self._image_artist.set_clim(0, 1)
        if reset_view:
            self.axes.set_xlim(0, width)
            self.axes.set_ylim(height, 0)
        else:
            self.axes.set_xlim(old_xlim)
            self.axes.set_ylim(old_ylim)
        self.axes.tick_params(colors="#616161", labelsize=8)
        for spine in self.axes.spines.values():
            spine.set_color("#bdbdbd")
        self._draw_overlays()
        processed = "已识别" if self.frame_processed[self.current_frame] else "未识别"
        self.axes.set_title(
            f"帧 {self.current_frame + 1}/{len(self.frame_records)} · {processed} · "
            f"原子 {len(self.frame_records[self.current_frame])}"
        )
        self.frame_label_var.set(f"帧 {self.current_frame + 1} / {len(self.frame_records)}")
        self._update_region_summary()
        self._refresh_point_tree()
        self._refresh_intensity_summary()
        self.canvas.draw_idle()

    def _draw_overlays(self) -> None:
        self._clear_overlay_artists()
        if self.stack is None:
            return
        region = self._active_region()
        if region is not None:
            edge_color = "#00e5ff" if self.selection_scope_var.get() == "all" else "#ffb300"
            if region.kind == "rectangle":
                (x0, y0), (x1, y1) = region.vertices[:2]
                x_min, x_max = sorted((x0, x1))
                y_min, y_max = sorted((y0, y1))
                patch = Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    fill=True,
                    facecolor=edge_color,
                    edgecolor=edge_color,
                    alpha=0.13,
                    linewidth=2.0,
                    linestyle="--",
                    zorder=3,
                )
            else:
                patch = Polygon(
                    np.asarray(region.vertices),
                    closed=True,
                    fill=True,
                    facecolor=edge_color,
                    edgecolor=edge_color,
                    alpha=0.13,
                    linewidth=2.0,
                    linestyle="--",
                    zorder=3,
                )
            self.axes.add_patch(patch)
            self._overlay_artists.append(patch)
        records = self.frame_records[self.current_frame]
        visible = visible_records(records, self.rules)
        for rule in self.rules:
            group = [record for record, matched_rule in visible if matched_rule is rule]
            if not group:
                continue
            points = np.array([(record.x, record.y) for record in group], dtype=np.float64)
            marker = MPL_MARKERS.get(rule.marker, "o")
            size = (self.marker_radius * 2.0) ** 2
            if rule.marker in {"cross", "plus", "star"}:
                artist = self.axes.scatter(
                    points[:, 0], points[:, 1], s=size, marker=marker, c=rule.color, linewidths=1.5, zorder=4
                )
            else:
                artist = self.axes.scatter(
                    points[:, 0],
                    points[:, 1],
                    s=size,
                    marker=marker,
                    facecolors="none",
                    edgecolors=rule.color,
                    linewidths=1.6,
                    zorder=4,
                )
            self._overlay_artists.append(artist)

        truncated_points = np.array(
            [
                (record.x, record.y)
                for record in records
                if record.truncated and np.isfinite(record.integrated_intensity)
            ],
            dtype=np.float64,
        )
        if len(truncated_points):
            artist = self.axes.scatter(
                truncated_points[:, 0],
                truncated_points[:, 1],
                s=(self.marker_radius * 2.6) ** 2,
                marker="o",
                facecolors="none",
                edgecolors="#9e9e9e",
                linewidths=0.9,
                linestyle="--",
                zorder=5,
            )
            self._overlay_artists.append(artist)

        if self.selected_atom_id is not None:
            selected = next((record for record in records if record.atom_id == self.selected_atom_id), None)
            if selected is not None:
                artist = self.axes.scatter(
                    [selected.x], [selected.y], s=(self.marker_radius * 2.8) ** 2, marker="o",
                    facecolors="none", edgecolors="#ffeb3b", linewidths=2.2, zorder=6
                )
                self._overlay_artists.append(artist)

        if self.show_ids:
            label_records = [record for record, _rule in visible]
            if len(label_records) <= self.MAX_DRAWN_LABELS:
                for record in label_records:
                    text = self.axes.text(
                        record.x + self.marker_radius,
                        record.y - self.marker_radius,
                        str(record.atom_id),
                        color="white",
                        fontsize=7,
                        ha="left",
                        va="bottom",
                        fontweight="bold",
                        zorder=7,
                        path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
                    )
                    self._label_artists.append(text)

        for index, (x, y) in enumerate(self._calibration_points):
            artist = self.axes.scatter(
                [x], [y], s=130, marker="o", facecolors="none", edgecolors="#00b0ff", linewidths=2.0, zorder=8
            )
            self._overlay_artists.append(artist)
            label = self.axes.text(x + 4, y - 4, str(index + 1), color="#00b0ff", fontsize=9, fontweight="bold", zorder=9)
            self._label_artists.append(label)

    def _active_region(self) -> DetectionRegion | None:
        if self.stack is None:
            return None
        if self.selection_scope_var.get() == "all":
            return self.stack_region
        if 0 <= self.current_frame < len(self.frame_regions):
            return self.frame_regions[self.current_frame]
        return None

    def _set_active_region(self, region: DetectionRegion | None) -> None:
        if self.stack is None:
            return
        if self.selection_scope_var.get() == "all":
            self.stack_region = region
        else:
            self.frame_regions[self.current_frame] = region
        self._update_region_summary()

    def _region_for_detection(self, frame_index: int, all_frames_job: bool) -> DetectionRegion | None:
        scope = self.selection_scope_var.get()
        if all_frames_job:
            return self.stack_region if scope == "all" else None
        if scope == "all":
            return self.stack_region
        return self.frame_regions[frame_index] if 0 <= frame_index < len(self.frame_regions) else None

    def _update_region_summary(self) -> None:
        if not hasattr(self, "region_summary_var"):
            return
        region = self._active_region()
        scope_text = "逐帧全部共用" if self.selection_scope_var.get() == "all" else f"仅帧 {self.current_frame + 1}"
        if region is None:
            self.region_summary_var.set(f"{scope_text}：当前没有选区")
            return
        if region.kind == "rectangle":
            (x0, y0), (x1, y1) = region.vertices[:2]
            self.region_summary_var.set(
                f"{scope_text}：矩形 {abs(x1 - x0):.0f}×{abs(y1 - y0):.0f} px"
            )
        else:
            self.region_summary_var.set(f"{scope_text}：自由选区 {len(region.vertices)} 个顶点")

    def _on_selection_scope_changed(self) -> None:
        self._cancel_region_drawing(switch_to_add=True)
        self._update_region_summary()
        if self.stack is not None:
            self._draw_overlays()
            self.canvas.draw_idle()

    def start_rectangle_selection(self) -> None:
        if self.stack is None:
            messagebox.showinfo("提示", "请先导入 TIFF 图像。")
            return
        if self._busy:
            return
        self._cancel_region_drawing(switch_to_add=False)
        self._calibration_points.clear()
        self.current_mode = "draw_rectangle"
        self.mode_var.set("add")
        self.btn_rectangle.configure(text="⬜ 请拖拽矩形")
        scope = "整个堆栈" if self.selection_scope_var.get() == "all" else f"当前帧 {self.current_frame + 1}"
        self.status.config(text=f"矩形选区：在图像上按住左键拖拽；选区将用于{scope}的自动识别")

    def start_polygon_selection(self) -> None:
        if self.stack is None:
            messagebox.showinfo("提示", "请先导入 TIFF 图像。")
            return
        if self._busy:
            return
        if self.current_mode == "draw_polygon" and len(self._polygon_vertices) >= 3:
            self.finish_polygon_selection()
            return
        self._cancel_region_drawing(switch_to_add=False)
        self._calibration_points.clear()
        self.current_mode = "draw_polygon"
        self.mode_var.set("add")
        self.btn_polygon.configure(text="完成自由选区")
        scope = "整个堆栈" if self.selection_scope_var.get() == "all" else f"当前帧 {self.current_frame + 1}"
        self.status.config(
            text=f"自由选区：逐点左击，双击/Enter/再次点“完成”闭合；右键撤销顶点；将用于{scope}"
        )

    def clear_active_region(self) -> None:
        if self.stack is None or self._busy:
            return
        self._cancel_region_drawing(switch_to_add=True)
        self._set_active_region(None)
        self._draw_overlays()
        self.canvas.draw_idle()
        scope = "逐帧全部" if self.selection_scope_var.get() == "all" else f"帧 {self.current_frame + 1}"
        self.status.config(text=f"✓ 已清除{scope}选区")

    def _remove_region_preview(self) -> None:
        if self._region_preview_artist is not None:
            try:
                self._region_preview_artist.remove()
            except Exception:
                pass
            try:
                self._overlay_artists.remove(self._region_preview_artist)
            except ValueError:
                pass
            self._region_preview_artist = None

    def _cancel_region_drawing(self, switch_to_add: bool) -> None:
        self._rectangle_start = None
        self._polygon_vertices.clear()
        self._remove_region_preview()
        if hasattr(self, "btn_rectangle"):
            self.btn_rectangle.configure(text="⬜ 矩形选区")
            self.btn_polygon.configure(text="✏ 自由选区")
        if switch_to_add and self.current_mode in {"draw_rectangle", "draw_polygon"}:
            self.current_mode = "add"
            self.mode_var.set("add")

    def _clip_point_to_image(self, x: float, y: float) -> tuple[float, float]:
        height, width = self._frame_array().shape
        return float(np.clip(x, 0, width - 1)), float(np.clip(y, 0, height - 1))

    def _update_polygon_preview(self, cursor: tuple[float, float] | None = None) -> None:
        self._remove_region_preview()
        if not self._polygon_vertices:
            self.canvas.draw_idle()
            return
        points = list(self._polygon_vertices)
        if cursor is not None:
            points.append(cursor)
        array = np.asarray(points, dtype=np.float64)
        line = self.axes.plot(
            array[:, 0],
            array[:, 1],
            color="#00e5ff",
            marker="o",
            markersize=4,
            linewidth=1.6,
            linestyle="-",
            zorder=9,
        )[0]
        self._region_preview_artist = line
        self._overlay_artists.append(line)
        self.canvas.draw_idle()

    def finish_polygon_selection(self) -> None:
        if self.current_mode != "draw_polygon":
            return
        vertices: list[tuple[float, float]] = []
        for point in self._polygon_vertices:
            clipped = self._clip_point_to_image(*point)
            if not vertices or np.linalg.norm(np.asarray(clipped) - np.asarray(vertices[-1])) > 0.5:
                vertices.append(clipped)
        if len(vertices) < 3:
            messagebox.showinfo("自由选区", "自由选区至少需要 3 个不同顶点。")
            return
        try:
            region = DetectionRegion("polygon", tuple(vertices)).validated()
        except Exception as error:
            messagebox.showerror("选区无效", str(error))
            return
        self._set_active_region(region)
        self._cancel_region_drawing(switch_to_add=True)
        self._draw_overlays()
        self.canvas.draw_idle()
        self.status.config(text=f"✓ 自由选区已完成：{len(vertices)} 个顶点")

    def _handle_enter(self) -> None:
        if self.current_mode == "draw_polygon":
            self.finish_polygon_selection()
        elif self.current_mode == "calibrate":
            self.finish_calibration()

    def _on_mouse_move(self, event) -> None:
        if self.stack is None or self._busy or event.inaxes != self.axes or event.xdata is None or event.ydata is None:
            return
        x, y = self._clip_point_to_image(float(event.xdata), float(event.ydata))
        if self.current_mode == "draw_rectangle" and self._rectangle_start is not None:
            if isinstance(self._region_preview_artist, Rectangle):
                x0, y0 = self._rectangle_start
                self._region_preview_artist.set_xy((min(x0, x), min(y0, y)))
                self._region_preview_artist.set_width(abs(x - x0))
                self._region_preview_artist.set_height(abs(y - y0))
                self.canvas.draw_idle()
        elif self.current_mode == "draw_polygon" and self._polygon_vertices:
            self._update_polygon_preview((x, y))

    def _on_image_release(self, event) -> None:
        if (
            self.stack is None
            or self._busy
            or self.current_mode != "draw_rectangle"
            or self._rectangle_start is None
            or getattr(event, "button", None) != 1
            or event.inaxes != self.axes
            or event.xdata is None
            or event.ydata is None
        ):
            return
        end = self._clip_point_to_image(float(event.xdata), float(event.ydata))
        start = self._rectangle_start
        if abs(end[0] - start[0]) < 3 or abs(end[1] - start[1]) < 3:
            self._cancel_region_drawing(switch_to_add=True)
            self._draw_overlays()
            self.canvas.draw_idle()
            messagebox.showinfo("矩形选区", "矩形选区宽度和高度至少需要 3 px。")
            return
        region = DetectionRegion("rectangle", (start, end)).validated()
        self._set_active_region(region)
        self._cancel_region_drawing(switch_to_add=True)
        self._draw_overlays()
        self.canvas.draw_idle()
        self.status.config(text=f"✓ 矩形选区已完成：{abs(end[0]-start[0]):.0f}×{abs(end[1]-start[1]):.0f} px")

    def _on_scroll(self, event) -> None:
        if self.stack is None or event.inaxes != self.axes or event.xdata is None or event.ydata is None:
            return
        scale = 0.85 if event.button == "up" else 1.0 / 0.85
        xlim, ylim = self.axes.get_xlim(), self.axes.get_ylim()
        x, y = event.xdata, event.ydata
        self.axes.set_xlim(x - (x - xlim[0]) * scale, x + (xlim[1] - x) * scale)
        self.axes.set_ylim(y - (y - ylim[0]) * scale, y + (ylim[1] - y) * scale)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # 手动添加、删除、撤销与选择
    # ------------------------------------------------------------------
    def _set_mode(self) -> None:
        mode = self.mode_var.get()
        if mode == "calibrate":
            self.start_calibration()
            return
        self._cancel_region_drawing(switch_to_add=False)
        self.current_mode = mode
        self._calibration_points.clear()
        hint = "左键添加原子，右键删除" if mode == "add" else "左键或右键删除最近原子"
        self.status.config(text=f"模式：{hint}  |  [A]添加 [D]删除 [C]校准 [Esc]取消")
        self._draw_overlays()
        self.canvas.draw_idle()

    def cancel_transient_mode(self) -> None:
        self._cancel_region_drawing(switch_to_add=False)
        self._calibration_points.clear()
        self.current_mode = "add"
        self.mode_var.set("add")
        self.status.config(text="已取消临时操作，回到添加模式")
        self._draw_overlays()
        self.canvas.draw_idle()

    def _save_undo_snapshot(self) -> None:
        if self.stack is None:
            return
        snapshot = [record.copy() for record in self.frame_records[self.current_frame]]
        self.undo_stack.append(
            (
                self.current_frame,
                snapshot,
                self.frame_processed[self.current_frame],
                self.frame_edited[self.current_frame],
                self.frame_intensity_params[self.current_frame],
            )
        )
        if len(self.undo_stack) > UNDO_HISTORY_LIMIT:
            self.undo_stack.pop(0)

    def _on_image_click(self, event) -> None:
        if self.stack is None or self._busy or event.inaxes != self.axes or event.xdata is None or event.ydata is None:
            return
        x, y = self._clip_point_to_image(float(event.xdata), float(event.ydata))

        if self.current_mode == "draw_rectangle":
            if event.button == 3:
                self.cancel_transient_mode()
                return
            if event.button != 1:
                return
            self._rectangle_start = (x, y)
            self._remove_region_preview()
            preview = Rectangle(
                (x, y),
                0,
                0,
                fill=True,
                facecolor="#00e5ff",
                edgecolor="#00e5ff",
                alpha=0.18,
                linewidth=1.8,
                linestyle="--",
                zorder=9,
            )
            self.axes.add_patch(preview)
            self._region_preview_artist = preview
            self._overlay_artists.append(preview)
            self.canvas.draw_idle()
            return

        if self.current_mode == "draw_polygon":
            if event.button == 3:
                if self._polygon_vertices:
                    self._polygon_vertices.pop()
                    self._update_polygon_preview()
                    self.status.config(text=f"自由选区：已选择 {len(self._polygon_vertices)} 个顶点")
                return
            if event.button != 1:
                return
            self._polygon_vertices.append((x, y))
            if getattr(event, "dblclick", False) and len(self._polygon_vertices) >= 3:
                self.finish_polygon_selection()
            else:
                self._update_polygon_preview()
                self.status.config(
                    text=f"自由选区：已选择 {len(self._polygon_vertices)} 个顶点；双击/Enter/再次点“完成”闭合"
                )
            return

        if event.button == 3:
            self._delete_nearest(x, y)
            return
        if event.button != 1:
            return
        if self.current_mode == "calibrate":
            if len(self._calibration_points) >= 4:
                self.status.config(text="校准点已达 4 个，请按 Enter 确认或 Esc 取消")
                return
            self._calibration_points.append((x, y))
            self.status.config(text=f"校准：已选 {len(self._calibration_points)} 个点；选 2–4 个后按 Enter")
            self._draw_overlays()
            self.canvas.draw_idle()
        elif self.current_mode == "delete":
            self._delete_nearest(x, y)
        else:
            self._add_point(x, y)

    def _all_known_ids(self) -> set[int]:
        return {record.atom_id for records in self.frame_records for record in records}

    def _add_point(self, x: float, y: float) -> None:
        # 卡片输入框是强度参数的唯一真相源：任何改动强度的操作前都先解析，
        # 避免手动加点沿用上次识别/重算的过期孔径。
        try:
            self.intensity_params, self.match_distance = self._parse_intensity_controls()
        except Exception as error:
            messagebox.showerror("参数错误", str(error))
            return
        try:
            # 检测用工作图走单帧缓存，避免每次点击都重算全帧分位数。
            refined_x, refined_y = refine_point_on_work(
                self._work_frame(self.current_frame), x, y, self.detection_params
            )
            records = self.frame_records[self.current_frame]
            if records:
                distances = np.hypot(
                    np.array([record.x for record in records]) - refined_x,
                    np.array([record.y for record in records]) - refined_y,
                )
                if distances.min(initial=np.inf) < max(1.0, self.detection_params.min_distance * 0.35):
                    messagebox.showinfo("重复位置", "点击位置附近已有原子点，未重复添加。")
                    return
            self._save_undo_snapshot()
            current_ids = {record.atom_id for record in records}
            catalog = {
                atom_id: point
                for atom_id, point in build_track_catalog(self.frame_records, self.current_frame).items()
                if atom_id not in current_ids
            }
            new_point = np.array([[refined_x, refined_y]], dtype=np.float64)
            next_id = max(self._all_known_ids(), default=0) + 1
            atom_id = int(match_points_to_catalog(new_point, catalog, self.match_distance, next_id)[0])
            records.append(AtomRecord(atom_id, refined_x, refined_y, source="manual"))
            self.frame_records[self.current_frame] = recalculate_record_intensities(
                self._frame_array(), records, self.intensity_params
            )
            self.frame_processed[self.current_frame] = True
            self.frame_edited[self.current_frame] = True
            self.frame_intensity_params[self.current_frame] = self.intensity_params
            self.selected_atom_id = atom_id
            self.show_frame(reset_view=False)
            self.status.config(text=f"✓ 当前帧添加原子 ID {atom_id}：({refined_x:.2f}, {refined_y:.2f})")
        except Exception as error:
            _log_exception("手动添加原子失败", error)
            messagebox.showerror("添加失败", str(error))

    def _delete_nearest(self, x: float, y: float) -> None:
        records = self.frame_records[self.current_frame] if self.stack is not None else []
        if not records:
            return
        distances = np.hypot(
            np.array([record.x for record in records]) - x,
            np.array([record.y for record in records]) - y,
        )
        index = int(np.argmin(distances))
        if distances[index] > self.pick_radius:
            self.status.config(text="点击位置附近没有可删除的原子点")
            return
        try:
            self.intensity_params, self.match_distance = self._parse_intensity_controls()
        except Exception as error:
            messagebox.showerror("参数错误", str(error))
            return
        self._save_undo_snapshot()
        deleted = records.pop(index)
        self.frame_records[self.current_frame] = recalculate_record_intensities(
            self._frame_array(), records, self.intensity_params
        )
        self.frame_processed[self.current_frame] = True
        self.frame_edited[self.current_frame] = True
        self.frame_intensity_params[self.current_frame] = self.intensity_params
        if self.selected_atom_id == deleted.atom_id:
            self.selected_atom_id = None
        self.show_frame(reset_view=False)
        self.status.config(text=f"✓ 仅从当前帧删除原子 ID {deleted.atom_id}")

    def undo(self) -> None:
        if not self.undo_stack or self._busy:
            self.status.config(text="没有可撤销的当前帧编辑")
            return
        frame_index, records, processed, edited, intensity_params = self.undo_stack.pop()
        self.frame_records[frame_index] = [record.copy() for record in records]
        self.frame_processed[frame_index] = processed
        self.frame_edited[frame_index] = edited
        if intensity_params is not None and frame_index < len(self.frame_intensity_params):
            # 快照里的强度值是用当时的参数算的，溯源标记必须一起回滚。
            self.frame_intensity_params[frame_index] = intensity_params
        self.current_frame = frame_index
        self.frame_scale.set(frame_index)
        self.selected_atom_id = None
        self.show_frame(reset_view=False)
        self.status.config(text=f"✓ 已撤销帧 {frame_index + 1} 的上一步编辑")

    def clear_current_frame(self) -> None:
        if self.stack is None or not self.frame_records[self.current_frame] or self._busy:
            return
        if not messagebox.askyesno("清空当前帧", "仅清空当前帧的全部原子点，其他帧不受影响。是否继续？"):
            return
        self._save_undo_snapshot()
        self.frame_records[self.current_frame] = []
        self.frame_processed[self.current_frame] = True
        self.frame_edited[self.current_frame] = True
        self.selected_atom_id = None
        self.show_frame(reset_view=False)
        self.status.config(text=f"✓ 已清空帧 {self.current_frame + 1} 的原子点")

    def _refresh_point_tree(self) -> None:
        for item in self.point_tree.get_children():
            self.point_tree.delete(item)
        if self.stack is None:
            return
        for record in sorted(self.frame_records[self.current_frame], key=lambda item: item.atom_id):
            intensity = "NaN" if not np.isfinite(record.integrated_intensity) else f"{record.integrated_intensity:.5g}"
            percentile = "NaN" if not np.isfinite(record.percentile) else f"{record.percentile:.2f}%"
            nearest = "—" if not np.isfinite(record.nearest_neighbor) else f"{record.nearest_neighbor:.2f}"
            item = self.point_tree.insert(
                "",
                tk.END,
                iid=str(record.atom_id),
                values=(
                    record.atom_id,
                    f"{record.x:.3f}",
                    f"{record.y:.3f}",
                    intensity,
                    percentile,
                    "是" if record.truncated else "",
                    "是" if record.neighbor_contaminated else "",
                    nearest,
                    "手动" if record.source == "manual" else "自动",
                ),
            )
            if record.atom_id == self.selected_atom_id:
                self.point_tree.selection_set(item)
                self.point_tree.see(item)

    def _on_point_selected(self, _event=None) -> None:
        selection = self.point_tree.selection()
        if not selection:
            return
        self.selected_atom_id = int(selection[0])
        self._draw_overlays()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # 校准与检测参数
    # ------------------------------------------------------------------
    def start_calibration(self) -> None:
        if self.stack is None:
            messagebox.showinfo("提示", "请先导入 TIFF 图像。")
            return
        if self._busy:
            messagebox.showinfo("任务进行中", "请先等待当前任务完成或取消任务。")
            return
        self._cancel_region_drawing(switch_to_add=False)
        self._calibration_points.clear()
        self.current_mode = "calibrate"
        self.mode_var.set("calibrate")
        self.status.config(text="🔵 校准：点击 2–4 个相邻原子，按 Enter 确认；按 Esc 取消")
        self._draw_overlays()
        self.canvas.draw_idle()

    def finish_calibration(self) -> None:
        if self.current_mode != "calibrate":
            return
        if len(self._calibration_points) < 2:
            messagebox.showinfo("校准", "请至少点击 2 个相邻原子。")
            return
        try:
            result = calibrate_from_points(
                self._frame_array(), self._calibration_points, bright=self.detection_params.bright
            )
        except Exception as error:
            messagebox.showerror("校准失败", str(error))
            return
        det = result.detection_params
        intensity = result.intensity_params
        action = self.show_detection_params(
            initial_detection=det,
            initial_intensity=intensity,
            initial_match_distance=result.match_distance,
            calibration_result=result,
        )
        if action == "recalibrate":
            self._calibration_points.clear()
            self.status.config(text="请重新点击 2–4 个相邻原子，按 Enter 确认")
            self._draw_overlays()
            self.canvas.draw_idle()
            return
        if action not in {"saved", "detect"}:
            self.cancel_transient_mode()
            self.status.config(text="已取消校准参数应用，保留原有参数")
            return
        self._calibration_points.clear()
        self.current_mode = "add"
        self.mode_var.set("add")
        self._draw_overlays()
        self.canvas.draw_idle()
        if action == "detect":
            self.start_detection(False)
        else:
            self.status.config(text="✓ 已应用校准后微调参数；可开始识别当前帧或逐帧识别全部")

    def _sync_parameter_vars(self) -> None:
        self.aperture_var.set(f"{self.intensity_params.aperture_radius:.3g}")
        self.bg_inner_var.set(f"{self.intensity_params.background_inner_radius:.3g}")
        self.bg_outer_var.set(f"{self.intensity_params.background_outer_radius:.3g}")
        self.match_distance_var.set(f"{self.match_distance:.3g}")

    def _update_detection_summary(self) -> None:
        params = self.detection_params
        polarity = "亮原子" if params.bright else "暗原子"
        method = "COM" if params.method == "com" else "2D Gaussian"
        if params.threshold is None:
            threshold = "自适应 P80"
        elif 0 < params.threshold < 1:
            threshold = f"P{params.threshold * 100:g} 分位"
        else:
            threshold = f"归一化绝对值 {params.threshold:g}"
        self.detect_summary_var.set(
            f"参数：间距 {params.min_distance:g} px，σ {params.sigma:g}，{method}，{polarity}，阈值 {threshold}"
        )

    @staticmethod
    def _validated_parameter_bundle(
        *,
        sigma: str,
        min_distance: str,
        window: str,
        threshold: str,
        bright: bool,
        method: str,
        aperture_radius: str,
        background_inner_radius: str,
        background_outer_radius: str,
        match_distance: str,
    ) -> tuple[DetectionParams, IntensityParams, float]:
        threshold_text = threshold.strip()
        try:
            window_value = int(str(window).strip())
        except (TypeError, ValueError):
            raise AtomicToolError(f"质心窗口必须是整数（当前输入：{window!r}）。") from None
        detection = DetectionParams(
            sigma=parse_finite_float(sigma, "高斯滤波 sigma"),
            min_distance=parse_finite_float(min_distance, "最小原子间距"),
            window=window_value,
            threshold=None if not threshold_text else parse_finite_float(threshold_text, "强度阈值"),
            bright=bool(bright),
            method=method,
        ).validated()
        intensity = IntensityParams(
            aperture_radius=parse_finite_float(aperture_radius, "积分孔径半径"),
            background_inner_radius=parse_finite_float(background_inner_radius, "背景环内半径"),
            background_outer_radius=parse_finite_float(background_outer_radius, "背景环外半径"),
            bright=detection.bright,
        ).validated()
        id_distance = parse_finite_float(match_distance, "跨帧 ID 关联距离")
        if id_distance <= 0:
            raise AtomicToolError("跨帧 ID 关联距离必须大于 0。")
        return detection, intensity, id_distance

    def show_detection_params(
        self,
        *,
        initial_detection: DetectionParams | None = None,
        initial_intensity: IntensityParams | None = None,
        initial_match_distance: float | None = None,
        calibration_result: CalibrationResult | None = None,
    ) -> str:
        """打开完整参数窗口；校准流程会把建议值预填后再让用户确认。"""
        start_detection_params = (initial_detection or self.detection_params).validated()
        start_intensity_params = (initial_intensity or self.intensity_params).validated()
        start_match_distance = self.match_distance if initial_match_distance is None else float(initial_match_distance)

        dialog = tk.Toplevel(self.root)
        dialog.title(
            f"校准后参数确认 · v{APP_VERSION}" if calibration_result is not None else f"自动原子识别参数 · v{APP_VERSION}"
        )
        dialog.geometry("620x700" if calibration_result is not None else "620x620")
        dialog.minsize(560, 590)
        dialog.transient(self.root)
        dialog.grab_set()

        sigma_var = tk.StringVar(value=str(start_detection_params.sigma))
        distance_var = tk.StringVar(value=str(start_detection_params.min_distance))
        window_var = tk.StringVar(value=str(start_detection_params.window))
        threshold_var = tk.StringVar(
            value="" if start_detection_params.threshold is None else str(start_detection_params.threshold)
        )
        bright_var = tk.BooleanVar(value=start_detection_params.bright)
        method_var = tk.StringVar(value=start_detection_params.method)
        aperture_var = tk.StringVar(value=str(start_intensity_params.aperture_radius))
        bg_inner_var = tk.StringVar(value=str(start_intensity_params.background_inner_radius))
        bg_outer_var = tk.StringVar(value=str(start_intensity_params.background_outer_radius))
        id_distance_var = tk.StringVar(value=str(start_match_distance))
        result = {"action": "cancel"}

        def restore_opening_values() -> None:
            sigma_var.set(str(start_detection_params.sigma))
            distance_var.set(str(start_detection_params.min_distance))
            window_var.set(str(start_detection_params.window))
            threshold_var.set(
                "" if start_detection_params.threshold is None else str(start_detection_params.threshold)
            )
            bright_var.set(start_detection_params.bright)
            method_var.set(start_detection_params.method)
            aperture_var.set(str(start_intensity_params.aperture_radius))
            bg_inner_var.set(str(start_intensity_params.background_inner_radius))
            bg_outer_var.set(str(start_intensity_params.background_outer_radius))
            id_distance_var.set(str(start_match_distance))

        if calibration_result is not None:
            peak_text = (
                "未可靠测得"
                if not np.isfinite(calibration_result.peak_min)
                else f"{calibration_result.peak_min:.4g} ～ {calibration_result.peak_max:.4g}（显示归一化、扣背景）"
            )
            clicked = len(self._calibration_points)
            if clicked and calibration_result.sample_count < clicked:
                sample_text = (
                    f"点击 {clicked} 点，{calibration_result.sample_count} 点参与统计（贴边或无效 ROI 已剔除）"
                )
            else:
                sample_text = f"参与统计 {calibration_result.sample_count} 点"
            contrast_text = (
                "\n⚠ 校准点峰值与噪声接近，参数建议保守，识别后请人工检查。"
                if calibration_result.low_contrast
                else ""
            )
            summary = ttk.LabelFrame(dialog, text="校准分析与建议", padding=8)
            summary.pack(fill=tk.X, padx=12, pady=(12, 6))
            ttk.Label(
                summary,
                text=(
                    f"{sample_text}  ·  最近邻间距 "
                    f"{calibration_result.nearest_spacing:.2f} px  ·  FWHM "
                    f"{calibration_result.median_fwhm:.2f} px\n"
                    f"校准点峰强度：{peak_text}{contrast_text}\n"
                    "下方已填入校准建议。可逐项修改，再选择仅应用或立即识别当前帧。"
                ),
                wraplength=565,
                justify=tk.LEFT,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Button(
                summary,
                text="恢复校准建议",
                command=restore_opening_values,
                style="Outline.TButton",
            ).pack(side=tk.RIGHT, padx=(8, 0))

        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 4))
        detection_tab = ttk.Frame(notebook, padding=12)
        intensity_tab = ttk.Frame(notebook, padding=12)
        notebook.add(detection_tab, text="识别与亚像素定位")
        notebook.add(intensity_tab, text="积分强度与跨帧 ID")

        preset_var = tk.StringVar(value="标准 HRTEM (默认)")
        ttk.Label(detection_tab, text="参数预设", font=("", 9, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 7)
        )
        preset_row = ttk.Frame(detection_tab)
        preset_row.grid(row=0, column=1, sticky="ew", pady=(0, 7))
        preset_box = ttk.Combobox(
            preset_row,
            textvariable=preset_var,
            values=list(DETECTION_PRESETS),
            state="readonly",
            width=29,
        )
        preset_box.pack(side=tk.LEFT, fill=tk.X, expand=True)

        def apply_preset() -> None:
            preset = DETECTION_PRESETS[preset_var.get()]
            sigma_var.set(str(preset.sigma))
            distance_var.set(str(preset.min_distance))
            window_var.set(str(preset.window))
            threshold_var.set("")
            bright_var.set(preset.bright)
            method_var.set(preset.method)

        ttk.Button(preset_row, text="应用", command=apply_preset, style="Outline.TButton").pack(
            side=tk.RIGHT, padx=(4, 0)
        )
        preset_box.bind("<<ComboboxSelected>>", lambda _event: apply_preset())

        detection_fields = [
            ("最小原子间距 (px)", distance_var),
            ("高斯预滤波 σ", sigma_var),
            ("质心精炼窗口（≥3 的奇数）", window_var),
            ("强度阈值（空=自适应 P80；0–1 之间按分位数 P{值×100} 解释）", threshold_var),
        ]
        for row, (label, variable) in enumerate(detection_fields, start=1):
            ttk.Label(detection_tab, text=label).grid(row=row, column=0, sticky="w", pady=8)
            ttk.Entry(detection_tab, textvariable=variable, width=16).grid(
                row=row, column=1, sticky="e", padx=(8, 0)
            )
        detection_tab.columnconfigure(0, weight=1)
        detection_tab.columnconfigure(1, weight=1)

        ttk.Label(detection_tab, text="亚像素定位方法").grid(row=5, column=0, sticky="w", pady=8)
        method_row = ttk.Frame(detection_tab)
        method_row.grid(row=5, column=1, sticky="e")
        ttk.Radiobutton(method_row, text="COM（快速）", variable=method_var, value="com").pack(side=tk.LEFT)
        ttk.Radiobutton(method_row, text="2D 高斯（精确）", variable=method_var, value="gaussian").pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Checkbutton(
            detection_tab,
            text="原子为亮点（暗原子请取消）",
            variable=bright_var,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=8)
        ttk.Label(
            detection_tab,
            text=(
                "算法流程：高斯预滤波 → 局部极值 → 选区候选过滤 → NMS → 亚像素精炼。\n"
                "减小最小间距会保留更密集的原子；降低阈值会检出更弱的原子，也可能增加噪声点。"
            ),
            foreground=self.colors["text_secondary"],
            wraplength=535,
            justify=tk.LEFT,
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(14, 0))

        intensity_fields = [
            ("积分孔径半径 (px)", aperture_var),
            ("局部背景环内半径 (px)", bg_inner_var),
            ("局部背景环外半径 (px)", bg_outer_var),
            ("跨帧 ID 关联距离 (px)", id_distance_var),
        ]
        for row, (label, variable) in enumerate(intensity_fields):
            ttk.Label(intensity_tab, text=label).grid(row=row, column=0, sticky="w", pady=10)
            ttk.Entry(intensity_tab, textvariable=variable, width=16).grid(
                row=row, column=1, sticky="e", padx=(8, 0)
            )
        intensity_tab.columnconfigure(0, weight=1)
        intensity_tab.columnconfigure(1, weight=1)
        ttk.Label(
            intensity_tab,
            text=(
                "必须满足：孔径半径 < 背景环内半径 < 背景环外半径。\n"
                "局部积分强度按孔径内像素相对背景环中位数求和；跨帧 ID 距离应小于最近邻原子间距，"
                "并大于漂移矫正后同一原子的残余位移。"
            ),
            foreground=self.colors["text_secondary"],
            wraplength=535,
            justify=tk.LEFT,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(18, 0))

        def commit(action: str) -> None:
            try:
                detection, intensity, id_distance = self._validated_parameter_bundle(
                    sigma=sigma_var.get(),
                    min_distance=distance_var.get(),
                    window=window_var.get(),
                    threshold=threshold_var.get(),
                    bright=bool(bright_var.get()),
                    method=method_var.get(),
                    aperture_radius=aperture_var.get(),
                    background_inner_radius=bg_inner_var.get(),
                    background_outer_radius=bg_outer_var.get(),
                    match_distance=id_distance_var.get(),
                )
            except Exception as error:
                messagebox.showerror("参数错误", str(error), parent=dialog)
                dialog.focus_set()
                return
            self.detection_params = detection
            self.intensity_params = intensity
            self.match_distance = id_distance
            self._sync_parameter_vars()
            self._update_detection_summary()
            result["action"] = action
            dialog.destroy()

        def close_with(action: str) -> None:
            result["action"] = action
            dialog.destroy()

        button_row = ttk.Frame(dialog)
        button_row.pack(fill=tk.X, padx=12, pady=(5, 12))
        if calibration_result is not None:
            ttk.Button(
                button_row,
                text="重新校准",
                command=lambda: close_with("recalibrate"),
                style="Outline.TButton",
            ).pack(side=tk.LEFT)
        ttk.Button(
            button_row,
            text="应用并识别当前帧",
            command=lambda: commit("detect"),
            style="Accent.TButton",
        ).pack(side=tk.RIGHT)
        ttk.Button(
            button_row,
            text="仅应用参数",
            command=lambda: commit("saved"),
            style="Outline.TButton",
        ).pack(side=tk.RIGHT, padx=6)
        ttk.Button(
            button_row,
            text="取消",
            command=lambda: close_with("cancel"),
            style="Outline.TButton",
        ).pack(side=tk.RIGHT)

        dialog.protocol("WM_DELETE_WINDOW", lambda: close_with("cancel"))
        dialog.focus_set()
        dialog.wait_window()
        action = result["action"]
        if calibration_result is None:
            if action == "detect":
                self.start_detection(False)
            elif action == "saved":
                self.status.config(text="✓ 识别与强度参数已保存；现有记录可按需重算强度")
        return action

    # ------------------------------------------------------------------
    # 后台逐帧检测、重算与导出任务
    # ------------------------------------------------------------------
    def _parse_intensity_controls(self) -> tuple[IntensityParams, float]:
        params = IntensityParams(
            aperture_radius=parse_finite_float(self.aperture_var.get(), "积分孔径半径"),
            background_inner_radius=parse_finite_float(self.bg_inner_var.get(), "背景环内半径"),
            background_outer_radius=parse_finite_float(self.bg_outer_var.get(), "背景环外半径"),
            bright=self.detection_params.bright,
        ).validated()
        match_distance = parse_finite_float(self.match_distance_var.get(), "跨帧 ID 关联距离")
        if match_distance <= 0:
            raise AtomicToolError("跨帧 ID 关联距离必须大于 0。")
        return params, match_distance

    def _set_busy(self, busy: bool, maximum: int = 1) -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        for button in [
            self.btn_import,
            self.btn_detect_current,
            self.btn_detect_all,
            self.btn_calibrate,
            self.btn_relink,
            self.btn_rectangle,
            self.btn_polygon,
            self.btn_clear_region,
            self.btn_recalc_current,
            self.btn_recalc_all,
            self.btn_save_marked,
            self.btn_export_stack,
            self.btn_export_csv,
            self.btn_save_session,
            self.btn_load_session,
        ]:
            button.configure(state=state)
        self.btn_cancel.configure(state=tk.NORMAL if busy else tk.DISABLED)
        self.progress.configure(maximum=max(1, maximum), value=0)

    def start_detection(self, all_frames: bool) -> None:
        if self.stack is None:
            messagebox.showinfo("提示", "请先导入 TIFF 图像。")
            return
        if self._busy:
            return
        if all_frames and any(self.frame_records):
            if not messagebox.askyesno(
                "重新识别全部帧",
                "逐帧识别全部会替换所有帧的现有点和手工修订。是否继续？",
            ):
                return
        elif not all_frames and self.frame_records[self.current_frame]:
            if not messagebox.askyesno(
                "重新识别当前帧",
                "识别当前帧会替换该帧现有点和手工修订，其他帧不受影响。是否继续？",
            ):
                return
        try:
            intensity_params, match_distance = self._parse_intensity_controls()
            detection_params = self.detection_params.validated()
        except Exception as error:
            messagebox.showerror("参数错误", str(error))
            return
        self.intensity_params = intensity_params
        self.match_distance = match_distance
        indices = list(range(len(self.frame_records))) if all_frames else [self.current_frame]
        regions_snapshot = {
            frame_index: self._region_for_detection(frame_index, all_frames)
            for frame_index in indices
        }
        active_region_count = sum(region is not None for region in regions_snapshot.values())
        self._job_generation += 1
        generation = self._job_generation
        self._cancel_event = threading.Event()
        self._set_busy(True, len(indices))
        region_text = "（应用选区）" if active_region_count else "（不限制选区）"
        self.status.config(text=f"正在后台{'逐帧识别全部' if all_frames else '识别当前帧'}{region_text}…")

        # 目录快照只用于单帧重检，避免线程读取正在变化的 GUI 状态。
        records_snapshot = [[record.copy() for record in records] for records in self.frame_records]
        stack_ref = self.stack

        def worker() -> None:
            try:
                frames_points: list[np.ndarray] = []
                for done, frame_index in enumerate(indices, start=1):
                    if self._cancel_event.is_set():
                        self._worker_queue.put(("cancelled", generation, None))
                        return
                    points = detect_atoms(
                        np.asarray(stack_ref[frame_index]),
                        detection_params,
                        region=regions_snapshot[frame_index],
                    )
                    frames_points.append(points)
                    self._worker_queue.put(("progress", generation, (done, len(indices), frame_index, len(points))))

                if all_frames:
                    frame_ids = link_frame_points(frames_points, match_distance, max_missed=TRACK_MAX_MISSED)
                    records_result = [
                        make_records(np.asarray(stack_ref[index]), points, ids, intensity_params)
                        for index, points, ids in zip(indices, frames_points, frame_ids)
                    ]
                    payload = (True, indices, records_result, detection_params, intensity_params)
                else:
                    frame_index = indices[0]
                    catalog = build_track_catalog(records_snapshot, exclude_frame=frame_index)
                    next_id = max(
                        (record.atom_id for frame in records_snapshot for record in frame),
                        default=0,
                    ) + 1
                    ids = match_points_to_catalog(frames_points[0], catalog, match_distance, next_id)
                    records_result = make_records(
                        np.asarray(stack_ref[frame_index]), frames_points[0], ids, intensity_params
                    )
                    payload = (False, indices, [records_result], detection_params, intensity_params)
                self._worker_queue.put(("detect_done", generation, payload))
            except Exception as error:
                self._worker_queue.put(("error", generation, f"识别失败：{error}"))

        threading.Thread(target=worker, daemon=True).start()

    def recalculate_intensities(self, all_frames: bool) -> None:
        if self.stack is None or self._busy:
            return
        try:
            params, match_distance = self._parse_intensity_controls()
        except Exception as error:
            messagebox.showerror("参数错误", str(error))
            return
        self.intensity_params = params
        self.match_distance = match_distance
        indices = list(range(len(self.frame_records))) if all_frames else [self.current_frame]
        if not all_frames:
            self._save_undo_snapshot()
            self.frame_records[self.current_frame] = recalculate_record_intensities(
                self._frame_array(), self.frame_records[self.current_frame], params
            )
            self.frame_intensity_params[self.current_frame] = params
            self.show_frame(reset_view=False)
            self.status.config(text=f"✓ 已重算帧 {self.current_frame + 1} 的积分强度与单帧百分位")
            return

        self._job_generation += 1
        generation = self._job_generation
        self._cancel_event = threading.Event()
        self._set_busy(True, len(indices))
        stack_ref = self.stack
        records_snapshot = [[record.copy() for record in records] for records in self.frame_records]

        def worker() -> None:
            try:
                result: list[list[AtomRecord]] = []
                for done, frame_index in enumerate(indices, start=1):
                    if self._cancel_event.is_set():
                        self._worker_queue.put(("cancelled", generation, None))
                        return
                    result.append(
                        recalculate_record_intensities(
                            np.asarray(stack_ref[frame_index]), records_snapshot[frame_index], params
                        )
                    )
                    self._worker_queue.put(("progress", generation, (done, len(indices), frame_index, len(result[-1]))))
                self._worker_queue.put(("recalc_done", generation, (result, params)))
            except Exception as error:
                self._worker_queue.put(("error", generation, f"强度重算失败：{error}"))

        threading.Thread(target=worker, daemon=True).start()

    def relink_all_ids(self) -> None:
        if self.stack is None or self._busy:
            return
        if not any(self.frame_records):
            messagebox.showinfo("提示", "当前没有可关联的原子点。")
            return
        try:
            intensity_params, match_distance = self._parse_intensity_controls()
        except Exception as error:
            messagebox.showerror("参数错误", str(error))
            return
        if not messagebox.askyesno(
            "重新关联 ID",
            "将按帧顺序和位置重新建立全部原子 ID；坐标与强度不会删除。是否继续？",
        ):
            return
        self.intensity_params = intensity_params
        self.match_distance = match_distance
        self._job_generation += 1
        generation = self._job_generation
        self._cancel_event = threading.Event()
        total = len(self.frame_records)
        self._set_busy(True, total)
        self.status.config(text="正在后台重新关联全部帧的原子 ID…")
        stack_ref = self.stack
        points_snapshot = [
            np.array([(record.x, record.y) for record in records], dtype=np.float64).reshape(-1, 2)
            for records in self.frame_records
        ]

        def worker() -> None:
            try:
                ids_by_frame = link_frame_points(points_snapshot, match_distance, max_missed=TRACK_MAX_MISSED)
                result: list[list[AtomRecord]] = []
                for frame_index, (points, ids) in enumerate(zip(points_snapshot, ids_by_frame)):
                    if self._cancel_event.is_set():
                        self._worker_queue.put(("cancelled", generation, None))
                        return
                    result.append(
                        make_records(np.asarray(stack_ref[frame_index]), points, ids, intensity_params)
                    )
                    self._worker_queue.put(("progress", generation, (frame_index + 1, total, frame_index, len(points))))
                self._worker_queue.put(("relink_done", generation, (result, intensity_params)))
            except Exception as error:
                self._worker_queue.put(("error", generation, f"重新关联失败：{error}"))

        threading.Thread(target=worker, daemon=True).start()

    def cancel_job(self) -> None:
        if self._busy:
            self._cancel_event.set()
            self.status.config(text="正在取消后台任务…")

    def _poll_worker_queue(self) -> None:
        try:
            while True:
                kind, generation, payload = self._worker_queue.get_nowait()
                if generation != self._job_generation:
                    continue
                if kind == "progress":
                    done, total, frame_index, count = payload
                    self.progress.configure(value=done)
                    self.status.config(text=f"处理中：帧 {frame_index + 1} · 检出/处理 {count} 个原子 · {done}/{total}")
                elif kind == "detect_done":
                    all_frames_flag, indices, results, detection_params, intensity_params = payload
                    for frame_index, records in zip(indices, results):
                        self.frame_records[frame_index] = records
                        self.frame_processed[frame_index] = True
                        # 记录被自动识别整体替换，该帧的手工修订溯源随之失效。
                        self.frame_edited[frame_index] = False
                        if frame_index < len(self.frame_detection_params):
                            self.frame_detection_params[frame_index] = detection_params
                        if frame_index < len(self.frame_intensity_params):
                            self.frame_intensity_params[frame_index] = intensity_params
                    affected = set(indices)
                    if all_frames_flag:
                        self.undo_stack.clear()
                    else:
                        # 单帧识别只替换该帧记录，其他帧的手工撤销历史必须保留。
                        self.undo_stack = [item for item in self.undo_stack if item[0] not in affected]
                    self._set_busy(False)
                    self.show_frame(reset_view=False)
                    total_atoms = sum(len(records) for records in results)
                    self.status.config(text=f"✓ 识别完成：{len(indices)} 帧，共 {total_atoms} 个逐帧原子记录")
                elif kind == "recalc_done":
                    result, params = payload
                    self.frame_records = result
                    for frame_index in range(len(self.frame_intensity_params)):
                        self.frame_intensity_params[frame_index] = params
                    # 全帧记录已被新参数整体替换，旧撤销快照会恢复出与新参数
                    # 不一致的强度值，直接失效。
                    self.undo_stack.clear()
                    self._set_busy(False)
                    self.show_frame(reset_view=False)
                    self.status.config(text="✓ 已重算全部帧的局部积分强度与单帧百分位（撤销历史已清空）")
                elif kind == "export_done":
                    self._set_busy(False)
                    self.status.config(text=f"✓ 已导出带标记堆栈：{os.path.basename(payload)}")
                    messagebox.showinfo("导出完成", f"带标记堆栈 TIFF 已保存：\n{payload}")
                elif kind == "csv_done":
                    self._set_busy(False)
                    self.status.config(text=f"✓ 已导出 CSV 与参数元数据：{os.path.basename(payload)}")
                    messagebox.showinfo("导出完成", f"CSV 与参数元数据已保存：\n{payload}")
                elif kind == "relink_done":
                    result, intensity_params = payload
                    self.frame_records = result
                    for frame_index in range(len(self.frame_intensity_params)):
                        self.frame_intensity_params[frame_index] = intensity_params
                    self.undo_stack.clear()
                    self.selected_atom_id = None
                    self._set_busy(False)
                    self.show_frame(reset_view=False)
                    self.status.config(text="✓ 已按位置重新关联全部帧的稳定原子 ID")
                elif kind == "cancelled":
                    self._set_busy(False)
                    self.status.config(text="后台任务已取消；未提交未完成的计算结果")
                elif kind == "error":
                    self._set_busy(False)
                    self.status.config(text=payload)
                    messagebox.showerror("任务失败", payload)
        except queue.Empty:
            pass
        except Exception:
            # 任何单条消息的处理异常都不允许终止轮询，也不允许界面永久卡在忙碌状态：
            # 兜底分支同样要恢复 _busy，否则按钮保持禁用，用户只能重启程序。
            _log_exception("后台任务消息处理失败", True)
            try:
                self._set_busy(False)
            except tk.TclError:
                pass
            try:
                self.status.config(text="内部错误：任务消息处理失败，详情已写入日志")
            except tk.TclError:
                pass
        finally:
            if getattr(self, "_poll_active", False):
                try:
                    self._poll_after_id = self.root.after(POLL_INTERVAL_MS, self._poll_worker_queue)
                except tk.TclError:
                    self._poll_after_id = None

    def _refresh_intensity_summary(self) -> None:
        if self.stack is None:
            self.intensity_summary_var.set("当前帧：尚无原子强度")
            return
        records = self.frame_records[self.current_frame]
        values = np.array(
            [record.integrated_intensity for record in records], dtype=np.float64
        )
        values = values[np.isfinite(values)]
        if len(values) == 0:
            self.intensity_summary_var.set("当前帧：尚无有效原子强度")
            return
        shown = len(visible_records(records, self.rules))
        truncated = sum(1 for record in records if record.truncated)
        truncation_note = f"；截断 {truncated} 个（强度偏低，仅标记不剔除）" if truncated else ""
        contaminated = sum(1 for record in records if record.neighbor_contaminated)
        contamination_note = f"；邻居污染 {contaminated} 个（孔径/背景环可能含相邻原子信号）" if contaminated else ""
        self.intensity_summary_var.set(
            f"当前帧：弱/中位/强 = {values.min():.5g} / {np.median(values):.5g} / {values.max():.5g}；"
            f"当前规则显示 {shown}/{len(values)}{truncation_note}{contamination_note}"
        )

    # ------------------------------------------------------------------
    # 百分位范围、颜色和标记形状
    # ------------------------------------------------------------------
    def _refresh_rule_tree(self, select_index: int | None = None) -> None:
        for item in self.rule_tree.get_children():
            self.rule_tree.delete(item)
        for index, rule in enumerate(self.rules):
            item = self.rule_tree.insert(
                "",
                tk.END,
                iid=str(index),
                values=(
                    f"{rule.low:g}%–{rule.high:g}%",
                    rule.color,
                    MARKER_LABEL_BY_KEY.get(rule.marker, rule.marker),
                    "是" if rule.visible else "否",
                ),
            )
            if select_index == index:
                self.rule_tree.selection_set(item)
                self.rule_tree.see(item)

    def _selected_rule_index(self) -> int | None:
        selection = self.rule_tree.selection()
        return int(selection[0]) if selection else None

    def _show_rule_dialog(self, existing: MarkerRule | None = None) -> MarkerRule | None:
        dialog = tk.Toplevel(self.root)
        dialog.title("编辑百分位标记规则" if existing else "添加百分位标记规则")
        dialog.geometry("390x360")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        default = existing or MarkerRule("弱强度 10–20%", 10.0, 20.0, "#ff9800", "circle", True)
        name_var = tk.StringVar(value=default.name)
        low_var = tk.StringVar(value=f"{default.low:g}")
        high_var = tk.StringVar(value=f"{default.high:g}")
        color_var = tk.StringVar(value=default.color)
        shape_var = tk.StringVar(value=MARKER_LABEL_BY_KEY.get(default.marker, "圆圈"))
        visible_var = tk.BooleanVar(value=default.visible)
        result: list[MarkerRule] = []

        content = ttk.Frame(dialog, padding=14)
        content.pack(fill=tk.BOTH, expand=True)
        rows = [
            ("规则名称", name_var),
            ("百分位下限", low_var),
            ("百分位上限", high_var),
        ]
        for row, (label, variable) in enumerate(rows):
            ttk.Label(content, text=label).grid(row=row, column=0, sticky="w", pady=7)
            ttk.Entry(content, textvariable=variable, width=22).grid(row=row, column=1, sticky="e", pady=7)
        ttk.Label(content, text="标记形状").grid(row=3, column=0, sticky="w", pady=7)
        ttk.Combobox(
            content,
            textvariable=shape_var,
            values=list(MARKER_LABELS),
            state="readonly",
            width=19,
        ).grid(row=3, column=1, sticky="e", pady=7)
        ttk.Label(content, text="标记颜色").grid(row=4, column=0, sticky="w", pady=7)
        color_row = ttk.Frame(content)
        color_row.grid(row=4, column=1, sticky="e", pady=7)
        color_label = tk.Label(color_row, textvariable=color_var, bg=color_var.get(), fg="white", width=10, relief="solid")
        color_label.pack(side=tk.LEFT)

        def choose_color() -> None:
            chosen = colorchooser.askcolor(color_var.get(), parent=dialog, title="选择标记颜色")[1]
            if chosen:
                color_var.set(chosen)
                color_label.configure(bg=chosen)

        ttk.Button(color_row, text="选择", command=choose_color, style="Outline.TButton").pack(side=tk.LEFT, padx=(5, 0))
        ttk.Checkbutton(content, text="启用并显示此范围", variable=visible_var).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=7
        )
        content.columnconfigure(1, weight=1)

        def save() -> None:
            try:
                rule = MarkerRule(
                    name=name_var.get().strip() or "未命名范围",
                    low=float(low_var.get()),
                    high=float(high_var.get()),
                    color=color_var.get(),
                    marker=MARKER_LABELS[shape_var.get()],
                    visible=bool(visible_var.get()),
                ).validated()
            except Exception as error:
                messagebox.showerror("规则错误", str(error), parent=dialog)
                return
            result.append(rule)
            dialog.destroy()

        button_row = ttk.Frame(dialog)
        button_row.pack(fill=tk.X, padx=14, pady=12)
        ttk.Button(button_row, text="保存", command=save, style="Accent.TButton").pack(side=tk.RIGHT)
        ttk.Button(button_row, text="取消", command=dialog.destroy, style="Outline.TButton").pack(side=tk.RIGHT, padx=5)
        dialog.wait_window()
        return result[0] if result else None

    def add_rule(self) -> None:
        rule = self._show_rule_dialog()
        if rule is None:
            return
        self.rules.append(rule)
        self._refresh_rule_tree(len(self.rules) - 1)
        self.show_frame(reset_view=False) if self.stack is not None else None

    def edit_rule(self) -> None:
        index = self._selected_rule_index()
        if index is None:
            messagebox.showinfo("提示", "请先选中一条标记规则。")
            return
        rule = self._show_rule_dialog(self.rules[index])
        if rule is None:
            return
        self.rules[index] = rule
        self._refresh_rule_tree(index)
        self.show_frame(reset_view=False) if self.stack is not None else None

    def toggle_rule(self) -> None:
        index = self._selected_rule_index()
        if index is None:
            return
        self.rules[index] = replace(self.rules[index], visible=not self.rules[index].visible)
        self._refresh_rule_tree(index)
        self.show_frame(reset_view=False) if self.stack is not None else None

    def delete_rule(self) -> None:
        index = self._selected_rule_index()
        if index is None:
            return
        self.rules.pop(index)
        self._refresh_rule_tree(min(index, len(self.rules) - 1) if self.rules else None)
        self.show_frame(reset_view=False) if self.stack is not None else None

    def move_rule(self, direction: int) -> None:
        index = self._selected_rule_index()
        if index is None:
            return
        target = index + direction
        if not 0 <= target < len(self.rules):
            return
        self.rules[index], self.rules[target] = self.rules[target], self.rules[index]
        self._refresh_rule_tree(target)
        self.show_frame(reset_view=False) if self.stack is not None else None

    def _update_marker_radius(self, value: str) -> None:
        self.marker_radius = round(float(value), 1)
        self.marker_radius_label.configure(text=f"{self.marker_radius:.1f} px")
        if self.stack is not None:
            self._schedule_overlay_refresh()

    def _schedule_overlay_refresh(self) -> None:
        """标记大小连续变化时防抖，只重绘 overlay 不重建整帧。"""
        if self._overlay_refresh_after_id is not None:
            try:
                self.root.after_cancel(self._overlay_refresh_after_id)
            except tk.TclError:
                pass
        self._overlay_refresh_after_id = self.root.after(REDRAW_DEBOUNCE_MS, self._debounced_overlay_refresh)

    def _debounced_overlay_refresh(self) -> None:
        self._overlay_refresh_after_id = None
        self._draw_overlays()
        self.canvas.draw_idle()

    def _toggle_ids(self) -> None:
        self.show_ids = bool(self.show_ids_var.get())
        if self.stack is not None:
            visible_count = len(visible_records(self.frame_records[self.current_frame], self.rules))
            if self.show_ids and visible_count > self.MAX_DRAWN_LABELS:
                self.status.config(
                    text=f"当前可见原子 {visible_count} 个，超过 {self.MAX_DRAWN_LABELS}；为保证速度，预览暂不画 ID，导出仍可画"
                )
            self._draw_overlays()
            self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # 保存与导出
    # ------------------------------------------------------------------
    @staticmethod
    def _check_writable_target(path: str, estimated_bytes: int = 0) -> None:
        """导出前的写权限与磁盘空间预检；失败时给出可直接理解的中文错误。"""
        target = Path(path).absolute()
        directory = target.parent
        if not directory.exists():
            raise AtomicToolError(f"目标目录不存在：{directory}")
        if not os.access(directory, os.W_OK):
            raise AtomicToolError(f"没有目录写入权限：{directory}")
        if estimated_bytes > 0:
            free = shutil.disk_usage(directory).free
            if free < estimated_bytes:
                raise AtomicToolError(
                    f"磁盘空间不足：预计需要约 {estimated_bytes / 1e9:.2f} GB，"
                    f"{directory} 所在磁盘仅剩 {free / 1e9:.2f} GB。"
                )

    @staticmethod
    def _part_path(path: str | Path) -> Path:
        """原子写入的临时同目录路径：写入成功后 os.replace 到目标。

        临时名保留目标真实扩展名（如 xxx.png -> xxx.tmp.png），
        否则 Pillow 等按扩展名推断格式的保存方写临时文件时
        会因 .part 后缀抛 unknown file extension。
        """
        target = Path(path).absolute()
        suffix = target.suffix or ".part"
        return target.with_name(f"{target.stem}.tmp{suffix}")

    def _ensure_not_source(self, path: str | Path) -> None:
        """拒绝把导出目标指向源 TIFF 本身，防止导出覆盖原始数据。

        先做 normcase 字符串比较（覆盖大小写、分隔符、相对/绝对路径差异），
        再用 os.path.samefile 复核「路径写法不同但指向同一文件」的情形
        （符号链接、硬链接、8.3 短名等）；任一命中即拒绝。
        """
        if self.stack_info is None:
            return
        source = Path(self.stack_info.path)
        target = Path(path)
        same = os.path.normcase(os.path.abspath(str(source))) == os.path.normcase(
            os.path.abspath(str(target))
        )
        if not same:
            try:
                same = (
                    source.exists()
                    and target.exists()
                    and os.path.samefile(source, target)
                )
            except OSError:
                same = False
        if same:
            raise AtomicToolError(
                f"导出路径与源图像相同：{target}\n为避免覆盖原始数据，请另选文件名。"
            )

    def save_current_marked_image(self) -> None:
        if self.stack is None:
            messagebox.showinfo("提示", "请先导入 TIFF 图像。")
            return
        default_base = Path(self.stack_info.path).stem if self.stack_info else "atomic"
        path = filedialog.asksaveasfilename(
            title="保存当前帧标记图",
            defaultextension=".png",
            initialfile=f"{default_base}_frame_{self.current_frame + 1:04d}_marked.png",
            filetypes=[("PNG 图像", "*.png"), ("TIFF 图像", "*.tif *.tiff")],
        )
        if not path:
            return
        try:
            from PIL import Image

            self._ensure_not_source(path)
            self._check_writable_target(path)
            rendered = render_marked_frame(
                self._frame_array(),
                self.frame_records[self.current_frame],
                list(self.rules),
                marker_radius=self.marker_radius,
                show_ids=self.show_ids,
            )
            part = self._part_path(path)
            Image.fromarray(rendered).save(part)
            os.replace(part, path)
        except Exception as error:
            _log_exception("保存标记图失败", error)
            messagebox.showerror("保存失败", str(error))
            return
        self.status.config(text=f"✓ 已保存当前帧标记图：{os.path.basename(path)}")

    def export_marked_stack(self) -> None:
        if self.stack is None or self._busy:
            return
        unprocessed = [index + 1 for index, processed in enumerate(self.frame_processed) if not processed]
        if unprocessed:
            preview = ", ".join(map(str, unprocessed[:8]))
            if len(unprocessed) > 8:
                preview += "…"
            if not messagebox.askyesno(
                "存在未识别帧",
                f"有 {len(unprocessed)} 帧尚未识别（帧 {preview}）。这些帧将不带原子标记，是否继续导出？",
            ):
                return
        default_base = Path(self.stack_info.path).stem if self.stack_info else "atomic"
        path = filedialog.asksaveasfilename(
            title="导出带标记堆栈 TIFF",
            defaultextension=".tif",
            initialfile=f"{default_base}_marked_stack.tif",
            filetypes=[("TIFF 堆栈", "*.tif *.tiff")],
        )
        if not path:
            return
        try:
            # 按未压缩大小保守预估，压缩实际占用更小；不足时宁可拒绝也不写一半。
            frame_height, frame_width = self.stack.shape[1:]
            self._ensure_not_source(path)
            estimated_bytes = (
                int(self.stack.shape[0]) * int(frame_height) * int(frame_width) * 3
            )
            self._check_writable_target(path, estimated_bytes)
        except Exception as error:
            messagebox.showerror("无法导出", str(error))
            return

        self._job_generation += 1
        generation = self._job_generation
        self._cancel_event = threading.Event()
        total = len(self.frame_records)
        self._set_busy(True, total)
        stack_ref = self.stack
        records_snapshot = [[record.copy() for record in records] for records in self.frame_records]
        rules_snapshot = list(self.rules)
        radius = self.marker_radius
        show_ids = self.show_ids
        height, width = self.stack.shape[1:]
        part_path = self._part_path(path)

        def worker() -> None:
            try:
                import tifffile

                def frames():
                    for index in range(total):
                        if self._cancel_event.is_set():
                            raise InterruptedError("用户取消导出")
                        rendered = render_marked_frame(
                            np.asarray(stack_ref[index]),
                            records_snapshot[index],
                            rules_snapshot,
                            marker_radius=radius,
                            show_ids=show_ids,
                        )
                        self._worker_queue.put(("progress", generation, (index + 1, total, index, len(records_snapshot[index]))))
                        yield rendered

                tifffile.imwrite(
                    part_path,
                    data=frames(),
                    shape=(total, height, width, 3),
                    dtype=np.uint8,
                    photometric="rgb",
                    metadata={"axes": "TYXS"},
                    compression="zlib",
                    bigtiff=estimated_bytes > 3_500_000_000,
                )
                os.replace(part_path, path)
                self._worker_queue.put(("export_done", generation, path))
            except InterruptedError:
                try:
                    part_path.unlink(missing_ok=True)
                except Exception:
                    pass
                self._worker_queue.put(("cancelled", generation, None))
            except Exception as error:
                try:
                    part_path.unlink(missing_ok=True)
                except Exception:
                    pass
                self._worker_queue.put(("error", generation, f"堆栈导出失败：{error}"))

        threading.Thread(target=worker, daemon=True).start()

    def _build_export_metadata(self) -> dict:
        """汇总导出 CSV 时的全部参数与溯源信息；在 GUI 线程上快照。"""
        per_frame_intensity = [
            asdict(params) if params is not None else None
            for params in self.frame_intensity_params
        ]
        # 一致性只看有原子记录的帧；有原子但强度参数未记录（None）也算不一致。
        used_params = [
            self.frame_intensity_params[index]
            for index in range(len(self.frame_records))
            if self.frame_records[index]
        ]
        if used_params:
            intensity_consistent = all(
                params is not None and params == used_params[0] for params in used_params
            )
        else:
            intensity_consistent = True
        return {
            "tool_version": APP_VERSION,
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "source_tiff": self.stack_info.path if self.stack_info else "",
            "frame_count": len(self.frame_records),
            "coordinate_convention": "image pixels: x right, y down; frames and CSV frame numbers are 1-based",
            "intensity_definition": (
                "circular aperture sum relative to median of a concentric local background annulus; "
                "bright: sum(I-bg), dark: sum(bg-I)"
            ),
            "detection_params": asdict(self.detection_params),
            "intensity_params": asdict(self.intensity_params),
            "intensity_params_consistent_across_frames": intensity_consistent,
            "per_frame_intensity_params": per_frame_intensity,
            "id_match_distance_px": self.match_distance,
            "track_max_missed_frames": TRACK_MAX_MISSED,
            "percentile_scope": "independently within each frame",
            "truncation_note": (
                "truncated: the aperture/annulus is clipped by the image edge or contains non-finite "
                "pixels; the intensity is biased low. Truncated atoms stay in the percentile ranking "
                "(v1.2 keeps v1.1 behavior) and are flagged in the CSV."
            ),
            "neighbor_contamination_note": (
                "nearest_neighbor is the distance to the closest other atom in the same frame "
                "(empty when the atom is alone). neighbor_contaminated means this distance is below "
                "aperture_radius + background_outer_radius, i.e. the neighbor's core signal may enter "
                "this atom's aperture or annulus; verify such intensities manually."
            ),
            "marker_rules": [asdict(rule) for rule in self.rules],
            "selection_scope": self.selection_scope_var.get(),
            "selection_scope_note": (
                "all: stack_region is applied to current-frame detection and every frame in detect-all; "
                "current: each frame can store its own region and it is applied only by current-frame detection"
            ),
            "stack_region": asdict(self.stack_region) if self.stack_region is not None else None,
            "frame_regions": [asdict(region) if region is not None else None for region in self.frame_regions],
            "per_frame_detection_params": [
                asdict(params) if params is not None else None
                for params in self.frame_detection_params
            ],
            "manually_edited_frames": [index + 1 for index, edited in enumerate(self.frame_edited) if edited],
        }

    def export_csv(self) -> None:
        if self.stack is None:
            messagebox.showinfo("提示", "请先导入 TIFF 图像。")
            return
        if not any(self.frame_records):
            messagebox.showinfo("提示", "当前没有可导出的原子记录。")
            return
        default_base = Path(self.stack_info.path).stem if self.stack_info else "atomic"
        path = filedialog.asksaveasfilename(
            title="导出坐标、ID 与原子柱强度",
            defaultextension=".csv",
            initialfile=f"{default_base}_atoms.csv",
            filetypes=[("CSV 文件", "*.csv")],
        )
        if not path:
            return
        try:
            self._check_writable_target(path)
        except Exception as error:
            messagebox.showerror("无法导出", str(error))
            return

        self._job_generation += 1
        generation = self._job_generation
        self._cancel_event = threading.Event()
        total = len(self.frame_records)
        self._set_busy(True, total)
        self.status.config(text="正在后台导出 CSV 与参数元数据…")
        records_snapshot = [[record.copy() for record in records] for records in self.frame_records]
        rules_snapshot = list(self.rules)
        intensity_by_frame = list(self.frame_intensity_params)
        metadata = self._build_export_metadata()
        part_path = self._part_path(path)

        def worker() -> None:
            try:
                with open(part_path, "w", newline="", encoding="utf-8-sig") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        [
                            "帧号(从1开始)",
                            "原子ID",
                            "x_px",
                            "y_px",
                            "局部背景扣除积分强度",
                            "单帧强度百分位",
                            "孔径像素数",
                            "背景像素数",
                            "背景中位数",
                            "孔径截断",
                            "邻居污染",
                            "最近邻距离(px)",
                            "来源",
                            "本帧强度参数",
                            "当前规则是否显示",
                            "命中规则",
                            "标记颜色",
                            "标记形状",
                        ]
                    )
                    for frame_index, records in enumerate(records_snapshot, start=1):
                        if self._cancel_event.is_set():
                            raise InterruptedError("用户取消导出")
                        fingerprint = _intensity_fingerprint(
                            intensity_by_frame[frame_index - 1]
                            if frame_index - 1 < len(intensity_by_frame) else None
                        )
                        for record in sorted(records, key=lambda item: item.atom_id):
                            rule = rule_for_percentile(record.percentile, rules_snapshot)
                            nearest = (
                                "" if not np.isfinite(record.nearest_neighbor)
                                else f"{record.nearest_neighbor:.6f}"
                            )
                            writer.writerow(
                                [
                                    frame_index,
                                    record.atom_id,
                                    f"{record.x:.6f}",
                                    f"{record.y:.6f}",
                                    "NaN" if not np.isfinite(record.integrated_intensity) else f"{record.integrated_intensity:.10g}",
                                    "NaN" if not np.isfinite(record.percentile) else f"{record.percentile:.6f}",
                                    record.aperture_pixels,
                                    record.background_pixels,
                                    "NaN" if not np.isfinite(record.background_level) else f"{record.background_level:.10g}",
                                    "是" if record.truncated else "否",
                                    "是" if record.neighbor_contaminated else "否",
                                    nearest,
                                    "手动" if record.source == "manual" else "自动",
                                    fingerprint,
                                    "是" if rule is not None else "否",
                                    _csv_safe_text(rule.name) if rule else "",
                                    rule.color if rule else "",
                                    MARKER_LABEL_BY_KEY.get(rule.marker, rule.marker) if rule else "",
                                ]
                            )
                os.replace(part_path, path)
                metadata_part = self._part_path(str(Path(path).with_suffix(".metadata.json")))
                metadata_path = str(Path(path).with_suffix(".metadata.json"))
                with open(metadata_part, "w", encoding="utf-8") as handle:
                    json.dump(metadata, handle, ensure_ascii=False, indent=2)
                os.replace(metadata_part, metadata_path)
                self._worker_queue.put(("csv_done", generation, path))
            except InterruptedError:
                try:
                    part_path.unlink(missing_ok=True)
                except Exception:
                    pass
                self._worker_queue.put(("cancelled", generation, None))
            except Exception as error:
                try:
                    part_path.unlink(missing_ok=True)
                except Exception:
                    pass
                self._worker_queue.put(("error", generation, f"CSV 导出失败：{error}"))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # 会话保存与加载
    # ------------------------------------------------------------------
    def save_session(self) -> bool:
        """把当前全部状态保存为 JSON 会话；返回是否成功（用户取消返回 False）。"""
        if self.stack is None or self.stack_info is None:
            messagebox.showinfo("提示", "请先导入 TIFF 图像。")
            return False
        default_base = Path(self.stack_info.path).stem
        path = filedialog.asksaveasfilename(
            title="保存会话",
            defaultextension=".session.json",
            initialfile=f"{default_base}_session.json",
            filetypes=[("会话文件", "*.session.json"), ("JSON 文件", "*.json")],
        )
        if not path:
            return False
        try:
            self._check_writable_target(path)
            part = self._part_path(path)
            with open(part, "w", encoding="utf-8") as handle:
                json.dump(self._serialize_session(), handle, ensure_ascii=False, indent=2)
            os.replace(part, path)
        except Exception as error:
            _log_exception("保存会话失败", error)
            messagebox.showerror("保存失败", str(error))
            return False
        self.status.config(text=f"✓ 会话已保存：{os.path.basename(path)}")
        return True

    def _serialize_session(self) -> dict:
        frames = []
        for index in range(len(self.frame_records)):
            frames.append(
                {
                    "processed": self.frame_processed[index],
                    "edited": self.frame_edited[index],
                    "region": asdict(self.frame_regions[index]) if self.frame_regions[index] is not None else None,
                    "detection_params": (
                        asdict(self.frame_detection_params[index])
                        if self.frame_detection_params[index] is not None
                        else None
                    ),
                    "intensity_params": (
                        asdict(self.frame_intensity_params[index])
                        if self.frame_intensity_params[index] is not None
                        else None
                    ),
                    "records": [self._record_to_json(record) for record in self.frame_records[index]],
                }
            )
        return {
            "format": SESSION_FORMAT,
            "tool_version": APP_VERSION,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "source_tiff": self.stack_info.path,
            "stack_shape": [int(value) for value in self.stack_info.stack_shape],
            "detection_params": asdict(self.detection_params),
            "intensity_params": asdict(self.intensity_params),
            "match_distance": self.match_distance,
            "marker_radius": self.marker_radius,
            "show_ids": self.show_ids,
            "selection_scope": self.selection_scope_var.get(),
            "rules": [asdict(rule) for rule in self.rules],
            "stack_region": asdict(self.stack_region) if self.stack_region is not None else None,
            "frames": frames,
        }

    @staticmethod
    def _record_to_json(record: AtomRecord) -> dict:
        return {
            "atom_id": int(record.atom_id),
            "x": _json_float(record.x),
            "y": _json_float(record.y),
            "integrated_intensity": _json_float(record.integrated_intensity),
            "percentile": _json_float(record.percentile),
            "aperture_pixels": int(record.aperture_pixels),
            "background_pixels": int(record.background_pixels),
            "background_level": _json_float(record.background_level),
            "truncated": bool(record.truncated),
            "source": str(record.source),
            "nearest_neighbor": _json_float(record.nearest_neighbor),
            "neighbor_contaminated": bool(record.neighbor_contaminated),
        }

    def open_session(self) -> None:
        if self._busy:
            messagebox.showinfo("任务进行中", "请先等待当前任务完成或取消任务。")
            return
        if any(self.frame_records) and not messagebox.askyesno(
            "打开会话",
            "打开会话会替换当前全部逐帧识别与手工修订结果。是否继续？",
        ):
            return
        path = filedialog.askopenfilename(
            title="打开会话",
            filetypes=[("会话文件", "*.session.json"), ("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as error:
            messagebox.showerror("打开会话失败", f"无法读取会话文件：{error}")
            return
        try:
            self._apply_session(data)
        except Exception as error:
            _log_exception("应用会话失败", error)
            messagebox.showerror("打开会话失败", str(error))
            return
        self.status.config(
            text=f"✓ 已打开会话（{len(self.frame_records)} 帧，"
            f"{sum(len(records) for records in self.frame_records)} 个原子）；"
            f"原始 TIFF：{os.path.basename(self.stack_info.path)}"
        )

    def _apply_session(self, data: dict) -> None:
        """校验并整体应用会话；任何字段非法都在安装状态前抛错，不留半截状态。"""
        if not isinstance(data, dict) or data.get("format") != SESSION_FORMAT:
            raise AtomicToolError(f"不是本工具的会话文件（format={data.get('format')!r}）。")

        source = str(data.get("source_tiff") or "")
        tiff_path = Path(source) if source else None
        if tiff_path is None or not tiff_path.is_file():
            relocated = filedialog.askopenfilename(
                title="找不到原始 TIFF，请重新定位",
                filetypes=[("TIFF 文件", "*.tif *.tiff"), ("所有文件", "*.*")],
            )
            if not relocated:
                raise AtomicToolError("未重新定位原始 TIFF，会话未加载。")
            tiff_path = Path(relocated)

        stack, info = load_tiff_stack(tiff_path)
        expected = tuple(int(value) for value in data.get("stack_shape", []))
        if expected and tuple(info.stack_shape) != expected:
            close_tiff_stack(stack)
            raise AtomicToolError(
                f"原始 TIFF 与会话不匹配：会话记录 {expected}，实际 {tuple(info.stack_shape)}。"
            )

        try:
            detection = _detection_params_from_json(data.get("detection_params"))
            intensity = _intensity_params_from_json(data.get("intensity_params"))
            if detection is None or intensity is None:
                raise AtomicToolError("会话缺少识别或强度参数。")
            match_distance = parse_finite_float(data.get("match_distance"), "跨帧 ID 关联距离")
            if match_distance <= 0:
                raise AtomicToolError("跨帧 ID 关联距离必须大于 0。")
            rules = [_rule_from_json(item) for item in data.get("rules", [])]
            marker_radius = parse_finite_float(data.get("marker_radius", 7.0), "标记大小")
            frames = data.get("frames")
            if not isinstance(frames, list) or len(frames) != int(info.stack_shape[0]):
                raise AtomicToolError(
                    f"会话帧数（{len(frames) if isinstance(frames, list) else '?'}）与"
                    f" TIFF 帧数（{info.stack_shape[0]}）不一致。"
                )
            parsed_frames = [
                {
                    "processed": bool(frame.get("processed", False)),
                    "edited": bool(frame.get("edited", False)),
                    "region": _region_from_json(frame.get("region")),
                    "detection_params": _detection_params_from_json(frame.get("detection_params")),
                    "intensity_params": _intensity_params_from_json(frame.get("intensity_params")),
                    "records": [_record_from_json(item) for item in frame.get("records", [])],
                }
                for frame in frames
            ]
        except Exception:
            close_tiff_stack(stack)
            raise

        self._install_stack(stack, info)
        self.detection_params = detection
        self.intensity_params = intensity
        self.match_distance = match_distance
        self.rules = rules
        self.marker_radius = float(marker_radius)
        self.show_ids = bool(data.get("show_ids", False))
        self.stack_region = _region_from_json(data.get("stack_region"))
        scope = str(data.get("selection_scope", "current"))
        self.selection_scope_var.set(scope if scope in {"current", "all"} else "current")
        for index, frame in enumerate(parsed_frames):
            self.frame_records[index] = frame["records"]
            self.frame_processed[index] = frame["processed"]
            self.frame_edited[index] = frame["edited"]
            self.frame_regions[index] = frame["region"]
            self.frame_detection_params[index] = frame["detection_params"]
            self.frame_intensity_params[index] = frame["intensity_params"]
        self._sync_parameter_vars()
        # 显示控件一并回填：否则打开会话后 Scale/复选框仍停留在旧位置，
        # 而 _update_marker_radius/_toggle_ids 又从绑定变量读值，控件与状态脱钩。
        # 实测对绑定变量 .set() 不会同步触发 Scale 的 -command，标签需一并同步。
        self.marker_radius_var.set(float(self.marker_radius))
        self.marker_radius_label.configure(text=f"{self.marker_radius:.1f} px")
        self.show_ids_var.set(bool(self.show_ids))
        self._update_detection_summary()
        self._refresh_rule_tree()
        self.show_frame(reset_view=True)


def _excepthook(exc_type, exc_value, exc_tb) -> None:
    _log_exception("主线程未捕获异常", (exc_type, exc_value, exc_tb))
    try:
        location = str(LOG_PATH) if LOG_PATH else "临时目录"
        messagebox.showerror("程序错误", f"发生致命错误，详情已写入日志文件：\n{location}\n\n{exc_value}")
    except Exception:
        pass


def main() -> None:
    _setup_logging()
    sys.excepthook = _excepthook
    root = tk.Tk()
    AtomicRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
