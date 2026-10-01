"""Tk desktop tool for precise atom-center point annotation."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from .annotations import (
    DEFAULT_METADATA,
    AnnotationDocument,
    AnnotationProject,
    ProjectRecord,
    export_yolo_dataset,
    point_in_regions,
    project_statistics,
    validate_document,
)
from .image_io import load_image, normalize_percentile

BUNDLED_FONT_FAMILY = "Noto Sans CJK SC"
_BUNDLED_FONT_CONFIGURED: bool | None = None

_COMMON_METADATA_FIELDS = (
    ("sample_id", "样品 ID", None),
    ("acquisition_id", "采集批次 ID", None),
    ("pixel_size", "像素尺寸", None),
    (
        "pixel_size_unit",
        "尺寸单位",
        ("angstrom_per_pixel", "nm_per_pixel", "pm_per_pixel"),
    ),
    ("accelerating_voltage_kv", "加速电压(kV)", None),
)
_MODALITY_METADATA_FIELDS = {
    "haadf_stem": (
        ("detector_inner_angle_mrad", "探测器内角(mrad)", None),
        ("detector_outer_angle_mrad", "探测器外角(mrad)", None),
    ),
    "hrtem": (
        ("defocus_nm", "离焦(nm)", None),
        ("spherical_aberration_mm", "球差 Cs(mm)", None),
        ("contrast_polarity", "原子对比", ("bright", "dark", "mixed")),
    ),
}
_DESCRIPTIVE_METADATA_FIELDS = (
    ("material", "材料/结构", None),
    ("zone_axis", "晶带轴", None),
    ("microscope_id", "显微镜 ID", None),
    ("annotator", "标注人", None),
    ("notes", "备注", None),
)


def metadata_fields_for_modality(
    modality: str,
) -> tuple[tuple[str, str, tuple[str, ...] | None], ...]:
    """Show only acquisition fields relevant to the current task type."""

    return (
        _COMMON_METADATA_FIELDS
        + _MODALITY_METADATA_FIELDS[modality]
        + _DESCRIPTIVE_METADATA_FIELDS
    )


def bundled_font_path() -> Path:
    return Path(__file__).resolve().parent / "assets" / "NotoSansCJKsc-AtomAnnotator.otf"


def configure_bundled_font() -> bool:
    """Load the bundled CJK font privately without modifying the host system."""

    global _BUNDLED_FONT_CONFIGURED
    if _BUNDLED_FONT_CONFIGURED is not None:
        return _BUNDLED_FONT_CONFIGURED
    font_path = bundled_font_path()
    if not font_path.is_file():
        _BUNDLED_FONT_CONFIGURED = False
        return False
    windows_registered = True
    try:
        import os

        if os.name == "nt":
            import ctypes

            FR_PRIVATE = 0x10
            windows_registered = bool(
                ctypes.windll.gdi32.AddFontResourceExW(str(font_path), FR_PRIVATE, 0)
            )
        from matplotlib import font_manager

        font_manager.fontManager.addfont(str(font_path))
    except Exception:
        _BUNDLED_FONT_CONFIGURED = False
        return False
    _BUNDLED_FONT_CONFIGURED = windows_registered
    return _BUNDLED_FONT_CONFIGURED


def apply_tk_font_defaults(root: object) -> None:
    import tkinter.font as tkfont

    configure_bundled_font()
    for name in (
        "TkDefaultFont",
        "TkTextFont",
        "TkMenuFont",
        "TkHeadingFont",
        "TkCaptionFont",
        "TkSmallCaptionFont",
        "TkIconFont",
        "TkTooltipFont",
    ):
        try:
            tkfont.nametofont(name, root=root).configure(family=BUNDLED_FONT_FAMILY)
        except Exception:
            continue


def calculate_window_geometry(
    requested_size: tuple[int, int],
    screen_size: tuple[int, int],
    *,
    minimum_size: tuple[int, int],
    maximum_fraction: tuple[float, float] = (0.94, 0.88),
) -> tuple[int, int, int, int]:
    """Return a centered window geometry that never exceeds the usable screen."""

    screen_width = max(1, int(screen_size[0]))
    screen_height = max(1, int(screen_size[1]))
    maximum_width = max(320, int(screen_width * maximum_fraction[0]))
    maximum_height = max(260, int(screen_height * maximum_fraction[1]))
    minimum_width = min(maximum_width, max(320, int(minimum_size[0])))
    minimum_height = min(maximum_height, max(260, int(minimum_size[1])))
    width = min(max(int(requested_size[0]), minimum_width), maximum_width)
    height = min(max(int(requested_size[1]), minimum_height), maximum_height)
    x = max(0, (screen_width - width) // 2)
    y = max(0, (screen_height - height) // 2)
    return width, height, x, y


def fit_window_to_screen(
    root: object,
    *,
    preferred_size: tuple[int, int],
    minimum_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Size a Tk window after layout, accounting for DPI and small displays."""

    root.update_idletasks()
    requested_size = (
        max(int(root.winfo_reqwidth()), int(preferred_size[0])),
        max(int(root.winfo_reqheight()), int(preferred_size[1])),
    )
    geometry = calculate_window_geometry(
        requested_size,
        (int(root.winfo_screenwidth()), int(root.winfo_screenheight())),
        minimum_size=minimum_size,
    )
    width, height, x, y = geometry
    root.geometry(f"{width}x{height}+{x}+{y}")
    root.minsize(min(width, minimum_size[0]), min(height, minimum_size[1]))
    return geometry


def find_task_project(selected_path: str | Path) -> Path:
    """Find one task project from a forgiving file or folder selection."""

    selected = Path(selected_path).expanduser().resolve()
    if selected.is_file():
        if selected.suffix.casefold() != ".json":
            raise ValueError("请选择任务文件夹，或选择标注项目 JSON。")
        return selected
    if not selected.is_dir():
        raise FileNotFoundError(f"文件夹不存在：{selected}")

    # Accept the task root, a folder inside it (for example images), or a parent
    # containing exactly one task folder.  Avoid an unbounded recursive scan.
    for folder in (selected, *tuple(selected.parents)[:3]):
        candidate = folder / "annotation_project.json"
        if candidate.is_file():
            return candidate

    def looks_like_project_json(path: Path) -> bool:
        import json

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            return False
        required = {
            "schema_version",
            "name",
            "modality",
            "image_root",
            "label_root",
            "manifest_path",
            "records",
        }
        return isinstance(payload, dict) and required.issubset(payload)

    legacy_candidates = sorted(
        path for path in selected.glob("*.json") if looks_like_project_json(path)
    )
    if len(legacy_candidates) == 1:
        return legacy_candidates[0]
    if len(legacy_candidates) > 1:
        raise ValueError("这个文件夹包含多个标注项目，请分别存放后再选择。")
    try:
        candidates = sorted(
            child / "annotation_project.json"
            for child in selected.iterdir()
            if child.is_dir() and (child / "annotation_project.json").is_file()
        )
    except OSError:
        candidates = []
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError("这个位置包含多个标注任务，请进入具体任务文件夹后再选择。")
    raise FileNotFoundError(
        "没有找到 annotation_project.json。\n\n"
        "请先完整解压负责人发送的任务包，再选择包含 images、labels 和 "
        "annotation_project.json 的任务文件夹。"
    )


def next_task_directory(
    parent_directory: str | Path,
    *,
    modality: str,
    timestamp: str | None = None,
) -> Path:
    """Choose a unique, human-readable child directory for a new task."""

    parent = Path(parent_directory).expanduser().resolve()
    if not parent.is_dir():
        raise FileNotFoundError(f"保存位置不存在：{parent}")
    if modality not in {"haadf_stem", "hrtem"}:
        raise ValueError(f"不支持的成像模式：{modality}")
    modality_name = "HAADF-STEM" if modality == "haadf_stem" else "HRTEM"
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    base = parent / f"{modality_name}标注任务_{stamp}"
    candidate = base
    counter = 2
    while candidate.exists():
        candidate = parent / f"{base.name}_{counter}"
        counter += 1
    return candidate


def _snapshot(
    document: AnnotationDocument,
) -> tuple[
    list[tuple[float, float]],
    list[tuple[float, float, float, float]],
]:
    return (copy.deepcopy(document.points_xy), copy.deepcopy(document.coverage_regions_xyxy))


class AnnotationApp:
    """Manual point annotator with explicit annotation coverage semantics."""

    AUTOSAVE_DELAY_MS = 700

    def __init__(self, root: object, project: AnnotationProject) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        apply_tk_font_defaults(root)
        self.project = project
        self.current_index = 0 if project.records else -1
        self.record: ProjectRecord | None = None
        self.document: AnnotationDocument | None = None
        self.raw_image: np.ndarray | None = None
        self._loading = False
        self._modified = False
        self._autosave_after_id: str | None = None
        self._undo_stack: list[object] = []
        self._redo_stack: list[object] = []
        self._dragging_index: int | None = None
        self._drag_original: tuple[float, float] | None = None
        self._point_artist = None
        self._region_artists: list[object] = []
        self._stats = project_statistics(project)

        self.root.title(f"原子中心手工标注 — {project.modality}")
        self.root.resizable(True, True)
        self._setup_ui()
        self.window_geometry = fit_window_to_screen(
            self.root,
            preferred_size=(1580, 940),
            minimum_size=(900, 600),
        )
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        if project.records:
            self._load_index(0)
        else:
            self._show_empty_project()

    def _setup_ui(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        import matplotlib

        matplotlib.use("TkAgg")
        matplotlib.rcParams["font.sans-serif"] = [
            BUNDLED_FONT_FAMILY,
            "Microsoft YaHei",
            "SimHei",
            "DejaVu Sans",
        ]
        matplotlib.rcParams["axes.unicode_minus"] = False
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg,
            NavigationToolbar2Tk,
        )
        from matplotlib.figure import Figure
        from matplotlib.widgets import RectangleSelector

        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)
        self.main_frame = main

        # Pack the fixed-width control shell first so the expanding image canvas
        # can only consume the remaining space on small or high-DPI screens.
        right_shell = ttk.Frame(main, width=390)
        right_shell.pack(side=tk.RIGHT, fill=tk.Y)
        right_shell.pack_propagate(False)
        self.sidebar_shell = right_shell

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_panel = left
        self.figure = Figure(figsize=(10, 8), dpi=100, facecolor="#171717")
        self.axes = self.figure.add_subplot(111)
        self.axes.set_facecolor("#171717")
        self.canvas = FigureCanvasTkAgg(self.figure, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, left, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill=tk.X)
        self.roi_selector = RectangleSelector(
            self.axes,
            self._on_roi_selected,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
        )
        self.roi_selector.set_active(False)

        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_motion)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        style = ttk.Style(self.root)
        sidebar_background = style.lookup("TFrame", "background") or "#f0f0f0"
        self.sidebar_canvas = tk.Canvas(
            right_shell,
            background=sidebar_background,
            borderwidth=0,
            highlightthickness=0,
            width=365,
        )
        sidebar_scrollbar = ttk.Scrollbar(
            right_shell,
            orient=tk.VERTICAL,
            command=self.sidebar_canvas.yview,
        )
        self.sidebar_canvas.configure(yscrollcommand=sidebar_scrollbar.set)
        sidebar_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.Frame(self.sidebar_canvas, padding=7)
        self._sidebar_window = self.sidebar_canvas.create_window(
            (0, 0), window=right, anchor=tk.NW
        )
        right.bind("<Configure>", self._update_sidebar_scrollregion)
        self.sidebar_canvas.bind("<Configure>", self._resize_sidebar_content)

        info = ttk.LabelFrame(right, text="图像与进度", padding=7)
        info.pack(fill=tk.X, pady=(0, 5))
        self.file_label = ttk.Label(info, text="", wraplength=330)
        self.file_label.pack(anchor=tk.W)
        self.frame_label = ttk.Label(info, text="")
        self.frame_label.pack(anchor=tk.W)
        self.progress_label = ttk.Label(info, text="")
        self.progress_label.pack(anchor=tk.W)
        self.count_label = ttk.Label(info, text="")
        self.count_label.pack(anchor=tk.W)
        self.progress_value = tk.DoubleVar(value=0.0)
        ttk.Progressbar(info, variable=self.progress_value, maximum=100).pack(
            fill=tk.X, pady=(4, 0)
        )

        navigation = ttk.LabelFrame(right, text="导航与审核", padding=7)
        navigation.pack(fill=tk.X, pady=(0, 5))
        row = ttk.Frame(navigation)
        row.pack(fill=tk.X)
        ttk.Button(row, text="◀ 上一帧", command=self._previous).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2)
        )
        ttk.Button(row, text="下一帧 ▶", command=self._next).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0)
        )
        ttk.Button(
            navigation,
            text="保存草稿  Ctrl+S",
            command=self._save_draft,
        ).pack(fill=tk.X, pady=(4, 0))
        ttk.Button(
            navigation,
            text="审核通过并下一帧  Ctrl+Enter",
            command=self._review_and_next,
        ).pack(fill=tk.X, pady=(4, 0))

        editing = ttk.LabelFrame(right, text="点位与完整标注范围", padding=7)
        editing.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(
            editing,
            text="先指定已完整标注的范围；范围外不参与训练。",
            foreground="#9b6500",
            wraplength=330,
        ).pack(anchor=tk.W)
        row = ttk.Frame(editing)
        row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(row, text="整图已完整标注", command=self._set_full_coverage).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2)
        )
        ttk.Button(row, text="框选完整区域", command=self._start_roi_selection).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0)
        )
        row = ttk.Frame(editing)
        row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(row, text="删除最后区域", command=self._remove_last_region).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2)
        )
        ttk.Button(row, text="清空区域", command=self._clear_regions).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0)
        )
        row = ttk.Frame(editing)
        row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(row, text="撤销  Ctrl+Z", command=self._undo).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2)
        )
        ttk.Button(row, text="重做  Ctrl+Y", command=self._redo).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0)
        )

        display = ttk.LabelFrame(right, text="显示（不改变原图）", padding=7)
        display.pack(fill=tk.X, pady=(0, 5))
        row = ttk.Frame(display)
        row.pack(fill=tk.X)
        ttk.Label(row, text="分位数").pack(side=tk.LEFT)
        self.low_var = tk.StringVar(value="1.0")
        self.high_var = tk.StringVar(value="99.0")
        ttk.Label(row, text="低").pack(side=tk.LEFT, padx=(7, 2))
        ttk.Entry(row, textvariable=self.low_var, width=4).pack(side=tk.LEFT)
        ttk.Label(row, text="高").pack(side=tk.LEFT, padx=(7, 2))
        ttk.Entry(row, textvariable=self.high_var, width=4).pack(side=tk.LEFT)
        ttk.Button(row, text="应用", command=self._apply_contrast).pack(
            side=tk.RIGHT
        )
        self.invert_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            display,
            text="反相显示",
            variable=self.invert_var,
            command=self._apply_contrast,
        ).pack(anchor=tk.W, pady=(3, 0))
        row = ttk.Frame(display)
        row.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(row, text="显示/导出框直径(px)").pack(side=tk.LEFT)
        self.diameter_var = tk.DoubleVar(value=6.0)
        self.diameter_var.trace_add("write", lambda *_args: self._diameter_changed())
        ttk.Spinbox(
            row,
            textvariable=self.diameter_var,
            from_=1.0,
            to=40.0,
            increment=0.5,
            width=7,
            command=self._diameter_changed,
        ).pack(side=tk.RIGHT)

        metadata = ttk.LabelFrame(right, text="采集信息（可选）", padding=7)
        metadata.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(
            metadata,
            text="不知道可以留空，不影响审核和导出。",
            foreground="#555555",
            wraplength=330,
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 3))
        self.metadata_vars: dict[str, object] = {}
        fields = metadata_fields_for_modality(self.project.modality)
        for row_index, (field_name, label, choices) in enumerate(fields):
            label_row = row_index * 2 + 1
            input_row = label_row + 1
            ttk.Label(metadata, text=label).grid(
                row=label_row,
                column=0,
                columnspan=2,
                sticky=tk.W,
                pady=(3 if row_index else 0, 1),
            )
            variable = tk.StringVar(value=DEFAULT_METADATA[field_name])
            self.metadata_vars[field_name] = variable
            if choices:
                widget = ttk.Combobox(
                    metadata,
                    textvariable=variable,
                    values=choices,
                    state="readonly",
                    width=22,
                )
            else:
                widget = ttk.Entry(metadata, textvariable=variable, width=24)
            widget.grid(
                row=input_row,
                column=0,
                columnspan=2,
                sticky=tk.EW,
                pady=(0, 1),
            )
            variable.trace_add("write", self._metadata_changed)
        metadata.columnconfigure(0, weight=1)
        ttk.Button(
            metadata,
            text="应用到同文件夹的未审核图像",
            command=self._apply_metadata_to_folder,
        ).grid(
            row=len(fields) * 2 + 1,
            column=0,
            columnspan=2,
            sticky=tk.EW,
            pady=(6, 0),
        )

        operations = ttk.LabelFrame(right, text="任务与数据", padding=7)
        operations.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(
            operations,
            text="① 打开图像文件夹",
            command=self._open_image_folder,
        ).pack(fill=tk.X)
        ttk.Button(
            operations,
            text="② 扫描新增图像",
            command=self._sync_project,
        ).pack(fill=tk.X, pady=(4, 0))
        ttk.Button(
            operations,
            text="导出已审核训练数据…",
            command=self._export_yolo,
        ).pack(fill=tk.X, pady=(4, 0))
        ttk.Label(
            operations,
            text=(
                "空任务先打开图像文件夹并复制原图，再扫描；"
                "左键添加/拖动，右键删除，滚轮缩放。"
            ),
            wraplength=330,
            foreground="#555555",
        ).pack(anchor=tk.W, pady=(4, 0))

        self.status_label = ttk.Label(
            right, text="就绪", relief=tk.SUNKEN, anchor=tk.W, wraplength=345
        )
        self.status_label.pack(fill=tk.X, pady=(0, 2))
        self._bind_sidebar_mousewheel(right)
        self.root.update_idletasks()
        desired_sidebar_width = (
            right.winfo_reqwidth() + sidebar_scrollbar.winfo_reqwidth() + 6
        )
        maximum_sidebar_width = max(
            390, int(self.root.winfo_screenwidth() * 0.55)
        )
        self.sidebar_shell_width = min(
            max(390, desired_sidebar_width), maximum_sidebar_width
        )
        right_shell.configure(width=self.sidebar_shell_width)

    def _update_sidebar_scrollregion(self, _event: object | None = None) -> None:
        bounds = self.sidebar_canvas.bbox("all")
        if bounds is not None:
            self.sidebar_canvas.configure(scrollregion=bounds)

    def _resize_sidebar_content(self, event: object) -> None:
        width = max(1, int(getattr(event, "width", 1)))
        self.sidebar_canvas.itemconfigure(self._sidebar_window, width=width)

    def _bind_sidebar_mousewheel(self, widget: object) -> None:
        widget.bind("<MouseWheel>", self._on_sidebar_mousewheel, add="+")
        for child in widget.winfo_children():
            self._bind_sidebar_mousewheel(child)

    def _on_sidebar_mousewheel(self, event: object) -> str | None:
        delta = int(getattr(event, "delta", 0))
        if not delta:
            return None
        direction = -1 if delta > 0 else 1
        self.sidebar_canvas.yview_scroll(direction * 3, "units")
        return "break"

    def _open_image_folder(self) -> None:
        import os
        import subprocess
        import sys
        from tkinter import messagebox

        self.project.image_root.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(str(self.project.image_root))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(self.project.image_root)])
            else:
                subprocess.Popen(["xdg-open", str(self.project.image_root)])
        except Exception as exc:
            messagebox.showerror(
                "无法打开图像文件夹",
                f"请手动打开：\n{self.project.image_root}\n\n{exc}",
                parent=self.root,
            )

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-s>", lambda _event: self._save_draft())
        self.root.bind("<Control-z>", lambda _event: self._undo())
        self.root.bind("<Control-y>", lambda _event: self._redo())
        self.root.bind("<Control-Return>", lambda _event: self._review_and_next())
        self.root.bind("<Prior>", lambda _event: self._previous())
        self.root.bind("<Next>", lambda _event: self._next())
        self.root.bind("<Alt-Left>", lambda _event: self._previous())
        self.root.bind("<Alt-Right>", lambda _event: self._next())
        self.root.bind("<Escape>", lambda _event: self._cancel_roi_selection())

    def _show_empty_project(self) -> None:
        self.axes.clear()
        self.axes.set_facecolor("#171717")
        self.axes.text(
            0.5,
            0.5,
            (
                "这个任务还没有图像\n\n"
                "① 点击右侧“打开图像文件夹”并复制原始图像\n"
                "② 返回软件，点击“扫描新增图像”\n\n"
                f"任务位置：{self.project.project_path.parent}"
            ),
            color="white",
            ha="center",
            va="center",
            wrap=True,
            transform=self.axes.transAxes,
        )
        self.axes.set_axis_off()
        self.canvas.draw_idle()
        self.file_label.config(text="没有图像")
        self.frame_label.config(text="")
        self.progress_label.config(text="进度：0 / 0")
        self.count_label.config(text="")
        self.status_label.config(text="等待原始图像：请按 ①、② 两步操作")

    def _load_index(self, index: int) -> None:
        from tkinter import messagebox

        if not self.project.records:
            self._show_empty_project()
            return
        if self._modified and not self._save_current(silent=True):
            return
        index = max(0, min(index, len(self.project.records) - 1))
        record = self.project.records[index]
        try:
            document = self.project.load_document(record)
            loaded = load_image(
                self.project.image_file(record),
                frame_index=record.frame_index,
                series_index=record.series_index,
                normalize=False,
            )
        except Exception as exc:
            messagebox.showerror("无法读取图像/标签", str(exc))
            self.status_label.config(text=f"读取失败：{exc}")
            return

        self.current_index = index
        self.record = record
        self.document = document
        self.raw_image = loaded.image
        self._modified = False
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._dragging_index = None
        self._loading = True
        try:
            for key, variable in self.metadata_vars.items():
                variable.set(document.metadata.get(key, ""))
            self.diameter_var.set(document.atom_diameter_px)
        finally:
            self._loading = False
        self._draw_new_image()
        self._update_info()
        self.status_label.config(text="已加载；修改会自动保存为草稿")

    def _draw_new_image(self) -> None:
        if self.raw_image is None:
            return
        # Axes.clear() removes every child artist and invalidates its remove
        # callback.  Drop our references first so _redraw_overlays() does not
        # try to remove already-detached artists while switching images.
        self._point_artist = None
        self._region_artists.clear()
        self.axes.clear()
        self.axes.set_facecolor("#171717")
        display = self._display_image()
        self.axes.imshow(display, cmap="gray", origin="upper", aspect="equal")
        height, width = display.shape
        self.axes.set_xlim(-0.5, width - 0.5)
        self.axes.set_ylim(height - 0.5, -0.5)
        self.axes.tick_params(colors="#aaaaaa", labelsize=8)
        self.roi_selector.disconnect_events()
        self.roi_selector = self.roi_selector.__class__(
            self.axes,
            self._on_roi_selected,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
        )
        self.roi_selector.set_active(False)
        self._redraw_overlays()

    def _display_image(self) -> np.ndarray:
        if self.raw_image is None:
            return np.zeros((2, 2), dtype=np.float32)
        try:
            low = float(self.low_var.get())
            high = float(self.high_var.get())
            display = normalize_percentile(self.raw_image, low=low, high=high)
        except Exception:
            display = normalize_percentile(self.raw_image, low=1.0, high=99.0)
        if self.invert_var.get():
            display = 1.0 - display
        return display

    def _redraw_overlays(self) -> None:
        from matplotlib.patches import Rectangle

        if self.document is None:
            return
        if self._point_artist is not None:
            try:
                self._point_artist.remove()
            except (ValueError, NotImplementedError):
                pass
            self._point_artist = None
        for artist in self._region_artists:
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        self._region_artists.clear()

        if self.document.points_xy:
            points = np.asarray(self.document.points_xy, dtype=np.float64)
            diameter = max(2.0, float(self.diameter_var.get()))
            self._point_artist = self.axes.scatter(
                points[:, 0],
                points[:, 1],
                s=(diameter + 4.0) ** 2,
                facecolors="none",
                edgecolors="#00ff72",
                linewidths=1.2,
                zorder=4,
            )
        for index, (x0, y0, x1, y1) in enumerate(
            self.document.coverage_regions_xyxy
        ):
            rectangle = Rectangle(
                (x0 - 0.5, y0 - 0.5),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="#ffd43b",
                linewidth=1.4,
                linestyle="--",
                zorder=3,
                label=f"coverage-{index}",
            )
            self.axes.add_patch(rectangle)
            self._region_artists.append(rectangle)
        state = {
            "unannotated": "未标注",
            "draft": "草稿",
            "reviewed": "已审核",
        }[self.document.review_status]
        self.axes.set_title(
            f"{len(self.document.points_xy)} 个原子 | {len(self.document.coverage_regions_xyxy)} 个完整区域 | {state}",
            color="white",
            fontsize=10,
        )
        self.canvas.draw_idle()

    def _toolbar_active(self) -> bool:
        return bool(str(getattr(self.toolbar, "mode", "")))

    def _nearest_point(self, event: object, threshold_px: float = 11.0) -> int | None:
        if self.document is None or not self.document.points_xy:
            return None
        event_x = getattr(event, "x", None)
        event_y = getattr(event, "y", None)
        if event_x is None or event_y is None:
            return None
        points = np.asarray(self.document.points_xy, dtype=np.float64)
        screen = self.axes.transData.transform(points)
        distances = np.hypot(screen[:, 0] - event_x, screen[:, 1] - event_y)
        index = int(np.argmin(distances))
        return index if distances[index] <= threshold_px else None

    def _point_is_editable(self, point: tuple[float, float]) -> bool:
        if self.document is None:
            return False
        height, width = self.document.image_shape
        x, y = point
        return (
            0.0 <= x < width
            and 0.0 <= y < height
            and point_in_regions(point, self.document.coverage_regions_xyxy)
        )

    def _on_mouse_press(self, event: object) -> None:
        if (
            self.document is None
            or getattr(event, "inaxes", None) is not self.axes
            or getattr(event, "xdata", None) is None
            or self.roi_selector.active
            or self._toolbar_active()
        ):
            return
        x = float(event.xdata)
        y = float(event.ydata)
        button = getattr(event, "button", None)
        if button == 1:
            nearest = self._nearest_point(event)
            if nearest is not None:
                self._dragging_index = nearest
                self._drag_original = self.document.points_xy[nearest]
                return
            if not self.document.coverage_regions_xyxy:
                self.status_label.config(text="请先选择“整图已完整标注”或框选完整区域")
                return
            if not self._point_is_editable((x, y)):
                self.status_label.config(text="该位置在完整标注范围之外")
                return
            self._push_history()
            self.document.points_xy.append((x, y))
            self._geometry_changed()
        elif button == 3:
            nearest = self._nearest_point(event, threshold_px=15.0)
            if nearest is not None:
                self._push_history()
                del self.document.points_xy[nearest]
                self._geometry_changed()

    def _on_mouse_motion(self, event: object) -> None:
        if (
            self.document is None
            or self._dragging_index is None
            or getattr(event, "xdata", None) is None
            or getattr(event, "ydata", None) is None
        ):
            return
        self.document.points_xy[self._dragging_index] = (
            float(event.xdata),
            float(event.ydata),
        )
        self._redraw_overlays()

    def _on_mouse_release(self, event: object) -> None:
        if self.document is None or self._dragging_index is None:
            return
        index = self._dragging_index
        original = self._drag_original
        self._dragging_index = None
        self._drag_original = None
        if original is None:
            return
        candidate = self.document.points_xy[index]
        if not self._point_is_editable(candidate):
            self.document.points_xy[index] = original
            self._redraw_overlays()
            self.status_label.config(text="点位不能移出图像或完整标注范围")
            return
        if np.hypot(candidate[0] - original[0], candidate[1] - original[1]) < 1e-6:
            return
        snapshot_points = copy.deepcopy(self.document.points_xy)
        snapshot_points[index] = original
        self._undo_stack.append(
            (snapshot_points, copy.deepcopy(self.document.coverage_regions_xyxy))
        )
        self._redo_stack.clear()
        self._geometry_changed()

    def _on_scroll(self, event: object) -> None:
        if getattr(event, "inaxes", None) is not self.axes:
            return
        x = getattr(event, "xdata", None)
        y = getattr(event, "ydata", None)
        if x is None or y is None:
            return
        x_limits = self.axes.get_xlim()
        y_limits = self.axes.get_ylim()
        factor = 0.8 if getattr(event, "button", "") == "up" else 1.25
        new_width = (x_limits[1] - x_limits[0]) * factor
        new_height = (y_limits[1] - y_limits[0]) * factor
        relative_x = (x - x_limits[0]) / (x_limits[1] - x_limits[0])
        relative_y = (y - y_limits[0]) / (y_limits[1] - y_limits[0])
        self.axes.set_xlim(x - new_width * relative_x, x + new_width * (1 - relative_x))
        self.axes.set_ylim(y - new_height * relative_y, y + new_height * (1 - relative_y))
        self.canvas.draw_idle()

    def _turn_off_toolbar_mode(self) -> None:
        mode = str(getattr(self.toolbar, "mode", "")).lower()
        if "pan" in mode:
            self.toolbar.pan()
        elif "zoom" in mode:
            self.toolbar.zoom()

    def _start_roi_selection(self) -> None:
        if self.document is None:
            return
        self._turn_off_toolbar_mode()
        self.roi_selector.set_active(True)
        self.status_label.config(text="拖动左键框选一个已完整标注区域；Esc 取消")

    def _cancel_roi_selection(self) -> None:
        self.roi_selector.set_active(False)
        self.status_label.config(text="已取消框选")

    def _on_roi_selected(self, start: object, end: object) -> None:
        if self.document is None:
            return
        self.roi_selector.set_active(False)
        values = (
            getattr(start, "xdata", None),
            getattr(start, "ydata", None),
            getattr(end, "xdata", None),
            getattr(end, "ydata", None),
        )
        if any(value is None for value in values):
            return
        sx, sy, ex, ey = (float(value) for value in values)
        height, width = self.document.image_shape
        left, right = sorted((sx, ex))
        top, bottom = sorted((sy, ey))
        x0 = float(max(0, int(np.floor(left))))
        x1 = float(min(width, int(np.ceil(right))))
        y0 = float(max(0, int(np.floor(top))))
        y1 = float(min(height, int(np.ceil(bottom))))
        if x1 - x0 < 2.0 or y1 - y0 < 2.0:
            self.status_label.config(text="标注区域过小，已忽略")
            return
        for rx0, ry0, rx1, ry1 in self.document.coverage_regions_xyxy:
            if min(x1, rx1) > max(x0, rx0) and min(y1, ry1) > max(y0, ry0):
                self.status_label.config(text="新区域与已有区域重叠，已忽略")
                return
        self._push_history()
        self.document.coverage_regions_xyxy.append((x0, y0, x1, y1))
        self._geometry_changed()

    def _set_full_coverage(self) -> None:
        if self.document is None:
            return
        height, width = self.document.image_shape
        self._push_history()
        self.document.coverage_regions_xyxy = [
            (0.0, 0.0, float(width), float(height))
        ]
        self._geometry_changed()

    def _remove_last_region(self) -> None:
        if self.document is None or not self.document.coverage_regions_xyxy:
            return
        self._push_history()
        self.document.coverage_regions_xyxy.pop()
        self._geometry_changed()

    def _clear_regions(self) -> None:
        if self.document is None or not self.document.coverage_regions_xyxy:
            return
        self._push_history()
        self.document.coverage_regions_xyxy.clear()
        self._geometry_changed()

    def _push_history(self) -> None:
        if self.document is None:
            return
        self._undo_stack.append(_snapshot(self.document))
        if len(self._undo_stack) > 200:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _undo(self) -> None:
        if self.document is None or not self._undo_stack:
            return
        self._redo_stack.append(_snapshot(self.document))
        points, regions = self._undo_stack.pop()
        self.document.points_xy = points
        self.document.coverage_regions_xyxy = regions
        self._geometry_changed(clear_redo=False)

    def _redo(self) -> None:
        if self.document is None or not self._redo_stack:
            return
        self._undo_stack.append(_snapshot(self.document))
        points, regions = self._redo_stack.pop()
        self.document.points_xy = points
        self.document.coverage_regions_xyxy = regions
        self._geometry_changed(clear_redo=False)

    def _geometry_changed(self, *, clear_redo: bool = True) -> None:
        if self.document is None:
            return
        if clear_redo:
            self._redo_stack.clear()
        self.document.review_status = "draft"
        self._modified = True
        self._redraw_overlays()
        self._update_info()
        self._schedule_autosave()

    def _metadata_changed(self, *_args: object) -> None:
        if self._loading or self.document is None:
            return
        for key, variable in self.metadata_vars.items():
            self.document.metadata[key] = str(variable.get())
        self.document.review_status = "draft"
        self._modified = True
        self._redraw_overlays()
        self._update_info()
        self._schedule_autosave()

    def _diameter_changed(self) -> None:
        if self._loading or self.document is None:
            return
        try:
            value = float(self.diameter_var.get())
        except Exception:
            return
        if value <= 0:
            return
        self.document.atom_diameter_px = value
        self.document.review_status = "draft"
        self._modified = True
        self._redraw_overlays()
        self._schedule_autosave()

    def _schedule_autosave(self) -> None:
        if self._autosave_after_id is not None:
            self.root.after_cancel(self._autosave_after_id)
        self._autosave_after_id = self.root.after(
            self.AUTOSAVE_DELAY_MS, lambda: self._save_current(silent=True)
        )

    def _save_current(self, *, silent: bool) -> bool:
        from tkinter import messagebox

        if self.document is None:
            return True
        if self._autosave_after_id is not None:
            self.root.after_cancel(self._autosave_after_id)
            self._autosave_after_id = None
        if not self._modified and silent:
            return True
        try:
            for key, variable in self.metadata_vars.items():
                self.document.metadata[key] = str(variable.get())
            self.document.atom_diameter_px = float(self.diameter_var.get())
            self.project.save_document(self.document)
            self._modified = False
            self._stats = project_statistics(self.project)
        except Exception as exc:
            if not silent:
                messagebox.showerror("保存失败", str(exc))
            self.status_label.config(text=f"自动保存失败：{exc}")
            return False
        if not silent:
            self.project.write_manifest()
            self.status_label.config(text="草稿已原子化保存，manifest 已更新")
        else:
            self.status_label.config(text="草稿已自动保存")
        self._update_info()
        return True

    def _save_draft(self) -> None:
        self._save_current(silent=False)

    def _review_and_next(self) -> None:
        from tkinter import messagebox

        if self.document is None:
            return
        for key, variable in self.metadata_vars.items():
            self.document.metadata[key] = str(variable.get())
        try:
            validate_document(self.document, for_review=True)
            self.document.review_status = "reviewed"
            self._modified = True
            if not self._save_current(silent=True):
                return
            self.project.write_manifest()
        except Exception as exc:
            self.document.review_status = "draft"
            messagebox.showerror("不能审核通过", str(exc))
            self.status_label.config(text=f"审核未通过：{exc}")
            return
        self.status_label.config(text="审核通过")
        self._update_info()
        if self.current_index < len(self.project.records) - 1:
            self._load_index(self.current_index + 1)

    def _previous(self) -> None:
        if self.current_index > 0:
            self._load_index(self.current_index - 1)

    def _next(self) -> None:
        if self.current_index < len(self.project.records) - 1:
            self._load_index(self.current_index + 1)

    def _apply_contrast(self) -> None:
        from tkinter import messagebox

        if self.raw_image is None:
            return
        try:
            low = float(self.low_var.get())
            high = float(self.high_var.get())
            if not (0.0 <= low < high <= 100.0):
                raise ValueError("分位数必须满足 0 ≤ 低 < 高 ≤ 100")
        except Exception as exc:
            messagebox.showerror("显示参数错误", str(exc))
            return
        limits = self.axes.get_xlim(), self.axes.get_ylim()
        self.axes.images[0].set_data(self._display_image())
        self.axes.set_xlim(*limits[0])
        self.axes.set_ylim(*limits[1])
        self.canvas.draw_idle()

    def _apply_metadata_to_folder(self) -> None:
        from tkinter import messagebox

        if self.document is None or self.record is None:
            return
        if not messagebox.askyesno(
            "批量应用元数据",
            "仅更新同一原始图像文件夹中的未审核帧；已审核帧不会改变。继续吗？",
        ):
            return
        if not self._save_current(silent=True):
            return
        current_parent = Path(self.record.image_path).parent
        values = {key: str(variable.get()) for key, variable in self.metadata_vars.items()}
        changed = 0
        for record in self.project.records:
            if Path(record.image_path).parent != current_parent:
                continue
            document = self.project.load_document(record)
            if document.review_status == "reviewed":
                continue
            document.metadata.update(values)
            if document.review_status == "unannotated":
                document.review_status = "draft"
            self.project.save_document(document)
            changed += 1
        self.project.write_manifest()
        self._stats = project_statistics(self.project)
        self.status_label.config(text=f"已更新同目录 {changed} 个未审核帧")
        self._load_index(self.current_index)

    def _sync_project(self) -> None:
        from tkinter import messagebox

        if self._modified and not self._save_current(silent=True):
            return
        added, failures = self.project.sync_images()
        self.project.save()
        self.project.write_manifest()
        self._stats = project_statistics(self.project)
        if self.current_index < 0 and self.project.records:
            self._load_index(0)
        self._update_info()
        messagebox.showinfo(
            "扫描完成",
            f"新增 {added} 个二维图像/帧；无法读取 {failures} 个文件。",
        )

    def _export_yolo(self) -> None:
        from tkinter import filedialog, messagebox

        if self._modified and not self._save_current(silent=True):
            return
        self.project.write_manifest()
        parent = filedialog.askdirectory(
            title="选择导出父目录（工具会新建带时间戳的子目录）",
            initialdir=str(self.project.project_path.parent),
        )
        if not parent:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = Path(parent) / f"{self.project.modality}_yolo_{stamp}"
        try:
            summary = export_yolo_dataset(
                self.project,
                target,
                box_size_px=float(self.diameter_var.get()),
                percentile_low=float(self.low_var.get()),
                percentile_high=float(self.high_var.get()),
            )
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))
            return
        messagebox.showinfo(
            "导出完成",
            f"{summary.source_images} 个原始帧 → {summary.derived_images} 个训练图，"
            f"共 {summary.points} 个点。\n\n{summary.output_dir}",
        )
        self.status_label.config(text=f"已导出：{summary.output_dir}")

    def _update_info(self) -> None:
        if self.document is None or self.record is None:
            return
        stats = self._stats
        self.file_label.config(text=f"文件：{self.record.image_path}")
        frame = "单帧" if self.record.frame_index is None else f"帧 {self.record.frame_index}"
        self.frame_label.config(
            text=f"series {self.record.series_index} / {frame} / {self.record.image_shape[1]}×{self.record.image_shape[0]}"
        )
        self.progress_label.config(
            text=f"进度：{self.current_index + 1} / {len(self.project.records)}；已审核 {stats.reviewed}"
        )
        self.count_label.config(
            text=f"当前 {len(self.document.points_xy)} 点；已审核总计 {stats.reviewed_points} 点"
        )
        self.progress_value.set(
            100.0 * stats.reviewed / max(1, len(self.project.records))
        )

    def _on_close(self) -> None:
        from tkinter import messagebox

        if self._modified and not self._save_current(silent=True):
            if not messagebox.askyesno("保存失败", "草稿未能保存，仍然退出吗？"):
                return
        try:
            self.project.save()
            self.project.write_manifest()
        except Exception as exc:
            if not messagebox.askyesno(
                "项目清单保存失败", f"{exc}\n\n仍然退出吗？"
            ):
                return
        self.root.destroy()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="原子中心点位手工标注工具")
    parser.add_argument(
        "task",
        nargs="?",
        type=Path,
        help="可选任务文件夹或 annotation_project.json（也支持拖放到 EXE）",
    )
    parser.add_argument("--project", type=Path, help="标注项目 JSON")
    parser.add_argument("--images", type=Path, help="新建项目时的原始图像目录")
    parser.add_argument("--labels", type=Path, help="新建项目时的精确点标签目录")
    parser.add_argument("--manifest", type=Path, help="自动生成的 CSV manifest")
    parser.add_argument("--modality", choices=("haadf_stem", "hrtem"))
    parser.add_argument("--smoke-test", type=Path, help=argparse.SUPPRESS)
    return parser


def create_portable_task(
    task_directory: str | Path,
    *,
    modality: str,
) -> AnnotationProject:
    """Create or reopen a self-contained task folder suitable for ZIP transfer."""

    task_root = Path(task_directory).expanduser().resolve()
    project_path = task_root / "annotation_project.json"
    if project_path.is_file():
        project = AnnotationProject.load(project_path)
        if project.modality != modality:
            raise ValueError(
                f"任务已是 {project.modality} 项目，不能改为 {modality}"
            )
        project.sync_images()
        project.save()
        project.write_manifest()
        return project
    return AnnotationProject.create(
        project_path,
        image_root=task_root / "images",
        modality=modality,
        label_root=task_root / "labels",
        manifest_path=task_root / "manifest.csv",
        name=f"{task_root.name} - {modality}",
    )


class StartupDialog:
    """Plain-language launcher shown when the application has no CLI arguments."""

    def __init__(self, root: object) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        apply_tk_font_defaults(root)
        self.project: AnnotationProject | None = None
        root.title("原子中心标注器")
        root.resizable(True, True)
        root.protocol("WM_DELETE_WINDOW", root.destroy)
        root.bind("<Escape>", lambda _event: root.destroy())

        frame = ttk.Frame(root, padding=22)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            frame,
            text="原子中心标注器",
            font=(BUNDLED_FONT_FAMILY, 18, "bold"),
        ).pack(anchor=tk.W)
        ttk.Label(
            frame,
            text="HAADF-STEM / HRTEM 原子中心精确点位标注",
            font=(BUNDLED_FONT_FAMILY, 10),
            foreground="#555555",
        ).pack(anchor=tk.W, pady=(2, 14))

        ttk.Label(
            frame,
            text="如果负责人发给你一个任务包，请使用这一项：",
            font=(BUNDLED_FONT_FAMILY, 11, "bold"),
        ).pack(anchor=tk.W, pady=(0, 5))
        self.open_button = ttk.Button(
            frame,
            text="打开收到的标注任务文件夹…",
            command=self._open_existing,
        )
        self.open_button.pack(fill=tk.X, ipady=9)
        ttk.Label(
            frame,
            text="不需要寻找 JSON 文件；直接选择解压后的整个任务文件夹。",
            foreground="#555555",
        ).pack(anchor=tk.W, pady=(5, 0))

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=14)
        ttk.Label(
            frame,
            text="任务负责人：创建新的空任务",
            font=(BUNDLED_FONT_FAMILY, 11, "bold"),
        ).pack(anchor=tk.W)
        ttk.Label(
            frame,
            text=(
                "只需选择保存位置，程序会自动新建任务文件夹和所需文件。"
            ),
            wraplength=660,
            foreground="#555555",
        ).pack(anchor=tk.W, pady=(3, 8))
        row = ttk.Frame(frame)
        row.pack(fill=tk.X)
        self.haadf_button = ttk.Button(
            row,
            text="新建 HAADF-STEM 任务…",
            command=lambda: self._new_task("haadf_stem"),
        )
        self.haadf_button.pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5), ipady=7
        )
        self.hrtem_button = ttk.Button(
            row,
            text="新建 HRTEM 任务…",
            command=lambda: self._new_task("hrtem"),
        )
        self.hrtem_button.pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0), ipady=7
        )

        ttk.Label(
            frame,
            text=(
                "提示：一个任务文件夹必须整体保存和交回，不要只发送 labels。"
            ),
            wraplength=660,
            foreground="#775500",
        ).pack(anchor=tk.W, pady=(14, 0))
        footer = ttk.Frame(frame)
        footer.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(footer, text="使用说明", command=self._show_help).pack(
            side=tk.LEFT
        )
        ttk.Button(footer, text="退出", command=root.destroy).pack(side=tk.RIGHT)

        self.primary_buttons = (
            self.open_button,
            self.haadf_button,
            self.hrtem_button,
        )
        root.update_idletasks()
        self.required_size = (root.winfo_reqwidth(), root.winfo_reqheight())
        self.window_geometry = fit_window_to_screen(
            root,
            preferred_size=(720, 0),
            minimum_size=(600, 380),
        )
        self.content_fits = (
            self.window_geometry[0] >= self.required_size[0]
            and self.window_geometry[1] >= self.required_size[1]
        )
        self.open_button.focus_set()

    def _show_help(self) -> None:
        from tkinter import messagebox

        messagebox.showinfo(
            "使用说明",
            "标注人员：\n"
            "1. 完整解压负责人发送的任务包。\n"
            "2. 点击“打开收到的标注任务文件夹”。\n"
            "3. 选择含 annotation_project.json、images 和 labels 的文件夹。\n"
            "4. 标注后退出，把整个任务文件夹压缩交回。\n\n"
            "任务负责人：\n"
            "选择 HAADF-STEM 或 HRTEM，再选择任务保存位置。",
            parent=self.root,
        )

    def _open_existing(self) -> None:
        from tkinter import filedialog, messagebox

        selected = filedialog.askdirectory(
            title="选择解压后的标注任务文件夹",
            mustexist=True,
        )
        if not selected:
            return
        try:
            project_path = find_task_project(selected)
            project = AnnotationProject.load(project_path)
            project.sync_images()
            project.save()
            project.write_manifest()
        except Exception as exc:
            messagebox.showerror(
                "这不是有效的标注任务",
                str(exc),
                parent=self.root,
            )
            return
        self.project = project
        self.root.destroy()

    def _new_task(self, modality: str) -> None:
        from tkinter import filedialog, messagebox

        parent = filedialog.askdirectory(
            title="选择新任务的保存位置（程序会自动新建文件夹）",
            mustexist=True,
        )
        if not parent:
            return
        try:
            task_directory = next_task_directory(parent, modality=modality)
            project = create_portable_task(task_directory, modality=modality)
        except Exception as exc:
            messagebox.showerror("任务无法创建", str(exc), parent=self.root)
            return
        messagebox.showinfo(
            "任务创建成功",
            f"任务文件夹：\n{project.project_path.parent}\n\n"
            "下一步进入主界面后，按 ①、② 两步添加原始图像。",
            parent=self.root,
        )
        self.project = project
        self.root.destroy()


def choose_project_interactively() -> AnnotationProject | None:
    import tkinter as tk

    root = tk.Tk()
    dialog = StartupDialog(root)
    root.mainloop()
    return dialog.project


def open_or_create_project(args: argparse.Namespace) -> AnnotationProject | None:
    import tkinter as tk
    from tkinter import messagebox

    project_path = args.project or args.task
    if project_path is None:
        return choose_project_interactively()

    if project_path.is_file() or project_path.is_dir():
        resolved_project = find_task_project(project_path)
        project = AnnotationProject.load(resolved_project)
        project.sync_images()
        project.save()
        project.write_manifest()
        return project
    if args.images is None or args.modality is None:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "缺少参数",
            "新建项目必须同时提供 --images 和 --modality。建议使用 scripts/start_annotation.ps1。",
        )
        root.destroy()
        return None
    return AnnotationProject.create(
        project_path,
        image_root=args.images,
        modality=args.modality,
        label_root=args.labels,
        manifest_path=args.manifest,
    )


def run_packaged_smoke_test(report_path: str | Path) -> int:
    """Exercise bundled Tk/TIFF/PNG/JSON code paths without user interaction."""

    import json
    import platform
    import sys
    import tempfile
    import traceback
    import tkinter as tk

    import tifffile

    from . import __version__
    from .annotations import export_yolo_dataset

    report_file = Path(report_path).expanduser().resolve()
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "ok": False,
        "version": __version__,
        "frozen": bool(getattr(sys, "frozen", False)),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "bundled_font_registered": configure_bundled_font(),
    }
    try:
        with tempfile.TemporaryDirectory(prefix="atom-annotator-smoke-") as temporary:
            task_root = Path(temporary) / "task"
            image_directory = task_root / "images" / "sample-smoke" / "run-smoke"
            image_directory.mkdir(parents=True)
            yy, xx = np.mgrid[:64, :80]
            image = (xx * 17 + yy * 9).astype(np.uint16)
            tifffile.imwrite(
                image_directory / "01-reviewed.tif",
                image,
                photometric="minisblack",
                metadata={"axes": "YX"},
            )
            next_yy, next_xx = np.mgrid[:72, :96]
            next_image = (next_xx * 11 + next_yy * 5).astype(np.uint16)
            tifffile.imwrite(
                image_directory / "02-unannotated.tif",
                next_image,
                photometric="minisblack",
                metadata={"axes": "YX"},
            )
            project = create_portable_task(task_root, modality="haadf_stem")
            record = project.records[0]
            document = project.load_document(record)
            document.coverage_regions_xyxy = [(0.0, 0.0, 80.0, 64.0)]
            document.points_xy = [(20.25, 18.5), (55.75, 42.25)]
            # Acquisition metadata is intentionally blank: packaged releases must
            # prove that a labeler can review and export without knowing it.
            document.metadata.update(
                {
                    "sample_id": "",
                    "acquisition_id": "",
                    "pixel_size": "",
                    "accelerating_voltage_kv": "",
                    "detector_inner_angle_mrad": "",
                    "detector_outer_angle_mrad": "",
                }
            )
            document.review_status = "reviewed"
            project.save_document(document)
            project.write_manifest()
            # Reproduce the normal hand-off sequence: export the reviewed first
            # image, then continue annotating the unreviewed second image.
            exported = export_yolo_dataset(project, task_root / "export")

            launcher_window = tk.Tk()
            launcher_window.withdraw()
            launcher_window.tk.call("tk", "scaling", 2.0)
            launcher = StartupDialog(launcher_window)
            launcher_window.update_idletasks()
            launcher_window.update()
            launcher_buttons_present = all(
                button.winfo_manager() for button in launcher.primary_buttons
            )
            launcher_window.destroy()

            window = tk.Tk()
            window.withdraw()
            window.attributes("-alpha", 0.0)
            window.tk.call("tk", "scaling", 2.0)
            app = AnnotationApp(window, project)
            window.geometry("1200x700")
            window.deiconify()
            window.update_idletasks()
            window.update()
            loaded_shape = list(app.raw_image.shape) if app.raw_image is not None else []
            sidebar_scrollable = bool(app.sidebar_canvas.cget("yscrollcommand"))
            sidebar_packed_first = (
                app.main_frame.pack_slaves()[0] is app.sidebar_shell
            )
            sidebar_content = app.sidebar_canvas.nametowidget(
                app.sidebar_canvas.itemcget(app._sidebar_window, "window")
            )
            sidebar_horizontal_fits = (
                sidebar_content.winfo_reqwidth()
                <= app.sidebar_canvas.winfo_width()
            )
            sidebar_bounds = app.sidebar_canvas.bbox("all")
            sidebar_vertical_overflow = (
                sidebar_bounds is not None
                and sidebar_bounds[3] > app.sidebar_canvas.winfo_height()
            )
            sidebar_visible_width = app.sidebar_canvas.winfo_width()
            sidebar_required_width = sidebar_content.winfo_reqwidth()
            app._next()
            window.update_idletasks()
            window.update()
            next_loaded_shape = (
                list(app.raw_image.shape) if app.raw_image is not None else []
            )
            image_switch_passed = (
                app.current_index == 1
                and app.document is not None
                and app.document.review_status == "unannotated"
                and next_loaded_shape == [72, 96]
                and app._point_artist is None
                and not app._region_artists
            )
            window.withdraw()
            window.destroy()

            interface_checks_passed = all(
                (
                    launcher.content_fits,
                    launcher_buttons_present,
                    sidebar_scrollable,
                    sidebar_packed_first,
                    sidebar_horizontal_fits,
                    sidebar_vertical_overflow,
                    image_switch_passed,
                )
            )
            report.update(
                {
                    "ok": interface_checks_passed,
                    "launcher_constructed": True,
                    "launcher_content_fits": launcher.content_fits,
                    "launcher_buttons_present": launcher_buttons_present,
                    "launcher_required_size": list(launcher.required_size),
                    "launcher_window_size": list(launcher.window_geometry[:2]),
                    "main_window_size": list(app.window_geometry[:2]),
                    "sidebar_scrollable": sidebar_scrollable,
                    "sidebar_packed_first": sidebar_packed_first,
                    "sidebar_horizontal_fits": sidebar_horizontal_fits,
                    "sidebar_vertical_overflow": sidebar_vertical_overflow,
                    "sidebar_shell_width": app.sidebar_shell_width,
                    "sidebar_visible_width": sidebar_visible_width,
                    "sidebar_required_width": sidebar_required_width,
                    "records": len(project.records),
                    "loaded_shape": loaded_shape,
                    "next_loaded_shape": next_loaded_shape,
                    "image_switch_passed": image_switch_passed,
                    "saved_points": len(project.load_document(record).points_xy),
                    "exported_images": exported.derived_images,
                    "exported_points": exported.points,
                }
            )
    except Exception as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    report_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if report["ok"] else 2


def main(argv: Sequence[str] | None = None) -> int:
    import tkinter as tk

    configure_bundled_font()
    args = _parser().parse_args(argv)
    if args.smoke_test is not None:
        return run_packaged_smoke_test(args.smoke_test)
    project = open_or_create_project(args)
    if project is None:
        return 0
    root = tk.Tk()
    AnnotationApp(root, project)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
