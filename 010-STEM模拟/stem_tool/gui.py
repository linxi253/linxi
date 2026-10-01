"""STEM(HAADF/ADF) 模拟工具主界面（Tk 三栏：结构与取向 / 电镜与探针 / 结果）。

界面设计的两条主线：

1. **代价前置**：HAADF 的耗时由"探测器外角→所需采样→网格×切片×探针数×声子组态"
   层层放大，跑完才知道要等半小时是不能接受的。因此左栏常驻「模拟计划」面板，
   参数一改就重算派生量与预估耗时，并在探测器外角超过采样可收集上限、
   声子组态过少、内角过小等情况下直接给出黄字警告。

2. **进度可见**：多进程 × 多线程的扫描按 (声子组态, 图像行块) 切分，
   进度条与状态栏实时反映"第几个组态、扫到第几行"。
"""

from __future__ import annotations

import os
import queue
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
from typing import Optional
import tkinter as tk

import numpy as np

from .export import default_basename, export_all
from .params import QUALITY_PRESETS, StemParams
from .render import apply_display, nice_scale_bar_length
from .series import parse_list, run_series
from .sim_core import SimPlan, StemSimResult, load_structure, parse_cif_meta, plan, run_simulation
from .worker import Worker

_TOOL_ROOT = Path(__file__).resolve().parent.parent
PRESETS_DIR = _TOOL_ROOT / "presets"
CIF_DIR = _TOOL_ROOT / "cif"
DOC_DIR = _TOOL_ROOT / "docs"


def make_root():
    try:
        import ttkbootstrap as ttkb

        root = ttkb.Window(themename="cosmo")
    except ImportError:
        root = tk.Tk()
    root.title("STEM-HAADF 模拟工具 · stem_sim 引擎（冻结声子多层法）")
    root.geometry("1560x900")
    return root


def tk_var(parent, value):
    if isinstance(value, bool):
        return tk.BooleanVar(master=parent, value=value)
    if isinstance(value, int):
        return tk.IntVar(master=parent, value=value)
    if isinstance(value, float):
        return tk.DoubleVar(master=parent, value=value)
    return tk.StringVar(master=parent, value=value)


class LabeledEntry(ttk.Frame):
    """label + entry(+unit) 的小工厂。"""

    def __init__(self, parent, label, default="", unit="", width=9, label_width=17):
        super().__init__(parent)
        self.var = tk_var(parent, default)
        ttk.Label(self, text=label, width=label_width, anchor="e").pack(side="left")
        ttk.Entry(self, textvariable=self.var, width=width).pack(side="left", padx=(4, 2))
        if unit:
            ttk.Label(self, text=unit, width=6, anchor="w").pack(side="left")

    def get(self) -> str:
        return self.var.get()


def tk_text(parent, height=4):
    text = tk.Text(parent, height=height, wrap="word", relief="flat",
                   background="#f2f2f2", font=("Microsoft YaHei", 9))
    text.config(state="disabled")
    return text


class STEMApp:
    def __init__(self, root):
        self.tk, self.ttk = tk, ttk
        self.root = root
        self.worker = Worker()
        self.series_worker = Worker()
        self.result: Optional[StemSimResult] = None
        self.plan_obj: Optional[SimPlan] = None
        self.structure = None
        self.cif_path = ""
        self._closing = False
        self._after_id = None
        self._plan_job = None

        self._build_toolbar()
        self._build_main()
        self._build_statusbar()
        self._refresh_presets()
        self._update_readout()
        self._after_id = self.root.after(80, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ==================================================================
    # 界面构建
    # ==================================================================
    def _build_toolbar(self):
        tk, ttk = self.tk, self.ttk
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(side="top", fill="x")
        ttk.Label(bar, text="结构文件:").pack(side="left")
        self.var_cif = tk.StringVar(value="(未加载)")
        ttk.Label(bar, textvariable=self.var_cif, width=40, anchor="w").pack(side="left", padx=4)
        ttk.Button(bar, text="加载 CIF/PDB/XYZ…", command=self.load_cif).pack(side="left", padx=4)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)

        ttk.Label(bar, text="质量档:").pack(side="left")
        self.var_quality = tk.StringVar(value="标准")
        qbox = ttk.Combobox(bar, textvariable=self.var_quality, width=7, state="readonly",
                            values=list(QUALITY_PRESETS.keys()))
        qbox.pack(side="left", padx=3)
        qbox.bind("<<ComboboxSelected>>", lambda e: self.apply_quality())

        ttk.Label(bar, text="预设:").pack(side="left", padx=(10, 0))
        self.var_preset = tk.StringVar()
        self.preset_box = ttk.Combobox(bar, textvariable=self.var_preset, width=30,
                                       state="readonly")
        self.preset_box.pack(side="left", padx=4)
        ttk.Button(bar, text="载入", command=self.apply_preset).pack(side="left", padx=2)
        ttk.Button(bar, text="保存为预设", command=self.save_preset).pack(side="left", padx=2)

        self.btn_sim = ttk.Button(bar, text="▶ 模 拟", command=self.start_simulation)
        self.btn_sim.pack(side="left", padx=14)
        self.btn_cancel = ttk.Button(bar, text="取消", command=self.cancel_work, state="disabled")
        self.btn_cancel.pack(side="left")
        ttk.Button(bar, text="批量系列…", command=self.open_series_dialog).pack(side="left", padx=12)
        ttk.Button(bar, text="参数约定说明", command=self.show_conventions).pack(side="right")

    def _build_main(self):
        tk, ttk = self.tk, self.ttk
        main = ttk.Frame(self.root, padding=(8, 2))
        main.pack(side="top", fill="both", expand=True)
        main.columnconfigure(2, weight=1)
        main.rowconfigure(0, weight=1)

        # ---------------- 左栏：结构与取向 ----------------
        left = ttk.Labelframe(main, text=" 结构与取向 ", padding=8)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 6))
        for attr, text in (("lbl_formula", "化学式: —"), ("lbl_cell", "晶格: —"),
                           ("lbl_sg", "空间群: —"), ("lbl_natoms", "单胞原子数: —")):
            lbl = ttk.Label(left, text=text, anchor="w", wraplength=250, justify="left")
            lbl.pack(anchor="w")
            setattr(self, attr, lbl)
        self.lbl_natoms.pack_configure(pady=(0, 8))

        zonef = ttk.Labelframe(left, text="带轴 [h k l]（电子束方向）", padding=6)
        zonef.pack(fill="x", pady=2)
        zrow = ttk.Frame(zonef)
        zrow.pack()
        self.var_h, self.var_k, self.var_l = tk.IntVar(value=1), tk.IntVar(value=0), tk.IntVar(value=0)
        for var in (self.var_h, self.var_k, self.var_l):
            ttk.Spinbox(zrow, from_=-9, to=9, textvariable=var, width=4,
                        command=self.schedule_plan).pack(side="left", padx=3)
        quick = ttk.Frame(zonef)
        quick.pack(pady=(6, 0))
        for label, z in (("[100]", (1, 0, 0)), ("[110]", (1, 1, 0)), ("[111]", (1, 1, 1)),
                         ("[210]", (2, 1, 0)), ("[211]", (2, 1, 1)), ("[0001]", (0, 0, 1))):
            ttk.Button(quick, text=label, width=6,
                       command=lambda z=z: self.set_zone(z)).pack(side="left", padx=1)

        self.ent_thickness = LabeledEntry(left, "样品厚度", "10.0", "nm")
        self.ent_thickness.pack(anchor="w", pady=(8, 1))
        for e in (self.ent_thickness,):
            e.var.trace_add("write", lambda *_: self.schedule_plan())

        scanf = ttk.Labelframe(left, text="扫描", padding=6)
        scanf.pack(fill="x", pady=4)
        self.ent_fov = LabeledEntry(scanf, "扫描视场", "30", "nm", label_width=10)
        self.ent_fov.pack(anchor="w", pady=1)
        self.ent_npts = LabeledEntry(scanf, "扫描点数", "48", "²", label_width=10)
        self.ent_npts.pack(anchor="w", pady=1)
        self.lbl_step = ttk.Label(scanf, text="步长: —", anchor="w")
        self.lbl_step.pack(anchor="w", pady=(2, 0))

        rotf = ttk.Frame(left)
        rotf.pack(fill="x", pady=(4, 0))
        self.ent_rotation = LabeledEntry(rotf, "面内旋转", "0", "°")
        self.ent_rotation.pack(anchor="w", pady=1)
        self.var_mirror = tk.BooleanVar(value=False)
        ttk.Checkbutton(rotf, text="水平镜像", variable=self.var_mirror,
                        command=self._rerender_display).pack(anchor="w")

        planf = ttk.Labelframe(left, text="模拟计划（实时预估）", padding=6)
        planf.pack(fill="x", pady=(6, 0))
        self.plan_text = tk_text(planf, height=13)
        self.plan_text.pack(fill="x")

        # ---------------- 中栏：电镜与探针 ----------------
        mid = ttk.Labelframe(main, text=" 电镜 / 探针 / 探测器 ", padding=8)
        mid.grid(row=0, column=1, sticky="nsw", padx=6)

        self.ent_kv = LabeledEntry(mid, "加速电压", "300", "kV")
        self.ent_kv.pack(anchor="w", pady=1)
        self.ent_kv.var.trace_add("write", lambda *_: self._update_readout())
        self.ent_cs = LabeledEntry(mid, "球差 Cs (可为负)", "0.01", "mm")
        self.ent_cs.pack(anchor="w", pady=1)
        self.ent_cs.var.trace_add("write", lambda *_: self._update_readout())

        prow = ttk.Frame(mid)
        prow.pack(anchor="w", pady=1)
        ttk.Label(prow, text="探针会聚半角", width=17, anchor="e").pack(side="left")
        self.ent_probe = LabeledEntry(prow, "", "25", "mrad", width=9, label_width=0)
        self.ent_probe.pack(side="left")
        self.btn_opt_probe = ttk.Button(prow, text="最优", width=5,
                                        command=self._fill_optimal_probe)
        self.btn_opt_probe.pack(side="left", padx=4)
        self.ent_probe.var.trace_add("write", lambda *_: self._update_readout())

        dfrow = ttk.Frame(mid)
        dfrow.pack(anchor="w", pady=1)
        ttk.Label(dfrow, text="探针离焦", width=17, anchor="e").pack(side="left")
        self.ent_df = LabeledEntry(dfrow, "", "-4.44", "nm", width=9, label_width=0)
        self.ent_df.pack(side="left")
        self.btn_scherzer = ttk.Button(dfrow, text="Scherzer", width=9,
                                       command=self._fill_scherzer)
        self.btn_scherzer.pack(side="left", padx=4)
        self.ent_df.var.trace_add("write", lambda *_: self._update_readout())

        tilt = ttk.Frame(mid)
        tilt.pack(anchor="w", pady=1)
        self.ent_tiltx = LabeledEntry(tilt, "束倾斜 x / y", "0", "mrad", label_width=10, width=7)
        self.ent_tiltx.pack(side="left")
        self.ent_tilty = LabeledEntry(tilt, "/", "0", "mrad", width=7, label_width=1)
        self.ent_tilty.pack(side="left", padx=(4, 0))
        self.ent_soft = LabeledEntry(mid, "光阑软化", "0", "mrad")
        self.ent_soft.pack(anchor="w", pady=1)

        detf = ttk.Labelframe(mid, text="环形探测器（HAADF/ADF）", padding=6)
        detf.pack(fill="x", pady=(8, 4))
        self.ent_din = LabeledEntry(detf, "内角", "50", "mrad", label_width=10)
        self.ent_din.pack(anchor="w", pady=1)
        self.ent_din.var.trace_add("write", lambda *_: self.schedule_plan())
        self.ent_dout = LabeledEntry(detf, "外角", "100", "mrad", label_width=10)
        self.ent_dout.pack(anchor="w", pady=1)
        self.ent_dout.var.trace_add("write", lambda *_: self.schedule_plan())
        self.ent_dsoft = LabeledEntry(detf, "软化", "0", "mrad", label_width=10)
        self.ent_dsoft.pack(anchor="w", pady=1)
        self.var_bf = tk.BooleanVar(value=True)
        ttk.Checkbutton(detf, text="同时输出 BF / ABF（几乎不增加耗时）",
                        variable=self.var_bf, command=self.schedule_plan).pack(anchor="w", pady=(4, 0))

        phf = ttk.Labelframe(mid, text="冻结声子（热漫散射）", padding=6)
        phf.pack(fill="x", pady=4)
        self.ent_nph = LabeledEntry(phf, "组态数", "8", "", label_width=10)
        self.ent_nph.pack(anchor="w", pady=1)
        self.ent_nph.var.trace_add("write", lambda *_: self.schedule_plan())
        self.ent_temp = LabeledEntry(phf, "样品温度", "300", "K", label_width=10)
        self.ent_temp.pack(anchor="w", pady=1)
        self.ent_temp.var.trace_add("write", lambda *_: self.schedule_plan())
        self.ent_theta = LabeledEntry(phf, "Debye 温度(0=表)", "0", "K", label_width=10)
        self.ent_theta.pack(anchor="w", pady=1)
        self.ent_sigma = LabeledEntry(phf, "位移 σ (0=模型)", "0", "Å", label_width=10)
        self.ent_sigma.pack(anchor="w", pady=1)
        self.ent_sigma.var.trace_add("write", lambda *_: self.schedule_plan())

        msf = ttk.Labelframe(mid, text="多层法网格 / 性能", padding=6)
        msf.pack(fill="x", pady=4)
        self.ent_sampling = LabeledEntry(msf, "采样 (0=自动)", "0", "Å/px", label_width=12)
        self.ent_sampling.pack(anchor="w", pady=1)
        self.ent_sampling.var.trace_add("write", lambda *_: self.schedule_plan())
        self.ent_slice = LabeledEntry(msf, "切片厚度", "2.0", "Å", label_width=12)
        self.ent_slice.pack(anchor="w", pady=1)
        self.ent_slice.var.trace_add("write", lambda *_: self.schedule_plan())
        self.ent_threads = LabeledEntry(msf, "FFT 线程(0=自动)", "0", "", label_width=12)
        self.ent_threads.pack(anchor="w", pady=1)
        prow2 = ttk.Frame(msf)
        prow2.pack(anchor="w", pady=1)
        ttk.Label(prow2, text="并行", width=12, anchor="e").pack(side="left")
        self.var_parallel = tk.StringVar(value="auto")
        pbox = ttk.Combobox(prow2, textvariable=self.var_parallel, width=8, state="readonly",
                            values=["auto", "off"])
        pbox.pack(side="left", padx=4)
        pbox.bind("<<ComboboxSelected>>", lambda e: self.schedule_plan())

        outf = ttk.Labelframe(mid, text="输出与显示", padding=6)
        outf.pack(fill="x", pady=4)
        self.ent_outsam = LabeledEntry(outf, "输出采样(0=原生)", "0", "Å/px", label_width=14)
        self.ent_outsam.pack(anchor="w", pady=1)
        prow3 = ttk.Frame(outf)
        prow3.pack(anchor="w", pady=1)
        ttk.Label(prow3, text="极性", width=14, anchor="e").pack(side="left")
        self.var_polarity = tk.StringVar(value="正常")
        pol = ttk.Combobox(prow3, textvariable=self.var_polarity, width=7, state="readonly",
                           values=["正常", "反转"])
        pol.pack(side="left", padx=4)
        pol.bind("<<ComboboxSelected>>", lambda e: self._rerender_display())
        self.ent_blur = LabeledEntry(outf, "显示模糊", "0", "Å", label_width=14)
        self.ent_blur.pack(anchor="w", pady=1)
        crow = ttk.Frame(outf)
        crow.pack(anchor="w", pady=1)
        ttk.Label(crow, text="对比度 %", width=14, anchor="e").pack(side="left")
        self.var_lo, self.var_hi = tk.DoubleVar(value=0.5), tk.DoubleVar(value=99.5)
        for var in (self.var_lo, self.var_hi):
            sb = ttk.Spinbox(crow, from_=0, to=100, textvariable=var, width=6)
            sb.pack(side="left", padx=2)
            sb.bind("<Return>", lambda e: self._rerender_display())
            sb.bind("<FocusOut>", lambda e: self._rerender_display())

        self.lbl_readout = tk_text(mid, height=6)
        self.lbl_readout.pack(fill="x", pady=(8, 0))

        # ---------------- 右栏：结果 ----------------
        right = ttk.Labelframe(main, text=" 模拟结果 ", padding=6)
        right.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        mode_row = ttk.Frame(right)
        mode_row.pack(fill="x")
        ttk.Label(mode_row, text="显示:").pack(side="left")
        self.var_mode = tk.StringVar(value="ADF")
        self.mode_box = ttk.Combobox(mode_row, textvariable=self.var_mode, width=12,
                                     state="readonly", values=["ADF", "BF", "ABF", "探针", "3D 强度曲面"])
        self.mode_box.pack(side="left", padx=6)
        self.mode_box.bind("<<ComboboxSelected>>", lambda e: self._rerender_display())
        ttk.Label(mode_row, text="（ADF 内/外角、BF/ABF 由探针角自动定标）",
                  foreground="#666").pack(side="left", padx=6)

        from .render import setup_matplotlib_cjk

        setup_matplotlib_cjk()
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure

        self.fig = Figure(figsize=(8.0, 7.0), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, right)
        self.toolbar.update()
        self.toolbar.pack(side="top", fill="x")

        exbar = ttk.Frame(right, padding=(0, 6))
        exbar.pack(side="bottom", fill="x")
        ttk.Label(exbar, text="文件名:").pack(side="left")
        self.var_basename = tk.StringVar(value="")
        ttk.Entry(exbar, textvariable=self.var_basename, width=38).pack(side="left", padx=4)
        self.var_export_all_det = tk.BooleanVar(value=True)
        ttk.Checkbutton(exbar, text="导出全部探测器", variable=self.var_export_all_det).pack(
            side="left", padx=6)
        ttk.Button(exbar, text="导出到文件夹…", command=self.export_to_folder).pack(side="left", padx=6)

    def _build_statusbar(self):
        tk, ttk = self.tk, self.ttk
        bar = ttk.Frame(self.root, padding=(8, 4))
        bar.pack(side="bottom", fill="x")
        self.progress = ttk.Progressbar(bar, length=420, mode="determinate", maximum=100)
        self.progress.pack(side="left")
        self.var_status = tk.StringVar(value="就绪 — 加载结构文件后即可模拟（左栏可先看耗时预估）")
        ttk.Label(bar, textvariable=self.var_status, anchor="w").pack(side="left", padx=10)

    # ==================================================================
    # 工具
    # ==================================================================
    def status(self, text: str):
        self.var_status.set(text)

    def set_busy(self, busy: bool):
        self.btn_sim.config(state="disabled" if busy else "normal")
        self.btn_cancel.config(state="normal" if busy else "disabled")

    def _f(self, entry, name, minimum=None, allow_zero=True):
        try:
            value = float(entry.get())
        except (ValueError, tk.TclError):
            raise ValueError(f"{name} 不是有效数字")
        if not np.isfinite(value):
            raise ValueError(f"{name} 必须为有限数值")
        if minimum is not None and (value < minimum or (value == 0 and not allow_zero)):
            raise ValueError(f"{name} 需 ≥ {minimum}")
        return value

    # ==================================================================
    # 结构加载
    # ==================================================================
    def load_cif(self):
        path = filedialog.askopenfilename(
            title="选择结构文件",
            initialdir=str(CIF_DIR if CIF_DIR.exists() else "."),
            filetypes=[("结构文件", "*.cif *.pdb *.xyz *.vasp"), ("CIF", "*.cif"),
                       ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            self.structure = load_structure(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("加载失败", f"{path}\n\n{exc}")
            return
        self.cif_path = path
        self.var_cif.set(Path(path).name)
        meta = parse_cif_meta(path)
        self.lbl_formula.config(text=f"化学式: {self.structure.chemical_formula()}")
        try:
            cell = self.structure.cell_lengths()
            self.lbl_cell.config(
                text=f"晶格: {meta['cell'] or ' / '.join(f'{v:.4f}' for v in cell)} Å")
        except Exception:  # noqa: BLE001
            self.lbl_cell.config(text="晶格: —")
        self.lbl_sg.config(text=f"空间群: {meta['spacegroup'] or '(未标注)'}")
        self.lbl_natoms.config(text=f"单胞原子数: {len(self.structure)}")
        self.schedule_plan()

    def set_zone(self, zone):
        self.var_h.set(zone[0]); self.var_k.set(zone[1]); self.var_l.set(zone[2])
        self.schedule_plan()

    def zone_tuple(self):
        try:
            return (int(self.var_h.get()), int(self.var_k.get()), int(self.var_l.get()))
        except (tk.TclError, ValueError):
            raise ValueError("带轴指数输入未完成（需为整数）")

    # ==================================================================
    # 参数收集 / 预设
    # ==================================================================
    def collect_params(self) -> StemParams:
        if self.structure is None or not self.cif_path:
            raise ValueError("请先加载结构文件（CIF/PDB/XYZ）")
        zone = self.zone_tuple()
        if all(v == 0 for v in zone):
            raise ValueError("带轴不能为 [0 0 0]")
        det = self.var_mode.get()
        if det in ("探针", "3D 强度曲面"):
            det = "ADF"
        return StemParams(
            cif_path=self.cif_path,
            zone=zone,
            thickness_nm=self._f(self.ent_thickness, "厚度", minimum=0.1),
            scan_fov_nm=self._f(self.ent_fov, "扫描视场", minimum=0.5),
            scan_points=int(float(self.ent_npts.get())),
            rotation_deg=self._f(self.ent_rotation, "面内旋转"),
            mirror=bool(self.var_mirror.get()),
            voltage_kv=self._f(self.ent_kv, "电压", minimum=1.0),
            cs_mm=self._f(self.ent_cs, "Cs"),
            defocus_nm=self._f(self.ent_df, "探针离焦"),
            probe_semiangle_mrad=self._f(self.ent_probe, "探针半角", minimum=0.1),
            probe_soft_mrad=self._f(self.ent_soft, "光阑软化", minimum=0.0),
            tilt_x_mrad=self._f(self.ent_tiltx, "束倾斜 x"),
            tilt_y_mrad=self._f(self.ent_tilty, "束倾斜 y"),
            detector_inner_mrad=self._f(self.ent_din, "探测器内角", minimum=0.0),
            detector_outer_mrad=self._f(self.ent_dout, "探测器外角", minimum=0.1),
            detector_soft_mrad=self._f(self.ent_dsoft, "探测器软化", minimum=0.0),
            with_bright_field=bool(self.var_bf.get()),
            display_detector=det,
            n_phonons=int(float(self.ent_nph.get())),
            temperature_k=self._f(self.ent_temp, "样品温度", minimum=1.0),
            debye_temperature_k=self._f(self.ent_theta, "Debye 温度", minimum=0.0),
            sigma_override_a=self._f(self.ent_sigma, "位移 σ", minimum=0.0),
            sampling_a=self._f(self.ent_sampling, "采样", minimum=0.0),
            slice_thickness_a=self._f(self.ent_slice, "切片厚度", minimum=0.2),
            parallel=self.var_parallel.get(),
            threads=int(float(self.ent_threads.get())),
            output_sampling_a=self._f(self.ent_outsam, "输出采样", minimum=0.0),
            polarity=1 if self.var_polarity.get() == "正常" else -1,
            display_blur_a=self._f(self.ent_blur, "显示模糊", minimum=0.0),
            contrast_lo_pct=float(self.var_lo.get()),
            contrast_hi_pct=float(self.var_hi.get()),
        )

    def apply_params(self, p: StemParams):
        self.cif_path = p.cif_path
        if p.cif_path and Path(p.cif_path).exists():
            try:
                self.structure = load_structure(p.cif_path)
                self.var_cif.set(Path(p.cif_path).name)
                meta = parse_cif_meta(p.cif_path)
                self.lbl_formula.config(text=f"化学式: {self.structure.chemical_formula()}")
                cell = self.structure.cell_lengths()
                self.lbl_cell.config(
                    text=f"晶格: {meta['cell'] or ' / '.join(f'{v:.4f}' for v in cell)} Å")
                self.lbl_sg.config(text=f"空间群: {meta['spacegroup'] or '(未标注)'}")
                self.lbl_natoms.config(text=f"单胞原子数: {len(self.structure)}")
            except Exception:  # noqa: BLE001
                pass
        elif p.cif_path:
            self.var_cif.set(f"{Path(p.cif_path).name} (缺失)")
        self.var_h.set(p.zone[0]); self.var_k.set(p.zone[1]); self.var_l.set(p.zone[2])
        self.ent_thickness.var.set(p.thickness_nm)
        self.ent_fov.var.set(p.scan_fov_nm)
        self.ent_npts.var.set(p.scan_points)
        self.ent_rotation.var.set(p.rotation_deg)
        self.var_mirror.set(p.mirror)
        self.ent_kv.var.set(p.voltage_kv)
        self.ent_cs.var.set(p.cs_mm)
        self.ent_df.var.set(p.defocus_nm)
        self.ent_probe.var.set(p.probe_semiangle_mrad)
        self.ent_soft.var.set(p.probe_soft_mrad)
        self.ent_tiltx.var.set(p.tilt_x_mrad)
        self.ent_tilty.var.set(p.tilt_y_mrad)
        self.ent_din.var.set(p.detector_inner_mrad)
        self.ent_dout.var.set(p.detector_outer_mrad)
        self.ent_dsoft.var.set(p.detector_soft_mrad)
        self.var_bf.set(p.with_bright_field)
        self.ent_nph.var.set(p.n_phonons)
        self.ent_temp.var.set(p.temperature_k)
        self.ent_theta.var.set(p.debye_temperature_k)
        self.ent_sigma.var.set(p.sigma_override_a)
        self.ent_sampling.var.set(p.sampling_a)
        self.ent_slice.var.set(p.slice_thickness_a)
        self.var_parallel.set(p.parallel)
        self.ent_threads.var.set(p.threads)
        self.ent_outsam.var.set(p.output_sampling_a)
        self.var_polarity.set("正常" if p.polarity >= 0 else "反转")
        self.ent_blur.var.set(p.display_blur_a)
        self.var_lo.set(p.contrast_lo_pct)
        self.var_hi.set(p.contrast_hi_pct)
        if p.display_detector in ("ADF", "BF", "ABF"):
            self.var_mode.set(p.display_detector)
        self._update_readout()
        self.schedule_plan()

    def apply_quality(self):
        name = self.var_quality.get()
        vals = QUALITY_PRESETS.get(name)
        if not vals:
            return
        self.ent_npts.var.set(vals["scan_points"])
        self.ent_nph.var.set(vals["n_phonons"])
        self.ent_slice.var.set(vals["slice_thickness_a"])
        self.ent_sampling.var.set(vals["sampling_a"])
        self.var_parallel.set(vals["parallel"])
        self.status(f"已套用质量档「{name}」：{vals['n_phonons']} 声子组态 / "
                    f"{vals['scan_points']}² 扫描点 / 切片 {vals['slice_thickness_a']:g} Å")
        self.schedule_plan()

    def _refresh_presets(self):
        names = []
        if PRESETS_DIR.exists():
            import json

            for f in sorted(PRESETS_DIR.glob("*.json")):
                try:
                    names.append(json.loads(f.read_text(encoding="utf-8")).get("name", f.stem))
                except Exception:  # noqa: BLE001
                    names.append(f.stem)
        self.preset_box.config(values=names)
        if names and not self.var_preset.get():
            self.var_preset.set(names[0])

    def apply_preset(self):
        name = self.var_preset.get()
        if not name:
            return
        import json

        for f in PRESETS_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if data.get("name", f.stem) != name:
                continue
            data = dict(data)
            cif = data.get("cif_path", "")
            if cif and not Path(cif).exists():
                local = _TOOL_ROOT / cif
                if local.exists():
                    data["cif_path"] = str(local)
                elif (CIF_DIR / Path(cif).name).exists():
                    data["cif_path"] = str(CIF_DIR / Path(cif).name)
            try:
                self.apply_params(StemParams.from_dict(data))
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("预设", f"预设「{name}」无法应用：\n{exc}")
                return
            self.status(f"已载入预设：{name}")
            return
        messagebox.showwarning("预设", f"找不到预设：{name}")

    def save_preset(self):
        try:
            p = self.collect_params()
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        name = simpledialog.askstring("保存预设", "预设名称：", parent=self.root)
        if not name:
            return
        import json

        from .export import sanitize

        PRESETS_DIR.mkdir(exist_ok=True)
        path = PRESETS_DIR / f"{sanitize(name)}.json"
        if path.exists() and not messagebox.askyesno(
                "保存预设", f"预设「{name}」已存在，覆盖？", parent=self.root):
            return
        data = p.to_dict()
        data["name"] = name
        try:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("保存预设", f"写入失败：{exc}", parent=self.root)
            return
        self._refresh_presets()
        self.var_preset.set(name)
        self.status(f"预设已保存：{path.name}")

    # ==================================================================
    # 只读显示 / 计划预估
    # ==================================================================
    def _update_readout(self):
        try:
            p = StemParams(
                voltage_kv=float(self.ent_kv.get() or 300),
                cs_mm=float(self.ent_cs.get() or 0.01),
                defocus_nm=float(self.ent_df.get() or 0),
                probe_semiangle_mrad=float(self.ent_probe.get() or 25),
            )
            sc = p.scope()
            lines = [
                f"λ = {sc.wavelength:.5f} Å   探针最优离焦 = {sc.scherzer_probe_defocus()/10:+.2f} nm",
                f"最优探针半角 = {sc.optimal_probe_semiangle():.1f} mrad   "
                f"探针尺寸(Rayleigh) = {sc.probe_size_rayleigh():.3f} Å",
                f"外角 {p.detector_outer_mrad:g} mrad 所需采样 ≤ {sc.required_sampling():.4f} Å/px",
                f"（TEM 相位衬度 Scherzer 为 {sc.scherzer_defocus()/10:+.1f} nm，"
                "与探针最优值不同）",
            ]
            self._set_text(self.lbl_readout, "\n".join(lines))
        except (ValueError, tk.TclError):
            pass

    def _fill_scherzer(self):
        try:
            sc = StemParams(voltage_kv=float(self.ent_kv.get()),
                            cs_mm=float(self.ent_cs.get())).scope()
            self.ent_df.var.set(round(sc.scherzer_probe_defocus() / 10.0, 3))
        except ValueError:
            pass

    def _fill_optimal_probe(self):
        try:
            sc = StemParams(voltage_kv=float(self.ent_kv.get()),
                            cs_mm=float(self.ent_cs.get())).scope()
            self.ent_probe.var.set(round(sc.optimal_probe_semiangle(), 2))
        except ValueError:
            pass

    def schedule_plan(self):
        """参数变化后延迟重算计划（避免每次按键都读结构文件）。"""
        if self._closing:
            return
        if self._plan_job is not None:
            try:
                self.root.after_cancel(self._plan_job)
            except Exception:  # noqa: BLE001
                pass
        self._plan_job = self.root.after(250, self.update_plan)

    def update_plan(self):
        self._plan_job = None
        try:
            step = float(self.ent_fov.get()) * 10.0 / max(int(float(self.ent_npts.get())), 1)
            self.lbl_step.config(text=f"步长: {step:.3f} Å")
        except (ValueError, tk.TclError):
            pass
        if self.structure is None:
            self._set_text(self.plan_text, "请先加载结构文件。")
            return
        try:
            p = self.collect_params()
        except ValueError as exc:
            self._set_text(self.plan_text, f"参数未就绪：\n{exc}")
            return
        try:
            pl = plan(p)
        except Exception as exc:  # noqa: BLE001
            self._set_text(self.plan_text, f"该配置不可用：\n{exc}")
            return
        self.plan_obj = pl
        lines = pl.summary_lines()
        if pl.sigma:
            lines.append("位移 σ_u: " + ", ".join(
                f"{k}={v:.4f} Å" for k, v in pl.sigma.items()))
        lines.append("探测器: " + ", ".join(
            f"{n} {a:g}–{b:.1f} mrad" for n, a, b in pl.detector_angles))
        if pl.n_workers:
            lines.append(f"并行: {pl.n_workers} 进程")
        self._set_text(self.plan_text, "\n".join(lines + [""] + pl.warnings))

    def _set_text(self, widget, text: str):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.config(state="disabled")

    def show_conventions(self):
        messagebox.showinfo(
            "参数与物理约定",
            "【探测角与采样】\n"
            "探测器外角决定了所需采样：Δx ≤ λ/(2·sinθ_外)，因为 FFT 网格的\n"
            "奈奎斯特频率对应 sinθ = λ/(2Δx)。外角越大、电压越低，要求的采样\n"
            "越细，计算量按网格点数平方增长——这是 HAADF 模拟最主要的代价来源。\n"
            "超过奈奎斯特的强度是混叠结果，工具会裁剪并提示。\n\n"
            "【离焦符号】正 = 过焦，负 = 欠焦。\n"
            "探针最优离焦 Δf = −√(|Cs|λ)；TEM 相位衬度的 Scherzer 离焦是\n"
            "−√(1.5|Cs|λ)，两者不同，界面上的「Scherzer」按钮给的是探针口径。\n\n"
            "【球差 Cs】单位 mm，1 mm = 1e7 Å；负值 = 校正器过校正。\n\n"
            "【冻结声子】HAADF 的高角信号主要来自热漫散射（TDS），必须用声子\n"
            "组态做非相干平均才能得到可靠 Z 衬度。组态数 < 4 时图像会残留格子\n"
            "条纹，建议 ≥ 8（或仅作预览）。热位移取 Debye 模型：B = 8π²⟨u²⟩，\n"
            "σ_u = √(B/8π²)；也可直接指定 σ 或 Debye 温度。\n\n"
            "【强度单位】输出的 ADF/BF 是**入射束流被收集的比例**（0–1），\n"
            "可跨厚度/参数直接比较。定量对比实验绝对强度时注意：本引擎口径\n"
            "比「直接用 dσ/dΩ=|f_e|² 的运动学估计」高约 2.2 倍（σ=2πγmeλ/h²\n"
            "与散射因子表口径的组合约定，见 README）。\n\n"
            "【取向】面内旋转为绕光轴逆时针；镜像为水平翻转，用于与实验图对齐。\n\n"
            "【未实现】部分相干（源尺寸/能量展宽）、分段探测器的对向差分 ABF、\n"
            "DPC 一阶矩、声子色散关联（Kikuchi 带）。见 README「已知边界」。",
        )

    # ==================================================================
    # 模拟执行
    # ==================================================================
    def start_simulation(self):
        if self.worker.running or self.series_worker.running:
            return
        try:
            params = self.collect_params()
            pl = plan(params)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("参数错误", str(exc))
            return
        if pl.warnings:
            if not messagebox.askyesno(
                "参数提醒",
                "以下问题会影响结果可靠性或耗时：\n\n" +
                "\n".join("• " + w for w in pl.warnings) + "\n\n仍要继续？",
            ):
                return
        if pl.estimate_s > 600 and not messagebox.askyesno(
                "耗时提醒",
                f"预估耗时约 {pl.estimate_s/60:.1f} 分钟（{pl.n_fft/1e6:.1f} M 次 FFT，"
                f"{pl.n_atoms} 原子）。\n\n继续？\n"
                "（可换用「预览」质量档、减小扫描点数或视场）"):
            return
        self.set_busy(True)
        self.status("模拟中…")
        self.progress.config(value=0)
        self.worker.start(lambda progress, stop: run_simulation(params, progress, stop, pl))

    def cancel_work(self):
        if self.worker.stop() or self.series_worker.stop():
            self.status("正在取消…")

    def _poll(self):
        if self._closing:
            return
        try:
            for worker in (self.worker, self.series_worker):
                try:
                    while True:
                        self._handle_event(worker.events.get_nowait())
                except queue.Empty:
                    pass
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            self.status("界面事件处理出错（详见控制台）")
        self._after_id = self.root.after(80, self._poll)

    def _handle_event(self, event):
        kind = event[0]
        if kind == "progress":
            _, frac, msg, tag = event
            self.progress.config(value=frac * 100)
            self.status(msg if tag != "series" else f"[系列] {msg}")
        elif kind == "result":
            obj, tag = event[1], event[2]
            if tag == "series":
                self._on_series_done(obj)
            else:
                self._on_sim_done(obj)
        elif kind == "cancelled":
            self.status("已取消")
            self.set_busy(False)
        elif kind == "error":
            self.set_busy(False)
            messagebox.showerror("出错", event[1][-1800:])
            self.status("出错，详见弹窗")
        elif kind == "done":
            if not self.worker.running and not self.series_worker.running:
                self.set_busy(False)

    def _on_sim_done(self, result: StemSimResult):
        self.result = result
        res = result.result
        names = result.detector_names()
        self.mode_box.config(values=names + ["探针", "3D 强度曲面"])
        if self.var_mode.get() not in names:
            self.var_mode.set(names[0] if names else "ADF")
        if not self.var_basename.get():
            self.var_basename.set(default_basename(result.params, result))
        lines = [
            f"超胞 {res.n_atoms} 原子 / {res.n_slices} 切片 × {res.dz:.3f} Å",
            f"实际厚度 {res.thickness_a/10:.3f} nm",
            f"采样 {res.sampling[0]:.4f} Å/px  输出 {result.out_sampling:.4f} Å/px",
            f"FFT {res.n_fft/1e6:.2f} M  用时 {res.elapsed_s:.1f} s"
            f"（{res.n_configs} 声子组态，实测 {res.fft_rate:.0f} FFT/s）",
            f"位移: " + ", ".join(f"{k} σ={v:.4f} Å" for k, v in res.sigma.items()),
        ]
        lines += res.notes
        self._set_text(self.plan_text, "\n".join(lines))
        self.progress.config(value=100)
        self.status(f"完成（{result.elapsed_s:.1f} s）— {result.formula} "
                    f"{result.params.zone_str} t={res.thickness_a/10:.2f} nm")
        self._rerender_display()

    # ==================================================================
    # 显示渲染
    # ==================================================================
    def _rerender_display(self):
        if self.result is None:
            return
        try:
            blur = float(self.ent_blur.get())
        except ValueError:
            blur = 0.0
        try:
            lo, hi = float(self.var_lo.get()), float(self.var_hi.get())
        except (ValueError, tk.TclError):
            lo, hi = 0.5, 99.5
        mode = self.var_mode.get()
        ax = self.ax
        ax.clear()
        result = self.result
        p = result.params
        if mode == "探针":
            from stem_sim import probe_intensity_profile

            sc = p.scope()
            r, prof = probe_intensity_profile(sc, (256, 256), p.effective_sampling())
            ax.plot(r, prof, lw=1.8)
            ax.set_xlim(0, min(8.0, r.max()))
            ax.set_xlabel("r (Å)")
            ax.set_ylabel("归一化强度")
            ax.set_ylim(0, 1.08)
            ax.set_title(
                f"探针强度剖面  α={p.probe_semiangle_mrad:g} mrad  "
                f"df={p.defocus_nm:+g} nm  Cs={p.cs_mm:g} mm  "
                f"(Rayleigh {sc.probe_size_rayleigh():.2f} Å)", fontsize=10)
        elif mode == "3D 强度曲面":
            det = names = result.detector_names()
            data = result.image("ADF")
            h, w = data.shape
            fov_x, fov_y = w * result.out_sampling, h * result.out_sampling
            disp = apply_display(data, result.out_sampling, polarity=p.polarity,
                                 blur_a=blur, lo_pct=lo, hi_pct=hi)
            yy, xx = np.mgrid[0:h, 0:w]
            ax.remove()
            self.ax = self.fig.add_subplot(111, projection="3d")
            ax = self.ax
            step = max(1, h // 140)
            ax.plot_surface(xx[::step] * result.out_sampling,
                            yy[::step] * result.out_sampling,
                            disp[::step, ::step], cmap="viridis", linewidth=0,
                            antialiased=True, rcount=200, ccount=200)
            ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("强度")
            ax.set_title(f"{result.formula} {p.zone_str} ADF 立体衬度", fontsize=10)
            ax.view_init(elev=42, azim=-58)
        else:
            if self.ax.name == "3d":
                self.ax.remove()
                self.ax = self.fig.add_subplot(111)
                ax = self.ax
            data = result.image(mode)
            disp = apply_display(data, result.out_sampling, polarity=p.polarity,
                                 blur_a=blur, lo_pct=lo, hi_pct=hi)
            h, w = disp.shape
            fov_x, fov_y = w * result.out_sampling, h * result.out_sampling
            ax.imshow(disp, cmap="gray", vmin=0, vmax=1, extent=(0, fov_x, fov_y, 0))
            self._draw_scale_bar(ax, fov_x, fov_y)
            spec = next((d for d in result.result.detector_specs
                         if d.name.upper() == mode.upper()), None)
            ang = f"{spec.inner:g}–{spec.outer:g} mrad" if spec else ""
            info = (f"{result.formula} {p.zone_str}  {mode}  {ang}\n"
                    f"t = {result.result.thickness_a/10:.2f} nm\n"
                    f"探针 {p.probe_semiangle_mrad:g} mrad  df = {p.defocus_nm:+g} nm\n"
                    f"{p.voltage_kv:g} kV  Cs = {p.cs_mm:g} mm\n"
                    f"{result.out_sampling:.4f} Å/px  {p.n_phonons} 声子组态")
            ax.text(0.015, 0.03, info, transform=ax.transAxes, va="bottom", ha="left",
                    color="white", fontsize=8.5,
                    bbox=dict(facecolor="black", alpha=0.55, pad=3, edgecolor="none"))
            ax.set_title(f"{result.formula} {p.zone_str}  STEM-{mode} 模拟", fontsize=11)
        ax.tick_params(labelsize=8)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _draw_scale_bar(self, ax, fov_x: float, fov_y: float):
        bar = nice_scale_bar_length(fov_x)
        x1 = fov_x - fov_x * 0.05 - bar
        y = fov_y - fov_y * 0.05
        ax.plot([x1, x1 + bar], [y, y], color="black", lw=5, solid_capstyle="butt")
        ax.plot([x1, x1 + bar], [y, y], color="white", lw=2.4, solid_capstyle="butt")
        label = f"{bar / 10:g} nm" if bar >= 10 else f"{bar:g} Å"
        ax.text(x1 + bar / 2, y - fov_y * 0.035, label, color="white", ha="center",
                fontsize=9, bbox=dict(facecolor="black", alpha=0.55, pad=2, edgecolor="none"))

    # ==================================================================
    # 导出
    # ==================================================================
    def export_to_folder(self):
        if self.result is None:
            messagebox.showinfo("导出", "请先完成一次模拟")
            return
        out_dir = filedialog.askdirectory(title="选择导出文件夹")
        if not out_dir:
            return
        name = self.var_basename.get().strip() or None
        dets = (self.result.detector_names() if self.var_export_all_det.get()
                else [self.var_mode.get()])
        dets = [d for d in dets if d in self.result.detector_names()] or \
            self.result.detector_names()
        kinds = {"tiff": True, "png": True, "npy": True, "json": True}
        try:
            paths = export_all(self.result, out_dir, basename=name, kind_map=kinds,
                               detectors=dets)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("导出失败", "".join(traceback.format_exception(exc))[-1500:])
            return
        self.status(f"已导出 {len(paths)} 个文件（{', '.join(dets)}）")
        if messagebox.askyesno(
                "导出完成",
                f"已导出 {len(paths)} 个文件（探测器：{', '.join(dets)}）。\n\n"
                "其中 .npy 是**物理强度**（收集比例 0–1，未归一化），"
                "可直接用于定量分析。\n\n打开输出文件夹？"):
            os.startfile(out_dir)  # noqa: S606

    # ==================================================================
    # 批量系列
    # ==================================================================
    def open_series_dialog(self):
        if self.structure is None:
            messagebox.showinfo("批量系列", "请先加载结构文件")
            return
        tk, ttk = self.tk, self.ttk
        win = tk.Toplevel(self.root)
        win.title("批量系列模拟")
        win.transient(self.root)
        win.grab_set()
        body = ttk.Frame(win, padding=12)
        body.pack(fill="both", expand=True)

        ttk.Label(body, text="系列类型").grid(row=0, column=0, sticky="w", pady=3)
        var_mode = tk.StringVar(value="厚度系列")
        box = ttk.Combobox(body, textvariable=var_mode, state="readonly", width=18,
                           values=["厚度系列", "离焦系列"])
        box.grid(row=0, column=1, sticky="w", pady=3)

        ttk.Label(body, text="厚度列表 (nm)\n如 2,4,6,8,10").grid(row=1, column=0, sticky="w", pady=3)
        ent_t = ttk.Entry(body, width=30)
        ent_t.insert(0, "2,5,10,15,20")
        ent_t.grid(row=1, column=1, sticky="w", pady=3)

        ttk.Label(body, text="离焦列表 (nm)\n如 -8,-4,0,4").grid(row=2, column=0, sticky="w", pady=3)
        ent_d = ttk.Entry(body, width=30)
        ent_d.insert(0, "-8,-4,0,4,8")
        ent_d.grid(row=2, column=1, sticky="w", pady=3)

        ttk.Label(body, text="输出文件夹").grid(row=3, column=0, sticky="w", pady=3)
        var_dir = tk.StringVar(value=str(Path.home() / "Desktop" / "stem_series"))
        ttk.Entry(body, textvariable=var_dir, width=40).grid(row=3, column=1, sticky="w")
        ttk.Button(body, text="浏览…", command=lambda: var_dir.set(
            filedialog.askdirectory(title="输出文件夹") or var_dir.get())
        ).grid(row=3, column=2, padx=4)

        fmt = ttk.LabelFrame(body, text="输出内容", padding=6)
        fmt.grid(row=4, column=0, columnspan=3, sticky="w", pady=6)
        var_tiff, var_png = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        var_npy, var_json = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
        var_montage = tk.BooleanVar(value=True)
        for text, var in (("16-bit TIFF", var_tiff), ("PNG", var_png), ("NPY(物理量)", var_npy),
                          ("参数 JSON", var_json), ("蒙太奇大图", var_montage)):
            ttk.Checkbutton(fmt, text=text, variable=var).pack(side="left", padx=6)

        ttk.Label(body, text="提示：厚度系列在引擎层是「一次传播顺带捕获」，\n"
                             "多个厚度与单个厚度耗时几乎相同；离焦系列则按个数线性增长。",
                  foreground="#555").grid(row=5, column=0, columnspan=3, sticky="w")

        def run():
            try:
                params = self.collect_params()
            except ValueError as exc:
                messagebox.showerror("参数错误", str(exc), parent=win)
                return
            try:
                mode = "thickness" if var_mode.get() == "厚度系列" else "defocus"
                thicks = parse_list(ent_t.get()) if mode == "thickness" else [params.thickness_nm]
                dfocs = parse_list(ent_d.get()) if mode == "defocus" else [params.defocus_nm]
                if any(t <= 0 for t in thicks):
                    raise ValueError("厚度必须 > 0")
            except ValueError as exc:
                messagebox.showerror("输入错误", str(exc), parent=win)
                return
            kinds = {"tiff": var_tiff.get(), "png": var_png.get(),
                     "npy": var_npy.get(), "json": var_json.get()}
            if not any(kinds.values()) and not var_montage.get():
                messagebox.showerror("输入错误", "请至少选择一种输出内容", parent=win)
                return
            if self.worker.running or self.series_worker.running:
                messagebox.showwarning("忙", "有任务正在运行", parent=win)
                return
            if params.n_phonons < 4 and not messagebox.askyesno(
                    "声子组态偏少",
                    f"当前只有 {params.n_phonons} 个声子组态，ADF 衬度不可靠。\n继续？",
                    parent=win):
                return
            out_dir = var_dir.get()
            self.set_busy(True)
            self.status("[系列] 启动…")
            self.progress.config(value=0)
            btn_run.config(state="disabled")

            def job(progress, stop):
                return run_series(params, mode, dfocs, thicks, out_dir, kinds=kinds,
                                  montage=var_montage.get(), progress=progress, stop=stop)

            self.series_worker.start(job, tag="series")
            win.destroy()

        bar = ttk.Frame(body)
        bar.grid(row=6, column=0, columnspan=3, pady=(10, 0), sticky="w")
        btn_run = ttk.Button(bar, text="运 行", command=run)
        btn_run.pack(side="left", padx=6)
        ttk.Button(bar, text="关 闭", command=win.destroy).pack(side="left")

    def _on_series_done(self, summary: dict):
        self.progress.config(value=100)
        text = (f"[系列] 完成：{summary['n_images']} 张 → {summary['out_dir']}")
        if summary.get("montage"):
            text += f"｜蒙太奇: {Path(summary['montage']).name}"
        self.status(text)
        if messagebox.askyesno(
                "批量系列完成",
                f"共导出 {summary['n_images']} 张 {summary['detector']} 图，"
                f"汇总表 {Path(summary['csv']).name}。\n\n打开输出文件夹？"):
            os.startfile(summary["out_dir"])  # noqa: S606

    # ==================================================================
    def _on_close(self):
        if (self.worker.running or self.series_worker.running) and not messagebox.askyesno(
                "退出", "模拟任务仍在运行，退出将丢失未完成的结果。\n确定退出？"):
            return
        self._closing = True
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:  # noqa: BLE001
                pass
        self.worker.stop()
        self.series_worker.stop()
        self.root.after(150, self.root.destroy)


def main() -> None:
    root = make_root()
    app = STEMApp(root)  # noqa: F841 — 保持引用
    root.mainloop()


if __name__ == "__main__":
    main()
