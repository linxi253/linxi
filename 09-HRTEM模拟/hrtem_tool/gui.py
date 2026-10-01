"""HRTEM 模拟工具主界面（Tk 三栏：结构 / 电镜参数 / 显示与导出）。"""

from __future__ import annotations

import os
import queue
import sys
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
from typing import Optional
import tkinter as tk

import numpy as np

from .export import default_basename, export_all
from .params import SimParams
from .render import apply_display, nice_scale_bar_length, oriented_image
from .series import run_series
from .sim_core import parse_cif_meta, run_simulation
from .worker import Worker
from tem_sim import zone_axis_info

TOOL_ROOT = Path(__file__).resolve().parent.parent
PRESETS_DIR = TOOL_ROOT / "presets"
CIF_DIR = TOOL_ROOT / "cif"


def make_root():
    try:
        import ttkbootstrap as ttkb
        root = ttkb.Window(themename="cosmo")
    except ImportError:
        root = tk.Tk()
    root.title("HRTEM 高分辨模拟工具 · tem_sim 引擎")
    root.geometry("1500x880")
    return root


class LabeledEntry(ttk.Frame):
    """label + entry(+unit) 的小工厂。"""

    def __init__(self, parent, label, default="", unit="", width=10, label_width=16):
        super().__init__(parent)
        self.var = tk_var(parent, default)
        ttk.Label(self, text=label, width=label_width, anchor="e").pack(side="left")
        self.entry = ttk.Entry(self, textvariable=self.var, width=width)
        self.entry.pack(side="left", padx=(4, 2))
        if unit:
            ttk.Label(self, text=unit, width=6, anchor="w").pack(side="left")

    def get(self) -> str:
        return self.var.get()


def tk_var(parent, value):
    if isinstance(value, bool):
        return tk.BooleanVar(master=parent, value=value)
    if isinstance(value, int):
        return tk.IntVar(master=parent, value=value)
    if isinstance(value, float):
        return tk.DoubleVar(master=parent, value=value)
    return tk.StringVar(master=parent, value=value)


class HRTEMApp:
    def __init__(self, root):
        self.tk, self.ttk = tk, ttk
        self.root = root
        self.worker = Worker()
        self.series_worker = Worker()
        self.result = None
        self.structure = None
        self._closing = False
        self._after_id = None
        self._orient_key = None  # 实时取向缓存键：(result id, 角度, 镜像, 输出采样)
        self.cif_path = ""

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
        ttk.Label(bar, textvariable=self.var_cif, width=46, anchor="w").pack(side="left", padx=4)
        ttk.Button(bar, text="加载 CIF/PDB/XYZ…", command=self.load_cif).pack(side="left", padx=4)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(bar, text="参数预设:").pack(side="left")
        self.var_preset = tk.StringVar()
        self.preset_box = ttk.Combobox(bar, textvariable=self.var_preset, width=34,
                                       state="readonly")
        self.preset_box.pack(side="left", padx=4)
        ttk.Button(bar, text="载入", command=self.apply_preset).pack(side="left", padx=2)
        ttk.Button(bar, text="保存为预设", command=self.save_preset).pack(side="left", padx=2)
        self.btn_sim = ttk.Button(bar, text="▶ 模 拟", command=self.start_simulation)
        self.btn_sim.pack(side="left", padx=14)
        self.btn_cancel = ttk.Button(bar, text="取消", command=self.cancel_work, state="disabled")
        self.btn_cancel.pack(side="left")
        ttk.Button(bar, text="批量系列…", command=self.open_series_dialog).pack(side="left", padx=14)
        ttk.Button(bar, text="参数约定说明", command=self.show_conventions).pack(side="right")

    def _build_main(self):
        tk, ttk = self.tk, self.ttk
        main = ttk.Frame(self.root, padding=(8, 2))
        main.pack(side="top", fill="both", expand=True)
        main.columnconfigure(2, weight=1)
        main.rowconfigure(0, weight=1)

        # ---------------- 左栏：结构 ----------------
        left = ttk.Labelframe(main, text=" 结构与取向 ", padding=8)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 6))
        self.lbl_formula = ttk.Label(left, text="化学式: —", anchor="w")
        self.lbl_formula.pack(anchor="w")
        self.lbl_cell = ttk.Label(left, text="晶格: —", anchor="w", wraplength=260, justify="left")
        self.lbl_cell.pack(anchor="w")
        self.lbl_sg = ttk.Label(left, text="空间群: —", anchor="w")
        self.lbl_sg.pack(anchor="w")
        self.lbl_natoms = ttk.Label(left, text="单胞原子数: —", anchor="w")
        self.lbl_natoms.pack(anchor="w", pady=(0, 8))

        zonef = ttk.Labelframe(left, text="带轴 [h k l]", padding=6)
        zonef.pack(fill="x", pady=2)
        zrow = ttk.Frame(zonef)
        zrow.pack()
        self.var_h = tk.IntVar(value=1)
        self.var_k = tk.IntVar(value=1)
        self.var_l = tk.IntVar(value=0)
        for var in (self.var_h, self.var_k, self.var_l):
            ttk.Spinbox(zrow, from_=-9, to=9, textvariable=var, width=4,
                        command=self.update_zone_info).pack(side="left", padx=3)
        quick = ttk.Frame(zonef)
        quick.pack(pady=(6, 0))
        for label, z in (("[100]", (1, 0, 0)), ("[110]", (1, 1, 0)), ("[111]", (1, 1, 1)),
                         ("[210]", (2, 1, 0)), ("[211]", (2, 1, 1)), ("[112]", (1, 1, 2))):
            ttk.Button(quick, text=label, width=5,
                       command=lambda z=z: self.set_zone(z)).pack(side="left", padx=2)

        self.ent_thickness = LabeledEntry(left, "样品厚度", "5.0", "nm")
        self.ent_thickness.pack(anchor="w", pady=(8, 1))
        self.ent_fov = LabeledEntry(left, "横向视场(目标)", "55", "Å")
        self.ent_fov.pack(anchor="w", pady=1)
        self.ent_rotation = LabeledEntry(left, "面内旋转", "0", "°")
        self.ent_rotation.pack(anchor="w", pady=1)
        self.var_mirror = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="水平镜像", variable=self.var_mirror,
                        command=self._rerender_display).pack(anchor="w", pady=(2, 8))
        # 旋转/镜像/输出采样在单次模拟后可实时重渲染（从 raw 重新取向）
        for ent in (self.ent_rotation,):
            ent.entry.bind("<Return>", lambda e: self._rerender_display())
            ent.entry.bind("<FocusOut>", lambda e: self._rerender_display())

        infof = ttk.Labelframe(left, text="带轴投影信息", padding=6)
        infof.pack(fill="x")
        self.zone_info_text = tk_text(infof, height=9)
        self.zone_info_text.pack(fill="x")

        # ---------------- 中栏：电镜参数 ----------------
        mid = ttk.Labelframe(main, text=" 电镜与模拟参数 ", padding=8)
        mid.grid(row=0, column=1, sticky="nsw", padx=6)
        self.ent_kv = LabeledEntry(mid, "加速电压", "300", "kV")
        self.ent_kv.pack(anchor="w", pady=1)
        self.ent_kv.var.trace_add("write", lambda *_: self._update_readout())
        self.ent_cs = LabeledEntry(mid, "球差 Cs (可为负)", "1.0", "mm")
        self.ent_cs.pack(anchor="w", pady=1)
        self.ent_cs.var.trace_add("write", lambda *_: self._update_readout())

        dfrow = ttk.Frame(mid)
        dfrow.pack(anchor="w", pady=1)
        ttk.Label(dfrow, text="离焦(负=欠焦)", width=16, anchor="e").pack(side="left")
        self.ent_df = LabeledEntry(dfrow, "", "-50", "nm", width=10, label_width=0)
        self.ent_df.pack(side="left")
        self.btn_scherzer = ttk.Button(dfrow, text="Scherzer", width=8, command=self._fill_scherzer)
        self.btn_scherzer.pack(side="left", padx=4)
        self.ent_df.var.trace_add("write", lambda *_: self._update_readout())

        aprow = ttk.Frame(mid)
        aprow.pack(anchor="w", pady=1)
        ttk.Label(aprow, text="物镜光阑(0=无)", width=16, anchor="e").pack(side="left")
        self.ent_ap = LabeledEntry(aprow, "", "20", "mrad", width=10, label_width=0)
        self.ent_ap.pack(side="left")
        self.btn_optap = ttk.Button(aprow, text="最佳", width=5, command=self._fill_optimal_aperture)
        self.btn_optap.pack(side="left", padx=4)

        self.ent_spread = LabeledEntry(mid, "离焦展宽", "30", "Å")
        self.ent_spread.pack(anchor="w", pady=1)
        self.ent_conv = LabeledEntry(mid, "束发散", "0.3", "mrad")
        self.ent_conv.pack(anchor="w", pady=1)
        astig = ttk.Frame(mid)
        astig.pack(anchor="w", pady=1)
        self.ent_astig = LabeledEntry(astig, "二重像散", "0", "Å", label_width=10)
        self.ent_astig.pack(side="left")
        self.ent_astig_az = LabeledEntry(astig, "@", "0", "°", width=6, label_width=1)
        self.ent_astig_az.pack(side="left", padx=(6, 0))

        msf = ttk.Labelframe(mid, text="多层法", padding=6)
        msf.pack(fill="x", pady=(8, 4))
        self.ent_sampling = LabeledEntry(msf, "采样", "0.07", "Å/px", label_width=10)
        self.ent_sampling.pack(anchor="w", pady=1)
        grow = ttk.Frame(msf)
        grow.pack(anchor="w", pady=1)
        ttk.Label(grow, text="像素数", width=10, anchor="e").pack(side="left")
        self.var_gpts = tk.StringVar(value="auto")
        ttk.Combobox(grow, textvariable=self.var_gpts, width=8, state="readonly",
                     values=["auto", "256", "512", "1024"]).pack(side="left", padx=4)
        trow = ttk.Frame(msf)
        trow.pack(anchor="w", pady=1)
        ttk.Label(trow, text="散射因子表", width=10, anchor="e").pack(side="left")
        self.var_table = tk.StringVar(value="Peng 标准")
        ttk.Combobox(trow, textvariable=self.var_table, width=18, state="readonly",
                     values=["Peng 标准", "gauss3 legacy"]).pack(side="left", padx=4)
        self.ent_slice = LabeledEntry(msf, "切片厚度", "2.0", "Å", label_width=10)
        self.ent_slice.pack(anchor="w", pady=1)

        outf = ttk.Labelframe(mid, text="输出与显示", padding=6)
        outf.pack(fill="x", pady=(4, 4))
        self.ent_outsam = LabeledEntry(outf, "输出采样(0=原生)", "0", "Å/px", label_width=12)
        self.ent_outsam.pack(anchor="w", pady=1)
        self.ent_outsam.entry.bind("<Return>", lambda e: self._rerender_display())
        self.ent_outsam.entry.bind("<FocusOut>", lambda e: self._rerender_display())
        prow = ttk.Frame(outf)
        prow.pack(anchor="w", pady=1)
        ttk.Label(prow, text="显示极性", width=12, anchor="e").pack(side="left")
        self.var_polarity = tk.StringVar(value="正常")
        pol_box = ttk.Combobox(prow, textvariable=self.var_polarity, width=7, state="readonly",
                               values=["正常", "反转"])
        pol_box.pack(side="left", padx=4)
        pol_box.bind("<<ComboboxSelected>>", lambda e: self._rerender_display())
        self.ent_blur = LabeledEntry(outf, "探测器模糊", "0", "Å", label_width=12)
        self.ent_blur.pack(anchor="w", pady=1)
        crow = ttk.Frame(outf)
        crow.pack(anchor="w", pady=1)
        ttk.Label(crow, text="对比度 %", width=12, anchor="e").pack(side="left")
        self.var_lo = tk.DoubleVar(value=0.5)
        self.var_hi = tk.DoubleVar(value=99.5)
        self.spin_lo = ttk.Spinbox(crow, from_=0, to=100, textvariable=self.var_lo, width=6)
        self.spin_lo.pack(side="left", padx=2)
        self.spin_hi = ttk.Spinbox(crow, from_=0, to=100, textvariable=self.var_hi, width=6)
        self.spin_hi.pack(side="left", padx=2)
        # Spinbox 的 command 只响应步进按钮；键入需绑定回车/失焦后重绘
        for sb in (self.spin_lo, self.spin_hi):
            sb.bind("<Return>", lambda e: self._rerender_display())
            sb.bind("<FocusOut>", lambda e: self._rerender_display())

        self.lbl_readout = tk_text(mid, height=5)
        self.lbl_readout.pack(fill="x", pady=(8, 0))

        # ---------------- 右栏：显示与导出 ----------------
        right = ttk.Labelframe(main, text=" 模拟结果 ", padding=6)
        right.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        mode_row = ttk.Frame(right)
        mode_row.pack(fill="x")
        ttk.Label(mode_row, text="显示:").pack(side="left")
        self.var_mode = tk.StringVar(value="hrtem")
        for text, value in (("HRTEM 像", "hrtem"), ("衍射花样", "diff"), ("CTF 曲线", "ctf")):
            ttk.Radiobutton(mode_row, text=text, value=value, variable=self.var_mode,
                            command=self._rerender_display).pack(side="left", padx=6)

        # 直接用 Figure + TkAgg 画布嵌入（不经 pyplot：避免全局后端状态
        # 与隐式 figure 管理器，工作线程中的 Agg 渲染也不会影响本画布）
        import matplotlib
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        self.fig = Figure(figsize=(7.6, 6.6), dpi=100)
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
        ttk.Entry(exbar, textvariable=self.var_basename, width=44).pack(side="left", padx=4)
        self.btn_exdir = ttk.Button(exbar, text="导出到文件夹…", command=self.export_to_folder)
        self.btn_exdir.pack(side="left", padx=8)

    def _build_statusbar(self):
        tk, ttk = self.tk, self.ttk
        bar = ttk.Frame(self.root, padding=(8, 4))
        bar.pack(side="bottom", fill="x")
        self.progress = ttk.Progressbar(bar, length=380, mode="determinate",
                                        maximum=100)
        self.progress.pack(side="left")
        self.var_status = tk.StringVar(value="就绪 — 请加载 CIF 并点击「模拟」")
        ttk.Label(bar, textvariable=self.var_status, anchor="w").pack(side="left", padx=10)

    # ==================================================================
    # 工具
    # ==================================================================
    def status(self, text: str):
        self.var_status.set(text)

    def set_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        self.btn_sim.config(state=state)
        self.btn_cancel.config(state="normal" if busy else "disabled")

    def _f(self, entry, name, minimum=None):
        try:
            value = float(entry.get())
        except (ValueError, tk.TclError):
            raise ValueError(f"{name} 不是有效数字")
        if not np.isfinite(value):
            raise ValueError(f"{name} 必须为有限数值")
        if minimum is not None and value < minimum:
            raise ValueError(f"{name} 需 ≥ {minimum}")
        return value

    # ==================================================================
    # 结构加载与带轴信息
    # ==================================================================
    def load_cif(self):
        path = filedialog.askopenfilename(
            title="选择结构文件",
            initialdir=str(CIF_DIR if CIF_DIR.exists() else "."),
            filetypes=[("结构文件", "*.cif *.pdb *.xyz *.vasp"), ("CIF", "*.cif"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            from .sim_core import load_structure
            self.structure = load_structure(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("加载失败", f"{path}\n\n{exc}")
            return
        self.cif_path = path
        self.var_cif.set(Path(path).name)
        meta = parse_cif_meta(path)
        self.lbl_formula.config(text=f"化学式: {self.structure.chemical_formula()}")
        cell = self.structure.cell_lengths()
        self.lbl_cell.config(text=f"晶格: {meta['cell'] or ' / '.join(f'{v:.4f}' for v in cell)} Å")
        self.lbl_sg.config(text=f"空间群: {meta['spacegroup'] or '(P1 展开/未标注)'}")
        self.lbl_natoms.config(text=f"单胞原子数: {len(self.structure)}")
        self.update_zone_info()

    def set_zone(self, zone):
        self.var_h.set(zone[0]); self.var_k.set(zone[1]); self.var_l.set(zone[2])
        self.update_zone_info()

    def zone_tuple(self):
        try:
            return (int(self.var_h.get()), int(self.var_k.get()), int(self.var_l.get()))
        except (tk.TclError, ValueError):
            raise ValueError("带轴指数输入未完成（需为整数）")

    def update_zone_info(self):
        if self.structure is None:
            return
        try:
            zone = self.zone_tuple()
        except ValueError as exc:
            self._set_zone_text(str(exc))
            return
        if all(v == 0 for v in zone):
            self.zone_info_text.config(state="normal")
            self.zone_info_text.delete("1.0", "end")
            self.zone_info_text.insert("end", "带轴不能为 [0 0 0]")
            self.zone_info_text.config(state="disabled")
            return
        try:
            info = zone_axis_info(self.structure, zone)
        except Exception as exc:  # noqa: BLE001
            self._set_zone_text(f"该带轴不可用：\n{exc}")
            return
        try:
            t_nm = float(self.ent_thickness.get())
        except ValueError:
            t_nm = float(self.ent_thickness.var.get() or 5.0)
        nz = max(1, int(-(-t_nm * 10 // info["lz"])))
        lines = [
            f"投影周期: Lx={info['lx']:.3f} Å  Ly={info['ly']:.3f} Å",
            f"沿束周期: Lz={info['lz']:.3f} Å（厚度台阶）",
            f"投影原胞: {info['unit_cell_atoms']} 原子",
            f"周期比 ×{info['volume_ratio']:.3f}" + ("" if info["commensurate"] else "  ⚠ 非公度"),
            f"目标厚度 {t_nm:g} nm → 实际 ≈ {nz * info['lz'] / 10:.3f} nm（{nz}×Lz）",
        ]
        if self.var_gpts.get() != "auto":
            g = int(self.var_gpts.get())
            note = "（各向异性 → 矩形像素）" if abs(info["lx"] - info["ly"]) > 1e-9 else ""
            lines.append(
                f"gpts={g}: x/y 采样 = {info['lx'] / g:.3f} / {info['ly'] / g:.3f} Å/px{note}"
            )
        self._set_zone_text("\n".join(lines))

    def _set_zone_text(self, text: str):
        self.zone_info_text.config(state="normal")
        self.zone_info_text.delete("1.0", "end")
        self.zone_info_text.insert("end", text)
        self.zone_info_text.config(state="disabled")

    # ==================================================================
    # 参数收集 / 预设 / 只读显示
    # ==================================================================
    def collect_params(self) -> SimParams:
        if self.structure is None:
            raise ValueError("请先加载结构文件（CIF/PDB/XYZ）")
        zone = self.zone_tuple()
        if all(v == 0 for v in zone):
            raise ValueError("带轴不能为 [0 0 0]")
        lo = self._f(self.spin_lo, "对比度下限", minimum=0.0)
        hi = self._f(self.spin_hi, "对比度上限", minimum=0.0)
        if not lo < hi:
            raise ValueError("对比度下限需小于上限")
        return SimParams(
            cif_path=self.cif_path,
            zone=zone,
            thickness_nm=self._f(self.ent_thickness, "厚度", minimum=0.01),
            target_xy_a=self._f(self.ent_fov, "视场", minimum=1.0),
            voltage_kv=self._f(self.ent_kv, "电压", minimum=1.0),
            cs_mm=self._f(self.ent_cs, "Cs"),
            defocus_nm=self._f(self.ent_df, "离焦"),
            aperture_mrad=self._f(self.ent_ap, "光阑", minimum=0.0),
            focal_spread_a=self._f(self.ent_spread, "离焦展宽", minimum=0.0),
            angular_spread_mrad=self._f(self.ent_conv, "束发散", minimum=0.0),
            astigmatism_a=self._f(self.ent_astig, "像散"),
            astigmatism_azimuth_deg=self._f(self.ent_astig_az, "像散方位角"),
            sampling_a=self._f(self.ent_sampling, "采样", minimum=0.01),
            slice_thickness_a=self._f(self.ent_slice, "切片厚度", minimum=0.2),
            gpts_mode=self.var_gpts.get(),
            table=("peng" if self.var_table.get() == "Peng 标准" else "gauss3"),
            rotation_deg=self._f(self.ent_rotation, "旋转"),
            mirror=bool(self.var_mirror.get()),
            output_sampling_a=self._f(self.ent_outsam, "输出采样", minimum=0.0),
            polarity=1 if self.var_polarity.get() == "正常" else -1,
            display_blur_a=self._f(self.ent_blur, "模糊", minimum=0.0),
            display_lo_pct=lo,
            display_hi_pct=hi,
        )

    def apply_params(self, p: SimParams):
        self.cif_path = p.cif_path
        if p.cif_path and Path(p.cif_path).exists():
            try:
                from .sim_core import load_structure
                self.structure = load_structure(p.cif_path)
                self.var_cif.set(Path(p.cif_path).name)
                meta = parse_cif_meta(p.cif_path)
                self.lbl_formula.config(text=f"化学式: {self.structure.chemical_formula()}")
                cell = self.structure.cell_lengths()
                self.lbl_cell.config(text=f"晶格: {meta['cell'] or ' / '.join(f'{v:.4f}' for v in cell)} Å")
                self.lbl_sg.config(text=f"空间群: {meta['spacegroup'] or '(P1 展开/未标注)'}")
                self.lbl_natoms.config(text=f"单胞原子数: {len(self.structure)}")
            except Exception:  # noqa: BLE001
                pass
        elif p.cif_path:
            self.var_cif.set(f"{Path(p.cif_path).name} (缺失)")
        self.var_h.set(p.zone[0]); self.var_k.set(p.zone[1]); self.var_l.set(p.zone[2])
        self.ent_thickness.var.set(p.thickness_nm)
        self.ent_fov.var.set(p.target_xy_a)
        self.ent_rotation.var.set(p.rotation_deg)
        self.var_mirror.set(p.mirror)
        self.ent_kv.var.set(p.voltage_kv)
        self.ent_cs.var.set(p.cs_mm)
        self.ent_df.var.set(p.defocus_nm)
        self.ent_ap.var.set(p.aperture_mrad)
        self.ent_spread.var.set(p.focal_spread_a)
        self.ent_conv.var.set(p.angular_spread_mrad)
        self.ent_astig.var.set(p.astigmatism_a)
        self.ent_astig_az.var.set(p.astigmatism_azimuth_deg)
        self.ent_sampling.var.set(p.sampling_a)
        self.ent_slice.var.set(p.slice_thickness_a)
        self.var_gpts.set(p.gpts_mode)
        self.var_table.set("Peng 标准" if p.table == "peng" else "gauss3 legacy")
        self.ent_outsam.var.set(p.output_sampling_a)
        self.var_polarity.set("正常" if p.polarity >= 0 else "反转")
        self.ent_blur.var.set(p.display_blur_a)
        self.var_lo.set(p.display_lo_pct)
        self.var_hi.set(p.display_hi_pct)
        self.update_zone_info()
        self._update_readout()

    def _refresh_presets(self):
        names = []
        if PRESETS_DIR.exists():
            for f in sorted(PRESETS_DIR.glob("*.json")):
                try:
                    import json
                    data = json.loads(f.read_text(encoding="utf-8"))
                    names.append(data.get("name", f.stem))
                except Exception:  # noqa: BLE001
                    names.append(f.stem)
        self.preset_box.config(values=names)
        if names and not self.var_preset.get():
            self.var_preset.set(names[0])

    def apply_preset(self):
        name = self.var_preset.get()
        if not name:
            return
        for f in PRESETS_DIR.glob("*.json"):
            try:
                import json
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if data.get("name", f.stem) == name:
                data = dict(data)
                cif = data.get("cif_path", "")
                if cif and not Path(cif).exists():
                    local = TOOL_ROOT / cif
                    if local.exists():
                        data["cif_path"] = str(local)
                self.apply_params(SimParams.from_dict(data))
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

        path = PRESETS_DIR / f"{sanitize(name)}.json"
        PRESETS_DIR.mkdir(exist_ok=True)
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

    def _update_readout(self):
        try:
            p = SimParams(
                voltage_kv=float(self.ent_kv.get() or 300),
                cs_mm=float(self.ent_cs.get() or 1.0),
                defocus_nm=float(self.ent_df.get() or 0),
            )
            scope = p.scope()
            lines = [
                f"λ = {scope.wavelength:.5f} Å",
                f"Scherzer 离焦 = {scope.scherzer_defocus():+.0f} Å = {scope.scherzer_defocus()/10:+.1f} nm",
                f"点分辨率 = {scope.point_resolution():.3f} Å",
            ]
            self.lbl_readout.config(state="normal")
            self.lbl_readout.delete("1.0", "end")
            self.lbl_readout.insert("end", "\n".join(lines))
            self.lbl_readout.config(state="disabled")
        except (ValueError, tk.TclError):
            pass

    def _fill_scherzer(self):
        try:
            scope = SimParams(
                voltage_kv=float(self.ent_kv.get()), cs_mm=float(self.ent_cs.get())
            ).scope()
        except ValueError:
            return
        self.ent_df.var.set(round(scope.scherzer_defocus() / 10.0, 2))

    def _fill_optimal_aperture(self):
        try:
            p = SimParams(
                voltage_kv=float(self.ent_kv.get()), cs_mm=float(self.ent_cs.get()),
                defocus_nm=float(self.ent_df.get()),
            )
            self.ent_ap.var.set(round(p.scope().optimal_aperture(), 2))
        except ValueError:
            pass

    def show_conventions(self):
        messagebox.showinfo(
            "参数约定（SimulaTEM 惯例）",
            "离焦：正 = 过焦，负 = 欠焦（Scherzer 为负值）。\n\n"
            "球差 Cs：单位 mm，1 mm = 1e7 Å；负值表示球差校正器过校正\n"
            "（如 -0.8 mm），校正区典型值 0.001–0.1 mm。\n\n"
            "厚度：超胞沿带轴按最小周期 Lz 向上取整，界面显示实际厚度。\n\n"
            "散射因子表：默认 Peng 1999 标准表（物理标度）；\n"
            "gauss3 legacy 仅用于复现 SimulaTEM/0820 交付口径\n"
            "（标度偏大 ~1.87×，见 README）。导出元数据会记录所用表。\n\n"
            "对比度：屏幕显示与导出 PNG 使用下方百分比（默认 0.5–99.5）；\n"
            "16-bit TIFF 固定 0.05–99.95（0820 交付口径），两种口径均\n"
            "写入文件元数据。物理强度 NPY 不做归一。\n\n"
            "面内旋转/镜像/输出采样：单次模拟完成后可实时调整\n"
            "（自动从原始强度重新渲染，无需重新模拟）。\n\n"
            "引擎：tem_sim（Cowley–Moody 多层法，SimulaTEM v1.3.2 物理重构，\n"
            "Peng 五高斯电子散射因子 + 误差函数像素积分）。",
        )

    # ==================================================================
    # 模拟执行
    # ==================================================================
    def _maybe_warn_grid(self, params: SimParams) -> bool:
        """网格规模过大时提示内存风险；返回 False 表示用户放弃运行。

        透射函数已惰性化（内存 O(N²) 而非 O(n_slices·N²)），这里按波场 +
        FFT 工作区约 7 个 N² 复数数组估算。
        """
        if self.structure is None:
            return True
        try:
            info = zone_axis_info(self.structure, params.zone)
        except Exception:  # noqa: BLE001 - 带轴不可用时引擎自会报错
            return True
        if params.gpts_mode == "auto":
            nx = max(32, int(round(info["lx"] / params.sampling_a)))
            ny = max(32, int(round(info["ly"] / params.sampling_a)))
        else:
            nx = ny = int(params.gpts_mode)
        px = nx * ny
        if px <= 16_000_000:  # 4096²：工作区约 2 GB 量级，可接受
            return True
        est_gb = 7.0 * px * 16 / 1e9
        return messagebox.askyesno(
            "网格规模确认",
            f"网格 {nx}×{ny}（{px / 1e6:.1f} MP）预计内存占用约 {est_gb:.0f} GB，\n"
            "可能导致系统卡顿或内存不足。\n仍要继续？",
        )

    def start_simulation(self):
        if self.worker.running or self.series_worker.running:
            return
        try:
            params = self.collect_params()
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        if not self._maybe_warn_grid(params):
            return
        self.set_busy(True)
        self.status("模拟中…")
        self.progress.config(value=0)
        self.worker.start(lambda progress, stop: run_simulation(params, progress, stop))

    def cancel_work(self):
        if self.worker.stop() or self.series_worker.stop():
            self.status("正在取消…")

    def _poll(self):
        if self._closing:
            return
        # 事件处理中的任何异常都不允许中断轮询循环，
        # 否则后续进度/结果/错误事件将永远无法送达界面
        try:
            for worker in (self.worker, self.series_worker):
                try:
                    while True:
                        event = worker.events.get_nowait()
                        self._handle_event(event)
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
            summary, detail = event[1], event[2]
            print(detail, file=sys.stderr)
            messagebox.showerror("出错", f"{summary}\n\n完整回溯已打印到控制台。")
            self.status(f"出错：{summary}")
        elif kind == "done":
            if not self.worker.running and not self.series_worker.running:
                self.set_busy(False)

    def _on_sim_done(self, result):
        self.result = result
        p = result.params
        info = result.zone_periods
        self._set_zone_text("\n".join([
            f"投影周期: Lx={info['lx']:.3f} Å  Ly={info['ly']:.3f} Å",
            f"沿束周期: Lz={info['lz']:.3f} Å",
            f"超胞: {result.n_atoms} 原子",
            f"实际厚度: {result.thickness_actual_nm:.3f} nm",
            f"原生采样: {result.sampling:.4f} Å/px  输出: {result.out_sampling:.4f} Å/px",
            f"用时: {result.elapsed_s:.1f} s",
        ]))
        if not self.var_basename.get():
            self.var_basename.set(default_basename(p, result))
        self.progress.config(value=100)
        self.status(f"完成（{result.elapsed_s:.1f} s）— {result.formula} {p.zone_str} "
                    f"t={result.thickness_actual_nm:.2f} nm")
        self._rerender_display()

    # ==================================================================
    # 显示渲染
    # ==================================================================
    def _current_oriented(self):
        """按当前 GUI 的旋转/镜像/输出采样从 raw 重新取向渲染。

        单次模拟保留了原始强度 raw，取向只是廉价后处理，因此这三个控件
        可以实时联动；参数无效时沿用上次渲染并提示。返回 (oriented, 采样)。
        系列结果不保留 raw，直接返回烘焙好的 oriented。
        """
        result = self.result
        if result is None:
            return None, 0.0
        if result.raw is None:
            return result.oriented, result.out_sampling
        try:
            angle = float(self.ent_rotation.get())
            mirror = bool(self.var_mirror.get())
            outsam = float(self.ent_outsam.get())
        except (ValueError, tk.TclError):
            self.status("取向参数无效，沿用上次渲染")
            return result.oriented, result.out_sampling
        key = (id(result), angle, mirror, outsam)
        if key != self._orient_key:
            try:
                result.oriented = oriented_image(
                    result.raw, result.sampling,
                    angle_deg=angle, mirror=mirror, output_sampling_a=outsam,
                )
            except ValueError as exc:
                self.status(f"取向渲染失败：{exc}")
                return result.oriented, result.out_sampling
            result.out_sampling = outsam if outsam > 0 else result.sampling
            # 同步进参数对象，导出文件名/元数据与屏幕所见一致
            result.params = result.params.replace(
                rotation_deg=angle, mirror=mirror, output_sampling_a=outsam
            )
            self._orient_key = key
        return result.oriented, result.out_sampling

    def _rerender_display(self):
        if self.result is None:
            return
        try:
            blur = float(self.ent_blur.get())
        except ValueError:
            blur = 0.0
        mode = self.var_mode.get()
        ax = self.ax
        ax.clear()
        result = self.result
        p = result.params
        if mode == "hrtem":
            oriented, out_sampling = self._current_oriented()
            try:
                lo_pct = float(self.var_lo.get())
                hi_pct = float(self.var_hi.get())
            except (ValueError, tk.TclError):
                lo_pct, hi_pct = 0.5, 99.5
            display = apply_display(
                oriented, out_sampling,
                polarity=1 if self.var_polarity.get() == "正常" else -1,
                blur_a=blur,
                lo_pct=lo_pct,
                hi_pct=hi_pct,
            )
            h, w = display.shape
            fov_x, fov_y = w * out_sampling, h * out_sampling
            ax.imshow(display, cmap="gray", vmin=0, vmax=1,
                      extent=(0, fov_x, fov_y, 0))
            self._draw_scale_bar(ax, fov_x, fov_y)
            info = (f"{result.formula} {p.zone_str}\n"
                    f"t = {result.thickness_actual_nm:.2f} nm\ndf = {p.defocus_nm:+g} nm\n"
                    f"{p.voltage_kv:g} kV  Cs = {p.cs_mm:g} mm\n"
                    f"{out_sampling:.4f} Å/px")
            ax.text(0.015, 0.03, info, transform=ax.transAxes, va="bottom", ha="left",
                    color="white", fontsize=8.5,
                    bbox=dict(facecolor="black", alpha=0.55, pad=3, edgecolor="none"))
            ax.set_title(f"{result.formula} {p.zone_str}  HRTEM 模拟", fontsize=11)
        elif mode == "diff":
            if result.exit_wave is None:
                self.status("该结果未保留出射波，无衍射花样（已切回 HRTEM 像）")
                self.var_mode.set("hrtem")
                return self._rerender_display()
            dp = result.diffraction()
            ny, nx = dp.shape
            # 衍射谱来自原生采样网格的出射波，坐标轴必须用原生采样
            # （out_sampling 是取向渲染后的重采样，k 轴标度与它无关）
            ky = np.fft.fftshift(np.fft.fftfreq(ny, d=result.sampling))
            kx = np.fft.fftshift(np.fft.fftfreq(nx, d=result.sampling))
            ax.imshow(dp, cmap="magma",
                      extent=(kx[0], kx[-1], ky[-1], ky[0]))
            ax.set_xlabel("kx (Å$^{-1}$)")
            ax.set_ylabel("ky (Å$^{-1}$)")
            ax.set_title(f"{result.formula} {p.zone_str} 衍射花样（log）", fontsize=11)
        else:
            from tem_sim.imaging import ctf_curve
            scope = p.scope()
            k_max = 1.4
            k, y = ctf_curve(scope, k_max=k_max)
            ax.plot(k, y, lw=1.8, label="-sinχ · E")
            ax.axhline(0, color="gray", lw=0.6)
            if p.aperture_mrad > 0:
                k_ap = p.aperture_mrad * 1e-3 / scope.wavelength
                ax.axvline(k_ap, color="crimson", ls="--", lw=1.2,
                           label=f"光阑 {p.aperture_mrad:g} mrad")
            ax.set_xlim(0, k_max)
            ax.set_xlabel("k (Å$^{-1}$)")
            ax.set_ylabel("CTF")
            ax.legend(fontsize=9)
            ax.set_title(f"CTF：{p.voltage_kv:g} kV, Cs={p.cs_mm:g} mm, df={p.defocus_nm:+g} nm",
                         fontsize=11)
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
        # 先同步实时取向（旋转/镜像/输出采样），保证导出与屏幕所见一致
        self._current_oriented()
        name = self.var_basename.get().strip() or None
        kinds = {"tiff": True, "png": True, "npy": True, "json": True}
        try:
            paths = export_all(self.result, out_dir, basename=name, kind_map=kinds)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("导出失败", "".join(traceback.format_exception(exc))[-1200:])
            return
        self.status("已导出: " + ", ".join(Path(v).name for v in paths.values()))
        if messagebox.askyesno("导出完成", "已导出 4 个文件（PNG/TIFF/NPY/JSON）。\n打开输出文件夹？"):
            self._open_folder(out_dir)

    @staticmethod
    def _open_folder(path: str):
        try:
            os.startfile(path)  # noqa: S606 (Windows)
        except (AttributeError, OSError) as exc:
            # 非 Windows 平台无 os.startfile，或资源管理器打开失败
            print(f"无法自动打开文件夹 {path}：{exc}", file=sys.stderr)

    # ==================================================================
    # 批量系列对话框
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
        var_mode = tk.StringVar(value="defocus")
        mode_box = ttk.Combobox(body, textvariable=var_mode, state="readonly", width=18,
                                values=["离焦系列", "厚度系列", "厚度×离焦矩阵"])
        mode_box.current(0)
        mode_box.grid(row=0, column=1, sticky="w", pady=3)

        def focus_widgets_state():
            is_defocus = var_mode.get() in ("离焦系列", "厚度×离焦矩阵")
            is_thick = var_mode.get() in ("厚度系列", "厚度×离焦矩阵")
            ent_df["state"] = "normal" if is_defocus else "disabled"
            ent_dt["state"] = "normal" if is_thick else "disabled"
        mode_box.bind("<<ComboboxSelected>>", lambda e: focus_widgets_state())

        ttk.Label(body, text="离焦列表 (nm)\n如 -50,-40,-30").grid(row=1, column=0, sticky="w", pady=3)
        ent_df = ttk.Entry(body, width=28)
        ent_df.insert(0, "-80,-60,-40,-20,0,20")
        ent_df.grid(row=1, column=1, sticky="w", pady=3)

        ttk.Label(body, text="厚度列表 (nm)\n如 2,4,6,8,10").grid(row=2, column=0, sticky="w", pady=3)
        ent_dt = ttk.Entry(body, width=28)
        ent_dt.insert(0, "2,4,6,8,10")
        ent_dt.grid(row=2, column=1, sticky="w", pady=3)

        ttk.Label(body, text="输出文件夹").grid(row=3, column=0, sticky="w", pady=3)
        var_dir = tk.StringVar(value=str(Path.home() / "Desktop" / "hrtem_series"))
        ttk.Entry(body, textvariable=var_dir, width=38).grid(row=3, column=1, sticky="w")
        ttk.Button(body, text="浏览…", command=lambda: var_dir.set(
            filedialog.askdirectory(title="输出文件夹") or var_dir.get())
        ).grid(row=3, column=2, padx=4)

        fmt = ttk.LabelFrame(body, text="输出内容", padding=6)
        fmt.grid(row=4, column=0, columnspan=3, sticky="w", pady=6)
        var_tiff = tk.BooleanVar(value=True)
        var_png = tk.BooleanVar(value=True)
        var_npy = tk.BooleanVar(value=False)
        var_json = tk.BooleanVar(value=False)
        var_montage = tk.BooleanVar(value=True)
        for text, var in (("16-bit TIFF", var_tiff), ("PNG", var_png), ("NPY", var_npy),
                          ("参数 JSON", var_json), ("蒙太奇大图", var_montage)):
            ttk.Checkbutton(fmt, text=text, variable=var).pack(side="left", padx=6)

        btnbar = ttk.Frame(body)
        btnbar.grid(row=5, column=0, columnspan=3, pady=(8, 0))
        var_status = tk.StringVar(value="")
        ttk.Label(body, textvariable=var_status).grid(row=6, column=0, columnspan=3, sticky="w")

        def run():
            try:
                params = self.collect_params()
            except ValueError as exc:
                messagebox.showerror("参数错误", str(exc), parent=win)
                return
            from .series import parse_list
            try:
                mode_map = {"离焦系列": "defocus", "厚度系列": "thickness", "厚度×离焦矩阵": "matrix"}
                mode = mode_map[var_mode.get()]
                defoci = parse_list(ent_df.get()) if mode in ("defocus", "matrix") else [params.defocus_nm]
                thicks = parse_list(ent_dt.get()) if mode in ("thickness", "matrix") else [params.thickness_nm]
                if any(t <= 0 for t in thicks):
                    raise ValueError("厚度必须 > 0")
            except ValueError as exc:
                messagebox.showerror("输入错误", str(exc), parent=win)
                return
            kinds = {"tiff": var_tiff.get(), "png": var_png.get(),
                     "npy": var_npy.get(), "json": var_json.get()}
            montage_flag = var_montage.get()  # 主线程读取；worker 线程访问 Tk 变量不安全
            if not any(kinds.values()) and not montage_flag:
                messagebox.showerror("输入错误", "请至少选择一种输出内容", parent=win)
                return
            if self.worker.running or self.series_worker.running:
                messagebox.showwarning("忙", "有任务正在运行", parent=win)
                return
            if not self._maybe_warn_grid(params):
                return
            out_dir = var_dir.get()
            self.set_busy(True)
            self.status("[系列] 启动…")
            self.progress.config(value=0)
            btn_run.config(state="disabled")

            def job(progress, stop):
                return run_series(params, mode, defoci, thicks, out_dir,
                                  kinds=kinds, montage=montage_flag,
                                  progress=progress, stop=stop)

            self.series_worker.start(job, tag="series")
            win.destroy()

        btn_run = ttk.Button(btnbar, text="运 行", command=run)
        btn_run.pack(side="left", padx=6)
        ttk.Button(btnbar, text="关 闭", command=win.destroy).pack(side="left")
        focus_widgets_state()

    def _on_series_done(self, summary: dict):
        self.progress.config(value=100)
        self.status(f"[系列] 完成：{summary['n_images']} 张 → {summary['out_dir']}")
        if summary.get("montage"):
            self.status(self.var_status.get() + f"｜蒙太奇: {Path(summary['montage']).name}")
        if messagebox.askyesno("批量系列完成",
                               f"共导出 {summary['n_images']} 张图。\n打开输出文件夹？"):
            self._open_folder(summary["out_dir"])

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


def tk_text(parent, height=4):
    text = tk.Text(parent, height=height, wrap="word", relief="flat",
                   background="#f0f0f0", font=("Microsoft YaHei", 9))
    text.config(state="disabled")
    return text
