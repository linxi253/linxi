"""批量系列模拟：离焦系列 / 厚度系列 / 厚度×离焦矩阵 + 蒙太奇大图。

厚度方向复用一次势场与传播（tem_sim.multislice_series 多厚度捕获）；
离焦方向复用同一出射波（只重算 CTF 成像）。
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from tem_sim import multislice, multislice_series
from tem_sim.imaging import hrtem_image

from .export import default_basename, export_all
from .params import SimParams
from .render import apply_display, oriented_image
from .sim_core import (
    CancelledError,
    SimResult,
    build_crystal,
    effective_gpts,
    effective_sampling,
    load_structure,
)


def parse_list(text: str, max_count: int = 400) -> List[float]:
    """解析 '2, 4, 6' 风格的数值列表（支持中文逗号与逗号两侧空格）。

    拒绝 NaN/Inf、无法解析的片段与超过 max_count 的列表（后者是批量渲染
    内存/时间代价的软上限）。
    """
    values: List[float] = []
    for part in text.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = float(part)
        except ValueError:
            raise ValueError(
                f"无法解析数值：'{part}'（多个数值请用逗号分隔）"
            ) from None
        if not math.isfinite(v):
            raise ValueError(f"数值必须为有限数：'{part}'")
        values.append(v)
    if not values:
        raise ValueError("数值列表为空")
    if len(values) > max_count:
        raise ValueError(
            f"列表长度 {len(values)} 超过上限 {max_count}（批量渲染内存/时间代价过大）"
        )
    return values


def _formula_of(params: SimParams) -> str:
    try:
        return load_structure(params.cif_path).chemical_formula()
    except Exception:  # noqa: BLE001
        return Path(params.cif_path).stem or "sim"


def run_series(
    params: SimParams,
    mode: str,                     # "defocus" / "thickness" / "matrix"
    defocus_list_nm: List[float],
    thickness_list_nm: List[float],
    out_dir: str,
    kinds: Optional[dict] = None,
    montage: bool = True,
    progress: Optional[Callable] = None,
    stop: Optional[Callable] = None,
) -> dict:
    """执行系列模拟并落盘，返回汇总 dict（含 rows，供 GUI 展示）。"""
    params.validate()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kinds = kinds or {"tiff": True, "png": True, "npy": False, "json": False}
    formula = _formula_of(params)

    def check_stop():
        if stop is not None and stop():
            raise CancelledError("用户取消")

    def report(frac, msg):
        if progress is not None:
            progress(frac, msg)

    # ---- 1. 波场：厚度序列一次传播捕获，离焦序列单厚度 ----
    structure = load_structure(params.cif_path)

    def ms_progress(base: float, span: float):
        def hook(frac: float, msg: str):
            check_stop()  # 势场/传播阶段同样响应取消
            report(base + span * frac, msg)
        return hook

    if mode in ("thickness", "matrix"):
        thicknesses = sorted(set(thickness_list_nm))
        p_max = params.replace(thickness_nm=max(thicknesses))
        crystal, info = build_crystal(p_max, structure)
        report(0.02, f"厚度序列势场：{info['n_atoms']} 原子，最长 {info['actual_thickness_a']/10:.1f} nm")
        waves = multislice_series(
            crystal, params.scope(),
            thicknesses=[t * 10.0 for t in thicknesses],
            sampling=effective_sampling(params, structure),
            gpts=effective_gpts(params),
            slice_thickness=params.slice_thickness_a,
            progress=ms_progress(0.05, 0.55),
            table=params.table,
        )
        # 渲染/记录使用出射波的真实采样（gpts 模式下与目标值不同）
        wave_list: List[Tuple[float, object, float]] = [
            (t_a / 10.0, w, float(w.sampling[0])) for t_a, w in waves
        ]
    else:  # defocus
        crystal, info = build_crystal(params, structure)
        report(0.02, f"势场：{info['n_atoms']} 原子，{info['actual_thickness_a']/10:.1f} nm")
        wave = multislice(
            crystal, params.scope(), sampling=effective_sampling(params, structure),
            gpts=effective_gpts(params),
            slice_thickness=params.slice_thickness_a, verbose=False,
            progress=ms_progress(0.05, 0.6), table=params.table,
        )
        wave_list = [(info["actual_thickness_a"] / 10.0, wave, float(wave.sampling[0]))]

    if mode == "thickness":
        defoci = [params.defocus_nm]
    else:
        defoci = list(defocus_list_nm)

    # ---- 2. 逐条件成像 + 导出 ----
    rows: List[dict] = []
    montage_entries: List[dict] = []
    total = len(wave_list) * len(defoci)
    counter = 0

    for t_nm, w, sampling_native in wave_list:
        for df in defoci:
            check_stop()
            counter += 1
            report(0.65 + 0.33 * counter / total, f"成像 t={t_nm:.2f} nm df={df:+g} nm")
            p = params.replace(thickness_nm=t_nm, defocus_nm=float(df))
            image = hrtem_image(w, p.scope())
            oriented = oriented_image(
                image, sampling_native,
                angle_deg=p.rotation_deg, mirror=p.mirror,
                output_sampling_a=p.output_sampling_a,
            )
            out_sampling = p.output_sampling_a if p.output_sampling_a > 0 else sampling_native
            result = SimResult(
                params=p, raw=None, oriented=oriented, exit_wave=None,
                thickness_actual_nm=t_nm, sampling=sampling_native,
                out_sampling=out_sampling, n_atoms=info["n_atoms"], formula=formula,
            )
            name = default_basename(p, result)
            export_all(result, out, name, kinds)
            rows.append(dict(
                formula=formula,
                table=p.table,
                thickness_nm=round(t_nm, 4), defocus_nm=df, zone=p.zone_str,
                voltage_kv=p.voltage_kv, cs_mm=p.cs_mm, aperture_mrad=p.aperture_mrad,
                sampling_a_per_px=sampling_native, out_sampling_a_per_px=out_sampling,
                rotation_deg=p.rotation_deg, mirror=p.mirror, polarity=p.polarity,
                file=name,
            ))
            montage_entries.append(dict(
                image=apply_display(oriented, out_sampling, polarity=p.polarity,
                                    blur_a=p.display_blur_a,
                                    lo_pct=p.display_lo_pct, hi_pct=p.display_hi_pct),
                title=f"t={t_nm:.2f} nm\ndf={df:+g} nm",
            ))

    # ---- 3. summary CSV + 蒙太奇 ----
    csv_path = out / "series_summary.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    montage_path = None
    if montage and montage_entries:
        report(0.99, "生成蒙太奇大图…")
        montage_path = str(out / "montage.png")
        save_montage(
            montage_entries, montage_path,
            n_cols=(len(defoci) if mode != "thickness" else 1),
            title=f"{formula} {params.zone_str} 系列（{params.voltage_kv:g} kV, Cs={params.cs_mm:g} mm）",
        )

    report(1.0, f"完成：{len(rows)} 张")
    return dict(out_dir=str(out), n_images=len(rows), csv=str(csv_path),
                montage=montage_path, rows=rows)


def save_montage(entries: List[dict], path: str, n_cols: int = 1, title: str = "") -> None:
    """网格蒙太奇 PNG（每格带条件标签）。

    用 Agg 画布直接渲染（不经 pyplot、不切换全局后端），因此可以安全地
    在 GUI 后台工作线程中调用，不会触碰主线程的 Tk 图窗。
    """
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    n = len(entries)
    n_cols = max(1, min(n_cols, n))
    n_rows = int(np.ceil(n / n_cols))
    fig = Figure(figsize=(2.5 * n_cols, 2.7 * n_rows), dpi=200)
    FigureCanvasAgg(fig)
    axes = fig.subplots(n_rows, n_cols, squeeze=False)
    for i, entry in enumerate(entries):
        ax = axes[i // n_cols][i % n_cols]
        ax.imshow(entry["image"], cmap="gray", vmin=0, vmax=1)
        ax.set_title(entry["title"], fontsize=8)
        ax.axis("off")
    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
