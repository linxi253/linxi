"""批量系列模拟：厚度系列（近零额外开销）、离焦系列，以及蒙太奇大图。

厚度系列的实现要点见 stem_sim.scan：多层法循环里每一片之后本来就要做一次
正变换，在这些位置顺带累加探测器信号即可得到任意切片边界的厚度序列，
既不重建势场也不重复传播——所以"10 个厚度"与"1 个厚度"的耗时几乎相同。

离焦系列则必须逐点重算（探针不同），代价按离焦个数线性增长。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .export import (default_basename, export_all, export_png, export_tiff16,
                     tiff_description, tiff_description as _td)
from .params import StemParams
from .render import apply_display, nice_scale_bar_length
from .sim_core import StemSimResult, run_defocus_series, run_thickness_series


def parse_list(text: str) -> List[float]:
    """解析 "1,2,3" / "1 2 3" / "1-5" 形式的数值列表。"""
    text = (text or "").strip()
    if not text:
        raise ValueError("列表为空")
    out: List[float] = []
    for chunk in text.replace("，", ",").replace(";", ",").replace("、", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk[1:] and "e" not in chunk.lower():
            lo_s, hi_s = chunk.split("-", 1)
            lo, hi = float(lo_s), float(hi_s)
            n = max(2, int(round(abs(hi - lo) / (1.0 if abs(hi - lo) >= 1 else 0.1))) + 1)
            out.extend(np.linspace(lo, hi, n).tolist())
        else:
            out.append(float(chunk))
    if not out:
        raise ValueError("未能解析出任何数值")
    return out


def run_series(
    params: StemParams,
    mode: str,
    defoci_nm: Sequence[float],
    thicknesses_nm: Sequence[float],
    out_dir,
    kinds: Optional[dict] = None,
    montage: bool = True,
    progress: Optional[Callable] = None,
    stop: Optional[Callable] = None,
) -> dict:
    """执行一条系列并导出，返回汇总 dict。

    mode : "thickness" | "defocus"
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kinds = kinds or {"tiff": True, "png": True, "npy": False, "json": False}

    if mode == "thickness":
        results = run_thickness_series(params, list(thicknesses_nm),
                                       progress=progress, stop=stop)
        labels = [f"t = {r.result.thickness_a / 10.0:.2f} nm" for r in results]
        title = (f"{results[0].formula if results else ''} {params.zone_str}  "
                 f"厚度系列   {params.voltage_kv:g} kV  "
                 f"探针 {params.probe_semiangle_mrad:g} mrad  "
                 f"探测器 {params.detector_inner_mrad:g}–{params.detector_outer_mrad:g} mrad  "
                 f"{params.n_phonons} 声子组态")
    elif mode == "defocus":
        results = run_defocus_series(params, list(defoci_nm),
                                     progress=progress, stop=stop)
        labels = [f"df = {r.params.defocus_nm:+g} nm" for r in results]
        title = (f"{results[0].formula if results else ''} {params.zone_str}  "
                 f"离焦系列   {params.voltage_kv:g} kV  "
                 f"t = {params.thickness_nm:g} nm  "
                 f"探测器 {params.detector_inner_mrad:g}–{params.detector_outer_mrad:g} mrad  "
                 f"{params.n_phonons} 声子组态")
    else:
        raise ValueError(f"未知系列类型 {mode!r}（可选 thickness / defocus）")

    det = (params.display_detector or "ADF").upper()
    exported: List[str] = []
    rows: List[dict] = []
    for i, r in enumerate(results):
        name = default_basename(r.params, r, det)
        export_all(r, out, basename=name, kind_map=kinds, detectors=[det])
        exported.append(name)
        img = r.image(det)
        rows.append(dict(
            file=name,
            thickness_nm=round(r.result.thickness_a / 10.0, 4),
            defocus_nm=r.params.defocus_nm,
            detector=det,
            mean=float(np.mean(img)),
            min=float(np.min(img)),
            max=float(np.max(img)),
            p99=float(np.percentile(img, 99.0)),
            stdev=float(np.std(img)),
        ))
        if progress is not None:
            progress(0.95 + 0.05 * (i + 1) / len(results), f"导出 {i + 1}/{len(results)}")

    csv_path = out / f"series_{mode}_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = dict(n_images=len(results), out_dir=str(out), csv=str(csv_path),
                   mode=mode, detector=det)
    if montage and results:
        p = out / f"montage_{mode}.png"
        make_montage(results, p, labels, title, det)
        summary["montage"] = str(p)
    return summary


def make_montage(
    results: List[StemSimResult],
    path,
    labels: Sequence[str],
    title: str,
    detector: str = "ADF",
    max_cols: int = 6,
    panel_in: float = 2.0,
) -> None:
    """S 系列风格的蒙太奇大图（统一对比度口径，便于直接比较）。"""
    import matplotlib

    matplotlib.use("Agg")   # 工作线程里渲染，必须用无界面后端
    import matplotlib.pyplot as plt

    from .render import setup_matplotlib_cjk

    setup_matplotlib_cjk()

    n = len(results)
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * panel_in + 0.8,
                                                  rows * panel_in + 1.1),
                             squeeze=False)
    params = results[0].params
    all_vals = np.concatenate([r.image(detector).ravel() for r in results])
    lo, hi = np.percentile(all_vals, params.contrast_lo_pct), \
        np.percentile(all_vals, params.contrast_hi_pct)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        ax.set_xticks([]); ax.set_yticks([])
        if i >= n:
            ax.axis("off")
            continue
        r = results[i]
        img = r.image(detector)
        disp = np.clip((img - lo) / max(hi - lo, 1e-30), 0, 1)
        if params.polarity < 0:
            disp = 1.0 - disp
        if params.display_blur_a > 0:
            from scipy.ndimage import gaussian_filter

            sigma = params.display_blur_a / r.out_sampling
            if sigma > 0.01:
                disp = gaussian_filter(disp, sigma)
        h, w = disp.shape
        fov_x, fov_y = w * r.out_sampling, h * r.out_sampling
        ax.imshow(disp, cmap="gray", vmin=0, vmax=1, extent=(0, fov_x, fov_y, 0))
        ax.set_title(labels[i] if i < len(labels) else "", fontsize=8)
        for s in ax.spines.values():
            s.set_color("0.6")
        bar = nice_scale_bar_length(fov_x)
        x1 = fov_x - fov_x * 0.06 - bar
        y = fov_y - fov_y * 0.06
        ax.plot([x1, x1 + bar], [y, y], color="black", lw=3.6, solid_capstyle="butt")
        ax.plot([x1, x1 + bar], [y, y], color="white", lw=1.7, solid_capstyle="butt")
        ax.text(x1 + bar / 2, y - fov_y * 0.05,
                f"{bar / 10:g} nm" if bar >= 10 else f"{bar:g} Å",
                color="white", ha="center", fontsize=6.5,
                bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))
    fig.suptitle(f"{title}    （对比度 {detector} 统一按 {params.contrast_lo_pct:g}–"
                 f"{params.contrast_hi_pct:g}% 归一）", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=170)
    plt.close(fig)
