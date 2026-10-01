"""生成示例图：Au [100] HAADF/BF/ABF 三联图 + 厚度系列蒙太奇。"""
from __future__ import annotations
import sys
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from stem_tool.params import StemParams
    from stem_tool.render import setup_matplotlib_cjk
    from stem_tool.sim_core import plan, run_simulation
    from stem_tool.series import run_series

    setup_matplotlib_cjk()
    OUT = ROOT / "examples"
    OUT.mkdir(exist_ok=True)

    p = StemParams(cif_path=str(ROOT / "cif" / "Au_fcc_Fm-3m.cif"),
                   zone=(1, 0, 0), thickness_nm=10.0, scan_fov_nm=2.5, scan_points=48,
                   voltage_kv=300.0, cs_mm=0.01, defocus_nm=-4.44,
                   probe_semiangle_mrad=25.0, detector_inner_mrad=50.0,
                   detector_outer_mrad=100.0, n_phonons=8, sampling_a=0.0,
                   slice_thickness_a=2.5, parallel="auto", with_bright_field=True)
    pl = plan(p)
    print(f"计划: 网格 {pl.grid[1]}² 切片 {pl.n_slices} FFT {pl.n_fft/1e6:.2f}M "
          f"预估 {pl.estimate_s/60:.1f} min 并行 {pl.n_workers}")
    res = run_simulation(p, progress=lambda f, m: None, plan_obj=pl)
    r = res.result
    print(f"完成 {r.elapsed_s:.0f}s，实测 {r.fft_rate:.0f} FFT/s")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.9))
    for ax, name in zip(axes, ("ADF", "ABF", "BF")):
        img = res.image(name)
        lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.5)
        ax.imshow(np.clip((img - lo) / (hi - lo), 0, 1), cmap="gray",
                  extent=(0, img.shape[1] * res.out_sampling,
                          img.shape[0] * res.out_sampling, 0))
        ax.set_title(f"STEM-{name}", fontsize=12)
        ax.set_xlabel("x (nm)" if False else "x (Å)")
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("y (Å)")
    fig.suptitle(
        f"Au [100]  {p.voltage_kv:g} kV  Cs={p.cs_mm:g} mm  df={p.defocus_nm:+g} nm  "
        f"探针 {p.probe_semiangle_mrad:g} mrad  t={r.thickness_a/10:.2f} nm  "
        f"ADF {p.detector_inner_mrad:g}–{p.detector_outer_mrad:g} mrad  "
        f"{p.n_phonons} 声子组态", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT / "Au100_HAADF_BF_ABF.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("示例图:", path)

    s = run_series(p.replace(scan_points=32, n_phonons=6), "thickness", [0.0],
                   [2.0, 5.0, 10.0, 15.0, 20.0], OUT / "series",
                   kinds={"tiff": True, "png": True, "npy": False, "json": False},
                   montage=True, progress=lambda f, m: None)
    print("厚度系列蒙太奇:", s["montage"])

    # 探针剖面图
    from stem_sim import probe_intensity_profile
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for alpha, df in ((20.0, -4.44), (25.0, -4.44), (30.0, -4.44)):
        sc = p.replace(probe_semiangle_mrad=alpha).scope()
        rr, prof = probe_intensity_profile(sc, (512, 512), 0.05)
        ax.plot(rr, prof, lw=1.6, label=f"α={alpha:g} mrad (r={sc.probe_size_rayleigh():.2f} Å)")
    ax.set_xlim(0, 4); ax.set_ylim(0, 1.05)
    ax.set_xlabel("r (Å)"); ax.set_ylabel("归一化强度")
    ax.set_title(f"探针强度剖面（Cs={p.cs_mm:g} mm, df={p.defocus_nm:+g} nm）", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "probe_profiles.png", dpi=160)
    plt.close(fig)
    print("探针剖面:", OUT / "probe_profiles.png")


if __name__ == "__main__":
    main()
