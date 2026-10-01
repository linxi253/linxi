"""GUI 冒烟测试：构建界面、套用预设、刷新计划与只读显示、渲染各显示模式。

不进入 mainloop，只做构建与渲染调用，用于捕捉界面代码的错误。
运行：python tests/test_gui_smoke.py
"""
from __future__ import annotations
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np


def main():
    import tkinter as tk
    from stem_tool.gui import STEMApp, make_root

    root = make_root()
    app = STEMApp(root)
    root.update()
    print("[1] 界面构建 OK")

    # 载入预设（会自动读结构、刷新计划）
    from stem_tool.params import StemParams
    from stem_tool.sim_core import load_structure, parse_cif_meta
    p = StemParams.load_json(ROOT / "presets" / "Au_100_300kV_校正_HAADF.json")
    p.cif_path = str(ROOT / "cif" / "Au_fcc_Fm-3m.cif")
    app.apply_params(p)
    root.update()
    app.update_plan()
    root.update()
    plan_txt = app.plan_text.get("1.0", "end")
    print("[2] 预设应用 + 计划刷新 OK")
    print("    计划首行:", plan_txt.splitlines()[0] if plan_txt.strip() else "(空)")
    assert "网格" in plan_txt and "预估" in plan_txt
    assert app.plan_obj is not None and app.plan_obj.n_atoms > 0

    # 质量档切换
    for q in ("预览", "标准", "高质量"):
        app.var_quality.set(q)
        app.apply_quality()
        root.update()
    print("[3] 质量档切换 OK，最终扫描点数 =", app.ent_npts.get())

    # 只读显示与 zone 快捷
    for z in ((1, 1, 0), (1, 1, 1), (0, 0, 1)):
        app.set_zone(z)
        root.update()
    app.set_zone((1, 0, 0))
    app._fill_scherzer(); app._fill_optimal_probe(); app._update_readout()
    root.update()
    print("[4] 带轴切换 / Scherzer / 最优探针角 OK；离焦 =", app.ent_df.get(),
          "探针 =", app.ent_probe.get())

    # 构造一个假结果，测试所有显示模式与导出
    from stem_tool.sim_core import StemSimResult
    from stem_sim import ScanGeometry, StemOptics, StemResult

    geom = ScanGeometry(25.0, 25.0, 24, 24)
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:24, 0:24]
    adf = 0.05 + 0.04 * (np.sin(xx * np.pi / 4) ** 2 * np.sin(yy * np.pi / 4) ** 2)
    images = {"ADF": adf, "BF": 1.0 - 0.3 * adf / adf.max(),
              "ABF": 0.5 + 0.1 * np.cos(xx * np.pi / 4)}
    optics = p.scope()
    dets = p.detectors()
    res = StemResult(images=images, captures={}, capture_thickness_a={},
                     sampling=(1.0, 1.0), extent=(25.0, 25.0), geometry=geom,
                     thickness_a=100.0, n_slices=41, dz=2.44, n_atoms=3456,
                     n_fft=1_000_000, elapsed_s=42.0, detector_specs=dets,
                     optics=optics, notes=["冒烟测试"], sigma={"Au": 0.0906},
                     phonon_summary="8 组态", n_configs=8)
    sim = StemSimResult(params=p, result=res, oriented=dict(images),
                        sampling=(1.0, 1.0), out_sampling=1.0,
                        formula="Au4", elapsed_s=42.0)
    app.result = sim
    app._on_sim_done(sim)
    root.update()
    print("[5] 结果注入 + 角标渲染 OK；探测器下拉 =", app.mode_box.cget("values"))

    for mode in list(app.mode_box.cget("values")):
        app.var_mode.set(mode)
        app._rerender_display()
        root.update()
        print(f"    显示模式 {mode!r} 渲染 OK")

    # 显示参数联动
    app.var_polarity.set("反转"); app._rerender_display()
    app.ent_blur.var.set(2.0); app._rerender_display()
    app.var_lo.set(2.0); app.var_hi.set(98.0); app._rerender_display()
    root.update()
    print("[6] 极性/模糊/对比度联动 OK")

    # 导出
    from stem_tool.export import export_all
    out = ROOT / "tests" / "output" / "gui_smoke"
    paths = export_all(sim, out, basename="gui_smoke",
                       kind_map={"tiff": True, "png": True, "npy": True, "json": True},
                       detectors=sim.detector_names())
    print(f"[7] 导出 {len(paths)} 个文件 OK")

    # 系列对话框构建（不运行）
    app.open_series_dialog()
    root.update()
    for w in root.winfo_children():
        if isinstance(w, tk.Toplevel):
            w.destroy()
    root.update()
    print("[8] 批量系列对话框构建 OK")

    app._on_close()
    root.update()
    print("\nGUI 冒烟测试通过 ✔")


if __name__ == "__main__":
    main()
