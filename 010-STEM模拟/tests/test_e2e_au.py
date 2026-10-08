"""端到端验收：Au [100] HAADF 模拟 + 导出 + 系列，检查全部产物。

两种运行方式都支持：

* ``python tests/test_e2e_au.py``  —— 脚本模式（约 3–6 分钟），产物写到
  ``tests/output/``；
* ``pytest tests/test_e2e_au.py``  —— pytest 模式。此前 ``test_single(pl)`` /
  ``test_display(res)`` 把上一个用例的返回值当 fixture 用，pytest 收集时报
  ``fixture 'pl' not found``（2 errors）。现改为真正的 pytest fixture（模块级），
  两个入口共用同一份单次/系列逻辑，不再 skip/ignore 伪绿。

pytest 模式下的产物写到 ``tmp_path``（不污染源码树）；脚本模式仍写 ``tests/output/``。
"""
from __future__ import annotations
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from stem_tool.export import export_all
from stem_tool.params import StemParams
from stem_tool.render import apply_display, oriented_image
from stem_tool.series import make_montage, run_series
from stem_tool.sim_core import plan, run_simulation, run_thickness_series

OUT = ROOT / "tests" / "output"


def base_params(**kw) -> StemParams:
    p = StemParams(
        cif_path=str(ROOT / "cif" / "Au_fcc_Fm-3m.cif"),
        zone=(1, 0, 0), thickness_nm=10.0, scan_fov_nm=2.5, scan_points=32,
        voltage_kv=300.0, cs_mm=0.01, defocus_nm=-4.44,
        probe_semiangle_mrad=25.0, detector_inner_mrad=50.0, detector_outer_mrad=100.0,
        n_phonons=8, temperature_k=300.0, sampling_a=0.1, slice_thickness_a=2.0,
        parallel="auto", with_bright_field=True, display_detector="ADF",
    )
    return p.replace(**kw) if kw else p


def build_plan():
    """构建并校验模拟计划（脚本与 pytest 共用）。"""
    p = base_params()
    pl = plan(p)
    print(f"[计划] 超胞 {np.round(pl.cell_a,2)} Å  {pl.n_atoms} 原子")
    print(f"       网格 {pl.grid[1]}×{pl.grid[0]}  采样 {pl.sampling[0]:.4f} Å/px")
    print(f"       切片 {pl.n_slices}  奈奎斯特 {pl.nyquist_mrad:.1f} mrad")
    print(f"       FFT {pl.n_fft/1e6:.2f} M  预估 {pl.estimate_s/60:.2f} min  并行 {pl.n_workers}")
    print(f"       位移 σ: {({k: round(v,4) for k,v in pl.sigma.items()})}")
    print(f"       警告: {pl.warnings}")
    assert pl.n_atoms > 1000 and pl.n_fft > 0
    assert not any("超过" in w and "外角" in w for w in pl.warnings), pl.warnings
    return pl


def run_single(pl, out_dir: Path):
    """跑单次模拟并导出（脚本与 pytest 共用）。"""
    p = base_params()
    res = run_simulation(p, progress=lambda f, m: None, plan_obj=pl)
    r = res.result
    print(f"[单次] 用时 {r.elapsed_s:.1f} s（预估 {pl.estimate_s:.1f} s）")
    for name in res.detector_names():
        img = res.image(name)
        print(f"   {name:>4s}: min={img.min():.3e} max={img.max():.3e} "
              f"mean={img.mean():.3e} 衬度(max/mean)={img.max()/max(img.mean(),1e-30):.2f}")
    adf = res.image("ADF")
    assert 0 < adf.max() < 1.0
    assert adf.max() / adf.mean() > 2.0, "ADF 衬度过低，可能未形成原子柱衬度"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = export_all(res, out_dir, basename="Au100_HAADF_demo",
                       kind_map={"tiff": True, "png": True, "npy": True, "json": True},
                       detectors=res.detector_names())
    for k, v in paths.items():
        print(f"   导出 {k:>9s} -> {Path(v).name}")
    assert len(paths) >= 8
    return res


def run_series_case(out_dir: Path):
    """厚度系列 + 蒙太奇（脚本与 pytest 共用）。"""
    p = base_params(scan_points=20, n_phonons=4)
    s = run_series(p, "thickness", [0.0], [2.0, 5.0, 10.0, 15.0], out_dir / "series",
                   kinds={"tiff": True, "png": True, "npy": False, "json": False},
                   montage=True, progress=lambda f, m: None)
    print(f"[厚度系列] {s['n_images']} 张 -> {s['out_dir']}")
    print(f"           蒙太奇 {Path(s['montage']).name}")
    import csv
    rows = list(csv.DictReader(open(s["csv"], encoding="utf-8-sig")))
    for r in rows:
        print(f"   t={float(r['thickness_nm']):5.2f} nm  mean={float(r['mean']):.3e}")
    means = [float(r["mean"]) for r in rows]
    assert all(b > a for a, b in zip(means, means[1:])), "ADF 均值未随厚度单调增"
    return s


def check_display(res):
    """显示管线（极性/模糊/对比度）与取向渲染不产生异常。"""
    img = res.image("ADF")
    for pol in (1, -1):
        d = apply_display(img, res.out_sampling, polarity=pol, blur_a=1.0,
                          lo_pct=0.5, hi_pct=99.5)
        assert d.min() >= 0.0 and d.max() <= 1.0 and d.shape == img.shape
    rot = oriented_image(img, res.out_sampling, angle_deg=17.0, mirror=True,
                         output_sampling_a=res.out_sampling)
    assert rot.shape == img.shape
    print(f"[显示] 极性/模糊/对比度归一化与 17° 旋转+镜像渲染 OK（{rot.shape}）")


# ---------------------------------------------------------------------------
# pytest 入口：真正的 fixture（此前把返回值当 fixture 用，导致 2 个 collect error）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def plan_obj():
    return build_plan()


@pytest.fixture(scope="module")
def single_result(plan_obj, tmp_path_factory):
    return run_single(plan_obj, tmp_path_factory.mktemp("au100_e2e"))


def test_plan(plan_obj):
    assert plan_obj.n_atoms > 1000 and plan_obj.n_fft > 0


def test_single(single_result):
    assert single_result is not None
    assert single_result.image("ADF").max() > 0


def test_display(single_result):
    check_display(single_result)


def test_series(tmp_path):
    run_series_case(tmp_path)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    p = build_plan()
    r = run_single(p, OUT)
    check_display(r)
    run_series_case(OUT)
    print("\n端到端验收通过 ✔")
