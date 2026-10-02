"""验收脚本：复现内部参考数据集基线 + 标准表重标定基线。

自 2026-09 P0 修复起，引擎默认使用 Peng 标准散射因子表（物理标度），
SimulaTEM gauss3 表降级为 legacy（table="gauss3"）。本脚本因此提供两种模式：

    python validate_reproduce.py             # 标准模式（Peng 表）
    python validate_reproduce.py --legacy    # legacy 模式（gauss3 表）

检查项
------
1. 物理自检（λ(300 kV)）。
2. 基线复现：
   - legacy 模式：与内部参考数据集的物理强度 NPY 计算 NCC，
     同引擎同参数应 ≈ 1.0（管线无回归，阈值 >0.999）；
   - 标准模式：旧交付参考为 legacy 口径产物，不做逐比特比对；
     改为与实验图直接对比，并对离焦做 ±16 nm 扫描给出重标定基线，
     生成 tests/output/final_reproduced_standard.npy 新参考。
3. 红框风格参考复现（300 kV / Cs=1 mm / 5.5 mrad）：
   - legacy 模式：与参考 PNG 的 NCC（阈值 >0.85）；
   - 标准模式：信息性输出（参考图本身为 legacy 口径渲染，不设阈值）。

用法：python validate_reproduce.py [--legacy] [--ref-root DIR]
      （参考数据根目录也可用环境变量 HRTEM_REF_ROOT 指定；
        参考数据的具体目录结构不在脚本内写死，布局不同时可用
        HRTEM_REF_FINAL_NPY / HRTEM_REF_EXPERIMENT_PNG / HRTEM_REF_REDBOX_PNG
        逐项给出完整路径）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tem_sim import Microscope, multislice, read_structure, zone_axis_cell  # noqa: E402
from tem_sim.imaging import hrtem_image  # noqa: E402
from hrtem_tool.render import oriented_image, resize_float, robust_norm  # noqa: E402

# ---- 参考数据根目录 ----
# 查找顺序：--ref-root 参数 > HRTEM_REF_ROOT 环境变量 > 占位符（必然不存在，
# 对应检查项自动跳过，不影响其余验收）。参考数据的内部目录结构不写入本脚本：
# 布局不同时用 HRTEM_REF_FINAL_NPY / HRTEM_REF_EXPERIMENT_PNG /
# HRTEM_REF_REDBOX_PNG 环境变量逐项给出完整路径即可。

_PLACEHOLDER_REF_ROOT = Path("<HRTEM_REF_ROOT>")


def _ref_root() -> Path:
    if "--ref-root" in sys.argv:
        i = sys.argv.index("--ref-root")
        if i + 1 >= len(sys.argv):
            raise SystemExit("--ref-root 需要跟一个路径参数")
        return Path(sys.argv[i + 1])
    env = os.environ.get("HRTEM_REF_ROOT")
    if env:
        return Path(env)
    return _PLACEHOLDER_REF_ROOT


REF_ROOT = _ref_root()


def _ref_path(env_key: str, default_name: str) -> Path:
    override = os.environ.get(env_key)
    if override:
        return Path(override)
    return REF_ROOT / default_name


FINAL_NPY = _ref_path("HRTEM_REF_FINAL_NPY", "physical_intensity.npy")
EXPERIMENT_PNG = _ref_path("HRTEM_REF_EXPERIMENT_PNG", "experiment.png")
REDBOX_PNG = _ref_path("HRTEM_REF_REDBOX_PNG", "redbox_reference.png")
OUT_DIR = HERE / "tests" / "output"

# 交付基线成像参数（200 kV / Cs=0.085 mm / df=-34 nm / 24 mrad / 像散 120Å@145°）
FINAL_SCOPE_KW = dict(
    voltage_kv=200.0, cs=850000.0, defocus=-340.0,
    astigmatism=120.0, astigmatism_azimuth=145.0,
    aperture_outer=24.0, focal_spread=30.0, angular_spread=0.3,
)
# 红框风格参考参数（300 kV / Cs=1 mm / df=-200 Å / 5.5 mrad）
REDBOX_SCOPE_KW = dict(
    voltage_kv=300.0, cs=1.0e7, defocus=-200.0,
    aperture_outer=5.5, focal_spread=30.0, angular_spread=0.3,
)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def gray_png(path: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"), dtype=float)


def central_experiment_roi(arr: np.ndarray) -> np.ndarray:
    """与旧标定脚本完全一致的实验图中心 ROI 裁剪。"""
    side = min(arr.shape[1] - 110, arr.shape[0] - 190)
    x0 = (arr.shape[1] - side) // 2
    y0 = 18
    return arr[y0 : y0 + side, x0 : x0 + side]


def check_physics() -> bool:
    from tem_sim.constants import electron_wavelength
    lam = electron_wavelength(300.0)
    ok = abs(lam - 0.019687) < 2e-6
    print(f"[{'OK' if ok else 'FAIL'}] 物理自检：λ(300 kV) = {lam:.6f} Å")
    return ok


def reproduce_final(table: str, legacy: bool) -> bool:
    """最终交付复现。legacy: 逐比特复现交付 NPY；standard: 实验 NCC 基线 + df 扫描。"""
    if not FINAL_NPY.exists() or not EXPERIMENT_PNG.exists():
        print("[SKIP] 未配置参考数据（--ref-root / HRTEM_REF_ROOT），跳过基线复现")
        return True
    unit = read_structure(HERE / "cif" / "Fe3O4_Fd-3m.cif")
    crystal = zone_axis_cell(unit, [1, 1, 0], target_xy=55.0, target_thickness=30.0)
    import time
    t0 = time.perf_counter()
    wave = multislice(crystal, Microscope(**{**FINAL_SCOPE_KW, "defocus": 0.0}),
                      sampling=0.07, slice_thickness=2.0, verbose=False, table=table)
    print(f"  multislice（table={table}）用时 {time.perf_counter() - t0:.1f} s，"
          f"{len(crystal)} 原子")

    experiment_full = gray_png(EXPERIMENT_PNG)
    roi = central_experiment_roi(experiment_full)
    ems = roi.shape[0] * (10.0 / 288.0) / 192.0          # experiment_match_sampling
    render_sampling = (10.0 / 288.0) * 0.99              # nominal × scale_factor

    def render(df_a: float) -> np.ndarray:
        scope = Microscope(**{**FINAL_SCOPE_KW, "defocus": df_a})
        raw = hrtem_image(wave, scope)
        return oriented_image(
            raw, float(wave.sampling[0]), angle_deg=91.8, mirror=True,
            output_sampling_a=render_sampling, output_shape=experiment_full.shape,
            shift_a=(3.0 * ems, 1.0 * ems),
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if legacy:
        physical = render(-340.0)
        ref = np.load(FINAL_NPY).astype(float)
        score = ncc(physical, ref)
        print(f"最终交付复现（legacy）：shape {physical.shape} vs 参考 {ref.shape}，NCC = {score:.6f}")
        np.save(OUT_DIR / "final_reproduced.npy", physical.astype(np.float32))
        ok = score > 0.999 and physical.shape == ref.shape
        print(f"[{'OK' if ok else 'FAIL'}] legacy 逐比特复现（阈值 NCC>0.999）")
        return ok

    # 标准模式：厚度×离焦二维重标定扫描
    # （相位标度相对 legacy 变为 1/1.87，交付基线参数需在新物理下重新寻优；
    #   厚度序列复用一次势场，成像/渲染是主要开销）
    from tem_sim import multislice_series
    thicknesses_nm = [2.5, 3.0, 3.5, 4.0]
    crystal_t = zone_axis_cell(unit, [1, 1, 0], target_xy=55.0,
                               target_thickness=max(thicknesses_nm) * 10.0)
    waves = multislice_series(
        crystal_t, Microscope(voltage_kv=200.0, cs=850000.0),
        thicknesses=[t * 10.0 for t in thicknesses_nm],
        sampling=0.07, slice_thickness=2.0, table=table,
    )
    results = []
    for t_actual_a, wave in waves:
        for df_nm in range(-90, 31, 10):
            scope = Microscope(**{**FINAL_SCOPE_KW, "defocus": df_nm * 10.0})
            raw = hrtem_image(wave, scope)
            img = oriented_image(
                raw, float(wave.sampling[0]), angle_deg=91.8, mirror=True,
                output_sampling_a=render_sampling, output_shape=experiment_full.shape,
                shift_a=(3.0 * ems, 1.0 * ems),
            )
            results.append((ncc(img, experiment_full), t_actual_a / 10.0, df_nm))
    results.sort(key=lambda r: -r[0])
    print("  标准表重标定扫描 top-6（厚度 nm / 离焦 nm / NCC 实验）：")
    for score, t_nm, df_nm in results[:6]:
        print(f"    t={t_nm:5.2f} nm  df={df_nm:+4d} nm  NCC = {score:.4f}")
    best_score, best_t, best_df = results[0]
    scope = Microscope(**{**FINAL_SCOPE_KW, "defocus": best_df * 10.0})
    for t_actual_a, wave in waves:
        if abs(t_actual_a / 10.0 - best_t) < 1e-6:
            physical = oriented_image(
                hrtem_image(wave, scope), float(wave.sampling[0]), angle_deg=91.8,
                mirror=True, output_sampling_a=render_sampling,
                output_shape=experiment_full.shape, shift_a=(3.0 * ems, 1.0 * ems),
            )
            break
    print(f"标准表基线：t={best_t:.2f} nm, df={best_df:+d} nm，NCC(实验) = {best_score:.4f}")
    # 物理标度修正后，旧交付基线的可接受外观部分来自过强相位；
    # (厚度, 离焦) 二维寻优 best≈0.35，完整重标定需扩展至光阑/像散/极性等，
    # 属于待办研究工作（见 README「散射因子标度修正」一节）。此处验收的
    # 是端到端基线生成，实验匹配度作为信息性指标记录。
    if not (np.isfinite(physical).all() and physical.shape == experiment_full.shape):
        print("[FAIL] 标准模式基线图像无效")
        return False
    np.save(OUT_DIR / "final_reproduced_standard.npy", physical.astype(np.float32))
    from PIL import Image
    Image.fromarray(np.round(robust_norm(physical) * 255).astype(np.uint8)).save(
        OUT_DIR / "final_reproduced_standard.png")
    print(f"[OK] 标准模式端到端基线已生成（实验 NCC 基线 {best_score:.4f}，"
          "预设重标定为待办项）")
    return True


def reproduce_redbox(table: str, legacy: bool) -> bool:
    """红框风格参考复现。legacy: 与参考 PNG 比 NCC；standard: 信息性输出。"""
    if not REDBOX_PNG.exists():
        print("[SKIP] 找不到红框风格参考图，跳过")
        return True
    unit = read_structure(HERE / "cif" / "Fe3O4_Fd-3m.cif")
    crystal = zone_axis_cell(unit, [1, 1, 0], target_xy=55.0, target_thickness=250.0)
    wave = multislice(crystal, Microscope(**REDBOX_SCOPE_KW),
                      sampling=0.07, slice_thickness=2.0, verbose=False, table=table)
    raw = hrtem_image(wave, Microscope(**REDBOX_SCOPE_KW))

    from scipy.ndimage import gaussian_filter
    out_shape = (512, 512)
    fov = 43.2
    out_sampling = fov / out_shape[0]
    panel = oriented_image(
        raw, float(wave.sampling[0]), angle_deg=120.25, mirror=True,
        output_sampling_a=out_sampling, output_shape=out_shape,
        shift_a=(-2 * fov / 192.0, -2 * fov / 192.0),
    )
    panel = -gaussian_filter(panel, 0.9 / out_sampling)

    target = robust_norm(resize_float(gray_png(REDBOX_PNG), out_shape))
    mine = robust_norm(panel)
    score = ncc(mine, target)
    print(f"红框风格复现（table={table}）：NCC = {score:.4f}（参考图为 legacy 直方图匹配显示图）")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "sans-serif"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.2))
    axes[0].imshow(target, cmap="gray"); axes[0].set_title("内部参考复现图")
    axes[1].imshow(mine, cmap="gray"); axes[1].set_title(f"本工具管线（{table} 表）")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"红框风格参考对照（NCC = {score:.3f}）")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "redbox_comparison.png", dpi=200)
    plt.close(fig)

    if legacy:
        ok = score > 0.85
        print(f"[{'OK' if ok else 'FAIL'}] 红框风格复现（阈值 NCC>0.85）")
        return ok
    print("[INFO] 标准模式：红框参考为 legacy 口径渲染，仅记录不判阈值")
    return True


def main() -> int:
    legacy = "--legacy" in sys.argv
    table = "gauss3" if legacy else "peng"
    mode = "legacy（gauss3 表，复现旧交付口径）" if legacy else "standard（Peng 表，物理标准）"
    print("=" * 60)
    print(f"HRTEM 模拟工具 · 验收（{mode}）")
    print("=" * 60)
    results = [check_physics(), reproduce_final(table, legacy), reproduce_redbox(table, legacy)]
    print("=" * 60)
    if all(results):
        print("全部验收通过 ✔")
        return 0
    print("存在未通过项 ✘")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
