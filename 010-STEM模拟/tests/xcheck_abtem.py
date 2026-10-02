"""与 abTEM（领域标准实现）的交叉验证：HAADF 图像一致性与绝对标度。

abTEM 独立实现了探针构造、多层法、冻结声子与环形探测器，是理想对照。
参数化用 abTEM 的 kirkland（该版本 abTEM 只支持无限投影的 Kirkland），
本工具用 Peng 1999；两者的投影势**积分**（即 f_e(0)、平均内电位）
一致到 0.1%，但高角形状不同（核区参数化差异，峰值相差约 1.5×），
故绝对强度允许 ~2× 量级差，重点看图像形态（NCC）与厚度/Z 衬度趋势。

运行：python tests/xcheck_abtem.py    （需 pip install abtem==1.0.0b34）
"""
from __future__ import annotations
import sys
from pathlib import Path

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import os
os.environ.setdefault('ABTEM_DISABLE_PBAR','1')

try:
    import abtem
except ImportError:
    print("[SKIP] 未安装 abtem，跳过交叉验证")
    sys.exit(0)

import stem_sim as ss

VOLT, CS, DF, ALPHA = 300.0, 1e4, -14.0, 25.0
INNER, OUTER = 50.0, 98.0
SIGMA_AU = 0.0906
NPH = 8


def _fcc(sym, a):
    from stem_sim import Structure
    return Structure([sym] * 4,
                     np.array([[0, 0, 0], [0, a / 2, a / 2],
                               [a / 2, 0, a / 2], [a / 2, a / 2, 0]]),
                     cell=np.diag([a, a, a]))


def _par():
    from stem_sim.scan import _resolve_parallel, spawn_safety
    if _resolve_parallel(None, NPH, 100) and not spawn_safety()[0]:
        return False
    return None


def ncc(x, y):
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


def run_abtem(a, nx, ny, nz, sampling, slice_a, sigma, t_end, step, detector):
    from ase import Atoms
    n = 4 * nx * ny * nz
    base = np.array([[0, 0, 0], [0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    pos = np.vstack([base + np.array([i, j, k]) * a
                     for i in range(nx) for j in range(ny) for k in range(nz)])
    atoms = Atoms("Au" * n, positions=pos,
                  cell=np.diag([a * nx, a * ny, a * nz]), pbc=True)
    frozen = abtem.FrozenPhonons(atoms, NPH, sigmas=sigma, seed=20260915)
    pot = abtem.Potential(frozen, sampling=sampling, slice_thickness=slice_a,
                          parametrization="kirkland", projection="infinite")
    probe = abtem.Probe(energy=VOLT * 1e3, semiangle_cutoff=ALPHA, defocus=DF,
                        Cs=CS, rolloff=0.0)
    scan = abtem.GridScan(start=(0, 0), end=(t_end, t_end), sampling=step)
    m = probe.scan(scan, detector, pot, pbar=False)
    arr = np.asarray(m.array, dtype=float)
    if arr.ndim == 3:
        arr = arr.mean(0)
    return arr


def main():
    fov, t_nm, samp, slice_a, step = 20.0, 8.0, 0.1, 2.0, 1.0
    cry = ss.zone_axis_cell(_fcc("Au", 4.1713), (1, 0, 0),
                            target_xy=fov, target_thickness=t_nm * 10)
    lx, ly, lz = cry.cell_lengths()
    nx, ny, nz = (int(round(lx / 4.1713)), int(round(ly / 4.1713)),
                  int(round(lz / 4.1713)))
    print(f"超胞 {lx:.2f} x {ly:.2f} x {lz:.2f} Å；{nx}x{ny}x{nz} 胞，{len(cry)} 原子")

    opt = ss.StemOptics(voltage_kv=VOLT, cs=CS, defocus=DF, probe_semiangle=ALPHA,
                        detector_inner=INNER, detector_outer=OUTER)
    npt = int(round(fov / step)) + 1
    geom = ss.ScanGeometry(fov, fov, npt, npt)
    res = ss.stem_scan(cry, opt, sampling=samp, geometry=geom, slice_thickness=slice_a,
                       phonons=ss.PhononConfig(n_configs=NPH, sigma_override=SIGMA_AU),
                       parallel=_par())
    mine = res.images["ADF"]
    print(f"[stem_sim] 网格 {res.grid if hasattr(res,'grid') else ''} 采样 "
          f"{res.sampling[0]:.4f} Å/px  扫描 {mine.shape}")

    theirs = run_abtem(4.1713, nx, ny, nz, samp, slice_a, SIGMA_AU, fov, step,
                       abtem.AnnularDetector(inner=INNER, outer=OUTER))
    print(f"[abTEM  ] 扫描 {theirs.shape}")

    m = min(mine.shape[0], theirs.shape[0]), min(mine.shape[1], theirs.shape[1])
    A, B = mine[:m[0], :m[1]], theirs[:m[0], :m[1]]
    print(f"\n--- ADF 图像对比（{m[0]}x{m[1]}）---")
    for nm, arr in (("stem_sim", A), ("abTEM", B)):
        print(f"  {nm:9s} min={arr.min():.4e} max={arr.max():.4e} mean={arr.mean():.4e}")
    print(f"  绝对标度比 stem_sim/abTEM: mean={A.mean()/B.mean():.3f}  "
          f"max={A.max()/B.max():.3f}")
    print(f"  归一化互相关 NCC = {ncc(A, B):.4f}")

    print("\n--- 厚度趋势对比（ADF 均值）---")
    for t in (4.0, 8.0, 16.0):
        c = ss.zone_axis_cell(_fcc("Au", 4.1713), (1, 0, 0),
                              target_xy=fov, target_thickness=t * 10)
        r = ss.stem_scan(c, opt, sampling=samp, geometry=geom, slice_thickness=slice_a,
                         phonons=ss.PhononConfig(n_configs=4, sigma_override=SIGMA_AU),
                         parallel=_par()).images["ADF"]
        lz_ = c.cell_lengths()[2]
        nz_ = int(round(lz_ / 4.1713))
        b = run_abtem(4.1713, nx, ny, nz_, samp, slice_a, SIGMA_AU, fov, step,
                      abtem.AnnularDetector(inner=INNER, outer=OUTER))
        print(f"  t={t:5.1f} nm: stem_sim {r.mean():.4e}   abTEM {np.mean(b):.4e}   "
              f"比 {r.mean()/np.mean(b):.3f}")

    print("\n--- Z 衬度指数（同晶格常数 4.0 Å，8 组态）---")
    els = {"Al": 13, "Si": 14, "Fe": 26, "Cu": 29, "Pd": 46, "Au": 79}
    for nm, engine in (("stem_sim", "mine"), ("abTEM", "abtem")):
        zs, vs = [], []
        for el, z in els.items():
            if engine == "mine":
                c = ss.zone_axis_cell(_fcc(el, 4.0), (1, 0, 0), target_xy=16.0,
                                      target_thickness=40.0)
                o = ss.StemOptics(voltage_kv=VOLT, cs=CS, defocus=DF, probe_semiangle=ALPHA,
                                  detector_inner=60.0, detector_outer=95.0)
                g = ss.ScanGeometry(16.0, 16.0, 17, 17)
                r = ss.stem_scan(c, o, sampling=0.1, geometry=g, slice_thickness=2.0,
                                 phonons=ss.PhononConfig(n_configs=8,
                                                         sigma_override=0.09),
                                 parallel=_par()).images["ADF"]
            else:
                from ase import Atoms
                a = 4.0
                base = np.array([[0, 0, 0], [0, a / 2, a / 2],
                                 [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
                pos = np.vstack([base + np.array([i, j, k]) * a
                                 for i in range(4) for j in range(4) for k in range(10)])
                atoms = Atoms(el * len(pos), positions=pos,
                              cell=np.diag([16.0, 16.0, 40.0]), pbc=True)
                fr = abtem.FrozenPhonons(atoms, 8, sigmas=0.09, seed=20260915)
                pot = abtem.Potential(fr, sampling=0.1, slice_thickness=2.0,
                                      parametrization="kirkland", projection="infinite")
                pr = abtem.Probe(energy=VOLT * 1e3, semiangle_cutoff=ALPHA,
                                 defocus=DF, Cs=CS, rolloff=0.0)
                sc = abtem.GridScan(start=(0, 0), end=(16.0, 16.0), sampling=1.0)
                r = np.asarray(pr.scan(
                    sc, abtem.AnnularDetector(inner=60.0, outer=95.0), pot, pbar=False
                ).array, dtype=float)
                if r.ndim == 3:
                    r = r.mean(0)
            v = float(r[r >= np.percentile(r, 95)].mean())
            vs.append(v)
            zs.append(z)
        alpha = float(np.polyfit(np.log(zs), np.log(vs), 1)[0])
        print(f"  {nm:9s}: α = {alpha:.2f}")
    print("\n完成。")


if __name__ == "__main__":
    main()
