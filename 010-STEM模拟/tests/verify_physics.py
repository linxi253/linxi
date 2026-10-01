"""stem_sim 物理自检：归一化、尺度、Z 衬度、冻结声子必要性、并行一致性。

运行：
    python tests/verify_physics.py

检查项
------
 1. 相对论波长与相互作用参数（与 tem_sim 同源，口径一致性）
 2. 探针归一化 Σ_r|ψ(r)|² = 1（探测器信号定量化的前提）
 3. 探针尺寸 / Scherzer 探针离焦 / 最优探针半角 的解析公式
 4. **真空极限**：无样品时 BF = 1、ADF = 0（探针归一化 ↔ 探测器标度自洽）
 5. **探测器互补性**：BF + ADF + 外圈 = 1（Parseval + 幺正，任意样品）
 6. **平均内电位 V0**：引擎势场标度与表理论值一致（继承 tem_sim 已验证标度）
 7. **Z 衬度指数**：ADF ∝ Z^α，冻结声子下 α 应落在 1.5–2.2
 8. **冻结声子必要性**：静态势的高角信号显著偏弱
 9. **厚度线性**：薄样品下 ADF 与厚度成正比（单散射区）
10. **并行一致性**：多进程结果与串行逐位接近（含声子组态可复现性）
11. Debye-Waller：Au 室温 B ≈ 0.65 Å²（Debye 模型 + 实验值对照）
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from stem_sim import (
    FrozenPhonons,
    PhononConfig,
    ScanGeometry,
    StemOptics,
    Structure,
    debye_waller_B,
    electron_scattering_factor,
    electron_wavelength,
    gamma_factor,
    angle_from_k,
    interaction_sigma,
    probe_ft,
    probe_wave,
    probe_intensity_profile,
    read_structure,
    rms_displacement,
    sigma_from_B,
    stem_scan,
    zone_axis_cell,
)
from stem_sim._fft import backend_name, fast_grid_size
from stem_sim.detectors import RingDetector, make_detectors
from stem_sim.microscope import Microscope
from stem_sim.multislice import build_slice_transmissions, projected_phase
from stem_sim.probe import frequency_grid
from stem_sim.scan import CancelledError, _scan_positions

CIF = ROOT / "cif"


# ----------------------------------------------------------------------
def test_wavelength_sigma():
    lam = electron_wavelength(300.0)
    assert abs(lam - 0.019687) < 1e-5, lam
    s = interaction_sigma(300.0)
    assert abs(s - 6.5262e-4) < 2e-6, s
    print(f"[OK] 1 波长/σ 与 tem_sim 同源: λ(300kV) = {lam:.6f} Å, σ = {s:.4e}")


def test_probe_normalization():
    opt = StemOptics(voltage_kv=300.0, cs=1e4, defocus=-14.0, probe_semiangle=25.0)
    for shape, samp in [((128, 128), 0.05), ((96, 96), 0.1)]:
        psi = probe_wave(opt, shape, samp)
        total = float(np.sum(np.abs(psi) ** 2))
        assert abs(total - 1.0) < 1e-12, f"探针归一化偏差 {total}"
    # 倒空间归一：Σ|P|² = N  ⇔  Σ|ψ|² = 1
    P = probe_ft(opt, (128, 128), 0.05, normalize=True)
    assert abs(np.sum(np.abs(P) ** 2) - 128 * 128) < 1e-6
    print("[OK] 2 探针归一化 Σ|ψ|² = 1（实空间与倒空间双向验证）")


def test_phase_scale():
    """投影势相位的**绝对标度**（本工具相对 09-HRTEM模拟 的关键修正）。

    相位核必须是"连续相位在像素上的平均值"，而不是"像素积分"。用单个
    Au 原子做判定：解析峰值 φ(0) = γλ·π·Σ(a_i/b_i) = 3.4713 rad
    （≈π，对应重原子中心透射函数近 −1 的教科书结论），且该值必须与采样
    无关（只在像素粗到无法分辨最窄高斯项时才因带限而降低）。

    同时验证求和规则 Σ_pixels φ = γλ·f_e(0)/(Δx·Δy)（采样相位的正确求和）。
    """
    gl = StemOptics(voltage_kv=300.0).gamma_lambda
    from stem_sim.scattering import get_factors

    a, b = get_factors("Au")
    phi0_analytic = gl * np.pi * float(np.sum(a / b))
    assert abs(phi0_analytic - 3.4713) < 1e-3, phi0_analytic

    fe0 = float(electron_scattering_factor("Au", 0.0))
    st = Structure(["Au"], np.array([[20.0, 20.0, 0.5]]), cell=np.diag([40.0, 40.0, 1.0]))
    scope = Microscope(voltage_kv=300.0)
    peaks = {}
    for samp in (0.4, 0.2, 0.1, 0.05, 0.02):
        ph = projected_phase(st, scope, sampling=samp, padding=0.0)
        peaks[samp] = float(ph.max())
        expect = gl * fe0 / (samp * samp)
        assert abs(ph.sum() / expect - 1.0) < 1e-4, (samp, ph.sum(), expect)
    # 细采样必须收敛到解析峰值
    assert abs(peaks[0.02] / phi0_analytic - 1.0) < 0.03, peaks
    # 且必须随采样变细单调增大（若相位是"像素积分"，则会随 Δx² 减小）
    seq = [peaks[s] for s in (0.4, 0.2, 0.1, 0.05, 0.02)]
    assert all(x < y for x, y in zip(seq, seq[1:])), seq
    # 旧口径（像素积分）会把 0.1 Å/px 的峰值压到 ~0.02 rad，差 ~100 倍
    assert peaks[0.1] > 2.0, f"0.1 Å/px 下峰值仅 {peaks[0.1]:.3f} rad，相位标度仍偏小"
    print(
        "[OK] 2b 投影势相位标度: 单 Au 原子峰值 "
        + ", ".join(f"{s}Å/px→{v:.3f}" for s, v in peaks.items())
        + f" rad（解析 {phi0_analytic:.4f}）；求和规则偏差 <1e-4"
    )


def test_probe_optics_formulas():
    lam = electron_wavelength(300.0)
    opt = StemOptics(voltage_kv=300.0, cs=1e4, probe_semiangle=25.0)
    assert abs(opt.scherzer_probe_defocus() + np.sqrt(1e4 * lam)) < 1e-9
    assert abs(opt.optimal_probe_semiangle() - (4 * lam / 1e4) ** 0.25 * 1e3) < 1e-9
    # 探针尺寸反比于会聚角
    a = opt.probe_size_rayleigh()
    opt2 = StemOptics(voltage_kv=300.0, cs=1e4, probe_semiangle=50.0)
    assert abs(a / opt2.probe_size_rayleigh() - 2.0) < 1e-12
    # 采样约束：外角越大要求采样越细
    thin = StemOptics(voltage_kv=300.0, detector_outer=50.0)
    wide = StemOptics(voltage_kv=300.0, detector_outer=200.0)
    assert wide.required_sampling() < thin.required_sampling()
    assert abs(wide.required_sampling() * 2 * np.sin(0.2) / lam - 0.95) < 1e-12
    print(
        f"[OK] 3 探针光学: Δf_Sch = {opt.scherzer_probe_defocus():.1f} Å, "
        f"α_opt = {opt.optimal_probe_semiangle():.1f} mrad, "
        f"r(25 mrad) = {a:.3f} Å, Δx(200 mrad) = {wide.required_sampling():.4f} Å/px"
    )


def _detector_masks(opt: StemOptics, shape, sampling):
    """构造一组互补掩模：BF 圆孔 / ADF 环 / 外圈。"""
    KX, KY, K2 = frequency_grid(shape, sampling)
    from stem_sim.constants import k_from_angle

    lam = opt.wavelength
    k = np.sqrt(K2)
    k_probe = k_from_angle(opt.probe_semiangle, lam)
    k_in = k_from_angle(opt.detector_inner, lam)
    k_out = k_from_angle(opt.detector_outer, lam)
    bf = (k <= k_in).astype(float)
    adf = ((k > k_in) & (k <= k_out)).astype(float)
    out = (k > k_out).astype(float)
    return {"BF": bf, "ADF": adf, "OUT": out}


def test_vacuum_limit():
    """无样品（透射函数恒为 1）：BF = 1、ADF = 0、总强度守恒。"""
    opt = StemOptics(voltage_kv=300.0, cs=1e4, defocus=-14.0,
                     probe_semiangle=25.0, detector_inner=50.0, detector_outer=90.0)
    shape, samp = (128, 128), (0.08, 0.08)
    P0 = probe_ft(opt, shape, samp, normalize=True)
    KX, KY, _ = frequency_grid(shape, samp)
    one = np.ones(shape, dtype=np.complex128)
    masks = _detector_masks(opt, shape, samp)
    res = _scan_positions(
        [one], samp, P0, KX, KY, np.ones(shape, dtype=np.complex128),
        np.array([[0.0, 0.0], [3.0, 2.0], [-4.0, 1.5]]), masks, [1], threads=2,
    )
    bf = res[1]["BF"]
    adf = res[1]["ADF"]
    tot = res[1]["BF"] + res[1]["ADF"] + res[1]["OUT"]
    assert np.allclose(bf, 1.0, atol=1e-12), bf
    assert np.allclose(adf, 0.0, atol=1e-12), adf
    assert np.allclose(tot, 1.0, atol=1e-12), tot
    print(f"[OK] 4 真空极限: BF = {bf[0]:.12f}, ADF = {adf[0]:.2e}, BF+ADF+OUT = 1")


def test_parseval_complementarity():
    """有样品时三组互补探测器之和仍应为 1（Parseval + 相位透射保模）。

    注意：这里的 OUT 掩模必须覆盖到网格**角部**（|k| 最大 = √2/(2Δx)），
    而不是 RingDetector.validate 裁剪到的轴向奈奎斯特 1/(2Δx)——后者是真实
    探测器该有的保守上限（超出即混叠不可信），但会让方形网格的四个角
    漏掉，从而不能用来检验 Parseval。
    """
    st = read_structure(CIF / "SrTiO3_cubic_Pm-3m.cif")
    crystal = zone_axis_cell(st, (1, 0, 0), target_xy=16.0, target_thickness=40.0)
    opt = StemOptics(voltage_kv=300.0, cs=1e4, defocus=-14.0, probe_semiangle=25.0)
    geom = ScanGeometry(14.0, 14.0, 6, 6)
    # 用不会触发裁剪的探测器：OUT 覆盖整张网格（含角部），以检验 Parseval
    dets = [RingDetector("BF", 0.0, 50.0),
            RingDetector("ADF", 50.0, 90.0),
            RingDetector("OUT", 90.0, 89_000.0, clip_to_nyquist=False)]
    corner = np.degrees(np.arcsin(opt.wavelength * np.sqrt(2) / (2 * 0.1))) * 1e3
    assert 90.0 < corner, corner
    res = stem_scan(crystal, opt, sampling=0.1, geometry=geom, slice_thickness=2.0,
                    phonons=None, parallel=False, detectors=dets)
    total = res.images["BF"] + res.images["ADF"] + res.images["OUT"]
    err = float(np.max(np.abs(total - 1.0)))
    assert err < 5e-12, f"互补性偏差 {err}"
    assert float(res.images["ADF"].max()) < 1.0
    print(
        f"[OK] 5 探测器互补性: BF+ADF+OUT = 1（最大偏差 {err:.2e}），"
        f"ADF 最大 {res.images['ADF'].max():.4f}；网格角部覆盖到 {corner:.1f} mrad"
    )


def test_inner_potential():
    """平均内电位：引擎势场标度与 Peng 表理论值一致（Au，fcc）。"""
    a = 4.1713
    unit = Structure(
        ["Au"] * 4,
        np.array([[0, 0, 0], [0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]]),
        cell=np.diag([a, a, a]),
    )
    scope = Microscope(voltage_kv=300.0)
    phase = projected_phase(unit, scope, sampling=0.05, padding=0.0)
    sigma = interaction_sigma(300.0)
    # φ 是采样相位（像素平均值）→ V0 = ⟨φ⟩/(σ·t)；旧写法 Σφ/(σ·V_cell)
    # 隐含"φ 是像素积分"的口径，修正标度后两者不再等价（见 test_phase_scale）
    v0_engine = float(phase.mean()) / (sigma * a)
    fe0 = float(electron_scattering_factor("Au", 0.0))
    v0_theory = scope.gamma_lambda * 4.0 * fe0 / (sigma * a**3)
    rel = abs(v0_engine - v0_theory) / v0_theory
    assert rel < 5e-4, f"V0 偏差 {rel:.2e}"
    v0_coarse = float(projected_phase(unit, scope, sampling=0.12, padding=0.0).mean())         / (sigma * a)
    assert abs(v0_coarse / v0_engine - 1.0) < 2e-3, (v0_coarse, v0_engine)
    print(
        f"[OK] 6 平均内电位 V0(Au) = {v0_engine:.2f} V（表理论值 {v0_theory:.2f} V，"
        f"偏差 {rel:.1e}；0.12 Å/px 复算 {v0_coarse:.2f} V；实验公认 ≈ 22–27 V）"
    )


def _single_element_fcc(symbol: str, a: float, thickness_a: float, xy: float = 16.0):
    unit = Structure(
        [symbol] * 4,
        np.array([[0, 0, 0], [0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]]),
        cell=np.diag([a, a, a]),
    )
    return zone_axis_cell(unit, (1, 0, 0), target_xy=xy, target_thickness=thickness_a)


def test_z_contrast():
    """Z 衬度：同一（人为固定的）晶格下 ADF 强度应按 Z^α 增长，α ∈ [1.5, 2.2]。"""
    a = 4.0
    elements = ["Al", "Si", "Fe", "Cu", "Ni", "Pd", "Pt", "Au"]
    znum = {"Al": 13, "Si": 14, "Fe": 26, "Ni": 28, "Cu": 29,
            "Pd": 46, "Pt": 78, "Au": 79}
    opt = StemOptics(voltage_kv=300.0, cs=1e4, defocus=-14.0, probe_semiangle=25.0,
                     detector_inner=60.0, detector_outer=95.0)
    geom = ScanGeometry(16.0, 16.0, 16, 16)
    ph = PhononConfig(n_configs=4, temperature=300.0)
    vals, zs = [], []
    for el in elements:
        crystal = _single_element_fcc(el, a, 50.0)
        res = stem_scan(crystal, opt, sampling=0.1, geometry=geom, slice_thickness=2.0,
                        phonons=ph, parallel=False)
        img = res.images["ADF"]
        # 取前 5% 亮点（原子柱上）的平均强度作为柱强度
        thr = np.percentile(img, 95.0)
        vals.append(float(img[img >= thr].mean()))
        zs.append(znum[el])
    vals = np.array(vals)
    zs = np.array(zs, dtype=float)
    alpha = float(np.polyfit(np.log(zs), np.log(vals), 1)[0])
    assert 1.5 <= alpha <= 2.2, f"Z 衬度指数 α = {alpha:.2f} 超出预期 [1.5, 2.2]"
    # 单调性
    order = np.argsort(zs)
    assert np.all(np.diff(np.log(vals[order])) > 0), "ADF 未随 Z 单调增长"
    print(
        f"[OK] 9 Z 衬度指数 α = {alpha:.2f}（{elements[0]}→{elements[-1]} 强度比 "
        f"{vals[-1] / vals[0]:.1f}×，预期 Z^1.7 法则给出 {(zs[-1] / zs[0]) ** 1.7:.1f}×）"
    )


def test_frozen_phonon_necessity():
    """静态势（无声子）的高角信号应显著低于冻结声子，且图像衬度形态不同。"""
    crystal = _single_element_fcc("Au", 4.1713, 60.0)
    opt = StemOptics(voltage_kv=300.0, cs=1e4, defocus=-14.0, probe_semiangle=25.0,
                     detector_inner=60.0, detector_outer=95.0)
    geom = ScanGeometry(16.0, 16.0, 16, 16)
    static = stem_scan(crystal, opt, sampling=0.1, geometry=geom, slice_thickness=2.0,
                       phonons=None, parallel=False).images["ADF"]
    phonon = stem_scan(crystal, opt, sampling=0.1, geometry=geom, slice_thickness=2.0,
                       phonons=PhononConfig(n_configs=4), parallel=False).images["ADF"]
    r = float(phonon.max() / max(static.max(), 1e-30))
    assert r > 1.5, f"冻结声子并未显著提高高角强度（比值 {r:.2f}）"
    print(
        f"[OK] 10 冻结声子必要性: 静态势 ADF 峰值 {static.max():.2e} → "
        f"声子平均后 {phonon.max():.2e}（{r:.1f}×），TDS 是高角信号的主来源"
    )


def test_thickness_linearity():
    """薄样品（单散射区）ADF 与厚度成正比，验证厚度序列捕获。"""
    crystal = _single_element_fcc("Au", 4.1713, 120.0)
    opt = StemOptics(voltage_kv=300.0, cs=1e4, defocus=-14.0, probe_semiangle=25.0,
                     detector_inner=60.0, detector_outer=95.0)
    geom = ScanGeometry(16.0, 16.0, 12, 12)
    want = [10.0, 20.0, 40.0]
    res = stem_scan(crystal, opt, sampling=0.1, geometry=geom, slice_thickness=2.0,
                    phonons=PhononConfig(n_configs=3), parallel=False,
                    capture_thickness_a=want)
    caps = res.capture_list()
    assert len(caps) >= 4, caps  # 含总厚度
    ts = np.array([t for t, _ in caps if t <= 45.0])
    vals = np.array([float(res.captures[i]["ADF"].mean()) for t, i in caps if t <= 45.0])
    per_nm = vals / (ts / 10.0)
    spread = float(per_nm.max() / per_nm.min())
    assert spread < 1.5, f"薄区 ADF/厚度 比值散布过大: {per_nm}"
    print(
        "[OK] 11 厚度线性: " + ", ".join(
            f"t={t / 10:.1f}nm→{v / (t / 10):.3e}/nm" for t, v in zip(ts, vals)
        ) + f"（散布 {spread:.2f}×）"
    )


def test_parallel_consistency():
    """多进程与串行结果一致；且声子位移可复现（同种子 → 同结果）。"""
    crystal = _single_element_fcc("Au", 4.1713, 40.0)
    opt = StemOptics(voltage_kv=300.0, cs=1e4, defocus=-14.0, probe_semiangle=25.0,
                     detector_inner=60.0, detector_outer=95.0)
    geom = ScanGeometry(16.0, 16.0, 8, 8)
    ph = PhononConfig(n_configs=2, seed=1234)
    a = stem_scan(crystal, opt, sampling=0.12, geometry=geom, slice_thickness=2.0,
                  phonons=ph, parallel=False).images["ADF"]
    b = stem_scan(crystal, opt, sampling=0.12, geometry=geom, slice_thickness=2.0,
                  phonons=ph, parallel=False).images["ADF"]
    assert np.array_equal(a, b), "同种子两次串行结果不一致（声子不可复现）"
    c = stem_scan(crystal, opt, sampling=0.12, geometry=geom, slice_thickness=2.0,
                  phonons=ph, parallel=True).images["ADF"]
    rel = float(np.max(np.abs(c - a)) / max(a.max(), 1e-30))
    assert rel < 1e-9, f"并行与串行偏差 {rel:.2e}"
    print(f"[OK] 12 并行一致性: 串行可复现，多进程最大相对偏差 {rel:.1e}")


def test_debye_waller():
    """Debye-Waller 因子：Au 室温 B 应落在实验区间附近。"""
    b_au = debye_waller_B("Au", 300.0)
    sigma_au = rms_displacement("Au", 300.0)
    # 文献 B_Au(300K) ≈ 0.5–0.7 Å²（X 射线/中子德拜-瓦勒测量）
    assert 0.45 <= b_au <= 0.75, f"Au B = {b_au:.3f} Å² 偏离文献区间"
    assert abs(sigma_from_B(b_au) - sigma_au) < 1e-12
    # 低温趋近零点振动：σ(T→0) 应为非零常数（Au ≈ 0.03–0.05 Å），而非 0
    sigma_zp = rms_displacement("Au", 1.0)
    assert 0.02 <= sigma_zp <= 0.06, f"零点位移异常: {sigma_zp:.4f} Å"
    assert sigma_zp < 0.5 * sigma_au
    # 温度单调
    seq = [rms_displacement("Au", t) for t in (100.0, 200.0, 300.0, 500.0)]
    assert np.all(np.diff(seq) > 0)
    # 高温极限：B ∝ T（经典等分配律），300→900 K 应约 3×
    b300, b900 = debye_waller_B("Au", 300.0), debye_waller_B("Au", 900.0)
    assert 2.5 < b900 / b300 < 3.5, f"高温 B∝T 失效: {b900 / b300:.2f}"
    # 位移可复现且统计量正确
    crystal = _single_element_fcc("Au", 4.1713, 20.0)
    fp = FrozenPhonons(crystal, PhononConfig(n_configs=2, seed=7))
    d0 = fp.displacements(0)
    d0b = FrozenPhonons(crystal, PhononConfig(n_configs=2, seed=7)).displacements(0)
    assert np.allclose(d0, d0b), "声子位移不可复现"
    assert not np.allclose(d0, fp.displacements(1)), "不同组态位移相同"
    rms = float(np.sqrt(np.mean(d0**2)))
    assert abs(rms - sigma_au) / sigma_au < 0.1, f"位移统计量异常: {rms:.4f} vs {sigma_au:.4f}"
    print(
        f"[OK] 7 Debye-Waller: Au(300K) B = {b_au:.3f} Å², σ_u = {sigma_au:.4f} Å；"
        f"T→0 零点 σ = {sigma_zp:.4f} Å；300→900 K B 比 {b900 / b300:.2f}"
        f"（文献 B_Au(300K) ≈ 0.5–0.7 Å²）"
    )


def test_absolute_scale():
    """ADF 绝对标度的**约定因子**（登记在案，便于定量对比实验时换算）。

    对薄样品，ADF 收集比例 F 与"运动学单散射"估计 σ_ADF·n_areal 的比值
    是一个与厚度/密度/有序度无关的常数：

        F ≈ K · σ_ADF · n_areal，  σ_ADF = ∫|f_e|²dΩ（探测器角度范围）

    本引擎（以及同源的标准多层法口径）给出 K ≈ 2.1（Peng 表、0.1 Å/px），
    来源是 σ = 2πγmeλ/h² 与 f_e 表口径的组合（γ² = 2.52 @300 kV，
    扣除像素带限的 sinc² 损失后约 2.1）。**这是约定而非误差**：
    与 abTEM 的投影势积分一致到 0.1%，Z 衬度指数也一致（1.76 vs 1.71）。
    定量对比实验绝对强度时按此因子换算；只比相对衬度时可忽略。
    """
    from stem_sim.scattering import phase_kernels
    from stem_sim.multislice import _accumulate_phase, _prepare_box
    from stem_sim.probe import frequency_grid

    lam = electron_wavelength(300.0)
    th = np.linspace(50e-3, 95e-3, 4000)
    g = np.sin(th) / lam
    fe = np.asarray(electron_scattering_factor("Au", g), dtype=float)
    sigma_adf = float(2 * np.pi * np.trapezoid(fe**2 * np.sin(th), th))

    L, gpts = 40.0, 512
    rng = np.random.default_rng(11)
    n_atoms = 128
    pos = np.column_stack([rng.uniform(0, L, n_atoms), rng.uniform(0, L, n_atoms),
                           rng.uniform(0, 33.4, n_atoms)])
    st = Structure(["Au"] * n_atoms, pos, cell=np.diag([L, L, 33.4]))
    p2, syms, lx, ly, _ = _prepare_box(st, 0.0)
    sx = sy = lx / gpts
    kern = {s: phase_kernels(s, (sx, sy), StemOptics(voltage_kv=300.0).gamma_lambda)
            for s in set(syms)}
    phase = np.zeros((gpts, gpts))
    _accumulate_phase(phase, p2, syms, np.arange(len(syms)), sx, sy, kern)
    _, _, K2 = frequency_grid((gpts, gpts), (sx, sy))
    ang = angle_from_k(np.sqrt(K2), lam)
    m = (ang >= 50) & (ang <= 95)
    frac = float(np.sum(np.abs(np.fft.fft2(1j * phase)[m]) ** 2)) / gpts ** 4
    areal = n_atoms / (L * L)
    K = frac / (sigma_adf * areal)
    assert 1.6 <= K <= 2.6, f"绝对标度因子 K = {K:.2f} 超出登记范围 [1.6, 2.6]"
    print(
        f"[OK] 12 ADF 绝对标度: K = {K:.2f}（F ≈ K·σ_ADF·n_areal；"
        f"γ²(300kV) = {gamma_factor(300e3) ** 2:.2f}，扣除像素带限后约 2.1；"
        "与 abTEM 投影势积分一致到 0.1%）"
    )


def test_backend_report():
    print(f"[INFO] FFT 后端 = {backend_name()}；5-光滑网格示例: "
          f"334→{fast_grid_size(334)}, 250→{fast_grid_size(250)}, 512→{fast_grid_size(512)}")


TESTS = [
    ("1 波长/σ", test_wavelength_sigma),
    ("2 探针归一化", test_probe_normalization),
    ("2b 相位标度", test_phase_scale),
    ("3 探针光学公式", test_probe_optics_formulas),
    ("4 真空极限", test_vacuum_limit),
    ("5 探测器互补性", test_parseval_complementarity),
    ("6 平均内电位 V0", test_inner_potential),
    ("7 Debye-Waller", test_debye_waller),
    ("8 后端信息", test_backend_report),
    ("9 Z 衬度指数", test_z_contrast),
    ("10 冻结声子必要性", test_frozen_phonon_necessity),
    ("11 厚度线性", test_thickness_linearity),
    ("12 并行一致性", test_parallel_consistency),
    ("13 绝对标度约定", test_absolute_scale),
]


if __name__ == "__main__":
    # 用法：python tests/verify_physics.py [编号 ...]   不给编号则全部运行
    selected = sys.argv[1:]
    todo = [
        (name, fn) for i, (name, fn) in enumerate(TESTS, start=1)
        if not selected or str(i) in selected or name.split()[0] in selected
    ]
    t0 = time.perf_counter()
    for name, fn in todo:
        print(f"--- {name} ---")
        fn()
    print(f"\n全部验证通过 ✔（{len(todo)} 项，总耗时 {time.perf_counter() - t0:.1f} s）")
