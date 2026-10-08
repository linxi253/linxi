"""
tem_sim 物理正确性验证（重构自 SimulaTEM 后的自检）。

检查项：
  1. 相对论电子波长（300 kV ≈ 0.01969 Å）
  2. 相互作用参数 σ（300 kV ≈ 6.5262e-4 rad/(V·Å)）
  3. 散射因子 f_e(0) = Σ a_i
  4. 单原子相位核的傅里叶变换 ≈ γλ·f_e(g)/(Δx·Δy)（解析关系数值验证）
  4b. 投影势相位绝对标度：单 Au 原子解析峰值 γλ·π·Σ(a_i/b_i) = 3.4713 rad
      （与采样无关；2026-10 相位标度修正的回归判据）
  5. Multislice 波函数的幺正性（|ψ|² 总强度守恒）
  6. Scherzer 离焦与点分辨率（Williams-Carter 公式）
  7. [110] 带轴重构保持原子密度（防止含负分量的新晶胞漏原子）
  8. 散射因子绝对标度（氢第一性原理 + Mott-Bethe 双锚点，Peng 表）
  9. 平均内电位 V0（fcc Al：引擎管线 vs 表理论值）
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from tem_sim import (
    Microscope,
    Structure,
    electron_wavelength,
    interaction_sigma,
    electron_scattering_factor,
    multislice,
    projected_phase,
)
from tem_sim.scattering import get_factors, phase_kernels
from tem_sim.structure import zone_axis_cell


def test_wavelength():
    lam300 = electron_wavelength(300.0)
    assert abs(lam300 - 0.019687) < 1e-5, f"300 kV 波长错误: {lam300}"
    lam80 = electron_wavelength(80.0)
    assert abs(lam80 - 0.041761) < 5e-5, f"80 kV 波长错误: {lam80}"
    print(f"[OK] 波长: 300 kV → {lam300:.6f} Å; 80 kV → {lam80:.6f} Å")


def test_sigma():
    s300 = interaction_sigma(300.0)
    # σ(300 kV) ≈ 6.527×10⁻⁴ rad/(V·Å)（= 6.527×10⁶ rad/(V·m)）
    assert abs(s300 - 6.5262e-4) < 2e-6, f"σ(300kV) 错误: {s300}"
    print(f"[OK] 相互作用参数 σ(300 kV) = {s300:.4e} rad/(V·Å)")


def test_scattering_factor():
    a, b = get_factors("Au")
    fe0 = electron_scattering_factor("Au", 0.0)
    assert abs(fe0 - a.sum()) < 1e-12
    fe1 = electron_scattering_factor("Au", 1.0)
    expect = sum(ai * np.exp(-bi * 1.0) for ai, bi in zip(a, b))
    assert abs(fe1 - expect) < 1e-12
    print(f"[OK] 散射因子: f_Au(0) = {fe0:.4f} Å = Σa_i")


def test_phase_kernel_fft():
    """相位核 FT 应等于 γλ·f_e(g)（投影势-散射因子解析关系）。"""
    scope = Microscope(voltage_kv=300.0)
    gl = scope.gamma_lambda
    sampling = 0.02  # 精细采样
    kernels = phase_kernels("Au", sampling, gl, cutoff=14.0)
    # 把各高斯核放进大网格中心再 FFT
    n = 1024
    grid = np.zeros((n, n))
    for k in kernels:
        kh, kw = k.shape
        i0, j0 = n // 2 - kh // 2, n // 2 - kw // 2
        grid[i0 : i0 + kh, j0 : j0 + kw] += k
    # erf 核为相位在像素上的平均值（rad），等于像素积分除以 (sx·sy)，
    # 其 DFT 相应等于连续傅里叶变换乘同一 1/(sx·sy) 因子
    ft = np.fft.fftshift(np.abs(np.fft.fft2(grid)))
    kx = np.fft.fftshift(np.fft.fftfreq(n, d=sampling))
    g = np.abs(kx)  # 沿中心行（ky = 0）
    row = ft[n // 2]
    # 像素平均核的 DFT 满足泊松求和模型：
    #   DFT ≈ γλ·f_e(g)·sinc(π·g·s)/(sx·sy) + Σ_{n≠0} 混叠项。
    # 混叠在 g ≪ 1/(2s)（奈奎斯特）处可忽略；Peng 表最窄 g 项频谱较宽，
    # g 接近奈奎斯特时折叠项可达百分之几，故比对限制在 g < 15 Å⁻¹
    # （奈奎斯特 25 Å⁻¹ 的 60%，HRTEM 物镜光阑也远低于此）。
    sinc_env = np.sinc(kx * sampling)
    pixel_area = sampling * sampling
    fe = electron_scattering_factor("Au", g) * gl * sinc_env / pixel_area
    m = (fe > 1e-5 * fe.max()) & (g < 15.0)
    rel = np.max(np.abs(row[m] - fe[m]) / np.abs(fe[m]))
    assert rel < 3e-3, f"相位核 FT 与 γλ f_e(g)/(sx·sy) 偏差过大: {rel:.3e}"
    print(f"[OK] 相位核 FT ≈ γλ·f_e(g)·sinc/(sx·sy)（最大相对偏差 {rel:.2e}）")


def test_phase_scale():
    """投影势相位的**绝对标度**（2026-10 相位标度修正的回归判据）。

    相位核必须是"连续相位在像素上的平均值"，而不是"像素积分"。用单个
    Au 原子做判定：解析峰值 φ(0) = γλ·π·Σ(a_i/b_i) = 3.4713 rad
    （≈π，对应重原子中心透射函数近 −1 的教科书结论），且该值必须与采样
    无关（只在像素粗到无法分辨最窄高斯项时才因带限而降低）。

    同时验证求和规则 Σ_pixels φ = γλ·f_e(0)/(Δx·Δy)（采样相位的正确求和）。
    本判据对绝对标度敏感：若把核的 /(sx·sy) 去掉（退回旧"像素积分"口径），
    峰值随 Δx² 缩小、求和规则差 s² 倍，本测试立即失败。
    （写法与 010-STEM模拟/tests/verify_physics.py 第 2b 项一致。）
    """
    gl = Microscope(voltage_kv=300.0).gamma_lambda
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
        "[OK] 4b 相位标度: 单 Au 原子峰值 "
        + ", ".join(f"{s}Å/px→{v:.3f}" for s, v in peaks.items())
        + f" rad（解析 {phi0_analytic:.4f}）；求和规则偏差 <1e-4"
    )


def test_multislice_unitarity():
    """自由传播与相位透射都保模，出射波总强度应 = 网格点数。"""
    atoms = Structure(
        ["Au", "Au", "Au", "Au"],
        np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 1.0],
                [0.0, 2.0, 2.0],
                [2.0, 2.0, 3.0],
            ]
        ),
        cell=None,
    )
    scope = Microscope(voltage_kv=300.0, cs=1e7)
    ew = multislice(atoms, scope, sampling=0.1, slice_thickness=1.0, padding=5.0, verbose=False)
    total = np.sum(np.abs(ew.array) ** 2)
    n_pix = ew.array.size
    rel = abs(total - n_pix) / n_pix
    assert rel < 1e-10, f"幺正性破坏: 相对偏差 {rel:.3e}"
    print(f"[OK] Multislice 幺正性: Σ|ψ|² = {total:.6f} / {n_pix}（偏差 {rel:.1e}）")


def test_scherzer():
    scope = Microscope(voltage_kv=300.0, cs=1e7)  # Cs = 1 mm
    df_sch = scope.scherzer_defocus()
    d_pt = scope.point_resolution()
    lam = scope.wavelength
    assert abs(df_sch + np.sqrt(1.5 * 1e7 * lam)) < 1e-9
    # 经典数值：300 kV, Cs=1mm → Δf_Sch = -sqrt(3/2·Cs·λ) ≈ -543 Å（abTEM 约定），
    # 点分辨率 d = 0.66·Cs^¼·λ^¾ ≈ 1.95 Å（Williams & Carter）
    assert abs(df_sch - (-543.4)) < 3.0, f"Scherzer: {df_sch}"
    assert abs(d_pt - 1.95) < 0.02, f"点分辨率: {d_pt}"
    print(f"[OK] Scherzer 离焦 = {df_sch:.1f} Å；点分辨率 = {d_pt:.3f} Å（300 kV, Cs = 1 mm）")


def test_zone_axis_density():
    """简单立方 [110] 新晶胞体积为原胞 2 倍，必须恰含 2 个原子。"""
    a = 4.0
    unit = Structure(["Fe"], np.array([[0.0, 0.0, 0.0]]), cell=np.eye(3) * a)
    zone = zone_axis_cell(unit, [1, 1, 0], target_xy=1.0, target_thickness=1.0)
    expected_lengths = np.array([a, np.sqrt(2.0) * a, np.sqrt(2.0) * a])
    assert len(zone) == 2, f"简单立方 [110] 应含 2 个原子，实际 {len(zone)}"
    assert np.allclose(zone.cell_lengths(), expected_lengths)
    rho0 = len(unit) / abs(np.linalg.det(unit.cell))
    rho1 = len(zone) / abs(np.linalg.det(zone.cell))
    assert abs(rho1 / rho0 - 1.0) < 1e-12, f"带轴重构密度错误: {rho1 / rho0}"
    print("[OK] [110] 带轴重构: 2 倍体积 / 2 个原子，原子密度守恒")


def test_scattering_scale():
    """散射因子绝对标度：氢原子第一性原理 f_e(0) 与 Mott-Bethe 形状双锚点。

    氢 1s 静电势可解析积分：∫φ d³r = (e/ε0)·a0²/2，
    f_e(0) = (2π m0 e / h²)·∫φ d³r = 0.5292 Å（Born 极限精确值）。
    """
    M0 = 9.1093837015e-31
    E_CH = 1.602176634e-19
    H_PLANCK = 6.62607015e-34
    EPS0 = 8.8541878128e-12
    A0_SI = 5.29177210903e-11
    A0 = A0_SI * 1e10  # Å

    fe0_ref = 2.0 * np.pi * M0 * E_CH / H_PLANCK**2 * (E_CH / EPS0 * A0_SI**2 / 2.0) * 1e10
    assert abs(fe0_ref - 0.5292) < 5e-4, f"第一性原理参考值异常: {fe0_ref}"
    fe0_tab = float(electron_scattering_factor("H", 0.0))
    assert abs(fe0_tab - fe0_ref) < 5e-4, (
        f"Peng 表 f_H(0) = {fe0_tab:.4f} Å 偏离第一性原理 {fe0_ref:.4f} Å"
    )

    # Mott-Bethe 形状锚点: f_H(g) = (m0e²/(2πh²ε0))·(1 − f_X(g))/g²，
    # 1s X 射线形状因子 f_X(g) = (1 + π²a0²g²)^-2
    const = M0 * E_CH**2 / (2.0 * np.pi * H_PLANCK**2 * EPS0) * 1e-10  # → Å·Å⁻¹ 换算
    fx1 = (1.0 + np.pi**2 * A0**2) ** -2
    fe1_ref = const * (1.0 - fx1)
    fe1_tab = float(electron_scattering_factor("H", 1.0))
    assert abs(fe1_tab - fe1_ref) / fe1_ref < 0.02, (
        f"Peng 表 f_H(1 Å⁻¹) = {fe1_tab:.4f} 偏离 Mott-Bethe {fe1_ref:.4f}"
    )

    a_legacy, _ = get_factors("H", "gauss3")
    print(
        f"[OK] 散射因子标度: f_H(0) = {fe0_tab:.4f} Å（第一性原理 {fe0_ref:.4f}），"
        f"f_H(1) 相对 Mott-Bethe 偏差 {abs(fe1_tab - fe1_ref) / fe1_ref:.1%}"
    )
    print(
        f"[INFO] legacy gauss3: f_H(0) = {a_legacy.sum():.4f} Å = "
        f"{a_legacy.sum() / fe0_ref:.2f}× 物理值（已知标度问题，见 README；"
        "仅限复现 SimulaTEM 口径使用）"
    )


def test_mean_inner_potential():
    """平均内电位 V0：fcc Al 引擎管线值必须与表理论值一致（采样相位口径）。

    φ 是采样相位（像素平均值）→ V0 = ⟨φ⟩/(σ·t)；旧写法 Σφ/(σ·V_cell)
    隐含"φ 是像素积分"的口径，修正标度后两者不再等价（见 test_phase_scale）。
    实验公认 V0(Al) ≈ 13.4 V；Peng 表预测 ≈ 17 V（参数化自身偏差量级），
    此处仅断言引擎与表自洽，实验对照打印供人工核查。
    """
    a = 4.0495
    unit = Structure(
        ["Al"] * 4,
        np.array(
            [[0, 0, 0], [0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]],
            dtype=float,
        ),
        cell=np.diag([a, a, a]),
    )
    scope = Microscope(voltage_kv=300.0)
    phase = projected_phase(unit, scope, sampling=0.05, padding=0.0)
    sigma = interaction_sigma(300.0)
    # φ 是采样相位（像素平均值）→ V0 = ⟨φ⟩/(σ·t)；旧写法 Σφ/(σ·V_cell)
    # 隐含"φ 是像素积分"的口径，修正标度后两者不再等价（见 test_phase_scale）
    v0_engine = float(phase.mean()) / (sigma * a)
    fe0_al = float(electron_scattering_factor("Al", 0.0))
    v0_theory = scope.gamma_lambda * 4.0 * fe0_al / (sigma * a**3)
    rel = abs(v0_engine - v0_theory) / v0_theory
    # phase_floor 截断（宽尾项按绝对相位阈值截边）带来 ≤1e-4 量级的设计性损失
    assert rel < 5e-4, f"引擎 V0 与表理论值偏差 {rel:.2e}"
    print(
        f"[OK] 平均内电位: 引擎 V0(Al) = {v0_engine:.2f} V = 表理论值"
        f"（实验公认 ≈ 13.4 V，Peng 参数化预测 ≈ {v0_theory:.1f} V）"
    )


if __name__ == "__main__":
    test_wavelength()
    test_sigma()
    test_scattering_factor()
    test_scattering_scale()
    test_phase_kernel_fft()
    test_phase_scale()
    test_multislice_unitarity()
    test_scherzer()
    test_zone_axis_density()
    test_mean_inner_potential()
    print("\n全部验证通过 ✔")
