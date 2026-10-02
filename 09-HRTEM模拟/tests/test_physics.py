"""pytest 封装：verify_physics 中的全部物理自检。

运行：python -m pytest tests/ -v
（无外部数据依赖；E 盘相关的 0820 复现请运行 validate_reproduce.py）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_physics as vp  # noqa: E402


def test_wavelength():
    vp.test_wavelength()


def test_sigma():
    vp.test_sigma()


def test_scattering_factor():
    vp.test_scattering_factor()


def test_scattering_scale():
    vp.test_scattering_scale()


def test_phase_kernel_fft():
    vp.test_phase_kernel_fft()


def test_phase_scale():
    """绝对标度回归：核峰值收敛到解析峰值 γλ·π·Σ(a_i/b_i)（去掉 /(sx·sy) 必失败）。"""
    vp.test_phase_scale()


def test_multislice_unitarity():
    vp.test_multislice_unitarity()


def test_scherzer():
    vp.test_scherzer()


def test_zone_axis_density():
    vp.test_zone_axis_density()


def test_mean_inner_potential():
    vp.test_mean_inner_potential()


def test_accumulate_phase_matches_reference():
    """切片累加实现与 np.add.at 参考实现逐元素一致（含回绕边界）。"""
    import numpy as np

    from tem_sim.multislice import _accumulate_phase
    from tem_sim.scattering import phase_kernels

    from tem_sim import Microscope

    ny, nx, s = 32, 32, 0.1
    gl = Microscope(voltage_kv=300.0).gamma_lambda
    kerns = phase_kernels("Au", (s, s), gl, table="gauss3")  # gauss3 无 floor，核精确
    kc = {"Au": kerns}
    # 覆盖角点/边界的回绕路径
    pos = np.array(
        [[0.0, 0.0, 0.0], [0.05, 1.55, 0.5], [1.55, 0.05, 1.0], [1.6, 1.6, 1.5]],
        dtype=float,
    )
    ids = np.arange(len(pos))

    def reference(phase):
        ny_, nx_ = phase.shape
        out = phase.copy()
        for j in ids:
            jx = int(np.floor(pos[j, 0] / s + 0.5)) % nx_
            iy = int(np.floor(pos[j, 1] / s + 0.5)) % ny_
            for kern in kerns:
                kh, kw = kern.shape
                di = (iy + np.arange(-(kh // 2), kh - kh // 2)) % ny_
                dj = (jx + np.arange(-(kw // 2), kw - kw // 2)) % nx_
                np.add.at(out, np.ix_(di, dj), kern)
        return out

    new = np.zeros((ny, nx))
    _accumulate_phase(new, pos, ["Au"] * len(pos), ids, s, s, kc)
    ref = reference(np.zeros((ny, nx)))
    assert np.allclose(new, ref, rtol=0, atol=1e-15), (
        f"切片累加与参考实现差异 max={np.abs(new - ref).max():.2e}"
    )
    # 相对截断 cutoff=12（e^-12 ≈ 6e-6 边界幅度）带来 ~1e-6 量级的
    # 设计性总积分亏缺，与累加实现无关。核为像素平均值口径：
    # Σ_pixels K = γλ·f_e(0)/(s·s)（旧"像素积分"口径少除 s²，此断言必失败）
    expect = gl * float(vp.electron_scattering_factor("Au", 0.0, "gauss3")) / (s * s)
    assert abs(new.sum() - expect * len(pos)) / (expect * len(pos)) < 1e-5
