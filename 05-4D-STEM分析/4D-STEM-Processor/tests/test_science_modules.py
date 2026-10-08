"""合成数据验证测试：4D-STEM 应变/取向/叠层三个科学模块。

这三个模块曾因物理错误被重写（CoM 被当作位移场、假模板、实/倒空
间混用）。本文件用正演-反演一致性测试锁定其科学正确性。
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.orientation_mapping import (STRUCTURE_PRESETS, build_zone_axis_bank,
                                      index_diffraction_pattern,
                                      make_structure, reciprocal_matrix,
                                      reflection_allowed, zone_axis_template)
from core.peak_pairs import _fold_angle_difference, peak_pairs_mapping
from core.ptychography import initialize_probe, run_ptychography
from core.strain_mapping import (detect_bragg_disks,
                                 fit_displacement_gradient,
                                 run_strain_mapping, strain_from_gradient)


def soft_disk_factory(det, radius, edge_w=1.5):
    """连续软边盘（在像素中心采样），模拟源尺寸+探测器 PSF。"""
    yy, xx = np.mgrid[0:det, 0:det]

    def make(y, x, inten=100.0):
        d = np.sqrt((yy - y) ** 2 + (xx - x) ** 2)
        return inten * np.clip((radius + edge_w / 2 - d) / edge_w, 0, 1)

    return make


# ================= 应变 =================

def make_strained_cube(sy=10, sx=10, det=64, seed=7):
    """合成 NBED 应变数据立方：δg = -Eᵀḡ（ḡ 相对束心，FCC[001] 斑位）。

    应变包宽度选得使 (0:2,0:2) 参考角落的应变 <1e-4（参考区必须
    实际零应变，否则测得的是相对参考的偏差——这正是相对应变测量
    的语义）。
    """
    rng = np.random.default_rng(seed)
    cy = cx = det / 2
    disk = soft_disk_factory(det, 3.0)
    ang = np.deg2rad([0, 90, 180, 270, 45, 135, 225, 315])
    rad = np.array([18.0] * 4 + [25.4] * 4)
    base = np.stack([cy + rad * np.sin(ang), cx + rad * np.cos(ang)], axis=1)
    g_c = base - np.array([cy, cx])

    yy_ax, xx_ax = np.mgrid[0:sy, 0:sx]
    eps_xx = 0.02 * np.exp(-(((yy_ax - sy / 2) ** 2 + (xx_ax - sx / 2) ** 2)
                             / (2 * 1.8 ** 2)))
    eps_xy = -0.01 * np.exp(-(((yy_ax - sy / 2) ** 2 + (xx_ax - sx / 2) ** 2)
                              / (2 * 2.0 ** 2)))
    cube = np.zeros((sy, sx, det, det), np.float32)
    for i in range(sy):
        for j in range(sx):
            E = np.array([[eps_xx[i, j], eps_xy[i, j]],
                          [eps_xy[i, j], 0.0]])
            dg = -(E.T @ g_c.T).T
            dp = disk(cy, cx, 200.0)
            for k in range(len(base)):
                dp += disk(base[k, 0] + dg[k, 0], base[k, 1] + dg[k, 1], 80.0)
            cube[i, j] = dp + 0.5 * rng.random((det, det))
    return cube, (cy, cx), eps_xx, eps_xy


def test_strain_recovers_synthetic_field():
    cube, center, eps_xx_true, eps_xy_true = make_strained_cube()
    res = run_strain_mapping(cube, center, 3.0, ref_region=(0, 2, 0, 2),
                             verbose=False)
    st = res['strain']
    assert res['n_matched'].min() >= 4
    assert np.isfinite(st['eps_xx']).all()
    assert np.nanmedian(np.abs(st['eps_xx'] - eps_xx_true)) < 2e-3
    assert np.nanmedian(np.abs(st['eps_xy'] - eps_xy_true)) < 2e-3
    assert np.nanmedian(np.abs(st['omega'])) < 2e-3


def test_strain_rotation_sign():
    theta = 0.01
    E_true = np.array([[0.0, -theta], [theta, 0.0]])
    g = np.array([[18, 0], [0, 18], [-18, 0], [0, -18],
                  [18, 18], [-18, -18], [18, -18], [-18, 18.]], float)
    dg = -(E_true.T @ g.T).T
    E_fit, _ = fit_displacement_gradient(g, g + dg)
    comp = strain_from_gradient(E_fit)
    assert abs(comp['omega'] - theta) < 1e-9
    assert abs(comp['eps_xy']) < 1e-9


def test_detect_bragg_disks_subpixel_accuracy():
    det = 64
    disk = soft_disk_factory(det, 3.0)
    errs = []
    for dy in np.arange(0.0, 1.0, 0.2):
        dp = disk(30.0 + dy, 40.0)
        peaks = detect_bragg_disks(dp, 3.0, center=(det / 2, det / 2))
        assert len(peaks) == 1
        errs.append(peaks[0, 0] - (30.0 + dy))
    assert max(abs(np.asarray(errs))) < 0.05


def test_strain_legacy_methods_removed():
    import pytest
    with pytest.raises(ValueError):
        run_strain_mapping(np.zeros((2, 2, 8, 8)), (4, 4), 2.0,
                           method='com')


def test_peak_pairs_cancellation():
    det = 32
    yy, xx = np.mgrid[0:det, 0:det]
    base = (((yy - 10) ** 2 + (xx - 10) ** 2 <= 4).astype(float) * 50.0
            + ((yy - 22) ** 2 + (xx - 22) ** 2 <= 4).astype(float) * 30.0)
    cube = (np.repeat(base[None, None], 4, axis=0).repeat(4, axis=1)
            + 0.01 * np.random.default_rng(0).random((4, 4, det, det)))

    def stop():
        return True

    res = peak_pairs_mapping(cube, (16, 16), 10.0, should_stop=stop)
    assert res == {'cancelled': True}


def test_peak_pair_angle_folding_undirected():
    """工单21：角度差按「方向模 180 的无向峰对」折叠，|Δ|>180° 不得为负。

    旧公式 min(d, 180-d) 在 d>180° 时给出负值（Δ=190°→-10°，
    Δ=350°→-170°），令方向完全不匹配的峰对反而获得负惩罚。
    """
    assert abs(float(_fold_angle_difference(190.0)) - 10.0) < 1e-9
    assert abs(float(_fold_angle_difference(350.0)) - 10.0) < 1e-9
    for d, want in [(0.0, 0.0), (45.0, 45.0), (90.0, 90.0),
                    (135.0, 45.0), (180.0, 0.0), (270.0, 90.0),
                    (360.0, 0.0)]:
        assert abs(float(_fold_angle_difference(d)) - want) < 1e-9
    grid = _fold_angle_difference(np.arange(0.0, 360.01, 0.25))
    assert float(grid.min()) >= 0.0
    assert float(grid.max()) <= 90.0 + 1e-9


def _gauss_disk_factory(det, sigma=1.5):
    """高斯盘：严格单极大（软边盘的线性剖面在中心削顶成平台，
    find_peaks_in_dp 的 `dp == local_max` 会把平台逐像素都判成峰）。"""
    yy, xx = np.mgrid[0:det, 0:det]

    def make(y, x, inten=100.0):
        d2 = (yy - y) ** 2 + (xx - x) ** 2
        return inten * np.exp(-d2 / (2.0 * sigma ** 2))

    return make


def _dp_from_disks(det, disk, peaks):
    dp = np.zeros((det, det))
    for y, x, inten in peaks:
        dp += disk(y, x, inten)
    return dp


def test_peak_pairs_190_degree_trap_not_preferred():
    """工单21 管线级：Δ≈190° 的陷阱峰对不得凭负分抢走最佳匹配。

    参考峰对方向 29.74°（间距 24.19 px）。位置 (0,0) 的局域 DP 含
    真匹配对（34.99°，24.41 px，有向差 5.25°）与陷阱对（-160.83°，
    24.35 px，有向差 190.57°≈190°）。旧公式对陷阱对记负惩罚
    （score≈-0.89 < 真匹配 0.75）而选错；按模 180 折叠后陷阱对差
    10.57°（score≈1.22），真匹配以 0.75 胜出。
    """
    det = 64
    disk = _gauss_disk_factory(det)
    ref_dp = _dp_from_disks(det, disk,
                            [(32, 32, 50.0), (44, 53, 30.0)])
    cube = np.repeat(ref_dp[None, None], 4, axis=0).repeat(4, axis=1)
    cube[0, 0] = _dp_from_disks(
        det, disk, [(32, 32, 100.0), (46, 52, 60.0), (24, 9, 40.0)])

    res = peak_pairs_mapping(cube, (32, 32), 15.0)
    assert res is not None and res['n_pairs'] == 1
    # (0,0) 必须选中真匹配对（≈35.0°），而非 -160.8° 的陷阱对
    assert abs(res['pair_angles'][0, 0, 0] - 34.99) < 3.0
    # 其余位置正常匹配参考对（29.74°）
    assert (np.abs(res['pair_angles'][1:, :, 0] - 29.74) < 1.0).all()


def test_peak_pairs_350_degree_trap_rejected():
    """工单21 管线级：Δ≈350° 的峰对按模 180 折叠后方向仅差 9.46°，
    但间距失配 ~19.7 px 必须被 `score < 5` 判据拒绝。

    旧公式对 Δ=350.54° 记负惩罚 -17.05，把 score 压到 ≈2.6 而将
    间距差近 20 px 的峰对错误放行（负分恒满足 <5）。
    """
    det = 64
    disk = _gauss_disk_factory(det)
    ref_dp = _dp_from_disks(det, disk,
                            [(32, 50, 50.0), (28, 26, 30.0)])  # -170.54°
    cube = np.repeat(ref_dp[None, None], 4, axis=0).repeat(4, axis=1)
    # 陷阱对：+180.0°（arctan2(0,-44)），间距 44 px → Δ=350.54°≈350°
    cube[0, 0] = _dp_from_disks(det, disk,
                                [(32, 50, 100.0), (32, 6, 60.0)])

    res = peak_pairs_mapping(cube, (32, 50), 25.0)  # max_distance=50
    assert res is not None and res['n_pairs'] == 1
    assert res['pair_distances'][0, 0, 0] == 0.0
    assert res['pair_angles'][0, 0, 0] == 0.0
    # 其余 15 个位置与参考对完全一致，正常匹配
    assert (res['pair_distances'][1:, :, 0] > 0).all()


# ================= 取向 =================

def test_crystallography_basics():
    st = make_structure('Au (FCC)')
    B = reciprocal_matrix(st)
    assert abs(np.linalg.norm(B.T @ np.array([2, 0, 0]))
               - 2 / 4.0782) < 1e-9
    assert abs(np.linalg.norm(B.T @ np.array([2, 2, 0]))
               - 2 * math.sqrt(2) / 4.0782) < 1e-9
    assert not reflection_allowed((1, 0, 0), 'F')
    assert not reflection_allowed((1, 1, 0), 'F')
    assert reflection_allowed((1, 1, 1), 'F')
    assert reflection_allowed((2, 0, 0), 'F')
    assert not reflection_allowed((2, 2, 2), 'D')   # 金刚石 222 禁
    assert reflection_allowed((1, 1, 1), 'D')
    # 预设不得被函数默认参数覆盖，且高阶反射不可按 gcd 过滤：
    # FCC [111] 的首环是二阶反射 {2,-2,0}（0.6935），禁戒的 {110}
    # （0.3468）和被 gcd 错杀的二阶都不应改变它
    t111 = zone_axis_template((1, 1, 1), st)
    assert abs(t111['r'][0] - 2 * math.sqrt(2) / 4.0782) < 1e-9
    # HCP 预设必须带 γ=120°：GaN (100) 的 d = a√3/2
    st_h = make_structure('GaN (HCP)')
    assert abs(st_h['gamma'] - 120.0) < 1e-9
    B_h = reciprocal_matrix(st_h)
    assert abs(np.linalg.norm(B_h.T @ np.array([1, 0, 0]))
               - 2 / (3.1890 * math.sqrt(3))) < 1e-9


def _synthetic_dp(st, uvw, s, delta_rad, det=128, seed=1):
    disk = soft_disk_factory(det, 3.0)
    rng = np.random.default_rng(seed)
    cy = cx = det / 2
    tmpl = zone_axis_template(uvw, st)
    dp = 2.0 * rng.random((det, det))
    for gi in range(len(tmpl['r'])):
        R = s * tmpl['r'][gi]
        if R > 56:
            break
        ang = tmpl['phi'][gi] + delta_rad
        dp += disk(cy - R * math.sin(ang), cx + R * math.cos(ang),
                   120.0 / (1 + gi * 0.15))
    return dp, (cy, cx)


def _cubic_equiv(a, b):
    return tuple(sorted(map(abs, a))) == tuple(sorted(map(abs, b)))


def test_orientation_indexing_recovers_zone_axis():
    st = make_structure('Au (FCC)')
    bank = build_zone_axis_bank(st, max_index=2)
    assert len(bank) > 10
    cases = [((0, 0, 1), 40.0, 23.0), ((1, 1, 1), 28.0, 110.0),
             ((2, 1, 0), 36.0, 61.0)]
    for uvw_true, s_t, delta_t in cases:
        delta_t = math.radians(delta_t)
        dp, center = _synthetic_dp(st, uvw_true, s_t, delta_t)
        hit = index_diffraction_pattern(dp, center, 3.0, bank)
        rec = tuple(hit['uvw'])
        assert _cubic_equiv(rec, uvw_true), f"{rec} vs {uvw_true}"
        assert abs(hit['s'] - s_t) / s_t < 0.05
        assert hit['score'] > 0.5
        if rec == uvw_true:
            d_err = math.degrees(abs(math.atan2(
                math.sin(hit['delta'] - delta_t),
                math.cos(hit['delta'] - delta_t))))
            # 中心对称花样面内角只定义到 mod 180°
            assert min(d_err, 180 - d_err) < 3.0


def test_structure_presets_complete():
    for name in ('Au (FCC)', 'Fe (BCC)', 'Si (金刚石)', 'GaN (HCP)'):
        st = make_structure(name)
        tmpl = zone_axis_template((0, 0, 1), st)
        assert tmpl is not None and len(tmpl['r']) >= 3


# ================= 叠层 ePIE =================

def test_epie_recovers_object_amplitude_and_phase():
    rng = np.random.default_rng(5)
    sy = sx = 8
    det = 24
    alpha = 6.0
    cy = cx = det // 2

    probe = initialize_probe(det, det, center=(cy, cx), alpha=alpha)
    amp_p = np.abs(probe)
    assert np.unravel_index(amp_p.argmax(), amp_p.shape) == (cy, cx)

    obj_y = sy - 1 + det
    gy, gx = np.mgrid[0:obj_y, 0:obj_y]
    amp = 0.7 + 0.3 * np.cos(2 * np.pi * gy / 9.0)
    phase = 0.8 * np.sin(2 * np.pi * gx / 11.0)
    obj = amp * np.exp(1j * phase)

    data = np.zeros((sy, sx, det, det))
    for i in range(sy):
        for j in range(sx):
            psi = probe * obj[i:i + det, j:j + det]
            Psi = np.fft.fft2(np.fft.ifftshift(psi))
            data[i, j] = np.roll(np.abs(Psi) ** 2, (cy, cx), axis=(0, 1))
    data = rng.poisson(data * (2e4 / data.mean())).astype(float)

    res = run_ptychography(data, (cy, cx), alpha, n_iterations=30,
                           step_size=0.5, verbose=False)

    def corr(a, b):
        a = a.ravel() - a.mean()
        b = b.ravel() - b.mean()
        return float(a @ b / np.sqrt((a @ a) * (b @ b)))

    true_a = amp[cy:cy + sy, cx:cx + sx]
    true_p = phase[cy:cy + sy, cx:cx + sx]
    assert corr(res['amplitude'], true_a) > 0.85
    assert corr(res['phase'], true_p) > 0.85
    assert res['errors'][-1] < res['errors'][0] * 0.5


def test_epie_cancellation():
    data = np.random.default_rng(0).random((4, 4, 12, 12)) + 1.0
    calls = {'n': 0}

    def stop():
        calls['n'] += 1
        return calls['n'] > 3

    res = run_ptychography(data, (6, 6), 4.0, n_iterations=10,
                           verbose=False, should_stop=stop)
    assert res['cancelled'] is True


def test_ptychography_wdd_removed():
    import pytest
    with pytest.raises(ValueError):
        run_ptychography(np.zeros((2, 2, 8, 8)), (4, 4), 2.0, method='wdd')
