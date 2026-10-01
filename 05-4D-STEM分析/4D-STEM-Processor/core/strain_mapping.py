"""
strain_mapping.py - Strain mapping from 4D-STEM data.

物理原理
--------
纳米束电子衍射 (NBED) 应变测量的正确路线：局部晶格畸变使每个
Bragg 盘相对参考区域移动 δg，小应变下

    δg = -Eᵀ ḡ,   E = ∇u（位移梯度张量，2×2）

其中 ḡ 为参考区倒易矢量（探测器像素坐标即可，应变无量纲、与绝对
标定无关）。每个扫描像素用 ≥2 个非共线斑点即可最小二乘解出 E，
应变张量取对称部分 ε = (E + Eᵀ)/2，旋转取反对称部分。

旧版把 CoM（束偏转 ∝ 电场）当作位移场求梯度、把整图互相关平移当
位移场，均不成立，已移除。

流程：圆盘模板匹配滤波检峰 → COM 亚像素定位 → 参考区中值建
ḡ 列表 → 逐像素最近邻配对 → 鲁棒最小二乘拟合 E。
"""
import json
import os

import numpy as np
from scipy import ndimage

# 直接相关的运算量上限（dp.size * template.size）。低于它直接用
# ndimage.correlate；高于它换 FFT 卷积——大探测器（如 512x512）+ 大盘径
# 模板的直接相关是 O(N·K)，逐像素检测时完全不可用。
_DIRECT_CORRELATION_LIMIT = 8_000_000


def _template_correlate(dp, template):
    """与（对称）圆盘模板的匹配滤波相关。

    两条路径数值等价：模板奇数尺寸、零填充边界下，中心对齐的线性
    卷积与 ndimage.correlate(mode='constant') 一致。
    """
    if dp.size * template.size <= _DIRECT_CORRELATION_LIMIT:
        return ndimage.correlate(dp, template, mode='constant')
    from scipy.signal import fftconvolve
    return fftconvolve(dp, template, mode='same')


# ========== Bragg 盘检测 ==========

def _disk_template(radius):
    """软边实心圆盘相关模板（matched filter 用）。

    必须是填充盘而非边缘环：环模板的相关峰在两个相邻整数中心间
    会出现精确平局，破坏亚像素定位。
    """
    r = int(max(2, round(radius)))
    yy, xx = np.mgrid[-r - 1:r + 2, -r - 1:r + 2]
    d = np.sqrt(yy ** 2 + xx ** 2)
    # 线性缓变边缘，宽度 ~2 px
    return np.clip((radius + 1.0 - d) / 2.0, 0.0, 1.0)


def detect_bragg_disks(dp, disk_radius, min_separation=None,
                       rel_threshold=0.06, abs_sigma=8.0, max_peaks=64,
                       exclude_radius=None, center=None):
    """
    用圆盘模板的匹配滤波在单张衍射图上检测 Bragg 盘位置。

    Parameters
    ----------
    dp : ndarray (det_y, det_x)
        单张衍射图（强度）
    disk_radius : float
        盘半径（像素），通常取 BF 盘半径
    min_separation : float, optional
        峰间最小间距，默认 disk_radius
    rel_threshold : float
        相对相关图最大值的阈值（去掉弱响应）
    abs_sigma : float
        相关图噪声 σ 的倍数阈值（robust MAD 估计）
    max_peaks : int
        最多返回峰数
    exclude_radius : float, optional
        中心束排除半径；None 时排除 disk_radius
    center : (cy, cx), optional
        中心束位置，默认图中心

    Returns
    -------
    ndarray (n, 3)：[y, x, 强度]（亚像素，含中心束）
    """
    dp = np.asarray(dp, dtype=np.float64)
    det_y, det_x = dp.shape
    if center is None:
        center = (det_y / 2.0, det_x / 2.0)
    if min_separation is None:
        min_separation = max(2.0, disk_radius)

    # 匹配滤波：与盘模板互相关（模板对称，flip 无所谓）
    template = _disk_template(disk_radius)
    corr = _template_correlate(dp, template)

    # robust 噪声估计（MAD），阈值取相对/绝对两者较大者
    med = np.median(corr)
    sigma = 1.4826 * np.median(np.abs(corr - med))
    thr = max(rel_threshold * corr.max(), med + abs_sigma * sigma)
    if thr > corr.max():
        return np.zeros((0, 3))

    local_max = ndimage.maximum_filter(corr, size=max(3, int(disk_radius)))
    peaks_mask = (corr >= local_max) & (corr > thr)
    coords = np.argwhere(peaks_mask)
    if len(coords) == 0:
        return np.zeros((0, 3))

    vals = corr[coords[:, 0], coords[:, 1]]
    order = np.argsort(vals)[::-1]
    coords, vals = coords[order], vals[order]

    # COM 亚像素定位：窗口必须完整包住盘（±(r+1)），截断的质心会被
    # 拉向整数峰中心，导致亚像素位移被系统性缩小
    win = int(round(disk_radius)) + 1
    result = []
    for (py, px), v in zip(coords, vals):
        y0, y1 = max(0, py - win), min(det_y, py + win + 1)
        x0, x1 = max(0, px - win), min(det_x, px + win + 1)
        patch = dp[y0:y1, x0:x1]
        if patch.max() <= 0:
            continue
        # 抬高底噪以压缩质心拖尾
        w = np.clip(patch - np.median(patch), 0, None)
        if w.sum() <= 0:
            continue
        gy, gx = np.mgrid[y0:y1, x0:x1]
        result.append((float((w * gy).sum() / w.sum()),
                       float((w * gx).sum() / w.sum()),
                       float(v)))

    # 峰间距约束：贪心保留强者
    selected = []
    min_sep2 = min_separation ** 2
    for p in result:
        ok = True
        for q in selected:
            if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < min_sep2:
                ok = False
                break
        if ok:
            selected.append(p)
        if len(selected) >= max_peaks:
            break
    if not selected:
        return np.zeros((0, 3))
    peaks = np.array(selected)

    # 排除中心束：其位置由束流倾斜/扫描误差主导，不含晶格应变信息
    if exclude_radius is None:
        exclude_radius = disk_radius
    r2 = (peaks[:, 0] - center[0]) ** 2 + (peaks[:, 1] - center[1]) ** 2
    peaks = peaks[r2 > exclude_radius ** 2]
    return peaks


# ========== 应变张量拟合 ==========

def fit_displacement_gradient(ref_g, loc_g, outlier_reject=4.0):
    """
    最小二乘解位移梯度张量 E：δg = -Eᵀ ḡ。

    Parameters
    ----------
    ref_g, loc_g : ndarray (n, 2)
        参考区与当前位置的倒易矢量（任意一致单位，如探测器像素）

    Returns
    -------
    E : ndarray (2, 2)
        位移梯度 ∇u（行 = 对 u 的分量求导方向）
    residual : float
        拟合后平均每斑点残差（像素），用于质量评估
    """
    ref_g = np.asarray(ref_g, dtype=np.float64)
    loc_g = np.asarray(loc_g, dtype=np.float64)
    delta = loc_g - ref_g

    def _solve(ref, dlt):
        # 逐行堆叠 δgᵀ = -ḡᵀE ⟹ Ḡ·E = -ΔG，lstsq 直接解出 E
        A = ref                      # (n, 2)
        B = -dlt                     # (n, 2)
        E, *_ = np.linalg.lstsq(A, B, rcond=None)  # (2, 2)
        return E, B

    E, B = _solve(ref_g, delta)
    if outlier_reject is not None and len(ref_g) >= 5:
        pred = -(E.T @ ref_g.T).T    # 预测 δg
        res = np.linalg.norm(loc_g - (ref_g + pred), axis=1)
        med = np.median(res)
        if med > 0:
            keep = res <= outlier_reject * med
            if keep.sum() >= 4 and keep.sum() < len(ref_g):
                E, _ = _solve(ref_g[keep], delta[keep])
    pred = -(E.T @ ref_g.T).T
    residual = float(np.mean(np.linalg.norm((loc_g - ref_g) - pred, axis=1)))
    return E, residual


def strain_from_gradient(E):
    """由位移梯度 E 计算应变/旋转/主应变等派生量（保持旧输出键名）。"""
    E = np.asarray(E)
    eps_xx = E[..., 0, 0]
    eps_yy = E[..., 1, 1]
    eps_xy = 0.5 * (E[..., 0, 1] + E[..., 1, 0])
    omega = 0.5 * (E[..., 1, 0] - E[..., 0, 1])  # 刚体旋转（y 向下的图像坐标）

    eps_mean = 0.5 * (eps_xx + eps_yy)
    R = np.sqrt(((eps_xx - eps_yy) / 2) ** 2 + eps_xy ** 2)
    return {
        'eps_xx': eps_xx, 'eps_yy': eps_yy, 'eps_xy': eps_xy,
        'omega': omega,
        'eps_1': eps_mean + R, 'eps_2': eps_mean - R,
        'gamma_max': 2 * R,
        'eps_mean': eps_mean,
        'eps_vm': np.sqrt(eps_xx ** 2 + eps_yy ** 2 - eps_xx * eps_yy
                          + 3 * eps_xy ** 2),
    }


def build_reference_from_list(all_peaks, weights, match_tol=3.0,
                              max_spots=24, min_count=2):
    """
    从参考区的峰列表（含重复检测）聚类出 ḡ。

    Parameters
    ----------
    all_peaks : ndarray (n, 3)
        [y, x, 强度]
    weights : ndarray (n,)
        每个峰的计数（来自多少张参考 DP），用于频次过滤
    """
    order = np.argsort(all_peaks[:, 2])[::-1]
    clusters = []
    for idx in order:
        p = all_peaks[idx, :2]
        placed = False
        for c in clusters:
            if (p[0] - c['y']) ** 2 + (p[1] - c['x']) ** 2 < match_tol ** 2:
                c['ys'].append(p[0])
                c['xs'].append(p[1])
                c['ints'].append(all_peaks[idx, 2])
                c['count'] += weights[idx]
                placed = True
                break
        if not placed:
            clusters.append({'y': p[0], 'x': p[1], 'ys': [p[0]],
                             'xs': [p[1]], 'ints': [all_peaks[idx, 2]],
                             'count': weights[idx]})
    good = [c for c in clusters if c['count'] >= min_count]
    good.sort(key=lambda c: np.mean(c['ints']), reverse=True)
    good = good[:max_spots]
    if not good:
        return np.zeros((0, 3))
    ref = np.array([[np.median(c['ys']), np.median(c['xs']),
                     np.mean(c['ints'])] for c in good])
    return ref


def match_peaks_to_reference(peaks, ref_spots, match_tol=3.0):
    """
    把当前位置检测峰与参考 ḡ 配对（每个参考斑取容差内最强者）。

    Returns
    -------
    ref_idx, loc_idx : 配对索引数组
    """
    if len(peaks) == 0 or len(ref_spots) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    # 距离矩阵 (n_ref, n_loc)，广播计算
    d2 = ((ref_spots[:, None, :2] - peaks[None, :, :2]) ** 2).sum(-1)
    valid = d2 <= match_tol ** 2
    # 容差内按峰强度取最强；无效置 -inf
    scores = np.where(valid, peaks[None, :, 2], -np.inf)
    best = scores.argmax(axis=1)            # 每个参考斑的最佳局部峰
    has = np.isfinite(scores.max(axis=1)) & (scores.max(axis=1) > -np.inf)
    ref_idx = np.nonzero(has)[0]
    # 同一局部峰不允许配两个参考斑：按分数降序贪心占用
    order = ref_idx[np.argsort(scores[ref_idx, best[ref_idx]])[::-1]]
    used = set()
    ri_out, li_out = [], []
    for r in order:
        li = best[r]
        # 若被占用，在剩余有效峰中取次强
        if li in used:
            cand = np.nonzero(valid[r] & np.array(
                [j not in used for j in range(len(peaks))]))[0]
            if len(cand) == 0:
                continue
            li = cand[np.argmax(peaks[cand, 2])]
        used.add(li)
        ri_out.append(r)
        li_out.append(li)
    return np.array(ri_out, int), np.array(li_out, int)


# ========== 主入口 ==========

def strain_from_bragg_disks(datacube, center, disk_radius, ref_region=None,
                            match_tol=None, min_peaks=4, max_spots=24,
                            should_stop=None, verbose=True, progress=None):
    """
    Bragg 盘跟踪应变成像（NBED 应变测量的标准路线）。

    Parameters
    ----------
    datacube : ndarray (scan_y, scan_x, det_y, det_x)
    center : (cy, cx) 中心束位置（像素）
    disk_radius : float 盘半径（像素，可用 find_alpha_from_radial 估计）
    ref_region : tuple (y0, y1, x0, x1), optional
        零应变参考区（探测器坐标外的扫描区）；None 取整幅中值——
        此时输出为相对全场平均的应变，不是绝对应变
    match_tol : float 配对容差（像素），默认 max(2, disk_radius/2)
    min_peaks : int 拟合所需最少配对斑数
    progress : callable(str), optional
        细粒度进度回调（每 ~10% 一次），供 GUI 日志；None 时退回 print

    Returns
    -------
    dict：E 场 (2,2,scan_y,scan_x)、strain 各分量（无效点 NaN）、
    n_matched、fit_residual、reference_spots
    """
    def _report(msg):
        if progress is not None:
            progress(msg)
        elif verbose:
            print(msg)

    datacube = np.asarray(datacube)
    scan_y, scan_x, det_y, det_x = datacube.shape
    cy, cx = center
    if match_tol is None:
        match_tol = max(2.0, disk_radius * 0.5)

    # 1) 逐位置检测（含中心束，供参考区统计；拟合时再排除）
    detections = np.empty((scan_y, scan_x), dtype=object)
    total = scan_y * scan_x
    done = 0
    for sy in range(scan_y):
        if should_stop is not None and should_stop():
            return {'cancelled': True}
        for sx in range(scan_x):
            detections[sy, sx] = detect_bragg_disks(
                datacube[sy, sx], disk_radius, center=(cy, cx),
                rel_threshold=0.06, max_peaks=64)
            done += 1
            if done % max(1, total // 10) == 0:
                _report(f"    disk detection {done}/{total}")

    # 2) 参考区 ḡ：对参考区 DP 的峰做联合聚类
    if ref_region is not None:
        y0, y1, x0, x1 = ref_region
        sel = detections[y0:y1, x0:x1].ravel()
    else:
        sel = detections.ravel()
    ref_peaks = [p for p in sel if len(p)]
    if ref_peaks:
        cat = np.vstack(ref_peaks)
        counts = np.ones(len(cat))
        ref_spots = build_reference_from_list(cat, counts,
                                              match_tol=match_tol,
                                              max_spots=max_spots,
                                              min_count=2)
        # 排除中心束簇
        r2 = (ref_spots[:, 0] - cy) ** 2 + (ref_spots[:, 1] - cx) ** 2
        ref_spots = ref_spots[r2 > disk_radius ** 2]
    else:
        ref_spots = np.zeros((0, 3))

    # 3) 逐像素配对 + 拟合（E 存为 (scan_y, scan_x, 2, 2)）
    E_map = np.full((scan_y, scan_x, 2, 2), np.nan)
    n_matched = np.zeros((scan_y, scan_x), dtype=int)
    residual_map = np.full((scan_y, scan_x), np.nan)
    center_arr = np.array([cy, cx])
    if len(ref_spots) >= 2:
        done = 0
        for sy in range(scan_y):
            if should_stop is not None and should_stop():
                return {'cancelled': True}
            for sx in range(scan_x):
                done += 1
                if done % max(1, total // 10) == 0:
                    _report(f"    strain fitting {done}/{total}")
                peaks = detections[sy, sx]
                ri, li = match_peaks_to_reference(peaks, ref_spots,
                                                  match_tol=match_tol)
                if len(ri) < min_peaks:
                    continue
                g = ref_spots[ri, :2] - center_arr
                # 共线检验：参考斑近似共线时 E 不可辨识
                sv = np.linalg.svd(g - g.mean(0), compute_uv=False)
                if sv[-1] < 1e-6 * max(sv[0], 1e-12):
                    continue
                E, res = fit_displacement_gradient(
                    g, peaks[li, :2] - center_arr)
                E_map[sy, sx] = E
                n_matched[sy, sx] = len(ri)
                residual_map[sy, sx] = res

    strain = strain_from_gradient(E_map)

    return {
        'E': E_map,
        'strain': strain,
        'n_matched': n_matched,
        'fit_residual': residual_map,
        'reference_spots': ref_spots,
        'cancelled': False,
    }


def plot_strain_results(strain_result, name, output_dir, scan_scale_nm=1.0):
    """Generate strain mapping visualization.

    ``strain_result`` is the full dict returned by
    :func:`strain_from_bragg_disks`, so the fitted E field, per-pixel match
    counts, fit residuals and the reference spot list are persisted in the
    .npz alongside the strain components (needed to audit fit quality).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    strain_data = strain_result['strain']

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    components = [
        ('eps_xx', 'εxx'), ('eps_yy', 'εyy'), ('eps_xy', 'εxy'),
        ('eps_1', 'ε₁ (max principal)'), ('eps_2', 'ε₂ (min principal)'),
        ('gamma_max', 'γmax (max shear)'),
        ('omega', 'ω (rotation)'), ('eps_mean', 'εmean (hydrostatic)'),
        ('eps_vm', 'εvm (von Mises)'),
    ]

    for idx, (key, label) in enumerate(components):
        ax = axes[idx // 3, idx % 3]
        data = strain_data[key]

        vmax = np.nanmax(np.abs(data))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1

        im = ax.imshow(data, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'{label}\n[{np.nanmin(data):.4f}, {np.nanmax(data):.4f}]')
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(f'Strain Mapping: {name} (Bragg disk tracking)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{name}_strain.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    arrays = dict(strain_data)
    for key in ('E', 'n_matched', 'fit_residual', 'reference_spots'):
        if strain_result.get(key) is not None:
            arrays[key] = strain_result[key]
    np.savez(os.path.join(output_dir, f'{name}_strain.npz'), **arrays)
    # Summary statistics are only meaningful for 2D per-pixel maps.
    summary = {}
    for key, arr in strain_data.items():
        arr = np.asarray(arr)
        if arr.ndim != 2:
            continue
        summary[key] = {
            'min': float(np.nanmin(arr)),
            'max': float(np.nanmax(arr)),
            'mean': float(np.nanmean(arr)),
            'std': float(np.nanstd(arr)),
        }
    report = {
        'method': strain_result.get('method', 'bragg'),
        'reference': strain_result.get(
            'reference', 'whole-image median (relative strain)'),
        'component_stats': summary,
    }
    with open(os.path.join(output_dir, f'{name}_strain_metadata.json'),
              'w', encoding='utf-8') as fj:
        json.dump(report, fj, indent=2, ensure_ascii=False)

    return save_path


def run_strain_mapping(datacube, center, alpha, method='bragg',
                       ref_region=None, name='sample', output_dir=None,
                       should_stop=None, verbose=True, progress=None,
                       **_legacy):
    """
    Main strain mapping function.

    Parameters
    ----------
    datacube : ndarray (scan_y, scan_x, det_y, det_x)
    center : tuple (cy, cx) - beam center
    alpha : float - BF disk radius（用作盘检测模板半径）
    method : str - 'bragg'（Bragg 盘跟踪 + 最小二乘畸变拟合）
    ref_region : (y0, y1, x0, x1) - 零应变参考扫描区；None 用整幅中值
        （输出为相对全场平均的应变）
    progress : callable(str), optional - 细粒度进度回调（GUI 日志用）

    旧参数 scan_scale_nm / det_scale_mrad 已无作用（应变无量纲），
    为兼容旧调用保留在 **_legacy 中忽略。
    """
    if method not in ('bragg', 'disk'):
        raise ValueError(
            f"未知方法 '{method}'。可用方法：'bragg'（Bragg 盘跟踪）。"
            "旧版 'com'/'xcorr' 把束偏转或整图平移当作位移场，物理上"
            "不成立，已移除。")

    results = {'method': 'bragg',
               'reference': ('scan region ' + str(ref_region)
                             if ref_region is not None
                             else 'whole-image median (relative strain)')}
    result = strain_from_bragg_disks(datacube, center, alpha,
                                     ref_region=ref_region,
                                     should_stop=should_stop,
                                     verbose=verbose,
                                     progress=progress)
    if result.get('cancelled'):
        results['cancelled'] = True
        return results
    results.update(result)

    if output_dir and 'strain' in results:
        plot_strain_results(results, name, output_dir)

    return results
