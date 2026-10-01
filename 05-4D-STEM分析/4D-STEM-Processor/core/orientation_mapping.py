"""
orientation_mapping.py - Crystal orientation mapping from 4D-STEM data.

物理原理
--------
对每个扫描像素的衍射花样，检测 Bragg 斑（角度 + 强度），与按真实
晶体学模拟的带轴模板做尺度不变匹配：

1. 由晶格参数（任意晶系）构造倒易格点，按格子类型施加运动学消光
   条件（P/I/F/金刚石）；
2. 枚举候选带轴 [uvw]，取满足-zone 定律 hu+kv+lw=0 的 ZOLZ 反射
   g 投影到垂直于 u 的平面，得到模板斑点极坐标 (|g|, φ)；
3. 对每个模板：(a) 用 (实测斑, 模板斑) 半径比投票估计未知的相机
   常数 s [px·Å]；(b) 在 s 固定后用角度差投票估计面内旋转 δ；
   (c) 以"实测强度被解释的比例"减"模板强斑缺失比例"打分；
4. 取最高分模板为该像素的带轴，输出带轴图、可靠性、面内旋转与
   相机常数图。

旧版的"模板"是按 alpha 缩放的人造高斯环加角向调制，与任何晶体
结构无关，其输出不是晶体取向，已移除。

运动学近似下花样具 Friedel 对称（±u 镜像不可分），且图像 y 轴向
下；本模块按 x 向右、y 向上的右手系报告面内角，镜像解需结合倾转
台或 CBED 判别。
"""
import math
import os

import numpy as np
from scipy import ndimage

try:
    from core.strain_mapping import detect_bragg_disks
except ImportError:
    from strain_mapping import detect_bragg_disks


# ========== 晶体学 ==========

# 常见相预设：a,b,c (Å)，夹角 (deg)，格子类型
STRUCTURE_PRESETS = {
    'Au (FCC)':        dict(a=4.0782, lattice_type='F'),
    'Ag (FCC)':        dict(a=4.0857, lattice_type='F'),
    'Cu (FCC)':        dict(a=3.6149, lattice_type='F'),
    'Pt (FCC)':        dict(a=3.9231, lattice_type='F'),
    'Ni (FCC)':        dict(a=3.5242, lattice_type='F'),
    'Al (FCC)':        dict(a=4.0495, lattice_type='F'),
    'Fe (BCC)':        dict(a=2.8664, lattice_type='I'),
    'W (BCC)':         dict(a=3.1648, lattice_type='I'),
    'Cr (BCC)':        dict(a=2.8840, lattice_type='I'),
    'Si (金刚石)':      dict(a=5.4309, lattice_type='D'),
    'Ge (金刚石)':      dict(a=5.6575, lattice_type='D'),
    'Ti (HCP)':        dict(a=2.9508, c=4.6855, lattice_type='P',
                            alpha=90.0, beta=90.0, gamma=120.0),
    'Mg (HCP)':        dict(a=3.2094, c=5.2108, lattice_type='P',
                            alpha=90.0, beta=90.0, gamma=120.0),
    'Zn (HCP)':        dict(a=2.6650, c=4.9468, lattice_type='P',
                            alpha=90.0, beta=90.0, gamma=120.0),
    'GaN (HCP)':       dict(a=3.1890, c=5.1850, lattice_type='P',
                            alpha=90.0, beta=90.0, gamma=120.0),
}


def make_structure(name=None, a=None, b=None, c=None,
                   alpha=None, beta=None, gamma=None, lattice_type=None):
    """构造结构参数 dict；name 取预设，显式非 None 参数可覆盖预设值。

    注意默认值必须是 None 而不是 90/'P'，否则会覆盖预设（如 HCP 的
    γ=120°、FCC 的 'F'）。
    """
    params = dict(a=1.0, b=None, c=None, alpha=90.0, beta=90.0,
                  gamma=90.0, lattice_type='P', name='custom')
    if name is not None:
        if name not in STRUCTURE_PRESETS:
            raise ValueError(f"未知预设 '{name}'，可用：{list(STRUCTURE_PRESETS)}")
        params.update(STRUCTURE_PRESETS[name])
        params['name'] = name
    for k, v in dict(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                     lattice_type=lattice_type).items():
        if v is not None:
            params[k] = v
    params['b'] = params['b'] or params['a']
    params['c'] = params['c'] or params['a']
    return params


def lattice_matrix(structure):
    """直接格子矩阵 A（行向量 a1,a2,a3，Å）→ 倒易基满足 aᵢ·bⱼ=δᵢⱼ。"""
    a, b, c = structure['a'], structure['b'], structure['c']
    al, be, ga = (math.radians(structure[k]) for k in ('alpha', 'beta', 'gamma'))
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([b * math.cos(ga), b * math.sin(ga), 0.0])
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz2 = c * c - cx * cx - cy * cy
    if cz2 <= 0:
        raise ValueError("非法晶格参数（夹角不自洽）")
    a3 = np.array([cx, cy, math.sqrt(cz2)])
    return np.vstack([a1, a2, a3])


def reciprocal_matrix(structure):
    """倒易格子矩阵 B（行向量 b1,b2,b3，1/Å）：B = (A⁻¹)ᵀ。"""
    return np.linalg.inv(lattice_matrix(structure)).T


def reflection_allowed(hkl, lattice_type):
    """运动学消光条件（按格子类型）。"""
    h, k, l = hkl
    t = lattice_type.upper()
    if t == 'P':
        return True
    if t == 'I':
        return (h + k + l) % 2 == 0
    if t == 'F':
        return (h % 2 == k % 2 == l % 2)
    if t == 'D':  # 金刚石：F 条件 + 全偶时 h+k+l=4n
        if not (h % 2 == k % 2 == l % 2):
            return False
        if (h % 2) == 1:
            return True
        return (h + k + l) % 4 == 0
    raise ValueError(f"未知格子类型 '{lattice_type}'（P/I/F/D）")


def _gcd3(h, k, l):
    return math.gcd(math.gcd(abs(h), abs(k)), abs(l))


def enumerate_zone_axes(max_index=2):
    """枚举互不平行的带轴 [uvw]，规范符号（首个非零分量为正）。"""
    axes = []
    seen = set()
    for u in range(-max_index, max_index + 1):
        for v in range(-max_index, max_index + 1):
            for w in range(-max_index, max_index + 1):
                if (u, v, w) == (0, 0, 0) or _gcd3(u, v, w) != 1:
                    continue
                # 规范化平行方向符号
                for t in (u, v, w):
                    if t != 0:
                        if t < 0:
                            u, v, w = -u, -v, -w
                        break
                key = (u, v, w)
                if key not in seen:
                    seen.add(key)
                    axes.append(key)
    return axes


def zone_axis_template(uvw, structure, max_hkl=8, g_max=2.2):
    """
    生成带轴模板：ZOLZ 反射在垂直于 u 的平面内的极坐标。

    g_max：模板保留的最大 |g|（1/Å）。运动学强度随散射角快速衰减，
    d < ~0.45 Å 的高阶反射实际不可见，保留只会制造退化匹配。

    Returns
    -------
    dict：'uvw', 'r'（|g|，1/Å，升序）, 'phi'（弧度，右手系），
    'g_vec'（3D，1/Å）, 'hkl'；不足 3 个反射时返回 None
    """
    u = np.array(uvw, dtype=np.float64)
    u = u / np.linalg.norm(u)
    B = reciprocal_matrix(structure)
    hmax = max_hkl

    gs, hkls = [], []
    for h in range(-hmax, hmax + 1):
        for k in range(-hmax, hmax + 1):
            for l in range(-hmax, hmax + 1):
                if (h, k, l) == (0, 0, 0):
                    continue
                if h * uvw[0] + k * uvw[1] + l * uvw[2] != 0:
                    continue                      # zone 定律
                if not reflection_allowed((h, k, l), structure['lattice_type']):
                    continue
                # 注意：不做 gcd 约化过滤——同方向不同阶是不同衍射斑
                # （FCC 的 (1,-1,0) 禁止但其二阶 (2,-2,0) 允许）
                g = B.T @ np.array([h, k, l], dtype=np.float64)  # h b1 + k b2 + l b3
                if np.linalg.norm(g) > g_max:
                    continue
                gs.append(g)
                hkls.append((h, k, l))
    if len(gs) < 3:
        return None
    gs = np.array(gs)

    # 面内右手系：e1 取最短 g，e2 = û × ê1
    i0 = np.argmin(np.linalg.norm(gs, axis=1))
    e1 = gs[i0] / np.linalg.norm(gs[i0])
    e2 = np.cross(u, e1)
    e2 /= np.linalg.norm(e2)

    r = np.linalg.norm(gs, axis=1)
    phi = np.arctan2(gs @ e2, gs @ e1)
    order = np.argsort(r)
    return {'uvw': uvw, 'r': r[order], 'phi': phi[order],
            'g_vec': gs[order], 'hkl': [hkls[i] for i in order]}


def build_zone_axis_bank(structure, max_index=2, max_hkl=8):
    """枚举带轴并生成模板库（列表按 uvw 排序保证确定性）。"""
    axes = sorted(enumerate_zone_axes(max_index))
    bank = []
    for uvw in axes:
        t = zone_axis_template(uvw, structure, max_hkl=max_hkl)
        if t is not None:
            bank.append(t)
    return bank


# ========== 花样匹配（尺度不变投票） ==========

def match_pattern_to_template(R, Phi, W, template, n_scale_bins=48,
                              r_tol=0.08, ang_tol_deg=8.0,
                              min_visible_radius=0.0):
    """
    把一组实测斑点与单个带轴模板匹配（尺度 s 与面内旋转 δ 均未知）。

    Parameters
    ----------
    R, Phi, W : ndarray (n,)
        实测斑半径（px，相对中心）、角度（弧度，右手系）、强度权重
    template : zone_axis_template 输出
    min_visible_radius : float
        中心束排除半径（px）。若某尺度 s 把模板最内环映射到该半径
        以内，则最强环不可见，此解物理上不可能，直接禁投。

    Returns
    -------
    dict：score ∈ [0,1]（越大越好）、s（px·Å）、delta（弧度）、
    matched_fraction、missing_fraction；不可解时 score=0
    """
    if len(R) < 3:
        return {'score': 0.0}
    tr = template['r']
    # (a) 尺度投票：s = R_j / r_i，且要求最内环映射出中心束排除区
    S = R[:, None] / tr[None, :]                # (n, m)
    Wm = W[:, None] * np.ones_like(S)
    s_lo = R.min() / tr.max()
    s_hi = R.max() / tr.min()
    if not (np.isfinite(s_lo) and np.isfinite(s_hi) and s_hi > s_lo > 0):
        return {'score': 0.0}
    # 物理可行域：s ≥ min_visible_radius / r_min（最内环必须可见）
    if min_visible_radius > 0:
        s_lo = max(s_lo, min_visible_radius / tr.min())
        if s_hi <= s_lo:
            return {'score': 0.0}
        Wm = np.where(S >= s_lo, Wm, 0.0)
    edges = np.linspace(np.log(s_lo), np.log(s_hi), n_scale_bins + 1)
    idx = np.clip(np.digitize(np.log(S.ravel()), edges) - 1, 0, n_scale_bins - 1)
    votes = np.bincount(idx, weights=Wm.ravel(), minlength=n_scale_bins)

    def _evaluate(s_hat):
        """给定尺度做旋转投票并计分。"""
        pred_r = s_hat * tr
        compat = np.abs(R[:, None] - pred_r[None, :]) <= np.maximum(
            r_tol * R[:, None], 1.0)             # 半径容差（相对或 1px）
        if not compat.any():
            return None
        dphi = Phi[:, None] - template['phi'][None, :]
        # 差角必须 wrap 到 [-π, π] 再进直方图，否则 ±2π 附近的票被
        # clip 进边缘 bin，把真解的旋转峰完全抹掉
        dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
        w_pair = W[:, None] * compat
        n_ang = 180
        ang_bins = np.linspace(-np.pi, np.pi, n_ang + 1)
        aidx = np.clip(np.digitize(dphi.ravel(), ang_bins) - 1, 0, n_ang - 1)
        avotes = np.bincount(aidx, weights=w_pair.ravel(), minlength=n_ang)
        best = avotes.argmax()
        delta = 0.5 * (ang_bins[best] + ang_bins[best + 1])

        rot = dphi - delta
        rot = np.arctan2(np.sin(rot), np.cos(rot))
        matched = compat & (np.abs(rot) <= np.deg2rad(ang_tol_deg))
        matched_fraction = float(W[matched.any(axis=1)].sum() / W.sum())
        # 模板端：落入最大实测半径内的反射是否被匹配
        inside = pred_r <= R.max() * (1 + r_tol)
        missing_fraction = float((inside & ~matched.any(axis=0)).sum()
                                 / max(inside.sum(), 1))
        score = max(0.0, matched_fraction - 0.5 * missing_fraction)
        return {'score': score, 's': float(s_hat), 'delta': float(delta),
                'matched_fraction': matched_fraction,
                'missing_fraction': missing_fraction}

    # 尺度投票只作候选生成：高阶反射常造成倍/半尺度简并，且组合
    # 投票会产生多个伪峰。取投票直方图全部局部极大值（>15% 主峰，
    # 上限 12 个）分别做完整旋转匹配，取终分最高者
    cand = []
    v_max = votes.max()
    for b in range(n_scale_bins):
        v = votes[b]
        if v < 0.15 * v_max:
            continue
        left = votes[b - 1] if b > 0 else -1.0
        right = votes[b + 1] if b < n_scale_bins - 1 else -1.0
        if v >= left and v >= right:
            cand.append(b)
    cand.sort(key=lambda b: -votes[b])
    cand = cand[:12]
    best = None
    for b in cand:
        s_hat = math.exp(0.5 * (edges[b] + edges[b + 1]))
        m = _evaluate(s_hat)
        if m is not None and (best is None or m['score'] > best['score']):
            best = m
    if best is None:
        return {'score': 0.0}
    return best


def index_diffraction_pattern(dp, center, disk_radius, bank,
                              max_spots=12, **kw):
    """对单张花样检测斑点并在模板库中择优。"""
    peaks = detect_bragg_disks(dp, disk_radius, center=center,
                               exclude_radius=0.0, max_peaks=max_spots + 1,
                               rel_threshold=0.12, abs_sigma=12.0)
    cy, cx = center
    keep = np.sqrt((peaks[:, 0] - cy) ** 2 + (peaks[:, 1] - cx) ** 2) \
        > disk_radius * 1.2
    peaks = peaks[keep][:max_spots]
    if len(peaks) < 3:
        return None
    dy = -(peaks[:, 0] - cy)          # 图像 y 向下 → 右手系取负
    dx = peaks[:, 1] - cx
    R = np.hypot(dy, dx)
    Phi = np.arctan2(dy, dx)
    W = peaks[:, 2]
    if W.max() > 0:
        W = W / W.max()

    best = None
    for ti, t in enumerate(bank):
        m = match_pattern_to_template(R, Phi, W, t,
                                      min_visible_radius=1.2 * disk_radius,
                                      **kw)
        if best is None or m['score'] > best[1]['score']:
            best = (ti, m)
    ti, m = best
    if m['score'] <= 0:
        return None
    return {'template_index': ti, 'uvw': bank[ti]['uvw'], **m}


def orientation_from_templates(datacube, center, alpha, structure,
                               max_index=2, should_stop=None, verbose=True,
                               progress=None):
    """
    逐像素衍射花样指标化 → 带轴/面内旋转/可靠性/相机常数图。

    progress : callable(str), optional - 细粒度进度回调（每 ~10% 一次，
    供 GUI 日志）；None 时退回 print。

    Returns
    -------
    dict：zone_index (int, -1=未指标化), reliability, rotation_deg,
    scale (px·Å), zone_labels（库中带轴列表）, structure
    """
    def _report(msg):
        if progress is not None:
            progress(msg)
        elif verbose:
            print(msg)

    datacube = np.asarray(datacube)
    scan_y, scan_x = datacube.shape[:2]
    bank = build_zone_axis_bank(structure, max_index=max_index)
    if not bank:
        raise ValueError("模板库为空：检查晶格参数/格子类型")
    _report(f"    zone-axis bank: {len(bank)} templates "
            f"({structure.get('name', 'custom')})")

    zone_index = np.full((scan_y, scan_x), -1, dtype=int)
    reliability = np.zeros((scan_y, scan_x))
    rotation_deg = np.full((scan_y, scan_x), np.nan)
    scale = np.full((scan_y, scan_x), np.nan)

    total = scan_y * scan_x
    done = 0
    for sy in range(scan_y):
        if should_stop is not None and should_stop():
            return {'cancelled': True}
        for sx in range(scan_x):
            hit = index_diffraction_pattern(datacube[sy, sx], center,
                                            alpha, bank)
            if hit is not None:
                zone_index[sy, sx] = hit['template_index']
                reliability[sy, sx] = hit['score']
                rotation_deg[sy, sx] = math.degrees(hit['delta'])
                scale[sy, sx] = hit['s']
            done += 1
            if done % max(1, total // 10) == 0:
                _report(f"    indexing {done}/{total}")

    return {
        'zone_index': zone_index,
        'reliability': reliability,
        'rotation_deg': rotation_deg,
        'scale': scale,
        'zone_labels': [t['uvw'] for t in bank],
        'structure': structure,
        'cancelled': False,
    }


# ========== 环检测（多晶平均花样） ==========

def detect_rings(avg_dp, n_rings=5, threshold=None, center=None):
    """Detect diffraction rings in average diffraction pattern."""
    det_y, det_x = avg_dp.shape
    if center is None:
        center = (det_y // 2, det_x // 2)
    cy, cx = center

    r = np.sqrt((np.arange(det_y)[:, None] - cy) ** 2
                + (np.arange(det_x)[None, :] - cx) ** 2)
    r_int = r.astype(int)
    max_r = min(cy, cx, det_y - cy, det_x - cx)

    radial = np.bincount(r_int.ravel(), weights=avg_dp.ravel())[:max_r]
    counts = np.bincount(r_int.ravel())[:max_r]
    radial = radial / np.maximum(counts, 1)

    radial_smooth = ndimage.gaussian_filter1d(radial, sigma=2)

    if threshold is None:
        threshold = radial_smooth.mean() + radial_smooth.std()

    from scipy.signal import find_peaks
    peaks, _ = find_peaks(radial_smooth, height=threshold, distance=5)

    rings = [{'radius': float(p), 'intensity': float(radial_smooth[p])}
             for p in peaks[:n_rings]]
    return rings, radial_smooth


# ========== 旧各向异性描述符（非晶体取向，仅作衬度参考） ==========

def orientation_from_symmetry(datacube, center, alpha, n_sectors=8):
    """
    BF 盘环形区域的角分布各向异性描述符。

    注意：这不是晶体取向。输出仅为环形强度分布的一/二次谐波，
    可用作取向衬度的定性参考。
    """
    scan_y, scan_x, det_y, det_x = datacube.shape
    cy, cx = center

    ky, kx = np.meshgrid(np.arange(det_y), np.arange(det_x), indexing='ij')
    r = np.sqrt((ky - cy) ** 2 + (kx - cx) ** 2)
    theta = np.arctan2(ky - cy, kx - cx)

    annular = ((r >= alpha * 0.3) & (r <= alpha)).astype(np.float64)

    sector_edges = np.linspace(-np.pi, np.pi, n_sectors + 1)
    sector_centers = (sector_edges[:-1] + sector_edges[1:]) / 2

    # Accumulate sector intensities with one (S x P) @ (P x sectors) matmul
    # instead of n_sectors full-cube temporaries. Pixels outside the annulus
    # (or exactly at theta = pi) belong to no sector, as in the old loop.
    sector_idx = np.full((det_y, det_x), -1, dtype=np.int64)
    annulus = annular > 0
    for i in range(n_sectors):
        in_sector = annulus & (theta >= sector_edges[i]) \
            & (theta < sector_edges[i + 1])
        sector_idx[in_sector] = i
    flat = datacube.reshape(scan_y * scan_x, det_y * det_x)
    onehot = np.zeros((det_y * det_x, n_sectors))
    valid = sector_idx.ravel() >= 0
    onehot[np.nonzero(valid)[0], sector_idx.ravel()[valid]] = 1.0
    sector_intensities = (flat @ onehot).reshape(scan_y, scan_x, n_sectors)

    total = np.maximum(np.sum(sector_intensities, axis=2, keepdims=True), 1)
    sector_norm = sector_intensities / total

    cos_component = np.sum(sector_norm
                           * np.cos(sector_centers)[np.newaxis, np.newaxis, :],
                           axis=2)
    sin_component = np.sum(sector_norm
                           * np.sin(sector_centers)[np.newaxis, np.newaxis, :],
                           axis=2)
    orientation_angle = np.arctan2(sin_component, cos_component)
    anisotropy = np.sqrt(cos_component ** 2 + sin_component ** 2)

    cos2 = np.sum(sector_norm * np.cos(2 * sector_centers)[np.newaxis,
                                                           np.newaxis, :], axis=2)
    sin2 = np.sum(sector_norm * np.sin(2 * sector_centers)[np.newaxis,
                                                           np.newaxis, :], axis=2)
    symmetry_4fold = np.sqrt(cos2 ** 2 + sin2 ** 2)
    angle_4fold = np.arctan2(sin2, cos2) / 2

    return {
        'orientation_angle': orientation_angle,
        'orientation_deg': np.degrees(orientation_angle),
        'anisotropy': anisotropy,
        'symmetry_4fold': symmetry_4fold,
        'angle_4fold': np.degrees(angle_4fold),
        'sector_intensities': sector_intensities,
        'sector_norm': sector_norm,
    }


# ========== 可视化 ==========

def plot_orientation_results(orient_data, name, output_dir):
    """带轴指标化结果可视化（模板法）。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    zi = orient_data['zone_index']
    rel = orient_data['reliability']
    rot = orient_data['rotation_deg']
    labels = orient_data['zone_labels']
    present = sorted(set(zi.ravel().tolist()) - {-1})

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    im = axes[0, 0].imshow(zi, cmap='tab20', vmin=-0.5,
                           vmax=max(len(labels) - 0.5, 0.5))
    axes[0, 0].set_title('Zone-axis index map')
    ticks = present if present else [0]
    axes[0, 0].axis('off')
    cbar = plt.colorbar(im, ax=axes[0, 0], shrink=0.8, ticks=ticks)
    cbar.ax.set_yticklabels(
        [f"[{labels[t][0]}{labels[t][1]}{labels[t][2]}]" for t in ticks])

    im = axes[0, 1].imshow(rel, cmap='inferno', vmin=0, vmax=1)
    axes[0, 1].set_title('Reliability (match score)')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], shrink=0.8)

    finite = np.isfinite(rot)
    if finite.any():
        im = axes[1, 0].imshow(np.where(finite, rot, np.nan), cmap='hsv',
                               vmin=-180, vmax=180)
        axes[1, 0].set_title('In-plane rotation (deg)')
    else:
        axes[1, 0].set_title('In-plane rotation (no solution)')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], shrink=0.8)

    sc = orient_data.get('scale')
    if sc is not None and np.isfinite(sc).any():
        im = axes[1, 1].imshow(sc, cmap='viridis')
        axes[1, 1].set_title('Camera scale s (px·Å)')
        plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
    else:
        axes[1, 1].set_title('Camera scale (no solution)')
    axes[1, 1].axis('off')

    struct = orient_data.get('structure', {})
    plt.suptitle(f"Orientation Mapping: {name} "
                 f"({struct.get('name', 'custom')}, template matching)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{name}_orientation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    np.savez(os.path.join(output_dir, f'{name}_orientation.npz'),
             zone_index=zi, reliability=rel, rotation_deg=rot,
             scale=orient_data.get('scale', np.full_like(rel, np.nan)),
             zone_labels=np.array([''.join(map(str, l)) for l in labels]))

    return save_path


def plot_symmetry_results(orient_data, name, output_dir):
    """旧环形各向异性结果可视化（定性衬度参考）。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    angle_deg = orient_data['orientation_deg']
    anisotropy = orient_data['anisotropy']

    im = axes[0, 0].imshow(angle_deg, cmap='hsv', vmin=-180, vmax=180)
    axes[0, 0].set_title('Annular anisotropy angle (NOT crystal orientation)')
    axes[0, 0].axis('off')
    plt.colorbar(im, ax=axes[0, 0], shrink=0.8)

    im = axes[0, 1].imshow(anisotropy, cmap='inferno')
    axes[0, 1].set_title('Anisotropy magnitude')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], shrink=0.8)

    sym4 = orient_data['symmetry_4fold']
    im = axes[1, 0].imshow(sym4, cmap='viridis')
    axes[1, 0].set_title('4-fold symmetry metric')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], shrink=0.8)

    sector_avg = np.mean(orient_data['sector_norm'], axis=(0, 1))
    n_sectors = len(sector_avg)
    angles = np.linspace(0, 360, n_sectors + 1)
    axes[1, 1].bar(angles[:-1], sector_avg, width=360 / n_sectors,
                   edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Angle (deg)')
    axes[1, 1].set_ylabel('Normalized intensity')
    axes[1, 1].set_title('Average sector pattern')

    plt.suptitle(f'Annular anisotropy: {name}', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{name}_anisotropy.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def run_orientation_mapping(datacube, center, alpha, method='template',
                            structure=None, name='sample', output_dir=None,
                            should_stop=None, verbose=True, progress=None):
    """
    Main orientation mapping function.

    Parameters
    ----------
    method : 'template'（真实晶体学模板指标化，默认）或
             'symmetry'（环形各向异性，定性衬度参考，非取向）
    structure : dict 或预设名，如 'Au (FCC)'；None 时用 make_structure()
                的默认（PoC 用；建议显式给定）
    progress : callable(str), optional - 细粒度进度回调（GUI 日志用）
    """
    results = {'method': method}
    avg_dp = np.mean(datacube, axis=(0, 1))
    results['rings'], results['radial_profile'] = detect_rings(
        avg_dp, center=center)

    if method == 'template':
        if structure is None:
            structure = make_structure('Au (FCC)')
        elif isinstance(structure, str):
            structure = make_structure(structure)
        orient = orientation_from_templates(datacube, center, alpha,
                                            structure,
                                            should_stop=should_stop,
                                            verbose=verbose,
                                            progress=progress)
        results['orientation'] = orient
        if orient.get('cancelled'):
            return results
        if output_dir:
            plot_orientation_results(orient, name, output_dir)
    elif method == 'symmetry':
        orient = orientation_from_symmetry(datacube, center, alpha,
                                           n_sectors=12)
        results['orientation'] = orient
        if output_dir:
            plot_symmetry_results(orient, name, output_dir)
    else:
        raise ValueError(f"未知方法 '{method}'（'template' 或 'symmetry'）")

    return results
