"""
peak_pairs.py - Peak pairs analysis for 4D-STEM data.

Identifies pairs of diffraction peaks and computes their spacing/direction.
Useful for lattice spacing measurement and local structure analysis.
"""
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks


def _fold_angle_difference(ang_diff_deg):
    """把两峰对方向的夹角绝对值 |Δ|（度，∈[0,360]）按无向语义折叠到 [0,90]。

    语义（定明）：峰对 = 两衍射斑连线，无方向性——θ 与 θ+180° 视为
    同一方向（方向模 180），故折叠区间为 [0,90] 而非 [0,180]。

    必须「先对 180 取模、再取 min(d, 180-d)」：若对 |Δ|>180° 直接算
    180-|Δ| 会得到负值（Δ=190° → -10°，Δ=270° → -90°，Δ=350° → -170°），
    负值乘 0.1 权重后变成负惩罚，令方向完全不匹配的峰对反而得分更低、
    被 argmin 选为最佳匹配。Δ=190° 与 Δ=350° 折叠后均为 10°。
    """
    d = np.asarray(ang_diff_deg, dtype=float) % 180.0
    return np.minimum(d, 180.0 - d)


def _robust_threshold(dp, threshold_frac):
    """组合阈值：max 的 fraction 与 mean+2σ 取较大者。

    背景扣除数据 max 可能 ≈ 0，纯比例阈值会落到噪声里选出海量伪峰；
    mean+2σ 兜底。完全无正信号时返回 None（调用方应跳过该图）。
    """
    if dp.max() <= 0:
        return None
    return max(threshold_frac * dp.max(), dp.mean() + 2.0 * dp.std())


def find_peaks_in_dp(dp, threshold=None, min_distance=5, max_peaks=20):
    """
    Find peaks in a single diffraction pattern.
    
    Returns list of (y, x, intensity) tuples.
    """
    if threshold is None:
        threshold = dp.mean() + 2 * dp.std()
    
    # Local maximum detection
    local_max = ndimage.maximum_filter(dp, size=min_distance)
    peaks_mask = (dp == local_max) & (dp > threshold)
    
    # Get coordinates
    coords = np.argwhere(peaks_mask)
    intensities = [dp[c[0], c[1]] for c in coords]
    
    # Sort by intensity
    sorted_idx = np.argsort(intensities)[::-1][:max_peaks]
    
    peaks = []
    for idx in sorted_idx:
        y, x = coords[idx]
        peaks.append((float(y), float(x), float(intensities[idx])))
    
    return peaks


def compute_peak_pairs(peaks, center, max_distance=None):
    """
    Compute all pairs of peaks and their properties.
    
    Parameters
    ----------
    peaks : list of (y, x, intensity)
    center : tuple (cy, cx) - beam center
    max_distance : float - maximum pair distance
    
    Returns
    -------
    list of dicts with pair properties
    """
    n = len(peaks)
    pairs = []
    
    for i in range(n):
        for j in range(i+1, n):
            y1, x1, I1 = peaks[i]
            y2, x2, I2 = peaks[j]
            
            # Distance between peaks
            dy = y2 - y1
            dx = x2 - x1
            distance = np.sqrt(dy**2 + dx**2)
            
            if max_distance and distance > max_distance:
                continue
            
            # Direction (angle)
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Midpoint
            mid_y = (y1 + y2) / 2
            mid_x = (x1 + x2) / 2
            
            # Distance from center to midpoint
            mid_r = np.sqrt((mid_y - center[0])**2 + (mid_x - center[1])**2)
            
            # g-vector (reciprocal space vector)
            g_magnitude = distance  # in pixels
            
            pairs.append({
                'peak1': (y1, x1, I1),
                'peak2': (y2, x2, I2),
                'distance': distance,
                'angle': angle,
                'midpoint': (mid_y, mid_x),
                'mid_radius': mid_r,
                'g_magnitude': g_magnitude,
                'intensity_product': I1 * I2,
            })
    
    # Sort by intensity product
    pairs.sort(key=lambda p: p['intensity_product'], reverse=True)
    return pairs


def peak_pairs_mapping(datacube, center, alpha, n_peaks=10,
                        threshold_frac=0.3, max_pairs=20, should_stop=None):
    """
    Perform peak pairs analysis across all scan positions.

    Parameters
    ----------
    datacube : ndarray (scan_y, scan_x, det_y, det_x)
    center : tuple (cy, cx)
    alpha : float
    n_peaks : int - max peaks per DP
    threshold_frac : float - threshold as fraction of max
    max_pairs : int - max pairs to track
    should_stop : callable, optional - cooperative cancellation, checked
        per scan row; returns {'cancelled': True} when triggered

    Returns
    -------
    dict with pair statistics and maps, or {'cancelled': True}
    """
    scan_y, scan_x, det_y, det_x = datacube.shape

    # Find reference peaks from average DP
    avg_dp = np.mean(datacube, axis=(0, 1))
    threshold = _robust_threshold(avg_dp, threshold_frac)
    ref_peaks = (find_peaks_in_dp(avg_dp, threshold=threshold,
                                  max_peaks=n_peaks)
                 if threshold is not None else [])

    if len(ref_peaks) < 2:
        return None

    # Compute reference pairs
    ref_pairs = compute_peak_pairs(ref_peaks, center, max_distance=alpha*2)
    ref_pairs = ref_pairs[:max_pairs]

    if len(ref_pairs) == 0:
        return None

    # Track pairs across scan
    n_pairs = len(ref_pairs)
    pair_distances = np.zeros((scan_y, scan_x, n_pairs))
    pair_angles = np.zeros((scan_y, scan_x, n_pairs))
    pair_intensities = np.zeros((scan_y, scan_x, n_pairs))

    # Reference pair properties as arrays for vectorized matching
    ref_dist = np.array([rp['distance'] for rp in ref_pairs])[:, None]
    ref_ang = np.array([rp['angle'] for rp in ref_pairs])[:, None]

    for sy in range(scan_y):
        if should_stop is not None and should_stop():
            return {'cancelled': True}
        for sx in range(scan_x):
            dp = datacube[sy, sx]
            thr = _robust_threshold(dp, threshold_frac)
            if thr is None:
                continue
            peaks = find_peaks_in_dp(dp, threshold=thr, max_peaks=n_peaks)

            if len(peaks) < 2:
                continue
            local_pairs = compute_peak_pairs(peaks, center, max_distance=alpha*2)
            if not local_pairs:
                continue

            # Score every (reference, local) pair at once:
            # score = |dist diff| + 0.1 * folded angle diff (degrees,
            # undirected: folded to [0,90] by direction mod 180 — see
            # _fold_angle_difference; a bare min(d,180-d) here would go
            # negative for |Δ|>180° and rank anti-parallel pairs best).
            lp_dist = np.array([lp['distance'] for lp in local_pairs])
            lp_ang = np.array([lp['angle'] for lp in local_pairs])
            lp_int = np.array([lp['intensity_product'] for lp in local_pairs])

            ang_diff = _fold_angle_difference(
                np.abs(lp_ang[None, :] - ref_ang))
            scores = np.abs(lp_dist[None, :] - ref_dist) + ang_diff * 0.1

            best = scores.argmin(axis=1)
            best_scores = scores[np.arange(n_pairs), best]
            ok = best_scores < 5
            pair_distances[sy, sx, ok] = lp_dist[best[ok]]
            pair_angles[sy, sx, ok] = lp_ang[best[ok]]
            pair_intensities[sy, sx, ok] = lp_int[best[ok]]

    # Statistics (circular mean/std for the angular quantities: a plain
    # mean of e.g. 179 deg and -179 deg would report 0 deg)
    pair_stats = []
    for pi in range(n_pairs):
        d = pair_distances[:, :, pi]
        valid = d > 0
        if valid.sum() > 0:
            ang = np.deg2rad(pair_angles[:, :, pi][valid])
            c, s = np.cos(ang).mean(), np.sin(ang).mean()
            mean_angle = float(np.degrees(np.arctan2(s, c)))
            resultant = min(1.0, float(np.hypot(c, s)))
            std_angle = float(np.degrees(
                np.sqrt(-2.0 * np.log(max(resultant, 1e-12)))))
            std_angle = min(std_angle, 180.0)
            pair_stats.append({
                'mean_distance': float(d[valid].mean()),
                'std_distance': float(d[valid].std()),
                'mean_angle': mean_angle,
                'std_angle': std_angle,
                'coverage': float(valid.sum()) / (scan_y * scan_x),
                'ref_distance': ref_pairs[pi]['distance'],
                'ref_angle': ref_pairs[pi]['angle'],
            })

    return {
        'reference_peaks': ref_peaks,
        'reference_pairs': ref_pairs,
        'pair_distances': pair_distances,
        'pair_angles': pair_angles,
        'pair_intensities': pair_intensities,
        'pair_stats': pair_stats,
        'n_pairs': n_pairs,
    }


def plot_peak_pairs_results(pp_data, name, output_dir):
    """Visualize peak pairs analysis."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    n_pairs = min(pp_data['n_pairs'], 6)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i in range(min(n_pairs, 6)):
        ax = axes[i // 3, i % 3]
        d = pp_data['pair_distances'][:, :, i]
        valid = d > 0
        
        if valid.sum() > 0:
            im = ax.imshow(d, cmap='viridis')
            stats = pp_data['pair_stats'][i]
            ax.set_title(f"Pair {i+1}: d={stats['mean_distance']:.1f}±{stats['std_distance']:.2f} px\n"
                        f"θ={stats['mean_angle']:.1f}° (cov={stats['coverage']:.0%})")
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.set_title(f'Pair {i+1}: No data')
        ax.axis('off')
    
    plt.suptitle(f'Peak Pairs Analysis: {name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{name}_peak_pairs.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save numeric results alongside the figure.
    np.save(os.path.join(output_dir, f'{name}_peak_pair_distances.npy'),
            pp_data['pair_distances'])
    np.save(os.path.join(output_dir, f'{name}_peak_pair_angles.npy'),
            pp_data['pair_angles'])
    with open(os.path.join(output_dir, f'{name}_peak_pairs_metadata.json'),
              'w', encoding='utf-8') as fj:
        json.dump({'pair_stats': pp_data['pair_stats'],
                   'n_pairs': int(pp_data['n_pairs'])},
                  fj, indent=2, ensure_ascii=False)
    
    return save_path
