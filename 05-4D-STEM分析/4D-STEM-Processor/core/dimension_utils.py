"""
dimension_utils.py - Utilities for verifying and fixing 4D-STEM data dimensions.

The DM4 format can store 4D-STEM data in different dimension orders depending on
the acquisition software and detector. This module provides tools to:
1. Verify the correct dimension order
2. Detect and fix dimension swaps
3. Validate data quality after dimension correction
"""
import numpy as np
from scipy import ndimage


def verify_dimensions(data, expected_scan=None, expected_det=None):
    """
    Verify that data dimensions are correctly assigned.
    
    Parameters
    ----------
    data : ndarray
        4D data array with shape (a, b, c, d)
    expected_scan : tuple, optional
        Expected scan dimensions (scan_y, scan_x)
    expected_det : tuple, optional
        Expected detector dimensions (det_y, det_x)
    
    Returns
    -------
    dict
        Verification results including:
        - is_correct: bool, whether dimensions appear correct
        - bf_image: ndarray, bright field image
        - avg_dp: ndarray, average diffraction pattern
        - bf_contrast: float, contrast of BF image
        - bf_peaks: int, number of peaks in BF image
        - dp_range: tuple, intensity range of average DP
        - recommendation: str, suggested action
    """
    a, b, c, d = data.shape
    
    # Interpretation 1: (scan_y, scan_x, det_y, det_x)
    bf1 = np.sum(data, axis=(2, 3))  # Sum over detector
    avg_dp1 = np.mean(data, axis=(0, 1))  # Mean over scan
    
    # Interpretation 2: (det_y, det_x, scan_y, scan_x) - swapped
    bf2 = np.sum(data, axis=(0, 1))  # Sum over "detector" (axes 0,1)
    avg_dp2 = np.mean(data, axis=(2, 3))  # Mean over "scan" (axes 2,3)
    
    # Analyze both interpretations
    result1 = _analyze_bf_image(bf1, "Interpretation 1")
    result2 = _analyze_bf_image(bf2, "Interpretation 2")
    
    # Determine which is correct
    # A correct BF image should have:
    # 1. Higher contrast (std/mean)
    # 2. More peaks (atomic columns)
    # 3. Clear periodicity in FFT
    
    score1 = result1['contrast'] * 10 + result1['n_peaks'] + result1['fft_ratio'] / 10
    score2 = result2['contrast'] * 10 + result2['n_peaks'] + result2['fft_ratio'] / 10

    is_correct = score1 >= score2

    common = {
        'score_scan_first': score1,
        'score_det_first': score2,
    }

    if is_correct:
        return {
            'is_correct': True,
            'bf_image': bf1,
            'avg_dp': avg_dp1,
            'bf_contrast': result1['contrast'],
            'bf_peaks': result1['n_peaks'],
            'dp_range': (avg_dp1.min(), avg_dp1.max()),
            'fft_ratio': result1['fft_ratio'],
            'score': score1,
            'recommendation': 'Dimensions appear correct',
            'interpretation': '(scan_y, scan_x, det_y, det_x)',
            **common,
        }
    else:
        return {
            'is_correct': False,
            'bf_image': bf2,
            'avg_dp': avg_dp2,
            'bf_contrast': result2['contrast'],
            'bf_peaks': result2['n_peaks'],
            'dp_range': (avg_dp2.min(), avg_dp2.max()),
            'fft_ratio': result2['fft_ratio'],
            'score': score2,
            'recommendation': 'Dimensions appear SWAPPED! Use data.transpose(2, 3, 0, 1)',
            'interpretation': '(det_y, det_x, scan_y, scan_x) -> needs transpose',
            **common,
        }


def _analyze_bf_image(bf, name):
    """Analyze a BF image for periodic structure.

    The image is min-shifted before scoring: background-subtracted data can
    have ~50% negative pixels, and scoring raw values (contrast = std/mean)
    collapses to 0 whenever mean <= 0, which makes the swap heuristic a coin
    flip exactly for the datasets that need it most.
    """
    bf = np.asarray(bf, dtype=np.float64)
    bf_shifted = bf - bf.min()

    # Contrast
    mean = bf_shifted.mean()
    contrast = bf_shifted.std() / mean if mean > 0 else 0

    # Peak detection
    bf_smooth = ndimage.gaussian_filter(bf_shifted, sigma=1)
    local_max = ndimage.maximum_filter(bf_smooth, size=5)
    threshold = bf_smooth.mean() + bf_smooth.std()
    peaks = (bf_smooth == local_max) & (bf_smooth > threshold)
    n_peaks = np.sum(peaks)

    # FFT periodicity check
    fft = np.fft.fft2(bf_shifted)
    fft_mag = np.abs(np.fft.fftshift(fft))
    cy, cx = bf.shape[0]//2, bf.shape[1]//2
    fft_mag[cy-2:cy+2, cx-2:cx+2] = 0  # Remove DC
    fft_ratio = fft_mag.max() / fft_mag.mean() if fft_mag.mean() > 0 else 0

    return {
        'contrast': contrast,
        'n_peaks': n_peaks,
        'fft_ratio': fft_ratio
    }


def fix_dimensions(data, method='auto', swap_margin=1.25):
    """
    Fix data dimensions if they are swapped.

    Parameters
    ----------
    data : ndarray
        4D data array
    method : str
        'auto': automatically detect and fix
        'swap': always swap dimensions
        'none': never swap
    swap_margin : float
        In 'auto' mode the data is only transposed when the swapped
        interpretation scores decisively higher (score_det_first >=
        swap_margin * score_scan_first; calibrated so the synthetic
        forward-inverse test's true swap at ratio ~1.46 still passes).
        The DM4 header already fixes the layout, so the heuristic must
        only override it on strong evidence: on low-contrast
        (amorphous/vacuum) scans a plain score comparison is a coin flip,
        and a wrong transpose of a shape-asymmetric cube silently destroys
        every downstream result (and inflates memory up to 16x).

    Returns
    -------
    ndarray
        Data with correct dimensions (scan_y, scan_x, det_y, det_x)
    dict
        Information about the fix applied, including both heuristic
        scores so the decision can be audited from saved metadata.
    """
    if method == 'none':
        return data, {'swapped': False, 'reason': 'method=none'}

    if method == 'swap':
        fixed = data.transpose(2, 3, 0, 1)
        return fixed, {'swapped': True, 'reason': 'method=swap'}

    # Auto detection (verify_dimensions is expensive; compute it at most
    # once per interpretation and reuse the result)
    result = verify_dimensions(data)
    score_scan = result['score_scan_first']
    score_det = result['score_det_first']

    info = {
        'score_scan_first': score_scan,
        'score_det_first': score_det,
    }

    if score_scan >= score_det:
        return data, {
            'swapped': False,
            'reason': 'Dimensions verified correct',
            'score': score_scan,
            **info,
        }

    if score_det < swap_margin * score_scan:
        # Ambiguous evidence: keep the header-derived layout. The DM4
        # header already fixes the dimension order; the heuristic must
        # only override it on strong evidence, otherwise low-contrast
        # (amorphous/vacuum) scans get silently transposed — which both
        # destroys every downstream result and can inflate memory by up
        # to 16x on shape-asymmetric cubes.
        return data, {
            'swapped': False,
            'reason': (f'Heuristic prefers swap but not decisively '
                       f'(score_det_first={score_det:.4f} < '
                       f'{swap_margin} x score_scan_first={score_scan:.4f});'
                       f' keeping DM4 header layout'),
            'score': score_scan,
            **info,
        }

    fixed = data.transpose(2, 3, 0, 1)
    return fixed, {
        'swapped': True,
        'reason': 'Dimensions were swapped',
        'score': score_det,
        **info,
    }


def validate_data_quality(data, name="Data"):
    """
    Validate 4D-STEM data quality.
    
    Parameters
    ----------
    data : ndarray
        4D data array (scan_y, scan_x, det_y, det_x)
    name : str
        Name for reporting
    
    Returns
    -------
    dict
        Quality metrics
    """
    scan_y, scan_x, det_y, det_x = data.shape
    
    # Basic statistics
    stats = {
        'name': name,
        'shape': data.shape,
        'dtype': str(data.dtype),
        'min': float(data.min()),
        'max': float(data.max()),
        'mean': float(data.mean()),
        'std': float(data.std()),
    }
    
    # Negative values
    neg_count = np.sum(data < 0)
    stats['negative_pixels'] = int(neg_count)
    stats['negative_fraction'] = float(neg_count / data.size)
    
    # Total intensity per scan position
    total_intensity = np.sum(data, axis=(2, 3))
    stats['total_intensity_min'] = float(total_intensity.min())
    stats['total_intensity_max'] = float(total_intensity.max())
    stats['total_intensity_mean'] = float(total_intensity.mean())
    stats['negative_scan_positions'] = int(np.sum(total_intensity < 0))
    
    # BF image analysis
    bf = np.sum(data, axis=(2, 3))
    bf_analysis = _analyze_bf_image(bf, "BF")
    stats['bf_contrast'] = bf_analysis['contrast']
    stats['bf_peaks'] = bf_analysis['n_peaks']
    stats['bf_fft_ratio'] = bf_analysis['fft_ratio']
    
    # Average DP analysis
    avg_dp = np.mean(data, axis=(0, 1))
    stats['avg_dp_min'] = float(avg_dp.min())
    stats['avg_dp_max'] = float(avg_dp.max())
    stats['avg_dp_mean'] = float(avg_dp.mean())
    
    # Center vs corner ratio (BF disk detection)
    cy, cx = det_y // 2, det_x // 2
    center_val = avg_dp[max(0,cy-5):cy+5, max(0,cx-5):cx+5].mean()
    corner_val = avg_dp[0:10, 0:10].mean()
    stats['center_corner_ratio'] = float(center_val / corner_val) if corner_val > 0 else 0
    
    # Quality assessment
    issues = []
    if stats['negative_fraction'] > 0.1:
        issues.append(f"High negative fraction: {stats['negative_fraction']:.1%}")
    if stats['negative_scan_positions'] > 0:
        issues.append(f"Negative total intensity at {stats['negative_scan_positions']} positions")
    if stats['bf_peaks'] < 10:
        issues.append(f"Few peaks in BF image: {stats['bf_peaks']}")
    if stats['center_corner_ratio'] < 1.2:
        issues.append(f"Low center/corner ratio: {stats['center_corner_ratio']:.2f}")
    
    stats['issues'] = issues
    stats['quality'] = 'GOOD' if len(issues) == 0 else 'WARNING' if len(issues) < 3 else 'POOR'
    
    return stats


def print_quality_report(stats):
    """Print a formatted quality report."""
    print(f"\n{'='*70}")
    print(f"Data Quality Report: {stats['name']}")
    print(f"{'='*70}")
    print(f"Shape: {stats['shape']}")
    print(f"Dtype: {stats['dtype']}")
    print(f"Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    print(f"Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
    print(f"\nNegative pixels: {stats['negative_pixels']} ({stats['negative_fraction']:.2%})")
    print(f"Negative scan positions: {stats['negative_scan_positions']}")
    print(f"\nBF image:")
    print(f"  Contrast: {stats['bf_contrast']:.3f}")
    print(f"  Peaks: {stats['bf_peaks']}")
    print(f"  FFT ratio: {stats['bf_fft_ratio']:.2f}")
    print(f"\nAverage DP:")
    print(f"  Range: [{stats['avg_dp_min']:.2f}, {stats['avg_dp_max']:.2f}]")
    print(f"  Center/Corner ratio: {stats['center_corner_ratio']:.2f}")
    print(f"\nQuality: {stats['quality']}")
    if stats['issues']:
        print(f"Issues:")
        for issue in stats['issues']:
            print(f"  - {issue}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    # Test with DPC Demo data
    import os
    
    base = os.environ.get('STEM4D_DATA',
                          r'D:\data\4dSTEM\20260707-Au')
    data_path = os.path.join(base, 'analysis', 'data',
                             'DPC_Demo_Data_crop128.npy')
    if os.path.exists(data_path):
        data = np.load(data_path)
        print(f"Loaded data: {data.shape}")
        
        # Verify dimensions
        result = verify_dimensions(data)
        print(f"\nDimension verification:")
        print(f"  Is correct: {result['is_correct']}")
        print(f"  Recommendation: {result['recommendation']}")
        print(f"  BF contrast: {result['bf_contrast']:.3f}")
        print(f"  BF peaks: {result['bf_peaks']}")
        
        # Fix if needed
        fixed_data, fix_info = fix_dimensions(data, method='auto')
        print(f"\nFix applied: {fix_info}")
        print(f"Fixed shape: {fixed_data.shape}")
        
        # Validate quality
        stats = validate_data_quality(fixed_data, "DPC Demo (fixed)")
        print_quality_report(stats)
