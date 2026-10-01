"""
diagnose_au.py - Diagnose Au data quality and result reliability.
"""
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.dpc_core import compute_com_fast, find_alpha_from_radial

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.environ.get('STEM4D_DATA',
                      r'D:\data\4dSTEM\20260707-Au')
DATA_DIR = os.environ.get('STEM4D_ANALYSIS_DATA',
                          os.path.join(BASE, 'analysis', 'data'))
OUTPUT_DIR = os.environ.get('STEM4D_OUTPUT',
                            os.path.join(BASE, 'analysis', 'results', 'v2'))

def diagnose_dataset(name, filepath):
    """Comprehensive diagnosis of a dataset."""
    print(f'\n{"="*70}')
    print(f'Diagnosing: {name}')
    print(f'{"="*70}')
    
    data = np.load(filepath)
    scan_y, scan_x, det_y, det_x = data.shape
    
    print(f'\n[1] Basic Statistics')
    print(f'  Shape: {data.shape}')
    print(f'  Dtype: {data.dtype}')
    print(f'  Min: {data.min():.1f}')
    print(f'  Max: {data.max():.1f}')
    print(f'  Mean: {data.mean():.1f}')
    print(f'  Std: {data.std():.1f}')
    print(f'  Median: {np.median(data):.1f}')
    
    # Check for saturation
    saturated = np.sum(data >= 65535)
    total = data.size
    print(f'\n[2] Saturation Check')
    print(f'  Saturated pixels (>=65535): {saturated} ({100*saturated/total:.2f}%)')
    
    # Check for zeros/dead pixels
    zeros = np.sum(data == 0)
    print(f'  Zero pixels: {zeros} ({100*zeros/total:.2f}%)')
    
    # Average diffraction pattern
    avg_dp = np.mean(data, axis=(0, 1))
    print(f'\n[3] Average Diffraction Pattern')
    print(f'  Center value: {avg_dp[det_y//2, det_x//2]:.1f}')
    print(f'  Corner value: {avg_dp[0, 0]:.1f}')
    print(f'  Max value: {avg_dp.max():.1f} at {np.unravel_index(avg_dp.argmax(), avg_dp.shape)}')
    print(f'  Min value: {avg_dp.min():.1f}')
    
    # Background estimation (corners)
    corner_size = 3
    corners = [
        avg_dp[:corner_size, :corner_size],
        avg_dp[:corner_size, -corner_size:],
        avg_dp[-corner_size:, :corner_size],
        avg_dp[-corner_size:, -corner_size:],
    ]
    bg_estimate = np.mean([c.mean() for c in corners])
    print(f'  Background estimate (corners): {bg_estimate:.1f}')
    
    # Signal-to-background ratio
    signal = avg_dp.max() - bg_estimate
    print(f'  Signal (max - bg): {signal:.1f}')
    print(f'  Signal/Background ratio: {signal/bg_estimate:.2f}' if bg_estimate > 0 else '  S/B: N/A')
    
    # CoM analysis
    print(f'\n[4] Center of Mass Analysis')
    com_result = compute_com_fast(data)
    center = com_result['center']
    print(f'  Beam center: ({center[0]:.2f}, {center[1]:.2f})')
    print(f'  Expected center: ({det_y/2:.1f}, {det_x/2:.1f})')
    print(f'  Offset from center: ({center[0]-det_y/2:.2f}, {center[1]-det_x/2:.2f})')
    
    # CoM statistics
    com_y = com_result['com_y']
    com_x = com_result['com_x']
    print(f'\n  CoM-Y: mean={com_y.mean():.4f}, std={com_y.std():.4f}, range=[{com_y.min():.4f}, {com_y.max():.4f}]')
    print(f'  CoM-X: mean={com_x.mean():.4f}, std={com_x.std():.4f}, range=[{com_x.min():.4f}, {com_x.max():.4f}]')
    
    # Check if CoM values are reasonable (should be small fraction of detector)
    com_magnitude = np.sqrt(com_y**2 + com_x**2)
    print(f'  |CoM|: mean={com_magnitude.mean():.4f}, max={com_magnitude.max():.4f}')
    print(f'  |CoM|/detector_size: {com_magnitude.mean()/det_y*100:.2f}%')
    
    # Alpha estimation
    alpha, radial, counts = find_alpha_from_radial(data, center)
    print(f'\n[5] BF Disk Estimation')
    print(f'  Alpha: {alpha} pixels')
    print(f'  Alpha/detector: {alpha/det_y*100:.1f}%')
    print(f'  BF disk area: ~{np.pi*alpha**2:.0f} pixels ({np.pi*alpha**2/(det_y*det_x)*100:.1f}% of detector)')
    
    # Radial profile analysis
    print(f'\n[6] Radial Profile')
    print(f'  Peak intensity: {radial.max():.1f} at r={np.argmax(radial)}')
    print(f'  Intensity at alpha: {radial[alpha-1] if alpha > 0 else 0:.1f}')
    print(f'  Intensity at edge: {radial[-1]:.1f}')
    print(f'  Peak/Edge ratio: {radial.max()/radial[-1]:.1f}' if radial[-1] > 0 else '  Peak/Edge: N/A')
    
    # Scan position analysis
    print(f'\n[7] Scan Position Analysis')
    total_intensity = np.sum(data, axis=(2, 3))
    print(f'  Total intensity per scan position:')
    print(f'    Mean: {total_intensity.mean():.1f}')
    print(f'    Std: {total_intensity.std():.1f}')
    print(f'    Variation: {total_intensity.std()/total_intensity.mean()*100:.2f}%')
    
    # Check for scan artifacts
    row_means = np.mean(total_intensity, axis=1)
    col_means = np.mean(total_intensity, axis=0)
    print(f'  Row intensity variation: {row_means.std()/row_means.mean()*100:.2f}%')
    print(f'  Col intensity variation: {col_means.std()/col_means.mean()*100:.2f}%')
    
    # Create diagnostic plot
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Average diffraction pattern
    im0 = axes[0, 0].imshow(avg_dp, cmap='viridis')
    axes[0, 0].set_title(f'Avg Diffraction Pattern\nCenter: ({center[0]:.1f}, {center[1]:.1f})')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Log scale
    im1 = axes[0, 1].imshow(np.log10(avg_dp + 1), cmap='inferno')
    axes[0, 1].set_title('Avg DP (log10)')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Radial profile
    axes[0, 2].plot(radial, 'b-', label='Radial avg')
    axes[0, 2].axvline(alpha, color='r', linestyle='--', label=f'alpha={alpha}')
    axes[0, 2].set_xlabel('Radius (pixels)')
    axes[0, 2].set_ylabel('Intensity')
    axes[0, 2].set_title('Radial Profile')
    axes[0, 2].legend()
    
    # Histogram of pixel values
    axes[0, 3].hist(avg_dp.ravel(), bins=100, color='blue', alpha=0.7)
    axes[0, 3].set_xlabel('Intensity')
    axes[0, 3].set_ylabel('Count')
    axes[0, 3].set_title('Pixel Value Distribution')
    
    # CoM maps
    im4 = axes[1, 0].imshow(com_y, cmap='RdBu_r')
    axes[1, 0].set_title('CoM-Y')
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].imshow(com_x, cmap='RdBu_r')
    axes[1, 1].set_title('CoM-X')
    plt.colorbar(im5, ax=axes[1, 1])
    
    im6 = axes[1, 2].imshow(com_magnitude, cmap='inferno')
    axes[1, 2].set_title('|CoM|')
    plt.colorbar(im6, ax=axes[1, 2])
    
    # Total intensity map
    im7 = axes[1, 3].imshow(total_intensity, cmap='viridis')
    axes[1, 3].set_title('Total Intensity')
    plt.colorbar(im7, ax=axes[1, 3])
    
    # BF image
    bf_intensity = com_result['bf_intensity']
    im8 = axes[2, 0].imshow(bf_intensity, cmap='gray')
    axes[2, 0].set_title('BF Intensity')
    plt.colorbar(im8, ax=axes[2, 0])
    
    # Single diffraction patterns
    axes[2, 1].imshow(data[0, 0], cmap='viridis')
    axes[2, 1].set_title('DP at (0,0)')
    
    axes[2, 2].imshow(data[scan_y//2, scan_x//2], cmap='viridis')
    axes[2, 2].set_title('DP at center')
    
    axes[2, 3].imshow(data[-1, -1], cmap='viridis')
    axes[2, 3].set_title('DP at (-1,-1)')
    
    plt.suptitle(f'Diagnostics: {name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'{name}_diagnostics.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  Diagnostic plot saved: {save_path}')
    
    # Recommendations
    print(f'\n[8] Recommendations')
    issues = []
    if saturated > 0:
        issues.append(f'  - {saturated} saturated pixels detected')
    if bg_estimate > avg_dp.max() * 0.5:
        issues.append(f'  - High background ({bg_estimate:.0f}), consider background subtraction')
    if alpha < 3:
        issues.append(f'  - Very small BF disk (alpha={alpha}), limited angular information')
    if com_magnitude.mean() > det_y * 0.1:
        issues.append(f'  - Large CoM shifts, possible beam drift or strong fields')
    if total_intensity.std()/total_intensity.mean() > 0.2:
        issues.append(f'  - High intensity variation across scan, possible thickness variation')
    
    if issues:
        print('  Potential issues:')
        for issue in issues:
            print(issue)
    else:
        print('  No major issues detected')
    
    return {
        'shape': data.shape,
        'center': center,
        'alpha': alpha,
        'bg_estimate': bg_estimate,
        'com_std': (com_y.std(), com_x.std()),
    }


def main():
    print('='*70)
    print('Au Data Quality Diagnostics')
    print('='*70)
    
    datasets = {
        'Au_SI19': os.path.join(DATA_DIR, 'Au_crop128.npy'),
        'Au_SI20': os.path.join(DATA_DIR, 'Au_SI_data_20_crop128.npy'),
        'Au_SI21': os.path.join(DATA_DIR, 'Au_SI_data_21_crop128.npy'),
    }
    
    results = {}
    for name, path in datasets.items():
        if os.path.exists(path):
            results[name] = diagnose_dataset(name, path)
    
    # Summary comparison
    print(f'\n{"="*70}')
    print('Summary Comparison')
    print(f'{"="*70}')
    print(f'{"Dataset":<12} {"Shape":<20} {"Center":<18} {"Alpha":<8} {"Background":<12}')
    print('-'*70)
    for name, r in results.items():
        print(f'{name:<12} {str(r["shape"]):<20} ({r["center"][0]:.1f},{r["center"][1]:.1f}){"":<8} {r["alpha"]:<8} {r["bg_estimate"]:.1f}')


if __name__ == '__main__':
    main()
