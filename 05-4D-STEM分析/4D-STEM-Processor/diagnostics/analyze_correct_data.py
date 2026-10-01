"""
analyze_correct_data.py - Analyze the correctly extracted Au data.
"""
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.environ.get('STEM4D_DATA',
                      r'D:\data\4dSTEM\20260707-Au')
DATA_DIR = os.environ.get('STEM4D_ANALYSIS_DATA',
                          os.path.join(BASE, 'analysis', 'data'))
OUTPUT_DIR = os.environ.get('STEM4D_OUTPUT',
                            os.path.join(BASE, 'analysis', 'results', 'v2'))

def analyze_dataset(name, filepath):
    """Analyze a correctly extracted dataset."""
    print(f'\n{"="*70}')
    print(f'Analyzing: {name}')
    print(f'{"="*70}')
    
    data = np.load(filepath)
    scan_y, scan_x, det_y, det_x = data.shape
    print(f'Shape: {data.shape}')
    print(f'Min: {data.min():.2f}, Max: {data.max():.2f}')
    print(f'Mean: {data.mean():.2f}, Std: {data.std():.2f}')
    
    # Check total intensity per scan position
    total_intensity = np.sum(data, axis=(2, 3))
    print(f'\nTotal intensity per scan position:')
    print(f'  Min: {total_intensity.min():.2f}')
    print(f'  Max: {total_intensity.max():.2f}')
    print(f'  Mean: {total_intensity.mean():.2f}')
    print(f'  Negative positions: {np.sum(total_intensity < 0)}')
    
    # Check average diffraction pattern
    avg_dp = np.mean(data, axis=(0, 1))
    print(f'\nAverage diffraction pattern:')
    print(f'  Min: {avg_dp.min():.2f}, Max: {avg_dp.max():.2f}')
    print(f'  Mean: {avg_dp.mean():.2f}')
    print(f'  Total: {avg_dp.sum():.2f}')
    
    # Find beam center using different methods
    # Method 1: Center of mass (if total > 0)
    total = avg_dp.sum()
    if total > 0:
        y_coords, x_coords = np.mgrid[0:det_y, 0:det_x]
        com_y = np.sum(y_coords * avg_dp) / total
        com_x = np.sum(x_coords * avg_dp) / total
        print(f'\nBeam center (CoM): ({com_y:.2f}, {com_x:.2f})')
    else:
        print(f'\nTotal intensity is negative, CoM unreliable')
        com_y, com_x = det_y/2, det_x/2
    
    # Method 2: Maximum position
    max_pos = np.unravel_index(avg_dp.argmax(), avg_dp.shape)
    print(f'Maximum position: {max_pos}')
    
    # Method 3: Center of positive values only
    pos_mask = avg_dp > 0
    if pos_mask.sum() > 0:
        pos_avg = avg_dp * pos_mask
        pos_total = pos_avg.sum()
        if pos_total > 0:
            com_y_pos = np.sum(y_coords * pos_avg) / pos_total
            com_x_pos = np.sum(x_coords * pos_avg) / pos_total
            print(f'Center (positive only): ({com_y_pos:.2f}, {com_x_pos:.2f})')
    
    # Try different data processing approaches
    print(f'\n{"="*70}')
    print('Testing different processing approaches')
    print(f'{"="*70}')
    
    approaches = {}
    
    # Approach 1: Shift to positive (subtract min)
    data_shifted = data - data.min()
    approaches['shifted'] = data_shifted
    print(f'\n1. Shifted (data - min):')
    print(f'   Min: {data_shifted.min():.2f}, Max: {data_shifted.max():.2f}')
    print(f'   Mean: {data_shifted.mean():.2f}')
    
    # Approach 2: Clip negatives to zero
    data_clipped = np.maximum(data, 0)
    approaches['clipped'] = data_clipped
    print(f'\n2. Clipped (max(data, 0)):')
    print(f'   Min: {data_clipped.min():.2f}, Max: {data_clipped.max():.2f}')
    print(f'   Mean: {data_clipped.mean():.2f}')
    print(f'   Zero pixels: {np.sum(data_clipped == 0)} ({100*np.sum(data_clipped == 0)/data_clipped.size:.1f}%)')
    
    # Approach 3: Use absolute value
    data_abs = np.abs(data)
    approaches['abs'] = data_abs
    print(f'\n3. Absolute value:')
    print(f'   Min: {data_abs.min():.2f}, Max: {data_abs.max():.2f}')
    print(f'   Mean: {data_abs.mean():.2f}')
    
    # Approach 4: Square (emphasizes large values)
    data_sq = data ** 2
    approaches['squared'] = data_sq
    print(f'\n4. Squared:')
    print(f'   Min: {data_sq.min():.2f}, Max: {data_sq.max():.2f}')
    print(f'   Mean: {data_sq.mean():.2f}')
    
    # Test CoM calculation with each approach
    print(f'\n{"="*70}')
    print('Testing CoM calculation')
    print(f'{"="*70}')
    
    for app_name, app_data in approaches.items():
        # Calculate CoM for center scan position
        dp = app_data[scan_y//2, scan_x//2]
        total = dp.sum()
        if total > 0:
            y_coords, x_coords = np.mgrid[0:det_y, 0:det_x]
            com_y = np.sum(y_coords * dp) / total
            com_x = np.sum(x_coords * dp) / total
            print(f'\n{app_name}:')
            print(f'  Total intensity: {total:.2f}')
            print(f'  CoM: ({com_y:.2f}, {com_x:.2f})')
            
            # Check if CoM is reasonable (should be near center)
            offset = np.sqrt((com_y - det_y/2)**2 + (com_x - det_x/2)**2)
            print(f'  Offset from center: {offset:.2f} pixels')
        else:
            print(f'\n{app_name}: Total intensity <= 0, CoM unreliable')
    
    # Create visualization
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Row 1: Different processing approaches (average DP)
    for idx, (app_name, app_data) in enumerate(approaches.items()):
        ax = axes[0, idx]
        avg = np.mean(app_data, axis=(0, 1))
        im = ax.imshow(avg, cmap='viridis')
        ax.set_title(f'{app_name}\nmean={avg.mean():.0f}')
        plt.colorbar(im, ax=ax)
    
    # Row 2: CoM maps for shifted data
    shifted = approaches['shifted']
    
    # Calculate CoM maps
    com_y_map = np.zeros((scan_y, scan_x))
    com_x_map = np.zeros((scan_y, scan_x))
    bf_map = np.zeros((scan_y, scan_x))
    
    y_coords, x_coords = np.mgrid[0:det_y, 0:det_x]
    
    for sy in range(scan_y):
        for sx in range(scan_x):
            dp = shifted[sy, sx]
            total = dp.sum()
            bf_map[sy, sx] = total
            if total > 0:
                com_y_map[sy, sx] = np.sum(y_coords * dp) / total
                com_x_map[sy, sx] = np.sum(x_coords * dp) / total
    
    im4 = axes[1, 0].imshow(bf_map, cmap='gray')
    axes[1, 0].set_title('BF (total intensity)')
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].imshow(com_y_map, cmap='RdBu_r')
    axes[1, 1].set_title('CoM-Y')
    plt.colorbar(im5, ax=axes[1, 1])
    
    im6 = axes[1, 2].imshow(com_x_map, cmap='RdBu_r')
    axes[1, 2].set_title('CoM-X')
    plt.colorbar(im6, ax=axes[1, 2])
    
    com_mag = np.sqrt((com_y_map - com_y_map.mean())**2 + (com_x_map - com_x_map.mean())**2)
    im7 = axes[1, 3].imshow(com_mag, cmap='inferno')
    axes[1, 3].set_title('|CoM| magnitude')
    plt.colorbar(im7, ax=axes[1, 3])
    
    # Row 3: Radial profiles and histograms
    # Radial profile for different approaches
    r = np.sqrt((np.arange(det_y)[:, None] - det_y/2)**2 + (np.arange(det_x)[None, :] - det_x/2)**2)
    r_int = r.astype(int)
    
    for app_name, app_data in approaches.items():
        avg = np.mean(app_data, axis=(0, 1))
        radial = np.bincount(r_int.ravel(), weights=avg.ravel())
        counts = np.bincount(r_int.ravel())
        radial = radial / np.maximum(counts, 1)
        axes[2, 0].plot(radial, label=app_name)
    
    axes[2, 0].set_xlabel('Radius (pixels)')
    axes[2, 0].set_ylabel('Intensity')
    axes[2, 0].set_title('Radial Profiles')
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    
    # Histogram of shifted data
    axes[2, 1].hist(shifted.ravel(), bins=100, color='blue', alpha=0.7)
    axes[2, 1].set_xlabel('Intensity')
    axes[2, 1].set_ylabel('Count')
    axes[2, 1].set_title('Histogram (shifted)')
    
    # Single DP comparison
    dp_orig = data[scan_y//2, scan_x//2]
    dp_shift = shifted[scan_y//2, scan_x//2]
    
    im10 = axes[2, 2].imshow(dp_orig, cmap='RdBu_r')
    axes[2, 2].set_title(f'Original DP\nmean={dp_orig.mean():.0f}')
    plt.colorbar(im10, ax=axes[2, 2])
    
    im11 = axes[2, 3].imshow(dp_shift, cmap='viridis')
    axes[2, 3].set_title(f'Shifted DP\nmean={dp_shift.mean():.0f}')
    plt.colorbar(im11, ax=axes[2, 3])
    
    plt.suptitle(f'{name}: Data Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'{name}_correct_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {save_path}')
    
    return approaches


def main():
    datasets = {
        'Au_SI19': os.path.join(DATA_DIR, 'Au_SI19_correct.npy'),
        'Au_SI20': os.path.join(DATA_DIR, 'Au_SI20_correct.npy'),
        'Au_SI21': os.path.join(DATA_DIR, 'Au_SI21_correct.npy'),
    }
    
    for name, path in datasets.items():
        if os.path.exists(path):
            analyze_dataset(name, path)
        else:
            print(f'\nFile not found: {path}')


if __name__ == '__main__':
    main()
