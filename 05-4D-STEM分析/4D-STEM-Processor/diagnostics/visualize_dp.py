"""
visualize_dp.py - Visualize diffraction patterns to understand data issues.
"""
import numpy as np
import os
import matplotlib

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.environ.get('STEM4D_DATA', '').strip()


def _resolve_output_dir():
    """输出目录必须由环境变量提供（缺省报错并打印用法）。"""
    output_dir = os.environ.get('STEM4D_OUTPUT', '').strip() or \
        (os.path.join(BASE, 'analysis', 'results', 'v2') if BASE else '')
    if not output_dir:
        raise SystemExit(
            '[visualize_dp] 未配置输出目录：请设置 STEM4D_OUTPUT'
            '（或数据根目录 STEM4D_DATA）后重跑，例如：\n'
            '  PowerShell: $env:STEM4D_OUTPUT = \'<输出目录>\'\n'
            '  cmd:        set STEM4D_OUTPUT=<输出目录>')
    return output_dir


OUTPUT_DIR = ''


def main():
    global OUTPUT_DIR
    OUTPUT_DIR = _resolve_output_dir()
    # Load the test data
    test_data = np.load(os.path.join(OUTPUT_DIR, 'test_data_correct.npy'))
    print(f'Test data shape: {test_data.shape}')
    print(f'Min: {test_data.min():.2f}, Max: {test_data.max():.2f}')

    # Get average DP
    avg_dp = np.mean(test_data, axis=(0, 1))
    print(f'\nAverage DP:')
    print(f'  Shape: {avg_dp.shape}')
    print(f'  Min: {avg_dp.min():.2f}, Max: {avg_dp.max():.2f}')
    print(f'  Mean: {avg_dp.mean():.2f}')

    # Check different regions
    print(f'\nRegion analysis:')
    print(f'  Center pixel (16,16): {avg_dp[16, 16]:.2f}')
    print(f'  Center 4x4: {avg_dp[14:18, 14:18].mean():.2f}')
    print(f'  Corner (0,0): {avg_dp[0, 0]:.2f}')
    print(f'  Corner 4x4: {avg_dp[0:4, 0:4].mean():.2f}')
    print(f'  Edge (0,16): {avg_dp[0, 16]:.2f}')
    print(f'  Edge (16,0): {avg_dp[16, 0]:.2f}')

    # Check if beam is centered
    # Find the actual maximum
    max_pos = np.unravel_index(avg_dp.argmax(), avg_dp.shape)
    print(f'\nMax position: {max_pos}, value: {avg_dp.max():.2f}')

    # Find center of mass of the DP
    total = np.sum(avg_dp)
    if total > 0:
        y_coords, x_coords = np.mgrid[0:32, 0:32]
        com_y = np.sum(y_coords * avg_dp) / total
        com_x = np.sum(x_coords * avg_dp) / total
        print(f'Center of mass: ({com_y:.2f}, {com_x:.2f})')
    else:
        print(f'Total intensity is negative or zero: {total:.2f}')

    # Check histogram
    print(f'\nHistogram:')
    hist, edges = np.histogram(avg_dp.ravel(), bins=20)
    for i in range(len(hist)):
        print(f'  [{edges[i]:.0f}, {edges[i+1]:.0f}): {hist[i]}')

    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    # 1. Average DP (raw)
    im0 = axes[0, 0].imshow(avg_dp, cmap='RdBu_r')
    axes[0, 0].set_title(f'Average DP (raw)\nmin={avg_dp.min():.0f}, max={avg_dp.max():.0f}')
    plt.colorbar(im0, ax=axes[0, 0])

    # 2. Average DP (shifted to positive)
    dp_shifted = avg_dp - avg_dp.min()
    im1 = axes[0, 1].imshow(dp_shifted, cmap='viridis')
    axes[0, 1].set_title(f'Average DP (shifted)\nmin=0, max={dp_shifted.max():.0f}')
    plt.colorbar(im1, ax=axes[0, 1])

    # 3. Log scale (shifted)
    im2 = axes[0, 2].imshow(np.log10(dp_shifted + 1), cmap='inferno')
    axes[0, 2].set_title('Average DP (log10)')
    plt.colorbar(im2, ax=axes[0, 2])

    # 4. Single DP
    dp_single = test_data[0, 0]
    im3 = axes[0, 3].imshow(dp_single, cmap='RdBu_r')
    axes[0, 3].set_title(f'Single DP (0,0)\nmin={dp_single.min():.0f}, max={dp_single.max():.0f}')
    plt.colorbar(im3, ax=axes[0, 3])

    # 5. Radial profile
    r = np.sqrt((np.arange(32)[:, None] - 16)**2 + (np.arange(32)[None, :] - 16)**2)
    r_int = r.astype(int)
    radial = np.bincount(r_int.ravel(), weights=avg_dp.ravel())
    counts = np.bincount(r_int.ravel())
    radial = radial / np.maximum(counts, 1)

    axes[1, 0].plot(radial, 'b-o', label='Raw')
    radial_shifted = np.bincount(r_int.ravel(), weights=dp_shifted.ravel()) / np.maximum(counts, 1)
    axes[1, 0].plot(radial_shifted, 'r-s', label='Shifted')
    axes[1, 0].set_xlabel('Radius (pixels)')
    axes[1, 0].set_ylabel('Intensity')
    axes[1, 0].set_title('Radial Profile')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 6. Histogram
    axes[1, 1].hist(avg_dp.ravel(), bins=100, color='blue', alpha=0.7)
    axes[1, 1].axvline(0, color='red', linestyle='--', label='Zero')
    axes[1, 1].set_xlabel('Intensity')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Pixel Distribution')
    axes[1, 1].legend()

    # 7. BF image (sum over detector)
    bf = np.sum(test_data, axis=(2, 3))
    im6 = axes[1, 2].imshow(bf, cmap='gray')
    axes[1, 2].set_title(f'BF Image (sum)\nmin={bf.min():.0f}, max={bf.max():.0f}')
    plt.colorbar(im6, ax=axes[1, 2])

    # 8. ADF image (outer detector)
    adf = np.sum(test_data[:, :, 20:, 20:], axis=(2, 3))
    im7 = axes[1, 3].imshow(adf, cmap='gray')
    axes[1, 3].set_title(f'ADF Image (corner)\nmin={adf.min():.0f}, max={adf.max():.0f}')
    plt.colorbar(im7, ax=axes[1, 3])

    # 9-12. Individual scan positions
    for idx, (sy, sx) in enumerate([(0,0), (0,3), (3,0), (3,3)]):
        ax = axes[2, idx]
        dp = test_data[sy, sx]
        im = ax.imshow(dp, cmap='RdBu_r')
        ax.set_title(f'DP ({sy},{sx})\nmean={dp.mean():.0f}')
        plt.colorbar(im, ax=ax)

    plt.suptitle('Au SI19 Diffraction Pattern Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'dp_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {save_path}')

    # Now check if the issue is with the data extraction
    # The original extraction used >u2 (big-endian uint16)
    # But the correct format is <i2 (little-endian int16)

    # Let's also check what the original extracted data looks like
    print('\n' + '='*70)
    print('Comparing with original extraction')
    print('='*70)

    orig_path = os.path.join(BASE, 'analysis', 'data', 'Au_crop128.npy')
    if os.path.exists(orig_path):
        orig_data = np.load(orig_path)
        print(f'Original data shape: {orig_data.shape}')
        print(f'Original min: {orig_data.min():.2f}, max: {orig_data.max():.2f}')
        print(f'Original mean: {orig_data.mean():.2f}')
    
        # Check a DP
        orig_dp = orig_data[64, 64]
        print(f'\nOriginal DP at (64,64):')
        print(f'  Min: {orig_dp.min():.2f}, Max: {orig_dp.max():.2f}')
        print(f'  Mean: {orig_dp.mean():.2f}')
        print(f'  Center 4x4: {orig_dp[14:18, 14:18].mean():.2f}')
        print(f'  Corner 4x4: {orig_dp[0:4, 0:4].mean():.2f}')
    
        # The original data was extracted with wrong dtype
        # Let's see if we can recover the correct data by reinterpreting
        # Original: >u2 (big-endian uint16)
        # Correct: <i2 (little-endian int16)
    
        # Actually, the bytes are the same, just interpreted differently
        # Let's check the relationship
        print(f'\nByte-level comparison:')
        # Take a small region from both
        test_correct = test_data[0, 0, 0, 0]  # First pixel
        # The original extraction would have read this differently
    
        # The issue is that the original extraction used:
        # - dtype: >u2 (big-endian unsigned)
        # - But the file is little-endian signed
    
        # This means every 2-byte value was byte-swapped AND interpreted as unsigned
        # Let's verify by checking if we can convert
        print(f'Correct value at (0,0,0,0): {test_correct:.2f}')


if __name__ == '__main__':
    main()
