"""
verify_dtype.py - Verify correct data type for Au DM4 files.
"""
import numpy as np
import os
import sys

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import dm4_io


def _require_env(name: str, hint: str) -> str:
    """缺环境变量即报错并打印用法；脚本不内置任何本机默认路径。"""
    value = os.environ.get(name, '').strip()
    if not value:
        raise SystemExit(
            f'[verify_dtype] 缺少环境变量 {name}（{hint}）。\n'
            f'用法：先设置环境变量再重跑，例如：\n'
            f'  PowerShell: $env:{name} = \'<路径>\'\n'
            f'  cmd:        set {name}=<路径>')
    return value


def main():
    dm4_path = _require_env('STEM4D_DM4', '待检查的原始 DM4 文件完整路径')

    if not os.path.exists(dm4_path):
        print(f'File not found: {dm4_path}')
        raise SystemExit(1)
    meta = dm4_io.read_dm4_metadata(dm4_path)
    offset = meta['offset']
    det_z2, det_z = meta['det_y'], meta['det_x']
    scan_y, scan_x = meta['scan_y'], meta['scan_x']
    elem_size = meta['dtype'].itemsize

    print('='*70)
    print('Data Type Verification')
    print('='*70)

    # From DM4 header: byte_order=1 means little-endian
    # From metadata: dataType=2 means float32 (int16 came from the old
    # mis-sourced dtype table; corrected 2026-10, see core/dm4_io.py)

    # Test different interpretations
    with open(dm4_path, 'rb') as f:
        f.seek(offset)
        raw_bytes = f.read(10000)

    print('\nTesting data type interpretations:')
    print('-'*70)

    interpretations = [
        ('>u2 (BE uint16) - CURRENT', np.dtype('>u2')),
        ('<u2 (LE uint16)', np.dtype('<u2')),
        ('>i2 (BE int16)', np.dtype('>i2')),
        ('<i2 (LE int16) - CORRECT?', np.dtype('<i2')),
        ('<f4 (LE float32)', np.dtype('<f4')),
    ]

    for name, dtype in interpretations:
        arr = np.frombuffer(raw_bytes[:2000], dtype=dtype)
        print(f'\n{name}:')
        print(f'  First 10: {arr[:10]}')
        print(f'  Min: {arr.min():.2f}, Max: {arr.max():.2f}')
        print(f'  Mean: {arr.mean():.2f}, Std: {arr.std():.2f}')
    
        # Check if this looks like diffraction data
        # Good diffraction data should have:
        # - Positive values (or small negative if background subtracted)
        # - Clear structure (not random noise)
        # - Reasonable dynamic range
        if arr.min() >= 0 and arr.max() < 100000:
            print(f'  Assessment: Positive values, reasonable range')
        elif arr.min() < 0 and arr.max() > 0:
            print(f'  Assessment: Mixed positive/negative (possible background-subtracted)')
        else:
            print(f'  Assessment: Suspicious values')

    # Now test with actual 4D-STEM structure
    print('\n' + '='*70)
    print('Testing 4D-STEM structure with <i2 (LE int16)')
    print('='*70)

    # Read a small datacube: 4x4 scan positions, full 32x32 detector
    # Shape in file: (det_z2=32, det_z=32, scan_y=2048, scan_x=2048)
    # We need to read it in the correct order

    # Read first 4x4 scan positions
    test_scan = 4
    data_test = np.zeros((test_scan, test_scan, det_z2, det_z), dtype=np.float32)

    with open(dm4_path, 'rb') as f:
        for dz2 in range(det_z2):
            for dz in range(det_z):
                base = offset + elem_size * (dz2 * det_z * scan_y * scan_x + dz * scan_y * scan_x)
                for sy in range(test_scan):
                    for sx in range(test_scan):
                        pos = base + elem_size * (sy * scan_x + sx)
                        f.seek(pos)
                        val = np.frombuffer(f.read(elem_size), dtype=np.dtype('<i2'))[0]
                        data_test[sy, sx, dz2, dz] = val

    print(f'\nExtracted test datacube: {data_test.shape}')
    print(f'Min: {data_test.min():.1f}, Max: {data_test.max():.1f}')
    print(f'Mean: {data_test.mean():.1f}, Std: {data_test.std():.1f}')

    # Check a single diffraction pattern
    dp = data_test[0, 0]
    print(f'\nSingle DP at (0,0):')
    print(f'  Min: {dp.min():.1f}, Max: {dp.max():.1f}')
    print(f'  Mean: {dp.mean():.1f}')
    print(f'  Center 4x4: {dp[14:18, 14:18].mean():.1f}')
    print(f'  Corner 4x4: {dp[0:4, 0:4].mean():.1f}')
    print(f'  Ratio (center/corner): {dp[14:18, 14:18].mean() / dp[0:4, 0:4].mean():.3f}')

    # Check if there's a clear BF disk
    avg_dp = np.mean(data_test, axis=(0, 1))
    print(f'\nAverage DP:')
    print(f'  Min: {avg_dp.min():.1f}, Max: {avg_dp.max():.1f}')
    print(f'  Center value: {avg_dp[16, 16]:.1f}')
    print(f'  Corner value: {avg_dp[0, 0]:.1f}')

    # Radial profile
    cy, cx = 16, 16
    r = np.sqrt((np.arange(det_z2)[:, None] - cy)**2 + (np.arange(det_z)[None, :] - cx)**2)
    r_int = r.astype(int)
    radial = np.bincount(r_int.ravel(), weights=avg_dp.ravel())
    counts = np.bincount(r_int.ravel())
    radial = radial / np.maximum(counts, 1)

    print(f'\nRadial profile:')
    for i in range(0, 16, 2):
        print(f'  r={i}: {radial[i]:.1f}')

    # Find alpha (where intensity drops)
    if radial.max() > 0:
        threshold = 0.3 * radial.max()
        alpha_candidates = np.where(radial < threshold)[0]
        alpha = alpha_candidates[0] if len(alpha_candidates) > 0 else 16
        print(f'\nEstimated alpha: {alpha} pixels')
        print(f'Peak/Edge ratio: {radial.max() / radial[-1]:.2f}')

    # Save visualization
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Average DP
    im0 = axes[0, 0].imshow(avg_dp, cmap='viridis')
    axes[0, 0].set_title('Average DP (<i2)')
    plt.colorbar(im0, ax=axes[0, 0])

    # Log scale
    im1 = axes[0, 1].imshow(np.log10(avg_dp - avg_dp.min() + 1), cmap='inferno')
    axes[0, 1].set_title('Average DP (log)')
    plt.colorbar(im1, ax=axes[0, 1])

    # Single DP
    im2 = axes[0, 2].imshow(data_test[2, 2], cmap='viridis')
    axes[0, 2].set_title('Single DP at (2,2)')
    plt.colorbar(im2, ax=axes[0, 2])

    # Radial profile
    axes[1, 0].plot(radial, 'b-o')
    axes[1, 0].set_xlabel('Radius (pixels)')
    axes[1, 0].set_ylabel('Intensity')
    axes[1, 0].set_title('Radial Profile')
    axes[1, 0].grid(True)

    # Histogram
    axes[1, 1].hist(avg_dp.ravel(), bins=100, color='blue', alpha=0.7)
    axes[1, 1].set_xlabel('Intensity')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Pixel Distribution')

    # BF image (sum over detector)
    bf = np.sum(data_test, axis=(2, 3))
    im5 = axes[1, 2].imshow(bf, cmap='gray')
    axes[1, 2].set_title('BF Image (sum)')
    plt.colorbar(im5, ax=axes[1, 2])

    plt.suptitle('Data Type Verification: <i2 (LE int16)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    OUTPUT_DIR = os.environ.get('STEM4D_OUTPUT',
                                os.path.join(BASE, 'analysis', 'results', 'v2'))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, 'dtype_verification.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {save_path}')


if __name__ == '__main__':
    main()
