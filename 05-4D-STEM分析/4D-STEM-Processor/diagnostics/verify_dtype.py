"""
verify_dtype.py - Verify the data type used to decode an Au DM4 file.

The authoritative decode is the DM4 metadata data-type code (via
``core.dm4_io.read_dm4_metadata``), which also cross-checks the tag tree's
declared byte count and the file size before anything memmaps the data block.

The "interpretations" section below is a *diagnostic comparison* only: it shows
what the first bytes would look like under several dtypes. It never decides how
the real datacube is read.

Environment:
    STEM4D_DM4      (required) full path of the DM4 file to inspect
    STEM4D_OUTPUT   (optional) directory for the PNG; falls back to
                    <STEM4D_DATA>/analysis/results/v2
"""
import os
import sys

import numpy as np

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


def _output_dir() -> str:
    """Resolve the PNG directory without ever evaluating an undefined name.

    Order: ``STEM4D_OUTPUT``, then ``<STEM4D_DATA>/analysis/results/v2``.
    Neither configured is an actionable error (the public script deliberately
    carries no machine default).
    """
    explicit = os.environ.get('STEM4D_OUTPUT', '').strip()
    if explicit:
        return explicit
    base = os.environ.get('STEM4D_DATA', '').strip()
    if base:
        return os.path.join(base, 'analysis', 'results', 'v2')
    raise SystemExit(
        '[verify_dtype] 未配置输出目录：请设置 STEM4D_OUTPUT（推荐）或 STEM4D_DATA。\n'
        '  PowerShell: $env:STEM4D_OUTPUT = \'<输出目录>\'\n'
        '  或         $env:STEM4D_DATA   = \'<数据根目录>\'\n'
        '脚本不内置任何本机默认路径。')


def _reject_unsupported_complex(dtype, dtype_name: str) -> None:
    """Complex data is refused outright rather than silently reduced to real."""
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        raise SystemExit(
            f'[verify_dtype] {dtype_name} 是复数数据，本诊断不支持：脚本不会偷偷取实部。\n'
            '请先把实部/虚部分别导出为独立文件后再检查。')


def _diagnostic_interpretations(raw_bytes: bytes, n_items: int = 10) -> None:
    """Show how the leading bytes would read under several dtypes.

    Diagnostic only -- the real datacube is decoded from the metadata dtype and
    this function never influences its length.

    Each interpretation consumes only the largest prefix that is a whole number
    of *its* itemsize. When the data block is not a multiple of that itemsize
    (e.g. a 1-byte uint8 block read as int16/float32) the interpretation is
    skipped with an explanation instead of raising ``ValueError``.
    """
    print('\nTesting data type interpretations (diagnostic only):')
    print('-' * 70)
    if len(raw_bytes) == 0:
        print('  (no data bytes available to compare)')
        return
    interpretations = [
        ('>u2 (BE uint16)', np.dtype('>u2')),
        ('<u2 (LE uint16)', np.dtype('<u2')),
        ('>i2 (BE int16)', np.dtype('>i2')),
        ('<i2 (LE int16)', np.dtype('<i2')),
        ('<f4 (LE float32)', np.dtype('<f4')),
    ]
    for name, dtype in interpretations:
        itemsize = dtype.itemsize
        usable = len(raw_bytes) - (len(raw_bytes) % itemsize)
        if usable < itemsize:
            print(f'\n{name}:')
            print(f'  (skipped: needs a multiple of {itemsize} bytes, '
                  f'have {len(raw_bytes)})')
            continue
        arr = np.frombuffer(raw_bytes[:usable], dtype=dtype)
        print(f'\n{name}:')
        print(f'  First {min(n_items, arr.size)}: {arr[:n_items]}')
        print(f'  Min: {arr.min():.2f}, Max: {arr.max():.2f}')
        print(f'  Mean: {arr.mean():.2f}, Std: {arr.std():.2f}')
        if arr.min() >= 0 and arr.max() < 100000:
            print('  Assessment: Positive values, reasonable range')
        elif arr.min() < 0 < arr.max():
            print('  Assessment: Mixed positive/negative '
                  '(possible background-subtracted)')
        else:
            print('  Assessment: Suspicious values')


def _safe_window(size: int, fraction: int = 4, maximum: int = 4) -> int:
    """Window length that cannot exceed ``size`` (no hardcoded detector size)."""
    return max(1, min(maximum, max(1, size // fraction)))


def main():
    dm4_path = _require_env('STEM4D_DM4', '待检查的原始 DM4 文件完整路径')

    if not os.path.exists(dm4_path):
        print(f'File not found: {dm4_path}')
        raise SystemExit(1)

    # Validates the type code against the tag tree's declared byte count and the
    # file size, so a drifted table or truncated file fails loudly here.
    meta = dm4_io.read_dm4_metadata(dm4_path)
    offset = meta['offset']
    dtype = np.dtype(meta['dtype'])
    dtype_name = meta.get('dtype_name') or dm4_io.dtype_display_name(dtype)
    det_y, det_x = int(meta['det_y']), int(meta['det_x'])
    scan_y, scan_x = int(meta['scan_y']), int(meta['scan_x'])

    _reject_unsupported_complex(dtype, dtype_name)

    print('=' * 70)
    print('Data Type Verification')
    print('=' * 70)
    print(f'File      : {dm4_path}')
    print(f'Metadata  : dtype code -> {dtype_name} ({dtype.str}), '
          f'{dtype.itemsize} bytes/element')
    print(f'Scan      : {scan_y} x {scan_x}')
    print(f'Detector  : {det_y} x {det_x}')

    # Diagnostic comparison on the leading bytes of the data block (bounded by
    # the real block size so we never read past EOF).
    block_bytes = det_y * det_x * scan_y * scan_x * dtype.itemsize
    probe_len = int(min(2000, block_bytes))
    with open(dm4_path, 'rb') as handle:
        handle.seek(offset)
        raw_bytes = handle.read(probe_len)
    _diagnostic_interpretations(raw_bytes)

    # Real datacube: metadata dtype, small scan crop, actual detector size.
    test_scan = max(1, min(4, scan_y, scan_x))
    print('\n' + '=' * 70)
    print(f'Reading real datacube with metadata dtype {dtype_name}: '
          f'{test_scan} x {test_scan} scan positions')
    print('=' * 70)
    data_test = dm4_io.extract_4d_data(dm4_path, meta, scan_crop=test_scan)

    print(f'\nExtracted test datacube: {data_test.shape} '
          f'(scan_y, scan_x, det_y, det_x)')
    print(f'Min: {data_test.min():.1f}, Max: {data_test.max():.1f}')
    print(f'Mean: {data_test.mean():.1f}, Std: {data_test.std():.1f}')

    cy, cx = det_y // 2, det_x // 2
    win = _safe_window(min(det_y, det_x))
    dp = data_test[0, 0]
    print(f'\nSingle DP at (0,0):')
    print(f'  Min: {dp.min():.1f}, Max: {dp.max():.1f}')
    print(f'  Mean: {dp.mean():.1f}')
    center = dp[cy - win // 2:cy - win // 2 + win, cx - win // 2:cx - win // 2 + win]
    corner = dp[0:win, 0:win]
    print(f'  Center {win}x{win} ({cy},{cx}): {center.mean():.1f}')
    print(f'  Corner {win}x{win}: {corner.mean():.1f}')
    corner_mean = corner.mean()
    if corner_mean:
        print(f'  Ratio (center/corner): {center.mean() / corner_mean:.3f}')
    else:
        print('  Ratio (center/corner): n/a (corner mean is 0)')

    avg_dp = np.mean(data_test, axis=(0, 1))
    print(f'\nAverage DP:')
    print(f'  Min: {avg_dp.min():.1f}, Max: {avg_dp.max():.1f}')
    print(f'  Center value ({cy},{cx}): {avg_dp[cy, cx]:.1f}')
    print(f'  Corner value (0,0): {avg_dp[0, 0]:.1f}')

    # Radial profile over the actual detector size.
    yy, xx = np.indices((det_y, det_x))
    r_int = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    radial_sum = np.bincount(r_int.ravel(), weights=avg_dp.ravel())
    radial_cnt = np.bincount(r_int.ravel())
    radial = radial_sum / np.maximum(radial_cnt, 1)

    print(f'\nRadial profile:')
    for i in range(0, min(len(radial), 16), 2):
        print(f'  r={i}: {radial[i]:.1f}')

    alpha = None
    if radial.max() > 0:
        threshold = 0.3 * radial.max()
        candidates = np.where(radial < threshold)[0]
        alpha = int(candidates[0]) if len(candidates) > 0 else len(radial) - 1
        edge = radial[-1]
        print(f'\nEstimated alpha: {alpha} pixels')
        if edge:
            print(f'Peak/Edge ratio: {radial.max() / edge:.2f}')
        else:
            print('Peak/Edge ratio: n/a (edge intensity is 0)')

    # Resolve the destination *before* building the figure: a configuration
    # error must not leave a figure behind.
    output_dir = _output_dir()

    # Save visualization
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    try:
        im0 = axes[0, 0].imshow(avg_dp, cmap='viridis')
        axes[0, 0].set_title(f'Average DP ({dtype_name})')
        plt.colorbar(im0, ax=axes[0, 0])

        im1 = axes[0, 1].imshow(np.log10(avg_dp - avg_dp.min() + 1), cmap='inferno')
        axes[0, 1].set_title('Average DP (log)')
        plt.colorbar(im1, ax=axes[0, 1])

        im2 = axes[0, 2].imshow(data_test[min(2, test_scan - 1), min(2, test_scan - 1)],
                                cmap='viridis')
        axes[0, 2].set_title(f'Single DP at '
                             f'({min(2, test_scan - 1)},{min(2, test_scan - 1)})')
        plt.colorbar(im2, ax=axes[0, 2])

        axes[1, 0].plot(radial, 'b-o')
        axes[1, 0].set_xlabel('Radius (pixels)')
        axes[1, 0].set_ylabel('Intensity')
        axes[1, 0].set_title('Radial Profile')
        axes[1, 0].grid(True)

        axes[1, 1].hist(avg_dp.ravel(), bins=100, color='blue', alpha=0.7)
        axes[1, 1].set_xlabel('Intensity')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Pixel Distribution')

        bf = np.sum(data_test, axis=(2, 3))
        im5 = axes[1, 2].imshow(bf, cmap='gray')
        axes[1, 2].set_title('BF Image (sum)')
        plt.colorbar(im5, ax=axes[1, 2])

        plt.suptitle(f'Data Type Verification: {dtype_name} ({dtype.str})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'dtype_verification.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    finally:
        plt.close(fig)
    print(f'\nSaved: {save_path}')


if __name__ == '__main__':
    main()
