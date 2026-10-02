"""
batch_dpc.py - Batch DPC/iDPC analysis for all 4D-STEM DM4 datasets.

Unlike the old version, this script:
- discovers DM4 files by walking an input directory (no hard-coded offsets,
  shapes or endian guesses);
- reads dtype/byte-order from the DM4 header via ``core.dm4_io``;
- uses the vectorized robust CoM (positive-only, outlier-rejected);
- saves phase/CoM/virtual images as .npy and one summary figure per dataset.

Usage:
    python processing/batch_dpc.py --input-dir D:\\data --output-dir out
"""
import argparse
import os
import sys
import time

import numpy as np
import matplotlib

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.pipeline import prepare_dataset
from core.dpc_core import (compute_com_robust, idpc_reconstruct,
                           compute_virtual_images, plot_dpc_results)


def find_dm4_files(input_dir):
    """Recursively collect and sort all .dm4 files."""
    files = []
    for root, dirs, names in os.walk(input_dir):
        for f in names:
            if f.lower().endswith('.dm4'):
                files.append(os.path.join(root, f))
    return sorted(files)


def process_one(dm4_path, output_dir, scan_crop=128):
    """Run the DPC pipeline on one file and save results."""
    name = os.path.splitext(os.path.basename(dm4_path))[0]
    name = name.replace(' ', '_').replace('(', '').replace(')', '')
    os.makedirs(output_dir, exist_ok=True)

    print(f'\n--- {name} ---', flush=True)
    t0 = time.time()

    # Extraction, dimension fix, preprocessing and beam-centre estimation
    # are shared with the GUI via core.pipeline.
    ds = prepare_dataset(dm4_path, scan_crop=scan_crop,
                         log=lambda m: print(m, flush=True))
    data, center, alpha = ds['data'], ds['center'], ds['alpha']
    det_y, det_x = data.shape[2], data.shape[3]

    com = compute_com_robust(data, center=center, radius=alpha)
    phase = idpc_reconstruct(com)
    vimg = compute_virtual_images(data, center, alpha)

    np.save(os.path.join(output_dir, f'{name}_dpc_phase.npy'), phase)
    np.save(os.path.join(output_dir, f'{name}_com_x.npy'), com['com_x'])
    np.save(os.path.join(output_dir, f'{name}_com_y.npy'), com['com_y'])
    np.save(os.path.join(output_dir, f'{name}_bf.npy'), vimg['bf'])
    np.save(os.path.join(output_dir, f'{name}_adf.npy'), vimg['adf'])

    result = {
        'com_result': com,
        'phase': phase,
        'virtual_images': vimg,
        'metadata': {
            'label': name,
            'scan_shape': (data.shape[0], data.shape[1]),
            'det_shape': (det_y, det_x),
            'alpha_pixels': int(alpha),
            'center': center,
            'time_sec': time.time() - t0,
        },
    }
    plot_dpc_results(result,
                     save_path=os.path.join(output_dir, f'{name}_dpc.png'))
    print(f'  Done in {time.time() - t0:.1f}s')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Batch DPC analysis for DM4 4D-STEM datasets.')
    parser.add_argument('--input-dir', required=True,
                        help='Folder scanned recursively for .dm4 files.')
    parser.add_argument('--output-dir', required=True,
                        help='Folder for per-dataset results.')
    parser.add_argument('--crop', type=int, default=128,
                        help='Centre crop size in scan pixels.')
    args = parser.parse_args(argv)

    files = find_dm4_files(args.input_dir)
    if not files:
        print(f'No DM4 files found under: {args.input_dir}')
        return 1
    print(f'Found {len(files)} DM4 files')

    results = {}
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            results[name] = process_one(path, args.output_dir, args.crop)
        except Exception as exc:
            print(f'  ERROR processing {path}: {exc}')
            import traceback
            traceback.print_exc()

    if results:
        n = len(results)
        fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
        if n == 1:
            axes = axes.reshape(1, -1)
        for i, (name, res) in enumerate(results.items()):
            phase = res['phase']
            vmax = max(abs(phase.min()), abs(phase.max()))
            axes[i, 0].imshow(res['virtual_images']['adf'], cmap='gray')
            axes[i, 0].set_title(f'{name}: ADF')
            axes[i, 0].axis('off')
            axes[i, 1].imshow(res['com_result']['com_magnitude'],
                              cmap='inferno')
            axes[i, 1].set_title(f'{name}: |CoM|')
            axes[i, 1].axis('off')
            axes[i, 2].imshow(phase, cmap='viridis')
            axes[i, 2].set_title(f'{name}: iDPC Phase')
            axes[i, 2].axis('off')
            axes[i, 3].imshow(phase, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[i, 3].set_title(f'{name}: Phase +/-')
            axes[i, 3].axis('off')
        plt.tight_layout()
        summary_path = os.path.join(args.output_dir, '00_SUMMARY_ALL.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'\nSummary saved: {summary_path}')

    print(f'\nAll results saved to: {args.output_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
