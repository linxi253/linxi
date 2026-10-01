"""
extract_au_correct.py - Extract 4D-STEM data from DM4 files.

Uses ``core.dm4_io`` so the dtype and byte order come from the DM4 header
itself (ncempy), never from a file-size heuristic. Each file is centre-cropped
to ``--crop`` scan positions and saved as ``{name}_correct.npy``.

Usage:
    python processing/extract_au_correct.py
    python processing/extract_au_correct.py --base-dir D:\\data --crop 128
    python processing/extract_au_correct.py file1.dm4 file2.dm4 --out-dir out
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import dm4_io


DEFAULT_DATASETS = {
    'Au_SI19': os.path.join('SI data (19)', '007_STEM SI.dm4'),
    'Au_SI20': os.path.join('SI data (20)', '008_STEM SI.dm4'),
    'Au_SI21': os.path.join('SI data (21)', '009_STEM SI.dm4'),
}


def extract_with_dm4io(dm4_path, scan_crop=128, outpath=None):
    """Extract a centre-cropped datacube and report its statistics."""
    print(f'\n{"=" * 70}')
    print(f'Extracting: {os.path.basename(dm4_path)}')
    print(f'{"=" * 70}')

    meta = dm4_io.read_dm4_metadata(dm4_path)
    obj = meta['objects'][meta['4d_index']]
    print(f"4D-STEM object index: {meta['4d_index']}")
    print(f"  Scan: {meta['scan_y']}x{meta['scan_x']}")
    print(f"  Detector: {meta['det_y']}x{meta['det_x']}")
    print(f"  dtype: {meta['dtype_name']} ({meta['dtype']}), "
          f"offset: {meta['offset']}")
    print(f"  Objects in file: {meta['numObjects']} "
          f"(ndim={obj['ndim']})")

    actual_crop = min(int(scan_crop), meta['scan_y'], meta['scan_x'])
    print(f'Crop region: central {actual_crop}x{actual_crop} scan positions')

    t0 = time.time()
    cropped = dm4_io.extract_4d_data(dm4_path, meta, scan_crop)
    print(f'Extraction time: {time.time() - t0:.1f}s')

    print(f'\nExtracted data:')
    print(f'  Shape: {cropped.shape}')
    print(f'  Dtype: {cropped.dtype}')
    print(f'  Min: {cropped.min():.2f}')
    print(f'  Max: {cropped.max():.2f}')
    print(f'  Mean: {cropped.mean():.2f}')
    print(f'  Std: {cropped.std():.2f}')

    neg_pct = 100.0 * np.sum(cropped < 0) / cropped.size
    print(f'  Negative pixels: {neg_pct:.2f}%')
    if neg_pct > 1:
        print('  WARNING: significant negative values - the data appears to '
              'be background-subtracted.')

    if outpath:
        os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
        np.save(outpath, cropped)
        print(f'\nSaved: {outpath} '
              f'({os.path.getsize(outpath) / 1024 ** 2:.1f} MB)')
    return cropped


def main(argv=None):
    default_base = os.environ.get(
        'STEM4D_DATA', r'D:\data\4dSTEM\20260707-Au')
    parser = argparse.ArgumentParser(
        description='Extract 4D-STEM datacubes from DM4 files.')
    parser.add_argument('files', nargs='*',
                        help='Explicit DM4 files (default: known Au datasets '
                             'under --base-dir).')
    parser.add_argument('--base-dir', default=default_base,
                        help='Base data directory (default: STEM4D_DATA env '
                             'or the original F:\\ path).')
    parser.add_argument('--out-dir', default=None,
                        help='Output directory (default: '
                             '<base>/analysis/data).')
    parser.add_argument('--crop', type=int, default=128,
                        help='Centre crop size in scan pixels.')
    args = parser.parse_args(argv)

    out_dir = args.out_dir or os.path.join(args.base_dir, 'analysis', 'data')
    os.makedirs(out_dir, exist_ok=True)

    if args.files:
        datasets = {os.path.splitext(os.path.basename(p))[0]: p
                    for p in args.files}
    else:
        datasets = {name: os.path.join(args.base_dir, rel)
                    for name, rel in DEFAULT_DATASETS.items()}

    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f'\nFile not found, skipping: {path}')
            continue
        outpath = os.path.join(out_dir, f'{name}_correct.npy')
        extract_with_dm4io(path, scan_crop=args.crop, outpath=outpath)

    print(f'\nAll extractions written to: {out_dir}')


if __name__ == '__main__':
    main()
