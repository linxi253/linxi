"""
check_raw_data.py - Check raw DM4 data to diagnose extraction issues.
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import dm4_io

def main():
    # Check the raw DM4 file
    BASE = os.environ.get('STEM4D_DATA',
                          r'D:\data\4dSTEM\20260707-Au')
    dm4_path = os.path.join(BASE, 'SI data (19)', '007_STEM SI.dm4')

    # Read the offset and dtype from the DM4 header itself.
    if os.path.exists(dm4_path):
        meta = dm4_io.read_dm4_metadata(dm4_path)
        offset = meta['offset']
        print(f'Header dtype: {meta["dtype_name"]}, offset: {offset}')
    else:
        print(f'File not found: {dm4_path}')
        raise SystemExit(1)

    print('='*70)
    print('Raw DM4 Data Check')
    print('='*70)
    print(f'File: {dm4_path}')
    print(f'File size: {os.path.getsize(dm4_path):,} bytes')
    print(f'Offset: {offset}')

    # Read data at offset with different interpretations
    with open(dm4_path, 'rb') as f:
        f.seek(offset)
        raw_bytes = f.read(10000)

    print(f'\n[1] Data at offset {offset}:')

    # Try different dtypes
    for dtype_name, dtype in [
        ('>u2 (BE uint16)', np.dtype('>u2')),
        ('<u2 (LE uint16)', np.dtype('<u2')),
        ('>i2 (BE int16)', np.dtype('>i2')),
        ('<i2 (LE int16)', np.dtype('<i2')),
        ('>u4 (BE uint32)', np.dtype('>u4')),
        ('<u4 (LE uint32)', np.dtype('<u4')),
        ('>f4 (BE float32)', np.dtype('>f4')),
        ('<f4 (LE float32)', np.dtype('<f4')),
    ]:
        try:
            arr = np.frombuffer(raw_bytes[:1000], dtype=dtype)
            print(f'  {dtype_name}: first 10 = {arr[:10]}')
            print(f'    min={arr.min():.2f}, max={arr.max():.2f}, mean={arr.mean():.2f}')
        except:
            pass

    # Check if there's a pattern
    print(f'\n[2] Checking for patterns...')
    arr_u2_be = np.frombuffer(raw_bytes[:2000], dtype=np.dtype('>u2'))
    arr_u2_le = np.frombuffer(raw_bytes[:2000], dtype=np.dtype('<u2'))

    print(f'  BE uint16: unique values in first 100: {len(np.unique(arr_u2_be[:100]))}')
    print(f'  LE uint16: unique values in first 100: {len(np.unique(arr_u2_le[:100]))}')

    # Check histogram
    print(f'\n[3] Value distribution (BE uint16, first 5000 values):')
    arr = np.frombuffer(raw_bytes[:10000], dtype=np.dtype('>u2'))
    hist, edges = np.histogram(arr, bins=10)
    for i in range(len(hist)):
        print(f'  [{edges[i]:.0f}-{edges[i+1]:.0f}]: {hist[i]}')

    # Check the extracted data
    print(f'\n[4] Checking extracted data...')
    extracted_path = os.path.join(BASE, 'analysis', 'data', 'Au_crop128.npy')
    if os.path.exists(extracted_path):
        data = np.load(extracted_path)
        print(f'  Shape: {data.shape}')
        print(f'  Min: {data.min():.1f}, Max: {data.max():.1f}')
        print(f'  Mean: {data.mean():.1f}, Std: {data.std():.1f}')
    
        # Check a single diffraction pattern
        dp = data[64, 64]  # center scan position
        print(f'\n  Single DP at (64,64):')
        print(f'    Min: {dp.min():.1f}, Max: {dp.max():.1f}')
        print(f'    Mean: {dp.mean():.1f}')
        print(f'    Center pixel: {dp[16, 16]:.1f}')
        print(f'    Corner pixel: {dp[0, 0]:.1f}')
    
        # Check if there's actual structure
        print(f'\n  DP structure check:')
        print(f'    Center 4x4 mean: {dp[14:18, 14:18].mean():.1f}')
        print(f'    Edge 4x4 mean: {dp[0:4, 0:4].mean():.1f}')
        print(f'    Ratio: {dp[14:18, 14:18].mean() / dp[0:4, 0:4].mean():.3f}')

    # Try reading with different offsets to find the actual data
    print(f'\n[5] Searching for correct data offset...')
    with open(dm4_path, 'rb') as f:
        # Check a few offsets around the expected one
        for test_offset in [offset - 1000, offset - 100, offset, offset + 100, offset + 1000]:
            f.seek(test_offset)
            test_data = np.frombuffer(f.read(2000), dtype=np.dtype('>u2'))
            mean_val = test_data.mean()
            std_val = test_data.std()
            # Good data should have reasonable std (not all same value)
            print(f'  Offset {test_offset}: mean={mean_val:.1f}, std={std_val:.1f}, first5={test_data[:5]}')

    # Check DM4 header for data type info
    print(f'\n[6] DM4 Header Analysis...')
    with open(dm4_path, 'rb') as f:
        header = f.read(100)
        print(f'  First 20 bytes (hex): {header[:20].hex()}')
        print(f'  First 20 bytes (int): {list(header[:20])}')
    
        # DM4 magic number check
        if header[:4] == b'\x00\x00\x00\x03':
            print('  DM4 magic number: OK (version 3)')
        elif header[:4] == b'\x00\x00\x00\x04':
            print('  DM4 magic number: OK (version 4)')
        else:
            print(f'  DM4 magic number: Unknown ({header[:4].hex()})')

    print(f'\n[7] Checking standard data for comparison...')
    STANDARD_DIR = os.environ.get('STEM4D_STANDARD',
                                  r'D:\data\4dSTEM\standard\data')
    standard_path = os.path.join(STANDARD_DIR, 'Cu foil-grain boundary.dm4')
    if os.path.exists(standard_path):
        with open(standard_path, 'rb') as f:
            # Find data offset (this is known to work)
            # From previous analysis: offset=156783 for Cu foil
            f.seek(156783)
            cu_data = np.frombuffer(f.read(2000), dtype=np.dtype('>u2'))
            print(f'  Cu foil at offset 156783: mean={cu_data.mean():.1f}, std={cu_data.std():.1f}')
            print(f'    first 10: {cu_data[:10]}')


if __name__ == '__main__':
    main()
