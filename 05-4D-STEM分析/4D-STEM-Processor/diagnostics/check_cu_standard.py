"""
check_cu_standard.py - Verify Cu foil standard data quality.
"""
import numpy as np
import os
from ncempy.io import dm

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _require_env(name: str, hint: str) -> str:
    """缺环境变量即报错并打印用法；脚本不内置任何本机默认路径。"""
    value = os.environ.get(name, '').strip()
    if not value:
        raise SystemExit(
            f'[check_cu_standard] 缺少环境变量 {name}（{hint}）。\n'
            f'用法：先设置环境变量再重跑，例如：\n'
            f'  PowerShell: $env:{name} = \'<路径>\'\n'
            f'  cmd:        set {name}=<路径>')
    return value


def main():
    # Cu foil standard data
    standard_dir = _require_env('STEM4D_STANDARD', '标准样品数据目录')
    cu_path = os.path.join(standard_dir, 'Cu foil-grain boundary.dm4')

    print('='*70)
    print('Cu Foil Standard Data Quality Check')
    print('='*70)

    # Open with ncempy
    f = dm.fileDM(cu_path)
    print(f'\nFile: {cu_path}')
    print(f'Number of objects: {f.numObjects}')
    print(f'Data shapes: {f.dataShape}')
    print(f'Data types: {f.dataType}')

    # Find 4D-STEM object
    for i in range(f.numObjects):
        ndim = f.dataShape[i]
        print(f'\nObject {i}: ndim={ndim}, dtype={f.dataType[i]}')
        print(f'  xSize={f.xSize[i]}, ySize={f.ySize[i]}')
        print(f'  zSize={f.zSize[i]}, zSize2={f.zSize2[i]}')
    
        if ndim == 4 or (f.zSize[i] > 1 and f.zSize2[i] > 1):
            print(f'  -> 4D-STEM data')
        
            # Get offset and dtype
            offset = f.dataOffset[i]
            dtype_code = f.dataType[i]

            # DM4 image dataType mapping (image data body semantics).
            # Source: ncempy.io.dm._DM2NPDataTypes / Gatan dm4io.h
            # GatanDataType enum. NOT the tag-level encoded-type table
            # (_EncodedTypeDTypes): that one maps code 2 to int16 and
            # code 10 to uint8, which silently corrupts image data.
            DM4_DTYPES = {
                1: np.int16, 2: np.float32, 3: np.complex64,
                6: np.uint8, 7: np.int32, 9: np.int8,
                10: np.uint16, 11: np.uint32, 12: np.float64,
                13: np.complex128,
            }
            if dtype_code not in DM4_DTYPES:
                raise RuntimeError(
                    f'Unsupported DM4 dataType code {dtype_code}; supported '
                    f'codes: {sorted(DM4_DTYPES)}')
            base_dtype = DM4_DTYPES[dtype_code]

            # The tag tree declares the byte count of the data block; it
            # must match element count x itemsize of the mapped dtype,
            # otherwise decoding would silently produce garbage.
            n_elements = (f.xSize[i] * f.ySize[i] * f.zSize[i] * f.zSize2[i])
            data_sizes = getattr(f, 'dataSize', None)
            if data_sizes is not None and i < len(data_sizes):
                declared = int(data_sizes[i])
                if declared != n_elements * np.dtype(base_dtype).itemsize:
                    raise RuntimeError(
                        f'dataType {dtype_code} -> {np.dtype(base_dtype)} '
                        f'implies {n_elements * np.dtype(base_dtype).itemsize} '
                        f'bytes but the tag tree declares {declared}')
        
            # Check byte order
            with open(cu_path, 'rb') as fp:
                fp.seek(12)
                byte_order = int.from_bytes(fp.read(4), 'big')
                endian = '<' if byte_order == 1 else '>'
        
            print(f'  Byte order: {byte_order} ({"little" if byte_order == 1 else "big"}-endian)')
            print(f'  Base dtype: {base_dtype}')
        
            # Create memmap
            det_y = f.zSize2[i]
            det_x = f.zSize[i]
            scan_y = f.ySize[i]
            scan_x = f.xSize[i]
        
            if base_dtype == np.uint8:
                full_dtype = np.dtype('u1')  # uint8 has no endianness
            elif base_dtype == np.int16:
                full_dtype = np.dtype(f'{endian}i2')
            elif base_dtype == np.uint16:
                full_dtype = np.dtype(f'{endian}u2')
            else:
                full_dtype = np.dtype(f'{endian}{np.dtype(base_dtype).str[1:]}')
        
            print(f'  Full dtype: {full_dtype}')
            print(f'  Shape: ({det_y}, {det_x}, {scan_y}, {scan_x})')
        
            # Read data
            data = np.memmap(cu_path, dtype=full_dtype, mode='r', offset=offset,
                            shape=(det_y, det_x, scan_y, scan_x))
        
            # Convert to (scan_y, scan_x, det_y, det_x)
            data = np.array(data.transpose(2, 3, 0, 1)).astype(np.float32)
            print(f'\n  Data shape: {data.shape}')
            print(f'  Min: {data.min():.2f}')
            print(f'  Max: {data.max():.2f}')
            print(f'  Mean: {data.mean():.2f}')
            print(f'  Std: {data.std():.2f}')
        
            # Check for negative values
            neg_count = np.sum(data < 0)
            print(f'  Negative pixels: {neg_count} ({100*neg_count/data.size:.2f}%)')
        
            # Check total intensity per scan position
            total_intensity = np.sum(data, axis=(2, 3))
            print(f'\n  Total intensity per scan position:')
            print(f'    Min: {total_intensity.min():.2f}')
            print(f'    Max: {total_intensity.max():.2f}')
            print(f'    Mean: {total_intensity.mean():.2f}')
            print(f'    Negative positions: {np.sum(total_intensity < 0)}')
        
            # Check average diffraction pattern
            avg_dp = np.mean(data, axis=(0, 1))
            print(f'\n  Average diffraction pattern:')
            print(f'    Min: {avg_dp.min():.2f}, Max: {avg_dp.max():.2f}')
            print(f'    Mean: {avg_dp.mean():.2f}')
        
            # Find beam center
            total = avg_dp.sum()
            if total > 0:
                y_coords, x_coords = np.mgrid[0:det_y, 0:det_x]
                com_y = np.sum(y_coords * avg_dp) / total
                com_x = np.sum(x_coords * avg_dp) / total
                print(f'    Center of mass: ({com_y:.2f}, {com_x:.2f})')
        
            # Check center vs corner
            cy, cx = det_y//2, det_x//2
            center_val = avg_dp[cy-5:cy+5, cx-5:cx+5].mean()
            corner_val = avg_dp[0:10, 0:10].mean()
            print(f'    Center 10x10: {center_val:.2f}')
            print(f'    Corner 10x10: {corner_val:.2f}')
            print(f'    Ratio: {center_val/corner_val:.3f}' if corner_val > 0 else '    Ratio: N/A')
        
            # Radial profile
            r = np.sqrt((np.arange(det_y)[:, None] - cy)**2 + (np.arange(det_x)[None, :] - cx)**2)
            r_int = r.astype(int)
            radial = np.bincount(r_int.ravel(), weights=avg_dp.ravel())
            counts = np.bincount(r_int.ravel())
            radial = radial / np.maximum(counts, 1)
        
            print(f'\n  Radial profile:')
            for j in range(0, min(50, len(radial)), 10):
                print(f'    r={j}: {radial[j]:.2f}')
        
            # Find alpha
            if radial.max() > 0:
                threshold = 0.3 * radial.max()
                alpha_candidates = np.where(radial < threshold)[0]
                alpha = alpha_candidates[0] if len(alpha_candidates) > 0 else len(radial)//2
                print(f'\n  Estimated alpha: {alpha} pixels')
                print(f'  Peak/Edge ratio: {radial.max() / radial[-1]:.2f}' if radial[-1] > 0 else '  Peak/Edge: N/A')
        
            # Compare with Au data characteristics
            print(f'\n  {"="*60}')
            print(f'  Comparison with Au data issues:')
            print(f'  {"="*60}')
            print(f'  {"Metric":<30} {"Cu foil":<15} {"Au SI19":<15}')
            print(f'  {"-"*60}')
            print(f'  {"Data type":<30} {base_dtype.__name__:<15} {"float32":<15}')
            print(f'  (dtype labels follow the image dataType table; the old')
            print(f'   "Cu=uint8 / Au=int16" labels came from the wrong')
            print(f'   encoded-type table: Au dataType=2 -> float32.)')
            print(f'  {"Negative pixels":<30} {"0%":<15} {"~50%":<15}')
            print(f'  {"Mean intensity":<30} {data.mean():<15.1f} {"~0 (raw)":<15}')
            print(f'  {"Center/Corner ratio":<30} {center_val/corner_val:<15.3f} {"~1.0":<15}')
            print(f'  {"Detector size":<30} {f"{det_y}x{det_x}":<15} {"32x32":<15}')
            print(f'  {"Scan size":<30} {f"{scan_y}x{scan_x}":<15} {"2048x2048":<15}')
        
            break

    f.fid.close()

    # Also check the previously extracted Cu data
    print(f'\n{"="*70}')
    print('Checking previously extracted Cu data')
    print('='*70)

    BASE = os.environ.get('STEM4D_DATA', '').strip()
    data_dir = os.environ.get('STEM4D_ANALYSIS_DATA', '').strip() or \
        (os.path.join(BASE, 'analysis', 'data') if BASE else '')
    if not data_dir:
        print('\nSkipped extracted-data check '
              '(set STEM4D_ANALYSIS_DATA or STEM4D_DATA to enable)')
        return
    extracted_path = os.path.join(data_dir, 'Cu_foil_crop128.npy')
    if os.path.exists(extracted_path):
        cu_extracted = np.load(extracted_path)
        print(f'\nExtracted Cu data:')
        print(f'  Shape: {cu_extracted.shape}')
        print(f'  Min: {cu_extracted.min():.2f}, Max: {cu_extracted.max():.2f}')
        print(f'  Mean: {cu_extracted.mean():.2f}')
        print(f'  Negative pixels: {np.sum(cu_extracted < 0)}')
    else:
        print(f'\nExtracted file not found: {extracted_path}')

        # Check what files exist
        print(f'\nFiles in {data_dir}:')
        for fname in os.listdir(data_dir):
            if fname.endswith('.npy'):
                fpath = os.path.join(data_dir, fname)
                size_mb = os.path.getsize(fpath) / 1024**2
                print(f'  {fname}: {size_mb:.1f} MB')


if __name__ == '__main__':
    main()
