"""
process_au_v3.py - Process Au 4D-STEM data with correct data extraction.
Uses correctly extracted data (int16 little-endian) with proper handling of negative values.
"""
import numpy as np
import os, sys, time, json, argparse

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dpc_core import (compute_com_fast, find_alpha_from_radial,
                           idpc_reconstruct, compute_virtual_images,
                           run_dpc_pipeline, plot_dpc_results)
from core.ssb_core import ssb_reconstruct

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuration
# 不内置任何本机默认数据路径：数据目录必须经 STEM4D_DATA /
# STEM4D_ANALYSIS_DATA 环境变量或命令行参数提供，缺失时报错并打印用法。
DEFAULT_BASE = os.environ.get('STEM4D_DATA', '').strip()
DATA_DIR = os.environ.get(
    'STEM4D_ANALYSIS_DATA', '').strip() or (
    os.path.join(DEFAULT_BASE, 'analysis', 'data') if DEFAULT_BASE else '')
OUTPUT_DIR = os.environ.get(
    'STEM4D_OUTPUT', '').strip() or (
    os.path.join(DEFAULT_BASE, 'analysis', 'results', 'v2') if DEFAULT_BASE
    else '')

def get_datasets(data_dir):
    """Au datasets - using correctly extracted data."""
    return {
        'Au_SI19': {
            'path': os.path.join(data_dir, 'Au_SI19_correct.npy'),
            'description': 'Au SI data 19 (correct extraction)',
            'scale_nm': 0.212,  # nm/pixel from metadata
        },
        'Au_SI20': {
            'path': os.path.join(data_dir, 'Au_SI20_correct.npy'),
            'description': 'Au SI data 20 (correct extraction)',
            'scale_nm': 0.484,
        },
        'Au_SI21': {
            'path': os.path.join(data_dir, 'Au_SI21_correct.npy'),
            'description': 'Au SI data 21 (correct extraction)',
            'scale_nm': 0.182,
        },
    }


def preprocess_data(data, method='shift'):
    """
    Preprocess background-subtracted data to make it suitable for DPC/SSB.
    
    Args:
        data: Raw data with negative values (background-subtracted)
        method: 'shift' (subtract min), 'clip' (max with 0), 'abs' (absolute value)
    
    Returns:
        Preprocessed data with positive values
    """
    if method == 'shift':
        # Shift to make all values positive
        return data - data.min()
    elif method == 'clip':
        # Clip negative values to zero
        return np.maximum(data, 0)
    elif method == 'abs':
        # Take absolute value
        return np.abs(data)
    else:
        raise ValueError(f"Unknown method: {method}")


def process_single_dataset(name, config):
    """Process a single Au dataset with DPC and SSB."""
    print(f'\n{"="*70}')
    print(f'Processing: {name} - {config["description"]}')
    print(f'{"="*70}')
    
    # Load correctly extracted data
    print(f'Loading data from: {config["path"]}')
    raw_data = np.load(config['path'])
    scan_y, scan_x, det_y, det_x = raw_data.shape
    print(f'  Shape: {raw_data.shape}')
    print(f'  Raw range: [{raw_data.min():.1f}, {raw_data.max():.1f}]')
    print(f'  Raw mean: {raw_data.mean():.1f}')
    print(f'  Negative pixels: {np.sum(raw_data < 0)} ({100*np.sum(raw_data < 0)/raw_data.size:.1f}%)')
    
    # Preprocess: shift to positive
    print(f'\nPreprocessing: shifting data to positive...')
    data = preprocess_data(raw_data, method='shift')
    print(f'  Processed range: [{data.min():.1f}, {data.max():.1f}]')
    print(f'  Processed mean: {data.mean():.1f}')
    
    t0 = time.time()
    
    # Step 1: Find beam center
    print(f'\n[1] Finding beam center (CoM)...')
    com_result = compute_com_fast(data)
    center = com_result['center']
    print(f'  Center: ({center[0]:.2f}, {center[1]:.2f})')
    print(f'  CoM-Y range: [{com_result["com_y"].min():.4f}, {com_result["com_y"].max():.4f}]')
    print(f'  CoM-X range: [{com_result["com_x"].min():.4f}, {com_result["com_x"].max():.4f}]')
    
    # Step 2: Estimate alpha
    print(f'\n[2] Estimating BF disk radius...')
    alpha, radial, counts = find_alpha_from_radial(data, center)
    print(f'  Alpha: {alpha} pixels (detector: {det_y}x{det_x})')
    
    # Step 3: Run DPC pipeline
    print(f'\n[3] Running DPC pipeline...')
    dpc_result = run_dpc_pipeline(
        data,
        alpha_pixels=alpha,
        center=center,
        label=name,
        regularization=1e-3,
        outdir=OUTPUT_DIR,
    )
    
    # Step 4: Run SSB reconstruction
    print(f'\n[4] Running SSB reconstruction...')
    ssb_result = ssb_reconstruct(
        data,
        alpha_pixels=alpha,
        center=center,
        regularization=1e-2,
        verbose=True
    )
    
    # Save results
    np.save(os.path.join(OUTPUT_DIR, f'{name}_v3_dpc_phase.npy'), dpc_result['phase'])
    np.save(os.path.join(OUTPUT_DIR, f'{name}_v3_ssb_phase.npy'), ssb_result['phase'])
    np.save(os.path.join(OUTPUT_DIR, f'{name}_v3_ssb_amplitude.npy'), ssb_result['amplitude'])
    
    # Plot SSB results
    plot_ssb_results(ssb_result, name)
    
    # Step 5: Compare DPC vs SSB
    print(f'\n[5] Comparing DPC vs SSB...')
    compare_results(dpc_result, ssb_result, name, config)
    
    # Step 6: Electric field analysis
    print(f'\n[6] Computing electric field maps...')
    compute_electric_field(com_result, name, config)
    
    elapsed = time.time() - t0
    print(f'\nTotal processing time: {elapsed:.1f}s')
    
    # Save metadata
    metadata = {
        'name': name,
        'description': config['description'],
        'scan_shape': [scan_y, scan_x],
        'det_shape': [det_y, det_x],
        'center': [float(center[0]), float(center[1])],
        'alpha_pixels': int(alpha),
        'scale_nm': config.get('scale_nm'),
        'preprocessing': 'shift (data - min)',
        'raw_data_range': [float(raw_data.min()), float(raw_data.max())],
        'processed_data_range': [float(data.min()), float(data.max())],
        'dpc_phase_range': [float(dpc_result['phase'].min()), float(dpc_result['phase'].max())],
        'ssb_phase_range': [float(ssb_result['phase'].min()), float(ssb_result['phase'].max())],
        'processing_time_sec': elapsed,
    }
    with open(os.path.join(OUTPUT_DIR, f'{name}_v3_metadata.json'),
              'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return dpc_result, ssb_result


def plot_ssb_results(result, name):
    """Plot SSB reconstruction results."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Amplitude
    im0 = axes[0, 0].imshow(result['amplitude'], cmap='gray')
    axes[0, 0].set_title('SSB Amplitude')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)
    
    # Phase
    im1 = axes[0, 1].imshow(result['phase'], cmap='viridis')
    axes[0, 1].set_title('SSB Phase')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    
    # Phase (+/-)
    vmax = max(abs(result['phase'].min()), abs(result['phase'].max()))
    im2 = axes[0, 2].imshow(result['phase'], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0, 2].set_title('SSB Phase (+/-)')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], shrink=0.8)
    
    # Complex real/imag
    im3 = axes[1, 0].imshow(np.real(result['complex_obj']), cmap='RdBu_r')
    axes[1, 0].set_title('Re[ψ]')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
    
    im4 = axes[1, 1].imshow(np.imag(result['complex_obj']), cmap='RdBu_r')
    axes[1, 1].set_title('Im[ψ]')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
    
    # Aperture
    axes[1, 2].imshow(result['aperture'], cmap='gray')
    axes[1, 2].set_title('Aperture')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'SSB Results (v3): {name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'{name}_v3_ssb.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def compare_results(dpc_result, ssb_result, name, config):
    """Compare DPC and SSB results."""
    phase_dpc = dpc_result['phase']
    phase_ssb = ssb_result['phase']
    
    if phase_dpc.shape != phase_ssb.shape:
        print(f'  Shape mismatch, skipping comparison')
        return
    
    diff = phase_dpc - phase_ssb
    corr = np.corrcoef(phase_dpc.ravel(), phase_ssb.ravel())[0, 1]
    
    print(f'  DPC phase range: [{phase_dpc.min():.4f}, {phase_dpc.max():.4f}]')
    print(f'  SSB phase range: [{phase_ssb.min():.4f}, {phase_ssb.max():.4f}]')
    print(f'  Correlation: {corr:.4f}')
    
    # Plot comparison
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: DPC results
    axes[0, 0].imshow(dpc_result['virtual_images']['adf'], cmap='gray')
    axes[0, 0].set_title('Virtual ADF')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(dpc_result['com_result']['com_magnitude'], cmap='inferno')
    axes[0, 1].set_title('|CoM| Magnitude')
    axes[0, 1].axis('off')
    
    vmax_dpc = max(abs(phase_dpc.min()), abs(phase_dpc.max()))
    im2 = axes[0, 2].imshow(phase_dpc, cmap='RdBu_r', vmin=-vmax_dpc, vmax=vmax_dpc)
    axes[0, 2].set_title('iDPC Phase')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], shrink=0.8)
    
    axes[0, 3].imshow(dpc_result['virtual_images']['bf'], cmap='gray')
    axes[0, 3].set_title('Virtual BF')
    axes[0, 3].axis('off')
    
    # Row 2: SSB results
    axes[1, 0].imshow(ssb_result['amplitude'], cmap='gray')
    axes[1, 0].set_title('SSB Amplitude')
    axes[1, 0].axis('off')
    
    vmax_ssb = max(abs(phase_ssb.min()), abs(phase_ssb.max()))
    im5 = axes[1, 1].imshow(phase_ssb, cmap='RdBu_r', vmin=-vmax_ssb, vmax=vmax_ssb)
    axes[1, 1].set_title('SSB Phase')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)
    
    vmax_diff = max(abs(diff.min()), abs(diff.max()))
    im6 = axes[1, 2].imshow(diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
    axes[1, 2].set_title('DPC - SSB')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2], shrink=0.8)
    
    # Scatter plot
    axes[1, 3].scatter(phase_dpc.ravel()[::5], phase_ssb.ravel()[::5], s=1, alpha=0.3)
    axes[1, 3].set_xlabel('DPC Phase')
    axes[1, 3].set_ylabel('SSB Phase')
    axes[1, 3].set_title(f'Correlation: {corr:.4f}')
    
    plt.suptitle(f'DPC vs SSB (v3): {name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'{name}_v3_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def compute_electric_field(com_result, name, config):
    """Compute electric field maps from CoM data."""
    com_y = com_result['com_y']
    com_x = com_result['com_x']
    
    scale_nm = config.get('scale_nm', 1.0)
    
    # Electric field magnitude
    e_magnitude = np.sqrt(com_x**2 + com_y**2)
    
    # Charge density (divergence of E field)
    dEx_dx = np.gradient(com_x, axis=1) / scale_nm
    dEy_dy = np.gradient(com_y, axis=0) / scale_nm
    charge_density = dEx_dx + dEy_dy
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # E_x
    vmax_x = max(abs(com_x.min()), abs(com_x.max()))
    im0 = axes[0, 0].imshow(com_x, cmap='RdBu_r', vmin=-vmax_x, vmax=vmax_x)
    axes[0, 0].set_title('E_x (CoM-X)')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)
    
    # E_y
    vmax_y = max(abs(com_y.min()), abs(com_y.max()))
    im1 = axes[0, 1].imshow(com_y, cmap='RdBu_r', vmin=-vmax_y, vmax=vmax_y)
    axes[0, 1].set_title('E_y (CoM-Y)')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    
    # |E|
    im2 = axes[0, 2].imshow(e_magnitude, cmap='inferno')
    axes[0, 2].set_title('|E| Magnitude')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], shrink=0.8)
    
    # Charge density
    vmax_rho = max(abs(charge_density.min()), abs(charge_density.max()))
    im3 = axes[1, 0].imshow(charge_density, cmap='RdBu_r', vmin=-vmax_rho, vmax=vmax_rho)
    axes[1, 0].set_title('Charge Density (∇·E)')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
    
    # Vector field
    step = max(1, com_x.shape[0] // 20)
    y, x = np.mgrid[0:com_x.shape[0]:step, 0:com_x.shape[1]:step]
    axes[1, 1].quiver(x, y, com_x[::step, ::step], -com_y[::step, ::step], 
                      e_magnitude[::step, ::step], cmap='viridis', alpha=0.8)
    axes[1, 1].set_title('Electric Field Vectors')
    axes[1, 1].set_aspect('equal')
    axes[1, 1].invert_yaxis()
    
    # Histogram
    axes[1, 2].hist(e_magnitude.ravel(), bins=100, color='blue', alpha=0.7)
    axes[1, 2].set_xlabel('|E| (pixels)')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('Field Magnitude Distribution')
    
    plt.suptitle(f'Electric Field Analysis (v3): {name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'{name}_v3_efield.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')
    
    # Save data
    np.save(os.path.join(OUTPUT_DIR, f'{name}_v3_charge_density.npy'), charge_density)


def create_summary(all_results):
    """Create summary comparison of all datasets."""
    print(f'\n{"="*70}')
    print('Creating Summary')
    print(f'{"="*70}')
    
    n = len(all_results)
    if n == 0:
        return
    
    fig, axes = plt.subplots(n, 5, figsize=(25, 5*n))
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i, (name, (dpc, ssb)) in enumerate(all_results.items()):
        phase_dpc = dpc['phase']
        phase_ssb = ssb['phase']
        
        # ADF
        axes[i, 0].imshow(dpc['virtual_images']['adf'], cmap='gray')
        axes[i, 0].set_title(f'{name}: ADF')
        axes[i, 0].axis('off')
        
        # |CoM|
        axes[i, 1].imshow(dpc['com_result']['com_magnitude'], cmap='inferno')
        axes[i, 1].set_title(f'{name}: |CoM|')
        axes[i, 1].axis('off')
        
        # iDPC Phase
        vmax = max(abs(phase_dpc.min()), abs(phase_dpc.max()))
        axes[i, 2].imshow(phase_dpc, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[i, 2].set_title(f'{name}: iDPC Phase')
        axes[i, 2].axis('off')
        
        # SSB Phase
        vmax_ssb = max(abs(phase_ssb.min()), abs(phase_ssb.max()))
        axes[i, 3].imshow(phase_ssb, cmap='RdBu_r', vmin=-vmax_ssb, vmax=vmax_ssb)
        axes[i, 3].set_title(f'{name}: SSB Phase')
        axes[i, 3].axis('off')
        
        # SSB Amplitude
        axes[i, 4].imshow(ssb['amplitude'], cmap='gray')
        axes[i, 4].set_title(f'{name}: SSB Amplitude')
        axes[i, 4].axis('off')
    
    plt.suptitle('Au 4D-STEM Analysis Summary (v3 - Correct Data)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, '00_SUMMARY_v3.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Summary saved: {save_path}')


def main(argv=None):
    global DATA_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser(
        description='Run the v3 DPC/SSB analysis on the extracted Au data.')
    parser.add_argument('--data-dir', default=DATA_DIR,
                        help='Directory with the *_correct.npy files.')
    parser.add_argument('--output-dir', default=OUTPUT_DIR,
                        help='Directory for results.')
    args = parser.parse_args(argv)

    if not args.data_dir:
        parser.error('未配置数据目录：请传 --data-dir，或设置环境变量 '
                     'STEM4D_ANALYSIS_DATA / STEM4D_DATA 后重跑')
    if not args.output_dir:
        parser.error('未配置输出目录：请传 --output-dir，或设置环境变量 '
                     'STEM4D_OUTPUT / STEM4D_DATA 后重跑')

    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    AU_DATASETS = get_datasets(DATA_DIR)

    print('='*70)
    print('Au 4D-STEM Data Processing (v3 - Correct Data Extraction)')
    print('='*70)
    print(f'Data directory: {DATA_DIR}')
    print(f'Output directory: {OUTPUT_DIR}')
    print(f'\nNOTE: Using correctly extracted data (int16 little-endian)')
    print(f'      with shift preprocessing to handle negative values.')
    
    all_results = {}
    
    for name, config in AU_DATASETS.items():
        if not os.path.exists(config['path']):
            print(f'\nSkipping {name}: file not found at {config["path"]}')
            continue
        
        try:
            dpc_result, ssb_result = process_single_dataset(name, config)
            all_results[name] = (dpc_result, ssb_result)
        except Exception as e:
            print(f'\nERROR processing {name}: {e}')
            import traceback
            traceback.print_exc()
    
    # Create summary
    create_summary(all_results)
    
    print(f'\n{"="*70}')
    print(f'All processing complete!')
    print(f'Results saved to: {OUTPUT_DIR}')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
