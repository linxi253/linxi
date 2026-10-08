"""
dpc_core.py - Differential Phase Contrast (DPC) and iDPC analysis for 4D-STEM data.

DPC: Center-of-mass (CoM) of each diffraction pattern → electric field
iDPC: Integrated DPC → phase reconstruction via Fourier integration

Reference: Lazic et al., Ultramicroscopy 160, 265-280 (2016)
           Mueller-Caspary et al., Nature Communications 8, 371 (2017)
"""
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, time, json


# ============================================================
# 1. CENTER OF MASS (CoM) ANALYSIS
# ============================================================

def compute_com_robust(datacube, center=None, radius=None, positive_only=True,
                       outlier_sigma=3.0, should_stop=None):
    """
    Vectorized, memory-bounded CoM computation.

    - masks to a BF disk around ``center`` (auto-detected if ``center`` is None)
    - clips negative pixels to zero when ``positive_only`` (required for
      background-subtracted data)
    - processes in row-chunks to keep memory bounded
    - rejects outlier CoM values with a 3x3 median filter when
      ``outlier_sigma > 0``
    - supports cooperative cancellation via ``should_stop`` (the returned
      dict carries ``cancelled=True`` and the unprocessed rows stay zero)
    """
    scan_y, scan_x, det_y, det_x = datacube.shape

    ky, kx = np.meshgrid(np.arange(det_y), np.arange(det_x), indexing='ij')

    if center is None:
        avg_cbed = np.mean(datacube, axis=(0, 1))
        total = np.sum(avg_cbed)
        if not total > 0:
            raise RuntimeError(
                "Cannot auto-detect the beam centre: the average diffraction "
                "pattern has non-positive total intensity. The data appears "
                "to be background-subtracted; shift/clip it to positive "
                "values first, or pass an explicit centre.")
        center = (np.sum(ky * avg_cbed) / total, np.sum(kx * avg_cbed) / total)

    cy, cx = center
    if radius is not None:
        r2 = (ky - cy) ** 2 + (kx - cx) ** 2
        mask = (r2 <= radius ** 2).astype(np.float64)
    else:
        mask = np.ones((det_y, det_x), dtype=np.float64)

    ky_shifted = (ky - cy).astype(np.float64)
    kx_shifted = (kx - cx).astype(np.float64)

    bf_intensity = np.zeros((scan_y, scan_x), dtype=np.float64)
    com_y_map = np.zeros((scan_y, scan_x), dtype=np.float64)
    com_x_map = np.zeros((scan_y, scan_x), dtype=np.float64)
    cancelled = False

    # Memory budget: ~64 MB of float64 temporaries per chunk
    chunk_elements = 8_000_000
    chunk_rows = max(1, chunk_elements // (det_y * det_x))

    for row_start in range(0, scan_y, chunk_rows):
        if should_stop is not None and should_stop():
            cancelled = True
            break
        row_end = min(row_start + chunk_rows, scan_y)
        chunk = datacube[row_start:row_end] * mask[np.newaxis, np.newaxis, :, :]
        if positive_only:
            chunk = np.maximum(chunk, 0)
        bf_intensity[row_start:row_end] = np.sum(chunk, axis=(2, 3))
        com_y_map[row_start:row_end] = np.einsum('ijkl,kl->ij', chunk, ky_shifted)
        com_x_map[row_start:row_end] = np.einsum('ijkl,kl->ij', chunk, kx_shifted)

    valid = bf_intensity > 0
    com_y = np.zeros_like(com_y_map)
    com_x = np.zeros_like(com_x_map)
    com_y[valid] = com_y_map[valid] / bf_intensity[valid]
    com_x[valid] = com_x_map[valid] / bf_intensity[valid]

    # Outlier rejection: replace CoM values far from the median with the
    # local median (scipy is a hard dependency of this project).
    if outlier_sigma > 0:
        com_mag = np.sqrt(com_y ** 2 + com_x ** 2)
        med = np.median(com_mag)
        std = np.std(com_mag)
        if std > 0:
            from scipy.ndimage import median_filter
            outliers = com_mag > med + outlier_sigma * std
            if np.any(outliers):
                com_y[outliers] = median_filter(com_y, size=3)[outliers]
                com_x[outliers] = median_filter(com_x, size=3)[outliers]

    com_magnitude = np.sqrt(com_y ** 2 + com_x ** 2)

    return {
        'com_y': com_y,
        'com_x': com_x,
        'com_magnitude': com_magnitude,
        'bf_intensity': bf_intensity,
        'center': center,
        'mask': mask,
        'cancelled': cancelled,
    }


def compute_com_fast(datacube, mask=None, center=None, radius=None):
    """Deprecated alias for :func:`compute_com_robust` with the legacy
    numeric behaviour (no positive clipping, no outlier rejection).

    Kept only for the older processing/diagnostics scripts; new code should
    call :func:`compute_com_robust` directly.
    """
    if mask is not None:
        raise TypeError("compute_com_fast no longer accepts a custom mask; "
                        "use compute_com_robust with radius instead")
    return compute_com_robust(datacube, center=center, radius=radius,
                              positive_only=False, outlier_sigma=0)


# ============================================================
# 2. iDPC PHASE RECONSTRUCTION
# ============================================================

def idpc_reconstruct(com_result, regularization=1e-3):
    """
    Reconstruct phase from CoM via Fourier integration (iDPC).

    For numpy's FFT convention exp(-2πi k·x):
        ∇φ = CoM
        FFT: (2πi k) · φ̃ = CoM̃
        φ̃ = -i (k·CoM̃) / (2π k² + ε)

    Note on ``regularization``: ε is added to 2πk² where k is in
    cycles/pixel, so its damping effect is *not* scale invariant. With the
    default 1e-3 the lowest non-zero frequency of a 128-px scan
    (2π/128² ≈ 3.8e-4) is attenuated to ~28% — long-range phase is
    deliberately suppressed. Reduce it (e.g. 1e-5) when the low-frequency
    component matters and the CoM field is trusted; the value used is
    recorded by the GUI metadata.

    Parameters:
        com_result: dict from compute_com_robust()
        regularization: Tikhonov regularization parameter

    Returns:
        phase: reconstructed phase, shape (scan_y, scan_x)
    """
    com_y = com_result['com_y']
    com_x = com_result['com_x']
    scan_y, scan_x = com_y.shape
    
    # FFT of CoM components
    com_y_k = fft2(com_y)
    com_x_k = fft2(com_x)
    
    # Spatial frequency grids (in cycles/pixel: -0.5 to 0.5)
    qy, qx = np.meshgrid(
        np.fft.fftfreq(scan_y),
        np.fft.fftfreq(scan_x),
        indexing='ij'
    )
    q2 = qy**2 + qx**2
    
    # Integration in Fourier space
    # φ̃ = -i(k·CoM̃) / (2π·k² + ε)
    TWOPI = 2.0 * np.pi
    phase_k = np.zeros((scan_y, scan_x), dtype=np.complex128)
    valid = q2 > 0
    phase_k[valid] = -1j * (qy[valid] * com_y_k[valid] + qx[valid] * com_x_k[valid]) / (TWOPI * q2[valid] + regularization)
    
    # Inverse FFT
    phase = np.real(ifft2(phase_k))
    
    # Detrend: remove linear plane (fits: phase = a*x + b*y + c)
    yy, xx = np.mgrid[:scan_y, :scan_x]
    X = np.column_stack([xx.ravel(), yy.ravel(), np.ones(scan_y * scan_x)])
    coeffs = np.linalg.lstsq(X, phase.ravel(), rcond=None)[0]
    plane = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
    phase = phase - plane
    
    # Remove mean
    phase -= np.mean(phase)
    
    return phase


def idpc_reconstruct_rotated(com_result, rotation_deg=0, regularization=1e-3):
    """
    Reconstruct phase with scan-detector rotation correction.
    """
    com_y = com_result['com_y']
    com_x = com_result['com_x']
    
    # Rotate CoM
    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    com_y_rot = com_y * cos_t - com_x * sin_t
    com_x_rot = com_y * sin_t + com_x * cos_t
    
    result_rot = {
        'com_y': com_y_rot,
        'com_x': com_x_rot,
        'com_magnitude': np.sqrt(com_y_rot**2 + com_x_rot**2),
        'bf_intensity': com_result['bf_intensity'],
        'center': com_result['center'],
        'mask': com_result['mask'],
    }
    
    return idpc_reconstruct(result_rot, regularization)


# ============================================================
# 3. VIRTUAL DETECTOR IMAGES
# ============================================================

def compute_virtual_images(datacube, center, alpha_pixels, chunk_rows=None):
    """Compute BF, ADF, and ABF virtual detector images.

    The annular ranges are adapted to the actual detector size to ensure
    valid masks even for small detectors.

    Masks are applied with a chunked ``einsum`` (same scheme as
    :func:`compute_com_robust`): the old ``datacube * mask`` broadcast
    materialized a full float64 copy of the cube per detector — four times
    in a row — which made large-detector datasets OOM.
    """
    scan_y, scan_x, det_y, det_x = datacube.shape
    ky, kx = np.meshgrid(np.arange(det_y), np.arange(det_x), indexing='ij')
    cy, cx = center
    r2 = (ky - cy)**2 + (kx - cx)**2
    r = np.sqrt(r2)

    # Calculate maximum possible radius from center to detector edge
    max_radius = max(cy, cx, det_y - cy, det_x - cx)

    # BF: inside alpha
    bf_mask = (r2 <= alpha_pixels**2).astype(np.float64)

    # ADF: outside alpha, up to detector edge
    # Use adaptive inner radius: start just outside BF disk
    adf_inner = min(int(alpha_pixels * 1.2), int(max_radius * 0.9))
    adf_inner = max(adf_inner, int(alpha_pixels) + 1)  # Ensure at least 1 pixel outside BF
    adf_mask = (r >= adf_inner).astype(np.float64)

    # ABF: annular bright field (inner part of BF disk)
    abf_inner = int(alpha_pixels * 0.4)
    abf_outer = int(alpha_pixels * 0.9)
    abf_mask = ((r >= abf_inner) & (r <= abf_outer)).astype(np.float64)

    # HAADF: high-angle ADF (if detector is large enough)
    haadf_inner = min(int(alpha_pixels * 2.0), int(max_radius * 0.7))
    if haadf_inner < max_radius:
        haadf_mask = (r >= haadf_inner).astype(np.float64)
    else:
        haadf_mask = adf_mask  # Fallback to ADF if detector too small

    if chunk_rows is None:
        # Memory budget: ~64 MB of float64 temporaries per chunk
        chunk_rows = max(1, 8_000_000 // (det_y * det_x))

    images = {key: np.zeros((scan_y, scan_x), dtype=np.float64)
              for key in ('bf', 'adf', 'abf', 'haadf')}
    masks = {'bf': bf_mask, 'adf': adf_mask, 'abf': abf_mask,
             'haadf': haadf_mask}
    for row_start in range(0, scan_y, chunk_rows):
        row_end = min(row_start + chunk_rows, scan_y)
        chunk = datacube[row_start:row_end]
        for key, mask in masks.items():
            images[key][row_start:row_end] = np.einsum('ijkl,kl->ij',
                                                       chunk, mask)

    return {
        'bf': images['bf'],
        'adf': images['adf'],
        'abf': images['abf'],
        'haadf': images['haadf'],
        'bf_mask': bf_mask,
        'adf_mask': adf_mask,
        'adf_inner_radius': adf_inner,
    }


# ============================================================
# 4. COMPLETE DPC PIPELINE
# ============================================================

def run_dpc_pipeline(datacube, alpha_pixels, center=None, rotation_deg=0,
                     regularization=1e-3, label='Dataset', outdir=None):
    """
    Complete DPC analysis pipeline.
    """
    scan_y, scan_x, det_y, det_x = datacube.shape
    
    print(f'\n{"="*60}')
    print(f'DPC Analysis: {label}')
    print(f'{"="*60}')
    print(f'  Shape: ({scan_y}, {scan_x}, {det_y}, {det_x})')
    print(f'  Alpha: {alpha_pixels}, Rotation: {rotation_deg}°')
    
    t0 = time.time()
    
    # 1. CoM computation
    print(f'  Computing CoM...', flush=True)
    com_result = compute_com_robust(datacube, center=center, radius=alpha_pixels)
    print(f'    CoM range: y=[{com_result["com_y"].min():.3f}, {com_result["com_y"].max():.3f}]')
    print(f'               x=[{com_result["com_x"].min():.3f}, {com_result["com_x"].max():.3f}]')
    
    # 2. iDPC reconstruction
    print(f'  Reconstructing phase (iDPC)...', flush=True)
    if rotation_deg != 0:
        phase = idpc_reconstruct_rotated(com_result, rotation_deg, regularization)
    else:
        phase = idpc_reconstruct(com_result, regularization)
    print(f'    Phase range: [{phase.min():.4f}, {phase.max():.4f}]')
    
    # 3. Virtual images
    print(f'  Computing virtual images...', flush=True)
    vimg = compute_virtual_images(datacube, com_result['center'], alpha_pixels)
    
    elapsed = time.time() - t0
    print(f'  Done in {elapsed:.1f}s')
    
    result = {
        'com_result': com_result,
        'phase': phase,
        'virtual_images': vimg,
        'metadata': {
            'label': label,
            'scan_shape': (scan_y, scan_x),
            'det_shape': (det_y, det_x),
            'alpha_pixels': alpha_pixels,
            'center': com_result['center'],
            'rotation_deg': rotation_deg,
            'regularization': regularization,
            'time_sec': elapsed,
        }
    }
    
    # Save and plot
    if outdir:
        # Save numeric data
        np.save(os.path.join(outdir, f'{label}_dpc_phase.npy'), phase)
        np.save(os.path.join(outdir, f'{label}_com_y.npy'), com_result['com_y'])
        np.save(os.path.join(outdir, f'{label}_com_x.npy'), com_result['com_x'])
        
        # 文本模式 open 必须显式 encoding：中文 Windows 默认 cp936，会把
        # ensure_ascii=False 的中文/符号 label 写成 GBK 或直接抛
        # UnicodeEncodeError。
        with open(os.path.join(outdir, f'{label}_dpc_meta.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(result['metadata'], f, indent=2, default=str,
                      ensure_ascii=False)
        
        # Plot
        plot_dpc_results(result, save_path=os.path.join(outdir, f'{label}_dpc.png'))
    
    return result


# ============================================================
# 5. VISUALIZATION
# ============================================================

def plot_dpc_results(result, save_path=None):
    """Plot DPC analysis results comprehensively."""
    com = result['com_result']
    phase = result['phase']
    vimg = result['virtual_images']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Virtual images
    im0 = axes[0, 0].imshow(vimg['bf'], cmap='gray')
    axes[0, 0].set_title('Virtual BF')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)
    
    im1 = axes[0, 1].imshow(vimg['adf'], cmap='gray')
    axes[0, 1].set_title('Virtual ADF')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    
    im2 = axes[0, 2].imshow(vimg['abf'], cmap='gray')
    axes[0, 2].set_title('Virtual ABF')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], shrink=0.8)
    
    # CoM magnitude map
    im3 = axes[0, 3].imshow(com['com_magnitude'], cmap='inferno')
    axes[0, 3].set_title('|CoM| Magnitude')
    axes[0, 3].axis('off')
    plt.colorbar(im3, ax=axes[0, 3], shrink=0.8)
    
    # Row 2: DPC and Phase
    im4 = axes[1, 0].imshow(com['com_y'], cmap='RdBu_r')
    axes[1, 0].set_title('CoM-Y (vertical field)')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)
    
    im5 = axes[1, 1].imshow(com['com_x'], cmap='RdBu_r')
    axes[1, 1].set_title('CoM-X (horizontal field)')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)
    
    # iDPC Phase
    vmax = max(abs(phase.min()), abs(phase.max()))
    im6 = axes[1, 2].imshow(phase, cmap='viridis')
    axes[1, 2].set_title('iDPC Phase')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2], shrink=0.8)
    
    # Phase with +/- colormap
    im7 = axes[1, 3].imshow(phase, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1, 3].set_title('iDPC Phase (+/-)')
    axes[1, 3].axis('off')
    plt.colorbar(im7, ax=axes[1, 3], shrink=0.8)
    
    plt.suptitle(result['metadata'].get('label', 'DPC Results'), fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    else:
        plt.show()
    plt.close()


def plot_comparison(dpc_result, ssb_result=None, save_path=None):
    """Compare DPC and SSB results side by side."""
    nrows = 2 if ssb_result is None else 3
    fig, axes = plt.subplots(nrows, 4, figsize=(20, 5*nrows))
    
    com = dpc_result['com_result']
    phase_dpc = dpc_result['phase']
    vimg = dpc_result['virtual_images']
    
    # DPC row
    axes[0, 0].imshow(vimg['adf'], cmap='gray')
    axes[0, 0].set_title('ADF')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(com['com_magnitude'], cmap='inferno')
    axes[0, 1].set_title('|CoM|')
    axes[0, 1].axis('off')
    
    vmax_dpc = max(abs(phase_dpc.min()), abs(phase_dpc.max()))
    axes[0, 2].imshow(phase_dpc, cmap='viridis')
    axes[0, 2].set_title('iDPC Phase (viridis)')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(phase_dpc, cmap='RdBu_r', vmin=-vmax_dpc, vmax=vmax_dpc)
    axes[0, 3].set_title('iDPC Phase (+/-)')
    axes[0, 3].axis('off')
    
    # SSB row if available
    if ssb_result is not None:
        phase_ssb = ssb_result['phase']
        vmax_ssb = max(abs(phase_ssb.min()), abs(phase_ssb.max()))
        
        axes[1, 0].imshow(ssb_result['amplitude'], cmap='gray')
        axes[1, 0].set_title('SSB Amplitude')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(ssb_result['aperture'], cmap='gray')
        axes[1, 1].set_title('SSB Aperture')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(phase_ssb, cmap='viridis')
        axes[1, 2].set_title('SSB Phase (viridis)')
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(phase_ssb, cmap='RdBu_r', vmin=-vmax_ssb, vmax=vmax_ssb)
        axes[1, 3].set_title('SSB Phase (+/-)')
        axes[1, 3].axis('off')
        
        # Difference row (only meaningful when the shapes match)
        if phase_dpc.shape == phase_ssb.shape:
            diff = phase_dpc - phase_ssb
            vmax_diff = max(abs(diff.min()), abs(diff.max()))
            axes[2, 0].text(0.5, 0.5, 'DPC vs SSB\nPhase Difference',
                           ha='center', va='center', transform=axes[2, 0].transAxes)
            axes[2, 0].axis('off')
            axes[2, 1].axis('off')
            axes[2, 2].imshow(diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
            axes[2, 2].set_title('DPC - SSB Phase')
            axes[2, 2].axis('off')
            axes[2, 3].hist(diff.ravel(), bins=100)
            axes[2, 3].set_title('Phase Difference Histogram')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved comparison: {save_path}')
    plt.close()


# ============================================================
# 6. BATCH PROCESSING
# ============================================================

def find_alpha_from_radial(datacube, center):
    """Estimate BF disk radius from radial average (vectorized).
    
    Uses a combination of threshold and gradient methods to robustly
    detect the BF disk edge.
    """
    det_y, det_x = datacube.shape[-2], datacube.shape[-1]
    avg_cbed = np.mean(datacube, axis=(0, 1))
    
    cy, cx = center
    max_r = min(int(cy), int(cx), det_y - int(cy), det_x - int(cx))
    if max_r < 4:
        # Centre too close to the detector edge for a radial profile; return
        # a conservative fallback instead of a degenerate/negative radius.
        return max(3, min(det_y, det_x) // 8), np.zeros(0), np.zeros(0)
    
    # Vectorized radial average computation
    ky, kx = np.meshgrid(np.arange(det_y), np.arange(det_x), indexing='ij')
    r = np.sqrt((ky - cy)**2 + (kx - cx)**2)
    r_int = r.astype(np.int32)
    
    # Only consider pixels within max_r
    valid_pixels = r_int < max_r
    
    radial = np.zeros(max_r, dtype=np.float64)
    counts = np.zeros(max_r, dtype=np.float64)
    
    # Use np.bincount for fast radial averaging
    r_flat = r_int[valid_pixels].ravel()
    intensity_flat = avg_cbed[valid_pixels].ravel()
    
    radial[:max_r] = np.bincount(r_flat, weights=intensity_flat, minlength=max_r)[:max_r]
    counts[:max_r] = np.bincount(r_flat, minlength=max_r)[:max_r]
    
    radial[counts > 0] /= counts[counts > 0]
    
    # Method 1: Find edge where intensity drops to 30% of max (original method)
    threshold = radial.max() * 0.3
    above = np.where(radial > threshold)[0]
    alpha_threshold = above[-1] + 1 if len(above) > 0 else det_y // 4
    
    # Method 2: Find maximum gradient (steepest drop)
    if len(radial) > 5:
        # Smooth radial profile slightly using numpy convolution
        kernel = np.ones(3) / 3.0
        radial_smooth = np.convolve(radial, kernel, mode='same')
        
        gradient = np.abs(np.diff(radial_smooth))
        # Find the first significant gradient peak (BF disk edge)
        # Look in the range [5, max_r-5] to avoid edge effects
        search_start = min(5, max_r // 4)
        search_end = max_r - 2
        if search_end > search_start:
            grad_region = gradient[search_start:search_end]
            if len(grad_region) > 0:
                # Find the maximum gradient in the search region
                max_grad_idx = np.argmax(grad_region) + search_start
                alpha_gradient = max_grad_idx
            else:
                alpha_gradient = alpha_threshold
        else:
            alpha_gradient = alpha_threshold
    else:
        alpha_gradient = alpha_threshold
    
    # Use the smaller of the two estimates (more conservative)
    # This avoids including too much diffuse scattering
    alpha = min(alpha_threshold, alpha_gradient + 2)  # +2 for margin
    
    # Sanity check: alpha should be reasonable fraction of detector
    alpha = max(alpha, 3)  # At least 3 pixels
    alpha = min(alpha, max_r - 1)  # At most max_r - 1
    
    return alpha, radial, counts
