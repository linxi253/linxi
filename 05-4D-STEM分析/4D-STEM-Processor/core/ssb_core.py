"""
ssb_core.py - Single-Side-Band (SSB) Ptychography implementation for 4D-STEM data.

Algorithm:
  1. FFT over scan positions for each detector pixel
  2. Construct double-overlap (trotter) masks in (Kf, Q) space
  3. Extract phase information from single-side-band regions
  4. Inverse FFT to get complex transmission function
  5. Output amplitude and phase images

Reference: Pennycook et al., Ultramicroscopy 151, 160-167 (2015)
           Yang et al., Nature Communications 7, 12532 (2016)
"""
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os, time, json

# ============================================================
# 1. SYNTHETIC DATA GENERATION (for testing)
# ============================================================

def generate_probe(det_shape, alpha_pixels, defocus_rad=0.0, Cs_rad=0.0):
    """
    Generate a probe-forming aperture and probe function.
    
    Parameters:
        det_shape: (det_y, det_x) - detector dimensions in pixels
        alpha_pixels: convergence semi-angle in pixels (radius of bright-field disk)
        defocus_rad: defocus in radians at aperture edge
        Cs_rad: spherical aberration in radians at aperture edge
    
    Returns:
        aperture: binary mask of the probe-forming aperture
        probe_k: complex probe in Fourier (detector) plane
    """
    det_y, det_x = det_shape
    ky, kx = np.ogrid[-det_y//2:det_y//2, -det_x//2:det_x//2]
    k2 = (kx**2 + ky**2).astype(np.float64)
    k_rad = np.sqrt(k2)
    
    # Normalized spatial frequency (0 to 1 at aperture edge)
    k_norm = np.zeros_like(k_rad)
    if alpha_pixels > 0:
        k_norm = k_rad / alpha_pixels
    
    # Aperture mask
    aperture = (k_rad <= alpha_pixels).astype(np.float64)
    
    # Aberration phase: χ(k) = π·λ·Δf·k² + π/2·Cs·λ³·k⁴ + ...
    # Using normalized k for simplicity
    chi = defocus_rad * k_norm**2 + Cs_rad * k_norm**4
    chi[~aperture.astype(bool)] = 0
    
    probe_k = aperture * np.exp(1j * chi)
    
    return aperture, probe_k


def generate_phase_object(scan_shape, feature_size=3.0, amplitude=0.5):
    """
    Generate a synthetic phase object for testing.
    
    Parameters:
        scan_shape: (scan_y, scan_x)
        feature_size: characteristic feature size in pixels
        amplitude: phase modulation amplitude in radians
    
    Returns:
        obj: complex transmission function ψ(r) = exp(i*phase)
    """
    scan_y, scan_x = scan_shape
    rng = np.random.RandomState(42)
    
    # Create low-frequency phase variation
    ky, kx = np.ogrid[-scan_y//2:scan_y//2, -scan_x//2:scan_x//2]
    k2 = (kx**2 / feature_size**2 + ky**2 / feature_size**2).astype(np.float64)
    
    # Random phase object with 1/f spectrum
    noise = rng.randn(scan_y, scan_x) + 1j * rng.randn(scan_y, scan_x)
    noise_k = fftshift(fft2(noise))
    noise_k = noise_k / (1 + k2)  # 1/f filter
    phase = np.real(ifft2(ifftshift(noise_k)))
    
    # Normalize and scale
    phase = phase - phase.min()
    phase = phase / phase.max() * amplitude
    
    # Add some structure: circles/bumps
    for _ in range(5):
        cy, cx = rng.randint(0, scan_y), rng.randint(0, scan_x)
        ry, rx = rng.randint(2, 5), rng.randint(2, 5)
        y, x = np.ogrid[-cy:scan_y-cy, -cx:scan_x-cx]
        bump = np.exp(-((x/rx)**2 + (y/ry)**2))
        phase += bump * amplitude * 0.5
    
    obj = np.exp(1j * phase)
    return obj


def generate_4dstem_fast(scan_shape, det_shape, alpha_pixels, defocus_rad=0.0):
    """
    Generate synthetic 4D-STEM data using Fourier-space convolution (fast).
    
    This implements the forward model:
        I(K, R) = |FFT[P(r-R) · ψ(r)]|²
    
    Parameters:
        scan_shape: (scan_y, scan_x)
        det_shape: (det_y, det_x) 
        alpha_pixels: convergence semi-angle in detector pixels
        defocus_rad: defocus aberration
    
    Returns:
        datacube: shape (scan_y, scan_x, det_y, det_x)
        obj: ground truth object
        probe_k: probe in Fourier plane
    """
    scan_y, scan_x = scan_shape
    det_y, det_x = det_shape
    
    # Generate object in scan coordinates
    obj = generate_phase_object(scan_shape, feature_size=3.0, amplitude=0.8)
    
    # Generate probe in REAL SPACE at scan resolution
    # The probe is a small focused spot, approximated as a Gaussian
    # Real-space probe width is inversely proportional to alpha
    # probe_width_pixels = scan_pixel_size / (alpha_in_rad * lambda) * ...
    # For synthetic data, use a simple Gaussian
    
    probe_width = scan_y / (alpha_pixels * 2.5)  # empirical scaling
    cy, cx = scan_y // 2, scan_x // 2
    ry, rx = np.ogrid[:scan_y, :scan_x]
    r2 = ((ry - cy)**2 + (rx - cx)**2) / (2 * probe_width**2)
    probe_r = np.exp(-r2).astype(np.complex128)
    
    # Add defocus phase in real space: exp(i * defocus * r^2)
    if defocus_rad != 0:
        phase_r = defocus_rad * ((ry - cy)**2 + (rx - cx)**2) / (scan_y/4)**2
        probe_r *= np.exp(1j * phase_r)
    
    # Normalize probe
    probe_r /= np.sqrt(np.sum(np.abs(probe_r)**2))
    
    # Probe in Fourier (detector) plane
    probe_k = fftshift(fft2(probe_r))
    
    # Generate 4D-STEM data
    datacube = np.zeros((scan_y, scan_x, det_y, det_x), dtype=np.float64)
    
    print(f'    Generating {scan_y}x{scan_x} diffraction patterns...', flush=True)
    
    # Pre-compute object FFT for faster convolution
    for sy in range(scan_y):
        for sx in range(scan_x):
            # Shifted probe: roll probe to scan position
            probe_shifted = np.roll(probe_r, (sy - cy, sx - cx), axis=(0, 1))
            
            # Exit wave
            exit_wave = probe_shifted * obj
            
            # Diffraction pattern
            exit_k = fft2(exit_wave)
            
            # Crop to detector size from center
            # If det > scan, pad; if det < scan, crop
            if det_y <= scan_y and det_x <= scan_x:
                dy0 = scan_y // 2 - det_y // 2
                dx0 = scan_x // 2 - det_x // 2
                diffraction = np.abs(exit_k[dy0:dy0+det_y, dx0:dx0+det_x])**2
            else:
                # Pad to detector size
                pad_y = (det_y - scan_y) // 2
                pad_x = (det_x - scan_x) // 2
                diffraction = np.pad(np.abs(exit_k)**2, 
                                    ((pad_y, det_y-scan_y-pad_y), 
                                     (pad_x, det_x-scan_x-pad_x)))
            
            datacube[sy, sx] = diffraction
        
        if sy % 8 == 0:
            print(f'      scan row {sy}/{scan_y}', flush=True)
    
    return datacube, obj, probe_k


# ============================================================
# 2. SSB PTYCHOGRAPHY ALGORITHM
# ============================================================

def find_center_of_mass(datacube):
    """
    Find the center of the bright-field disk using center-of-mass method.
    Returns (center_y, center_x) in detector coordinates.

    Raises RuntimeError when the total intensity is non-positive
    (background-subtracted data): dividing by it would silently return a
    mirrored centre or NaN.
    """
    scan_y, scan_x, det_y, det_x = datacube.shape

    # Sum over all scan positions to get average CBED
    avg_cbed = np.sum(datacube, axis=(0, 1))

    # Center of mass
    ky, kx = np.meshgrid(np.arange(det_y), np.arange(det_x), indexing='ij')
    total = np.sum(avg_cbed)
    if not total > 0:
        raise RuntimeError(
            "Cannot auto-detect the beam centre: the average diffraction "
            "pattern has non-positive total intensity. The data appears to "
            "be background-subtracted; shift/clip it to positive values "
            "first, or pass an explicit centre.")
    center_y = np.sum(ky * avg_cbed) / total
    center_x = np.sum(kx * avg_cbed) / total

    return center_y, center_x


def create_aperture_mask(det_shape, center, alpha_pixels):
    """Create circular aperture mask centered at `center`."""
    det_y, det_x = det_shape
    ky, kx = np.ogrid[:det_y, :det_x]
    r2 = (ky - center[0])**2 + (kx - center[1])**2
    mask = (r2 <= alpha_pixels**2).astype(np.float64)
    return mask


def double_overlap_plus(aperture, center, alpha_pixels, qy, qx):
    """DO+(Kf, Q) mask: aperture(Kf) ∩ aperture(Kf-Q), minus the
    triple-overlap region where aperture(Kf+Q) also overlaps.

    The shifted apertures are built analytically at the shifted centres.
    ``np.roll`` must NOT be used here: it wraps disk pixels around the
    detector edges and fabricates overlap whenever 2*alpha is comparable
    to the detector size (under-sampled detectors).
    """
    det_y, det_x = aperture.shape
    aperture_shifted = create_aperture_mask(
        (det_y, det_x), (center[0] - qy, center[1] - qx), alpha_pixels)
    aperture_shifted_neg = create_aperture_mask(
        (det_y, det_x), (center[0] + qy, center[1] + qx), alpha_pixels)
    do_plus = aperture * aperture_shifted
    triple = aperture * aperture_shifted * aperture_shifted_neg
    return do_plus * (1 - triple)


def _generate_q_indices(scan_size, info_limit):
    """
    Generate all FFT indices whose corresponding frequency |q| <= info_limit.
    Covers both positive and negative frequencies.
    
    In numpy FFT convention:
      - indices 0..N//2 correspond to frequencies 0..N//2
      - indices N//2+1..N-1 correspond to frequencies -(N//2-1)..-1
    
    Returns list of (fft_index, signed_frequency) tuples, excluding DC.
    """
    max_q = int(info_limit) + 1
    indices = []

    # Positive frequencies: index 1 to min(max_q, N//2). For even N the
    # Nyquist bin N//2 is its own negative frequency, so it belongs to the
    # positive list only (listing it in both made the Q loop process it
    # twice with mirrored DO+ masks, the second overwriting the first).
    for q in range(1, min(max_q + 1, scan_size // 2 + 1)):
        indices.append((q, q))

    # Negative frequencies: index (N - q); exclude the Nyquist bin (even N)
    for q in range(1, min(max_q + 1, (scan_size - 1) // 2 + 1)):
        indices.append((scan_size - q, -q))

    return indices


def ssb_reconstruct(datacube, alpha_pixels, center=None, defocus_rad=0.0, Cs_rad=0.0, 
                    regularization=1e-3, verbose=True, should_stop=None):
    """
    Single-Side-Band ptychography reconstruction.
    
    Parameters:
        datacube: 4D-STEM data, shape (scan_y, scan_x, det_y, det_x)
        alpha_pixels: convergence semi-angle in detector pixels
        center: (cy, cx) bright-field disk center; auto-detected if None
        defocus_rad: defocus in radians at aperture edge
        Cs_rad: spherical aberration in radians at aperture edge
        regularization: Tikhonov regularization parameter
    
    Returns:
        result: dict with 'amplitude', 'phase', 'complex_obj', 'aperture', 'metadata'
    """
    scan_y, scan_x, det_y, det_x = datacube.shape
    
    if verbose:
        print(f'SSB Reconstruction:')
        print(f'  Scan: {scan_y}x{scan_x}, Detector: {det_y}x{det_x}')
        print(f'  Alpha: {alpha_pixels} pixels')
    
    # Step 0: Find center if not provided
    if center is None:
        center = find_center_of_mass(datacube)
        if verbose:
            print(f'  Center (auto): ({center[0]:.1f}, {center[1]:.1f})')
    
    # Step 1: Create aperture in pixel coordinates (centered at actual beam center)
    aperture = create_aperture_mask((det_y, det_x), center, alpha_pixels)

    # Aberration phase chi(k), measured from an arbitrary disk centre. Kept
    # as a function so the shifted probe P(Kf-Q) can be evaluated analytically
    # at the shifted centre instead of np.roll (which wraps disk pixels
    # around the detector edges and invents false overlap regions).
    has_aberrations = (defocus_rad != 0 or Cs_rad != 0)
    if has_aberrations:
        def _chi(cy_c, cx_c):
            ky_full, kx_full = np.meshgrid(np.arange(det_y), np.arange(det_x),
                                           indexing='ij')
            k_r = np.sqrt((ky_full - cy_c) ** 2 + (kx_full - cx_c) ** 2)
            k_norm = np.zeros_like(k_r)
            if alpha_pixels > 0:
                k_norm = k_r / alpha_pixels
            return defocus_rad * k_norm ** 2 + Cs_rad * k_norm ** 4

        chi = _chi(center[0], center[1])
        chi[~aperture.astype(bool)] = 0
        probe_phase = np.exp(1j * chi)
    else:
        probe_phase = np.ones((det_y, det_x), dtype=np.complex128)
    
    # Step 2: FFT over scan positions for each detector pixel
    # G(Kf, Qp) = FFT_{R→Q}[I(Kf, R)]
    if verbose:
        print(f'  Computing FFT over scan positions...')
    
    # Vectorized FFT: fft2 over the first two axes (scan dimensions).
    # Keep the native precision (complex64 for float32 input): an extra
    # astype(complex128) would double the size of the largest array in
    # the whole pipeline for no accuracy benefit at 8-bit/16-bit data.
    G = fft2(datacube, axes=(0, 1))
    
    if verbose:
        print(f'  FFT complete. Building SSB masks...')
    
    # Step 3: SSB reconstruction for each spatial frequency Q
    # Ψ_s(Q) = Σ_{Kf∈DO+(Q)} G(Kf, Q) · Γ*_+(Kf, Q) / Σ |Γ_+|^2
    
    # Output: complex object in Fourier space
    Psi_s = np.zeros((scan_y, scan_x), dtype=np.complex128)
    weight_sum = np.zeros((scan_y, scan_x), dtype=np.float64)
    
    # Information limit: maximum transferred spatial frequency = 2*alpha
    info_limit = 2 * alpha_pixels
    
    # Generate all Q indices covering BOTH positive and negative frequencies
    q_y_list = _generate_q_indices(scan_y, info_limit)
    q_x_list = _generate_q_indices(scan_x, info_limit)
    
    # Also include zero in one direction for axis-aligned Q vectors
    q_y_all = [(0, 0)] + q_y_list  # (fft_index, signed_q)
    q_x_all = [(0, 0)] + q_x_list
    
    # Total Q vectors to process (estimate)
    total_q_est = len(q_y_all) * len(q_x_all)
    if verbose:
        print(f'  Processing Q vectors: {len(q_y_all)} x {len(q_x_all)} = {total_q_est} candidates')
        print(f'  Info limit: {info_limit:.1f} pixels')
    
    q_count = 0
    cancelled = False

    for qy_idx, qy in q_y_all:
        for qx_idx, qx in q_x_all:
            if should_stop is not None and should_stop():
                cancelled = True
                break
            # Skip DC
            if qy == 0 and qx == 0:
                continue
            
            # Check if |Q| is within information limit
            q_magnitude = np.sqrt(qy**2 + qx**2)
            if q_magnitude > info_limit:
                continue
            
            # Build DO_plus mask: Kf where both Kf and Kf-Q are inside the
            # aperture (analytic shifted apertures — see double_overlap_plus).
            do_plus = double_overlap_plus(aperture, center, alpha_pixels,
                                          qy, qx)

            if np.sum(do_plus) < 1:
                continue

            # Probe overlap weight: Γ_+(Kf, Q) = P(Kf) · P*(Kf-Q)
            # With aberrations the shifted probe is the aberration phase
            # evaluated at the shifted centre, masked by the shifted aperture
            # (identical to the rolled array inside DO_plus, without wrap).
            if has_aberrations:
                aperture_shifted = create_aperture_mask(
                    (det_y, det_x), (center[0] - qy, center[1] - qx),
                    alpha_pixels)
                chi_s = _chi(center[0] - qy, center[1] - qx)
                chi_s[aperture_shifted <= 0] = 0
                gamma = do_plus * probe_phase * np.conj(np.exp(1j * chi_s))
            else:
                gamma = do_plus
            
            # Weighted sum over DO_plus region
            G_at_Q = G[qy_idx, qx_idx, :, :]
            
            psi_val = np.sum(G_at_Q * np.conj(gamma))
            w_val = np.sum(np.abs(gamma)**2)
            
            Psi_s[qy_idx, qx_idx] = psi_val
            weight_sum[qy_idx, qx_idx] = w_val
            
            q_count += 1
            if verbose and q_count % 500 == 0:
                print(f'    Q progress: {q_count}', flush=True)
        if cancelled:
            break
    
    if verbose:
        print(f'  Processed {q_count} Q vectors (of {total_q_est} candidates)')

    metadata = {
        'scan_shape': (scan_y, scan_x),
        'det_shape': (det_y, det_x),
        'alpha_pixels': alpha_pixels,
        'center': center,
        'defocus_rad': defocus_rad,
        'Cs_rad': Cs_rad,
        'regularization': regularization,
        'q_vectors_processed': q_count,
    }

    if cancelled:
        # A partial Psi_s is missing Q vectors; inverse-transforming it would
        # produce a *wrong* image rather than an incomplete one, so no
        # reconstruction output is returned.
        if verbose:
            print('  SSB cancelled - no reconstruction output.')
        return {'amplitude': None, 'phase': None, 'complex_obj': None,
                'aperture': aperture, 'metadata': metadata,
                'cancelled': True}

    # Normalize
    valid_mask = weight_sum > 0
    if np.any(valid_mask):
        Psi_s[valid_mask] /= (weight_sum[valid_mask] + regularization * np.max(weight_sum))

    # Step 4: Inverse FFT to get real-space complex object
    if verbose:
        print(f'  Computing inverse FFT...')

    psi_r = ifft2(Psi_s)

    # Step 5: Extract amplitude and phase
    amplitude = np.abs(psi_r)
    phase = np.angle(psi_r)

    result = {
        'amplitude': amplitude,
        'phase': phase,
        'complex_obj': psi_r,
        'aperture': aperture,
        'metadata': metadata,
        'cancelled': cancelled,
    }

    return result


# ============================================================
# 3. VISUALIZATION
# ============================================================

def plot_results(result, save_path=None):
    """Plot SSB reconstruction results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Amplitude
    im0 = axes[0].imshow(result['amplitude'], cmap='gray')
    axes[0].set_title('SSB Amplitude')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    
    # Phase
    im1 = axes[1].imshow(result['phase'], cmap='viridis')
    axes[1].set_title('SSB Phase')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    
    # Aperture
    im2 = axes[2].imshow(result['aperture'], cmap='gray')
    axes[2].set_title('Probe Aperture')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved figure to: {save_path}')
    else:
        plt.show()
    
    plt.close()


# ============================================================
# 4. MAIN - Test with synthetic data
# ============================================================

def test_with_synthetic():
    """Test SSB with synthetic 4D-STEM data."""
    print('='*60)
    print('SSB Ptychography - Synthetic Data Test')
    print('='*60)
    
    # Parameters
    scan_shape = (64, 64)        # Scan grid
    det_shape = (128, 128)       # Detector pixels
    alpha_pixels = 30            # BF disk radius in pixels
    defocus_rad = 2.0            # Small defocus for phase contrast
    
    # Generate synthetic data
    print('\n[1] Generating synthetic data...')
    t0 = time.time()
    
    # Use a smaller test configuration
    # det must be <= scan since detector records subset of scan FFT
    test_scan = (64, 64)
    test_det = (64, 64)  # Same as scan for simplicity
    test_alpha = 12  # BF disk radius in detector pixels
    
    datacube, obj_small, probe_k = generate_4dstem_fast(
        test_scan, test_det, test_alpha, defocus_rad=defocus_rad
    )
    
    print(f'  Datacube shape: {datacube.shape}, memory: {datacube.nbytes/1024/1024:.1f} MB')
    print(f'  Generated in {time.time()-t0:.1f}s')
    
    # Run SSB reconstruction
    print('\n[2] Running SSB reconstruction...')
    t1 = time.time()
    
    result = ssb_reconstruct(
        datacube, 
        alpha_pixels=test_alpha,
        defocus_rad=defocus_rad,
        regularization=1e-3,
        verbose=True
    )
    
    print(f'  Reconstruction completed in {time.time()-t1:.1f}s')
    
    # Plot results
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'results', 'ssb_test')
    os.makedirs(outdir, exist_ok=True)
    
    print('\n[3] Visualizing results...')
    plot_results(result, save_path=os.path.join(outdir, 'ssb_synthetic.png'))
    
    # Also plot ground truth for comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(np.abs(obj_small), cmap='gray')
    axes[0].set_title('Ground Truth Amplitude')
    axes[0].axis('off')
    
    axes[1].imshow(np.angle(obj_small), cmap='viridis')
    axes[1].set_title('Ground Truth Phase')
    axes[1].axis('off')
    
    # Diffraction pattern (average)
    avg_cbed = np.mean(datacube, axis=(0, 1))
    axes[2].imshow(np.log10(avg_cbed + 1), cmap='inferno')
    axes[2].set_title('Avg Diffraction (log10)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'ssb_ground_truth.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print('\nDone! Results saved to:', outdir)
    
    return result


if __name__ == '__main__':
    test_with_synthetic()
