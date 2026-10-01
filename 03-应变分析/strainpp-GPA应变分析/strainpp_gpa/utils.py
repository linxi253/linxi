"""
Core utility functions for Geometric Phase Analysis.

Provides FFT wrappers, window functions, phase wrapping, coordinate transforms,
and other mathematical utilities used throughout the GPA pipeline.

All public FFT operations keep zero frequency at array index
``(rows // 2, columns // 2)`` for both even and odd image dimensions.
"""

import os

import numpy as np
from scipy import fft as _sfft
from scipy.fft import fftshift, ifftshift
from typing import Tuple, Optional

__all__ = [
    'forward_fft', 'backward_fft', 'hann_window_2d', 'apply_hann_window',
    'extract_phase', 'wrap_to_pi', 'unwrap_phase_2d', 'reference_plane',
    'gaussian_mask', 'estimate_g_vector_radius', 'detect_bragg_peaks',
    'rotation_matrix_2d', 'phase_derivative_fourier', 'phase_gradient',
    'estimate_peak_bytes',
]


# ==============================================================================
# FFT Utilities
# ==============================================================================

def _default_fft_workers() -> int:
    """Worker count for parallel FFTs; ``STRAINPP_FFT_WORKERS`` overrides.

    The default of ``-1`` uses every core, which speeds up large images.
    Results are bit-identical regardless of the worker count.
    """
    raw = os.environ.get('STRAINPP_FFT_WORKERS', '-1')
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return -1
    if value == 0:
        return 1
    return value


def forward_fft(image: np.ndarray, workers: Optional[int] = None) -> np.ndarray:
    """
    Compute the 2D forward FFT of a real or complex image.

    Uses ``fftshift(fft2(image))`` so centering is exact for both even and odd
    image dimensions. Multiplication by ``(-1) ** (i+j)`` is not equivalent for
    odd dimensions and causes severe spectral leakage.

    Parameters
    ----------
    image : np.ndarray (M, N), real or complex
        Input image.
    workers : int, optional
        Parallel FFT worker count. ``None`` uses the ``STRAINPP_FFT_WORKERS``
        environment variable (default: all cores); ``-1`` also means all cores.
        The result does not depend on this value.

    Returns
    -------
    np.ndarray (M, N), complex
        FFT of the image with zero-frequency centered.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"FFT input must be 2D, got shape {image.shape}.")
    if workers is None:
        workers = _default_fft_workers()
    return fftshift(_sfft.fft2(image, workers=workers))


def backward_fft(fft_data: np.ndarray,
                 workers: Optional[int] = None) -> np.ndarray:
    """
    Compute the 2D inverse FFT, undoing the pre-FFT shift.

    Parameters
    ----------
    fft_data : np.ndarray (M, N), complex
        FFT data (with zero-frequency assumed centered).
    workers : int, optional
        Parallel FFT worker count (see :func:`forward_fft`).

    Returns
    -------
    np.ndarray (M, N), complex
        Inverse FFT result.
    """
    fft_data = np.asarray(fft_data)
    if fft_data.ndim != 2:
        raise ValueError(f"Inverse FFT input must be 2D, got shape {fft_data.shape}.")
    if workers is None:
        workers = _default_fft_workers()
    return _sfft.ifft2(ifftshift(fft_data), workers=workers)


def estimate_peak_bytes(shape: Tuple[int, int]) -> int:
    """
    Conservative peak-memory estimate (bytes) for the full GPA pipeline.

    Budget per pixel, based on the buffers ``GPA.load_image`` and
    ``GPA.compute`` hold simultaneously on a 2-g-vector analysis:
    the shared FFT and two per-phase complex fringe fields (3 x 16 B),
    image + power spectrum + two raw/wrapped/unwrapped phase sets and
    amplitudes (~13 x 8 B), the masked-FFT transient (16 B), and the
    output tensor/displacement fields with their ``np.where`` masking
    temporaries (~13 x 8 B).
    """
    pixels = int(shape[0]) * int(shape[1])
    if pixels <= 0:
        raise ValueError(f"shape must be positive, got {shape}.")
    return pixels * (3 * 16 + 26 * 8 + 16)


# ==============================================================================
# Window Functions
# ==============================================================================

def hann_window_2d(shape: Tuple[int, int]) -> np.ndarray:
    """
    Create a separable 2D Hann (Hanning) window.

    w(i, j) = h_y(i) * h_x(j)
    where h(i) = 0.5 * (1 - cos(2*pi*i / (N-1)))

    Parameters
    ----------
    shape : (M, N)
        Height and width of the window.

    Returns
    -------
    np.ndarray (M, N)
        2D Hann window.
    """
    M, N = shape
    h_x = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N) / max(N - 1, 1)))
    h_y = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(M) / max(M - 1, 1)))
    return np.outer(h_y, h_x)


def apply_hann_window(image: np.ndarray, window: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply a 2D Hann window to an image to reduce FFT edge artifacts.

    Parameters
    ----------
    image : np.ndarray (M, N)
        Input image.
    window : np.ndarray (M, N), optional
        Pre-computed window. If None, a new window is created.

    Returns
    -------
    np.ndarray (M, N)
        Windowed image.
    """
    if window is None:
        window = hann_window_2d(image.shape)
    return image.astype(np.float64) * window


# ==============================================================================
# Phase Utilities
# ==============================================================================

def extract_phase(complex_data: np.ndarray) -> np.ndarray:
    """
    Extract the phase (argument) of complex data.

    Parameters
    ----------
    complex_data : np.ndarray
        Complex-valued data.

    Returns
    -------
    np.ndarray
        Phase in radians, in range [-pi, pi].
    """
    return np.arctan2(complex_data.imag, complex_data.real)


def wrap_to_pi(phase: np.ndarray) -> np.ndarray:
    """
    Wrap phase values to the range [-pi, pi].

    Uses the formula: wrapped = phase - round(phase / (2*pi)) * 2*pi
    This matches the point-wise wrapping approach in the original Strain++.

    Parameters
    ----------
    phase : np.ndarray
        Unwrapped phase values.

    Returns
    -------
    np.ndarray
        Phase wrapped to [-pi, pi].
    """
    return phase - np.round(phase / (2.0 * np.pi)) * (2.0 * np.pi)


def unwrap_phase_2d(phase: np.ndarray) -> np.ndarray:
    """
    Unwrap a smooth 2D phase field for displacement reconstruction.

    Strategy for residues / branch cuts
    -----------------------------------
    This routine is intentionally simple and only intended for smooth GPA
    phase maps. It does **not** implement branch-cut placement: residues
    (pixels where the wrapped phase circulates by ±2π, e.g. at dislocation
    cores or inside the Fourier-mask dilated ring) are unwrapped along
    whatever path the sequential pass takes.

    The residual risk is mitigated by three upstream mechanisms:

    1. The wrapped gradient used for strain is computed wrap-safely
       (:func:`phase_derivative_fourier`), so strain never depends on this
       unwrapping; only the displacement *display* does.
    2. ``Phase.quality_mask`` flags residue-dilated pixels as unreliable, and
       consumers are expected to read displacement together with that mask.
    3. Two axis orders (x-then-y and y-then-x) are unwrapped and the one with
       the smaller wrapped-gradient residual is kept, which rejects the worse
       path on noisy fields.

    Phase fields with significant residue density need a quality-guided
    unwrapper; the current result should then be treated as qualitative.
    """
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2:
        raise ValueError(f"Phase must be 2D, got shape {phase.shape}.")
    if not np.all(np.isfinite(phase)):
        raise ValueError("Phase contains NaN or infinite values.")

    xy = np.unwrap(np.unwrap(phase, axis=1), axis=0)
    yx = np.unwrap(np.unwrap(phase, axis=0), axis=1)

    def residual(candidate: np.ndarray) -> float:
        dx = np.diff(candidate, axis=1)
        dy = np.diff(candidate, axis=0)
        wrapped_dx = np.angle(np.exp(1j * np.diff(phase, axis=1)))
        wrapped_dy = np.angle(np.exp(1j * np.diff(phase, axis=0)))
        return float(
            np.mean(np.abs(dx - wrapped_dx))
            + np.mean(np.abs(dy - wrapped_dy))
        )

    selected = xy if residual(xy) <= residual(yx) else yx
    # Remove the arbitrary 2π offset while preserving relative displacement.
    return selected - selected[0, 0] + phase[0, 0]


def reference_plane(M: int, N: int, gx: float, gy: float) -> np.ndarray:
    """
    Create the reference phase plane for a perfect lattice.

    P_ref(i, j) = 2*pi * (j * gx / N + i * gy / M)

    This is the expected phase contribution from a perfect lattice with
    reciprocal lattice vector g = (gx, gy) in cycles/image units.
    Subtracting this from the raw phase isolates the deformation-induced
    phase variation (geometric phase).

    Parameters
    ----------
    M, N : int
        Image dimensions (rows, columns).
    gx, gy : float
        Reciprocal lattice vector components in cycles/image units
        (i.e., FFT pixel coordinates relative to DC).

    Returns
    -------
    np.ndarray (M, N)
        Reference phase plane.
    """
    i_idx = np.arange(M).reshape(-1, 1)  # row index
    j_idx = np.arange(N).reshape(1, -1)  # column index
    # Normalize by image dimensions: gx cycles over N pixels, gy cycles over M pixels
    return 2.0 * np.pi * (j_idx * gx / N + i_idx * gy / M)


# ==============================================================================
# Gaussian Mask
# ==============================================================================

def gaussian_mask(shape: Tuple[int, int], center: Tuple[float, float],
                  sigma: float) -> np.ndarray:
    """
    Create a 2D Gaussian mask centered at (cx, cy).

    mask(i, j) = exp(-0.5 * ((i - cy)^2 + (j - cx)^2) / sigma^2)

    Parameters
    ----------
    shape : (M, N)
        Shape of the mask (rows, columns).
    center : (cx, cy)
        Center position (column, row / x, y).
    sigma : float
        Standard deviation of the Gaussian.

    Returns
    -------
    np.ndarray (M, N)
        Gaussian mask with values in [0, 1].
    """
    M, N = shape
    cx, cy = center

    # Wrap coordinates for periodic boundary conditions
    # In the FFT, pixel coordinates are periodic
    i = np.arange(M).reshape(-1, 1)
    j = np.arange(N).reshape(1, -1)

    # Compute distances with periodic wrapping
    di = np.minimum(np.abs(i - cy), M - np.abs(i - cy))
    dj = np.minimum(np.abs(j - cx), N - np.abs(j - cx))
    dist_sq = di**2 + dj**2

    return np.exp(-0.5 * dist_sq / (sigma**2))


# ==============================================================================
# G-Vector Detection
# ==============================================================================

def estimate_g_vector_radius(power_spectrum: np.ndarray) -> float:
    """
    Estimate the radius of the smallest reciprocal lattice vector (Bragg peak).

    The input may be a log-magnitude spectrum. Local prominence relative to a
    smooth background is used instead of thresholding absolute log values,
    which is invalid when the spectrum contains negative values.

    Parameters
    ----------
    power_spectrum : np.ndarray (M, N)
        Log power spectrum (zero-frequency centered).

    Returns
    -------
    float
        Estimated radius (in pixels) of the first Bragg peak from the center.
    """
    ps = np.asarray(power_spectrum, dtype=np.float64)
    if ps.ndim != 2:
        raise ValueError(f"Power spectrum must be 2D, got shape {ps.shape}.")
    if min(ps.shape) < 8:
        raise ValueError("Image is too small for reliable Bragg peak detection.")
    if not np.any(np.isfinite(ps)):
        raise ValueError("Power spectrum contains no finite values.")

    from scipy.ndimage import gaussian_filter, maximum_filter

    finite_floor = float(np.nanmin(ps[np.isfinite(ps)]))
    ps = np.nan_to_num(ps, nan=finite_floor, posinf=finite_floor, neginf=finite_floor)
    rows, cols = ps.shape
    cy, cx = rows // 2, cols // 2
    yy, xx = np.ogrid[:rows, :cols]
    radius_grid = np.hypot(yy - cy, xx - cx)
    max_radius = min(cy, cx, rows - 1 - cy, cols - 1 - cx)
    min_radius = max(3.0, min(rows, cols) * 0.02)

    background_sigma = max(2.0, min(rows, cols) / 64.0)
    prominence = ps - gaussian_filter(ps, sigma=background_sigma, mode="nearest")
    local_max = ps == maximum_filter(ps, size=3, mode="nearest")
    search_mask = (
        local_max
        & (radius_grid >= min_radius)
        & (radius_grid <= max_radius)
    )

    values = prominence[search_mask]
    if values.size == 0:
        raise ValueError("No Bragg peak candidates were found.")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = max(1.4826 * mad, np.finfo(float).eps)
    threshold = max(median + 6.0 * robust_sigma, 0.15 * float(values.max()))
    # A log spectrum can give numerical FFT noise deceptively high local
    # prominence when the surrounding floor is near machine precision. Require
    # a candidate to also lie within four decades of the strongest non-DC local
    # maximum. Peaks weaker than that are not a reliable automatic choice.
    local_intensities = ps[search_mask]
    intensity_threshold = max(
        float(local_intensities.max()) - 4.0,
        float(np.quantile(local_intensities, 0.90)),
    )
    candidates = (
        search_mask
        & (prominence >= threshold)
        & (ps >= intensity_threshold)
    )
    if not np.any(candidates):
        raise ValueError("No statistically significant Bragg peaks were found.")

    candidate_r = radius_grid[candidates]
    candidate_p = prominence[candidates]
    radial_scores = np.bincount(
        np.rint(candidate_r).astype(int),
        weights=candidate_p,
        minlength=int(max_radius) + 2,
    )
    if radial_scores.max() <= 0:
        raise ValueError("Bragg peak prominence is too low.")

    strong_rings = np.where(radial_scores >= 0.25 * radial_scores.max())[0]
    strong_rings = strong_rings[strong_rings >= int(np.floor(min_radius))]
    if strong_rings.size == 0:
        raise ValueError("No reliable Bragg ring was found.")
    return float(strong_rings.min())


def detect_bragg_peaks(power_spectrum: np.ndarray, radius: float,
                       n_peaks: int = 10) -> list:
    """
    Detect Bragg peak candidates in the power spectrum using vectorized operations.

    Uses scipy.ndimage.maximum_filter for fast local maximum detection,
    replacing the slow pure-Python nested loop approach.

    Parameters
    ----------
    power_spectrum : np.ndarray (M, N)
        Log power spectrum (DC centered at M/2, N/2).
    radius : float
        Estimated Bragg peak radius from center.
    n_peaks : int
        Maximum number of peaks to return.

    Returns
    -------
    list of (gx, gy, intensity, r)
        Peak candidates sorted by intensity (descending).
        gx, gy are relative to DC center.
    """
    ps = np.asarray(power_spectrum, dtype=np.float64)
    if ps.ndim != 2:
        raise ValueError(f"Power spectrum must be 2D, got shape {ps.shape}.")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"Radius must be finite and positive, got {radius}.")
    if n_peaks <= 0:
        return []

    from scipy.ndimage import gaussian_filter, maximum_filter

    finite_floor = float(np.nanmin(ps[np.isfinite(ps)]))
    ps = np.nan_to_num(ps, nan=finite_floor, posinf=finite_floor, neginf=finite_floor)
    M, N = ps.shape
    cy, cx = M // 2, N // 2
    yy, xx = np.ogrid[:M, :N]
    dist = np.hypot(yy - cy, xx - cx)
    # Search beyond the shortest ring so rectangular/anisotropic lattices can
    # contribute a non-collinear family whose spacing differs substantially.
    # These are candidates for user review, never automatically selected Gs.
    r_min = max(3.0, 0.5 * radius)
    r_max = min(
        min(cx, cy, N - 1 - cx, M - 1 - cy),
        3.0 * radius,
    )
    if r_max <= r_min:
        return []

    annular_mask = (dist >= r_min) & (dist <= r_max)
    background = gaussian_filter(
        ps,
        sigma=max(2.0, min(M, N) / 64.0),
        mode="nearest",
    )
    prominence = ps - background
    local_max = maximum_filter(ps, size=3, mode="nearest")
    is_peak = (ps == local_max) & annular_mask
    candidate_prominence = prominence[is_peak]
    if candidate_prominence.size == 0:
        return []
    prominence_threshold = max(
        float(np.quantile(candidate_prominence, 0.75)),
        0.10 * float(candidate_prominence.max()),
    )
    candidate_intensity = ps[is_peak]
    intensity_threshold = max(
        float(candidate_intensity.max()) - 4.0,
        float(np.quantile(candidate_intensity, 0.90)),
    )
    is_peak &= (
        (prominence >= prominence_threshold)
        & (ps >= intensity_threshold)
    )

    # Extract peak coordinates
    peak_rows, peak_cols = np.where(is_peak)

    if len(peak_rows) == 0:
        return []

    # Compute properties for all peaks (vectorized)
    peak_gx = peak_cols.astype(float) - cx
    peak_gy = peak_rows.astype(float) - cy
    peak_intensity = ps[peak_rows, peak_cols]
    peak_prominence = prominence[peak_rows, peak_cols]
    peak_r = dist[peak_rows, peak_cols]

    # Sort by local prominence first, then absolute intensity.
    sort_idx = np.lexsort((-peak_intensity, -peak_prominence))

    # Spatial non-maximum suppression removes multiple pixels along the same
    # broad/streaked Bragg spot while retaining its centrosymmetric partner.
    min_separation = max(3.0, 0.15 * radius)
    peaks = []
    selected_coordinates = []
    for idx in sort_idx:
        coordinate = np.array([peak_cols[idx], peak_rows[idx]], dtype=float)
        if any(
            np.linalg.norm(coordinate - selected) < min_separation
            for selected in selected_coordinates
        ):
            continue
        selected_coordinates.append(coordinate)
        peaks.append((
            float(peak_gx[idx]),
            float(peak_gy[idx]),
            float(peak_intensity[idx]),
            float(peak_r[idx])
        ))
        if len(peaks) >= n_peaks:
            break

    return peaks


# ==============================================================================
# Rotation Matrix
# ==============================================================================

def rotation_matrix_2d(theta_rad: float) -> np.ndarray:
    """
    Create a 2D rotation matrix.

    R = [[cos(theta), -sin(theta)],
         [sin(theta),  cos(theta)]]

    Parameters
    ----------
    theta_rad : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray (2, 2)
        Rotation matrix.
    """
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    return np.array([[cos_t, -sin_t],
                     [sin_t,  cos_t]])


# ==============================================================================
# Derivative Computation
# ==============================================================================

def phase_derivative_fourier(phase: np.ndarray, axis: int,
                              pixel_size: float = 1.0) -> np.ndarray:
    """
    Compute the derivative of a wrapped phase map without periodic edge coupling.

    The historical function name is retained for API compatibility. Adjacent
    phase steps are computed as principal arguments, then centered by averaging
    neighbouring steps. This gives the correct sign, avoids the ``sin(dP)``
    approximation, and uses one-sided differences at image boundaries.

    Parameters
    ----------
    phase : np.ndarray (M, N)
        Wrapped phase (in radians, range [-π, π]).
    axis : int
        Axis along which to differentiate (0 for y/row, 1 for x/column).
    pixel_size : float
        Physical pixel size for scaling the derivative.

    Returns
    -------
    np.ndarray (M, N)
        Phase derivative dP/dx or dP/dy.
    """
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2:
        raise ValueError(f"Phase must be 2D, got shape {phase.shape}.")
    if axis not in (0, 1):
        raise ValueError(f"Axis must be 0 or 1, got {axis}.")
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError(f"Pixel size must be finite and positive, got {pixel_size}.")
    if phase.shape[axis] < 2:
        raise ValueError("At least two pixels are required along the derivative axis.")
    if not np.all(np.isfinite(phase)):
        raise ValueError("Phase contains NaN or infinite values.")

    steps = np.angle(np.exp(1j * np.diff(phase, axis=axis))) / pixel_size
    derivative = np.empty_like(phase, dtype=np.float64)
    if axis == 1:
        derivative[:, 0] = steps[:, 0]
        derivative[:, -1] = steps[:, -1]
        if phase.shape[1] > 2:
            derivative[:, 1:-1] = 0.5 * (steps[:, :-1] + steps[:, 1:])
    else:
        derivative[0, :] = steps[0, :]
        derivative[-1, :] = steps[-1, :]
        if phase.shape[0] > 2:
            derivative[1:-1, :] = 0.5 * (steps[:-1, :] + steps[1:, :])
    return derivative


def phase_gradient(phase: np.ndarray, pixel_size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the gradient of a wrapped phase map.

    Parameters
    ----------
    phase : np.ndarray (M, N)
        Wrapped phase.
    pixel_size : float
        Physical pixel size.

    Returns
    -------
    (dP_dx, dP_dy) : tuple of np.ndarray
        Phase derivatives in x (column) and y (row) directions.
    """
    dP_dx = phase_derivative_fourier(phase, axis=1, pixel_size=pixel_size)
    dP_dy = phase_derivative_fourier(phase, axis=0, pixel_size=pixel_size)
    return dP_dx, dP_dy
