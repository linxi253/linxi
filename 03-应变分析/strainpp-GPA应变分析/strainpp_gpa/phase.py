"""
Phase analysis for a single reciprocal lattice vector (g-vector).

This module implements the core GPA phase extraction pipeline:
1. Apply Gaussian mask to a Bragg spot in the FFT
2. Compute inverse FFT to get the complex lattice fringe image
3. Extract and unwrap the geometric phase
4. Refine the g-vector position using a reference region
5. Compute phase derivatives via Fourier-domain convolution
"""

import numpy as np
from typing import Tuple, Optional

from .utils import (
    forward_fft, backward_fft, extract_phase, wrap_to_pi,
    reference_plane, gaussian_mask,
    phase_derivative_fourier, phase_gradient, unwrap_phase_2d
)

__all__ = ['ComputationCancelled', 'Phase']


class ComputationCancelled(RuntimeError):
    """
    Raised when a cooperative cancellation check fires during phase work.

    :class:`~strainpp_gpa.gpa.GPAComputationCancelled` subclasses this so a
    single ``except ComputationCancelled`` covers both phase-level and
    GPA-level cancellation.
    """


class Phase:
    """
    Geometric phase analysis for a single g-vector.

    Manages the complete pipeline from FFT masking through phase extraction
    to differentiation, for one reciprocal lattice vector.

    Attributes
    ----------
    gx, gy : float
        Reciprocal lattice vector components (cycles per image).
    sigma : float
        Gaussian mask width (standard deviation in pixels).
    raw_phase : np.ndarray or None
        Raw (unwrapped) phase after reference plane subtraction.
    wrapped_phase : np.ndarray or None
        Phase wrapped to [-π, π].
    refined : bool
        Whether the g-vector has been refined.
    """

    def __init__(self):
        """Initialize an empty Phase object."""
        self.gx: float = 0.0
        self.gy: float = 0.0
        self.sigma: float = 5.0
        self.raw_phase: Optional[np.ndarray] = None
        self.wrapped_phase: Optional[np.ndarray] = None
        self.unwrapped_phase: Optional[np.ndarray] = None
        self.amplitude: Optional[np.ndarray] = None
        self._fft: Optional[np.ndarray] = None
        self._h_prime: Optional[np.ndarray] = None  # H'_g(r) = IFFT(masked FFT)
        self._image_shape: Tuple[int, int] = (0, 0)
        self.refined: bool = False

    # ==========================================================================
    # Core Pipeline
    # ==========================================================================

    def set_g_vector(self, gx: float, gy: float, sigma: float = 5.0):
        """
        Set the reciprocal lattice vector and mask width.

        Parameters
        ----------
        gx, gy : float
            G-vector components in pixels (in the FFT coordinate system).
        sigma : float
            Gaussian mask sigma in pixels. Default 5.0.
            (The full mask radius is approximately 3*sigma.)
        """
        if not np.isfinite(gx) or not np.isfinite(gy):
            raise ValueError("G-vector components must be finite.")
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"Gaussian sigma must be finite and positive, got {sigma}.")
        self.gx = float(gx)
        self.gy = float(gy)
        self.sigma = float(sigma)
        self.refined = False

    def set_fft(self, fft_data: np.ndarray, check_finite: bool = True):
        """
        Set the pre-computed FFT of the image.

        Parameters
        ----------
        fft_data : np.ndarray (M, N), complex
            FFT of the original image (zero-frequency centered).
        check_finite : bool
            Validate that the FFT contains no NaN or infinite values. Callers
            that already validated the same buffer (e.g. ``GPA.load_image``
            validating the source image) can pass False to skip the redundant
            full-array scan.

        Notes
        -----
        Stores a reference (not a copy) since the FFT data is only read
        during mask application, never modified in-place. This saves
        significant memory for large images (~64MB per 2048x2048 complex128).
        """
        fft_data = np.asarray(fft_data)
        if fft_data.ndim != 2:
            raise ValueError(f"FFT data must be 2D, got shape {fft_data.shape}.")
        if check_finite and not np.all(np.isfinite(fft_data)):
            raise ValueError("FFT data contains NaN or infinite values.")
        self._fft = fft_data  # shared reference, read-only usage
        self._image_shape = fft_data.shape
        self.raw_phase = None
        self.wrapped_phase = None
        self.unwrapped_phase = None
        self.amplitude = None
        self._h_prime = None

    def compute_from_image(self, image: np.ndarray, gx: float, gy: float,
                           sigma: float = 5.0, use_hann: bool = False):
        """
        Run the full phase extraction pipeline from a raw image.

        Parameters
        ----------
        image : np.ndarray (M, N)
            Input TEM image.
        gx, gy : float
            G-vector components in the FFT.
        sigma : float
            Gaussian mask sigma.
        use_hann : bool
            Whether to apply a Hann window before FFT.
        """
        from .utils import hann_window_2d, apply_hann_window

        self.set_g_vector(gx, gy, sigma)

        if use_hann:
            window = hann_window_2d(image.shape)
            image = apply_hann_window(image, window)

        self._fft = forward_fft(image.astype(np.float64))
        self._image_shape = image.shape

        self.compute()

    def compute(self):
        """
        Run the complete phase-extraction pipeline for the current g-vector:
        apply the Gaussian mask, compute the inverse FFT, extract the raw
        phase (minus the reference plane), and wrap it to [-pi, pi].

        Requires the FFT to be set first via :meth:`set_fft` (or
        :meth:`compute_from_image`) and the g-vector via :meth:`set_g_vector`.
        """
        self._apply_mask_and_invert()
        self._extract_phase()
        self._wrap_phase()

    def _apply_mask_and_invert(self):
        """
        Apply the Gaussian mask to the FFT and compute the inverse FFT.

        H'_g(r) = IFFT{ FFT(image) * Gaussian_mask(g_pos, sigma) }
        """
        if self._fft is None:
            raise RuntimeError("FFT not set. Call set_fft() or compute_from_image() first.")

        M, N = self._image_shape
        # forward_fft uses fftshift, so for both even and odd dimensions the
        # DC component sits exactly at array position (M // 2, N // 2).
        # The g-vector (gx, gy) is relative to DC, so the array position of
        # the Bragg peak is (N // 2 + gx, M // 2 + gy).
        mask = gaussian_mask(
            (M, N),
            (self.gx + N // 2, self.gy + M // 2),
            self.sigma,
        )

        masked_fft = self._fft * mask
        self._h_prime = backward_fft(masked_fft)
        self.amplitude = np.abs(self._h_prime)

    def _extract_phase(self):
        """
        Extract the raw geometric phase and subtract the reference plane.

        P_g(r) = atan2(Im[H'_g(r)], Re[H'_g(r)]) - 2*pi * (i*g_y + j*g_x)
        """
        if self._h_prime is None:
            raise RuntimeError("H'_g not computed. Call _apply_mask_and_invert() first.")

        M, N = self._image_shape
        raw = extract_phase(self._h_prime)
        ref = reference_plane(M, N, self.gx, self.gy)

        self.raw_phase = raw - ref

    def _wrap_phase(self):
        """
        Wrap the raw phase to [-pi, pi].

        P_wrapped(i) = P(i) - round(P(i) / (2*pi)) * 2*pi

        Unwrapping is intentionally lazy because strain uses wrapped-safe phase
        differences and does not require a global displacement reconstruction.
        """
        if self.raw_phase is None:
            raise RuntimeError("Raw phase not computed.")
        self.wrapped_phase = wrap_to_pi(self.raw_phase)
        self.unwrapped_phase = None

    def ensure_unwrapped_phase(self) -> np.ndarray:
        """Compute and cache the continuous phase needed for displacement."""
        if self.wrapped_phase is None:
            raise RuntimeError("Wrapped phase not computed.")
        if self.unwrapped_phase is None:
            self.unwrapped_phase = unwrap_phase_2d(self.wrapped_phase)
        return self.unwrapped_phase

    # ==========================================================================
    # G-Vector Refinement
    # ==========================================================================

    def refine_g_vector(self, mask: np.ndarray) -> Tuple[float, float]:
        """
        Refine the g-vector by fitting the residual phase gradient in a
        homogeneous reference region.

        The selected region (mask) should correspond to an area of the sample
        with known homogeneous (e.g., unstrained) lattice spacing. The
        residual phase gradient inside the region is estimated robustly as
        the median of wrapped-safe phase differences (robust to isolated
        phase residues and low-amplitude pixels), and that slope provides
        the correction to the g-vector:

            delta_gx = median(dP/dx) * N / (2*pi)
            delta_gy = median(dP/dy) * M / (2*pi)

        Parameters
        ----------
        mask : np.ndarray (M, N), bool
            Binary mask where True indicates the reference region.

        Returns
        -------
        (delta_gx, delta_gy) : float
            Correction applied to the g-vector.
        """
        if self.wrapped_phase is None:
            raise RuntimeError("Phase not computed. Call compute_from_image() first.")

        M, N = self._image_shape
        mask = np.asarray(mask)
        if mask.shape != (M, N):
            raise ValueError(
                f"Reference mask shape {mask.shape} does not match phase shape {(M, N)}."
            )
        mask = mask.astype(bool, copy=False)
        rows, cols = np.where(mask)

        if len(rows) < 9:
            raise ValueError("Reference region must contain at least 9 pixels.")
        if np.ptp(rows) < 2 or np.ptp(cols) < 2:
            raise ValueError("Reference region must span at least 3 rows and 3 columns.")

        # Estimate the residual phase-plane slope from wrapped-safe gradients.
        # The median is robust to isolated phase residues and low-amplitude pixels.
        dphase_dx, dphase_dy = phase_gradient(self.wrapped_phase)
        if self.amplitude is not None:
            quality = self.quality_mask()
            sample_mask = mask & quality
            if np.count_nonzero(sample_mask) < 9:
                raise ValueError(
                    "Too few reliable pixels remain in the reference region."
                )
        else:
            sample_mask = mask

        slope_x = float(np.median(dphase_dx[sample_mask]))
        slope_y = float(np.median(dphase_dy[sample_mask]))

        # gx/gy are FFT pixel coordinates, while slopes are radians per image
        # pixel, so conversion requires the corresponding image dimension.
        delta_gx = slope_x * N / (2.0 * np.pi)
        delta_gy = slope_y * M / (2.0 * np.pi)

        # Update g-vector
        self.gx += delta_gx
        self.gy += delta_gy
        self.refined = True

        # Recompute the full pipeline with the refined g-vector:
        # re-apply mask at new position → IFFT → extract phase → wrap
        self.compute()

        return delta_gx, delta_gy

    def quality_mask(self, relative_threshold: float = 0.10,
                     percentile: float = 0.0) -> np.ndarray:
        """
        Return pixels whose complex Bragg amplitude supports a reliable phase.

        The threshold is a fraction of the maximum amplitude. An optional
        percentile floor can be requested by API callers, but defaults to zero
        so a fixed proportion of otherwise valid pixels is never discarded.
        Phase-circulation residues are also dilated by roughly one third of the
        Fourier-mask sigma. This removes Hann-window boundaries, weak-signal
        areas, zeros, and phase-singular cores.
        """
        if self.amplitude is None:
            raise RuntimeError("Bragg amplitude has not been computed.")
        if not np.isfinite(relative_threshold) or relative_threshold < 0:
            raise ValueError("relative_threshold must be finite and non-negative.")
        if not 0 <= percentile < 100:
            raise ValueError("percentile must be in [0, 100).")
        finite = np.isfinite(self.amplitude)
        if not np.any(finite):
            return np.zeros(self.amplitude.shape, dtype=bool)
        values = self.amplitude[finite]
        threshold = max(
            float(values.max()) * relative_threshold,
            float(np.percentile(values, percentile)),
            np.finfo(np.float64).tiny,
        )
        valid = finite & (self.amplitude >= threshold)

        if self.wrapped_phase is not None and min(self.wrapped_phase.shape) >= 2:
            from scipy.ndimage import binary_dilation

            phase = self.wrapped_phase
            step_x = np.angle(np.exp(1j * np.diff(phase, axis=1)))
            step_y = np.angle(np.exp(1j * np.diff(phase, axis=0)))
            circulation = (
                step_x[:-1, :]
                + step_y[:, 1:]
                - step_x[1:, :]
                - step_y[:, :-1]
            )
            residues = np.abs(circulation) > np.pi
            if np.any(residues):
                singular = np.zeros(phase.shape, dtype=bool)
                singular[:-1, :-1] |= residues
                singular[:-1, 1:] |= residues
                singular[1:, :-1] |= residues
                singular[1:, 1:] |= residues
                singular = binary_dilation(
                    singular,
                    iterations=max(1, int(np.ceil(self.sigma / 3.0))),
                )
                valid &= ~singular
        return valid

    def refine_iterative(self, mask: np.ndarray, max_iter: int = 5,
                         tolerance: float = 1e-4,
                         cancel_check=None) -> list:
        """
        Iteratively refine the g-vector until convergence.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask for the reference region.
        max_iter : int
            Maximum number of iterations.
        tolerance : float
            Convergence tolerance on the correction magnitude.
        cancel_check : callable, optional
            A zero-argument callable returning True when the refinement
            should stop early. It is polled before each iteration and raises
            :class:`ComputationCancelled`.

        Returns
        -------
        list of (delta_gx, delta_gy)
            Corrections applied at each iteration.
        """
        corrections = []
        for _ in range(max_iter):
            if cancel_check is not None and cancel_check():
                raise ComputationCancelled("G-vector refinement cancelled.")
            dgx, dgy = self.refine_g_vector(mask)
            corrections.append((dgx, dgy))
            if np.sqrt(dgx**2 + dgy**2) < tolerance:
                break
        return corrections

    # ==========================================================================
    # Differentiation
    # ==========================================================================

    def derivative_x(self, pixel_size: float = 1.0) -> np.ndarray:
        """
        Compute dP/dx using Fourier-domain convolution.

        Parameters
        ----------
        pixel_size : float
            Physical pixel size for scaling.

        Returns
        -------
        np.ndarray (M, N)
        """
        if self.wrapped_phase is None:
            raise RuntimeError("Wrapped phase not computed.")
        return phase_derivative_fourier(self.wrapped_phase, axis=1, pixel_size=pixel_size)

    def derivative_y(self, pixel_size: float = 1.0) -> np.ndarray:
        """
        Compute dP/dy using Fourier-domain convolution.

        Parameters
        ----------
        pixel_size : float
            Physical pixel size for scaling.

        Returns
        -------
        np.ndarray (M, N)
        """
        if self.wrapped_phase is None:
            raise RuntimeError("Wrapped phase not computed.")
        return phase_derivative_fourier(self.wrapped_phase, axis=0, pixel_size=pixel_size)

    def gradient(self, pixel_size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the phase gradient (dP/dx, dP/dy).

        Parameters
        ----------
        pixel_size : float
            Physical pixel size.

        Returns
        -------
        (dP_dx, dP_dy)
        """
        if self.wrapped_phase is None:
            raise RuntimeError("Wrapped phase not computed.")
        return phase_gradient(self.wrapped_phase, pixel_size)

    # ==========================================================================
    # Properties
    # ==========================================================================

    @property
    def shape(self) -> Tuple[int, int]:
        """Image shape (M, N)."""
        return self._image_shape
