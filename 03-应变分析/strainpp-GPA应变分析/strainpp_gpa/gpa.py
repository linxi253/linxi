"""
Geometric Phase Analysis (GPA) controller.

Manages two Phase objects (one per g-vector) and computes the full strain,
distortion, rotation, and dilatation tensor fields from HRTEM images.

The fundamental GPA relation is:
    P_g(r) = -2π * g · u(r)

where P_g is the geometric phase for reciprocal lattice vector g,
and u(r) is the displacement field.

With two non-colinear g-vectors, the displacement field is:
    [u_x, u_y]^T = -(1/(2π)) * A · [P_g1, P_g2]^T

where A = (G^T)^(-1) and G = [[g1x, g1y], [g2x, g2y]].

Reference:
    Hytch, M. J., Snoeck, E. & Kilaas, R. Ultramicroscopy 74, 131-146 (1998).
"""

import numpy as np
import warnings
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from .phase import Phase, ComputationCancelled
from .utils import (
    forward_fft,
    estimate_g_vector_radius, rotation_matrix_2d,
    phase_gradient, estimate_peak_bytes
)

__all__ = ['GPA', 'GPAOutput', 'GPAComputationCancelled']

# Above this estimated footprint GPA.load_image warns before allocating.
_LARGE_IMAGE_WARNING_BYTES = 4 * (1 << 30)


class GPAComputationCancelled(ComputationCancelled):
    """Raised internally when a cooperative cancellation check fires."""


@dataclass
class GPAOutput:
    """Container for all GPA output fields."""
    # Distortion tensor (asymmetric)
    e_xx: np.ndarray
    e_xy: np.ndarray
    e_yx: np.ndarray
    e_yy: np.ndarray

    # Strain tensor (symmetric part of distortion)
    eps_xx: np.ndarray   # = e_xx
    eps_xy: np.ndarray   # = 0.5*(e_xy + e_yx)
    eps_yy: np.ndarray   # = e_yy

    # Rotation tensor (anti-symmetric part of distortion)
    omega_xy: np.ndarray  # = 0.5*(e_xy - e_yx)

    # Dilatation (trace of distortion)
    dilatation: np.ndarray  # = e_xx + e_yy

    # Displacement field
    u_x: Optional[np.ndarray] = None
    u_y: Optional[np.ndarray] = None

    # Phase fields
    phase1: Optional[np.ndarray] = None
    phase2: Optional[np.ndarray] = None
    quality_mask: Optional[np.ndarray] = None
    g_condition_number: Optional[float] = None


class GPA:
    """
    Geometric Phase Analysis controller.

    Manages the complete GPA workflow:
    1. Load an image and compute its FFT
    2. Auto-detect Bragg peak positions
    3. Set up two non-colinear g-vectors
    4. Compute phase maps for each g-vector
    5. Refine g-vectors using reference regions
    6. Compute the full distortion/strain/rotation tensor fields

    Usage
    -----
    >>> gpa = GPA()
    >>> gpa.load_image(image_array)
    >>> gpa.set_g1(gx1, gy1, sigma=5.0)
    >>> gpa.set_g2(gx2, gy2, sigma=5.0)
    >>> result = gpa.compute()
    >>> print(result.eps_xx)  # strain ε_xx map
    """

    def __init__(self):
        """Initialize an empty GPA object."""
        self._image: Optional[np.ndarray] = None
        self._fft: Optional[np.ndarray] = None
        self._power_spectrum: Optional[np.ndarray] = None
        self._image_shape: Tuple[int, int] = (0, 0)

        self.phase1 = Phase()
        self.phase2 = Phase()

        self._rotation_angle: float = 0.0  # radians
        self._use_hann: bool = False
        self._pixel_size: float = 1.0
        self._pixel_size_x: float = 1.0
        self._pixel_size_y: float = 1.0
        self._last_g_condition_number: Optional[float] = None

    # ==========================================================================
    # Image Loading
    # ==========================================================================

    def load_image(self, image: np.ndarray, pixel_size=1.0,
                   use_hann: bool = False):
        """
        Load an image and compute its FFT and power spectrum.

        Parameters
        ----------
        image : np.ndarray (M, N)
            Input TEM image (real-valued, float).
        pixel_size : float or (y_size, x_size)
            Physical pixel size in nm (or any consistent unit). A two-element
            value supports anisotropic pixels.
        use_hann : bool
            Apply Hann window to reduce FFT edge artifacts.
        """
        from .utils import hann_window_2d, apply_hann_window

        # Input validation
        if image is None:
            raise ValueError("Image cannot be None.")
        if image.ndim != 2:
            raise ValueError(f"Image must be 2D, got {image.ndim}D array.")
        if image.shape[0] < 4 or image.shape[1] < 4:
            raise ValueError(f"Image too small: {image.shape}. Minimum 4x4 required.")
        if not np.all(np.isfinite(image)):
            raise ValueError(
                "Image contains NaN or infinite values. Repair or mask the "
                "source data explicitly before GPA analysis."
            )
        if np.ptp(image) == 0:
            raise ValueError(
                "Image has no contrast (constant values). GPA requires "
                "lattice fringes, i.e. a non-zero intensity range."
            )
        estimated_bytes = estimate_peak_bytes(image.shape)
        if estimated_bytes > _LARGE_IMAGE_WARNING_BYTES:
            warnings.warn(
                f"Estimated peak memory for a {image.shape[1]}x{image.shape[0]} "
                f"GPA analysis is ~{estimated_bytes / (1 << 30):.1f} GiB. "
                "Consider cropping the analysis region or closing other "
                "applications; the computation may fail or swap on "
                "memory-constrained machines.",
                RuntimeWarning,
                stacklevel=2,
            )

        pixel_values = np.asarray(pixel_size, dtype=np.float64)
        if pixel_values.ndim == 0:
            pixel_y = pixel_x = float(pixel_values)
        elif pixel_values.shape == (2,):
            pixel_y, pixel_x = map(float, pixel_values)
        else:
            raise ValueError(
                "pixel_size must be a positive scalar or a (y_size, x_size) pair."
            )
        if not np.isfinite(pixel_x) or not np.isfinite(pixel_y) \
                or pixel_x <= 0 or pixel_y <= 0:
            raise ValueError(f"Pixel sizes must be finite and positive, got {pixel_size}.")

        self._image_shape = image.shape
        self._pixel_size_x = pixel_x
        self._pixel_size_y = pixel_y
        self._pixel_size = (
            pixel_x if np.isclose(pixel_x, pixel_y) else (pixel_y, pixel_x)
        )
        self._use_hann = use_hann

        work_image = image.astype(np.float64)

        if use_hann:
            window = hann_window_2d(image.shape)
            work_image = apply_hann_window(work_image, window)

        self._image = work_image
        self._fft = forward_fft(work_image)

        # Compute power spectrum directly from the already-computed FFT
        # (avoids a redundant second FFT call)
        mag = np.abs(self._fft)
        mag = np.maximum(mag, 1e-30)
        self._power_spectrum = np.log10(mag)

        # Reset phases
        self.phase1 = Phase()
        self.phase2 = Phase()

        # Set FFT on each phase object. phase1 performs the full finiteness
        # scan; phase2 shares the identical buffer so it can skip it.
        self.phase1.set_fft(self._fft)
        self.phase2.set_fft(self._fft, check_finite=False)
        self._last_g_condition_number = None

    def load_image_from_file(self, filepath: str, pixel_size: float = 1.0,
                             use_hann: bool = False):
        """
        Load an image from a file (TIFF, DM3, or DM4).

        Parameters
        ----------
        filepath : str
            Path to image file.
        pixel_size : float
            Physical pixel size.
        use_hann : bool
            Apply Hann window.
        """
        from .dm_reader import read_dm_file, is_dm_file, read_tiff
        import os

        ext = os.path.splitext(filepath)[1].lower()

        if ext in ('.dm3', '.dm4') or is_dm_file(filepath):
            image, metadata = read_dm_file(filepath)
            if 'pixel_size' in metadata:
                pixel_size = metadata['pixel_size']
        elif ext in ('.tif', '.tiff'):
            image = read_tiff(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        self.load_image(image, pixel_size, use_hann)

    # ==========================================================================
    # G-Vector Setup
    # ==========================================================================

    def set_g1(self, gx: float, gy: float, sigma: float = 5.0):
        """
        Set the first g-vector and compute its phase.

        Parameters
        ----------
        gx, gy : float
            G-vector position in the FFT (pixel coordinates, zero-centered).
        sigma : float
            Gaussian mask sigma (must be > 0).
        """
        if self._fft is None:
            raise RuntimeError("No image loaded. Call load_image() first.")
        self._validate_g_vector(gx, gy, sigma)

        self.phase1.set_g_vector(gx, gy, sigma)
        self.phase1.set_fft(self._fft)
        self.phase1.compute()

    def set_g2(self, gx: float, gy: float, sigma: float = 5.0):
        """
        Set the second g-vector and compute its phase.

        The second g-vector must be non-colinear with the first.

        Parameters
        ----------
        gx, gy : float
            G-vector position in the FFT.
        sigma : float
            Gaussian mask sigma (must be > 0).
        """
        if self._fft is None:
            raise RuntimeError("No image loaded. Call load_image() first.")
        self._validate_g_vector(gx, gy, sigma)

        self.phase2.set_g_vector(gx, gy, sigma)
        self.phase2.set_fft(self._fft)
        self.phase2.compute()

    def set_rotation(self, theta_rad: float):
        """
        Set a rotation angle for the coordinate system.

        The distortion/strain tensors will be rotated by this angle.
        This is equivalent to rotating the reference lattice.

        Parameters
        ----------
        theta_rad : float
            Rotation angle in radians.
        """
        if not np.isfinite(theta_rad):
            raise ValueError("Rotation angle must be finite.")
        self._rotation_angle = float(theta_rad)

    def _validate_g_vector(self, gx: float, gy: float, sigma: float):
        """Validate a reciprocal vector against the loaded FFT grid."""
        if self._fft is None:
            raise RuntimeError("No image loaded. Call load_image() first.")
        if not np.isfinite(gx) or not np.isfinite(gy):
            raise ValueError("G-vector components must be finite.")
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"Gaussian sigma must be finite and positive, got {sigma}.")

        rows, cols = self._image_shape
        cy, cx = rows // 2, cols // 2
        if not (-cx <= gx <= cols - 1 - cx) or \
                not (-cy <= gy <= rows - 1 - cy):
            raise ValueError(
                f"G-vector ({gx}, {gy}) lies outside the FFT coordinate range "
                f"x=[{-cx}, {cols - 1 - cx}], y=[{-cy}, {rows - 1 - cy}]."
            )
        radius = float(np.hypot(gx, gy))
        if radius < 2.0:
            raise ValueError("G-vector is too close to the DC component.")
        if sigma > min(rows, cols) / 4:
            raise ValueError("Gaussian sigma is too large for the image dimensions.")
        if 3.0 * sigma >= radius:
            warnings.warn(
                "The Gaussian mask reaches the DC component; reduce sigma to "
                "avoid mixing the central beam with the Bragg peak.",
                RuntimeWarning,
                stacklevel=2,
            )

    # ==========================================================================
    # G-Vector Refinement
    # ==========================================================================

    def refine_g1(self, mask: np.ndarray) -> Tuple[float, float]:
        """
        Refine g1 using a reference region mask.

        Parameters
        ----------
        mask : np.ndarray (M, N), bool
            Binary mask of the homogeneous reference region.

        Returns
        -------
        (delta_gx, delta_gy)
        """
        return self.phase1.refine_g_vector(mask)

    def refine_g2(self, mask: np.ndarray) -> Tuple[float, float]:
        """
        Refine g2 using a reference region mask.

        Parameters
        ----------
        mask : np.ndarray (M, N), bool
            Binary mask of the homogeneous reference region.

        Returns
        -------
        (delta_gx, delta_gy)
        """
        return self.phase2.refine_g_vector(mask)

    # ==========================================================================
    # G-Vector Auto-Detection
    # ==========================================================================

    def estimate_mask_radius(self) -> float:
        """
        Estimate the optimal radius for the Gaussian mask based on the
        radial profile of the power spectrum.

        Returns the estimated distance (in pixels) from the DC component
        to the first Bragg peak, which can be used to set an appropriate
        mask sigma (a conservative start is typically radius / 15).

        Returns
        -------
        float
            Estimated radius in pixels.
        """
        if self._power_spectrum is None:
            raise RuntimeError("No image loaded. Call load_image() first.")
        return estimate_g_vector_radius(self._power_spectrum)

    def get_power_spectrum(self) -> np.ndarray:
        """
        Get the log power spectrum of the loaded image.

        Returns
        -------
        np.ndarray (M, N)
        """
        if self._power_spectrum is None:
            raise RuntimeError("No image loaded.")
        return self._power_spectrum

    # ==========================================================================
    # Core Computation
    # ==========================================================================

    def _get_a_matrix(self) -> np.ndarray:
        """
        Compute the A matrix for mapping phase gradients to strain.

        The g-vectors are in cycles/image (FFT pixel index) units. To correctly
        relate phase gradients (radians/pixel) to dimensionless strain, we
        normalize g by image dimensions to get cycles/pixel:
            g_x_normalized = g_x / N
            g_y_normalized = g_y / M

        From P_g = -2π * (g_x/N * u_x + g_y/M * u_y), taking gradients:
            [dP1/dx; dP2/dx] = -2π * G_norm * [e_xx; e_yx]

        Solving: [e_xx; e_yx] = -(1/(2π)) * G_norm^(-1) * [dP1/dx; dP2/dx]

        Returns
        -------
        np.ndarray (2, 2)
        """
        M, N = self._image_shape

        # Normalize g-vectors to cycles/pixel
        G = np.array([[self.phase1.gx / N, self.phase1.gy / M],
                       [self.phase2.gx / N, self.phase2.gy / M]])

        if not np.all(np.isfinite(G)):
            raise RuntimeError("G-vector matrix contains non-finite values.")
        norms = np.linalg.norm(G, axis=1)
        if np.any(norms <= 0):
            raise RuntimeError("Both G-vectors must be non-zero.")
        sin_angle = abs(np.linalg.det(G)) / float(norms[0] * norms[1])
        min_sin_angle = np.sin(np.radians(5.0))
        if sin_angle < min_sin_angle:
            raise RuntimeError(
                f"G-vectors are too close to colinear "
                f"(included angle ≈ {np.degrees(np.arcsin(np.clip(sin_angle, 0, 1))):.2f}°). "
                "Choose Bragg spots separated by at least 5°; 60°–120° is preferred."
            )

        condition_number = float(np.linalg.cond(G))
        if not np.isfinite(condition_number) or condition_number > 1e4:
            raise RuntimeError(
                f"G-vector matrix is ill-conditioned (condition number "
                f"{condition_number:.2e}). Choose peaks with comparable radii "
                "and a wider included angle."
            )
        self._last_g_condition_number = condition_number

        return np.linalg.inv(G)

    def _rotate_displacement(self, u_x: np.ndarray, u_y: np.ndarray):
        """Rotate vector components into the requested output coordinate system."""
        if self._rotation_angle == 0.0:
            return u_x, u_y
        R = rotation_matrix_2d(self._rotation_angle)
        return (
            R[0, 0] * u_x + R[0, 1] * u_y,
            R[1, 0] * u_x + R[1, 1] * u_y,
        )

    def _rotate_distortion(self, e_xx, e_xy, e_yx, e_yy):
        """Apply E' = R E R^T, preserving trace and tensor invariants."""
        if self._rotation_angle == 0.0:
            return e_xx, e_xy, e_yx, e_yy
        c = float(np.cos(self._rotation_angle))
        s = float(np.sin(self._rotation_angle))
        a00 = c * e_xx - s * e_yx
        a01 = c * e_xy - s * e_yy
        a10 = s * e_xx + c * e_yx
        a11 = s * e_xy + c * e_yy
        return (
            c * a00 - s * a01,
            s * a00 + c * a01,
            c * a10 - s * a11,
            s * a10 + c * a11,
        )

    def _quality_mask(self) -> np.ndarray:
        return self.phase1.quality_mask() & self.phase2.quality_mask()

    def _compute_tensor_fields(self, p1: np.ndarray, p2: np.ndarray,
                               A: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute the distortion/strain/rotation/dilatation fields from two
        wrapped phase maps. Shared by :meth:`compute` and
        :meth:`compute_strain_only` so the tensor algebra stays in one place.

        Returns unmasked arrays; callers apply the quality mask.
        """
        a1x, a1y = A[0, 0], A[0, 1]
        a2x, a2y = A[1, 0], A[1, 1]
        factor = -1.0 / (2.0 * np.pi)

        # Phase gradients in radians/pixel (pixel_size=1.0 for dimensionless
        # strain); strain = du/dx where both u and x are in pixels.
        dP1_dx, dP1_dy = phase_gradient(p1, pixel_size=1.0)
        dP2_dx, dP2_dy = phase_gradient(p2, pixel_size=1.0)

        # Distortion tensor components
        # e_xx = ∂u_x/∂x, e_xy = ∂u_x/∂y
        e_xx = factor * (a1x * dP1_dx + a1y * dP2_dx)
        e_xy = factor * (a1x * dP1_dy + a1y * dP2_dy)
        e_yx = factor * (a2x * dP1_dx + a2y * dP2_dx)
        e_yy = factor * (a2x * dP1_dy + a2y * dP2_dy)
        # Convert cross derivatives for anisotropic physical pixels.
        e_xy *= self._pixel_size_x / self._pixel_size_y
        e_yx *= self._pixel_size_y / self._pixel_size_x
        e_xx, e_xy, e_yx, e_yy = self._rotate_distortion(
            e_xx, e_xy, e_yx, e_yy
        )

        return {
            'e_xx': e_xx, 'e_xy': e_xy, 'e_yx': e_yx, 'e_yy': e_yy,
            'eps_xx': e_xx,
            'eps_xy': 0.5 * (e_xy + e_yx),
            'eps_yy': e_yy,
            'omega_xy': 0.5 * (e_xy - e_yx),
            'dilatation': e_xx + e_yy,
        }

    @staticmethod
    def _mask_invalid(data: np.ndarray, valid: np.ndarray) -> np.ndarray:
        return np.where(valid, data, np.nan)

    def compute(self, include_displacement: bool = True,
                cancel_check=None, mask_results: bool = True) -> GPAOutput:
        """
        Compute the full GPA output: displacement field, distortion,
        strain, rotation, and dilatation tensors.

        The computation follows the standard GPA derivation:

        1. From the two phase fields and the g-matrix:
           [u_x, u_y]^T = -(1/(2π)) · A · [P_1, P_2]^T

        2. Distortion tensor e_ij = ∂u_i/∂x_j:
           [[∂u_x/∂x, ∂u_x/∂y],
            [∂u_y/∂x, ∂u_y/∂y]]

        3. Strain = symmetric part: ε = 0.5 · (e + e^T)
        4. Rotation = anti-symmetric part: ω = 0.5 · (e - e^T)
        5. Dilatation = trace: Δ = e_xx + e_yy

        Parameters
        ----------
        include_displacement : bool
            If False, skip unwrapped displacement reconstruction while still
            returning all distortion, strain, rotation, and dilatation fields.
        cancel_check : callable, optional
            A zero-argument callable returning True when the computation should
            stop early.  It is polled between major FFT/phase stages.
        mask_results : bool
            If True (default), pixels flagged unreliable by the quality mask
            are set to NaN in every returned tensor/displacement field, so the
            arrays are directly safe for quantitative use. If False, raw
            values are returned for every pixel (including edges and
            low-amplitude regions, matching the original Strain++ display);
            combine the output with ``quality_mask`` at the display layer.

        Returns
        -------
        GPAOutput
            All computed fields.

        Raises
        ------
        GPAComputationCancelled
            If ``cancel_check`` returns True.
        """
        if cancel_check is not None and cancel_check():
            raise GPAComputationCancelled("GPA computation cancelled before start.")

        p1 = self.phase1.wrapped_phase
        p2 = self.phase2.wrapped_phase

        if p1 is None or p2 is None:
            raise RuntimeError("Both g-vectors must be set and phases computed.")

        if cancel_check is not None and cancel_check():
            raise GPAComputationCancelled("GPA computation cancelled before g-matrix evaluation.")

        A = self._get_a_matrix()
        a1x, a1y = A[0, 0], A[0, 1]
        a2x, a2y = A[1, 0], A[1, 1]
        factor = -1.0 / (2.0 * np.pi)

        u_x = u_y = None
        if include_displacement:
            if cancel_check is not None and cancel_check():
                raise GPAComputationCancelled("GPA computation cancelled before phase unwrapping.")
            # Continuous displacement field from unwrapped phases.
            up1 = self.phase1.ensure_unwrapped_phase()
            if cancel_check is not None and cancel_check():
                raise GPAComputationCancelled("GPA computation cancelled after g1 phase unwrapping.")
            up2 = self.phase2.ensure_unwrapped_phase()
            u_x = factor * (a1x * up1 + a1y * up2) * self._pixel_size_x
            u_y = factor * (a2x * up1 + a2y * up2) * self._pixel_size_y
            u_x, u_y = self._rotate_displacement(u_x, u_y)

        if cancel_check is not None and cancel_check():
            raise GPAComputationCancelled("GPA computation cancelled before tensor-field evaluation.")

        fields = self._compute_tensor_fields(p1, p2, A)
        quality_mask = self._quality_mask()
        if mask_results:
            if u_x is not None:
                u_x = self._mask_invalid(u_x, quality_mask)
                u_y = self._mask_invalid(u_y, quality_mask)
            for key in (
                'e_xx', 'e_xy', 'e_yx', 'e_yy',
                'eps_xx', 'eps_xy', 'eps_yy',
                'omega_xy', 'dilatation',
            ):
                fields[key] = self._mask_invalid(fields[key], quality_mask)

        return GPAOutput(
            u_x=u_x, u_y=u_y,
            e_xx=fields['e_xx'], e_xy=fields['e_xy'],
            e_yx=fields['e_yx'], e_yy=fields['e_yy'],
            eps_xx=fields['eps_xx'], eps_xy=fields['eps_xy'],
            eps_yy=fields['eps_yy'],
            omega_xy=fields['omega_xy'],
            dilatation=fields['dilatation'],
            phase1=p1, phase2=p2,
            quality_mask=quality_mask,
            g_condition_number=self._last_g_condition_number,
        )

    def compute_strain_only(self, cancel_check=None) -> Dict[str, np.ndarray]:
        """
        Compute only the strain/rotation/dilatation fields, skipping the
        displacement field computation (saves memory and a small amount of time).

        Returns
        -------
        dict with keys: eps_xx, eps_xy, eps_yy, dilatation, omega_xy,
        quality_mask
        """
        if cancel_check is not None and cancel_check():
            raise GPAComputationCancelled("GPA strain-only computation cancelled.")

        p1 = self.phase1.wrapped_phase
        p2 = self.phase2.wrapped_phase

        if p1 is None or p2 is None:
            raise RuntimeError("Both g-vectors must be set and phases computed.")

        if cancel_check is not None and cancel_check():
            raise GPAComputationCancelled("GPA strain-only computation cancelled before g-matrix evaluation.")

        fields = self._compute_tensor_fields(p1, p2, self._get_a_matrix())
        quality_mask = self._quality_mask()

        return {
            'eps_xx': self._mask_invalid(fields['eps_xx'], quality_mask),
            'eps_xy': self._mask_invalid(fields['eps_xy'], quality_mask),
            'eps_yy': self._mask_invalid(fields['eps_yy'], quality_mask),
            'dilatation': self._mask_invalid(fields['dilatation'], quality_mask),
            'omega_xy': self._mask_invalid(fields['omega_xy'], quality_mask),
            'quality_mask': quality_mask,
        }

    # ==========================================================================
    # Properties
    # ==========================================================================

    @property
    def shape(self) -> Tuple[int, int]:
        """Image shape (M, N)."""
        return self._image_shape

    @property
    def g1(self) -> Tuple[float, float]:
        """First g-vector components."""
        return (self.phase1.gx, self.phase1.gy)

    @property
    def g2(self) -> Tuple[float, float]:
        """Second g-vector components."""
        return (self.phase2.gx, self.phase2.gy)

    @property
    def power_spectrum(self) -> Optional[np.ndarray]:
        """Log power spectrum of the loaded image."""
        return self._power_spectrum

    @property
    def pixel_size(self):
        """Physical pixel size as a scalar or ``(y_size, x_size)`` pair."""
        return self._pixel_size


# Backward compatibility: read_tiff is now in dm_reader.py
from .dm_reader import read_tiff  # noqa: F401
