"""Public package API for Strain++ GPA."""

from .utils import (
    apply_hann_window,
    backward_fft,
    detect_bragg_peaks,
    estimate_g_vector_radius,
    extract_phase,
    forward_fft,
    gaussian_mask,
    hann_window_2d,
    phase_derivative_fourier,
    phase_gradient,
    reference_plane,
    rotation_matrix_2d,
    unwrap_phase_2d,
    wrap_to_pi,
)
from .phase import ComputationCancelled, Phase
from .gpa import GPA, GPAComputationCancelled, GPAOutput
from .dm_reader import (
    DM3Error,
    is_dm_file,
    read_dm_file,
    read_dm_file_simple,
    read_tiff,
)

__all__ = [
    'ComputationCancelled',
    'DM3Error',
    'GPA',
    'GPAComputationCancelled',
    'GPAOutput',
    'Phase',
    'apply_hann_window',
    'backward_fft',
    'detect_bragg_peaks',
    'estimate_g_vector_radius',
    'extract_phase',
    'forward_fft',
    'gaussian_mask',
    'hann_window_2d',
    'is_dm_file',
    'phase_derivative_fourier',
    'phase_gradient',
    'read_dm_file',
    'read_dm_file_simple',
    'read_tiff',
    'reference_plane',
    'rotation_matrix_2d',
    'unwrap_phase_2d',
    'wrap_to_pi',
]

try:
    from _version import __version__
except ImportError:  # pragma: no cover - only reached in broken installs
    __version__ = '0.0.0'
