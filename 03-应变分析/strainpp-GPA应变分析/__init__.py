"""
Strain++ - Geometric Phase Analysis (GPA) for strain measurement from HRTEM images.

This package implements the Geometric Phase Analysis algorithm based on:
    Hytch, M. J., Snoeck, E. & Kilaas, R. "Quantitative measurement of displacement
    and strain fields from HREM micrographs." Ultramicroscopy 74, 131-146 (1998).

Core modules:
    utils       - FFT wrappers, Hann window, phase wrapping, math utilities
    phase       - Single g-vector phase analysis (masking, unwrapping, differentiation)
    gpa         - GPA controller (two g-vectors, tensor calculation, g-vector detection)
    dm_reader   - DM3/DM4 (Gatan DigitalMicrograph) file reader
    cli         - Command-line entry point, TIFF I/O, full pipeline
"""

try:
    from _version import __version__
except ImportError:  # pragma: no cover - only reached in broken installs
    __version__ = "0.0.0"
__author__ = "Based on Strain++ by J.J.P. Peters"

try:
    from .utils import *
    from .phase import Phase
    from .gpa import GPA, GPAOutput
except ImportError:
    # Support direct source-tree loading where this file is imported as a
    # top-level module rather than as a package.
    from utils import *
    from phase import Phase
    from gpa import GPA, GPAOutput
