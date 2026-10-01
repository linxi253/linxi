"""Shared atom-center model development foundation."""

from .interfaces import (
    AtomDetectorProvider,
    CandidateBackend,
    CandidateSet,
    DetectionResult,
)

__all__ = [
    "AtomDetectorProvider",
    "CandidateBackend",
    "CandidateSet",
    "DetectionResult",
]

__version__ = "0.3.0"
