"""Domain-specific exceptions shared across the processing pipeline."""


class StemEnhancerError(Exception):
    """Base exception for expected application errors."""


class OperationCancelled(StemEnhancerError):
    """Raised when a user-requested cancellation is observed."""


class InputValidationError(StemEnhancerError, ValueError):
    """Raised when an input file or frame is unsupported or inconsistent."""


class OutputValidationError(StemEnhancerError):
    """Raised when a completed output fails structural verification."""
