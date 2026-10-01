"""Backward-compatibility shim for :mod:`strainpp_gpa.dm_reader`."""
from strainpp_gpa.dm_reader import *  # noqa: F401,F403
from strainpp_gpa.dm_reader import (  # noqa: F401  (private helpers used by tests)
    _DMReader,
    _convert_ncempy_result,
    _extract_pixel_size_heuristic,
)
