"""Backward-compatibility shim.

The real implementation lives in :mod:`strainpp_gpa.gpa`.  This file only
exists so that legacy source-tree workflows (``python gpa.py``, direct
``from gpa import GPA``) keep working without injecting generic module names
into site-packages.  New code should use ``from strainpp_gpa import GPA``.
"""
from strainpp_gpa.gpa import *  # noqa: F401,F403

