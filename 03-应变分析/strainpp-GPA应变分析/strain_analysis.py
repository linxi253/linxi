#!/usr/bin/env python3
"""Backward-compatibility shim.

The CLI implementation lives in :mod:`strainpp_gpa.cli`.  This file only
exists so that legacy source-tree workflows (``python strain_analysis.py``,
``from strain_analysis import main``) keep working.  New code should use
``from strainpp_gpa.cli import main`` or the ``strainpp-gpa`` console script.
"""
from strainpp_gpa.cli import *  # noqa: F401,F403
from strainpp_gpa.cli import main  # noqa: F401

if __name__ == '__main__':
    main()
