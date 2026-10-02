#!/usr/bin/env python3
"""Backward-compatibility shim.

The batch CLI implementation lives in :mod:`strainpp_gpa.batch_cli`.  This
file only exists so that legacy source-tree workflows (``python
strain_batch.py``, ``from strain_batch import main``) keep working.  New code
should use ``from strainpp_gpa.batch_cli import main`` or the
``strainpp-gpa-batch`` console script.
"""
from strainpp_gpa.batch_cli import *  # noqa: F401,F403
from strainpp_gpa.batch_cli import main  # noqa: F401

if __name__ == '__main__':
    main()
