# -*- coding: utf-8 -*-
"""Backward-compatibility shim.

The GUI implementation lives in :mod:`strainpp_gpa.gui`.  This file only
exists so that legacy source-tree workflows (``python run.py``, ``from
strain_gui import StrainGUI``) keep working.  New code should use ``from
strainpp_gpa.gui import StrainGUI, main`` or the ``strainpp-gpa-gui``
console script.
"""
from strainpp_gpa.gui import *  # noqa: F401,F403
from strainpp_gpa.gui import main, StrainGUI  # noqa: F401

if __name__ == '__main__':
    main()
