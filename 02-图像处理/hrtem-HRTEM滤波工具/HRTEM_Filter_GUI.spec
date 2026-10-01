# -*- mode: python ; coding: utf-8 -*-
"""Reproducible PyInstaller target for the v5 graphical application."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH)

imagecodecs_hidden = collect_submodules("imagecodecs")
imagecodecs_binaries = collect_dynamic_libs("imagecodecs")

a = Analysis(
    [str(ROOT / "butter.py")],
    pathex=[str(ROOT / "src")],
    binaries=imagecodecs_binaries,
    datas=[],
    hiddenimports=["matplotlib.backends.backend_tkagg", *imagecodecs_hidden],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython", "pytest", "psutil",
        # Tk/Matplotlib do not require pywin32; keep this build independent of
        # an optional, potentially broken pythoncom installation.
        "pythoncom", "pywintypes", "win32api", "win32com", "win32gui",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HRTEM_Filter_GUI_v5",
    console=False,
    debug=False,
    upx=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="HRTEM_Filter_GUI_v5",
)
