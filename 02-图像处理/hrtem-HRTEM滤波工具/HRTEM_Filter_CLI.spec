# -*- mode: python ; coding: utf-8 -*-
"""Reproducible PyInstaller target for the v5 command-line application."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH)

imagecodecs_hidden = collect_submodules("imagecodecs")
imagecodecs_binaries = collect_dynamic_libs("imagecodecs")

a = Analysis(
    [str(ROOT / "cli_launcher.py")],
    pathex=[str(ROOT / "src")],
    binaries=imagecodecs_binaries,
    datas=[],
    hiddenimports=imagecodecs_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib", "IPython", "pytest", "psutil",
        # The application does not use pywin32.  Excluding it also avoids a
        # broken pythoncom installation in otherwise valid Conda environments.
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
    name="HRTEM_Filter_CLI_v5",
    console=True,
    debug=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name="HRTEM_Filter_CLI_v5")
