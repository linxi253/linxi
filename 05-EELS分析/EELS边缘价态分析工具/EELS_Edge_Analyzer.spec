# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller configuration for the standalone EELS Edge Analyzer GUI."""

from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("ncempy")
# 内置参数预设必须随包分发，GUI/CLI 的 --preset 默认项才能找到它。
datas += [("src/eels_edge_analyzer/presets/cu_l23.json", "eels_edge_analyzer/presets")]

a = Analysis(
    ["run.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "ncempy",
        "ncempy.io",
        "ncempy.io.dm",
        "scipy",
        "scipy.ndimage",
        "scipy.optimize",
        "scipy.signal",
        "matplotlib.backends.backend_agg",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Development/test and interactive-shell packages pulled in by optional
        # dependency discovery. They are not used by the analysis application.
        "pytest", "py", "IPython", "dask", "numba", "llvmlite",
        "setuptools", "pkg_resources", "jaraco", "more_itertools",
        # Avoid a broken optional pywin32/COM installation in the build host.
        "win32com", "pythoncom", "pywintypes", "win32api", "win32con",
        "win32gui", "win32process", "win32security",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="EELS边缘价态分析工具",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
