# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

# imagecodecs 通过 __getattr__ 惰性加载各编解码扩展模块（如 _lzw/_packbits），
# 静态分析会漏掉。用 collect_submodules 全量收集 imagecodecs 子模块，
# 并显式收集其动态库（.pyd/.dll），避免打包后 LZW/PackBits 等压缩读不出来。
imagecodecs_submodules = collect_submodules("imagecodecs")
imagecodecs_binaries = collect_dynamic_libs("imagecodecs")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=imagecodecs_binaries,
    datas=[],
    hiddenimports=[
        "main_window",
        "contrast",
        "errors",
        "filters",
        "param_panel",
        "pipeline",
        "preview_canvas",
        "self_test",
        "tiff_handler",
        "version",
        "worker",
        "cv2",
        "defusedxml",
        "imagecodecs",
        "imagecodecs._deflate",
        "imagecodecs._imcd",
        "imagecodecs._jpeg2k",
        "imagecodecs._jpeg8",
        "imagecodecs._shared",
        "imagecodecs._zstd",
        *imagecodecs_submodules,
        "numpy",
        "scipy.fft",
        "scipy.ndimage",
        "tifffile",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PIL",
        "matplotlib",
        "pandas",
        "pytest",
        "tkinter.test",
        "win32com",
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="STEM图像优化工具v2.1",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version="version_info.txt",
    uac_admin=False,
    uac_uiaccess=False,
)
