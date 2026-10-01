# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules

# setuptools>=81 的 pkg_resources 不再自带 vendored 依赖
# （jaraco.text / appdirs / ...），缺任一则冻结程序启动即崩
# （pyi_rth_pkgres 钩子）。pkg_resources 和 setuptools 各有独立的
# _vendor 目录，两棵都要整体收集。
vendor_hidden = (collect_submodules('setuptools._vendor')
                 + collect_submodules('pkg_resources._vendor'))

a = Analysis(
    ['应力面积统计.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'seaborn',
        'matplotlib',
        'matplotlib.backends.backend_agg',
        'tifffile',
        'skimage',
        'skimage.filters',
        'skimage.morphology',
        'scipy',
        'scipy.ndimage',
        'scipy.signal',
        'scipy.stats',
        'numpy',
    ] + vendor_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'pythoncom',
        'pywin32',
        'win32com',
        'win32api',
        'win32con',
        'pywintypes',
        '_win32sysloader',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='HAADF-STEM特征区域演化分析工具',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
