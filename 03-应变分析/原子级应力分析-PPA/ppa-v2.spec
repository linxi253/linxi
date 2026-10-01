# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules


# pkg_resources loads its vendored packages dynamically via ``pkg_resources.extern``.
# Explicit collection keeps the frozen app robust when the build machine has a
# newer user-site setuptools alongside the Conda copy of pkg_resources.
pkg_resources_vendor = collect_submodules(
    'pkg_resources._vendor',
    filter=lambda name: name != 'pkg_resources._vendor.pyparsing.diagram',
)

a = Analysis(
    ['ppa.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'numpy',
        'scipy',
        'scipy.spatial',
        'scipy.ndimage',
        'scipy.interpolate',
        'matplotlib',
        'matplotlib.backends.backend_tkagg',
        'tifffile',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'ppa_core',
        'ppa_core.strain',
        'ppa_core.image_io',
        'ppa_core.project_store',
    ] + pkg_resources_vendor,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['hook_numpy_fix.py'],
    excludes=[
        # Optional scientific/notebook/cloud stacks discovered from the broad
        # Conda build environment.  PPA does not import them; excluding them
        # keeps the one-file release small and avoids unrelated DLL warnings.
        'pytest',
        '_pytest',
        'py',
        'IPython',
        'jedi',
        'parso',
        'pandas',
        'openpyxl',
        'dask',
        'distributed',
        'cloudpickle',
        'fsspec',
        'partd',
        'locket',
        'numba',
        'llvmlite',
        'boto3',
        'botocore',
        's3transfer',
        'h5py',
        'zarr',
        'numcodecs',
        'keyring',
        'cryptography',
        'lxml',
        'conda',
        'pythoncom',
        'pywin32',
        'win32com',
        'win32api',
        'win32evtlogutil',
        'win32evtlog',
        'win32pdh',
        'win32con',
        'winerror',
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
    name='PPA原子级位移应变分析工具v3.4',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='ppa_version_info.txt',
)
