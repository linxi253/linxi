# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

project_dir = Path(SPECPATH)
datas = [
    (str(project_dir / 'strainpp.ico'), '.'),
    (str(project_dir / 'LICENSE'), '.'),
]
datas += collect_data_files('ttkbootstrap')
# ncempy 在打开 DM3/DM4 时才会被导入。这里收集其数据文件并显式列出
# h5py/hdf5plugin hidden imports，避免 PyInstaller 漏掉 DM4 读取链路。
datas += collect_data_files('ncempy')
datas += collect_data_files('hdf5plugin')

a = Analysis(
    [str(project_dir / 'run.py')],
    pathex=[str(project_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'ttkbootstrap',
        'tifffile',
        'scipy.ndimage',
        # strainpp_gpa.utils 顶层导入 scipy.fft（多核 FFT 后端）
        'scipy.fft',
        'ncempy.io.dm',
        'h5py',
        'hdf5plugin',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # setuptools/pkg_resources 被排除：本项目及全部运行时依赖（numpy/scipy/
    # matplotlib/tifffile/ttkbootstrap/ncempy/h5py）均不使用 pkg_resources。
    # setuptools >= 70 的 pkg_resources 需要 jaraco.text 等外部包，PyInstaller
    # 的 pyi_rth_pkgres 运行时钩子收集不全会导致 exe 启动即 ImportError。
    excludes=[
        'win32com', 'pythoncom', 'pywintypes', 'win32api',
        'setuptools', 'pkg_resources',
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
    name='Strain++GPA',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(project_dir / 'strainpp.ico')],
    version=str(project_dir / 'version_info.txt'),
)
