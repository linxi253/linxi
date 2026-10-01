# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import runpy
from PyInstaller.utils.win32.versioninfo import (
    VSVersionInfo, FixedFileInfo, StringFileInfo, StringTable,
    StringStruct, VarFileInfo, VarStruct,
)

app_version = runpy.run_path(str(Path(SPECPATH) / 'version.py'))
version = app_version['__version__']
version_tuple = tuple(int(part) for part in version.split('.')) + (0,)
version_resource = VSVersionInfo(
    ffi=FixedFileInfo(filevers=version_tuple, prodvers=version_tuple,
                      mask=0x3f, flags=0, OS=0x40004, fileType=1,
                      subtype=0, date=(0, 0)),
    kids=[StringFileInfo([StringTable('080404b0', [
        StringStruct('FileDescription', app_version['APP_TITLE']),
        StringStruct('FileVersion', version),
        StringStruct('ProductName', app_version['APP_NAME']),
        StringStruct('ProductVersion', version),
        StringStruct('InternalName', 'DelocCleaner'),
        StringStruct('OriginalFilename', 'DelocCleaner.exe'),
    ])]), VarFileInfo([VarStruct('Translation', [2052, 1200])])],
)

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=['.'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 本工具只用到 matplotlib 的 TkAgg 后端，其余后端/示例数据全部排除
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.backends.backend_qt5',
        'matplotlib.backends.backend_qtagg',
        'matplotlib.backends.backend_webagg',
        'matplotlib.backends.backend_wx',
        'matplotlib.backends.backend_gtk3',
        'matplotlib.backends._backend_gtk3agg',
        'matplotlib.backends.backend_pdf',
        'matplotlib.backends.backend_ps',
        'matplotlib.backends.backend_svg',
        'matplotlib.tests',
        'matplotlib.sphinxext',
        'mpl_toolkits.basemap',
        'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'wx',
        'win32com', 'pythoncom', 'pywintypes', 'win32api',
        'win32con', 'win32comext', 'IPython', 'jupyter', 'notebook',
        'pytest', 'sphinx', 'pandas', 'skimage', 'PIL.SpiderImagePlugin',
        # 运行时无任何依赖加载 pkg_resources（numpy/scipy 仅为注释/测试引用）。
        # 新版 setuptools 的 pkg_resources 需要外部 jaraco.text，静态分析
        # 收集不到，会导致 exe 启动即报 "Failed to execute script"。
        'pkg_resources',
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
    name='DelocCleaner',
    version=version_resource,
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
)
