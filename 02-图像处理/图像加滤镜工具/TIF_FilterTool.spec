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
        StringStruct('InternalName', 'TIF_FilterTool'),
        StringStruct('OriginalFilename', 'TIF_FilterTool.exe'),
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
        'matplotlib', 'win32com', 'pythoncom', 'pywintypes', 'win32api',
        'win32con', 'win32comext', 'IPython', 'jupyter', 'notebook',
        'pytest', 'sphinx', 'pandas', 'PIL.SpiderImagePlugin',
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
    name='TIF_FilterTool',
    version=version_resource,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX 压缩 numpy/scipy/tifffile 的原生 DLL 有杀软误报和个别 DLL
    # 损坏前科，PyInstaller 新版已默认对部分二进制禁用；这里整体关闭。
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
