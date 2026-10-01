# 历史版本打包配置，仅供对照。请勿用本 spec 打包当前代码（版本资源与产物名均不匹配）；当前构建请使用 TIFF漂移矫正工具v7.3.spec。
# -*- mode: python ; coding: utf-8 -*-
import os

# tkinterdnd2 需要携带 tkdnd 原生库（DLL + tcl 脚本），PyInstaller 不会自动收集
_datas = []
_hiddenimports = [
    'drift_core', 'cv2', 'numpy', 'matplotlib',
    'matplotlib.backends.backend_tkagg', 'tifffile',
]
try:
    import tkinterdnd2
    _tkdnd_dir = os.path.dirname(tkinterdnd2.__file__)
    # Python 模块本身由 hidden import 收集；仅携带 Windows x64 所需原生 Tcl/Tk 资源，
    # 避免把 Linux/macOS 资源一起塞入 Windows 单文件包。
    for _relative in ('tkdnd/win-x64', 'tkdnd/win-x64-tcl9'):
        _source = os.path.join(_tkdnd_dir, _relative)
        if os.path.isdir(_source):
            _datas.append((_source, os.path.join('tkinterdnd2', _relative)))
    _hiddenimports.append('tkinterdnd2')
except ImportError:
    pass

# 缺少版本资源文件时不应让整次打包失败
_version_file = 'version_info.txt' if os.path.isfile('version_info.txt') else None

a = Analysis(
    ['drift_correction.py'],
    pathex=[],
    binaries=[],
    datas=_datas,
    hiddenimports=_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # 构建工具链不是运行时依赖。显式排除可避免受全局 Python 环境中损坏或
    # 混装的 setuptools/pkg_resources（例如缺失 jaraco.text）影响。
    excludes=['pythoncom', 'pywintypes', 'win32com', 'IPython', 'notebook', 'pytest',
              'pandas', 'PyQt5', 'PySide2', 'zmq', 'pip', 'setuptools',
              'pkg_resources', 'jaraco'],
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
    name='TIFF漂移矫正工具v7.2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX 对 numpy/scipy/cv2 的 pyd 压缩收益小、启动解压慢且杀软误报率高，关闭
    upx=False,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version=_version_file,
)
