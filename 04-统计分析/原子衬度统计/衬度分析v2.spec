# -*- mode: python ; coding: utf-8 -*-
"""TIF图像衬度分析工具 PyInstaller 打包配置（版本号单源于 contrast_core.py）"""
import re
from PyInstaller.utils.hooks import collect_all, collect_submodules

# 从 contrast_core.__version__ 读取版本（用正则避免在打包期 import numpy）。
# 此前 exe 名里的版本号是手写死的，与 __version__ 双源易漏改。
with open('contrast_core.py', encoding='utf-8') as _f:
    _m = re.search(r'__version__\s*=\s*"([^"]+)"', _f.read())
assert _m, "contrast_core.py 中未找到 __version__"
APP_VERSION = _m.group(1)

datas = []
binaries = []
hiddenimports = []

# 收集完整依赖包（含资源文件）。
# 说明：CSV 导出已改用标准库 csv，运行时不依赖 pandas，因此不再收集 pandas。
for pkg in ['ttkbootstrap', 'matplotlib', 'tifffile', 'numpy', 'PIL']:
    tmp_ret = collect_all(pkg)
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

hiddenimports += ['matplotlib.backends.backend_tkagg']

# setuptools>=81 的 pkg_resources 不再自带 vendored 依赖
# （jaraco.text / appdirs / ...），缺任一则冻结程序启动即崩
# （pyi_rth_pkgres 钩子）。pkg_resources 和 setuptools 各有独立的
# _vendor 目录，两棵都要整体收集。
hiddenimports += (collect_submodules('setuptools._vendor')
                  + collect_submodules('pkg_resources._vendor'))

# 只使用 TkAgg 后端；排除其余 GUI 后端、示例数据与开发期依赖，控制体积。
excludes = [
    'pythoncom', 'pywin32', 'win32com', 'win32api', 'pywintypes', '_win32sysloader',
    # 其他 matplotlib 后端（本工具固定 TkAgg）
    'matplotlib.backends.backend_qt', 'matplotlib.backends.backend_qt5',
    'matplotlib.backends.backend_qtagg', 'matplotlib.backends.backend_qt5agg',
    'matplotlib.backends.backend_gtk', 'matplotlib.backends.backend_gtk3',
    'matplotlib.backends.backend_gtk3agg', 'matplotlib.backends.backend_gtk4',
    'matplotlib.backends.backend_gtk4agg', 'matplotlib.backends.backend_wx',
    'matplotlib.backends.backend_wxagg', 'matplotlib.backends.backend_webagg',
    'matplotlib.backends.backend_webagg_core', 'matplotlib.backends.backend_nbagg',
    'matplotlib.backends.backend_tkcairo', 'matplotlib.backends.backend_template',
    'matplotlib.backends.backend_pdf', 'matplotlib.backends.backend_svg',
    'matplotlib.backends.backend_ps',
    'matplotlib.testing', 'matplotlib.tests',
    'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'gtk', 'wx',
    'IPython', 'jupyter', 'notebook', 'nbformat', 'ipykernel',
    # 运行时不再依赖（导出用标准库 csv；pandas 仅测试脚本使用）
    'pandas',
    'pytest', '_pytest', 'pytest_cov',
    # scipy 由 PyInstaller 的 hook-numpy 防御性引入，本工具与
    # matplotlib/tifffile/PIL/ttkbootstrap 均无 scipy 运行时导入
    # （已逐包核对），排除可显著减小体积（scipy 安装体积约 118MB）。
    'scipy',
]

a = Analysis(
    ['tif图像衬度分析工具.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    # 必须为 0：optimize=2 会剥离 docstring，numpy 的 array_function_dispatch
    # 依赖 dispatcher.__doc__ 注入文档，收到 None 会抛
    # TypeError: argument docstring of add_docstring should be a str，
    # 导致冻结程序启动即崩。
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name=f'TIF图像衬度分析工具v{APP_VERSION}',
    debug=False, bootloader_ignore_signals=False, strip=False,
    upx=True, upx_exclude=[], runtime_tmpdir=None,
    console=False, disable_windowed_traceback=False,
    argv_emulation=False, target_arch=None,
    codesign_identity=None, entitlements_file=None,
)
