# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置 - 电镜晶体/非晶区域统计分析工具 v5.3

打包命令:
    pyinstaller 非晶面积统计工具.spec
"""

import os
import ttkbootstrap
from PyInstaller.utils.hooks import collect_submodules

# 动态获取 ttkbootstrap 资源路径（修复打包后找不到 bootstrap.ttf 的问题）
ttkb_root = os.path.dirname(ttkbootstrap.__file__)
ttkb_assets = os.path.join(ttkb_root, 'assets')

# setuptools>=81 的 pkg_resources 不再自带 vendored 依赖
# （jaraco.text / appdirs / ...），缺任一则冻结程序启动即崩
# （pyi_rth_pkgres 钩子）。pkg_resources 和 setuptools 各有独立的
# _vendor 目录，两棵都要整体收集。
vendor_hidden = (collect_submodules('setuptools._vendor')
                 + collect_submodules('pkg_resources._vendor'))

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        # ttkbootstrap 字体和图标资源（必须包含）
        (ttkb_assets, 'ttkbootstrap/assets'),
    ],
    hiddenimports=[
        # GUI
        'ttkbootstrap',
        'ttkbootstrap.themes',
        'ttkbootstrap.themes.standard',
        'ttkbootstrap.style',
        'ttkbootstrap.window',
        'ttkbootstrap.dialogs',
        'ttkbootstrap.widgets',
        'ttkbootstrap.localization',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'PIL.ImageDraw',
        'PIL.ImageFont',
        # 图像处理（v5.2 移除 scikit-image：全项目无引用的死依赖）
        'cv2',
        'numpy',
        'scipy',
        'scipy.ndimage',
        # 文件IO
        'tifffile',
        # 数据分析
        'pandas',
        'pandas.io.excel',
        'pandas._libs',
        'openpyxl',
        # 项目模块
        'constants',
        'core',
        'core.segmentation',
        'core.measurement',
        'core.analysis',
        'core.logger',
        'gui',
        'gui.app',
        'gui.parameter_panel',
        'gui.canvas_viewer',
        'io_utils',
        'io_utils.tiff_handler',
        'io_utils.exporter',
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
        'matplotlib',
        'seaborn',
        'IPython',
        'notebook',
        'pytest',
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
    name='电镜晶体非晶区域统计分析工具v5.3',
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
