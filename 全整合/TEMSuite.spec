# -*- mode: python ; coding: utf-8 -*-
"""TEM Suite 打包配置。

打包范围
--------
只打包整合层（temsuite）与全部 Python 依赖，**不打包各工具的源码**。

理由：整合层的核心设计是「不修改原项目、按需引用其源码」，各工具作为独立
git 仓库持续演进。若把源码一并封进 exe，每次上游改动都需重新打包，
反而丧失了该设计的主要收益。

因此 exe 运行时从工作区目录读取工具源码。工作区定位规则见
``temsuite/registry.py`` 的 ``_detect_workspace_root()``：
先看环境变量 ``TEMSUITE_WORKSPACE``，否则从 exe 所在目录逐级向上查找。

构建后请把 exe 放在 AIforTEM 目录内（任意层级均可）。
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# setuptools>=81 的 pkg_resources 不再自带 vendored 依赖
# （jaraco.text / appdirs / ...），缺任一则冻结程序启动即崩
# （pyi_rth_pkgres 钩子，2026-08-23 各工具重建后仅本整合层未修）。
# pkg_resources 和 setuptools 各有独立的 _vendor 目录，两棵都要整体收集。
vendor_hidden = (collect_submodules('setuptools._vendor')
                 + collect_submodules('pkg_resources._vendor'))

# ttkbootstrap 2.x 把主题所需的字体与图标放在包内 assets/ 下，
# 且没有官方 PyInstaller hook。缺失时 Window() 初始化即抛
# FileNotFoundError: ttkbootstrap/assets/icons/bootstrap.ttf
_datas = collect_data_files("ttkbootstrap")
_datas += collect_data_files("tkinterdnd2")

a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=[],
    datas=_datas,
    hiddenimports=[
        # ---- GUI ----
        'ttkbootstrap',
        'ttkbootstrap.themes',
        'ttkbootstrap.style',
        'ttkbootstrap.constants',
        'tkinterdnd2',
        # ---- 数值与图像 ----
        'numpy',
        'scipy',
        'scipy.ndimage',
        'scipy.signal',
        'scipy.stats',
        'scipy.spatial',
        'scipy.interpolate',
        'scipy.fft',
        'cv2',
        'tifffile',
        'imagecodecs',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'skimage',
        'skimage.filters',
        'skimage.morphology',
        'skimage.measure',
        # ---- 绘图与数据 ----
        'matplotlib',
        'matplotlib.backends.backend_agg',
        'matplotlib.backends.backend_tkagg',
        'pandas',
        'seaborn',
        'openpyxl',
        # ---- 其他 ----
        'defusedxml',
        'ncempy',
        'ncempy.io',
        # ---- 外部工具源码引用的标准库子模块 ----
        # 各工具源码不打包、运行时才从工作区加载，PyInstaller 扫描不到它们的
        # import，因此这些子模块必须显式声明（缺任一则对应工具加载即 ImportError）
        'tkinter.colorchooser',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.simpledialog',
        'tkinter.font',
        'tkinter.ttk',
        # stem-optimize worker.py 运行期才 import（拖拽文件路径解析）
        'ctypes.wintypes',
    ] + vendor_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # pywin32 在本机 conda 环境中损坏（_win32sysloader 无法加载），
        # 且所有工具都不需要它
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
    name='TEM Suite',
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
