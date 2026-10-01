# -*- mode: python ; coding: utf-8 -*-

import os

from PyInstaller.utils.hooks import collect_submodules

project_root = os.path.dirname(os.path.abspath(SPEC))

# setuptools>=81 的 pkg_resources 不再自带 vendored 依赖
# （jaraco.text / appdirs / ...），缺任一则冻结程序启动即崩
# （pyi_rth_pkgres 钩子）。pkg_resources 和 setuptools 各有独立的
# _vendor 目录，两棵都要整体收集。
vendor_hidden = (collect_submodules('setuptools._vendor')
                 + collect_submodules('pkg_resources._vendor'))

a = Analysis(
    ['stem_processor_gui.py'],
    pathex=[project_root, os.path.join(project_root, 'core')],
    binaries=[],
    datas=[],
    hiddenimports=['ncempy', 'ncempy.io', 'ncempy.io.dm',
                   'scipy', 'scipy.ndimage'] + vendor_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'pythoncom', 'pywin32', 'win32com', 'win32api', 'pywintypes', '_win32sysloader',
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
    name='4D-STEM_Processor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX compresses to a smaller exe but is a well-known antivirus
    # false-positive trigger on Windows; keep it off for distribution.
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
