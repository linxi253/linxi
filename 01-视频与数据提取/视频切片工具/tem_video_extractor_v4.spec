# -*- mode: python ; coding: utf-8 -*-
import re
from pathlib import Path

from PyInstaller.utils.hooks import copy_metadata

ROOT = Path(SPECPATH)
# 版本单点来源：video_extractor/__init__.py 的 __version__
version_match = re.search(
    r'__version__ = "([^"]+)"',
    (ROOT / "video_extractor" / "__init__.py").read_text(encoding="utf-8"),
)
if not version_match:
    raise RuntimeError("无法从 video_extractor/__init__.py 读取 __version__")
VERSION = version_match.group(1)
EXE_NAME = f"TEMVideoExtractor-v{VERSION}"
FFMPEG_DIR = ROOT / "tools" / "ffmpeg"
required_names = [
    "ffmpeg.exe", "ffprobe.exe",
    "avcodec-62.dll", "avdevice-62.dll", "avfilter-11.dll",
    "avformat-62.dll", "avutil-60.dll", "swresample-6.dll", "swscale-9.dll",
]
required = [FFMPEG_DIR / name for name in required_names]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise RuntimeError("拒绝构建：缺少已验证的 FFmpeg 运行时：" + ", ".join(missing))

binaries = [(str(path), ".") for path in required]
runtime_doc_names = [
    "LICENSE", "LICENSE-LGPLv3.txt", "PROVENANCE.md", "README.md",
    "THIRD_PARTY_NOTICES.md",
]
datas = [(str(FFMPEG_DIR / name), "tools/ffmpeg") for name in runtime_doc_names]
# 确保 package metadata 存在（修复 importlib.metadata.PackageNotFoundError）
for pkg in ("tifffile", "numpy"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

a = Analysis(
    ["run_app.py"],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "tifffile", "numpy",
        "video_extractor", "video_extractor.app", "video_extractor.ui",
        "video_extractor.cli", "video_extractor.runner",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PIL", "matplotlib", "scipy", "cv2", "imageio", "imageio_ffmpeg"],
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
    name=EXE_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
