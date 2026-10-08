# -*- coding: utf-8 -*-
"""Assert the CI FFmpeg download destination is the one the consumer actually reads.

The pipeline job (10-DSH集成) and the video job (01-视频与数据提取/视频切片工具)
both need FFmpeg. ``video_extractor.FFmpegManager`` resolves ``tools/ffmpeg``
relative to its own ``__file__`` root, and the pipeline reuses that same module
through ``sys.path``. So downloading into ``<matrix.dir>/tools/ffmpeg`` is wrong
for the pipeline job -- it would place the asset somewhere the resolver never
looks, and PATH is not extended either.

This script is called from ci.yml right after the download to prove, at run
time, that the destination equals the resolver's expected directory. The
companion test (``.tools/tests/test_ci_contract.py``) checks the same contract
statically so a ci.yml edit cannot drift unnoticed.

Usage::

    python .tools/verify_ffmpeg_asset.py --expect-dir <downloaded dir>
    python .tools/verify_ffmpeg_asset.py --print-consumer-dir
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent

# The module that actually resolves FFmpeg, relative to the repository root.
CONSUMER_PACKAGE_PARENT = Path("01-视频与数据提取") / "视频切片工具"
FFMPEG_SUBDIR = Path("tools") / "ffmpeg"


def consumer_dir() -> Path:
    """Directory ``FFmpegManager`` searches (root/tools/ffmpeg)."""
    return ROOT / CONSUMER_PACKAGE_PARENT


def consumer_candidates() -> list[Path]:
    """Mirror of ``video_extractor.ffmpeg.FFmpegManager._bundled_candidates``."""
    root = consumer_dir()
    return [root / "ffmpeg.exe",
            root / FFMPEG_SUBDIR / "ffmpeg.exe",
            root / FFMPEG_SUBDIR / "ffmpeg"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--print-consumer-dir", action="store_true")
    parser.add_argument("--expect-dir", type=Path, default=None,
                        help="directory the CI download step wrote to")
    args = parser.parse_args(argv)

    expected = (consumer_dir() / FFMPEG_SUBDIR).resolve()
    if args.print_consumer_dir:
        print(expected)
        return 0

    if args.expect_dir is None:
        parser.error("either --expect-dir or --print-consumer-dir is required")
    actual = args.expect_dir.resolve()
    if actual != expected:
        print(f"[FAIL] FFmpeg was downloaded to {actual}", file=sys.stderr)
        print(f"       but the consumer resolves {expected}", file=sys.stderr)
        print("       (video_extractor.FFmpegManager reads tools/ffmpeg next to "
              "its own package root; PATH is not extended)", file=sys.stderr)
        return 1

    # The source-mode layout is root/tools/ffmpeg/{ffmpeg,ffprobe}.exe; the
    # frozen layout (root/ffmpeg.exe next to the executable) is produced by the
    # packager, not by this CI step, so only the source-mode pair is required.
    required = [expected / "ffmpeg.exe", expected / "ffprobe.exe"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        print(f"[FAIL] consumer directory is missing expected binaries: {missing}",
              file=sys.stderr)
        return 1
    print(f"[ok  ] FFmpeg asset directory matches the consumer: {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
