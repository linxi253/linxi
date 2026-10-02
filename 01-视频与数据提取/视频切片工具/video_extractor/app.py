import argparse
import os

from .ffmpeg import FFmpegManager
from .ui import ExtractorApp

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



def main(argv: list[str] | None = None) -> int:
    if os.environ.get("TEM_VIDEO_EXTRACTOR_SMOKE") == "1":
        FFmpegManager().resolve()
        return 0
    parser = argparse.ArgumentParser(description="TEM Video Extractor")
    parser.add_argument(
        "--headless-smoke",
        action="store_true",
        help="Verify the bundled FFmpeg runtime and exit without opening the GUI.",
    )
    args = parser.parse_args(argv)
    if args.headless_smoke:
        runtime = FFmpegManager().resolve()
        print(f"FFmpeg runtime verified: {'.'.join(map(str, runtime.version))}")
        return 0
    ExtractorApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
