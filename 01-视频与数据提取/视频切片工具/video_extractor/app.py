import argparse
import os

from .ffmpeg import FFmpegManager
from .ui import ExtractorApp


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
