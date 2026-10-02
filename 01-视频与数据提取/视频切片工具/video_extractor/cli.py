from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .models import BitDepth, ColorMode, ExtractOptions, JobState, Sampling, SamplingMode
from .runner import JobRunner

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"TEM Video Extractor v{__version__} — 视频转 TIFF 堆栈",
    )
    parser.add_argument("--input", required=True, type=Path, help="包含视频的输入目录")
    parser.add_argument("--output", required=True, type=Path, help="输出目录")
    parser.add_argument("--ffmpeg", help="ffmpeg 可执行文件的路径")
    parser.add_argument("--mode", choices=[item.value for item in SamplingMode], default=SamplingMode.ALL.value)
    parser.add_argument("--value", type=float, help="目标 FPS 或时间间隔（秒）；全部帧模式下不接受")
    parser.add_argument("--color", choices=[item.value for item in ColorMode], default=ColorMode.PRESERVE.value)
    parser.add_argument("--bit-depth", choices=[item.value for item in BitDepth], default=BitDepth.UINT8.value)
    parser.add_argument(
        "--lenient-decode", action="store_true",
        help="宽松解码：损坏帧跳过而非任务失败（结果可能缺帧且位置不可知，仅建议抢救性提取使用）",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        input_root = args.input.expanduser()
        output_root = args.output.expanduser()
        if not input_root.is_dir():
            raise ValueError(f"输入目录不存在: {input_root}")
        if output_root.exists() and not output_root.is_dir():
            raise ValueError(f"输出路径已存在但不是目录: {output_root}")
        options = ExtractOptions(
            sampling=Sampling(SamplingMode(args.mode), args.value),
            color_mode=ColorMode(args.color),
            bit_depth=BitDepth(args.bit_depth),
            lenient_decode=args.lenient_decode,
        )
        runner = JobRunner(options, args.ffmpeg)

        def report(event: dict) -> None:
            # 进度/告警事件走 stderr，stdout 只保留最终结果数组，
            # 使 `python -m video_extractor.cli ... | python -m json.tool` 可直接解析
            print(json.dumps(event, ensure_ascii=False), flush=True, file=sys.stderr)

        results = runner.run_batch(input_root, output_root, report)
    except KeyboardInterrupt:
        print("已取消", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1
    print(json.dumps([result.as_dict() for result in results], ensure_ascii=False, indent=2))
    states = [result.state for result in results]
    if all(state is JobState.COMPLETED for state in states):
        return 0
    # 取消与失败使用不同退出码，脚本可以区分"没做完"和"做错了"
    if any(state is JobState.CANCELLED for state in states):
        return 130
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
