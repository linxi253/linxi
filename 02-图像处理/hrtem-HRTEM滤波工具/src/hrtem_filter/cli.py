"""Command-line entry point for repeatable, non-interactive processing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._version import __version__
from .core import HRTEMFilter
from .geometry import Roi
from .params import FilterParams, OutputEncoding, SaveOptions
from .pipeline import ProcessingCancelled, StackProcessor
from .tiff_io import inspect_tiff


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HRTEM/STEM Kilaas filter v5")
    parser.add_argument("--version", action="version", version=f"HRTEM/STEM filter {__version__}")
    parser.add_argument("input", type=Path, nargs="?", help="输入 TIFF")
    parser.add_argument("output", type=Path, nargs="?", help="输出 TIFF")
    parser.add_argument("--inspect", action="store_true", help="仅显示 TIFF 结构，不处理")
    parser.add_argument("--mode", choices=("wiener", "absf", "butterworth"), default="wiener")
    parser.add_argument("--stem", action="store_true", help="为所有请求的输出应用 STEM 十字掩膜")
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--delta", type=float, default=5.0)
    parser.add_argument("--cycles", type=int, default=99)
    parser.add_argument("--bw-order", type=int, default=4)
    parser.add_argument("--bw-ro", type=float, default=0.30)
    parser.add_argument("--no-butterworth", action="store_true")
    parser.add_argument("--low-freq-percent", type=float, default=0.0)
    parser.add_argument("--crosshair-width", type=int, default=4)
    parser.add_argument("--crosshair-hole-radius", type=int, default=4)
    parser.add_argument("--crosshair-bw-ro", type=float, default=0.03)
    parser.add_argument("--rotation-method", choices=("fast_radial_bin", "dm_compatible"), default="fast_radial_bin")
    parser.add_argument("--fft-workers", type=int, default=1, help="FFT 线程数（默认 1；大图/多帧可调高）")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("TOP", "LEFT", "BOTTOM", "RIGHT"))
    parser.add_argument(
        "--encoding",
        choices=tuple(item.value for item in OutputEncoding),
        default=OutputEncoding.FLOAT32.value,
        help="float32 为默认定量输出；uint8-display 仅用于显示",
    )
    parser.add_argument(
        "--display-range",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        help="uint8-display 的全局显示范围（仅该编码使用）",
    )
    parser.add_argument(
        "--input-hash",
        action="store_true",
        help="在处理记录中写入输入文件的 SHA-256（归档级标识；需要对输入做一次完整读取）",
    )
    return parser


def _progress(phase: str, current: int, total: int) -> None:
    label = "计算全局显示范围" if phase == "scan" else "写入"
    print(f"\r{label}: {current}/{total}", end="", flush=True)
    if current == total:
        print()


def _warn_large_memory(info, roi) -> None:
    """大图处理前给出不可交互的内存提示（CLI 无法弹窗，只能提前告知）。"""
    shape = roi.shape if roi is not None else info.frame_shape
    estimate = HRTEMFilter.estimate_peak_bytes(shape)
    if estimate > 4 * 1024**3:
        print(
            f"警告: 单帧峰值内存估计约 {estimate / 1024**2:.0f} MiB，"
            "可能耗尽内存或触发交换；可先用 --roi 在小区域上验证参数。",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.inspect:
        if args.input is None:
            _parser().error("--inspect 需要输入 TIFF")
        info = inspect_tiff(args.input)
        print(f"文件: {info.path}")
        print(f"帧数: {info.frame_count}")
        print(f"灰度尺寸: {info.frame_shape[1]}×{info.frame_shape[0]}")
        print(f"类型: {info.dtype}")
        print(f"series axes: {info.metadata.axes}")
        return 0
    if args.input is None or args.output is None:
        _parser().error("需要 INPUT 和 OUTPUT，或使用 --inspect")
    try:
        params = FilterParams(
            step=args.step,
            delta=args.delta,
            cycles=args.cycles,
            bw_order=args.bw_order,
            bw_ro=args.bw_ro,
            apply_butterworth=not args.no_butterworth,
            primary_output=args.mode,
            stem_filter=args.stem,
            crosshair_width=args.crosshair_width,
            crosshair_hole_radius=args.crosshair_hole_radius,
            crosshair_bw_ro=args.crosshair_bw_ro,
            low_freq_percent=args.low_freq_percent,
            rotation_method=args.rotation_method,
        ).validated()
        options = SaveOptions(
            OutputEncoding(args.encoding), tuple(args.display_range) if args.display_range else None
        ).validated()
        roi = Roi(*args.roi) if args.roi else None
        _warn_large_memory(inspect_tiff(args.input), roi)
        filter_processor = HRTEMFilter(workers=args.fft_workers)
        report = StackProcessor(filter_processor=filter_processor).process_tiff(
            args.input,
            args.output,
            params=params,
            save_options=options,
            roi=roi,
            progress=_progress,
            entry="cli",
            input_hash=args.input_hash,
        )
    except ProcessingCancelled as exc:
        print(f"已取消: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"处理失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"完成: {report.output_path}")
    print(f"处理记录: {report.provenance_path}")
    if report.display_range is not None:
        low, high = report.display_range
        print(f"全局显示范围: {low:.6g} .. {high:.6g}（uint8 线性映射）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
