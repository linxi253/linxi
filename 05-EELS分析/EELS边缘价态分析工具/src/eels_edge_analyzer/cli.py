"""命令行入口：适合批处理、复现实验和无 GUI 环境。"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

from .dm4io import format_dataset_table, infer_dataset_indices, inspect_dm4
from .models import AnalysisConfig, ReferenceSpec
from .pipeline import run_analysis
from .presets import config_from_saved, load_preset, parse_distance_bins_text
from .references import discover_cu_references
from .reporting import export_artifacts

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# CLI 参数的 argparse dest → AnalysisConfig 字段。
_VALUE_FIELDS: dict[str, str] = {
    "fit_min_ev": "fit_min_ev",
    "fit_max_ev": "fit_max_ev",
    "background_min_ev": "background_min_ev",
    "background_max_ev": "background_max_ev",
    "smoothing_ev": "smoothing_ev",
    "regularization": "deconvolution_regularization",
    "along_surface_segment_nm": "along_surface_segment_nm",
}

# 从 --config 复跑时排除的字段（由输入/输出/参考谱参数单独解析）。
_SAVED_EXCLUDED_FIELDS = ("input_path", "output_dir", "references")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EELS Edge Analyzer v1: Cu-L Dual-EELS 边缘价态分析",
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="输入 DM3/DM4 Spectrum Image（使用 --config 复跑时可省略）",
    )
    parser.add_argument("--output-dir", type=Path, help="独立结果输出目录")
    parser.add_argument("--references-dir", type=Path, help="含 Cu Metal / Cu2O / CuO 参考谱的文件夹")
    parser.add_argument("--cu0", type=Path, help="Cu0 参考谱")
    parser.add_argument("--cu1", type=Path, help="Cu1 (Cu2O) 参考谱")
    parser.add_argument("--cu2", type=Path, help="Cu2 (CuO) 参考谱")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="从上次导出的 analysis_config.json 复现全部参数（与 --preset/--no-preset 互斥）",
    )
    parser.add_argument(
        "--preset",
        type=Path,
        default=None,
        help="参数预设 JSON；缺省使用内置 Cu L2,3 预设",
    )
    parser.add_argument("--no-preset", action="store_true", help="不加载预设，全部使用程序默认值")
    parser.add_argument("--fit-min-ev", type=float, help="拟合窗口下限 (eV)")
    parser.add_argument("--fit-max-ev", type=float, help="拟合窗口上限 (eV)")
    parser.add_argument("--background-min-ev", type=float, help="幂律背景拟合下限 (eV)")
    parser.add_argument("--background-max-ev", type=float, help="幂律背景拟合上限 (eV)")
    parser.add_argument("--smoothing-ev", type=float, help="Savitzky-Golay 平滑宽度 (eV)")
    parser.add_argument("--regularization", type=float, help="去卷积正则化参数")
    parser.add_argument(
        "--distance-bins",
        type=str,
        help='距离分层，如 "E1:0:2.2,E2:2.2:4.4,Bulk:11"（末层省略上限即开区间）',
    )
    parser.add_argument(
        "--along-surface-segment-nm",
        type=float,
        help="沿表面分段宽度 (nm)；Cu L2,3 内置预设为 11.0，--no-preset 时必须显式提供",
    )
    parser.add_argument("--orientation", choices=("auto", "top", "bottom", "left", "right"), default=None)
    parser.add_argument("--survey-dataset", type=int)
    parser.add_argument("--low-loss-dataset", type=int)
    parser.add_argument("--high-loss-dataset", type=int)
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--no-sensitivity", action="store_true")
    parser.add_argument("--no-injection", action="store_true")
    parser.add_argument(
        "--npz-plain",
        action="store_true",
        help="processed_arrays.npz 不做 zlib 压缩，牺牲磁盘空间换取大 SI 的写盘速度",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖输出目录中已存在的既往分析结果（默认拒绝静默覆盖）",
    )
    parser.add_argument("--fast", action="store_true", help="用于试运行：100 次 bootstrap，关闭注入恢复")
    parser.add_argument("--inspect", action="store_true", help="仅列出 DM4 对象及自动推荐，不运行分析")
    return parser


def _references(
    args: argparse.Namespace,
    saved: AnalysisConfig | None,
) -> tuple[ReferenceSpec, ...]:
    if args.references_dir is not None:
        return discover_cu_references(args.references_dir)
    if all((args.cu0, args.cu1, args.cu2)):
        return (
            ReferenceSpec("Cu0", 0, args.cu0),
            ReferenceSpec("Cu1", 1, args.cu1),
            ReferenceSpec("Cu2", 2, args.cu2),
        )
    if saved is not None and saved.references:
        return saved.references
    raise ValueError("请提供 --references-dir，或同时提供 --cu0、--cu1、--cu2。")


def _config_overrides(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> dict:
    """合并 preset/--config 与显式 CLI 参数；显式参数优先级最高。

    运行控制类参数（bootstrap、敏感性、注入恢复）只在用户显式给出或基础
    配置未提供时才落到默认值，因此 --config 复跑与预设中写入的值不会被
    默认值静默冲掉。
    """

    if args.no_preset and args.preset is not None:
        parser.error("--no-preset 与 --preset 不能同时指定。")
    if args.config is not None:
        saved = config_from_saved(args.config)
        overrides: dict = {
            field.name: getattr(saved, field.name)
            for field in fields(AnalysisConfig)
            if field.name not in _SAVED_EXCLUDED_FIELDS
        }
    elif args.no_preset:
        overrides = {}
    else:
        overrides = load_preset(args.preset)
    for arg_name, field in _VALUE_FIELDS.items():
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field] = value
    if args.distance_bins is not None:
        overrides["distance_bins"] = parse_distance_bins_text(args.distance_bins)
    # 运行控制类参数：显式 CLI > preset/--config > 程序默认。
    if args.fast:
        overrides["bootstrap_resamples"] = 100
        overrides["injection_simulations"] = 0
    else:
        if args.bootstrap is not None:
            overrides["bootstrap_resamples"] = args.bootstrap
        elif "bootstrap_resamples" not in overrides:
            overrides["bootstrap_resamples"] = 500
        if args.no_injection:
            overrides["injection_simulations"] = 0
        elif "injection_simulations" not in overrides:
            overrides["injection_simulations"] = 200
    if args.no_sensitivity:
        overrides["run_sensitivity"] = False
    elif "run_sensitivity" not in overrides:
        overrides["run_sensitivity"] = True
    return overrides


def _progress(message: str, fraction: float | None) -> None:
    prefix = f"[{100 * fraction:5.1f}%] " if fraction is not None else ""
    print(prefix + message, flush=True)


def _run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.fast and args.bootstrap is not None:
        parser.error("--fast 与 --bootstrap 不能同时指定；--fast 固定使用 100 次 bootstrap。")
    if args.inspect:
        if args.input is None:
            parser.error("--inspect 需要提供输入 DM3/DM4 文件。")
        try:
            items = inspect_dm4(args.input)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"DM4 检查失败：{exc}", file=sys.stderr)
            return 2
        survey, low, high = infer_dataset_indices(items)
        print(format_dataset_table(items, survey, low, high))
        return 0

    saved = None
    if args.config is not None:
        if args.preset is not None or args.no_preset:
            parser.error("--config 与 --preset/--no-preset 不能同时指定。")
        saved = config_from_saved(args.config)
    if args.input is None and saved is None:
        parser.error("必须提供输入 DM3/DM4 文件，或使用 --config 指定复跑配置。")
    source = args.input if args.input is not None else saved.input_path
    default_output = source.with_name(f"{source.stem}_eels_edge_analysis")
    output = args.output_dir or (saved.output_dir if saved is not None else default_output)

    def _dataset_index(cli_value: int | None, saved_value: int | None) -> int | None:
        return cli_value if cli_value is not None else saved_value

    orientation = args.orientation
    if orientation is None:
        orientation = saved.surface_orientation if saved is not None else "auto"

    config = AnalysisConfig(
        input_path=source,
        output_dir=output,
        references=_references(args, saved),
        survey_dataset=_dataset_index(args.survey_dataset, saved.survey_dataset if saved else None),
        low_loss_dataset=_dataset_index(args.low_loss_dataset, saved.low_loss_dataset if saved else None),
        high_loss_dataset=_dataset_index(args.high_loss_dataset, saved.high_loss_dataset if saved else None),
        surface_orientation=orientation,
        **_config_overrides(parser, args),
    )
    artifacts = run_analysis(config, progress=_progress)
    paths = export_artifacts(artifacts, config, compress_arrays=not args.npz_plain, overwrite=args.overwrite)
    print(f"完成。报告：{paths['report']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return _run(parser, args)
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
