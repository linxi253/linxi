"""针对 2026-09 代码审查修复项的回归测试。

覆盖：输入校验收紧、CLI 友好错误、CSV 注入清洗、输出目录覆盖守卫、
多正则化去卷积等价性、shift 越界守卫、真配对 bootstrap、块宽建议、
config 复跑 loader、以及此前缺少直接单测的核心数值函数。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from test_core import _gaussian, _synthetic_dual_eels

from eels_edge_analyzer import reporting
from eels_edge_analyzer.cli import main as cli_main
from eels_edge_analyzer.fitting import moving_block_indices, paired_block_indices, suggested_block_width
from eels_edge_analyzer.models import MAX_BOOTSTRAP_RESAMPLES, AnalysisConfig, ReferenceSpec
from eels_edge_analyzer.presets import config_from_saved
from eels_edge_analyzer.processing import (
    auto_orientation_confident,
    column_spectra_for_mask,
    fourier_ratio_deconvolution,
    fourier_ratio_deconvolution_multi,
    low_loss_thickness,
    power_law_subtract,
    process_region_spectrum,
)


def _base_config(tmp_path: Path, **overrides) -> AnalysisConfig:
    source = tmp_path / "input.dm4"
    source.touch()
    for state in range(3):
        (tmp_path / f"cu{state}.dm4").touch()
    references = tuple(
        ReferenceSpec(f"Cu{state}", state, tmp_path / f"cu{state}.dm4") for state in range(3)
    )
    kwargs: dict = {
        "input_path": source,
        "output_dir": tmp_path / "output",
        "references": references,
        # 与样品/预设绑定的参数不再有数值默认，测试显式给一个中性值。
        "along_surface_segment_nm": 5.0,
    }
    kwargs.update(overrides)
    return AnalysisConfig(**kwargs)


# ---------------------------------------------------------------------------
# 输入校验收紧
# ---------------------------------------------------------------------------


def test_config_rejects_empty_output_directory(tmp_path) -> None:
    # Path("") 与 Path(".") 都指向当前目录，旧版会让结果静默写入启动目录。
    with pytest.raises(ValueError, match="输出目录不能为空"):
        _base_config(tmp_path, output_dir=Path("")).validate()
    with pytest.raises(ValueError, match="输出目录不能为空"):
        _base_config(tmp_path, output_dir=Path(".")).validate()


def test_config_rejects_empty_input_path(tmp_path) -> None:
    with pytest.raises(ValueError, match="输入 DM3/DM4 文件路径不能为空"):
        _base_config(tmp_path, input_path=Path("")).validate()


def test_config_rejects_negative_dataset_indices(tmp_path) -> None:
    with pytest.raises(ValueError, match="不能为负数"):
        _base_config(tmp_path, low_loss_dataset=-1).validate()
    with pytest.raises(ValueError, match="不能为负数"):
        _base_config(tmp_path, high_loss_dataset=-3).validate()


def test_config_rejects_absurd_bootstrap_count(tmp_path) -> None:
    with pytest.raises(ValueError, match="bootstrap 重采样次数不能超过"):
        _base_config(tmp_path, bootstrap_resamples=MAX_BOOTSTRAP_RESAMPLES + 1).validate()
    _base_config(tmp_path, bootstrap_resamples=500).validate()


# ---------------------------------------------------------------------------
# CLI：友好错误与参数合并
# ---------------------------------------------------------------------------


def test_cli_reports_errors_friendly(capsys) -> None:
    code = cli_main(["missing.dm4", "--cu0", "a.dm4", "--cu1", "b.dm4", "--cu2", "c.dm4"])
    assert code == 2
    captured = capsys.readouterr()
    assert "错误" in captured.err
    assert "Traceback" not in captured.err


def test_cli_rejects_config_with_preset() -> None:
    with pytest.raises(SystemExit):
        cli_main(["in.dm4", "--config", "saved.json", "--preset", "p.json"])


def test_cli_config_run_control_not_clobbered(tmp_path) -> None:
    import json

    saved = _base_config(
        tmp_path, bootstrap_resamples=777, run_sensitivity=False, injection_simulations=0
    )
    saved_path = tmp_path / "analysis_config.json"
    saved_path.write_text(json.dumps(saved.to_dict(), ensure_ascii=False), encoding="utf-8")

    from eels_edge_analyzer.cli import _config_overrides, _parser

    parser = _parser()
    args = parser.parse_args(["--config", str(saved_path)])
    overrides = _config_overrides(parser, args)
    # --config 提供的运行控制参数不应被 CLI 默认值覆盖。
    assert overrides["bootstrap_resamples"] == 777
    assert overrides["run_sensitivity"] is False
    assert overrides["injection_simulations"] == 0
    # 显式 --fast 仍然优先。
    args = parser.parse_args(["--config", str(saved_path), "--fast"])
    overrides = _config_overrides(parser, args)
    assert overrides["bootstrap_resamples"] == 100
    assert overrides["injection_simulations"] == 0


# ---------------------------------------------------------------------------
# R9：--config 复跑必须真正走到 run_analysis
# ---------------------------------------------------------------------------


def _saved_config_path(tmp_path, **overrides):
    """写出一份可被 --config 读取的保存配置，返回其路径。"""
    import json

    source = tmp_path / "in.dm4"
    source.write_bytes(b"synthetic")
    refs = []
    for index in range(3):
        path = tmp_path / f"cu{index}.dm4"
        path.write_bytes(b"ref")
        refs.append(ReferenceSpec(f"Cu{index}", index, path))
    config = _base_config(tmp_path, input_path=source, references=tuple(refs), **overrides)
    saved = tmp_path / "analysis_config.json"
    saved.write_text(json.dumps(config.to_dict(), ensure_ascii=False), encoding="utf-8")
    return saved, config


def test_cli_config_reaches_run_analysis(tmp_path, monkeypatch) -> None:
    """回归 2026-10-03 R9：--config 复跑此前在 run_analysis 之前就 TypeError。

    ``_run`` 曾把 survey/low/high/surface_orientation 显式传给 AnalysisConfig，
    同时 ``_config_overrides`` 又从保存配置里带回同名字段，于是报
    ``got multiple values for keyword argument 'survey_dataset'``。本用例真正调用
    ``cli_main`` 并断言 run_analysis 被调用、方向来自显式 CLI。
    """
    from eels_edge_analyzer import cli

    saved, _config = _saved_config_path(tmp_path)
    seen: list = []

    monkeypatch.setattr(cli, "run_analysis",
                        lambda config, **kw: seen.append(config) or object())
    monkeypatch.setattr(cli, "export_artifacts",
                        lambda *a, **kw: {"report": tmp_path / "r.md"})

    rc = cli.main(["--config", str(saved), "--orientation", "right"])

    assert rc == 0
    assert seen, "run_analysis 必须被真正调用"
    assert seen[0].surface_orientation == "right"


def test_cli_config_keeps_saved_dataset_indices(tmp_path, monkeypatch) -> None:
    """保存配置里的数据对象编号必须保留，而不是被 CLI 默认 None 冲掉。"""
    from eels_edge_analyzer import cli

    saved, config = _saved_config_path(
        tmp_path, survey_dataset=4, low_loss_dataset=5, high_loss_dataset=6)
    seen: list = []
    monkeypatch.setattr(cli, "run_analysis",
                        lambda c, **kw: seen.append(c) or object())
    monkeypatch.setattr(cli, "export_artifacts",
                        lambda *a, **kw: {"report": tmp_path / "r.md"})

    assert cli.main(["--config", str(saved)]) == 0
    assert seen[0].survey_dataset == config.survey_dataset
    assert seen[0].low_loss_dataset == config.low_loss_dataset
    assert seen[0].high_loss_dataset == config.high_loss_dataset


def test_cli_explicit_value_beats_saved_config(tmp_path, monkeypatch) -> None:
    """显式 CLI > 保存配置；且 along_surface_segment_nm 也走同一优先级。"""
    from eels_edge_analyzer import cli

    saved, config = _saved_config_path(tmp_path, fit_min_ev=925.0)
    seen: list = []
    monkeypatch.setattr(cli, "run_analysis",
                        lambda c, **kw: seen.append(c) or object())
    monkeypatch.setattr(cli, "export_artifacts",
                        lambda *a, **kw: {"report": tmp_path / "r.md"})

    rc = cli.main(["--config", str(saved), "--fit-min-ev", "930",
                   "--along-surface-segment-nm", "7.5"])
    assert rc == 0
    assert seen[0].fit_min_ev == 930.0
    assert seen[0].along_surface_segment_nm == 7.5
    assert seen[0].fit_min_ev != config.fit_min_ev


def test_cli_config_missing_input_is_reported(tmp_path, capsys) -> None:
    """保存配置指向的输入缺失时，错误码与提示必须是可诊断的。"""
    import json

    saved = tmp_path / "analysis_config.json"
    saved.write_text(json.dumps({"input_path": str(tmp_path / "gone.dm4"),
                                 "output_dir": str(tmp_path / "out"),
                                 "references": []}), encoding="utf-8")
    from eels_edge_analyzer import cli
    rc = cli.main(["--config", str(saved)])
    assert rc != 0


# ---------------------------------------------------------------------------
# R1：配对能量轴校验（色散一致但起点可不同；非均匀/非有限必须拒绝）
# ---------------------------------------------------------------------------


def _axis(size: int, step: float, origin: float) -> np.ndarray:
    return np.arange(size, dtype=float) * step + origin


def test_paired_axes_accept_same_dispersion_different_origin() -> None:
    """同色散、不同能量起点必须通过：低损含 ZLP、高损从吸收边开始是常态。"""
    from eels_edge_analyzer.models import validate_paired_energy_axes

    validate_paired_energy_axes(_axis(64, 0.25, -5.0), _axis(64, 0.25, 900.0))


def test_paired_axes_reject_unequal_dispersion() -> None:
    from eels_edge_analyzer.models import validate_paired_energy_axes

    with pytest.raises(ValueError, match="色散不一致"):
        validate_paired_energy_axes(_axis(64, 0.25, -5.0), _axis(64, 1.0, 900.0))


def test_paired_axes_reject_channel_count_mismatch() -> None:
    from eels_edge_analyzer.models import validate_paired_energy_axes

    with pytest.raises(ValueError, match="通道数不一致"):
        validate_paired_energy_axes(_axis(64, 0.25, -5.0), _axis(32, 0.25, 900.0))


def test_paired_axes_reject_non_uniform_sampling() -> None:
    """非均匀采样必须被拒绝，而不是被静默当成等间距使用。"""
    from eels_edge_analyzer.models import validate_paired_energy_axes

    high = _axis(64, 0.25, 900.0)
    high[20:] += 0.5
    with pytest.raises(ValueError, match="不均匀"):
        validate_paired_energy_axes(_axis(64, 0.25, -5.0), high)


def test_paired_axes_reject_non_finite_and_short() -> None:
    from eels_edge_analyzer.models import validate_paired_energy_axes

    high = _axis(64, 0.25, 900.0)
    high[3] = np.nan
    with pytest.raises(ValueError, match="非有限"):
        validate_paired_energy_axes(_axis(64, 0.25, -5.0), high)

    with pytest.raises(ValueError, match="只有 1 个通道"):
        validate_paired_energy_axes(np.array([0.0]), np.array([1.0]))


def test_paired_axes_reject_non_positive_step_and_2d() -> None:
    from eels_edge_analyzer.models import validate_paired_energy_axes

    descending = _axis(64, 0.25, 900.0)[::-1].copy()
    with pytest.raises(ValueError):
        validate_paired_energy_axes(_axis(64, 0.25, -5.0), descending)

    with pytest.raises(ValueError, match="一维"):
        validate_paired_energy_axes(_axis(64, 0.25, -5.0),
                                    np.zeros((4, 16), dtype=float))


def test_paired_axes_tolerate_float32_representation_noise() -> None:
    """float32 存储的标称色散不得被误判为不一致。

    两条轴同为 0.25 eV/ch、只有能量起点不同，但都经 float32 往返；在 1000 eV
    量级上 float32 的相对误差约 1e-7，绝对误差约 1e-4 eV。校验必须容忍这种
    表示噪声，否则合法数据会被误拒。
    """
    from eels_edge_analyzer.models import validate_paired_energy_axes

    step = np.float32(0.25)
    low = (np.arange(1000, dtype=np.float32) * step - np.float32(50.0)).astype(float)
    high = (np.arange(1000, dtype=np.float32) * step + np.float32(900.0)).astype(float)
    # 前提：两条轴的名义色散确实相同（只差浮点表示）
    assert abs(np.median(np.diff(low)) - np.median(np.diff(high))) < 1e-3
    validate_paired_energy_axes(low, high)   # 不应抛异常


@pytest.mark.parametrize("step_low,step_high,origin", [
    (0.01, 0.0103, 900.0),      # 3% 失配（返工中曾漏过）
    (0.001, 0.0012, 900.0),     # 20% 失配（返工中曾漏过）
    (0.01, 0.011, 3000.0),      # 10% 失配（返工中曾漏过）
])
def test_paired_axes_reject_small_relative_mismatches(
        step_low: float, step_high: float, origin: float) -> None:
    """小相对失配必须被拒绝——容差不得随原点量级无界放大。

    返工的中间实现把端点估计误差按 4*quantum 计（未除以 n-1），使容差可接近
    100% 步长，于是上表三例全部错误通过。这里逐个钉住。
    """
    from eels_edge_analyzer.models import validate_paired_energy_axes

    low = np.arange(128, dtype=float) * step_low
    high = (origin + np.arange(128, dtype=float) * step_high).astype(np.float32).astype(float)
    with pytest.raises(ValueError):
        validate_paired_energy_axes(low, high)


def test_paired_axes_reject_huge_origin_with_local_jump() -> None:
    """超大原点 + 半数通道跳变必须被拒绝（不得被精度容差吞掉）。"""
    from eels_edge_analyzer.models import validate_paired_energy_axes

    low = np.arange(128, dtype=float) * 0.25
    high = (1e6 + np.arange(128, dtype=float) * 0.25).astype(np.float32).astype(float)
    high[60:] += 0.125
    with pytest.raises(ValueError):
        validate_paired_energy_axes(low, high)


def test_paired_axes_precision_limit_is_reported_honestly() -> None:
    """精度不足时给出的必须是"坐标精度不足"，不能谎称色散确实不等。"""
    from eels_edge_analyzer.models import validate_paired_energy_axes

    low = np.arange(128, dtype=float) * 0.001
    high = (900.0 + np.arange(128, dtype=float) * 0.001).astype(np.float32).astype(float)
    with pytest.raises(ValueError, match="坐标精度不足"):
        validate_paired_energy_axes(low, high)


@pytest.mark.parametrize("step", [0.25, 0.5, 1.0])
def test_paired_axes_accept_float32_and_float64_eels_ranges(step: float) -> None:
    """常规 EELS 色散在 float32/float64、不同原点下都必须接受。"""
    from eels_edge_analyzer.models import validate_paired_energy_axes

    for dtype in (np.float64, np.float32):
        low = (-60.0 + np.arange(128) * step).astype(dtype).astype(float)
        high = (900.0 + np.arange(128) * step).astype(dtype).astype(float)
        validate_paired_energy_axes(low, high)


def test_single_axis_validation_is_reusable() -> None:
    """单轴校验是公开接口，供只处理低损谱的调用方复用。"""
    from eels_edge_analyzer.models import validate_single_energy_axis

    assert validate_single_energy_axis(np.arange(64) * 0.5, "低损对象") == 0.5
    with pytest.raises(ValueError, match="非有限值"):
        validate_single_energy_axis(np.array([0.0, np.nan, 1.0]), "低损对象")
    with pytest.raises(ValueError, match="严格递增"):
        validate_single_energy_axis(np.array([0.0, 1.0, 0.5]), "低损对象")
    nonuniform = np.arange(64) * 0.5
    nonuniform[10:] += 0.25
    with pytest.raises(ValueError, match="不均匀"):
        validate_single_energy_axis(nonuniform, "低损对象")


def test_deconvolution_validates_low_axis_without_high_axis() -> None:
    """不传高损轴时，低损轴仍必须被完整校验（兑现 docstring 的承诺）。"""
    high = np.ones((32, 2, 2), dtype=float)
    low = np.ones((32, 2, 2), dtype=float)
    bad = np.arange(32, dtype=float) * 0.25
    bad[5] = np.nan
    with pytest.raises(ValueError, match="非有限值"):
        fourier_ratio_deconvolution_multi(high, low, bad, (0.003,),
                                          (-30.0, -10.0), (-3.0, 3.0))


def test_deconvolution_rejects_axis_length_mismatch_with_si() -> None:
    """两条轴彼此等长但与 SI 通道数不符时，必须给出可诊断的错误。"""
    high = np.ones((32, 2, 2), dtype=float)
    low = np.ones((32, 2, 2), dtype=float)
    short = np.arange(16, dtype=float) * 0.25
    with pytest.raises(ValueError, match="与 SI 能量通道数"):
        fourier_ratio_deconvolution_multi(high, low, short, (0.003,),
                                          (-30.0, -10.0), (-3.0, 3.0))


def test_deconvolution_checks_paired_axes_when_high_axis_given() -> None:
    """公开 processing 接口传入高损轴时必须校验；不传则不声称已检查。"""
    high = np.ones((32, 2, 2), dtype=float)
    low = np.ones((32, 2, 2), dtype=float)
    low_axis = _axis(32, 0.25, -4.0)
    with pytest.raises(ValueError, match="色散不一致"):
        fourier_ratio_deconvolution(high, low, low_axis, 0.003, (-30.0, -10.0),
                                    (-3.0, 3.0), high_energy_ev=_axis(32, 1.0, 900.0))


def test_run_analysis_rejects_mismatched_dispersion(tmp_path, monkeypatch) -> None:
    """端到端：run_analysis 必须在去卷积之前拒绝不等色散（R1 反例）。"""
    from eels_edge_analyzer import pipeline
    from eels_edge_analyzer.models import LoadedDataset

    config = _base_config(tmp_path)
    low = LoadedDataset(1, np.ones((32, 2, 2)),
                        (_axis(32, 0.25, -4.0), np.arange(2), np.arange(2)),
                        ("eV", "nm", "nm"), "low")
    high = LoadedDataset(2, np.ones((32, 2, 2)),
                         (_axis(32, 1.0, 900.0), np.arange(2), np.arange(2)),
                         ("eV", "nm", "nm"), "high")
    monkeypatch.setattr(pipeline, "inspect_dm4", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "infer_dataset_indices", lambda *a, **k: (None, 1, 2))
    monkeypatch.setattr(pipeline, "read_dataset",
                        lambda path, index: low if index == 1 else high)

    with pytest.raises(ValueError, match="色散不一致"):
        pipeline.run_analysis(config)


# ---------------------------------------------------------------------------
# CSV 注入清洗与输出目录守卫
# ---------------------------------------------------------------------------


def test_write_csv_sanitizes_formula_cells(tmp_path) -> None:
    path = tmp_path / "table.csv"
    reporting.write_csv(
        path,
        [
            {"bin": "=SUM(A1:A2)", "snr": 3.5},
            {"bin": "@cmd", "snr": -1.5},
            {"bin": "E1", "snr": 2.0},
        ],
    )
    text = path.read_text(encoding="utf-8-sig")
    assert "'=SUM(A1:A2)" in text
    assert "'@cmd" in text
    # 数值型的负号不构成公式注入，必须保持原样。
    assert "-1.5" in text


def test_output_directory_state(tmp_path) -> None:
    assert reporting.output_directory_state(tmp_path / "absent") == "absent"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert reporting.output_directory_state(empty) == "empty"
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.txt").touch()
    assert reporting.output_directory_state(other) == "other_content"
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "edge_analysis_summary.json").touch()
    assert reporting.output_directory_state(previous) == "previous_results"


# ---------------------------------------------------------------------------
# 去卷积：多正则化与单正则化数值等价
# ---------------------------------------------------------------------------


def test_deconvolution_multi_matches_single() -> None:
    high, low, energy = _synthetic_dual_eels(seed=19, ny=3, nx=5)
    kwargs = {
        "baseline_range_ev": (-10.0, -5.0),
        "zlp_window_ev": (-2.0, 2.0),
    }
    regs = (0.001, 0.003, 0.01)
    cubes, diagnostics = fourier_ratio_deconvolution_multi(high, low, energy, regs, **kwargs)
    assert set(cubes) == set(regs)
    assert diagnostics["fft_length"] >= 2048
    for reg in regs:
        single, single_diag = fourier_ratio_deconvolution(high, low, energy, reg, **kwargs)
        tolerance = 1e-6 * max(1.0, float(np.nanmax(np.abs(single))))
        assert np.allclose(cubes[reg], single, rtol=1e-6, atol=tolerance, equal_nan=True)
        assert single_diag["regularization"] == reg
        assert single_diag["negative_value_fraction"] == pytest.approx(
            diagnostics["negative_value_fraction_by_regularization"][f"{reg:g}"]
        )


# ---------------------------------------------------------------------------
# 此前缺少直接单测的核心数值函数
# ---------------------------------------------------------------------------


def test_power_law_subtract_recovers_exponent() -> None:
    energy = np.linspace(850.0, 925.0, 200)
    truth_exponent = 3.2
    amplitude = 5e9
    spectrum = amplitude * energy ** (-truth_exponent)
    corrected, details = power_law_subtract(spectrum, energy, 850.0, 925.0)
    assert details["exponent"] == pytest.approx(truth_exponent, abs=0.02)
    assert details["log_fit_r2"] > 0.999
    # 扣除背景后残差应远小于原始信号。
    assert float(np.max(np.abs(corrected))) < 1e-6 * float(np.max(spectrum))


def test_low_loss_thickness_log_ratio() -> None:
    energy = np.linspace(-20.0, 200.0, 1101)
    zlp = _gaussian(energy, 0.0, 0.5) * 1e6
    tail = np.where(energy > 5.0, 2e3, 0.0)
    low = (zlp + tail).astype(np.float32)[:, None, None]
    thickness, diagnostics = low_loss_thickness(
        low, energy, (-20.0, -10.0), (-3.0, 3.0)
    )
    zero = float(np.trapezoid(zlp[(energy >= -3) & (energy <= 3)], energy[(energy >= -3) & (energy <= 3)]))
    total = zero + 2e3 * (200.0 - 5.0)
    expected = float(np.log(total / zero))
    assert thickness[0, 0] == pytest.approx(expected, rel=0.05)
    assert diagnostics["total_integral_max_ev"] == 200.0


def test_column_spectra_for_mask_orientation() -> None:
    cube = np.arange(4 * 3 * 5, dtype=float).reshape(4, 3, 5)
    mask = np.zeros((3, 5), dtype=bool)
    mask[1, :] = True
    mask[2, :] = True
    values = column_spectra_for_mask(cube, mask, "top")
    # 输出布局为 (沿表面条数, 能量通道)：每条 x 扫描线在所选行上平均。
    expected = np.array([cube[:, 1:3, x].mean(axis=1) for x in range(5)])
    assert values.shape == (5, 4)
    assert np.allclose(values, expected)


def test_suggested_block_width_detects_correlation() -> None:
    rng = np.random.default_rng(5)
    # 缓慢变化的沿表面信号 → 相关长度明显大于 1 列。
    correlated = np.cumsum(rng.normal(size=(120, 10)), axis=0)
    assert suggested_block_width(correlated) >= 3
    # 白噪声 → 建议块宽为 1。
    white = rng.normal(size=(120, 10))
    assert suggested_block_width(white) == 1
    # 常数谱（零方差）不应崩溃。
    assert suggested_block_width(np.ones((10, 8))) == 1


# ---------------------------------------------------------------------------
# shift 越界守卫与 auto 方向置信度
# ---------------------------------------------------------------------------


def test_process_region_spectrum_rejects_out_of_range_shift() -> None:
    energy = np.arange(850.0, 1000.5, 0.5)
    rng = np.random.default_rng(9)
    # 列谱布局为 (沿表面条数, 能量通道)；加入小噪声使 SNR 估计有限。
    single = 5e9 * energy ** (-3.0) + rng.normal(0, 0.5, energy.size)
    columns = np.tile(single, (6, 1))
    fit_grid = np.arange(925.0, 970.25, 0.5)
    with pytest.raises(ValueError, match="能量平移"):
        process_region_spectrum(columns, energy, fit_grid, 850.0, 925.0, 1.5, 3, shift_ev=80.0)
    spectrum, details = process_region_spectrum(
        columns, energy, fit_grid, 850.0, 925.0, 1.5, 3, shift_ev=0.5
    )
    assert spectrum.shape == fit_grid.shape
    assert np.isfinite(details["snr"])


def test_auto_orientation_confident() -> None:
    assert auto_orientation_confident(None) is None
    assert auto_orientation_confident(()) is None
    good = ({"orientation": "top", "contrast_quality": 0.8}, {"orientation": "bottom", "contrast_quality": -np.inf})
    assert auto_orientation_confident(good) is True
    bad = ({"orientation": "top", "contrast_quality": 0.0}, {"orientation": "bottom", "contrast_quality": -1.0})
    assert auto_orientation_confident(bad) is False


# ---------------------------------------------------------------------------
# 真配对 bootstrap
# ---------------------------------------------------------------------------


def test_paired_block_indices_shares_positions_when_equal() -> None:
    rng = np.random.default_rng(7)
    edge, bulk = paired_block_indices(20, 20, 5, rng)
    assert np.array_equal(edge, bulk)
    edge, bulk = paired_block_indices(20, 12, 5, rng)
    assert edge.shape == (20,)
    assert bulk.shape == (12,)
    # 独立抽样时两者通常不同，且各自都是合法的循环块索引。
    assert set(edge.tolist()) <= set(range(20))
    assert set(bulk.tolist()) <= set(range(12))


def test_moving_block_indices_bounds() -> None:
    rng = np.random.default_rng(3)
    indices = moving_block_indices(17, 5, rng)
    assert indices.shape == (17,)
    assert indices.min() >= 0 and indices.max() < 17


# ---------------------------------------------------------------------------
# config 复跑 loader
# ---------------------------------------------------------------------------


def test_config_from_saved_roundtrip(tmp_path) -> None:
    config = _base_config(
        tmp_path,
        bootstrap_resamples=777,
        surface_orientation="left",
        survey_dataset=1,
        low_loss_dataset=2,
        high_loss_dataset=3,
        random_seed=42,
    )
    import json

    saved_path = tmp_path / "analysis_config.json"
    saved_path.write_text(json.dumps(config.to_dict(), ensure_ascii=False), encoding="utf-8")
    loaded = config_from_saved(saved_path)
    assert loaded.input_path == config.input_path
    assert loaded.output_dir == config.output_dir
    assert loaded.bootstrap_resamples == 777
    assert loaded.surface_orientation == "left"
    assert loaded.survey_dataset == 1
    assert loaded.random_seed == 42
    assert [item.label for item in loaded.references] == ["Cu0", "Cu1", "Cu2"]
    assert [item.path for item in loaded.references] == [item.path for item in config.references]
    assert loaded.distance_bins == config.distance_bins
    assert loaded.zlp_window_ev == config.zlp_window_ev
    assert loaded.injection_fractions == config.injection_fractions
    loaded.validate()


def test_config_from_saved_rejects_unknown_keys(tmp_path) -> None:
    import json

    saved_path = tmp_path / "bad.json"
    saved_path.write_text(json.dumps({"input_path": "a.dm4", "typo_key": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="未知键"):
        config_from_saved(saved_path)


# --------------------------------------------------------------------------
# 2026-09-28 第二轮审查修复
# --------------------------------------------------------------------------
def test_jsonable_sanitizes_nonfinite_numpy_scalars_and_arrays() -> None:
    """numpy 标量/数组里的 NaN/Inf 也必须清洗成 null。

    此前 ndarray 走 tolist()、np.generic 走 .item()，均绕过末尾的有限性
    检查：坏像素产生的 NaN 会以 ``NaN`` 字面量写进 summary JSON，严格解析
    器（jq/JS/pandas strict）会拒绝整个文件。
    """
    import json

    from eels_edge_analyzer.models import jsonable

    payload = {
        "scalar_nan": np.float64("nan"),
        "scalar_inf": np.float32("inf"),
        "array": np.array([1.0, np.nan, 2.0]),
        "matrix": np.array([[np.inf, 1.0]]),
        "keep": np.float32(0.5),
    }
    text = json.dumps(jsonable(payload))
    assert "NaN" not in text and "Infinity" not in text
    restored = json.loads(text)
    assert restored["scalar_nan"] is None
    assert restored["scalar_inf"] is None
    assert restored["array"] == [1.0, None, 2.0]
    assert restored["matrix"] == [[None, 1.0]]
    assert restored["keep"] == pytest.approx(0.5)


def test_parse_distance_bins_reports_bad_bounds_with_context() -> None:
    """下限/上限非数字时必须给带上下文的中文错误，而不是原生英文异常。"""
    from eels_edge_analyzer.presets import parse_distance_bins_text

    with pytest.raises(ValueError, match="距离分层上限不是数字"):
        parse_distance_bins_text("E1:0:abc")
    with pytest.raises(ValueError, match="距离分层下限不是数字"):
        parse_distance_bins_text("E1:xx:2.2")
    # 合法输入不受影响，空上限仍表示开区间层。
    bins = parse_distance_bins_text("E1:0:2.2,Bulk:11")
    assert [item.label for item in bins] == ["E1", "Bulk"]
    assert bins[1].high_nm is None
