from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from eels_edge_analyzer import dm4io
from eels_edge_analyzer.dm4io import HashWatch, _coords_from_calibration
from eels_edge_analyzer.fitting import fit_all_cu_models, injection_recovery
from eels_edge_analyzer.models import (
    AnalysisConfig,
    DatasetInfo,
    DistanceBin,
    LoadedDataset,
    ReferenceSpec,
)
from eels_edge_analyzer.pipeline import run_analysis
from eels_edge_analyzer.processing import (
    fourier_ratio_deconvolution,
    inward_distance_nm,
    masks_for_distance_bins,
    trace_surface,
)


def _gaussian(energy: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((energy - center) / width) ** 2)


def test_coords_from_calibration_matches_ncempy_formula() -> None:
    assert np.allclose(_coords_from_calibration(1.0, 0.0, 5), np.arange(5, dtype=float))
    expected = np.round(
        np.linspace(0, 0.25 * 3, 4) + round(-1.0 * 2.0 * 0.25, ndigits=4), decimals=4
    )
    assert np.allclose(_coords_from_calibration(0.25, 2.0, 4), expected)
    # 与 dmReader 一致：长度为 1 的轴返回标量 0。
    assert _coords_from_calibration(1.0, 5.0, 1) == 0
    # 负 origin 等价于能量轴整体右移。
    shifted = _coords_from_calibration(0.5, -4.0, 6)
    assert np.isclose(shifted[0], round(-1.0 * -4.0 * 0.5, ndigits=4))


def test_three_component_mlls_recovers_known_mixture() -> None:
    energy = np.linspace(925.0, 970.0, 301)
    refs = np.column_stack(
        (
            _gaussian(energy, 933.0, 1.5) + 0.35 * _gaussian(energy, 953.0, 2.3),
            _gaussian(energy, 932.3, 1.3) + 0.55 * _gaussian(energy, 952.5, 2.1),
            0.8 * _gaussian(energy, 935.5, 2.0) + _gaussian(energy, 954.5, 2.8),
        )
    )
    refs /= np.trapezoid(refs, energy, axis=0)[None, :]
    truth = np.array((0.64, 0.20, 0.16))
    spectrum = refs @ truth
    fit = fit_all_cu_models(spectrum, refs)["M012_Cu0_Cu1_Cu2"]
    assert np.allclose(fit.fractions, truth, atol=1e-4)
    assert fit.r2 > 0.99999


def test_cu1_injection_has_increasing_selection_rate() -> None:
    energy = np.linspace(925.0, 970.0, 151)
    refs = np.column_stack(
        (
            _gaussian(energy, 933.0, 1.3),
            _gaussian(energy, 934.5, 1.3),
            _gaussian(energy, 954.0, 2.0),
        )
    )
    refs /= np.trapezoid(refs, energy, axis=0)[None, :]
    baseline = fit_all_cu_models(refs @ np.array((0.7, 0.0, 0.3)), refs)["M02_Cu0_Cu2"]
    rows, _ = injection_recovery(
        baseline,
        refs,
        energy,
        simulations=40,
        rng=np.random.default_rng(3),
        bic_threshold=2.0,
        fractions=(0.05, 0.40),
    )
    assert rows[1]["recovered_cu1_median"] > rows[0]["recovered_cu1_median"]
    assert (
        rows[1]["cu1_selection_rate_delta_bic_ge_threshold"]
        >= rows[0]["cu1_selection_rate_delta_bic_ge_threshold"]
    )


def test_cu1_injection_is_invariant_to_baseline_intensity_scale() -> None:
    energy = np.linspace(925.0, 970.0, 151)
    refs = np.column_stack(
        (
            _gaussian(energy, 933.0, 1.3),
            _gaussian(energy, 934.5, 1.3),
            _gaussian(energy, 954.0, 2.0),
        )
    )
    refs /= np.trapezoid(refs, energy, axis=0)[None, :]
    spectrum = refs @ np.array((0.72, 0.0, 0.28))
    unit_baseline = fit_all_cu_models(spectrum, refs)["M02_Cu0_Cu2"]
    scaled_baseline = fit_all_cu_models(750.0 * spectrum, refs)["M02_Cu0_Cu2"]
    kwargs = {
        "references": refs,
        "energy_ev": energy,
        "simulations": 60,
        "bic_threshold": 2.0,
        "fractions": (0.10, 0.40),
    }
    unit_rows, unit_limit = injection_recovery(unit_baseline, rng=np.random.default_rng(11), **kwargs)
    scaled_rows, scaled_limit = injection_recovery(scaled_baseline, rng=np.random.default_rng(11), **kwargs)
    keys = (
        "recovered_cu1_p2_5",
        "recovered_cu1_median",
        "recovered_cu1_p97_5",
        "cu1_selection_rate_delta_bic_ge_threshold",
    )
    unit_values = np.array([[row[key] for key in keys] for row in unit_rows])
    scaled_values = np.array([[row[key] for key in keys] for row in scaled_rows])
    assert np.allclose(unit_values, scaled_values, atol=1e-10)
    assert unit_limit == scaled_limit
    assert scaled_rows[-1]["recovered_cu1_median"] > 0.30


def test_surface_trace_and_distance_bins() -> None:
    ny, nx = 30, 40
    expected = np.rint(4 + 1.5 * np.sin(np.linspace(0, 2 * np.pi, nx))).astype(int)
    image = np.zeros((ny, nx), dtype=float)
    for column, row in enumerate(expected):
        image[row:, column] = 100.0
    image += np.random.default_rng(1).normal(0, 0.8, image.shape)
    boundary = trace_surface(image, "top", max_depth=10, smoothness_penalty=0.02)
    assert np.mean(np.abs(boundary.boundary - expected)) <= 1.0
    assert trace_surface(image, "auto", max_depth=10, smoothness_penalty=0.02).orientation == "top"
    distance = inward_distance_nm(boundary, image.shape, y_step_nm=1.0, x_step_nm=1.0)
    masks = masks_for_distance_bins(
        distance,
        (DistanceBin("E1", 0.0, 2.0), DistanceBin("Bulk", 2.0, None)),
    )
    assert np.all(~(masks["E1"] & masks["Bulk"]))
    assert masks["E1"].sum() > 0
    assert masks["Bulk"].sum() > 0


def test_config_rejects_ambiguous_distance_bins() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        source = root / "input.dm4"
        source.touch()
        references = []
        for state in range(3):
            path = root / f"cu{state}.dm4"
            path.touch()
            references.append(ReferenceSpec(f"Cu{state}", state, path))
        invalid_sets = (
            (DistanceBin("Bulk", 0.0, None), DistanceBin("E1", 1.0, 2.0)),
            (DistanceBin("E1", 0.0, 2.0), DistanceBin("E1", 2.0, None)),
        )
        for bins in invalid_sets:
            config = AnalysisConfig(
                input_path=source,
                output_dir=root / "output",
                references=tuple(references),
                distance_bins=bins,
                along_surface_segment_nm=5.0,
            )
            try:
                config.validate()
            except ValueError:
                pass
            else:
                raise AssertionError("无上限层位置或重复标签未被拒绝。")


def test_config_rejects_invalid_injection_grid() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        source = root / "input.dm4"
        source.touch()
        references = []
        for state in range(3):
            path = root / f"cu{state}.dm4"
            path.touch()
            references.append(ReferenceSpec(f"Cu{state}", state, path))
        invalid_settings = (
            {"injection_fractions": (0.20, 0.10)},
            {"injection_fractions": (0.10, 0.10)},
            {"injection_fractions": (0.0, 0.20)},
            {"injection_residual_block_channels": 0},
        )
        for settings in invalid_settings:
            config = AnalysisConfig(
                input_path=source,
                output_dir=root / "output",
                references=tuple(references),
                along_surface_segment_nm=5.0,
                **settings,
            )
            try:
                config.validate()
            except ValueError:
                pass
            else:
                raise AssertionError("无效的注入网格或残差块宽度未被拒绝。")


def test_config_validates_windows_and_scalars() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        source = root / "input.dm4"
        source.touch()
        references = tuple(
            ReferenceSpec(f"Cu{state}", state, root / f"cu{state}.dm4") for state in range(3)
        )
        for state in range(3):
            (root / f"cu{state}.dm4").touch()

        # 默认值组合必须通过校验（保证新增规则没有误伤合法设置）。
        # along_surface_segment_nm 是唯一不设数值默认的样品/预设绑定参数，
        # 这里显式补一个中性值，其余全部走默认。
        AnalysisConfig(input_path=source, output_dir=root / "out", references=references,
                       along_surface_segment_nm=5.0).validate()

        invalid_settings = (
            {"savgol_polyorder": 0},
            {"surface_smoothness_penalty": -0.01},
            {"zlp_window_ev": (3.0, -3.0)},
            {"zlp_window_ev": (-3.0, 3.0, 5.0)},
            # 基线整体低于 ZLP 窗口的要求：默认 ZLP 下界 -3，基线上界 -1 违反。
            {"low_loss_baseline_ev": (-2.0, -1.0)},
            {"sensitivity_background_mins_ev": (0.0,)},
            {"sensitivity_background_mins_ev": (930.0,)},
        )
        for settings in invalid_settings:
            config = AnalysisConfig(
                input_path=source,
                output_dir=root / "out",
                references=references,
                along_surface_segment_nm=5.0,
                **settings,
            )
            try:
                config.validate()
            except ValueError:
                continue
            raise AssertionError(f"无效设置未被拒绝: {settings}")

        # 输出位置已存在且是文件时必须报错。
        existing_file = root / "occupied"
        existing_file.touch()
        config = AnalysisConfig(
            input_path=source,
            output_dir=existing_file,
            references=references,
            along_surface_segment_nm=5.0,
        )
        try:
            config.validate()
        except ValueError:
            pass
        else:
            raise AssertionError("已存在的文件路径作为输出目录未被拒绝。")


def test_fourier_ratio_deconvolution_is_finite_and_shape_preserving() -> None:
    energy = np.linspace(-10.0, 20.0, 96)
    zlp = _gaussian(energy, 0.0, 0.5)
    low = np.tile(zlp[:, None, None], (1, 3, 4)) * 5000
    high = np.tile((_gaussian(energy, 8.0, 1.3) * 200)[:, None, None], (1, 3, 4))
    restored, diagnostics = fourier_ratio_deconvolution(
        high.astype(np.float32),
        low.astype(np.float32),
        energy,
        regularization=0.003,
        baseline_range_ev=(-10.0, -5.0),
        zlp_window_ev=(-2.0, 2.0),
    )
    assert restored.shape == high.shape
    assert np.isfinite(restored).all()
    assert diagnostics["fft_length"] >= 2048


def _pixel_loop_deconvolution_reference(
    high_loss: np.ndarray,
    low_loss: np.ndarray,
    low_energy_ev: np.ndarray,
    regularization: float,
    baseline_range_ev: tuple[float, float],
    zlp_window_ev: tuple[float, float],
) -> np.ndarray:
    """旧版逐像素实现的参考副本，用于验证批量 FFT 路径的数值等价性。"""

    baseline_mask = (low_energy_ev >= baseline_range_ev[0]) & (low_energy_ev <= baseline_range_ev[1])
    zero_window = (low_energy_ev >= zlp_window_ev[0]) & (low_energy_ev <= zlp_window_ev[1])
    low_corrected = np.maximum(
        low_loss - np.median(low_loss[baseline_mask], axis=0, keepdims=True),
        0.0,
    )
    zero_index = int(np.argmin(np.abs(low_energy_ev)))
    zero_candidates = np.flatnonzero(zero_window)
    n_energy, ny, nx = high_loss.shape
    n_fft = 1 << max(1, int(max(2 * n_energy, 2048) - 1).bit_length())
    output = np.empty_like(high_loss, dtype=np.float32)
    for row in range(ny):
        for column in range(nx):
            low = low_corrected[:, row, column].astype(float, copy=True)
            peak = int(zero_candidates[np.argmax(low[zero_window])])
            channel_shift = zero_index - peak
            if channel_shift:
                low = np.roll(low, channel_shift)
            kernel = np.zeros(n_fft, dtype=float)
            kernel[: n_energy - zero_index] = low[zero_index:]
            kernel[n_fft - zero_index :] = low[:zero_index]
            kernel /= float(np.sum(kernel))
            zero = low.copy()
            zero[~zero_window] = 0.0
            zero_kernel = np.zeros(n_fft, dtype=float)
            zero_kernel[: n_energy - zero_index] = zero[zero_index:]
            zero_kernel[n_fft - zero_index :] = zero[:zero_index]
            zero_kernel /= float(np.sum(zero_kernel))
            padded_high = np.zeros(n_fft, dtype=float)
            padded_high[:n_energy] = high_loss[:, row, column]
            low_fft = np.fft.rfft(kernel)
            zero_fft = np.fft.rfft(zero_kernel)
            high_fft = np.fft.rfft(padded_high)
            wiener_ratio = zero_fft * np.conj(low_fft) / (np.abs(low_fft) ** 2 + float(regularization))
            output[:, row, column] = np.fft.irfft(high_fft * wiener_ratio, n_fft)[:n_energy]
    return output


def _synthetic_dual_eels(seed: int = 7, ny: int = 4, nx: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    energy = np.linspace(-10.0, 25.0, 64)
    zlp = _gaussian(energy, 0.0, 0.5) * 8000
    core = _gaussian(energy, 8.0, 1.3) * 200 + _gaussian(energy, 14.0, 2.2) * 60
    low = np.zeros((energy.size, ny, nx), dtype=np.float32)
    high = np.zeros_like(low)
    for row in range(ny):
        for column in range(nx):
            shift = int(rng.integers(-3, 4))
            low[:, row, column] = np.roll(zlp, shift) + rng.normal(0, 2.0, energy.size)
            high[:, row, column] = np.roll(core, shift) + rng.normal(0, 0.5, energy.size)
    return high, low, energy


def test_fourier_ratio_deconvolution_matches_pixel_reference() -> None:
    high, low, energy = _synthetic_dual_eels()
    kwargs = {
        "regularization": 0.003,
        "baseline_range_ev": (-10.0, -5.0),
        "zlp_window_ev": (-2.0, 2.0),
    }
    batched, diagnostics = fourier_ratio_deconvolution(high, low, energy, **kwargs)
    reference = _pixel_loop_deconvolution_reference(high, low, energy, **kwargs)
    tolerance = 1e-6 * float(np.max(np.abs(reference)))
    assert np.allclose(batched, reference, rtol=1e-6, atol=tolerance)
    assert diagnostics["selected_pixel_fraction"] == 1.0
    assert diagnostics["fft_length"] >= 2048


def test_deconvolution_pixel_subset_matches_full() -> None:
    high, low, energy = _synthetic_dual_eels(seed=11, ny=4, nx=6)
    kwargs = {
        "regularization": 0.003,
        "baseline_range_ev": (-10.0, -5.0),
        "zlp_window_ev": (-2.0, 2.0),
    }
    full, _ = fourier_ratio_deconvolution(high, low, energy, **kwargs)
    subset_mask = np.zeros((4, 6), dtype=bool)
    subset_mask[:, ::2] = True
    indices = np.flatnonzero(subset_mask.ravel())
    subset, diagnostics = fourier_ratio_deconvolution(
        high, low, energy, **kwargs, pixel_indices=indices
    )
    expected_fraction = indices.size / subset_mask.size
    assert abs(diagnostics["selected_pixel_fraction"] - expected_fraction) < 1e-12
    flat_subset = subset.reshape(energy.size, -1)
    flat_full = full.reshape(energy.size, -1)
    tolerance = 1e-6 * float(np.max(np.abs(full)))
    assert np.allclose(flat_subset[:, indices], flat_full[:, indices], rtol=1e-6, atol=tolerance)
    assert np.isnan(flat_subset[:, ~subset_mask.ravel()]).all()


def test_hash_watch_avoids_redundant_hashing(tmp_path, monkeypatch) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"payload")
    watch = HashWatch(path)
    first = watch.current()
    assert first == dm4io.file_sha256(path)
    calls = {"count": 0}
    real_sha = dm4io.file_sha256

    def counting_sha(target, block_size=1024 * 1024):
        calls["count"] += 1
        return real_sha(target, block_size)

    monkeypatch.setattr(dm4io, "file_sha256", counting_sha)
    # stat 未变化：直接复用缓存哈希，不做全量读取。
    assert watch.current() == first
    assert calls["count"] == 0
    # mtime 变化：重算一次，结果一致。
    status = path.stat()
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns + 1_000_000))
    assert watch.current() == first
    assert calls["count"] == 1
    # 内容变化：重算且哈希不同。
    path.write_bytes(b"payload-v2")
    assert watch.current() != first
    assert calls["count"] == 2


# ---------------------------------------------------------------------------
# 管线端到端冒烟测试：mock DM4 读取，真实跑 run_analysis + export_artifacts。
# ---------------------------------------------------------------------------

def _synthetic_si_dataset(
    index: int,
    energy: np.ndarray,
    spatial_shape: tuple[int, int],
    spectrum_of_row,
    unit: str = "nm",
    step: float = 1.1,
) -> LoadedDataset:
    data = np.empty((energy.size, *spatial_shape), dtype=np.float32)
    for row in range(spatial_shape[0]):
        data[:, row, :] = spectrum_of_row(row)[:, None]
    coords = (
        energy,
        np.arange(spatial_shape[0], dtype=float) * step,
        np.arange(spatial_shape[1], dtype=float) * step,
    )
    units = ("eV", unit, unit)
    titles = {0: "survey", 1: "low_loss", 2: "high_loss"}
    return LoadedDataset(index=index, data=data, coords=coords, units=units, title=titles.get(index, f"ds{index}"))


def test_pipeline_end_to_end_on_synthetic_si(tmp_path, monkeypatch) -> None:
    rng = np.random.default_rng(42)
    energy = np.linspace(-30.0, 1000.0, 520)
    ny, nx = 24, 20
    boundary_row = 2

    def reference_profiles(center_l3: float, width_l3: float, l3_to_l2: float, l2_shift: float):
        def profile(ev: np.ndarray) -> np.ndarray:
            l3 = _gaussian(ev, center_l3, width_l3)
            l2 = l3_to_l2 * _gaussian(ev, center_l3 + l2_shift, width_l3 * 1.3)
            return l3 + l2

        return profile

    cu0_shape = reference_profiles(933.0, 1.5, 0.5, 20.0)
    cu1_shape = reference_profiles(932.2, 1.3, 0.7, 19.4)
    cu2_shape = reference_profiles(935.5, 2.0, 1.1, 20.4)

    def make_reference_csv(path, shape):
        intensity = shape(energy[(energy >= 900) & (energy <= 990)])
        grid = energy[(energy >= 900) & (energy <= 990)]
        lines = ["energy_ev,intensity"]
        for ev, value in zip(grid, intensity, strict=True):
            lines.append(f"{ev:.3f},{value:.6f}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    reference_dir = tmp_path / "references"
    reference_dir.mkdir()
    make_reference_csv(reference_dir / "cu0.csv", cu0_shape)
    make_reference_csv(reference_dir / "cu1.csv", cu1_shape)
    make_reference_csv(reference_dir / "cu2.csv", cu2_shape)
    references = (
        ReferenceSpec("Cu0", 0, reference_dir / "cu0.csv"),
        ReferenceSpec("Cu1", 1, reference_dir / "cu1.csv"),
        ReferenceSpec("Cu2", 2, reference_dir / "cu2.csv"),
    )

    def power_law(ev: np.ndarray) -> np.ndarray:
        return 5e9 * np.maximum(ev, 1.0) ** (-3.0)

    def high_spectrum_of_row(row: int) -> np.ndarray:
        distance = max(0, row - boundary_row)
        cu1_weight = 0.30 if distance <= 3 else (0.05 if distance <= 8 else 0.0)
        weights = np.array((0.6 - cu1_weight / 2, cu1_weight, 0.4 - cu1_weight / 2))
        if row <= boundary_row:
            mixture = np.zeros_like(energy)
        else:
            mixture = weights[0] * cu0_shape(energy) + weights[1] * cu1_shape(energy) + weights[2] * cu2_shape(energy)
        # 真实高损谱在 ~10 eV 以下被漂移管/截止阀抑制；用平滑截止代替硬截断，
        # 避免巨大的低能阶跃在去卷积中产生振铃。
        smooth_cut = 1.0 / (1.0 + np.exp(-(energy - 12.0) / 2.0))
        spectrum = power_law(np.maximum(energy, 1.0)) * smooth_cut + 200.0 * mixture
        spectrum[energy < 0] = 0.0
        return spectrum + rng.normal(0, 0.05, energy.size)

    def low_spectrum_of_row(row: int) -> np.ndarray:
        # 物理量级：ZLP 必须远强于 5 eV 以上的损失尾部，否则去卷积核的
        # 质量不在零损峰上，Fourier-ratio 比率会失效。
        if row <= boundary_row:
            spectrum = 5e7 * _gaussian(energy, 0.0, 0.5)
        else:
            spectrum = 4e7 * _gaussian(energy, 0.0, 0.5) + 3e4 * np.power(
                np.maximum(energy, 1.0), -1.5
            ) * (energy > 5)
        return spectrum + rng.normal(0, 0.02, energy.size)

    survey = np.zeros((ny, nx), dtype=np.float32)
    survey[boundary_row + 1 :, :] = 100.0
    survey += rng.normal(0, 0.3, survey.shape).astype(np.float32)

    datasets = {
        0: LoadedDataset(
            index=0, data=survey, coords=(np.arange(ny, dtype=float), np.arange(nx, dtype=float)),
            units=("nm", "nm"), title="survey",
        ),
        1: _synthetic_si_dataset(1, energy, (ny, nx), low_spectrum_of_row),
        2: _synthetic_si_dataset(2, energy, (ny, nx), high_spectrum_of_row),
    }

    source = tmp_path / "synthetic.dm4"
    source.write_bytes(b"synthetic-dual-eels-fixture")

    import eels_edge_analyzer.pipeline as pipeline_module

    def fake_read_dataset(path, index):
        return datasets[index]

    def fake_inspect(path, max_datasets=None):
        return [
            DatasetInfo(0, (ny, nx), "float32", None, None, (ny, nx), "survey_image", "survey"),
            DatasetInfo(1, (energy.size, ny, nx), "float32", float(energy.min()), float(energy.max()), (ny, nx), "low_loss_si", "low"),
            DatasetInfo(2, (energy.size, ny, nx), "float32", float(energy.min()), float(energy.max()), (ny, nx), "core_loss_si", "high"),
        ]

    monkeypatch.setattr(pipeline_module, "read_dataset", fake_read_dataset)
    monkeypatch.setattr(pipeline_module, "inspect_dm4", fake_inspect)
    monkeypatch.setattr(pipeline_module, "all_tags", lambda path: {})

    output = tmp_path / "results"
    config = AnalysisConfig(
        input_path=source,
        output_dir=output,
        references=references,
        survey_dataset=0,
        low_loss_dataset=1,
        high_loss_dataset=2,
        bootstrap_resamples=50,
        injection_simulations=0,
        along_surface_segment_nm=5.0,
        # 距离分层不再有 Cu 专用默认：需要表层剖面时必须显式提供
        #（此处沿用内置 Cu 预设的分箱，测试表层/内部的 Cu1 权重差异）。
        distance_bins=(
            DistanceBin("E1", 0.0, 2.2),
            DistanceBin("E2", 2.2, 4.4),
            DistanceBin("E3", 4.4, 6.6),
            DistanceBin("E4", 6.6, 11.0),
            DistanceBin("Bulk", 11.0, None),
        ),
    )
    artifacts = run_analysis(config)
    paths = dm4io  # noqa: F841 - 占位避免误导，真实检查见下方 reporting 导入
    from eels_edge_analyzer.reporting import export_artifacts

    written = export_artifacts(artifacts, config)

    import json

    summary = json.loads(Path(written["summary"]).read_text(encoding="utf-8"))
    assert summary["surface_registration"]["orientation"] in {"top", "bottom", "left", "right"}
    assert summary["surface_registration"]["selection"] is not None
    assert len(summary["surface_registration"]["selection"]) == 4
    assert summary["surface_registration"]["auto_orientation_confident"] is True
    assert summary["application"]["timestamp_utc"]
    assert summary["environment"]["numpy"]
    assert np.isfinite(summary["thickness_quality"]["median_t_over_lambda_sample"])
    assert summary["thickness_quality"]["warning_threshold"] > 0
    assert summary["along_surface_block_width"]["configured_columns"] == 5
    assert summary["along_surface_block_width"]["suggested_from_autocorrelation"]
    assert summary["edge_minus_bulk_bootstrap_method"] in {
        "shared_block_positions",
        "independent_resampling",
    }
    assert summary["global_shift_scan_limit_ev"] > 0
    assert summary["input"]["spatial_step_details"]["y"]["unit_kind"] == "nm"
    assert summary["input"]["spatial_step_details"]["y"]["step_nm"] == 1.1
    assert summary["distance_binning"]["total_si_pixels"] == ny * nx
    assert summary["distance_binning"]["unassigned_fraction"] < 0.5
    assert summary["low_loss"]["total_integral_max_ev"] == 200.0
    assert summary["detection_limit_cu1_fraction"] is None  # 注入恢复已关闭

    profile = {row["bin"]: row for row in summary["profile"]}
    assert set(profile) == {"E1", "E2", "E3", "E4", "Bulk"}
    for row in summary["profile"]:
        for label in ("Cu0", "Cu1", "Cu2"):
            assert np.isfinite(row[f"{label}_fraction"])
        assert abs(sum(row[f"{label}_fraction"] for label in ("Cu0", "Cu1", "Cu2")) - 1.0) < 1e-6
    # 表层注入了 Cu1 信号，内部以 Cu0 为主：E1 的 Cu1 权重应高于 Bulk。
    assert profile["E1"]["Cu1_fraction"] > profile["Bulk"]["Cu1_fraction"] + 0.05
    assert profile["E1"]["Cu0_fraction"] < profile["Bulk"]["Cu0_fraction"]

    for key in (
        "report", "summary", "profile", "models", "sensitivity", "segments", "detection",
        "boundary", "shift_scan", "arrays", "integrity", "run_log",
        "references_plot", "region_fits_plot",
    ):
        assert Path(written[key]).is_file(), key
    pngs = [value for key, value in written.items() if str(value).endswith(".png")]
    assert len(pngs) == 10
    assert (output / "edge_analysis_report.md").is_file()
    run_log_text = (output / "run_log.txt").read_text(encoding="utf-8")
    assert "EELS Edge Analyzer run log" in run_log_text
    assert "environment" in run_log_text

    # 覆盖守卫：同目录再次导出必须显式允许覆盖。
    try:
        export_artifacts(artifacts, config)
    except RuntimeError as exc:
        assert "既往分析结果" in str(exc)
    else:
        raise AssertionError("对包含既往结果的输出目录未拒绝静默覆盖。")
    export_artifacts(artifacts, config, overwrite=True)
