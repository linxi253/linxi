# -*- coding: utf-8 -*-
"""Publication_Data.csv / Statistical_Summary.md / 科学修正回归测试。

覆盖 2026-09 审查修复：
- CSV 增加强度 95% CI 半宽列（此前 intensity_confidence 是死数据）；
- 处理元数据（背景基准、形态学参数、阈值模式、坏像素统计）随摘要导出；
- Welch 功率谱（去趋势 + 加窗）替代原始 |FFT| 幅度谱；
- 滑动窗口相关对常量窗口返回 NaN（而非误读为"测得零相关"）。
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

evolution = importlib.import_module("应力面积统计")


def _make_frames(n_frames=6):
    """背景 + 中央亮特征，面积随帧漂移便于产生有效统计。"""
    rng = np.random.default_rng(9)
    yy, xx = np.mgrid[:48, :48]
    frames = []
    for i in range(n_frames):
        img = rng.normal(100.0, 5.0, size=(48, 48))
        r = 8 + i
        mask = (yy - 24) ** 2 + (xx - 24) ** 2 <= r ** 2
        img[mask] = 200 + rng.normal(0, 5, size=int(mask.sum()))
        frames.append(np.clip(img, 0, 65535).astype(np.uint16))
    return frames


@pytest.fixture(scope="module")
def pipeline_result(tmp_path_factory):
    import tifffile
    tmp = tmp_path_factory.mktemp("pub")
    path = tmp / "stack.tif"
    tifffile.imwrite(str(path), np.stack(_make_frames()), photometric="minisblack")
    meta = {}
    result = evolution.process_stem_stack(str(path), 0.2, meta=meta)
    return result, meta, tmp


class TestPublicationDataCsv:
    def test_csv_has_eight_columns_with_ci_and_neff(self, pipeline_result):
        result, meta, tmp = pipeline_result
        times, areas, intens, ici, aci, thresh, neff, n = result
        files = evolution.save_publication_data(
            times, areas, intens, thresh, aci, str(tmp),
            shared_threshold=False, intensity_ci=ici, n_eff=neff, meta=meta)
        csv_path = [f for f in files if f.endswith(".csv")][0]

        header = Path(csv_path).read_text(encoding="utf-8").splitlines()[0]
        assert "Intensity_CI95_halfwidth" in header
        assert "N_eff" in header
        assert "Frame_Valid" in header

        data = np.loadtxt(csv_path, delimiter=",", comments="#")
        assert data.shape[1] == 8
        assert data.shape[0] == n
        # CI / N_eff 列逐帧等于处理返回值（CSV 以 %.6f 写出，比较需容差），且非全零
        assert np.allclose(data[:, 5], ici, atol=1e-6)
        assert np.all(data[:, 5] > 0)
        assert np.allclose(data[:, 6], neff, rtol=1e-6)
        # 该数据集全部帧分割成功 → Frame_Valid 全 1
        assert np.all(data[:, 7] == 1)

    def test_summary_contains_processing_parameters(self, pipeline_result):
        result, meta, tmp = pipeline_result
        times, areas, intens, ici, aci, thresh, neff, n = result
        files = evolution.save_publication_data(
            times, areas, intens, thresh, aci, str(tmp),
            shared_threshold=False, intensity_ci=ici, n_eff=neff, meta=meta)
        md = [f for f in files if f.endswith(".md")][0]
        text = Path(md).read_text(encoding="utf-8")
        assert "## Processing Parameters" in text
        assert "Background I0" in text
        assert "remove_small_objects(min_size=4)" in text
        assert "per-frame Otsu" in text
        # p 值自相关警示与强度 CI 口径警示
        assert "时间自相关" in text
        assert "I₀ 估计误差" in text or "I0 估计误差" in text

    def test_constant_series_reports_undefined_correlation(self, pipeline_result):
        """常量序列的相关必须写 undefined，不得写 r=0.000（误读为测得零相关）。"""
        result, meta, tmp = pipeline_result
        times, areas, intens, ici, aci, thresh, neff, n = result
        constant_intens = np.full_like(intens, 1.5)
        sub = tmp / "constant_case"
        sub.mkdir(exist_ok=True)
        files = evolution.save_publication_data(
            times, areas, constant_intens, thresh, aci, str(sub))
        md = [f for f in files if f.endswith(".md")][0]
        text = Path(md).read_text(encoding="utf-8")
        assert "undefined" in text
        assert "r = 0.000" not in text


class TestProcessingMeta:
    def test_meta_populated(self, pipeline_result):
        result, meta, _ = pipeline_result
        times, areas, intens, ici, aci, thresh, neff, n = result
        assert meta["n_frames"] == n
        assert meta["bg_intensity"] > 0
        assert meta["min_size"] == 4
        assert meta["closing_structure"] == "3x3"
        assert "Otsu" in meta["threshold_mode"]
        assert meta["version"] == evolution.__version__
        assert meta["invalid_frames"] == 0
        assert meta["empty_mask_frames"] == 0


class TestComputePowerSpectrum:
    def test_recovers_known_frequency(self):
        fs_dt = 0.1
        t = np.arange(0, 100, fs_dt)
        signal = np.sin(2 * np.pi * 0.5 * t) + 0.2 * np.sin(2 * np.pi * 2.0 * t)
        freqs, psd = evolution.compute_power_spectrum(signal, fs_dt)
        assert np.all(np.isfinite(freqs)) and np.all(np.isfinite(psd))
        assert np.all(psd[1:] > 0)
        # 主峰应落在已知主频 0.5 Hz 附近
        main_freq = freqs[np.argmax(psd[1:]) + 1]
        assert abs(main_freq - 0.5) < 0.1

    def test_linear_trend_suppressed(self):
        # 强线性趋势 + 弱周期信号：Welch 去趋势后低频端不应被趋势主导
        t = np.arange(0, 100, 0.1)
        signal = 0.5 * t + np.sin(2 * np.pi * 1.0 * t)
        freqs, psd = evolution.compute_power_spectrum(signal, 0.1)
        main_freq = freqs[np.argmax(psd[1:]) + 1]
        assert abs(main_freq - 1.0) < 0.2

    def test_invalid_dt_raises(self):
        with pytest.raises(ValueError):
            evolution.compute_power_spectrum(np.ones(100), 0.0)


class TestRollingCorrelation:
    def test_constant_window_returns_nan(self):
        areas = np.full(20, 5.0)
        intens = np.arange(20.0)
        corr = evolution.rolling_correlation(areas, intens, 5)
        assert np.all(np.isnan(corr))

    def test_window_with_nan_returns_nan(self):
        # 无效帧（NaN）混入窗口 → 该窗口相关为 NaN，不参与统计
        areas = np.arange(20.0)
        intens = areas * 2.0
        intens[3] = np.nan
        corr = evolution.rolling_correlation(areas, intens, 5)
        assert np.isnan(corr[0]) and np.isnan(corr[3])
        assert not np.isnan(corr[10])

    def test_perfect_correlation(self):
        rng = np.random.default_rng(0)
        areas = rng.normal(0, 1, 20)
        corr = evolution.rolling_correlation(areas, areas, 5)
        assert np.allclose(corr, 1.0)
