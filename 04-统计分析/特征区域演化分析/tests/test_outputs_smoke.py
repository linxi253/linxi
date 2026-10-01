# -*- coding: utf-8 -*-
"""全流程端到端冒烟测试：main() 产出全部文件、CSV 8 列、退出码正确。

覆盖 2026-09 P3 重构后的回归防线：
- 面板渲染函数抽取（单图/组合图共用）后全图仍可产出；
- AnalysisConfig / --output / --dpi CLI 参数贯通；
- 退化输入（面积恒定）走"跳过拟合"分支时流程同样完整；
- 新增输出：Segmentation_QA.png（分割质检）与 Processing_Metadata.json
  （机器可读元数据 + 输入 SHA-256）；
- 无效帧/空掩膜帧的 NaN 语义不破坏全流程（图线自然断开而非崩溃）。
"""

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

evolution = importlib.import_module("应力面积统计")

EXPECTED_OUTPUTS = {
    'Fig1A_FeatureAreaEvolution.png', 'Fig1B_ColumnIntensity.png',
    'Fig1C_AreaIntensityCorrelation.png', 'Fig1D_ParallelEvolution.png',
    'Fig1E_PhasePortrait.png', 'Fig1F_StatisticalSummary.png',
    'Fig1_Combined.png',
    'Fig2A_PowerSpectrum.png', 'Fig2B_MovingWindowCorrelation.png',
    'Fig2C_CumulativeDistribution.png', 'Fig2D_StateTransitions.png',
    'Fig2E_ParameterSpaceTrajectory.png', 'Fig2F_ChangePointDetection.png',
    'Fig2_Combined.png',
    'Supplementary_DualAxis.png', 'Supplementary_Boxplot.png',
    'Segmentation_QA.png',
    'Publication_Data.csv', 'Statistical_Summary.md', 'ANALYSIS_REPORT.md',
    'Processing_Metadata.json',
}


def _make_drifting_stack(n_frames, tmp_path, name="stack.tif"):
    """背景 + 半径随时间正弦漂移的中央亮特征。"""
    rng = np.random.default_rng(21)
    yy, xx = np.mgrid[:64, :64]
    frames = []
    for i in range(n_frames):
        img = rng.normal(100.0, 5.0, size=(64, 64))
        r = 10 + 4 * np.sin(i / 6)
        mask = (yy - 32) ** 2 + (xx - 32) ** 2 <= r ** 2
        img[mask] = 200 + rng.normal(0, 5, size=int(mask.sum()))
        frames.append(np.clip(img, 0, 65535).astype(np.uint16))
    path = tmp_path / name
    tifffile.imwrite(str(path), np.stack(frames), photometric="minisblack")
    return str(path)


def _make_constant_frame_stack(n_frames, tmp_path, name="frozen.tif"):
    """冻结视野（逐帧相同）→ 面积恒定 → Fig1C 走"跳过拟合"分支。"""
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[:48, :48]
    frame = rng.normal(100.0, 5.0, size=(48, 48))
    mask = (yy - 24) ** 2 + (xx - 24) ** 2 <= 10 ** 2
    frame[mask] = 200 + rng.normal(0, 5, size=int(mask.sum()))
    stack = np.stack([np.clip(frame, 0, 65535).astype(np.uint16)] * n_frames)
    path = tmp_path / name
    tifffile.imwrite(str(path), stack, photometric="minisblack")
    return str(path)


@pytest.fixture
def no_open_explorer(monkeypatch):
    """阻止 main() 结束时弹出系统文件管理器。"""
    monkeypatch.setattr(evolution, "open_output_directory", lambda *_: None)


class TestFullPipelineSmoke:
    def test_main_produces_all_outputs(self, tmp_path, monkeypatch, no_open_explorer):
        tif = _make_drifting_stack(40, tmp_path)
        out = tmp_path / "out"
        # --output / --dpi 同时验证 CLI 贯通；低 dpi 加速渲染
        monkeypatch.setattr(sys, "argv",
                            ["prog", "--file", tif, "--output", str(out),
                             "--dt", "0.5", "--dpi", "150"])
        rc = evolution.main()
        assert rc == 0
        assert EXPECTED_OUTPUTS.issubset(set(os.listdir(out)))

        data = np.loadtxt(out / "Publication_Data.csv", delimiter=",", comments="#")
        assert data.shape == (40, 8)
        assert np.all(np.isfinite(data))

        report = (out / "ANALYSIS_REPORT.md").read_text(encoding="utf-8")
        assert evolution.__version__ in report

    def test_metadata_json_contains_sha256_and_summary(self, tmp_path, monkeypatch,
                                                       no_open_explorer):
        tif = _make_drifting_stack(12, tmp_path)
        out = tmp_path / "out_json"
        monkeypatch.setattr(sys, "argv",
                            ["prog", "--file", tif, "--output", str(out), "--dpi", "150"])
        assert evolution.main() == 0

        payload = json.loads((out / "Processing_Metadata.json")
                             .read_text(encoding="utf-8"))
        meta, summary = payload["meta"], payload["summary"]
        assert meta["n_frames"] == 12
        assert len(meta["input_sha256"]) == 64
        assert summary["valid_frames"] == 12
        assert summary["pearson_r"] is None or np.isfinite(summary["pearson_r"])
        # qa_samples 含图像数组，不应进入 JSON
        assert "qa_samples" not in meta

    def test_degenerate_constant_area_completes(self, tmp_path, monkeypatch, no_open_explorer):
        path = _make_constant_frame_stack(12, tmp_path)
        out = tmp_path / "out2"
        monkeypatch.setattr(sys, "argv",
                            ["prog", "--file", str(path), "--output", str(out), "--dpi", "150"])
        assert evolution.main() == 0
        assert EXPECTED_OUTPUTS.issubset(set(os.listdir(out)))
        # 常量面积：报告不得写 r = 0.000（应写 undefined）
        report = (out / "ANALYSIS_REPORT.md").read_text(encoding="utf-8")
        assert "undefined" in report

    def test_constant_and_invalid_frames_yield_nan_not_fake_data(self, tmp_path, monkeypatch,
                                                                 no_open_explorer):
        """无效帧语义（旧版伪造 area=0/intensity=1.0 会污染统计）：

        - 常量帧（skimage>=0.25 不抛错）→ 空掩膜：面积=0（真实测量）、
          强度/CI/N_eff=NaN、Frame_Valid=1；
        - threshold_otsu 抛 ValueError（旧版 skimage 行为）→ 无效帧：
          面积/强度/CI/N_eff/阈值全 NaN、Frame_Valid=0。
        """
        rng = np.random.default_rng(6)
        frames = []
        for i in range(8):
            if i == 4:
                frames.append(np.full((48, 48), 100, dtype=np.uint16))  # 常量帧
            else:
                img = rng.normal(100.0, 5.0, size=(48, 48))
                yy, xx = np.mgrid[:48, :48]
                mask = (yy - 24) ** 2 + (xx - 24) ** 2 <= 9 ** 2
                img[mask] = 200
                frames.append(np.clip(img, 0, 65535).astype(np.uint16))
        path = tmp_path / "with_constant.tif"
        tifffile.imwrite(str(path), np.stack(frames), photometric="minisblack")

        result = evolution.process_stem_stack(str(path), 0.2)
        times, areas, intens, ici, aci, thresh, neff, n = result
        assert n == 8
        # 常量帧 → 空掩膜：面积 0、强度 NaN、阈值有限
        assert areas[4] == 0
        assert np.isnan(intens[4]) and np.isnan(neff[4]) and np.isnan(ici[4])
        assert np.isfinite(thresh[4])
        assert np.all(np.isfinite(np.delete(areas, 4)))
        assert np.all(np.isfinite(np.delete(intens, 4)))

        # 模拟旧版 skimage：常量图抛 ValueError → 无效帧全 NaN
        real_otsu = evolution.threshold_otsu

        def raising_otsu(img):
            if np.ptp(img) == 0:
                raise ValueError("simulated: threshold_otsu requires 2 distinct values")
            return real_otsu(img)

        monkeypatch.setattr(evolution, "threshold_otsu", raising_otsu)
        result2 = evolution.process_stem_stack(str(path), 0.2)
        times2, areas2, intens2, _, _, thresh2, neff2, n2 = result2
        assert n2 == 8
        assert np.isnan(areas2[4]) and np.isnan(intens2[4])
        assert np.isnan(thresh2[4]) and np.isnan(neff2[4])
        assert np.all(np.isfinite(np.delete(areas2, 4)))
        # 手动恢复（不能用 monkeypatch.undo()：会连带撤销同 fixture 的
        # no_open_explorer 补丁，导致 main() 真的弹出资源管理器）
        monkeypatch.setattr(evolution, "threshold_otsu", real_otsu)

        out = tmp_path / "out3"
        monkeypatch.setattr(sys, "argv",
                            ["prog", "--file", str(path), "--output", str(out), "--dpi", "150"])
        assert evolution.main() == 0
        data = np.loadtxt(out / "Publication_Data.csv", delimiter=",", comments="#")
        assert data.shape == (8, 8)
        assert data[4, 7] == 1   # 空掩膜帧分割成功 → Frame_Valid = 1
        assert data[4, 1] == 0   # 面积 = 0（真实测量值）
        assert np.isnan(data[4, 2])  # 强度 = NaN（无特征可测）

    def test_min_size_config_threading(self, tmp_path):
        # --min-size 贯通到处理管线：min_size 大于特征面积时应得到空掩膜帧
        tif = _make_drifting_stack(4, tmp_path, name="small.tif")
        big = evolution.AnalysisConfig(min_size=10000)
        meta = {}
        result = evolution.process_stem_stack(tif, 0.2, meta=meta, config=big)
        times, areas, intens, ici, aci, thresh, neff, n = result
        assert n == 4
        assert all(a == 0 for a in areas)
        # 空掩膜：面积=0（真实测量），强度/CI/N_eff=NaN（无特征可测）
        assert np.all(np.isnan(intens))
        assert np.all(np.isnan(neff))
        # 分割本身成功 → 阈值有限、Frame_Valid 可推导为 1
        assert np.all(np.isfinite(thresh))
        assert meta["empty_mask_frames"] == 4
        assert meta["invalid_frames"] == 0
