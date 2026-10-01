# -*- coding: utf-8 -*-
"""process_stem_stack 阈值处理端到端回归测试。

覆盖 2026-09-05 修复：
- 每帧实际使用的分割阈值随结果返回（可复现）；
- --shared-threshold 共享阈值模式使用逐帧 Otsu 的中位数，
  消除阈值漂移混入面积序列的噪声。
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

evolution = importlib.import_module("应力面积统计")


def _write_stack(tmp_path, frames):
    import tifffile
    path = tmp_path / "stack.tif"
    tifffile.imwrite(str(path), np.stack(frames), photometric="minisblack")
    return str(path)


def _make_frames(n_frames=5, backgrounds=(100.0,)):
    """背景 + 中央亮特征（含噪声），可选逐帧背景漂移制造阈值漂移。"""
    rng = np.random.default_rng(7)
    frames = []
    for i in range(n_frames):
        bg = backgrounds[i % len(backgrounds)]
        img = rng.normal(bg, 5.0, size=(48, 48))
        img[16:32, 16:32] = bg + 100 + rng.normal(0, 5.0, size=(16, 16))
        frames.append(np.clip(img, 0, 65535).astype(np.uint16))
    return frames


def _run(path, shared):
    return evolution.process_stem_stack(path, time_interval=0.2,
                                                   shared_threshold=shared)


class TestPerFrameThresholds:
    def test_thresholds_recorded_and_finite(self, tmp_path):
        path = _write_stack(tmp_path, _make_frames())
        times, areas, intens, ici, aci, thresh, neff, n = _run(path, shared=False)
        assert n == 5
        assert len(times) == len(areas) == len(thresh) == 5
        assert np.all(np.isfinite(thresh))
        # 归一化后背景≈1、特征≈2，Otsu 阈值应落在两者之间
        assert np.all(thresh > 1.0) and np.all(thresh < 2.0)

    def test_shared_threshold_equals_median_of_per_frame(self, tmp_path):
        # 背景逐帧漂移 → 逐帧 Otsu 阈值漂移，共享模式应取其中位数
        frames = _make_frames(n_frames=6, backgrounds=(100.0, 140.0))
        path = _write_stack(tmp_path, frames)
        *_, thresh_per_frame, _, _ = _run(path, shared=False)
        *_, thresh_shared, _, _ = _run(path, shared=True)
        expected = np.median(thresh_per_frame)
        assert np.allclose(thresh_shared, expected)
        # 共享模式下每帧记录的就是实际使用的统一阈值
        assert np.all(thresh_shared == thresh_shared[0])

    def test_shared_mode_threshold_stabler_than_per_frame(self, tmp_path):
        frames = _make_frames(n_frames=6, backgrounds=(100.0, 150.0))
        path = _write_stack(tmp_path, frames)
        *_, thresh_per_frame, _, _ = _run(path, shared=False)
        *_, thresh_shared, _, _ = _run(path, shared=True)
        assert np.std(thresh_shared) < np.std(thresh_per_frame)
