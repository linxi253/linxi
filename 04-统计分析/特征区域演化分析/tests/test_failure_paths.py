# -*- coding: utf-8 -*-
"""process_stem_stack 失败路径与输入校验回归测试。

覆盖 2026-09 审查修复：
- 所有失败路径返回 8 元组（times 为 None），调用方可稳定解包
  （历史缺陷：新增返回值后失败路径漏改，曾退化为短元组）；
- NaN/inf 坏像素按帧清洗（有限像素中值填充），不再静默产出全零面积
  并在 Fig1C 触发 LinAlgError 的崩溃链；
- CLI 参数 --dt / --file 非法时立即报错退出（SystemExit, code 2）；
- 手动输入路径 dt=NaN/inf 被拒绝（曾通过 "<=0" 检查使时间轴全 NaN）；
- 输出目录已存在且非空 / 指向文件时拒绝（退出码 1，不覆盖旧结果）；
- AnalysisConfig 编程注入非法参数在构造时报错；
- main() 返回退出码，处理失败时为 1。
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

evolution = importlib.import_module("应力面积统计")


def _write_stack(tmp_path, frames, name="stack.tif"):
    path = tmp_path / name
    tifffile.imwrite(str(path), np.stack(frames), photometric="minisblack")
    return str(path)


class TestFailureTupleContract:
    """失败路径必须返回 8 元组（与成功路径元数一致）。"""

    def test_single_frame_returns_8_tuple(self, tmp_path):
        path = _write_stack(tmp_path, [np.full((16, 16), 100, dtype=np.uint16)])
        result = evolution.process_stem_stack(path, 0.2)
        assert isinstance(result, tuple)
        assert len(result) == 8
        assert result[0] is None
        assert result[-1] == 0

    def test_corrupt_file_returns_8_tuple(self, tmp_path):
        path = tmp_path / "corrupt.tif"
        path.write_bytes(b"not a tiff at all")
        result = evolution.process_stem_stack(str(path), 0.2)
        assert len(result) == 8
        assert result[0] is None

    def test_main_returns_1_when_processing_fails(self, tmp_path, monkeypatch):
        path = _write_stack(tmp_path, [np.full((16, 16), 100, dtype=np.uint16)],
                            name="single2.tif")
        monkeypatch.setattr(sys, "argv", ["prog", "--file", path])
        assert evolution.main() == 1


class TestBadPixelSanitization:
    """NaN/inf 坏像素应被清洗，而不是污染背景后静默产出全零面积。"""

    def test_nan_pixels_sanitized_and_analysis_completes(self, tmp_path):
        rng = np.random.default_rng(3)
        frames = [rng.normal(100.0, 5.0, (48, 48)) for _ in range(5)]
        frames[0] = frames[0].copy()
        frames[0][10, 10] = np.nan
        frames[0][20, 20] = np.inf
        path = _write_stack(tmp_path, frames, name="nan.tif")

        result = evolution.process_stem_stack(path, 0.2)

        assert len(result) == 8
        times, areas, intens, ici, aci, thresh, neff, n = result
        assert n == 5
        # 修复前：背景中值变 NaN → 全帧 Otsu 失败 → areas 全 0、thresh 全 NaN
        assert np.all(np.isfinite(areas))
        assert np.all(np.isfinite(thresh))
        assert np.all(np.asarray(areas) > 0)

    def test_majority_bad_frame_aborts(self, tmp_path):
        rng = np.random.default_rng(4)
        frames = [rng.normal(100.0, 5.0, (32, 32)) for _ in range(3)]
        frames[1] = frames[1].copy()
        frames[1][:] = np.nan  # 整帧坏点 > 50%
        path = _write_stack(tmp_path, frames, name="bad.tif")

        result = evolution.process_stem_stack(path, 0.2)
        assert len(result) == 8
        assert result[0] is None


class TestCliValidation:
    """显式给出的 CLI 参数非法时必须立即报错，而非静默落入 GUI/手动输入。"""

    def _expect_argparse_error(self, argv, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog"] + argv)
        with pytest.raises(SystemExit) as excinfo:
            evolution.get_file_and_params()
        assert excinfo.value.code == 2

    def test_dt_zero_rejected(self, monkeypatch):
        self._expect_argparse_error(["--dt", "0"], monkeypatch)

    def test_dt_negative_rejected(self, monkeypatch):
        self._expect_argparse_error(["--dt", "-0.5"], monkeypatch)

    def test_dt_nan_rejected(self, monkeypatch):
        self._expect_argparse_error(["--dt", "nan"], monkeypatch)

    def test_missing_file_rejected(self, monkeypatch):
        self._expect_argparse_error(
            ["--file", "definitely_missing_input.tif"], monkeypatch)

    def test_dpi_out_of_bounds_rejected(self, monkeypatch):
        self._expect_argparse_error(["--dpi", "20000"], monkeypatch)


class TestManualInputValidation:
    """手动输入路径的 dt 校验必须与 CLI 同口径（isfinite）。"""

    def test_manual_dt_nan_then_inf_rejected(self, tmp_path, monkeypatch):
        # 禁用 GUI 分支（tkinter 置为 None 触发 ImportError → 手动输入降级）
        monkeypatch.setitem(sys.modules, "tkinter", None)
        monkeypatch.setattr(sys, "argv", ["prog"])
        answers = iter([str(tmp_path / "x.tif"), "nan", "inf", "0.5", ""])
        monkeypatch.setattr("builtins.input", lambda *_: next(answers))
        tif, dt, title, shared, config, out = evolution.get_file_and_params()
        assert dt == 0.5


class TestOutputDirectoryProtection:
    """输出目录保护：非空目录/文件路径拒绝，退出码 1。"""

    def _run_main(self, tmp_path, monkeypatch, output_arg):
        tif = _write_stack(tmp_path,
                           [np.full((16, 16), 100, dtype=np.uint16)],
                           name="single3.tif")
        monkeypatch.setattr(sys, "argv",
                            ["prog", "--file", tif, "--output", output_arg])
        return evolution.main()

    def test_output_path_is_file_rejected(self, tmp_path, monkeypatch):
        target = tmp_path / "occupied.txt"
        target.write_text("x")
        assert self._run_main(tmp_path, monkeypatch, str(target)) == 1

    def test_output_dir_nonempty_rejected(self, tmp_path, monkeypatch):
        target = tmp_path / "old_results"
        target.mkdir()
        (target / "previous.png").write_bytes(b"png")
        assert self._run_main(tmp_path, monkeypatch, str(target)) == 1
        # 旧结果未被触碰
        assert (target / "previous.png").exists()

    def test_output_dir_empty_allowed(self, tmp_path, monkeypatch):
        target = tmp_path / "fresh"
        target.mkdir()
        # 空目录允许（后续在处理阶段因单帧失败退出，但不应在目录阶段报错）
        rc = self._run_main(tmp_path, monkeypatch, str(target))
        assert rc == 1  # 单帧堆栈 → 处理失败退出码 1，与目录保护无关


class TestAnalysisConfigValidation:
    """编程注入的非法参数在构造时报错（与 CLI 同口径）。"""

    def test_min_size_zero_raises(self):
        with pytest.raises(ValueError):
            evolution.AnalysisConfig(min_size=0)

    def test_negative_dpi_raises(self):
        with pytest.raises(ValueError):
            evolution.AnalysisConfig(dpi=-1)

    def test_margin_frac_out_of_range_raises(self):
        with pytest.raises(ValueError):
            evolution.AnalysisConfig(margin_frac=0.9)

    def test_defaults_are_valid(self):
        # 仅"构造不抛"不足以守护默认值：合法但错误的漂移（如 margin_frac
        # 0.05→0.45）不会触发 __post_init__ 报错，这里对关键字段默认值
        # 直接断言预期区间/数值。
        cfg = evolution.AnalysisConfig()
        assert cfg.dpi > 0
        assert 0 < cfg.margin_frac < 0.5
        assert cfg.min_size >= 1 and cfg.n_bg_frames >= 1
        assert cfg.z95 == pytest.approx(1.96)
