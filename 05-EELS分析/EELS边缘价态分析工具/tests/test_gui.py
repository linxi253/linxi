# -*- coding: utf-8 -*-
"""EELS GUI 层回归测试。

只覆盖「一旦错了会静默毁数据 / 静默跑错分析」的行为：配置组装校验、
覆盖既往结果前的确认、DM4 对象编号随文件切换重置。需要可用的 Tk
显示；没有显示环境时整个模块跳过（与界面外观有关的断言一概不做）。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

tk = pytest.importorskip("tkinter")

try:
    import eels_edge_analyzer.gui as gui_module
    from eels_edge_analyzer.gui import EELSEdgeAnalyzerApp
except (ImportError, tk.TclError) as exc:  # pragma: no cover - 环境相关
    # 只兜环境类失败：依赖缺失（ImportError）或无显示环境（TclError）。
    # 代码级异常（SyntaxError/NameError/AttributeError 等）必须直接失败，
    # 否则 GUI 回归整模块静默假绿。
    pytest.skip(f"无法导入 GUI 模块: {exc}", allow_module_level=True)

from eels_edge_analyzer.models import DatasetInfo


@pytest.fixture(scope="module")
def _tk_root():
    """整个模块共用一个 Tk root（反复建毁会踩 Tcl 的库初始化问题）。"""
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - 无显示环境
        pytest.skip(f"没有可用的 Tk 显示: {exc}")
    root.geometry("240x200+-4000+-4000")  # 映射但挪到屏幕外
    try:
        yield root
    finally:
        root.destroy()


@pytest.fixture()
def app(_tk_root):
    """复位到「刚启动」状态的界面，供每个用例独立使用。"""
    application = EELSEdgeAnalyzerApp(_tk_root)
    _tk_root.update()
    return application


def _fake_items() -> list[DatasetInfo]:
    return [
        DatasetInfo(0, (200, 180), "uint8", None, None, (200, 180), "survey_image", "survey"),
        DatasetInfo(1, (2048, 40, 50), "float32", -200.0, 2000.0, (40, 50), "low_loss_si", "low"),
        DatasetInfo(2, (2048, 40, 50), "float32", 850.0, 1000.0, (40, 50), "core_loss_si", "high"),
    ]


def _fill_valid_paths(app, tmp_path: Path) -> None:
    source = tmp_path / "input.dm4"
    source.touch()
    refs = []
    for state in range(3):
        ref = tmp_path / f"ref{state}.dm4"
        ref.touch()
        refs.append(str(ref))
    app.input_var.set(str(source))
    app.output_var.set(str(tmp_path / "out"))
    app.cu0_var.set(refs[0])
    app.cu1_var.set(refs[1])
    app.cu2_var.set(refs[2])


# --------------------------------------------------------------------------
# 配置组装：坏输入必须在启动分析前被拦下
# --------------------------------------------------------------------------
def test_make_config_requires_input_and_output(app, tmp_path):
    app.input_var.set("")
    with pytest.raises(ValueError, match="原始 DM3/DM4"):
        app._make_config()

    app.input_var.set(str(tmp_path / "a.dm4"))
    app.output_var.set("")
    with pytest.raises(ValueError, match="输出目录"):
        app._make_config()   # Path("") 等价当前目录，曾把结果写进启动目录


def test_make_config_rejects_partial_reference_files(app, tmp_path):
    _fill_valid_paths(app, tmp_path)
    app.cu1_var.set("")      # 只填了 Cu0/Cu2
    with pytest.raises(ValueError, match="三个文件全部指定"):
        app._make_config()


def test_make_config_rejects_bad_numbers(app, tmp_path):
    _fill_valid_paths(app, tmp_path)
    app.bootstrap_var.set("abc")
    with pytest.raises(ValueError, match="Bootstrap 次数必须是整数"):
        app._make_config()

    app.bootstrap_var.set("500")
    app.low_index_var.set("x")
    with pytest.raises(ValueError, match="低损编号必须是整数"):
        app._make_config()


# --------------------------------------------------------------------------
# 启动分析：无效配置不产生工作线程；覆盖既往结果必须先确认
# --------------------------------------------------------------------------
def test_start_invalid_config_shows_error_and_starts_nothing(app, tmp_path, monkeypatch):
    errors = []
    monkeypatch.setattr(gui_module.messagebox, "showerror",
                        lambda *a, **k: errors.append(a[-1]))
    app.input_var.set("")    # 其余留空同样无效
    app._start()
    assert errors, "无效配置必须弹错误框而不是静默不动"
    assert app._processing is False
    assert app._worker_thread is None


def test_start_declines_overwrite_of_previous_results(app, tmp_path, monkeypatch):
    _fill_valid_paths(app, tmp_path)
    output = Path(app.output_var.get())
    output.mkdir(parents=True)
    (output / "edge_analysis_summary.json").write_text("{}", encoding="utf-8")

    asked = []
    called = []
    monkeypatch.setattr(gui_module.messagebox, "askyesno",
                        lambda *a, **k: asked.append(a[-1]) or False)

    def _must_not_run(*a, **k):
        called.append("run")
        raise AssertionError("用户拒绝覆盖后不得启动分析")

    monkeypatch.setattr(gui_module, "run_analysis", _must_not_run)
    app._start()
    assert asked, "输出目录含既往结果时必须先确认"
    assert app._processing is False and not called, "用户拒绝后不得启动分析"


def test_start_runs_in_background_after_consent(app, tmp_path, monkeypatch):
    _fill_valid_paths(app, tmp_path)
    output = Path(app.output_var.get())
    output.mkdir(parents=True)
    (output / "edge_analysis_summary.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(gui_module.messagebox, "askyesno", lambda *a, **k: True)
    monkeypatch.setattr(gui_module.messagebox, "showinfo", lambda *a, **k: None)
    monkeypatch.setattr(gui_module, "run_analysis",
                        lambda config, progress=None, cancel=None: object())
    monkeypatch.setattr(gui_module, "export_artifacts",
                        lambda artifacts, config, **k: {"report": str(output / "edge_analysis_report.md")})
    app._start()
    assert app._processing and app._worker_thread is not None

    deadline = time.time() + 5
    while time.time() < deadline and app._processing:
        app.root.update()
        time.sleep(0.02)
    assert app._processing is False, "后台任务应已完成并经消息队列收尾"
    assert app._result_paths is not None


# --------------------------------------------------------------------------
# DM4 对象编号只对当前文件有效：换文件必须重置并重新自动填充
# --------------------------------------------------------------------------
def test_browse_input_resets_stale_dataset_indices(app, tmp_path, monkeypatch):
    app.survey_index_var.set("9")
    app.low_index_var.set("8")
    app.high_index_var.set("7")   # 旧文件的手工编号，对新文件是危险的

    new_file = tmp_path / "new.dm4"
    new_file.touch()
    monkeypatch.setattr(gui_module.filedialog, "askopenfilename",
                        lambda *a, **k: str(new_file))
    monkeypatch.setattr(gui_module, "inspect_dm4", lambda _path: _fake_items())

    app._browse_input()

    deadline = time.time() + 3
    while time.time() < deadline and app._inspecting:
        app.root.update()
        time.sleep(0.02)
    app.root.update()

    assert (app.survey_index_var.get(), app.low_index_var.get(), app.high_index_var.get()) == (
        "0", "1", "2"
    ), "换文件后旧编号不得残留，应由新文件的对象检查结果重新填充"
