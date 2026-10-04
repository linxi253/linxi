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
# R5：沿表面分段宽度必须能从界面显式给出（回归 2026-10-03）
# --------------------------------------------------------------------------
def test_segment_width_field_exists(app):
    """界面必须有沿表面分段宽度的输入项，且默认留空（交给预设）。"""
    assert hasattr(app, "segment_nm_var")
    assert app.segment_nm_var.get() == "", "默认必须留空，否则会覆盖预设值"


def test_explicit_segment_width_used_without_preset(app, tmp_path):
    """取消预设 + 显式填写宽度 → 必须组装出合法配置。

    回归前该组合直接抛"沿表面分段宽度必须显式提供"，因为界面上根本没有输入口。
    """
    _fill_valid_paths(app, tmp_path)
    app.use_preset_var.set(False)
    app.segment_nm_var.set("2.5")
    config = app._make_config()
    config.validate()
    assert config.along_surface_segment_nm == 2.5


def test_preset_supplies_segment_width_when_field_empty(app, tmp_path):
    """勾选预设且字段留空 → 使用预设值（内置 Cu L2,3 为 11.0），不被界面覆盖。"""
    _fill_valid_paths(app, tmp_path)
    app.use_preset_var.set(True)
    app.segment_nm_var.set("")
    config = app._make_config()
    config.validate()
    assert config.along_surface_segment_nm == 11.0


def test_explicit_segment_width_beats_preset(app, tmp_path):
    """非空显式输入优先于预设值。"""
    _fill_valid_paths(app, tmp_path)
    app.use_preset_var.set(True)
    app.segment_nm_var.set("3.25")
    config = app._make_config()
    config.validate()
    assert config.along_surface_segment_nm == 3.25


def test_no_preset_and_empty_segment_width_is_reported(app, tmp_path):
    """取消预设且留空 → 必须给出可诊断的提示，而不是静默用某个默认值。

    提示由 ``AnalysisConfig.validate()`` 给出（``_start`` 在组装后立即调用它），
    因此这里复现同一条链路：``_make_config()`` 再 ``validate()``。
    """
    _fill_valid_paths(app, tmp_path)
    app.use_preset_var.set(False)
    app.segment_nm_var.set("")
    with pytest.raises(ValueError, match="沿表面分段宽度"):
        app._make_config().validate()


def test_start_reports_missing_segment_width(app, tmp_path, monkeypatch):
    """真实 _start 链路：缺宽度必须弹错误框且不启动分析。"""
    _fill_valid_paths(app, tmp_path)
    app.use_preset_var.set(False)
    app.segment_nm_var.set("")
    errors = []
    monkeypatch.setattr(gui_module.messagebox, "showerror",
                        lambda *a, **k: errors.append(a[-1]))
    app._start()
    assert errors, "缺少分段宽度必须给出错误提示"
    assert app._processing is False and app._worker_thread is None


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "0", "-1", "abc", "1e999"])
def test_segment_width_rejects_bad_values(app, tmp_path, bad):
    """NaN/Inf/0/负数/非数值都必须被拒绝，不能静默传播进距离分层。"""
    _fill_valid_paths(app, tmp_path)
    app.use_preset_var.set(False)
    app.segment_nm_var.set(bad)
    with pytest.raises(ValueError, match="沿表面分段宽度"):
        app._make_config()


# --------------------------------------------------------------------------
# R5 布局：分段宽度字段不得与既有控件同格遮挡（回归 2026-10-03）
# --------------------------------------------------------------------------
def _walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from _walk(child)

def _grid_cells(widget):
    """返回该控件占用的 (row, column) 集合（展开 columnspan）。"""
    info = widget.grid_info()
    if not info:
        return set()
    row = int(info["row"])
    column = int(info["column"])
    span = int(info.get("columnspan", 1))
    return {(row, column + offset) for offset in range(span)}


def test_segment_width_field_does_not_collide_with_other_widgets(app):
    """分段宽度字段必须独占网格单元，且渲染后不与任何兄弟控件重叠。

    初稿把新 Label/提示放在 row 2，与原有的"执行 Cu1 注入恢复检验"及边缘
    提示同格，在真实 Tk 布局下互相遮挡。本用例同时检查 grid 单元占用与
    实际几何矩形相交，避免只测 _make_config 而漏掉布局缺陷。
    """
    entry = next(w for w in _walk(app.root)
                 if "textvariable" in w.keys()
                 and str(w.cget("textvariable")) == str(app.segment_nm_var))
    parent = entry.master
    siblings = [w for w in parent.winfo_children() if w is not entry]

    # 1) 不得有兄弟控件与本字段同格
    mine = _grid_cells(entry)
    for other in siblings:
        shared = mine & _grid_cells(other)
        assert not shared, (
            f"分段宽度输入框与 {other.winfo_class()}"
            f"“{other.cget('text') if 'text' in other.keys() else ''}”"
            f" 共用网格单元 {shared}")

    # 2) 标签与提示也不得与兄弟控件同格
    for widget in parent.winfo_children():
        if widget is entry or "text" not in widget.keys():
            continue
        text = str(widget.cget("text"))
        if "分段宽度" not in text and "留空采用" not in text:
            continue
        for other in parent.winfo_children():
            if other is widget:
                continue
            shared = _grid_cells(widget) & _grid_cells(other)
            assert not shared, f"“{text}”与兄弟控件共用网格单元 {shared}"

    # 3) 实际几何不得重叠
    app.root.update_idletasks()
    rects = []
    for widget in parent.winfo_children():
        width, height = widget.winfo_width(), widget.winfo_height()
        if width <= 1 or height <= 1:      # 未映射的控件不参与几何比较
            continue
        rects.append((widget, widget.winfo_x(), widget.winfo_y(), width, height))
    for index, (left, x, y, w, h) in enumerate(rects):
        for right, xx, yy, ww, hh in rects[index + 1:]:
            overlap = (min(x + w, xx + ww) > max(x, xx)
                       and min(y + h, yy + hh) > max(y, yy))
            label = lambda wid: str(wid.cget("text")) if "text" in wid.keys() else wid.winfo_class()  # noqa: E731
            assert not overlap, f"控件几何重叠：{label(left)} / {label(right)}"


def test_segment_width_hint_matches_behaviour(app, tmp_path):
    """提示文案必须与实现一致：留空用预设、填写覆盖预设、未启用预设必填。"""
    texts = [str(w.cget("text")) for w in _walk(app.root) if "text" in w.keys()]
    hint = next(t for t in texts if "留空采用" in t)
    assert "覆盖预设" in hint, "提示必须说明填写会覆盖预设"
    assert "未启用预设时必须填写" in hint, "提示必须说明未启用预设时必填"
    assert "以预设为准" not in hint, "旧文案与‘非空输入覆盖预设’的实现矛盾"


def test_segment_width_reaches_start_chain(app, tmp_path, monkeypatch):
    """真实 _start 链路：显式宽度必须进入传给 run_analysis 的配置。"""
    import threading as _threading

    _fill_valid_paths(app, tmp_path)
    app.use_preset_var.set(False)
    app.segment_nm_var.set("4.5")
    seen = []
    monkeypatch.setattr(gui_module, "output_directory_state", lambda *a, **k: "empty")

    class _Immediate:
        def __init__(self, target=None, args=(), daemon=None, **kw):
            self._target, self._args = target, args

        def start(self):
            seen.append(self._args[0])        # 同步取配置，不起真线程

        def is_alive(self):
            return False

        def join(self, timeout=None):
            return None

    monkeypatch.setattr(gui_module.threading, "Thread", _Immediate)
    assert _threading is not None
    app._start()
    assert seen, "_start 必须组装配置并交给工作线程"
    assert seen[0].along_surface_segment_nm == 4.5


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
