# -*- coding: utf-8 -*-
"""统计面积 GUI 方法回归（不启动 Tk 主循环）。

覆盖合并进来的保护（回归 2026-10-03 / 2026-10-04），用真实 pyplot figure 验证：

* ``on_close`` 只关闭本工具持有的 figure —— ``plt.close('all')`` 会清掉同进程
  内其他工具在 Suite 里创建的图窗；``fig is None`` 时不得调用
  ``plt.close(None)``（那会关掉**当前**旁图）；
* ``load_project`` 对顶层非 JSON 对象的文件给出友好提示，而不是未捕获的
  ``AttributeError``，并且不污染当前项目状态。

说明（后端与 CI）：被测模块在 import 时按工具自身约定执行
``matplotlib.use('TkAgg')``；本文件随后显式切到 ``Agg``，因为测试只创建
figure 对象、不创建 Tk 窗口。Windows CI 的 runner 有桌面会话，TkAgg 本身
也能加载；这里的 Agg 是为了让断言不依赖窗口系统，而不是"无显示 CI 必需"。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

pytest.importorskip("matplotlib")
import matplotlib  # noqa: E402

matplotlib.use("Agg")           # 测试不建 Tk 窗口；模块级 TkAgg 已被覆盖
from matplotlib import pyplot as plt  # noqa: E402


def _load_tool_module():
    """以独立模块名加载 统计面积.py（文件名是中文，不能用 import 语句）。"""
    spec = importlib.util.spec_from_file_location(
        "area_tool_under_test", PROJECT / "统计面积.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["area_tool_under_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return _load_tool_module()


def _bare_viewer(tool, **attrs):
    """构造只含被测方法所需字段的实例（不跑 __init__/Tk）。"""
    viewer = tool.TiffStackViewer.__new__(tool.TiffStackViewer)
    viewer.fig = None
    viewer.root = mock.Mock()
    viewer._dirty = False
    viewer._load_after_id = None
    viewer._load_generation = 0
    viewer.tiff_stack = None
    for key, value in attrs.items():
        setattr(viewer, key, value)
    return viewer


# ---------------------------------------------------------------------------
# on_close：只关闭自己的 figure
# ---------------------------------------------------------------------------
def test_on_close_closes_only_own_figure(tool):
    """其他工具/标签页的 figure 必须存活（真实 pyplot figure 验证）。"""
    own = plt.figure()
    other_a = plt.figure()
    other_b = plt.figure()
    try:
        viewer = _bare_viewer(tool, fig=own)
        viewer.on_close()

        assert not plt.fignum_exists(own.number), "本工具的 figure 应被关闭"
        assert plt.fignum_exists(other_a.number), "其他 figure 被误关"
        assert plt.fignum_exists(other_b.number), "其他 figure 被误关"
        assert viewer.root.destroy.called, "窗口仍应被销毁"
    finally:
        for figure in (own, other_a, other_b):
            plt.close(figure)


def test_on_close_without_figure_still_destroys_root(tool):
    """没有 figure（尚未绘图）时也必须正常销毁窗口。

    关键回归：``plt.close(None)`` 等价于关闭**当前** figure，会把旁边的图窗
    误关。这里放一张旁图，断言它在 fig=None 的关闭路径中存活。
    """
    bystander = plt.figure()
    try:
        viewer = _bare_viewer(tool, fig=None)
        viewer.on_close()
        assert viewer.root.destroy.called, "窗口仍应被销毁"
        assert plt.fignum_exists(bystander.number), (
            "fig=None 时不得调用 plt.close(None) 而误关当前旁图")
    finally:
        plt.close(bystander)


def test_on_close_cancels_pending_poll_and_bumps_generation(tool):
    """关闭时必须取消待执行的加载轮询并递增代次（既有行为保留）。"""
    viewer = _bare_viewer(tool, fig=plt.figure(), _load_after_id="after#7")
    try:
        viewer.on_close()
        viewer.root.after_cancel.assert_called_with("after#7")
        assert viewer._load_after_id is None
        assert viewer._load_generation == 1
    finally:
        plt.close("all")


def test_on_close_respects_unsaved_changes_prompt(tool):
    """未保存修改且用户选择取消时：不关图、不销毁窗口。"""
    own = plt.figure()
    try:
        viewer = _bare_viewer(tool, fig=own, _dirty=True)
        with mock.patch.object(tool.messagebox, "askyesno", return_value=False):
            viewer.on_close()
        assert plt.fignum_exists(own.number), "取消时不得关闭 figure"
        assert not viewer.root.destroy.called
    finally:
        plt.close(own)


# ---------------------------------------------------------------------------
# load_project：顶层对象校验
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("payload,label", [
    ([], "list"), (123, "int"), ("x", "str"), (None, "NoneType"),
    (True, "bool"), (1.5, "float"),
])
def test_load_project_rejects_non_object_root(tool, tmp_path, payload, label):
    """JSON 合法但顶层不是对象：友好提示、不抛异常、不污染当前项目。"""
    path = tmp_path / f"{label}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    viewer = _bare_viewer(tool)
    viewer._confirm_discard_changes = lambda: True
    viewer.status_var = mock.Mock()
    viewer._pending_project = "SENTINEL"

    with mock.patch.object(tool.filedialog, "askopenfilename", return_value=str(path)), \
            mock.patch.object(tool.messagebox, "showerror") as error:
        viewer.load_project()

    assert error.called, f"{label} 顶层非法时未给出错误提示"
    assert "JSON 对象" in str(error.call_args), "提示应说明期望的对象类型"
    assert viewer._pending_project == "SENTINEL", "不得污染/改动挂起项目"
    viewer.status_var.set.assert_called()


def test_load_project_accepts_object_root(tool, tmp_path):
    """合法对象：不报格式错误，进入既有加载流程。"""
    path = tmp_path / "ok.json"
    path.write_text(json.dumps({"file_path": str(tmp_path / "missing.tif")}),
                    encoding="utf-8")
    viewer = _bare_viewer(tool)
    viewer._confirm_discard_changes = lambda: True
    viewer.status_var = mock.Mock()
    viewer._pending_project = None
    loaded: list = []
    viewer.load_tiff_file = lambda tiff: loaded.append(tiff)

    with mock.patch.object(tool.filedialog, "askopenfilename", return_value=str(path)), \
            mock.patch.object(tool.messagebox, "showerror") as error, \
            mock.patch.object(tool.messagebox, "showwarning"), \
            mock.patch.object(tool.filedialog, "askopenfilename",
                              side_effect=[str(path), str(tmp_path / "replacement.tif")]), \
            mock.patch.object(Path, "exists", return_value=True):
        viewer.load_project()

    assert not error.called, "合法项目不应报格式错误"
    assert loaded, "合法项目应进入图像加载流程"


def test_load_project_cancel_does_not_touch_state(tool):
    """用户在文件对话框取消：直接返回，不改动状态。"""
    viewer = _bare_viewer(tool)
    viewer.status_var = mock.Mock()
    viewer._pending_project = "SENTINEL"
    with mock.patch.object(tool.filedialog, "askopenfilename", return_value=""):
        viewer.load_project()
    assert viewer._pending_project == "SENTINEL"
    viewer.status_var.set.assert_not_called()


def test_load_project_invalid_json_shows_error(tool, tmp_path):
    """文件根本不是 JSON：沿用既有的加载失败提示，不抛异常。"""
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    viewer = _bare_viewer(tool)
    viewer._confirm_discard_changes = lambda: True
    viewer.status_var = mock.Mock()
    with mock.patch.object(tool.filedialog, "askopenfilename", return_value=str(path)), \
            mock.patch.object(tool.messagebox, "showerror") as error:
        viewer.load_project()
    assert error.called
