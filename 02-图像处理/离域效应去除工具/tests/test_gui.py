# -*- coding: utf-8 -*-
"""GUI 层回归测试。

只覆盖那些「一旦错了会静默毁掉用户数据」的行为，不做界面外观断言。
需要可用的 Tk 显示；没有显示环境时整个模块跳过。
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

tk = pytest.importorskip("tkinter")
tifffile = pytest.importorskip("tifffile")

try:
    import main as app_main
except (ImportError, tk.TclError) as exc:  # pragma: no cover - 环境相关
    # 只兜环境类失败：依赖缺失（ImportError）或无显示环境（TclError）。
    # 代码级异常（SyntaxError/NameError/AttributeError 等）必须直接失败，
    # 否则 GUI 回归整模块静默假绿。
    pytest.skip(f"无法导入 GUI 模块: {exc}", allow_module_level=True)

import roi_model  # noqa: E402


@pytest.fixture(scope="module")
def _tk_app():
    """整个模块共用一个 Tk root。

    反复 Tk() + destroy() 会踩 Tcl 的 `invalid command name "tcl_findLibrary"`，
    让后续用例偶发被跳过（覆盖率悄悄掉）。共用 root 也顺带让焦点状态更稳定。
    """
    try:
        root = tk.Tk()
    except tk.TclError as exc:              # pragma: no cover - 无显示环境
        pytest.skip(f"没有可用的 Tk 显示: {exc}")
    root.geometry("200x200+-4000+-4000")    # 映射但挪到屏幕外，避免测试时弹窗
    application = app_main.DelocApp(root)
    root.update()
    try:
        yield application
    finally:
        root.destroy()


@pytest.fixture
def app(_tk_app):
    """把应用状态复位到"刚打开程序"的样子，供每个用例独立使用。"""
    a = _tk_app
    a._path = None
    a._frame = 0
    a.img = None
    a.info = None
    a.roi = roi_model.RoiSet()
    a.engine = app_main.CleanEngine()
    a.mask = None
    a.result = None
    a._points = []
    a._drawing = False
    a._start = None
    a._cursor = None
    a._pan = None
    a._band_artists = []
    a._overlay = []
    # 结果有效性状态（审查项03/06）：夹具重置为"尚未计算"，需要导出的
    # 用例用 _mark_result_valid() 显式声明，模拟一次成功的 _recompute。
    a._result_valid = False
    a._result_caption = "尚未计算"
    a._keep_fraction = 0.0
    a.var_fmin.set(f"{app_main.DEFAULT_FMIN:.4f}")
    a.var_fmax.set(f"{app_main.DEFAULT_FMAX:.4f}")
    a.var_dilate.set(6.0)
    a.var_feather.set(8.0)
    a.var_strength.set(1.0)
    a.var_band_feather.set(2.0)
    a.view.set("结果")
    a.tool.set("polygon")
    a.subtract.set(False)
    a.root.update()
    return a


def _load_stack(app, tmp_path, frames=3):
    """写入一个多帧栈并载入第 1 帧（用于验证切帧行为）。

    photometric="minisblack" 必须显式给：tifffile 会把 (3|4, H, W) 的 uint8
    数组默认当成 planar RGB(A) 写，那样 inspect 会把它误报成"3 帧"。
    """
    stack = np.stack([np.full((16, 20), i, np.uint8) for i in range(frames)])
    path = str(tmp_path / "stack.tif")
    tifffile.imwrite(path, stack, photometric="minisblack")
    app._path = path
    app._frame = 1
    app._load_current()
    return path


def _wait_for(pred, timeout_s=1.5):
    """轮询等待某个焦点状态成立（Windows 的焦点转移是异步的，不能只 update 一次）。"""
    deadline = time.time() + timeout_s
    while True:
        if pred():
            return True
        if time.time() >= deadline:
            return False
        time.sleep(0.02)


def _focus_entry(app):
    """把焦点放到频带输入框；拿不到焦点就跳过（无显示/窗口未映射）。"""
    entry = app._band_entries[0]

    def focused():
        entry.focus_force()
        app.root.update()
        return app._typing_in_entry()

    if not _wait_for(focused):
        pytest.skip("当前环境无法让 Entry 取得焦点")
    return entry


def _focus_root(app):
    """把焦点移回顶层窗口；做不到就跳过。"""
    def focused():
        app.root.focus_force()
        app.root.update()
        return not app._typing_in_entry()

    if not _wait_for(focused):
        pytest.skip("当前环境无法把焦点从 Entry 移回顶层窗口")


# --------------------------------------------------------------------------
# P1：输入框里的按键不能触发全局快捷键
# --------------------------------------------------------------------------
def test_typing_in_entry_detects_entry_focus(app):
    _focus_entry(app)
    assert app._typing_in_entry() is True

    _focus_root(app)
    assert app._typing_in_entry() is False


def test_shortcut_is_skipped_while_typing(app, monkeypatch):
    calls = []
    handler = app._shortcut(lambda: calls.append("ran"))

    monkeypatch.setattr(app, "_typing_in_entry", lambda: True)
    handler()
    assert calls == []

    monkeypatch.setattr(app, "_typing_in_entry", lambda: False)
    handler()
    assert calls == ["ran"]


def test_arrow_key_in_entry_does_not_switch_frame_or_wipe_roi(app, tmp_path):
    """P1 回归：在频带输入框里按 ←/→ 曾经会切帧，而切帧会重建 RoiSet，
    把用户画好的 ROI 和撤销历史一起清掉。"""
    _load_stack(app, tmp_path, frames=3)
    assert app.roi.add(roi_model.RoiShape("rect", [(2, 2), (10, 10)]))
    assert app._frame == 1 and len(app.roi) == 1

    entry = _focus_entry(app)
    entry.event_generate("<Left>")
    app.root.update()

    assert app._frame == 1, "在输入框里按 ← 不该切帧"
    assert len(app.roi) == 1, "ROI 被切帧清掉了"
    assert app.roi.has_keep


def test_arrow_key_on_canvas_still_switches_frame(app, tmp_path):
    """对照：焦点不在输入框时，←/→ 仍然要能切帧。"""
    _load_stack(app, tmp_path, frames=3)
    _focus_root(app)
    assert app._typing_in_entry() is False

    app.root.event_generate("<Left>")
    app.root.update()
    assert app._frame == 0


# --------------------------------------------------------------------------
# P1：只有挖除区时必须拦住保存
# --------------------------------------------------------------------------
def _mark_result_valid(app, keep_fraction=0.5):
    """模拟一次成功的 _recompute：置有效标志并给出保留覆盖率。"""
    app._result_valid = True
    app._keep_fraction = keep_fraction
    app._result_caption = app._param_caption()


def test_require_result_blocks_subtract_only_roi(app, monkeypatch):
    warnings = []
    monkeypatch.setattr(app_main.messagebox, "showwarning",
                        lambda *a, **k: warnings.append(a[-1]))

    app.img = np.zeros((32, 32), np.float32)
    app.result = np.zeros((32, 32), np.float32)
    app.roi = roi_model.RoiSet()
    app.roi.add(roi_model.RoiShape("rect", [(2, 2), (10, 10)], subtract=True))
    assert not app.roi.has_keep
    _mark_result_valid(app)

    assert app._require_result() is False
    assert warnings and "保留区" in warnings[-1]


def test_require_result_allows_normal_roi(app, monkeypatch):
    monkeypatch.setattr(app_main.messagebox, "showwarning", lambda *a, **k: None)
    app.img = np.zeros((32, 32), np.float32)
    app.result = np.zeros((32, 32), np.float32)
    app.roi = roi_model.RoiSet()
    app.roi.add(roi_model.RoiShape("rect", [(2, 2), (10, 10)]))
    _mark_result_valid(app)
    assert app._require_result() is True


# --------------------------------------------------------------------------
# 审查项03：参数失效 / 结果过期必须拦住导出
# --------------------------------------------------------------------------
def test_require_result_blocks_stale_result(app, monkeypatch):
    """频带改无效后 result 仍是旧数组：必须拒绝导出（图注会描述错参数）。"""
    warnings = []
    monkeypatch.setattr(app_main.messagebox, "showwarning",
                        lambda *a, **k: warnings.append(a[-1]))
    app.img = np.zeros((32, 32), np.float32)
    app.result = np.zeros((32, 32), np.float32)
    app.roi = roi_model.RoiSet()
    app.roi.add(roi_model.RoiShape("rect", [(2, 2), (10, 10)]))
    _mark_result_valid(app)

    app.var_fmin.set("0.2")
    app.var_fmax.set("0.1")          # 下限 >= 上限 -> 无效
    app.apply_band()                 # 应把结果标记为过期
    assert app._result_valid is False
    assert app._require_result() is False
    assert warnings and "旧结果" in warnings[-1]


def test_recompute_marks_result_valid_and_captures_coverage(app):
    """成功的重算必须置有效、记录覆盖率，并把参数快照进 caption。"""
    app.img = np.ones((40, 40), np.float32) * 100.0
    app.roi = roi_model.RoiSet()
    app.roi.add(roi_model.RoiShape("rect", [(10, 10), (30, 30)]))
    app.var_dilate.set(0.0)
    app.var_feather.set(0.0)
    app._recompute()

    assert app._result_valid is True
    assert 0.2 < app._keep_fraction < 0.6      # 20×20 / 40×40 = 0.25
    assert "fmin=" in app._result_caption and "带宽羽化" in app._result_caption
    assert app._result_caption == app._param_caption()


def test_require_result_blocks_zero_coverage(app, monkeypatch):
    """审查项06：保留区被挖除区完全吃掉时 has_keep=True 但有效覆盖率为 0。"""
    warnings = []
    monkeypatch.setattr(app_main.messagebox, "showwarning",
                        lambda *a, **k: warnings.append(a[-1]))
    app.img = np.zeros((40, 40), np.float32)
    app.result = np.zeros((40, 40), np.float32)
    app.roi = roi_model.RoiSet()
    app.roi.add(roi_model.RoiShape("rect", [(10, 10), (30, 30)]))
    app.roi.add(roi_model.RoiShape("rect", [(5, 5), (35, 35)], subtract=True))
    assert app.roi.has_keep                  # 形状标志通过
    app.var_dilate.set(0.0)
    app.var_feather.set(0.0)
    app._recompute()
    assert app._keep_fraction <= 1e-6        # 但实际覆盖率是 0
    assert app._require_result() is False
    assert warnings and "有效覆盖率" in warnings[-1]


# --------------------------------------------------------------------------
# 审查项05：没有 ROI 时预览必须是原图，不能是已删晶格的 base
# --------------------------------------------------------------------------
def test_preview_shows_original_when_no_roi(app):
    """空 RoiSet + 强度 1 的结果等于 base（晶格已被删）。

    默认视图是"结果"，若直接显示 result，用户会在一张已抑制晶格的图上
    圈"晶体真实边界"，与状态栏「此时不做任何抑制」完全相反。
    """
    rng = np.random.default_rng(0)
    app.img = (100 + 20 * rng.standard_normal((48, 48))).astype(np.float32)
    app.roi = roi_model.RoiSet()             # 空 ROI
    app.view.set("结果")
    app.var_dilate.set(0.0)
    app.var_feather.set(0.0)
    app._recompute()

    shown = app._image_artist.get_array()
    assert np.allclose(np.asarray(shown), app.img, atol=1e-4), \
        "无 ROI 时预览必须是原图"


# --------------------------------------------------------------------------
# 审查项04：滚轮缩放不能因反向 Y 轴跨度符号而跳飞
# --------------------------------------------------------------------------
def test_scroll_zoom_keeps_view_sane_on_reversed_y_axis(app):
    app.img = np.zeros((128, 128), np.float32)
    app.fit_view()
    before = (app.ax.get_xlim(), app.ax.get_ylim())
    assert before[1][1] < before[1][0]       # Y 轴确实是反向（跨度为负）

    class _Event:
        inaxes = app.ax
        xdata = 64.0
        ydata = 64.0
        button = "up"

    app._on_scroll(_Event())
    xlim, ylim = app.ax.get_xlim(), app.ax.get_ylim()
    assert all(abs(v) < 1e4 for v in (*xlim, *ylim)), \
        f"视图跳到了异常坐标: xlim={xlim} ylim={ylim}"
    # 放大：跨度必须变小，且锚点 64 仍留在视图内
    assert (ylim[1] - ylim[0]) > (before[1][1] - before[1][0])   # 负跨度，绝对值变小
    assert ylim[0] > 64.0 > ylim[1]


def test_after_roi_edit_status_warns_about_subtract_only(app):
    app.img = np.zeros((32, 32), np.float32)
    app.result = np.zeros((32, 32), np.float32)
    app.roi = roi_model.RoiSet()
    app.roi.add(roi_model.RoiShape("rect", [(2, 2), (10, 10)], subtract=True))
    app._after_roi_edit()
    assert "只有挖除区" in app.status.get()


# --------------------------------------------------------------------------
# P1：写盘前先提示裁剪，而不是写完再告知
# --------------------------------------------------------------------------
def test_clipping_is_asked_before_choosing_path(app, tmp_path, monkeypatch):
    order = []
    app.img = np.zeros((16, 16), np.float32)
    app.result = np.full((16, 16), 300.0, np.float32)   # 远超 uint8 上限
    app.roi = roi_model.RoiSet()
    app.roi.add(roi_model.RoiShape("rect", [(2, 2), (10, 10)]))
    app._path = str(tmp_path / "src.tif")
    tifffile.imwrite(app._path, np.zeros((16, 16), np.uint8))
    app.info = app_main.tif_io.inspect(app._path)
    _mark_result_valid(app)

    questions = []

    def fake_askyesno(title, message, *a, **k):
        questions.append(message)
        order.append("askyesno")
        return False          # 不转 float32，也不改位深

    def fake_saveas(*a, **k):
        order.append("asksaveasfilename")
        return str(tmp_path / "out.tif")

    monkeypatch.setattr(app_main.messagebox, "askyesno", fake_askyesno)
    monkeypatch.setattr(app_main.messagebox, "showinfo", lambda *a, **k: None)
    monkeypatch.setattr(app_main.filedialog, "asksaveasfilename", fake_saveas)

    app.save_result()

    assert "asksaveasfilename" in order
    clip_q = [q for q in questions if "裁剪" in q]
    assert clip_q, "有裁剪风险时必须先问用户"
    assert order.index("askyesno") < order.index("asksaveasfilename")
    # 提示的 dtype 必须与实际写出的一致
    assert "uint8" in clip_q[0]


def test_no_clipping_question_when_values_fit(app, tmp_path, monkeypatch):
    questions = []
    app.img = np.zeros((16, 16), np.float32)
    app.result = np.full((16, 16), 100.0, np.float32)
    app.roi = roi_model.RoiSet()
    app.roi.add(roi_model.RoiShape("rect", [(2, 2), (10, 10)]))
    app._path = str(tmp_path / "src.tif")
    tifffile.imwrite(app._path, np.zeros((16, 16), np.uint8))
    app.info = app_main.tif_io.inspect(app._path)
    _mark_result_valid(app)

    monkeypatch.setattr(app_main.messagebox, "askyesno",
                        lambda t, m, *a, **k: questions.append(m) or False)
    monkeypatch.setattr(app_main.messagebox, "showinfo", lambda *a, **k: None)
    monkeypatch.setattr(app_main.filedialog, "asksaveasfilename",
                        lambda *a, **k: str(tmp_path / "out.tif"))
    app.save_result()

    assert len(questions) == 1, "没有裁剪风险就不该多问一次"


# --------------------------------------------------------------------------
# 损坏的 ROI JSON 不能让载入变成"点了没反应"
# --------------------------------------------------------------------------
def test_load_roi_tolerates_bad_extra_fields(app, tmp_path, monkeypatch):
    app.img = np.zeros((32, 32), np.float32)
    app.result = np.zeros((32, 32), np.float32)
    app._path = str(tmp_path / "src.tif")

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "shapes": [{"kind": "rect", "points": [[2, 2], [10, 10]]}],
        "fmin": None, "dilate": "abc", "image_shape": [478],
    }), encoding="utf-8")

    monkeypatch.setattr(app_main.filedialog, "askopenfilename", lambda *a, **k: str(bad))
    monkeypatch.setattr(app_main.messagebox, "askyesno", lambda *a, **k: True)

    app.load_roi()                       # 不允许抛异常

    assert len(app.roi) == 1
    assert "忽略无效字段" in app.status.get()
    assert app.var_fmin.get() == f"{app_main.DEFAULT_FMIN:.4f}"   # 坏值没写进去


# --------------------------------------------------------------------------
# P1：打开/翻帧/关闭不能静默撕裂状态或丢工作
# --------------------------------------------------------------------------
def test_open_file_cancelled_keeps_previous_state(app, tmp_path, monkeypatch):
    """帧号对话框取消后，_path/_frame/img 必须维持原状（曾经 _path 已被改）。"""
    _load_stack(app, tmp_path, frames=3)          # 当前：stack.tif 第 1 帧
    before = (app._path, app._frame, app.img)

    other = str(tmp_path / "other.tif")
    tifffile.imwrite(other, np.stack([np.zeros((8, 8), np.uint8)] * 2),
                     photometric="minisblack")
    monkeypatch.setattr(app_main.filedialog, "askopenfilename", lambda *a, **k: other)
    monkeypatch.setattr(app_main.simpledialog, "askinteger", lambda *a, **k: None)

    app.open_file()
    assert (app._path, app._frame, app.img) == before


def test_open_file_load_failure_rolls_back(app, tmp_path, monkeypatch):
    """载入失败必须回滚 _path/_frame：防覆盖 guard 与默认文件名才不会张冠李戴。"""
    _load_stack(app, tmp_path, frames=3)
    before_path, before_frame = app._path, app._frame

    other = str(tmp_path / "other.tif")
    tifffile.imwrite(other, np.zeros((8, 8), np.uint8))
    monkeypatch.setattr(app_main.filedialog, "askopenfilename", lambda *a, **k: other)
    monkeypatch.setattr(app_main.messagebox, "showerror", lambda *a, **k: None)

    def boom(_path, _frame):
        raise RuntimeError("simulated load failure")

    monkeypatch.setattr(app_main.tif_io, "load", boom)
    app.open_file()
    assert app._path == before_path and app._frame == before_frame


def test_step_frame_with_roi_asks_and_keeps_state_on_decline(app, tmp_path, monkeypatch):
    """有 ROI 时翻帧必须先确认；拒绝则帧号与 ROI 都不动。"""
    _load_stack(app, tmp_path, frames=3)          # 当前第 1 帧（共 0..2）
    assert app.roi.add(roi_model.RoiShape("rect", [(2, 2), (10, 10)]))

    asked = []
    monkeypatch.setattr(app_main.messagebox, "askyesno",
                        lambda t, m, *a, **k: asked.append(m) or False)
    app.step_frame(1)
    assert asked, "有 ROI 时翻帧必须先确认"
    assert app._frame == 1
    assert len(app.roi) == 1


def test_step_frame_without_roi_switches_silently(app, tmp_path, monkeypatch):
    _load_stack(app, tmp_path, frames=3)
    asked = []
    monkeypatch.setattr(app_main.messagebox, "askyesno",
                        lambda t, m, *a, **k: asked.append(m) or True)
    app.step_frame(1)
    assert not asked, "没有 ROI 时翻帧不该打断用户"
    assert app._frame == 2


def test_on_close_with_unsaved_roi_asks_and_aborts(app, tmp_path, monkeypatch):
    _load_stack(app, tmp_path, frames=3)
    assert app.roi.add(roi_model.RoiShape("rect", [(2, 2), (10, 10)]))
    monkeypatch.setattr(app_main.messagebox, "askyesno", lambda *a, **k: False)

    app.on_close()
    assert app.root.winfo_exists()        # 用户拒绝：窗口还在，ROI 还在
    assert len(app.roi) == 1


# --------------------------------------------------------------------------
# P2：band_feather 必须随 ROI JSON 往返，并同步到 engine
# --------------------------------------------------------------------------
def test_roi_json_roundtrips_band_feather(app, tmp_path, monkeypatch):
    """带宽羽化原先不写入也不回读 ROI JSON：换图复现时 FFT 参数缺一角。"""
    app.img = np.zeros((32, 32), np.float32)
    app._path = str(tmp_path / "src.tif")
    assert app.roi.add(roi_model.RoiShape("rect", [(2, 2), (10, 10)]))
    app.var_band_feather.set(3.5)

    roi_path = tmp_path / "roi.json"
    monkeypatch.setattr(app_main.filedialog, "asksaveasfilename",
                        lambda *a, **k: str(roi_path))
    app.save_roi()
    assert json.loads(roi_path.read_text(encoding="utf-8"))["band_feather"] == 3.5

    app.var_band_feather.set(5.0)
    monkeypatch.setattr(app_main.filedialog, "askopenfilename",
                        lambda *a, **k: str(roi_path))
    monkeypatch.setattr(app_main.messagebox, "askyesno", lambda *a, **k: True)
    app.load_roi()

    assert app.var_band_feather.get() == 3.5
    assert app.engine.band_feather == 3.5, "载入后 engine 副本必须同步，否则 render 用旧值"


# --------------------------------------------------------------------------
# 第二轮审查项 15/18：图像 artist 累积泄漏、切帧失败回滚
# --------------------------------------------------------------------------
def test_repeated_image_loads_do_not_accumulate_artists(app, tmp_path):
    """审查项15：_load_current 只把 _image_artist 置 None，旧对象仍挂在
    axes 上——实测 5 次换图后 axes 里有 5 张叠画图像（内存+渲染翻倍）。"""
    path = str(tmp_path / "a.tif")
    tifffile.imwrite(path, np.zeros((32, 32), np.uint8))
    app._path = path
    app._frame = 0

    for _ in range(5):
        app._load_current()
        app.root.update()

    assert len(app.ax.get_images()) == 1, "换图后 axes 只应保留一个图像 artist"


def test_step_frame_load_failure_rolls_back_frame(app, tmp_path, monkeypatch):
    """审查项18：切帧时载入失败必须回滚帧号，否则 _frame 指向新帧、
    img 还是旧帧像素，导出时张冠李戴。"""
    _load_stack(app, tmp_path, frames=3)          # 当前第 1 帧
    assert app._frame == 1
    monkeypatch.setattr(app_main.messagebox, "showerror", lambda *a, **k: None)

    def boom(_path, _frame):
        raise RuntimeError("simulated frame load failure")

    monkeypatch.setattr(app_main.tif_io, "load", boom)
    app.step_frame(1)
    assert app._frame == 1, "载入失败后帧号必须回滚"
