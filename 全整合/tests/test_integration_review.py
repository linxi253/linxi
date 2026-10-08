# -*- coding: utf-8 -*-
"""TEM Suite 集成层回归测试（回归 2026-10-03 R3）。

覆盖此前真实 Tk 复现的缺陷：事件转发语义、按页菜单恢复、关闭生命周期、
宿主能力判定、子进程输出捕获。全部使用真实 Tk 与真实 SuiteApp/ToolHost，
只在"是否结束进程""用户确认框"这类不可自动化处使用 mock —— 绝不真的退出
验收进程。

需要可用的 Tk 显示；没有显示环境时整个模块跳过。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

tk = pytest.importorskip("tkinter")

from temsuite.app import SuiteApp                       # noqa: E402
from temsuite.hostframe import ToolHost                  # noqa: E402
from temsuite.mplbackend import force_headless_backend   # noqa: E402

force_headless_backend()


def _new_tk_root():
    """创建一个新的 Tk root，能容忍 Tcl 的残留默认根状态。

    销毁某个 root 之后 ``tkinter._default_root`` 可能仍指向它，下一次 ``Tk()``
    会抛 ``Can't find a usable tk.tcl`` / ``invalid command name "tcl_findLibrary"``。
    这是 Tk/pytest 的交互问题（已用纯 tkinter 最小复现确认，与本套件代码无关），
    重试前清掉该引用即可恢复。**不**把它当成 skip —— 那会掩盖真实失败。
    """
    last: tk.TclError | None = None
    for attempt in range(3):
        try:
            return tk.Tk()
        except tk.TclError as exc:
            last = exc
            stale = tk._default_root
            tk._default_root = None
            if stale is not None:
                try:
                    stale.destroy()
                except tk.TclError:
                    pass
    raise tk.TclError(f"无法创建 Tk root（重试 3 次）：{last}")


@pytest.fixture(scope="module")
def _shared_root():
    """整个模块共用一个 Tk root。

    反复 ``Tk()``/``destroy()`` 会在某些环境下让 Tcl 库搜索状态损坏，表现为
    后续用例被误跳过。共用一个 root 可大幅减少这种抖动；需要真实销毁 root 的
    用例（程序级关闭）改用 :func:`disposable_root`。
    """
    try:
        window = _new_tk_root()
    except tk.TclError as exc:  # pragma: no cover - 无显示环境
        pytest.skip(f"没有可用的 Tk 显示: {exc}")
    window.geometry("300x200+-4000+-4000")
    try:
        yield window
    finally:
        try:
            for aid in window.tk.call("after", "info"):
                window.after_cancel(aid)
            window.update_idletasks()
            window.destroy()
        except tk.TclError:
            pass
        tk._default_root = None


@pytest.fixture()
def root(_shared_root):
    """把共用 root 复位成"干净"状态供单个用例使用。

    上一个用例留下的标签页/部件会被清掉，但**不销毁 root 本身**。
    """
    for child in list(_shared_root.winfo_children()):
        try:
            child.destroy()
        except tk.TclError:
            pass
    try:
        _shared_root.update()
    except tk.TclError:
        pass
    yield _shared_root
    for child in list(_shared_root.winfo_children()):
        try:
            child.destroy()
        except tk.TclError:
            pass


@pytest.fixture()
def disposable_root():
    """用例自己拥有、允许被销毁的 root（程序级关闭用例专用）。

    这些用例会真的销毁 root（正是被测行为），因此不能借用模块共用的 root。
    销毁后清掉 ``tkinter._default_root``，避免下一次 ``Tk()`` 复用已失效解释器。
    """
    try:
        window = _new_tk_root()
    except tk.TclError as exc:  # pragma: no cover - 无显示环境
        pytest.skip(f"没有可用的 Tk 显示: {exc}")
    window.geometry("300x200+-4000+-4000")
    try:
        yield window
    finally:
        try:
            if window.winfo_exists():
                for aid in window.tk.call("after", "info"):
                    window.after_cancel(aid)
                window.update_idletasks()
                window.destroy()
        except tk.TclError:
            pass
        tk._default_root = None


def _capture_file():
    """给子进程 stdout 用的具名临时文件。

    不用 ``tempfile.TemporaryFile``：在 Windows 上它与 pytest 的捕获文件描述符
    相互干扰（整模块运行时报 ``ValueError: I/O operation on closed file``）。
    具名文件由 :meth:`SuiteApp._close_subprocess_log` 关闭，测试负责删除。
    """
    import tempfile

    fd, name = tempfile.mkstemp(prefix="temsuite-probe-", suffix=".log")
    handle = os.fdopen(fd, "w+", encoding="utf-8", errors="replace")
    handle._probe_name = name  # type: ignore[attr-defined]
    return handle


def _discard_capture(handle):
    """删除具名临时捕获文件（句柄可能已被 _close_subprocess_log 关闭）。"""
    name = getattr(handle, "_probe_name", None)
    try:
        handle.close()
    except Exception:
        pass
    if name:
        try:
            os.unlink(name)
        except OSError:
            pass


def _notebook(root):
    from tkinter import ttk

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True)
    return nb


def _hosts(root):
    """建两个标签页，各含一个 Entry，返回 (notebook, h1, h2, e1, e2)。"""
    from tkinter import ttk

    nb = _notebook(root)
    h1 = ToolHost(nb, tool_id="one")
    h2 = ToolHost(nb, tool_id="two")
    nb.add(h1, text="one")
    nb.add(h2, text="two")
    e1 = ttk.Entry(h1)
    e1.pack()
    e2 = ttk.Entry(h2)
    e2.pack()
    root.update()
    return nb, h1, h2, e1, e2


def _send(root, widget, sequence="<<SuiteProbe>>"):
    root.update()
    widget.event_generate(sequence)
    root.update()


# ---------------------------------------------------------------------------
# 事件转发：真实子部件 event_generate
# ---------------------------------------------------------------------------
def test_only_active_tab_receives_events(root):
    nb, h1, h2, e1, e2 = _hosts(root)
    calls = []
    h1.bind("<<SuiteProbe>>", lambda e: calls.append("one"))
    h2.bind("<<SuiteProbe>>", lambda e: calls.append("two"))

    nb.select(h1)
    _send(root, e1)
    nb.select(h2)
    _send(root, e2)
    assert calls == ["one", "two"]


def test_bind_without_add_replaces_and_with_add_appends(root):
    nb, h1, _h2, e1, _e2 = _hosts(root)
    calls = []
    h1.bind("<<SuiteProbe>>", lambda e: calls.append("old"))
    h1.bind("<<SuiteProbe>>", lambda e: calls.append("new"))
    _send(root, e1)
    assert calls == ["new"], "未给 add 时必须替换本页旧回调"

    calls.clear()
    h1.bind("<<SuiteProbe>>", lambda e: calls.append("a"))
    h1.bind("<<SuiteProbe>>", lambda e: calls.append("b"), add="+")
    _send(root, e1)
    assert calls == ["a", "b"], "add='+' 必须追加，且按登记顺序调用"


def test_bind_returning_break_stops_chain(root):
    nb, h1, _h2, e1, _e2 = _hosts(root)
    calls = []
    h1.bind("<<SuiteProbe>>", lambda e: calls.append("a") or "break")
    h1.bind("<<SuiteProbe>>", lambda e: calls.append("b"), add="+")
    _send(root, e1)
    assert calls == ["a"], "返回 'break' 后不得继续调用后续回调"


def test_unbind_by_funcid_keeps_other_callbacks(root):
    nb, h1, _h2, e1, _e2 = _hosts(root)
    calls = []
    first = h1.bind("<<SuiteProbe>>", lambda e: calls.append("a"))
    h1.bind("<<SuiteProbe>>", lambda e: calls.append("b"), add="+")
    h1.unbind("<<SuiteProbe>>", first)
    _send(root, e1)
    assert calls == ["b"], "按 funcid 解绑只应移除该回调"


def test_bind_all_is_per_tab_and_does_not_leak(root):
    nb, h1, h2, e1, e2 = _hosts(root)
    calls = []
    h1.bind_all("<<SuiteProbe>>", lambda e: calls.append("one"))
    h2.bind_all("<<SuiteProbe>>", lambda e: calls.append("two"))
    nb.select(h1)
    _send(root, e1)
    nb.select(h2)
    _send(root, e2)
    assert calls == ["one", "two"], "bind_all 不得串扰到非激活页"


def test_destroyed_tab_binding_does_not_fire_and_root_survives(root):
    nb, h1, h2, _e1, e2 = _hosts(root)
    calls = []
    root.bind("<<SuiteProbe>>", lambda e: calls.append("root"))
    h1.bind_all("<<SuiteProbe>>", lambda e: calls.append("one"), add="+")
    h2.bind_all("<<SuiteProbe>>", lambda e: calls.append("two"), add="+")

    h1._destroy_now()
    nb.select(h2)
    _send(root, e2)
    assert calls == ["root", "two"], (
        "关闭一页后：本页回调必须停止，宿主 root 与邻页绑定必须保留")


# ---------------------------------------------------------------------------
# 菜单：按页保存与恢复
# ---------------------------------------------------------------------------
def test_menu_restored_per_tab(root):
    app = SuiteApp(root)
    h1 = ToolHost(app.notebook, tool_id="one", on_menu_change=app._on_tool_menu_change)
    h2 = ToolHost(app.notebook, tool_id="two", on_menu_change=app._on_tool_menu_change)
    app.notebook.add(h1, text="one")
    app.notebook.add(h2, text="two")
    m1 = tk.Menu(h1, tearoff=False)
    m1.add_command(label="ONE")
    m2 = tk.Menu(h2, tearoff=False)
    m2.add_command(label="TWO")

    h1.configure(menu=m1)
    h2.configure(menu=m2)
    seen = []
    for host in (h1, h2, h1):
        app.notebook.select(host)
        root.update()
        seen.append(str(root.cget("menu")))
    assert seen == [str(m1), str(m2), str(m1)]


def test_menu_cleared_for_tab_without_menu(root):
    app = SuiteApp(root)
    h1 = ToolHost(app.notebook, tool_id="one", on_menu_change=app._on_tool_menu_change)
    h2 = ToolHost(app.notebook, tool_id="two", on_menu_change=app._on_tool_menu_change)
    app.notebook.add(h1, text="one")
    app.notebook.add(h2, text="two")
    m1 = tk.Menu(h1, tearoff=False)
    h1.configure(menu=m1)

    app.notebook.select(h1)
    root.update()
    assert str(root.cget("menu")) == str(m1)

    app.notebook.select(h2)          # 无菜单的工具
    root.update()
    assert str(root.cget("menu")) == "", "切到无菜单页必须清空菜单栏"

    app.notebook.select(app._welcome)  # 起始页
    root.update()
    assert str(root.cget("menu")) == ""


def test_background_load_does_not_steal_menu(root):
    app = SuiteApp(root)
    h1 = ToolHost(app.notebook, tool_id="one", on_menu_change=app._on_tool_menu_change)
    app.notebook.add(h1, text="one")
    app.notebook.select(h1)
    root.update()
    m1 = tk.Menu(h1, tearoff=False)
    h1.configure(menu=m1)

    # 后台页（未激活）声明自己的菜单
    h2 = ToolHost(app.notebook, tool_id="two", on_menu_change=app._on_tool_menu_change)
    app.notebook.add(h2, text="two")
    m2 = tk.Menu(h2, tearoff=False)
    h2.configure(menu=m2)
    app.notebook.select(h1)
    root.update()
    assert str(root.cget("menu")) == str(m1), "后台加载不得夺走前台菜单"


def test_closing_tab_clears_its_menu(root):
    app = SuiteApp(root)
    h1 = ToolHost(app.notebook, tool_id="one", on_menu_change=app._on_tool_menu_change)
    app.notebook.add(h1, text="one")
    app.notebook.select(h1)
    m1 = tk.Menu(h1, tearoff=False)
    h1.configure(menu=m1)
    root.update()
    assert str(root.cget("menu")) == str(m1)

    h1._destroy_now()
    app._apply_active_menu()
    root.update()
    assert str(root.cget("menu")) == "", "关闭活动页不得留下已销毁的 Tcl 菜单"


# ---------------------------------------------------------------------------
# 关闭生命周期：真实 SuiteApp
# ---------------------------------------------------------------------------
def _app_with_tools(root, mode, *, close_delay_ms=100):
    """构造带两个工具页的 SuiteApp；one 的关闭行为由 mode 决定。"""
    app = SuiteApp(root)
    calls = []

    def instantiate(spec, host):
        calls.append(spec.tool_id)
        if spec.tool_id == "one":
            if mode == "cancel":
                host.protocol("WM_DELETE_WINDOW", lambda: None)
            elif mode == "async":
                host.protocol("WM_DELETE_WINDOW", lambda: host.after(close_delay_ms, host.destroy))
            else:
                host.protocol("WM_DELETE_WINDOW", host.destroy)
        else:
            host.protocol("WM_DELETE_WINDOW", host.destroy)
        return SimpleNamespace(root=host)

    app._instantiate = instantiate
    spec = lambda name: SimpleNamespace(  # noqa: E731
        available=True, run_mode="embed", tool_id=name, name=name, tab_label=name)
    app.open_tool(spec("one"))
    app.open_tool(spec("two"))
    root.update()
    return app, calls


def test_cancel_close_keeps_tab_and_registration(root):
    """用户在确认框选取消：标签页与登记必须原样保留。"""
    app, _ = _app_with_tools(root, "cancel")
    one = app._open_tools["one"][0]
    app._close_tab_at(app.notebook.index(one))
    root.update()
    assert one.winfo_exists(), "取消关闭后部件必须仍在"
    assert "one" in app._open_tools, "取消关闭后登记必须保留"
    assert str(one) in app.notebook.tabs()


def test_synchronous_close_removes_only_target_tab(root):
    """同步 destroy：只关目标页，绝不能误删邻页标签。"""
    app, _ = _app_with_tools(root, "sync")
    one = app._open_tools["one"][0]
    two = app._open_tools["two"][0]
    app._close_tab_at(app.notebook.index(one))
    root.update()
    assert not one.winfo_exists()
    assert "one" not in app._open_tools
    assert two.winfo_exists() and str(two) in app.notebook.tabs(), (
        "关闭一页不得摘掉邻页标签（回归：forget(index) 会误删后一页）")


def test_async_close_keeps_tab_until_tool_destroys(root):
    """延迟收尾：部件与登记保留到工具自己销毁为止，避免截断写出。"""
    app, _ = _app_with_tools(root, "async", close_delay_ms=120)
    one = app._open_tools["one"][0]
    app._close_tab_at(app.notebook.index(one))
    assert one.winfo_exists(), "工具尚未销毁时不得提前摘除"
    assert "one" in app._open_tools

    end = time.monotonic() + 0.6
    while time.monotonic() < end and "one" in app._open_tools:
        root.update()
        time.sleep(0.01)
    assert "one" not in app._open_tools, "工具销毁后登记必须清理"
    assert not one.winfo_exists()


def test_tool_destroy_then_reopen_same_id(root):
    """工具自行 destroy 后，同 ID 必须能重新构造出新实例。"""
    app, calls = _app_with_tools(root, "sync")
    one = app._open_tools["one"][0]
    one.destroy()          # 工具自己销毁（等价于点它自己的退出）
    root.update()
    assert "one" not in app._open_tools, "工具自毁后登记不得残留"

    spec = SimpleNamespace(available=True, run_mode="embed", tool_id="one",
                           name="one", tab_label="one")
    app.open_tool(spec)    # 不得抛 TclError
    root.update()
    assert calls.count("one") == 2
    new_one = app._open_tools["one"][0]
    assert new_one is not one and new_one.winfo_exists()


def test_app_close_respects_cancel_and_keeps_waiting(disposable_root):
    """程序级关闭：工具拒绝退出时必须继续等待，不得立刻销毁 root。

    走真实 ``host.protocol`` 注册的处理器（不 mock ``run_close_callbacks``），
    确保被测的就是生产路径。
    """
    app, _ = _app_with_tools(disposable_root, "cancel")
    app.on_app_close()
    disposable_root.update()
    assert disposable_root.winfo_exists(), (
        "工具未完成清理时不得无条件结束（会截断写文件）")
    assert app._closing_hosts, "等待集合必须被冻结保留，不能在下一轮变空"

    # 收尾：工具自己完成清理（等价于它稍后 destroy 自己），全部完成后才关窗
    for host in list(app._closing_hosts):
        if host.winfo_exists():
            host.destroy()
    _pump_until(disposable_root, lambda: _root_gone(disposable_root))


def test_app_close_waits_for_async_tool_then_exits(disposable_root):
    app, _ = _app_with_tools(disposable_root, "async", close_delay_ms=120)
    app.on_app_close()
    end = time.monotonic() + 0.8
    while time.monotonic() < end:
        try:
            if not disposable_root.winfo_exists():
                break
            disposable_root.update()
        except tk.TclError:
            break            # root 已销毁：正是期望结果
        time.sleep(0.01)
    with pytest.raises(tk.TclError):
        disposable_root.winfo_exists()


def test_app_close_frozen_wait_set_is_not_emptied(disposable_root):
    """回归：主源码先 clear(_open_tools) 再 after 调自己，下一轮等待集合为空。"""
    app, _ = _app_with_tools(disposable_root, "cancel")
    app.on_app_close()
    frozen = list(app._closing_hosts)
    assert frozen, "首次调用必须冻结等待集合"
    app.on_app_close()               # 第二轮：集合不得被重新求值成空
    assert app._closing_hosts == frozen


# ---------------------------------------------------------------------------
# 关闭边界：超时中止、处理器异常、无处理器、退出期间禁止新任务
# ---------------------------------------------------------------------------
def _patch_close_timeout(monkeypatch, seconds):
    import temsuite.app as module

    monkeypatch.setattr(module, "_CLOSE_WAIT_TIMEOUT_S", seconds)


def _root_gone(root) -> bool:
    """root 是否已被销毁（销毁后 winfo 调用会抛 TclError，也算已销毁）。"""
    try:
        return not root.winfo_exists()
    except tk.TclError:
        return True


def _pump_until(root, predicate, timeout=3.0):
    """泵事件循环直到 predicate 成立或超时（root 已销毁时提前结束）。

    超时判定与提示发生在 ``after`` 回调里，只调用一次 ``on_app_close()``
    不会触发它，必须让事件循环跑起来。
    """
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            if predicate():
                return True
            root.update()
        except tk.TclError:
            return predicate()
        time.sleep(0.005)
    return predicate()


def test_app_close_timeout_aborts_and_keeps_window(disposable_root, monkeypatch):
    """超时必须**中止本次退出**并保留窗口，不得无条件 destroy（回归 round10）。"""
    import temsuite.app as module

    errors = []
    monkeypatch.setattr(module.messagebox, "showerror",
                        lambda *a, **k: errors.append(a))
    _patch_close_timeout(monkeypatch, 0.05)

    app, _ = _app_with_tools(disposable_root, "cancel")
    app.on_app_close()
    _pump_until(disposable_root, lambda: bool(errors))

    assert disposable_root.winfo_exists(), "取消/超时后不得销毁主窗口"
    assert "one" in app._open_tools, "仍在收尾的工具必须保留登记"
    assert errors, "必须提示用户仍有工具未关闭"
    assert app._closing_hosts is None, "中止后必须允许安全重试"


def test_app_close_handler_error_keeps_page_and_reports(disposable_root, monkeypatch):
    """处理器抛异常不得被吞掉后强退：必须记录根因、提示并保留该页。

    用的是真实 ``host.protocol`` 注册的处理器，异常必须能穿透
    ``ToolHost.run_close_callbacks`` 传到协调器（该路径此前被 suppress 吞掉）。
    """
    import temsuite.app as module

    errors = []
    monkeypatch.setattr(module.messagebox, "showerror",
                        lambda *a, **k: errors.append(a))
    _patch_close_timeout(monkeypatch, 0.05)

    app = SuiteApp(disposable_root)

    def instantiate(spec, host):
        def bad():
            raise RuntimeError("SYNTHETIC_CLEANUP_FAILURE")
        host.protocol("WM_DELETE_WINDOW", bad)
        return object()

    app._instantiate = instantiate
    app.open_tool(SimpleNamespace(available=True, run_mode="embed", tool_id="one",
                                 name="one", tab_label="one"))
    disposable_root.update()
    host = app._open_tools["one"][0]

    app.on_app_close()
    _pump_until(disposable_root, lambda: bool(errors))
    assert host.winfo_exists(), "处理器异常后必须保留该页"
    assert "one" in app._open_tools
    assert disposable_root.winfo_exists()
    assert errors, "必须把未关闭原因告知用户"
    joined = "\n".join(str(e) for e in errors)
    assert "SYNTHETIC_CLEANUP_FAILURE" in joined, (
        "提示必须包含处理器异常根因，不能只说'未完成清理'")
    assert "one" in joined, "提示必须点名出错的工具"


def test_run_close_callbacks_propagates_handler_exception(root):
    """ToolHost.run_close_callbacks 必须让 WM_DELETE_WINDOW 异常向上传播。"""
    nb, h1, _h2, _e1, _e2 = _hosts(root)

    def bad():
        raise RuntimeError("ROOT_CAUSE")

    h1.protocol("WM_DELETE_WINDOW", bad)
    with pytest.raises(RuntimeError, match="ROOT_CAUSE"):
        h1.run_close_callbacks()


def test_run_close_callbacks_ignores_other_protocol_names(root):
    """只调用 WM_DELETE_WINDOW，不得顺带跑其它 protocol 名目。"""
    nb, h1, _h2, _e1, _e2 = _hosts(root)
    calls = []
    h1.protocol("WM_DELETE_WINDOW", lambda: calls.append("delete"))
    h1.protocol("WM_TAKE_FOCUS", lambda: calls.append("focus"))
    h1.run_close_callbacks()
    assert calls == ["delete"]


def test_single_tab_close_handler_error_is_reported(root, monkeypatch):
    """单页关闭走真实处理器异常时也必须提示，不能静默吞掉。"""
    import temsuite.app as module

    errors = []
    monkeypatch.setattr(module.messagebox, "showerror",
                        lambda *a, **k: errors.append(a))
    app = SuiteApp(root)

    def instantiate(spec, host):
        def bad():
            raise RuntimeError("PAGE_CLOSE_BOOM")
        host.protocol("WM_DELETE_WINDOW", bad)
        return object()

    app._instantiate = instantiate
    app.open_tool(SimpleNamespace(available=True, run_mode="embed", tool_id="one",
                                 name="one", tab_label="one"))
    root.update()
    host = app._open_tools["one"][0]

    app._close_tab_at(app.notebook.index(host))
    assert host.winfo_exists(), "处理器出错时必须保留该页"
    assert "one" in app._open_tools
    assert errors, "单页关闭失败必须提示"
    assert "PAGE_CLOSE_BOOM" in str(errors[0])


# ---------------------------------------------------------------------------
# 关闭重入：连续/排队关闭不得产生第二条 after 链
# ---------------------------------------------------------------------------
def test_double_close_request_runs_handler_once(disposable_root, monkeypatch):
    """连续两次 on_app_close 不得重复调用处理器（回归 round15）。"""
    import temsuite.app as module

    monkeypatch.setattr(module.messagebox, "showerror", lambda *a, **k: None)
    _patch_close_timeout(monkeypatch, 0.05)
    app, calls = _app_with_tools(disposable_root, "cancel")

    app.on_app_close()
    app.on_app_close()          # 第二次：必须被忽略
    _pump_until(disposable_root, lambda: app._closing_hosts is None)

    assert calls.count("one") == 1, f"处理器被调用 {calls.count('one')} 次"
    assert disposable_root.winfo_exists()
    assert app._closing_hosts is None, "中止后状态必须清空以便安全重试"


def test_queued_close_request_does_not_start_second_chain(disposable_root, monkeypatch):
    """排队到达的关闭请求不得在事件循环里重入并再跑一遍处理器。"""
    import temsuite.app as module

    monkeypatch.setattr(module.messagebox, "showerror", lambda *a, **k: None)
    _patch_close_timeout(monkeypatch, 0.05)
    app, calls = _app_with_tools(disposable_root, "cancel")

    disposable_root.after(0, app.on_app_close)   # 排队请求
    app.on_app_close()                           # 同时的显式请求
    _pump_until(disposable_root, lambda: app._closing_hosts is None)

    assert calls.count("one") == 1, f"处理器被调用 {calls.count('one')} 次"
    assert disposable_root.winfo_exists()


def test_expired_close_poll_does_not_restart_close(disposable_root, monkeypatch):
    """过期的轮询回调不得启动新的关闭流程。"""
    import temsuite.app as module

    monkeypatch.setattr(module.messagebox, "showerror", lambda *a, **k: None)
    _patch_close_timeout(monkeypatch, 0.05)
    app, calls = _app_with_tools(disposable_root, "cancel")

    app.on_app_close()
    _pump_until(disposable_root, lambda: app._closing_hosts is None)
    before = calls.count("one")

    app._poll_app_close()        # 手工触发一个已过期的轮询
    disposable_root.update()
    assert calls.count("one") == before, "过期轮询不得重新调用处理器"
    assert app._closing_hosts is None


def test_at_most_one_close_poll_scheduled(disposable_root, monkeypatch):
    """任一时刻至多存在一条关闭轮询链（唯一 after_id）。"""
    import temsuite.app as module

    monkeypatch.setattr(module.messagebox, "showerror", lambda *a, **k: None)
    _patch_close_timeout(monkeypatch, 30.0)
    app, _ = _app_with_tools(disposable_root, "cancel")

    app.on_app_close()
    assert app._close_after_id is not None, "应已安排唯一轮询"
    first = app._close_after_id
    app._schedule_close_poll()          # 重复安排必须替换，而不是叠加
    assert app._close_after_id is not None

    pending = [a for a in disposable_root.tk.call("after", "info")]
    assert len(pending) >= 1            # 至少有我们自己的轮询
    app._cancel_close_poll()
    assert app._close_after_id is None
    assert first is not None


def test_app_close_closes_page_without_handler_immediately(disposable_root, monkeypatch):
    """无 WM_DELETE_WINDOW 处理器的页应立即正常关闭，而不是等到超时。"""
    import temsuite.app as module

    _patch_close_timeout(monkeypatch, 30.0)   # 长超时：若在等待就会卡住
    app = SuiteApp(disposable_root)
    app._instantiate = lambda spec, host: object()   # 不注册任何关闭处理器
    app.open_tool(SimpleNamespace(available=True, run_mode="embed", tool_id="one",
                                 name="one", tab_label="one"))
    disposable_root.update()

    started = time.monotonic()
    app.on_app_close()
    elapsed = time.monotonic() - started
    assert elapsed < 5.0, "无处理器页不得走超时等待路径"
    with pytest.raises(tk.TclError):
        disposable_root.winfo_exists()      # root 已关闭


def test_new_tool_blocked_during_close(disposable_root, monkeypatch):
    """退出流程进行中不得打开新工具，否则新页不在待关闭集合内。"""
    import temsuite.app as module

    _patch_close_timeout(monkeypatch, 0.05)
    app, calls = _app_with_tools(disposable_root, "cancel")
    app.on_app_close()
    disposable_root.update()

    spec = SimpleNamespace(available=True, run_mode="embed", tool_id="three",
                           name="three", tab_label="three")
    app.open_tool(spec)
    assert "three" not in app._open_tools, "退出期间不得打开新工具"
    assert "three" not in calls, "不得真正构造新工具"


def test_new_subprocess_blocked_during_close(disposable_root, monkeypatch):
    """退出流程进行中不得启动新子进程。"""
    import temsuite.app as module

    _patch_close_timeout(monkeypatch, 0.05)
    app, _ = _app_with_tools(disposable_root, "cancel")
    app.on_app_close()
    disposable_root.update()

    launched = []
    monkeypatch.setattr(module.subprocess, "Popen",
                        lambda *a, **k: launched.append(a) or pytest.fail(
                            "退出期间不得启动子进程"))
    spec = SimpleNamespace(tool_id="sp", name="sp", project_dir=PROJECT,
                           extra_paths=(), subprocess_module="", subprocess_args=(),
                           entry="run.py", resolve_python_exe=lambda: Path(sys.executable))
    app._launch_subprocess(spec)
    assert not launched


def test_app_close_retry_does_not_repeat_callbacks(disposable_root, monkeypatch):
    """中止后重试必须安全：不得重复调用关闭处理器（避免二次弹框/二次写配置）。"""
    import temsuite.app as module

    monkeypatch.setattr(module.messagebox, "showerror", lambda *a, **k: None)
    _patch_close_timeout(monkeypatch, 0.05)

    app = SuiteApp(disposable_root)
    calls = []

    def instantiate(spec, host):
        host.protocol("WM_DELETE_WINDOW", lambda: calls.append(spec.tool_id))
        return object()

    app._instantiate = instantiate
    app.open_tool(SimpleNamespace(available=True, run_mode="embed", tool_id="one",
                                 name="one", tab_label="one"))
    disposable_root.update()

    app.on_app_close()
    assert calls == ["one"]
    app.on_app_close()          # 重试：第一轮之后不再重复回调
    assert calls == ["one"], "重试不得重复调用关闭处理器"


def test_app_close_completes_when_all_hosts_destroy(disposable_root, monkeypatch):
    """所有 host 真正完成后才关闭 root（全部完成即关闭）。"""
    import temsuite.app as module

    _patch_close_timeout(monkeypatch, 5.0)
    app = SuiteApp(disposable_root)

    def instantiate(spec, host):
        # 延迟到事件循环里销毁自己，模拟正常的异步收尾
        host.protocol("WM_DELETE_WINDOW", lambda: host.after(30, host.destroy))
        return object()

    app._instantiate = instantiate
    app.open_tool(SimpleNamespace(available=True, run_mode="embed", tool_id="one",
                                 name="one", tab_label="one"))
    disposable_root.update()

    app.on_app_close()
    end = time.monotonic() + 3.0
    while time.monotonic() < end:
        try:
            if not disposable_root.winfo_exists():
                break
            disposable_root.update()
        except tk.TclError:
            break
        time.sleep(0.01)
    with pytest.raises(tk.TclError):
        disposable_root.winfo_exists()


# ---------------------------------------------------------------------------
# 源码模式缺工具环境：明确报缺失并停止启动，不回退 Suite Python
# ---------------------------------------------------------------------------
def test_missing_tool_env_stops_launch_without_fallback(root, monkeypatch):
    """缺 .venv 时必须报环境缺失并停止，不得用套件解释器顶替（回归 round10）。"""
    import temsuite.app as module

    errors = []
    monkeypatch.setattr(module.messagebox, "showerror",
                        lambda *a, **k: errors.append(a))
    launched = []
    monkeypatch.setattr(module.subprocess, "Popen",
                        lambda *a, **k: launched.append(a) or pytest.fail(
                            "缺环境时不得尝试启动"))

    app = SuiteApp(root)
    spec = SimpleNamespace(
        tool_id="missing", name="missing", project_dir=PROJECT, extra_paths=(),
        subprocess_module="does_not_exist", subprocess_args=(),
        entry="run.py", resolve_python_exe=lambda: None,
    )
    app._launch_subprocess(spec)

    assert not launched, "缺工具环境时不得调用 Popen"
    assert errors, "必须告知用户环境缺失"
    assert "环境缺失" in errors[0][0]


def test_frozen_runtime_branch_args_preserved(root, monkeypatch):
    """冻结 EXE 的打包运行时分支保留：以 --run-script/--run-module 委派。"""
    import temsuite.app as module

    recorded = {}

    class _FakeProc:
        returncode = 0

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        recorded["cmd"] = cmd
        recorded["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(module.sys, "frozen", True, raising=False)
    monkeypatch.setattr(module, "tempfile", module.tempfile)

    app = SuiteApp(root)
    base = dict(tool_id="atom_annotator", name="atom", project_dir=PROJECT,
                extra_paths=(), subprocess_args=(), resolve_python_exe=lambda: None)

    app._launch_subprocess(SimpleNamespace(**base, subprocess_module="atom_center.annotator_gui",
                                          entry="src\\atom_center\\annotator_gui.py"))
    assert recorded["cmd"][1] == "--run-module"
    assert recorded["cmd"][2] == "atom_center.annotator_gui"

    recorded.clear()
    app._launch_subprocess(SimpleNamespace(**base, subprocess_module="", entry="run.py"))
    assert recorded["cmd"][1] == "--run-script"
    assert recorded["cmd"][2] == str(PROJECT / "run.py")


def test_launch_failure_closes_log_handle(root, monkeypatch):
    """启动失败必须关闭已创建的日志句柄，不能泄漏。"""
    import temsuite.app as module

    created = []
    real_tempfile = module.tempfile.TemporaryFile

    def tracking_tempfile(*a, **k):
        handle = real_tempfile(*a, **k)
        created.append(handle)
        return handle

    monkeypatch.setattr(module.tempfile, "TemporaryFile", tracking_tempfile)
    monkeypatch.setattr(module.subprocess, "Popen",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    monkeypatch.setattr(module.messagebox, "showerror", lambda *a, **k: None)

    app = SuiteApp(root)
    spec = SimpleNamespace(tool_id="t", name="t", project_dir=PROJECT, extra_paths=(),
                           subprocess_module="", subprocess_args=(), entry="run.py",
                           resolve_python_exe=lambda: Path(sys.executable))
    app._launch_subprocess(spec)

    assert created, "应当先创建日志句柄"
    assert created[0].closed, "启动失败后必须关闭日志句柄"


# ---------------------------------------------------------------------------
# 子进程日志：退出后只读有界 tail；并行退出互不干扰
# ---------------------------------------------------------------------------
def _ended_failing_proc(root, text):
    """起一个已结束的非 0 子进程，把 ``text`` 写进它的捕获文件。"""
    log_file = _capture_file()
    proc = subprocess.Popen(
        [sys.executable, "-c", f"import sys; sys.stdout.write({text!r}); sys.exit(3)"],
        stdout=log_file, stderr=subprocess.STDOUT)
    proc.wait()
    return proc, log_file


def test_two_parallel_failed_subprocesses_report_their_own_tail(root, monkeypatch):
    """两个已结束的非 0 子进程各自回报自己的 tail，互不覆盖（回归 round10）。"""
    import temsuite.app as module

    messages = []
    monkeypatch.setattr(module.messagebox, "showerror",
                        lambda *a, **k: messages.append(a[-1]))

    app = SuiteApp(root)
    p1, f1 = _ended_failing_proc(root, "FIRST-TOOL-OUTPUT")
    p2, f2 = _ended_failing_proc(root, "SECOND-TOOL-OUTPUT")
    app._subprocesses.extend([p1, p2])
    app._subprocess_logs[p1] = f1
    app._subprocess_logs[p2] = f2

    app._watch_subprocess(p1, SimpleNamespace(tool_id="t1", name="t1"))
    # 第一个 watch 不得关掉第二个仍在等待 watch 的句柄
    assert p2 in app._subprocess_logs, (
        "_reap_subprocesses 不得关闭另一个待 watch 子进程的日志")

    app._watch_subprocess(p2, SimpleNamespace(tool_id="t2", name="t2"))
    joined = "\n".join(messages)
    assert "FIRST-TOOL-OUTPUT" in joined, "第一个子进程的 tail 必须被回报"
    assert "SECOND-TOOL-OUTPUT" in joined, "第二个子进程的 tail 必须被回报"
    _discard_capture(f1)
    _discard_capture(f2)


def test_tail_read_is_bounded(root):
    """tail 读取必须有上限，不能把整个日志读进内存。"""
    app = SuiteApp(root)
    log_file = _capture_file()
    log_file.write("PREFIX-" + "x" * 50000 + "-SUFFIX")
    log_file.flush()

    class _Proc:
        returncode = 1

        def poll(self):
            return 1

    proc = _Proc()
    app._subprocess_logs[proc] = log_file
    tail = app._read_subprocess_tail(proc, limit=200)
    assert len(tail) <= 200, "tail 必须受 limit 约束"
    assert tail.endswith("-SUFFIX"), "tail 必须包含日志末尾"
    assert "PREFIX" not in tail, "tail 不应包含超出上限的开头部分"
    _discard_capture(log_file)


def test_real_launch_with_large_output_terminates(root, tmp_path, monkeypatch):
    """真实 _launch_subprocess 路径：大量输出的子进程必须能结束（无管道卡死）。

    脚本写在 ``tmp_path``（不污染源码 tests/），并让 ``project_dir`` 指向该临时
    目录，从而走的是真实 ``_launch_subprocess`` 接线（命令拼装、日志文件重定向、
    ``after`` 轮询），而不是测试自造的旁路。
    """
    import temsuite.app as module

    monkeypatch.setattr(module.messagebox, "showerror", lambda *a, **k: None)
    app = SuiteApp(root)

    script = tmp_path / "noisy_child.py"
    script.write_text(
        "import sys\nsys.stdout.write('y' * 200000)\nsys.stdout.flush()\nsys.exit(0)\n",
        encoding="utf-8")

    noisy_spec = SimpleNamespace(
        tool_id="noisy", name="noisy", project_dir=tmp_path, extra_paths=(),
        subprocess_module="", subprocess_args=(), entry="noisy_child.py",
        resolve_python_exe=lambda: Path(sys.executable),
    )
    app._launch_subprocess(noisy_spec)
    assert app._subprocesses, "应已启动子进程"
    proc = app._subprocesses[0]
    end = time.monotonic() + 30
    while time.monotonic() < end and proc.poll() is None:
        root.update()
        time.sleep(0.02)
    assert proc.poll() is not None, "大量输出不得导致子进程永不退出"
    assert proc.returncode == 0
    app._watch_subprocess(proc, noisy_spec)
    assert proc not in app._subprocesses


# ---------------------------------------------------------------------------
# loader：标记对象按身份清理
# ---------------------------------------------------------------------------
def test_loader_marker_survives_cached_str_and_foreign_inserts():
    """同一 Path 复用的缓存 str + 外部插入都必须保留（回归 round10）。"""
    from temsuite.loader import ProjectLoader

    loader = ProjectLoader()
    saved = list(sys.path)
    target = PROJECT
    cached = str(target)                 # str(Path) 有缓存，可能多次返回同一对象
    assert cached is str(target)
    foreign = "FOREIGN_MARKER"
    try:
        for _ in range(3):               # 同一值插入多次，验证按身份精确清理
            sys.path[:] = saved
            with loader._project_on_path(target):
                sys.path.insert(0, cached)      # 外部插入同值缓存 str
                sys.path.insert(1, foreign)
            assert sys.path == [cached, foreign] + saved, (
                f"实际前缀 {sys.path[:3]}")
    finally:
        sys.path[:] = saved


def test_loader_nested_and_exception_restore_exactly():
    """嵌套与异常路径都必须精确还原，不残留也不误删。"""
    from temsuite.loader import ProjectLoader

    loader = ProjectLoader()
    saved = list(sys.path)
    target = PROJECT
    try:
        with pytest.raises(RuntimeError):
            with loader._project_on_path(target):
                with loader._project_on_path(target, (target,)):
                    raise RuntimeError("boom")
        assert sys.path == saved, f"异常路径未还原: {sys.path[:4]}"

        with loader._project_on_path(target):
            pass
        assert sys.path == saved, f"单层未还原: {sys.path[:4]}"
    finally:
        sys.path[:] = saved


def test_loader_preexisting_duplicate_preserved():
    """进入前就存在的同值条目（缓存 str）不得被当成注入项删除。"""
    from temsuite.loader import ProjectLoader

    loader = ProjectLoader()
    saved = list(sys.path)
    cached = str(PROJECT)
    try:
        sys.path.insert(0, cached)          # 进入前已存在
        before = list(sys.path)
        with loader._project_on_path(PROJECT):
            pass
        assert sys.path == before, "进入前的同值条目被误删"
        assert sys.path.count(cached) == before.count(cached)
    finally:
        sys.path[:] = saved


# ---------------------------------------------------------------------------
# 宿主能力：内嵌禁止结束进程
# ---------------------------------------------------------------------------
def test_toolhost_reports_embedded_capability(root):
    nb, h1, _h2, _e1, _e2 = _hosts(root)
    assert h1.is_embedded() is True
    assert h1.embedded_in_suite is True
    assert h1.can_terminate_process() is False, "内嵌宿主不得允许结束进程"


def test_standalone_root_has_no_embedded_marker(root):
    assert not hasattr(root, "can_terminate_process")
    assert getattr(root, "embedded_in_suite", False) is False


@pytest.mark.parametrize("tool_id,rel_dir,module,cls_name", [
    ("stem_optimize", r"02-图像处理\stem-optimize-STEM图像优化", "main_window", "MainWindow"),
    ("drift_correct", r"02-图像处理\drift-correction-v7", "drift_correction", "DriftCorrectionApp"),
])
def test_embedded_force_close_does_not_exit_process(
        root, monkeypatch, tool_id, rel_dir, module, cls_name):
    """内嵌时即使选"强制关闭"也绝不调用 os._exit（mock，不真结束进程）。"""
    from temsuite.loader import ProjectLoader

    loader = ProjectLoader()
    nb = _notebook(root)
    host = ToolHost(nb, tool_id=tool_id)
    nb.add(host, text=tool_id)

    mod = loader.load_module(tool_id, PROJECT.parent / rel_dir, module)
    cls = getattr(mod, cls_name)
    app = cls.__new__(cls)
    app.root = host
    app._close_deadline = 0
    if tool_id == "stem_optimize":
        app._background_tasks_active = lambda: True
        app._poll_after_id = None
        app._reader = None
    else:
        from unittest.mock import Mock
        app.worker = Mock()
        app.worker.is_alive.return_value = True
        app._close_force_prompted = False
        app.status_var = tk.StringVar(master=root)

    with monkeypatch.context() as ctx:
        ctx.setattr(mod.os, "_exit", lambda *a, **k: pytest.fail(
            f"{tool_id} 在内嵌模式下调用了 os._exit"))
        ctx.setattr(mod.messagebox, "askyesno", lambda *a, **k: True)
        app._finish_close()          # 不得抛异常，也不得请求结束进程


def test_standalone_force_close_still_exits(monkeypatch):
    """独立运行（真实 Tk root）时保留原退出语义：允许 os._exit。"""
    from unittest.mock import Mock
    from temsuite.loader import ProjectLoader

    loader = ProjectLoader()
    mod = loader.load_module(
        "drift_correct", PROJECT.parent / r"02-图像处理\drift-correction-v7",
        "drift_correction")
    cls = getattr(mod, "DriftCorrectionApp")
    standalone = tk.Tk()
    standalone.withdraw()
    try:
        app = cls.__new__(cls)
        app.root = standalone           # 真正的 Tk root，无宿主能力标记
        app._close_deadline = 0
        app.worker = Mock()
        app.worker.is_alive.return_value = True
        app._close_force_prompted = False
        app.status_var = tk.StringVar(master=standalone)

        called = []
        with monkeypatch.context() as ctx:
            ctx.setattr(mod.os, "_exit", lambda code=0: called.append(code))
            ctx.setattr(mod.messagebox, "askyesno", lambda *a, **k: True)
            app._finish_close()
        assert called == [0], "独立运行时必须保留 os._exit 退出语义"
    finally:
        try:
            standalone.destroy()
        except tk.TclError:
            pass


# ---------------------------------------------------------------------------
# 注册表：扁平与 pythonProject 两种布局
# ---------------------------------------------------------------------------
def test_registry_prefers_flat_layout(root, tmp_path, monkeypatch):
    """候选扁平布局必须优先解析（不得因回退逻辑改变现有行为）。"""
    import importlib
    import temsuite.registry as registry

    flat = tmp_path / "04-统计分析" / "非晶面积统计"
    (flat / "gui").mkdir(parents=True)
    (flat / "gui" / "app.py").write_text("", encoding="utf-8")
    (tmp_path / "01-视频与数据提取").mkdir()
    monkeypatch.setenv("TEMSUITE_WORKSPACE", str(tmp_path))
    reloaded = importlib.reload(registry)
    try:
        spec = reloaded.get_tool("amorphous_area")
        assert spec.project_dir == flat.resolve()
        assert spec.available
    finally:
        monkeypatch.delenv("TEMSUITE_WORKSPACE", raising=False)
        importlib.reload(registry)


def test_registry_falls_back_to_pythonproject_layout(tmp_path, monkeypatch):
    """主源码布局（多一层 pythonProject）必须也能被发现，无需搬迁目录。"""
    import importlib
    import temsuite.registry as registry

    nested = tmp_path / "04-统计分析" / "非晶面积统计" / "pythonProject"
    (nested / "gui").mkdir(parents=True)
    (nested / "gui" / "app.py").write_text("", encoding="utf-8")
    (tmp_path / "01-视频与数据提取").mkdir()
    monkeypatch.setenv("TEMSUITE_WORKSPACE", str(tmp_path))
    reloaded = importlib.reload(registry)
    try:
        spec = reloaded.get_tool("amorphous_area")
        assert spec.project_dir == nested.resolve()
        assert spec.available
    finally:
        monkeypatch.delenv("TEMSUITE_WORKSPACE", raising=False)
        importlib.reload(registry)


# ---------------------------------------------------------------------------
# SuiteApp 必须只使用调用方传入的 root（回归 2026-10-08：CI 隐式 root 缺陷）
# ---------------------------------------------------------------------------
def test_suite_app_uses_only_the_supplied_root(root, monkeypatch):
    """构造 SuiteApp 时不得依赖 tkinter 的隐式默认 root。

    反例（修复前的真实失败）：``disposable_root`` 用例结束后
    ``tk._default_root`` 被清空，随后借用共用 root 的用例构造 SuiteApp 时，
    ``ttk.Style()`` / ``tkfont.nametofont`` / ``tk.StringVar`` 会去创建**新的**
    Tcl 解释器；在 CI 上那一步直接抛
    ``_tkinter.TclError: Can't find a usable init.tcl``，
    即使传入的 root 仍然存活。

    这里把隐式创建路径彻底堵死：任何 ``tk.Tk()`` 都确定性失败，且
    ``tk._default_root`` 置空。修复后 SuiteApp 只用传入的解释器，
    构造成功且样式/变量都落在该解释器上。
    """
    def _forbid_implicit_tk(*args, **kwargs):
        raise AssertionError(
            "SuiteApp 不得创建隐式默认 root；必须使用调用方传入的 root")

    monkeypatch.setattr(tk, "Tk", _forbid_implicit_tk)
    # 用 monkeypatch 而不是直接赋值：本用例结束后自动恢复这个进程级全局属性，
    # 不给后续用例留下副作用（模拟 disposable_root 之后的残留状态）。
    monkeypatch.setattr(tk, "_default_root", None)

    app = SuiteApp(root)             # 只能用传入的 root，否则上面的 AssertionError

    # 变量必须挂在传入的解释器上（而不是另起的隐式 root）
    assert app.desc_var._root is root
    assert app.status_var._root is root
    assert app.desc_var.get() == "双击工具名称即可打开。"
    assert app.status_var.get() == "就绪"

    # 样式配置必须真的落到传入 root 的 ttk 样式库里
    rowheight = root.tk.call("ttk::style", "configure", "Suite.Treeview", "-rowheight")
    assert rowheight, "Suite.Treeview 的 rowheight 未配置到传入的 root 上"
    assert int(rowheight) > 0
