"""Tk 根窗口代理 —— 让独立 GUI 工具无改动地嵌入标签页。

背景
----
被整合的 13 个工具原本都是独立程序，其 GUI 类要么接收一个 ``tk.Tk`` 根窗口，
要么在 ``__init__`` 内部自建 ``tk.Tk()``。而一个 Python 进程只允许存在一个
Tk 根窗口，因此无法直接把它们塞进同一个主窗口。

解决方式
--------
``ToolHost`` 继承 ``ttk.Frame``（可作为标签页内容 pack 进 Notebook），
同时模拟出 Tk 根窗口独有的那部分接口。经统计，所有工具对 root 的调用共涉及
17 个方法，其中 9 个是 ``ttk.Frame`` 自带的（bind/after/config/destroy 等）可直接
透传，只有 8 类是 Tk/Wm 专有需要模拟的：

    title       -> 转为更新标签页标题
    geometry    -> 忽略（尺寸由主窗口统一管理）
    minsize     -> 记录但不生效
    mainloop    -> 忽略（主窗口已持有唯一事件循环）
    protocol    -> 注册为标签页关闭回调
    iconbitmap  -> 忽略
    state       -> 覆写，避免与 ttk 的部件状态语义冲突
    拖放接口    -> 转发给真实顶层窗口
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from contextlib import suppress
from tkinter import ttk
from typing import Any

# ttk.Frame 支持的配置项之外的 Tk 根窗口专有选项，config() 时需要剔除
_WM_ONLY_CONFIG_KEYS = frozenset({"menu", "screen", "use", "container", "colormap", "visual"})

# 分派器登记表挂在**目标部件自己**身上（而不是模块级字典）：Tk 部件的名字会
# 复用（每个新建的 Tk() 都叫 "."），用名字做全局键会让上一个已销毁 root 的
# 分派器被下一个 root 复用，从而绑定失败、事件全部丢失（实测复现）。
_DISPATCHER_ATTR = "_temsuite_binding_dispatchers"


class _BindingDispatcher:
    """某个 (顶层窗口, sequence) 上唯一的转发器，负责按页分派事件。

    每个标签页只在登记表里出现一次；摘除页面仅从登记表移除，**从不**操作 Tcl
    绑定脚本 —— 因为 tkinter 的 ``unbind(sequence)`` 会清空该 sequence 的**整段**
    脚本，宿主 root 自己的绑定会被一起删掉（实测复现）。因此分派器一旦装上就
    留着，没有登记页时只是一个极廉价的空转。
    """

    def __init__(self, target: tk.Misc, sequence: str) -> None:
        self._target = target
        self._sequence = sequence
        self._key_hosts: list[ToolHost] = []
        self._all_hosts: list[ToolHost] = []
        self._funcid: str | None = None

    # -- 登记 ---------------------------------------------------------------
    def register(self, host: "ToolHost", *, all_bindings: bool = False) -> None:
        hosts = self._all_hosts if all_bindings else self._key_hosts
        if host not in hosts:
            hosts.append(host)
        self._install()

    def unregister(self, host: "ToolHost", *, all_bindings: bool = False) -> None:
        """只从登记表移除；不触碰 Tcl 脚本，避免误删宿主 root 的绑定。"""
        hosts = self._all_hosts if all_bindings else self._key_hosts
        with suppress(ValueError):
            hosts.remove(host)

    def _installed(self) -> bool:
        """确认分派器仍在脚本里（宿主 root 可能用替换语义覆盖过它）。"""
        if self._funcid is None:
            return False
        try:
            return self._funcid in (self._target.bind(self._sequence) or "")
        except tk.TclError:
            return False

    def _install(self) -> None:
        if self._installed():
            return
        try:
            # add="+" 保留宿主 root 可能已有的绑定（不覆盖、不删除）
            self._funcid = str(self._target.bind(self._sequence, self._dispatch, add="+"))
        except tk.TclError:
            self._funcid = None

    # -- 分派 ---------------------------------------------------------------
    def _dispatch(self, event: tk.Event) -> str | None:
        # 键绑定与 bind_all 都只由**当前激活页**响应；回调按登记顺序调用。
        for host, all_bindings in (
            *((h, False) for h in list(self._key_hosts)),
            *((h, True) for h in list(self._all_hosts)),
        ):
            try:
                if not host._alive() or not host._is_active():
                    continue
            except tk.TclError:
                continue
            bindings = host._all_bindings if all_bindings else host._key_bindings
            for _funcid, callback in list(bindings.get(self._sequence, [])):
                with suppress(Exception):  # noqa: BLE001 - 单个回调失败不影响其他
                    if callback(event) == "break":
                        return "break"
        return None


def _dispatcher_for(target: tk.Misc, sequence: str) -> _BindingDispatcher:
    """取（必要时创建）该部件上该 sequence 的分派器。

    登记表存在目标部件对象上，随部件一起销毁，避免跨 root 串用。
    """
    registry = getattr(target, _DISPATCHER_ATTR, None)
    if registry is None:
        registry = {}
        try:
            setattr(target, _DISPATCHER_ATTR, registry)
        except (AttributeError, tk.TclError):  # pragma: no cover - 极端只读部件
            registry = {}
    dispatcher = registry.get(sequence)
    if dispatcher is None:
        dispatcher = _BindingDispatcher(target, sequence)
        registry[sequence] = dispatcher
    return dispatcher


# 常见的窗口管理方法，统一以「静默忽略」处理
_IGNORED_WM_METHODS = (
    "iconbitmap",
    "iconphoto",
    "iconname",
    "iconify",
    "deiconify",
    "withdraw",
    "overrideredirect",
    "maxsize",
    "aspect",
    "focusmodel",
    "group",
    "transient",
)


class ToolHost(ttk.Frame):
    """承载单个工具 GUI 的容器，对外伪装成 Tk 根窗口。

    Parameters
    ----------
    master:
        父级部件，通常是主窗口的 ``ttk.Notebook``。
    on_title_change:
        工具调用 ``root.title("...")`` 时触发，用于同步标签页文字。
    tool_id:
        工具标识，仅用于日志与错误提示。
    on_close_request:
        工具调用 ``root.destroy()``/``quit()`` 时触发，由主窗口执行统一的
        关页流程（清理登记 + 摘除标签页）。未提供时退化为直接销毁自身。
    on_menu_change:
        工具用 ``root.config(menu=...)`` 挂菜单栏时触发。主窗口据此在**本页
        处于激活状态**时把菜单挂到真实顶层窗口，并保存下来供切换时恢复。
    """

    def __init__(
        self,
        master: tk.Misc,
        *,
        on_title_change: Callable[[str], None] | None = None,
        tool_id: str = "",
        on_close_request: Callable[["ToolHost"], None] | None = None,
        on_menu_change: Callable[["ToolHost", tk.Menu | None], None] | None = None,
    ) -> None:
        super().__init__(master)
        self._tool_id = tool_id
        self._on_title_change = on_title_change
        self._on_close_request = on_close_request
        self._on_menu_change = on_menu_change
        self._title = ""
        self._minsize: tuple[int, int] | None = None
        self._close_callbacks: dict[str, Callable[[], Any]] = {}
        # 工具在 __init__ 里可能读写这些属性，预先备好避免 AttributeError
        self._real_toplevel = self.winfo_toplevel()
        # 本标签页登记的键绑定：sequence -> [(funcid, func)]（保持登记顺序）
        self._key_bindings: dict[str, list[tuple[str, Callable[..., Any]]]] = {}
        # 本标签页登记的 bind_all：sequence -> [(funcid, func)]
        self._all_bindings: dict[str, list[tuple[str, Callable[..., Any]]]] = {}
        self._bind_counter = 0
        self._closing = False
        # 主窗口正在处理本标签页的关闭流程（防止 destroy() 重入）
        self.close_in_progress = False
        # 本页的菜单栏（工具用 config(menu=...) 挂上来的）
        self.menu: tk.Menu | None = None
        # 宿主能力标记：本对象是 Suite 内嵌容器，而不是独立 Tk 根窗口。
        # 工具据此决定"强退"是否允许真正结束进程（见各工具 _finish_close）。
        self.embedded_in_suite = True

    # ------------------------------------------------------------------
    # 宿主能力：工具判断自己是否被内嵌
    # ------------------------------------------------------------------
    def is_embedded(self) -> bool:
        """本容器是否为 Suite 内嵌宿主（独立运行时工具拿到的是真正的 Tk）。"""
        return True

    def can_terminate_process(self) -> bool:
        """内嵌标签页**不允许**终止整个进程。

        工具的超时"强退"在独立运行时可以 ``os._exit``（进程只属于它自己），
        但在 Suite 里这样做会连带杀掉其他已打开工具，并可能截断正在写出的
        文件。宿主能力必须在 ``destroy()`` **之前**读取，destroy 之后就问不到了。
        """
        return False

    # ------------------------------------------------------------------
    # 标题：转为标签页文字
    # ------------------------------------------------------------------
    def title(self, string: str | None = None) -> str:
        if string is None:
            return self._title
        self._title = string
        if self._on_title_change is not None:
            self._on_title_change(string)
        return string

    def wm_title(self, string: str | None = None) -> str:
        return self.title(string)

    # ------------------------------------------------------------------
    # 几何：主窗口统一管理，此处仅记录
    # ------------------------------------------------------------------
    def geometry(self, newGeometry: str | None = None) -> str:
        if newGeometry is None:
            return self._real_toplevel.geometry()
        return newGeometry

    wm_geometry = geometry

    def minsize(self, width: int | None = None, height: int | None = None):
        if width is None and height is None:
            return self._minsize or (1, 1)
        self._minsize = (int(width or 1), int(height or 1))
        return self._minsize

    wm_minsize = minsize

    def resizable(self, width: bool | None = None, height: bool | None = None):
        if width is None and height is None:
            return (True, True)
        return (bool(width), bool(height))

    wm_resizable = resizable

    # ------------------------------------------------------------------
    # 事件循环：主窗口已持有唯一 mainloop
    # ------------------------------------------------------------------
    def mainloop(self, n: int = 0) -> None:  # noqa: ARG002 - 保持签名兼容
        return None

    def quit(self) -> None:
        """工具请求退出 —— 只关闭自己，不终止整个程序。"""
        self.run_close_callbacks()

    # ------------------------------------------------------------------
    # 关闭协议：注册为标签页关闭回调
    # ------------------------------------------------------------------
    def protocol(self, name: str | None = None, func: Callable[[], Any] | None = None):
        if name is None:
            return list(self._close_callbacks)
        if func is None:
            return self._close_callbacks.get(name)
        self._close_callbacks[name] = func
        return func

    wm_protocol = protocol

    def run_close_callbacks(self) -> None:
        """由主窗口在关闭标签页前调用，让工具执行自身的清理逻辑。

        只调用 ``WM_DELETE_WINDOW`` 这一个协议回调（工具就是在这条路径上做
        取消任务 / 关闭文件等收尾），不触碰其它 protocol 名目。

        回调抛出的异常**向上传播**给协调器：本方法不做 suppress —— 之前在这里
        吞掉异常，使 SuiteApp 永远收不到根因，只能报告"未完成清理"
        （回归 2026-10-03 R3）。调用方负责记录、提示并保留该页。
        """
        handler = self._close_callbacks.get("WM_DELETE_WINDOW")
        if handler is None:
            return
        handler()

    def destroy(self) -> None:  # type: ignore[override]
        """销毁本标签页。

        工具（十余个）在关闭按钮/退出路径上最终都执行 ``self.root.destroy()``。
        内嵌时 ``self.root`` 是本对象，若直接销毁 Frame，Notebook 会**自动移除**
        该标签页，而 ``SuiteApp._open_tools`` 的登记不会同步 —— 再次打开同一
        工具时会拿到失效部件并抛 TclError（回归 2026-09-29 P0）。

        因此有 ``on_close_request`` 时改为请求主窗口走统一的关页流程；主窗口
        再以 ``_destroy_now()`` 真正销毁，避免递归。
        """
        if self._on_close_request is not None and not self._closing:
            self._closing = True
            try:
                self._on_close_request(self)
            finally:
                self._closing = False
            return
        self._destroy_now()

    def _destroy_now(self) -> None:
        """真正销毁自身：先注销绑定转发器与菜单，再交给 ttk.Frame.destroy()。"""
        self._clear_bindings()
        self._detach_menu()
        super().destroy()

    # ------------------------------------------------------------------
    # 键绑定：转发到真实顶层窗口，按标签页隔离
    #
    # 为什么不用 tkinter 的 unbind(sequence, funcid)：它的实现是
    # ``self.tk.call('bind', w, sequence, '')`` **再** deletecommand —— 即先清空
    # 该 sequence 的整段脚本，只删一个 funcid 的语义根本没实现（实测：宿主 root
    # 自己的绑定与相邻标签页的转发器会被一起清掉）。
    #
    # 因此这里改为：每个 (顶层窗口, sequence) 只装**一个**共享分派器，分派器按
    # 登记表逐个询问各标签页。摘除某个标签页 = 从登记表移除它，完全不触碰 Tcl
    # 绑定脚本，因此宿主 root 与其他标签页的绑定都安然无恙。
    # ------------------------------------------------------------------
    def _is_active(self) -> bool:
        """本标签页是否为 Notebook 当前选中页（非 Notebook 场景恒为真）。"""
        parent = self.master
        if not isinstance(parent, ttk.Notebook):
            return True
        try:
            return parent.select() == str(self)
        except tk.TclError:
            return False

    def _alive(self) -> bool:
        try:
            return bool(self.winfo_exists())
        except tk.TclError:
            return False

    def _next_funcid(self) -> str:
        self._bind_counter += 1
        return f"host{self._bind_counter}"

    def _dispatcher(self, sequence: str) -> "_BindingDispatcher":
        return _dispatcher_for(self._real_toplevel, sequence)

    def bind(self, sequence=None, func=None, add=None):  # type: ignore[override]
        """把工具的快捷键绑定转发到真实顶层窗口。

        转发器只在本标签页**激活**时调用工具的回调，从而让十余个工具各自的
        快捷键互不干扰（Tk 的 bind 是替换语义，直接绑到顶层会互相顶掉）。

        语义与 Tk 对齐：
        * 未给 ``add`` 时**替换**本页在该 sequence 上已有的回调；
        * ``add="+"`` 时**追加**；
        * 回调按登记顺序调用（不是逆序）；
        * 任一回调返回 ``"break"`` 即停止后续回调。
        返回本页自己的 funcid，供 :meth:`unbind` 精确解绑。
        """
        if sequence is None or func is None or not callable(func):
            return super().bind(sequence, func, add)

        funcid = self._next_funcid()
        if str(add or "") == "+":
            self._key_bindings.setdefault(sequence, []).append((funcid, func))
        else:
            self._key_bindings[sequence] = [(funcid, func)]
        self._dispatcher(sequence).register(self)
        return funcid

    def unbind(self, sequence: str, funcid: str | None = None) -> None:  # type: ignore[override]
        """按 funcid 解绑；未给 funcid 时清除本页在该 sequence 上的全部回调。

        只影响**本页**的回调；宿主 root 与其他标签页的绑定不受影响。
        """
        if funcid is None:
            self._key_bindings.pop(sequence, None)
        else:
            remaining = [
                (fid, cb) for fid, cb in self._key_bindings.get(sequence, [])
                if fid != funcid
            ]
            if remaining:
                self._key_bindings[sequence] = remaining
            else:
                self._key_bindings.pop(sequence, None)
        if not self._key_bindings.get(sequence):
            self._dispatcher(sequence).unregister(self)

    # ------------------------------------------------------------------
    # bind_all：同样按标签页隔离，避免污染进程级 all 绑定
    # ------------------------------------------------------------------
    def bind_all(self, sequence=None, func=None, add=None):  # type: ignore[override]
        """转发全局绑定（如滚轮），但只在本标签页激活时生效。

        直接 ``bind_all`` 会写进 Tcl 的 ``all`` 绑定标签：多数工具从不解绑，
        且 Tk 默认替换语义会让后打开的工具静默顶掉先打开的工具，甚至在标签页
        关闭后仍然触发已销毁页面的回调（回归 2026-09-29 P0）。
        """
        if sequence is None or func is None or not callable(func):
            return super().bind_all(sequence, func, add)

        funcid = self._next_funcid()
        if str(add or "") == "+":
            self._all_bindings.setdefault(sequence, []).append((funcid, func))
        else:
            self._all_bindings[sequence] = [(funcid, func)]
        self._dispatcher(sequence).register(self, all_bindings=True)
        return funcid

    def unbind_all(self, sequence: str, funcid: str | None = None) -> None:  # type: ignore[override]
        """解除本标签页登记的全局绑定（工具按需解绑时调用）。"""
        if funcid is None:
            self._all_bindings.pop(sequence, None)
        else:
            remaining = [
                (fid, cb) for fid, cb in self._all_bindings.get(sequence, [])
                if fid != funcid
            ]
            if remaining:
                self._all_bindings[sequence] = remaining
            else:
                self._all_bindings.pop(sequence, None)
        if not self._all_bindings.get(sequence):
            self._dispatcher(sequence).unregister(self, all_bindings=True)

    def _clear_bindings(self) -> None:
        """标签页关闭时摘除**本页**的登记，不动其他页与宿主 root 的绑定。"""
        for sequence in list(self._key_bindings):
            self._dispatcher(sequence).unregister(self)
        for sequence in list(self._all_bindings):
            self._dispatcher(sequence).unregister(self, all_bindings=True)
        self._key_bindings.clear()
        self._all_bindings.clear()

    # ------------------------------------------------------------------
    # 菜单栏：本页保存，激活时由主窗口挂上
    # ------------------------------------------------------------------
    def set_menu(self, menu: tk.Menu | None) -> None:
        """由主窗口在标签页切换时调用：把本页菜单挂到真实顶层窗口。

        仅当本页是当前激活页时才真正挂载，避免后台加载的工具夺走前台菜单。
        """
        self.menu = menu
        if not self._alive():
            return
        with suppress(tk.TclError):
            self._real_toplevel.configure(menu=menu if menu is not None else "")

    def _detach_menu(self) -> None:
        """销毁前清掉挂在真实顶层窗口上的本页菜单，避免留下已销毁的 Tcl 菜单。"""
        if self.menu is None:
            return
        try:
            if str(self._real_toplevel.cget("menu")) == str(self.menu):
                self._real_toplevel.configure(menu="")
        except tk.TclError:
            pass
        self.menu = None

    # ------------------------------------------------------------------
    # 状态：避免与 ttk 部件状态语义冲突
    # ------------------------------------------------------------------
    def state(self, newstate: str | None = None):  # type: ignore[override]
        if newstate is None:
            return "normal"
        return newstate

    wm_state = state

    def attributes(self, *_args, **_kwargs):  # type: ignore[override]
        """``-topmost`` 等属性设置对嵌入式标签页无意义，静默忽略。"""
        return ""

    wm_attributes = attributes

    # ------------------------------------------------------------------
    # 配置：剔除 Frame 不认识的窗口专有选项
    # ------------------------------------------------------------------
    def configure(self, cnf: dict | None = None, **kwargs):  # type: ignore[override]
        """透传配置，剔除 Frame 不支持的窗口专有选项。

        ``menu`` 需要特别处理：工具以 ``self.root.config(menu=menubar)`` 挂菜单栏
        （stem-optimize / GPA / 衬度统计等），而 Frame 不支持 ``menu``。直接丢弃
        会让整个 File/Edit 菜单静默消失，因此交给主窗口保存并在本页激活时挂上
        真实顶层窗口；主源码曾直接挂到顶层且**从不恢复**，导致切页后菜单栏
        停留在上一个工具（回归 2026-10-03 R3）。
        """
        if cnf is not None and not isinstance(cnf, dict):
            # 位置参数形式：('bg',) 查询或 ('bg', 'red') 设置，交由 ttk 处理
            return super().configure(cnf, **kwargs)
        if cnf:
            kwargs.update(cnf)

        menu_requested = "menu" in kwargs
        menu = kwargs.pop("menu", None)
        if menu_requested:
            # 本页自己保存菜单（切换回来时要恢复）；主窗口据此决定何时挂到顶层。
            self.menu = menu if isinstance(menu, tk.Menu) else None
            if self._on_menu_change is not None:
                self._on_menu_change(self, self.menu)
            elif self._is_active():
                # 无主窗口协调时（独立使用/测试）直接挂到真实顶层窗口
                with suppress(tk.TclError):
                    self._real_toplevel.configure(
                        menu=self.menu if self.menu is not None else ""
                    )
        cleaned = {k: v for k, v in kwargs.items() if k not in _WM_ONLY_CONFIG_KEYS}
        if not cleaned:
            return None
        try:
            return super().configure(**cleaned)
        except tk.TclError:
            # 个别工具会传入 Frame 不支持的选项，忽略即可
            return None

    config = configure  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # 拖放（tkinterdnd2）：转发给真实顶层窗口
    # ------------------------------------------------------------------
    def drop_target_register(self, *args, **kwargs):
        target = self._real_toplevel
        if hasattr(target, "drop_target_register"):
            return target.drop_target_register(*args, **kwargs)
        return None

    def dnd_bind(self, *args, **kwargs):
        target = self._real_toplevel
        if hasattr(target, "dnd_bind"):
            return target.dnd_bind(*args, **kwargs)
        return None


def _make_ignored_wm_method(name: str):
    def _ignored(self: ToolHost, *args, **kwargs):  # noqa: ANN001, ARG001
        return None

    _ignored.__name__ = name
    _ignored.__doc__ = f"窗口管理方法 {name}() 对嵌入式标签页无意义，静默忽略。"
    return _ignored


for _name in _IGNORED_WM_METHODS:
    setattr(ToolHost, _name, _make_ignored_wm_method(_name))
    setattr(ToolHost, f"wm_{_name}", _make_ignored_wm_method(f"wm_{_name}"))

del _name
