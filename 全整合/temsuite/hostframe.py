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
    """

    def __init__(
        self,
        master: tk.Misc,
        *,
        on_title_change: Callable[[str], None] | None = None,
        tool_id: str = "",
    ) -> None:
        super().__init__(master)
        self._tool_id = tool_id
        self._on_title_change = on_title_change
        self._title = ""
        self._minsize: tuple[int, int] | None = None
        self._close_callbacks: dict[str, Callable[[], Any]] = {}
        # 工具在 __init__ 里可能读写这些属性，预先备好避免 AttributeError
        self._real_toplevel = self.winfo_toplevel()

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
        """由主窗口在关闭标签页前调用，让工具执行自身的清理逻辑。"""
        for func in list(self._close_callbacks.values()):
            with suppress(Exception):  # noqa: BLE001 - 清理失败不应阻塞关闭
                func()

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
        if cnf:
            kwargs.update(cnf)
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
