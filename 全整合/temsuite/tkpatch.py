"""Tk 构造函数拦截 —— 让自建根窗口的工具无需改动即可嵌入。

三个工具（视频帧提取、STEM 图像优化、TIFF 面积测量）的 GUI 类在
``__init__`` 内部直接执行 ``self.root = tk.Tk()``，没有对外暴露注入点。

与其修改这些原项目的代码（会污染各自独立的 git 仓库、并与上游后续改动冲突），
这里在实例化期间临时把 ``tkinter.Tk`` 替换为返回既有 ``ToolHost`` 的工厂。
工具拿到的「根窗口」就是标签页容器本身，其余代码路径完全不变。
"""

from __future__ import annotations

import contextlib
import logging
import tkinter as tk
from collections.abc import Iterator
from tkinter import ttk
from typing import Any

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def ttk_theme_frozen() -> Iterator[None]:
    """在上下文内阻止工具切换全局 ttk 主题。

    背景
    ----
    TIF 滤镜工具在 ``FilterApp.__init__`` 中执行 ``ttk.Style().theme_use('clam')``。
    独立运行时这只是设定自己的外观，但在整合环境下 ttk 主题是**进程级全局资源**，
    切换它会：

    1. 清空所有已注册的自定义样式配置（例如主窗口为工具树设定的 rowheight，
       被清空后中文行距不足而出现文字垂直重叠）；
    2. 把其余所有工具的外观一并从 ttkbootstrap 主题改成 clam。

    因此加载工具期间屏蔽主题切换，外观统一由主窗口负责。
    无参调用（查询当前主题）照常放行。
    """
    original = ttk.Style.theme_use

    def _blocked(self: ttk.Style, themename: str | None = None):
        if themename is None:
            return original(self)  # 查询语义，放行
        logger.debug("已忽略工具的 ttk 主题切换请求: %s", themename)
        return None

    ttk.Style.theme_use = _blocked  # type: ignore[method-assign]
    try:
        yield
    finally:
        ttk.Style.theme_use = original  # type: ignore[method-assign]


@contextlib.contextmanager
def tk_root_redirected(host: Any) -> Iterator[None]:
    """在上下文内，把工具的**首次** ``tk.Tk()`` 调用重定向为 *host*。

    同时拦截 ``ttkbootstrap.Window``（它是 ``tk.Tk`` 的子类，
    单独 patch ``tkinter.Tk`` 对已定义的子类无效）。

    为何只拦截首次
    --------------
    工具只会创建一个根窗口，但同一构造过程中的第三方库也可能调用 ``tk.Tk()``。
    典型案例是 matplotlib 的 TkAgg 后端：``plt.subplots()`` 内部会执行
    ``tk.Tk(className="matplotlib")`` 来承载图窗。若把它也重定向到 host，
    matplotlib 会误将标签页容器当作顶层窗口，进而调用 ``wm_frame()`` 等
    仅存在于真实窗口上的接口而失败。

    因此这里采用一次性语义：首次调用交出 host，后续调用放行给真正的
    ``tkinter.Tk``，让 matplotlib 照常创建它自己的隐藏图窗容器。

    Notes
    -----
    仅拦截根窗口构造，``tk.Toplevel`` 不受影响 —— 工具弹出的独立对话框
    应当保持为真正的顶层窗口。
    """
    original_tk = tk.Tk
    consumed = False

    def _factory(*args: Any, **kwargs: Any) -> Any:
        nonlocal consumed
        if consumed:
            # 已交出 host，后续调用（如 matplotlib）使用真实 Tk
            return original_tk(*args, **kwargs)
        consumed = True
        return host

    patched: list[tuple[Any, str, Any]] = [(tk, "Tk", original_tk)]
    tk.Tk = _factory  # type: ignore[assignment,misc]

    # ttkbootstrap.Window 继承 tk.Tk，需要单独处理。
    # 其构造参数（themename 等）与 tk.Tk 不同，因此后备要落回原始 Window。
    try:
        import ttkbootstrap as ttkb
    except ImportError:
        pass
    else:
        original_window = ttkb.Window

        def _window_factory(*args: Any, **kwargs: Any) -> Any:
            nonlocal consumed
            if consumed:
                return original_window(*args, **kwargs)
            consumed = True
            return host

        patched.append((ttkb, "Window", original_window))
        ttkb.Window = _window_factory  # type: ignore[assignment,misc]

    try:
        yield
    finally:
        for module, attr, original in patched:
            setattr(module, attr, original)
