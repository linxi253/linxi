"""matplotlib 后端治理 —— 消除多 Tcl 解释器冲突。

问题
----
半数工具在模块顶层执行 ``matplotlib.use('TkAgg')``。在独立运行时这没问题，
但整合进单进程后会引发致命冲突：

    4D-STEM 处理 与 TIFF 面积测量 使用 ``plt.subplots()``（pyplot 全局接口）。
    TkAgg 后端为每个 pyplot figure 创建 ``tk.Tk(className="matplotlib")``，
    也就是**第二个 Tcl 解释器实例**。两个解释器各有独立事件循环，
    在同一进程中共存会导致进程整体崩溃。

解决
----
把全局后端固定为 ``Agg``（非交互），并在加载工具期间屏蔽其 ``matplotlib.use()``
调用。这样做安全的依据是：

1. 所有工具的图形都通过 ``FigureCanvasTkAgg(fig, master)`` **显式嵌入**到自己的
   部件里 —— 该类不依赖全局后端设置，照常工作。
2. 全部工具均未使用 ``plt.show()``（已逐一核查），因此 Agg 的非交互特性
   不影响任何功能。
3. ``plt.subplots()`` 在 Agg 下只创建 Figure 对象、不创建 GUI 窗口，
   恰好消除了多余的 Tcl 解释器。

附带收益：特征区域演化分析本身就要求 Agg，固定后端后它不再与其他工具冲突。
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator

logger = logging.getLogger(__name__)

BACKEND = "Agg"


def force_headless_backend() -> None:
    """在加载任何工具之前调用，把 matplotlib 固定为 Agg。"""
    try:
        import matplotlib
    except ImportError:
        return
    matplotlib.use(BACKEND, force=True)
    # pyplot 的后端是**惰性解析**的：模块导入不加载具体后端，首次 plt.* 调用
    # 才经 _get_backend_mod() -> switch_backend() 解析。若该首次解析发生在
    # backend_locked() 期间（switch_backend 被屏蔽），pyplot 将拿不到后端模块，
    # 之后任何 plt.subplots() 都会以 "'NoneType' object has no attribute
    # 'FigureCanvas'" 崩溃。故须在此（无锁状态）立即把 pyplot 解析到 Agg。
    import matplotlib.pyplot

    matplotlib.pyplot.switch_backend(BACKEND)
    logger.debug("matplotlib 后端已固定为 %s", BACKEND)


@contextlib.contextmanager
def backend_locked() -> Iterator[None]:
    """在上下文内屏蔽 ``matplotlib.use()`` 与 ``pyplot.switch_backend()``。

    工具模块在 import 阶段就会尝试切换后端，因此该上下文需要覆盖整个加载过程。
    """
    try:
        import matplotlib
    except ImportError:
        yield
        return

    original_use = matplotlib.use
    restore: list[tuple[object, str, object]] = [(matplotlib, "use", original_use)]

    def _blocked_use(backend: str | None = None, *args, **kwargs) -> None:  # noqa: ARG001
        if backend and str(backend).lower() != BACKEND.lower():
            logger.debug("已忽略工具的后端切换请求: %s", backend)

    matplotlib.use = _blocked_use  # type: ignore[assignment]

    # pyplot 可能尚未导入；已导入则同时屏蔽 switch_backend
    pyplot = getattr(matplotlib, "pyplot", None)
    if pyplot is not None:
        restore.append((pyplot, "switch_backend", pyplot.switch_backend))
        pyplot.switch_backend = _blocked_use  # type: ignore[assignment]

    try:
        yield
    finally:
        for module, attr, original in restore:
            setattr(module, attr, original)
