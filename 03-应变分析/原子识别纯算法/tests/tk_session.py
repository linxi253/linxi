# -*- coding: utf-8 -*-
"""跨测试模块共享的单个 Tk 会话。

本机 Miniconda 环境下，销毁 Tk root 后再次创建 ``tk.Tk()`` 会因
``init.tcl`` 读取失败而崩溃，因此整个测试进程只创建一次 root 与
应用实例，进程退出时统一清理。文件名不以 test_ 开头，避免被收集。
"""
from __future__ import annotations

import atexit
import tkinter as tk

_SESSION: dict = {}


def get_tk_session() -> tuple[tk.Tk, object]:
    """返回 (root, app)；Tk 不可用时抛出 tk.TclError，由调用方转为 SkipTest。"""
    if "app" not in _SESSION:
        root = tk.Tk()
        root.withdraw()
        from atomic_app import AtomicRecognitionApp

        _SESSION["root"] = root
        _SESSION["app"] = AtomicRecognitionApp(root)
        atexit.register(_cleanup)
    return _SESSION["root"], _SESSION["app"]


def _cleanup() -> None:
    app = _SESSION.pop("app", None)
    root = _SESSION.pop("root", None)
    if app is not None:
        try:
            # 直接走关停路径：on_close 在有未保存修订时会弹模态确认框，
            # 会把整个测试进程挂死。
            app._shutdown()
        except Exception:
            pass
    elif root is not None:
        try:
            root.destroy()
        except Exception:
            pass
