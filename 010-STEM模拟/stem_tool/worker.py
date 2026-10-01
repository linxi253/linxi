"""后台工作线程：模拟在子线程执行，进度/结果经队列投递到 Tk 主线程。"""

from __future__ import annotations

import queue
import sys
import threading
import traceback
from typing import Callable, Optional

from .sim_core import CancelledError


def _format_exception(exc: BaseException) -> str:
    # Python 3.10+ 支持单参形式；3.9 需要旧签名
    if sys.version_info >= (3, 10):
        return "".join(traceback.format_exception(exc))
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


class Worker:
    """把一个 callable(fn, progress, stop) 放到后台线程执行。

    事件通过 queue 投递，GUI 用 after() 轮询：
      ("progress", frac, msg) / ("log", msg) / ("result", obj)
      ("error", text) / ("cancelled",) / ("done",)
    """

    def __init__(self):
        self.events: "queue.Queue[tuple]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> bool:
        """请求取消（协作式：模拟循环内轮询）。"""
        if self.running:
            self._stop.set()
            return True
        return False

    def start(self, fn: Callable, tag: str = "") -> None:
        if self.running:
            raise RuntimeError("已有任务在运行")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(fn, tag), daemon=True, name="SimWorker"
        )
        self._thread.start()

    # ------------------------------------------------------------------
    def _run(self, fn: Callable, tag: str) -> None:
        def progress(frac: float, msg: str) -> None:
            self.events.put(("progress", min(1.0, max(0.0, frac)), msg, tag))

        def is_stopped() -> bool:
            return self._stop.is_set()

        try:
            result = fn(progress, is_stopped)
        except CancelledError:
            self.events.put(("cancelled", tag))
            self.events.put(("done", tag))
            return
        except Exception as exc:  # noqa: BLE001
            self.events.put(("error", _format_exception(exc)))
            self.events.put(("done", tag))
            return
        if self._stop.is_set():
            self.events.put(("cancelled", tag))
        else:
            self.events.put(("result", result, tag))
        self.events.put(("done", tag))
