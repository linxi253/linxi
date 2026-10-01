"""Pure cancel-state helpers for the GUI.

These helpers live outside ``gui.py`` so test suites can exercise the
cancel-state logic without importing Tkinter (which may be absent in headless
CI environments).  The GUI uses them directly, keeping tests aligned with the
production code path.
"""

from __future__ import annotations

from threading import Event


def reset_cancel_event_for_preview(cancel_event: Event | None) -> None:
    """Return the cancel event to use when a preview starts.

    Previews run as a short FFT task and do not support cancellation; any
    event left over from a previous save run would make the Cancel button
    report "正在取消" even though setting it has no effect.  Preview therefore
    always starts with ``None`` (returned implicitly).
    """

    del cancel_event


def cancel_status_text(cancel_event: Event | None) -> str:
    """Human-readable cancel button status for the current worker kind."""

    if cancel_event is not None:
        return "正在取消；当前帧完成后将删除临时输出…"
    return "预览无法中断，正在等待当前 FFT 完成…"
