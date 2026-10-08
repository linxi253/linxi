"""Frozen application entry point with a user-visible crash report."""

from __future__ import annotations

import os
import tempfile
import traceback
from datetime import datetime
from multiprocessing import freeze_support
from pathlib import Path

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



def _write_crash_report(details: str) -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
    log_directory = base / "AtomCenterAnnotator" / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_directory / f"crash_{stamp}.log"
    path.write_text(details, encoding="utf-8")
    return path


def main() -> int:
    freeze_support()
    try:
        from atom_center.annotator_gui import main as application_main

        return application_main()
    except Exception:
        details = traceback.format_exc()
        try:
            report = _write_crash_report(details)
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "原子中心标注器发生错误",
                f"错误报告已保存到：\n{report}\n\n请把该文件发给软件维护者。",
                parent=root,
            )
            root.destroy()
        except Exception:
            pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
