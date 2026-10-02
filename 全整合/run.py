"""开发模式启动入口。

用法::

    python run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


sys.path.insert(0, str(Path(__file__).resolve().parent))

from temsuite.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
