"""HRTEM 模拟工具入口。

用法：
    python hrtem_tool/main.py        （双击 run.bat 即此方式）
    python -m hrtem_tool.main
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


_TOOLS_ROOT = Path(__file__).resolve().parent.parent
if str(_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ROOT))


def main() -> None:
    # 直接以脚本方式运行时无包上下文，统一走绝对导入
    from hrtem_tool.gui import HRTEMApp, make_root

    root = make_root()
    app = HRTEMApp(root)  # noqa: F841 — 保持引用
    root.mainloop()


if __name__ == "__main__":
    main()
