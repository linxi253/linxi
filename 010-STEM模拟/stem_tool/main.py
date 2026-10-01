"""STEM-HAADF 模拟工具入口。

用法：
    python stem_tool/main.py          （双击 run.bat 即此方式）
    python -m stem_tool.main

注意：文件末尾的 ``if __name__ == "__main__":`` 守卫是**必须**的
—— Windows 上多进程只能用 spawn，子进程会重新执行主模块的顶层代码，
缺少守卫会导致子进程再次建池（重复模拟甚至进程爆炸）。引擎侧对不安全
上下文会自动退回串行执行并给出提示（见 stem_sim.scan.spawn_safety）。
"""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_ROOT = Path(__file__).resolve().parent.parent
if str(_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ROOT))


def main() -> None:
    from stem_tool.gui import STEMApp, make_root

    root = make_root()
    app = STEMApp(root)  # noqa: F841 — 保持引用
    root.mainloop()


if __name__ == "__main__":
    main()
