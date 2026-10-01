"""开发模式启动入口。

用法::

    python run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from temsuite.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
