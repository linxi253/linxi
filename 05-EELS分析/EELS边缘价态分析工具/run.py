"""独立 GUI 开发入口。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from eels_edge_analyzer.gui import main

if __name__ == "__main__":
    raise SystemExit(main())
