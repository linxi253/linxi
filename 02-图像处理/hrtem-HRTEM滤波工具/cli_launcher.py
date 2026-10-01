"""Minimal PyInstaller target for the command-line executable."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from hrtem_filter.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
