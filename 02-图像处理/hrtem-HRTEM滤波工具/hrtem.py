"""Compatibility entry point for v5.

Run without arguments to open the safe GUI.  Pass input/output paths for the
repeatable command-line workflow, for example::

    python hrtem.py input.tif output.tif --mode wiener --stem
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main() -> int:
    if len(sys.argv) == 1:
        from hrtem_filter.gui import main as gui_main

        return gui_main()
    from hrtem_filter.cli import main as cli_main

    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
