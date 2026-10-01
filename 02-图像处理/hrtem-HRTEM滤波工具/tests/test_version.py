from __future__ import annotations

import re
import unittest
from pathlib import Path

from hrtem_filter._version import __version__

_ROOT = Path(__file__).resolve().parent.parent


class VersionSyncTests(unittest.TestCase):
    """pyproject.toml 与 _version.py 是两个版本号来源，必须保持一致。"""

    def test_pyproject_version_matches_package_version(self) -> None:
        text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        self.assertIsNotNone(match, "pyproject.toml 缺少 version 字段")
        self.assertEqual(match.group(1), __version__)


if __name__ == "__main__":
    unittest.main()
