import re
import unittest
from pathlib import Path

from version import APP_VERSION

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



class PackagingMetadataTests(unittest.TestCase):
    def test_pyproject_project_version_matches_release_version(self):
        """pyproject [project].version 必须与 version.APP_VERSION 一致。"""
        text = (
            Path(__file__).resolve().parent.parent / "pyproject.toml"
        ).read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        self.assertIsNotNone(match, "pyproject.toml 缺少 project.version")
        self.assertEqual(match.group(1), APP_VERSION)


if __name__ == "__main__":
    unittest.main()
