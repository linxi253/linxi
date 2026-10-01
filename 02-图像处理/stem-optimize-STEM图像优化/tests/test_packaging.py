import re
import unittest
from pathlib import Path

from version import APP_VERSION


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
