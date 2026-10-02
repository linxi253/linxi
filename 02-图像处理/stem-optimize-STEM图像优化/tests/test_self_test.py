import unittest

from self_test import run_self_test

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



class PackagedSelfTestTests(unittest.TestCase):
    def test_runtime_self_test(self):
        result = run_self_test()
        self.assertEqual(result["frames"], 3)
        self.assertEqual(result["dtype"], "uint16")
        self.assertEqual(
            result["codecs"],
            ["deflate", "jpeg", "jpeg2000", "lzw", "packbits", "zstd"],
        )


if __name__ == "__main__":
    unittest.main()
