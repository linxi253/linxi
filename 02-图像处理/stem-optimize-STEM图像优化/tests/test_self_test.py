import unittest

from self_test import run_self_test


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
