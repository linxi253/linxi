import unittest

import numpy as np

from pipeline import DEFAULT_PARAMS, process_frame, validate_params

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



class PipelineTests(unittest.TestCase):
    def test_unknown_parameters_are_rejected(self):
        params = dict(DEFAULT_PARAMS)
        params["typo_parameter"] = 1
        with self.assertRaises(ValueError):
            validate_params(params)

    def test_equal_global_range_is_rejected(self):
        image = np.ones((16, 16), dtype=np.uint16)
        with self.assertRaises(ValueError):
            process_frame(image, dict(DEFAULT_PARAMS), (10.0, 10.0))


if __name__ == "__main__":
    unittest.main()
