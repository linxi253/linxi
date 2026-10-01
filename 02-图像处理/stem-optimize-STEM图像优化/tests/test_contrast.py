import threading
import unittest

import numpy as np

from contrast import (
    DEFAULT_HIGH_PERCENTILE,
    DEFAULT_LOW_PERCENTILE,
    apply_clahe,
    apply_clahe_uint16,
    estimate_global_range,
)
from errors import OperationCancelled


class _SyntheticReader:
    num_frames = 100
    dtype = np.dtype(np.uint16)

    def read_frame(self, index):
        frame = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) % 101
        if index == 50:
            frame[8:56, 8:56] = 10000
        return frame


class _ConstantUint16Reader:
    num_frames = 2
    dtype = np.dtype(np.uint16)

    def read_frame(self, index):
        return np.full((16, 16), 1234, dtype=np.uint16)


class ContrastTests(unittest.TestCase):
    def test_default_percentile_constants_are_sane(self):
        self.assertLess(DEFAULT_LOW_PERCENTILE, DEFAULT_HIGH_PERCENTILE)

    def test_clahe_uint8_preserves_dtype_and_contrast(self):
        image = np.linspace(0, 255, 4096, dtype=np.uint8).reshape(64, 64)
        result = apply_clahe(image, clip_limit=2.0)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, image.shape)
        self.assertGreater(float(result.std()), 0.0)

    def test_clahe_uint16_preserves_dtype(self):
        image = np.linspace(0, 65535, 4096, dtype=np.uint16).reshape(64, 64)
        result = apply_clahe_uint16(image, clip_limit=2.0)
        self.assertEqual(result.dtype, np.uint16)
        self.assertEqual(result.shape, image.shape)
        self.assertTrue(np.isfinite(result).all())

    def test_global_integer_histogram_sees_every_frame(self):
        vmin, vmax = estimate_global_range(_SyntheticReader())
        self.assertLessEqual(vmin, 1)
        self.assertGreater(vmax, 100)

    def test_global_range_honors_cancellation(self):
        event = threading.Event()
        event.set()
        with self.assertRaises(OperationCancelled):
            estimate_global_range(_SyntheticReader(), cancel_event=event)

    def test_constant_integer_stack_uses_full_native_range(self):
        with self.assertLogs("contrast", level="WARNING"):
            result = estimate_global_range(_ConstantUint16Reader())
        self.assertEqual(result, (0.0, 65535.0))


if __name__ == "__main__":
    unittest.main()
