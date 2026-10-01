from __future__ import annotations

import unittest

import numpy as np

from butter import HRTEMFilter as LegacyFilter


class FacadeTests(unittest.TestCase):
    """butter.py 的 v4 兼容 facade：键名合同不能因 v5 键名修正而破坏。"""

    def test_stem_butterworth_maps_back_to_v4_key(self) -> None:
        """v4 始终用 butterworth_filtered 键承载 BW 输出，即使应用了十字掩膜。"""
        image = np.random.default_rng(11).normal(size=(64, 64)).astype(np.float32)
        proc = LegacyFilter()
        proc.params["stem_filter"] = True
        proc.params["apply_wiener"] = False
        proc.params["apply_absf"] = False
        out = proc.process_image(image)
        self.assertIn("butterworth_filtered", out)
        self.assertNotIn("stem_butterworth_filtered", out)

    def test_default_keys_match_v4_contract(self) -> None:
        image = np.random.default_rng(16).normal(size=(64, 64)).astype(np.float32)
        out = LegacyFilter().process_image(image)
        for key in ("original", "fft", "wiener_filtered", "absf_filtered", "butterworth_filtered"):
            self.assertIn(key, out)


if __name__ == "__main__":
    unittest.main()
