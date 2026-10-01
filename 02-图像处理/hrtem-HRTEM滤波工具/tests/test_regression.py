from __future__ import annotations

import unittest

import numpy as np

from hrtem_filter import FilterParams, HRTEMFilter


class RegressionTests(unittest.TestCase):
    def test_default_fast_mode_golden_samples(self) -> None:
        """Protect the validated v5 default against accidental algorithm drift."""
        image = np.random.default_rng(2026).normal(size=(128, 128)).astype(np.float32)
        output = HRTEMFilter().process_image(image, FilterParams()).primary
        expected = {
            (0, 0): -0.21300022,
            (1, 63): -0.01628669,
            (23, 97): 0.14727417,
            (64, 64): 0.15274517,
            (127, 127): -0.05979720,
        }
        # Relative tolerance survives BLAS/platform rounding differences while
        # still catching real algorithm drift.
        for index, value in expected.items():
            np.testing.assert_allclose(float(output[index]), value, rtol=5e-4, atol=1e-6)
        np.testing.assert_allclose(float(output.mean()), 0.0, atol=1e-5)
        np.testing.assert_allclose(float(output.std()), 0.20225593, rtol=5e-4)

    def test_dm_compatible_golden_samples(self) -> None:
        """Protect the dm_compatible sampling path against accidental drift."""
        image = np.random.default_rng(2026).normal(size=(128, 128)).astype(np.float32)
        output = HRTEMFilter().process_image(
            image, FilterParams(rotation_method="dm_compatible")
        ).primary
        expected = {
            (0, 0): -0.21683821,
            (1, 63): -0.00788816,
            (23, 97): 0.15594697,
            (64, 64): 0.14300528,
            (127, 127): -0.07130323,
        }
        for index, value in expected.items():
            np.testing.assert_allclose(float(output[index]), value, rtol=5e-4, atol=1e-6)
        np.testing.assert_allclose(float(output.mean()), 0.0, atol=1e-5)
        np.testing.assert_allclose(float(output.std()), 0.20168415, rtol=5e-4)


if __name__ == "__main__":
    unittest.main()
