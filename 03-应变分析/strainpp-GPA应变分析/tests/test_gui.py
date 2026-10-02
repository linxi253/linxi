"""Tests for StrainGUI static helpers that do not require a Tk root."""
import unittest

import numpy as np

from strain_gui import StrainGUI

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



class FakeResult:
    """Minimal stand-in for GPAOutput with raw (unmasked) fields."""

    def __init__(self):
        self.quality_mask = np.array(
            [[True, False], [False, True]], dtype=bool
        )
        self.eps_xx = np.array([[0.01, 0.50], [-0.75, 0.02]])
        self.phase1 = np.array([[0.1, 9.9], [-8.0, 0.2]])


class FieldDataTests(unittest.TestCase):
    def test_numeric_export_is_always_quality_masked(self):
        """GUI results are raw; numeric exports must mask regardless."""
        result = FakeResult()
        data = StrainGUI._field_data(result, 'eps_xx')
        self.assertTrue(np.isfinite(data[0, 0]))
        self.assertTrue(np.isnan(data[0, 1]))
        self.assertTrue(np.isnan(data[1, 0]))
        self.assertTrue(np.isfinite(data[1, 1]))

    def test_phase_fields_are_masked_too(self):
        result = FakeResult()
        data = StrainGUI._field_data(result, 'phase1')
        self.assertTrue(np.isnan(data[0, 1]))
        self.assertAlmostEqual(data[1, 1], 0.2)

    def test_quality_mask_exports_as_float(self):
        result = FakeResult()
        data = StrainGUI._field_data(result, 'quality_mask')
        self.assertEqual(data.dtype, np.float32)
        np.testing.assert_allclose(data, [[1.0, 0.0], [0.0, 1.0]])

    def test_none_result_and_missing_fields_return_none(self):
        self.assertIsNone(StrainGUI._field_data(None, 'eps_xx'))
        self.assertIsNone(StrainGUI._field_data(FakeResult(), 'u_x'))


class DisplayDataTests(unittest.TestCase):
    def test_block_mean_avoids_aliasing(self):
        values = np.array(
            [[1.0, 2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0],
             [9.0, 10.0, 11.0, 12.0],
             [13.0, 14.0, 15.0, 16.0]]
        )
        reduced = StrainGUI._display_data(values, max_side=2)
        np.testing.assert_allclose(reduced, [[3.5, 5.5], [11.5, 13.5]])

    def test_all_nan_block_stays_nan(self):
        values = np.full((4, 4), np.nan)
        values[0, 0] = 1.0
        reduced = StrainGUI._display_data(values, max_side=2)
        self.assertAlmostEqual(reduced[0, 0], 1.0)
        self.assertTrue(np.isnan(reduced[0, 1]))

    def test_no_downsampling_returns_original_array(self):
        values = np.ones((100, 100))
        self.assertIs(StrainGUI._display_data(values), values)


class AutomaticLimitsTests(unittest.TestCase):
    def test_heavy_tail_does_not_stretch_symmetric_limits(self):
        """82% outliers must not wash out the background colour scale."""
        rng = np.random.default_rng(0)
        values = np.concatenate([
            rng.normal(0.028, 0.028, size=8200),
            rng.uniform(-3.0, 3.0, size=1800),
        ])
        _low, high = StrainGUI._automatic_limits(
            'eps_xx', values, symmetric=True
        )
        quantile_high = float(np.quantile(np.abs(values), 0.995))
        # MAD bound stays in the background-noise magnitude (original
        # Strain++ ships fixed +/-0.2 limits), unlike the long-tail quantile.
        self.assertLess(high, 0.25)
        self.assertLess(high, quantile_high / 2.0)
        self.assertAlmostEqual(high, -_low)

    def test_constant_field_falls_back_to_max_abs(self):
        values = np.array([0.1, 0.1, 0.1, 5.0])
        low, high = StrainGUI._automatic_limits(
            'eps_xx', values, symmetric=True
        )
        self.assertEqual(high, 5.0)
        self.assertEqual(low, -5.0)

    def test_empty_values_return_unit_range(self):
        self.assertEqual(
            StrainGUI._automatic_limits('eps_xx', np.array([]), True),
            (0.0, 1.0),
        )


class PowerSpectrumCoordinateTests(unittest.TestCase):
    """Power-spectrum click convention: pixel centres on integer data coords.

    Regression for the audit finding: with ``extent=(0, N, M, 0)`` pixel
    centres sit at half-integer coordinates while the click back-calculation
    assumes integer array indices, so every picked G vector came out +0.5 px
    in both components.
    """

    def _power_extent(self, rows, cols):
        """Evaluate the extent expressions found in _show_power_spectrum."""
        import inspect
        import re

        source = inspect.getsource(StrainGUI._show_power_spectrum)
        expressions = re.findall(r"extent=\(([^)]*)\)", source)
        expressions += re.findall(r"set_extent\(\(([^)]*)\)\)", source)
        self.assertEqual(
            len(expressions), 2,
            'power spectrum extent should appear exactly twice '
            '(imshow + set_extent)',
        )
        extents = {
            eval(f'({expression})', {'N': cols, 'M': rows})
            for expression in expressions
        }
        self.assertEqual(len(extents), 1, 'extent expressions disagree')
        return extents.pop()

    def test_extent_places_pixel_centers_on_integer_coordinates(self):
        """Pixel (i, j) centre must sit at data coordinates (j, i)."""
        rows, cols = 64, 96
        self.assertEqual(
            self._power_extent(rows, cols),
            (-0.5, cols - 0.5, rows - 0.5, -0.5),
        )

    def test_click_back_calculation_has_no_half_pixel_offset(self):
        """Clicking the centre of a Bragg peak must yield its exact G."""
        import inspect
        import re

        source = inspect.getsource(StrainGUI._on_power_click)
        gx_expr = next(
            line.strip().split('=', 1)[1]
            for line in source.splitlines()
            if re.match(r'\s*gx\s*=', line)
        )
        gy_expr = next(
            line.strip().split('=', 1)[1]
            for line in source.splitlines()
            if re.match(r'\s*gy\s*=', line)
        )

        rows = cols = 128
        gx_true, gy_true = 9, 6
        row, col = rows // 2 + gy_true, cols // 2 + gx_true
        extent = self._power_extent(rows, cols)
        # Data coordinate of the clicked pixel centre under the extent.
        x = extent[0] + (col + 0.5) / cols * (extent[1] - extent[0])
        y = extent[3] + (row + 0.5) / rows * (extent[2] - extent[3])
        gx = eval(gx_expr, {'x': x, 'N': cols})
        gy = eval(gy_expr, {'y': y, 'M': rows})
        self.assertEqual((gx, gy), (gx_true, gy_true))


if __name__ == '__main__':
    unittest.main()
