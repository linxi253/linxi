import unittest
import warnings

import numpy as np

from gpa import GPA
from phase import ComputationCancelled, Phase
from utils import (
    detect_bragg_peaks,
    estimate_g_vector_radius,
    forward_fft,
    backward_fft,
    phase_derivative_fourier,
    wrap_to_pi,
)


class FFTTests(unittest.TestCase):
    def test_fft_round_trip_even_and_odd_shapes(self):
        rng = np.random.default_rng(42)
        for shape in ((64, 80), (65, 79)):
            with self.subTest(shape=shape):
                image = rng.normal(size=shape)
                restored = backward_fft(forward_fft(image))
                np.testing.assert_allclose(restored.real, image, atol=1e-12)
                np.testing.assert_allclose(restored.imag, 0.0, atol=1e-12)

    def test_odd_constant_image_has_single_centered_dc_peak(self):
        image = np.ones((65, 67), dtype=np.float64)
        spectrum = forward_fft(image)
        expected = np.zeros_like(spectrum)
        expected[65 // 2, 67 // 2] = image.size
        np.testing.assert_allclose(spectrum, expected, atol=1e-10)


class PhaseDerivativeTests(unittest.TestCase):
    def test_linear_wrapped_phase_has_positive_exact_derivative(self):
        rows, cols = 32, 128
        slope = 0.05
        phase = np.broadcast_to(
            wrap_to_pi(slope * np.arange(cols)[None, :]),
            (rows, cols),
        )
        derivative = phase_derivative_fourier(phase, axis=1)
        np.testing.assert_allclose(derivative, slope, atol=1e-12)

    def test_invalid_axis_and_pixel_size_are_rejected(self):
        phase = np.zeros((8, 8))
        with self.assertRaises(ValueError):
            phase_derivative_fourier(phase, axis=2)
        with self.assertRaises(ValueError):
            phase_derivative_fourier(phase, axis=0, pixel_size=0)

    def test_phase_residue_is_excluded_from_quality_mask(self):
        rows = cols = 33
        y, x = np.mgrid[:rows, :cols]
        phase = Phase()
        phase.sigma = 3.0
        phase.amplitude = np.ones((rows, cols), dtype=float)
        phase.wrapped_phase = np.arctan2(y - rows // 2, x - cols // 2)
        quality = phase.quality_mask()
        self.assertFalse(quality[rows // 2, cols // 2])
        self.assertTrue(quality[2, 2])


class GPAMathTests(unittest.TestCase):
    @staticmethod
    def _gpa_with_affine_phase(E, shape=(128, 128), pixel_size=1.0):
        rows, cols = shape
        y, x = np.mgrid[:rows, :cols]
        ux = E[0, 0] * x + E[0, 1] * y
        uy = E[1, 0] * x + E[1, 1] * y

        gpa = GPA()
        # Non-constant carrier (a constant image is rejected by load_image);
        # the phases below are injected analytically and overwrite whatever
        # the FFT pipeline would have produced.
        carrier = (
            np.cos(2 * np.pi * 8 * x / cols)
            + np.cos(2 * np.pi * 10 * y / rows)
        )
        gpa.load_image(carrier, pixel_size=pixel_size)
        gpa.phase1.gx, gpa.phase1.gy = 8.0, 0.0
        gpa.phase2.gx, gpa.phase2.gy = 0.0, 10.0

        p1 = -2.0 * np.pi * (8.0 / cols * ux)
        p2 = -2.0 * np.pi * (10.0 / rows * uy)
        gpa.phase1.wrapped_phase = wrap_to_pi(p1)
        gpa.phase2.wrapped_phase = wrap_to_pi(p2)
        gpa.phase1.unwrapped_phase = p1
        gpa.phase2.unwrapped_phase = p2
        gpa.phase1._h_prime = np.ones(shape, dtype=np.complex128)
        gpa.phase2._h_prime = np.ones(shape, dtype=np.complex128)
        gpa.phase1.amplitude = np.ones(shape, dtype=np.float64)
        gpa.phase2.amplitude = np.ones(shape, dtype=np.float64)
        return gpa

    def test_affine_distortion_is_recovered_with_correct_sign(self):
        expected = np.array([[0.01, 0.02], [-0.015, -0.005]])
        result = self._gpa_with_affine_phase(expected).compute()
        actual = np.array([
            [np.nanmedian(result.e_xx), np.nanmedian(result.e_xy)],
            [np.nanmedian(result.e_yx), np.nanmedian(result.e_yy)],
        ])
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_coordinate_rotation_preserves_trace(self):
        E = np.array([[0.02, 0.01], [0.03, -0.005]])
        gpa = self._gpa_with_affine_phase(E)
        base = gpa.compute()
        angle = np.radians(30.0)
        gpa.set_rotation(angle)
        rotated = gpa.compute()
        rotation = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        expected = rotation @ E @ rotation.T
        actual = np.array([
            [np.nanmedian(rotated.e_xx), np.nanmedian(rotated.e_xy)],
            [np.nanmedian(rotated.e_yx), np.nanmedian(rotated.e_yy)],
        ])
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_allclose(
            rotated.dilatation,
            base.dilatation,
            atol=1e-12,
            equal_nan=True,
        )

    def test_anisotropic_pixel_scaling_is_applied_once_everywhere(self):
        pixel_space_E = np.array([[0.01, 0.02], [-0.03, 0.04]])
        gpa = self._gpa_with_affine_phase(
            pixel_space_E, pixel_size=(2.0, 1.0)
        )
        expected = np.array([[0.01, 0.01], [-0.06, 0.04]])
        result = gpa.compute()
        actual = np.array([
            [np.nanmedian(result.e_xx), np.nanmedian(result.e_xy)],
            [np.nanmedian(result.e_yx), np.nanmedian(result.e_yy)],
        ])
        np.testing.assert_allclose(actual, expected, atol=1e-12)

        strain_only = gpa.compute_strain_only()
        self.assertAlmostEqual(
            float(np.nanmedian(strain_only['eps_xy'])),
            0.5 * (expected[0, 1] + expected[1, 0]),
            places=12,
        )

    def test_displacement_uses_unwrapped_phase(self):
        E = np.array([[0.20, 0.0], [0.0, 0.0]])
        result = self._gpa_with_affine_phase(E).compute()
        row = result.u_x[result.u_x.shape[0] // 2]
        self.assertLess(np.nanmax(np.abs(np.diff(row))), 1.0)
        self.assertAlmostEqual(float(row[-1] - row[0]), 0.20 * 127, places=10)

    def test_no_displacement_skips_global_phase_unwrap(self):
        rows = cols = 64
        y, x = np.mgrid[:rows, :cols]
        image = (
            np.cos(2 * np.pi * 8 * x / cols)
            + np.cos(2 * np.pi * 10 * y / rows)
        )
        gpa = GPA()
        gpa.load_image(image)
        gpa.set_g1(8, 0, sigma=1.5)
        gpa.set_g2(0, 10, sigma=1.5)
        result = gpa.compute(include_displacement=False)
        self.assertIsNone(result.u_x)
        self.assertIsNone(result.u_y)
        self.assertIsNone(gpa.phase1.unwrapped_phase)
        self.assertIsNone(gpa.phase2.unwrapped_phase)

    def test_mask_results_false_returns_raw_values_with_mask(self):
        """mask_results=False keeps raw values; the mask ships alongside."""
        rows = cols = 96
        y, x = np.mgrid[:rows, :cols]
        # Periodic lattice -> no Hann -> edge amplitudes still finite, but the
        # mask must still exclude singular/low-amplitude regions if any.
        image = (
            np.cos(2 * np.pi * 12 * x / cols)
            + np.cos(2 * np.pi * 12 * y / rows)
        )
        gpa = GPA()
        gpa.load_image(image)
        gpa.set_g1(12, 0, sigma=1.5)
        gpa.set_g2(0, 12, sigma=1.5)

        masked = gpa.compute(include_displacement=False)
        raw = gpa.compute(include_displacement=False, mask_results=False)

        self.assertTrue(np.isfinite(raw.eps_xx).all())
        self.assertIsInstance(raw.quality_mask, np.ndarray)
        self.assertEqual(raw.quality_mask.dtype, bool)
        self.assertIs(raw.phase1, masked.phase1)
        # Wherever the mask is valid, both variants must agree exactly.
        inside = raw.quality_mask
        np.testing.assert_allclose(
            raw.eps_xx[inside], masked.eps_xx[inside], rtol=0, atol=0
        )
        # Wherever the mask is invalid, the masked variant is NaN while the
        # raw variant retains a finite value.
        if np.any(~inside):
            self.assertTrue(np.isnan(masked.eps_xx[~inside]).all())
            self.assertTrue(np.isfinite(raw.eps_xx[~inside]).all())

    def test_end_to_end_synthetic_lattice_sign_and_magnitude(self):
        """FFT masking -> phase extraction -> strain/displacement end-to-end check.

        The affine unit tests inject phase fields directly and therefore cannot
        catch a sign or scaling error in the mask/IFFT/phase pipeline itself.
        A real cosine lattice with a known uniform deformation is used here so
        the full chain is exercised: a lattice constructed with a phase shift
        of ``+s*x`` has compressed fringes (strain ``-s``) and vice versa.
        """
        rows = cols = 256
        y, x = np.mgrid[:rows, :cols]
        g1 = (32.0, 0.0)
        g2 = (0.0, 32.0)
        for phase_shift_sign in (1.0, -1.0):
            with self.subTest(phase_shift_sign=phase_shift_sign):
                expected_strain = -phase_shift_sign * 0.02
                image = (
                    np.cos(
                        2 * np.pi * g1[0] / cols
                        * (x + phase_shift_sign * 0.02 * x)
                    )
                    + np.cos(2 * np.pi * g2[1] / rows * y)
                )

                gpa = GPA()
                gpa.load_image(image)
                gpa.set_g1(*g1, sigma=3.0)
                gpa.set_g2(*g2, sigma=3.0)
                result = gpa.compute()

                self.assertAlmostEqual(
                    float(np.nanmedian(result.eps_xx)),
                    expected_strain,
                    places=3,
                )
                # Central region is free of FFT leakage at the image edges.
                self.assertAlmostEqual(
                    float(np.nanmean(result.eps_xx[64:192, 64:192])),
                    expected_strain,
                    places=3,
                )
                self.assertAlmostEqual(
                    float(np.nanmedian(result.eps_yy)), 0.0, places=3,
                )
                self.assertAlmostEqual(
                    float(np.nanmedian(result.eps_xy)), 0.0, places=3,
                )
                u_slope = float(np.median(
                    np.gradient(result.u_x[64:192, 64:192], axis=1)
                ))
                self.assertAlmostEqual(u_slope, expected_strain, places=3)

                # Refinement must recover the shifted peak with the same sign
                # convention that produced the strain above.
                correction = gpa.refine_g1(
                    np.ones((rows, cols), dtype=bool)
                )
                self.assertAlmostEqual(
                    correction[0],
                    32.0 * 0.02 * phase_shift_sign,
                    places=5,
                )
                self.assertAlmostEqual(correction[1], 0.0, places=5)

    def test_refining_perfect_lattice_does_not_move_g_vector(self):
        rows = cols = 128
        y, x = np.mgrid[:rows, :cols]
        image = (
            np.cos(2 * np.pi * 8 * x / cols)
            + np.cos(2 * np.pi * 10 * y / rows)
        )
        gpa = GPA()
        gpa.load_image(image)
        gpa.set_g1(8, 0, sigma=1.5)
        mask = np.zeros((rows, cols), dtype=bool)
        mask[24:104, 24:104] = True
        correction = gpa.refine_g1(mask)
        np.testing.assert_allclose(correction, (0.0, 0.0), atol=1e-10)
        np.testing.assert_allclose(gpa.g1, (8.0, 0.0), atol=1e-10)

    def test_collinear_and_nonfinite_g_vectors_are_rejected(self):
        gpa = GPA()
        gpa.load_image(
            np.cos(2 * np.pi * 8 * np.arange(64) / 64)[None, :] *
            np.ones((64, 1))
        )
        with self.assertRaises(ValueError):
            gpa.set_g1(np.nan, 2)
        gpa.set_g1(8, 0, sigma=1.5)
        gpa.set_g2(16, 0.01, sigma=1.5)
        with self.assertRaises(RuntimeError):
            gpa.compute()

    def test_refine_iterative_honours_cancellation(self):
        phase = Phase()
        phase.wrapped_phase = np.zeros((16, 16))
        mask = np.ones((16, 16), dtype=bool)
        with self.assertRaises(ComputationCancelled):
            phase.refine_iterative(mask, cancel_check=lambda: True)

    def test_gpa_computation_cancelled_shares_cancellation_base(self):
        from gpa import GPAComputationCancelled
        self.assertTrue(issubclass(GPAComputationCancelled, ComputationCancelled))
        self.assertTrue(issubclass(GPAComputationCancelled, RuntimeError))

    def test_nearly_equal_pixel_sizes_are_normalized_to_scalar(self):
        gpa = GPA()
        carrier = np.cos(np.arange(32 * 32).reshape(32, 32) * 0.37)
        gpa.load_image(carrier, pixel_size=(2.0, 2.0 + 1e-12))
        self.assertIsInstance(gpa.pixel_size, float)
        gpa.load_image(carrier, pixel_size=(2.0, 1.0))
        self.assertEqual(gpa.pixel_size, (2.0, 1.0))

    def test_constant_image_is_rejected(self):
        gpa = GPA()
        with self.assertRaises(ValueError) as raised:
            gpa.load_image(np.full((32, 32), 7.5))
        self.assertIn('contrast', str(raised.exception))

    def test_large_image_emits_memory_warning(self):
        from strainpp_gpa import gpa as gpa_module
        carrier = np.cos(np.arange(16 * 16).reshape(16, 16) * 0.3)
        original = gpa_module._LARGE_IMAGE_WARNING_BYTES
        gpa_module._LARGE_IMAGE_WARNING_BYTES = 1  # force the estimate over
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                gpa_module.GPA().load_image(carrier)
            self.assertTrue(any(
                issubclass(entry.category, RuntimeWarning)
                and 'memory' in str(entry.message).lower()
                for entry in caught
            ))
        finally:
            gpa_module._LARGE_IMAGE_WARNING_BYTES = original

    def test_estimate_peak_bytes_is_monotonic_and_validated(self):
        from utils import estimate_peak_bytes
        small = estimate_peak_bytes((64, 64))
        large = estimate_peak_bytes((128, 128))
        self.assertGreater(small, 0)
        self.assertGreater(large, small)
        self.assertAlmostEqual(
            large / small, 4.0, places=6
        )
        with self.assertRaises(ValueError):
            estimate_peak_bytes((0, 10))

    def test_fft_worker_count_does_not_change_result(self):
        rng = np.random.default_rng(7)
        image = rng.normal(size=(96, 80))
        single = forward_fft(image, workers=1)
        multi = forward_fft(image, workers=-1)
        np.testing.assert_array_equal(single, multi)


class BraggDetectionTests(unittest.TestCase):
    def test_first_bragg_ring_is_detected(self):
        rows = cols = 128
        y, x = np.mgrid[:rows, :cols]
        image = (
            np.cos(2 * np.pi * 30 * x / cols)
            + np.cos(2 * np.pi * 30 * y / rows)
        )
        spectrum = np.log10(np.maximum(np.abs(forward_fft(image)), 1e-30))
        radius = estimate_g_vector_radius(spectrum)
        self.assertAlmostEqual(radius, 30.0, delta=1.0)
        peaks = detect_bragg_peaks(spectrum, radius, n_peaks=8)
        self.assertGreaterEqual(len(peaks), 4)
        self.assertTrue(all(abs(peak[3] - 30.0) <= 1.0 for peak in peaks[:4]))

    def test_numerical_fft_floor_is_not_mistaken_for_a_bragg_ring(self):
        rows, cols = 96, 128
        y, x = np.mgrid[:rows, :cols]
        image = (
            np.cos(2 * np.pi * 12 * x / cols)
            + np.cos(2 * np.pi * 15 * y / rows)
        ).astype(np.float32)
        spectrum = np.log10(np.maximum(np.abs(forward_fft(image)), 1e-30))
        radius = estimate_g_vector_radius(spectrum)
        self.assertEqual(radius, 12.0)
        peaks = detect_bragg_peaks(spectrum, radius, n_peaks=4)
        self.assertEqual(len(peaks), 4)
        detected_radii = sorted(round(peak[3]) for peak in peaks)
        self.assertEqual(detected_radii, [12, 12, 15, 15])


if __name__ == "__main__":
    unittest.main()
