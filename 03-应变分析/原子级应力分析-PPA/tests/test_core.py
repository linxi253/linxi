from __future__ import annotations

import hashlib
import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from atom_detector.annotate.label_review import LabelReviewApp
from ppa import (
    assign_unique_lattice_indices,
    estimate_reference_vectors_from_chains,
    fit_lattice_vector_from_chain,
    greedy_nms,
    infer_atom_chain_from_anchors,
    refine_lattice_basis_in_region,
    find_lattice_holes,
    strain_value_for_display,
)
from ppa_stats import (
    compute_deformation_mode,
    compute_strain_gradient,
    detect_physical_convention,
    load_displacement_csv,
    load_strain_csv,
    weighted_mean_std,
)
from atom_detector.model_security import ModelVerificationError, load_allowlist, verify_model
from ppa_core.image_io import ImageLoadError, load_analysis_image
from ppa_core.project_store import ProjectValidationError, load_project, save_project
from ppa_core.strain import AnalysisError, compute_cst_strain, compute_local_peak_pair_strain

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



def square_grid(size: int = 5) -> np.ndarray:
    return np.array([(x, y) for y in range(size) for x in range(size)], dtype=float)


class StrainCoreTests(unittest.TestCase):
    def test_affine_stretch_uses_reference_coordinates(self):
        reference = square_grid()
        F = np.array([[1.20, 0.0], [0.0, 0.90]])
        deformed = reference @ F.T + np.array([3.0, -2.0])
        result = compute_cst_strain(reference, deformed)
        np.testing.assert_allclose(result.deformation_gradient, np.broadcast_to(F, result.deformation_gradient.shape), atol=1e-12)
        np.testing.assert_allclose(result.small_xx, 0.20, atol=1e-12)
        np.testing.assert_allclose(result.small_yy, -0.10, atol=1e-12)
        np.testing.assert_allclose(result.green_xx, 0.22, atol=1e-12)
        np.testing.assert_allclose(result.green_yy, -0.095, atol=1e-12)

    def test_rigid_rotation_has_zero_green_lagrange_strain(self):
        reference = square_grid()
        angle = math.radians(30.0)
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        result = compute_cst_strain(reference, reference @ rotation.T)
        np.testing.assert_allclose(result.green_xx, 0.0, atol=1e-12)
        np.testing.assert_allclose(result.green_yy, 0.0, atol=1e-12)
        np.testing.assert_allclose(result.green_xy, 0.0, atol=1e-12)
        np.testing.assert_allclose(result.polar_rotation, angle, atol=1e-12)

    def test_simple_shear_has_tensor_shear_half_engineering_shear(self):
        reference = square_grid()
        gamma = 0.1
        F = np.array([[1.0, gamma], [0.0, 1.0]])
        result = compute_cst_strain(reference, reference @ F.T)
        np.testing.assert_allclose(result.small_xy, gamma / 2.0, atol=1e-12)

    def test_collinear_or_duplicate_inputs_are_rejected(self):
        collinear = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        with self.assertRaises(AnalysisError):
            compute_cst_strain(collinear, collinear)
        duplicated = square_grid()
        duplicated[1] = duplicated[0]
        with self.assertRaises(AnalysisError):
            compute_cst_strain(duplicated, duplicated)

    def test_local_peak_pairs_recovers_affine_transform(self):
        reference = square_grid(5)
        indices = reference.astype(int)
        F = np.array([[1.05, 0.03], [0.02, 0.97]])
        result = compute_local_peak_pair_strain(indices, reference, reference @ F.T)
        np.testing.assert_allclose(result.deformation_gradient, np.broadcast_to(F, result.deformation_gradient.shape), atol=1e-12)
        self.assertEqual(len(result.locations), 25)
        self.assertTrue(result.quality_mask.all())
        self.assertEqual(int(np.sum(result.site_quality == "A-symmetric")), 9)

    def test_local_peak_pairs_uses_fallback_for_border_sites(self):
        reference = square_grid(5)
        indices = reference.astype(int)
        result = compute_local_peak_pair_strain(indices, reference, reference)
        self.assertIsNotNone(result.site_indices)
        np.testing.assert_array_equal(result.site_indices, np.arange(len(reference)))
        self.assertTrue(result.quality_mask.all())
        self.assertEqual(int(np.sum(result.site_quality == "A-symmetric")), 9)
        self.assertEqual(int(np.sum(result.site_quality == "B-local-fit")), 16)
        self.assertEqual(int(np.sum(result.site_quality == "C-minimal-fit")), 0)

    def test_local_peak_pairs_keeps_unsolved_site_as_nan_with_reason(self):
        reference = square_grid(4)
        indices = reference.astype(int)
        # Retain one isolated accepted atom plus a separate 3x3 block that is solvable.
        keep = ((indices[:, 0] < 3) & (indices[:, 1] < 3)) | ((indices[:, 0] == 3) & (indices[:, 1] == 3))
        reference = reference[keep]
        indices = indices[keep]
        result = compute_local_peak_pair_strain(indices, reference, reference)
        isolated = int(np.flatnonzero(np.all(indices == [3, 3], axis=1))[0])
        self.assertEqual(len(result.locations), len(reference))
        self.assertFalse(bool(result.quality_mask[isolated]))
        self.assertTrue(np.isnan(result.small_xx[isolated]))
        self.assertEqual(result.invalid_reasons[isolated], "insufficient_noncollinear_neighbors")


class MultiPointReferenceVectorTests(unittest.TestCase):
    def test_two_noisy_chains_recover_length_and_angle_from_all_atoms(self):
        rng = np.random.default_rng(20260831)
        a_true = np.array([9.7, 1.8])
        b_true = np.array([-2.1, 10.4])
        index = np.arange(10, dtype=float)[:, None]
        a_chain = np.array([15.0, 20.0]) + index * a_true + rng.normal(0.0, 0.10, (10, 2))
        b_chain = np.array([80.0, 35.0]) + index * b_true + rng.normal(0.0, 0.10, (10, 2))
        # One imperfect centroid should be downweighted without changing the
        # integer sequence or reducing the calculation to endpoint subtraction.
        a_chain[5] += np.array([0.55, -0.45])

        a_fit, b_fit = estimate_reference_vectors_from_chains(a_chain, b_chain)

        np.testing.assert_allclose(a_fit.vector, a_true, atol=0.045)
        np.testing.assert_allclose(b_fit.vector, b_true, atol=0.045)
        self.assertEqual(a_fit.n_points, 10)
        self.assertEqual(b_fit.n_points, 10)
        self.assertLess(np.linalg.norm(a_fit.vector - a_true),
                        np.linalg.norm((a_chain[1] - a_chain[0]) - a_true))

    def test_vector_fit_handles_angle_wraparound(self):
        angle = np.radians(179.5)
        expected = 12.0 * np.array([np.cos(angle), np.sin(angle)])
        index = np.arange(8, dtype=float)[:, None]
        chain = np.array([100.0, 50.0]) + index * expected
        chain[:, 1] += np.array([0.02, -0.03, 0.01, 0.00, -0.01, 0.03, -0.02, 0.01])

        fit = fit_lattice_vector_from_chain(chain)

        np.testing.assert_allclose(fit.vector, expected, atol=0.01)

    def test_chain_fit_rejects_skipped_atom_and_backtracking(self):
        skipped = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0],
                            [40.0, 0.0], [50.0, 0.0]])
        with self.assertRaisesRegex(AnalysisError, "跳过|不连续"):
            fit_lattice_vector_from_chain(skipped)

        reversed_order = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0],
                                   [10.0, 0.0], [30.0, 0.0]])
        with self.assertRaisesRegex(AnalysisError, "回退"):
            fit_lattice_vector_from_chain(reversed_order)

    def test_two_chains_must_define_two_dimensions(self):
        a_chain = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        b_chain = np.array([[0.0, 5.0], [10.0, 5.0], [20.0, 5.0]])
        with self.assertRaises(AnalysisError):
            estimate_reference_vectors_from_chains(a_chain, b_chain)

    def test_nonadjacent_anchors_expand_to_every_horizontal_atom(self):
        all_points = np.array([(10.0 * x, 10.0 * y)
                               for y in range(4) for x in range(12)], dtype=float)
        anchors = [0, 4, 8]

        inferred = infer_atom_chain_from_anchors(
            all_points, anchors, label="a 方向原子列")
        fit = fit_lattice_vector_from_chain(all_points[inferred], min_points=2)

        np.testing.assert_array_equal(inferred, np.arange(9))
        np.testing.assert_allclose(fit.vector, [10.0, 0.0], atol=1e-12)

    def test_two_rotated_anchors_find_chain_without_neighboring_row(self):
        rng = np.random.default_rng(19)
        a_vec = np.array([9.0, 4.0])
        b_vec = np.array([-4.0, 9.0])
        all_points = np.array([
            np.array([30.0, 25.0]) + i * a_vec + j * b_vec
            for j in range(5) for i in range(9)
        ])
        all_points += rng.normal(0.0, 0.08, all_points.shape)
        inferred = infer_atom_chain_from_anchors(
            all_points, [0, 8], label="a 方向原子列")
        fit = fit_lattice_vector_from_chain(all_points[inferred], min_points=2)

        np.testing.assert_array_equal(inferred, np.arange(9))
        np.testing.assert_allclose(fit.vector, a_vec, atol=0.08)

    def test_reverse_anchor_direction_returns_reverse_vector(self):
        all_points = np.array([(8.0 * x, 11.0 * y)
                               for y in range(3) for x in range(10)], dtype=float)

        inferred = infer_atom_chain_from_anchors(
            all_points, [8, 4, 0], label="a 方向原子列")
        fit = fit_lattice_vector_from_chain(all_points[inferred], min_points=2)

        np.testing.assert_array_equal(inferred, np.arange(8, -1, -1))
        np.testing.assert_allclose(fit.vector, [-8.0, 0.0], atol=1e-12)

    def test_off_line_anchor_is_rejected_instead_of_widening_band(self):
        all_points = np.array([(10.0 * x, 10.0 * y)
                               for y in range(5) for x in range(10)], dtype=float)
        with self.assertRaisesRegex(AnalysisError, "同一条原子列|偏离"):
            infer_atom_chain_from_anchors(
                all_points, [0, 5, 16], label="a 方向原子列")

    def test_sparse_anchors_on_two_oblique_axes_produce_unique_lattice_indices(self):
        # Regression for the packaged-v3.2 failure: each clicked gap spans four
        # periods, but automatic line completion must recover one-period a/b.
        origin = np.array([80.0, 60.0])
        a_true = np.array([55.0, -1.0])
        b_true = np.array([31.5, 45.0])
        n_a = n_b = 10
        all_points = np.array([
            origin + i * a_true + j * b_true
            for j in range(n_b) for i in range(n_a)
        ])
        a_indices = infer_atom_chain_from_anchors(all_points, [0, 4, 8])
        b_indices = infer_atom_chain_from_anchors(
            all_points, [0, 4 * n_a, 8 * n_a])
        a_fit, b_fit = estimate_reference_vectors_from_chains(
            all_points[a_indices], all_points[b_indices], min_points=2)

        np.testing.assert_allclose(a_fit.vector, a_true, atol=1e-12)
        np.testing.assert_allclose(b_fit.vector, b_true, atol=1e-12)
        assignment = assign_unique_lattice_indices(
            all_points, origin, a_fit.vector, b_fit.vector)
        self.assertEqual(
            len({tuple(index) for index in assignment.lattice_indices}),
            len(all_points))


class ReferenceRegionRefinementTests(unittest.TestCase):
    """Reference-region basis refinement must cut noise bias without eating strain."""

    @staticmethod
    def _strained_lattice(seed=7, size=12, strain=0.02, noise=0.10):
        """Left half unstrained reference region; right half uniformly stretched in x."""
        rng = np.random.default_rng(seed)
        a_true = np.array([10.0, 0.0])
        b_true = np.array([0.0, 10.0])
        ii, jj = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        n_idx, m_idx = ii.ravel(), jj.ravel()
        ideal = n_idx[:, None] * a_true + m_idx[:, None] * b_true
        pts = ideal.astype(float).copy()
        half = size // 2
        stretched = n_idx >= half
        pts[stretched, 0] += (n_idx[stretched] - half) * a_true[0] * strain
        pts += rng.normal(0, noise, pts.shape)
        region_mask = n_idx < half   # unstrained half
        return pts, region_mask, a_true, b_true

    def test_region_refinement_reduces_basis_error(self):
        pts, region_mask, a_true, b_true = self._strained_lattice()
        # Three-point basis: same construction the GUI produces from clicks.
        origin = pts[0].copy()
        a0 = pts[12] - pts[0]   # +1 along n (size=12 row stride)
        b0 = pts[1] - pts[0]    # +1 along m
        error_before = np.linalg.norm(a0 - a_true) / np.linalg.norm(a_true)

        o_r, a_r, b_r, applied, n_used = refine_lattice_basis_in_region(
            pts, origin, a0, b0, region_mask)
        self.assertTrue(applied)
        self.assertEqual(n_used, int(region_mask.sum()))
        error_after = np.linalg.norm(a_r - a_true) / np.linalg.norm(a_true)
        self.assertLess(error_after, error_before / 3.0)
        self.assertLess(error_after, 0.005)

    def test_region_refinement_preserves_strain_outside_region(self):
        strain = 0.02
        pts, region_mask, a_true, b_true = self._strained_lattice(strain=strain)
        origin = pts[0].copy()
        a0, b0 = pts[12] - pts[0], pts[1] - pts[0]
        o_r, a_r, b_r, applied, _ = refine_lattice_basis_in_region(
            pts, origin, a0, b0, region_mask)
        self.assertTrue(applied)
        # Refined basis must stay at the unstrained spacing, not absorb the 2%.
        self.assertLess(abs(np.linalg.norm(a_r) - np.linalg.norm(a_true)) / np.linalg.norm(a_true), 0.005)

        indices = np.round((pts - o_r) @ np.linalg.inv(np.column_stack((a_r, b_r))).T).astype(int)
        ideal = o_r + indices[:, 0:1] * a_r + indices[:, 1:2] * b_r
        result = compute_local_peak_pair_strain(indices, ideal, pts)
        outside = np.array([not region_mask[i] for i in result.site_indices])
        self.assertGreater(outside.sum(), 5)
        np.testing.assert_allclose(np.median(result.small_xx[outside]), strain, atol=0.006)

    def test_refinement_skipped_when_region_too_small(self):
        pts, _, _, _ = self._strained_lattice()
        origin, a0, b0 = pts[0].copy(), pts[12] - pts[0], pts[1] - pts[0]
        tiny = np.zeros(len(pts), dtype=bool)
        tiny[:5] = True
        o_r, a_r, b_r, applied, n_used = refine_lattice_basis_in_region(pts, origin, a0, b0, tiny)
        self.assertFalse(applied)
        self.assertEqual(n_used, 5)
        np.testing.assert_array_equal(a_r, a0)   # unchanged input on refusal
        self.assertFalse(refine_lattice_basis_in_region(pts, origin, a0, b0, None)[3])


class LatticeAssignmentAndHoleTests(unittest.TestCase):
    def test_global_assignment_resolves_rounding_collision_without_dropping_points(self):
        origin = np.array([0.0, 0.0])
        a_vec = np.array([10.0, 0.0])
        b_vec = np.array([0.0, 10.0])
        raw_points = np.array([[4.9, 0.0], [5.1, 0.0], [12.0, 0.0]])
        original = raw_points.copy()

        assignment = assign_unique_lattice_indices(raw_points, origin, a_vec, b_vec)

        np.testing.assert_array_equal(raw_points, original)
        self.assertEqual(len(assignment.lattice_indices), len(raw_points))
        self.assertEqual(len({tuple(row) for row in assignment.lattice_indices}), len(raw_points))
        np.testing.assert_array_equal(assignment.source_indices, np.arange(len(raw_points)))
        self.assertEqual(assignment.n_conflicts, 1)
        self.assertGreaterEqual(int(assignment.reassigned_mask.sum()), 1)

    def test_unique_naive_indices_are_preserved(self):
        origin = np.array([2.0, -3.0])
        a_vec = np.array([8.0, 1.0])
        b_vec = np.array([-1.0, 7.0])
        indices = np.array([[0, 0], [1, 0], [0, 1], [2, 2]])
        points = origin + indices[:, 0:1] * a_vec + indices[:, 1:2] * b_vec
        assignment = assign_unique_lattice_indices(points, origin, a_vec, b_vec)
        np.testing.assert_array_equal(assignment.lattice_indices, indices)
        self.assertFalse(assignment.reassigned_mask.any())
        self.assertFalse(assignment.low_confidence_mask.any())

    def test_assignment_is_deterministic_and_flags_forced_distant_match(self):
        origin = np.zeros(2)
        a_vec = np.array([10.0, 0.0])
        b_vec = np.array([0.0, 10.0])
        points = np.array([[0.0, 0.0], [1.0, 0.0]])
        first = assign_unique_lattice_indices(points, origin, a_vec, b_vec)
        second = assign_unique_lattice_indices(points, origin, a_vec, b_vec)
        np.testing.assert_array_equal(first.lattice_indices, second.lattice_indices)
        self.assertEqual(int(first.low_confidence_mask.sum()), 1)

    def test_lattice_hole_detection(self):
        gi, gj = np.meshgrid(np.arange(6), np.arange(6), indexing='ij')
        full = np.column_stack([gi.ravel(), gj.ravel()])
        self.assertEqual(find_lattice_holes(full), [])
        missing = ~((full[:, 0] == 3) & (full[:, 1] == 2))
        self.assertEqual(find_lattice_holes(full[missing]), [(3, 2)])


class GreedyNmsTests(unittest.TestCase):
    def test_nms_matches_brute_force_reference(self):
        rng = np.random.default_rng(0)
        coords = rng.integers(0, 60, size=(300, 2)).astype(float)
        selected = greedy_nms(coords, min_distance=4.0)
        brute = []
        for c in coords:
            if all(np.linalg.norm(c - s) > 4.0 for s in brute):
                brute.append(c)
        np.testing.assert_array_equal(selected, np.asarray(brute))

    def test_nms_suppresses_close_peaks_in_strength_order(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
        selected = greedy_nms(coords, min_distance=3.0)
        np.testing.assert_array_equal(selected, coords[[0, 2]])


class CentroidBatchTests(unittest.TestCase):
    def test_batch_com_matches_brute_force(self):
        from ppa import AtomMarkerApp

        rng = np.random.default_rng(3)
        image = rng.normal(size=(48, 48))
        ys, xs = np.mgrid[0:48, 0:48]
        for cy, cx in [(10.2, 12.7), (25.5, 28.1), (5.3, 33.9)]:
            image = image + 8.0 * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / 2.0)
        coords = np.array([[10, 13], [26, 28], [5, 34]])

        bx, by = AtomMarkerApp._refine_centroids_batch(None, image, coords, window=5, iterations=2)

        hw = 2
        padded = np.pad(image, hw, mode="reflect")
        centers_r = coords[:, 0].astype(int)
        centers_c = coords[:, 1].astype(int)
        ref_x = centers_c.astype(float)
        ref_y = centers_r.astype(float)
        for _ in range(2):
            rows, cols = centers_r + hw, centers_c + hw
            new_x = np.zeros(len(coords))
            new_y = np.zeros(len(coords))
            grid_y, grid_x = np.mgrid[0:5, 0:5]
            for i in range(len(coords)):
                roi = padded[rows[i] - hw:rows[i] + hw + 1, cols[i] - hw:cols[i] + hw + 1]
                roi_sub = np.maximum(roi - np.percentile(roi, 5), 0)
                total = roi_sub.sum()
                if total > 0:
                    new_x[i] = cols[i] - 2 * hw + (grid_x * roi_sub).sum() / total
                    new_y[i] = rows[i] - 2 * hw + (grid_y * roi_sub).sum() / total
                else:
                    new_x[i] = cols[i] - hw
                    new_y[i] = rows[i] - hw
            centers_c = np.clip(np.round(new_x).astype(int), 0, image.shape[1] - 1)
            centers_r = np.clip(np.round(new_y).astype(int), 0, image.shape[0] - 1)
            ref_x, ref_y = new_x, new_y
        np.testing.assert_allclose(bx, ref_x, atol=1e-10)
        np.testing.assert_allclose(by, ref_y, atol=1e-10)


class PhysicalCoordinateImportTests(unittest.TestCase):
    """v3 exports physical y-up coordinates; ppa_stats must plot in display y-down."""

    STRAIN_HEADER = ("原子位点编号,位置x,位置y,ε_xx,ε_yy,ε_xy (张量剪应变),"
                     "ε_eq (von Mises),极分解旋转 θ (rad)\n")

    def test_convention_detection(self):
        self.assertTrue(detect_physical_convention(np.array([-10.0, -50.0, 0.0])))
        self.assertFalse(detect_physical_convention(np.array([10.0, 50.0])))
        self.assertFalse(detect_physical_convention(np.array([])))
        # Sidecar metadata wins even when values are ambiguous (all zero).
        self.assertTrue(detect_physical_convention(
            np.array([0.0, 0.0]),
            {'coordinate_convention': 'physical-cartesian-x-right-y-up; image-display-y-down'}))
        # 显式显示坐标约定按显示坐标处理, 不再落入启发式
        self.assertFalse(detect_physical_convention(
            np.array([-5.0]),
            {'coordinate_convention': 'image-display-y-down'}))
        # 参考原点在视场中部 → 物理 y 有正有负, 行号不可能为负, 应判为物理坐标
        self.assertTrue(detect_physical_convention(np.array([-30.0, 10.0, 55.0])))

    def test_convention_note_only_without_sidecar(self):
        from ppa_stats import convention_detection_note

        sidecar = {'coordinate_convention': 'physical-cartesian-x-right-y-up; image-display-y-down'}
        self.assertIsNone(convention_detection_note(np.array([-10.0]), sidecar))
        note = convention_detection_note(np.array([-10.0]), {})
        self.assertIsNotNone(note)
        self.assertIn("物理坐标", note)
        note_display = convention_detection_note(np.array([10.0]), {})
        self.assertIsNotNone(note_display)
        self.assertIn("显示坐标", note_display)

    def test_strain_csv_physical_y_is_flipped_with_shear_and_rotation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "strain.csv"
            csv_path.write_text(
                self.STRAIN_HEADER + "1,100.0,-120.0,0.01,-0.005,0.002,0.011,0.0003\n",
                encoding="utf-8")
            loaded = load_strain_csv(csv_path)
            self.assertTrue(loaded['physical_source'])
            np.testing.assert_allclose(loaded['centroids'], [[100.0, 120.0]])
            np.testing.assert_allclose(loaded['strain_xy'], [-0.002])
            np.testing.assert_allclose(loaded['rotation'], [-0.0003])
            np.testing.assert_allclose(loaded['strain_xx'], [0.01])  # normal strain unchanged

    def test_display_coordinate_csv_is_left_untouched(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "strain_display.csv"
            csv_path.write_text(
                self.STRAIN_HEADER + "1,100.0,120.0,0.01,-0.005,0.002,0.011,0.0003\n",
                encoding="utf-8")
            loaded = load_strain_csv(csv_path)
            self.assertFalse(loaded['physical_source'])
            np.testing.assert_allclose(loaded['centroids'], [[100.0, 120.0]])
            np.testing.assert_allclose(loaded['strain_xy'], [0.002])

    def test_gl_columns_are_parsed_and_shear_flipped_for_physical_source(self):
        header = (self.STRAIN_HEADER.rstrip("\n")
                  + ",ε_xx (GL),ε_yy (GL),ε_xy (GL),ε_eq (GL von Mises)\n")
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "strain_gl.csv"
            csv_path.write_text(
                header + "1,10.0,-20.0,0.01,-0.005,0.002,0.011,0.0003,0.012,-0.004,0.003,0.013\n",
                encoding="utf-8")
            loaded = load_strain_csv(csv_path)
            self.assertTrue(loaded['physical_source'])
            np.testing.assert_allclose(loaded['gl_xx'], [0.012])
            np.testing.assert_allclose(loaded['gl_yy'], [-0.004])
            np.testing.assert_allclose(loaded['gl_xy'], [-0.003])  # 剪切随 y 翻转变号
            np.testing.assert_allclose(loaded['gl_eq'], [0.013])   # 不变量不变号
            self.assertIsNotNone(loaded['convention_note'])        # 无 sidecar → 附判定提示

    def test_displacement_csv_physical_import_and_sidecar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "disp.csv"
            csv_path.write_text(
                "编号,实际x（物理）,实际y（物理）,参考x（物理）,参考y（物理）,"
                "位移dx（物理）,位移dy（物理）,位移幅值\n"
                "1,10.0,-20.0,10.5,-20.5,-0.5,0.5,0.7071\n", encoding="utf-8")
            (Path(temp_dir) / "disp.metadata.json").write_text(
                '{"coordinate_convention": "physical-cartesian-x-right-y-up; image-display-y-down"}',
                encoding="utf-8")
            loaded = load_displacement_csv(csv_path)
            self.assertTrue(loaded['physical_source'])
            np.testing.assert_allclose(loaded['actual_pos'], [[10.0, 20.0]])
            np.testing.assert_allclose(loaded['ideal_pos'], [[10.5, 20.5]])
            np.testing.assert_allclose(loaded['displacements'], [[-0.5, -0.5]])
            np.testing.assert_allclose(loaded['distortions'], [0.7071])  # magnitude invariant

    def test_csv_import_rejects_nan_and_short_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bad_strain = Path(temp_dir) / "strain_nan.csv"
            bad_strain.write_text(
                "三角形编号,位置x,位置y,ε_xx,ε_yy,ε_xy (张量剪应变),ε_eq (von Mises),极分解旋转 θ (rad)\n"
                "1,0,0,0.1,nan,0,0.1,0\n",
                encoding="utf-8")
            with self.assertRaises(ValueError):
                load_strain_csv(bad_strain)

            bad_disp = Path(temp_dir) / "disp_short.csv"
            bad_disp.write_text(
                "编号,x,y\n1,0,0\n",
                encoding="utf-8")
            with self.assertRaises(ValueError):
                load_displacement_csv(bad_disp)

    def test_site_strain_csv_keeps_invalid_atom_row_with_nan_reason(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "site_strain.csv"
            csv_path.write_text(
                "原子位点编号,位置x,位置y,ε_xx,ε_yy,ε_xy (张量剪应变),ε_eq (von Mises),极分解旋转 θ (rad),"
                "原始点编号,晶格n,晶格m,应变有效,计算质量,无效原因\n"
                "1,0,0,0.01,0.02,0.0,0.02,0.0,1,0,0,1,A-symmetric,\n"
                "2,1,0,nan,nan,nan,nan,nan,2,1,0,0,invalid,insufficient_noncollinear_neighbors\n",
                encoding="utf-8",
            )
            loaded = load_strain_csv(csv_path)
            self.assertEqual(loaded['n'], 2)
            np.testing.assert_array_equal(loaded['valid_mask'], [True, False])
            self.assertTrue(np.isnan(loaded['strain_xx'][1]))
            np.testing.assert_array_equal(loaded['source_indices'], [0, 1])
            np.testing.assert_array_equal(loaded['lattice_indices'], [[0, 0], [1, 0]])
            self.assertEqual(loaded['quality_grades'][1], 'invalid')
            self.assertEqual(loaded['invalid_reasons'][1], 'insufficient_noncollinear_neighbors')

    def test_site_strain_csv_rejects_finite_value_marked_invalid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "false_invalid.csv"
            csv_path.write_text(
                "原子位点编号,位置x,位置y,ε_xx,ε_yy,ε_xy (张量剪应变),ε_eq (von Mises),极分解旋转 θ (rad),应变有效\n"
                "1,0,0,0,0,0,0,0,0\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_strain_csv(csv_path)


class StatisticsEdgeCaseTests(unittest.TestCase):
    def test_weighted_mean_std_ignores_nan_without_weights(self):
        mean, std = weighted_mean_std(np.array([0.0, 1.0, np.nan]))
        self.assertAlmostEqual(mean, 0.5)
        self.assertTrue(np.isfinite(std))

    def test_weighted_mean_std_all_invalid_returns_nan(self):
        mean, std = weighted_mean_std(np.array([np.nan]), np.array([1.0]))
        self.assertTrue(np.isnan(mean) and np.isnan(std))

    def test_zero_second_principal_strain_is_uniaxial_tension(self):
        modes, _ = compute_deformation_mode(np.array([0.10]), np.array([0.0]))
        self.assertEqual(int(modes[0]), 2)
        modes, _ = compute_deformation_mode(np.array([0.10]), np.array([0.05]))
        self.assertEqual(int(modes[0]), 2)
        modes, _ = compute_deformation_mode(np.array([0.10]), np.array([0.09]))
        self.assertEqual(int(modes[0]), 3)


class DisplayConventionTests(unittest.TestCase):
    """物理 y-up 的剪切/旋转在 y-down 显示轴上必须变号；正应变/等效应变不变。"""

    def test_helper_flips_only_shear_and_rotation(self):
        values = np.array([0.01, -0.02])
        np.testing.assert_array_equal(strain_value_for_display('xy', values), -values)
        np.testing.assert_array_equal(strain_value_for_display('gl_xy', values), -values)
        np.testing.assert_array_equal(strain_value_for_display('rot', values), -values)
        np.testing.assert_array_equal(strain_value_for_display('xx', values), values)
        np.testing.assert_array_equal(strain_value_for_display('eq', values), values)
        self.assertIsNone(strain_value_for_display('xy', None))

    def test_interpolated_grids_use_display_convention(self):
        import ppa

        fields = {
            'tri_centroids': square_grid(4) * 100.0,
            'strain_xx': np.full(16, 0.01),
            'strain_yy': np.full(16, -0.02),
            'strain_xy': np.full(16, 0.03),
            'strain_eq': np.full(16, 0.04),
            'rotation': np.full(16, 0.05),
            'strain_gl_xx': np.full(16, 0.01),
            'strain_gl_yy': np.full(16, 0.01),
            'strain_gl_xy': np.full(16, 0.06),
            'strain_gl_eq': np.full(16, 0.01),
        }
        strain_grid, strain_grid_gl, extent = ppa._interpolate_strain_grids(
            fields, image_shape=(400, 400), grid_size=15)
        self.assertEqual(extent, (0, 399, 0, 399))
        # 凸包外保持 NaN 是设计行为; 只比较凸包内的有效格点
        valid = ~np.isnan(strain_grid['xx'])

        def inside(grid):
            return grid[valid]

        np.testing.assert_allclose(inside(strain_grid['xx']), 0.01, atol=1e-9)
        np.testing.assert_allclose(inside(strain_grid['eq']), 0.04, atol=1e-9)
        np.testing.assert_allclose(inside(strain_grid['xy']), -0.03, atol=1e-9)
        np.testing.assert_allclose(inside(strain_grid['rot']), -0.05, atol=1e-9)
        np.testing.assert_allclose(inside(strain_grid_gl['gl_xy']), -0.06, atol=1e-9)


class PointEditInvalidationTests(unittest.TestCase):
    """任何修改点表的操作都必须使分析结果与选择状态失效。"""

    def _bare_app(self):
        from ppa import AtomMarkerApp

        app = AtomMarkerApp.__new__(AtomMarkerApp)
        app.points = [(0.0, 0.0), (0.5, 0.0), (10.0, 0.0), (10.5, 0.0), (20.0, 10.0)]
        app.undo_stack = []
        app.detect_min_dist = 10.0
        app.selected_point_idx = 1
        app.ref_select_indices = [0, 1, 2]
        app._job_generation = 0  # _clear_analysis_results 会递增它使后台任务失效
        app._clear_analysis_results()
        app.displacements = np.ones(5)
        app.distortions = np.ones(5)
        app._refresh_overlay = lambda: None
        app.status = type("Status", (), {"config": staticmethod(lambda **kw: None)})()
        return app

    def test_dedup_invalidates_stale_results_and_selection(self):
        app = self._bare_app()
        removed = app.deduplicate_points()
        self.assertEqual(removed, 2)          # 0.5/10.5 距邻近点 < 10*0.3
        self.assertEqual(len(app.points), 3)
        self.assertIsNone(app.displacements)  # 旧结果不得残留
        self.assertIsNone(app.distortions)
        self.assertEqual(app.ref_select_indices, [])
        self.assertIsNone(app.selected_point_idx)

    def test_undo_invalidates_selection_after_index_shift(self):
        app = self._bare_app()
        app.undo_stack.append(("clear", list(app.points)))
        app.points.clear()
        app.undo_last()
        self.assertEqual(len(app.points), 5)
        self.assertEqual(app.ref_select_indices, [])

    def test_fourth_click_after_cancel_restarts_selection(self):
        from ppa import AtomMarkerApp

        app = AtomMarkerApp.__new__(AtomMarkerApp)
        app.points = [(0.0, 0.0), (50.0, 0.0), (0.0, 50.0), (50.0, 50.0)]
        app.pick_radius = 10.0
        app.ref_select_indices = [0, 1, 2]   # 确认框「取消」后残留的待确认状态
        app.redraw_points = lambda: None
        app.canvas = type("Canvas", (), {"draw_idle": staticmethod(lambda: None)})()
        app.status = type("Status", (), {"config": staticmethod(lambda **kw: None)})()
        event = type("Event", (), {"xdata": 50.0, "ydata": 50.0})()
        app._handle_ref_select_click(event)
        # 第 4 次点击不得把选择列表推到长度 4 (高亮颜色表只有 3 色)
        self.assertEqual(app.ref_select_indices, [3])


class MultiPointReferenceGuiStateTests(unittest.TestCase):
    def test_confirmed_two_chain_fit_sets_reference_and_reproducibility_metadata(self):
        import ppa as ppa_module
        from ppa import AtomMarkerApp
        from unittest import mock

        class _Var:
            def __init__(self, value=None):
                self.value = value

            def set(self, value):
                self.value = value

        class _Widget:
            def __init__(self):
                self.last = {}

            def config(self, **kwargs):
                self.last = kwargs

        app = AtomMarkerApp.__new__(AtomMarkerApp)
        a_chain = [(10.0 * i, 0.0) for i in range(6)]
        b_chain = [(5.0, 20.0 + 12.0 * i) for i in range(6)]
        app.points = a_chain + b_chain
        # Only sparse anchors are clicked; the GUI must automatically include
        # the intermediate detected atoms on both lines.
        app.ref_multi_indices = [[0, 3, 5], [6, 9, 11]]
        app.ref_multi_stage = 1
        app.reference_vecs = None
        app.ref_origin = None
        app.reference_metadata = None
        app.ref_region = None
        app._job_generation = 0  # _clear_analysis_results 会递增它使后台任务失效
        app.mode_var = _Var("select_ref")
        app.root = type("Root", (), {"unbind": staticmethod(lambda *args: None)})()
        app.status = _Widget()
        app.ref_status_label = _Widget()
        app.ref_region_label = _Widget()
        app.redraw_points = lambda: None
        app.canvas = type("Canvas", (), {"draw_idle": staticmethod(lambda: None)})()

        with mock.patch.object(ppa_module.messagebox, "askyesnocancel", return_value=True):
            app._confirm_ref_multi_select()

        np.testing.assert_allclose(app.reference_vecs[0], [10.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(app.reference_vecs[1], [0.0, 12.0], atol=1e-12)
        np.testing.assert_allclose(app.ref_origin, a_chain[0], atol=1e-12)
        self.assertEqual(app.reference_metadata['method'], 'two-chain-robust-fit')
        self.assertEqual(app.reference_metadata['a_anchor_count'], 3)
        self.assertEqual(app.reference_metadata['a_point_count'], 6)
        self.assertEqual(app.ref_multi_indices, [[], []])
        self.assertEqual(app.current_mode, 'add')


class EvalDirResolutionTests(unittest.TestCase):
    def test_resolve_eval_dirs_returns_explicit_image_and_label_paths(self):
        from atom_detector.infer.benchmark import resolve_eval_dirs
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "images" / "test").mkdir(parents=True)
            (root / "labels" / "test").mkdir(parents=True)
            img_dir, lbl_dir = resolve_eval_dirs(str(root), "test")
            self.assertEqual(img_dir, root / "images" / "test")
            self.assertEqual(lbl_dir, root / "labels" / "test")

    def test_resolve_eval_dirs_falls_back_to_simple_structure(self):
        from atom_detector.infer.benchmark import resolve_eval_dirs
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            img_dir, lbl_dir = resolve_eval_dirs(str(root), "test")
            self.assertEqual(img_dir, root / "images")
            self.assertEqual(lbl_dir, root / "labels")


class ModelSecurityTests(unittest.TestCase):
    def test_unknown_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model.pt"
            model.write_bytes(b"fake torch weights")
            allowlist = Path(temp_dir) / "allowlist.json"
            allowlist.write_text(
                '{"sha256": ["' + "0" * 64 + '"]}', encoding="utf-8")
            with self.assertRaises(ModelVerificationError):
                verify_model(model, allowlist)

    def test_allowlist_rejects_non_hex_digest(self):
        # 64 位长度但含非十六进制字符的摘要必须整体被拒 (fail-closed)
        with tempfile.TemporaryDirectory() as temp_dir:
            allowlist = Path(temp_dir) / "allowlist.json"
            allowlist.write_text(
                '{"sha256": ["' + "z" * 64 + '"]}', encoding="utf-8")
            with self.assertRaises(ModelVerificationError):
                load_allowlist(allowlist)

    def test_allowlisted_model_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model.pt"
            model.write_bytes(b"fake torch weights")
            digest = hashlib.sha256(b"fake torch weights").hexdigest()
            allowlist = Path(temp_dir) / "allowlist.json"
            allowlist.write_text(
                f'{{"sha256": ["{digest}"]}}', encoding="utf-8")
            self.assertEqual(verify_model(model, allowlist), digest)


class PersistenceAndImageTests(unittest.TestCase):
    def test_stack_requires_explicit_frame_and_preserves_frame_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "stack.tif"
            stack = np.stack((np.zeros((4, 5), dtype=np.uint16), np.ones((4, 5), dtype=np.uint16) * 8))
            tifffile.imwrite(image_path, stack)
            with self.assertRaises(ImageLoadError):
                load_analysis_image(image_path)
            loaded = load_analysis_image(image_path, frame_index=1)
            self.assertEqual(loaded.frame_count, 2)
            self.assertEqual(loaded.frame_index, 1)
            self.assertEqual(loaded.pixels.shape, (4, 5))

    def test_project_requires_identical_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.tif"
            project_path = Path(temp_dir) / "case.json"
            tifffile.imwrite(image_path, np.arange(25, dtype=np.uint16).reshape(5, 5))
            save_project(project_path, {"points": [[1.0, 2.0]], "algorithm": {"id": "lattice-cst"}}, image_path)
            loaded = load_project(project_path)
            self.assertEqual(loaded["schema_version"], 2)
            tifffile.imwrite(image_path, np.zeros((5, 5), dtype=np.uint16))
            with self.assertRaises(ProjectValidationError):
                load_project(project_path)

    def test_preprocess_settings_survive_save_load_roundtrip(self):
        # save_project 写出 'preprocess' 块, load_project 必须把它恢复回来;
        # 此前只存不读, 重开项目后滤波方法/参数/开关全部丢失。
        import ppa as ppa_module
        from ppa import AtomMarkerApp
        from unittest import mock

        class _Var:
            def __init__(self, value=None):
                self._value = value

            def get(self):
                return self._value

            def set(self, value):
                self._value = value

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "img.tif"
            project_path = Path(temp_dir) / "project.json"
            tifffile.imwrite(image_path, np.arange(16, dtype=np.uint16).reshape(4, 4))

            app = AtomMarkerApp.__new__(AtomMarkerApp)
            app.image = np.zeros((4, 4))
            app.image_path = str(image_path)
            app.image_frame_index = None
            app.image_metadata = None
            app.points = [(1.0, 1.0)]
            app.detect_roi = None
            app.detect_polygon = None
            app.reference_vecs = (np.array([10.0, 0.0]), np.array([0.0, 10.0]))
            app.ref_origin = np.array([1.0, 1.0])
            app.reference_metadata = {
                'method': 'two-chain-robust-fit',
                'estimator': '2d-huber-linear-fit-v1',
                'a_point_count': 10,
                'b_point_count': 9,
            }
            app.ref_region = None
            app.detect_sigma = 1.5
            app.detect_min_dist = 6.0
            app.detect_window = 5
            app.centroid_method = "com"
            app.preprocess_method = "gaussian"
            app.preprocess_params = {
                'gaussian_sigma': 2.5, 'median_size': 3, 'bw_order': 4,
                'bw_cutoff': 0.3, 'delta': 5.0, 'cycles': 99, 'step': 2,
            }
            app.use_preprocessed = _Var(True)
            app.analysis_method = "peak_pairs"
            app.von_mises_coeff = 4.0 / 9.0
            app.outlier_mask = None
            app.lattice_indices = np.array([[0, 0]])
            app.lattice_coords = np.array([[0.05, -0.02]])
            app.assignment_residuals = np.array([0.05385])
            app.assignment_residuals_px = np.array([0.2])
            app.assignment_reassigned_mask = np.array([False])
            app.assignment_low_confidence_mask = np.array([False])
            app.assignment_conflict_count = 0
            app.strain_quality_grades = np.array(['A-symmetric'])
            app.strain_invalid_reasons = np.array([''])
            app.colorbar_manual = False
            app.colorbar_vmin = _Var(0.0)
            app.colorbar_vmax = _Var(1.0)
            app.status = type("Status", (), {"config": staticmethod(lambda **kw: None)})()
            fake_dialog = type("FD", (), {
                "asksaveasfilename": staticmethod(lambda **kw: str(project_path))})()
            with mock.patch.object(ppa_module, "filedialog", fake_dialog):
                AtomMarkerApp.save_project(app)

            data = load_project(project_path)
            self.assertEqual(data["preprocess"]["method"], "gaussian")
            self.assertEqual(data["preprocess"]["params"]["gaussian_sigma"], 2.5)
            self.assertTrue(data["preprocess"]["use_preprocessed"])
            self.assertEqual(data["reference"]["estimation"]["method"], 'two-chain-robust-fit')
            self.assertEqual(data["reference"]["estimation"]["a_point_count"], 10)
            assignment = data["analysis"]["lattice_assignment"]
            self.assertEqual(assignment["lattice_indices"], [[0, 0]])
            self.assertEqual(assignment["strain_quality"], ['A-symmetric'])

            app2 = AtomMarkerApp.__new__(AtomMarkerApp)
            app2.preprocess_method = "none"
            app2.preprocess_params = {
                'gaussian_sigma': 1.0, 'median_size': 3, 'bw_order': 4,
                'bw_cutoff': 0.3, 'delta': 5.0, 'cycles': 99, 'step': 2,
            }
            app2.preprocess_method_var = _Var("无")
            app2.use_preprocessed = _Var(False)
            app2._on_preprocess_method_changed = lambda: None
            app2._restore_preprocess_settings(data)
            self.assertEqual(app2.preprocess_method, "gaussian")
            self.assertEqual(app2.preprocess_method_var.get(), "Gaussian 高斯模糊")
            self.assertEqual(app2.preprocess_params["gaussian_sigma"], 2.5)
            self.assertTrue(app2.use_preprocessed.get())


class ExportCompletenessTests(unittest.TestCase):
    def test_displacement_export_has_one_row_per_confirmed_atom(self):
        import ppa as ppa_module
        from ppa import AtomMarkerApp
        from unittest import mock

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "displacement.csv"
            app = AtomMarkerApp.__new__(AtomMarkerApp)
            app.displacements = np.array([[0.1, 0.0], [0.0, 0.2], [-0.1, 0.0]])
            app.matched_actual = np.array([[0.1, 0.0], [1.0, 0.2], [1.9, 0.0]])
            app.ideal_grid = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
            app.distortions = np.linalg.norm(app.displacements, axis=1)
            app.lattice_indices = np.array([[0, 0], [1, 0], [2, 0]])
            app.assignment_residuals = np.array([0.1, 0.2, 0.1])
            app.assignment_residuals_px = np.array([0.1, 0.2, 0.1])
            app.assignment_reassigned_mask = np.array([False, True, False])
            app.assignment_low_confidence_mask = np.array([False, False, False])
            app.analysis_method = 'peak_pairs'
            app.image_path = 'synthetic.tif'
            app.image_frame_index = None
            app.status = type("Status", (), {"config": staticmethod(lambda **kw: None)})()
            fake_dialog = type("FD", (), {
                "asksaveasfilename": staticmethod(lambda **kw: str(csv_path))})()
            with mock.patch.object(ppa_module, "filedialog", fake_dialog):
                AtomMarkerApp.save_displacement_csv(app)
            with csv_path.open('r', encoding='utf-8', newline='') as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(len(rows) - 1, 3)
            self.assertIn('晶格n', rows[0])
            self.assertEqual(rows[2][-2:], ['1', '0'])

    def test_site_strain_export_keeps_invalid_atom_as_nan_row(self):
        import ppa as ppa_module
        from ppa import AtomMarkerApp
        from unittest import mock

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "strain.csv"
            app = AtomMarkerApp.__new__(AtomMarkerApp)
            values = np.array([0.01, np.nan, 0.02])
            app.tri_centroids = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
            app.result_locations_physical = app.tri_centroids.copy()
            app.strain_xx = values.copy()
            app.strain_yy = values.copy()
            app.strain_xy = values.copy()
            app.strain_eq = values.copy()
            app.rotation = values.copy()
            app.strain_gl_xx = values.copy()
            app.strain_gl_yy = values.copy()
            app.strain_gl_xy = values.copy()
            app.strain_gl_eq = values.copy()
            app.tri_edge_mask = None
            app.element_area = None
            app.strain_geometry = 'sites'
            app.lattice_indices = np.array([[0, 0], [1, 0], [2, 0]])
            app.strain_quality_grades = np.array(['A-symmetric', 'invalid', 'B-local-fit'])
            app.strain_invalid_reasons = np.array(['', 'insufficient_noncollinear_neighbors', ''])
            app.analysis_method = 'peak_pairs'
            app.von_mises_coeff = 4.0 / 9.0
            app.image_path = 'synthetic.tif'
            app.image_frame_index = None
            app.status = type("Status", (), {"config": staticmethod(lambda **kw: None)})()
            fake_dialog = type("FD", (), {
                "asksaveasfilename": staticmethod(lambda **kw: str(csv_path))})()
            with mock.patch.object(ppa_module, "filedialog", fake_dialog):
                AtomMarkerApp.save_strain_csv(app)
            loaded = load_strain_csv(csv_path)
            self.assertEqual(loaded['n'], 3)
            np.testing.assert_array_equal(loaded['valid_mask'], [True, False, True])
            self.assertTrue(np.isnan(loaded['strain_xx'][1]))
            self.assertEqual(loaded['invalid_reasons'][1], 'insufficient_noncollinear_neighbors')


class StatisticsAndLabelReviewTests(unittest.TestCase):
    def test_gradient_is_distance_normalized(self):
        centroids = square_grid(5)
        exx = 2.0 * centroids[:, 0] - 3.0 * centroids[:, 1]
        eyy = -4.0 * centroids[:, 0]
        exy = 0.5 * centroids[:, 1]
        gxx, gyy, gxy = compute_strain_gradient(exx, eyy, exy, centroids)
        interior = np.array([6, 7, 8, 11, 12, 13, 16, 17, 18])
        np.testing.assert_allclose(gxx[interior], math.sqrt(13.0), atol=1e-12)
        np.testing.assert_allclose(gyy[interior], 4.0, atol=1e-12)
        np.testing.assert_allclose(gxy[interior], 0.5, atol=1e-12)

    def test_gradient_rank_deficient_neighbors_are_nan_not_zero(self):
        # 共线邻域无法定梯度: 0 是合法梯度值会污染统计, 必须记 NaN
        centroids = np.array([[float(x), 0.0] for x in range(6)])
        gxx, _, _ = compute_strain_gradient(np.arange(6.0), np.arange(6.0), np.arange(6.0), centroids)
        self.assertTrue(np.isnan(gxx).all())
        # 点数不足以构造平面拟合时同样全 NaN (2 点无法定梯度)
        tiny = np.array([[0.0, 0.0], [1.0, 1.0]])
        gxx2, _, _ = compute_strain_gradient(tiny[:, 0], tiny[:, 0], tiny[:, 0], tiny)
        self.assertTrue(np.isnan(gxx2).all())

    def test_poisson_contraction_is_not_labelled_pure_shear(self):
        modes, _ = compute_deformation_mode(np.array([0.10]), np.array([-0.03]))
        self.assertEqual(int(modes[0]), 2)

    def test_area_weighted_statistics_and_extended_strain_csv(self):
        mean, std = weighted_mean_std(np.array([0.0, 1.0]), np.array([1.0, 3.0]))
        self.assertAlmostEqual(mean, 0.75)
        self.assertAlmostEqual(std, math.sqrt(0.1875))
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "strain.csv"
            csv_path.write_text(
                "三角形编号,位置x,位置y,ε_xx,ε_yy,ε_xy (张量剪应变),ε_eq (von Mises),极分解旋转 θ (rad),边缘三角形,参考三角形面积\n"
                "1,0,0,0.1,0,0,0.1,0,0,2.5\n", encoding="utf-8")
            loaded = load_strain_csv(csv_path)
            np.testing.assert_allclose(loaded['element_area'], [2.5])
            np.testing.assert_array_equal(loaded['edge_mask'], [False])

    def test_drag_records_original_position_and_clear_is_undoable(self):
        app = LabelReviewApp.__new__(LabelReviewApp)
        app.points = [(10.0, 10.0), (30.0, 30.0)]
        app.undo_stack = []
        app._dragging_idx = 0
        app._drag_original = (10.0, 10.0)
        app._modified = False
        app._redraw = lambda: None
        app._update_info = lambda: None
        event = type("Event", (), {"xdata": 20.0, "ydata": 20.0})()
        LabelReviewApp._on_motion(app, event)
        LabelReviewApp._on_release(app, event)
        self.assertEqual(app.points[0], (20.0, 20.0))
        self.assertEqual(app.undo_stack[-1], ('move', 0, (10.0, 10.0)))
        LabelReviewApp._undo(app)
        self.assertEqual(app.points[0], (10.0, 10.0))
        LabelReviewApp._clear_all(app)
        LabelReviewApp._undo(app)
        self.assertEqual(app.points, [(10.0, 10.0), (30.0, 30.0)])


if __name__ == "__main__":
    unittest.main()
