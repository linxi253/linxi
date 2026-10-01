import itertools
import numpy as np
import pytest
from atom_center.coordinates import Letterbox, points_to_yolo, yolo_to_points
from atom_center.interfaces import CandidateSet
from atom_center.metrics import match_points
from atom_center.pipeline import DetectionPipeline, PipelineConfig
from atom_center.refinement import refine_with_diagnostics
from atom_center.splitting import grouped_split_records


def test_half_pixel_letterbox_and_serialized_round_trip():
    points = np.array([[0., 0.], [11.25, 15.75], [92.1, 28.3]])
    transform = Letterbox.create((37, 101), (128, 128))
    np.testing.assert_allclose(transform.inverse_centers(transform.forward_points(points)), points, atol=1e-12)
    text = np.array([[float(f"{v:.8f}") for v in row] for row in points_to_yolo(points, 101, 37)])
    np.testing.assert_allclose(yolo_to_points(text, 101, 37), points, atol=1e-5)


def test_cardinality_has_priority_over_sum_of_distances():
    predicted = np.column_stack((np.arange(5), np.zeros(5)))
    actual = match_points(predicted, predicted+[-.99, 0], max_distance_px=1)
    assert actual.true_positives == 5


def test_matching_matches_independent_brute_force_objective():
    rng = np.random.default_rng(321)
    for _ in range(15):
        predicted, truth = rng.random((4, 2)), rng.random((4, 2))
        distances = np.linalg.norm(predicted[:, None]-truth, axis=2)
        choices = range(-1, 4)
        solutions = []
        for assignment in itertools.product(choices, repeat=4):
            used = [j for j in assignment if j >= 0]
            if len(used) != len(set(used)):
                continue
            if any(j >= 0 and distances[i, j] > .5 for i, j in enumerate(assignment)):
                continue
            solutions.append((-len(used), sum(distances[i, j] for i, j in enumerate(assignment) if j >= 0)))
        actual = match_points(predicted, truth, max_distance_px=.5)
        expected = min(solutions)
        assert actual.true_positives == -expected[0]
        assert actual.distances_px.sum() == pytest.approx(expected[1])


def test_flat_and_boundary_fits_preserve_fractional_candidates():
    points = [[12.3, 14.7], [.2, .4]]
    result = refine_with_diagnostics(np.ones((32, 32)), points, method="gaussian")
    np.testing.assert_array_equal(result.points, points)
    assert [q["status"] for q in result.diagnostics] == ["flat", "image_boundary"]


def test_raw_full_image_refinement_has_roi_context():
    yy, xx = np.mgrid[:64, :64]
    true = np.array([[20.35, 28.65]])
    raw = 20 + 1000*np.exp(-((xx-true[0, 0])**2+(yy-true[0, 1])**2)/(2*1.3**2))

    class Backend:
        name = "fixed-candidate"
        model_sha256 = None
        def predict(self, patch):
            # Candidate is one pixel from ROI boundary, but has full-image context.
            return CandidateSet([[1., 9.]], [.9])

    config = PipelineConfig(tile_size=64, refinement_window=9,
                            refinement_method="gaussian", refinement_polarity="bright")
    result = DetectionPipeline(Backend(), config).detect(raw, roi=(19, 20, 40, 44))
    assert np.linalg.norm(result.points-true) < .01
    assert result.metadata["point_quality"][0]["status"] == "refined"


def test_group_minimums_and_whitespace_are_enforced():
    rows = [{"acquisition_id": f" group-{i} ", "id": str(i)} for i in range(6)]
    result = grouped_split_records(rows, min_groups_per_split={"test": 2})
    assert result.group_counts["test"] >= 2
    assert all(group == group.strip() for group in result.group_to_split)
    with pytest.raises(ValueError):
        grouped_split_records(rows[:3], min_groups_per_split={"test": 2})


def test_matching_large_separated_grid_and_invalid_threshold():
    x, y = np.meshgrid(np.arange(80)*8., np.arange(60)*8.)
    points = np.column_stack((x.ravel(), y.ravel()))
    result = match_points(points+[.1, -.1], points, max_distance_px=.3)
    assert result.true_positives == 4800
    for distance in [float("nan"), float("inf"), -1]:
        with pytest.raises(ValueError):
            match_points(points[:1], points[:1], max_distance_px=distance)


def test_excessive_refinement_shift_retains_candidate_and_diagnosis():
    yy, xx = np.mgrid[:32, :32]
    image = 100+1000*np.exp(-((xx-17.5)**2+(yy-16.)**2)/2)
    candidate = [[15.7, 16.2]]
    result = refine_with_diagnostics(image, candidate, window_size=9, max_shift_px=.4)
    np.testing.assert_array_equal(result.points, candidate)
    assert result.diagnostics[0]["status"] == "excessive_shift"
    assert result.diagnostics[0]["proposed_shift_px"] > .4
