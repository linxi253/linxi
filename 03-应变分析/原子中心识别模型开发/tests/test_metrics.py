import math

import numpy as np

from atom_center.metrics import (
    evaluate_points,
    nearest_neighbor_spacing,
    strain_noise_bound,
)


def test_point_metrics_include_detection_and_localization() -> None:
    truth = np.asarray([[0.0, 0.0], [10.0, 0.0]])
    predictions = np.asarray([[0.1, 0.0], [10.0, 0.2], [20.0, 20.0]])
    metrics = evaluate_points(predictions, truth, max_distance_px=0.5)
    assert (metrics.true_positives, metrics.false_positives, metrics.false_negatives) == (2, 1, 0)
    assert metrics.precision == 2 / 3
    assert metrics.recall == 1.0
    assert math.isclose(metrics.localization_rmse_px, math.sqrt(0.025))


def test_fraction_threshold_uses_lattice_spacing() -> None:
    truth = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    assert nearest_neighbor_spacing(truth) == 10.0
    metrics = evaluate_points(
        truth + [1.0, 0.0], truth, max_distance_fraction=0.15
    )
    assert metrics.match_distance_px == 1.5
    assert metrics.true_positives == 3


def test_strain_noise_bound() -> None:
    assert math.isclose(strain_noise_bound(0.1, 10.0), math.sqrt(2) * 0.01)
