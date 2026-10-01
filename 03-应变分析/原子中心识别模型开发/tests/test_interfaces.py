import numpy as np
import pytest

from atom_center.interfaces import CandidateSet, DetectionResult


def test_candidate_set_normalizes_empty_shape() -> None:
    result = CandidateSet([], [])
    assert result.points.shape == (0, 2)
    assert result.confidences.shape == (0,)


def test_detection_result_copies_and_freezes_arrays() -> None:
    source = np.asarray([[1.0, 2.0]])
    result = DetectionResult(source, [0.9], provider="fake")
    source[0, 0] = 100.0
    assert result.points[0, 0] == 1.0
    with pytest.raises(ValueError):
        result.points[0, 0] = 3.0


@pytest.mark.parametrize(
    ("points", "confidences"),
    [([[1.0, 2.0, 3.0]], [0.5]), ([[1.0, 2.0]], []), ([[1.0, 2.0]], [1.1])],
)
def test_candidate_validation(points: object, confidences: object) -> None:
    with pytest.raises(ValueError):
        CandidateSet(points, confidences)


def test_detection_result_validates_sha256() -> None:
    with pytest.raises(ValueError):
        DetectionResult([], [], provider="fake", model_sha256="not-a-digest")
