"""Point-localization metrics designed for dense atomic lattices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _points(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array with shape (N, 2)")
    return array


@dataclass(frozen=True)
class PointMatchResult:
    matched_indices: NDArray[np.int64]
    distances_px: NDArray[np.float64]
    unmatched_prediction_indices: NDArray[np.int64]
    unmatched_ground_truth_indices: NDArray[np.int64]

    @property
    def true_positives(self) -> int:
        return len(self.matched_indices)

    @property
    def false_positives(self) -> int:
        return len(self.unmatched_prediction_indices)

    @property
    def false_negatives(self) -> int:
        return len(self.unmatched_ground_truth_indices)


@dataclass(frozen=True)
class PointMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    localization_mae_px: float
    localization_rmse_px: float
    match_distance_px: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "localization_mae_px": self.localization_mae_px,
            "localization_rmse_px": self.localization_rmse_px,
            "match_distance_px": self.match_distance_px,
        }


def nearest_neighbor_spacing(points: ArrayLike) -> float:
    """Return median nearest-neighbor spacing for a point lattice."""

    coordinates = _points(points, "points")
    if len(coordinates) < 2:
        raise ValueError("at least two points are required to estimate lattice spacing")
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(coordinates).query(coordinates, k=2)
    spacing = float(np.median(distances[:, 1]))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("nearest-neighbor spacing is not positive")
    return spacing


def resolve_match_distance(
    ground_truth: ArrayLike,
    *,
    max_distance_px: float | None = None,
    max_distance_fraction: float | None = None,
    lattice_spacing_px: float | None = None,
) -> float:
    """Resolve a matching threshold in pixels, preferably from lattice spacing."""

    if (max_distance_px is None) == (max_distance_fraction is None):
        raise ValueError(
            "provide exactly one of max_distance_px and max_distance_fraction"
        )
    if max_distance_px is not None:
        if not np.isfinite(max_distance_px) or max_distance_px < 0.0:
            raise ValueError("max_distance_px cannot be negative")
        return float(max_distance_px)
    assert max_distance_fraction is not None
    if not np.isfinite(max_distance_fraction) or max_distance_fraction <= 0.0:
        raise ValueError("max_distance_fraction must be positive")
    spacing = (
        float(lattice_spacing_px)
        if lattice_spacing_px is not None
        else nearest_neighbor_spacing(ground_truth)
    )
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("lattice_spacing_px must be positive")
    return float(max_distance_fraction * spacing)


def match_points(
    predictions: ArrayLike, ground_truth: ArrayLike, *, max_distance_px: float,
) -> PointMatchResult:
    """Maximum-cardinality, then minimum-distance matching in spatial components.

    A normalized valid edge costs <= 1; an unmatched prediction costs K+1
    (K=min(component sizes)). Thus dropping one match can never reduce the cost,
    even if every other matching edge changes. No full-image dense matrix.
    """
    predicted = _points(predictions, "predictions")
    truth = _points(ground_truth, "ground_truth")
    if not np.isfinite(max_distance_px) or max_distance_px < 0:
        raise ValueError("max_distance_px must be finite and non-negative")
    pairs = []
    if len(predicted) and len(truth):
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial import cKDTree
        adjacent = cKDTree(predicted).query_ball_tree(cKDTree(truth), max_distance_px)
        reverse = [[] for _ in truth]
        for p, neighbours in enumerate(adjacent):
            for t in neighbours:
                reverse[t].append(p)
        visited = set()
        for seed, neighbours in enumerate(adjacent):
            if seed in visited or not neighbours:
                continue
            ps, ts, pending = set(), set(), [seed]
            while pending:
                p = pending.pop()
                if p in ps:
                    continue
                ps.add(p)
                for t in adjacent[p]:
                    if t not in ts:
                        ts.add(t)
                        pending.extend(reverse[t])
            visited.update(ps)
            pi, ti = sorted(ps), sorted(ts)
            if len(pi) == len(ti) == 1:
                pairs.append((pi[0], ti[0]))
                continue
            penalty = float(min(len(pi), len(ti))+1)
            costs = np.full((len(pi), len(ti)+len(pi)), penalty)
            costs[:, :len(ti)] = np.inf
            columns = {t: j for j, t in enumerate(ti)}
            for i, p in enumerate(pi):
                for t in adjacent[p]:
                    costs[i, columns[t]] = np.linalg.norm(predicted[p]-truth[t])/max(max_distance_px, 1.)
            rows, cols = linear_sum_assignment(costs)
            pairs.extend((pi[i], ti[j]) for i, j in zip(rows, cols) if j < len(ti))
    matched = np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)
    distances = (np.linalg.norm(predicted[matched[:, 0]]-truth[matched[:, 1]], axis=1)
                 if len(matched) else np.empty(0, dtype=np.float64))
    return PointMatchResult(
        matched, distances,
        np.setdiff1d(np.arange(len(predicted), dtype=np.int64), matched[:, 0]),
        np.setdiff1d(np.arange(len(truth), dtype=np.int64), matched[:, 1]))


def evaluate_points(
    predictions: ArrayLike,
    ground_truth: ArrayLike,
    *,
    max_distance_px: float | None = None,
    max_distance_fraction: float | None = None,
    lattice_spacing_px: float | None = None,
) -> PointMetrics:
    threshold = resolve_match_distance(
        ground_truth,
        max_distance_px=max_distance_px,
        max_distance_fraction=max_distance_fraction,
        lattice_spacing_px=lattice_spacing_px,
    )
    matches = match_points(
        predictions, ground_truth, max_distance_px=threshold
    )
    tp = matches.true_positives
    fp = matches.false_positives
    fn = matches.false_negatives
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    if len(matches.distances_px):
        mae = float(np.mean(matches.distances_px))
        rmse = float(np.sqrt(np.mean(matches.distances_px**2)))
    else:
        mae = float("nan")
        rmse = float("nan")
    return PointMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        localization_mae_px=mae,
        localization_rmse_px=rmse,
        match_distance_px=threshold,
    )


def strain_noise_bound(localization_sigma_px: float, lattice_spacing_px: float) -> float:
    """Approximate one-sigma pair-spacing strain noise from point uncertainty."""

    if localization_sigma_px < 0.0:
        raise ValueError("localization_sigma_px cannot be negative")
    if lattice_spacing_px <= 0.0:
        raise ValueError("lattice_spacing_px must be positive")
    return float(np.sqrt(2.0) * localization_sigma_px / lattice_spacing_px)
