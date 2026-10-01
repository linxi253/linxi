"""Stable contracts shared by training, evaluation, and PPA integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
ROI = tuple[float, float, float, float]


def _validated_points(value: ArrayLike) -> FloatArray:
    points = np.asarray(value, dtype=np.float64)
    if points.size == 0:
        points = np.empty((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite values")
    points = np.array(points, dtype=np.float64, copy=True)
    points.setflags(write=False)
    return points


def _validated_confidences(value: ArrayLike, count: int) -> FloatArray:
    confidences = np.asarray(value, dtype=np.float64)
    if confidences.size == 0 and count == 0:
        confidences = np.empty((0,), dtype=np.float64)
    if confidences.ndim != 1 or len(confidences) != count:
        raise ValueError(
            "confidences must have shape (N,) matching points; "
            f"got {confidences.shape} for {count} points"
        )
    if not np.isfinite(confidences).all():
        raise ValueError("confidences must contain only finite values")
    if np.any((confidences < 0.0) | (confidences > 1.0)):
        raise ValueError("confidences must be within [0, 1]")
    confidences = np.array(confidences, dtype=np.float64, copy=True)
    confidences.setflags(write=False)
    return confidences


def _validated_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError("model_sha256 must be a 64-character hexadecimal digest")
    return normalized


@dataclass(frozen=True)
class CandidateSet:
    """Candidate points returned by a model backend for one image patch.

    Coordinates are always ``(x, y)`` in patch-local pixels with the image
    origin at the top-left corner.
    """

    points: ArrayLike
    confidences: ArrayLike

    def __post_init__(self) -> None:
        points = _validated_points(self.points)
        confidences = _validated_confidences(self.confidences, len(points))
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "confidences", confidences)

    @classmethod
    def empty(cls) -> "CandidateSet":
        return cls(np.empty((0, 2)), np.empty((0,)))


@dataclass(frozen=True)
class DetectionResult:
    """Validated, image-global atom-center detections."""

    points: ArrayLike
    confidences: ArrayLike
    provider: str
    model_sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = _validated_points(self.points)
        confidences = _validated_confidences(self.confidences, len(points))
        provider = self.provider.strip()
        if not provider:
            raise ValueError("provider must be a non-empty string")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "confidences", confidences)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model_sha256", _validated_sha256(self.model_sha256))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@runtime_checkable
class CandidateBackend(Protocol):
    """A model-specific backend that predicts patch-local candidates."""

    name: str
    model_sha256: str | None

    def predict(self, image: NDArray[np.floating]) -> CandidateSet:
        ...


@runtime_checkable
class AtomDetectorProvider(Protocol):
    """PPA-facing provider contract independent of the ML framework."""

    def detect(self, image: ArrayLike, *, roi: ROI | None = None) -> DetectionResult:
        ...
