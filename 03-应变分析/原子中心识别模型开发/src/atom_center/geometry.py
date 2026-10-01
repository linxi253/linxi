"""Coordinate, ROI, tiling, and point-merging helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .interfaces import CandidateSet, ROI


@dataclass(frozen=True)
class PixelROI:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def slices(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y1), slice(self.x0, self.x1)


@dataclass(frozen=True)
class Tile(PixelROI):
    pass


def clip_roi(roi: ROI | None, image_shape: tuple[int, int]) -> PixelROI:
    """Convert a floating ROI to clipped, half-open integer pixel bounds."""

    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image shape: {image_shape}")
    if roi is None:
        return PixelROI(0, 0, width, height)
    values = np.asarray(roi, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("roi must contain four finite values (x0, y0, x1, y1)")
    x0 = max(0, min(width, math.floor(float(values[0]))))
    y0 = max(0, min(height, math.floor(float(values[1]))))
    x1 = max(0, min(width, math.ceil(float(values[2]))))
    y1 = max(0, min(height, math.ceil(float(values[3]))))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"roi is empty after clipping: {(x0, y0, x1, y1)}")
    return PixelROI(x0, y0, x1, y1)


def _axis_positions(length: int, tile: int, overlap: float) -> list[int]:
    if tile <= 0:
        raise ValueError("tile dimensions must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")
    if length <= tile:
        return [0]
    step = max(1, int(round(tile * (1.0 - overlap))))
    last = length - tile
    positions = list(range(0, last + 1, step))
    if positions[-1] != last:
        positions.append(last)
    return positions


def generate_tiles(
    image_shape: tuple[int, int],
    *,
    tile_size: int | tuple[int, int],
    overlap: float,
) -> list[Tile]:
    """Generate deterministic tiles that cover every pixel exactly or by overlap."""

    height, width = image_shape
    if isinstance(tile_size, int):
        tile_height = tile_width = tile_size
    else:
        tile_height, tile_width = tile_size
    ys = _axis_positions(height, tile_height, overlap)
    xs = _axis_positions(width, tile_width, overlap)
    return [
        Tile(
            x0=x0,
            y0=y0,
            x1=min(width, x0 + tile_width),
            y1=min(height, y0 + tile_height),
        )
        for y0 in ys
        for x0 in xs
    ]


def offset_candidates(candidates: CandidateSet, dx: float, dy: float) -> CandidateSet:
    if len(candidates.points) == 0:
        return candidates
    offset = np.asarray([dx, dy], dtype=np.float64)
    return CandidateSet(candidates.points + offset, candidates.confidences)


def merge_close_points(
    points: ArrayLike,
    confidences: ArrayLike,
    *,
    min_distance: float,
) -> CandidateSet:
    """Confidence-ordered point NMS for overlapping inference tiles."""

    candidates = CandidateSet(points, confidences)
    keep = point_nms_indices(candidates.points, candidates.confidences, min_distance=min_distance)
    return CandidateSet(candidates.points[keep], candidates.confidences[keep])


def point_nms_indices(points, confidences, *, min_distance, radii=None):
    """Indices retained by deterministic confidence-ordered point NMS."""
    candidates = CandidateSet(points, confidences)
    if radii is not None:
        radii = np.asarray(radii,dtype=float)
        if radii.shape != (len(candidates.points),) or not np.isfinite(radii).all() or np.any(radii<0):
            raise ValueError("radii must be finite nonnegative values matching points")
    if len(candidates.points) <= 1 or min_distance <= 0:
        return np.arange(len(candidates.points), dtype=int)
    order = np.argsort(-candidates.confidences, kind="stable")
    sorted_points = candidates.points[order]
    sorted_confidences = candidates.confidences[order]
    sorted_radii = radii[order] if radii is not None else None
    suppressed = np.zeros(len(sorted_points), dtype=bool)
    keep: list[int] = []
    squared_threshold = float(min_distance) ** 2
    for index in range(len(sorted_points)):
        if suppressed[index]:
            continue
        keep.append(index)
        deltas = sorted_points[index + 1 :] - sorted_points[index]
        squared_distances = np.einsum("ij,ij->i", deltas, deltas)
        threshold = squared_threshold if sorted_radii is None else np.maximum(sorted_radii[index], sorted_radii[index+1:])**2
        suppressed[index + 1 :] |= squared_distances < threshold
    return order[keep]


def tiles_cover_image(tiles: Iterable[Tile], image_shape: tuple[int, int]) -> bool:
    """Return whether a tile set covers every image pixel (used by validation)."""

    height, width = image_shape
    coverage = np.zeros((height, width), dtype=bool)
    for tile in tiles:
        coverage[tile.y0 : tile.y1, tile.x0 : tile.x1] = True
    return bool(coverage.all())
