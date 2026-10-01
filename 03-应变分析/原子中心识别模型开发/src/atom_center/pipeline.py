"""Model-agnostic inference; input is the unmodified full grayscale image."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
from .geometry import clip_roi, generate_tiles, merge_close_points, offset_candidates, point_nms_indices
from .interfaces import CandidateBackend, CandidateSet, DetectionResult, ROI
from .refinement import Polarity, RefinementMethod, refine_with_diagnostics


@dataclass(frozen=True)
class PipelineConfig:
    tile_size: int | tuple[int, int] = 640
    tile_overlap: float = .1
    merge_distance_px: float = 3.
    refine: bool = True
    refinement_window: int = 7
    refinement_method: RefinementMethod = "com"
    refinement_polarity: Polarity = "auto"
    min_atom_distance_px: float | None = None
    refinement_max_shift_px: float | None = None
    merge_after_refinement: bool = False
    adaptive_merge_sigma: float | None = None

    def __post_init__(self):
        generate_tiles((32, 32), tile_size=self.tile_size, overlap=self.tile_overlap)
        if type(self.refinement_window) is not int or self.refinement_window < 3 or self.refinement_window % 2 != 1:
            raise ValueError("refinement_window must be an odd integer >= 3")
        if self.refinement_method not in {"com", "com_continuous", "gaussian", "adaptive_blob"} or self.refinement_polarity not in {"auto", "bright", "dark"}:
            raise ValueError("unknown refinement method/polarity")
        if type(self.merge_after_refinement) is not bool:
            raise ValueError("merge_after_refinement must be boolean")
        if self.refinement_method == "adaptive_blob" and self.refinement_polarity != "bright":
            raise ValueError("adaptive_blob requires explicit bright polarity")
        if self.adaptive_merge_sigma is not None and (self.refinement_method != 'adaptive_blob' or not np.isfinite(self.adaptive_merge_sigma) or self.adaptive_merge_sigma <= 0):
            raise ValueError("adaptive_merge_sigma requires adaptive_blob and a positive finite multiplier")
        if self.refinement_max_shift_px is not None and (not np.isfinite(self.refinement_max_shift_px) or self.refinement_max_shift_px <= 0):
            raise ValueError("refinement_max_shift_px must be positive and finite")
        if not np.isfinite(self.merge_distance_px) or self.merge_distance_px < 0:
            raise ValueError("merge_distance_px must be finite and non-negative")
        if self.min_atom_distance_px is not None:
            d = self.min_atom_distance_px
            if not np.isfinite(d) or d <= 0:
                raise ValueError("min_atom_distance_px must be finite and positive")
            if self.merge_distance_px >= d/2:
                raise ValueError("merge_distance_px must be below half the minimum atom distance")


class DetectionPipeline:
    """Backends normalize raw tiles; refinement always uses the full raw image."""

    def __init__(self, backend: CandidateBackend, config: PipelineConfig | None = None):
        if not getattr(backend, "name", "").strip():
            raise ValueError("backend.name must be non-empty")
        self.backend = backend
        self.config = config or PipelineConfig()

    def _predict_region(self, image):
        tiles = generate_tiles(image.shape, tile_size=self.config.tile_size,
                               overlap=self.config.tile_overlap)
        batches, diagnostics = [], []
        for tile in tiles:
            local = self.backend.predict(image[tile.slices])
            diagnostics.append(dict(getattr(self.backend, "last_diagnostics", {})))
            if not len(local.points):
                continue
            inside = ((local.points[:, 0] >= 0) & (local.points[:, 0] < tile.width)
                      & (local.points[:, 1] >= 0) & (local.points[:, 1] < tile.height))
            shifted = offset_candidates(
                CandidateSet(local.points[inside], local.confidences[inside]), tile.x0, tile.y0)
            if len(shifted.points):
                batches.append(shifted)
        if not batches:
            return CandidateSet.empty(), len(tiles), diagnostics
        return merge_close_points(
            np.concatenate([b.points for b in batches]),
            np.concatenate([b.confidences for b in batches]),
            min_distance=self.config.merge_distance_px), len(tiles), diagnostics

    def detect(self, image: ArrayLike, *, roi: ROI | None = None) -> DetectionResult:
        raw = np.asarray(image, dtype=np.float64)
        if raw.ndim != 2 or not np.isfinite(raw).all():
            raise ValueError("DetectionPipeline expects one finite, unmodified 2-D image")
        bounds = clip_roi(roi, raw.shape)
        candidates, count, tile_diagnostics = self._predict_region(raw[bounds.slices])
        points = candidates.points + [bounds.x0, bounds.y0]
        quality = []
        if self.config.refine and len(points):
            window = self.config.refinement_window
            limit = self.config.refinement_max_shift_px
            if self.config.min_atom_distance_px is not None:
                distance = self.config.min_atom_distance_px
                window = min(window, max(3, 2*int(distance/2)+1))
                limit = min(limit if limit is not None else window//2*.8, distance*.4)
            refined = refine_with_diagnostics(
                raw, points, window_size=window, method=self.config.refinement_method,
                polarity=self.config.refinement_polarity, max_shift_px=limit)
            points, quality = refined.points, list(refined.diagnostics)
        inside = ((points[:, 0] >= bounds.x0) & (points[:, 0] < bounds.x1)
                  & (points[:, 1] >= bounds.y0) & (points[:, 1] < bounds.y1))
        if quality and len(points) > 1 and self.config.merge_distance_px > 0:
            from scipy.spatial import cKDTree
            for a, b in cKDTree(points).query_pairs(self.config.merge_distance_px):
                quality[a]["close_after_refinement"] = True
                quality[b]["close_after_refinement"] = True
        final_points, final_scores = points[inside], candidates.confidences[inside]
        final_quality = [q for q, keep in zip(quality, inside) if keep]
        removed = 0
        if self.config.refine and self.config.merge_after_refinement:
            radii = None
            if self.config.adaptive_merge_sigma is not None:
                radii = [max(self.config.merge_distance_px, q.get('estimated_sigma_px',0.)*self.config.adaptive_merge_sigma) for q in final_quality]
            keep = point_nms_indices(final_points, final_scores, min_distance=self.config.merge_distance_px, radii=radii)
            removed = len(final_points)-len(keep)
            final_points, final_scores = final_points[keep], final_scores[keep]
            if final_quality:
                final_quality = [final_quality[i] for i in keep]
        return DetectionResult(
            points=final_points, confidences=final_scores,
            provider=self.backend.name, model_sha256=getattr(self.backend, "model_sha256", None),
            metadata={
                "coordinate_order": "xy", "pixel_centers": "integer_coordinates",
                "roi_xyxy": [bounds.x0, bounds.y0, bounds.x1, bounds.y1],
                "tile_count": count, "tile_diagnostics": tile_diagnostics,
                "refined": self.config.refine,
                "refinement_method": self.config.refinement_method if self.config.refine else None,
                "refinement_input": "unmodified_full_image",
                "point_quality": final_quality,
                "duplicates_removed_after_refinement": removed,
                "outside_roi_after_refinement": int((~inside).sum()),
            })
