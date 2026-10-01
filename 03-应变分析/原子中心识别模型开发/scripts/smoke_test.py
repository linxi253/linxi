"""Model-free end-to-end smoke test: TIFF -> candidates -> refine -> metrics."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import tifffile

from atom_center.image_io import load_image
from atom_center.interfaces import CandidateSet
from atom_center.metrics import evaluate_points
from atom_center.pipeline import DetectionPipeline, PipelineConfig


class SeedBackend:
    name = "synthetic-seed-backend"
    model_sha256 = None

    def __init__(self, seeds: np.ndarray):
        self.seeds = np.asarray(seeds, dtype=np.float64)

    def predict(self, image: np.ndarray) -> CandidateSet:
        return CandidateSet(self.seeds, np.full(len(self.seeds), 0.95))


def synthetic_image(
    shape: tuple[int, int], centers: np.ndarray, sigma: float = 1.25
) -> np.ndarray:
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]]
    image = np.full(shape, 0.02, dtype=np.float64)
    for x, y in centers:
        image += np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2.0 * sigma**2))
    rng = np.random.default_rng(20260831)
    image += rng.normal(0.0, 0.002, size=shape)
    image -= image.min()
    image /= image.max()
    return image


def main() -> int:
    truth = np.asarray([[25.35, 30.65], [68.20, 63.40]], dtype=np.float64)
    image = synthetic_image((96, 96), truth)
    uint16_image = np.round(image * np.iinfo(np.uint16).max).astype(np.uint16)
    with tempfile.TemporaryDirectory() as directory:
        image_path = Path(directory) / "synthetic_atoms.tif"
        tifffile.imwrite(image_path, uint16_image, photometric="minisblack")
        loaded = load_image(image_path, normalize=False)
        seeds = np.round(truth)
        pipeline = DetectionPipeline(
            SeedBackend(seeds),
            PipelineConfig(
                tile_size=128,
                refine=True,
                refinement_window=9,
                refinement_method="gaussian",
                refinement_polarity="bright",
            ),
        )
        result = pipeline.detect(loaded.image)
        metrics = evaluate_points(result.points, truth, max_distance_px=0.75)

    report = {
        "image_shape": list(loaded.image.shape),
        "provider": result.provider,
        "points_xy": result.points.tolist(),
        "metrics": metrics.to_dict(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if metrics.f1 != 1.0 or metrics.localization_rmse_px >= 0.25:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
