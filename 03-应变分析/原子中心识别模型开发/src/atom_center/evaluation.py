"""Point metrics on immutable full-intensity source planes and reviewed regions."""
from __future__ import annotations
import time
import numpy as np
from .data_workflow import verify_dataset
from .image_io import load_image
from .metrics import match_points
from .storage import read_json, write_json


def evaluate_dataset(provider, data, *, split="val", max_distance_px=2.):
    manifest, root = verify_dataset(data)
    rows, all_deltas = [], []
    for source in manifest["sources"]:
        if source["split"] != split:
            continue
        truth = read_json(root/source["truth"])
        raw = load_image(root/source["raw"], normalize=False).image
        expected = np.asarray(truth["points_xy"], dtype=float).reshape(-1, 2)
        for region_index, roi in enumerate(truth["regions_xyxy"]):
            x0, y0, x1, y1 = roi
            gt = expected[(expected[:, 0] >= x0) & (expected[:, 0] < x1)
                          & (expected[:, 1] >= y0) & (expected[:, 1] < y1)]
            start = time.perf_counter()
            result = provider.detect(raw, roi=roi)
            duration = time.perf_counter()-start
            matches = match_points(result.points, gt, max_distance_px=max_distance_px)
            pairs = matches.matched_indices
            delta = result.points[pairs[:, 0]]-gt[pairs[:, 1]]
            all_deltas.extend(delta.tolist())
            rows.append({"record_key": source["record_key"], "isolation_group": source["isolation_group"],
                "roi_index": region_index, "tp": matches.true_positives,
                "fp": matches.false_positives, "fn": matches.false_negatives,
                "time_s": duration, "rmse_px": float(np.sqrt(np.mean(matches.distances_px**2))) if len(pairs) else None,
                "metadata": dict(result.metadata)})
    if not rows:
        raise ValueError(f"no reviewed regions in split {split}")
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    deltas = np.asarray(all_deltas).reshape(-1, 2)
    distances = np.linalg.norm(deltas, axis=1)
    return {"schema_version": 1, "purpose": "development_evaluation", "split": split,
        "dataset_content_sha256": manifest["content_sha256"], "match_distance_px": max_distance_px,
        "tp": tp, "fp": fp, "fn": fn, "precision": tp/(tp+fp) if tp+fp else 1.,
        "recall": tp/(tp+fn) if tp+fn else 1., "f1": 2*tp/(2*tp+fp+fn) if tp+fp+fn else 1.,
        "rmse_px": float(np.sqrt(np.mean(distances**2))) if tp else None,
        "p95_px": float(np.percentile(distances, 95)) if tp else None,
        "bias_xy_px": deltas.mean(axis=0).tolist() if tp else None,
        "total_time_s": sum(r["time_s"] for r in rows), "regions": rows,
        "note": "Per-region observations; no claim of scientific acceptance or independent-atom confidence intervals."}
