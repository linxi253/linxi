"""Validate a completed real-data workflow without changing source annotations."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import numpy as np
from atom_center.annotations import AnnotationProject
from atom_center.backends import TorchBackend, OnnxBackend, onnx_pipeline, decode_output
from atom_center.configuration import pipeline_config
from atom_center.data_workflow import verify_dataset
from atom_center.evaluation import evaluate_dataset
from atom_center.image_io import load_image
from atom_center.metrics import match_points
from atom_center.model_manifest import sha256_file
from atom_center.pipeline import DetectionPipeline
from atom_center.storage import read_json, write_json
from atom_center.training import checkpoint_for_run, code_provenance

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=root/"runs/real-workflow-20260909")
    parser.add_argument("--data", type=Path, default=root/"data/processed/real_workflow_20260909")
    parser.add_argument("--output", type=Path, default=root/"reports/real_workflow_validation_20260909.json")
    args = parser.parse_args()
    run = args.run.resolve()
    dataset, data_root = verify_dataset(args.data)
    training, checkpoint, digest = checkpoint_for_run(run/"yolov8s_640")
    assert code_provenance()["source_sha256"] == training["code"]["source_sha256"]
    pt = TorchBackend(checkpoint, expected_sha256=digest, contract=training["contract"],
                      inference=training["config"]["inference"])
    ort = OnnxBackend(run/"onnx/model_manifest.json")
    pt_pipeline = DetectionPipeline(pt, pipeline_config(training["config"]))
    ort_pipeline = onnx_pipeline(run/"onnx/model_manifest.json")
    audit = read_json(data_root/"audit_report.json")
    source_by_key = {s["record_key"]: s for s in audit["records"]}
    identities = []
    for source in dataset["sources"]:
        original = source_by_key[source["record_key"]]
        assert sha256_file(original["label_file"]) == original["label_sha256"]
        assert sha256_file(original["source_image"]) == original["source_identity"]["file_sha256"]
        identities.append(source["record_key"])
    cases = []
    # Check every reviewed native-resolution region against both backends.
    for source in dataset["sources"]:
        raw = load_image(data_root/source["raw"], normalize=False).image
        truth = read_json(data_root/source["truth"])
        for region_index, roi in enumerate(truth["regions_xyxy"]):
            x0, y0, x1, y1 = map(int, roi)
            patch = raw[y0:y1, x0:x1]
            # Tensor contract checked on a native raw tile at the ROI origin.
            patch = patch[:dataset["tile_size"], :dataset["tile_size"]]
            tensor, transform = pt.prepare(patch)
            a, b = pt.raw(tensor), ort.raw(tensor)
            box_delta = float(np.max(np.abs(a[:, :4]-b[:, :4])))
            score_delta = float(np.max(np.abs(a[:, 4]-b[:, 4])))
            p, q = pt_pipeline.detect(raw, roi=roi), ort_pipeline.detect(raw, roi=roi)
            matching = match_points(p.points, q.points, max_distance_px=.01)
            # Additional nonempty point probe with NMS disabled and confidence=0.
            # This checks every finite in-image decoded center without pretending
            # these low-confidence boxes are real detections or accepted outputs.
            pd, _ = decode_output(a, transform, conf=0., iou=1., max_det=20000)
            qd, _ = decode_output(b, transform, conf=0., iou=1., max_det=20000)
            probe = match_points(pd.points, qd.points, max_distance_px=.01)
            maximum = float(probe.distances_px.max()) if len(probe.distances_px) else None
            cases.append({"record_key": source["record_key"], "source": Path(source_by_key[source["record_key"]]["source_image"]).name,
                "region_index": region_index, "split": source["split"], "roi": roi,
                "raw_box_max_delta_px": box_delta, "raw_score_max_delta": score_delta,
                "raw_max_confidence": float(a[:, 4].max()),
                "default_torch_points": len(p.points), "default_onnx_points": len(q.points),
                "probe_only_no_nms": True, "probe_points": len(pd.points),
                "probe_point_max_delta_px": maximum,
                "passed": box_delta < .01 and score_delta < 1e-5
                    and matching.false_positives == matching.false_negatives == 0
                    and probe.false_positives == probe.false_negatives == 0})
    metrics = evaluate_dataset(ort_pipeline, data_root, split="val", max_distance_px=2.)
    write_json(run/"validation_points.json", metrics)
    with (run/"yolov8s_640/training/results.csv").open() as handle:
        epochs = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]
    assert len(epochs) == 3 and all(np.isfinite(v) for e in epochs for v in e.values())
    state = read_json(run/"yolov8s_640/state.json")
    assert state["completed_epochs"] == 3 and state["status"] == "completed"
    assert state["attempts"][1]["from_completed_epochs"] == 1
    projects = []
    project_paths = sorted({source_by_key[s["record_key"]]["source_project"] for s in dataset["sources"]})
    for path in project_paths:
        project = AnnotationProject.load(path)
        docs = [project.load_document(record) for record in project.records]
        projects.append({"path": str(path), "review_states": dict(Counter(d.review_status for d in docs)),
            "drafts_excluded": [{"image": d.image_path, "point_count": len(d.points_xy)}
                               for d in docs if d.review_status != "reviewed"]})
    report = {"purpose": "real_data_training_workflow_only", "date": "2026-09-09",
        "formal_audit_passed": dataset["formal_audit_passed"], "scientific_acceptance": False,
        "dataset_content_sha256": dataset["content_sha256"],
        "data": {"projects": projects, "source_counts": dict(Counter(s["split"] for s in dataset["sources"])),
            "crop_counts": dict(Counter(c["split"] for c in dataset["crops"])),
            "atom_counts": {split: sum(c["point_count"] for c in dataset["crops"] if c["split"]==split) for split in ("train","val")},
            "reviewed_regions": len(cases), "max_train_atoms_per_crop": dataset["max_train_atoms_per_crop"],
            "background_crops": sum(c["point_count"]==0 for c in dataset["crops"]),
            "original_images_and_labels_unchanged": len(identities)},
        "training": state, "epochs": epochs,
        "onnx_model_sha256": ort.model_sha256, "real_image_parity": cases,
        "all_real_cases_passed": all(c["passed"] for c in cases),
        "validation_points": {k: metrics[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1", "rmse_px")},
        "limitations": ["Missing acquisition identifiers; task isolation does not prove independent acquisitions.",
                       "Legacy identity observed now; annotation-time image hashes unavailable.",
                       "Three images remain draft; no reviewed blank backgrounds.",
                       "pixel_size=1024 angstrom_per_pixel requires correction/confirmation; no physical-unit metrics.",
                       "Three random-initialized epochs validate execution, not model accuracy."] }
    write_json(args.output, report)
    print({"regions": len(cases), "passed": report["all_real_cases_passed"],
           "val_counts": report["validation_points"]})
    if not report["all_real_cases_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
