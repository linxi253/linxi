"""Static FP32 detector export and actual Torch/ONNX equivalence checks."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import numpy as np
from .backends import TorchBackend, OnnxBackend, onnx_pipeline
from .configuration import model_contract, pipeline_config
from .metrics import match_points
from .model_manifest import ModelManifest
from .pipeline import DetectionPipeline
from .storage import new_artifact_directory, write_json, read_json
from .training import checkpoint_for_run, software_versions


def validation_images(size=128):
    rng = np.random.default_rng(731)
    cases = []
    for name, shape, amplitude, spacing in (
        ("blank", (size, size), 0., 16),
        ("dense", (size, size), 4000., 8),
        ("weak", (size, size), 4., 16),
        ("non_square", (size*3//4, size+27), 3000., 16),
        ("overlap_tiles", (size+33, size*2+17), 3000., 16),
    ):
        yy, xx = np.mgrid[:shape[0], :shape[1]]
        raw = np.full(shape, 1000.) if not amplitude else 1000.+rng.normal(0, 1., shape)
        for y in np.arange(10.2, shape[0]-6, spacing):
            for x in np.arange(10.35, shape[1]-6, spacing):
                raw += amplitude*np.exp(-((xx-x)**2+(yy-y)**2)/(2*1.1**2))
        cases.append((name, raw, None))
    cases.append(("roi", cases[-1][1], (5, 7, size+21, size+19)))
    return cases


def compare_backends(run, manifest_path, *, checkpoint="best.pt"):
    training, path, digest = checkpoint_for_run(run, checkpoint)
    config = training["config"]
    torch = TorchBackend(path, expected_sha256=digest, contract=training["contract"],
                         inference=config["inference"], device="cpu")
    onnx = OnnxBackend(manifest_path)
    if (onnx.contract != training["contract"]
            or dict(onnx.manifest.inference) != config["inference"]):
        raise ValueError("bundle configuration differs from source run")
    pt_pipeline = DetectionPipeline(torch, pipeline_config(config))
    ort_pipeline = onnx_pipeline(manifest_path)
    rows = []
    for name, raw, roi in validation_images(config["training"]["imgsz"]):
        tensor, _ = torch.prepare(raw)
        a, b = torch.raw(tensor), onnx.raw(tensor)
        box_error = float(np.max(np.abs(a[:, :4]-b[:, :4])))
        score_error = float(np.max(np.abs(a[:, 4]-b[:, 4])))
        p, q = pt_pipeline.detect(raw, roi=roi), ort_pipeline.detect(raw, roi=roi)
        matched = match_points(p.points, q.points, max_distance_px=.01)
        pairs = matched.matched_indices
        point_delta = float(matched.distances_px.max()) if len(pairs) else None
        confidence_delta = (float(np.max(np.abs(p.confidences[pairs[:, 0]]-q.confidences[pairs[:, 1]])))
                            if len(pairs) else None)
        rows.append({"case": name, "tensor_box_max_abs_px": box_error,
            "tensor_score_max_abs": score_error, "torch_points": len(p.points), "onnx_points": len(q.points),
            "matched_points": len(pairs), "unmatched_torch": matched.false_positives,
            "unmatched_onnx": matched.false_negatives, "point_max_delta_px": point_delta,
            "confidence_max_delta": confidence_delta,
            "threshold_near_predictions": int(np.sum(np.abs(a[:, 4]-config["inference"]["conf"]) < 1e-6)),
            "passed": box_error <= .01 and score_error <= 1e-5
                      and matched.false_positives == matched.false_negatives == 0
                      and (confidence_delta is None or confidence_delta <= 1e-5)})
    report = {"schema_version": 1, "purpose": "software_equivalence_only",
              "checkpoint_sha256": digest, "onnx_sha256": onnx.model_sha256,
              "providers": onnx.session.get_providers(), "cases": rows,
              "passed": all(r["passed"] for r in rows),
              "nonempty_cases": sum(r["torch_points"] > 0 for r in rows),
              "note": "Agreement validates computation, not atom localization accuracy."}
    return report


def export_run(run, output, *, checkpoint="best.pt"):
    import torch
    from copy import deepcopy
    import onnx
    training, checkpoint_path, digest = checkpoint_for_run(run, checkpoint)
    config = training["config"]
    backend = TorchBackend(checkpoint_path, expected_sha256=digest, contract=training["contract"],
                           inference=config["inference"])
    model = deepcopy(backend.model).eval().float()
    head = model.model[-1]
    head.export, head.format, head.dynamic = True, "onnx", False
    size = config["training"]["imgsz"]
    target = Path(output).resolve()
    with new_artifact_directory(target) as temp:
        model_path = temp/"model.onnx"
        torch.onnx.export(model, torch.zeros(1, 3, size, size), str(model_path),
            input_names=["images"], output_names=["predictions"], opset_version=17,
            dynamic_axes=None, do_constant_folding=True, dynamo=False)
        graph = onnx.load(str(model_path))
        # Native checker failures must not crash the calling process. Some old
        # Windows Python/CRT combinations crash even on an Identity graph.
        import subprocess
        import sys
        checked = subprocess.run([sys.executable, "-c",
            "import onnx,sys; onnx.checker.check_model(sys.argv[1],full_check=True); print(onnx.__version__)",
            str(model_path)], capture_output=True, text=True, timeout=60)
        if checked.returncode:
            raise RuntimeError("ONNX native checker failed in this Python runtime; on this Windows project use scripts/atom-center.ps1. "
                               + checked.stderr[-1000:])
        if any(node.op_type == "NonMaxSuppression" for node in graph.graph.node):
            raise ValueError("unexpected embedded NMS")
        manifest = ModelManifest.create(model_path, model_id=Path(run).name, modality=config["modality"],
            version="development-0.3.0", provider="onnxruntime-cpu-yolov8",
            input=training["contract"], inference=config["inference"], refinement=config["refinement"],
            metrics={"purpose": "development", "scientific_acceptance": False,
                     "training_purpose": training["purpose"], "source_checkpoint_sha256": digest,
                     "source_checkpoint_name": checkpoint,
                     "dataset_content_sha256": training["dataset_content_sha256"]},
            software=software_versions(), git_commit=training["code"]["git_commit"])
        manifest.write(temp/"model_manifest.json")
        report = compare_backends(run, temp/"model_manifest.json", checkpoint=checkpoint)
        write_json(temp/"onnx_validation.json", report)
        if not report["passed"]:
            # Preserve a diagnostic outside the unpublished bundle.
            write_json(Path(run)/"onnx_validation.failed.json", report)
            raise ValueError("ONNX parity failed; see the run's onnx_validation.failed.json")
        manifest = replace(manifest, metrics={**manifest.metrics, "software_equivalence_passed": True,
                                              "nonempty_validation_cases": report["nonempty_cases"]})
        manifest.write(temp/"model_manifest.json")
    return target/"model_manifest.json"
