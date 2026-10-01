"""Actual YOLOv8 Torch and CPU ONNX backends; importing this module needs no torch."""
from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
from .interfaces import CandidateSet
from .model_manifest import sha256_file, verify_model_bundle, ModelVerificationError
from .preprocessing import prepare_tile, input_tensor


def decode_output(output, transform, *, conf, iou, max_det):
    """Decode a single-class [1,5,N] raw tensor; shared deterministic NumPy NMS."""
    raw = np.asarray(output)
    if raw.ndim != 3 or raw.shape[:2] != (1, 5):
        raise ValueError(f"expected YOLOv8 [1,5,N] output, got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("model output contains NaN/Inf")
    if not 0 <= conf <= 1 or not 0 <= iou <= 1 or type(max_det) is not int or max_det < 1:
        raise ValueError("invalid decoding thresholds/max_det")
    rows = raw[0].T
    # Resolve FP32 score ties identically across Torch/ORT, including tile merging.
    # The confidence contract is six decimal places; tensor parity uses raw scores.
    rows = rows.copy()
    rows[:, 4] = np.round(rows[:, 4].astype(np.float64), 6)
    valid = (rows[:, 4] >= conf) & (rows[:, 4] <= 1) & np.all(rows[:, 2:4] > 0, axis=1)
    rows = rows[valid].astype(np.float64)
    points = transform.inverse_centers(rows[:, :2])
    h, w = transform.source_hw
    inside = ((points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h))
    rows, points = rows[inside], points[inside]
    boxes = np.concatenate((rows[:, :2]-rows[:, 2:4]/2, rows[:, :2]+rows[:, 2:4]/2), axis=1)
    areas = rows[:, 2]*rows[:, 3]
    order = np.argsort(-rows[:, 4], kind="stable")
    keep = []
    while len(order) and len(keep) < max_det:
        index, rest = order[0], order[1:]
        keep.append(index)
        left = np.maximum(boxes[index, :2], boxes[rest, :2])
        right = np.minimum(boxes[index, 2:], boxes[rest, 2:])
        intersection = np.maximum(0., right-left).prod(axis=1)
        overlap = intersection/np.maximum(areas[index]+areas[rest]-intersection, 1e-12)
        order = rest[overlap <= iou]
    diagnostics = {"candidates_above_threshold": int(valid.sum()),
                   "candidates_in_image": len(rows), "returned": len(keep),
                   "max_det": max_det, "hit_max_det": len(keep) == max_det,
                   "truncated": bool(len(order)),
                   "near_threshold_count": int(np.sum(np.abs(raw[0, 4]-conf) <= 1e-6))}
    if diagnostics["hit_max_det"]:
        warnings.warn(f"detections reached max_det={max_det}; inspect density/cap", RuntimeWarning)
    return CandidateSet(points[keep], rows[keep, 4]), diagnostics


class _Backend:
    def _configure(self, contract, inference):
        if contract.get("contract") != "raw_tile_percentile_uint8_letterbox_v1":
            raise ModelVerificationError("unsupported preprocessing contract")
        if contract.get("output") != "yolov8_single_class_xywh_score_no_nms":
            raise ModelVerificationError("unsupported output contract")
        fixed = {"layout": "NCHW", "channels": 3, "batch": 1, "dtype": "float32",
                 "padding": 114, "confidence_decimals": 6}
        if any(contract.get(k) != v for k, v in fixed.items()):
            raise ModelVerificationError("unsupported input/postprocessing contract")
        self.contract, self.inference = contract, inference
        self.last_diagnostics = {}

    def prepare(self, image):
        norm = self.contract["normalization"]
        gray, transform = prepare_tile(image, self.contract["input_size"], low=norm["low"], high=norm["high"])
        return input_tensor(gray), transform

    def predict(self, image):
        tensor, transform = self.prepare(image)
        candidates, self.last_diagnostics = decode_output(self.raw(tensor), transform,
            **{key: self.inference[key] for key in ("conf", "iou", "max_det")})
        return candidates


class TorchBackend(_Backend):
    name = "pytorch-yolov8"

    def __init__(self, checkpoint, *, expected_sha256, contract, inference, device="cpu"):
        path = Path(checkpoint).resolve()
        if sha256_file(path) != expected_sha256:
            raise ModelVerificationError("checkpoint SHA-256 mismatch")
        self.model_sha256 = expected_sha256
        self._configure(contract, inference)
        import torch
        from ultralytics import YOLO
        self.device = torch.device(device)
        self.model = YOLO(str(path), task="detect").model.to(self.device).float().eval()
        if self.model.model[-1].nc != 1:
            raise ModelVerificationError("v1 requires a one-class YOLOv8 model")

    def raw(self, tensor):
        import torch
        with torch.inference_mode():
            output = self.model(torch.from_numpy(tensor).to(self.device))
        return (output[0] if isinstance(output, tuple) else output).detach().cpu().numpy()


class OnnxBackend(_Backend):
    name = "onnxruntime-cpu-yolov8"

    def __init__(self, manifest_path):
        manifest, path = verify_model_bundle(manifest_path)
        if manifest.model_format != "onnx":
            raise ModelVerificationError("expected an ONNX bundle")
        self.manifest, self.model_sha256 = manifest, manifest.model_sha256
        self._configure(dict(manifest.input), dict(manifest.inference))
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        size = self.contract["input_size"]
        if (len(inputs) != 1 or inputs[0].shape != [1, 3, size, size]
                or inputs[0].type != "tensor(float)" or len(outputs) != 1):
            raise ModelVerificationError("ONNX input/output violates static FP32 contract")
        self.input_name = inputs[0].name

    def raw(self, tensor):
        return self.session.run(None, {self.input_name: tensor})[0]


def onnx_pipeline(manifest_path):
    from .pipeline import DetectionPipeline, PipelineConfig
    backend = OnnxBackend(manifest_path)
    inf, ref = backend.manifest.inference, backend.manifest.refinement
    config = PipelineConfig(tile_size=tuple(inf["tile_size"]), tile_overlap=inf["tile_overlap"],
        merge_distance_px=inf["merge_distance_px"], min_atom_distance_px=inf["min_atom_distance_px"],
        refine=ref["enabled"], refinement_method=ref["method"], refinement_window=ref["window_size"],
        refinement_polarity=ref["polarity"], refinement_max_shift_px=ref["max_shift_px"],
        merge_after_refinement=inf.get("merge_after_refinement",False),adaptive_merge_sigma=ref.get("merge_sigma"))
    return DetectionPipeline(backend, config)
