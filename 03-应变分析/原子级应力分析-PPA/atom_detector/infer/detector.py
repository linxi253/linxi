"""Legacy optional detector, sharing atom-center coordinates and raw-image refinement.

PPA GUI integration is a separate step. Existing model allowlist verification remains required.
"""
from pathlib import Path
import warnings
import numpy as np


class AtomDetector:
    def __init__(self, model_path, conf=.5, iou=.45, imgsz=640, device="0",
                 refine=True, refine_window=5, refine_method="com", sahi=True,
                 sahi_slice=640, sahi_overlap=.1, allowlist_path=None, max_det=3000,
                 merge_distance_px=3., min_atom_distance_px=None, polarity="auto"):
        # 参数范围校验必须在任何重型导入之前: 错误的 conf/iou 会让
        # 后续推理静默产出空结果或全量框, 而不是立即失败。
        if not (0.0 < float(conf) < 1.0):
            raise ValueError(f"conf 必须在 (0, 1) 内, 收到 {conf}")
        if not (0.0 < float(iou) < 1.0):
            raise ValueError(f"iou 必须在 (0, 1) 内, 收到 {iou}")
        if type(imgsz) is not int or imgsz < 32:
            raise ValueError(f"imgsz 必须是 >=32 的整数, 收到 {imgsz}")
        if type(sahi_slice) is not int or sahi_slice < 32:
            raise ValueError(f"sahi_slice 必须是 >=32 的整数, 收到 {sahi_slice}")
        if not (0.0 <= float(sahi_overlap) < 1.0):
            raise ValueError(f"sahi_overlap 必须在 [0, 1) 内, 收到 {sahi_overlap}")
        if float(merge_distance_px) <= 0:
            raise ValueError(f"merge_distance_px 必须为正, 收到 {merge_distance_px}")
        if type(max_det) is not int or max_det < 1:
            raise ValueError("max_det must be a positive integer")
        from atom_center.pipeline import PipelineConfig
        self.model_path, self.allowlist_path = Path(model_path), allowlist_path
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device
        self.max_det, self.sahi, self.sahi_slice = max_det, sahi, sahi_slice
        self.config = PipelineConfig(tile_size=sahi_slice, tile_overlap=sahi_overlap,
            merge_distance_px=merge_distance_px, min_atom_distance_px=min_atom_distance_px,
            refine=refine, refinement_window=refine_window, refinement_method=refine_method,
            refinement_polarity=polarity)
        self.last_diagnostics = {}
        self.last_result = None
        self.name = "legacy-ultralytics"
        self._load_model()

    def _load_model(self):
        from ..model_security import verify_model
        self.model_sha256 = verify_model(self.model_path, allowlist_path=self.allowlist_path)
        from ultralytics import YOLO
        self._model = YOLO(str(self.model_path), task="detect")

    def predict(self, raw):
        from atom_center.interfaces import CandidateSet
        return CandidateSet(*self._detect_single(raw))

    def _detect_single(self, raw):
        from atom_center.image_io import normalize_percentile
        gray = np.rint(normalize_percentile(raw)*255).astype(np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=2)
        results = self._model.predict(rgb, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
            device=self.device, verbose=False, max_det=self.max_det)
        points, confidences = [], []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy, scores = boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()
            # Results already contains patch coordinates: only undo the +0.5 encoding.
            points = ((xyxy[:, :2]+xyxy[:, 2:])/2-.5).tolist()
            confidences = scores.tolist()
        self.last_diagnostics = {"max_det": self.max_det, "returned": len(points),
                                 "hit_max_det": len(points) == self.max_det}
        if self.last_diagnostics["hit_max_det"]:
            warnings.warn(f"detections reached max_det={self.max_det}", RuntimeWarning)
        return points, confidences

    def detect(self, image, roi=None):
        from dataclasses import replace
        from atom_center.pipeline import DetectionPipeline
        raw = np.asarray(image)
        config = self.config if self.sahi else replace(self.config, tile_size=raw.shape)
        self.last_result = DetectionPipeline(self, config).detect(raw, roi=roi)
        return self.last_result.points.tolist(), self.last_result.confidences.tolist()

    def _load_image_file(self, path, *, series_index=0, frame_index=None):
        from atom_center.image_io import load_image
        return load_image(path, normalize=False, series_index=series_index, frame_index=frame_index).image

    def detect_from_file(self, image_path, roi=None, *, series_index=0, frame_index=None):
        return self.detect(self._load_image_file(image_path, series_index=series_index,
                                                frame_index=frame_index), roi)

    @classmethod
    def from_config(cls, config_path=None):
        import yaml
        root = Path(__file__).resolve().parent.parent
        cfg = yaml.safe_load(Path(config_path or root/"config.yaml").read_text(encoding="utf-8"))["inference"]
        allowlist = cfg.get("model_allowlist")
        return cls(root/cfg["model_path"], conf=cfg.get("conf", .5), iou=cfg.get("iou", .45),
            imgsz=cfg.get("imgsz", 640), refine=cfg.get("refine_centroid", True),
            refine_window=cfg.get("refine_window", 5), sahi_slice=cfg.get("sahi_slice_size", 640),
            sahi_overlap=cfg.get("sahi_overlap", .1), max_det=cfg.get("max_det", 3000),
            allowlist_path=root/allowlist if allowlist else None)


def quick_detect(image_path, model_path=None, conf=.5):
    path = model_path or Path(__file__).resolve().parent.parent/"models"/"best.pt"
    return AtomDetector(path, conf=conf).detect_from_file(image_path)
