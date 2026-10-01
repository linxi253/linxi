"""Strict layered configuration; validation never loads a model."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import math
import yaml

DEFAULTS = {
    "schema_version": 1, "project": "atom-center-model-development",
    "modality": "haadf_stem", "class_names": ["atom_column"], "seed": 20260831,
    "coordinates": {"order": "xy", "origin": "top_left", "pixel_centers": "integer_coordinates"},
    "image": {"tiff_frame_policy": "explicit", "expected_polarity": "bright",
              "normalization": {"method": "percentile", "low": 1., "high": 99.,
                                "output_range": [0., 1.]}},
    "dataset": {"group_key": "acquisition_id", "split_ratios": {"train": .8, "val": .1, "test": .1}},
    "inference": {"tile_size": [640, 640], "tile_overlap": .1, "merge_distance_px": 3.,
                  "conf": .25, "iou": .45, "max_det": 3000, "min_atom_distance_px": None,
                  "merge_after_refinement": False},
    "refinement": {"enabled": True, "method": "com", "window_size": 7,
                   "polarity": "bright", "max_shift_px": None, "merge_sigma": None},
    "augmentation": {"brightness": .15, "contrast": .15, "probability": .5},
    "point_validation": {"enabled": False, "interval_epochs": 5, "match_distance_px": 2.},
    "training": {
        "model": "yolov8s.yaml", "epochs": 200, "imgsz": 640, "batch": 8,
        "patience": 30, "device": "0", "workers": 0, "optimizer": "AdamW",
        "lr0": .001, "lrf": .01, "momentum": .937, "weight_decay": .0005,
        "warmup_epochs": 3., "warmup_momentum": .8, "warmup_bias_lr": .1,
        "box": 12., "cls": .5, "dfl": 2., "cos_lr": True, "amp": False,
        "deterministic": True, "cache": False, "plots": False, "save_period": 1,
        "mosaic": 0., "mixup": 0., "copy_paste": 0., "shear": 0.,
        "perspective": 0., "hsv_h": 0., "hsv_s": 0., "hsv_v": 0.,
        "fliplr": .5, "flipud": .5, "degrees": 5., "scale": .1, "translate": .05,
        "close_mosaic": 0, "nbs": 64, "rect": False, "multi_scale": 0.,
    },
    "recommended_metadata": [],
}


def _merge(base, override, prefix=""):
    if not isinstance(override, dict):
        raise ValueError(f"{prefix or 'config'} must be a mapping")
    for key, value in override.items():
        if key not in base:
            raise ValueError(f"unknown configuration key: {prefix}{key}")
        if isinstance(base[key], dict):
            _merge(base[key], value, f"{prefix}{key}.")
        else:
            base[key] = deepcopy(value)


def load_config(common=None, modality=None, overrides=None):
    result = deepcopy(DEFAULTS)
    for path in (common, modality):
        if path:
            _merge(result, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    for key, value in (overrides or {}).items():
        node = result
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ValueError(f"unknown configuration key: {key}")
            node = node[part]
        _merge(node, {parts[-1]: value}, ".".join(parts[:-1])+".")
    validate_config(result)
    return result


def validate_config(config, *, ultralytics=False):
    baseline = deepcopy(DEFAULTS)
    _merge(baseline, config)
    if config["schema_version"] != 1 or config["modality"] not in {"haadf_stem", "hrtem"}:
        raise ValueError("unsupported schema or modality")
    if config["class_names"] != ["atom_column"] or config["coordinates"] != DEFAULTS["coordinates"]:
        raise ValueError("v1 requires one atom_column class and integer pixel centers (xy)")
    norm, train, inf, aug = (config[k] for k in ("image", "training", "inference", "augmentation"))
    norm = norm["normalization"]
    if norm["method"] != "percentile" or norm["output_range"] != [0., 1.]:
        raise ValueError("only percentile normalization to [0,1] is supported")
    if not 0 <= norm["low"] < norm["high"] <= 100:
        raise ValueError("invalid normalization percentiles")
    for field in ("imgsz", "batch", "epochs"):
        if type(train[field]) is not int or train[field] <= 0:
            raise ValueError(f"training.{field} must be a positive integer")
    if train["imgsz"] % 32 or inf["tile_size"] != [train["imgsz"]]*2:
        raise ValueError("imgsz must be divisible by 32 and equal both tile_size dimensions")
    if type(inf["max_det"]) is not int or inf["max_det"] < 1:
        raise ValueError("inference.max_det must be a positive integer")
    for key in ("conf", "iou"):
        if not 0 <= inf[key] <= 1:
            raise ValueError(f"inference.{key} must be in [0,1]")
    for key in ("brightness", "contrast", "probability"):
        if not isinstance(aug[key], (int, float)) or not 0 <= aug[key] <= 1:
            raise ValueError(f"augmentation.{key} must be in [0,1]")
    if any(train[k] != 0 for k in ("hsv_h", "hsv_s", "hsv_v", "mosaic", "mixup", "copy_paste")):
        raise ValueError("v1 grayscale training requires hsv/mosaic/mixup/copy_paste = 0")
    if train["multi_scale"] or train["rect"]:
        raise ValueError("v1 uses fixed square inputs")
    if train["cache"] is not False:
        raise ValueError("v1 reads verified image/label files directly; training.cache must be false")
    point = config.get("point_validation", DEFAULTS["point_validation"])
    if (type(point["enabled"]) is not bool or type(point["interval_epochs"]) is not int
            or point["interval_epochs"] < 1 or isinstance(point["match_distance_px"], bool)
            or not isinstance(point["match_distance_px"], (int, float))
            or not math.isfinite(point["match_distance_px"]) or point["match_distance_px"] <= 0):
        raise ValueError("invalid point_validation settings: boolean enabled, positive interval and match distance required")
    from .pipeline import PipelineConfig
    pipeline_config(config)
    if ultralytics:
        from ultralytics.cfg import get_cfg
        get_cfg(overrides={**train, "seed": config["seed"], "max_det": inf["max_det"],
                           "iou": inf["iou"]})
    return config


def pipeline_config(config):
    from .pipeline import PipelineConfig
    inf, ref = config["inference"], config["refinement"]
    return PipelineConfig(tile_size=tuple(inf["tile_size"]), tile_overlap=inf["tile_overlap"],
        merge_distance_px=inf["merge_distance_px"], min_atom_distance_px=inf["min_atom_distance_px"],
        refine=ref["enabled"], refinement_method=ref["method"],
        refinement_window=ref["window_size"], refinement_polarity=ref["polarity"],
        merge_after_refinement=inf.get("merge_after_refinement", False),
        adaptive_merge_sigma=ref.get("merge_sigma"),
        refinement_max_shift_px=ref["max_shift_px"])


def model_contract(config):
    return {"contract": "raw_tile_percentile_uint8_letterbox_v1", "input_size": config["training"]["imgsz"],
            "layout": "NCHW", "channels": 3, "dtype": "float32", "batch": 1,
            "padding": 114, "normalization": config["image"]["normalization"],
            "confidence_decimals": 6,
            "output": "yolov8_single_class_xywh_score_no_nms", "opset": 17}
