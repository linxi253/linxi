"""Identical raw-tile normalization and letterboxing for datasets and inference."""
from __future__ import annotations
import numpy as np
from .coordinates import Letterbox
from .image_io import normalize_percentile


def prepare_tile(image, input_size, *, low=1., high=99.):
    import cv2
    raw = np.asarray(image)
    if raw.ndim != 2 or not raw.size or not np.isfinite(raw).all():
        raise ValueError("expected one nonempty finite grayscale plane")
    if not 0 <= low < high <= 100:
        raise ValueError("invalid normalization percentiles")
    shape = (input_size, input_size) if isinstance(input_size, int) else tuple(input_size)
    transform = Letterbox.create(raw.shape, shape)
    gray = np.rint(normalize_percentile(raw, low=low, high=high)*255).astype(np.uint8)
    rh, rw = transform.resized_hw
    if gray.shape != (rh, rw):
        gray = cv2.resize(gray, (rw, rh), interpolation=cv2.INTER_LINEAR)
    output = np.full(transform.target_hw, 114, dtype=np.uint8)
    px, py = transform.padding_xy
    output[py:py+rh, px:px+rw] = gray
    return output, transform


def input_tensor(gray):
    return np.ascontiguousarray(np.repeat(gray[None, None], 3, axis=1), dtype=np.float32)/255.
