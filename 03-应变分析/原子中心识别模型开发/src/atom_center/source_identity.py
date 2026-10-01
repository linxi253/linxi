"""Identity of the original file and selected, unnormalized image plane."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from .image_io import load_image
from .model_manifest import sha256_file
from .storage import canonical_json


class SourceChangedError(ValueError):
    pass


def observe_source(path, *, series_index=0, frame_index=None, basis="observed_at_discovery"):
    path = Path(path).resolve()
    before = sha256_file(path)
    loaded = load_image(path, series_index=series_index, frame_index=frame_index,
                        normalize=False, preserve_dtype=True)
    array = np.asarray(loaded.image)
    little = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
    header = {"shape": list(little.shape), "dtype": little.dtype.str}
    digest = hashlib.sha256(canonical_json(header).encode("utf-8"))
    digest.update(little.tobytes())
    after = sha256_file(path)
    if before != after:
        raise SourceChangedError(f"source changed while reading: {path}")
    return {
        "file_sha256": before, "file_size": path.stat().st_size,
        "plane_sha256": digest.hexdigest(), "shape": list(array.shape),
        "dtype": array.dtype.str, "series_index": series_index,
        "frame_index": frame_index, "basis": basis,
    }


def verify_source(path, identity):
    if not identity:
        return False
    path = Path(path)
    if (not path.is_file() or path.stat().st_size != identity["file_size"]
            or sha256_file(path) != identity["file_sha256"]):
        raise SourceChangedError(f"原图内容已变化，不能继续使用旧标注：{path}")
    return True
