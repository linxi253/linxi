"""Processing provenance written next to, rather than hidden inside, TIFF data."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import tifffile

from ._version import __version__
from .params import FilterParams, SaveOptions


def compute_file_sha256(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Streamed SHA-256 of a file; used for optional archival input identity.

    The stat signature in ``input.integrity`` detects mid-run changes but is
    spoofable (mtime/size can be preserved deliberately).  A content hash is
    the archival-grade identifier, at the cost of one full read.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def build_provenance(
    *,
    input_path: str | Path,
    input_shape: tuple[int, int],
    input_dtype: np.dtype,
    frame_count: int,
    params: FilterParams,
    save_options: SaveOptions,
    roi: tuple[int, int, int, int] | None,
    display_range: tuple[float, float] | None,
    source_metadata: dict[str, Any],
    input_integrity: dict[str, Any] | None = None,
    output_dtype: np.dtype | None = None,
    output_frame_shape: tuple[int, int] | None = None,
    elapsed_ms: int | None = None,
    entry: str | None = None,
    fft_workers: int | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "software": "hrtem-filter",
        "software_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "entry": entry,
        "input": {
            "path": str(Path(input_path).resolve()),
            "frame_shape_yx": list(input_shape),
            "frame_count": frame_count,
            "dtype": str(input_dtype),
            "integrity": input_integrity or {},
        },
        "output": {
            "dtype": str(output_dtype) if output_dtype is not None else None,
            "frame_shape_yx": list(output_frame_shape) if output_frame_shape is not None else None,
        },
        "elapsed_ms": elapsed_ms,
        "roi_tlbr": list(roi) if roi is not None else None,
        "source_tiff_metadata": source_metadata,
        "filter_parameters": params.as_dict(),
        "save_options": {
            "encoding": save_options.encoding.value,
            "display_range": list(display_range) if display_range else None,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "tifffile": tifffile.__version__,
            "fft_workers": fft_workers,
        },
    }


def restore_json_bytes(path: str | Path, previous: bytes | None) -> Path | None:
    """Restore the previous provenance payload after a failed publish.

    A re-run replaces the existing ``.processing.json`` before the TIFF is
    committed; if the commit then fails, the published TIFF is still the old
    one and must keep its old record.  ``previous`` is the exact bytes read
    before the replacement, or ``None`` when no file existed (the file is
    removed).  Writing back is atomic for the same reason as
    :func:`write_json_atomically`.
    """
    target = Path(path)
    if previous is None:
        with contextlib.suppress(FileNotFoundError):
            target.unlink()
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".partial", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(previous)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
    return target


def write_json_atomically(path: str | Path, data: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".partial", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
    return target
