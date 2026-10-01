"""Packaged-runtime smoke test for codecs and the transactional pipeline."""

import json
import queue
import tempfile
import threading
from pathlib import Path

import numpy as np
import tifffile
from imagecodecs import (
    deflate_decode,
    jpeg2k_decode,
    jpeg_decode,
    lzw_decode,
    packbits_decode,
    zstd_decode,
)

from pipeline import DEFAULT_PARAMS
from worker import ProcessingWorker


def run_self_test() -> dict:
    """Exercise LZW decoding, OME output, filtering, and provenance."""
    codecs = {
        "lzw": lzw_decode,
        "deflate": deflate_decode,
        "packbits": packbits_decode,
        "jpeg": jpeg_decode,
        "jpeg2000": jpeg2k_decode,
        "zstd": zstd_decode,
    }
    if not all(callable(codec) for codec in codecs.values()):
        raise RuntimeError("one or more supported TIFF codecs are unavailable")

    with tempfile.TemporaryDirectory(prefix="stem_enhancer_self_test_") as directory:
        root = Path(directory)
        input_path = root / "input.ome.tif"
        output_path = root / "output.ome.tif"
        data = np.arange(3 * 16 * 24, dtype=np.uint16).reshape(3, 16, 24)
        tifffile.imwrite(
            input_path,
            data,
            ome=True,
            metadata={"axes": "TYX", "PhysicalSizeX": 0.25},
            compression="lzw",
        )

        messages = queue.Queue()
        worker = ProcessingWorker(
            str(input_path),
            str(output_path),
            dict(DEFAULT_PARAMS),
            messages,
            threading.Event(),
        )
        worker.run()
        events = []
        while not messages.empty():
            events.append(messages.get())
        if "done" not in [event[0] for event in events]:
            error = next(
                (event[1] for event in events if event[0] == "error"),
                "unknown packaged self-test failure",
            )
            raise RuntimeError(error)

        with tifffile.TiffFile(output_path) as tif:
            if not tif.is_ome:
                raise RuntimeError("packaged self-test output lost OME metadata")
            if tif.series[0].shape != data.shape:
                raise RuntimeError("packaged self-test output shape mismatch")
            if tif.series[0].dtype != np.dtype(np.uint16):
                raise RuntimeError("packaged self-test output dtype mismatch")
            decoded = tif.asarray()
            if decoded.shape != data.shape or not np.isfinite(decoded).all():
                raise RuntimeError("packaged self-test output cannot be decoded")

        sidecar = Path(str(output_path) + ".stem.json")
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        if provenance.get("status") != "complete":
            raise RuntimeError("packaged self-test provenance is incomplete")

        return {
            "frames": int(data.shape[0]),
            "shape": list(data.shape),
            "dtype": str(decoded.dtype),
            "events": len(events),
            "codecs": sorted(codecs),
        }
