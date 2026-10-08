"""Versioned project persistence with image identity checks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from .validation import ProjectValidationError, validate_project_data


SCHEMA_VERSION = 2
APP_VERSION = "3.4.1"
MAX_PROJECT_BYTES = 32*1024*1024


def image_identity(path: str | Path, *, frame_index: int | None = None) -> dict[str, Any]:
    file_path = Path(path).resolve()
    if not file_path.is_file():
        raise ProjectValidationError(f"Image does not exist: {file_path}")
    initial_stat = file_path.stat()
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    final_stat = file_path.stat()
    if (initial_stat.st_size, initial_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
        raise ProjectValidationError("Image changed while computing its identity.")
    return {"path": str(file_path), "sha256": digest.hexdigest(), "size_bytes": final_stat.st_size, "frame_index": frame_index}


def build_project(payload: dict[str, Any], image_path: str | Path, *, frame_index: int | None = None,
                  expected_identity: dict | None = None) -> dict[str, Any]:
    data = validate_project_data(payload)
    identity = image_identity(image_path, frame_index=frame_index)
    if expected_identity and any(identity[key] != expected_identity.get(key) for key in ('sha256', 'size_bytes', 'frame_index')):
        raise ProjectValidationError("Original image changed since it was opened; reload it before saving.")
    data.update(
        {
            "schema_version": SCHEMA_VERSION,
            "app_version": APP_VERSION,
            "coordinate_convention": "physical-cartesian-x-right-y-up; image-display-y-down",
            "image": identity,
        }
    )
    return data


def save_project(path: str | Path, payload: dict[str, Any], image_path: str | Path, *, frame_index: int | None = None,
                 expected_identity: dict | None = None) -> None:
    """Save JSON atomically so interruption cannot leave a half-written project."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = build_project(payload, image_path, frame_index=frame_index, expected_identity=expected_identity)
    temp_name = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(target.parent), suffix=".tmp") as handle:
            temp_name = handle.name
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temp_name, target)
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def load_project(path: str | Path, *, require_image: bool = True) -> dict[str, Any]:
    if Path(path).stat().st_size > MAX_PROJECT_BYTES:
        raise ProjectValidationError("Project exceeds the supported size limit.")
    with Path(path).open("r", encoding="utf-8") as handle:
        def reject_constant(value):
            raise ProjectValidationError(f"Non-finite JSON value: {value}")
        data = json.load(handle, parse_constant=reject_constant)
    data = validate_project_data(data)
    version = data.get("schema_version", 1)
    if version != SCHEMA_VERSION:
        if version == 1:
            data["legacy_unverified"] = True
            return data
        raise ProjectValidationError(f"Unsupported project schema version: {version}")
    image = data.get("image")
    if not isinstance(image, dict):
        raise ProjectValidationError("Project has no image identity metadata.")
    if require_image:
        current = image_identity(image.get("path", ""), frame_index=image.get("frame_index"))
        if current["sha256"] != image.get("sha256") or current["size_bytes"] != image.get("size_bytes"):
            raise ProjectValidationError("Project image does not match its saved SHA-256 identity.")
    return data
