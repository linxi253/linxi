"""Versioned project persistence with image identity checks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


SCHEMA_VERSION = 2
APP_VERSION = "3.4.0"


class ProjectValidationError(ValueError):
    pass


def image_identity(path: str | Path, *, frame_index: int | None = None) -> dict[str, Any]:
    file_path = Path(path).resolve()
    if not file_path.is_file():
        raise ProjectValidationError(f"Image does not exist: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(file_path), "sha256": digest.hexdigest(), "size_bytes": file_path.stat().st_size, "frame_index": frame_index}


def build_project(payload: dict[str, Any], image_path: str | Path, *, frame_index: int | None = None) -> dict[str, Any]:
    data = dict(payload)
    data.update(
        {
            "schema_version": SCHEMA_VERSION,
            "app_version": APP_VERSION,
            "coordinate_convention": "physical-cartesian-x-right-y-up; image-display-y-down",
            "image": image_identity(image_path, frame_index=frame_index),
        }
    )
    return data


def save_project(path: str | Path, payload: dict[str, Any], image_path: str | Path, *, frame_index: int | None = None) -> None:
    """Save JSON atomically so interruption cannot leave a half-written project."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = build_project(payload, image_path, frame_index=frame_index)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(target.parent), suffix=".tmp") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        temp_name = handle.name
    try:
        os.replace(temp_name, target)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def load_project(path: str | Path, *, require_image: bool = True) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
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
