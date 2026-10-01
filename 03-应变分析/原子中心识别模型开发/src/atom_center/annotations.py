"""Lossless point-annotation projects for atomic-resolution microscopy.

Exact ``(x, y)`` centers and explicitly completed coverage regions are the
source of truth.  Detector boxes and normalized training images are derived
artifacts and can therefore be regenerated with different settings.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .image_io import inspect_image_series, load_image, normalize_percentile
from .coordinates import points_to_yolo
from .source_identity import observe_source, verify_source

ANNOTATION_SCHEMA_VERSION = 1
PROJECT_SCHEMA_VERSION = 2
SUPPORTED_IMAGE_SUFFIXES = frozenset(
    {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
)
VALID_MODALITIES = frozenset({"haadf_stem", "hrtem"})
VALID_REVIEW_STATES = frozenset({"unannotated", "draft", "reviewed"})
MODALITY_RECOMMENDED_METADATA: Mapping[str, tuple[str, ...]] = {
    "haadf_stem": (
        "sample_id",
        "acquisition_id",
        "pixel_size",
        "accelerating_voltage_kv",
        "detector_inner_angle_mrad",
        "detector_outer_angle_mrad",
    ),
    "hrtem": (
        "sample_id",
        "acquisition_id",
        "pixel_size",
        "accelerating_voltage_kv",
        "defocus_nm",
        "spherical_aberration_mm",
        "contrast_polarity",
    ),
}

DEFAULT_METADATA: Mapping[str, str] = {
    "sample_id": "",
    "acquisition_id": "",
    "parent_field_id": "",
    "parent_source_sha256": "",
    "material": "",
    "zone_axis": "",
    "microscope_id": "",
    "pixel_size": "",
    "pixel_size_unit": "angstrom_per_pixel",
    "accelerating_voltage_kv": "",
    "detector_inner_angle_mrad": "",
    "detector_outer_angle_mrad": "",
    "defocus_nm": "",
    "spherical_aberration_mm": "",
    "contrast_polarity": "bright",
    "annotator": "",
    "notes": "",
}

MANIFEST_FIELDS = (
    "image_id",
    "sample_id",
    "acquisition_id",
    "group_id_source",
    "missing_metadata_fields",
    "modality",
    "image_path",
    "label_path",
    "series_index",
    "frame_index",
    "image_height",
    "image_width",
    "pixel_size",
    "pixel_size_unit",
    "accelerating_voltage_kv",
    "detector_inner_angle_mrad",
    "detector_outer_angle_mrad",
    "defocus_nm",
    "spherical_aberration_mm",
    "contrast_polarity",
    "material",
    "zone_axis",
    "microscope_id",
    "annotator",
    "review_status",
    "point_count",
    "coverage_region_count",
    "coverage_fraction",
    "notes",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _json_text(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _resolve_stored_path(value: str, base: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _store_path(path: Path, base: Path) -> str:
    absolute = path.expanduser().resolve()
    try:
        return absolute.relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(absolute)


def _manifest_path(path: Path, manifest_parent: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), manifest_parent.resolve())).as_posix()
    except ValueError:
        return str(path.resolve())


def _safe_stem(text: str) -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", text, flags=re.UNICODE).strip("._")
    return (cleaned or "image")[:48]


def make_image_id(
    relative_path: str,
    *,
    series_index: int,
    frame_index: int | None,
) -> str:
    """Create a stable, readable identifier for one selectable 2-D plane."""

    normalized = Path(relative_path).as_posix()
    key = f"{normalized.casefold()}|series={series_index}|frame={frame_index}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    stem = _safe_stem(Path(normalized).stem)
    suffix = f"s{series_index}"
    if frame_index is not None:
        suffix += f"f{frame_index:04d}"
    return f"{stem}__{suffix}__{digest}"


@dataclass(frozen=True)
class ProjectRecord:
    image_id: str
    image_path: str
    series_index: int
    frame_index: int | None
    image_shape: tuple[int, int]
    source_identity: dict[str, object] | None = None

    def __post_init__(self):
        if self.source_identity:
            identity = self.source_identity
            if (identity.get("series_index") != self.series_index
                    or identity.get("frame_index") != self.frame_index
                    or tuple(identity.get("shape", ())) != tuple(self.image_shape)):
                raise ValueError("record frame/shape differs from its observed source identity")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ProjectRecord":
        shape = payload.get("image_shape", ())
        if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
            raise ValueError("record image_shape must be [height, width]")
        values = tuple(int(value) for value in shape)
        if len(values) != 2:
            raise ValueError("record image_shape must contain two values")
        raw_frame = payload.get("frame_index")
        return cls(
            image_id=str(payload["image_id"]),
            image_path=str(payload["image_path"]),
            series_index=int(payload.get("series_index", 0)),
            frame_index=None if raw_frame is None else int(raw_frame),
            image_shape=(values[0], values[1]),
            source_identity=dict(payload["source_identity"]) if payload.get("source_identity") else None,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "series_index": self.series_index,
            "frame_index": self.frame_index,
            "image_shape": list(self.image_shape),
            "source_identity": self.source_identity,
        }


def _coerce_points(payload: object) -> list[tuple[float, float]]:
    if payload is None:
        return []
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError("points must be a list")
    points: list[tuple[float, float]] = []
    for item in payload:
        if isinstance(item, Mapping):
            point = (float(item["x"]), float(item["y"]))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            if len(item) != 2:
                raise ValueError("each point must contain x and y")
            point = (float(item[0]), float(item[1]))
        else:
            raise ValueError("each point must be an object or [x, y]")
        points.append(point)
    return points


def _coerce_regions(payload: object) -> list[tuple[float, float, float, float]]:
    if payload is None:
        return []
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError("coverage_regions_xyxy must be a list")
    regions: list[tuple[float, float, float, float]] = []
    for item in payload:
        if isinstance(item, Mapping):
            region = tuple(float(item[name]) for name in ("x0", "y0", "x1", "y1"))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            if len(item) != 4:
                raise ValueError("each coverage region must contain x0, y0, x1, y1")
            region = tuple(float(value) for value in item)
        else:
            raise ValueError("invalid coverage region")
        regions.append(region)  # type: ignore[arg-type]
    return regions


@dataclass
class AnnotationDocument:
    image_id: str
    image_path: str
    series_index: int
    frame_index: int | None
    image_shape: tuple[int, int]
    modality: str
    points_xy: list[tuple[float, float]] = field(default_factory=list)
    coverage_regions_xyxy: list[tuple[float, float, float, float]] = field(
        default_factory=list
    )
    metadata: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_METADATA)
    )
    review_status: str = "unannotated"
    atom_diameter_px: float = 6.0
    revision: int = 0
    created_utc: str = field(default_factory=utc_now)
    updated_utc: str = field(default_factory=utc_now)

    @classmethod
    def new(
        cls,
        record: ProjectRecord,
        *,
        modality: str,
        metadata: Mapping[str, object] | None = None,
    ) -> "AnnotationDocument":
        merged = dict(DEFAULT_METADATA)
        if metadata:
            merged.update({str(key): str(value) for key, value in metadata.items()})
        if modality == "hrtem" and merged["contrast_polarity"] == "bright":
            merged["contrast_polarity"] = "mixed"
        return cls(
            image_id=record.image_id,
            image_path=record.image_path,
            series_index=record.series_index,
            frame_index=record.frame_index,
            image_shape=record.image_shape,
            modality=modality,
            metadata=merged,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AnnotationDocument":
        schema_version = int(payload.get("schema_version", 0))
        if schema_version != ANNOTATION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported annotation schema version: {schema_version}"
            )
        raw_shape = payload.get("image_shape", ())
        if not isinstance(raw_shape, Sequence) or isinstance(raw_shape, (str, bytes)):
            raise ValueError("image_shape must be [height, width]")
        shape = tuple(int(value) for value in raw_shape)
        if len(shape) != 2:
            raise ValueError("image_shape must contain two values")
        raw_metadata = payload.get("metadata", {})
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("metadata must be an object")
        metadata = dict(DEFAULT_METADATA)
        metadata.update({str(key): str(value) for key, value in raw_metadata.items()})
        raw_frame = payload.get("frame_index")
        document = cls(
            image_id=str(payload["image_id"]),
            image_path=str(payload["image_path"]),
            series_index=int(payload.get("series_index", 0)),
            frame_index=None if raw_frame is None else int(raw_frame),
            image_shape=(shape[0], shape[1]),
            modality=str(payload["modality"]),
            points_xy=_coerce_points(
                payload.get("points_xy", payload.get("points", []))
            ),
            coverage_regions_xyxy=_coerce_regions(
                payload.get("coverage_regions_xyxy", [])
            ),
            metadata=metadata,
            review_status=str(payload.get("review_status", "unannotated")),
            atom_diameter_px=float(payload.get("atom_diameter_px", 6.0)),
            revision=int(payload.get("revision", 0)),
            created_utc=str(payload.get("created_utc", utc_now())),
            updated_utc=str(payload.get("updated_utc", utc_now())),
        )
        validate_document(document, for_review=False)
        return document

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "image_id": self.image_id,
            "image_path": self.image_path,
            "series_index": self.series_index,
            "frame_index": self.frame_index,
            "image_shape": list(self.image_shape),
            "coordinate_order": "xy",
            "modality": self.modality,
            "points_xy": [
                {"x": float(x), "y": float(y)} for x, y in self.points_xy
            ],
            "coverage_regions_xyxy": [
                {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
                for x0, y0, x1, y1 in self.coverage_regions_xyxy
            ],
            "metadata": dict(self.metadata),
            "review_status": self.review_status,
            "atom_diameter_px": float(self.atom_diameter_px),
            "revision": int(self.revision),
            "created_utc": self.created_utc,
            "updated_utc": self.updated_utc,
        }


def point_in_regions(
    point: tuple[float, float],
    regions: Sequence[tuple[float, float, float, float]],
) -> bool:
    x, y = point
    return any(x0 <= x < x1 and y0 <= y < y1 for x0, y0, x1, y1 in regions)


def missing_recommended_metadata(document: AnnotationDocument) -> tuple[str, ...]:
    """Return useful-but-optional acquisition fields that are still blank."""

    return tuple(
        field_name
        for field_name in MODALITY_RECOMMENDED_METADATA[document.modality]
        if not document.metadata.get(field_name, "").strip()
    )


def validate_document(
    document: AnnotationDocument,
    *,
    for_review: bool,
) -> None:
    """Validate labels and any acquisition metadata the user chose to provide."""

    if document.modality not in VALID_MODALITIES:
        raise ValueError(f"unsupported modality: {document.modality!r}")
    if document.review_status not in VALID_REVIEW_STATES:
        raise ValueError(f"invalid review_status: {document.review_status!r}")
    height, width = document.image_shape
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if not math.isfinite(document.atom_diameter_px) or document.atom_diameter_px <= 0:
        raise ValueError("atom_diameter_px must be positive and finite")

    seen: set[tuple[int, int]] = set()
    for index, (x, y) in enumerate(document.points_xy):
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"point {index} is not finite")
        if not (0.0 <= x < width and 0.0 <= y < height):
            raise ValueError(
                f"point {index} ({x:.3f}, {y:.3f}) is outside {width}x{height}"
            )
        quantized = (round(x * 1_000_000), round(y * 1_000_000))
        if quantized in seen:
            raise ValueError(f"point {index} duplicates an existing point")
        seen.add(quantized)

    for index, (x0, y0, x1, y1) in enumerate(document.coverage_regions_xyxy):
        values = (x0, y0, x1, y1)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"coverage region {index} is not finite")
        if any(abs(value - round(value)) > 1e-9 for value in values):
            raise ValueError(
                f"coverage region {index} must use integer pixel bounds"
            )
        if not (0.0 <= x0 < x1 <= width and 0.0 <= y0 < y1 <= height):
            raise ValueError(
                f"coverage region {index} is outside {width}x{height}: {values}"
            )
        for previous_index, (px0, py0, px1, py1) in enumerate(
            document.coverage_regions_xyxy[:index]
        ):
            overlap_width = min(x1, px1) - max(x0, px0)
            overlap_height = min(y1, py1) - max(y0, py0)
            if overlap_width > 0.0 and overlap_height > 0.0:
                raise ValueError(
                    f"coverage regions {previous_index} and {index} overlap; "
                    "use non-overlapping regions to avoid duplicate training labels"
                )

    if for_review or document.review_status == "reviewed":
        if not document.coverage_regions_xyxy:
            raise ValueError("审核前必须明确整图或至少一个完整标注区域")
        numeric_fields = {
            "pixel_size",
            "accelerating_voltage_kv",
            "detector_inner_angle_mrad",
            "detector_outer_angle_mrad",
            "defocus_nm",
            "spherical_aberration_mm",
        }
        for field_name in MODALITY_RECOMMENDED_METADATA[document.modality]:
            raw_value = document.metadata.get(field_name, "").strip()
            if field_name not in numeric_fields or not raw_value:
                continue
            try:
                numeric_value = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"{field_name} 必须是数值") from exc
            if not math.isfinite(numeric_value):
                raise ValueError(f"{field_name} 必须是有限数值")
            if field_name in {"pixel_size", "accelerating_voltage_kv"} and numeric_value <= 0:
                raise ValueError(f"{field_name} 必须大于 0")
        inner_value = document.metadata.get("detector_inner_angle_mrad", "").strip()
        outer_value = document.metadata.get("detector_outer_angle_mrad", "").strip()
        if document.modality == "haadf_stem" and inner_value and outer_value:
            inner = float(inner_value)
            outer = float(outer_value)
            if inner < 0 or outer <= inner:
                raise ValueError("HAADF 探测器外收集角必须大于内收集角，且内角不能为负")
        outside = [
            point
            for point in document.points_xy
            if not point_in_regions(point, document.coverage_regions_xyxy)
        ]
        if outside:
            raise ValueError(
                f"有 {len(outside)} 个原子点位于完整标注区域之外"
            )


def _infer_path_metadata(relative_path: str) -> dict[str, str]:
    """Infer IDs only from the documented ``sample/acquisition/file`` layout."""

    parts = Path(relative_path).parts
    if len(parts) >= 3:
        return {"sample_id": parts[0], "acquisition_id": parts[1]}
    return {}


@dataclass
class AnnotationProject:
    project_path: Path
    name: str
    modality: str
    image_root: Path
    label_root: Path
    manifest_path: Path
    records: list[ProjectRecord] = field(default_factory=list)
    project_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_utc: str = field(default_factory=utc_now)
    updated_utc: str = field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        project_path: str | Path,
        *,
        image_root: str | Path,
        modality: str,
        label_root: str | Path | None = None,
        manifest_path: str | Path | None = None,
        name: str | None = None,
    ) -> "AnnotationProject":
        if modality not in VALID_MODALITIES:
            raise ValueError(
                f"modality must be one of {sorted(VALID_MODALITIES)}, got {modality!r}"
            )
        project_file = Path(project_path).expanduser().resolve()
        images = Path(image_root).expanduser().resolve()
        labels = (
            Path(label_root).expanduser().resolve()
            if label_root is not None
            else (project_file.parent / "labels").resolve()
        )
        manifest = (
            Path(manifest_path).expanduser().resolve()
            if manifest_path is not None
            else project_file.with_suffix(".csv")
        )
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        project = cls(
            project_path=project_file,
            name=name or f"{modality} atom-center annotations",
            modality=modality,
            image_root=images,
            label_root=labels,
            manifest_path=manifest,
        )
        project.sync_images()
        project.save()
        project.write_manifest()
        return project

    @classmethod
    def load(cls, project_path: str | Path) -> "AnnotationProject":
        project_file = Path(project_path).expanduser().resolve()
        payload = json.loads(project_file.read_text(encoding="utf-8"))
        version = int(payload.get("schema_version", 0))
        if version not in {1, PROJECT_SCHEMA_VERSION}:
            raise ValueError(f"unsupported project schema version: {version}")
        base = project_file.parent
        project = cls(
            project_path=project_file,
            name=str(payload["name"]),
            modality=str(payload["modality"]),
            image_root=_resolve_stored_path(str(payload["image_root"]), base),
            label_root=_resolve_stored_path(str(payload["label_root"]), base),
            manifest_path=_resolve_stored_path(str(payload["manifest_path"]), base),
            records=[
                ProjectRecord.from_dict(item)
                for item in payload.get("records", [])
            ],
            project_id=str(payload.get("project_id") or "legacy-" + hashlib.sha256(
                f"{payload.get('created_utc')}|{payload.get('modality')}|{payload.get('name')}".encode("utf-8")
            ).hexdigest()[:24]),
            created_utc=str(payload.get("created_utc", utc_now())),
            updated_utc=str(payload.get("updated_utc", utc_now())),
        )
        if project.modality not in VALID_MODALITIES:
            raise ValueError(f"unsupported project modality: {project.modality!r}")
        return project

    def to_dict(self) -> dict[str, object]:
        base = self.project_path.parent
        return {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "project_id": self.project_id,
            "name": self.name,
            "modality": self.modality,
            "image_root": _store_path(self.image_root, base),
            "label_root": _store_path(self.label_root, base),
            "manifest_path": _store_path(self.manifest_path, base),
            "records": [record.to_dict() for record in self.records],
            "created_utc": self.created_utc,
            "updated_utc": self.updated_utc,
        }

    def save(self) -> Path:
        self.updated_utc = utc_now()
        _atomic_write_text(self.project_path, _json_text(self.to_dict()))
        return self.project_path

    def image_file(self, record: ProjectRecord) -> Path:
        return (self.image_root / Path(record.image_path)).resolve()

    def label_file(self, record: ProjectRecord) -> Path:
        return (self.label_root / f"{record.image_id}.json").resolve()

    def sync_images(self) -> tuple[int, int]:
        """Add newly discovered image planes; never delete existing records."""

        self.image_root.mkdir(parents=True, exist_ok=True)
        existing = {
            (
                Path(record.image_path).as_posix().casefold(),
                record.series_index,
                record.frame_index,
            )
            for record in self.records
        }
        added = 0
        failures = 0
        files = sorted(
            (
                path
                for path in self.image_root.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            ),
            key=lambda path: path.as_posix().casefold(),
        )
        for image_file in files:
            relative = image_file.relative_to(self.image_root).as_posix()
            try:
                descriptions = inspect_image_series(image_file)
            except Exception:
                failures += 1
                continue
            for description in descriptions:
                frame_indices: Iterable[int | None]
                if description.frame_count == 1:
                    frame_indices = (None,)
                else:
                    frame_indices = range(description.frame_count)
                for frame_index in frame_indices:
                    key = (
                        relative.casefold(),
                        description.series_index,
                        frame_index,
                    )
                    if key in existing:
                        continue
                    self.records.append(
                        ProjectRecord(
                            image_id=make_image_id(
                                relative,
                                series_index=description.series_index,
                                frame_index=frame_index,
                            ),
                            image_path=relative,
                            series_index=description.series_index,
                            frame_index=frame_index,
                            image_shape=description.plane_shape,
                            source_identity=observe_source(
                                image_file, series_index=description.series_index,
                                frame_index=frame_index),
                        )
                    )
                    existing.add(key)
                    added += 1
        self.records.sort(
            key=lambda record: (
                record.image_path.casefold(),
                record.series_index,
                -1 if record.frame_index is None else record.frame_index,
            )
        )
        return added, failures

    def load_document(self, record: ProjectRecord) -> AnnotationDocument:
        verify_source(self.image_file(record), record.source_identity)
        path = self.label_file(record)
        if path.is_file():
            document = AnnotationDocument.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
            if document.image_id != record.image_id:
                raise ValueError(f"label image_id mismatch: {path}")
            if document.image_shape != record.image_shape:
                raise ValueError(f"label image shape mismatch: {path}")
            if (document.series_index != record.series_index or document.frame_index != record.frame_index
                    or document.image_path != record.image_path or document.modality != self.modality):
                raise ValueError(f"label source frame/path/modality mismatch: {path}")
            return document
        return AnnotationDocument.new(
            record,
            modality=self.modality,
            metadata=_infer_path_metadata(record.image_path),
        )

    def effective_group_ids(
        self, document: AnnotationDocument
    ) -> tuple[str, str, str]:
        """Return non-empty split IDs without pretending optional metadata was supplied.

        When the task owner did not provide group IDs and the documented folder
        layout cannot supply them, every such image in this portable task is kept
        in one stable fallback group.  This is conservative: it avoids accidental
        train/test leakage while the original blank fields remain blank in JSON.
        """

        sample_id = document.metadata.get("sample_id", "").strip()
        acquisition_id = document.metadata.get("acquisition_id", "").strip()
        if sample_id and acquisition_id:
            return sample_id, acquisition_id, "provided_or_path"
        token_source = f"{self.created_utc}\n{self.modality}\n{self.name}"
        token = hashlib.sha256(token_source.encode("utf-8")).hexdigest()[:12]
        return (
            sample_id or f"auto-sample-{token}",
            acquisition_id or f"auto-acquisition-{token}",
            "project_fallback",
        )

    def save_document(self, document: AnnotationDocument) -> Path:
        matching = next((r for r in self.records if r.image_id == document.image_id), None)
        if matching is not None:
            verify_source(self.image_file(matching), matching.source_identity)
            if (document.image_shape != matching.image_shape or document.image_path != matching.image_path
                    or document.series_index != matching.series_index or document.frame_index != matching.frame_index
                    or document.modality != self.modality):
                raise ValueError("label source frame/path/modality mismatch")
        record_ids = {record.image_id for record in self.records}
        if document.image_id not in record_ids:
            raise ValueError(f"document is not part of this project: {document.image_id}")
        validate_document(
            document,
            for_review=document.review_status == "reviewed",
        )
        document.revision += 1
        document.updated_utc = utc_now()
        path = self.label_root / f"{document.image_id}.json"
        _atomic_write_text(path, _json_text(document.to_dict()))
        return path

    def write_manifest(self) -> Path:
        rows: list[dict[str, object]] = []
        manifest_parent = self.manifest_path.parent
        for record in self.records:
            document = self.load_document(record)
            metadata = document.metadata
            sample_id, acquisition_id, group_id_source = self.effective_group_ids(
                document
            )
            area = sum(
                (x1 - x0) * (y1 - y0)
                for x0, y0, x1, y1 in document.coverage_regions_xyxy
            )
            image_area = record.image_shape[0] * record.image_shape[1]
            rows.append(
                {
                    "image_id": record.image_id,
                    "sample_id": sample_id,
                    "acquisition_id": acquisition_id,
                    "group_id_source": group_id_source,
                    "missing_metadata_fields": ";".join(
                        missing_recommended_metadata(document)
                    ),
                    "modality": self.modality,
                    "image_path": _manifest_path(
                        self.image_file(record), manifest_parent
                    ),
                    "label_path": _manifest_path(
                        self.label_file(record), manifest_parent
                    ),
                    "series_index": record.series_index,
                    "frame_index": "" if record.frame_index is None else record.frame_index,
                    "image_height": record.image_shape[0],
                    "image_width": record.image_shape[1],
                    "pixel_size": metadata.get("pixel_size", ""),
                    "pixel_size_unit": metadata.get("pixel_size_unit", ""),
                    "accelerating_voltage_kv": metadata.get(
                        "accelerating_voltage_kv", ""
                    ),
                    "detector_inner_angle_mrad": metadata.get(
                        "detector_inner_angle_mrad", ""
                    ),
                    "detector_outer_angle_mrad": metadata.get(
                        "detector_outer_angle_mrad", ""
                    ),
                    "defocus_nm": metadata.get("defocus_nm", ""),
                    "spherical_aberration_mm": metadata.get(
                        "spherical_aberration_mm", ""
                    ),
                    "contrast_polarity": metadata.get("contrast_polarity", ""),
                    "material": metadata.get("material", ""),
                    "zone_axis": metadata.get("zone_axis", ""),
                    "microscope_id": metadata.get("microscope_id", ""),
                    "annotator": metadata.get("annotator", ""),
                    "review_status": document.review_status,
                    "point_count": len(document.points_xy),
                    "coverage_region_count": len(document.coverage_regions_xyxy),
                    "coverage_fraction": min(1.0, area / image_area),
                    "notes": metadata.get("notes", ""),
                }
            )
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
        _atomic_write_text(self.manifest_path, buffer.getvalue())
        return self.manifest_path


@dataclass(frozen=True)
class ProjectStatistics:
    image_planes: int
    drafts: int
    reviewed: int
    total_points: int
    reviewed_points: int
    samples: int
    acquisitions: int


def project_statistics(project: AnnotationProject) -> ProjectStatistics:
    drafts = reviewed = total_points = reviewed_points = 0
    samples: set[str] = set()
    acquisitions: set[str] = set()
    for record in project.records:
        document = project.load_document(record)
        total_points += len(document.points_xy)
        if document.review_status == "draft":
            drafts += 1
        elif document.review_status == "reviewed":
            reviewed += 1
            reviewed_points += len(document.points_xy)
            sample, acquisition, _source = project.effective_group_ids(document)
            samples.add(sample)
            acquisitions.add(acquisition)
    return ProjectStatistics(
        image_planes=len(project.records),
        drafts=drafts,
        reviewed=reviewed,
        total_points=total_points,
        reviewed_points=reviewed_points,
        samples=len(samples),
        acquisitions=len(acquisitions),
    )


@dataclass(frozen=True)
class YoloExportSummary:
    output_dir: Path
    source_images: int
    derived_images: int
    points: int


def export_yolo_dataset(
    project: AnnotationProject,
    output_dir: str | Path,
    *,
    box_size_px: float = 8.0,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> YoloExportSummary:
    """Export reviewed coverage regions as normalized PNG crops plus YOLO boxes.

    The target must not already exist.  This protects earlier derived exports
    and makes every export an immutable, auditable snapshot.
    """

    if not math.isfinite(box_size_px) or box_size_px <= 0:
        raise ValueError("box_size_px must be positive and finite")
    if not (0.0 <= percentile_low < percentile_high <= 100.0):
        raise ValueError("invalid normalization percentiles")

    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"export target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    ).resolve()
    images_dir = temporary / "images"
    labels_dir = temporary / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    source_rows: list[dict[str, object]] = []
    source_images = derived_images = point_count = 0
    try:
        from PIL import Image

        for record in project.records:
            document = project.load_document(record)
            if document.review_status != "reviewed":
                continue
            validate_document(document, for_review=True)
            sample_id, acquisition_id, group_id_source = project.effective_group_ids(
                document
            )
            loaded = load_image(
                project.image_file(record),
                frame_index=record.frame_index,
                series_index=record.series_index,
                normalize=False,
            )
            raw = loaded.image
            source_images += 1
            for region_index, region in enumerate(document.coverage_regions_xyxy):
                x0, y0, x1, y1 = region
                left = max(0, int(math.floor(x0)))
                top = max(0, int(math.floor(y0)))
                right = min(raw.shape[1], int(math.ceil(x1)))
                bottom = min(raw.shape[0], int(math.ceil(y1)))
                if right <= left or bottom <= top:
                    raise ValueError(
                        f"empty integer crop for {record.image_id}, region {region_index}"
                    )
                crop = raw[top:bottom, left:right]
                normalized = normalize_percentile(
                    crop, low=percentile_low, high=percentile_high
                )
                pixels = np.rint(normalized * 255.0).astype(np.uint8)
                export_id = f"{record.image_id}__r{region_index:03d}"
                Image.fromarray(pixels).save(images_dir / f"{export_id}.png")

                crop_height, crop_width = pixels.shape
                region_points = [
                    (x - left, y - top)
                    for x, y in document.points_xy
                    if x0 <= x < x1 and y0 <= y < y1
                ]
                box_width = min(1.0, box_size_px / crop_width)
                box_height = min(1.0, box_size_px / crop_height)
                lines = [
                    "0 "
                    f"{center[0]:.8f} {center[1]:.8f} "
                    f"{box_width:.8f} {box_height:.8f}"
                    for center in points_to_yolo(region_points, crop_width, crop_height)
                ]
                _atomic_write_text(
                    labels_dir / f"{export_id}.txt",
                    ("\n".join(lines) + "\n") if lines else "",
                )
                point_count += len(region_points)
                derived_images += 1
                source_rows.append(
                    {
                        "export_id": export_id,
                        "image_id": record.image_id,
                        "source_image": str(project.image_file(record)),
                        "series_index": record.series_index,
                        "frame_index": ""
                        if record.frame_index is None
                        else record.frame_index,
                        "roi_x0": left,
                        "roi_y0": top,
                        "roi_x1": right,
                        "roi_y1": bottom,
                        "sample_id": sample_id,
                        "acquisition_id": acquisition_id,
                        "group_id_source": group_id_source,
                        "missing_metadata_fields": ";".join(
                            missing_recommended_metadata(document)
                        ),
                        "point_count": len(region_points),
                    }
                )

        if source_images == 0:
            raise ValueError(
                "当前任务还没有已审核的标注。请先为图片指定完整标注范围，"
                "然后点击“审核通过并下一帧”。"
            )

        source_fields = (
            "export_id",
            "image_id",
            "source_image",
            "series_index",
            "frame_index",
            "roi_x0",
            "roi_y0",
            "roi_x1",
            "roi_y1",
            "sample_id",
            "acquisition_id",
            "group_id_source",
            "missing_metadata_fields",
            "point_count",
        )
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(source_fields))
        writer.writeheader()
        writer.writerows(source_rows)
        _atomic_write_text(temporary / "source_map.csv", buffer.getvalue())
        _atomic_write_text(
            temporary / "export_manifest.json",
            _json_text(
                {
                    "schema_version": 1,
                    "created_utc": utc_now(),
                    "source_project": str(project.project_path),
                    "modality": project.modality,
                    "coordinate_order": "xy",
                    "source_coordinate_origin": "top-left pixel center",
                    "yolo_center_conversion": (
                        "normalized=((source_xy - crop_xy0) + 0.5) / crop_wh"
                    ),
                    "class_names": {"0": "atom"},
                    "box_size_px": box_size_px,
                    "normalization_percentiles": [
                        percentile_low,
                        percentile_high,
                    ],
                    "source_images": source_images,
                    "derived_images": derived_images,
                    "point_count": point_count,
                    "split_warning": (
                        "Not split. Group source_map.csv by acquisition_id before "
                        "creating train/validation/test sets. Rows marked "
                        "project_fallback use a conservative task-level group because "
                        "the original IDs were left blank."
                    ),
                }
            ),
        )
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return YoloExportSummary(
        output_dir=target,
        source_images=source_images,
        derived_images=derived_images,
        points=point_count,
    )
