"""Reproducible model manifests and SHA-256 bundle verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class ModelVerificationError(RuntimeError):
    """Raised when model bytes do not match their release manifest."""


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_digest(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"{field_name} must be a 64-character hexadecimal digest")
    return normalized


@dataclass(frozen=True)
class ModelManifest:
    schema_version: int
    model_id: str
    modality: str
    version: str
    model_file: str
    model_format: str
    model_sha256: str
    provider: str
    created_utc: str
    dataset_manifest_sha256: str | None = None
    input: Mapping[str, Any] = field(default_factory=dict)
    inference: Mapping[str, Any] = field(default_factory=dict)
    refinement: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    software: Mapping[str, str] = field(default_factory=dict)
    git_commit: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported manifest schema: {self.schema_version}")
        if self.modality not in {"haadf_stem", "hrtem"}:
            raise ValueError("modality must be 'haadf_stem' or 'hrtem'")
        if not self.model_id.strip() or not self.version.strip() or not self.provider.strip():
            raise ValueError("model_id, version, and provider must be non-empty")
        model_path = Path(self.model_file)
        if model_path.is_absolute() or model_path.name != self.model_file:
            raise ValueError("model_file must be a basename without directory traversal")
        expected_format = model_path.suffix.lower().lstrip(".")
        if self.model_format.lower().lstrip(".") != expected_format:
            raise ValueError("model_format must match the model_file suffix")
        model_digest = _validate_digest(self.model_sha256, "model_sha256")
        if model_digest is None:
            raise ValueError("model_sha256 is required")
        object.__setattr__(self, "model_sha256", model_digest)
        object.__setattr__(
            self,
            "dataset_manifest_sha256",
            _validate_digest(
                self.dataset_manifest_sha256, "dataset_manifest_sha256"
            ),
        )

    @classmethod
    def create(
        cls,
        model_path: str | Path,
        *,
        model_id: str,
        modality: str,
        version: str,
        provider: str,
        dataset_manifest_path: str | Path | None = None,
        input: Mapping[str, Any] | None = None,
        inference: Mapping[str, Any] | None = None,
        refinement: Mapping[str, Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
        software: Mapping[str, str] | None = None,
        git_commit: str | None = None,
    ) -> "ModelManifest":
        path = Path(model_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        dataset_digest = (
            sha256_file(dataset_manifest_path)
            if dataset_manifest_path is not None
            else None
        )
        return cls(
            schema_version=1,
            model_id=model_id,
            modality=modality,
            version=version,
            model_file=path.name,
            model_format=path.suffix.lower().lstrip("."),
            model_sha256=sha256_file(path),
            provider=provider,
            created_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            dataset_manifest_sha256=dataset_digest,
            input=dict(input or {}),
            inference=dict(inference or {}),
            refinement=dict(refinement or {}),
            metrics=dict(metrics or {}),
            software=dict(software or {}),
            git_commit=git_commit,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        (output_path.parent / "sha256.txt").write_text(
            f"{self.model_sha256}  {self.model_file}\n", encoding="ascii"
        )
        return output_path

    @classmethod
    def read(cls, path: str | Path) -> "ModelManifest":
        manifest_path = Path(path)
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelVerificationError(
                f"cannot parse model manifest {manifest_path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ModelVerificationError("model manifest root must be a JSON object")
        try:
            return cls(**payload)
        except (TypeError, ValueError) as exc:
            raise ModelVerificationError(f"invalid model manifest: {exc}") from exc


def verify_model_bundle(manifest_path: str | Path) -> tuple[ModelManifest, Path]:
    """Verify a release bundle and return its manifest and resolved model path."""

    path = Path(manifest_path).resolve()
    manifest = ModelManifest.read(path)
    model_path = path.parent / manifest.model_file
    if not model_path.is_file():
        raise ModelVerificationError(f"model file does not exist: {model_path}")
    actual = sha256_file(model_path)
    if actual != manifest.model_sha256:
        raise ModelVerificationError(
            f"model SHA-256 mismatch for {model_path}: "
            f"expected {manifest.model_sha256}, got {actual}"
        )
    return manifest, model_path.resolve()
