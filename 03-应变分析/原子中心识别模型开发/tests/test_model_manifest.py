from pathlib import Path

import pytest

from atom_center.model_manifest import (
    ModelManifest,
    ModelVerificationError,
    verify_model_bundle,
)


def test_model_manifest_round_trip_and_verification(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"verified-model-bytes")
    manifest = ModelManifest.create(
        model,
        model_id="haadf-baseline",
        modality="haadf_stem",
        version="v1",
        provider="onnxruntime",
        input={"normalization": "percentile-1-99"},
    )
    manifest_path = manifest.write(tmp_path / "model_manifest.json")
    loaded, verified_path = verify_model_bundle(manifest_path)
    assert loaded.model_sha256 == manifest.model_sha256
    assert verified_path == model.resolve()
    assert (tmp_path / "sha256.txt").is_file()


def test_tampered_model_is_rejected(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"original")
    manifest = ModelManifest.create(
        model,
        model_id="hrtem-baseline",
        modality="hrtem",
        version="v1",
        provider="onnxruntime",
    )
    path = manifest.write(tmp_path / "model_manifest.json")
    model.write_bytes(b"tampered")
    with pytest.raises(ModelVerificationError):
        verify_model_bundle(path)


def test_manifest_rejects_directory_traversal() -> None:
    with pytest.raises(ValueError):
        ModelManifest(
            schema_version=1,
            model_id="bad",
            modality="hrtem",
            version="v1",
            model_file="../model.onnx",
            model_format="onnx",
            model_sha256="a" * 64,
            provider="onnxruntime",
            created_utc="2026-08-31T00:00:00Z",
        )
