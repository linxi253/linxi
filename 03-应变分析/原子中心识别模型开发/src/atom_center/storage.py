"""Atomic artifact writes, canonical content hashes, and portable paths."""
from __future__ import annotations
import hashlib
import json
import math
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return clean_json(value.item())
    return value


def canonical_json(value):
    return json.dumps(clean_json(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def content_digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(clean_json(value), stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
    return path


def relative_path(path, base):
    try:
        return Path(os.path.relpath(Path(path).resolve(), Path(base).resolve())).as_posix()
    except ValueError:
        return str(Path(path).resolve())


def resolve_path(value, base):
    path = Path(value)
    return (path if path.is_absolute() else Path(base)/path).resolve()


@contextmanager
def new_artifact_directory(path):
    """Publish a new directory only after every component succeeds."""
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    # TemporaryDirectory owns only this freshly created, resolved child.
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.build-", dir=target.parent) as name:
        temporary = Path(name).resolve()
        if temporary.parent != target.parent:
            raise ValueError("temporary output escaped its parent")
        yield temporary
        if target.exists():
            raise FileExistsError(target)
        os.rename(temporary, target)
