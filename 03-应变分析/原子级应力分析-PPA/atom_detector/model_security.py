"""Model loading security guard for optional atom detectors.

Only model files whose SHA-256 digest appears in the allowlist may be
loaded.  PyTorch checkpoints can execute arbitrary code during
deserialization, so loading an arbitrary ``.pt`` file is treated as a
security boundary rather than a convenience feature.

Allowlist file format (JSON)::

    {
      "description": "PPA atom-detector model allowlist",
      "sha256": ["<64-hex digest>", ...]
    }
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


class ModelVerificationError(RuntimeError):
    """Raised when a model file is missing, unknown, or unverifiable."""


DEFAULT_ALLOWLIST = Path(__file__).resolve().parent / "models" / "allowlist.json"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_allowlist(allowlist_path: str | Path | None = None) -> dict:
    path = Path(allowlist_path) if allowlist_path else DEFAULT_ALLOWLIST
    if not path.is_file():
        raise ModelVerificationError(f"模型白名单不存在: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelVerificationError(f"模型白名单无法解析: {path} ({exc})") from exc
    digests = data.get("sha256") if isinstance(data, dict) else None
    if not isinstance(digests, list):
        expected = '{"sha256": ["<64-character hex digest>"]}'
        raise ModelVerificationError(
            f"模型白名单格式无效: {path}（需要 {expected} 列表）"
        )
    hex_digits = set("0123456789abcdefABCDEF")
    for entry in digests:
        if (
            not isinstance(entry, str)
            or len(entry) != 64
            or any(c not in hex_digits for c in entry)
        ):
            expected = '{"sha256": ["<64-character hex digest>"]}'
            raise ModelVerificationError(
                f"模型白名单格式无效: {path} 中的 {entry!r} 不是 64 位十六进制 "
                f"SHA-256 摘要（需要 {expected}）"
            )
    return data


def verify_model(model_path: str | Path, allowlist_path: str | Path | None = None) -> str:
    """Verify a model against the allowlist; return its SHA-256 on success."""
    path = Path(model_path).resolve()
    if not path.is_file():
        raise ModelVerificationError(f"模型文件不存在: {path}")
    allowlist = load_allowlist(allowlist_path)
    digest = sha256_file(path)
    if digest not in set(allowlist["sha256"]):
        raise ModelVerificationError(
            f"模型未在白名单中，已拒绝加载: {path}\n"
            f"SHA-256: {digest}\n"
            f"请确认模型来源与哈希后，运行:\n"
            f"  python -m atom_detector.model_security {path} --allow"
        )
    return digest


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("用法: python -m atom_detector.model_security <model.pt> [--allow]")
        raise SystemExit(1)
    target = Path(sys.argv[1]).resolve()
    if not target.is_file():
        raise SystemExit(f"模型文件不存在: {target}")
    digest = sha256_file(target)
    print(f"SHA-256: {digest}")
    print(f"路径: {target}")
    if "--allow" in sys.argv:
        allowlist = DEFAULT_ALLOWLIST
        allowlist.parent.mkdir(parents=True, exist_ok=True)
        if allowlist.is_file():
            data = load_allowlist(allowlist)
        else:
            data = {"description": "PPA 原子检测模型白名单", "sha256": []}
        if digest not in data["sha256"]:
            data["sha256"].append(digest)
            allowlist.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"已加入白名单: {allowlist}")
        else:
            print("该模型已在白名单中。")
