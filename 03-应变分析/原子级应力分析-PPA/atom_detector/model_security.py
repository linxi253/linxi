"""Model loading security guard for optional atom detectors.

Only model files whose SHA-256 digest appears in the allowlist may be
loaded.  PyTorch checkpoints can execute arbitrary code during
deserialization, so loading an arbitrary ``.pt`` file is treated as a
security boundary rather than a convenience feature.

Allowlist file format (JSON)::

    {
      "description": "PPA atom-detector model allowlist",
      "sha256": ["<64-hex digest>", ...],
      "records": [
        {"sha256": "<64-hex digest>", "source": "<来源说明>",
         "added_at": "<ISO8601 时间戳>"}
      ]
    }

工单78: 白名单是「首次使用即信任 (TOFU)」性质的防误加载措施, 不是来源认证;
``--allow`` 登记必须经 ``--source`` 显式声明模型来源, 并写入 ``records`` 留痕。
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



class ModelVerificationError(RuntimeError):
    """Raised when a model file is missing, unknown, or unverifiable."""


DEFAULT_ALLOWLIST = Path(__file__).resolve().parent / "models" / "allowlist.json"
# 与 models/allowlist.json 出货文案一致; 白名单文件被删后按此重建 (工单78)
DEFAULT_DESCRIPTION = (
    "PPA 原子检测模型白名单。只有 SHA-256 在此列表中的模型才允许被加载。"
    "白名单是「首次使用即信任(TOFU)」性质的防误加载措施，不是来源认证；"
    "加白须经 python -m atom_detector.model_security <model.pt> --allow "
    "--source <来源说明> 登记并在 records 留痕。"
)


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
        # 工单78: 不把可直接复制执行的加白命令作为标准修复步骤呈现——加白
        # 必须按文档流程显式登记(声明来源并留痕), 而不是遇阻就一键 --allow。
        raise ModelVerificationError(
            f"模型未在白名单中，已拒绝加载: {path}\n"
            f"SHA-256: {digest}\n"
            f"白名单是「首次使用即信任(TOFU)」性质的防误加载措施，不是来源认证。\n"
            f"请先自行核对模型来源与上述哈希；确认无误后，按 atom_detector/README.md\n"
            f"「模型安全」一节的登记流程显式加白（需声明来源并留痕）。"
        )
    return digest


def register_allow(
    digest: str,
    source: str,
    allowlist_path: str | Path = DEFAULT_ALLOWLIST,
) -> dict | None:
    """把 ``digest`` 连同来源说明写入白名单并留痕; 已存在时返回 None。

    工单78: ``--allow`` 不再只登记裸哈希——来源说明与登记时间写入
    ``records`` 数组, 使加白成为一次留痕的显式操作。
    """
    path = Path(allowlist_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        data = load_allowlist(path)
    else:
        data = {"description": DEFAULT_DESCRIPTION, "sha256": [], "records": []}
    if digest in data["sha256"]:
        return None
    data["sha256"].append(digest)
    record = {
        "sha256": digest,
        "source": str(source),
        "added_at": datetime.now().isoformat(timespec="seconds"),
    }
    data.setdefault("records", []).append(record)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return record


def _parse_flag_value(flag: str) -> str | None:
    """返回命令行中 ``flag`` 的值; 未提供时返回 None。"""
    argv = sys.argv
    if flag in argv:
        idx = argv.index(flag)
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("用法: python -m atom_detector.model_security <model.pt>")
        print("      python -m atom_detector.model_security <model.pt> --allow --source <来源说明>")
        raise SystemExit(1)
    target = Path(sys.argv[1]).resolve()
    if not target.is_file():
        raise SystemExit(f"模型文件不存在: {target}")
    digest = sha256_file(target)
    print(f"SHA-256: {digest}")
    print(f"路径: {target}")
    if "--allow" in sys.argv:
        # 工单78: 加白不再是一次跑完即永久信任的无痕操作——必须显式声明来源,
        # 交互终端二次确认, 来源/时间戳写入 allowlist.json 的 records 留痕。
        source = _parse_flag_value("--source")
        if not source:
            raise SystemExit(
                "拒绝登记: --allow 必须同时提供 --source <来源说明>\n"
                "(如下载 URL / 版本号 / 训练任务号 / 审批人)。\n"
                "白名单是防误加载措施而非来源认证，请先自行核对模型来源与哈希。")
        if sys.stdin is not None and sys.stdin.isatty():
            print(f"即将登记模型: {target}")
            print(f"  SHA-256: {digest}")
            print(f"  来源: {source}")
            print(f"  白名单: {DEFAULT_ALLOWLIST}")
            try:
                answer = input("确认登记? (输入 yes 继续): ").strip().lower()
            except (EOFError, OSError):
                # fail-closed: 读不到确认输入一律视为未确认, 不静默放行
                raise SystemExit("无法读取确认输入，已取消，未做任何更改。")
            if answer != "yes":
                raise SystemExit("已取消, 未做任何更改。")
        record = register_allow(digest, source)
        if record is not None:
            print(f"已加入白名单: {DEFAULT_ALLOWLIST}")
            print(f"已留痕: {record}")
        else:
            print("该模型已在白名单中。")
