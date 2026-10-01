from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def short_hash(value: str, length: int = 10) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def file_sha256(path: Path, chunk_size: int = 1024 * 1024, cancelled=None) -> str:
    """计算文件 SHA-256；传入 threading.Event 可在哈希大文件时响应取消。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError("哈希计算已取消")
            digest.update(chunk)
    return digest.hexdigest()


def safe_component(value: str, fallback: str = "video", max_length: int = 96) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(". ")
    if not cleaned:
        cleaned = fallback
    # Windows 对“保留名 + 任意扩展名”同样保留（CON.txt、AUX.avi 等）
    if cleaned.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES:
        cleaned = f"_{cleaned}"
    return cleaned[:max_length].rstrip(". ") or fallback


def output_stem(input_root: Path, video_path: Path) -> tuple[Path, str]:
    """计算输出父目录与去歧义的文件名主干。

    视频可能是指向输入根之外的链接/junction（resolve 后逃逸 relative_to），
    此时退化为字符串相对路径继续工作，而不是让整批任务在扫描阶段崩溃。
    """
    try:
        relative = video_path.resolve().relative_to(input_root.resolve())
        parent_parts = list(relative.parent.parts)
        unique_source = str(relative).casefold()
    except ValueError:
        try:
            text = os.path.relpath(video_path, input_root)
        except ValueError:
            # 跨盘符等无法计算相对路径的情况：统一放进 _external 目录
            text = os.path.join("_external", video_path.name)
        parent_parts = [
            part if part not in (os.pardir, os.curdir) else "_" + part.lstrip(".")
            for part in Path(text).parts[:-1]
        ]
        unique_source = str(video_path).casefold()
    safe_parent = Path(*(safe_component(part) for part in parent_parts))
    stem = safe_component(video_path.stem)
    unique = short_hash(unique_source)
    return safe_parent, f"{stem}__{unique}"


def fsync_directory(directory: Path) -> None:
    """尽力持久化目录项（Windows 跳过；Linux/Unix 上 fsync 目录 fd）。"""
    if os.name == "nt":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def fsync_file(path: Path) -> None:
    """把文件已写入的数据刷到物理磁盘，而不是只留在 OS 页缓存。

    memmap.flush()/文件关闭只保证进入页缓存；断电时未 fsync 的内容可能
    丢失或损坏。Windows 的 FlushFileBuffers 需要可写句柄，因此以 O_RDWR
    打开（文件此时代码路径中已关闭，不会被其他句柄占用）。
    """
    fd = os.open(path, os.O_RDWR)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp"
        ) as handle:
            temp_name = handle.name
            # allow_nan=False：NaN/Inf 会写出非标准 JSON（json.tool 等严格解析器
            # 拒绝），采样参数校验已拦上游，这里兜底防止污染清单文件
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
        fsync_directory(path.parent)
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def ensure_same_filesystem_stage(output_root: Path, job_id: str) -> Path:
    stage = output_root / ".partial" / job_id
    stage.mkdir(parents=True, exist_ok=False)
    return stage
