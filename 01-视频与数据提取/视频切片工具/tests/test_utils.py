import json
from pathlib import Path

import pytest

from video_extractor.utils import atomic_write_json, output_stem, safe_component


def test_safe_component_handles_windows_reserved_and_empty_names() -> None:
    assert safe_component("CON") == "_CON"
    assert safe_component("CON.txt") == "_CON.txt"
    assert safe_component("aux.avi") == "_aux.avi"
    assert safe_component("...") == "video"
    assert safe_component("a:b?c") == "a_b_c"


def test_output_stem_preserves_relative_parent_and_disambiguates() -> None:
    root = Path("C:/input")
    parent, stem = output_stem(root, root / "nested" / "same.mp4")
    assert parent == Path("nested")
    assert stem.startswith("same__")


def test_atomic_write_json_writes_and_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "cfg.json"
    atomic_write_json(target, {"a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    atomic_write_json(target, {"b": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"b": 2}


def test_atomic_write_json_replace_failure_cleans_temp(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    target.mkdir()  # os.replace onto an existing directory fails on POSIX
    with pytest.raises(OSError):
        atomic_write_json(target, {"a": 1})
    leftovers = [item for item in tmp_path.iterdir() if item.name != "out.json"]
    assert leftovers == []


def test_fsync_file_persists_written_data(tmp_path: Path) -> None:
    from video_extractor.utils import fsync_file

    target = tmp_path / "data.bin"
    target.write_bytes(b"pixel-data")
    fsync_file(target)  # Windows 上 FlushFileBuffers 需要可写句柄，O_RDWR 打开必须成功
    assert target.read_bytes() == b"pixel-data"


def test_atomic_write_json_rejects_nan_payload(tmp_path: Path) -> None:
    """NaN 会写出非标准 JSON；校验兜底必须在写文件阶段拒绝。"""
    import pytest as _pytest

    target = tmp_path / "bad.json"
    with _pytest.raises(ValueError):
        atomic_write_json(target, {"value": float("nan")})
    assert not target.exists()


def test_output_stem_survives_escape_from_input_root() -> None:
    """链接/junction 指向输入根之外时不能让整批扫描崩溃（退化而非失败）。"""
    root = Path("C:/input")
    # 同盘符逃逸：relpath 回退，.. 被映射为安全目录名
    parent, stem = output_stem(root, Path("C:/elsewhere/video.mp4"))
    assert stem.startswith("video__")
    assert all(part != ".." for part in parent.parts)
    # 跨盘符逃逸：relpath 也失败，退到 _external 目录
    parent2, stem2 = output_stem(root, Path("D:/other/video2.mp4"))
    assert stem2.startswith("video2__")
    assert list(parent2.parts)[0] == "_external"
    # 正常相对路径行为保持不变
    parent3, stem3 = output_stem(root, root / "nested" / "same.mp4")
    assert parent3 == Path("nested")
