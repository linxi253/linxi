# -*- coding: utf-8 -*-
"""export_io 模块的单元测试：源保护和原子导出。"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import export_io  # noqa: E402


def test_protect_source_blocks_case_variants(tmp_path):
    """审查项 01 的核心复现：Windows 大小写不敏感，SourceCase.tif 与
    sourcecase.tif 是同一个文件，纯 abspath 字符串比较拦不住。"""
    src = tmp_path / "SourceCase.tif"
    src.write_bytes(b"source pixels")
    with pytest.raises(ValueError):
        export_io.protect_source(str(tmp_path / "sourcecase.tif"), str(src))
    # 大小写敏感文件系统上 normcase 也应归一（Linux 上此测试同样通过）
    with pytest.raises(ValueError):
        export_io.protect_source(str(src), str(src))
    # 不同文件不受影响
    export_io.protect_source(str(tmp_path / "out.tif"), str(src))


def test_atomic_path_writes_and_replaces_without_leftovers(tmp_path):
    target = tmp_path / "out.json"
    with export_io.atomic_path(str(target)) as tmp:
        assert tmp != str(target) and os.path.exists(tmp)
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write("first")
    assert target.read_text(encoding="utf-8") == "first"

    with export_io.atomic_path(str(target)) as tmp:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write("second")
    assert target.read_text(encoding="utf-8") == "second"
    leftovers = [f for f in os.listdir(tmp_path) if f.startswith(".deloc_save_")]
    assert leftovers == []


def test_atomic_path_cleans_up_on_failure(tmp_path):
    target = tmp_path / "out.json"
    with pytest.raises(RuntimeError):
        with export_io.atomic_path(str(target)) as tmp:
            raise RuntimeError("boom")
    assert not target.exists()
    leftovers = [f for f in os.listdir(tmp_path) if f.startswith(".deloc_save_")]
    assert leftovers == []


def test_write_json_rejects_nan_and_is_atomic(tmp_path):
    target = tmp_path / "data.json"
    with pytest.raises(ValueError):          # allow_nan=False：非有限值是错误
        export_io.write_json(str(target), {"bad": float("nan")})
    assert not target.exists()               # 失败不留半成品
    export_io.write_json(str(target), {"ok": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": 1}


def test_write_json_refuses_source(tmp_path):
    src = tmp_path / "src.tif"
    src.write_bytes(b"original")
    with pytest.raises(ValueError):
        export_io.write_json(str(src), {"a": 1}, str(src))
    assert src.read_bytes() == b"original"   # 源文件未被破坏
