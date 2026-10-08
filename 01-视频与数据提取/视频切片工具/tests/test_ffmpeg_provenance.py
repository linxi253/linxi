# -*- coding: utf-8 -*-
"""随包 FFmpeg/ffprobe 的 PROVENANCE 校验回归（2026-10-04）。

背景：候选此前完全没有运行期哈希校验，把同名 ffmpeg.exe 放进随包目录即可被
选中执行（Codex 探针 ffmpeg-provenance-before：bad_ffmpeg / bad_ffprobe 均被
接受）。本文件用**伪可执行字节**（不执行任何真实程序）验证：

* 匹配的随包对：通过；
* ffmpeg 被替换 / ffprobe 被替换：拒绝，且**不调用** ``_version``；
* 缺 PROVENANCE.md 或缺记录：按明确记录的兼容行为放行（不声称来源可信）；
* 读取异常：转为 FFmpegError，不崩溃；
* 外部显式路径与 ``allow_unsafe``：不强制校验（沿用既有政策）；
* imageio 回退仍参与候选（保留候选功能）。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from video_extractor.ffmpeg import FFmpegError, FFmpegManager


def _write_pair(directory: Path, *, ffmpeg_bytes=b"ffmpeg-bytes",
                ffprobe_bytes=b"ffprobe-bytes", provenance: bool = True,
                record: tuple[bool, bool] = (True, True)) -> tuple[Path, Path]:
    """写一对伪二进制；record 控制是否在 PROVENANCE.md 里登记各自哈希。"""
    directory.mkdir(parents=True, exist_ok=True)
    ffmpeg = directory / "ffmpeg.exe"
    ffprobe = directory / "ffprobe.exe"
    ffmpeg.write_bytes(ffmpeg_bytes)
    ffprobe.write_bytes(ffprobe_bytes)
    if provenance:
        rows = []
        for name, data, include in (("ffmpeg.exe", ffmpeg_bytes, record[0]),
                                    ("ffprobe.exe", ffprobe_bytes, record[1])):
            if include:
                rows.append(f"| {name} | {hashlib.sha256(data).hexdigest()} |")
        table = "| file | sha256 |\n|---|---|\n" + "\n".join(rows) + "\n"
        (directory / "PROVENANCE.md").write_text(table, encoding="utf-8")
    return ffmpeg, ffprobe


def _resolve_with(directory: Path, ffmpeg: Path, *, allow_unsafe=False,
                  calls: list | None = None):
    """在受控候选集上调用 resolve；version 被 mock，绝不执行真实程序。"""
    calls = calls if calls is not None else []

    def version(path):
        calls.append(Path(path).name)
        return (8, 1, 3)

    manager = FFmpegManager(allow_unsafe=allow_unsafe)
    with patch.object(FFmpegManager, "_candidates", return_value=[ffmpeg]), \
            patch.object(FFmpegManager, "_bundled_candidates", return_value=[ffmpeg]), \
            patch.object(FFmpegManager, "_version", side_effect=version):
        return manager.resolve()


# ---------------------------------------------------------------------------
# 匹配 / 错误哈希
# ---------------------------------------------------------------------------
def test_matching_bundled_pair_resolves(tmp_path):
    ffmpeg, _ = _write_pair(tmp_path / "bundle")
    calls: list = []
    paths = _resolve_with(tmp_path / "bundle", ffmpeg, calls=calls)
    assert paths.ffmpeg.name == "ffmpeg.exe"
    assert paths.version == (8, 1, 3)
    assert calls == ["ffmpeg.exe", "ffprobe.exe"]


@pytest.mark.parametrize("bad_name", ["ffmpeg.exe", "ffprobe.exe"])
def test_replaced_binary_is_rejected_before_version(tmp_path, bad_name):
    """被替换的随包二进制必须拒绝，且**先于**任何 -version 调用。"""
    directory = tmp_path / bad_name.replace(".", "_")
    ffmpeg, _ = _write_pair(directory)
    (directory / bad_name).write_bytes(b"changed bytes, identical mocked version")

    calls: list = []
    with pytest.raises(FFmpegError) as raised:
        _resolve_with(directory, ffmpeg, calls=calls)
    assert calls == [], f"校验应在运行任何二进制之前完成，实际调用 {calls}"
    assert "PROVENANCE.md" in str(raised.value)
    assert bad_name in str(raised.value)


def test_missing_provenance_is_permissive_and_documented(tmp_path):
    """缺 PROVENANCE.md：放行（兼容），且这不算"来源可信"。"""
    directory = tmp_path / "no-provenance"
    ffmpeg, _ = _write_pair(directory, provenance=False)
    calls: list = []
    paths = _resolve_with(directory, ffmpeg, calls=calls)
    assert paths.ffmpeg.name == "ffmpeg.exe"
    assert calls == ["ffmpeg.exe", "ffprobe.exe"]


def test_unrecorded_binary_is_permissive(tmp_path):
    """PROVENANCE.md 里没有该文件的记录：放行（兼容）。"""
    directory = tmp_path / "unrecorded"
    ffmpeg, _ = _write_pair(directory, record=(False, False))
    calls: list = []
    paths = _resolve_with(directory, ffmpeg, calls=calls)
    assert paths.ffmpeg.name == "ffmpeg.exe"
    assert calls == ["ffmpeg.exe", "ffprobe.exe"]


def test_partial_record_only_checks_recorded_file(tmp_path):
    """只登记 ffprobe：ffprobe 不匹配时仍拒绝，ffmpeg 无记录不阻断。"""
    directory = tmp_path / "partial"
    ffmpeg, ffprobe = _write_pair(directory, record=(False, True))
    ffprobe.write_bytes(b"replaced ffprobe")
    calls: list = []
    with pytest.raises(FFmpegError):
        _resolve_with(directory, ffmpeg, calls=calls)
    assert calls == []


# ---------------------------------------------------------------------------
# 读取异常不得崩溃
# ---------------------------------------------------------------------------
def test_hash_read_error_becomes_ffmpeg_error(tmp_path, monkeypatch):
    """计算哈希时的 IO 异常必须转成 FFmpegError（收集进 failures）。"""
    directory = tmp_path / "io-error"
    ffmpeg, _ = _write_pair(directory)

    import video_extractor.ffmpeg as module

    def boom(path, chunk_size=1024 * 1024):
        raise OSError("SYNTHETIC_READ_FAILURE")

    monkeypatch.setattr(module, "_sha256_file", boom)
    with pytest.raises(FFmpegError) as raised:
        _resolve_with(directory, ffmpeg)
    assert "SYNTHETIC_READ_FAILURE" in str(raised.value)


def test_unreadable_provenance_is_ffmpeg_error_before_version(tmp_path, monkeypatch):
    """PROVENANCE.md **存在但不可读**：FFmpegError，且不执行 -version。

    回归 2026-10-04 R21：此前 PermissionError 被 _parse_provenance_hashes 吞成
    空表 → 当成"没有记录"放行 → 仍然执行了 -version。存在却读不了是真实错误，
    必须与"确实不存在"区分。
    """
    directory = tmp_path / "unreadable"
    ffmpeg, _ = _write_pair(directory)

    real_read_text = Path.read_text

    def guarded(self, *args, **kwargs):
        if self.name == "PROVENANCE.md":
            raise PermissionError("SYNTHETIC_UNREADABLE")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    calls: list = []
    with pytest.raises(FFmpegError) as raised:
        _resolve_with(directory, ffmpeg, calls=calls)
    assert calls == [], f"读取失败时不得执行任何二进制，实际 {calls}"
    assert "无法读取或解码" in str(raised.value)


def test_invalid_utf8_provenance_is_ffmpeg_error_before_version(tmp_path):
    """PROVENANCE.md 存在但 UTF-8 损坏：FFmpegError，且不执行 -version。

    此前会原样抛 UnicodeDecodeError（不是 FFmpegError，未按契约收集）。
    """
    directory = tmp_path / "bad-utf8"
    ffmpeg, _ = _write_pair(directory)
    (directory / "PROVENANCE.md").write_bytes(b"\xff\xfe\x80")

    calls: list = []
    with pytest.raises(FFmpegError) as raised:
        _resolve_with(directory, ffmpeg, calls=calls)
    assert calls == [], f"解码失败时不得执行任何二进制，实际 {calls}"
    assert "无法读取或解码" in str(raised.value)


def test_missing_provenance_still_permissive_after_r21(tmp_path):
    """确实**不存在** PROVENANCE.md 时维持已声明的兼容政策（不阻断）。

    这条与上面的"存在但读不了"成对，确保修复没有把兼容行为一起收紧。
    """
    directory = tmp_path / "absent"
    ffmpeg, _ = _write_pair(directory, provenance=False)
    calls: list = []
    paths = _resolve_with(directory, ffmpeg, calls=calls)
    assert paths.ffmpeg.name == "ffmpeg.exe"
    assert calls == ["ffmpeg.exe", "ffprobe.exe"]


# ---------------------------------------------------------------------------
# 既有政策不被偷偷更改
# ---------------------------------------------------------------------------
def test_external_explicit_path_is_not_forced_through_provenance(tmp_path):
    """用户显式指定的外部路径：不做随包哈希校验（沿用既有政策）。"""
    external = tmp_path / "external"
    external.mkdir()
    ffmpeg = external / "ffmpeg.exe"
    ffmpeg.write_bytes(b"external ffmpeg")
    (external / "ffprobe.exe").write_bytes(b"external ffprobe")
    # 故意写一份与内容不符的 PROVENANCE，证明它不影响外部路径
    (external / "PROVENANCE.md").write_text(
        "| file | sha256 |\n|---|---|\n| ffmpeg.exe | " + "0" * 64 + " |\n",
        encoding="utf-8")

    calls: list = []
    manager = FFmpegManager(str(ffmpeg))

    def version(path):
        calls.append(Path(path).name)
        return (8, 1, 3)

    with patch.object(FFmpegManager, "_version", side_effect=version):
        paths = manager.resolve()
    assert paths.ffmpeg.name == "ffmpeg.exe"
    assert calls == ["ffmpeg.exe", "ffprobe.exe"]


def test_allow_unsafe_skips_provenance_check(tmp_path):
    """allow_unsafe（历史逃生舱）：跳过随包哈希校验。"""
    directory = tmp_path / "unsafe"
    ffmpeg, _ = _write_pair(directory)
    (directory / "ffmpeg.exe").write_bytes(b"replaced under allow_unsafe")
    calls: list = []
    paths = _resolve_with(directory, ffmpeg, allow_unsafe=True, calls=calls)
    assert paths.ffmpeg.name == "ffmpeg.exe"
    assert calls == ["ffmpeg.exe", "ffprobe.exe"]


# ---------------------------------------------------------------------------
# 保留候选功能
# ---------------------------------------------------------------------------
def test_imageio_fallback_still_participates(tmp_path):
    """imageio_ffmpeg 回退仍在候选集中（候选功能不得被本次合并移除）。"""
    fallback = tmp_path / "imageio" / "ffmpeg.exe"
    fallback.parent.mkdir(parents=True)
    fallback.write_bytes(b"imageio ffmpeg")
    (fallback.parent / "ffprobe.exe").write_bytes(b"imageio ffprobe")

    import video_extractor.ffmpeg as module

    fake = type("Fake", (), {"get_ffmpeg_exe": staticmethod(lambda: str(fallback))})
    with patch.dict("sys.modules", {"imageio_ffmpeg": fake}):
        candidates = FFmpegManager()._candidates()
    assert fallback in candidates, "imageio_ffmpeg 回退候选丢失"


def test_bundled_detection_ignores_non_bundled_paths(tmp_path):
    """_is_bundled 只认随包目录，不把任意同目录文件当随包。"""
    bundled = tmp_path / "tools" / "ffmpeg" / "ffmpeg.exe"
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b"x")
    other = tmp_path / "elsewhere" / "ffmpeg.exe"
    other.parent.mkdir()
    other.write_bytes(b"x")

    manager = FFmpegManager()
    with patch.object(FFmpegManager, "_bundled_candidates", return_value=[bundled]):
        assert manager._is_bundled(bundled) is True
        assert manager._is_bundled(other) is False
