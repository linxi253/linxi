import json
from pathlib import Path

import pytest

import video_extractor.ffmpeg as ffmpeg_module
from video_extractor.ffmpeg import (
    FFmpegError,
    FFmpegManager,
    _parse_duration,
    _resolve_bits_per_sample,
    is_variable_fps,
    parse_rate,
    probe_video,
)


class _FakeProbeProcess:
    def __init__(
        self,
        stdout: bytes | None,
        stderr: bytes | None = b"",
        returncode: int = 0,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    def communicate(self, timeout: int) -> tuple[bytes | None, bytes | None]:
        return self._stdout, self._stderr

    def poll(self) -> int:
        return self.returncode


def _valid_probe_payload() -> bytes:
    payload = {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 64,
                "height": 48,
                "avg_frame_rate": "30/1",
                "r_frame_rate": "30/1",
                "pix_fmt": "yuv420p",
            }
        ],
        "format": {
            "filename": "样品.mp4",
            "duration": "2.0",
        },
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def test_probe_video_captures_utf8_json_in_binary_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    popen_kwargs: dict[str, object] = {}

    def fake_popen(command: list[str], **kwargs: object) -> _FakeProbeProcess:
        popen_kwargs.update(kwargs)
        return _FakeProbeProcess(_valid_probe_payload())

    monkeypatch.setattr(ffmpeg_module.subprocess, "Popen", fake_popen)

    info = probe_video(Path("ffprobe"), Path("样品.mp4"))

    assert "text" not in popen_kwargs
    assert "encoding" not in popen_kwargs
    assert info.width == 64
    assert info.height == 48
    assert info.duration_s == 2.0


def test_probe_video_reports_missing_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ffmpeg_module.subprocess,
        "Popen",
        lambda command, **kwargs: _FakeProbeProcess(None),
    )

    with pytest.raises(FFmpegError, match="未返回探测数据"):
        probe_video(Path("ffprobe"), Path("sample.mp4"))


def test_probe_video_reports_non_utf8_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ffmpeg_module.subprocess,
        "Popen",
        lambda command, **kwargs: _FakeProbeProcess(b"\xff"),
    )

    with pytest.raises(FFmpegError, match="UTF-8"):
        probe_video(Path("ffprobe"), Path("sample.mp4"))


def test_probe_video_reports_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ffmpeg_module.subprocess,
        "Popen",
        lambda command, **kwargs: _FakeProbeProcess(b"not-json"),
    )

    with pytest.raises(FFmpegError, match="JSON 无效"):
        probe_video(Path("ffprobe"), Path("sample.mp4"))


def test_version_parser_accepts_btb_n_prefixed_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = ffmpeg_module.subprocess.CompletedProcess(
        ["ffmpeg", "-version"],
        0,
        stdout=b"ffmpeg version n8.1.2-44-g7c533d0f86-20260821\n",
        stderr=b"",
    )
    monkeypatch.setattr(ffmpeg_module.subprocess, "run", lambda *args, **kwargs: completed)
    assert FFmpegManager._version(Path("ffmpeg")) == (8, 1, 2)


def test_parse_rate_handles_fraction_integer_and_invalid() -> None:
    assert parse_rate("30000/1001") == 30000 / 1001
    assert parse_rate("30/1") == 30.0
    assert parse_rate("30") == 30.0
    assert parse_rate("0/0") is None
    assert parse_rate("") is None
    assert parse_rate(None) is None
    assert parse_rate("bad") is None
    assert parse_rate("1/0") is None


def test_ntsc_whitelist_is_not_vfr() -> None:
    assert is_variable_fps("30000/1001", "30/1") is False
    assert is_variable_fps("30/1", "30000/1001") is False
    assert is_variable_fps("24000/1001", "24/1") is False
    assert is_variable_fps("60000/1001", "60/1") is False


def test_genuine_rate_difference_is_vfr() -> None:
    assert is_variable_fps("30000/1001", "25/1") is True
    assert is_variable_fps("10/1", "20/1") is True


def test_missing_rate_is_not_vfr() -> None:
    assert is_variable_fps(None, "30/1") is False
    assert is_variable_fps("30/1", "0/0") is False


def test_parse_duration_rejects_invalid_stream_duration() -> None:
    assert _parse_duration("N/A") is None
    assert _parse_duration(2.0) == 2.0
    assert _parse_duration(0.0) is None
    assert _parse_duration(-1) is None


def test_resolve_bits_prefers_declared_component_depth() -> None:
    assert _resolve_bits_per_sample(
        {"bits_per_raw_sample": "10", "pix_fmt": "yuv420p10le"}
    ) == (10, "declared")
    assert _resolve_bits_per_sample(
        {"bits_per_sample": "8", "pix_fmt": "yuv420p"}
    ) == (8, "declared")


def test_resolve_bits_infers_from_pix_fmt_when_fields_missing() -> None:
    """字段缺失/为 0 时按像素格式推断，防止高位深源被静默按 8 位处理。"""
    assert _resolve_bits_per_sample({"pix_fmt": "yuv420p10le"}) == (10, "pix_fmt")
    assert _resolve_bits_per_sample({"bits_per_raw_sample": "0", "pix_fmt": "yuv420p10le"}) == (10, "pix_fmt")
    assert _resolve_bits_per_sample({"pix_fmt": "p010le"}) == (10, "pix_fmt")
    assert _resolve_bits_per_sample({"pix_fmt": "gray16le"}) == (16, "pix_fmt")
    assert _resolve_bits_per_sample({"pix_fmt": "rgb48le"}) == (16, "pix_fmt")
    assert _resolve_bits_per_sample({"pix_fmt": "yuv420p9le"}) == (9, "pix_fmt")


def test_resolve_bits_ignores_packed_total_depth() -> None:
    """rgb24 报告的 bits_per_sample=24 是打包总位深而非组件位深，不可信。"""
    assert _resolve_bits_per_sample(
        {"bits_per_sample": "24", "pix_fmt": "rgb24"}
    ) == (8, "default")


def test_resolve_bits_defaults_to_8_when_nothing_known() -> None:
    assert _resolve_bits_per_sample({}) == (8, "default")
    assert _resolve_bits_per_sample({"pix_fmt": "yuv420p"}) == (8, "default")
    assert _resolve_bits_per_sample({"bits_per_raw_sample": "N/A"}) == (8, "default")


def _probe_popen_with_payload(monkeypatch: pytest.MonkeyPatch, payload: bytes):
    monkeypatch.setattr(
        ffmpeg_module.subprocess,
        "Popen",
        lambda command, **kwargs: _FakeProbeProcess(payload),
    )


def test_probe_parses_rotation_from_display_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 64,
                "height": 48,
                "avg_frame_rate": "30/1",
                "r_frame_rate": "30/1",
                "pix_fmt": "yuv420p",
                "side_data_list": [
                    {"side_data_type": "Display Matrix", "rotation": -90},
                ],
            }
        ],
    }
    _probe_popen_with_payload(
        monkeypatch, json.dumps(payload).encode("utf-8")
    )
    info = probe_video(Path("ffprobe"), Path("rotated.mp4"))
    assert info.rotation_degrees == -90


def test_probe_rejects_nonpositive_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "streams": [
            {"index": 0, "codec_type": "video", "width": 0, "height": 48,
             "avg_frame_rate": "30/1", "r_frame_rate": "30/1", "pix_fmt": "yuv420p"}
        ],
    }
    _probe_popen_with_payload(monkeypatch, json.dumps(payload).encode("utf-8"))
    with pytest.raises(FFmpegError, match="非正值"):
        probe_video(Path("ffprobe"), Path("broken.mp4"))


def test_probe_video_ordinal_counts_attached_pictures(monkeypatch: pytest.MonkeyPatch) -> None:
    """封面图流也是视频流：v:N 序号必须把它计入，否则首帧探测会选错流。"""
    payload = {
        "streams": [
            {"index": 0, "codec_type": "video", "width": 8, "height": 8,
             "avg_frame_rate": "1/1", "r_frame_rate": "1/1", "pix_fmt": "yuv420p",
             "disposition": {"attached_pic": 1}},
            {"index": 1, "codec_type": "video", "width": 64, "height": 48,
             "avg_frame_rate": "30/1", "r_frame_rate": "30/1", "pix_fmt": "yuv420p"},
        ],
    }
    _probe_popen_with_payload(monkeypatch, json.dumps(payload).encode("utf-8"))
    info = probe_video(Path("ffprobe"), Path("with_cover.mkv"))
    assert info.stream_count == 1
    assert info.stream_index == 1
    assert info.video_stream_ordinal == 1


def test_probe_video_counts_multiple_usable_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    """多视频流（含封面）：stream_count 只计可用流，首流选择不受收集影响。"""
    payload = {
        "streams": [
            {"index": 0, "codec_type": "video", "width": 8, "height": 8,
             "avg_frame_rate": "1/1", "r_frame_rate": "1/1", "pix_fmt": "yuv420p",
             "disposition": {"attached_pic": 1}},
            {"index": 1, "codec_type": "video", "width": 64, "height": 48,
             "avg_frame_rate": "30/1", "r_frame_rate": "30/1", "pix_fmt": "yuv420p"},
            {"index": 2, "codec_type": "video", "width": 32, "height": 24,
             "avg_frame_rate": "15/1", "r_frame_rate": "15/1", "pix_fmt": "yuv420p"},
        ],
    }
    _probe_popen_with_payload(monkeypatch, json.dumps(payload).encode("utf-8"))
    info = probe_video(Path("ffprobe"), Path("multi_stream.mkv"))
    assert info.stream_count == 2
    # 首选流仍是第一个非封面流（解码语义保持不变）
    assert info.stream_index == 1
    assert info.video_stream_ordinal == 1
    assert info.width == 64
    assert info.height == 48


def test_probe_first_frame_pts_parses_first_line(monkeypatch: pytest.MonkeyPatch) -> None:
    from video_extractor.ffmpeg import probe_first_frame_pts

    completed = ffmpeg_module.subprocess.CompletedProcess(
        [], 0, stdout=b"1.000000\n1.100000\n", stderr=b""
    )
    monkeypatch.setattr(
        ffmpeg_module.subprocess, "run", lambda *a, **k: completed
    )
    assert probe_first_frame_pts(Path("ffprobe"), Path("v.mp4")) == 1.0


def test_probe_first_frame_pts_returns_none_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from video_extractor.ffmpeg import probe_first_frame_pts

    completed = ffmpeg_module.subprocess.CompletedProcess(
        [], 1, stdout=b"", stderr=b"boom"
    )
    monkeypatch.setattr(
        ffmpeg_module.subprocess, "run", lambda *a, **k: completed
    )
    assert probe_first_frame_pts(Path("ffprobe"), Path("v.mp4")) is None

    completed_ok = ffmpeg_module.subprocess.CompletedProcess(
        [], 0, stdout=b"N/A\n", stderr=b""
    )
    monkeypatch.setattr(
        ffmpeg_module.subprocess, "run", lambda *a, **k: completed_ok
    )
    assert probe_first_frame_pts(Path("ffprobe"), Path("v.mp4")) is None
