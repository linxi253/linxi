"""集成/回归测试共享 fixture。

ffmpeg_paths 依赖本机可解析到的真实 FFmpeg/ffprobe 二进制对（≥8.0.3 且版本
一致）。干净检出上没有 tools/ffmpeg/*.exe、PATH 也没有可用 FFmpeg 时，
此前会直接抛 FFmpegError 使整个模块 ERROR；这里统一兜底为 SKIP，
并给出自解释的跳过原因。本地跑这套用例前请先按 tools/ffmpeg/README.md
放置 FFmpeg，或在 CI 现场下载。
"""

from __future__ import annotations

import pytest

from video_extractor.ffmpeg import FFmpegError, FFmpegManager


@pytest.fixture(scope="module")
def ffmpeg_paths():
    try:
        return FFmpegManager().resolve()
    except FFmpegError as exc:
        pytest.skip(f"需要 FFmpeg ≥8.0.3：{exc}")
