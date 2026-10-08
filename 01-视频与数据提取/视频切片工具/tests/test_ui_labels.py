"""UI 状态标签与 JobState 的完整性契约（回归 2026-10-03）。

STATE_LABELS 曾漏掉 "probing"：runner 会用
``VideoResult(state=JobState.PROBING)`` 进入该状态，缺失标签时界面只能退化为
原始英文状态名。这里以 ``JobState`` 为唯一来源断言覆盖完整，避免再次漏项。

终态（completed/cancelled/failed）由 ui.py 的独立分支处理（L465 起），
不经过 STATE_LABELS，因此不在断言范围内。
"""
from __future__ import annotations

from video_extractor.models import JobState
from video_extractor.ui import STATE_LABELS

# ui.py 用独立分支渲染的终态，不经 STATE_LABELS
TERMINAL_STATES = {"completed", "cancelled", "failed"}


def test_every_non_terminal_job_state_has_a_label() -> None:
    missing = [state.value for state in JobState
               if state.value not in TERMINAL_STATES
               and state.value not in STATE_LABELS]
    assert not missing, f"STATE_LABELS 缺少这些状态的界面文案: {missing}"


def test_state_labels_are_chinese_and_non_empty() -> None:
    for key, label in STATE_LABELS.items():
        assert label and label.strip(), f"{key} 的标签不能为空"
        assert any("\u4e00" <= ch <= "\u9fff" for ch in label), (
            f"{key} 的标签应为中文界面文案，实际为 {label!r}"
        )


def test_probing_label_present() -> None:
    """显式断言本轮修复的状态（runner.py 会设置 PROBING）。"""
    assert STATE_LABELS["probing"] == "正在探测视频信息…"
