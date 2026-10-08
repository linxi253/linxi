# -*- coding: utf-8 -*-
"""pytest 配置：注册 slow 标记（真实完整链测试）。

分层运行建议：
* 快速单元/边界：``pytest tests -q -m "not slow"``
* 真实完整链：  ``pytest tests -q -m slow``（需要 FFmpeg，缺失时明确 skip）
"""
from __future__ import annotations


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: 真实子进程完整链测试（需要上游 FFmpeg 解析成功）",
    )
