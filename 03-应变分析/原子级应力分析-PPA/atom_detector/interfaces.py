"""Optional detector interface; the core application has no ML runtime dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    points: np.ndarray
    confidences: np.ndarray
    provider: str
    model_sha256: str | None = None


class AtomDetectorProvider(Protocol):
    """Contract for future detector providers such as a verified YOLO/ONNX model."""

    def detect(self, image: np.ndarray, *, roi: tuple[float, float, float, float] | None = None) -> DetectionResult:
        ...


class DetectorUnavailable(RuntimeError):
    """Raised when a user requests an optional provider that is not installed/configured."""


def get_optional_detector(*args, **kwargs) -> AtomDetectorProvider:
    """Intentional placeholder until the ML pipeline is separately validated.

    The application must not load arbitrary .pt files: PyTorch checkpoints can
    execute code during deserialization.  A future provider must verify a known
    SHA-256 allowlist and expose a reproducible model manifest before replacing
    this placeholder.
    """
    raise DetectorUnavailable(
        "深度学习检测框架已预留，但当前稳定版未启用模型加载。"
        "请使用内置传统检测或等待经过验证的模型提供方。"
    )
