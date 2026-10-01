# -*- coding: utf-8 -*-
"""
应力面积统计核心纯函数模块（无 GUI/图像 I/O 依赖）。

从 `应力面积统计.py` 中抽出，便于单元测试：
- safe_corrcoef / safe_cov：常量数据保护
- safe_divide
- get_smooth_curve：Savitzky-Golay 平滑（短序列保护）
- normalize_frame / rgb_to_gray：单页图像维度规范化
"""

import numpy as np
from scipy.signal import savgol_filter


def safe_corrcoef(x, y) -> float:
    """安全计算 Pearson 相关系数，常量数据返回 0.0 而非 NaN。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return 0.0
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def corrcoef_or_nan(x, y) -> float:
    """相关系数的展示口径：无法定义时返回 NaN 而非 0.0。

    safe_corrcoef 的 0.0 填充值面向数值管线；但在报告/图表中把"常量序列
    或有效样本不足"显示为 r=0.000 会被误读为"测得零相关"（与
    rolling_correlation 已修复的语义问题一致）。展示用本函数：
    - 先剔除任一序列中非有限的样本点；
    - 有效样本 <2 或任一序列常量 → NaN。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"corrcoef_or_nan 要求同形序列，得到 {x.shape} 与 {y.shape}")
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    x, y = x[finite], y[finite]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_cov(x, y) -> float:
    """安全计算协方差，常量数据返回 0.0 而非 NaN/警告。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return 0.0
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.cov(x, y)[0, 1])


def safe_divide(numerator, denominator, fill=0.0):
    """安全除法，分母为零或接近零时返回 fill。"""
    try:
        if abs(denominator) < 1e-12:
            return fill
        return numerator / denominator
    except (TypeError, ValueError):
        return fill


def get_smooth_curve(times, data, max_window=21, polyorder=3):
    """
    自适应窗口的 Savitzky-Golay 滤波。

    window_length 必须为奇数且 > polyorder；短序列直接返回原始数据。
    max_window/polyorder 可由 AnalysisConfig 注入，默认即历史行为。
    """
    times = np.asarray(times)
    data = np.asarray(data)
    n = len(data)

    if n < 3:
        return times, data

    window = min(max_window, n if n % 2 == 1 else n - 1)
    if window < polyorder + 2:  # savgol_filter 要求 window_length > polyorder
        return times, data

    smoothed = savgol_filter(data, window_length=window, polyorder=polyorder)
    return times, smoothed


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """将 (..., H, W, 3/4) RGB(A) 转灰度。

    返回与输入相同 dtype 的灰度数组；浮点输入直接加权，整型输入 clip 到
    原始整型范围。"""
    rgb = np.asarray(rgb)
    if rgb.ndim < 3 or rgb.shape[-1] not in (3, 4):
        raise ValueError(f"rgb_to_gray 需要 (..., H, W, 3/4) 输入，得到 {rgb.shape}")

    channels = rgb[..., :3].astype(np.float64)
    weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)
    gray = channels @ weights

    dtype = rgb.dtype
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(gray, 0, info.max).astype(dtype)
    return np.clip(gray, 0.0, None).astype(dtype)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """把单页 TIF 数据规范化为 2D 灰度图像 ``(H, W)``。

    支持：
    - ``(H, W)`` 灰度
    - ``(H, W, 3/4)`` RGB(A)
    - ``(1, H, W)`` 单通道伪装成 3D
    - ``(1, H, W, 1)`` 等 4D 单页
    """
    frame = np.asarray(frame)

    if frame.ndim == 2:
        return frame

    if frame.ndim == 3:
        if frame.shape[-1] in (3, 4):
            return rgb_to_gray(frame)
        if frame.shape[0] == 1:
            return frame[0]
        # 页迭代中基本不会出现 (T,H,W) 的多页堆栈被传到这里
        raise ValueError(
            f"单页规范化失败：3D 形状 {frame.shape} 既不是 (H,W,C=3/4) 也不是 (1,H,W)")

    if frame.ndim == 4:
        if frame.shape[0] == 1 and frame.shape[-1] == 1:
            return frame[0, :, :, 0]
        if frame.shape[0] == 1 and frame.shape[-1] in (3, 4):
            return rgb_to_gray(frame[0])
        raise ValueError(f"单页规范化失败：4D 形状 {frame.shape} 不是单页彩色/灰度")

    raise ValueError(f"单页规范化失败：不支持的维度 {frame.ndim}D，形状 {frame.shape}")
