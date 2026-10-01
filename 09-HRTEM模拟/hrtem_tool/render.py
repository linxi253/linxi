"""图像后处理渲染：面内旋转/镜像/重采样、对比度、标尺。

oriented_image 与 20260820 验收脚本中的 oriented_rect 语义一致：
周期延拓 → 镜像 → 三次样条旋转（wrap 边界）→ 中心裁剪，无缝拼缝。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, rotate as ndimage_rotate


def robust_norm(arr: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    """按百分位数归一化到 [0, 1]。"""
    values = np.asarray(arr, dtype=float)
    lo, hi = np.percentile(values, (low, high))
    return np.clip((values - lo) / max(hi - lo, 1e-30), 0.0, 1.0)


def resize_float(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """LANCZOS 重采样（与 0820 验收管线一致，float32 精度）。"""
    return np.asarray(
        Image.fromarray(np.asarray(arr, np.float32), mode="F").resize(
            (shape[1], shape[0]), Image.Resampling.LANCZOS
        ),
        dtype=float,
    )


def oriented_image(
    arr: np.ndarray,
    sampling_a: float,
    angle_deg: float,
    mirror: bool,
    output_sampling_a: float = 0.0,
    output_shape: Optional[Tuple[int, int]] = None,
    shift_a: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """周期超胞的取向渲染：重采样 → 镜像 → 旋转 → 无缝中心裁剪。

    Parameters
    ----------
    arr : 原生采样下的模拟强度（假定 x/y 周期边界）。
    sampling_a : arr 的采样 Å/px。
    angle_deg, mirror : 面内旋转角（逆时针为正）与水平镜像。
    output_sampling_a : 输出采样；0 = 保持原生采样。
    output_shape : 输出形状；None = 与重采样后的整幅相同。
    shift_a : (y, x) 平移量 Å（裁剪中心偏移，匹配实验图时使用）。

    Note
    ----
    输出为中心裁剪：输入视场先重采样到 output_shape+16 px 再裁回
    output_shape，因此输出视场 ≈ 输入 FOV × output/(output+16)（约 1–2%）。
    该约定与 20260820 验收口径耦合（实验匹配中的采样标度已含此效应），
    修改会破坏既有标定，请勿轻易改动。
    """
    if output_sampling_a != 0 and not (
        np.isfinite(output_sampling_a) and output_sampling_a > 0
    ):
        raise ValueError(
            f"输出采样 {output_sampling_a!r} 无效（需 0=原生 或正的有限值）"
        )
    out_sampling = output_sampling_a if output_sampling_a > 0 else sampling_a
    if not np.isfinite(out_sampling) or out_sampling < 0.02:
        raise ValueError(
            f"输出采样 {out_sampling:g} Å/px 过小（< 0.02），渲染尺寸将超出内存合理范围"
        )
    if output_shape is None:
        output_shape = (
            max(8, int(round(arr.shape[0] * sampling_a / out_sampling))),
            max(8, int(round(arr.shape[1] * sampling_a / out_sampling))),
        )
    full_shape = (
        max(output_shape[0] + 16, int(round(arr.shape[0] * sampling_a / out_sampling))),
        max(output_shape[1] + 16, int(round(arr.shape[1] * sampling_a / out_sampling))),
    )
    full = resize_float(arr, full_shape)
    if mirror:
        full = full[:, ::-1]
    pad = max(full.shape) // 2 + 24
    tiled = np.pad(full, ((pad, pad), (pad, pad)), mode="wrap")
    turned = ndimage_rotate(tiled, angle_deg, reshape=False, order=3, mode="wrap")
    shift_y_px = shift_a[0] / out_sampling
    shift_x_px = shift_a[1] / out_sampling
    y0 = int(round((turned.shape[0] - output_shape[0]) / 2.0 - shift_y_px))
    x0 = int(round((turned.shape[1] - output_shape[1]) / 2.0 - shift_x_px))
    if y0 < 0 or x0 < 0 or y0 + output_shape[0] > turned.shape[0] \
            or x0 + output_shape[1] > turned.shape[1]:
        raise ValueError(
            f"shift_a={shift_a} 超出可用视场，裁剪窗口越界"
            f"（可用平移余量约 ±{min(turned.shape) - max(output_shape) // 2} px）"
        )
    return turned[y0 : y0 + output_shape[0], x0 : x0 + output_shape[1]]


def apply_display(
    oriented: np.ndarray,
    out_sampling_a: float,
    polarity: int = 1,
    blur_a: float = 0.0,
    lo_pct: float = 0.5,
    hi_pct: float = 99.5,
) -> np.ndarray:
    """对比度/极性/探测器模糊 → [0,1] 显示图（不改变取向）。"""
    data = np.asarray(oriented, dtype=float)
    if blur_a > 0:
        sigma_px = blur_a / out_sampling_a
        if sigma_px > 0.01:
            data = gaussian_filter(data, sigma_px)
    normed = robust_norm(data, lo_pct, hi_pct)
    return normed if polarity >= 0 else 1.0 - normed


def nice_scale_bar_length(fov_a: float) -> float:
    """选一个覆盖约 1/6 视场的 '好看' 标尺长度（Å）。"""
    nice = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500], dtype=float)
    target = fov_a / 6.0
    for value in nice:
        if value >= target:
            return float(value)
    return float(nice[-1])
