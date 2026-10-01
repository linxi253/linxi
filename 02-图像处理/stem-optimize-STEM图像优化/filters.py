"""
Adaptive spectral denoising for two-dimensional HRTEM/STEM frames.

The implementation uses a rotationally averaged power spectrum as a background
estimate, a bounded Wiener-style attenuation mask, and a true local-standard-
deviation texture mask.  It is intentionally described as spectral denoising:
without a measured PSF/CTF it is not a physical deconvolution.
"""

import logging
from functools import lru_cache
from numbers import Real

import cv2
import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift, next_fast_len
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


def validate_real(name: str, value, *, minimum=None, maximum=None) -> float:
    """Validate one finite real parameter against optional inclusive bounds."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} 必须是数值")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} 必须是有限数值")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} 必须大于或等于 {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} 必须小于或等于 {maximum}")
    return result


@lru_cache(maxsize=2)
def _radial_grid(rows: int, cols: int):
    """Return a radius-index grid and bin counts for one FFT shape."""
    y = np.arange(rows, dtype=np.float32)[:, None] - rows // 2
    x = np.arange(cols, dtype=np.float32)[None, :] - cols // 2
    radius = np.sqrt(x * x + y * y).astype(np.int32)
    counts = np.bincount(radius.ravel())
    logger.debug("缓存频谱半径网格: %sx%s", rows, cols)
    return radius, counts


def clear_filter_caches():
    """Release cached FFT-size grids."""
    _radial_grid.cache_clear()


def radial_cache_info():
    """Expose bounded-cache statistics for diagnostics and tests."""
    return _radial_grid.cache_info()


def get_rotational_average(power_spectrum: np.ndarray) -> np.ndarray:
    """Return the rotational average expanded to the input shape."""
    spectrum = np.asarray(power_spectrum)
    if spectrum.ndim != 2:
        raise ValueError("功率谱必须是二维数组")
    if not np.issubdtype(spectrum.dtype, np.number):
        raise ValueError("功率谱必须是数值数组")
    if not np.isfinite(spectrum).all():
        raise ValueError("功率谱包含 NaN 或 Inf")

    rows, cols = spectrum.shape
    radius, counts = _radial_grid(rows, cols)
    sums = np.bincount(
        radius.ravel(), weights=spectrum.astype(np.float64, copy=False).ravel()
    )
    profile = sums / np.maximum(counts, 1)
    return profile[radius].astype(np.float32, copy=False)


def _texture_mask(raw_img: np.ndarray) -> np.ndarray:
    """Calculate a normalized local-standard-deviation texture mask."""
    height, width = raw_img.shape
    if height < 2 or width < 2:
        return np.zeros_like(raw_img, dtype=np.float32)

    small_width = max(1, width // 2)
    small_height = max(1, height // 2)
    small = cv2.resize(
        raw_img, (small_width, small_height), interpolation=cv2.INTER_AREA
    )
    # 对常规 uint8/uint16/float [0,1] 输入保持原 float32 计算路径不变；
    # 对大数值浮点图先按 max(abs) 稳健缩放到 [0,1] 再算方差，避免
    # float32 平方溢出产生 inf/nan。纹理掩码是 scale-invariant 的，
    # 因此缩放不改变最终掩码。
    max_abs = float(np.max(np.abs(small))) if small.size else 0.0
    compute_small = small
    if max_abs > 65535.0:
        compute_small = small / np.float32(max_abs)
    local_mean = cv2.GaussianBlur(
        compute_small,
        (0, 0),
        sigmaX=3.0,
        sigmaY=3.0,
        borderType=cv2.BORDER_REFLECT_101,
    )
    local_mean_sq = cv2.GaussianBlur(
        compute_small * compute_small,
        (0, 0),
        sigmaX=3.0,
        sigmaY=3.0,
        borderType=cv2.BORDER_REFLECT_101,
    )
    variance = np.maximum(local_mean_sq - local_mean * local_mean, 0.0)
    local_std = np.sqrt(variance, dtype=np.float32)
    scale = float(np.percentile(local_std, 99.5))
    if not np.isfinite(scale) or scale <= np.finfo(np.float32).eps:
        return np.zeros_like(raw_img, dtype=np.float32)

    mask_small = np.clip(local_std / scale, 0.0, 1.0)
    return cv2.resize(
        mask_small, (width, height), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32, copy=False)


def adaptive_spectral_filter(
    frame: np.ndarray,
    k_factor: float = 1.2,
    blend_ratio: float = 0.5,
    delta: float = 3.0,
) -> np.ndarray:
    """
    Apply adaptive rotational-background spectral denoising to one 2D frame.

    ``blend_ratio`` is the maximum filtered contribution.  The actual
    per-pixel contribution is additionally modulated by local texture.

    Inputs whose magnitude exceeds the uint16 range are internally
    normalized by ``max(abs(I))`` before the FFT so the float32 power
    spectrum cannot overflow; the filtered result is scaled back linearly.
    """
    k_factor = validate_real("k_factor", k_factor, minimum=1e-6)
    blend_ratio = validate_real(
        "blend_ratio", blend_ratio, minimum=0.0, maximum=1.0
    )
    delta = validate_real("delta", delta, minimum=1e-6)

    source = np.asarray(frame)
    if source.ndim != 2:
        raise ValueError(
            f"输入图像必须是二维数组，收到 {source.ndim}D (shape={source.shape})"
        )
    if source.size == 0:
        raise ValueError("输入图像不能为空")
    if not np.issubdtype(source.dtype, np.number) or np.iscomplexobj(source):
        raise ValueError(f"不支持的图像数据类型: {source.dtype}")

    raw_img = source.astype(np.float32)
    if not np.isfinite(raw_img).all():
        if np.isfinite(source).all():
            raise ValueError(
                "输入数值超出 float32 表示范围；请先按比例缩放图像强度"
            )
        raise ValueError("输入图像包含 NaN 或 Inf")
    if blend_ratio == 0.0 or min(raw_img.shape) < 2:
        return raw_img.copy()

    height, width = raw_img.shape
    spatial_mask = _texture_mask(raw_img)

    # float32 功率谱对大动态范围输入平方后会溢出为 inf。滤波是线性的、
    # 衰减掩码在同步缩放 epsilon 后近似尺度不变，因此先按 max(|I|) 归一、
    # 结束时再线性放大回去。仅当幅值超出常规 uint16 量级时启用，保证常规
    # 输入的数值路径与既有版本逐位一致。
    fft_scale = float(np.max(np.abs(raw_img)))
    if fft_scale > 65535.0:
        compute_img = raw_img / np.float32(fft_scale)
    else:
        fft_scale = 1.0
        compute_img = raw_img

    pad = max(32, int(0.05 * max(height, width)))
    target_height = next_fast_len(height + 2 * pad)
    target_width = next_fast_len(width + 2 * pad)
    pad_top = (target_height - height) // 2
    pad_bottom = target_height - height - pad_top
    pad_left = (target_width - width) // 2
    pad_right = target_width - width - pad_left
    padded = np.pad(
        compute_img,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="reflect",
    )

    shifted = fftshift(fft2(padded, workers=1))
    power = (shifted.real * shifted.real + shifted.imag * shifted.imag).astype(
        np.float32, copy=False
    )
    smoothed_power = gaussian_filter(power, sigma=delta, mode="nearest").astype(
        np.float32, copy=False
    )
    background_power = get_rotational_average(smoothed_power)
    signal_power = np.maximum(smoothed_power - background_power, 0.0)

    epsilon = np.finfo(np.float32).eps
    denominator = signal_power + k_factor * background_power + epsilon
    attenuation = (signal_power / denominator).astype(np.float32, copy=False)
    attenuation = cv2.GaussianBlur(
        attenuation, (15, 15), 5, borderType=cv2.BORDER_REFLECT_101
    )

    rows, cols = attenuation.shape
    dc_radius = max(2, min(50, int(0.015 * min(rows, cols))))
    cv2.circle(attenuation, (cols // 2, rows // 2), dc_radius, 1.0, -1)

    inverse = ifft2(ifftshift(shifted * attenuation), workers=1)
    filtered = inverse.real[
        pad_top : pad_top + height, pad_left : pad_left + width
    ].astype(np.float32, copy=False)

    weight = spatial_mask * np.float32(blend_ratio)
    blended = filtered * weight + compute_img * (1.0 - weight)
    if fft_scale != 1.0:
        blended *= np.float32(fft_scale)
    return np.ascontiguousarray(blended, dtype=np.float32)
