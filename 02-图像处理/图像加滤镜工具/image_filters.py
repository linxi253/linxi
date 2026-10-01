# -*- coding: utf-8 -*-
"""TIFF 图像滤镜核心。

滤镜只接受位于 ``[0, 1]`` 的浮点数组。TIFF 的数据类型、通道布局和
元数据由 :mod:`tif_io` 负责；该模块不再对任意浮点 TIFF 做隐式归一化，
从而避免默认保存时改变科学图像的数值含义。
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.ndimage import gaussian_filter


SUPPORTED_INTEGER_DTYPES = (np.uint8, np.uint16, np.uint32)

# 摄影风格系数集中管理：数值即调参档案，命名让每处用途可追溯。
ADAPTIVE_SIGMA_REF_DIM = 1000.0   # sigma 相对化的参考画面短边
ADAPTIVE_SIGMA_FLOOR = 0.5        # 避免 sigma 过小退化为无操作
GAUSSIAN_BLUR_DIVISOR = 8.0       # 滑块 100 时 sigma = 短边/8
USM_SIGMA = 1.5                   # USM/纹理/清晰度/去雾各自的模糊尺度
USM_GAIN_DIVISOR = 40.0           # 滑块 100 时锐化增量 = 2.5×细节
TEXTURE_SIGMA = 1.0
TEXTURE_DETAIL_GAIN = 0.8
TEXTURE_DEAD_ZONE = 0.01          # 细节软阈值，抑制背景噪声被纹理放大
CLARITY_SIGMA = 8.0
CLARITY_DETAIL_GAIN = 1.2
DEHAZE_SIGMA = 15.0
DEHAZE_ATMOSPHERE_FLOOR = 0.3     # 暗背景透射下限的基准光
DEHAZE_MIN_TRANSMISSION = 0.08    # 除法放大倍数上限约 12.5×
DEHAZE_ADD_HAZE_STRENGTH = 0.6    # 负值（加雾）折算系数
WHITE_BALANCE_SCALE_LIMIT = 2.0   # 通道增益钳制，防止小均值爆量程
TEMPERATURE_MAX_SHIFT = 0.15      # ±100 时 R 增/B 减的偏移量
TINT_MAX_SHIFT = 0.15             # ±100 时 G 通道偏移量
EXPOSURE_STOPS_AT_MAX = 2.0       # ±100 对应 ±2 档（4×/1/4×）
CONTRAST_POS_SCALE = 1.5          # 正向对比度斜率上限 2.5
CONTRAST_NEG_SCALE = 0.8          # 负向斜率下限 1/1.8
TONAL_GAIN_HIGHLIGHTS = 0.4       # 高光/阴影的最大权重
TONAL_GAIN_SHADOWS = 0.4
TONAL_GAIN_WHITES = 0.3           # 白色/黑色的最大权重
TONAL_GAIN_BLACKS = 0.3


def to_float(img: np.ndarray) -> np.ndarray:
    """将无符号整型图像精确映射到 ``[0, 1]``。

    ``uint32`` 使用 float64，避免 float32 精度不足和最大值回写溢出。
    已经是处理空间的浮点数组必须由调用者保证范围；这里不会按最大值
    自动归一化。
    """
    arr = np.asarray(img)
    if arr.dtype.type in SUPPORTED_INTEGER_DTYPES:
        work_dtype = np.float64 if arr.dtype == np.uint32 else np.float32
        return arr.astype(work_dtype) / float(np.iinfo(arr.dtype).max)
    if np.issubdtype(arr.dtype, np.floating):
        if not np.all(np.isfinite(arr)):
            raise ValueError('图像包含 NaN 或无穷值，无法安全应用滤镜。')
        return np.asarray(arr, dtype=np.float64 if arr.dtype == np.float64 else np.float32)
    raise ValueError(f'不支持的数据类型: {arr.dtype}；仅支持 uint8/uint16/uint32。')


def from_float(img_float: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    """将 ``[0, 1]`` 浮点数组安全转换回无符号整数数据类型。"""
    dtype = np.dtype(target_dtype)
    if dtype.type not in SUPPORTED_INTEGER_DTYPES:
        raise ValueError(f'不支持的目标数据类型: {dtype}。')
    # 乘法及取整统一用 float64，尤其避免 uint32 最大值转换为 2**32 后回绕。
    work = np.asarray(img_float, dtype=np.float64)
    work = np.clip(work, 0.0, 1.0)
    maximum = float(np.iinfo(dtype).max)
    values = np.rint(work * maximum)
    values = np.clip(values, 0.0, maximum)
    return values.astype(dtype)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """平滑阶梯函数，支持反向边界。"""
    if abs(edge1 - edge0) < 1e-12:
        return np.zeros_like(x)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """高斯模糊，不跨颜色通道混合；多余通道（如 alpha）原样保留。"""
    if sigma <= 0:
        return img.copy()
    if img.ndim == 3:
        if img.shape[2] > 3:
            rgb = gaussian_filter(img[:, :, :3], sigma=[sigma, sigma, 0], mode='reflect')
            return np.concatenate([rgb, img[:, :, 3:]], axis=2)
        return gaussian_filter(img, sigma=[sigma, sigma, 0], mode='reflect')
    return gaussian_filter(img, sigma=sigma, mode='reflect')


def _adaptive_sigma(img: np.ndarray, base: float) -> float:
    """按画面比例缩放 sigma，使预览和全分辨率效果一致。"""
    min_dim = min(img.shape[0], img.shape[1])
    return max(ADAPTIVE_SIGMA_FLOOR, base * min_dim / ADAPTIVE_SIGMA_REF_DIM)


def gaussian_blur(img: np.ndarray, value: float) -> np.ndarray:
    if value <= 0:
        return img
    # 与其他局部滤镜一样按画面比例缩放，避免大图预览/导出不一致。
    return _blur(img, _adaptive_sigma(img, value / GAUSSIAN_BLUR_DIVISOR))


def usm_sharpen(img: np.ndarray, value: float) -> np.ndarray:
    if value <= 0:
        return img
    blurred = _blur(img, _adaptive_sigma(img, USM_SIGMA))
    return img + value / USM_GAIN_DIVISOR * (img - blurred)


def _is_rgb(img: np.ndarray) -> bool:
    return img.ndim == 3 and img.shape[2] >= 3


def white_balance(img: np.ndarray, value: float) -> np.ndarray:
    if value <= 0 or not _is_rgb(img):
        return img
    strength = value / 100.0
    avg = img[:, :, :3].mean(axis=(0, 1))
    if np.any(avg < 1e-6):
        return img
    scale = np.clip(avg.mean() / avg, 1.0 / WHITE_BALANCE_SCALE_LIMIT, WHITE_BALANCE_SCALE_LIMIT)
    result = img.copy()
    corrected = result[:, :, :3] * scale.reshape(1, 1, 3)
    result[:, :, :3] = result[:, :, :3] * (1.0 - strength) + corrected * strength
    return result


def temperature(img: np.ndarray, value: float) -> np.ndarray:
    if value == 0 or not _is_rgb(img):
        return img
    result = img.copy()
    shift = value / 100.0 * TEMPERATURE_MAX_SHIFT
    result[:, :, 0] += shift
    result[:, :, 2] -= shift
    return result


def tint(img: np.ndarray, value: float) -> np.ndarray:
    if value == 0 or not _is_rgb(img):
        return img
    result = img.copy()
    result[:, :, 1] -= value / 100.0 * TINT_MAX_SHIFT
    return result


def exposure(img: np.ndarray, value: float) -> np.ndarray:
    if value == 0:
        return img
    return img * (2.0 ** (value / 100.0 * EXPOSURE_STOPS_AT_MAX))


def contrast(img: np.ndarray, value: float) -> np.ndarray:
    if value == 0:
        return img
    v = value / 100.0
    factor = (1.0 + v * CONTRAST_POS_SCALE if v >= 0
              else 1.0 / (1.0 - v * CONTRAST_NEG_SCALE))
    return (img - 0.5) * factor + 0.5


def highlights(img: np.ndarray, value: float) -> np.ndarray:
    if value == 0:
        return img
    return img + value / 100.0 * TONAL_GAIN_HIGHLIGHTS * _smoothstep(0.5, 0.95, img)


def shadows(img: np.ndarray, value: float) -> np.ndarray:
    if value == 0:
        return img
    return img + value / 100.0 * TONAL_GAIN_SHADOWS * _smoothstep(0.5, 0.05, img)


def whites(img: np.ndarray, value: float) -> np.ndarray:
    if value == 0:
        return img
    return img + value / 100.0 * TONAL_GAIN_WHITES * _smoothstep(0.7, 1.0, img)


def blacks(img: np.ndarray, value: float) -> np.ndarray:
    if value == 0:
        return img
    return img + value / 100.0 * TONAL_GAIN_BLACKS * _smoothstep(0.3, 0.0, img)


def texture(img: np.ndarray, value: float) -> np.ndarray:
    if value == 0:
        return img
    detail = img - _blur(img, _adaptive_sigma(img, TEXTURE_SIGMA))
    detail = np.sign(detail) * np.maximum(np.abs(detail) - TEXTURE_DEAD_ZONE, 0.0)
    return img + value / 100.0 * TEXTURE_DETAIL_GAIN * detail


def clarity(img: np.ndarray, value: float) -> np.ndarray:
    if value == 0:
        return img
    detail = img - _blur(img, _adaptive_sigma(img, CLARITY_SIGMA))
    weight = np.clip(1.0 - np.abs(img - 0.5) * 2.0, 0.0, 1.0)
    return img + value / 100.0 * CLARITY_DETAIL_GAIN * detail * weight


def dehaze(img: np.ndarray, value: float) -> np.ndarray:
    if value == 0:
        return img
    amount = value / 100.0
    dark = img if img.ndim == 2 else img[:, :, :3].min(axis=2)
    dark_smooth = gaussian_filter(dark, sigma=_adaptive_sigma(img, DEHAZE_SIGMA), mode='reflect')
    atmosphere = max(float(np.percentile(dark_smooth, 99.5)), DEHAZE_ATMOSPHERE_FLOOR)
    if amount > 0:
        transmission = np.clip(1.0 - amount * dark_smooth / atmosphere,
                               DEHAZE_MIN_TRANSMISSION, 1.0)
        if img.ndim == 3:
            transmission = transmission[:, :, np.newaxis]
        return (img - (1.0 - transmission) * atmosphere) / transmission
    haze_strength = -amount * DEHAZE_ADD_HAZE_STRENGTH
    haze_color: np.ndarray | float = (
        np.ones((1, 1, 3), dtype=img.dtype) if img.ndim == 3 else 1.0
    )
    return img * (1.0 - haze_strength) + haze_color * haze_strength


def vibrance(img: np.ndarray, value: float) -> np.ndarray:
    if value == 0 or not _is_rgb(img):
        return img
    rgb = img[:, :, :3]
    maxc = rgb.max(axis=2, keepdims=True)
    minc = rgb.min(axis=2, keepdims=True)
    saturation = np.zeros_like(maxc)
    np.divide(maxc - minc, maxc, out=saturation, where=maxc > 1e-6)
    gray = rgb.mean(axis=2, keepdims=True)
    result = img.copy()
    result[:, :, :3] = gray + (rgb - gray) * (1.0 + value / 100.0 * (1.0 - saturation))
    return result


def saturation(img: np.ndarray, value: float) -> np.ndarray:
    if value == 0 or not _is_rgb(img):
        return img
    rgb = img[:, :, :3]
    gray = rgb.mean(axis=2, keepdims=True)
    result = img.copy()
    result[:, :, :3] = gray + (rgb - gray) * (1.0 + value / 100.0)
    return result


# (key, 中文名, 英文名, min, max, 默认值, 步长)
FILTER_PARAMS = [
    ('gaussian_blur', '高斯模糊',   'Gaussian Blur',   0,    100, 0, 1),
    ('usm',           'USM锐化',    'Unsharp Mask',    0,    100, 0, 1),
    ('white_balance', '白平衡',     'White Balance',   0,    100, 0, 1),
    ('temperature',   '色温',       'Temperature',     -100, 100, 0, 1),
    ('tint',          '色调',       'Tint',            -100, 100, 0, 1),
    ('exposure',      '曝光',       'Exposure',        -100, 100, 0, 1),
    ('contrast',      '对比度',     'Contrast',        -100, 100, 0, 1),
    ('highlights',    '高光',       'Highlights',      -100, 100, 0, 1),
    ('shadows',       '阴影',       'Shadows',         -100, 100, 0, 1),
    ('whites',        '白色',       'Whites',          -100, 100, 0, 1),
    ('blacks',        '黑色',       'Blacks',          -100, 100, 0, 1),
    ('texture',       '纹理',       'Texture',         -100, 100, 0, 1),
    ('clarity',       '清晰度',     'Clarity',         -100, 100, 0, 1),
    ('dehaze',        '去除薄雾',   'Dehaze',          -100, 100, 0, 1),
    ('vibrance',      '自然饱和度', 'Vibrance',        -100, 100, 0, 1),
    ('saturation',    '饱和度',     'Saturation',      -100, 100, 0, 1),
]

FILTER_FUNCS = {
    'gaussian_blur': gaussian_blur,
    'usm': usm_sharpen,
    'white_balance': white_balance,
    'temperature': temperature,
    'tint': tint,
    'exposure': exposure,
    'contrast': contrast,
    'highlights': highlights,
    'shadows': shadows,
    'whites': whites,
    'blacks': blacks,
    'texture': texture,
    'clarity': clarity,
    'dehaze': dehaze,
    'vibrance': vibrance,
    'saturation': saturation,
}

# 滤镜执行顺序独立于 UI 排列，锐化固定在最后。
PROCESS_ORDER = (
    'white_balance', 'temperature', 'tint',
    'exposure', 'contrast', 'highlights', 'shadows', 'whites', 'blacks',
    'texture', 'clarity', 'dehaze', 'vibrance', 'saturation',
    'gaussian_blur', 'usm',
)

_PARAM_BOUNDS = {key: (minimum, maximum, default)
                 for key, _, _, minimum, maximum, default, _ in FILTER_PARAMS}

# 三处清单必须严格一致：漏掉 PROCESS_ORDER 的滤镜会被静默跳过，
# 漏掉注册表的滤镜无法被调用。import 期显式失败，杜绝人工同步遗漏。
_PARAM_KEYS = set(_PARAM_BOUNDS)
if set(FILTER_FUNCS) != _PARAM_KEYS or set(PROCESS_ORDER) != _PARAM_KEYS:
    raise RuntimeError(
        '滤镜定义不一致，键集合必须完全相同；'
        f'仅 FILTER_PARAMS 有: {sorted(_PARAM_KEYS - set(FILTER_FUNCS))}，'
        f'仅 FILTER_FUNCS 有: {sorted(set(FILTER_FUNCS) - _PARAM_KEYS)}，'
        f'仅 PROCESS_ORDER 有: {sorted(set(PROCESS_ORDER) - _PARAM_KEYS)}。'
    )
if len(PROCESS_ORDER) != len(_PARAM_KEYS):
    raise RuntimeError('PROCESS_ORDER 存在重复项。')


def default_params() -> dict[str, int]:
    return {key: default for key, _, _, _, _, default, _ in FILTER_PARAMS}


def validate_params(params: Mapping[str, float] | None) -> dict[str, float]:
    """验证参数名称、有限性和范围；未知参数不能被静默忽略。"""
    supplied = {} if params is None else dict(params)
    unknown = set(supplied) - set(_PARAM_BOUNDS)
    if unknown:
        raise ValueError(f'未知滤镜参数: {", ".join(sorted(unknown))}')
    validated: dict[str, float] = {}
    for key, (minimum, maximum, default) in _PARAM_BOUNDS.items():
        value = supplied.get(key, default)
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{key} 不是有效数值。') from exc
        if not np.isfinite(value):
            raise ValueError(f'{key} 必须是有限数值。')
        if not minimum <= value <= maximum:
            raise ValueError(f'{key} 超出允许范围 [{minimum}, {maximum}]。')
        validated[key] = value
    return validated


def is_default_params(params: Mapping[str, float] | None) -> bool:
    return all(value == _PARAM_BOUNDS[key][2]
               for key, value in validate_params(params).items())


def process_image(img_float: np.ndarray, params: Mapping[str, float] | None) -> np.ndarray:
    """按固定顺序应用滤镜并输出有限的 ``[0, 1]`` 浮点数组。

    超过 3 个通道时只对前 3 个通道应用滤镜，其余通道（如 alpha）原样保留。
    仅接受浮点数组：整型输入必须先经 :func:`to_float` 转换，否则直接报错，
    避免把原始数值误当作 ``[0, 1]`` 处理而静默破坏数据。
    """
    result = np.asarray(img_float)
    if result.ndim not in (2, 3):
        raise ValueError(f'滤镜仅支持二维灰度或三维彩色帧，实际 shape={result.shape}')
    if not np.all(np.isfinite(result)):
        raise ValueError('滤镜输入包含 NaN 或无穷值。')
    if not np.issubdtype(result.dtype, np.floating):
        raise ValueError(
            f'滤镜仅接受 [0,1] 浮点数组，实际 dtype={result.dtype}；'
            '请先使用 to_float() 转换。'
        )
    if result.dtype not in (np.float32, np.float64):
        result = result.astype(np.float32)
    values = validate_params(params)
    if result.ndim == 3 and result.shape[2] > 3:
        core = _apply_filters(result[:, :, :3].copy(), values)
        return np.concatenate([core, result[:, :, 3:]], axis=2)
    return _apply_filters(result, values)


def _apply_filters(result: np.ndarray, values: Mapping[str, float]) -> np.ndarray:
    for key in PROCESS_ORDER:
        result = FILTER_FUNCS[key](result, values[key])
    return np.clip(result, 0.0, 1.0)
