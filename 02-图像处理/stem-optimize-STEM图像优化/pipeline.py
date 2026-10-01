"""Shared preview/export processing pipeline and parameter validation."""

from typing import Any

import numpy as np

from contrast import (
    apply_clahe,
    apply_clahe_uint16,
    normalize_to_uint8,
    normalize_to_uint16,
)
from filters import adaptive_spectral_filter, validate_real

DEFAULT_PARAMS = {
    "k_factor": 1.2,
    "blend_ratio": 0.5,
    "delta": 3.0,
    "clahe_clip": 1.2,
    "output_bit_depth": 16,
    "apply_clahe": False,
}


def validate_params(params: dict[str, Any]) -> dict:
    """Return a validated, immutable-by-convention parameter snapshot."""
    if not isinstance(params, dict):
        raise ValueError("处理参数必须是字典")
    unknown = sorted(set(params) - set(DEFAULT_PARAMS))
    if unknown:
        raise ValueError(f"存在未知处理参数: {', '.join(unknown)}")
    values = dict(DEFAULT_PARAMS)
    values.update(params)
    values["k_factor"] = validate_real("k_factor", values["k_factor"], minimum=1e-6)
    values["blend_ratio"] = validate_real(
        "blend_ratio", values["blend_ratio"], minimum=0.0, maximum=1.0
    )
    values["delta"] = validate_real("delta", values["delta"], minimum=1e-6)
    values["clahe_clip"] = validate_real(
        "clahe_clip", values["clahe_clip"], minimum=0.1, maximum=20.0
    )
    bit_depth = values.get("output_bit_depth")
    if bit_depth not in (8, 16):
        raise ValueError("输出位深只能是 8 或 16")
    values["output_bit_depth"] = int(bit_depth)
    if not isinstance(values.get("apply_clahe"), bool):
        raise ValueError("apply_clahe 必须是布尔值")
    return values


def process_frame(
    frame: np.ndarray,
    params: dict[str, Any],
    global_range: tuple[float, float],
) -> tuple[np.ndarray, dict]:
    """Process one frame using the exact pipeline shared by preview/export."""
    values = validate_params(params)
    vmin, vmax = (float(global_range[0]), float(global_range[1]))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("全局强度范围无效：最大值必须大于最小值")

    filtered = adaptive_spectral_filter(
        frame,
        k_factor=values["k_factor"],
        blend_ratio=values["blend_ratio"],
        delta=values["delta"],
    )
    pixel_count = int(filtered.size)
    below = int(np.count_nonzero(filtered < vmin))
    above = int(np.count_nonzero(filtered > vmax))

    if values["output_bit_depth"] == 8:
        output = normalize_to_uint8(filtered, vmin, vmax)
        if values["apply_clahe"]:
            output = apply_clahe(output, values["clahe_clip"])
    else:
        output = normalize_to_uint16(filtered, vmin, vmax)
        if values["apply_clahe"]:
            output = apply_clahe_uint16(output, values["clahe_clip"])

    return output, {
        "pixels": pixel_count,
        "clipped_low": below,
        "clipped_high": above,
    }
