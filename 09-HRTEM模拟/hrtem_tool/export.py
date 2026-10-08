"""结果导出：PNG / 16-bit TIFF（参数写入 ImageDescription）/ NPY / 参数 JSON。

TIFF 描述串沿用旧交付基线的惯例：
  "Fe3O4 [110]; t=3.58 nm; df=-34.0 nm; 200 kV; Cs=0.085 mm; aperture=24 mrad; ..."
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

import tem_sim

from . import __version__ as TOOL_VERSION
from .params import SimParams
from .render import robust_norm
from .sim_core import SimResult


def tiff_description(params: SimParams, result: Optional[SimResult] = None) -> str:
    """TIFF ImageDescription 参数串（分号分隔，供下游读取完整模拟口径）。"""
    t = f"{result.thickness_actual_nm:.3f}" if result else f"{params.thickness_nm:g}"
    sampling = result.sampling if result else params.sampling_a
    out_sampling = (
        result.out_sampling if result
        else (params.output_sampling_a if params.output_sampling_a > 0 else params.sampling_a)
    )
    formula = result.formula if result else ""
    head = f"{formula} " if formula else ""
    parts = [
        f"{head}{params.zone_str}",
        f"t={t} nm",
        f"df={params.defocus_nm:+g} nm",
        f"{params.voltage_kv:g} kV",
        f"Cs={params.cs_mm:g} mm",
        f"aperture={params.aperture_mrad:g} mrad",
        f"focal_spread={params.focal_spread_a:g} A",
        f"angular_spread={params.angular_spread_mrad:g} mrad",
        f"astig={params.astigmatism_a:g} A@{params.astigmatism_azimuth_deg:g} deg",
        f"sampling={sampling:g} A/px",
        f"out_sampling={out_sampling:g} A/px",
        f"slice={params.slice_thickness_a:g} A",
        f"table={params.table}",
        f"polarity={params.polarity:+d}",
        f"blur={params.display_blur_a:g} A",
        f"contrast={params.display_lo_pct:g}-{params.display_hi_pct:g} pct (PNG)",
        "norm=0.05-99.95 pct (this TIFF)",
        f"rot={params.rotation_deg:g} deg",
        f"mirror={params.mirror}",
        f"engine=tem_sim {tem_sim.__version__}",
    ]
    return "; ".join(parts)


def sanitize(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", name)


def default_basename(params: SimParams, result: Optional[SimResult] = None) -> str:
    formula = result.formula if result else Path(params.cif_path).stem
    if not formula:
        formula = "sim"
    t = result.thickness_actual_nm if result else params.thickness_nm
    zone = f"{params.zone[0]}{params.zone[1]}{params.zone[2]}"
    return sanitize(
        f"{formula}_{zone}_t{t:.2f}nm_df{params.defocus_nm:+.1f}nm"
        f"_{params.voltage_kv:g}kV"
    )


def export_tiff16(
    oriented: np.ndarray,
    path,
    description: str = "",
    lo_pct: float = 0.05,
    hi_pct: float = 99.95,
) -> None:
    """16-bit TIFF（robust_norm 0.05–99.95，与 0820 交付一致）。"""
    from PIL import Image

    norm = robust_norm(np.asarray(oriented, float), lo_pct, hi_pct)
    # 注意：不要给 fromarray 传 mode=（Pillow 13 起弃用）；
    # uint16 二维数组自动映射为 I;16
    Image.fromarray(np.round(norm * 65535.0).astype(np.uint16)).save(
        path, compression="tiff_lzw", description=description
    )


def export_npy(oriented: np.ndarray, path) -> None:
    np.save(path, np.asarray(oriented, dtype=np.float32))


def export_params_json(params: SimParams, path, result: Optional[SimResult] = None) -> None:
    """参数 + 结果元数据 JSON（含引擎/工具版本与散射因子表，保证可复现）。"""
    data = dict(
        parameters=params.to_dict(),
        meta=dict(
            generated_at=datetime.now().isoformat(timespec="seconds"),
            tool_version=TOOL_VERSION,
            engine_version=tem_sim.__version__,
            table=params.table,
        ),
    )
    if result is not None:
        data["result"] = dict(
            formula=result.formula,
            thickness_actual_nm=result.thickness_actual_nm,
            sampling_a_per_px=result.sampling,
            output_sampling_a_per_px=result.out_sampling,
            n_atoms=result.n_atoms,
            zone_periods={k: v for k, v in result.zone_periods.items()},
            elapsed_s=round(result.elapsed_s, 2),
            engine=f"tem_sim {tem_sim.__version__} (Cowley-Moody multislice)",
        )
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )


def export_all(
    result: SimResult,
    out_dir,
    basename: Optional[str] = None,
    kind_map: Optional[dict] = None,
) -> dict:
    """按开关导出一组文件，返回 {kind: path}。

    PNG 使用 result.params 的对比度百分位（与屏幕显示一致）；
    TIFF 固定 0.05–99.95 归一（0820 交付口径），两者口径均写入元数据。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not os.access(out, os.W_OK):
        raise PermissionError(f"输出文件夹不可写：{out}")
    name = basename or default_basename(result.params, result)
    kinds = kind_map or {"tiff": True, "png": True, "npy": False, "json": True}
    paths: dict = {}
    if kinds.get("tiff"):
        p = out / f"{name}.tif"
        export_tiff16(result.oriented, p, tiff_description(result.params, result))
        paths["tiff"] = str(p)
    if kinds.get("png"):
        from PIL import Image

        from .render import apply_display

        display = apply_display(
            result.oriented, result.out_sampling,
            polarity=result.params.polarity,
            blur_a=result.params.display_blur_a,
            lo_pct=result.params.display_lo_pct,
            hi_pct=result.params.display_hi_pct,
        )
        p = out / f"{name}.png"
        Image.fromarray(np.round(display * 255.0).astype(np.uint8)).save(p)
        paths["png"] = str(p)
    if kinds.get("npy"):
        p = out / f"{name}.npy"
        export_npy(result.oriented, p)
        paths["npy"] = str(p)
    if kinds.get("json"):
        p = out / f"{name}_params.json"
        export_params_json(result.params, p, result)
        paths["json"] = str(p)
    return paths
