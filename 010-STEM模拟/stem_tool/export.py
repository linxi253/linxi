"""结果导出：PNG / 16-bit TIFF（参数写入 ImageDescription）/ NPY / 参数 JSON。

数值口径（与 09-HRTEM模拟 保持一致，便于两工具产物混用）：
  - PNG：robust_norm(0.5–99.5) 后的 8 位显示图（可加极性/模糊）
  - TIFF：robust_norm(0.05–99.95) 的 16 位图，参数摘要写入 description
  - NPY：**物理量**（float32，探测器收集比例 0–1），不做任何归一化，
    供后续定量分析（如厚度拟合、A/B 原子柱强度比）
  - JSON：完整参数 + 结果元信息
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .params import StemParams
from .render import apply_display, robust_norm
from .sim_core import StemSimResult


def tiff_description(params: StemParams, result: Optional[StemSimResult] = None,
                     detector: str = "ADF") -> str:
    t = f"{params.thickness_nm:.3f}"
    sampling = params.sampling_a
    if result is not None:
        t = f"{result.result.thickness_a / 10.0:.3f}"
        sampling = result.sampling[0]
    parts = [
        f"{params.zone_str}", f"detector={detector}", f"t={t} nm",
        f"{params.voltage_kv:g} kV", f"Cs={params.cs_mm:g} mm",
        f"df={params.defocus_nm:+g} nm",
        f"probe={params.probe_semiangle_mrad:g} mrad",
        f"detector_angles={params.detector_inner_mrad:g}-{params.detector_outer_mrad:g} mrad",
        f"phonons={params.n_phonons}@{params.temperature_k:g}K",
        f"grid_sampling={sampling:g} A/px",
        f"pixel={result.out_sampling if result else params.scan_step_a:g} A/px",
        f"slice={params.slice_thickness_a:g} A",
        f"scan={params.scan_fov_nm:g} nm/{params.scan_points}^2",
        f"polarity={params.polarity:+d}",
        f"rot={params.rotation_deg:g} deg", f"mirror={params.mirror}",
        "unit=fraction_of_incident", "engine=stem_sim",
    ]
    return "; ".join(parts)


def sanitize(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", name)


def default_basename(params: StemParams, result: Optional[StemSimResult] = None,
                     detector: Optional[str] = None) -> str:
    formula = (result.formula if result else Path(params.cif_path).stem) or "sim"
    t = (result.result.thickness_a / 10.0) if result else params.thickness_nm
    zone = f"{params.zone[0]}{params.zone[1]}{params.zone[2]}"
    det = (detector or params.display_detector or "ADF").upper()
    return sanitize(
        f"{formula}_{zone}_t{t:.2f}nm_{det}_df{params.defocus_nm:+.1f}nm"
        f"_{params.voltage_kv:g}kV"
    )


def export_tiff16(data: np.ndarray, path, description: str = "",
                  lo_pct: float = 0.05, hi_pct: float = 99.95) -> None:
    """16-bit TIFF（robust_norm 0.05–99.95，与 09-HRTEM模拟 交付口径一致）。"""
    from PIL import Image

    norm = robust_norm(np.asarray(data, float), lo_pct, hi_pct)
    Image.fromarray(np.round(norm * 65535.0).astype(np.uint16)).save(
        path, compression="tiff_lzw", description=description
    )


def export_npy(data: np.ndarray, path) -> None:
    np.save(path, np.asarray(data, dtype=np.float32))


def export_png(data: np.ndarray, path) -> None:
    from PIL import Image

    Image.fromarray(np.round(np.clip(data, 0.0, 1.0) * 255.0).astype(np.uint8)).save(path)


def export_params_json(params: StemParams, path, result: Optional[StemSimResult] = None,
                       detectors: Optional[List[str]] = None) -> None:
    data = dict(parameters=params.to_dict())
    if result is not None:
        res = result.result
        data["result"] = dict(
            formula=result.formula,
            thickness_actual_nm=res.thickness_a / 10.0,
            sampling_a_per_px=[float(v) for v in res.sampling],
            output_sampling_a_per_px=result.out_sampling,
            scan_grid=[int(v) for v in res.geometry.shape],
            scan_fov_nm=res.geometry.fov_x / 10.0,
            scan_step_a=[res.geometry.step_x, res.geometry.step_y],
            n_atoms=res.n_atoms,
            n_slices=res.n_slices,
            slice_thickness_a=res.dz,
            n_fft=res.n_fft,
            elapsed_s=round(res.elapsed_s, 2),
            detectors={d.name: [d.inner, d.outer] for d in res.detector_specs},
            detector_units="fraction of incident intensity (0-1)",
            phonons=dict(n_configs=res.n_configs, summary=res.phonon_summary,
                         sigma_u_a={k: round(v, 5) for k, v in res.sigma.items()}),
            notes=res.notes,
            engine="stem_sim (frozen-phonon multislice, Peng 1999)",
        )
        if detectors:
            data["result"]["exported_detectors"] = list(detectors)
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )


def export_all(
    result: StemSimResult,
    out_dir,
    basename: Optional[str] = None,
    kind_map: Optional[dict] = None,
    detectors: Optional[List[str]] = None,
) -> Dict[str, str]:
    """按开关导出一组文件，返回 {kind[:探测器]: path}。

    kind_map: {"tiff":bool, "png":bool, "npy":bool, "json":bool}
    导出的是**物理量**（NPY）与按统一对比度口径归一化的图像（PNG/TIFF）。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kinds = kind_map or {"tiff": True, "png": True, "npy": False, "json": True}
    names = [d.upper() for d in (detectors or [result.params.display_detector])]
    paths: Dict[str, str] = {}
    lo, hi = result.params.contrast_lo_pct, result.params.contrast_hi_pct
    for det in names:
        data = result.image(det)          # 取向渲染后的原生强度
        base = basename or default_basename(result.params, result, det)
        if len(names) > 1 and basename:
            base = f"{base}_{det}"
        if kinds.get("tiff"):
            p = out / f"{base}.tif"
            export_tiff16(data, p, tiff_description(result.params, result, det), 0.05, 99.95)
            paths[f"tiff:{det}"] = str(p)
        if kinds.get("png"):
            display = apply_display(
                data, result.out_sampling, polarity=result.params.polarity,
                blur_a=result.params.display_blur_a, lo_pct=lo, hi_pct=hi,
            )
            p = out / f"{base}.png"
            export_png(display, p)
            paths[f"png:{det}"] = str(p)
        if kinds.get("npy"):
            p = out / f"{base}.npy"
            export_npy(data, p)
            paths[f"npy:{det}"] = str(p)
    if kinds.get("json"):
        base = basename or default_basename(result.params, result, names[0])
        p = out / f"{base}_params.json"
        export_params_json(result.params, p, result, names)
        paths["json"] = str(p)
    return paths
