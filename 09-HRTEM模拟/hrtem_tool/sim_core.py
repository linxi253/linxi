"""模拟核心管线：CIF → 带轴超胞 → Multislice → CTF 成像 → 取向渲染。

GUI 与命令行/验收脚本共用同一条路径，保证界面出图与
内部参考数据集基线结果物理一致。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from tem_sim import (
    ExitWave,
    Microscope,
    Structure,
    multislice,
    read_structure,
    zone_axis_cell,
    zone_axis_info,
)
from tem_sim.imaging import diffraction_pattern, hrtem_image

from .params import SimParams
from .render import oriented_image


class CancelledError(RuntimeError):
    """用户取消。"""


# ----------------------------------------------------------------------
# 结构缓存（ase 读 CIF 较慢，按路径+修改时间缓存；固定容量，最旧先出）
# ----------------------------------------------------------------------
_structure_cache: Dict[Tuple[str, float], Structure] = {}
_STRUCTURE_CACHE_MAX = 8


def _cache_put(key: Tuple[str, float], structure: Structure) -> None:
    _structure_cache[key] = structure
    while len(_structure_cache) > _STRUCTURE_CACHE_MAX:
        _structure_cache.pop(next(iter(_structure_cache)))


def load_structure(path: str) -> Structure:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"结构文件不存在：{p}")
    key = (str(p.resolve()), p.stat().st_mtime)
    if key not in _structure_cache:
        _cache_put(key, read_structure(p))
    return _structure_cache[key]


def parse_cif_meta(path: str) -> dict:
    """从 CIF 文本抽取展示用元数据（晶格/空间群；化学式另由结构统计）。"""
    meta = dict(cell="", spacegroup="")
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return meta
    lines = text.splitlines()
    values: Dict[str, str] = {}
    for i, line in enumerate(lines):
        token = line.strip()
        if token.startswith("_") and i + 1 < len(lines):
            key = token.split()[0].lower()
            nxt = lines[i + 1].strip()
            if not nxt.startswith("_") and nxt:
                values[key] = nxt.strip("'\"()")
        elif token.startswith("_"):
            parts = token.split(None, 1)
            if len(parts) == 2:
                values[parts[0].lower()] = parts[1].strip("'\"()")

    def num(name: str) -> Optional[float]:
        raw = values.get(name, "").split()
        if not raw:
            return None
        try:
            return float(raw[0])
        except ValueError:
            return None

    lengths = [v for v in (num("_cell_length_a"), num("_cell_length_b"), num("_cell_length_c")) if v]
    angles = [v for v in (num("_cell_angle_alpha"), num("_cell_angle_beta"), num("_cell_angle_gamma")) if v]
    if lengths:
        meta["cell"] = " × ".join(f"{v:.4f}" for v in lengths)
        if len(angles) == 3 and any(abs(a - 90.0) > 0.01 for a in angles):
            meta["cell"] += "  " + "/".join(f"{a:.2f}°" for a in angles)
    meta["spacegroup"] = values.get("_symmetry_space_group_name_h-m",
                                    values.get("_space_group_name_h-m_alt", ""))
    return meta


def effective_gpts(params: SimParams) -> Optional[int]:
    """gpts 模式返回精确像素数（直接传给引擎），auto 模式返回 None。"""
    return None if params.gpts_mode == "auto" else int(params.gpts_mode)


def effective_sampling(params: SimParams, structure: Structure) -> Optional[float]:
    """auto 模式返回目标采样；gpts 模式返回 None（引擎按像素数构网）。"""
    if params.gpts_mode == "auto":
        return params.sampling_a
    return None


# ----------------------------------------------------------------------
# 单次模拟结果
# ----------------------------------------------------------------------
@dataclass
class SimResult:
    params: SimParams
    raw: np.ndarray                      # 原生采样 HRTEM 强度
    oriented: np.ndarray                 # 取向渲染后（原生或重采样）物理强度
    exit_wave: ExitWave
    thickness_actual_nm: float
    sampling: float                      # 原生采样 Å/px
    out_sampling: float                  # oriented 的采样 Å/px
    n_atoms: int
    zone_periods: dict = field(default_factory=dict)
    elapsed_s: float = 0.0
    formula: str = "sim"

    def diffraction(self, log_boost: float = 1e5) -> np.ndarray:
        if self.exit_wave is None:
            raise ValueError(
                "该结果未保留出射波（厚度系列/批量模式），无法计算衍射花样"
            )
        cached = getattr(self, "_dp_cache", None)
        if cached is not None and cached[0] == log_boost:
            return cached[1]
        dp = diffraction_pattern(self.exit_wave)
        dp = np.log1p(dp / (dp.max() + 1e-30) * log_boost) / np.log1p(log_boost)
        self._dp_cache = (log_boost, dp)  # 衍射谱只依赖出射波，缓存避免重复 FFT
        return dp


# ----------------------------------------------------------------------
# 主管线
# ----------------------------------------------------------------------
def _progress_adapter(progress: Optional[Callable], stop: Optional[Callable]):
    def hook(frac: float, msg: str):
        if stop is not None and stop():
            raise CancelledError("用户取消")
        if progress is not None:
            progress(frac, msg)
    return hook


def build_crystal(params: SimParams, structure: Structure) -> Tuple[Structure, dict]:
    """带轴超胞构建，返回 (超胞, 信息 dict)。"""
    info = zone_axis_info(structure, params.zone)
    crystal = zone_axis_cell(
        structure,
        params.zone,
        target_xy=params.target_xy_a,
        target_thickness=params.thickness_nm * 10.0,
    )
    box = crystal.cell_lengths()
    info["actual_thickness_a"] = float(box[2])
    info["actual_fov_a"] = (float(box[0]), float(box[1]))
    info["n_atoms"] = len(crystal)
    return crystal, info


def run_simulation(
    params: SimParams,
    progress: Optional[Callable] = None,
    stop: Optional[Callable] = None,
    keep_exit_wave: bool = True,
) -> SimResult:
    """执行一次完整 HRTEM 模拟。"""
    t0 = time.perf_counter()
    params.validate()
    hook = _progress_adapter(progress, stop)

    def stage(frac, msg):
        if progress is not None:
            progress(frac, msg)
        if stop is not None and stop():
            raise CancelledError("用户取消")

    structure = load_structure(params.cif_path)
    stage(0.02, "构建带轴超胞…")
    crystal, info = build_crystal(params, structure)

    scope = params.scope()
    stage(0.05, f"Multislice：{info['n_atoms']} 原子 / {info['actual_thickness_a']:.1f} Å 厚")

    def ms_progress(frac, msg):
        hook(0.05 + 0.85 * frac, f"Multislice {msg}")

    wave = multislice(
        crystal,
        scope,
        sampling=effective_sampling(params, structure),
        gpts=effective_gpts(params),
        slice_thickness=params.slice_thickness_a,
        verbose=False,
        progress=ms_progress,
        table=params.table,
    )
    stage(0.92, "CTF 成像…")
    image = hrtem_image(wave, scope)

    stage(0.96, "取向渲染…")
    oriented = oriented_image(
        image,
        float(wave.sampling[0]),
        angle_deg=params.rotation_deg,
        mirror=params.mirror,
        output_sampling_a=params.output_sampling_a,
    )
    out_sampling = (
        params.output_sampling_a if params.output_sampling_a > 0
        else float(wave.sampling[0])
    )

    return SimResult(
        params=params,
        formula=structure.chemical_formula(),
        raw=image,
        oriented=oriented,
        exit_wave=wave if keep_exit_wave else None,
        thickness_actual_nm=info["actual_thickness_a"] / 10.0,
        sampling=float(wave.sampling[0]),
        out_sampling=out_sampling,
        n_atoms=info["n_atoms"],
        zone_periods=info,
        elapsed_s=time.perf_counter() - t0,
    )
