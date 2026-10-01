"""模拟核心管线：CIF → 带轴超胞 → 冻结声子 STEM 扫描 → 取向渲染。

GUI、命令行与系列模拟共用同一条路径，保证界面出图与脚本结果一致。

与 09-HRTEM模拟 的差异：STEM 的"预览/预估"信息量更大（网格、奈奎斯特
上角、探测器裁剪、时间预估），因此把派生量集中在 `plan()` 里，界面在
点「模拟」之前就能看到全部约束是否自洽。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from stem_sim import (
    CancelledError,
    ScanGeometry,
    StemOptics,
    StemResult,
    Structure,
    read_structure,
    stem_scan,
    zone_axis_cell,
    zone_axis_info,
)
from stem_sim._fft import fast_grid_size
from stem_sim.multislice import _prepare_box

from .params import StemParams
from .render import oriented_image


# ----------------------------------------------------------------------
# 结构缓存（ase 读 CIF 较慢，按路径 + 修改时间缓存）
# ----------------------------------------------------------------------
_structure_cache: Dict[Tuple[str, float], Structure] = {}


def load_structure(path: str) -> Structure:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"结构文件不存在：{p}")
    key = (str(p.resolve()), p.stat().st_mtime)
    if key not in _structure_cache:
        _structure_cache[key] = read_structure(p)
    return _structure_cache[key]


def parse_cif_meta(path: str) -> dict:
    """从 CIF 文本抽取展示用元数据（晶格 / 空间群）。"""
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

    lengths = [v for v in (num("_cell_length_a"), num("_cell_length_b"),
                           num("_cell_length_c")) if v]
    angles = [v for v in (num("_cell_angle_alpha"), num("_cell_angle_beta"),
                          num("_cell_angle_gamma")) if v]
    if lengths:
        meta["cell"] = " × ".join(f"{v:.4f}" for v in lengths)
        if len(angles) == 3 and any(abs(a - 90.0) > 0.01 for a in angles):
            meta["cell"] += "  " + "/".join(f"{a:.2f}°" for a in angles)
    meta["spacegroup"] = values.get(
        "_symmetry_space_group_name_h-m",
        values.get("_space_group_name_h-m_alt", ""),
    )
    return meta


# ----------------------------------------------------------------------
# 模拟计划（不做实际计算的派生量）
# ----------------------------------------------------------------------
@dataclass
class SimPlan:
    """一次 STEM 模拟的派生量：网格、约束、代价。"""

    cell_a: Tuple[float, float, float]
    n_atoms: int
    n_slices: int
    dz: float
    thickness_a: float
    grid: Tuple[int, int]
    sampling: Tuple[float, float]
    nyquist_mrad: float
    scan_step_a: float
    n_probes: int
    n_fft: int
    estimate_s: float
    sigma: Dict[str, float] = field(default_factory=dict)
    detector_angles: List[Tuple[str, float, float]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    n_workers: int = 0

    def summary_lines(self) -> List[str]:
        lx, ly, lz = self.cell_a
        sx, sy = self.sampling
        lines = [
            f"超胞: {lx:.2f} × {ly:.2f} × {lz:.2f} Å   {self.n_atoms} 原子",
            f"网格: {self.grid[1]} × {self.grid[0]}   采样 {sx:.4f} Å/px",
            f"切片: {self.n_slices} 片 × {self.dz:.3f} Å",
            f"扫描: {self.n_probes} 点，步长 {self.scan_step_a:.3f} Å",
            f"可收集半角上限: {self.nyquist_mrad:.1f} mrad",
            f"FFT 次数: {self.n_fft/1e6:.2f} M   预估耗时: {_fmt_time(self.estimate_s)}",
        ]
        return lines


def _fmt_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds/60:.1f} min"
    return f"{seconds/3600:.1f} h"


def plan(params: StemParams, progress: Optional[Callable] = None) -> SimPlan:
    """计算派生量而不做实际模拟（界面实时显示 + 时间预估）。

    用 zone_axis_info 直接解析投影周期，避免为了预览而构建大超胞。
    """
    params.validate()
    structure = load_structure(params.cif_path)
    if structure.cell is None:
        raise ValueError("结构没有晶胞，无法做带轴 STEM 模拟（请用 CIF/PDB 晶胞结构）")

    info = zone_axis_info(structure, params.zone)
    if not info["commensurate"]:
        raise ValueError(
            f"带轴 {params.zone_str} 为非公度取向（新晶胞体积比为 "
            f"{info['volume_ratio']:.4f}）。STEM 的周期扫描要求公度取向，"
            "请改用低指数带轴，或手工构建取向超胞后用 PDB/XYZ 导入。"
        )
    lx, ly, lz = float(info["lx"]), float(info["ly"]), float(info["lz"])

    n_x = max(1, int(np.ceil(params.scan_fov_a / lx)))
    n_y = max(1, int(np.ceil(params.scan_fov_a / ly)))
    n_z = max(1, int(np.ceil(params.thickness_nm * 10.0 / lz)))
    cell = (n_x * lx, n_y * ly, n_z * lz)
    n_atoms = int(info["unit_cell_atoms"]) * n_x * n_y * n_z

    optics = params.scope()
    target_sampling = params.effective_sampling()
    ny_g = fast_grid_size(int(np.ceil(cell[1] / target_sampling)))
    nx_g = fast_grid_size(int(np.ceil(cell[0] / target_sampling)))
    sampling = (cell[0] / nx_g, cell[1] / ny_g)

    dz = params.slice_thickness_a
    n_slices = max(1, int(np.ceil(cell[2] / dz)))
    dz_actual = cell[2] / n_slices

    notes: List[str] = []
    warnings: List[str] = []
    nyq = min(float(optics.nyquist_angle_mrad(sampling[0])),
              float(optics.nyquist_angle_mrad(sampling[1])))
    if params.detector_outer_mrad > nyq + 1e-9:
        warnings.append(
            f"探测器外角 {params.detector_outer_mrad:g} mrad > 可收集上限 "
            f"{nyq:.1f} mrad，多出的部分会被丢弃。"
            f"若需完整收集，采样需 ≤ {optics.required_sampling():.4f} Å/px。"
        )
    if sampling[0] > target_sampling * 1.3 or sampling[1] > target_sampling * 1.3:
        warnings.append("实际采样明显粗于目标值，请检查超胞横向尺寸与网格的匹配")
    if params.probe_semiangle_mrad > nyq:
        warnings.append(
            f"探针会聚角 {params.probe_semiangle_mrad:g} mrad 超过网格可表示范围 "
            f"{nyq:.1f} mrad，探针将被截断"
        )
    if params.n_phonons < 4:
        warnings.append(
            f"声子组态仅 {params.n_phonons} 个：ADF 图会残留格子条纹、Z 衬度不可靠，"
            "建议 ≥ 8（或仅作预览）"
        )
    if params.detector_inner_mrad < 1.5 * params.probe_semiangle_mrad:
        warnings.append(
            f"探测器内角 {params.detector_inner_mrad:g} mrad < 1.5× 探针角 "
            f"({1.5 * params.probe_semiangle_mrad:g} mrad)：会收集到较多弹性/布拉格"
            "散射，Z 衬度会减弱（HAADF 常规取内角 ≈ 2–3× 探针角）"
        )
    trans_gb = n_slices * ny_g * nx_g * 16 / 1024 ** 3
    if trans_gb > 2.4:      # 默认预算 3 GB 的 80%
        raise ValueError(
            f"透射函数将占用 {trans_gb:.1f} GB（{n_slices} 切片 × {nx_g}×{ny_g} 网格），"
            "超出可用内存预算。请减小扫描视场、增大采样间隔，或增大切片厚度。"
        )
    if max(ny_g, nx_g) > 4096:
        raise ValueError(
            f"网格将达到 {nx_g}×{ny_g}（上限 4096/轴）：扫描视场 "
            f"{params.scan_fov_nm:g} nm 配采样 {target_sampling:g} Å/px 过大。"
            "原子分辨率 HAADF 视场通常 2–5 nm——请减小扫描视场或增大采样。"
        )
    if params.scan_step_a > 1.0:
        warnings.append(
            f"扫描步长 {params.scan_step_a:.2f} Å 偏大（原子柱间距通常 2–4 Å），"
            "图像会出现明显欠采样；建议扫描点数 ≥ 视场(Å)/0.5。"
        )
    if n_atoms > 3_000_000:
        warnings.append(f"超胞原子数达 {n_atoms:,}，势场构建会很慢，建议减小视场/厚度")

    # 位移幅度（用于界面显示）
    sigma: Dict[str, float] = {}
    try:
        from stem_sim import sigma_table

        sigma, sigma_notes = sigma_table(
            structure.symbols,
            temperature=params.temperature_k,
            debye_temperature=(params.debye_temperature_k
                               if params.debye_temperature_k > 0 else None),
            override_sigma=(params.sigma_override_a
                            if params.sigma_override_a > 0 else None),
        )
        notes.extend(sigma_notes)
    except (KeyError, ValueError) as exc:
        warnings.append(f"热位移无法计算：{exc}")

    n_probes = int(params.scan_points) ** 2
    n_configs = int(params.n_phonons)
    n_fft = n_probes * n_configs * (2 * n_slices - 1)

    from stem_sim.scan import (_resolve_parallel, estimate_seconds,
                               memory_budget, plan_batches, spawn_safety)

    n_workers = _resolve_parallel(
        {"auto": None, "off": False}[params.parallel], n_configs, n_configs * params.scan_points
    )
    if n_workers:
        ok, reason = spawn_safety()
        if not ok:
            n_workers = 0
            notes.append(f"多进程不可用（{reason}），预估按串行计算")
    batch, n_workers, mem_notes = plan_batches(ny_g, nx_g, n_slices, n_workers,
                                               memory_budget(None))
    warnings.extend(mem_notes)
    est = estimate_seconds(params.geometry(), n_slices, n_configs,
                           (ny_g, nx_g), batch=batch, n_workers=n_workers)

    det_angles = [(d.name, d.inner, min(d.outer, nyq)) for d in params.detectors()]
    return SimPlan(
        cell_a=cell, n_atoms=n_atoms, n_slices=n_slices, dz=dz_actual,
        thickness_a=cell[2], grid=(ny_g, nx_g), sampling=sampling,
        nyquist_mrad=nyq, scan_step_a=params.scan_step_a, n_probes=n_probes,
        n_fft=n_fft, estimate_s=est, sigma=sigma, detector_angles=det_angles,
        notes=notes, warnings=warnings, n_workers=n_workers,
    )


# ----------------------------------------------------------------------
# 单次模拟结果
# ----------------------------------------------------------------------
@dataclass
class StemSimResult:
    params: StemParams
    result: StemResult
    oriented: Dict[str, np.ndarray]      # 探测器名 → 取向渲染后的强度
    sampling: Tuple[float, float]        # 多层法网格采样（Å/px，探针传播用）
    out_sampling: float                  # 图像像素尺寸 = 扫描步长（Å/px）
    plan: Optional[SimPlan] = None
    formula: str = "sim"
    elapsed_s: float = 0.0

    def image(self, name: Optional[str] = None) -> np.ndarray:
        key = (name or self.params.display_detector).upper()
        for k, v in self.oriented.items():
            if k.upper() == key:
                return v
        return next(iter(self.oriented.values()))

    def detector_names(self) -> List[str]:
        return list(self.oriented.keys())


# ----------------------------------------------------------------------
# 主管线
# ----------------------------------------------------------------------
def build_crystal(params: StemParams, structure: Optional[Structure] = None):
    """带轴超胞构建（横向目标尺寸取扫描视场，使网格不被浪费）。"""
    if structure is None:
        structure = load_structure(params.cif_path)
    crystal = zone_axis_cell(
        structure,
        params.zone,
        target_xy=params.scan_fov_a,
        target_thickness=params.thickness_nm * 10.0,
    )
    return crystal


def _progress_adapter(progress: Optional[Callable], stop: Optional[Callable]):
    def hook(frac: float, msg: str):
        if stop is not None and stop():
            raise CancelledError("用户取消")
        if progress is not None:
            progress(min(1.0, max(0.0, frac)), msg)
    return hook


def run_simulation(
    params: StemParams,
    progress: Optional[Callable] = None,
    stop: Optional[Callable] = None,
    plan_obj: Optional[SimPlan] = None,
) -> StemSimResult:
    """执行一次完整 STEM 模拟。"""
    t0 = time.perf_counter()
    params.validate()
    hook = _progress_adapter(progress, stop)

    structure = load_structure(params.cif_path)
    hook(0.02, "构建带轴超胞…")
    crystal = build_crystal(params, structure)

    optics = params.scope()
    sampling = params.effective_sampling()
    geometry = params.geometry()
    phonons = params.phonons()

    hook(0.05, f"STEM 扫描：{len(crystal)} 原子 / {params.scan_points}² 探针 / "
               f"{params.n_phonons} 声子组态")

    def sc_progress(frac, msg):
        hook(0.05 + 0.9 * frac, msg)

    result = stem_scan(
        crystal, optics, sampling=sampling, geometry=geometry,
        slice_thickness=params.slice_thickness_a,
        detectors=params.detectors(),
        phonons=phonons,
        table="peng",
        threads=(params.threads if params.threads > 0 else None),
        parallel={"auto": None, "off": False}[params.parallel],
        progress=sc_progress,
        stop=stop,
    )

    hook(0.96, "取向渲染…")
    # 注意：STEM 图像的像素尺寸是**扫描步长**（视场/点数），不是多层法网格的
    # 采样间隔（网格采样只在计算探针传播时用）。两者相差一个数量级，
    # 混用会让标尺、重采样与导出元数据全部错位。
    pixel_a = geometry.step_x
    oriented: Dict[str, np.ndarray] = {}
    for name, img in result.images.items():
        oriented[name] = oriented_image(
            img, pixel_a,
            angle_deg=params.rotation_deg, mirror=params.mirror,
            output_sampling_a=params.output_sampling_a,
        )
    out_sampling = (params.output_sampling_a if params.output_sampling_a > 0 else pixel_a)

    return StemSimResult(
        params=params,
        result=result,
        oriented=oriented,
        sampling=result.sampling,
        out_sampling=out_sampling,
        plan=plan_obj,
        formula=structure.chemical_formula(),
        elapsed_s=time.perf_counter() - t0,
    )


def run_thickness_series(
    params: StemParams,
    thicknesses_nm: List[float],
    progress: Optional[Callable] = None,
    stop: Optional[Callable] = None,
) -> List[StemSimResult]:
    """厚度序列：一次传播捕获多个厚度（引擎层几乎零额外开销）。

    实现方式是把每个目标厚度换算成切片序号，在传播循环的对应位置顺带
    记录探测器信号——不重复计算势场，也不重复传播。
    """
    params.validate()
    hook = _progress_adapter(progress, stop)
    structure = load_structure(params.cif_path)

    thick = [float(t) for t in thicknesses_nm]
    if not thick or any(t <= 0 for t in thick):
        raise ValueError("厚度列表必须为正值")
    max_nm = max(thick)
    p_max = params.replace(thickness_nm=max_nm)
    hook(0.02, "构建带轴超胞…")
    crystal = build_crystal(p_max, structure)

    optics = params.scope()
    geometry = params.geometry()
    capture_a = [t * 10.0 for t in thick]

    def sc_progress(frac, msg):
        hook(0.05 + 0.9 * frac, msg)

    res = stem_scan(
        crystal, optics, sampling=params.effective_sampling(), geometry=geometry,
        slice_thickness=params.slice_thickness_a,
        detectors=params.detectors(),
        phonons=params.phonons(),
        capture_thickness_a=capture_a,
        threads=(params.threads if params.threads > 0 else None),
        parallel={"auto": None, "off": False}[params.parallel],
        progress=sc_progress,
        stop=stop,
    )

    hook(0.96, "取向渲染…")
    pixel_a = geometry.step_x          # 像素尺寸 = 扫描步长（见 run_simulation）
    out: List[StemSimResult] = []
    for t_actual_a, key in res.capture_list():
        images = res.captures[key]
        oriented = {
            name: oriented_image(img, pixel_a,
                                 angle_deg=params.rotation_deg, mirror=params.mirror,
                                 output_sampling_a=params.output_sampling_a)
            for name, img in images.items()
        }
        # 每个厚度复制一份把 thickness_nm 设为实际值，避免图表标注串味
        sub = StemResult(
            images=images, captures=res.captures, capture_thickness_a=res.capture_thickness_a,
            sampling=res.sampling, extent=res.extent, geometry=res.geometry,
            thickness_a=t_actual_a, n_slices=key, dz=res.dz, n_atoms=res.n_atoms,
            n_fft=res.n_fft, elapsed_s=res.elapsed_s, detector_specs=res.detector_specs,
            optics=res.optics, notes=res.notes, sigma=res.sigma,
            phonon_summary=res.phonon_summary, n_configs=res.n_configs,
        )
        out.append(StemSimResult(
            params=params.replace(thickness_nm=t_actual_a / 10.0),
            result=sub,
            oriented=oriented,
            sampling=res.sampling,
            out_sampling=(params.output_sampling_a if params.output_sampling_a > 0
                          else pixel_a),
            formula=structure.chemical_formula(),
            elapsed_s=res.elapsed_s,
        ))
    return out


def run_defocus_series(
    params: StemParams,
    defoci_nm: List[float],
    progress: Optional[Callable] = None,
    stop: Optional[Callable] = None,
) -> List[StemSimResult]:
    """探针离焦序列（每个离焦一次完整扫描，代价按离焦个数线性增长）。"""
    params.validate()
    hook = _progress_adapter(progress, stop)
    structure = load_structure(params.cif_path)
    crystal = build_crystal(params, structure)
    list_df = [float(d) for d in defoci_nm]
    if not list_df:
        raise ValueError("离焦列表不能为空")

    results: List[StemSimResult] = []
    for i, df in enumerate(list_df):
        sub = params.replace(defocus_nm=df)

        def sc_progress(frac, msg, i=i, df=df, n=len(list_df)):
            hook((i + frac) / n * 0.95,
                 f"df = {df:+g} nm ({i + 1}/{n})：{msg}")

        res = stem_scan(
            crystal, sub.scope(), sampling=sub.effective_sampling(),
            geometry=sub.geometry(), slice_thickness=sub.slice_thickness_a,
            detectors=sub.detectors(), phonons=sub.phonons(),
            threads=(sub.threads if sub.threads > 0 else None),
            parallel={"auto": None, "off": False}[sub.parallel],
            progress=sc_progress, stop=stop,
        )
        pixel_a = sub.geometry().step_x   # 像素尺寸 = 扫描步长
        oriented = {
            name: oriented_image(img, pixel_a,
                                 angle_deg=sub.rotation_deg, mirror=sub.mirror,
                                 output_sampling_a=sub.output_sampling_a)
            for name, img in res.images.items()
        }
        results.append(StemSimResult(
            params=sub, result=res, oriented=oriented, sampling=res.sampling,
            out_sampling=(sub.output_sampling_a if sub.output_sampling_a > 0
                          else pixel_a),
            formula=structure.chemical_formula(), elapsed_s=res.elapsed_s,
        ))
    return results
