"""
Cowley-Moody Multislice 算法（SimulaTEM 核心计算逻辑的 Python 实现）。

算法流程（与 SimulaTEM HLP "Theoretical information" 一节对应）：

1. 沿电子束方向 (z) 把样品切成厚度为 Δz 的薄片（SimulaTEM 的 Slices 对话框）；
2. 每个薄片用高斯拟合的原子散射因子解析地计算投影势相位
   φ(x,y) = γλ · Σ_atoms Σ_i (π a_i/b_i) exp(-π²|ρ-ρ_j|²/b_i)，
   得到透射函数 t(x,y) = exp(i φ)（相位光栅）；
3. 波函数依次通过各薄片并在片间做菲涅尔传播：
   ψ → ψ·t_j → F⁻¹[ F[ψ] · exp(-iπ λ Δz k²) ]；
4. 出射波用于计算衍射花样或经 CTF 成像（见 imaging 模块）。

单位：长度 Å，相位 rad。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .microscope import Microscope
from .scattering import phase_kernels
from .structure import Structure


@dataclass
class ExitWave:
    """Multislice 出射波函数。

    Attributes
    ----------
    array : ndarray, complex
        出射波 ψ(x, y)，形状 (ny, nx)。
    sampling : (float, float)
        x/y 方向采样间隔，Å/px。
    extent : (float, float)
        x/y 方向总尺寸，Å。
    voltage_kv : float
        模拟使用的加速电压。
    """

    array: np.ndarray
    sampling: Tuple[float, float]
    extent: Tuple[float, float]
    voltage_kv: float

    @property
    def shape(self) -> Tuple[int, int]:
        return self.array.shape

    @property
    def intensity(self) -> np.ndarray:
        return np.abs(self.array) ** 2


# ----------------------------------------------------------------------
# 内部工具
# ----------------------------------------------------------------------
def _prepare_box(
    structure: Structure, padding: float
) -> Tuple[np.ndarray, List[str], float, float, float]:
    """确定模拟盒子：有晶胞则用晶胞，否则用包围盒 + 真空边界。

    Returns
    -------
    positions : ndarray (N, 3)，已平移进盒子内
    symbols   : 元素符号列表
    lx, ly, lz: 盒子尺寸（Å）
    """
    pos = np.asarray(structure.positions, dtype=float).copy()
    if structure.cell is not None:
        cell = structure.cell
        # 只支持 x/y 方向正交的盒子（带轴重构后的结构满足该条件）。
        # 注意必须检查全部非对角项：单斜晶胞（如唯一轴沿 x，c[1,2]≠0）
        # 仅查 c01/c02 会漏检，导致 z 向周期化静默出错。
        off_xy = (
            abs(cell[0, 1]) + abs(cell[1, 0])
            + abs(cell[0, 2]) + abs(cell[2, 0])
            + abs(cell[1, 2]) + abs(cell[2, 1])
        )
        if off_xy > 1e-6 * abs(cell[0, 0]):
            raise ValueError(
                "晶胞在 x/y 平面非正交；请先用 structure.zone_axis_cell "
                "或手工构造正交超胞"
            )
        lx, ly, lz = abs(cell[0, 0]), abs(cell[1, 1]), abs(cell[2, 2])
        pos[:, 0] %= lx
        pos[:, 1] %= ly
        pos[:, 2] %= lz
    else:
        lo, hi = pos.min(axis=0), pos.max(axis=0)
        pos += padding - lo  # 四周留 padding 真空
        lx, ly, lz = (hi - lo) + 2.0 * padding
    return pos, list(structure.symbols), lx, ly, lz


def _grid_size(length: float, sampling: Optional[float], gpts: Optional[int]) -> int:
    if gpts is not None:
        return int(gpts)
    if sampling is None:
        sampling = 0.05
    return max(32, int(round(length / sampling)))


def _grid_shape(
    lx: float, ly: float, sampling: Optional[float], gpts
) -> Tuple[int, int]:
    """确定网格形状 (ny, nx)。

    gpts 可为 None（按 sampling 自动）、标量（两轴相同，对应 SimulaTEM 的
    256/512/1024 档）或 (ny, nx) 二元组——STEM 扫描需要后者：x/y 周期
    （Lx、Ly）一般不相等，两轴必须各自向上取到 FFT 友好的点数，
    才能同时满足目标采样且不浪费网格。
    """
    if gpts is None:
        if sampling is None:
            sampling = 0.05
        return (max(32, int(round(ly / sampling))),
                max(32, int(round(lx / sampling))))
    try:
        ny, nx = int(gpts[0]), int(gpts[1])
    except (TypeError, IndexError):
        return int(gpts), int(gpts)
    return ny, nx


def _fresnel_propagator(
    lam: float, dz: float, nx: int, ny: int, sx: float, sy: float
) -> np.ndarray:
    """菲涅尔自由空间传播子 P(k) = exp(-i π λ Δz k²)。"""
    kx = np.fft.fftfreq(nx, d=sx)
    ky = np.fft.fftfreq(ny, d=sy)
    k2 = kx[None, :] ** 2 + ky[:, None] ** 2
    return np.exp(-1j * np.pi * lam * dz * k2)


def _band_limit_mask(
    nx: int, ny: int, sx: float, sy: float, ratio: float = 2.0 / 3.0
) -> np.ndarray:
    """反混叠带限掩模（Kirkland 惯例：限制在 ratio×奈奎斯特以内）。

    厚样品传播时高角散射会越过网格奈奎斯特频率并折叠回低频造成伪影，
    带限是标准对策。默认不启用（band_limit=False），以免改变既有结果口径。
    """
    kx = np.fft.fftfreq(nx, d=sx)
    ky = np.fft.fftfreq(ny, d=sy)
    k = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    return k <= ratio * min(1.0 / (2.0 * sx), 1.0 / (2.0 * sy))


def _slice_indices(positions: np.ndarray, lz: float, n_slices: int) -> List[np.ndarray]:
    """把原子按 z 坐标分配到各切片。"""
    z = positions[:, 2]
    idx = np.clip((z / (lz / n_slices)).astype(int), 0, n_slices - 1)
    return [np.where(idx == i)[0] for i in range(n_slices)]


def _add_wrapped(phase: np.ndarray, kern: np.ndarray, iy: int, jx: int) -> None:
    """把核按 (iy, jx) 为中心加到相位网格上，越界部分周期回绕。

    等价于 np.add.at + 取模索引，但用连续切片累加（快一个量级以上）。
    核可能大于网格（宽尾项），按 ny/nx 分块后每块至多回绕一次。
    """
    ny, nx = phase.shape
    kh, kw = kern.shape
    base_i = (iy - kh // 2) % ny
    base_j = (jx - kw // 2) % nx
    r = 0
    while r < kh:
        h = min(ny, kh - r)
        r0 = (base_i + r) % ny
        c = 0
        while c < kw:
            w = min(nx, kw - c)
            c0 = (base_j + c) % nx
            block = kern[r : r + h, c : c + w]
            r1, c1 = r0 + h, c0 + w
            ri, ci = min(r1, ny), min(c1, nx)
            phase[r0:ri, c0:ci] += block[: ri - r0, : ci - c0]
            if r1 > ny:
                phase[: r1 - ny, c0:ci] += block[h - (r1 - ny) :, : ci - c0]
            if c1 > nx:
                phase[r0:ri, : c1 - nx] += block[: ri - r0, w - (c1 - nx) :]
            if r1 > ny and c1 > nx:
                phase[: r1 - ny, : c1 - nx] += block[h - (r1 - ny) :, w - (c1 - nx) :]
            c += w
        r += h


def _accumulate_phase(
    phase: np.ndarray,
    positions: np.ndarray,
    symbols: List[str],
    atom_ids: np.ndarray,
    sx: float,
    sy: float,
    kernel_cache: dict,
) -> None:
    """把指定原子的投影势相位核累加到相位网格（周期性边界）。

    原子被放置到最近像素的中心（半像素精度），与 SimulaTEM 的
    离散采样处理方式一致。
    """
    for j in atom_ids:
        x, y = positions[j, 0], positions[j, 1]
        # 最近像素中心对应的像素索引
        jx = int(np.floor(x / sx + 0.5)) % phase.shape[1]
        iy = int(np.floor(y / sy + 0.5)) % phase.shape[0]
        for kern in kernel_cache[symbols[j]]:
            _add_wrapped(phase, kern, iy, jx)


def build_slice_transmissions(
    structure: Structure,
    scope: Microscope,
    sampling: Optional[float] = 0.05,
    gpts: Optional[int] = None,
    slice_thickness: float = 2.0,
    padding: float = 5.0,
    kernel_cutoff: float = 12.0,
    table: str = "peng",
) -> Tuple[List[np.ndarray], Tuple[float, float], Tuple[float, float, float]]:
    """构造所有薄片的透射函数 t_j(x,y) = exp(i φ_j)。

    Parameters
    ----------
    table : str
        散射因子表："peng"（默认，物理标准）或 "gauss3"（SimulaTEM legacy，
        仅用于复现旧口径）。

    Returns
    -------
    transmissions : list of complex ndarray
    sampling : (sx, sy) Å/px
    extent : (lx, ly, lz) Å
    """
    pos, symbols, lx, ly, lz = _prepare_box(structure, padding)

    ny, nx = _grid_shape(lx, ly, sampling, gpts)
    sx, sy = lx / nx, ly / ny
    if gpts is None and abs(sx - sy) > 0.05 * (0.5 * (sx + sy)):
        # auto 模式下 x/y 采样不应有显著差异（显式 gpts 时各向异性是用户选择，
        # 且 phase_kernels 已支持矩形像素精确积分）
        raise ValueError(
            f"x/y 采样差异过大（{sx:.4f} vs {sy:.4f} Å），请调整采样或 gpts"
        )

    n_slices = max(1, int(np.ceil(lz / slice_thickness)))
    dz = lz / n_slices
    slices = _slice_indices(pos, lz, n_slices)

    gamma_lam = scope.gamma_lambda
    unique_symbols = sorted(set(symbols))
    kernel_cache = {
        sym: phase_kernels(sym, (sx, sy), gamma_lam, cutoff=kernel_cutoff, table=table)
        for sym in unique_symbols
    }

    transmissions: List[np.ndarray] = []
    for atom_ids in slices:
        phase = np.zeros((ny, nx), dtype=float)
        if len(atom_ids) > 0:
            _accumulate_phase(phase, pos, symbols, atom_ids, sx, sy, kernel_cache)
        transmissions.append(np.exp(1j * phase).astype(np.complex128))
    return transmissions, (sx, sy), (lx, ly, lz)


# ----------------------------------------------------------------------
# 对外接口
# ----------------------------------------------------------------------
def multislice(
    structure: Structure,
    scope: Microscope,
    sampling: Optional[float] = 0.05,
    gpts: Optional[int] = None,
    slice_thickness: float = 2.0,
    padding: float = 5.0,
    transmissions: Optional[List[np.ndarray]] = None,
    verbose: bool = True,
    progress=None,
    table: str = "peng",
    band_limit: bool = False,
) -> ExitWave:
    """Multislice 模拟：平面波入射，返回出射波。

    Parameters
    ----------
    structure : Structure
        原子结构（PDB/XYZ/CIF 读入，或 ase 转换）。
    scope : Microscope
        显微镜参数（只用其中的电压决定波长）。
    sampling : float, 可选
        目标采样间隔 Å/px（与 gpts 二选一）。
    gpts : int, 可选
        每方向采样点数（对应 SimulaTEM 的 256/512/1024 选项）。
    slice_thickness : float
        切片厚度 Å（对应 SimulaTEM 的 slice width，其默认档为 2.0 Å）。
    padding : float
        非周期结构周围的真空边界 Å。
    transmissions : list, 可选
        预先计算的透射函数（STEM 扫描等重复利用场景）。
    verbose : bool
        是否打印进度。
    progress : callable, 可选
        progress(fraction, message) 回调（GUI 进度条）。
    table : str
        散射因子表："peng"（默认，物理标准）或 "gauss3"（SimulaTEM legacy）。
    band_limit : bool
        是否启用反混叠带限（2/3 奈奎斯特，Kirkland 惯例）。厚样品建议开启；
        默认关闭以保持既有结果口径。

    Returns
    -------
    ExitWave
    """
    if transmissions is None:
        transmissions, (sx, sy), (lx, ly, lz) = build_slice_transmissions(
            structure, scope, sampling, gpts, slice_thickness, padding, table=table
        )
        nx, ny = transmissions[0].shape[1], transmissions[0].shape[0]
    else:
        ny, nx = transmissions[0].shape
        pos, symbols, lx, ly, lz = _prepare_box(structure, padding)
        sx, sy = lx / nx, ly / ny

    lam = scope.wavelength
    n_slices = len(transmissions)
    dz = lz / n_slices
    prop = _fresnel_propagator(lam, dz, nx, ny, sx, sy)
    bl_mask = _band_limit_mask(nx, ny, sx, sy) if band_limit else None

    psi = np.ones((ny, nx), dtype=np.complex128)  # 平面波入射，振幅 1
    for i, t in enumerate(transmissions):
        psi *= t
        psi_ft = np.fft.fft2(psi)
        if bl_mask is not None:
            psi_ft *= bl_mask
        psi = np.fft.ifft2(psi_ft * prop)
        if verbose and n_slices > 4 and (i + 1) % max(1, n_slices // 8) == 0:
            print(f"  multislice: 切片 {i + 1}/{n_slices}")
        if progress is not None:
            progress((i + 1) / n_slices, f"切片 {i + 1}/{n_slices}")

    return ExitWave(array=psi, sampling=(sx, sy), extent=(lx, ly), voltage_kv=scope.voltage_kv)


def multislice_series(
    structure: Structure,
    scope: Microscope,
    thicknesses: List[float],
    sampling: Optional[float] = 0.05,
    gpts: Optional[int] = None,
    slice_thickness: float = 2.0,
    padding: float = 5.0,
    progress=None,
    table: str = "peng",
    band_limit: bool = False,
) -> List[Tuple[float, ExitWave]]:
    """厚度序列：构建一次势场、一次传播，在请求的各厚度捕获出射波。

    捕获厚度取最接近请求值的切片边界，返回值为 (实际厚度 Å, ExitWave) 列表，
    按厚度升序排列。structure 应按最大厚度构建（厚度序列无需重复计算势场）。
    table：散射因子表（"peng" 默认 / "gauss3" legacy，见 build_slice_transmissions）；
    band_limit：反混叠带限（默认关，见 multislice）。
    """
    transmissions, (sx, sy), (lx, ly, lz) = build_slice_transmissions(
        structure, scope, sampling, gpts, slice_thickness, padding, table=table
    )
    n_slices = len(transmissions)
    dz = lz / n_slices
    ny, nx = transmissions[0].shape
    lam = scope.wavelength
    prop = _fresnel_propagator(lam, dz, nx, ny, sx, sy)
    bl_mask = _band_limit_mask(nx, ny, sx, sy) if band_limit else None

    targets: dict = {}
    for t in sorted(set(float(x) for x in thicknesses)):
        i = int(round(t / dz))
        i = min(max(1, i), n_slices)
        targets.setdefault(i, i * dz)  # 实际捕获厚度

    psi = np.ones((ny, nx), dtype=np.complex128)
    captured: dict = {}
    for i, t in enumerate(transmissions):
        psi *= t
        psi_ft = np.fft.fft2(psi)
        if bl_mask is not None:
            psi_ft *= bl_mask
        psi = np.fft.ifft2(psi_ft * prop)
        if (i + 1) in targets:
            captured[i + 1] = psi.copy()
        if progress is not None:
            progress((i + 1) / n_slices, f"切片 {i + 1}/{n_slices}")

    return [
        (targets[k], ExitWave(array=captured[k], sampling=(sx, sy),
                              extent=(lx, ly), voltage_kv=scope.voltage_kv))
        for k in sorted(captured)
    ]


def projected_phase(
    structure: Structure,
    scope: Microscope,
    sampling: float = 0.05,
    gpts: Optional[int] = None,
    padding: float = 5.0,
    table: str = "peng",
) -> np.ndarray:
    """计算整个样品沿 z 投影的总相位 φ(x,y)（单片近似，用于快速预览）。

    对应 SimulaTEM 默认 "Single slice" 模式。
    """
    pos, symbols, lx, ly, lz = _prepare_box(structure, padding)
    nx = _grid_size(lx, sampling, gpts)
    ny = _grid_size(ly, sampling, gpts)
    sx, sy = lx / nx, ly / ny

    gamma_lam = scope.gamma_lambda
    kernel_cache = {
        sym: phase_kernels(sym, (sx, sy), gamma_lam, table=table)
        for sym in set(symbols)
    }
    phase = np.zeros((ny, nx), dtype=float)
    _accumulate_phase(phase, pos, symbols, np.arange(len(symbols)), sx, sy, kernel_cache)
    return phase
