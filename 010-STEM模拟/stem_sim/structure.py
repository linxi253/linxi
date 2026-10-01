"""
原子结构输入与预处理（对应 SimulaTEM 的 Load structure 功能）。

支持的输入格式（与 SimulaTEM 相同）：
  - PDB（Brookhaven 格式；SimulaTEM 只读取 ATOM/HETATM、原子符号与 x/y/z 坐标）
  - XYZ（XMOL 格式：首行原子数、次行注释、其后 "元素 x y z"，坐标单位 Å）
  - CIF（需要安装 ase，作为扩展格式）
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Structure:
    """原子结构容器。

    Attributes
    ----------
    symbols : list of str
        每个原子的元素符号。
    positions : ndarray, shape (N, 3)
        原子笛卡尔坐标，单位 Å。z 轴为电子束方向。
    cell : ndarray, shape (3, 3) 或 None
        晶胞向量（按行存储）。None 表示非周期结构（自动加真空边界）。
    """

    symbols: List[str]
    positions: np.ndarray
    cell: Optional[np.ndarray] = None

    def __post_init__(self):
        self.positions = np.asarray(self.positions, dtype=float)
        if self.cell is not None:
            self.cell = np.asarray(self.cell, dtype=float)
        if len(self.symbols) != len(self.positions):
            raise ValueError("symbols 与 positions 数量不一致")

    def __len__(self) -> int:
        return len(self.symbols)

    # ------------------------------------------------------------------
    # 基本几何信息
    # ------------------------------------------------------------------
    @property
    def bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (min, max) 包围盒坐标 (Å)。"""
        return self.positions.min(axis=0), self.positions.max(axis=0)

    def cell_lengths(self) -> np.ndarray:
        if self.cell is None:
            lo, hi = self.bbox
            return hi - lo
        return np.linalg.norm(self.cell, axis=1)

    def chemical_formula(self) -> str:
        counts = {}
        for s in self.symbols:
            counts[s] = counts.get(s, 0) + 1
        return "".join(f"{k}{v}" for k, v in sorted(counts.items()))

    # ------------------------------------------------------------------
    # 变换操作
    # ------------------------------------------------------------------
    def translated(self, offset: Sequence[float]) -> "Structure":
        return Structure(list(self.symbols), self.positions + np.asarray(offset), self.cell)

    def repeated(self, n: Tuple[int, int, int]) -> "Structure":
        """沿晶胞三个方向重复（要求有 cell）。"""
        if self.cell is None:
            raise ValueError("Structure 没有晶胞，无法 repeat")
        nx, ny, nz = n
        new_pos = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    shift = i * self.cell[0] + j * self.cell[1] + k * self.cell[2]
                    new_pos.append(self.positions + shift)
        return Structure(
            list(self.symbols) * (nx * ny * nz),
            np.vstack(new_pos),
            self.cell * np.array([[nx], [ny], [nz]]),
        )

    def sorted_by_z(self) -> "Structure":
        """按 z 坐标升序排序（Multislice 切片需要）。"""
        order = np.argsort(self.positions[:, 2], kind="stable")
        return Structure(
            [self.symbols[i] for i in order],
            self.positions[order],
            self.cell,
        )


# ----------------------------------------------------------------------
# 文件读取
# ----------------------------------------------------------------------
def _normalize_symbol(token: str) -> str:
    """把 PDB/XYZ 的原子名（如 'AU', 'FE2+', '1HG'）规范化为元素符号。"""
    token = re.sub(r"[^A-Za-z]", "", token)
    if not token:
        raise ValueError("无法解析元素符号")
    return token[0].upper() + token[1:2].lower()


def read_pdb(path: str | Path) -> Structure:
    """读取 SimulaTEM 兼容的 PDB 文件。

    按 SimulaTEM HLP 的定义解析：关键字在 1-6 列（ATOM/HETATM），
    原子符号 13-16 列，X/Y/Z 坐标分别在 31-38 / 39-46 / 47-54 列。
    若固定列解析失败则回退到现代 PDB 的宽松解析。
    """
    symbols: List[str] = []
    positions: List[List[float]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            record = line[:6].strip()
            if record not in ("ATOM", "HETATM"):
                continue
            try:
                # SimulaTEM 固定列格式（1-based 列号 → 0-based 切片）
                name = line[12:16].strip()
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                # PDB 规范的元素符号列（77-78 列，0-based 76:78）优先：
                # 原子名的启发式解析会把 "1HG"（氢）误判为 Hg（汞）
                elem = line[76:78].strip() if len(line.rstrip()) >= 77 else ""
                if elem.isalpha() and len(elem) <= 2:
                    name = elem
            except (ValueError, IndexError):
                parts = line.split()
                if len(parts) < 5:
                    continue
                name = parts[2]
                x, y, z = float(parts[-3]), float(parts[-2]), float(parts[-1])
            symbols.append(_normalize_symbol(name))
            positions.append([x, y, z])
    if not symbols:
        raise ValueError(f"{path}: 未找到 ATOM/HETATM 记录")
    return Structure(symbols, np.array(positions), cell=None)


def read_xyz(path: str | Path) -> Structure:
    """读取 XMOL .xyz 文件（SimulaTEM 支持的第二种格式）。

    格式：首行原子数；次行注释；之后每行 "元素 x y z"（Å）。
    """
    symbols: List[str] = []
    positions: List[List[float]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    n_atoms = int(lines[0].split()[0])
    for line in lines[2:]:
        parts = line.replace("\t", " ").split()
        if len(parts) < 4:
            continue
        symbols.append(_normalize_symbol(parts[0]))
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(symbols) < n_atoms:
        raise ValueError(f"{path}: 声明 {n_atoms} 个原子，实际读到 {len(symbols)} 个")
    return Structure(symbols[:n_atoms], np.array(positions[:n_atoms]), cell=None)


def read_cif(path: str | Path) -> Structure:
    """读取 CIF 文件（依赖 ase）。"""
    try:
        from ase.io import read as ase_read
    except ImportError as exc:  # pragma: no cover
        raise ImportError("读取 CIF 需要安装 ase：pip install ase") from exc
    atoms = ase_read(str(path))
    return from_ase(atoms)


def from_ase(atoms) -> Structure:
    """由 ase.Atoms 对象构造 Structure（与现有 abtem/ase 工作流兼容）。"""
    cell = np.array(atoms.get_cell())
    if np.allclose(cell, 0.0):
        cell = None
    return Structure(list(atoms.get_chemical_symbols()), atoms.get_positions(), cell)


def read_structure(path: str | Path) -> Structure:
    """按扩展名自动分发读取 PDB / XYZ / CIF。"""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdb":
        return read_pdb(p)
    if suffix == ".xyz":
        return read_xyz(p)
    if suffix in (".cif", ".vasp", ".xsf"):
        return read_cif(p)
    raise ValueError(f"不支持的文件格式: {suffix}（支持 .pdb / .xyz / .cif）")


# ----------------------------------------------------------------------
# 带轴取向重构（面向立方/正交等晶系的常用带轴模拟）
# ----------------------------------------------------------------------
def _iter_lattice_vectors(cell: np.ndarray, n: int = 4):
    """枚举晶格平移矢量 u·a + v·b + w·c（u,v,w ∈ [-n, n]）。"""
    from itertools import product

    for u, v, w in product(range(-n, n + 1), repeat=3):
        if u == 0 and v == 0 and w == 0:
            continue
        yield u * cell[0] + v * cell[1] + w * cell[2]


def _shortest_parallel(cell: np.ndarray, direction: np.ndarray, n: int = 4) -> Optional[float]:
    """求沿给定方向的最短晶格平移矢量长度（周期性判定）。"""
    d = direction / np.linalg.norm(direction)
    best = None
    for t in _iter_lattice_vectors(cell, n):
        norm = np.linalg.norm(t)
        if np.linalg.norm(np.cross(t, d)) > 1e-5 * norm:
            continue  # 不平行
        if best is None or norm < best:
            best = norm
    return best


def _shortest_perpendicular(cell: np.ndarray, direction: np.ndarray, n: int = 4):
    """求垂直于给定方向的最短晶格平移矢量。"""
    d = direction / np.linalg.norm(direction)
    best = None
    for t in _iter_lattice_vectors(cell, n):
        norm = np.linalg.norm(t)
        if abs(np.dot(t, d)) > 1e-5 * norm:
            continue  # 不垂直
        if best is None or norm < np.linalg.norm(best):
            best = t
    return best


def _zone_axis_basis(
    cell: np.ndarray,
    zone: Sequence[int],
    lattice_search_n: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """求解带轴正交基 (x̂, ŷ, ẑ) 与投影原胞周期 (Lx, Ly, Lz)。

    zone_axis_cell 与 zone_axis_info 共用的晶格搜索逻辑。
    """
    h, k, l = [int(x) for x in zone]
    a = cell
    z_vec = h * a[0] + k * a[1] + l * a[2]
    if np.linalg.norm(z_vec) < 1e-9:
        raise ValueError(f"带轴 {zone} 对应的实空间方向为零矢量")
    z_hat = z_vec / np.linalg.norm(z_vec)

    # 1) 垂直于光束的最短晶格矢量 → x 方向与周期 Lx
    t_x = _shortest_perpendicular(a, z_hat, lattice_search_n)
    if t_x is None:
        raise ValueError(
            f"带轴 {zone}：找不到垂直于光束的晶格平移矢量，"
            "请增大 lattice_search_n 或手工构建取向超胞"
        )
    lx = float(np.linalg.norm(t_x))
    x_hat = t_x / lx

    # 2) y 方向 = ẑ × x̂；沿 y 的最短晶格矢量给出周期 Ly
    y_hat = np.cross(z_hat, x_hat)
    y_hat /= np.linalg.norm(y_hat)
    ly = _shortest_parallel(a, y_hat, lattice_search_n)
    if ly is None:
        raise ValueError(
            f"带轴 {zone}：y 方向 ({y_hat}) 上找不到晶格周期，"
            "该带轴可能为非公度取向，请手工构建取向超胞"
        )

    # 3) 沿光束方向的晶格周期 Lz（厚度方向的周期）
    lz = _shortest_parallel(a, z_hat, lattice_search_n)
    if lz is None:
        lz = abs(np.dot(a[0], z_hat)) + abs(np.dot(a[1], z_hat)) + abs(np.dot(a[2], z_hat))

    return x_hat, y_hat, z_hat, lx, ly, float(lz)


def zone_axis_info(
    structure: Structure,
    zone: Sequence[int],
    lattice_search_n: int = 4,
) -> dict:
    """查询带轴投影几何信息（不构建超胞，供界面预览）。

    Returns
    -------
    dict
        lx, ly, lz : 垂直/沿光束的投影周期（Å）；
        unit_cell_atoms : 投影原胞内原子数（整数，公度取向时）；
        volume_ratio : 投影原胞与原晶胞的体积比（非整数表示非公度取向）；
        thickness_grain : 沿光束的最小厚度台阶 = lz（Å）。
    """
    if structure.cell is None:
        raise ValueError("输入结构没有晶胞，无法做带轴分析")
    _, _, _, lx, ly, lz = _zone_axis_basis(structure.cell, zone, lattice_search_n)
    volume_ratio = lx * ly * lz / abs(np.linalg.det(structure.cell))
    return dict(
        zone=[int(v) for v in zone],
        lx=lx,
        ly=ly,
        lz=lz,
        volume_ratio=volume_ratio,
        unit_cell_atoms=int(round(volume_ratio * len(structure))),
        commensurate=bool(np.isclose(volume_ratio, round(volume_ratio), atol=1e-6)),
    )


def zone_axis_cell(
    structure: Structure,
    zone: Sequence[int],
    target_xy: float = 32.0,
    target_thickness: float = 32.0,
    lattice_search_n: int = 4,
) -> Structure:
    """把晶体重构为沿指定带轴的正交周期性超胞。

    思路与常见的 abtem 工作流一致：

    1. 由带轴指数 [h k l] 确定电子束方向 ẑ = h·a + k·b + l·c；
    2. 搜索晶格平移矢量，精确确定垂直于光束的 x/y 周期 Lx、Ly
       以及沿光束的周期 Lz（对立方、正交、四方等晶系的常见带轴均可精确求解）；
    3. 将原胞重复成大块，在新正交基下截取 [0,1) 分数坐标内的原子；
    4. 沿 x/y/z 重复到目标横向尺寸与厚度。

    Parameters
    ----------
    structure : Structure
        带晶胞的输入结构（如 CIF 读入的单胞）。
    zone : (h, k, l)
        带轴指数（实空间方向 h·a + k·b + l·c）。
    target_xy, target_thickness : float
        目标横向尺寸与厚度（Å）。
    lattice_search_n : int
        晶格矢量搜索范围 [-n, n]。
    """
    if structure.cell is None:
        raise ValueError("输入结构没有晶胞，无法做带轴重构")
    a = structure.cell
    x_hat, y_hat, z_hat, lx, ly, lz = _zone_axis_basis(a, zone, lattice_search_n)

    # 4) 把新基下的原胞内容截出来。
    #
    # 旧实现只沿原胞的正方向 repeat，然后把该块的中心当作新原点。
    # 对 [110] 这类含负笛卡尔分量的横向矢量，这个正向块并不能覆盖
    # 完整的新晶胞，Fe3O4 [110] 会从理论应有的 112 个原子漏到 70 个，
    # 密度和化学计量虽看似仍正确，投影势却少了 37.5%。这里改为围绕
    # 原点枚举正、负晶格平移，再在完整的平行六面体 [0,1)^3 内截取。
    basis_dir = np.array([x_hat, y_hat, z_hat])  # 正交归一基（行向量）
    new_cell_cart = np.diag([lx, ly, lz]) @ basis_dir
    inv_new_cell = np.linalg.inv(new_cell_cart)
    inv_a = np.linalg.inv(a)
    coeffs = new_cell_cart @ inv_a
    bound = int(np.ceil(np.abs(coeffs).max())) + 2

    frac_blocks = []
    symbol_blocks: List[str] = []
    tol = 1e-8
    for u in range(-bound, bound + 1):
        for v in range(-bound, bound + 1):
            for w in range(-bound, bound + 1):
                shift = u * a[0] + v * a[1] + w * a[2]
                frac_block = (structure.positions + shift) @ inv_new_cell
                # Snap numerical boundary values before using the half-open
                # interval.  Keeping exactly one side prevents duplicate atoms.
                frac_block[np.isclose(frac_block, 0.0, atol=tol)] = 0.0
                frac_block[np.isclose(frac_block, 1.0, atol=tol)] = 1.0
                inside = np.all((frac_block >= 0.0) & (frac_block < 1.0), axis=1)
                if np.any(inside):
                    frac_blocks.append(frac_block[inside])
                    symbol_blocks.extend(
                        structure.symbols[i] for i in np.where(inside)[0]
                    )

    if not frac_blocks:
        raise ValueError("带轴重构未截取到任何原子")
    frac_all = np.vstack(frac_blocks)

    # Deduplicate only numerical boundary coincidences.  Include the element in
    # the key so chemically distinct sites can never be merged accidentally.
    unique_frac = []
    sym_in: List[str] = []
    seen = set()
    for symbol, f in zip(symbol_blocks, frac_all):
        key = (symbol, *np.round(f, 10))
        if key in seen:
            continue
        seen.add(key)
        sym_in.append(symbol)
        unique_frac.append(f)
    frac = np.asarray(unique_frac, dtype=float)

    # A commensurate reconstructed cell must preserve atomic number density.
    # Fail loudly if a future change again clips a portion of the cell.
    volume_ratio = abs(np.linalg.det(new_cell_cart) / np.linalg.det(a))
    expected = int(round(volume_ratio * len(structure)))
    if not np.isclose(volume_ratio, round(volume_ratio), atol=1e-6):
        raise ValueError(
            f"带轴 {zone} 的新晶胞体积比 {volume_ratio:.8g} 不是整数，"
            "当前正交重构不适用于该非公度取向"
        )
    if len(sym_in) != expected:
        raise ValueError(
            f"带轴 {zone} 重构原子数错误：得到 {len(sym_in)}，应为 {expected}；"
            "请增大 lattice_search_n 或检查输入晶胞边界"
        )

    zone_cell = Structure(
        sym_in,
        frac @ np.diag([lx, ly, lz]),
        cell=np.diag([lx, ly, lz]),
    )

    # 5) 重复到目标尺寸
    nx = max(1, int(np.ceil(target_xy / lx)))
    ny = max(1, int(np.ceil(target_xy / ly)))
    nz = max(1, int(np.ceil(target_thickness / lz)))
    return zone_cell.repeated((nx, ny, nz))
