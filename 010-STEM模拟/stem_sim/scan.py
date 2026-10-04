"""STEM 扫描核心：批量探针多层法 + 冻结声子 + 多探测器同步输出。

算法（对每个探针位置、每个声子组态）：

    1. 探针 ψ₀(r) = F⁻¹[A(k)·exp(−iχ(k))]·e^{−2πi k·r_p}，Σ|ψ₀|² = 1
    2. 逐片：ψ ← ψ·t_j（乘透射函数）；ψ ← F⁻¹[F[ψ]·P(Δz)]（菲涅尔传播）
    3. 出射波倒空间强度 |F[ψ_exit]|² 与探测器掩模内积 → 该点的信号
    4. 对所有声子组态的**强度**求平均（不是对波函数求平均）

工程上的三个关键点：

* **批量探针**：一次 FFT 同时传播 B 个探针位置（三维数组、对末两轴变换），
  配合 pyfftw 多线程，比逐探针循环快近一个量级。
* **透射函数跨探针复用**：t_j(x,y) 只与结构和声子组态有关，与探针位置无关；
  一次构建后所有探针共用。相位累加（原子核卷积）因此只占总耗时的很小部分。
* **厚度序列免费**：循环中每一片之后都要做一次正变换，故在这些位置顺带累加
  探测器信号即可得到任意切片边界的厚度序列，无需重复传播。

并行策略：任务按 (声子组态, 图像行块) 切分，交给进程池。填充因子独立，
故每个子进程各自构建自己组态的透射函数（相位累加很快，且完全并行），
FFT 线程数按 `threads` 控制。实测（24 物理核）3 进程 × 8 线程相对
单进程 × 24 线程仍有约 2.4× 吞吐收益。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._fft import fast_grid_size, fft2, ifft2, get_threads, set_threads
from .detectors import RingDetector, make_detectors
from .multislice import _fresnel_propagator, build_slice_transmissions
from .phonons import FrozenPhonons, PhononConfig
from .probe import StemOptics, frequency_grid, probe_ft
from .structure import Structure

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



class CancelledError(RuntimeError):
    """用户取消。"""


# 内存预算 ---------------------------------------------------------------
# 每个批次在 _scan_positions 内同时在世的数组：相位(float64) + 相位斜坡、
# 探针乘积、psi、psi_k、传播乘积(各 complex128) → 约 80 B/像素/批元素。
# 透射函数本身另有 n_slices × ny × nx × 16 B，且**每个子进程各持一份**
# （Windows spawn 无共享内存），故并行时必须按进程数分摊预算，
# 否则 4 个子进程会各自按整机预算取批，很快耗尽内存。
BYTES_PER_BATCH_PIXEL = 80
DEFAULT_MEMORY_BUDGET = 2 * 1024 ** 3
# 单轴网格上限：4096² 的 complex128 数组已是 268 MB，再大不实用
MAX_GRID = 4096


def available_memory() -> Optional[int]:
    """可用物理内存（字节）；取不到时返回 None。"""
    try:  # Windows
        import ctypes

        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullAvailPhys)
    except Exception:  # noqa: BLE001
        pass
    try:  # POSIX
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except Exception:  # noqa: BLE001
        return None


def memory_budget(explicit: Optional[int] = None) -> int:
    """本次模拟允许使用的内存上限（字节）。

    优先级：显式传参 > 环境变量 STEM_MEMORY_MB > 可用内存的 40%
    （上限 3 GB，下限 512 MB）。留出余量给父进程的晶体、结果与 Python 本身。
    """
    if explicit and explicit > 0:
        return int(explicit)
    env = os.environ.get("STEM_MEMORY_MB")
    if env:
        try:
            return max(256, int(float(env))) * 1024 * 1024
        except ValueError:
            pass
    avail = available_memory()
    if avail is None:
        return DEFAULT_MEMORY_BUDGET
    return int(min(max(avail * 0.4, 512 * 1024 ** 2), 3 * 1024 ** 3))


def plan_batches(
    ny: int,
    nx: int,
    n_slices: int,
    n_workers: int,
    budget_bytes: int,
) -> Tuple[int, int, List[str]]:
    """按进程数与内存预算定出批大小与最终进程数。

    Returns
    -------
    (batch, n_workers, notes)
    """
    notes: List[str] = []
    per_px = ny * nx * 16
    trans_bytes = max(1, n_slices * per_px)

    if n_workers > 1:
        # 透射函数在每个子进程中各有一份；给它们留出不超过 55% 的预算
        cap = max(1, int(budget_bytes * 0.55 / trans_bytes))
        if cap < n_workers:
            notes.append(
                f"内存预算 {budget_bytes / 1024 ** 3:.1f} GB 只够 {cap} 个进程各持一份"
                f"透射函数（{trans_bytes / 1024 ** 2:.0f} MB/进程），"
                f"进程数由 {n_workers} 降为 {cap}"
            )
            n_workers = cap
        if n_workers <= 1:
            n_workers = 0

    share = budget_bytes if n_workers <= 1 else budget_bytes / n_workers
    room = max(share - (trans_bytes if n_workers > 1 else 0), 2 * per_px)
    batch = int(max(1, min(8192, room / (ny * nx * BYTES_PER_BATCH_PIXEL))))
    return batch, n_workers, notes


# ----------------------------------------------------------------------
# 扫描几何
# ----------------------------------------------------------------------
@dataclass
class ScanGeometry:
    """扫描区域与采样点数。

    fov_x/fov_y 为扫描视场（Å，正方形时两者相等），nx/ny 为两个方向的
    扫描点数，步长 = 视场/点数。扫描区域在超胞内**居中**放置，因此超胞
    可以比扫描视场大（留出周期性相邻晶体的贡献）。
    """

    fov_x: float
    fov_y: float
    nx: int
    ny: int

    def validate(self) -> None:
        if self.fov_x <= 0 or self.fov_y <= 0:
            raise ValueError("扫描视场需 > 0")
        if int(self.nx) < 2 or int(self.ny) < 2:
            raise ValueError("扫描点数每边需 ≥ 2")
        if int(self.nx) * int(self.ny) > 4_000_000:
            raise ValueError("扫描点数过多（上限约 400 万点）")

    @property
    def step_x(self) -> float:
        return self.fov_x / self.nx

    @property
    def step_y(self) -> float:
        return self.fov_y / self.ny

    @property
    def shape(self) -> Tuple[int, int]:
        return (int(self.ny), int(self.nx))

    def positions(self, cell_lx: float, cell_ly: float) -> np.ndarray:
        """探针位置数组 (ny·nx, 2)，行优先（与图像的 (ny, nx) 对应）。"""
        if self.fov_x > cell_lx + 1e-6 or self.fov_y > cell_ly + 1e-6:
            raise ValueError(
                f"扫描视场 ({self.fov_x:.2f}, {self.fov_y:.2f}) Å 超出超胞横向尺寸 "
                f"({cell_lx:.2f}, {cell_ly:.2f}) Å，请减小视场或增大超胞"
            )
        x0 = 0.5 * (cell_lx - self.fov_x)
        y0 = 0.5 * (cell_ly - self.fov_y)
        xs = x0 + (np.arange(self.nx) + 0.5) * self.step_x
        ys = y0 + (np.arange(self.ny) + 0.5) * self.step_y
        XX, YY = np.meshgrid(xs, ys)
        return np.stack([XX.ravel(), YY.ravel()], axis=1)

    def row_positions(self, cell_lx: float, cell_ly: float,
                      row_lo: int, row_hi: int) -> np.ndarray:
        """仅取 [row_lo, row_hi) 行探针位置（并行任务按行切分）。"""
        if self.fov_x > cell_lx + 1e-6 or self.fov_y > cell_ly + 1e-6:
            raise ValueError("扫描视场超出超胞横向尺寸")
        x0 = 0.5 * (cell_lx - self.fov_x)
        y0 = 0.5 * (cell_ly - self.fov_y)
        xs = x0 + (np.arange(self.nx) + 0.5) * self.step_x
        ys = y0 + (np.arange(row_lo, row_hi) + 0.5) * self.step_y
        XX, YY = np.meshgrid(xs, ys)
        return np.stack([XX.ravel(), YY.ravel()], axis=1)


# ----------------------------------------------------------------------
# 结果容器
# ----------------------------------------------------------------------
@dataclass
class StemResult:
    """一次 STEM 扫描的全部输出。"""

    images: Dict[str, np.ndarray]                 # 全厚度图像：探测器名 → (ny, nx)
    captures: Dict[int, Dict[str, np.ndarray]]    # 切片序号 → 探测器名 → 图像
    capture_thickness_a: Dict[int, float]         # 切片序号 → 实际厚度 Å
    sampling: Tuple[float, float]                 # (sx, sy) Å/px
    extent: Tuple[float, float]                   # 超胞横向尺寸 Å
    geometry: ScanGeometry
    thickness_a: float                            # 超胞总厚度 Å
    n_slices: int
    dz: float
    n_atoms: int
    n_fft: int
    elapsed_s: float
    detector_specs: List[RingDetector] = field(default_factory=list)
    optics: Optional[StemOptics] = None
    notes: List[str] = field(default_factory=list)
    sigma: Dict[str, float] = field(default_factory=dict)
    phonon_summary: str = ""
    n_configs: int = 1
    fft_rate: float = 0.0          # 实测变换速率（FFT/s），用于核对时间预估

    @property
    def thickness_nm(self) -> float:
        return self.thickness_a / 10.0

    def image(self, name: str) -> np.ndarray:
        key = name.upper()
        for k in self.images:
            if k.upper() == key:
                return self.images[k]
        raise KeyError(f"没有名为 {name} 的探测器输出（已有 {list(self.images)}）")

    def detector_names(self) -> List[str]:
        return list(self.images.keys())

    def capture_list(self) -> List[Tuple[float, int]]:
        """(实际厚度 Å, 切片序号) 列表，按厚度升序。"""
        return sorted((self.capture_thickness_a[i], i) for i in self.captures)


# ----------------------------------------------------------------------
# 单块扫描（一个声子组态 × 一段图像行）
# ----------------------------------------------------------------------
def _batch_size(shape: Tuple[int, int], bytes_budget: int) -> int:
    ny, nx = shape
    per = max(ny * nx * 16, 1)  # complex128
    return int(max(1, min(8192, bytes_budget // per)))


def _scan_positions(
    transmissions: List[np.ndarray],
    sampling: Tuple[float, float],
    P0: np.ndarray,
    KX: np.ndarray,
    KY: np.ndarray,
    prop: np.ndarray,
    positions: np.ndarray,
    masks: Dict[str, np.ndarray],
    capture_slices: Sequence[int],
    threads: Optional[int],
    batch: Optional[int] = None,
    stop: Optional[Callable[[], bool]] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """在一块探针位置上跑多层法，返回 {切片序号: {探测器名: 一维信号}}。

    batch = 一次同时传播的探针数（内存与 FFT 效率的折中，由 plan_batches 定）。
    """
    n_pos = len(positions)
    n_sl = len(transmissions)
    caps = [i for i in sorted(set(int(c) for c in capture_slices)) if 1 <= i <= n_sl]
    if n_sl not in caps:
        caps.append(n_sl)

    out: Dict[int, Dict[str, np.ndarray]] = {
        i: {name: np.zeros(n_pos, dtype=float) for name in masks} for i in caps
    }
    cap_set = set(caps)

    batch = max(1, int(batch)) if batch else _batch_size(transmissions[0].shape,
                                                         DEFAULT_MEMORY_BUDGET)
    twopi = 2.0 * np.pi
    for start in range(0, n_pos, batch):
        if stop is not None and stop():
            raise CancelledError("用户取消")
        end = min(start + batch, n_pos)
        pos = positions[start:end]
        # 探针平移 = 倒空间相位斜坡 exp(-2πi k·r_p)（自动满足周期边界）
        phase = KX[None, :, :] * pos[:, 0, None, None]
        phase += KY[None, :, :] * pos[:, 1, None, None]
        phase *= -twopi
        ramp = np.exp(1j * phase)
        del phase
        psi = ifft2(P0[None, :, :] * ramp, threads=threads)
        del ramp

        for i, t in enumerate(transmissions):
            psi *= t
            psi_k = fft2(psi, threads=threads)
            if (i + 1) in cap_set:
                inten = np.abs(psi_k) ** 2
                norm = float(psi_k.shape[-1] * psi_k.shape[-2])
                for name, mask in masks.items():
                    out[i + 1][name][start:end] = np.einsum(
                        "...ij,ij->...", inten, mask
                    ) / norm
                del inten
            if i + 1 < n_sl:
                psi = ifft2(psi_k * prop, threads=threads)
            del psi_k
    return out


# ----------------------------------------------------------------------
# 多进程任务（模块级函数，Windows spawn 需要可导入）
# ----------------------------------------------------------------------
_WORKER: Dict[str, object] = {}


def _worker_init(threads: int, run_key: str, payload: dict) -> None:
    set_threads(threads)
    _WORKER.clear()
    _WORKER["run_key"] = run_key
    _WORKER["payload"] = payload
    _WORKER["trans"] = {}
    _WORKER["config"] = None


def _worker_transmissions(config_index: int):
    """按需为声子组态构建透射函数（每个子进程缓存最近一个组态）。"""
    if _WORKER["config"] == config_index:
        return _WORKER["trans"]
    payload = _WORKER["payload"]
    structure = payload["structure"]
    if payload["phonons"] is not None:
        structure = payload["phonons"][config_index]
    trans, sampling, extent = build_slice_transmissions(
        structure,
        payload["optics"],
        sampling=payload["sampling"],
        gpts=payload["gpts"],
        slice_thickness=payload["slice_thickness"],
        padding=0.0,
        table=payload["table"],
    )
    _WORKER["trans"] = trans
    _WORKER["sampling_xy"] = sampling
    _WORKER["extent"] = extent
    _WORKER["config"] = config_index
    return trans


def _worker_task(task: Tuple[int, int, int]):
    """任务 = (声子组态序号, 图像行起, 图像行止)。"""
    config_index, row_lo, row_hi = task
    payload = _WORKER["payload"]
    trans = _worker_transmissions(config_index)
    sampling = _WORKER["sampling_xy"]
    geom: ScanGeometry = payload["geometry"]
    positions = geom.row_positions(payload["cell_lx"], payload["cell_ly"], row_lo, row_hi)
    res = _scan_positions(
        trans,
        sampling,
        payload["P0"],
        payload["KX"],
        payload["KY"],
        payload["prop"],
        positions,
        payload["masks"],
        payload["capture_slices"],
        threads=None,  # 已由 _worker_init 设定进程级线程数
        batch=payload["batch"],
    )
    return config_index, row_lo, row_hi, res


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def _main_module_guarded() -> bool:
    """判断主模块是否带 ``if __name__ == "__main__"`` 守卫。

    Windows 上多进程只能用 spawn，子进程会以 ``__mp_main__`` 之名**重新执行
    主模块的顶层代码**。若主模块没有守卫，子进程会再次走到建池那一步，
    导致重复模拟甚至进程爆炸。无法可靠地在运行期探测"顶层是否已执行完"，
    故用源码里的守卫语句作为判据；取不到源码（交互式、``python -c``、
    冻结的 exe）时保守判为不安全，退回串行。
    """
    import inspect
    import re
    import sys as _sys

    mod = _sys.modules.get("__main__")
    if mod is None:
        return False
    try:
        src = inspect.getsource(mod)
    except (OSError, TypeError):
        return False
    return bool(re.search(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]', src))


def spawn_safety() -> Tuple[bool, str]:
    """返回 (是否可安全建池, 原因)。"""
    if os.environ.get("STEM_NO_MP"):
        return False, "环境变量 STEM_NO_MP 已设置"
    try:
        import multiprocessing as mp
    except Exception as exc:  # noqa: BLE001
        return False, f"multiprocessing 不可用（{exc}）"
    if mp.parent_process() is not None:
        return False, "当前进程已是 spawn 子进程（嵌套建池会让子进程重跑主模块）"
    if not _main_module_guarded():
        return False, '主模块缺少 if __name__ == "__main__" 守卫'
    return True, ""


def _resolve_parallel(parallel: Optional[bool], n_configs: int, n_tasks: int) -> int:
    """决定工作进程数；0 表示不并行。"""
    cpus = os.cpu_count() or 1
    if parallel is False or cpus <= 2 or n_tasks <= 1:
        return 0
    if parallel is True:
        want = min(n_tasks, cpus)
    else:  # None = 自动
        if n_configs <= 1 or cpus < 4:
            return 0
        # 实测（24 核）：225² 网格下 8 进程 × 3 线程相对单进程 8 线程有
        # 3.8× 吞吐；线程再多元益，故进程数取 cpus//3 且不超过 8，
        # 每个进程保留 3 个 FFT 线程。内存预算按进程数分摊（见 plan_batches）。
        want = min(n_tasks, max(1, cpus // 3), 8)
    return int(max(0, want))


def parallel_gain(n_workers: int, grid_n: int = 256) -> float:
    """多进程相对单进程（8 FFT 线程）的吞吐倍率（经验模型）。

    实测（24 物理核，batch 64，纯 FFT 与扫描核心一致）：
      225² 网格：1×8 → 1036 FFT/s；3×8 → 2.37×；4×6 → 2.77×；
                 6×4 → 3.28×；8×3 → 3.81×；12×2 → 3.70×
      360² 网格：16×1 进程的完整扫描 → 约 3.8×（相对 1×8 的扫描核心速率）
    故取"满额 3.8×，按进程数线性逼近"，8 进程即达满额。
    换机器/换网格有 ±50% 偏差，仅用于给量级。
    """
    if n_workers <= 1:
        return 1.0
    return 1.0 + 2.8 * min(1.0, (n_workers - 1) / 7.0)


# 扫描核心相对"纯 FFT"的额外开销倍率：每个切片还要做两次逐元素复数乘
# （psi *= t、psi_k * prop）、探针相位斜坡与探测器内积，数组又反复新分配
# （缺页 + 缓存冷启动）。实测 256²/51 切片：纯 FFT 5232 FFT/s、扫描核心
# 2520 FFT/s → 2.08×；但多进程下内存带宽与预算被摊满，实际还会再慢一些，
# 故取 2.6 作为折中（两个实测配置的预估偏差都在 ±30% 内）。
SCAN_OVERHEAD = 2.6


def stem_scan(
    crystal: Structure,
    optics: StemOptics,
    sampling: float,
    geometry: ScanGeometry,
    slice_thickness: float = 2.0,
    detectors: Optional[Sequence[RingDetector]] = None,
    phonons: Optional[PhononConfig] = None,
    capture_thickness_a: Optional[Sequence[float]] = None,
    table: str = "peng",
    threads: Optional[int] = None,
    parallel: Optional[bool] = None,
    mem_budget_bytes: Optional[int] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    stop: Optional[Callable[[], bool]] = None,
) -> StemResult:
    """对周期超胞做 STEM 扫描（HAADF/ADF/BF/ABF 同步输出）。

    Parameters
    ----------
    crystal : Structure
        已重构为沿带轴的正交周期超胞（见 structure.zone_axis_cell）。
    optics : StemOptics
        探针与探测器参数。
    sampling : float
        目标采样 Å/px；实际网格会向上取到 FFT 友好尺寸，故实际采样
        略小于该值（只会上采样，不会变粗）。
    geometry : ScanGeometry
        扫描视场与点数。
    slice_thickness : float
        切片厚度 Å。
    detectors : sequence of RingDetector, 可选
        None = 按光学参数自动生成 ADF/BF/ABF。
    phonons : PhononConfig, 可选
        None = 不做热位移（纯弹性，会低估高角信号，仅用于快速预览/对比）。
    capture_thickness_a : sequence of float, 可选
        需要额外捕获的厚度（Å）。厚度序列在此实现为"传播到该切片边界时
        顺带记录探测器信号"，几乎不增加耗时。
    table : str
        散射因子表（"peng" 默认 / "gauss3" legacy）。
    mem_budget_bytes : int, 可选
        本次模拟的内存上限（字节）。None = 自动（可用内存的一半，封顶 4 GB）。
        并行时该预算要分摊到各子进程（每个进程各持一份透射函数），
        预算不足会自动减少进程数或批大小。
    threads : int, 可选
        FFT 线程数；None = 引擎默认。
    parallel : bool, 可选
        是否多进程。None = 自动（组态数 ≥ 2 且核数足够时启用）。

    Returns
    -------
    StemResult
    """
    t0 = time.perf_counter()
    optics.validate()
    geometry.validate()
    if crystal.cell is None:
        raise ValueError("STEM 扫描要求周期结构（请先用 zone_axis_cell 构建超胞）")
    if sampling <= 0:
        raise ValueError("采样需 > 0")

    notes: List[str] = []
    lx, ly, lz = (float(v) for v in crystal.cell_lengths())

    # 网格：两轴各自向上取 5-光滑数（避开 FFTW 的慢质因子）
    ny_g = fast_grid_size(int(np.ceil(ly / sampling)))
    nx_g = fast_grid_size(int(np.ceil(lx / sampling)))
    if max(ny_g, nx_g) > MAX_GRID:
        raise ValueError(
            f"网格将达到 {nx_g}×{ny_g}（上限 {MAX_GRID}/轴）：横向尺寸 "
            f"{lx:.1f}×{ly:.1f} Å 配采样 {sampling:g} Å/px 过大。"
            "HAADF 的原子分辨率视场通常只需几十 Å——请减小扫描视场，"
            "或增大采样间隔（并相应降低探测器外角）。"
        )
    sy, sx = ly / ny_g, lx / nx_g
    if sx > sampling * 1.5 or sy > sampling * 1.5:
        notes.append(
            f"实际采样 ({sx:.3f}, {sy:.3f}) Å/px 明显粗于请求值 {sampling:g}，"
            "请检查超胞横向尺寸是否为网格的合理倍数"
        )

    # 探测器（含奈奎斯特裁剪）
    if detectors is None:
        det_list, det_notes = make_detectors(optics, min(sx, sy), with_bright_field=True)
    else:
        det_list = list(detectors)
        det_notes = []
        for det in det_list:
            det_notes.extend(det.validate(optics, min(sx, sy)))
    notes.extend(det_notes)
    masks = {d.name: d.mask((ny_g, nx_g), (sx, sy), optics) for d in det_list}
    if any(float(m.max()) <= 0.0 for m in masks.values()):
        raise ValueError("探测器掩模在当前采样/角度下为空，请检查探测器内角与外角")

    # 探针与传播子
    P0 = probe_ft(optics, (ny_g, nx_g), (sx, sy), normalize=True)
    KX, KY, K2 = frequency_grid((ny_g, nx_g), (sx, sy))

    # 声子组态
    n_configs = 1
    phonon_iter = None
    sigma: Dict[str, float] = {}
    phonon_summary = "未启用（纯弹性）"
    if phonons is not None and phonons.n_configs >= 1:
        phonons.validate()
        fp = FrozenPhonons(crystal, phonons)
        n_configs = len(fp)
        phonon_iter = fp
        sigma = dict(fp.sigma)
        phonon_summary = phonons.summary()
        notes.extend(fp.notes)
        if n_configs == 1:
            notes.append(
                "只用了 1 个声子组态：ADF 图像会残留格子条纹、Z 衬度不可靠，"
                "仅建议用于快速预览"
            )

    # 厚度捕获点
    n_slices = max(1, int(np.ceil(lz / slice_thickness)))
    dz = lz / n_slices
    capture_slices: List[int] = []
    capture_thickness: Dict[int, float] = {}
    if capture_thickness_a:
        for t in sorted(set(float(x) for x in capture_thickness_a)):
            i = int(min(max(1, round(t / dz)), n_slices))
            capture_slices.append(i)
            capture_thickness[i] = i * dz
    capture_slices.append(n_slices)
    capture_thickness[n_slices] = lz
    capture_slices = sorted(set(capture_slices))

    # 并行任务切分：按 (组态, 行块)
    n_workers = _resolve_parallel(parallel, n_configs, n_configs * geometry.ny)
    if n_workers:
        ok, reason = spawn_safety()
        if not ok:
            notes.append(f"未启用多进程（{reason}），本次为串行执行")
            n_workers = 0

    budget = memory_budget(mem_budget_bytes)
    trans_bytes = n_slices * ny_g * nx_g * 16
    if trans_bytes > 0.8 * budget:
        raise ValueError(
            f"透射函数需要 {trans_bytes / 1024 ** 3:.1f} GB（{n_slices} 切片 × "
            f"{nx_g}×{ny_g} 网格），超过内存预算 {budget / 1024 ** 3:.1f} GB。"
            "请减小扫描视场、增大采样间隔（并相应降低探测器外角），"
            "或增大切片厚度。HAADF 的原子分辨率视场通常只需几 nm。"
        )
    batch, n_workers, mem_notes = plan_batches(
        ny_g, nx_g, n_slices, n_workers, budget
    )
    notes.extend(mem_notes)

    if n_workers:
        rows_per_block = max(1, int(np.ceil(geometry.ny / max(1, 4 * n_workers // n_configs))))
    else:
        rows_per_block = geometry.ny
    blocks = [(r, min(r + rows_per_block, geometry.ny))
              for r in range(0, geometry.ny, rows_per_block)]
    tasks = [(c, lo, hi) for c in range(n_configs) for (lo, hi) in blocks]

    threads_eff = get_threads() if threads is None else int(threads)
    if n_workers:
        threads_eff = max(1, min(8, (os.cpu_count() or 1) // n_workers))
        threads_eff = min(threads_eff, 8 if threads is None else int(threads))

    # 累加缓冲（每个探测器一张全图 + 每个捕获厚度一张）
    total: Dict[int, Dict[str, np.ndarray]] = {
        i: {name: np.zeros(geometry.shape, dtype=float) for name in masks}
        for i in capture_slices
    }
    n_fft = len(geometry.positions(lx, ly)) * n_configs * (2 * n_slices - 1)

    def _accumulate(row_lo: int, row_hi: int, res: Dict[int, Dict[str, np.ndarray]]):
        for i, per_det in res.items():
            for name, vec in per_det.items():
                total[i][name][row_lo:row_hi, :] += vec.reshape(row_hi - row_lo, geometry.nx)

    if n_workers:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        payload = dict(
            structure=crystal,
            phonons=phonon_iter,
            optics=optics,
            sampling=sampling,
            gpts=(ny_g, nx_g),
            slice_thickness=slice_thickness,
            table=table,
            geometry=geometry,
            cell_lx=lx,
            cell_ly=ly,
            P0=P0,
            KX=KX,
            KY=KY,
            prop=_fresnel_propagator(
                optics.wavelength, dz, nx_g, ny_g, sx, sy
            ),
            masks=masks,
            capture_slices=capture_slices,
            batch=batch,
        )
        run_key = f"{id(crystal)}-{ny_g}x{nx_g}x{n_slices}-{n_configs}"
        done = 0
        try:
            with ctx.Pool(
                n_workers,
                initializer=_worker_init,
                initargs=(threads_eff, run_key, payload),
            ) as pool:
                for _c, row_lo, row_hi, res in pool.imap_unordered(
                    _worker_task, tasks, chunksize=1
                ):
                    _accumulate(row_lo, row_hi, res)
                    done += 1
                    if stop is not None and stop():
                        pool.terminate()
                        raise CancelledError("用户取消")
                    if progress is not None:
                        progress(
                            0.06 + 0.9 * done / len(tasks),
                            f"STEM 扫描 {done}/{len(tasks)} 块（{n_workers} 进程）",
                        )
        except CancelledError:
            raise
        except (OSError, RuntimeError) as exc:
            # 进程池不可用（受限环境等）时退回串行，不中断模拟
            notes.append(f"多进程不可用（{exc}），已改用串行执行")
            total = {i: {name: np.zeros(geometry.shape) for name in masks}
                     for i in capture_slices}
            n_workers = 0
    if not n_workers:
        prop = _fresnel_propagator(optics.wavelength, dz, nx_g, ny_g, sx, sy)
        for c in range(n_configs):
            sub = crystal if phonon_iter is None else phonon_iter[c]
            trans, _s, _e = build_slice_transmissions(
                sub, optics, sampling=sampling, gpts=(ny_g, nx_g),
                slice_thickness=slice_thickness, padding=0.0, table=table,
            )
            for (lo, hi) in blocks:
                positions = geometry.row_positions(lx, ly, lo, hi)
                res = _scan_positions(
                    trans, (sx, sy), P0, KX, KY, prop, positions, masks,
                    capture_slices, threads=threads_eff, batch=batch,
                    stop=stop,
                )
                _accumulate(lo, hi, res)
                if progress is not None:
                    frac = (c + (hi / geometry.ny)) / n_configs
                    progress(0.06 + 0.9 * float(frac),
                             f"STEM 扫描 声子组态 {c + 1}/{n_configs}，行 {hi}/{geometry.ny}")

    for i in capture_slices:
        for name in total[i]:
            total[i][name] /= float(n_configs)

    images = {name: total[n_slices][name] for name in masks}
    elapsed = time.perf_counter() - t0
    if progress is not None:
        progress(1.0, f"完成（{elapsed:.1f} s）")

    return StemResult(
        images=images,
        captures={i: total[i] for i in capture_slices},
        capture_thickness_a=dict(capture_thickness),
        sampling=(sx, sy),
        extent=(lx, ly),
        geometry=geometry,
        thickness_a=lz,
        n_slices=n_slices,
        dz=dz,
        n_atoms=len(crystal),
        n_fft=n_fft,
        elapsed_s=elapsed,
        detector_specs=det_list,
        optics=optics,
        notes=notes,
        sigma=sigma,
        phonon_summary=phonon_summary,
        n_configs=n_configs,
        fft_rate=float(n_fft / max(elapsed, 1e-9)),
    )


# ----------------------------------------------------------------------
# 时间预估
# ----------------------------------------------------------------------
def estimate_seconds(
    geometry: ScanGeometry,
    n_slices: int,
    n_configs: int,
    grid_shape: Tuple[int, int],
    batch: Optional[int] = None,
    n_workers: int = 0,
) -> float:
    """预估耗时（秒）。

    做法是**在真实网格尺寸与批大小上实测一次批量 FFT**，再按总变换次数外推：
        总变换次数 = 探针数 × 组态数 × (2·切片数 − 1)
    这样自动包含网格尺度、批大小摊薄、线程效率等所有因素。实测校准：
    360² 网格 / 51 切片 / 8 组态 / 32² 探针（16 进程）预估 324 s、实跑 261 s；
    256² 网格同规模（8 进程）预估 108 s、实跑 136 s——即偏差约 ±30%，
    且大网格偏保守、小网格偏乐观。

    多进程的收益用 parallel_gain 的经验模型（见其文档，实测校准）。
    """
    from ._fft import fft_throughput

    ny, nx = grid_shape
    n_pos = geometry.nx * geometry.ny
    n_fft = n_pos * max(1, n_configs) * max(1, 2 * n_slices - 1)
    if batch is None:
        batch, _w, _n = plan_batches(ny, nx, n_slices, n_workers, memory_budget(None))
    probe_batch = int(max(1, min(batch, 64)))
    base = fft_throughput(ny, probe_batch)         # (probe_batch, ny, ny) 批量变换耗时
    per_fft = base / probe_batch * SCAN_OVERHEAD
    return float(n_fft * per_fft / parallel_gain(n_workers, ny))
