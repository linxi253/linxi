"""FFT 后端抽象：优先 pyfftw（多线程、批量），回退 scipy/numpy。

STEM 扫描的耗时几乎全在 FFT 上：每个探针位置、每个切片都要做一次
正变换和一次逆变换。实测（24 物理核，batch=64，complex128 二维变换）：

    numpy.fft            300²: 1130 µs/FFT（单线程，不释放 GIL）
    scipy.fft workers=-1 300²: 1130 µs/FFT（本机 pocketfft 线程化无收益）
    pyfftw threads=8     306²:  325 µs/FFT
    pyfftw threads=24    306²:  256 µs/FFT

因此默认走 pyfftw；缺失时自动回退，功能不受影响（只是慢）。

另一个同样重要的结论：**网格尺寸必须避开大质因子**。306 = 2·3²·17，
FFTW 无法用其 radix 核，实测比 320（2⁶·5）慢 1.57 倍。`fast_grid_size`
把网格向上取整到 5-光滑数，代价是采样略微变细（物理上只会更好）。
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np

try:  # pragma: no cover - 取决于环境
    import pyfftw
    import pyfftw.interfaces.numpy_fft as _pf

    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(120.0)
    HAVE_PYFFTW = True
except Exception:  # noqa: BLE001 - 任何导入期异常都退回 scipy
    HAVE_PYFFTW = False
    _pf = None

from scipy import fft as _scipy_fft


# ----------------------------------------------------------------------
# 线程数
# ----------------------------------------------------------------------
def default_threads() -> int:
    """默认 FFT 线程数。

    实测线程收益在 8 线程后趋于饱和（8→24 线程仅再快 1.27 倍），
    且高线程数会和冻结声子的多进程并行争抢核心，故默认封顶。
    可用环境变量 STEM_FFT_THREADS 覆盖。
    """
    env = os.environ.get("STEM_FFT_THREADS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    cpus = os.cpu_count() or 1
    return max(1, min(8, cpus))


_THREADS = default_threads()
_lock = threading.Lock()


def _thread_count(threads: Optional[int]) -> int:
    return _THREADS if threads is None else max(1, int(threads))


def backend_name() -> str:
    return "pyfftw" if HAVE_PYFFTW else "scipy"


def set_threads(n: int) -> None:
    """设置本进程使用的 FFT 线程数（多进程并行时每个进程调小）。"""
    global _THREADS
    with _lock:
        _THREADS = max(1, int(n))


def get_threads() -> int:
    return _THREADS


# ----------------------------------------------------------------------
# 批量二维变换（对最后两轴）
# ----------------------------------------------------------------------
def fft2(a: np.ndarray, threads: Optional[int] = None) -> np.ndarray:
    """批量二维正变换（对最后两个轴），保持输入形状。"""
    t = _thread_count(threads)
    if HAVE_PYFFTW and a.ndim >= 2:
        return _pf.fft2(a, axes=(-2, -1), threads=t, overwrite_input=False)
    return _scipy_fft.fft2(a, axes=(-2, -1), workers=t)


def ifft2(a: np.ndarray, threads: Optional[int] = None) -> np.ndarray:
    """批量二维逆变换（对最后两个轴）。"""
    t = _thread_count(threads)
    if HAVE_PYFFTW and a.ndim >= 2:
        return _pf.ifft2(a, axes=(-2, -1), threads=t, overwrite_input=False)
    return _scipy_fft.ifft2(a, axes=(-2, -1), workers=t)


# ----------------------------------------------------------------------
# 网格尺寸
# ----------------------------------------------------------------------
def _is_fast(n: int) -> bool:
    """是否为 5-光滑数（质因子只有 2/3/5）。"""
    if n < 1:
        return False
    for p in (2, 3, 5):
        while n % p == 0:
            n //= p
    return n == 1


def fast_grid_size(n: int, extra_factor: float = 1.0) -> int:
    """返回 ≥ n·extra_factor 的最小 5-光滑整数（避开 FFTW 慢质因子）。

    上界保护：最多向上找 200 个整数，找不到就退回原值（实际不会发生）。
    """
    target = max(1, int(np.ceil(n * extra_factor - 1e-9)))
    for m in range(target, target + 200):
        if _is_fast(m):
            return m
    return target


def fft_throughput(n: int = 256, batch: int = 32, threads: Optional[int] = None) -> float:
    """实测一次 (batch,n,n) 批量二维变换的耗时（秒），用于时间预估。

    结果按 (n,batch,threads) 缓存；单次计时开销约几十毫秒，相对整场
    模拟可忽略，但换来的是自动包含网格尺度、批摊薄与线程效率的预估。
    """
    key = (n, batch, _thread_count(threads))
    hit = _THROUGHPUT_CACHE.get(key)
    if hit is not None:
        return hit
    a = np.ones((batch, n, n), dtype=np.complex128)
    fft2(a, threads=threads)  # 预热 / 建计划
    import time

    t0 = time.perf_counter()
    reps = 3
    for _ in range(reps):
        fft2(a, threads=threads)
    dt = (time.perf_counter() - t0) / reps
    _THROUGHPUT_CACHE[key] = dt
    return dt


_THROUGHPUT_CACHE: dict = {}
