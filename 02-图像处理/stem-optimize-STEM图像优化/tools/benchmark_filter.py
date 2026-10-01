"""Benchmark one synthetic frame and report elapsed time and peak working set."""

import argparse
import ctypes
import json
import os
import time

import numpy as np

from filters import adaptive_spectral_filter
from tiff_handler import estimate_peak_memory_bytes


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak_working_set_bytes() -> int | None:
    if os.name != "nt":
        return None
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory_info.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    )
    get_memory_info.restype = ctypes.c_int
    if not get_memory_info(
        get_current_process(),
        ctypes.byref(counters),
        counters.cb,
    ):
        return None
    return int(counters.PeakWorkingSetSize)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=3072)
    args = parser.parse_args()
    if args.size < 2:
        parser.error("--size must be at least 2")

    rng = np.random.default_rng(20260729)
    frame = rng.integers(0, 65536, size=(args.size, args.size), dtype=np.uint16)
    started = time.perf_counter()
    result = adaptive_spectral_filter(frame)
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "shape": list(frame.shape),
                "elapsed_seconds": elapsed,
                "peak_working_set_bytes": peak_working_set_bytes(),
                "estimated_peak_bytes": estimate_peak_memory_bytes(frame.shape),
                "finite": bool(np.isfinite(result).all()),
                "output_dtype": str(result.dtype),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
