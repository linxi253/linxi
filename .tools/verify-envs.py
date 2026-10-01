# -*- coding: utf-8 -*-
"""Run each project's own test suite inside its freshly provisioned venv.

pytest is a *dev* dependency and is absent from most runtime locks, so it is
installed temporarily. Afterwards the venv is restored to the exact package set
it had before -- leaving stray transitive dependencies behind (Pygments,
pluggy, ...) would silently break lock fidelity, and uninstalling pytest
outright would break projects whose requirements legitimately include it.
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
BOOTSTRAP = {"pip", "setuptools", "wheel"}

# project dir, venv, command (None -> pytest), label
JOBS = [
    (r"02-图像处理\hrtem-HRTEM滤波工具", ".venv", None, "pytest"),
    (r"03-应变分析\strainpp-GPA应变分析", ".venv", None, "pytest"),
    (r"03-应变分析\原子级应力分析-PPA", ".venv", None, "pytest"),
    (r"03-应变分析\原子识别纯算法", ".venv", None, "pytest"),
    (r"04-统计分析\原子衬度统计", ".venv", None, "pytest"),
    (r"04-统计分析\特征区域演化分析", ".venv", None, "pytest"),
    (r"04-统计分析\统计面积", ".venv", None, "pytest"),
    (r"05-4D-STEM分析\4D-STEM-Processor", ".venv", None, "pytest"),
    (r"05-EELS分析\EELS边缘价态分析工具", ".venv", None, "pytest"),
    (r"01-视频与数据提取\视频切片工具", ".venv", None, "pytest"),
    (r"02-图像处理\drift-correction-v7", ".venv-build", None, "pytest"),
    (r"02-图像处理\stem-optimize-STEM图像优化", ".venv-build", None, "pytest"),
    (r"02-图像处理\离域效应去除工具", ".venv-build", None, "pytest"),
    (r"03-应变分析\原子中心识别模型开发", ".venv", None, "pytest"),
    (r"09-HRTEM模拟", ".venv", ["tests/verify_physics.py"], "verify_physics"),
    (r"010-STEM模拟", ".venv", ["tests/verify_physics.py"], "verify_physics"),
    # the suite ships a standalone smoke script, not pytest tests
    (r"全整合", ".venv", ["tests/smoke_test.py"], "smoke_test"),
]


def run(cmd, cwd=None, timeout=1200):
    try:
        return subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def has_module(py: Path, mod: str) -> bool:
    return run([str(py), "-c", f"import {mod}"], timeout=120).returncode == 0


def snapshot(py: Path) -> set[str]:
    """Exact installed package set, so the venv can be restored afterwards."""
    r = run([str(py), "-m", "pip", "freeze"], timeout=180)
    return {
        l.strip() for l in r.stdout.splitlines()
        if l.strip() and not l.startswith("#")
        and re.split(r"[=<>\s]", l.strip(), 1)[0].lower() not in BOOTSTRAP
    }


def restore(py: Path, before: set[str]) -> None:
    """Drop what we added and re-add anything we removed."""
    after = snapshot(py)
    extra = sorted(after - before)
    missing = sorted(before - after)
    if extra:
        names = [re.split(r"[=<>\s]", e, 1)[0] for e in extra]
        run([str(py), "-m", "pip", "uninstall", "-y", "-q", *names], timeout=300)
    if missing:
        run([str(py), "-m", "pip", "install", "-q",
             "--disable-pip-version-check", "-i", MIRROR, *missing], timeout=900)


def main() -> int:
    print(f"workspace: {ROOT}")
    print("=" * 78)
    rows = []
    for rel, venv, cmd, label in JOBS:
        proj = ROOT / rel
        py = proj / venv / "Scripts" / "python.exe"
        if not py.is_file():
            rows.append((rel, label, "NO VENV", ""))
            print(f"[SKIP] {rel:<40} no venv")
            continue

        before = snapshot(py)
        if cmd is None and not has_module(py, "pytest"):
            r = run([str(py), "-X", "utf8", "-m", "pip", "install", "-q",
                     "--disable-pip-version-check", "-i", MIRROR, "pytest"])
            if r is None or r.returncode != 0:
                rows.append((rel, label, "NO PYTEST", ""))
                print(f"[SKIP] {rel:<40} could not install pytest")
                continue

        argv = [str(py), "-X", "utf8", "-m", "pytest", "-q"] if cmd is None \
            else [str(py), "-X", "utf8", *cmd]
        t0 = time.time()
        r = run(argv, cwd=proj)
        dur = time.time() - t0

        restore(py, before)

        if r is None:
            rows.append((rel, label, "TIMEOUT", f"{dur:.0f}s"))
            print(f"[TIME] {rel:<40} {label} timed out after {dur:.0f}s")
            continue

        summary = ""
        for line in reversed([l for l in r.stdout.splitlines() if l.strip()]):
            if re.search(r"(\d+ (passed|failed|error)|通过|OK\b)", line):
                summary = line.strip()[:64]
                break
        status = "PASS" if r.returncode == 0 else "FAIL"
        rows.append((rel, label, status, summary))
        print(f"[{status}] {rel:<40} {label:<14} {dur:5.0f}s  {summary}")

    print("=" * 78)
    bad = [r for r in rows if r[2] != "PASS"]
    print(f"passed {len(rows) - len(bad)} / {len(rows)}")
    for rel, label, status, summary in bad:
        print(f"  {status:<9} {rel}  {summary}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
