# -*- coding: utf-8 -*-
"""Run each project's own test suite inside its freshly provisioned venv.

pytest is a *dev* dependency and is absent from most runtime locks, so it is
installed temporarily. Afterwards the venv is restored to the exact package set
it had before -- leaving stray transitive dependencies behind (Pygments,
pluggy, ...) would silently break lock fidelity, and uninstalling pytest
outright would break projects whose requirements legitimately include it.

Audit 127: the project list below must stay identical to the ``dir`` entries of
the ``test`` matrix in ``.github/workflows/ci.yml``. CI guards this invariant
with ``python .tools/verify-envs.py --check-matrix .github/workflows/ci.yml``
(pure stdlib text comparison, no venv/subprocess involved).

Usage::

    python .tools/verify-envs.py                    # run every project's tests
    python .tools/verify-envs.py --check-matrix [path-to-ci.yml]
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
BOOTSTRAP = {"pip", "setuptools", "wheel"}

# project dir, venv, command (None -> pytest), label
# 审计 127：与 .github/workflows/ci.yml test 矩阵的 19 个 dir 一一对应；
# 由 --check-matrix 在 CI 守门，改任一侧须同步另一侧。
JOBS = [
    (r"02-图像处理\hrtem-HRTEM滤波工具", ".venv", None, "pytest"),
    (r"02-图像处理\图像加滤镜工具", ".venv", None, "pytest"),
    (r"03-应变分析\strainpp-GPA应变分析", ".venv", None, "pytest"),
    (r"03-应变分析\原子级应力分析-PPA", ".venv", None, "pytest"),
    (r"03-应变分析\原子识别纯算法", ".venv", None, "pytest"),
    (r"03-应变分析\原子中心识别模型开发", ".venv", None, "pytest"),
    (r"04-统计分析\原子衬度统计", ".venv", None, "pytest"),
    (r"04-统计分析\特征区域演化分析", ".venv", None, "pytest"),
    (r"04-统计分析\统计面积", ".venv", None, "pytest"),
    (r"04-统计分析\非晶面积统计", ".venv", None, "pytest"),
    (r"05-4D-STEM分析\4D-STEM-Processor", ".venv", None, "pytest"),
    (r"05-EELS分析\EELS边缘价态分析工具", ".venv", None, "pytest"),
    (r"01-视频与数据提取\视频切片工具", ".venv", None, "pytest"),
    (r"02-图像处理\drift-correction-v7", ".venv-build", None, "pytest"),
    (r"02-图像处理\stem-optimize-STEM图像优化", ".venv-build", None, "pytest"),
    (r"02-图像处理\离域效应去除工具", ".venv-build", None, "pytest"),
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
    # 审计 65：run() 超时返回 None；此前直接 .returncode 会 AttributeError。
    # 超时按「模块不可用」处理，让上层走安全的降级分支。
    r = run([str(py), "-c", f"import {mod}"], timeout=120)
    return r is not None and r.returncode == 0


def snapshot(py: Path) -> set[str] | None:
    """Exact installed package set, so the venv can be restored afterwards.

    审计 65：pip freeze 超时返回 None。不能返回空集合充当「快照」——那会让
    restore() 把整个 site-packages 当作多余物全部卸掉；调用方必须判 None。
    """
    r = run([str(py), "-m", "pip", "freeze"], timeout=180)
    if r is None:
        return None
    return {
        l.strip() for l in r.stdout.splitlines()
        if l.strip() and not l.startswith("#")
        and re.split(r"[=<>\s]", l.strip(), 1)[0].lower() not in BOOTSTRAP
    }


def restore(py: Path, before: set[str]) -> None:
    """Drop what we added and re-add anything we removed."""
    after = snapshot(py)
    if after is None:
        # 审计 65：拿不到当前快照就无法安全 diff；宁可不动作也不把 venv 卸空
        print(f"    warn  pip freeze timed out in {py}; venv left untouched")
        return
    extra = sorted(after - before)
    missing = sorted(before - after)
    if extra:
        names = [re.split(r"[=<>\s]", e, 1)[0] for e in extra]
        run([str(py), "-m", "pip", "uninstall", "-y", "-q", *names], timeout=300)
    if missing:
        run([str(py), "-m", "pip", "install", "-q",
             "--disable-pip-version-check", "-i", MIRROR, *missing], timeout=900)


def load_provision_dirs() -> set[str]:
    """Directory set of .tools/provision-envs.py JOBS (module import is
    side-effect free: its top level only defines constants and functions)."""
    import importlib.util
    path = Path(__file__).with_name("provision-envs.py")
    spec = importlib.util.spec_from_file_location("provision_envs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {str(rel).replace("\\", "/") for _, rel, *_ in mod.JOBS}


def check_matrix(ci_path: Path) -> int:
    """审计 127：ci.yml test 矩阵必须与 .tools 清单收敛为单一来源。

    约束：verify-envs JOBS 的目录集合 == 矩阵的 dir 集合；矩阵每个 dir 都
    必须出现在 provision-envs JOBS（后者可以更广，如无测试的 10-DSH集成）。
    纯文本比对，无任何 venv/子进程操作。
    """
    if not ci_path.is_file():
        print(f"ci.yml not found: {ci_path}")
        return 2
    matrix = sorted(set(re.findall(r"dir:\s*'([^']+)'", ci_path.read_text(encoding="utf-8"))))
    jobs = sorted({rel.replace("\\", "/") for rel, *_ in JOBS})
    provision = load_provision_dirs()
    ok = True
    for tag, extra in (
        ("dir only in ci.yml matrix", set(matrix) - set(jobs)),
        ("dir only in verify-envs JOBS", set(jobs) - set(matrix)),
    ):
        for d in sorted(extra):
            ok = False
            print(f"MISMATCH {tag}: {d}")
    for d in sorted(set(matrix) - provision):
        ok = False
        print(f"MISMATCH dir missing from provision-envs JOBS: {d}")
    if ok:
        print(f"ci.yml matrix <-> .tools JOBS aligned: {len(matrix)} projects")
        return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--check-matrix" in args:
        i = args.index("--check-matrix")
        if i + 1 < len(args):
            return check_matrix(ROOT / args[i + 1])
        return check_matrix(ROOT / ".github" / "workflows" / "ci.yml")
    if args:
        print(f"unknown arguments: {' '.join(args)} "
              "(supported: --check-matrix [ci.yml path])")
        return 2

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
        if before is None:
            # 审计 65：pip freeze 超时——记为失败跳过该 job，而不是在
            # restore() 时把 venv 卸空
            rows.append((rel, label, "TIMEOUT", "pip freeze"))
            print(f"[TIME] {rel:<40} {label} pip freeze timed out")
            continue
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
