# -*- coding: utf-8 -*-
"""Provision one isolated venv + lock file per tool project under the workspace root.

Design notes
------------
* The base interpreter (the venv *seed*) is resolved at startup: the
  ``AIFORTEM_BASE_PY`` environment variable wins, then a list of common
  Miniconda / conda-env / CPython install locations is probed; the first
  candidate whose interpreter runs and reports Python >= 3.10 is used, and a
  clear error is raised before any side effect if none qualifies. Nothing is
  ever installed into it.
* Each project gets its own ``.venv`` so a tool can never see another tool's
  site-packages.
* requirements files in this workspace mix UTF-8 and GBK encodings, so they are
  decoded explicitly, comment-stripped and re-emitted as pure-ASCII temp files
  before being handed to pip. That removes every encoding failure mode.
* A ``requirements.lock.txt`` is written from ``pip freeze`` so the environment
  can be rebuilt bit-for-bit later.
"""
from __future__ import annotations

import concurrent.futures as futures
import functools
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# venv seed interpreter: env var wins, then common locations; every candidate
# must run and report Python >= 3.10 (all projects here require >= 3.10).
# See resolve_base_python().
MIN_BASE_VERSION = (3, 10)
BASE_PY_ENV_VAR = "AIFORTEM_BASE_PY"
BASE_PY_CANDIDATES = (
    r"C:\ProgramData\Miniconda3\python.exe",
    r"C:\ProgramData\Anaconda3\python.exe",
    "~/miniconda3/python.exe",
    "~/anaconda3/python.exe",
    "~/.conda/envs/py312/python.exe",   # 常见 conda 用户级环境（不存在时自动跳过）
    "~/AppData/Local/Programs/Python/Python313/python.exe",
    "~/AppData/Local/Programs/Python/Python312/python.exe",
    "~/AppData/Local/Programs/Python/Python311/python.exe",
    "~/AppData/Local/Programs/Python/Python310/python.exe",
    r"C:\Python313\python.exe",
    r"C:\Python312\python.exe",
    r"C:\Python311\python.exe",
    r"C:\Python310\python.exe",
)
ROOT = Path(__file__).resolve().parent.parent
MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
FALLBACK_INDEX = "https://pypi.org/simple"

BOOTSTRAP = {"pip", "setuptools", "wheel"}

# name, project dir, venv, install source, lock output, verify imports
JOBS = [
    ("hrtem-HRTEM滤波工具", r"02-图像处理\hrtem-HRTEM滤波工具", ".venv",
     "requirements.txt", "requirements.lock.txt",
     ["numpy", "scipy", "tifffile", "imagecodecs", "matplotlib"]),
    ("图像加滤镜工具", r"02-图像处理\图像加滤镜工具", ".venv",
     "requirements.txt", "requirements.lock.txt",
     ["numpy", "scipy", "PIL", "tifffile", "imagecodecs"]),
    ("strainpp-GPA应变分析", r"03-应变分析\strainpp-GPA应变分析", ".venv",
     "requirements-lock.txt", None,
     ["numpy", "scipy", "matplotlib", "tifffile", "ttkbootstrap", "ncempy"]),
    ("原子级应力分析-PPA", r"03-应变分析\原子级应力分析-PPA", ".venv",
     "requirements.txt", "requirements.lock.txt",
     ["numpy", "scipy", "matplotlib", "tifffile", "PIL"]),
    ("原子识别纯算法", r"03-应变分析\原子识别纯算法", ".venv",
     "requirements.txt", "requirements.lock.txt",
     ["numpy", "scipy", "matplotlib", "tifffile", "PIL"]),
    ("特征区域演化分析", r"04-统计分析\特征区域演化分析", ".venv",
     "requirements.txt", "requirements.lock.txt",
     ["numpy", "scipy", "matplotlib", "tifffile", "seaborn", "skimage"]),
    ("统计面积", r"04-统计分析\统计面积", ".venv",
     "requirements.txt", "requirements.lock.txt",
     ["tifffile", "numpy", "pandas", "matplotlib", "PIL", "openpyxl"]),
    ("原子衬度统计", r"04-统计分析\原子衬度统计", ".venv",
     "requirements.txt", "requirements.lock.txt",
     ["numpy", "matplotlib", "tifffile", "ttkbootstrap", "PIL"]),
    ("非晶面积统计", r"04-统计分析\非晶面积统计\pythonProject", ".venv",
     "requirements.txt", "requirements.lock.txt",
     ["cv2", "numpy", "scipy", "tifffile", "PIL", "pandas", "openpyxl",
      "ttkbootstrap"]),
    ("4D-STEM-Processor", r"05-4D-STEM分析\4D-STEM-Processor", ".venv",
     "requirements.txt", "requirements.lock.txt",
     ["numpy", "scipy", "matplotlib", "ncempy"]),
    ("EELS边缘价态分析工具", r"05-EELS分析\EELS边缘价态分析工具", ".venv",
     "requirements.txt", "requirements.lock.txt",
     ["numpy", "scipy", "matplotlib", "ncempy"]),
    ("全整合(TEM Suite)", r"全整合", ".venv",
     "requirements.txt", "requirements.lock.txt",
     ["numpy", "scipy", "matplotlib", "pandas", "tifffile", "PIL", "skimage",
      "seaborn", "openpyxl", "imagecodecs", "defusedxml", "ttkbootstrap",
      "ncempy", "tkinterdnd2", "cv2"]),
    ("09-HRTEM模拟", r"09-HRTEM模拟", ".venv",
     "requirements.txt", "requirements.lock.txt",
     ["numpy", "scipy", "matplotlib", "PIL", "tifffile", "ase",
      "ttkbootstrap"]),
    ("010-STEM模拟", r"010-STEM模拟", ".venv",
     "requirements.txt", "requirements.lock.txt",
     ["numpy", "scipy", "matplotlib", "PIL", "tifffile", "ase",
      "ttkbootstrap", "pyfftw"]),
    # video_extractor 要求 Python>=3.11：base 母本须为 3.11/3.12
    # （本机即 py312，或用 AIFORTEM_BASE_PY 指定），否则依赖装得上、提取阶段跑不了。
    ("10-DSH集成(TEM视频流水线)", r"10-DSH集成", ".venv",
     "requirements.txt", "requirements.lock.txt",
     ["numpy", "cv2", "tifffile", "matplotlib"]),
    ("开发中-01-原位数据集", r"开发中\01-原位数据集", ".venv",
     "requirements.txt", "requirements.lock.txt",
     ["numpy", "tifffile", "matplotlib"]),
    ("开发中-02-对象追踪与动力学", r"开发中\02-对象追踪与动力学", ".venv",
     "requirements.txt", "requirements.lock.txt",
     ["numpy", "tifffile", "matplotlib"]),
    ("开发中-03-漂移矫正升级", r"开发中\03-漂移矫正升级", ".venv",
     "requirements.txt", "requirements.lock.txt",
     ["numpy", "tifffile", "matplotlib"]),
    ("开发中-05-帧质量与事件检测", r"开发中\05-帧质量与事件检测", ".venv",
     "requirements.txt", "requirements.lock.txt",
     ["numpy", "tifffile", "matplotlib"]),
    ("开发中-自动识别晶面取向", r"开发中\自动识别晶面取向", ".venv",
     "requirements.lock.txt", None,
     ["numpy", "scipy", "matplotlib", "cv2", "skimage", "PIL"]),
]

# venv whose freeze becomes the *dev* lock of an existing project
DEV_LOCKS = [
    ("图像加滤镜工具", r"02-图像处理\图像加滤镜工具", ".venv-build",
     "requirements-dev.lock.txt"),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def decode(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "gbk", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", "replace")


def flatten(path: Path, _seen: set | None = None) -> list[str]:
    """Return pure-ASCII requirement specs from a (possibly GBK) req file."""
    _seen = _seen or set()
    path = path.resolve()
    if path in _seen:
        return []
    _seen.add(path)
    out: list[str] = []
    for line in decode(path).splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        s = re.sub(r"\s+#.*$", "", s).strip()   # inline comment
        if not s:
            continue
        if s.startswith(("-r ", "--requirement")):
            inc = s.split(None, 1)[1].strip().strip('"')
            out += flatten(path.parent / inc, _seen)
            continue
        if s.startswith("-"):                    # other pip options: keep as-is
            out.append(s)
            continue
        out.append(s)
    return out


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )


def _interpreter_version(py: Path) -> tuple[int, int] | None:
    """Return (major, minor) if the interpreter runs, else None."""
    r = run([str(py), "-c", "import sys;print(sys.version_info.major, sys.version_info.minor)"])
    if r.returncode != 0:
        return None
    try:
        major, minor = (int(part) for part in r.stdout.split()[:2])
    except ValueError:
        return None
    return major, minor


def resolve_base_python() -> Path:
    """Pick the venv seed interpreter: env var first, then common locations.

    A candidate qualifies only if it runs and reports Python >= 3.10. A bad
    *env-var* candidate is an explicit user decision, so it exits with a clear
    error instead of silently falling back; probed candidates that fail are
    skipped. If nothing qualifies, exit with a clear, actionable error --
    before any venv, install or lock file is touched.
    """
    candidates: list[Path] = []
    env_value = os.environ.get(BASE_PY_ENV_VAR, "").strip()
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates += [Path(p).expanduser() for p in BASE_PY_CANDIDATES]
    candidates.append(Path(sys.executable))   # interpreter running this script

    tried: list[str] = []
    for i, cand in enumerate(candidates):
        reason = None
        if not cand.is_file():
            reason = "文件不存在"
        else:
            ver = _interpreter_version(cand)
            if ver is None:
                reason = "无法运行或版本不可读"
            elif ver < MIN_BASE_VERSION:
                reason = (f"Python {ver[0]}.{ver[1]} < "
                          f"{MIN_BASE_VERSION[0]}.{MIN_BASE_VERSION[1]}，不满足")
        if reason is None:
            return cand
        tried.append(f"  - {cand}（{reason}）")
        if i == 0 and env_value:
            # 显式指定的解释器不合格：立即报错，绝不静默换用其它解释器
            raise SystemExit(
                f"[provision-envs] {BASE_PY_ENV_VAR} 指向的解释器不合格"
                f"（{reason}），已退出（未创建/改动任何环境）。\n"
                f"  {cand}\n"
                f"处理办法：修正 {BASE_PY_ENV_VAR}，或删除该环境变量改用自动探测。"
            )
    raise SystemExit(
        f"[provision-envs] 未找到 Python >= {MIN_BASE_VERSION[0]}."
        f"{MIN_BASE_VERSION[1]} 的母本解释器，已退出（未创建/改动任何环境）。\n"
        "探测记录（按顺序）：\n" + "\n".join(tried) + "\n"
        f"处理办法：设置环境变量 {BASE_PY_ENV_VAR} 指向一个合格解释器后重跑，例如：\n"
        f"  set {BASE_PY_ENV_VAR}=%USERPROFILE%\\.conda\\envs\\py312\\python.exe\n"
        "或在上述常见位置安装 Miniconda / Python >= 3.10。"
    )


def pip_install(py: Path, req: Path, cwd: Path) -> tuple[bool, str]:
    base = [str(py), "-X", "utf8", "-m", "pip", "install",
            "--disable-pip-version-check", "--no-input", "-q"]
    for index in (MIRROR, FALLBACK_INDEX):
        r = run(base + ["-i", index, "-r", str(req)], cwd=cwd)
        if r.returncode == 0:
            return True, ""
    return False, (r.stdout or "")[-1500:]


def freeze(py: Path) -> list[str]:
    r = run([str(py), "-m", "pip", "freeze"])
    lines = []
    for raw in r.stdout.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        name = re.split(r"[=<>\s]", s, 1)[0].strip().lower()
        if name in BOOTSTRAP:
            continue
        if s.startswith("-e ") or s.startswith("--editable"):
            continue
        lines.append(s)
    return sorted(set(lines), key=str.lower)


def write_lock(py: Path, dest: Path, source: str, venv: str) -> int:
    ver = run([str(py), "-c",
               "import sys;print('%d.%d.%d' % sys.version_info[:3])"]
              ).stdout.strip()
    lines = freeze(py)
    header = (
        "# 运行环境锁定（自动生成，请勿手改）\n"
        f"# 来源：{source} 在干净 venv 中解析；Python {ver}\n"
        f"# 复现：python -m venv {venv}; "
        f"{venv}\\Scripts\\python -X utf8 -m pip install -r {dest.name}\n\n"
    )
    dest.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def provision(job, base_py: Path) -> dict:
    name, rel, venv, source, lock_out, verify = job
    proj = ROOT / rel
    venv_dir = proj / venv
    py = venv_dir / "Scripts" / "python.exe"
    t0 = time.time()
    result = {"name": name, "ok": False, "detail": ""}
    try:
        if not proj.is_dir():
            result["detail"] = "project dir missing"
            return result

        src = proj / source
        if not src.is_file():
            result["detail"] = f"missing {source}"
            return result

        # 1. venv
        if not py.is_file():
            r = run([str(base_py), "-m", "venv", str(venv_dir)])
            if r.returncode != 0:
                result["detail"] = "venv creation failed: " + r.stdout[-500:]
                return result
        if not py.is_file():
            result["detail"] = "venv python missing after creation"
            return result

        # 2. install (ASCII temp requirements to dodge GBK/UTF-8 issues)
        specs = flatten(src)
        if not specs:
            result["detail"] = "no requirements parsed"
            return result
        with tempfile.NamedTemporaryFile(
                "w", suffix=".txt", delete=False, encoding="ascii") as fh:
            fh.write("\n".join(specs) + "\n")
            tmp = Path(fh.name)
        try:
            ok, err = pip_install(py, tmp, proj)
        finally:
            tmp.unlink(missing_ok=True)
        if not ok:
            result["detail"] = "pip install failed: " + err
            return result

        # 3. verify imports
        code = ("import importlib,sys\n"
                "bad=[]\n"
                f"for m in {verify!r}:\n"
                "    try: importlib.import_module(m)\n"
                "    except Exception as e: bad.append(m+':'+type(e).__name__)\n"
                "print('BAD=' + ','.join(bad) if bad else 'OK')\n")
        r = run([str(py), "-c", code])
        verdict = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "?"
        if verdict != "OK":
            result["detail"] = "import check: " + verdict
            return result

        # 4. lock
        n = 0
        if lock_out:
            n = write_lock(py, proj / lock_out, source, venv)

        result["ok"] = True
        result["detail"] = (f"{len(specs)} specs -> {n} locked pkgs, "
                            f"{time.time() - t0:.0f}s")
        return result
    except Exception as exc:                      # noqa: BLE001
        result["detail"] = f"{type(exc).__name__}: {exc}"
        return result


def main() -> int:
    base_py = resolve_base_python()
    log(f"base interpreter : {base_py}")
    log(f"index            : {MIRROR}")
    log(f"jobs             : {len(JOBS)}")
    log("=" * 78)

    results = []
    with futures.ThreadPoolExecutor(max_workers=3) as pool:
        for res in pool.map(functools.partial(provision, base_py=base_py), JOBS):
            results.append(res)
            flag = "OK  " if res["ok"] else "FAIL"
            log(f"[{flag}] {res['name']:<26} {res['detail']}")
            sys.stdout.flush()

    # dev locks from already-existing build venvs
    log("-" * 78)
    for name, rel, venv, out in DEV_LOCKS:
        proj = ROOT / rel
        py = proj / venv / "Scripts" / "python.exe"
        if not py.is_file():
            log(f"[SKIP] {name:<26} no {venv}")
            continue
        n = write_lock(py, proj / out, venv, venv)
        log(f"[OK  ] {name:<26} {out} ({n} pkgs)")

    log("=" * 78)
    bad = [r for r in results if not r["ok"]]
    log(f"success {len(results) - len(bad)} / {len(results)}")
    for r in bad:
        log(f"  FAILED {r['name']}: {r['detail']}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
