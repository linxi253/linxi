# -*- coding: utf-8 -*-
"""Provision one isolated venv + lock file per tool project under the workspace root.

Design notes
------------
* The base interpreter is the existing Miniconda 3.10 (used only as the venv
  *seed*). Nothing is ever installed into it.
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
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BASE_PY = Path(r"C:\ProgramData\Miniconda3\python.exe")
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


def provision(job) -> dict:
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
            r = run([str(BASE_PY), "-m", "venv", str(venv_dir)])
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
    log(f"base interpreter : {BASE_PY}")
    log(f"index            : {MIRROR}")
    log(f"jobs             : {len(JOBS)}")
    log("=" * 78)

    results = []
    with futures.ThreadPoolExecutor(max_workers=3) as pool:
        for res in pool.map(provision, JOBS):
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
