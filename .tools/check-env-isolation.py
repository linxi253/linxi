# -*- coding: utf-8 -*-
"""Audit the AIforTEM workspace for environment-isolation violations.

Checks
------
1. The Miniconda base interpreter is clean (``pip check`` passes, no ``~*``
   leftover distribution directories in site-packages).
2. Every active project that declares dependencies also has a project-local
   venv and a lock file.
3. Every project venv is genuinely isolated
   (``include-system-site-packages = false``) and its interpreter runs.
4. No launcher/build script falls back to a bare ``python`` from PATH.

Usage::

    python <仓库根>\\.tools\\check-env-isolation.py

Exit code 0 = clean, 1 = violations found.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_PY = Path(r"C:\ProgramData\Miniconda3\python.exe")

VENV_NAMES = (".venv", ".venv-build", ".venv-run")
SKIP_DIR_PARTS = {
    "08-历史版本", "site-packages", "node_modules", "__pycache__", "build",
    "dist", "runs", "legacy", "release", "_internal", ".runtime", ".tools",
    ".venv", ".venv-build", ".venv-run", ".review-tmp", "07-文档资料",
    "06-独立脚本",
}
SCRIPT_SUFFIXES = (".bat", ".cmd", ".ps1")

# Nested dependency manifests that intentionally have no environment of their
# own. Keep this list explicit and justified -- never silence a real gap.
DEFERRED = {
    r"03-应变分析\原子级应力分析-PPA\atom_detector":
        "深度学习子项目；PPA 稳定版不加载任何 .pt 模型（见其 README），"
        "需要时单独建环境",
    r"开发中\自动识别晶面取向\diffract_indexer":
        "该 requirements.txt 是父项目锁的来源，环境为父目录 .venv",
}

# a bare `python` token: not preceded by a path separator or a $ (PS variable)
BARE_PYTHON = re.compile(r"(?<![\w\\./\-$])python(?:\.exe)?(?![\w.\-])")
SAFE_LINE = re.compile(
    r"^\s*(rem|::|#)"                     # comment (any case)
    r"|^\s*echo\b"                        # echo text, not a command
    r"|\$python"                          # PowerShell variable
    r"|where\s+python"                    # existence probe
    r"|python\s+-m\s+venv"                # one-time venv bootstrap
    r"|-m\s+venv"                         # same, other word order
    r"|BOOTSTRAP\w*"                      # documented bootstrap variable
    r"|Join-Path.*python"                 # building a path, not invoking
    r"|-eq\s*['\"]python"                 # string comparison
    r"|python\s+-c\s+['\"]import sys",    # version probe
    re.IGNORECASE,
)

problems: list[str] = []


def decode(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "gbk", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", "replace")


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace")


def skipped(path: Path) -> bool:
    return any(part in SKIP_DIR_PARTS for part in path.parts)


def check_base() -> None:
    print("[1] Miniconda base cleanliness")
    sp = Path(r"C:\ProgramData\Miniconda3\Lib\site-packages")
    if sp.is_dir():
        junk = [d.name for d in sp.iterdir() if d.is_dir() and d.name.startswith("~")]
        if junk:
            problems.append(f"base has {len(junk)} leftover distribution dirs: "
                            + ", ".join(junk))
            print(f"    FAIL  leftover dirs: {', '.join(junk)}")
        else:
            print("    ok    no leftover distribution dirs")

    r = run([str(BASE_PY), "-m", "pip", "check"])
    if r.returncode == 0:
        print("    ok    pip check clean")
    else:
        problems.append("base pip check failed: " + r.stdout.strip()[:300])
        print("    FAIL  pip check:")
        for line in r.stdout.strip().splitlines()[:5]:
            print("          " + line)

    r = run([str(BASE_PY), "-c", "import sys;print(sys.version.split()[0])"])
    print(f"    info  base interpreter Python {r.stdout.strip()}")


def find_projects() -> list[Path]:
    found = []
    for req in ROOT.rglob("requirements*.txt"):
        if not skipped(req):
            found.append(req.parent)
    for proj in ROOT.rglob("pyproject.toml"):
        if not skipped(proj):
            found.append(proj.parent)
    return sorted(set(found))


def check_projects() -> None:
    print("\n[2] Project environments")
    for proj in find_projects():
        rel = proj.relative_to(ROOT)
        key = str(rel)
        if key in DEFERRED:
            print(f"    skip  {key}\n          {DEFERRED[key]}")
            continue
        if not (proj / "requirements.txt").is_file() and \
                not any((proj / n).is_dir() for n in VENV_NAMES):
            continue                       # pyproject-only, no env expected

        venvs = [proj / n for n in VENV_NAMES if (proj / n).is_dir()]
        locks = [p for p in proj.iterdir()
                 if p.is_file() and "lock" in p.name.lower()
                 and p.suffix in (".txt", ".lock")]
        locks += [p for p in proj.glob("requirements/*lock*") if p.is_file()]

        issues = []
        if not venvs:
            issues.append("no venv")
        if not locks:
            issues.append("no lock file")
        if issues:
            problems.append(f"{rel}: {', '.join(issues)}")
            print(f"    FAIL  {rel}: {', '.join(issues)}")
            continue

        detail = []
        for v in venvs:
            cfg = v / "pyvenv.cfg"
            if not cfg.is_file():
                problems.append(f"{rel}\\{v.name}: missing pyvenv.cfg")
                detail.append(f"{v.name}=BROKEN")
                continue
            text = cfg.read_text(encoding="utf-8", errors="replace")
            if "include-system-site-packages" not in text or \
                    "false" not in text.split("include-system-site-packages")[1][:20]:
                problems.append(f"{rel}\\{v.name}: leaks system site-packages")
                detail.append(f"{v.name}=LEAKY")
                continue
            py = v / "Scripts" / "python.exe"
            r = run([str(py), "-c", "import sys;print(sys.version.split()[0])"])
            ver = r.stdout.strip() if r.returncode == 0 else "BROKEN"
            if ver == "BROKEN":
                problems.append(f"{rel}\\{v.name}: interpreter does not run")
            detail.append(f"{v.name}={ver}")
        print(f"    ok    {rel}  [{' '.join(detail)}]  lock={locks[0].name}")


def check_scripts() -> None:
    print("\n[3] Scripts invoking bare python")
    hits = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SCRIPT_SUFFIXES:
            continue
        if skipped(path):
            continue
        for i, line in enumerate(decode(path).splitlines(), 1):
            if not BARE_PYTHON.search(line) or SAFE_LINE.search(line):
                continue
            problems.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()}")
            print(f"    FAIL  {path.relative_to(ROOT)}:{i}")
            print(f"          {line.strip()}")
            hits += 1
    if not hits:
        print("    ok    no bare-python fallbacks found")


def check_pth() -> None:
    """``.pth`` files must stay pure ASCII.

    CPython's ``site.py`` reads ``.pth`` files using the *locale* encoding, so a
    path containing Chinese characters is readable only in one mode: written as
    GBK it breaks under ``-X utf8``, written as UTF-8 it breaks in the default
    mode. A pure-ASCII ``.pth`` (deriving the path at runtime) works in both.
    """
    print("\n[4] .pth files inside venvs")
    bad = []
    for p in ROOT.rglob("*.pth"):
        if "site-packages" not in p.parts or skipped(p):
            continue
        if any(b >= 128 for b in p.read_bytes()):
            bad.append(p)
    if bad:
        for p in bad:
            problems.append(f"non-ASCII .pth: {p.relative_to(ROOT)}")
            print(f"    FAIL  {p.relative_to(ROOT)}")
        print("          -> rewrite it as an ASCII-only `import` line")
    else:
        print("    ok    all .pth files are ASCII-only")


def check_pip_config() -> None:
    """The user-level pip.ini must stay pure ASCII and must actually parse.

    pip reads its config using the system locale encoding (cp936 here), so any
    non-ASCII character makes pip reject the whole file with
    "Configuration file contains invalid cp936 characters" -- silently losing
    the mirror setting. Verify by asking pip to parse it.
    """
    print("\n[6] pip user-level configuration")
    cfg = Path.home() / "AppData" / "Roaming" / "pip" / "pip.ini"
    if not cfg.is_file():
        print("    info  no user-level pip.ini (not required)")
        return
    if any(b >= 128 for b in cfg.read_bytes()):
        problems.append(f"pip.ini is not ASCII-only: {cfg}")
        print(f"    FAIL  {cfg} contains non-ASCII bytes")
        print("          -> pip will reject the whole file; keep it ASCII-only")
        return
    r = run([str(BASE_PY), "-m", "pip", "config", "list"])
    if r.returncode != 0 or "invalid" in r.stdout.lower():
        problems.append("pip rejected its configuration file")
        print("    FAIL  pip rejects the config:")
        for line in r.stdout.strip().splitlines()[:3]:
            print("          " + line)
        return
    entries = [l.strip() for l in r.stdout.splitlines() if "=" in l]
    print(f"    ok    {len(entries)} setting(s) parsed")
    for line in entries:
        print(f"          {line}")


def check_lock_fidelity() -> None:
    """For auto-generated locks, verify the venv still matches the lock.

    Only locks carrying the generated header are compared: hand-maintained
    locks (pip-compile style, curated subsets, cu128 variants) legitimately
    describe a different set than what is currently installed.
    """
    print("\n[5] venv matches its auto-generated lock")
    checked = 0
    for lock in sorted(ROOT.rglob("requirements.lock.txt")):
        if skipped(lock):
            continue
        text = lock.read_text(encoding="utf-8", errors="replace")
        if "自动生成，请勿手改" not in text:
            continue
        proj = lock.parent
        venv = proj / ".venv"
        py = venv / "Scripts" / "python.exe"
        if not py.is_file():
            continue
        locked = sorted({l.strip() for l in text.splitlines()
                         if l.strip() and not l.startswith("#")})
        r = run([str(py), "-m", "pip", "freeze"])
        # Editable installs of the project itself are reported by pip freeze as
        # a "# Editable install ..." comment plus an "-e <path>" line. They are
        # not third-party dependencies and are deliberately absent from the
        # lock, so they must be filtered out or every editable project would
        # report a false mismatch. (The path is written in the locale encoding,
        # so it is not even valid UTF-8 -- run() already decodes with
        # errors="replace", and we drop the line right after.)
        installed = sorted({
            l.strip() for l in r.stdout.splitlines()
            if l.strip() and not l.startswith("#") and not l.startswith("-e ")
            and re.split(r"[=<>\s]", l.strip(), 1)[0].lower()
            not in ("pip", "setuptools", "wheel")
        })
        diff = set(installed) ^ set(locked)
        checked += 1
        if diff:
            problems.append(f"{proj.relative_to(ROOT)}: venv differs from "
                            f"{lock.name} by {len(diff)} package(s)")
            print(f"    FAIL  {proj.relative_to(ROOT)}")
            for item in sorted(diff)[:6]:
                print(f"          {item}")
        else:
            print(f"    ok    {proj.relative_to(ROOT)} ({len(locked)} pkgs)")
    if not checked:
        print("    info  no auto-generated locks found")


def main() -> int:
    print(f"workspace : {ROOT}")
    print("=" * 74)
    check_base()
    check_projects()
    check_scripts()
    check_pth()
    check_lock_fidelity()
    check_pip_config()
    print("=" * 74)
    if problems:
        print(f"RESULT: {len(problems)} problem(s) found")
        return 1
    print("RESULT: clean - environment isolation convention is intact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
