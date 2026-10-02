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

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# base（母本）解释器：环境变量 AIFORTEM_BASE_PY 优先，其次探测常见安装位置
# （与 .tools/provision-envs.py 同一约定）。都不可用时 BASE_PY 为 None，
# [1]/[6] 两项检查把它作为一条违规报告（不崩溃）。
BASE_PY_ENV_VAR = "AIFORTEM_BASE_PY"
_BASE_PY_CANDIDATES = (
    r"C:\ProgramData\Miniconda3\python.exe",
    r"C:\ProgramData\Anaconda3\python.exe",
    "~/miniconda3/python.exe",
    "~/anaconda3/python.exe",
    "~/AppData/Local/Programs/Python/Python313/python.exe",
    "~/AppData/Local/Programs/Python/Python312/python.exe",
    "~/AppData/Local/Programs/Python/Python311/python.exe",
    "~/AppData/Local/Programs/Python/Python310/python.exe",
    r"C:\Python313\python.exe",
    r"C:\Python312\python.exe",
    r"C:\Python311\python.exe",
    r"C:\Python310\python.exe",
)


def _resolve_base_py() -> Path | None:
    env_value = os.environ.get(BASE_PY_ENV_VAR, "").strip()
    candidates = [Path(env_value).expanduser()] if env_value else []
    candidates += [Path(p).expanduser() for p in _BASE_PY_CANDIDATES]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


BASE_PY = _resolve_base_py()

VENV_NAMES = (".venv", ".venv-build", ".venv-run")
# 「08-历史版本」是旧文档约定的本机归档目录名；该目录并不存在于本仓库，
# 列在此处仅为兼容旧检出布局，目录缺席时跳过逻辑本身无害。
SKIP_DIR_PARTS = {
    "08-历史版本", "site-packages", "node_modules", "__pycache__", "build",
    "dist", "runs", "legacy", "release", "_internal", ".runtime", ".tools",
    ".venv", ".venv-build", ".venv-run", ".review-tmp", "07-文档资料",
    "06-独立脚本",
}
SCRIPT_SUFFIXES = (".bat", ".cmd", ".ps1")

# Nested dependency manifests that intentionally have no environment of their
# own. Keep this list explicit and justified -- never silence a real gap.
# 本机特有的内部/在研项目路径不写入本公开脚本：需要时把「相对路径 -> 理由」
# 写入一个本地 JSON 对象文件，并用环境变量 AIFORTEM_DEFERRED_EXTRA 指向它。
DEFERRED = {
    r"03-应变分析\原子级应力分析-PPA\atom_detector":
        "深度学习子项目；PPA 稳定版不加载任何 .pt 模型（见其 README），"
        "需要时单独建环境",
}
DEFERRED_EXTRA_ENV_VAR = "AIFORTEM_DEFERRED_EXTRA"


def load_deferred_extra() -> dict:
    path_value = os.environ.get(DEFERRED_EXTRA_ENV_VAR, "").strip()
    if not path_value:
        return {}
    import json
    raw = json.loads(Path(path_value).expanduser().read_text(encoding="utf-8"))
    return {str(key): str(reason) for key, reason in raw.items()}

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
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace")
    except OSError as exc:
        # 解释器缺失/无法启动必须转成报告行，而不是让审计脚本本身崩溃
        return subprocess.CompletedProcess(cmd, 127, f"[OSError] {exc}\n", None)


def skipped(path: Path) -> bool:
    return any(part in SKIP_DIR_PARTS for part in path.parts)


def check_base() -> None:
    print("[1] Miniconda base cleanliness")
    if BASE_PY is None:
        problems.append("base interpreter not found "
                        f"（可用 {BASE_PY_ENV_VAR} 指定）")
        print("    FAIL  base interpreter not found; set AIFORTEM_BASE_PY "
              "or install Miniconda/Python >= 3.10 in a common location; "
              "pip check / leftover-dir scan skipped")
        return
    sp = BASE_PY.parent / "Lib" / "site-packages"
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
    if r.returncode == 0:
        print(f"    info  base interpreter Python {r.stdout.strip()}")
    else:
        problems.append("base interpreter does not run: "
                        + r.stdout.strip()[:200])
        print("    FAIL  base interpreter does not run")


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
    deferred = {**DEFERRED, **load_deferred_extra()}
    for proj in find_projects():
        rel = proj.relative_to(ROOT)
        key = str(rel)
        if key in deferred:
            print(f"    skip  {key}\n          {deferred[key]}")
            continue
        venvs = [proj / n for n in VENV_NAMES if (proj / n).is_dir()]
        sub_locks = [p for p in proj.glob("requirements/*lock*") if p.is_file()]
        if not (proj / "requirements.txt").is_file() and not venvs and not sub_locks:
            # pyproject-only project, no env expected -- print the reason
            # instead of skipping silently
            print(f"    skip  {rel}: pyproject-only (no requirements.txt, "
                  "no venv, no requirements/ lock) - no env expected")
            continue

        locks = [p for p in proj.iterdir()
                 if p.is_file() and "lock" in p.name.lower()
                 and p.suffix in (".txt", ".lock")]
        locks += sub_locks

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
            if not py.is_file():
                problems.append(f"{rel}\\{v.name}: missing Scripts/python.exe")
                detail.append(f"{v.name}=NO_PYTHON")
                continue
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
    if BASE_PY is None:
        problems.append("base interpreter not found "
                        f"（可用 {BASE_PY_ENV_VAR} 指定）")
        print("    FAIL  base interpreter not found; pip config parse check skipped")
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

    Coverage is the full set of lock files -- top-level ``requirements*.lock*``
    plus everything inside a project's ``requirements/`` subdirectory (e.g.
    the 原子中心识别模型开发 locks) -- checked against every venv name in
    VENV_NAMES, not just the ``requirements.lock.txt`` + ``.venv`` pair. A
    lock passes when at least one project-local venv's ``pip freeze`` matches
    it exactly.
    """
    print("\n[5] venv matches its auto-generated lock")
    checked = 0
    hand_maintained: list[Path] = []

    def is_lock_candidate(p: Path) -> bool:
        try:
            if not p.is_file():
                return False
        except OSError:
            # Windows reserved names (e.g. a stray `nul`) raise WinError 1
            # on stat; ignore them instead of aborting the whole audit
            return False
        return ("lock" in p.name.lower() and p.suffix in (".txt", ".lock")
                and (p.parent.name == "requirements"
                     or p.name.startswith("requirements"))
                and not skipped(p))

    lock_paths = sorted(p for p in ROOT.rglob("*") if is_lock_candidate(p))
    for lock in lock_paths:
        text = lock.read_text(encoding="utf-8", errors="replace")
        if "自动生成，请勿手改" not in text:
            hand_maintained.append(lock)
            continue
        proj = lock.parent.parent if lock.parent.name == "requirements" \
            else lock.parent
        rel = proj.relative_to(ROOT)
        locked = sorted({l.strip() for l in text.splitlines()
                         if l.strip() and not l.startswith("#")})
        venvs = [proj / n for n in VENV_NAMES
                 if (proj / n / "Scripts" / "python.exe").is_file()]
        if not venvs:
            print(f"    info  {rel}: {lock.name} but no runnable venv "
                  "(venv presence is [2]'s job)")
            continue
        matched = None
        best_diff: set[str] | None = None
        for venv in venvs:
            r = run([str(venv / "Scripts" / "python.exe"), "-m", "pip", "freeze"])
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
            if not diff:
                matched = venv
                break
            if best_diff is None or len(diff) < len(best_diff):
                best_diff = diff
        checked += 1
        if matched is not None:
            print(f"    ok    {rel} ({len(locked)} pkgs, {matched.name})")
        else:
            problems.append(f"{rel}: no venv in {', '.join(VENV_NAMES)} matches "
                            f"{lock.name} ({len(best_diff)} package(s) differ)")
            print(f"    FAIL  {rel}: nothing in {', '.join(VENV_NAMES)} "
                  f"matches {lock.name}")
            for item in sorted(best_diff)[:6]:
                print(f"          {item}")
    if hand_maintained:
        shown = ", ".join(str(p.relative_to(ROOT)) for p in hand_maintained[:4])
        more = "" if len(hand_maintained) <= 4 \
            else f" ... (+{len(hand_maintained) - 4} more)"
        print(f"    info  {len(hand_maintained)} hand-maintained lock(s) not "
              f"compared (no auto header): {shown}{more}")
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
