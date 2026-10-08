# -*- coding: utf-8 -*-
"""Shared loader, interpreter discovery, lock checks and CI guard for the tools.

Single source of truth is ``.tools/projects.json`` (R7). ``provision-envs.py``,
``verify-envs.py`` and ``.tools/ci-project.py`` all read this module, so a
project can no longer be provisioned into one venv and tested in another.

Design notes
------------
* R4 — the seed interpreter is discovered **per project** from the manifest's
  declared Python version. An explicitly supplied interpreter (environment
  variable or CLI mapping) is authoritative: if it is missing, unrunnable or
  outside the project's range the run fails immediately, and it is never
  silently replaced by a discovered one. All selected projects are preflighted
  before the first side effect.
* R6 — nothing in this module writes or regenerates a lock. Installing always
  installs the project's frozen lock as-is.
* R7 — :func:`derive_ci_rows` renders the CI matrix from the manifest and
  :func:`check_ci_matrix` verifies ci.yml against it.

The public manifest deliberately contains no machine-specific interpreter
paths: it declares the Python version a project needs, and discovery finds a
matching interpreter on whatever machine is running.

Stdlib only.
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
MANIFEST_PATH = TOOLS_DIR / "projects.json"

# Set by load_manifest(); lets the tools work against a different checkout (and
# lets the isolation tests build a synthetic repo in a temp directory).
_ACTIVE_ROOT = REPO_ROOT

BASE_PY_ENV_VAR = "AIFORTEM_BASE_PY"
BASE_MAP_ENV_VAR = "AIFORTEM_PYTHONS"
DEFAULT_TIMEOUT = 900


def active_root() -> Path:
    return _ACTIVE_ROOT


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
class ManifestError(SystemExit):
    """Raised (as a clean CLI exit) when the manifest itself is unusable."""


class ExplicitInterpreterError(SystemExit):
    """An explicitly supplied interpreter is unusable; never fall back."""


def load_manifest(path: Path | None = None) -> dict:
    global _ACTIVE_ROOT
    path = Path(path) if path else MANIFEST_PATH
    if not path.is_file():
        raise ManifestError(f"[env] manifest not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"[env] manifest unreadable: {path}: {exc}")
    if not isinstance(data.get("projects"), list) or not data["projects"]:
        raise ManifestError(f"[env] manifest has no projects: {path}")
    data["_path"] = str(path)
    dirs = [e["dir"] for e in data["projects"]]
    dupes = sorted({d for d in dirs if dirs.count(d) > 1})
    if dupes:
        raise ManifestError(f"[env] duplicate project dir(s) in manifest: {dupes}")
    data["_by_dir"] = {e["dir"]: e for e in data["projects"]}
    _ACTIVE_ROOT = path.resolve().parent.parent
    data["_root"] = str(_ACTIVE_ROOT)
    return data


def project_list(manifest: dict) -> list[dict]:
    return list(manifest["projects"])


def find_projects(manifest: dict, selectors: list[str] | None) -> list[dict]:
    projects = project_list(manifest)
    if not selectors:
        return projects
    wanted = {s.strip() for s in selectors if s.strip()}
    picked, seen = [], set()
    for entry in projects:
        keys = {entry["name"], entry["dir"], entry["dir"].replace("/", "\\")}
        if keys & wanted:
            picked.append(entry)
            seen |= keys & wanted
    missing = sorted(wanted - seen)
    if missing:
        known = ", ".join(e["name"] for e in projects)
        raise ManifestError(
            f"[env] unknown project selector(s): {', '.join(missing)}\n"
            f"      known projects: {known}")
    return picked


def project_dir(entry: dict, root: Path | None = None) -> Path:
    base = Path(root) if root else _ACTIVE_ROOT
    return base / entry["dir"].replace("/", os.sep)


def venv_dir(entry: dict, venv_name: str | None = None, root: Path | None = None) -> Path:
    return project_dir(entry, root) / (venv_name or entry["venv"])


def venv_python(entry: dict, venv_name: str | None = None, root: Path | None = None) -> Path:
    return venv_dir(entry, venv_name, root) / "Scripts" / "python.exe"


# --------------------------------------------------------------------------- #
# process helpers (full output is kept, timeouts keep partial output)
# --------------------------------------------------------------------------- #
class Result:
    __slots__ = ("cmd", "code", "out", "timed_out", "cwd")

    def __init__(self, cmd, code, out, timed_out=False, cwd=None):
        self.cmd = cmd
        self.code = code
        self.out = out
        self.timed_out = timed_out
        self.cwd = cwd

    @property
    def ok(self) -> bool:
        return self.code == 0 and not self.timed_out

    def tail(self, n: int = 600) -> str:
        return (self.out or "")[-n:].strip()


def _write_log(log_path: Path | None, cmd, cwd, code, out) -> None:
    if not log_path:
        return
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = (f"\n===== {stamp}  exit={code}\n"
              f"$ {' '.join(str(c) for c in cmd)}\n"
              f"# cwd: {cwd}\n")
    with log_path.open("a", encoding="utf-8", newline="") as fh:
        fh.write(header)
        fh.write(out or "")
        if not (out or "").endswith("\n"):
            fh.write("\n")


def run(cmd, cwd: Path | None = None, timeout: int = DEFAULT_TIMEOUT,
        env: dict | None = None, log_path: Path | None = None) -> Result:
    """Run a command, keeping the complete stdout/stderr and partial output on timeout.

    Child interpreters are pinned to UTF-8 (PYTHONIOENCODING) so their output is
    captured verbatim rather than through the console code page.
    """
    printable = [str(c) for c in cmd]
    cwd_s = str(cwd) if cwd else None
    child_env = dict(os.environ if env is None else env)
    child_env.setdefault("PYTHONIOENCODING", "utf-8")
    # UTF-8 mode: without it, setuptools writes an editable install's .pth file
    # using the ANSI code page. Project paths here contain Chinese characters, and
    # site.py always reads .pth as UTF-8, so a GBK-encoded .pth makes every later
    # interpreter start fail with "init_import_site: Failed to import the site
    # module". UTF-8 mode also makes locale.getpreferredencoding() report utf-8.
    child_env.setdefault("PYTHONUTF8", "1")
    try:
        proc = subprocess.Popen(
            printable, cwd=cwd_s, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=child_env)
    except OSError as exc:
        out = f"{type(exc).__name__}: {exc}"
        _write_log(log_path, printable, cwd_s, 125, out)
        return Result(printable, 125, out, cwd=cwd_s)

    timed_out = False
    try:
        out, _ = proc.communicate(timeout=timeout)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        try:
            out, _ = proc.communicate(timeout=120)
        except Exception:                                    # noqa: BLE001
            out = ""
        code = 124
        out = (out or "") + f"\n[TIMEOUT after {timeout}s; partial output above]\n"
    out = out or ""
    _write_log(log_path, printable, cwd_s, code, out)
    return Result(printable, code, out, timed_out=timed_out, cwd=cwd_s)


# --------------------------------------------------------------------------- #
# interpreter discovery (R4)
# --------------------------------------------------------------------------- #
def parse_version(text: str | None):
    if not text:
        return None
    m = re.match(r"\s*(\d+)\.(\d+)", str(text))
    return (int(m.group(1)), int(m.group(2))) if m else None


def interpreter_version(py: Path, timeout: int = 120) -> tuple | None:
    """(major, minor) if the interpreter really runs, else None.

    A venv whose seed interpreter is gone still has a python.exe on disk; that
    is exactly the failure the old provisioner missed.
    """
    info = interpreter_info(py, timeout=timeout)
    return info.get("version_tuple") if info.get("runs") else None


def interpreter_info(py: Path, timeout: int = 120) -> dict:
    """Full probe: version, sys.prefix/base_prefix, and the raw failure text."""
    py = Path(py)
    info: dict = {"path": str(py), "exists": py.is_file(), "runs": False,
                  "version": None, "version_tuple": None, "prefix": None,
                  "base_prefix": None, "error": None}
    if not info["exists"]:
        info["error"] = "python.exe missing"
        return info
    code = ("import sys;print('%d.%d.%d' % sys.version_info[:3]);"
            "print(sys.prefix);print(sys.base_prefix)")
    res = run([py, "-c", code], timeout=timeout)
    if not res.ok:
        info["exit_code"] = res.code
        info["error"] = (res.out or "").strip()[:400] or f"exit {res.code}"
        return info
    lines = [l for l in res.out.splitlines() if l.strip()]
    if len(lines) >= 3:
        info["runs"] = True
        info["version"] = lines[0].strip()
        info["version_tuple"] = parse_version(lines[0])
        info["prefix"] = lines[1].strip()
        info["base_prefix"] = lines[2].strip()
    else:
        info["error"] = "unexpected probe output"
    return info


def satisfies(version: tuple | None, minimum: str | None, maximum: str | None) -> bool:
    if version is None:
        return False
    lo, hi = parse_version(minimum), parse_version(maximum)
    if lo and version < lo:
        return False
    if hi and version >= hi:
        return False
    return True


def parse_base_map(text: str) -> dict[str, Path]:
    """Parse ``3.10=C:\\path\\python.exe;3.12=C:\\other\\python.exe``.

    A non-empty input that yields no usable entry is an error, not an empty map:
    silently returning ``{}`` would let a typo'd or garbage value look like "no
    explicit mapping was requested" and quietly fall back to discovery.
    """
    out: dict[str, Path] = {}
    rejected: list[str] = []
    for chunk in re.split(r"[;\n]", text or ""):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            rejected.append(chunk)
            continue
        key, _, value = chunk.partition("=")
        key, value = key.strip(), value.strip().strip('"')
        if not key or not value:
            rejected.append(chunk)
            continue
        out[key] = Path(value).expanduser()
    if (text or "").strip() and not out:
        raise ManifestError(
            f"{BASE_MAP_ENV_VAR} was set but contains no usable 'version=path' entry: "
            f"{text!r}. Expected e.g. '3.10=C:\\Python310\\python.exe'. "
            "Refusing to treat this as 'no mapping given'.")
    if rejected:
        raise ManifestError(
            f"{BASE_MAP_ENV_VAR} contains unusable entr(ies): {rejected}. "
            "Expected 'version=path' pairs separated by ';'.")
    return out


def explicit_interpreters(version: str, base_map: dict[str, Path]) -> list[tuple[str, Path]]:
    """Explicitly supplied interpreters for one version, in priority order."""
    out = []
    if version in base_map:
        out.append((f"{BASE_MAP_ENV_VAR}[{version}]", base_map[version]))
    for key, path in base_map.items():
        if key.startswith("default"):
            out.append((f"{BASE_MAP_ENV_VAR}[{key}]", path))
    if os.environ.get(BASE_PY_ENV_VAR, "").strip():
        out.append((BASE_PY_ENV_VAR, Path(os.environ[BASE_PY_ENV_VAR].strip()).expanduser()))
    return out


def _win_launcher_candidates(version: str) -> list[tuple[str, Path]]:
    """Ask the Windows py launcher where a given version lives."""
    launcher = shutil.which("py")
    if not launcher:
        return []
    res = run([launcher, f"-{version}", "-c", "import sys;print(sys.executable)"],
              timeout=60)
    if not res.ok:
        return []
    line = res.out.strip().splitlines()[-1].strip() if res.out.strip() else ""
    return [(f"py -{version}", Path(line))] if line else []


def _generic_candidates(version: str) -> list[tuple[str, Path]]:
    """Portable locations, built from ~ and the standard install roots."""
    tag = version.replace(".", "")
    home = Path.home()
    local = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
    out: list[tuple[str, Path]] = _win_launcher_candidates(version)

    named = [
        ("conda-user", home / "miniconda3" / "python.exe"),
        ("conda-user", home / "anaconda3" / "python.exe"),
        ("conda-env-name", home / ".conda" / "envs" / f"py{tag}" / "python.exe"),
        # python.org installs use the compact tag (Python312), not Python3.12
        ("python-org", local / "Programs" / "Python" / f"Python{tag}" / "python.exe"),
        ("python-root", Path(rf"C:\Python{tag}") / "python.exe"),
        ("conda-system", Path(r"C:\ProgramData\Miniconda3") / "python.exe"),
        ("conda-system", Path(r"C:\ProgramData\Anaconda3") / "python.exe"),
        ("conda-env-name", Path(r"C:\ProgramData\Miniconda3") / "envs" / f"py{tag}" / "python.exe"),
    ]
    out += named

    # any conda env whose directory the user already has, matched by version later
    for envs_root in (home / ".conda" / "envs",
                      Path(r"C:\ProgramData\Miniconda3") / "envs",
                      home / "miniconda3" / "envs"):
        try:
            if not envs_root.is_dir():
                continue
            for child in sorted(envs_root.iterdir()):
                cand = child / "python.exe"
                if cand.is_file():
                    out.append((f"conda-env:{child.name}", cand))
        except OSError:
            continue
    return out


def base_candidates(manifest: dict, entry: dict,
                    base_map: dict[str, Path] | None = None) -> tuple[list, list]:
    """Ordered candidates for one project.

    Returns ``(explicit, discovered)``. Explicit entries are authoritative: if an
    explicit candidate exists but is unusable the caller must fail rather than
    fall through to discovery.
    """
    version = requested_version(manifest, entry)
    base_map = base_map if base_map is not None else parse_base_map(
        os.environ.get(BASE_MAP_ENV_VAR, ""))
    explicit = explicit_interpreters(version, base_map) if version else []
    discovered = _generic_candidates(version) if version else []
    return explicit, discovered


def requested_version(manifest: dict, entry: dict) -> str | None:
    """The Python version this project's seed interpreter must report.

    ``base`` is the version the project's lock was validated against, so it is
    the single source of truth for both the seed and the CI interpreter.
    """
    if entry.get("base"):
        return ".".join(str(entry["base"]).split(".")[:2])
    pmin = (entry.get("python") or {}).get("min")
    if pmin:
        return ".".join(str(pmin).split(".")[:2])
    return None


def resolve_base_python(manifest: dict, entry: dict,
                        base_map: dict[str, Path] | None = None) -> tuple[Path | None, str | None]:
    """Resolve the venv seed for one project.

    An explicitly supplied interpreter that is missing, unrunnable or outside
    the project's range is a hard failure; discovery is only consulted when no
    explicit interpreter was supplied at all (R4).
    """
    version = requested_version(manifest, entry)
    pmin = (entry.get("python") or {}).get("min")
    pmax = (entry.get("python") or {}).get("max")
    explicit, discovered = base_candidates(manifest, entry, base_map)

    if explicit:
        tried = []
        for source, cand in explicit:
            if not cand.is_file():
                tried.append(f"{source} -> {cand} (does not exist)")
                continue
            info = interpreter_info(cand)
            if not info["runs"]:
                tried.append(f"{source} -> {cand} (does not run: {info['error']})")
                continue
            if not satisfies(info["version_tuple"], pmin, pmax):
                tried.append(f"{source} -> {cand} (Python {info['version']} outside "
                             f"[{pmin or '-'}, {pmax or '-'}))")
                continue
            if version and info["version_tuple"] != parse_version(version):
                tried.append(f"{source} -> {cand} (Python {info['version']} is not the "
                             f"declared seed {version})")
                continue
            return cand, None
        return None, (
            f"{entry['name']}: the explicitly supplied base interpreter is unusable; "
            f"refusing to fall back to a discovered one.\n"
            + "\n".join(f"      - {t}" for t in tried) + "\n"
            f"      fix: point {BASE_PY_ENV_VAR} / {BASE_MAP_ENV_VAR} at a working "
            f"Python {version or pmin}, or unset it to allow discovery.")

    if not version:
        return None, (f"{entry['name']}: manifest declares neither 'base' nor python.min, "
                      f"so no seed interpreter can be chosen.")

    tried = []
    for source, cand in discovered:
        if not cand.is_file():
            continue
        info = interpreter_info(cand)
        if not info["runs"]:
            tried.append(f"{source} -> {cand} (does not run: {info['error']})")
            continue
        if info["version_tuple"] != parse_version(version):
            continue
        if not satisfies(info["version_tuple"], pmin, pmax):
            tried.append(f"{source} -> {cand} (Python {info['version']} outside "
                         f"[{pmin or '-'}, {pmax or '-'}))")
            continue
        return cand, None
    return None, (
        f"{entry['name']}: requires a Python {version} seed "
        f"(range [{pmin or '-'}, {pmax or '-'})) and none was found.\n"
        + ("\n".join(f"      - {t}" for t in tried) if tried else "")
        + f"\n      fix: set {BASE_MAP_ENV_VAR}='{version}=<path to python.exe>' "
          f"or {BASE_PY_ENV_VAR}=<path>, or install Python {version}.")


def preflight(manifest: dict, entries: list[dict],
              base_map: dict[str, Path] | None = None) -> tuple[dict, list[str]]:
    """Resolve every selected project's base before any side effect (R4)."""
    resolved, problems = {}, []
    for entry in entries:
        path, reason = resolve_base_python(manifest, entry, base_map)
        if reason:
            problems.append(reason)
        else:
            resolved[entry["name"]] = path
    return resolved, problems


# --------------------------------------------------------------------------- #
# locks (R6)
# --------------------------------------------------------------------------- #
def _logical_requirements(path: Path, _seen: set | None = None) -> list[str]:
    """Requirement lines with comments stripped, backslash continuations joined
    and ``-r`` includes expanded, so lock semantics survive inspection."""
    _seen = _seen if _seen is not None else set()
    path = Path(path).resolve()
    if path in _seen or not path.is_file():
        return []
    _seen.add(path)
    try:
        raw = path.read_bytes()
    except OSError:
        return []
    text = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("latin-1", "replace")

    out: list[str] = []
    buffer = ""
    for line in text.splitlines():
        stripped = line.split(" #", 1)[0].rstrip()
        if not stripped.strip():
            continue
        if stripped.lstrip().startswith("#"):
            continue
        buffer += stripped
        if buffer.rstrip().endswith("\\"):
            buffer = buffer.rstrip()[:-1]
            continue
        logical = buffer.strip()
        buffer = ""
        if not logical:
            continue
        m = re.match(r"^\s*(?:-r|--requirement)\s+(.+?)\s*$", logical)
        if m:
            inc = m.group(1).strip().strip('"').strip("'")
            out += _logical_requirements(path.parent / inc, _seen)
            continue
        out.append(logical)
    if buffer.strip():
        out.append(buffer.strip())
    return out


_REQ_PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*==\s*([^\s;,]+)")
# PEP 508 direct reference, e.g. "mypkg @ file:///H:/wheelhouse/mypkg-1.0.0-py3-none-any.whl"
# pip freeze prints these in the same shape, so they are compared by wheel name.
_REQ_URL_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*@\s*(\S+)")


def _url_key(url: str) -> str:
    """Stable comparison key for a direct reference (wheel/file name, lowercased)."""
    tail = url.rstrip("/").rsplit("/", 1)[-1].split("#", 1)[0]
    return f"url:{tail.lower()}"


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def lock_pins(path: Path) -> dict[str, str]:
    """Exact ``name -> version`` pins of a requirements/lock file.

    Direct references (``name @ url``) are kept as ``url:<file name>`` so a lock
    that installs from a local or archived artifact can still be audited against
    the installed set.
    """
    pins: dict[str, str] = {}
    for logical in _logical_requirements(path):
        if logical.lstrip().startswith(("-", "--")):
            continue
        m = _REQ_PIN_RE.match(logical)
        if m:
            pins[normalize_name(m.group(1))] = m.group(2)
            continue
        m = _REQ_URL_RE.match(logical)
        if m:
            pins[normalize_name(m.group(1))] = _url_key(m.group(2))
    return pins


def installed_pins(py: Path, cwd: Path, log_path: Path | None = None,
                   timeout: int = 300) -> tuple[dict[str, str], Result]:
    """Installed ``name -> version`` for a project environment.

    ``--all`` is required: plain ``pip freeze`` hides the packaging toolchain
    (``pip``/``setuptools``/``wheel``), yet ``setuptools`` is a genuine *active*
    runtime requirement of some packages (e.g. in the 4D-STEM and 010 closures).
    Hiding it would make a correctly pinned lock look like it has a missing
    package.
    """
    res = run([py, "-X", "utf8", "-m", "pip", "freeze", "--all"], cwd=cwd,
              timeout=timeout, log_path=log_path)
    pins: dict[str, str] = {}
    for line in (res.out or "").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("-"):
            continue
        m = _REQ_PIN_RE.match(s)
        if m:
            pins[normalize_name(m.group(1))] = m.group(2)
            continue
        m = _REQ_URL_RE.match(s)
        if m:
            pins[normalize_name(m.group(1))] = _url_key(m.group(2))
    return pins, res


def compare_pins(locked: dict[str, str], installed: dict[str, str]) -> dict:
    missing = sorted(k for k in locked if k not in installed)
    wrong = sorted((k, locked[k], installed[k]) for k in locked
                   if k in installed and installed[k] != locked[k])
    extra = sorted(k for k in installed if k not in locked)
    return {"missing": missing, "mismatched": wrong, "extra": extra,
            "ok": not missing and not wrong, "locked_count": len(locked),
            "installed_count": len(installed)}


def write_constraints(path: Path, pins: dict[str, str]) -> Path:
    """Pure ``name==version`` constraints from a lock, hash lines dropped.

    Direct references (``name @ url``) are skipped: they cannot be expressed as a
    version constraint, and the lock keeps installing them. The lock file itself
    is never modified; this only gives pip a way to keep an overlay (pytest and
    friends) from upgrading packages the lock already pins.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{name}=={version}" for name, version in sorted(pins.items())
             if not str(version).startswith("url:")]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def closure_report_command(py: Path, locked: dict[str, str]) -> list[str]:
    """Command that runs the shared closure probe for ``locked``."""
    import json as _json
    return [str(py), "-X", "utf8", str(TOOLS_DIR / "_closure_probe.py"),
            _json.dumps(sorted(normalize_name(k) for k in locked))]


# Every key the probe must return. A reply missing any of them is a FAILED
# probe, never "nothing is missing" (a truncated/failed subprocess must not be
# able to make an environment look complete).
CLOSURE_PROBE_KEYS = ("packaging", "roots_installed", "roots_missing",
                      "roots_mismatch", "unpinned", "unresolved", "unsatisfied",
                      "parse_errors", "inactive")


def parse_closure_probe(stdout: str, locked: dict[str, str]) -> dict:
    """Turn raw probe stdout into a coverage verdict.

    ``complete`` is True only when the probe itself succeeded *and* every active
    transitive requirement is declared, installed and satisfied. Anything else
    (probe failure, schema mismatch, unresolved/unsatisfied dependency) yields
    ``complete`` False or None -- never True.
    """
    import json as _json
    lines = [line for line in (stdout or "").splitlines() if line.strip()]
    if not lines:
        return {"error": "closure probe produced no output", "complete": None}
    try:
        data = _json.loads(lines[-1])
    except ValueError as exc:
        return {"error": f"closure probe output is not JSON: {exc}", "complete": None}
    if not isinstance(data, dict):
        return {"error": "closure probe output is not an object", "complete": None}
    missing_keys = [k for k in CLOSURE_PROBE_KEYS if k not in data]
    if missing_keys:
        return {"error": f"closure probe schema mismatch (missing {missing_keys})",
                "complete": None}
    if data.get("error"):
        return {"error": f"closure probe failed: {data['error']}", "complete": None}
    if not data.get("packaging"):
        return {"error": "packaging not importable in the audited interpreter",
                "complete": None}

    unpinned = list(data.get("unpinned") or [])
    unresolved = list(data.get("unresolved") or [])
    unsatisfied = list(data.get("unsatisfied") or [])
    roots_missing = list(data.get("roots_missing") or [])
    roots_mismatch = list(data.get("roots_mismatch") or [])
    parse_errors = list(data.get("parse_errors") or [])

    if parse_errors:
        return {"error": f"unparseable requirement(s): {parse_errors[:3]}",
                "complete": None, "parse_errors": parse_errors}
    return {
        "complete": not (unpinned or unresolved or unsatisfied
                         or roots_missing or roots_mismatch),
        "unpinned": unpinned,
        # A declared-but-absent or version-conflicting dependency is a broken
        # environment (pip check territory), not merely an unpinned one.
        "broken": sorted(set(roots_missing + unresolved
                             + [row[0] for row in unsatisfied] + roots_mismatch)),
        "unsatisfied": unsatisfied,
        "inactive": list(data.get("inactive") or []),
        "installed_roots": len(data.get("roots_installed") or {}),
        "roots_missing": roots_missing,
        "roots_mismatch": roots_mismatch,
    }


def is_reproducible(entry: dict) -> bool:
    """Whether installation is driven by a frozen lock (declared pins only).

    NOTE: this is **not** a claim that the full dependency closure is frozen.
    Several locks pin only their direct dependencies (R10), so a ``True`` here
    means "the declared packages resolve to pinned versions", not "every
    transitive dependency is pinned". Use :func:`reproducibility_coverage` when
    the distinction matters.
    """
    return bool(entry.get("install", {}).get("reproducible")) and bool(install_chain(entry))


def reproducibility_coverage(entry: dict, *, closure_complete: bool | None = None) -> dict:
    """Describe *how much* reproducibility a project actually has.

    ``declared-pins``  – the install chain exists and its pins are the ones
                         actually installed (what ``is_reproducible`` asserts).
    ``full-closure``   – additionally, every transitive runtime/install
                         dependency is pinned (only known once measured).
    """
    has_lock = bool(install_chain(entry))
    level = "declared-pins" if has_lock else "none"
    if has_lock and closure_complete is True:
        level = "full-closure"
    elif has_lock and closure_complete is None:
        level = "declared-pins-unmeasured"
    return {
        "declared": is_reproducible(entry),
        "has_lock": has_lock,
        "closure_measured": closure_complete is not None,
        "closure_complete": closure_complete,
        "level": level,
    }


def install_chain(entry: dict) -> list[str]:
    return list(entry.get("install", {}).get("chain") or [])


def primary_source(entry: dict) -> str | None:
    chain = install_chain(entry)
    return chain[0] if chain else None


def sha256_file(path: Path) -> str | None:
    try:
        path = Path(path)
        if not path.is_file():
            return None
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def hash_many(base: Path, rels: list[str]) -> dict[str, str | None]:
    return {rel: sha256_file(Path(base) / rel) for rel in rels}


def pip_install(py: Path, args: list[str], cwd: Path, index: str | None = None,
                timeout: int = DEFAULT_TIMEOUT, log_path: Path | None = None) -> Result:
    cmd = [py, "-X", "utf8", "-m", "pip", "install",
           "--disable-pip-version-check", "--no-input"]
    if index:
        cmd += ["-i", index]
    cmd += list(args)
    return run(cmd, cwd=cwd, timeout=timeout, log_path=log_path)


# --------------------------------------------------------------------------- #
# test overlay (never modifies the runtime lock)
# --------------------------------------------------------------------------- #
def pytest_spec(manifest: dict, entry: dict) -> str:
    """The overlay requirement, validated as a real spec (e.g. ``pytest==8.4.2``).

    ``pytest`` / ``pytest==1.2.3`` / ``pytest>=8`` are accepted; a bare version
    such as ``8.4.2`` is not a requirement and is rejected.
    """
    spec = str((manifest.get("test_overlay") or {}).get("pytest") or "pytest").strip()
    if re.fullmatch(r"[0-9][0-9A-Za-z.*+!-]*", spec):
        raise ManifestError(
            f"[env] test_overlay.pytest is a bare version, not a requirement spec: "
            f"{spec!r} (expected e.g. 'pytest==8.4.2')")
    if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*(\[[^\]]*\])?([=<>!~]=?[^\s,;]+)?", spec):
        raise ManifestError(
            f"[env] test_overlay.pytest is not a valid requirement spec: {spec!r} "
            f"(expected e.g. 'pytest==8.4.2')")
    return spec


def declared_pytest_constraint(entry: dict) -> str | None:
    req = project_dir(entry) / "requirements.txt"
    if not req.is_file():
        return None
    try:
        text = req.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(r"^\s*pytest\s*([=<>!~][^\s#]*)", text, re.M)
    return m.group(1).strip() if m else None


def constraint_allows(constraint: str | None, version: tuple | None) -> bool:
    """Small evaluator for the constraint shapes used by these projects.

    Handles single-component bounds such as ``<9`` as well as ``>=8,<10``.
    """
    if not constraint or version is None:
        return True
    ver = (version[0], version[1])
    for part in constraint.split(","):
        m = re.match(r"\s*(>=|<=|==|!=|>|<)\s*(\d+(?:\.\d+)*)", part)
        if not m:
            continue
        op = m.group(1)
        bits = [int(x) for x in m.group(2).split(".")]
        bound = (bits[0], bits[1] if len(bits) > 1 else 0)
        ok = {">=": ver >= bound, "<=": ver <= bound, "==": ver == bound,
              "!=": ver != bound, ">": ver > bound, "<": ver < bound}[op]
        if not ok:
            return False
    return True


# --------------------------------------------------------------------------- #
# contract / CI matrix (R7)
# --------------------------------------------------------------------------- #
def project_python(entry: dict) -> str | None:
    """The interpreter version this project's environment must use.

    CI derives its matrix ``py`` from exactly this value, so a manifest change
    cannot leave CI on a stale interpreter.
    """
    return requested_version(None, entry)


def test_commands(entry: dict) -> list[str]:
    out = []
    for spec in entry.get("tests") or []:
        argv = [str(a) for a in (spec.get("argv") or [])]
        if argv:
            out.append("python " + " ".join(argv))
    return out


def contract(entry: dict) -> dict:
    proj = project_dir(entry)
    source = primary_source(entry)
    test_venv = (entry.get("test") or {}).get("venv") or entry["venv"]
    return {
        "name": entry["name"],
        "dir": entry["dir"],
        "python": project_python(entry),
        "python_range": [entry.get("python", {}).get("min"),
                         entry.get("python", {}).get("max")],
        "base": entry.get("base"),
        "venv": entry["venv"],
        "venvs_to_create": entry.get("venvs_to_create") or [entry["venv"]],
        "test_venv": test_venv,
        "install_source": source,
        "install_source_exists": bool(source and (proj / source).is_file()),
        "install_chain": install_chain(entry),
        "lock_exists": {name: (proj / name).is_file() for name in install_chain(entry)},
        "reproducible": is_reproducible(entry),
        "reproducibility": reproducibility_coverage(entry),
        "editable": bool(entry.get("editable")),
        "editable_args": (entry.get("editable") or {}).get("args") or [],
        "test_deps": list(entry.get("test_deps") or []),
        "verify_imports": list(entry.get("verify_imports") or []),
        "test_commands": test_commands(entry),
        "test_labels": [s.get("label", "?") for s in entry.get("tests") or []],
    }


def derive_ci_rows(manifest: dict) -> list[dict]:
    """Rows CI needs: only what CI cannot derive at runtime (identity + version)."""
    rows = []
    for entry in project_list(manifest):
        ci = entry.get("ci")
        if not ci:
            continue
        row = {"name": ci["name"], "dir": entry["dir"]}
        version = project_python(entry)
        if version:
            row["py"] = version
        if ci.get("ffmpeg"):
            row["ffmpeg"] = True
        rows.append(row)
    return rows


_ROW_RE = re.compile(r"^\s*-\s*\{(.*)\}\s*$")
_KV_RE = re.compile(r"(\w+)\s*:\s*(?:'([^']*)'|\"([^\"]*)\"|(\w+))")


def parse_ci_rows(ci_path: Path) -> tuple[list[dict], str | None]:
    if not ci_path.is_file():
        return [], f"ci.yml not found: {ci_path}"
    text = ci_path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        m = _ROW_RE.match(line)
        if not m or "dir:" not in m.group(1):
            continue
        row: dict = {}
        for kv in _KV_RE.finditer(m.group(1)):
            key = kv.group(1)
            value = kv.group(2) if kv.group(2) is not None else (
                kv.group(3) if kv.group(3) is not None else kv.group(4))
            if value in ("true", "True"):
                row[key] = True
            elif value in ("false", "False"):
                row[key] = False
            else:
                row[key] = value
        if row:
            rows.append(row)
    if not rows:
        return [], "no matrix rows with 'dir:' parsed from ci.yml"
    return rows, None


DERIVED_FIELDS = ("test", "deps", "editable")


def check_ci_matrix(manifest: dict, ci_path: Path) -> tuple[bool, list[str]]:
    """Verify ci.yml against the manifest, including the real contract fields."""
    messages: list[str] = []
    ci_path = Path(ci_path)
    expected = derive_ci_rows(manifest)
    actual, err = parse_ci_rows(ci_path)
    if err:
        return False, [err]

    exp_by_dir = {r["dir"]: r for r in expected}
    act_by_dir: dict[str, dict] = {}
    for row in actual:
        d = row["dir"]
        if d in act_by_dir:
            messages.append(f"duplicate ci.yml matrix row for: {d}")
        act_by_dir[d] = row

    for d in sorted(set(exp_by_dir) - set(act_by_dir)):
        messages.append(f"project missing from ci.yml matrix: {d}")
    for d in sorted(set(act_by_dir) - set(exp_by_dir)):
        messages.append(f"ci.yml matrix lists a project the manifest does not: {d}")

    for d in sorted(set(exp_by_dir) & set(act_by_dir)):
        exp, act = exp_by_dir[d], act_by_dir[d]
        for key in sorted(set(exp) | set(act)):
            if key in DERIVED_FIELDS:
                continue
            if exp.get(key) != act.get(key):
                messages.append(f"{d}: {key} differs (manifest={exp.get(key)!r}, "
                                f"ci.yml={act.get(key)!r})")
        entry = manifest["_by_dir"][d]
        for field in DERIVED_FIELDS:
            if field in act:
                if field == "editable":
                    messages.append(
                        f"{d}: ci.yml pins 'editable' but it is derived from the "
                        f"manifest (entry.editable={bool(entry.get('editable'))}); "
                        f"remove the duplicate source")
                else:
                    messages.append(
                        f"{d}: ci.yml pins '{field}' which is derived from the "
                        f"manifest; remove the duplicate source so a manifest "
                        f"change cannot silently diverge")

    text = ci_path.read_text(encoding="utf-8", errors="replace")
    if ".tools/ci-project.py" not in text:
        messages.append("ci.yml does not delegate install/test to .tools/ci-project.py "
                        "(the manifest-driven runner)")
    for marker, why in (
            ("verify-envs.py --check-matrix", "matrix guard step"),
            ("matrix.dir", "per-project working directory"),
    ):
        if marker not in text:
            messages.append(f"ci.yml is missing the {why} ({marker!r})")
    for action, why in (("install", "manifest-driven install step"),
                        ("test", "manifest-driven test step")):
        if not re.search(rf'ci-project\.py"?\s+{action}\b', text):
            messages.append(f"ci.yml is missing the {why} "
                            f"(no 'ci-project.py {action}' invocation)")
    # CI must not install into the job interpreter itself: every project step has
    # to go through the provisioner, otherwise a lock is bypassed. Comments are
    # ignored — they legitimately quote the old commands while explaining them.
    code_lines = [l for l in text.splitlines() if not l.lstrip().startswith("#")]
    code = "\n".join(code_lines)
    for bad, why in (
            ("pip install -r", "a raw 'pip install -r' bypasses the provisioner"),
            ("pip install -e .", "a raw editable install bypasses the manifest"),
            ("pip install pytest", "a bare pytest install ignores the lock/overlay"),
    ):
        if bad in code:
            messages.append(f"ci.yml contains {why} ({bad!r})")

    default_ci = default_py_from_ci(ci_path)
    default_manifest = default_py_from_manifest(manifest)
    if default_ci and default_ci != default_manifest:
        messages.append(f"default python-version differs (manifest={default_manifest}, "
                        f"ci.yml={default_ci})")

    for entry in project_list(manifest):
        proj = project_dir(entry)
        for name in install_chain(entry):
            if not (proj / name).is_file():
                messages.append(f"{entry['dir']}: install source missing on disk: {name}")
        declared = entry.get("venvs_to_create") or [entry["venv"]]
        test_venv = (entry.get("test") or {}).get("venv") or entry["venv"]
        if test_venv not in declared:
            messages.append(f"{entry['dir']}: test venv {test_venv!r} is never "
                            f"provisioned (provisioned: {declared})")
        for spec in entry.get("tests") or []:
            argv = spec.get("argv") or []
            if argv and not str(argv[0]).startswith("-"):
                if not (proj / str(argv[0])).is_file():
                    messages.append(f"{entry['dir']}: test script missing: {argv[0]}")

    return (not messages), messages


def default_py_from_ci(ci_path: Path) -> str | None:
    ci_path = Path(ci_path)
    if not ci_path.is_file():
        return None
    text = ci_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"python-version:\s*\$\{\{\s*matrix\.py\s*\|\|\s*'([^']+)'\s*\}\}", text)
    return m.group(1) if m else None


def default_py_from_manifest(manifest: dict) -> str:
    floors = [r["py"] for r in derive_ci_rows(manifest) if r.get("py")]
    if not floors:
        return "3.10"
    counts: dict[str, int] = {}
    for f in floors:
        counts[f] = counts.get(f, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def print_table(rows: list[list[str]], header: list[str]) -> None:
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def log(msg: str) -> None:
    print(msg, flush=True)
    sys.stdout.flush()


def ensure_utf8_stdio() -> None:
    """Force UTF-8 on our own stdout/stderr.

    Project names and paths are Chinese, and the Windows console defaults to a
    legacy code page; without this, captured output is mojibake.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError, OSError):
            pass
