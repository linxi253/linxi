# -*- coding: utf-8 -*-
"""Audit the workspace for environment-isolation violations (manifest-driven).

Scope
-----
This tool does **not** walk the workspace. It inspects:

* every project declared in ``.tools/projects.json`` -- including ones whose
  directory is absent, which are reported explicitly rather than skipped, and
* launcher scripts (``*.bat`` / ``*.cmd`` / ``*.ps1``) inside those project
  directories, pruning excluded subtrees **before** descending into them.

Excluded by directory pruning (never entered, never read): ``.venv`` and every
``.venv.*`` backup, history trees, data/experiment trees, build/dist/cache
outputs, ``site-packages`` and ``.review-tmp`` task artefacts. The project's own
source directories are always scanned, so a real bare-``python`` launcher in the
source is still reported.

Checks
------
1. Declared projects: directory present, lock present, venv present, venv
   isolated and runnable, Python inside the declared range.
2. Declared pins vs installed versions (never an equality test against the full
   ``pip freeze``: a test overlay legitimately adds packages).
3. Dependency-closure coverage, measured by a single metadata probe
   (``_closure_probe.py``, shared through ``_envcommon``). Only a fully
   successful probe that finds every active transitive requirement declared,
   installed and satisfied yields ``closure-complete``.
4. Launcher scripts must not fall back to a bare ``python`` from PATH.

Exit code 0 = clean, 1 = violations. ``--strict`` additionally fails when a
declared project has no venv (a fresh checkout); default mode reports that as
``info`` because "not provisioned here" is not a broken convention.

Base interpreter: a base explicitly requested on the command line or through
``AIFORTEM_BASE_PY`` / ``AIFORTEM_PYTHONS`` must exist, run and match the
declared version; there is no fallback to a discovered interpreter. Auditing
already-provisioned environments does not require any base to be present, and
this tool never installs anything.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent


def _load_envcommon():
    spec = importlib.util.spec_from_file_location("ec_isolation", TOOLS / "_envcommon.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ec_isolation"] = module
    spec.loader.exec_module(module)
    return module


ec = _load_envcommon()

ROOT = ec.REPO_ROOT                       # default; overridden by --manifest

SCRIPT_SUFFIXES = (".bat", ".cmd", ".ps1")

# Directory names never descended into. Prefixes cover renamed/timestamped
# variants of an environment or build tree, e.g. ``.venv.pre-rebuild-20261003``,
# ``.venv-backup-py311-20261004``, ``dist.old``. Only the *directory* is pruned:
# ordinary source directories are still scanned, so a genuine bare-``python``
# launcher in ``src/`` is still reported.
PRUNE_NAMES = {
    ".venv", ".venv-build", ".venv-run", "venv", "env",
    "site-packages", "node_modules", "__pycache__",
    "build", "dist", "release", "_internal", "cache", ".cache",
    "data", "dataset", "datasets", "runs", "outputs",
    "08-历史版本", "legacy", ".review-tmp", ".codex_tmp", ".runtime",
    ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
}
PRUNE_PREFIXES = (".venv.", ".venv-", "venv.", "venv-", "dist.", "build.")

BARE_PYTHON = re.compile(r"(?<![\w\\./\-$])python(?:\.exe)?(?![\w.\-])")
SAFE_LINE = re.compile(
    r"^\s*(rem|::|#)"
    r"|^\s*echo\b"
    r"|\$python"
    r"|where\s+python"
    r"|python\s+-m\s+venv"
    r"|-m\s+venv"
    r"|BOOTSTRAP\w*"
    r"|Join-Path.*python"
    r"|-eq\s*['\"]python"
    r"|python\s+-c\s+['\"]import sys",
    re.IGNORECASE,
)


class Findings:
    """Collects problems (fail) and notes (info) separately."""

    def __init__(self) -> None:
        self.problems: list[str] = []
        self.notes: list[str] = []

    def fail(self, message: str) -> None:
        self.problems.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 300):
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace",
                              cwd=str(cwd) if cwd else None, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(cmd, 127, f"[{type(exc).__name__}] {exc}\n", None)


def decode(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "gbk", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", "replace")


def display(path: Path, root: Path) -> str:
    """Path relative to ``root`` when possible, else absolute (never raises)."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# manifest-driven scope
# ---------------------------------------------------------------------------
def manifest_root(manifest: dict) -> Path:
    """Root the manifest was loaded from (NOT the tool's own location).

    Falls back to the loader's active root for a synthetic manifest dict that
    was constructed in-process rather than read from disk.
    """
    recorded = manifest.get("_root")
    if recorded:
        return Path(recorded)
    path = manifest.get("_path")
    if path:
        return Path(path).resolve().parent.parent
    return Path(ec.active_root())


def iter_scope_dirs(manifest: dict):
    """Yield ``(entry, project_dir, exists)`` for EVERY declared project."""
    root = manifest_root(manifest)
    for entry in ec.project_list(manifest):
        proj = root / entry["dir"].replace("/", os.sep)
        yield entry, proj, proj.is_dir()


def prune(dirs: list[str]) -> list[str]:
    """Sub-directory names to descend into (excluded trees removed up front)."""
    return [d for d in dirs
            if d not in PRUNE_NAMES
            and not any(d.startswith(p) for p in PRUNE_PREFIXES)]


def iter_launcher_scripts(proj: Path):
    """Walk ``proj`` for launcher scripts, pruning excluded trees before entry."""
    for current, dirnames, filenames in os.walk(proj):
        dirnames[:] = sorted(prune(dirnames))
        for name in sorted(filenames):
            if Path(name).suffix.lower() in SCRIPT_SUFFIXES:
                yield Path(current) / name


# ---------------------------------------------------------------------------
# base interpreter
# ---------------------------------------------------------------------------
def resolve_base(explicit: str | None, manifest: dict | None = None
                 ) -> tuple[Path | None, str | None]:
    """Resolve a *requested* base interpreter; never fall back.

    Uses ``_envcommon``'s declared-seed/range rules rather than a bare
    "is it a file / does it run" test: the requested interpreter must match the
    version requested by the projects it is meant to cover. There is no
    discovery and no fallback -- an unusable explicit value is an error.

    Returns ``(path, error)``; ``(None, None)`` means "nothing was requested",
    which is the normal case when merely auditing already-provisioned venvs.
    """
    value = (explicit or os.environ.get("AIFORTEM_BASE_PY", "")).strip()
    if not value:
        return None, None
    candidate = Path(value).expanduser()
    if not candidate.is_file():
        return None, (f"explicitly requested base interpreter does not exist: {candidate} "
                      "(fix AIFORTEM_BASE_PY / --base-python, or omit it)")
    info = ec.interpreter_info(candidate)
    if not info.get("runs"):
        return None, (f"explicitly requested base interpreter exists but is not runnable: "
                      f"{candidate} ({info.get('error')}); refusing to fall back")

    versions = requested_versions(manifest)
    if versions:
        declared = versions.get(info.get("version_tuple"))
        if declared is None:
            listed = ", ".join(sorted(v for v in versions.values() if v))
            return None, (
                f"explicitly requested base interpreter {candidate} runs Python "
                f"{info.get('version')}, which is not one of the versions the manifest "
                f"declares ({listed}); refusing to fall back. Point it at a declared "
                f"version, or omit it.")
    declared_minor = minor_of(info.get("version_tuple"))
    if declared_minor is None:
        return None, (f"explicitly requested base interpreter {candidate} reports an "
                      f"unparseable version ({info.get('version')!r})")
    return candidate, None


def minor_of(version: tuple | None) -> str | None:
    if not version or len(version) < 2:
        return None
    return f"{version[0]}.{version[1]}"


def requested_versions(manifest: dict | None) -> dict[tuple | None, str]:
    """Map ``version_tuple -> declared version`` across the manifest."""
    if not manifest:
        return {}
    out: dict[tuple | None, str] = {}
    for entry in ec.project_list(manifest):
        version = ec.requested_version(manifest, entry)
        if version:
            out[ec.parse_version(version)] = version
    return out


def is_default_key(key: str) -> bool:
    """``default`` / ``default310`` … are fallbacks, not version keys.

    Mirrors :func:`_envcommon.explicit_interpreters`, which treats any key
    starting with ``default`` as a fallback for projects whose requested version
    has no explicit entry.
    """
    return key.startswith("default")


def select_map_entry(entry: dict, manifest: dict, base_map: dict
                     ) -> tuple[str, Path | None] | None:
    """Which mapping entry actually covers ``entry`` (``None`` = not covered).

    A version-specific entry wins; a ``default*`` entry applies only when there
    is no explicit entry for the project's requested version. This mirrors the
    selection order in ``_envcommon.explicit_interpreters`` so that adding a
    ``default`` cannot mis-apply it to a project that has its own entry.
    """
    version = ec.requested_version(manifest, entry)
    if version and version in base_map:
        return (version, base_map[version])
    for key, path in base_map.items():
        if is_default_key(key):
            return (version or "-", path)
    return None


def check_base_map(manifest: dict | None = None
                   ) -> tuple[list[tuple[str, Path]], str | None]:
    """Validate every entry of ``AIFORTEM_PYTHONS`` that was explicitly given.

    Version keys must name a version and point at an interpreter of exactly that
    version. ``default*`` keys are supported (they are the manifest's documented
    fallback) and are checked **per project they actually cover**: the chosen
    interpreter must satisfy that project's declared seed and min/max range. A
    project that has its own version-specific entry is never validated against
    the default.
    """
    raw = os.environ.get("AIFORTEM_PYTHONS", "").strip()
    if not raw:
        return [], None
    try:
        base_map = ec.parse_base_map(raw)
    except (Exception, SystemExit) as exc:         # ManifestError is a SystemExit
        return [], f"AIFORTEM_PYTHONS is unusable: {exc}"
    if not base_map:
        return [], (f"AIFORTEM_PYTHONS was set to {raw!r} but no 'version=path' entry "
                    "could be read; refusing to claim it was verified")

    checked: list[tuple[str, Path]] = []
    problems: list[str] = []

    # 1) every explicit entry must exist, run, and self-describe correctly
    for key, path in sorted(base_map.items()):
        if is_default_key(key):
            continue                                # validated per covered project below
        declared = ec.parse_version(key)
        if declared is None:
            problems.append(f"AIFORTEM_PYTHONS key {key!r} is neither a version nor a "
                            "'default' fallback")
            continue
        if not path.is_file():
            problems.append(f"AIFORTEM_PYTHONS {key} -> {path} (does not exist)")
            continue
        info = ec.interpreter_info(path)
        if not info.get("runs"):
            problems.append(f"AIFORTEM_PYTHONS {key} -> {path} "
                            f"(does not run: {info.get('error')})")
            continue
        if info.get("version_tuple") != declared:
            problems.append(f"AIFORTEM_PYTHONS {key} -> {path} "
                            f"(runs Python {info.get('version')}, key says {key})")
            continue
        checked.append((key, path))

    # 2) each declared project must be covered by an entry that actually fits it
    if manifest:
        info_cache: dict[str, dict] = {}
        for entry in ec.project_list(manifest):
            selection = select_map_entry(entry, manifest, base_map)
            if selection is None:
                continue
            label, path = selection
            if path is None:
                continue
            key = f"{entry['name']}"
            if not path.is_file():
                problems.append(f"{key}: the entry covering it ({label} -> {path}) "
                                "does not exist")
                continue
            cache_key = str(path)
            if cache_key not in info_cache:
                info_cache[cache_key] = ec.interpreter_info(path)
            info = info_cache[cache_key]
            if not info.get("runs"):
                problems.append(f"{key}: the entry covering it ({label} -> {path}) "
                                f"does not run: {info.get('error')}")
                continue
            version = ec.requested_version(manifest, entry)
            pmin = (entry.get("python") or {}).get("min")
            pmax = (entry.get("python") or {}).get("max")
            if not ec.satisfies(info.get("version_tuple"), pmin, pmax):
                problems.append(
                    f"{key}: the entry covering it ({label} -> Python "
                    f"{info.get('version')}) is outside the declared range "
                    f"[{pmin or '-'}, {pmax or '-'})")
                continue
            if version and info.get("version_tuple") != ec.parse_version(version):
                problems.append(
                    f"{key}: the entry covering it ({label} -> Python "
                    f"{info.get('version')}) is not the declared seed {version}"
                    + (" (add an explicit '"
                       f"{version}=<path>' entry, or fix the default)"))
                continue
            if (label, path) not in checked:
                checked.append((label, path))

    if problems:
        return checked, ("AIFORTEM_PYTHONS does not cover the declared projects; "
                         "refusing to fall back:\n"
                         + "\n".join(f"      - {p}" for p in problems))
    return checked, None


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def check_projects(manifest: dict, findings: Findings, strict: bool) -> None:
    root = manifest_root(manifest)
    print("[1] Declared project environments (manifest scope)")
    for entry, proj, exists in iter_scope_dirs(manifest):
        name = entry["name"]
        rel = display(proj, root)
        venv_name = entry["venv"]
        py = proj / venv_name / "Scripts" / "python.exe"
        lock_names = ec.install_chain(entry)
        locks = [proj / n for n in lock_names if (proj / n).is_file()]

        if not exists:
            # Must never vanish from the report: a declared project that is not
            # checked out is a real finding under --strict.
            if strict:
                findings.fail(f"{rel}: declared project directory is missing (strict)")
                print(f"    FAIL  {rel}: project directory missing (strict)")
            else:
                print(f"    missing {rel}: declared in the manifest but not present "
                      "on this checkout")
            continue

        if not py.is_file():
            if strict:
                findings.fail(f"{rel}: no venv (strict)")
                print(f"    FAIL  {rel}: no venv at {venv_name} (strict)")
            else:
                print(f"    info  {rel}: not provisioned on this clone (no {venv_name})")
            if not locks and lock_names:
                findings.fail(f"{rel}: declared install chain {lock_names} has no lock file")
                print(f"    FAIL  {rel}: missing lock {lock_names}")
            continue

        detail: list[str] = []
        cfg = proj / venv_name / "pyvenv.cfg"
        if not cfg.is_file():
            findings.fail(f"{rel}: missing pyvenv.cfg")
            detail.append("BROKEN")
        else:
            text = cfg.read_text(encoding="utf-8", errors="replace")
            tail = text.split("include-system-site-packages", 1)
            if len(tail) < 2 or "false" not in tail[1][:20].lower():
                findings.fail(f"{rel}: venv leaks system site-packages")
                detail.append("LEAKY")
            else:
                detail.append("isolated")

        info = ec.interpreter_info(py)
        if not info.get("runs"):
            findings.fail(f"{rel}: interpreter does not run ({info.get('error')})")
            detail.append("NO_RUN")
        else:
            detail.append(info.get("version") or "?")
            pmin = (entry.get("python") or {}).get("min")
            pmax = (entry.get("python") or {}).get("max")
            if not ec.satisfies(ec.parse_version(info.get("version")), pmin, pmax):
                findings.fail(f"{rel}: Python {info.get('version')} outside "
                              f"declared range {pmin}..{pmax}")
                detail.append("OUT_OF_RANGE")

        bad = any(d in ("BROKEN", "LEAKY", "NO_RUN", "OUT_OF_RANGE") for d in detail)
        lock_note = locks[0].name if locks else "NO LOCK"
        print(f"    {'FAIL ' if bad else 'ok   '} {rel}  [{' '.join(detail)}]  "
              f"lock={lock_note}")
        if not locks and lock_names:
            findings.fail(f"{rel}: no lock file among {lock_names}")


def closure_report(py: Path, locked: dict[str, str]) -> dict:
    """Run the shared metadata probe once and interpret it via ``_envcommon``.

    A single probe call: the interpreter is asked for the whole answer in one go
    (no fragile two-round protocol). Any failure -- non-zero exit, unparseable
    or schema-mismatched output, unavailable ``packaging`` -- yields
    ``complete=None`` (unmeasured), never ``True``.
    """
    command = ec.closure_report_command(py, locked)
    proc = run(command, cwd=py.parent.parent)
    if proc.returncode != 0:
        return {"error": f"closure probe exited {proc.returncode}: "
                         f"{(proc.stdout or '').strip()[:160]}", "complete": None}
    return ec.parse_closure_probe(proc.stdout or "", locked)


def check_locks(manifest: dict, findings: Findings) -> list[dict]:
    root = manifest_root(manifest)
    print("\n[2] Declared pins + dependency-closure coverage")
    summary: list[dict] = []
    for entry, proj, exists in iter_scope_dirs(manifest):
        name = entry["name"]
        rel = display(proj, root)
        if not exists:
            continue
        locks = [proj / n for n in ec.install_chain(entry) if (proj / n).is_file()]
        py = proj / entry["venv"] / "Scripts" / "python.exe"
        if not locks:
            continue
        if not py.is_file():
            print(f"    info  {rel}: lock present, venv not provisioned (pin check skipped)")
            summary.append({"project": name, "dir": rel, "status": "no-venv"})
            continue

        locked: dict[str, str] = {}
        for lock in locks:
            locked.update(ec.lock_pins(lock))
        installed, _ = ec.installed_pins(py, proj)
        comparison = ec.compare_pins(locked, installed)
        mismatched = comparison.get("mismatched") or {}
        missing = comparison.get("missing") or []

        if mismatched or missing:
            findings.fail(f"{rel}: declared pins differ from installed "
                          f"(mismatch={sorted(mismatched)[:4]}, missing={missing[:4]})")
            print(f"    FAIL  {rel}: pin mismatch={sorted(mismatched)[:4]} "
                  f"missing={missing[:4]}")
            summary.append({"project": name, "dir": rel, "status": "pin-mismatch",
                            "mismatch": mismatched, "missing": missing})
            continue

        closure = closure_report(py, locked)
        row = {"project": name, "dir": rel, "locked": len(locked),
               "installed": len(installed), "closure": closure}
        if closure.get("complete") is True:
            row["status"] = "closure-complete"
            print(f"    ok    {rel}: {len(locked)} declared pins match; "
                  "dependency closure fully pinned")
        elif closure.get("error"):
            row["status"] = "closure-unmeasured"
            print(f"    UNMEASURED {rel}: {len(locked)} declared pins match, but the "
                  f"closure could not be measured ({closure['error'][:80]})")
        elif closure.get("broken"):
            row["status"] = "broken"
            broken = closure["broken"]
            findings.fail(f"{rel}: dependency problem(s): {broken[:6]}")
            print(f"    FAIL  {rel}: installed dependency problem(s): {broken[:6]}")
        else:
            unpinned = closure.get("unpinned") or []
            findings.note(f"{rel}: {len(unpinned)} dependency(ies) installed correctly "
                          f"but not pinned ({', '.join(unpinned[:6])})")
            print(f"    info  {rel}: {len(locked)} declared pins match, closure "
                  f"INCOMPLETE ({len(unpinned)} unpinned: {', '.join(unpinned[:6])}) "
                  "- to be locked in S1d")
            row["status"] = "incomplete-closure"
        summary.append(row)
    return summary


def check_scripts(manifest: dict, findings: Findings) -> None:
    root = manifest_root(manifest)
    print("\n[3] Launcher scripts inside declared projects")
    hits = 0
    for entry, proj, exists in iter_scope_dirs(manifest):
        if not exists:
            continue
        for path in iter_launcher_scripts(proj):
            for i, line in enumerate(decode(path).splitlines(), 1):
                if not BARE_PYTHON.search(line) or SAFE_LINE.search(line):
                    continue
                findings.fail(f"{display(path, root)}:{i}: bare python")
                print(f"    FAIL  {display(path, root)}:{i}")
                print(f"          {line.strip()}")
                hits += 1
    if not hits:
        print("    ok    no bare-python fallbacks in declared projects")


def check_pth(manifest: dict, findings: Findings) -> None:
    root = manifest_root(manifest)
    print("\n[4] .pth files in declared venvs")
    bad = []
    for entry, proj, exists in iter_scope_dirs(manifest):
        if not exists:
            continue
        site_packages = proj / entry["venv"] / "Lib" / "site-packages"
        if not site_packages.is_dir():
            continue
        for p in sorted(site_packages.glob("*.pth")):
            if p.name.endswith(".orig"):
                continue
            if any(b >= 128 for b in p.read_bytes()):
                bad.append(p)
    for p in bad:
        findings.fail(f"non-ASCII .pth: {display(p, root)}")
        print(f"    FAIL  {display(p, root)}")
    if bad:
        print("          -> .tools/normalise-pth.py rewrites the target editable file")
    else:
        print("    ok    all .pth files in declared venvs are ASCII-only")


def main(argv: list[str] | None = None) -> int:
    global ROOT
    parser = argparse.ArgumentParser(description="Manifest-scoped environment audit")
    parser.add_argument("--strict", action="store_true",
                        help="also fail when a declared project is absent or has no venv")
    parser.add_argument("--base-python", default=None,
                        help="explicit base interpreter (missing/unrunnable is a hard error)")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    ec.ensure_utf8_stdio()
    manifest = ec.load_manifest(Path(args.manifest) if args.manifest else None)
    ROOT = manifest_root(manifest)
    findings = Findings()

    print(f"workspace : {ROOT}")
    print(f"scope     : {len(ec.project_list(manifest))} declared projects "
          f"(backups/.venv.* and history trees pruned)")
    print(f"mode      : {'strict' if args.strict else 'default'}")
    print("=" * 74)

    base_py, base_error = resolve_base(args.base_python, manifest)
    checked_map, map_error = check_base_map(manifest)
    if base_error:
        findings.fail(base_error)
        print(f"[0] base interpreter\n    FAIL  {base_error}")
    elif base_py is not None:
        print(f"[0] base interpreter\n    ok    requested base runs: {base_py}")
    if map_error:
        findings.fail(map_error)
        print(f"[0] base map\n    FAIL  {map_error}")
    elif checked_map:
        print(f"[0] base map\n    ok    AIFORTEM_PYTHONS entries verified: "
              f"{', '.join(v for v, _ in checked_map)}")

    check_projects(manifest, findings, args.strict)
    lock_summary = check_locks(manifest, findings)
    check_scripts(manifest, findings)
    check_pth(manifest, findings)

    print("=" * 74)
    by_status: dict[str, int] = {}
    for row in lock_summary:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    if by_status:
        print("lock coverage: " + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())))
    if findings.notes:
        print(f"notes: {len(findings.notes)}")
        for note in findings.notes[:25]:
            print(f"  - {note}")
    if args.json:
        ec.write_json(Path(args.json), {
            "workspace": str(ROOT),
            "scope": "manifest projects only (directory-pruned)",
            "strict": args.strict,
            "problems": findings.problems,
            "notes": findings.notes,
            "locks": lock_summary,
        })
    if findings.problems:
        print(f"RESULT: {len(findings.problems)} problem(s) found")
        return 1
    print("RESULT: clean - environment isolation convention is intact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
