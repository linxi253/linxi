# -*- coding: utf-8 -*-
"""Provision one isolated venv per tool project, *from its frozen lock*.

Reconcile round, R4/R6/R7
-------------------------
* R4 — the seed interpreter is resolved **per project** from the manifest. An
  explicitly supplied interpreter that is unusable aborts the run; discovery is
  only used when nothing explicit was supplied. Every selected project is
  preflighted before the first side effect.
* R6 — installing never writes a lock. There is deliberately **no automatic
  lock-update mode**: regenerating a lock must happen in a separate clean
  resolution environment, and its diff must be reviewed on its own (see
  ``环境隔离说明.md``).
* R7 — project list, venv, install source, test venv and test commands all come
  from ``.tools/projects.json``, the same manifest the verifier and CI read.

Usage::

    python .tools/provision-envs.py --list
    python .tools/provision-envs.py --dry-run
    python .tools/provision-envs.py --check
    python .tools/provision-envs.py --project 全整合 --project 010-STEM模拟
    python .tools/provision-envs.py --project 全整合 --with-tests
    python .tools/provision-envs.py --reinstall --project 全整合

``--dry-run`` short-circuits every mutating branch: it only probes read-only
state and prints the plan.

Exit codes: 0 ok, 1 failures, 2 usage/manifest, 3 blocked (preflight failed, or
a venv exists whose interpreter cannot run).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _envcommon as ec  # noqa: E402
import importlib.util as _ilu  # noqa: E402


def _load_sibling(name: str):
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = _ilu.spec_from_file_location(f"_tools_{name.replace('-', '_')}", path)
    module = _ilu.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _MissingPthNormaliser:
    """Stand-in used ONLY when normalise-pth.py is genuinely absent.

    It exists so the module can still be imported in a partial checkout. It does
    **not** make normalisation succeed: ``find_editable_pth`` raises, and because
    ``install_editable`` treats a declared editable that cannot be normalised as
    a failure, the caller sees a real error rather than a silent pass.

    (The tool tests copy the real helper; this fallback must never be the reason
    a test passes.)
    """

    PthNormaliseError = RuntimeError

    @staticmethod
    def normalise_target_pth(target, expected_src, *, dry_run=False, log=None):
        raise RuntimeError("normalise-pth.py is missing; cannot normalise the editable .pth")

    @staticmethod
    def find_editable_pth(site_packages, package_name):
        raise RuntimeError("normalise-pth.py is missing; cannot locate the editable .pth")

    @staticmethod
    def site_packages_of(python_exe):
        candidate = Path(python_exe).parent.parent / "Lib" / "site-packages"
        return candidate if candidate.is_dir() else None


try:
    normalise_pth = _load_sibling("normalise-pth")
except FileNotFoundError:
    normalise_pth = _MissingPthNormaliser

MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
FALLBACK_INDEX = "https://pypi.org/simple"

OK = "OK"
PRESENT = "PRESENT"
BROKEN = "BROKEN"
FAILED = "FAILED"
PLANNED = "PLANNED"
UNREPRODUCIBLE = "UNREPRODUCIBLE"


def venv_state(entry: dict) -> dict:
    py = ec.venv_python(entry)
    vdir = ec.venv_dir(entry)
    info = ec.interpreter_info(py)
    info["venv_dir"] = str(vdir)
    info["venv_dir_exists"] = vdir.is_dir()
    info["isolated"] = bool(info.get("prefix") and info.get("base_prefix")
                            and info["prefix"] != info["base_prefix"])
    return info


def pip_check(py: Path, cwd: Path, log_path: Path) -> ec.Result:
    return ec.run([py, "-X", "utf8", "-m", "pip", "check"], cwd=cwd, timeout=300,
                  log_path=log_path)


def lock_consistency(entry: dict, py: Path, proj: Path, log_path: Path) -> dict:
    source = ec.primary_source(entry)
    if not source:
        return {"checked": False, "reason": "no frozen lock declared",
                "ok": None, "source": None}
    lock = proj / source
    locked = ec.lock_pins(lock)
    if not locked:
        return {"checked": False, "reason": f"no pins parsed from {source}",
                "ok": None, "source": source}
    installed, res = ec.installed_pins(py, proj, log_path=log_path)
    cmp = ec.compare_pins(locked, installed)
    cmp.update({"checked": True, "source": source, "freeze_exit": res.code})
    return cmp


def verify_imports(entry: dict, py: Path, proj: Path, log_path: Path) -> dict:
    modules = list(entry.get("verify_imports") or [])
    if not modules:
        return {"checked": False, "ok": None, "modules": [], "missing": []}
    code = ("import importlib\n"
            "bad=[]\n"
            f"for m in {modules!r}:\n"
            "    try: importlib.import_module(m)\n"
            "    except Exception as e: bad.append(m+':'+type(e).__name__)\n"
            "print('BAD=' + ','.join(bad) if bad else 'OK')\n")
    res = ec.run([py, "-c", code], cwd=proj, timeout=600, log_path=log_path)
    verdict = res.out.strip().splitlines()[-1] if res.out.strip() else "?"
    if verdict.startswith("OK"):
        return {"checked": True, "ok": True, "modules": modules, "missing": [],
                "verdict": verdict, "exit": res.code}
    missing = [m for m in (verdict[4:].split(",") if verdict.startswith("BAD=") else []) if m]
    return {"checked": True, "ok": False, "modules": modules, "missing": missing,
            "verdict": verdict, "exit": res.code}


def install_from_chain(py: Path, entry: dict, proj: Path, log_path: Path,
                       allow_fallback: bool) -> tuple[bool, str, list]:
    chain = ec.install_chain(entry)
    if not chain:
        return True, "no frozen lock declared (unreproducible project)", []
    missing = [name for name in chain if not (proj / name).is_file()]
    if missing:
        return False, f"declared lock missing on disk: {', '.join(missing)}", []
    source = chain[0]
    indexes = [MIRROR, FALLBACK_INDEX] if allow_fallback else [MIRROR]
    cmds, last = [], None
    for index in indexes:
        res = ec.pip_install(py, ["-r", source], cwd=proj, index=index, log_path=log_path)
        cmds.append(" ".join(res.cmd))
        last = res
        if res.ok:
            return True, f"installed from {source} (index {index})", cmds
    return False, (f"pip install -r {source} failed (exit {last.code if last else '?'}); "
                   f"see log for full output"), cmds


def install_editable(py: Path, entry: dict, proj: Path, log_path: Path) -> tuple[bool, str, list]:
    spec = entry.get("editable")
    if not spec:
        return True, "", []
    args = list(spec.get("args") or ["-e", "."])
    res = ec.pip_install(py, args, cwd=proj, index=None, log_path=log_path)
    if not res.ok:
        return False, f"pip install {' '.join(args)} failed (exit {res.code})", [" ".join(res.cmd)]

    # setuptools writes the editable .pth in the ANSI code page; with Chinese
    # paths that is not valid UTF-8 and breaks site.py on every later start.
    # Only the exact target editable file is touched, and it must point at the
    # declared source root -- see .tools/normalise-pth.py for the scope rules.
    note = f"editable install {' '.join(args)}"
    expected_src = spec.get("src") or entry.get("editable_src")
    package = spec.get("package")
    if not (expected_src and package):
        # Nothing declared -> nothing to normalise. This is not a success claim
        # for normalisation; it simply does not apply to this project.
        return True, note, [" ".join(res.cmd)]

    site_packages = (normalise_pth.site_packages_of(py)
                     or (py.parent.parent / "Lib" / "site-packages"))
    try:
        target = normalise_pth.find_editable_pth(site_packages, package)
        record = normalise_pth.normalise_target_pth(
            target, Path(proj) / expected_src, site_packages=site_packages,
            log=lambda r: None)
    except Exception as exc:                       # noqa: BLE001 - reported as failure
        # A declared editable whose .pth cannot be normalised is a real failure:
        # the environment may not start at all under some encodings. Report it
        # instead of recording success.
        if log_path:
            ec._write_log(log_path, ["normalise-pth"], str(proj), 1, str(exc))
        return False, f"{note}; editable .pth normalisation FAILED: {exc}", [" ".join(res.cmd)]

    if record.get("changed"):
        note += "; normalised editable .pth to ASCII"
    if log_path:
        ec._write_log(log_path, ["normalise-pth"], str(proj), 0,
                      json.dumps(record, ensure_ascii=False, indent=2))
    return True, note, [" ".join(res.cmd)]


def install_test_overlay(py: Path, entry: dict, proj: Path, manifest: dict,
                         scratch: Path, log_path: Path) -> tuple[bool, str, list, list]:
    """Install the explicit test overlay under the runtime lock's constraint.

    Two rules keep this from silently changing a verified environment:

    * the runtime lock is converted to a pure-version constraints file (the lock
      body, hashes included, is never rewritten) and passed to pip, so an overlay
      can never upgrade a package the lock already pins;
    * if the lock itself pins pytest, that pin wins over the overlay's preferred
      version, and any disagreement with the project's declared constraint in
      ``requirements.txt`` is reported rather than silently resolved.
    """
    actions, added, problems = [], [], []

    lock_source = ec.primary_source(entry)
    pins: dict[str, str] = {}
    constraints_file = None
    if lock_source and (proj / lock_source).is_file():
        pins = ec.lock_pins(proj / lock_source)
        if pins:
            constraints_file = ec.write_constraints(
                scratch / f"{entry['dir'].replace('/', '_')}.constraints.txt", pins)
            actions.append(f"constraints from {lock_source}: {len(pins)} pins")

    preferred = ec.pytest_spec(manifest, entry)
    locked_pytest = pins.get("pytest")
    declared = ec.declared_pytest_constraint(entry)

    if locked_pytest:
        want = f"pytest=={locked_pytest}"
        if declared and not ec.constraint_allows(
                declared, ec.parse_version(locked_pytest)):
            return False, (
                f"CONFLICT: the runtime lock ({lock_source}) pins "
                f"pytest=={locked_pytest}, which violates the project's declared "
                f"constraint '{declared}' in requirements.txt. Nothing was installed: "
                f"review the lock and the declaration together."), actions, problems
        if not preferred.endswith(locked_pytest):
            problems.append(
                f"runtime lock pins pytest=={locked_pytest}; overlay preference "
                f"'{preferred}' is not applied (the lock wins)")
    else:
        want = preferred

    has_pytest = ec.run([py, "-c", "import pytest;print(pytest.__version__)"],
                        cwd=proj, timeout=120, log_path=log_path)
    if has_pytest.ok and declared:
        current = ec.parse_version(has_pytest.out.strip())
        if not ec.constraint_allows(declared, current):
            return False, (
                f"CONFLICT: installed pytest {has_pytest.out.strip()} violates the "
                f"project's declared constraint '{declared}' (requirements.txt). "
                f"Nothing was changed; resolve the mismatch explicitly."), actions, problems

    if not has_pytest.ok:
        args = [want] + (["-c", str(constraints_file)] if constraints_file else [])
        res = ec.pip_install(py, args, cwd=proj, index=MIRROR, log_path=log_path)
        actions.append(" ".join(res.cmd))
        if not res.ok:
            return False, (f"test overlay '{want}' failed (exit {res.code}); "
                           f"see log"), actions, problems
        added.append(want)

    for dep in entry.get("test_deps") or []:
        name = dep.split("==")[0].split(">=")[0].strip()
        probe = ec.run([py, "-c", f"import importlib.util,sys;"
                                  f"sys.exit(0 if importlib.util.find_spec("
                                  f"{name.replace('-', '_')!r}) else 1)"],
                       cwd=proj, timeout=120, log_path=log_path)
        if probe.ok:
            continue
        args = [dep] + (["-c", str(constraints_file)] if constraints_file else [])
        res = ec.pip_install(py, args, cwd=proj, index=MIRROR, log_path=log_path)
        actions.append(" ".join(res.cmd))
        if not res.ok:
            return False, f"test dep '{dep}' failed (exit {res.code}); see log", actions, problems
        added.append(dep)

    detail = "; ".join(actions) if actions else "test overlay already satisfied"
    if problems:
        detail = " | ".join(problems) + " | " + detail
    return True, detail, actions, problems


def check_existing_env(entry: dict, py: Path, proj: Path, manifest: dict,
                       args, log_path: Path) -> dict:
    """Validate an already-runnable venv: version range, isolation, lock, imports."""
    info = ec.interpreter_info(py)
    pmin = (entry.get("python") or {}).get("min")
    pmax = (entry.get("python") or {}).get("max")
    base = ec.requested_version(manifest, entry)
    version_ok = ec.satisfies(info["version_tuple"], pmin, pmax)
    base_ok = bool(base) and info["version_tuple"] == ec.parse_version(base)
    isolated = bool(info.get("prefix") and info.get("base_prefix")
                    and info["prefix"] != info["base_prefix"])
    lock = lock_consistency(entry, py, proj, log_path)
    imports = verify_imports(entry, py, proj, log_path)
    pipchk = pip_check(py, proj, log_path)

    problems = []
    if not version_ok:
        problems.append(f"interpreter Python {info['version']} is outside the declared "
                        f"range [{pmin or '-'}, {pmax or '-'})")
    if not base_ok:
        problems.append(f"interpreter Python {info['version']} is not the declared seed "
                        f"Python {base} (the project's lock was validated on {base})")
    if not isolated:
        problems.append(f"not an isolated venv (prefix={info['prefix']}, "
                        f"base_prefix={info['base_prefix']})")
    if lock.get("checked"):
        if lock.get("missing"):
            problems.append(f"{len(lock['missing'])} locked package(s) missing: "
                            + ", ".join(lock["missing"][:8]))
        if lock.get("mismatched"):
            bad = ", ".join(f"{n} {got}!={want}" for n, want, got in lock["mismatched"][:8])
            problems.append(f"version mismatch vs lock: {bad}")
    else:
        problems.append(f"lock consistency not verifiable ({lock.get('reason')})")
    if imports.get("checked") and not imports.get("ok"):
        problems.append(f"declared imports unavailable: {imports.get('verdict')}")
    if not pipchk.ok:
        problems.append(f"pip check failed (exit {pipchk.code})")

    return {"info": info, "version_ok": version_ok, "base_ok": base_ok,
            "isolated": isolated, "lock": lock, "imports": imports,
            "pip_check_exit": pipchk.code, "problems": problems,
            "ok": not problems}


def _finish(row: dict, proj: Path, tracked: list) -> dict:
    """Always record the lock hashes after the run (dry-run included)."""
    row["lock_hashes_after"] = ec.hash_many(proj, tracked)
    row["lock_unchanged"] = row["lock_hashes_before"] == row["lock_hashes_after"]
    return row


def provision_one(entry: dict, manifest: dict, args, base: Path, log_path: Path,
                  scratch: Path) -> dict:
    name = entry["name"]
    proj = ec.project_dir(entry)
    coverage = ec.reproducibility_coverage(entry)
    row = {"project": name, "dir": entry["dir"], "venv": entry["venv"],
           "base": str(base), "base_kind": entry.get("base"),
           "state": FAILED, "detail": "",
           # Explicitly scoped: `reproducible` here means the DECLARED pins come
           # from a frozen lock. It does NOT mean the whole dependency closure is
           # pinned -- see `reproducibility` for the measured level.
           "reproducible": coverage["declared"],
           "reproducible_scope": "declared-pins",
           "reproducibility": coverage,
           "closure_measured": coverage["closure_measured"],
           "lock_hashes_before": {}, "lock_hashes_after": {}, "commands": [],
           "log": str(log_path)}

    tracked = list(ec.install_chain(entry))
    row["lock_hashes_before"] = ec.hash_many(proj, tracked)

    if not proj.is_dir():
        row["detail"] = f"project dir missing: {proj}"
        return _finish(row, proj, tracked)

    state = venv_state(entry)
    row["interpreter_before"] = {k: state.get(k) for k in
                                 ("version", "prefix", "base_prefix", "runs", "venv_dir_exists")}

    existing_ok = False
    existing_report = None
    if state["runs"]:
        if args.dry_run or args.reinstall:
            # --reinstall means "reinstall", not "reinstall only if broken"; the
            # real audit still happens after the install.
            existing_ok = None
        else:
            existing_report = check_existing_env(entry, Path(state["path"]), proj,
                                                 manifest, args, log_path)
            existing_ok = existing_report["ok"]
    elif state["venv_dir_exists"]:
        row["state"] = BROKEN
        row["detail"] = (
            f"venv at {state['venv_dir']} exists but its interpreter does not run: "
            f"{state.get('error')}. Back it up inside the project "
            f"(Move-Item .venv .venv.pre-rebuild-<date>) and rerun; this tool never "
            f"deletes or overwrites an existing venv.")
        row["probe_error"] = state.get("error")
        return _finish(row, proj, tracked)

    # ---------------------------------------------------------------- dry-run
    if args.dry_run:
        plan = []
        if not state["runs"]:
            plan.append(f"create venv {state['venv_dir']} with {base}")
            source = ec.primary_source(entry)
            plan.append(f"pip install -r {source}" if source
                        else "no frozen lock declared -> result not reproducible")
            if entry.get("editable"):
                plan.append("pip install " + " ".join(entry["editable"].get("args") or []))
        else:
            plan.append(f"reuse runnable venv (Python {state.get('version')})")
            plan.append("verify: interpreter range, isolation, lock consistency, imports, "
                        "pip check")
            if args.reinstall:
                plan.append("reinstall from the frozen lock (--reinstall)")
        if args.with_tests:
            plan.append(f"apply test overlay {ec.pytest_spec(manifest, entry)} "
                        f"(constrained by {ec.primary_source(entry)})")
        row["state"] = PLANNED
        row["plan"] = plan
        row["interpreter_range_ok"] = ec.satisfies(
            state.get("version_tuple"), (entry.get("python") or {}).get("min"),
            (entry.get("python") or {}).get("max"))
        row["detail"] = "; ".join(plan)
        return _finish(row, proj, tracked)

    # --------------------------------------------------------- create if absent
    created = False
    if not state["runs"]:
        vdir = ec.venv_dir(entry)
        vdir.parent.mkdir(parents=True, exist_ok=True)
        res = ec.run([base, "-m", "venv", str(vdir)], timeout=900, log_path=log_path)
        row["commands"].append(" ".join(res.cmd))
        if not res.ok or not ec.venv_python(entry).is_file():
            row["detail"] = f"venv creation failed (exit {res.code}); see log"
            return _finish(row, proj, tracked)
        created = True
        state = venv_state(entry)
    py = Path(state["path"]) if state.get("path") else ec.venv_python(entry)
    row["created"] = created
    row["interpreter"] = {k: state.get(k) for k in ("version", "prefix", "base_prefix")}

    # R4: the venv we are about to populate must satisfy the declared range
    if not ec.satisfies(state.get("version_tuple"),
                        (entry.get("python") or {}).get("min"),
                        (entry.get("python") or {}).get("max")):
        row["detail"] = (f"refusing to use venv: interpreter Python {state.get('version')} "
                         f"is outside the declared range")
        return _finish(row, proj, tracked)

    # ------------------------------------------------------------- installation
    reinstall = args.reinstall or created
    need_install = reinstall or existing_ok is False
    if existing_ok and not args.reinstall:
        row["pre_existing_check"] = {
            "ok": True, "pip_check_exit": existing_report["pip_check_exit"],
            "lock": existing_report["lock"], "imports": existing_report["imports"],
        }
    elif existing_ok is False and not reinstall and not args.with_tests:
        # The env exists and runs but failed its audit. Rebuilding in place would
        # leave stale packages behind, so report the concrete findings instead.
        # (With --with-tests we continue: the overlay is exactly what supplies the
        # missing test packages, and the audit runs again afterwards.)
        row["state"] = FAILED
        row["detail"] = ("existing venv failed its audit (not reinstalling): "
                         + "; ".join(existing_report["problems"])
                         + "; rerun with --reinstall to rebuild from the frozen lock, "
                           "or move .venv aside and provision a clean one")
        row["validation"] = {
            "version_ok": existing_report["version_ok"],
            "base_ok": existing_report["base_ok"],
            "isolated": existing_report["isolated"],
            "pip_check_exit": existing_report["pip_check_exit"],
            "lock": existing_report["lock"], "imports": existing_report["imports"],
            "problems": existing_report["problems"],
        }
        return _finish(row, proj, tracked)
    elif existing_ok is False and not reinstall:
        row["pre_existing_check"] = {
            "ok": False, "problems": existing_report["problems"],
            "note": "--with-tests was requested, so the test overlay is applied "
                    "before the environment is judged",
        }

    if need_install:
        ok, detail, cmds = install_from_chain(py, entry, proj, log_path,
                                              allow_fallback=not args.no_index_fallback)
        row["commands"] += cmds
        if not ok:
            row["detail"] = detail
            return _finish(row, proj, tracked)
        row["install"] = detail

        ok, detail, cmds = install_editable(py, entry, proj, log_path)
        row["commands"] += cmds
        if not ok:
            row["detail"] = detail
            return _finish(row, proj, tracked)
        if detail:
            row["editable"] = detail
    else:
        row["install"] = "reused existing venv (already consistent with the lock)"

    # ------------------------------------------------------------- test overlay
    # Applied before the final validation so that pip check, the lock audit and
    # the import probe all describe the environment that will actually be used.
    overlay_note = ""
    if args.with_tests:
        ok, detail, actions, problems = install_test_overlay(
            py, entry, proj, manifest, scratch, log_path)
        row["commands"] += actions
        row["test_overlay"] = detail
        row["test_overlay_warnings"] = problems
        if not ok:
            row["detail"] = detail
            return _finish(row, proj, tracked)
        overlay_note = "; overlay: " + detail

    # ------------------------------------------------------- lock-file integrity
    # Checked before the environment audit: a lock that changed during install is
    # the more serious finding and would otherwise be masked by the audit.
    _finish(row, proj, tracked)
    if not row["lock_unchanged"]:
        row["state"] = FAILED
        row["detail"] = ("lock file changed during install (must never happen): "
                         + ", ".join(k for k, v in row["lock_hashes_after"].items()
                                      if row["lock_hashes_before"].get(k) != v))
        return row

    # -------------------------------------------------- validate the final state
    report = check_existing_env(entry, py, proj, manifest, args, log_path)
    row["validation"] = {
        "version_ok": report["version_ok"], "isolated": report["isolated"],
        "pip_check_exit": report["pip_check_exit"], "lock": report["lock"],
        "imports": report["imports"], "problems": report["problems"],
    }
    if not report["ok"]:
        row["detail"] = "post-install validation failed: " + "; ".join(report["problems"])
        return _finish(row, proj, tracked)

    if not ec.is_reproducible(entry):
        row["state"] = UNREPRODUCIBLE
        row["detail"] = (f"env built from {ec.primary_source(entry)} (no frozen lock); "
                         f"NOT claimed reproducible{overlay_note}")
        return row

    row["state"] = OK if need_install else PRESENT
    row["detail"] = ((row.get("install") or "installed") +
                     f"; lock consistent ({report['lock'].get('locked_count')} pins), "
                     f"imports OK, pip check clean, locks unchanged{overlay_note}")
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Provision per-project venvs from their frozen locks.")
    ap.add_argument("--project", action="append", default=None,
                    help="project name or dir (repeatable); default: all")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="preflight only: interpreter, lock presence, venv state")
    ap.add_argument("--dry-run", action="store_true",
                    help="probe read-only state and print the plan; never installs")
    ap.add_argument("--reinstall", action="store_true",
                    help="reinstall even when an existing venv validates")
    ap.add_argument("--with-tests", action="store_true",
                    help="apply the explicit test overlay (lock-constrained)")
    ap.add_argument("--no-index-fallback", action="store_true")
    ap.add_argument("--python-map", default=None,
                    help="explicit version=path mapping, e.g. "
                         "'3.10=C:\\path\\python.exe;3.12=C:\\other\\python.exe'")
    ap.add_argument("--manifest", default=None,
                    help="use this projects.json instead of .tools/projects.json")
    ap.add_argument("--log", default=None, help="append full output to this log file")
    ap.add_argument("--json", default=None, help="write a JSON report here")
    args = ap.parse_args(argv)

    manifest = ec.load_manifest(Path(args.manifest) if args.manifest else None)
    try:
        entries = ec.find_projects(manifest, args.project)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.list:
        rows = [[e["name"], e["dir"], e.get("base", "-"),
                 ",".join(ec.install_chain(e)) or "(no lock)",
                 (e.get("test") or {}).get("venv") or e["venv"],
                 ec.reproducibility_coverage(e)["level"]] for e in entries]
        ec.print_table(rows, ["project", "dir", "base", "install from",
                              "test venv", "reproducibility"])
        print("reproducibility levels: full-closure (all transitive deps pinned) > "
              "declared-pins-unmeasured (lock exists, closure not measured) > "
              "declared-pins (lock exists) > none. A 'declared-pins*' level means "
              "the DECLARED packages are pinned, not that the closure is frozen.")
        return 0

    base_map = ec.parse_base_map(args.python_map) if args.python_map else None
    try:
        resolved, problems = ec.preflight(manifest, entries, base_map)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 3

    print(f"manifest         : {manifest['_path']}")
    print(f"projects selected: {len(entries)}")
    print(f"mode             : {'DRY-RUN (no side effects)' if args.dry_run else 'apply'}"
          + (" +test overlay" if args.with_tests else "")
          + (" +reinstall" if args.reinstall else ""))
    for e in entries:
        base = resolved.get(e["name"])
        print(f"  base {e['name']:<28} {base if base else 'UNRESOLVED'}")
    print("=" * 78)

    if problems:
        print("[provision-envs] preflight failed; nothing was created or changed.",
              file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        if args.json:
            ec.write_json(Path(args.json), {"stage": "preflight", "ok": False,
                                            "dry_run": args.dry_run,
                                            "problems": problems})
        return 3

    if args.check:
        rows = []
        for e in entries:
            proj = ec.project_dir(e)
            chain = ec.install_chain(e)
            missing = [n for n in chain if not (proj / n).is_file()]
            state = venv_state(e)
            verdict = ("NO FROZEN LOCK" if not chain else
                       ("LOCK MISSING" if missing else "lock present"))
            venv = ("runnable" if state["runs"] else
                    ("BROKEN" if state["venv_dir_exists"] else "absent"))
            rows.append([e["name"], resolved[e["name"]].name, verdict, venv,
                         ec.reproducibility_coverage(e)["level"]])
        ec.print_table(rows, ["project", "base", "install source", "venv",
                              "reproducibility"])
        print("reproducibility levels: full-closure > declared-pins-unmeasured > "
              "declared-pins > none. 'declared-pins*' means the DECLARED packages "
              "are pinned; it does NOT claim the whole dependency closure is frozen "
              "(closure is measured by .tools/check-env-isolation.py).")
        bad = [r for r in rows if r[2] == "LOCK MISSING"]
        print(f"preflight {'OK' if not bad else 'FAILED'} "
              f"({len(entries) - len(bad)}/{len(entries)})")
        if args.json:
            ec.write_json(Path(args.json), {"stage": "check", "ok": not bad,
                                            "rows": rows, "problems": problems})
        return 0 if not bad else 1

    log_path = Path(args.log) if args.log else None
    scratch = ec.active_root() / ".tools" / ".scratch"
    if not args.dry_run:
        scratch.mkdir(parents=True, exist_ok=True)

    results = []
    for entry in entries:
        row = provision_one(entry, manifest, args, resolved[entry["name"]],
                            log_path, scratch)
        results.append(row)
        print(f"[{row['state']:<14}] {row['project']:<28} {row['detail'][:110]}")
        sys.stdout.flush()

    print("=" * 78)
    counts: dict[str, int] = {}
    for row in results:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    print("summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    failed = [r for r in results if r["state"] in (FAILED, BROKEN)]
    for row in failed:
        print(f"  {row['state']:<14} {row['project']}: {row['detail']}")
    touched = [r["project"] for r in results if r.get("lock_unchanged") is False]
    if touched:
        print("  lock files changed during install (must be empty): " + ", ".join(touched))

    if args.dry_run:
        # A dry run reports the plan. A pre-existing venv whose interpreter no
        # longer runs is an expected, actionable finding (move it aside and
        # provision), not a failure of the plan itself.
        broken = [r for r in results if r["state"] == BROKEN]
        if broken:
            print(f"  note: {len(broken)} existing venv(s) must be moved aside "
                  f"before provisioning (see the BROKEN lines above)")
        ok = not [r for r in results if r["state"] in (FAILED,)]
        if args.json:
            ec.write_json(Path(args.json), {
                "stage": "provision", "ok": ok, "dry_run": True,
                "counts": counts, "results": results,
            })
        return 0 if ok else 1

    ok = not failed and not touched
    if args.json:
        ec.write_json(Path(args.json), {
            "stage": "provision", "ok": ok, "dry_run": False,
            "counts": counts, "results": results,
        })
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
