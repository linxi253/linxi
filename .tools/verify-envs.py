# -*- coding: utf-8 -*-
"""Run each project's own declared test command in that project's own venv.

What changed (reconcile round, R7)
----------------------------------
* Project list, venv name, Python range, install source and test command all come
  from ``.tools/projects.json`` — the same manifest ``provision-envs.py`` reads.
  The old failure mode (provision ``.venv``, then test a non-existent
  ``.venv-build``) is now structurally impossible, and the guard below catches it
  if anyone reintroduces the mismatch.
* This script **never installs or uninstalls anything**. A missing package, a
  missing venv, a broken interpreter or a timeout is reported as a non-pass, with
  an actionable hint, instead of being papered over with a temporary pip install.
* ``--project`` / ``--dry-run`` support分批执行, and the exit code is non-zero
  unless every selected command genuinely passed. SKIP is never counted as pass.

Usage::

    python .tools/verify-envs.py --list
    python .tools/verify-envs.py --dry-run
    python .tools/verify-envs.py --project 全整合
    python .tools/verify-envs.py --json results.json
    python .tools/verify-envs.py --check-matrix [.github/workflows/ci.yml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _envcommon as ec  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
ERROR = "ERROR"

CORE_IMPORTS = {
    "numpy": "numpy", "scipy": "scipy", "matplotlib": "matplotlib",
    "pandas": "pandas", "tifffile": "tifffile", "PIL": "PIL", "cv2": "cv2",
    "skimage": "skimage", "ase": "ase", "pyfftw": "pyfftw",
    "ncempy": "ncempy", "ttkbootstrap": "ttkbootstrap",
    "imagecodecs": "imagecodecs", "openpyxl": "openpyxl", "seaborn": "seaborn",
    "defusedxml": "defusedxml", "tkinterdnd2": "tkinterdnd2",
    "imageio_ffmpeg": "imageio_ffmpeg",
}


def quick_import_check(py: Path, proj: Path, modules: list[str]) -> tuple[bool, str]:
    if not modules:
        return True, ""
    code = ("import importlib,sys\n"
            "bad=[]\n"
            f"for m in {modules!r}:\n"
            "    try: importlib.import_module(m)\n"
            "    except Exception as e: bad.append(m+':'+type(e).__name__)\n"
            "print('BAD=' + ','.join(bad) if bad else 'OK')\n")
    res = ec.run([py, "-c", code], cwd=proj, timeout=300)
    if not res.ok:
        return False, f"import probe failed (exit {res.code}): {res.out.strip()[-200:]}"
    verdict = res.out.strip().splitlines()[-1] if res.out.strip() else "?"
    return verdict == "OK", verdict


def pip_install_hint(py: Path, proj: Path) -> str:
    return (f'& "{py}" -X utf8 -m pip install -r requirements.lock.txt   '
            f'(cwd {proj}); 或用 .tools/provision-envs.py --project '
            f'"{proj.name}" --with-tests')


def run_one(entry: dict, manifest: dict, results: list, dry_run: bool,
            log_path=None) -> None:
    proj = ec.project_dir(entry)
    venv_name = (entry.get("test") or {}).get("venv") or entry["venv"]
    py = ec.venv_python(entry, venv_name)

    if not proj.is_dir():
        results.append({"project": entry["name"], "dir": entry["dir"], "label": "-",
                        "status": ERROR, "detail": f"project dir missing: {proj}"})
        return

    if not entry.get("tests"):
        results.append({"project": entry["name"], "dir": entry["dir"], "label": "-",
                        "status": SKIP,
                        "detail": "no test command declared in the manifest "
                                  "(adapter project; not a pass)"})
        return

    info = ec.interpreter_info(py)
    modules = [CORE_IMPORTS[m] for m in (entry.get("verify_imports") or [])
               if m in CORE_IMPORTS]
    if not info["runs"]:
        detail = (f"interpreter not runnable: {info.get('error')}; "
                  f"provision it first: python .tools/provision-envs.py "
                  f'--project "{entry["name"]}"')
        results.append({"project": entry["name"], "dir": entry["dir"], "label": "-",
                        "status": SKIP, "detail": detail,
                        "interpreter": str(py)})
        return

    pmin = entry.get("python", {}).get("min")
    pmax = entry.get("python", {}).get("max")
    version = ec.parse_version(info["version"])
    if not ec.satisfies(version, pmin, pmax):
        results.append({
            "project": entry["name"], "dir": entry["dir"], "label": "-",
            "status": ERROR, "interpreter": str(py), "version": info["version"],
            "detail": f"interpreter {info['version']} outside declared range "
                      f"[{pmin or '-'}, {pmax or '-'})"})
        return

    if modules:
        ok, verdict = quick_import_check(py, proj, modules)
        if not ok:
            results.append({
                "project": entry["name"], "dir": entry["dir"], "label": "imports",
                "status": FAIL, "interpreter": str(py),
                "detail": f"declared imports unavailable: {verdict}. "
                          + pip_install_hint(py, proj)})
            return

    for spec in entry["tests"]:
        label = spec.get("label") or "test"
        argv = [str(py), "-X", "utf8", *[str(a) for a in spec.get("argv") or []]]
        if not spec.get("argv"):
            results.append({"project": entry["name"], "dir": entry["dir"],
                            "label": label, "status": ERROR,
                            "detail": "empty test argv in manifest"})
            continue
        if dry_run:
            results.append({"project": entry["name"], "dir": entry["dir"],
                            "label": label, "status": "PLANNED",
                            "cmd": argv, "interpreter": str(py)})
            continue
        res = ec.run(argv, cwd=proj, timeout=1800, log_path=log_path)
        summary = ""
        for line in reversed([l for l in res.out.splitlines() if l.strip()]):
            if ("passed" in line or "failed" in line or "error" in line
                    or line.strip().startswith(("OK", "FAIL", "PASS"))):
                summary = line.strip()[:120]
                break
        if res.timed_out:
            status, detail = FAIL, f"timeout: {res.out.strip()[-200:]}"
        elif res.ok:
            status, detail = PASS, summary or "exit 0"
        else:
            status, detail = FAIL, (f"exit {res.code}: "
                                    + (summary or res.out.strip()[-300:]))
        results.append({
            "project": entry["name"], "dir": entry["dir"], "label": label,
            "status": status, "detail": detail, "cmd": argv,
            "exit": res.code, "interpreter": str(py),
            "tail": res.out[-4000:],
        })


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run each project's declared tests.")
    ap.add_argument("--project", action="append", default=None,
                    help="project name or dir (repeatable); default: all")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would run (plus venv/import readiness)")
    ap.add_argument("--check-matrix", nargs="?", const="", default=None,
                    metavar="CI_YML",
                    help="compare the manifest-derived CI matrix with ci.yml")
    ap.add_argument("--manifest", default=None,
                    help="use this projects.json instead of .tools/projects.json")
    ap.add_argument("--log", default=None,
                    help="append the full stdout/stderr of every command here")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    manifest = ec.load_manifest(Path(args.manifest) if args.manifest else None)

    if args.check_matrix is not None:
        ci_path = Path(args.check_matrix) if args.check_matrix else \
            ec.active_root() / ".github" / "workflows" / "ci.yml"
        ok, messages = ec.check_ci_matrix(manifest, ci_path)
        default_manifest = ec.default_py_from_manifest(manifest)
        default_ci = ec.default_py_from_ci(ci_path)
        if default_ci and default_manifest != default_ci:
            ok = False
            messages.append(f"default python-version differs "
                            f"(manifest={default_manifest}, ci.yml={default_ci})")
        for m in messages:
            print(f"MISMATCH {m}")
        if ok:
            rows = ec.derive_ci_rows(manifest)
            print(f"manifest <-> ci.yml matrix aligned: {len(rows)} projects")
            print("compared: matrix identity/version fields (name, dir, py, ffmpeg), "
                  "each project's install source + lock presence + test venv + test "
                  "scripts, the CI install/test lifecycle steps, and the ban on raw "
                  "pip installs that would bypass a lock")
        return 0 if ok else 1

    try:
        entries = ec.find_projects(manifest, args.project)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.list:
        for entry in entries:
            tests = ", ".join(s.get("label", "?") for s in entry.get("tests") or []) or "-"
            print(f"{entry['name']:<28} {entry['dir']:<40} "
                  f"venv={(entry.get('test') or {}).get('venv') or entry['venv']:<10} "
                  f"tests={tests}")
        return 0

    print(f"repo root: {ec.active_root()}")
    print("=" * 78)
    log_path = Path(args.log) if args.log else None
    results: list = []
    for entry in entries:
        run_one(entry, manifest, results, args.dry_run, log_path)

    counts: dict[str, int] = {}
    for row in results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    for row in results:
        detail = row.get("detail") or ("would run: "
                                       + "; ".join(row.get("test_commands") or [])
                                       or row["status"])
        print(f"[{row['status']:<7}] {row['project']:<28} {row['label']:<14} "
              f"{detail[:90]}")

    print("=" * 78)
    print("summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    not_pass = [r for r in results if r["status"] not in (PASS, "PLANNED")]
    for row in not_pass:
        detail = row.get("detail") or row["status"]
        print(f"  {row['status']:<7} {row['project']} / {row['label']}: "
              f"{detail[:160]}")
    print(f"passed {counts.get(PASS, 0)} / {len(results)}"
          + ("  (SKIP is not a pass)" if counts.get(SKIP) else ""))
    if args.json:
        ec.write_json(Path(args.json), {
            "stage": "verify", "dry_run": args.dry_run, "counts": counts,
            "results": results,
        })
    if args.dry_run:
        return 0 if not any(r["status"] == ERROR for r in results) else 1
    return 0 if (results and not not_pass) else 1


if __name__ == "__main__":
    raise SystemExit(main())
