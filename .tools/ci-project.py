# -*- coding: utf-8 -*-
"""CI entry point: install and test one project, driven by .tools/projects.json.

This is what the CI matrix calls. It deliberately contains **no installer of its
own**: for ``install`` it delegates to ``.tools/provision-envs.py`` (the same
single implementation used locally), passing the CI interpreter as the project's
explicit seed via ``--python-map``. That keeps one code path for venv creation,
lock installation, the test overlay and validation, so CI cannot drift from
local provisioning (R7).

The CI matrix keeps only project identity and the interpreter version
(``name`` / ``dir`` / ``py`` / ``ffmpeg``), which CI cannot derive at runtime;
everything else is resolved from the manifest.

Usage (CI)::

    python .tools/ci-project.py install --dir "全整合" --log <path>
    python .tools/ci-project.py test    --dir "全整合" --log <path>
    python .tools/ci-project.py check-pytest --dir "全整合"

Exit codes: 0 ok, 1 failure, 2 usage/manifest, 3 blocked (wrong CI interpreter).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _envcommon as ec  # noqa: E402

PROVISIONER = Path(__file__).resolve().parent / "provision-envs.py"
MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
FALLBACK_INDEX = "https://pypi.org/simple"


def entry_for(manifest: dict, dir_value: str) -> dict:
    key = dir_value.replace("\\", "/").strip("/")
    for entry in ec.project_list(manifest):
        if entry["dir"] == key or entry["name"] == dir_value:
            return entry
    raise SystemExit(f"[ci-project] '{dir_value}' is not in the manifest "
                     f"({len(ec.project_list(manifest))} projects)")


def echo_log(log_path: Path | None) -> None:
    """Print the complete captured log, not just a failure tail."""
    if not log_path or not Path(log_path).is_file():
        return
    print(f"\n===== full log: {log_path} =====")
    try:
        print(Path(log_path).read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        print(f"[ci-project] could not read log: {exc}")


def do_install(entry: dict, manifest: dict, log_path: Path | None) -> int:
    version = ec.project_python(entry)
    have = (sys.version_info.major, sys.version_info.minor)
    if version and have != ec.parse_version(version):
        print(f"[ci-project] this job runs Python {have[0]}.{have[1]} but "
              f"{entry['dir']} needs Python {version}; fix the ci.yml matrix 'py' "
              f"for this project (setup-python installs matrix.py)", file=sys.stderr)
        return 3

    proj = ec.project_dir(entry)
    if not proj.is_dir():
        print(f"[ci-project] project dir missing: {proj}", file=sys.stderr)
        return 1

    venv_py = ec.venv_python(entry)
    print(f"[ci-project] install {entry['dir']} via {PROVISIONER.name} "
          f"(seed {sys.executable}, venv {venv_py})")

    cmd = [sys.executable, str(PROVISIONER),
           "--manifest", manifest["_path"],
           "--project", entry["dir"],
           "--with-tests",
           "--python-map", f"{version}={sys.executable}" if version else
                           f"default={sys.executable}",
           "--json", str((log_path.parent if log_path else proj) / "ci-install-report.json")]
    if log_path:
        cmd += ["--log", str(log_path)]

    res = ec.run(cmd, cwd=ec.active_root(), timeout=5400, log_path=log_path)
    print(res.out)
    if not res.ok:
        print(f"[ci-project] provisioning failed (exit {res.code})", file=sys.stderr)
        echo_log(log_path)
        return 1

    if not ec.venv_python(entry).is_file():
        print(f"[ci-project] provisioning reported success but {venv_py} is missing",
              file=sys.stderr)
        return 1
    print(f"[ci-project] project venv ready: {venv_py}")
    return 0


def do_test(entry: dict, manifest: dict, log_path: Path | None) -> int:
    proj = ec.project_dir(entry)
    venv_name = (entry.get("test") or {}).get("venv") or entry["venv"]
    py = ec.venv_python(entry, venv_name)
    if not py.is_file():
        print(f"[ci-project] project venv missing: {py}; run the install step first",
              file=sys.stderr)
        return 1
    specs = entry.get("tests") or []
    if not specs:
        print(f"[ci-project] {entry['dir']}: no test command declared in the "
              f"manifest; nothing to run (not a pass)", file=sys.stderr)
        return 1

    check = ec.run([py, "-X", "utf8", "-m", "pip", "check"], cwd=proj, timeout=300,
                   log_path=log_path)
    print(check.out)
    if not check.ok:
        print("[ci-project] pip check failed before tests", file=sys.stderr)
        echo_log(log_path)
        return 1

    log = log_path or (ec.active_root() / "ci-test.log")
    for spec in specs:
        argv = [str(py), "-X", "utf8", *[str(a) for a in spec.get("argv") or []]]
        print(f"[ci-project] run ({spec.get('label', '?')}): {' '.join(argv)}")
        res = ec.run(argv, cwd=proj, timeout=1800, log_path=log)
        print(res.out)
        if not res.ok:
            print(f"[ci-project] FAILED ({spec.get('label')}): exit {res.code}",
                  file=sys.stderr)
            echo_log(log_path)
            return 1
    print(f"[ci-project] all {len(specs)} declared command(s) passed")
    return 0


def do_check_pytest(entry: dict, manifest: dict) -> int:
    proj = ec.project_dir(entry)
    venv_name = (entry.get("test") or {}).get("venv") or entry["venv"]
    py = ec.venv_python(entry, venv_name)
    constraint = ec.declared_pytest_constraint(entry)
    if constraint is None:
        print("[ci-project] requirements.txt declares no pytest constraint; "
              "nothing to assert")
        return 0
    probe = ec.run([py, "-c", "import pytest;print(pytest.__version__)"], cwd=proj,
                   timeout=120)
    if not probe.ok:
        print(f"[ci-project] pytest is not installed but '{constraint}' is declared "
              f"in requirements.txt", file=sys.stderr)
        return 1
    version = ec.parse_version(probe.out.strip())
    if not ec.constraint_allows(constraint, version):
        print(f"[ci-project] pytest {probe.out.strip()} does not satisfy declared "
              f"'{constraint}'", file=sys.stderr)
        return 1
    print(f"[ci-project] pytest {probe.out.strip()} satisfies declared '{constraint}'")
    return 0


def do_contract(entry: dict, manifest: dict) -> int:
    import json
    print(json.dumps(ec.contract(entry), ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Manifest-driven CI install/test step.")
    ap.add_argument("action",
                    choices=["install", "test", "contract", "check-pytest"])
    ap.add_argument("--dir", required=True, help="project dir as written in the manifest")
    ap.add_argument("--manifest", default=None, help="override projects.json")
    ap.add_argument("--log", default=None, help="append the full command output here")
    args = ap.parse_args(argv)

    manifest = ec.load_manifest(Path(args.manifest) if args.manifest else None)
    entry = entry_for(manifest, args.dir)
    log_path = Path(args.log) if args.log else None

    if args.action == "install":
        return do_install(entry, manifest, log_path)
    if args.action == "test":
        return do_test(entry, manifest, log_path)
    if args.action == "check-pytest":
        return do_check_pytest(entry, manifest)
    return do_contract(entry, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
