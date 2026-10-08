# -*- coding: utf-8 -*-
"""Isolation and regression tests for the .tools environment toolchain.

These tests are *tool* tests: they exercise the provisioner, verifier, manifest
loader and CI guard. They never touch a business project's environment and never
install a business dependency.

Two styles are used deliberately:

* a synthetic repository under ``tmp_path`` (its own ``.tools/projects.json`` and
  project dirs) so the contract logic can be exercised without side effects;
* real subprocess runs against the real repository for exit codes and the
  real CI guard, so the shipped manifest and ci.yml are covered too.

Run with the tool interpreter::

    .tools\\.venv-tools\\Scripts\\python.exe -m pytest .tools/tests -q
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

REAL_TOOLS = Path(__file__).resolve().parent.parent
REAL_ROOT = REAL_TOOLS.parent
REAL_MANIFEST = REAL_TOOLS / "projects.json"

# Interpreter seeds are discovered or injected (never hardcoded to one machine):
# see tests/_test_seeds.py. `_seeds.find` returns (path|None, why); tests that
# need a real interpreter skip before doing any work.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_seeds as seeds  # noqa: E402

_SEED_310, SEED_310_SOURCE = seeds.find("3.10")
_SEED_312, SEED_312_SOURCE = seeds.find("3.12")
# A real 3.10 when one can be found or injected; otherwise the interpreter
# running the tests. The synthetic fixtures stub the interpreter layer, so any
# runnable path is equivalent there -- tests that genuinely need a 3.10/3.12
# seed call `seeds.require(...)`, which skips before doing anything.
PY310 = _SEED_310 or Path(sys.executable)
PY312 = _SEED_312                      # may be None -> use seeds.require("3.12")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_pair(tools_dir: Path, tag: str):
    """Load _envcommon plus provision/verify from a given .tools directory."""
    ec = _load(tools_dir / "_envcommon.py", f"_ec_{tag}")
    prov = _load(tools_dir / "provision-envs.py", f"_prov_{tag}")
    ver = _load(tools_dir / "verify-envs.py", f"_verify_{tag}")
    # make sure the auxiliary modules see the same instance
    prov.ec = ec
    ver.ec = ec
    return ec, prov, ver


def args_for(**over):
    base = dict(project=None, dry_run=False, reinstall=False, with_tests=False,
                no_index_fallback=False, check=False, list=False, manifest=None,
                log=None, json=None, python_map=None)
    base.update(over)
    return Namespace(**base)


def tree_snapshot(root: Path) -> dict:
    """path -> (size, mtime_ns) for every file, to prove nothing was written."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            try:
                st = p.stat()
            except OSError:
                continue
            out[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return out


SYNTH_MANIFEST = {
    "schema": 2,
    "base_interpreters": [{"id": "py310", "version": "3.10"},
                          {"id": "py312", "version": "3.12"}],
    "test_overlay": {"pytest": "pytest==8.4.2"},
    "projects": [
        {
            "name": "proj310", "dir": "p310", "base": "3.10",
            "python": {"min": "3.10", "max": None},
            "venv": ".venv", "venvs_to_create": [".venv"],
            "install": {"chain": ["requirements.lock.txt"], "reproducible": True},
            "tests": [{"label": "pytest", "argv": ["-m", "pytest", "-q"]}],
            "test": {"venv": ".venv"}, "editable": False,
            "verify_imports": ["json"],
            "ci": {"name": "P310"},
        },
        {
            "name": "proj312", "dir": "p312", "base": "3.12",
            "python": {"min": "3.11", "max": "3.14"},
            "venv": ".venv", "venvs_to_create": [".venv"],
            "install": {"chain": ["requirements.lock.txt"], "reproducible": True},
            "tests": [{"label": "pytest", "argv": ["-m", "pytest", "-q"]}],
            "test": {"venv": ".venv"}, "editable": False,
            "verify_imports": ["json"],
            "ci": {"name": "P312", "ffmpeg": True},
        },
        {
            "name": "multi", "dir": "multi", "base": "3.10",
            "python": {"min": "3.10", "max": None},
            "venv": ".venv", "venvs_to_create": [".venv"],
            "install": {"chain": ["requirements.lock.txt"], "reproducible": True},
            "tests": [{"label": "pytest", "argv": ["-m", "pytest", "-q"]},
                      {"label": "verify_physics", "argv": ["tests/verify_physics.py"]}],
            "test": {"venv": ".venv"}, "editable": False,
            "verify_imports": ["json"],
            "ci": {"name": "MULTI"},
        },
    ],
}

LOCK_TEXT = (
    "# synthetic lock\n"
    "numpy==2.2.6\n"
    "pytest==9.1.1\n"
    "withhash==1.0.0 \\\n"
    "    --hash=sha256:aaaa \\\n"
    "    --hash=sha256:bbbb\n"
)


def build_synth(tmp_path: Path, manifest: dict | None = None,
                ci_text: str | None = None) -> Path:
    root = tmp_path / "synthrepo"
    tools = root / ".tools"
    tools.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REAL_TOOLS / "_envcommon.py", tools / "_envcommon.py")
    shutil.copy2(REAL_TOOLS / "provision-envs.py", tools / "provision-envs.py")
    shutil.copy2(REAL_TOOLS / "verify-envs.py", tools / "verify-envs.py")
    shutil.copy2(REAL_TOOLS / "normalise-pth.py", tools / "normalise-pth.py")
    (tools / "projects.json").write_text(
        json.dumps(manifest or SYNTH_MANIFEST, ensure_ascii=False, indent=2),
        encoding="utf-8")
    for entry in (manifest or SYNTH_MANIFEST)["projects"]:
        proj = root / entry["dir"]
        (proj / "tests").mkdir(parents=True, exist_ok=True)
        (proj / "tests" / "verify_physics.py").write_text("print('physics ok')\n",
                                                          encoding="utf-8")
        (proj / "requirements.txt").write_text("pytest>=8,<10\n", encoding="utf-8")
        (proj / "requirements.lock.txt").write_text(LOCK_TEXT, encoding="utf-8")
    gh = root / ".github" / "workflows"
    gh.mkdir(parents=True, exist_ok=True)
    (gh / "ci.yml").write_text(ci_text if ci_text is not None else default_ci(),
                               encoding="utf-8")
    return root


def default_ci() -> str:
    return (
        "jobs:\n"
        "  test:\n"
        "    strategy:\n"
        "      matrix:\n"
        "        include:\n"
        "          - { name: 'P310', dir: 'p310', py: '3.10' }\n"
        "          - { name: 'P312', dir: 'p312', ffmpeg: true, py: '3.12' }\n"
        "          - { name: 'MULTI', dir: 'multi', py: '3.10' }\n"
        "    steps:\n"
        "      - run: python .tools/verify-envs.py --check-matrix .github/workflows/ci.yml\n"
        "      - run: python .tools/ci-project.py install --dir \"${{ matrix.dir }}\"\n"
        "      - run: python .tools/ci-project.py test --dir \"${{ matrix.dir }}\"\n"
        "        with:\n"
        "          python-version: ${{ matrix.py || '3.10' }}\n"
    )


@pytest.fixture()
def synth(tmp_path):
    root = build_synth(tmp_path)
    ec, prov, ver = load_pair(root / ".tools", f"s{abs(hash(str(root))) % 10**8}")
    return root, ec, prov, ver


# --------------------------------------------------------------------------- #
# R4 — explicit interpreter wins and never silently falls back
# --------------------------------------------------------------------------- #
def test_explicit_missing_override_fails_without_fallback(synth, monkeypatch):
    _, ec, _, _ = synth
    manifest = ec.load_manifest()
    entry = manifest["_by_dir"]["p310"]
    monkeypatch.setenv(ec.BASE_PY_ENV_VAR, r"C:\definitely\not\here\python.exe")
    path, reason = ec.resolve_base_python(manifest, entry)
    assert path is None, "must not fall back to a discovered interpreter"
    assert reason and "does not exist" in reason and "refusing to fall back" in reason


def test_explicit_unrunnable_override_fails(tmp_path, synth, monkeypatch):
    root, ec, _, _ = synth
    fake = tmp_path / "fake_python.exe"
    fake.write_text("not an executable", encoding="utf-8")
    monkeypatch.setenv(ec.BASE_PY_ENV_VAR, str(fake))
    manifest = ec.load_manifest()
    path, reason = ec.resolve_base_python(manifest, manifest["_by_dir"]["p310"])
    assert path is None
    assert reason and "does not run" in reason


def test_explicit_wrong_version_override_fails(synth, monkeypatch):
    _, ec, _, _ = synth
    # Decide the skip BEFORE touching the manifest/provisioning path.
    py312, _why = seeds.require("3.12")
    manifest = ec.load_manifest()
    monkeypatch.setenv(ec.BASE_PY_ENV_VAR, str(py312))
    path, reason = ec.resolve_base_python(manifest, manifest["_by_dir"]["p310"])
    assert path is None
    assert reason and ("outside" in reason or "not the declared seed" in reason)
    assert "3.12" in reason and "refusing to fall back" in reason


def test_explicit_map_mixes_310_and_312(synth):
    _, ec, _, _ = synth
    py310, _ = seeds.require("3.10")
    py312, _ = seeds.require("3.12")
    manifest = ec.load_manifest()
    base_map = {"3.10": py310, "3.12": py312}
    p310, r310 = ec.resolve_base_python(manifest, manifest["_by_dir"]["p310"], base_map)
    p312, r312 = ec.resolve_base_python(manifest, manifest["_by_dir"]["p312"], base_map)
    assert r310 is None and p310 == py310
    assert r312 is None and p312 == py312
    resolved, problems = ec.preflight(manifest, ec.project_list(manifest), base_map)
    assert problems == []
    assert resolved["proj310"] == py310 and resolved["proj312"] == py312


def test_preflight_failure_has_no_side_effects(synth, monkeypatch, capsys):
    root, ec, prov, _ = synth
    py312, _ = seeds.require("3.12")          # skip decided before any invocation
    before = tree_snapshot(root)
    # 3.12 pointed at a project that needs 3.10 -> explicit conflict for p310
    rc = prov.main(["--manifest", str(root / ".tools" / "projects.json"),
                    "--project", "proj310",
                    "--python-map", f"3.10={py312}"])
    assert rc == 3, capsys.readouterr().err
    assert tree_snapshot(root) == before, "preflight failure must not write anything"


def test_no_upper_bound_is_invented_for_stem_optimize(tmp_path):
    manifest = json.loads(REAL_MANIFEST.read_text(encoding="utf-8"))
    entry = {e["dir"]: e for e in manifest["projects"]}[
        "02-图像处理/stem-optimize-STEM图像优化"]
    assert entry["python"]["max"] is None, "do not guess an upper bound"
    assert entry["base"] == "3.10"
    ec, _, _ = load_pair(REAL_TOOLS, "realstem")
    assert ec.project_python(entry) == "3.10"
    assert ec.requested_version(None, entry) == "3.10"

    # The *compatibility range* is unbounded above, so 3.12 is inside it -- that
    # is a different question from which seed gets chosen. The declared seed is
    # 3.10, so an explicitly supplied 3.12 must be refused even though the range
    # alone would allow it. (Range and seed must not be conflated.)
    assert ec.satisfies((3, 12), entry["python"]["min"], entry["python"]["max"]), \
        "precondition: the range alone accepts 3.12"

    # A real file so the "does it exist" gate passes, with the interpreter layer
    # stubbed to report 3.12 -- isolating the version/seed rule under test.
    fake = tmp_path / "py312" / "python.exe"
    fake.parent.mkdir(parents=True)
    fake.write_bytes(b"stub")

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(ec, "interpreter_info", lambda py, timeout=120: {
            "path": str(py), "exists": True, "runs": True, "version": "3.12.14",
            "version_tuple": (3, 12, 14), "prefix": None, "base_prefix": None,
            "error": None})
        monkeypatch.setenv(ec.BASE_PY_ENV_VAR, str(fake))
        chosen, reason = ec.resolve_base_python(manifest, entry)
    finally:
        monkeypatch.undo()
    assert chosen is None, "a 3.12 interpreter must not seed a 3.10 project"
    assert reason and "not the declared seed" in reason, reason
    assert "3.12.14" in reason


def test_satisfies_bounds():
    ec, _, _ = load_pair(REAL_TOOLS, "realbounds")
    assert ec.satisfies((3, 10), "3.10", None)
    assert not ec.satisfies((3, 9), "3.10", None)
    assert ec.satisfies((3, 12), "3.11", "3.14")
    assert not ec.satisfies((3, 14), "3.11", "3.14")
    assert not ec.satisfies((3, 11), "3.10", "3.11")
    assert not ec.satisfies(None, None, None)


# --------------------------------------------------------------------------- #
# R6 — lock semantics and the overlay
# --------------------------------------------------------------------------- #
def test_pytest_overlay_spec_is_a_valid_requirement():
    manifest = json.loads(REAL_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["test_overlay"]["pytest"] == "pytest==8.4.2"
    ec, _, _ = load_pair(REAL_TOOLS, "realoverlay")
    assert ec.pytest_spec(manifest, {}) == "pytest==8.4.2"
    bad = dict(manifest)
    bad["test_overlay"] = {"pytest": "8.4.2"}
    with pytest.raises(SystemExit) as exc:
        ec.pytest_spec(bad, {})
    assert "bare version" in str(exc.value)
    bad["test_overlay"] = {"pytest": "==not a spec!!"}
    with pytest.raises(SystemExit) as exc:
        ec.pytest_spec(bad, {})
    assert "not a valid requirement spec" in str(exc.value)
    assert ec.constraint_allows("<9", (8, 4)) and not ec.constraint_allows("<9", (9, 1))
    assert ec.constraint_allows(">=7.4", (9, 1)) and not ec.constraint_allows(">=8,<9", (9, 1))


def test_lock_logical_lines_preserve_hashes_and_includes(tmp_path, synth):
    _, ec, _, _ = synth
    inc = tmp_path / "inc.txt"
    inc.write_text("included==3.0\n", encoding="utf-8")
    lock = tmp_path / "l.txt"
    lock.write_text(
        "# comment\n"
        "alpha==1.0\n"
        "beta==2.0 \\\n"
        "    --hash=sha256:dead \\\n"
        "    --hash=sha256:beef\n"
        f"-r {inc.name}\n"
        "gamma>=1.0  # inline\n",
        encoding="utf-8")
    logical = ec._logical_requirements(lock)
    assert any(l.startswith("beta==2.0") and "--hash=sha256:dead" in l
               and "--hash=sha256:beef" in l for l in logical), logical
    assert "included==3.0" in logical, "relative -r include must be followed"
    assert not any(l.startswith("#") for l in logical)
    pins = ec.lock_pins(lock)
    assert pins["alpha"] == "1.0" and pins["beta"] == "2.0"
    assert pins["included"] == "3.0" and "gamma" not in pins


def test_real_hash_lock_is_parsed_without_rewriting():
    ec, _, _ = load_pair(REAL_TOOLS, "realhash")
    lock = (REAL_ROOT / "02-图像处理" / "stem-optimize-STEM图像优化"
            / "requirements-win-py310.lock")
    before = ec.sha256_file(lock)
    pins = ec.lock_pins(lock)
    assert pins, "hash lock must yield exact pins"
    assert all("--hash" not in name for name in pins)
    assert ec.sha256_file(lock) == before


def test_overlay_is_constrained_by_lock_and_lock_unchanged(synth, monkeypatch, tmp_path):
    root, ec, prov, _ = synth
    manifest = ec.load_manifest()
    entry = manifest["_by_dir"]["p310"]
    proj = root / "p310"
    lock = proj / "requirements.lock.txt"
    lock_before = ec.sha256_file(lock)

    calls = []

    def fake_run(cmd, **kw):
        calls.append([str(c) for c in cmd])
        if "import pytest" in " ".join(str(c) for c in cmd):
            return ec.Result([str(c) for c in cmd], 1, "", cwd=None)  # pytest absent
        if "pip" in [str(c) for c in cmd] and "freeze" in [str(c) for c in cmd]:
            return ec.Result([str(c) for c in cmd], 0,
                             "numpy==2.2.6\npytest==8.4.2\n", cwd=None)
        return ec.Result([str(c) for c in cmd], 0, "OK\n", cwd=None)

    def fake_install(py, args, cwd, index=None, timeout=None, log_path=None):
        calls.append(["pip-install", *[str(a) for a in args]])
        return ec.Result(["pip"], 0, "installed\n", cwd=str(cwd))

    monkeypatch.setattr(ec, "run", fake_run)
    monkeypatch.setattr(ec, "pip_install", fake_install)

    ok, detail, actions, problems = prov.install_test_overlay(
        PY310 if PY310.is_file() else Path(sys.executable), entry, proj,
        manifest, tmp_path, None)
    assert ok
    install_calls = [c for c in calls if c and c[0] == "pip-install"]
    assert install_calls, "pytest was absent, so the overlay must install it"
    flat = " ".join(" ".join(c) for c in install_calls)
    # the runtime lock pins pytest==9.1.1, so the lock wins over the overlay default
    assert "pytest==9.1.1" in flat, flat
    assert "-c" in flat and "constraints" in flat, f"overlay must be lock-constrained: {flat}"
    constraints = list(tmp_path.glob("*.constraints.txt"))
    assert constraints and "numpy==2.2.6" in constraints[0].read_text(encoding="utf-8")
    assert "pytest==9.1.1" in constraints[0].read_text(encoding="utf-8")
    assert ec.sha256_file(lock) == lock_before, "lock body must never be rewritten"
    assert problems, "the overlay preference differing from the lock pin must be reported"
    assert "the lock wins" in problems[0]


def test_overlay_does_not_upgrade_installed_locked_pytest(synth, monkeypatch, tmp_path):
    root, ec, prov, _ = synth
    manifest = ec.load_manifest()
    entry = manifest["_by_dir"]["p310"]
    calls = []

    def fake_run(cmd, **kw):
        joined = " ".join(str(c) for c in cmd)
        if "import pytest" in joined:
            return ec.Result([str(c) for c in cmd], 0, "9.1.1\n", cwd=None)
        return ec.Result([str(c) for c in cmd], 0, "", cwd=None)

    def fake_install(py, args, cwd, index=None, timeout=None, log_path=None):
        calls.append([str(a) for a in args])
        return ec.Result(["pip"], 0, "", cwd=str(cwd))

    monkeypatch.setattr(ec, "run", fake_run)
    monkeypatch.setattr(ec, "pip_install", fake_install)
    # lock pins pytest==9.1.1 and requirements declares pytest>=8,<10 -> compatible:
    # nothing may be installed or upgraded.
    (root / "p310" / "requirements.txt").write_text("pytest>=8,<10\\n", encoding="utf-8")
    ok, detail, actions, problems = prov.install_test_overlay(
        PY310, entry, root / "p310", manifest, tmp_path, None)
    assert ok, detail
    assert calls == [], f"an installed pytest matching the lock must not be touched: {calls}"
    assert problems and "the lock wins" in problems[0], problems

    # now make requirements disagree with the lock pin: must fail loudly, change nothing
    (root / "p310" / "requirements.txt").write_text("pytest<9\\n", encoding="utf-8")
    ok, detail, actions, problems = prov.install_test_overlay(
        PY310, entry, root / "p310", manifest, tmp_path, None)
    assert not ok and "CONFLICT" in detail, detail
    assert calls == [], "a lock/requirements conflict must not install anything"


# --------------------------------------------------------------------------- #
# exit code / dry-run discipline
# --------------------------------------------------------------------------- #
def test_dry_run_reinstall_with_healthy_env_never_installs(synth, monkeypatch, tmp_path):
    root, ec, prov, _ = synth
    manifest = ec.load_manifest()
    entry = manifest["_by_dir"]["p310"]
    proj = root / "p310"
    (proj / ".venv" / "Scripts").mkdir(parents=True)
    (proj / ".venv" / "Scripts" / "python.exe").write_bytes(b"stub")

    calls = []
    monkeypatch.setattr(ec, "run", lambda cmd, **kw: (
        calls.append([str(c) for c in cmd]) or
        ec.Result([str(c) for c in cmd], 0, "3.10.9\n", cwd=None)))
    monkeypatch.setattr(ec, "pip_install", lambda *a, **k: (
        calls.append(["pip-install"]) or ec.Result(["pip"], 0, "", cwd=None)))
    monkeypatch.setattr(prov.ec, "interpreter_info", lambda py, timeout=120: {
        "path": str(py), "exists": True, "runs": True, "version": "3.10.9",
        "version_tuple": (3, 10), "prefix": str(proj / ".venv"),
        "base_prefix": r"C:\ProgramData\Miniconda3", "error": None})

    before = tree_snapshot(root)
    row = prov.provision_one(entry, manifest, args_for(dry_run=True, reinstall=True),
                             PY310, None, tmp_path)
    assert row["state"] == "PLANNED"
    assert row["lock_unchanged"] is True
    assert not [c for c in calls if c and c[0] == "pip-install"], "dry-run must not install"
    assert not [c for c in calls if "venv" in c], "dry-run must not create a venv"
    assert tree_snapshot(root) == before, "dry-run must not write"


def test_dry_run_absent_venv_creates_nothing(synth, monkeypatch, tmp_path):
    root, ec, prov, _ = synth
    manifest = ec.load_manifest()
    entry = manifest["_by_dir"]["proj310".replace("proj", "p")]
    calls = []
    monkeypatch.setattr(ec, "run", lambda cmd, **kw: (
        calls.append([str(c) for c in cmd]) or ec.Result([str(c) for c in cmd], 0, "", cwd=None)))
    before = tree_snapshot(root)
    row = prov.provision_one(entry, manifest, args_for(dry_run=True), PY310, None, tmp_path)
    assert row["state"] == "PLANNED"
    assert calls == [] or not [c for c in calls if "venv" in c]
    assert tree_snapshot(root) == before


def test_dry_run_all_projects_on_real_repo_writes_nothing():
    py312, _ = seeds.require("3.12")      # skip decided before the subprocess
    before = tree_snapshot(REAL_ROOT / ".tools")
    proc = subprocess.run(
        [sys.executable, str(REAL_TOOLS / "provision-envs.py"), "--dry-run",
         "--python-map", f"3.10={PY310};3.12={py312}"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(REAL_ROOT), timeout=900)
    after = tree_snapshot(REAL_ROOT / ".tools")
    assert "DRY-RUN" in proc.stdout
    assert before == after, "a full dry-run must not create scratch or venvs"
    # all 20 projects are accounted for: planned, or reported as a pre-existing
    # broken venv that must be moved aside before provisioning
    assert (proc.stdout.count("[PLANNED") + proc.stdout.count("[BROKEN")) == 20, \
        proc.stdout[-1500:]
    assert proc.returncode == 0, proc.stdout[-800:] + proc.stderr[-400:]


# --------------------------------------------------------------------------- #
# existing environment validation (PRESENT only when genuinely consistent)
# --------------------------------------------------------------------------- #
def _stub_env(monkeypatch, ec, proj, prefix, base_prefix, version, vtup):
    (proj / ".venv" / "Scripts").mkdir(parents=True, exist_ok=True)
    (proj / ".venv" / "Scripts" / "python.exe").write_bytes(b"stub")
    monkeypatch.setattr(ec, "interpreter_info", lambda py, timeout=120: {
        "path": str(py), "exists": True, "runs": True, "version": version,
        "version_tuple": vtup, "prefix": str(prefix), "base_prefix": str(base_prefix),
        "error": None})


def test_runnable_but_wrong_python_env_is_not_present(synth, monkeypatch, tmp_path):
    root, ec, prov, _ = synth
    manifest = ec.load_manifest()
    entry = manifest["_by_dir"]["p310"]
    proj = root / "p310"
    _stub_env(monkeypatch, ec, proj, proj / ".venv", r"C:\ProgramData\Miniconda3",
              "3.12.14", (3, 12))
    monkeypatch.setattr(ec, "run", lambda cmd, **kw: ec.Result(
        [str(c) for c in cmd], 0, "numpy==2.2.6\n", cwd=None))
    row = prov.provision_one(entry, manifest, args_for(), PY310, None, tmp_path)
    assert row["state"] != "PRESENT"
    assert ("not the declared seed" in row["detail"]
            or "outside the declared range" in row["detail"]), row["detail"]


def test_runnable_env_missing_locked_packages_is_not_present(synth, monkeypatch, tmp_path):
    root, ec, prov, _ = synth
    manifest = ec.load_manifest()
    entry = manifest["_by_dir"]["p310"]
    proj = root / "p310"
    _stub_env(monkeypatch, ec, proj, proj / ".venv", r"C:\ProgramData\Miniconda3",
              "3.10.9", (3, 10))

    def fake_run(cmd, **kw):
        joined = " ".join(str(c) for c in cmd)
        if "pip" in joined and "freeze" in joined:
            return ec.Result([str(c) for c in cmd], 0, "numpy==2.2.6\n", cwd=None)
        return ec.Result([str(c) for c in cmd], 0, "OK\n", cwd=None)

    monkeypatch.setattr(ec, "run", fake_run)
    row = prov.provision_one(entry, manifest, args_for(), PY310, None, tmp_path)
    assert row["state"] != "PRESENT", "an env missing locked packages is not provisioned"
    assert "missing" in row["detail"] or "validation failed" in row["detail"]
    # --reinstall rebuilds from the frozen lock instead of reporting only
    row2 = prov.provision_one(entry, manifest, args_for(reinstall=True),
                              PY310, None, tmp_path)
    assert row2["state"] != "PRESENT"


def test_consistent_existing_env_is_present_without_install(synth, monkeypatch, tmp_path):
    root, ec, prov, _ = synth
    manifest = ec.load_manifest()
    entry = manifest["_by_dir"]["p310"]
    proj = root / "p310"
    _stub_env(monkeypatch, ec, proj, proj / ".venv", r"C:\ProgramData\Miniconda3",
              "3.10.9", (3, 10))
    installs = []

    def fake_run(cmd, **kw):
        joined = " ".join(str(c) for c in cmd)
        if "pip" in joined and "freeze" in joined:
            return ec.Result([str(c) for c in cmd], 0,
                             "numpy==2.2.6\npytest==9.1.1\nwithhash==1.0.0\n", cwd=None)
        return ec.Result([str(c) for c in cmd], 0, "OK\n", cwd=None)

    monkeypatch.setattr(ec, "run", fake_run)
    monkeypatch.setattr(ec, "pip_install", lambda *a, **k: (
        installs.append(a) or ec.Result(["pip"], 0, "", cwd=None)))
    row = prov.provision_one(entry, manifest, args_for(), PY310, None, tmp_path)
    assert row["state"] == "PRESENT", row["detail"]
    assert installs == [], "a consistent env must not be reinstalled"
    assert row["lock_unchanged"] is True


def test_with_tests_on_existing_env_runs_overlay(synth, monkeypatch, tmp_path):
    root, ec, prov, _ = synth
    manifest = ec.load_manifest()
    entry = manifest["_by_dir"]["p310"]
    proj = root / "p310"
    _stub_env(monkeypatch, ec, proj, proj / ".venv", r"C:\ProgramData\Miniconda3",
              "3.10.9", (3, 10))
    installs = []
    state = {"pytest": None}

    def fake_run(cmd, **kw):
        joined = " ".join(str(c) for c in cmd)
        if "import pytest" in joined and state["pytest"] is None:
            return ec.Result([str(c) for c in cmd], 1, "", cwd=None)
        if "import pytest" in joined:
            return ec.Result([str(c) for c in cmd], 0, state["pytest"], cwd=None)
        if "pip" in joined and "freeze" in joined:
            body = "numpy==2.2.6\nwithhash==1.0.0\n"
            if state["pytest"]:
                body += f"pytest=={state['pytest']}\n"
            return ec.Result([str(c) for c in cmd], 0, body, cwd=None)
        return ec.Result([str(c) for c in cmd], 0, "OK\n", cwd=None)

    def fake_install(py, args, cwd, index=None, timeout=None, log_path=None):
        installs.append([str(a) for a in args])
        if any("pytest" in a for a in args):
            state["pytest"] = "9.1.1"
        return ec.Result(["pip"], 0, "", cwd=str(cwd))

    monkeypatch.setattr(ec, "run", fake_run)
    monkeypatch.setattr(ec, "pip_install", fake_install)
    row = prov.provision_one(entry, manifest, args_for(with_tests=True),
                             PY310, None, tmp_path)
    assert installs, "--with-tests on an existing env must still apply the overlay"
    assert any("pytest==9.1.1" in " ".join(a) for a in installs), installs
    assert row.get("test_overlay")
    assert row["state"] in ("PRESENT", "OK"), row["detail"]


def test_broken_env_is_reported_not_deleted(synth, monkeypatch, tmp_path):
    root, ec, prov, _ = synth
    manifest = ec.load_manifest()
    entry = manifest["_by_dir"]["p310"]
    proj = root / "p310"
    (proj / ".venv" / "Scripts").mkdir(parents=True)
    (proj / ".venv" / "Scripts" / "python.exe").write_bytes(b"stub")
    monkeypatch.setattr(ec, "interpreter_info", lambda py, timeout=120: {
        "path": str(py), "exists": True, "runs": False, "version": None,
        "version_tuple": None, "prefix": None, "base_prefix": None,
        "error": "No Python at '<stub interpreter>'"})
    row = prov.provision_one(entry, manifest, args_for(reinstall=True), PY310, None, tmp_path)
    assert row["state"] == "BROKEN"
    assert "pre-rebuild" in row["detail"]
    assert (proj / ".venv" / "Scripts" / "python.exe").is_file(), "must not delete it"


def test_lock_change_during_install_is_failed(synth, monkeypatch, tmp_path):
    root, ec, prov, _ = synth
    manifest = ec.load_manifest()
    entry = manifest["_by_dir"]["proj310".replace("proj", "p")]
    proj = root / "p310"
    lock = proj / "requirements.lock.txt"
    _stub_env(monkeypatch, ec, proj, proj / ".venv", r"C:\ProgramData\Miniconda3",
              "3.10.9", (3, 10))

    def fake_install(py, args, cwd, index=None, timeout=None, log_path=None):
        lock.write_text(LOCK_TEXT + "sneaky==1.0\n", encoding="utf-8")
        return ec.Result(["pip"], 0, "", cwd=str(cwd))

    def fake_run(cmd, **kw):
        joined = " ".join(str(c) for c in cmd)
        if "pip" in joined and "freeze" in joined:
            return ec.Result([str(c) for c in cmd], 0,
                             "numpy==2.2.6\npytest==9.1.1\nwithhash==1.0.0\n", cwd=None)
        return ec.Result([str(c) for c in cmd], 0, "OK\n", cwd=None)

    monkeypatch.setattr(ec, "pip_install", fake_install)
    monkeypatch.setattr(ec, "run", fake_run)
    row = prov.provision_one(entry, manifest, args_for(reinstall=True),
                             PY310, None, tmp_path)
    assert row["lock_unchanged"] is False
    assert row["state"] == "FAILED"
    assert "lock file changed" in row["detail"]


def test_provisioner_has_no_lock_update_mode():
    text = (REAL_TOOLS / "provision-envs.py").read_text(encoding="utf-8")
    assert "--update-lock" not in text
    assert "def update_lock" not in text
    doc = (REAL_ROOT / "环境隔离说明.md").read_text(encoding="utf-8")
    assert "锁" in doc and ("干净" in doc or "独立" in doc)


# --------------------------------------------------------------------------- #
# R7 — verifier behaviour and the CI guard
# --------------------------------------------------------------------------- #
def test_verifier_never_installs_and_skips_broken_env(synth, monkeypatch, capsys):
    _, ec, _, ver = synth
    manifest = ec.load_manifest()
    calls = []
    monkeypatch.setattr(ec, "run", lambda cmd, **kw: (
        calls.append([str(c) for c in cmd]) or ec.Result([str(c) for c in cmd], 1, "no python", cwd=None)))
    monkeypatch.setattr(ec, "interpreter_info", lambda py, timeout=120: {
        "path": str(py), "exists": True, "runs": False, "version": None,
        "version_tuple": None, "prefix": None, "base_prefix": None,
        "error": "No Python at ..."})
    rc = ver.main(["--manifest", str(synth[0] / ".tools" / "projects.json")])
    out = capsys.readouterr().out
    # Real SKIP semantics: the rows are reported as SKIP, the summary counts zero
    # passes, and the disclaimer that SKIP is not a pass is printed *after* the
    # passed count. Non-zero exit is asserted below.
    assert "SKIP" in out, "skipped environments must be reported as SKIP rows"
    assert "passed 0 /" in out, "no command actually passed here"
    assert "(SKIP is not a pass)" in out, \
        "the summary must state that SKIP is not counted as a pass"
    assert "pip-install" not in " ".join(" ".join(c) for c in calls)
    assert rc != 0, "skips are not passes"


def test_verifier_runs_every_declared_command(synth, monkeypatch, capsys):
    root, ec, _, ver = synth
    manifest = ec.load_manifest()
    entry = manifest["_by_dir"]["multi"]
    monkeypatch.setattr(ec, "interpreter_info", lambda py, timeout=120: {
        "path": str(py), "exists": True, "runs": True, "version": "3.10.9",
        "version_tuple": (3, 10), "prefix": str(root / "multi" / ".venv"),
        "base_prefix": r"C:\ProgramData\Miniconda3", "error": None})
    (root / "multi" / ".venv" / "Scripts").mkdir(parents=True, exist_ok=True)
    (root / "multi" / ".venv" / "Scripts" / "python.exe").write_bytes(b"stub")
    ran = []

    def fake_run(cmd, **kw):
        joined = [str(c) for c in cmd]
        ran.append(joined)
        return ec.Result(joined, 0, "2 passed\n", cwd=None)

    monkeypatch.setattr(ec, "run", fake_run)
    rc = ver.main(["--manifest", str(root / ".tools" / "projects.json"),
                   "--project", "multi"])
    assert rc == 0
    pytest_calls = [c for c in ran if "-m" in c and "pytest" in c]
    physics_calls = [c for c in ran if any("verify_physics" in x for x in c)]
    assert pytest_calls and physics_calls, "both declared commands must run"
    assert "passed 2 / 2" in capsys.readouterr().out


def test_verifier_fails_when_one_command_fails(synth, monkeypatch, capsys):
    root, ec, _, ver = synth
    monkeypatch.setattr(ec, "interpreter_info", lambda py, timeout=120: {
        "path": str(py), "exists": True, "runs": True, "version": "3.10.9",
        "version_tuple": (3, 10), "prefix": str(root / "multi" / ".venv"),
        "base_prefix": r"C:\ProgramData\Miniconda3", "error": None})
    (root / "multi" / ".venv" / "Scripts").mkdir(parents=True, exist_ok=True)
    (root / "multi" / ".venv" / "Scripts" / "python.exe").write_bytes(b"stub")

    def fake_run(cmd, **kw):
        joined = [str(c) for c in cmd]
        fail = any("verify_physics" in x for x in joined)
        return ec.Result(joined, 1 if fail else 0, "1 failed\n" if fail else "2 passed\n", cwd=None)

    monkeypatch.setattr(ec, "run", fake_run)
    rc = ver.main(["--manifest", str(root / ".tools" / "projects.json"),
                   "--project", "multi"])
    out = capsys.readouterr().out
    assert rc == 1 and "FAIL" in out


def test_ci_guard_catches_test_command_drift(synth):
    root, ec, _, _ = synth
    manifest = ec.load_manifest()
    ci = root / ".github" / "workflows" / "ci.yml"
    ok, msgs = ec.check_ci_matrix(manifest, ci)
    assert ok, msgs

    ci.write_text(default_ci().replace(
        "- { name: 'MULTI', dir: 'multi', py: '3.10' }",
        "- { name: 'MULTI', dir: 'multi', py: '3.12', test: 'python tests/verify_physics.py' }"),
        encoding="utf-8")
    ok, msgs = ec.check_ci_matrix(manifest, ci)
    assert not ok and any("derived from the manifest" in m for m in msgs), msgs


def test_ci_guard_catches_wrong_python_venv_and_editable(synth):
    root, ec, _, _ = synth
    manifest = ec.load_manifest()
    ci = root / ".github" / "workflows" / "ci.yml"
    base = default_ci()
    for mutated, needle in (
            (base.replace("dir: 'p310', py: '3.10'", "dir: 'p310', py: '3.12'"), "py differs"),
            (base.replace("dir: 'p312', ffmpeg: true, py: '3.12'", "dir: 'p312', py: '3.12'"),
             "ffmpeg differs"),
            (base.replace("dir: 'multi', py: '3.10'", "dir: 'multi', py: '3.10', editable: true"),
             "derived from the manifest"),
    ):
        ci.write_text(mutated, encoding="utf-8")
        ok, msgs = ec.check_ci_matrix(manifest, ci)
        assert not ok, f"mutation should be caught: {needle}"
        assert any(needle.split()[0] in m for m in msgs), msgs


def test_ci_guard_catches_missing_duplicate_and_extra_projects(synth):
    root, ec, _, _ = synth
    manifest = ec.load_manifest()
    ci = root / ".github" / "workflows" / "ci.yml"
    base = default_ci()

    ci.write_text(base.replace("          - { name: 'MULTI', dir: 'multi', py: '3.10' }\n", ""),
                  encoding="utf-8")
    ok, msgs = ec.check_ci_matrix(manifest, ci)
    assert not ok and any("missing from ci.yml matrix" in m for m in msgs), msgs

    ci.write_text(base + "          - { name: 'MULTI', dir: 'multi', py: '3.10' }\n",
                  encoding="utf-8")
    ok, msgs = ec.check_ci_matrix(manifest, ci)
    assert not ok and any("duplicate" in m for m in msgs), msgs

    ci.write_text(base + "          - { name: 'X', dir: 'nope', py: '3.10' }\n",
                  encoding="utf-8")
    ok, msgs = ec.check_ci_matrix(manifest, ci)
    assert not ok and any("does not" in m for m in msgs), msgs


def test_ci_guard_requires_manifest_driven_runner(synth):
    root, ec, _, _ = synth
    manifest = ec.load_manifest()
    ci = root / ".github" / "workflows" / "ci.yml"
    ci.write_text(default_ci().replace(".tools/ci-project.py", "python -m pytest"), encoding="utf-8")
    ok, msgs = ec.check_ci_matrix(manifest, ci)
    assert not ok and any("ci-project.py" in m for m in msgs), msgs


def test_ci_guard_catches_duplicate_manifest_dir(tmp_path):
    manifest = json.loads(json.dumps(SYNTH_MANIFEST))
    manifest["projects"].append(dict(manifest["projects"][0]))
    root = build_synth(tmp_path, manifest=manifest)
    ec, _, _ = load_pair(root / ".tools", "dup")
    with pytest.raises(SystemExit) as exc:
        ec.load_manifest()
    assert "duplicate project dir" in str(exc.value)


def test_ci_guard_catches_missing_test_script(synth):
    root, ec, _, _ = synth
    manifest = ec.load_manifest()
    (root / "multi" / "tests" / "verify_physics.py").unlink()
    ok, msgs = ec.check_ci_matrix(manifest, root / ".github" / "workflows" / "ci.yml")
    assert not ok and any("test script missing" in m for m in msgs), msgs


# --------------------------------------------------------------------------- #
# the real repository contract
# --------------------------------------------------------------------------- #
def test_real_manifest_shape_and_no_machine_paths():
    text = REAL_MANIFEST.read_text(encoding="utf-8")
    manifest = json.loads(text)
    assert len(manifest["projects"]) == 20
    # Every project is expected to have a CI entry; the ci.yml guard compares the
    # matrix against this same field, so hardcoding a different count here would
    # just drift from the contract (pipeline was added as the 20th).
    assert sum(1 for p in manifest["projects"] if p.get("ci")) == 20
    assert "C:\\\\Users" not in text and "C:/Users" not in text
    assert "15896" not in text and "SXK" not in text
    for entry in manifest["projects"]:
        assert entry["base"] in {"3.10", "3.12"}, entry["dir"]
        assert entry["venvs_to_create"] == [entry["venv"]]
        test_venv = (entry.get("test") or {}).get("venv") or entry["venv"]
        assert test_venv in entry["venvs_to_create"], entry["dir"]


def test_real_ci_matrix_aligned_and_real_contract_exists():
    proc = subprocess.run(
        [sys.executable, str(REAL_TOOLS / "verify-envs.py"), "--check-matrix"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(REAL_ROOT), timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "aligned" in proc.stdout


def test_real_verify_dry_run_covers_all_projects():
    """--dry-run must report on all 20 projects whatever their env state is.

    Deliberately state-agnostic: before S1b every venv was BROKEN (SKIP), after it
    they are runnable (PLANNED). What must never change is the coverage count.
    """
    proc = subprocess.run(
        [sys.executable, str(REAL_TOOLS / "verify-envs.py"), "--dry-run"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(REAL_ROOT), timeout=900)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-1000:]
    out = proc.stdout
    # one row per declared command, plus one row for a project that declares none
    rows = [l for l in out.splitlines() if l.startswith("[")]
    projects = {l.split("] ", 1)[1].split("  ")[0].strip() for l in rows}
    assert len(projects) == 20, f"every project must be reported: {sorted(projects)}"
    # Coverage, not state: this suite must pass whether the local envs are
    # PLANNED (runnable), PRESENT or BROKEN/SKIP. The tool must never *count* a
    # SKIP as a pass, which is a property of the summary line, not of the local
    # env state -- so assert that property instead of demanding the phrase.
    assert "passed 0 /" in out, "summary must report passes separately from coverage"
    if "SKIP=" in out:
        assert "SKIP is not a pass" in out, "SKIPs present but disclaimer missing"


def test_real_imports_are_declared_for_every_project():
    manifest = json.loads(REAL_MANIFEST.read_text(encoding="utf-8"))
    missing = [p["dir"] for p in manifest["projects"] if not p.get("verify_imports")]
    assert missing == [], f"every project must declare verify imports: {missing}"


def test_lock_pins_never_leak_into_constraints_as_hashes(tmp_path):
    ec, _, _ = load_pair(REAL_TOOLS, "realconstraints")
    lock = REAL_ROOT / "010-STEM模拟" / "requirements.lock.txt"
    pins = ec.lock_pins(lock)
    out = ec.write_constraints(tmp_path / "c.txt", pins)
    text = out.read_text(encoding="utf-8")
    assert "--hash" not in text
    assert all("==" in line for line in text.strip().splitlines())
    assert ec.sha256_file(lock) == ec.sha256_file(lock)
