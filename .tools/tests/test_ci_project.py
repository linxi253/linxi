# -*- coding: utf-8 -*-
"""End-to-end tests for the CI entry point (.tools/ci-project.py).

Unlike the mocked provisioner tests, these build a throwaway repository with its
own manifest and run ``ci-project.py install`` as a real subprocess, so the venv
lifecycle genuinely executes: a project venv is created, a real (tiny, offline)
dependency is installed into *that* venv, and nothing is installed into the
interpreter running the tool.

The one thing stubbed is the index URL: the manifest declares an empty install
chain is not usable for an "install happened" assertion, so a tiny local wheel is
built instead and referenced by absolute path, keeping the test offline.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REAL_TOOLS = Path(__file__).resolve().parent.parent
REAL_ROOT = REAL_TOOLS.parent
CI_PROJECT = REAL_TOOLS / "ci-project.py"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_seeds as seeds  # noqa: E402

_SEED_310, SEED_310_SOURCE = seeds.find("3.10")


def _require_seed_310() -> Path:
    """The fixture hardcodes base/CI 3.10, so require a real 3.10.

    Decided *before* anything is built or provisioned: falling back to
    ``sys.executable`` on a 3.12 runner would contradict the manifest and the
    documented "skip before provision" contract.
    """
    seed, _source = seeds.require("3.10")
    return seed


def write_local_wheel(tmp_path: Path) -> Path:
    """Write a trivial pure-Python wheel directly as a zip.

    Deliberately does **not** shell out to a build backend: that would run
    whatever interpreter happens to be the seed (on this machine the conda
    *base*) with its own setuptools, and the tool venv's pip is too old to drive
    ``pip wheel`` here. A wheel is just a zip with METADATA/WHEEL/RECORD, so the
    standard library alone produces one -- offline, deterministic, no packages
    installed anywhere, and no base interpreter executed.
    """
    import base64
    import hashlib
    import zipfile

    wheel_dir = tmp_path / "wheelhouse"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    wheel = wheel_dir / "tinypkg-1.0.0-py3-none-any.whl"
    dist_info = "tinypkg-1.0.0.dist-info"

    init_py = b"VALUE = 'tiny'\n"
    metadata = ("Metadata-Version: 2.1\n"
                "Name: tinypkg\n"
                "Version: 1.0.0\n"
                "Summary: tiny fixture package\n\n").encode("utf-8")
    wheel_meta = ("Wheel-Version: 1.0\n"
                  "Generator: aifortem-test-fixture\n"
                  "Root-Is-Purelib: true\n"
                  "Tag: py3-none-any\n\n").encode("utf-8")

    def record_line(name: str, data: bytes) -> str:
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
        return f"{name},sha256={digest.rstrip(b'=').decode()},{len(data)}"

    entries = [("tinypkg/__init__.py", init_py),
               (f"{dist_info}/METADATA", metadata),
               (f"{dist_info}/WHEEL", wheel_meta)]
    record = "\n".join(record_line(n, d) for n, d in entries)
    record += f"\n{dist_info}/RECORD,,\n"

    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries:
            zf.writestr(name, data)
        zf.writestr(f"{dist_info}/RECORD", record.encode("utf-8"))
    return wheel


def build_repo(tmp_path: Path, seed: Path, wheel: Path) -> tuple[Path, dict]:
    root = tmp_path / "cirepo"
    tools = root / ".tools"
    tools.mkdir(parents=True)
    for name in ("_envcommon.py", "provision-envs.py", "ci-project.py",
                 "verify-envs.py", "normalise-pth.py"):
        shutil.copy2(REAL_TOOLS / name, tools / name)

    proj = root / "tinyproj"
    (proj / "tests").mkdir(parents=True)
    # A lock that installs one real package from the local wheelhouse.
    (proj / "requirements.lock.txt").write_text(
        f"tinypkg @ file:///{wheel.as_posix()}\n", encoding="utf-8")
    (proj / "requirements.txt").write_text("pytest==8.4.2\n", encoding="utf-8")
    # The project's only test: prove it runs inside the project venv.
    (proj / "tests" / "verify_env.py").write_text(textwrap.dedent("""
        import sys, pathlib
        import tinypkg
        print("PREFIX=" + sys.prefix)
        print("TINY=" + tinypkg.VALUE)
        expected = pathlib.Path(__file__).resolve().parents[1] / ".venv"
        assert pathlib.Path(sys.prefix) == expected, (sys.prefix, str(expected))
        print("OK")
    """), encoding="utf-8")

    manifest = {
        "schema": 2,
        "test_overlay": {"pytest": "pytest==8.4.2"},
        "projects": [{
            "name": "tinyproj", "dir": "tinyproj", "base": "3.10",
            "python": {"min": "3.10", "max": None},
            "venv": ".venv", "venvs_to_create": [".venv"],
            "install": {"chain": ["requirements.lock.txt"], "reproducible": True},
            "tests": [{"label": "verify_env", "argv": ["tests/verify_env.py"]}],
            "test": {"venv": ".venv"},
            "editable": False,
            "verify_imports": ["tinypkg"],
            "ci": {"name": "TINY"},
        }],
    }
    (tools / "projects.json").write_text(json.dumps(manifest), encoding="utf-8")
    gh = root / ".github" / "workflows"
    gh.mkdir(parents=True)
    (gh / "ci.yml").write_text(
        "jobs:\n  test:\n    strategy:\n      matrix:\n        include:\n"
        "          - { name: 'TINY', dir: 'tinyproj', py: '3.10' }\n"
        "    steps:\n"
        "      - run: python .tools/verify-envs.py --check-matrix .github/workflows/ci.yml\n"
        "      - run: python .tools/ci-project.py install --dir \"${{ matrix.dir }}\"\n"
        "      - run: python .tools/ci-project.py test --dir \"${{ matrix.dir }}\"\n"
        "        with:\n          python-version: ${{ matrix.py || '3.10' }}\n",
        encoding="utf-8")
    return root, manifest


def run_ci(root: Path, seed: Path, *args, scratch: Path, script: Path | None = None):
    """Run the *synthetic* repo's own ci-project.py against its own manifest."""
    script = script or (root / ".tools" / "ci-project.py")
    env = dict(os.environ)
    env["PIP_CACHE_DIR"] = str(scratch / "pip-cache")
    env["TMP"] = str(scratch / "tmp")
    env["TEMP"] = str(scratch / "tmp")
    (scratch / "tmp").mkdir(parents=True, exist_ok=True)
    env["AIFORTEM_PYTHONS"] = f"3.10={seed}"
    return subprocess.run(
        [str(seed), str(script), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(root), timeout=1800, env=env)


@pytest.fixture()
def cirepo(tmp_path):
    seed = _require_seed_310()          # skip decided before any build/provision
    wheel = write_local_wheel(tmp_path)
    root, _ = build_repo(tmp_path, seed, wheel)
    return root, seed, tmp_path / "scratch"


def test_ci_install_creates_project_venv_and_does_not_touch_the_seed(cirepo):
    root, seed, scratch = cirepo
    proj = root / "tinyproj"
    assert not (proj / ".venv").exists(), "precondition: no venv in a clean checkout"

    before = subprocess.run(
        [str(seed), "-c", "import importlib.util,sys;"
                          "print('tinypkg' if importlib.util.find_spec('tinypkg') else 'absent')"],
        capture_output=True, text=True, timeout=120).stdout.strip()
    assert before == "absent", "seed interpreter must not have the test package yet"

    res = run_ci(root, seed, "install", "--dir", "tinyproj",
                 "--log", str(scratch / "install.log"), scratch=scratch)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-2000:]

    venv_py = proj / ".venv" / "Scripts" / "python.exe"
    assert venv_py.is_file(), "ci-project install must create the project venv"

    # the dependency landed in the project venv ...
    probe = subprocess.run(
        [str(venv_py), "-c", "import tinypkg,sys;print(tinypkg.VALUE);print(sys.prefix)"],
        capture_output=True, text=True, timeout=120)
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "tiny" in probe.stdout
    # ... and the venv is really the project's, seed unchanged
    assert str(proj / ".venv") in probe.stdout.replace("\\\\", "\\")
    after = subprocess.run(
        [str(seed), "-c", "import importlib.util;"
                          "print('tinypkg' if importlib.util.find_spec('tinypkg') else 'absent')"],
        capture_output=True, text=True, timeout=120).stdout.strip()
    assert after == "absent", "nothing may be installed into the CI/seed interpreter"


def test_ci_test_runs_in_project_venv_and_passes(cirepo):
    root, seed, scratch = cirepo
    res = run_ci(root, seed, "install", "--dir", "tinyproj",
                 "--log", str(scratch / "install.log"), scratch=scratch)
    assert res.returncode == 0, res.stdout[-2000:]

    res = run_ci(root, seed, "test", "--dir", "tinyproj",
                 "--log", str(scratch / "test.log"), scratch=scratch)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-2000:]
    assert "OK" in res.stdout
    assert "all 1 declared command(s) passed" in res.stdout
    # the full log is written, not just a failure tail
    assert (scratch / "test.log").is_file()
    assert "PREFIX=" in (scratch / "test.log").read_text(encoding="utf-8")


def test_ci_test_without_install_fails_clearly(cirepo):
    root, seed, scratch = cirepo
    res = run_ci(root, seed, "test", "--dir", "tinyproj", scratch=scratch)
    assert res.returncode == 1
    assert "venv missing" in res.stderr


def test_ci_install_rejects_wrong_interpreter(cirepo, tmp_path):
    root, seed, scratch = cirepo
    # claim the project needs 3.12 while this job runs 3.10
    manifest_path = root / ".tools" / "projects.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["projects"][0]["base"] = "3.12"
    manifest["projects"][0]["python"] = {"min": "3.11", "max": "3.14"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    res = run_ci(root, seed, "install", "--dir", "tinyproj", scratch=scratch)
    assert res.returncode == 3, res.stdout[-1500:]
    assert "needs Python 3.12" in res.stderr
    assert not (root / "tinyproj" / ".venv").exists(), "must fail before creating anything"


def test_ci_install_is_idempotent_and_second_run_reuses_env(cirepo):
    root, seed, scratch = cirepo
    first = run_ci(root, seed, "install", "--dir", "tinyproj", scratch=scratch)
    assert first.returncode == 0, first.stdout[-2000:]
    stamp = (root / "tinyproj" / ".venv" / "Scripts" / "python.exe").stat().st_mtime_ns

    second = run_ci(root, seed, "install", "--dir", "tinyproj", scratch=scratch)
    assert second.returncode == 0, second.stdout[-2000:]
    assert (root / "tinyproj" / ".venv" / "Scripts" / "python.exe").stat().st_mtime_ns == stamp
    assert "PRESENT" in second.stdout or "OK" in second.stdout


def test_ci_install_does_not_write_the_lock(cirepo):
    root, seed, scratch = cirepo
    lock = root / "tinyproj" / "requirements.lock.txt"
    before = lock.read_bytes()
    res = run_ci(root, seed, "install", "--dir", "tinyproj", scratch=scratch)
    assert res.returncode == 0, res.stdout[-2000:]
    assert lock.read_bytes() == before, "install must never rewrite the lock"


def test_ci_check_pytest_reports_declared_range(cirepo):
    root, seed, scratch = cirepo
    res = run_ci(root, seed, "install", "--dir", "tinyproj", scratch=scratch)
    assert res.returncode == 0, res.stdout[-2000:]
    res = run_ci(root, seed, "check-pytest", "--dir", "tinyproj", scratch=scratch)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "satisfies declared" in res.stdout


# --------------------------------------------------------------------------- #
# discovery paths
# --------------------------------------------------------------------------- #
def test_python_org_discovery_uses_compact_tag(monkeypatch, tmp_path):
    """python.org installs live in Python312, never Python3.12."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ec_paths", REAL_TOOLS / "_envcommon.py")
    ec = importlib.util.module_from_spec(spec)
    sys.modules["ec_paths"] = ec
    spec.loader.exec_module(ec)

    fake_local = tmp_path / "LocalAppData"
    monkeypatch.setenv("LOCALAPPDATA", str(fake_local))
    monkeypatch.setattr(ec, "_win_launcher_candidates", lambda version: [])

    cands = dict()
    for source, path in ec._generic_candidates("3.12"):
        cands.setdefault(source, []).append(str(path))

    org = cands["python-org"][0]
    assert org.endswith(r"Python\Python312\python.exe"), org
    assert "Python3.12" not in org
    assert cands["python-root"][0].endswith(r"C:\Python312\python.exe")

    cands310 = dict()
    for source, path in ec._generic_candidates("3.10"):
        cands310.setdefault(source, []).append(str(path))
    assert cands310["python-org"][0].endswith(r"Python\Python310\python.exe")


def test_python_org_path_is_found_when_it_exists(monkeypatch, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("ec_paths2", REAL_TOOLS / "_envcommon.py")
    ec = importlib.util.module_from_spec(spec)
    sys.modules["ec_paths2"] = ec
    spec.loader.exec_module(ec)

    fake_local = tmp_path / "LocalAppData"
    target = fake_local / "Programs" / "Python" / "Python312" / "python.exe"
    target.parent.mkdir(parents=True)
    shutil.copy2(Path(sys.executable), target)
    monkeypatch.setenv("LOCALAPPDATA", str(fake_local))
    monkeypatch.setattr(ec, "_win_launcher_candidates", lambda version: [])

    found = [p for _src, p in ec._generic_candidates("3.12") if p == target]
    assert found, "the python.org layout must be probed"
