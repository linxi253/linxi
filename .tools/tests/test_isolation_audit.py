# -*- coding: utf-8 -*-
"""Isolation-audit contract tests.

Covers the four defects found by the independent probe:

1. closure coverage: no package-name exemptions, real transitive walk over a
   tiny synthetic metadata graph, and any probe failure => unmeasured (never
   "complete").
2. launcher scan: excluded trees are pruned *before* descent, while a genuine
   bare-``python`` launcher in source is still found.
3. scope: absent manifest projects are reported (strict fails), and a custom
   manifest root is honoured instead of raising ValueError.
4. base resolution: an explicit but unrunnable/wrong-version interpreter is a
   hard error with no fallback.

Everything uses synthetic fixtures; nothing is installed or modified.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

TOOLS = Path(__file__).resolve().parent.parent
ROOT = TOOLS.parent

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_seeds as _seeds  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def iso():
    return _load(TOOLS / "check-env-isolation.py", "isolation_under_test")


@pytest.fixture(scope="module")
def ec():
    return _load(TOOLS / "_envcommon.py", "envcommon_under_test")


PROBE = TOOLS / "_closure_probe.py"


def _make_dist(root: Path, name: str, version: str, requires: list[str]) -> None:
    """Write a minimal installed-distribution metadata directory."""
    info = root / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    lines = [f"Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}"]
    lines += [f"Requires-Dist: {r}" for r in requires]
    (info / "METADATA").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_probe(dist_root: Path, roots: list[str]) -> dict:
    """Run the real probe against a synthetic metadata directory."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(dist_root)
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", str(PROBE), json.dumps(roots)],
        capture_output=True, text=True, encoding="utf-8", env=env, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# 1. closure coverage over a tiny synthetic metadata graph
# ---------------------------------------------------------------------------
def test_probe_walks_transitive_dependencies(tmp_path):
    """root -> mid -> leaf is followed; unpinned transitives are reported."""
    _make_dist(tmp_path, "root", "1.0", ["mid>=1.0"])
    _make_dist(tmp_path, "mid", "1.0", ["leaf>=2.0"])
    _make_dist(tmp_path, "leaf", "2.1", [])
    data = _run_probe(tmp_path, ["root"])
    assert data["packaging"] is True
    assert data["roots_missing"] == []
    assert set(data["unpinned"]) == {"mid", "leaf"}, data["unpinned"]
    assert data["unresolved"] == []


def test_probe_makes_no_package_name_exemptions(tmp_path):
    """setuptools/pip/wheel as real runtime deps must be reported, not skipped."""
    _make_dist(tmp_path, "root", "1.0", ["setuptools>=60", "pip", "wheel"])
    _make_dist(tmp_path, "setuptools", "80.0", [])
    _make_dist(tmp_path, "pip", "25.0", [])
    # wheel is deliberately NOT installed: it must surface as unresolved.
    data = _run_probe(tmp_path, ["root"])
    assert "setuptools" in data["unpinned"], data["unpinned"]
    assert "pip" in data["unpinned"], data["unpinned"]
    assert "wheel" in data["unresolved"], data["unresolved"]


def test_probe_treats_declared_dependency_as_pinned(tmp_path):
    """A transitive dependency that IS declared must not be reported."""
    _make_dist(tmp_path, "root", "1.0", ["mid>=1.0"])
    _make_dist(tmp_path, "mid", "1.0", [])
    data = _run_probe(tmp_path, ["root", "mid"])
    assert data["unpinned"] == []


def test_probe_skips_inactive_markers(tmp_path):
    """Requirements whose markers evaluate false must not count as missing.

    ``sys_platform == "linux"`` is false on this Windows runner, so both
    requirements below are inactive and must disappear from the walk.
    """
    _make_dist(tmp_path, "root", "1.0",
               ['linux-a>=1.0; sys_platform == "linux"',
                'linux-b>=1.0; platform_system == "Linux"'])
    data = _run_probe(tmp_path, ["root"])
    assert data["unpinned"] == []
    assert data["unresolved"] == [], data["unresolved"]
    assert "linux-a" in data["inactive"]
    assert "linux-b" in data["inactive"]


def test_probe_walks_requirements_whose_marker_is_true(tmp_path):
    """The complement of the previous test: an active marker is followed."""
    _make_dist(tmp_path, "root", "1.0", ['win-dep>=1.0; sys_platform == "win32"'])
    data = _run_probe(tmp_path, ["root"])
    assert "win-dep" in data["unresolved"]


def test_probe_reports_unresolved_and_unsatisfied(tmp_path):
    _make_dist(tmp_path, "root", "1.0", ["absent>=1.0", "old<1.0"])
    _make_dist(tmp_path, "old", "2.0", [])
    data = _run_probe(tmp_path, ["root"])
    assert data["unresolved"] == ["absent"]
    assert data["unsatisfied"] == [["old", "<1.0", "2.0"]]


def test_overlay_packages_do_not_count_as_missing(tmp_path):
    """Extra installed packages (a test overlay) are not reported as missing."""
    _make_dist(tmp_path, "root", "1.0", [])
    _make_dist(tmp_path, "pytest", "8.4.2", ["pluggy>=1.5"])
    _make_dist(tmp_path, "pluggy", "1.6.0", [])
    data = _run_probe(tmp_path, ["root"])
    # The walk starts from the *declared* pins only, so overlay packages simply
    # do not appear -- they are neither unpinned findings nor failures.
    assert data["unpinned"] == []
    assert data["unresolved"] == []


@pytest.mark.parametrize("payload,reason", [
    ("", "no output"),
    ("not json at all", "not JSON"),
    (json.dumps({"packaging": True}), "schema mismatch"),
    (json.dumps({k: [] for k in ("packaging", "roots_installed", "roots_missing",
                                 "roots_mismatch", "unpinned", "unresolved",
                                 "unsatisfied", "parse_errors", "inactive")}),
     "packaging false"),
    (json.dumps({"packaging": True, "error": "boom", "roots_installed": {},
                 "roots_missing": [], "roots_mismatch": [], "unpinned": [],
                 "unresolved": [], "unsatisfied": [], "parse_errors": [],
                 "inactive": []}), "probe error"),
])
def test_bad_probe_output_is_never_complete(ec, payload, reason):
    """A truncated/failed probe must yield unmeasured, never complete=True."""
    result = ec.parse_closure_probe(payload, {"root": "1"})
    assert result["complete"] is not True, reason
    assert result.get("error"), reason


def test_probe_with_unparseable_requirement_is_unmeasured(ec):
    """An unevaluable marker/requirement is not evidence of success."""
    payload = {"packaging": True, "roots_installed": {"root": "1"},
               "roots_missing": [], "roots_mismatch": [], "unpinned": [],
               "unresolved": [], "unsatisfied": [], "parse_errors": ["weird~~1"],
               "inactive": []}
    result = ec.parse_closure_probe(json.dumps(payload), {"root": "1"})
    assert result["complete"] is None
    assert "unparseable" in result["error"]


def test_closure_report_nonzero_exit_is_unmeasured(iso):
    """A probe that exits non-zero must not be reported as complete."""
    replies = [subprocess.CompletedProcess([], 2, "interpreter exploded")]
    with patch.object(iso, "run", side_effect=replies):
        report = iso.closure_report(Path(sys.executable), {"root": "1"})
    assert report["complete"] is None
    assert "exited 2" in report["error"]


def test_closure_report_single_probe_call(iso):
    """Exactly one subprocess per project (no fragile two-round protocol)."""
    good = {"packaging": True, "roots_installed": {"root": "1"}, "roots_missing": [],
            "roots_mismatch": [], "unpinned": [], "unresolved": [], "unsatisfied": [],
            "parse_errors": [], "inactive": []}
    calls: list = []

    def fake_run(cmd, cwd=None, timeout=300):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, json.dumps(good))

    with patch.object(iso, "run", side_effect=fake_run):
        report = iso.closure_report(Path(sys.executable), {"root": "1"})
    assert len(calls) == 1, f"expected a single probe call, got {len(calls)}"
    assert report["complete"] is True


def test_closure_report_missing_or_wrong_dep_is_broken(ec):
    payload = {"packaging": True, "roots_installed": {"root": "1"},
               "roots_missing": ["ghost"], "roots_mismatch": [], "unpinned": [],
               "unresolved": ["absent"], "unsatisfied": [], "parse_errors": [],
               "inactive": []}
    result = ec.parse_closure_probe(json.dumps(payload), {"root": "1", "ghost": "1"})
    assert result["complete"] is False
    assert set(result["broken"]) == {"ghost", "absent"}


# ---------------------------------------------------------------------------
# 2. directory pruning for launcher scripts
# ---------------------------------------------------------------------------
EXCLUDED = (".venv", ".venv.pre-rebuild-20261003", ".venv-backup-py311-20261004",
            "dist", "build", "08-历史版本",
            "data", "cache", ".review-tmp", "site-packages", "node_modules")


def test_launcher_scan_prunes_before_descending(iso, tmp_path):
    """Excluded trees are never entered, and a source launcher is still found."""
    proj = tmp_path / "proj"
    for name in EXCLUDED:
        d = proj / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "ignored.bat").write_text("python ignored.py\n", encoding="utf-8")
    src = proj / "src"
    src.mkdir(parents=True)
    (src / "real.ps1").write_text("python real.py\n", encoding="utf-8")

    visited: list[str] = []
    real_walk = os.walk

    def recording_walk(top, *a, **kw):
        for current, dirnames, filenames in real_walk(top, *a, **kw):
            visited.append(str(current))
            yield current, dirnames, filenames

    with patch.object(iso.os, "walk", recording_walk):
        found = sorted(str(p.relative_to(proj)) for p in iso.iter_launcher_scripts(proj))

    assert found == [str(Path("src") / "real.ps1")], found
    for name in EXCLUDED:
        assert not any(Path(v) == proj / name for v in visited), \
            f"{name} was descended into"
    assert any(Path(v) == src for v in visited), "source directory must be scanned"


def test_excluded_launchers_do_not_produce_findings(iso, tmp_path):
    """A synthetic manifest over a tree full of excluded launchers stays clean."""
    proj = tmp_path / "proj"
    proj.mkdir(parents=True)
    (proj / "requirements.lock.txt").write_text("pytest==8.4.2\n", encoding="utf-8")
    for name in EXCLUDED:
        d = proj / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "ignored.bat").write_text("python ignored.py\n", encoding="utf-8")
    manifest = {"projects": [{"name": "p", "dir": "proj", "venv": ".venv",
                              "install": {"chain": ["requirements.lock.txt"]}}],
                "_root": str(tmp_path)}
    findings = iso.Findings()
    with contextlib.redirect_stdout(io.StringIO()):
        iso.check_scripts(manifest, findings)
    assert findings.problems == [], findings.problems


def test_bare_python_in_source_is_still_reported(iso, tmp_path):
    proj = tmp_path / "proj"
    (proj / "tools").mkdir(parents=True)
    (proj / "tools" / "run.bat").write_text("python app.py\n", encoding="utf-8")
    manifest = {"projects": [{"name": "p", "dir": "proj", "venv": ".venv",
                              "install": {"chain": ["requirements.lock.txt"]}}],
                "_root": str(tmp_path)}
    findings = iso.Findings()
    with contextlib.redirect_stdout(io.StringIO()):
        iso.check_scripts(manifest, findings)
    assert len(findings.problems) == 1
    assert "bare python" in findings.problems[0]


def test_launcher_scan_ignores_unreadable_binary(iso, tmp_path):
    """An unrelated binary with a .bat name inside an excluded tree is not read."""
    proj = tmp_path / "proj"
    d = proj / ".venv"
    d.mkdir(parents=True)
    (d / "weird.bat").write_bytes(bytes(range(256)))
    assert list(iso.iter_launcher_scripts(proj)) == []


# ---------------------------------------------------------------------------
# 3. scope completeness and custom manifest root
# ---------------------------------------------------------------------------
def test_absent_project_is_reported_in_strict(iso):
    findings = iso.Findings()
    manifest = {"projects": [{"name": "gone", "dir": "absent-project", "venv": ".venv",
                              "install": {"chain": ["requirements.lock.txt"]}}],
                "_root": str(Path(iso.__file__).resolve().parent.parent)}
    with contextlib.redirect_stdout(io.StringIO()):
        iso.check_projects(manifest, findings, True)
    assert findings.problems, "an absent declared project must not be silently skipped"
    assert "missing" in findings.problems[0]


def test_absent_project_is_info_in_default_mode(iso, tmp_path):
    findings = iso.Findings()
    manifest = {"projects": [{"name": "gone", "dir": "absent-project", "venv": ".venv",
                              "install": {"chain": ["requirements.lock.txt"]}}],
                "_root": str(tmp_path)}
    with contextlib.redirect_stdout(io.StringIO()):
        iso.check_projects(manifest, findings, False)
    assert findings.problems == []


def test_manifest_root_prefers_recorded_root(iso, tmp_path):
    assert iso.manifest_root({"_root": str(tmp_path)}) == tmp_path
    assert iso.manifest_root({"_path": str(tmp_path / ".tools" / "projects.json")}) \
        == tmp_path


def test_custom_manifest_root_does_not_raise(iso, tmp_path):
    """A manifest outside the tool's own tree must not break relative paths."""
    fixture = tmp_path / "repo"
    (fixture / ".tools").mkdir(parents=True)
    (fixture / "proj").mkdir()
    (fixture / "proj" / "requirements.lock.txt").write_text("pytest==8.4.2\n",
                                                            encoding="utf-8")
    (fixture / ".tools" / "projects.json").write_text(json.dumps({
        "projects": [{"name": "p", "dir": "proj", "venv": ".venv",
                      "install": {"chain": ["requirements.lock.txt"]}}]}),
        encoding="utf-8")
    out = tmp_path / "report.json"
    with contextlib.redirect_stdout(io.StringIO()):
        code = iso.main(["--manifest", str(fixture / ".tools" / "projects.json"),
                         "--json", str(out)])
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["workspace"] == str(fixture)


def test_relpath_helper_falls_back_to_absolute(iso, tmp_path):
    outside = tmp_path / "elsewhere" / "x.bat"
    displayed = iso.display(outside, tmp_path / "other")
    assert displayed == str(outside)


# ---------------------------------------------------------------------------
# 4. explicit base interpreter
# ---------------------------------------------------------------------------
def test_missing_explicit_base_is_a_hard_error(iso, tmp_path):
    value, error = iso.resolve_base(str(tmp_path / "nope" / "python.exe"))
    assert value is None and error and "does not exist" in error


def test_existing_but_unrunnable_explicit_base_is_rejected(iso, tmp_path):
    """A text file named python.exe must not be accepted as an interpreter."""
    fake = tmp_path / "fake-python.exe"
    fake.write_bytes(b"not an executable")
    value, error = iso.resolve_base(str(fake))
    assert value is None
    assert error and "not runnable" in error


def test_no_request_means_no_base_required(iso, monkeypatch):
    monkeypatch.delenv("AIFORTEM_BASE_PY", raising=False)
    value, error = iso.resolve_base(None)
    assert value is None and error is None


def test_working_explicit_base_is_accepted(iso):
    value, error = iso.resolve_base(sys.executable)
    assert error is None
    assert value is not None and Path(value) == Path(sys.executable)


def test_bad_explicit_map_is_rejected_without_fallback(iso, monkeypatch, tmp_path):
    fake = tmp_path / "fake.exe"
    fake.write_bytes(b"nope")
    monkeypatch.setenv("AIFORTEM_PYTHONS", f"3.10={fake}")
    checked, error = iso.check_base_map()
    assert checked == []
    # A file that is not a working interpreter must be rejected either way.
    assert error and ("does not run" in error or "does not exist" in error)


def test_garbage_map_is_not_silently_empty(iso, monkeypatch):
    """A non-empty but unparseable map must be an error, not 'nothing given'."""
    monkeypatch.setenv("AIFORTEM_PYTHONS", "garbage")
    checked, error = iso.check_base_map()
    assert checked == []
    assert error, "a garbage map must never be reported as verified"
    assert "unusable" in error or "no 'version=path'" in error


def test_parse_base_map_rejects_garbage(ec):
    """The shared parser itself must not turn garbage into an empty mapping."""
    with pytest.raises(BaseException) as excinfo:
        ec.parse_base_map("garbage")
    assert "usable" in str(excinfo.value)
    with pytest.raises(BaseException):
        ec.parse_base_map("3.10=;3.12=")


def test_parse_base_map_accepts_real_entries(ec, tmp_path):
    parsed = ec.parse_base_map(f"3.10={tmp_path/'a.exe'};3.12={tmp_path/'b.exe'}")
    assert set(parsed) == {"3.10", "3.12"}


def test_good_explicit_map_is_verified(iso, monkeypatch):
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    monkeypatch.setenv("AIFORTEM_PYTHONS", f"{current}={sys.executable}")
    checked, error = iso.check_base_map()
    assert error is None
    assert [v for v, _ in checked] == [current]


def test_map_key_minor_must_match_the_interpreter(iso, monkeypatch):
    """A key whose interpreter reports a different minor is an error."""
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    wrong = "3.99" if current != "3.99" else "3.98"
    monkeypatch.setenv("AIFORTEM_PYTHONS", f"{wrong}={sys.executable}")
    checked, error = iso.check_base_map()
    assert checked == []
    assert error and "key says" in error


def test_wrong_minor_base_is_rejected_against_the_declared_manifest(iso, monkeypatch,
                                                                   tmp_path):
    """The requested base must match a version the manifest actually declares.

    Drives the real CLI with a synthetic manifest declaring a seed the running
    interpreter does not satisfy, so this covers the actual code path
    (manifest -> requested_version -> resolve_base) rather than comparing map
    keys in isolation. Deterministic regardless of what is installed here.
    """
    fixture = tmp_path / "repo"
    (fixture / ".tools").mkdir(parents=True)
    (fixture / "proj").mkdir()
    (fixture / "proj" / "requirements.lock.txt").write_text("pytest==8.4.2\n",
                                                            encoding="utf-8")
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    other = "3.99"
    (fixture / ".tools" / "projects.json").write_text(json.dumps({
        "projects": [{"name": "p", "dir": "proj", "venv": ".venv",
                      "base": other, "python": {"min": other, "max": None},
                      "install": {"chain": ["requirements.lock.txt"]}}]}),
        encoding="utf-8")

    out = tmp_path / "report.json"
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        code = iso.main(["--manifest", str(fixture / ".tools" / "projects.json"),
                         "--base-python", sys.executable, "--json", str(out)])
    assert code == 1, "a wrong-minor explicit base must fail the audit"
    text = captured.getvalue()
    assert "not one of the versions the manifest declares" in text
    assert other in text and current in text


def test_cli_accepts_a_matching_declared_base(iso, tmp_path):
    """The complement: a base matching the declared seed is accepted."""
    fixture = tmp_path / "repo"
    (fixture / ".tools").mkdir(parents=True)
    (fixture / "proj").mkdir()
    (fixture / "proj" / "requirements.lock.txt").write_text("pytest==8.4.2\n",
                                                            encoding="utf-8")
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    (fixture / ".tools" / "projects.json").write_text(json.dumps({
        "projects": [{"name": "p", "dir": "proj", "venv": ".venv",
                      "base": current, "python": {"min": current, "max": None},
                      "install": {"chain": ["requirements.lock.txt"]}}]}),
        encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        code = iso.main(["--manifest", str(fixture / ".tools" / "projects.json"),
                         "--base-python", sys.executable])
    assert code == 0, captured.getvalue()
    assert "requested base runs" in captured.getvalue()


def test_no_requested_base_leaves_existing_envs_auditable(iso, monkeypatch):
    """With no explicit base the audit must not discover or require one."""
    monkeypatch.delenv("AIFORTEM_BASE_PY", raising=False)
    manifest = iso.ec.load_manifest(iso.TOOLS / "projects.json")
    value, error = iso.resolve_base(None, manifest)
    assert value is None and error is None


def test_unset_map_is_not_checked(iso, monkeypatch):
    monkeypatch.delenv("AIFORTEM_PYTHONS", raising=False)
    checked, error = iso.check_base_map()
    assert checked == [] and error is None


# --- default= fallback (reviewer/default-map-round27.json) ------------------

def _all_310_manifest(iso):
    return {"_root": str(iso.TOOLS.parent),
            "projects": [{"name": "a", "dir": "x", "venv": ".venv", "base": "3.10",
                          "python": {"min": "3.10", "max": None},
                          "install": {"chain": ["requirements.lock.txt"]}}]}


def test_default_key_is_supported_not_rejected(iso, monkeypatch):
    """`default=<path>` is a documented fallback, not an invalid key."""
    py310, _ = _seeds.require("3.10")
    monkeypatch.setenv("AIFORTEM_PYTHONS", f"default={py310}")
    _checked, error = iso.check_base_map(_all_310_manifest(iso))
    assert error is None, error
    assert not (error and "is not a version" in error)


def test_default_covers_an_all_310_manifest(iso, monkeypatch):
    py310, _ = _seeds.require("3.10")
    monkeypatch.setenv("AIFORTEM_PYTHONS", f"default={py310}")
    checked, error = iso.check_base_map(_all_310_manifest(iso))
    assert error is None, error
    assert checked


def test_default_does_not_cover_a_mixed_manifest(iso, monkeypatch):
    """A 3.10 default cannot satisfy projects that declare 3.12."""
    py310, _ = _seeds.require("3.10")
    manifest = {"_root": str(iso.TOOLS.parent),
                "projects": [{"name": "p310", "dir": "x", "venv": ".venv", "base": "3.10",
                              "python": {"min": "3.10", "max": None},
                              "install": {"chain": ["requirements.lock.txt"]}},
                             {"name": "p312", "dir": "y", "venv": ".venv", "base": "3.12",
                              "python": {"min": "3.11", "max": "3.14"},
                              "install": {"chain": ["requirements.lock.txt"]}}]}
    monkeypatch.setenv("AIFORTEM_PYTHONS", f"default={py310}")
    _checked, error = iso.check_base_map(manifest)
    assert error, "a mixed manifest must require an explicit 3.12 entry"
    assert "p312" in error and "outside the declared range" in error
    assert "p310" not in error, "the 3.10 project IS covered by the default"


def test_explicit_entry_overrides_default(iso, monkeypatch):
    """With `default` plus an explicit 3.12, the mixed manifest is satisfied."""
    py310, _ = _seeds.require("3.10")
    py312, _ = _seeds.require("3.12")
    manifest = {"_root": str(iso.TOOLS.parent),
                "projects": [{"name": "p310", "dir": "x", "venv": ".venv", "base": "3.10",
                              "python": {"min": "3.10", "max": None},
                              "install": {"chain": ["requirements.lock.txt"]}},
                             {"name": "p312", "dir": "y", "venv": ".venv", "base": "3.12",
                              "python": {"min": "3.11", "max": "3.14"},
                              "install": {"chain": ["requirements.lock.txt"]}}]}
    monkeypatch.setenv("AIFORTEM_PYTHONS", f"default={py310};3.12={py312}")
    checked, error = iso.check_base_map(manifest)
    assert error is None, error
    labels = {label for label, _ in checked}
    assert "3.12" in labels


def test_explicit_wrong_minor_entry_is_not_masked_by_default(iso, monkeypatch):
    """An explicit key is still checked even when a default is present."""
    py310, _ = _seeds.require("3.10")
    py312, _ = _seeds.require("3.12")
    manifest = {"_root": str(iso.TOOLS.parent),
                "projects": [{"name": "p310", "dir": "x", "venv": ".venv", "base": "3.10",
                              "python": {"min": "3.10", "max": None},
                              "install": {"chain": ["requirements.lock.txt"]}}]}
    monkeypatch.setenv("AIFORTEM_PYTHONS", f"default={py310};3.10={py312}")
    _checked, error = iso.check_base_map(manifest)
    assert error, "3.10=3.12 must still be rejected"
    assert "key says 3.10" in error


def test_unknown_non_default_key_is_still_rejected(iso, monkeypatch):
    py310, _ = _seeds.require("3.10")
    monkeypatch.setenv("AIFORTEM_PYTHONS", f"nonsense={py310}")
    _checked, error = iso.check_base_map(_all_310_manifest(iso))
    assert error and "neither a version nor a 'default' fallback" in error


# ---------------------------------------------------------------------------
# 5. no installation side effects
# ---------------------------------------------------------------------------
def test_audit_never_invokes_a_installer():
    """The audit tool must not shell out to pip (read-only contract)."""
    text = (TOOLS / "check-env-isolation.py").read_text(encoding="utf-8")
    for forbidden in ("pip install", "pip\", \"install", "provision-envs"):
        assert forbidden not in text, f"audit must not install: found {forbidden!r}"
