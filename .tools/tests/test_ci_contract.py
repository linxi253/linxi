# -*- coding: utf-8 -*-
"""CI contract checks that the 20-matrix guard does not cover.

The matrix guard proves identity/version fields line up; it cannot prove that a
downloaded asset lands where the code looks for it, or that every declared test
entry is actually reachable. These checks are static (no network, no downloads).
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent
ROOT = TOOLS.parent
CI = ROOT / ".github" / "workflows" / "ci.yml"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def verifier():
    return _load(TOOLS / "verify_ffmpeg_asset.py", "verify_ffmpeg_asset")


@pytest.fixture(scope="module")
def ci_text():
    return CI.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# FFmpeg destination
# ---------------------------------------------------------------------------
def test_ffmpeg_download_target_is_the_consumer_directory(ci_text, verifier):
    """The download step must not *write* into ``matrix.dir``.

    That form puts the asset under 10-DSH集成 for the pipeline job, where
    FFmpegManager never looks (it resolves relative to its own package root).
    Comments may still mention the wrong form while explaining why it is wrong,
    so only executable lines are checked.
    """
    step = ci_text.split("Provide FFmpeg")[1].split("- name:")[0]
    code_lines = [ln for ln in step.splitlines()
                  if ln.strip() and not ln.lstrip().startswith("#")]
    offenders = [ln.strip() for ln in code_lines if "matrix.dir" in ln]
    assert offenders == [], f"FFmpeg step writes into matrix.dir: {offenders}"
    assert "FFMPEG_CONSUMER_DIR" in ci_text, \
        "the consumer directory must be declared once, explicitly"
    assert "verify_ffmpeg_asset.py" in ci_text, \
        "the download step must assert its destination matches the consumer"


def test_consumer_dir_matches_the_resolver(verifier):
    """The declared consumer dir must equal where FFmpegManager actually looks."""
    expected = ROOT / "01-视频与数据提取" / "视频切片工具" / "tools" / "ffmpeg"
    assert verifier.consumer_dir() / "tools" / "ffmpeg" == expected
    # Mirror of FFmpegManager._bundled_candidates(): root/tools/ffmpeg/ffmpeg.exe
    assert expected / "ffmpeg.exe" in verifier.consumer_candidates()


def test_verifier_rejects_a_wrong_destination(tmp_path, verifier, capsys):
    """A mismatched destination must fail loudly (not warn)."""
    rc = verifier.main(["--expect-dir", str(tmp_path)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "consumer resolves" in err


def test_verifier_accepts_the_consumer_directory(tmp_path, verifier, monkeypatch, capsys):
    """With the binaries present the check passes."""
    fake_root = tmp_path / "视频切片工具"
    binaries = fake_root / "tools" / "ffmpeg"
    binaries.mkdir(parents=True)
    (binaries / "ffmpeg.exe").write_bytes(b"x")
    (binaries / "ffprobe.exe").write_bytes(b"x")
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    monkeypatch.setattr(verifier, "CONSUMER_PACKAGE_PARENT", Path("视频切片工具"))
    rc = verifier.main(["--expect-dir", str(binaries)])
    assert rc == 0, capsys.readouterr().err


def test_pipeline_job_declares_ffmpeg(ci_text):
    """The 20th matrix entry (pipeline) needs FFmpeg for its real chain."""
    match = re.search(r"10-DSH集成',\s*ffmpeg:\s*(true|false)", ci_text)
    assert match and match.group(1) == "true", \
        "the pipeline matrix entry must request the FFmpeg asset"


# ---------------------------------------------------------------------------
# declared test entries are real and reachable
# ---------------------------------------------------------------------------
def test_declared_test_scripts_exist():
    """Every ``argv`` pointing at a file must reference a file that exists."""
    manifest = json.loads((TOOLS / "projects.json").read_text(encoding="utf-8"))
    missing = []
    for entry in manifest["projects"]:
        proj = ROOT / entry["dir"]
        for spec in entry.get("tests") or []:
            for token in spec.get("argv") or []:
                if token.startswith("-") or token in ("-m", "pytest"):
                    continue
                if token.endswith(".py") and not (proj / token).is_file():
                    missing.append(f"{entry['dir']}::{token}")
    assert missing == [], f"declared test entry points do not exist: {missing}"


def test_projects_with_test_suites_declare_a_runner():
    """A project shipping test files must declare *some* runner.

    Regression: Suite and 010 had pytest suites but the manifest still described
    them as script-only, so CI never ran those tests. Note that the runner does
    not have to be pytest: a standard-library ``unittest`` suite (PPA) is equally
    valid, so this checks for a real test entry rather than for the word pytest.
    """
    manifest = json.loads((TOOLS / "projects.json").read_text(encoding="utf-8"))
    undeclared = []
    for entry in manifest["projects"]:
        proj = ROOT / entry["dir"]
        tests_dir = proj / "tests"
        if not tests_dir.is_dir():
            continue
        suite_files = set(tests_dir.glob("test_*.py")) | set(tests_dir.rglob("test_*.py"))
        if not suite_files:
            continue
        specs = entry.get("tests") or []
        runs_a_suite = any(
            any(tok in ("pytest", "unittest") for tok in (spec.get("argv") or []))
            for spec in specs)
        if not runs_a_suite:
            undeclared.append(f"{entry['dir']} (has {len(suite_files)} test_*.py)")
    assert undeclared == [], f"test suites exist but no runner is declared: {undeclared}"


def test_ppa_keeps_the_unittest_entry():
    """PPA's original entry (`-m unittest discover -s tests -v`) must survive.

    Its independent 92-item evidence was produced through that entry point.
    """
    manifest = json.loads((TOOLS / "projects.json").read_text(encoding="utf-8"))
    ppa = next(e for e in manifest["projects"] if "PPA" in e["name"])
    argvs = [spec.get("argv") for spec in ppa.get("tests") or []]
    expected = ["-m", "unittest", "discover", "-s", "tests", "-v"]
    assert expected in argvs, f"PPA must keep the unittest discover entry: {argvs}"


def test_no_stale_claim_that_a_suite_is_not_pytest():
    """Notes must not claim a project 'has no pytest' when it declares a runner."""
    manifest = json.loads((TOOLS / "projects.json").read_text(encoding="utf-8"))
    for entry in manifest["projects"]:
        note = entry.get("note") or ""
        argvs = [spec.get("argv") or [] for spec in entry.get("tests") or []]
        has_runner = any(("pytest" in a) or ("unittest" in a) for a in argvs)
        if has_runner:
            assert "不是 pytest 套件" not in note, entry["dir"]
            assert "无法pytest" not in note and "无法 pytest" not in note, entry["dir"]
