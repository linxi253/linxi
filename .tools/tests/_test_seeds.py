# -*- coding: utf-8 -*-
"""Portable discovery of real CPython seeds for the tool tests.

Tests must not hardcode one machine's interpreter paths. A seed is resolved in
this order:

1. an explicit override environment variable (``AIFORTEM_TEST_PY310`` /
   ``AIFORTEM_TEST_PY312``) -- the "explicitly injected" route, used on CI or on
   any machine whose layout differs;
2. the shared discovery in :mod:`_envcommon` (the Windows ``py`` launcher and
   the standard install roots), verified to actually run *that* minor version.

If no suitable interpreter can be found the caller skips **before** doing any
work (never after a provisioning attempt). Pure-logic tests mock interpreter
versions instead of requiring a real one.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent


def _load_envcommon():
    existing = sys.modules.get("_envcommon_seeds")
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location("_envcommon_seeds",
                                                  TOOLS / "_envcommon.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_envcommon_seeds"] = module
    spec.loader.exec_module(module)
    return module


ec = _load_envcommon()

OVERRIDE_VARS = {"3.10": "AIFORTEM_TEST_PY310", "3.12": "AIFORTEM_TEST_PY312"}


def _runs_minor(path: Path, minor: str) -> bool:
    if not path.is_file():
        return False
    info = ec.interpreter_info(path)
    if not info.get("runs"):
        return False
    expected = ec.parse_version(minor)
    return bool(expected) and info.get("version_tuple") == expected


def find(minor: str) -> tuple[Path | None, str]:
    """Return ``(path, source)`` for a real interpreter of ``minor``."""
    var = OVERRIDE_VARS.get(minor)
    if var:
        value = os.environ.get(var, "").strip()
        if value:
            candidate = Path(value).expanduser()
            if _runs_minor(candidate, minor):
                return candidate, f"{var} override"
            return None, (f"{var} was set to {candidate}, which is not a runnable "
                          f"Python {minor}")
    for source, candidate in ec._generic_candidates(minor):
        if _runs_minor(candidate, minor):
            return candidate, f"discovered ({source})"
    return None, f"no runnable Python {minor} found"


def require(minor: str):
    """Skip (import-time caller decides) unless a real ``minor`` seed exists.

    Returns ``(path, source)``. Raises :class:`pytest.skip` via the caller's
    ``pytest`` module so the decision happens before any provisioning call.
    """
    import pytest

    path, source = find(minor)
    if path is None:
        pytest.skip(f"[seed] {source}; set {OVERRIDE_VARS.get(minor)} to provide one")
    return path, source


def find_312() -> tuple[Path | None, str]:
    return find("3.12")


def find_310() -> tuple[Path | None, str]:
    return find("3.10")
