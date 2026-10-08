# -*- coding: utf-8 -*-
"""Standalone dependency-closure probe. Run BY a project interpreter.

    python _closure_probe.py '["numpy","pytest",...]'

The argument is the set of *declared* pins (normalised names) that act as the
roots of the walk. The probe reports, as one JSON object on stdout:

  packaging           bool   -- ``packaging`` importable (specifier parsing)
  roots_installed     {name: version}
  roots_missing       [name]              declared but not installed
  roots_mismatch      [[name, want, got]] declared version != installed version
  unpinned            [name]              active transitive dep not declared
  unresolved          [name]              active transitive dep not installed
  unsatisfied         [[name, spec, got]] installed but violates the specifier
  parse_errors        [string]            requirement we could not parse
  inactive            [name]              requirement skipped by its marker

Deliberately **no** package-name exemptions: if ``setuptools``/``pip``/``wheel``
is an active runtime requirement it is reported like any other dependency.

Every key above is always present; a consumer must treat a reply missing any of
them as a failed probe rather than as "nothing missing".
"""
from __future__ import annotations

import json
import sys


def norm(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def main(argv: list[str]) -> int:
    roots = {norm(n) for n in json.loads(argv[1])} if len(argv) > 1 else set()

    result: dict = {
        "packaging": False,
        "roots_installed": {},
        "roots_missing": [],
        "roots_mismatch": [],
        "unpinned": [],
        "unresolved": [],
        "unsatisfied": [],
        "parse_errors": [],
        "inactive": [],
    }

    try:
        import importlib.metadata as md
        from packaging.requirements import Requirement
        from packaging.specifiers import SpecifierSet
    except Exception as exc:                       # noqa: BLE001 - reported, not raised
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(result))
        return 0
    result["packaging"] = True

    distributions: dict[str, object] = {}
    versions: dict[str, str] = {}
    for dist in md.distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        key = norm(name)
        distributions.setdefault(key, dist)
        versions.setdefault(key, dist.version)

    # --- roots -------------------------------------------------------------
    for root in sorted(roots):
        if root in versions:
            result["roots_installed"][root] = versions[root]
        else:
            result["roots_missing"].append(root)

    # --- transitive walk from installed roots ------------------------------
    seen: set[str] = set()
    queue = [r for r in sorted(roots) if r in versions]
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        dist = distributions.get(current)
        if dist is None:
            continue
        for raw in (dist.requires or []):
            try:
                requirement = Requirement(raw)
            except Exception:                      # noqa: BLE001
                result["parse_errors"].append(raw)
                continue
            if requirement.marker is not None:
                try:
                    if not requirement.marker.evaluate():
                        result["inactive"].append(norm(requirement.name))
                        continue
                except Exception:                  # noqa: BLE001
                    # An unevaluable marker is NOT evidence of success.
                    result["parse_errors"].append(raw)
                    continue
            dep = norm(requirement.name)
            if dep in versions:
                queue.append(dep)
                specifier = requirement.specifier
                if specifier and not SpecifierSet(str(specifier)).contains(
                        versions[dep], prereleases=True):
                    result["unsatisfied"].append([dep, str(specifier), versions[dep]])
                if dep not in roots:
                    result["unpinned"].append(dep)
            else:
                result["unresolved"].append(dep)

    for key in ("roots_missing", "unpinned", "unresolved", "parse_errors", "inactive"):
        result[key] = sorted(set(result[key]))
    result["unsatisfied"] = sorted({tuple(row) for row in result["unsatisfied"]})
    result["unsatisfied"] = [list(row) for row in result["unsatisfied"]]
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
