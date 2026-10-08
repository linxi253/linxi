# -*- coding: utf-8 -*-
"""Normalise ONE target editable ``.pth`` file to a safe ASCII form.

Why this exists (S1b/S1c, real defect):
    CPython's ``site.py`` reads every ``.pth`` with ``encoding="locale"``, while
    setuptools writes an editable install's ``__editable__.<pkg>.pth`` in the
    ANSI code page. The two agree only while both sides use the same encoding.
    Our tooling starts pip with ``-X utf8`` (UTF-8 mode), where "locale" means
    UTF-8, so a GBK-written ``.pth`` fails to decode and every interpreter start
    dies with ``init_import_site`` / ``UnicodeDecodeError``. Project paths here
    contain Chinese characters, so no single encoding works for both modes; the
    file must become pure ASCII.

Scope and safety:
    * Only the **exact target file** is ever written. Non-target ``.pth`` files
      are never touched (reported only).
    * The target must parse as exactly one recognised statement pointing at the
      **expected source root**. Anything else is a hard error -- we never guess
      encodings and never "repair" a file we do not understand.
    * Parsing uses :mod:`ast` + :func:`ast.literal_eval` (no ``exec``, no string
      partitioning), so already-normalised files parse back to the *same* path
      and the operation is idempotent.
    * The path is written with the builtin :func:`ascii` representation, which is
      valid for every code point (BMP, non-BMP/emoji, control bytes).
    * Call-form semantics are preserved: a legacy bare path line means *append*
      (that is what site.py does with a plain path line), so it becomes
      ``sys.path.append(...)``. An existing explicit ``insert`` keeps ``insert``
      with its original index.
    * The original bytes are backed up next to the file before any write.

Importable as a module (used by provision-envs.py) and runnable directly.

Usage::

    normalise-pth.py --site-packages <dir> --expected-src <abs src> --package <name>
    normalise-pth.py --site-packages <dir> --expected-src <src> --target <file>
"""
from __future__ import annotations

import argparse
import ast
import locale
import sys
from pathlib import Path


class PthNormaliseError(RuntimeError):
    """Raised when the target file is missing, ambiguous, or not understood."""


# ---------------------------------------------------------------------------
# parsing (strict AST; no exec, no string surgery)
# ---------------------------------------------------------------------------
def _parse_pth(text: str, target: Path) -> tuple[str, str, int | None]:
    """Return ``(call, path_text, index)`` for the single statement in ``text``.

    ``call`` is ``"append"`` (also used for a legacy bare path line, which
    site.py treats as an append) or ``"insert"`` (with its original index).

    site.py's own rule is followed: a line starting with ``import`` is executed
    as Python, any other non-comment line is a literal path. Raises
    :class:`PthNormaliseError` for anything we do not fully understand.
    """
    lines = [line for line in text.splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise PthNormaliseError(f"{target} has no effective line")
    if len(lines) != 1:
        raise PthNormaliseError(
            f"{target} must contain exactly one effective line, found {len(lines)}; "
            "refusing to rewrite an ambiguous file")
    line = lines[0].strip()

    if not line.startswith(("import ", "import\t")):
        # Legacy bare path line: site.py appends it verbatim (no Python parsing,
        # which is why a Windows path with backslashes is legal here).
        return ("append", line, None)

    try:
        tree = ast.parse(line, filename=str(target), mode="exec")
    except SyntaxError as exc:
        raise PthNormaliseError(
            f"{target} is not parseable as Python ({exc.msg}); refusing to rewrite") from exc

    # setuptools emits ``import sys; sys.path.insert(0, '<path>')``: an Import
    # plus exactly one expression. Accept that shape only.
    body = tree.body
    if len(body) == 2:
        first, statement = body
        if not (isinstance(first, ast.Import) and len(first.names) == 1
                and first.names[0].name == "sys" and first.names[0].asname is None):
            raise PthNormaliseError(
                f"{target} has an unexpected leading statement; refusing to rewrite")
    elif len(body) == 1:
        statement = body[0]
    else:
        raise PthNormaliseError(
            f"{target} must contain exactly one directive, found {len(body)} statements; "
            "refusing to rewrite an ambiguous file")

    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        raise PthNormaliseError(
            f"{target} contains an unsupported statement ({type(statement).__name__}); "
            "refusing to rewrite")
    call = statement.value
    func = call.func
    if not (isinstance(func, ast.Attribute)
            and func.attr in ("append", "insert")
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "path"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "sys"
            and isinstance(call.func.ctx, ast.Load)):
        raise PthNormaliseError(
            f"{target} calls something other than sys.path.append/insert; refusing to rewrite")
    if call.keywords:
        raise PthNormaliseError(f"{target} uses keyword arguments; refusing to rewrite")

    if func.attr == "insert":
        if len(call.args) != 2:
            raise PthNormaliseError(
                f"{target}: sys.path.insert needs (index, path); refusing to rewrite")
        index_arg, path_arg = call.args
        try:
            index = ast.literal_eval(index_arg)
        except (ValueError, SyntaxError) as exc:
            raise PthNormaliseError(
                f"{target}: insert index is not a literal; refusing to rewrite") from exc
        if not isinstance(index, int) or isinstance(index, bool):
            raise PthNormaliseError(
                f"{target}: insert index must be an int literal; refusing to rewrite")
    else:
        if len(call.args) != 1:
            raise PthNormaliseError(
                f"{target}: sys.path.append needs exactly one path; refusing to rewrite")
        index, path_arg = None, call.args[0]

    try:
        path_text = ast.literal_eval(path_arg)
    except (ValueError, SyntaxError) as exc:
        raise PthNormaliseError(
            f"{target}: path argument is not a string literal; refusing to rewrite") from exc
    if not isinstance(path_text, str):
        raise PthNormaliseError(
            f"{target}: path argument is not a string; refusing to rewrite")
    return (func.attr, path_text, index)


def _literal(value: str) -> str:
    """Canonical ASCII Python string literal for ``value``.

    The builtin :func:`ascii` already escapes every non-ASCII character
    (``\\uXXXX`` for BMP, ``\\UXXXXXXXX`` for non-BMP) and picks a valid quote
    character, so we return it unchanged rather than re-quoting by hand.
    """
    return ascii(value)


def _render(call: str, path: str, index: int | None) -> str:
    literal = _literal(path)
    if call == "append":
        # Includes the legacy bare-path form: site.py semantics are append.
        return f"import sys; sys.path.append({literal})"
    if call == "insert":
        return f"import sys; sys.path.insert({index}, {literal})"
    raise PthNormaliseError(f"unsupported call form: {call!r}")


# ---------------------------------------------------------------------------
# target resolution
# ---------------------------------------------------------------------------
def _resolve_recorded(path_text: str, site_packages: Path) -> Path:
    """Resolve the path recorded in the .pth.

    An absolute path is used as-is. A relative bare path is interpreted
    **relative to site-packages** (which is what a bare ``.pth`` line means) --
    never relative to the process working directory.
    """
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(site_packages) / candidate).resolve()


def normalise_target_pth(target: Path, expected_src: Path, *,
                         site_packages: Path | None = None,
                         dry_run: bool = False, log=None) -> dict:
    """Normalise ``target`` to ASCII, verifying it points at ``expected_src``."""
    target = Path(target)
    expected_src = Path(expected_src).resolve()
    site_packages = Path(site_packages) if site_packages else target.parent
    if not target.is_file():
        raise PthNormaliseError(f"target .pth does not exist: {target}")

    raw = target.read_bytes()
    try:
        text = raw.decode("ascii")
        decoded_as = "ascii"
    except UnicodeDecodeError:
        text = None
        decoded_as = None
        # Only the encodings we can justify: UTF-8 (what our tooling uses when it
        # writes) and the ANSI code page / GBK (what setuptools and this machine's
        # console actually use for Chinese paths). Never latin-1/cp1252: those
        # decode *any* byte sequence and would silently accept corrupt files.
        for encoding in ("utf-8", locale.getpreferredencoding(False) or "cp1252", "gbk"):
            try:
                text = raw.decode(encoding)
                decoded_as = encoding
                break
            except (UnicodeDecodeError, LookupError):
                continue
        if text is None:
            raise PthNormaliseError(
                f"cannot decode {target} as ascii/utf-8/ansi/gbk; refusing to guess "
                f"an encoding (first bytes: {raw[:16]!r})")

    call, path_text, index = _parse_pth(text, target)
    recorded = _resolve_recorded(path_text, site_packages)
    if recorded != expected_src:
        raise PthNormaliseError(
            f"{target} points at {recorded}, expected {expected_src}; refusing to rewrite")

    new_text = _render(call, str(expected_src), index) + "\n"
    try:
        new_text.encode("ascii")
    except UnicodeEncodeError as exc:              # pragma: no cover - defensive
        raise PthNormaliseError(f"rendered line is not ASCII: {exc}") from exc

    report = {"file": str(target), "decoded_as": decoded_as, "call_form": call,
              "index": index, "expected_src": str(expected_src),
              "recorded_src": str(recorded), "changed": False, "backup": None,
              "content": new_text.strip()[:200]}
    if new_text == text:
        return report                                  # already normalised
    if dry_run:
        report["changed"] = True
        report["dry_run"] = True
        return report

    backup = target.with_name(target.name + ".orig")
    if not backup.exists():
        backup.write_bytes(raw)                        # original bytes preserved
    report["backup"] = str(backup)
    target.write_text(new_text, encoding="ascii", newline="")
    report["changed"] = True
    if log:
        log(report)
    return report


def inspect_site_packages(site_packages: Path) -> list[dict]:
    """Report every ``.pth`` in ``site_packages`` without modifying any."""
    site_packages = Path(site_packages)
    if not site_packages.is_dir():
        return []
    rows = []
    for pth in sorted(site_packages.glob("*.pth")):
        raw = pth.read_bytes()
        try:
            raw.decode("ascii")
            ascii_ok = True
        except UnicodeDecodeError:
            ascii_ok = False
        rows.append({"file": str(pth), "bytes": len(raw), "ascii": ascii_ok})
    return rows


def site_packages_of(python_exe: Path) -> Path | None:
    python_exe = Path(python_exe)
    candidate = python_exe.parent.parent / "Lib" / "site-packages"
    return candidate if candidate.is_dir() else None


def find_editable_pth(site_packages: Path, package_name: str) -> Path:
    """Locate ``__editable__.<package_name>-*.pth`` (exactly one expected)."""
    matches = sorted(Path(site_packages).glob(f"__editable__.{package_name}-*.pth"))
    if not matches:
        raise PthNormaliseError(f"no __editable__.{package_name}-*.pth in {site_packages}")
    if len(matches) > 1:
        raise PthNormaliseError(
            f"multiple editable .pth for {package_name}: {[m.name for m in matches]}; "
            "refusing to guess")
    return matches[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Normalise one editable .pth to ASCII")
    parser.add_argument("--site-packages", required=True, type=Path)
    parser.add_argument("--expected-src", required=True, type=Path)
    parser.add_argument("--package", default=None)
    parser.add_argument("--target", default=None, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.target is not None:
        target = args.target
    elif args.package:
        target = find_editable_pth(args.site_packages, args.package)
    else:
        parser.error("one of --target or --package is required")

    before = inspect_site_packages(args.site_packages)
    report = normalise_target_pth(target, args.expected_src,
                                  site_packages=args.site_packages,
                                  dry_run=args.dry_run,
                                  log=lambda r: print(f"  [fixed] {r['file']} ({r['decoded_as']})"))
    after = {row["file"]: row for row in inspect_site_packages(args.site_packages)}
    untouched = sum(1 for row in before
                    if row["file"] != str(target)
                    and after[row["file"]]["bytes"] == row["bytes"]
                    and after[row["file"]]["ascii"] == row["ascii"])
    print(f"[ok  ] target={target}")
    print(f"[info] changed={report['changed']} call={report['call_form']} "
          f"backup={report['backup']}")
    print(f"[info] non-target .pth untouched: {untouched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
