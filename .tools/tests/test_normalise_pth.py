# -*- coding: utf-8 -*-
"""normalise-pth.py behaviour tests (real import via site.addsitedir).

Fixtures live in a throwaway directory; no project venv or real site-packages is
touched. The import checks use a real ``site.addsitedir`` on a synthetic
site-packages so the .pth is interpreted by CPython itself, not by a string
assertion.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
import sysconfig
from pathlib import Path

import pytest

REAL_TOOLS = Path(__file__).resolve().parent.parent


def _load(name: str = "normalise_pth_under_test"):
    path = REAL_TOOLS / "normalise-pth.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def npth():
    return _load()


def _make_package(site_packages: Path, src: Path, package: str = "demo_pkg") -> Path:
    """Create a real importable package and an editable-style .pth for it."""
    src.mkdir(parents=True, exist_ok=True)
    pkg = src / package
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("VALUE = 'from-src'\n", encoding="utf-8")
    target = site_packages / f"__editable__.{package}-0.0.pth"
    return target


def _import_from(site_packages: Path, package: str):
    """Import ``package`` with ``site.addsitedir`` semantics in a clean slate."""
    import site

    for module in [m for m in sys.modules if m == package or m.startswith(package + ".")]:
        del sys.modules[module]
    saved = list(sys.path)
    try:
        site.addsitedir(str(site_packages))
        module = importlib.import_module(package)
        return Path(module.__file__)
    finally:
        sys.path[:] = saved


# ---------------------------------------------------------------------------
# bare path line -> ASCII append, with real import
# ---------------------------------------------------------------------------
# Two fixtures, deliberately split so neither depends on the *runner's* locale:
#   * CHINESE_ONLY can be encoded in GBK byte-for-byte -> deterministic even when
#     the runner's preferred encoding is GBK (encoding an emoji there fails).
#   * EMOJI is written as UTF-8 (always encodable) and still exercises non-BMP
#     escaping in the helper.
CHINESE_ONLY = "编辑项目"
EMOJI = "编辑项目😀"


def _write_non_utf8_pth(target: Path, src: Path) -> bytes:
    """Write a bare-path .pth whose bytes are GBK (never valid UTF-8)."""
    raw = str(src).encode("gbk") + b"\r\n"
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")            # premise of the whole defect
    target.write_bytes(raw)
    return raw


def test_bare_chinese_path_becomes_ascii_append(npth, tmp_path):
    """裸路径（site 语义=append）→ ASCII append；GBK 夹具确定性可编码。"""
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    src = tmp_path / CHINESE_ONLY / "src"
    target = _make_package(site_packages, src)
    _write_non_utf8_pth(target, src)

    report = npth.normalise_target_pth(target, src, site_packages=site_packages)
    assert report["call_form"] == "append", "裸路径必须保留 append 语义"
    assert report["changed"] is True

    text = target.read_text(encoding="ascii")
    assert text.startswith("import sys; sys.path.append("), text
    assert all(ord(c) < 128 for c in text)
    assert src.is_dir()

    # Real import: the .pth must actually put src on sys.path.
    imported = _import_from(site_packages, "demo_pkg")
    assert imported.parent.parent == src.resolve()


def test_bare_emoji_path_escaping_is_non_bmp_safe(npth, tmp_path):
    """含 emoji（非 BMP）的路径：用 UTF-8 写夹具，转义必须仍可被 Python 读回。

    只用 UTF-8 写夹具，因此与 runner 的 locale 无关（GBK 无法编码 emoji）。
    """
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    src = tmp_path / EMOJI / "src"
    target = _make_package(site_packages, src)
    target.write_text(f"import sys; sys.path.insert(0, {ascii(str(src))})\n",
                      encoding="utf-8")

    report = npth.normalise_target_pth(target, src, site_packages=site_packages)
    assert report["recorded_src"] == str(src.resolve()), "非 BMP 转义必须往返一致"
    text = target.read_text(encoding="ascii")
    assert all(ord(c) < 128 for c in text), "结果必须纯 ASCII"
    assert "\\U0001f600" in text, f"非 BMP 应使用 \\UXXXXXXXX 转义: {text}"

    imported = _import_from(site_packages, "demo_pkg")
    assert imported.parent.parent == src.resolve()


def test_normalisation_is_idempotent(npth, tmp_path):
    """第二次运行必须解析出同一路径并判定"无需改动"（幂等）。"""
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    src = tmp_path / "编辑项目😀" / "src"
    target = _make_package(site_packages, src)
    target.write_text(f"import sys; sys.path.insert(0, {ascii(str(src))})\n",
                      encoding="ascii")

    first = npth.normalise_target_pth(target, src, site_packages=site_packages)
    assert first["changed"] is True
    after_first = target.read_bytes()

    second = npth.normalise_target_pth(target, src, site_packages=site_packages)
    assert second["changed"] is False, "第二次不应再改动"
    assert second["recorded_src"] == str(src.resolve())
    assert target.read_bytes() == after_first


def test_existing_insert_keeps_its_index(npth, tmp_path):
    """已登记的显式 insert 必须保留其原索引语义。"""
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    src = tmp_path / "编辑😀" / "src"
    target = _make_package(site_packages, src)
    target.write_text(f"import sys; sys.path.insert(3, {ascii(str(src))})\n",
                      encoding="ascii")

    report = npth.normalise_target_pth(target, src, site_packages=site_packages)
    assert report["call_form"] == "insert"
    assert report["index"] == 3
    assert target.read_text(encoding="ascii").startswith("import sys; sys.path.insert(3, ")


def test_original_bytes_are_backed_up(npth, tmp_path):
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    src = tmp_path / "编辑项目" / "src"          # no emoji: encodable in GBK
    target = _make_package(site_packages, src)
    # Valid Python literal (backslashes doubled) whose *bytes* are GBK, i.e. not
    # decodable as UTF-8 -- exactly the setuptools defect. GBK is chosen
    # explicitly so the fixture does not depend on this machine's ANSI code page.
    literal = "'" + str(src).replace("\\", "\\\\") + "'"
    original = ("import sys; sys.path.insert(0, " + literal + ")\r\n").encode("gbk")
    with pytest.raises(UnicodeDecodeError):
        original.decode("utf-8")               # the premise of the whole helper
    target.write_bytes(original)

    report = npth.normalise_target_pth(target, src, site_packages=site_packages)
    backup = Path(report["backup"])
    assert backup.is_file()
    assert backup.read_bytes() == original, "备份必须与原 bytes 完全一致"
    assert target.read_text(encoding="ascii").startswith("import sys; sys.path.insert(0, ")


# ---------------------------------------------------------------------------
# non-target files untouched; unknown content rejected
# ---------------------------------------------------------------------------
def test_non_target_and_corrupt_files_untouched(npth, tmp_path):
    """非目标 .pth（含损坏字节）必须完全不写、保持原 bytes。"""
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    src = tmp_path / "编辑😀" / "src"
    target = _make_package(site_packages, src)
    target.write_text(f"import sys; sys.path.append({ascii(str(src))})\n", encoding="ascii")

    third = site_packages / "third-party.pth"
    third.write_bytes(b"import sys; sys.path.insert(0, 'shared\\u4e2d\\u6587')\n")
    corrupt = site_packages / "unknown-corrupt.pth"
    corrupt.write_bytes(b"\xff\xfe\x00\x01\n")

    before = {p.name: p.read_bytes() for p in site_packages.glob("*.pth")}
    npth.normalise_target_pth(target, src, site_packages=site_packages)
    after = {p.name: p.read_bytes() for p in site_packages.glob("*.pth")}

    for name in ("third-party.pth", "unknown-corrupt.pth"):
        assert before[name] == after[name], f"{name} 被改写了"
    assert not (site_packages / "third-party.pth.orig").exists()
    assert not (site_packages / "unknown-corrupt.pth.orig").exists()


def test_wrong_expected_src_is_rejected(npth, tmp_path):
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    src = tmp_path / "编辑😀" / "src"
    target = _make_package(site_packages, src)
    target.write_text(f"import sys; sys.path.append({ascii(str(src))})\n", encoding="ascii")

    with pytest.raises(npth.PthNormaliseError, match="expected"):
        npth.normalise_target_pth(target, tmp_path / "other" / "src",
                                  site_packages=site_packages)
    assert target.read_text(encoding="ascii").startswith("import sys; sys.path.append(")


@pytest.mark.parametrize("payload,label", [
    ("import os; os.environ['X'] = '1'\n", "unrelated-import"),
    ("import sys; sys.path.insert(0, p)\n", "non-literal-arg"),
    ("import sys; sys.path.extend(['a', 'b'])\n", "unsupported-call"),
    ("import sys\nimport sys.path\n", "two-statements"),
    ("import sys; sys.path.append('a'); sys.path.append('b')\n", "two-calls"),
    ("import sys; sys.path.append('a')\nimport sys; sys.path.append('b')\n", "two-lines"),
    ("\xff\xfe\x80\n", "undecodable-bytes"),
])
def test_unrecognised_directives_are_rejected(npth, tmp_path, payload, label):
    """未知/异常 directive 必须明确失败，绝不盲猜编码或改写。"""
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    src = tmp_path / "编辑😀" / "src"
    target = _make_package(site_packages, src)
    if label == "undecodable-bytes":
        target.write_bytes(payload.encode("latin-1"))
    else:
        target.write_text(payload, encoding="utf-8")
    before = target.read_bytes()

    with pytest.raises(npth.PthNormaliseError):
        npth.normalise_target_pth(target, src, site_packages=site_packages)
    assert target.read_bytes() == before, "拒绝时不得改写"
    assert not (site_packages / (target.name + ".orig")).exists()


def test_relative_path_resolves_against_site_packages(npth, tmp_path, monkeypatch):
    """相对裸路径必须相对 site-packages 解释，而不是进程 cwd。"""
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    (site_packages / "rel_src").mkdir()
    target = site_packages / "__editable__.demo_pkg-0.0.pth"
    target.write_text("rel_src\n", encoding="ascii")

    monkeypatch.chdir(tmp_path.parent)          # cwd is NOT site-packages
    report = npth.normalise_target_pth(target, site_packages / "rel_src",
                                       site_packages=site_packages)
    assert report["recorded_src"] == str((site_packages / "rel_src").resolve())


def test_find_editable_pth_refuses_ambiguity(npth, tmp_path):
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    (site_packages / "__editable__.demo-1.pth").write_text("a\n", encoding="ascii")
    (site_packages / "__editable__.demo-2.pth").write_text("b\n", encoding="ascii")
    with pytest.raises(npth.PthNormaliseError, match="multiple"):
        npth.find_editable_pth(site_packages, "demo")


# ---------------------------------------------------------------------------
# the fixture itself: does a non-ASCII .pth really break site.addsitedir?
# ---------------------------------------------------------------------------
def test_non_ascii_pth_really_breaks_ansi_vs_utf8(tmp_path):
    """证明该 helper 要解决的问题真实存在（否则上面的修复无意义）。

    用**可 GBK 编码**的中文夹具，因此不依赖 runner 的 preferred encoding：
    同一份字节在 UTF-8 模式下必然读不出来；ASCII 化后两种模式都能读。
    """
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    src = tmp_path / CHINESE_ONLY / "src"
    _make_package(site_packages, src)
    target = site_packages / "__editable__.demo_pkg-0.0.pth"
    raw = str(src).encode("gbk") + b"\n"
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")            # UTF-8 mode cannot read this file
    target.write_bytes(raw)

    # site.py's own reader (utf-8 under -X utf8) fails on the raw bytes...
    import io
    with pytest.raises(UnicodeDecodeError):
        io.open(target, encoding="utf-8").read()

    # ...and after the helper's normalisation both decoding modes agree.
    module = _load("npth_breakage_probe")
    module.normalise_target_pth(target, src, site_packages=site_packages)
    text = target.read_bytes()
    assert text.decode("ascii", errors="strict")
    assert io.open(target, encoding="utf-8").read().strip().startswith("import sys")
