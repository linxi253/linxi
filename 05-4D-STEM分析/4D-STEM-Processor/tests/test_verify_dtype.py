# -*- coding: utf-8 -*-
"""verify_dtype diagnostic regressions.

The DM4 *tag tree* is mocked (ncempy needs a real DM4 object), but every byte
read is real: each test writes an actual binary datacube to a temp file and the
production code memmaps it. That is what makes these tests able to catch the
real defects they were written for:

  * the datacube was decoded with a hardcoded ``<i2`` regardless of the metadata
    dtype -- float32 data was silently mis-read, uint8 could raise;
  * ``BASE`` was undefined, so ``os.environ.get('STEM4D_OUTPUT', os.path.join(BASE, ...))``
    raised NameError even *with* the variable set, because the default argument
    is always evaluated;
  * detector indexing assumed 32x32 (``14:18``, ``[16, 16]``), so a small or
    rectangular detector overflowed.

No real experiment data is read.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from core import dm4_io  # noqa: E402


def _load_verify_dtype():
    path = PROJ / "diagnostics" / "verify_dtype.py"
    spec = importlib.util.spec_from_file_location("verify_dtype_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_dtype_under_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def vd():
    return _load_verify_dtype()


@pytest.fixture(autouse=True)
def _no_machine_defaults(request, monkeypatch):
    """Neutralise the main checkout's machine defaults for every test.

    The staged main-bound copy carries ``_MAIN_BASE_DEFAULT`` /
    ``_MAIN_DM4_DEFAULT`` (the main checkout's machine default path) so the main working tree keeps
    working without environment variables. Tests must never touch that path or
    write to its default output directory, so both are blanked here -- with
    ``raising=False`` because the public candidate does not define them at all.
    The defaults themselves are verified separately by the pure-path tests.
    """
    if "vd" not in request.fixturenames:
        return
    module = request.getfixturevalue("vd")
    monkeypatch.setattr(module, "_MAIN_BASE_DEFAULT", "", raising=False)
    monkeypatch.setattr(module, "_MAIN_DM4_DEFAULT", "", raising=False)


def _write_cube(path: Path, values: np.ndarray, dtype) -> dict:
    """Write a real binary datacube in DM4 order (det_y, det_x, scan_y, scan_x)."""
    scan_y, scan_x, det_y, det_x = values.shape
    cube = np.ascontiguousarray(
        values.transpose(2, 3, 0, 1).astype(dtype))       # (det_y, det_x, sy, sx)
    header = b"\x00" * 64
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(cube.tobytes())
    return {
        "offset": len(header),
        "dtype": np.dtype(dtype),
        "dtype_name": dm4_io.dtype_display_name(np.dtype(dtype)),
        "det_y": det_y, "det_x": det_x,
        "scan_y": scan_y, "scan_x": scan_x,
        "4d_index": 0,
    }


def _make_values(scan_y: int, scan_x: int, det_y: int, det_x: int, scale=1.0):
    """Distinct, exactly-representable values so mis-decoding is visible."""
    base = np.arange(scan_y * scan_x * det_y * det_x,
                     dtype=np.float64).reshape(scan_y, scan_x, det_y, det_x)
    return (base * scale).astype(np.float64)


@pytest.mark.parametrize("dtype,scale,tolerance", [
    (np.float32, 1.0, 1e-6),
    (np.int16, 1.0, 0.0),
    (np.uint8, 1.0, 0.0),
    (np.uint16, 1.0, 0.0),
])
def test_real_datacube_uses_the_metadata_dtype(vd, tmp_path, monkeypatch,
                                               dtype, scale, tolerance):
    """Values, shape and axis order must match the metadata dtype exactly."""
    values = _make_values(3, 4, 6, 5, scale)
    if dtype is np.uint8:
        values = values % 200                      # keep it in range
    path = tmp_path / "cube.dm4"
    meta = _write_cube(path, values, dtype)
    monkeypatch.setattr(dm4_io, "read_dm4_metadata", lambda p: meta)

    data = dm4_io.extract_4d_data(path, meta, scan_crop=min(4, 3, 4))
    # extract_4d_data takes one square centre crop for both scan axes.
    crop = min(4, 3, 4)
    sy0, sx0 = (3 - crop) // 2, (4 - crop) // 2
    expected = values.astype(dtype).astype(np.float32)[sy0:sy0 + crop, sx0:sx0 + crop]
    assert data.shape == (crop, crop, 6, 5)
    assert data.dtype == np.float32
    np.testing.assert_allclose(data, expected, rtol=0, atol=tolerance)


def test_float32_is_not_silently_misread_as_int16(vd, tmp_path, monkeypatch):
    """Regression: hardcoded '<i2' corrupted float32 data.

    A float32 value of 1.5 reinterpreted as int16 is meaningless; the decoded
    datacube must preserve 1.5.
    """
    values = np.zeros((2, 2, 4, 4), dtype=np.float64)
    values[0, 0, 0, 0] = 1.5
    values[1, 1, 3, 3] = 2.25
    path = tmp_path / "float.dm4"
    meta = _write_cube(path, values, np.float32)
    monkeypatch.setattr(dm4_io, "read_dm4_metadata", lambda p: meta)

    data = dm4_io.extract_4d_data(path, meta, scan_crop=2)
    assert data[0, 0, 0, 0] == pytest.approx(1.5)
    assert data[1, 1, 3, 3] == pytest.approx(2.25)


def test_short_data_raises_a_clear_error(vd, tmp_path, monkeypatch):
    """A truncated file must fail loudly, not decode garbage."""
    values = _make_values(2, 2, 4, 4)
    path = tmp_path / "short.dm4"
    meta = _write_cube(path, values, np.float32)
    with path.open("rb") as handle:
        blob = handle.read()
    path.write_bytes(blob[: len(blob) // 2])        # truncate the data block

    # read_dm4_metadata performs the size validation in production; here the
    # metadata is mocked, so the real check under test is the memmap read itself.
    with pytest.raises(Exception):
        dm4_io.extract_4d_data(path, meta, scan_crop=2)


def test_complex_dtype_is_refused_not_silently_realised(vd, monkeypatch, capsys):
    """Complex data must be rejected explicitly."""
    with pytest.raises(SystemExit) as excinfo:
        vd._reject_unsupported_complex(np.dtype(np.complex64), "complex64")
    assert "复数" in str(excinfo.value)
    assert "实部" in str(excinfo.value)


def test_real_dtype_passes_the_complex_guard(vd):
    for dtype in (np.float32, np.int16, np.uint8, np.uint16, np.float64):
        vd._reject_unsupported_complex(np.dtype(dtype), "x")   # must not raise


def test_safe_window_never_exceeds_small_detectors(vd):
    for size in (1, 2, 3, 4, 8, 16, 32, 512):
        win = vd._safe_window(size)
        assert 1 <= win <= size, (size, win)


@pytest.mark.parametrize("det", [(6, 10), (3, 3), (1, 1), (5, 40)])
def test_rectangular_and_tiny_detectors_do_not_overflow(vd, tmp_path, monkeypatch,
                                                        det, capsys):
    """Regression: 14:18 / [16,16] indexing assumed a 32x32 detector."""
    det_y, det_x = det
    values = _make_values(3, 3, det_y, det_x)
    path = tmp_path / f"det{det_y}x{det_x}.dm4"
    meta = _write_cube(path, values, np.float32)
    monkeypatch.setattr(dm4_io, "read_dm4_metadata", lambda p: meta)
    monkeypatch.setenv("STEM4D_DM4", str(path))
    out = tmp_path / "out"
    monkeypatch.setenv("STEM4D_OUTPUT", str(out))

    vd.main()                                     # must not raise IndexError
    printed = capsys.readouterr().out
    assert "Detector  : %d x %d" % (det_y, det_x) in printed
    assert (out / "dtype_verification.png").is_file()


def test_titles_report_the_real_dtype_not_int16(vd, tmp_path, monkeypatch, capsys):
    values = _make_values(2, 2, 8, 8)
    path = tmp_path / "t.dm4"
    meta = _write_cube(path, values, np.float32)
    monkeypatch.setattr(dm4_io, "read_dm4_metadata", lambda p: meta)
    monkeypatch.setenv("STEM4D_DM4", str(path))
    monkeypatch.setenv("STEM4D_OUTPUT", str(tmp_path / "out"))

    vd.main()
    printed = capsys.readouterr().out
    assert "float32" in printed
    assert "Metadata  : dtype code -> float32" in printed
    # the old, wrong claim must be gone from the summary line
    assert "Data Type Verification: <i2 (LE int16)" not in printed


@pytest.mark.parametrize("dtype,shape", [
    ("<f4", (3, 5, 5, 7)),
    ("<i2", (3, 3, 3, 3)),      # 162 bytes: not a multiple of 4
    ("u1", (3, 3, 3, 3)),       # 81 bytes: not a multiple of 2 or 4
    ("u1", (1, 1, 1, 1)),       # 1 byte: not a multiple of 2 or 4
])
def test_actual_main_preserves_values_and_exports(vd, dtype, shape, tmp_path,
                                                  monkeypatch):
    """Drive the REAL ``main()`` end to end; only metadata is stubbed.

    Ported from the reviewer's acceptance script. The datacube is read through
    the production path (spied, never mocked), so this is the test that the old
    hardcoded-``<i2`` code could not pass. Odd-sized data blocks also exercise
    the diagnostic prefix handling (162 / 81 / 1 bytes).
    """
    sy, sx, dy, dx = shape
    values = (np.arange(int(np.prod(shape))).reshape(shape) % 121 + 1).astype(dtype)
    if dtype == "<f4":
        values = values / 4
    src = tmp_path / "synthetic.dm4"
    src.write_bytes(b"\0" * 64 + values.transpose(2, 3, 0, 1).copy().tobytes())
    meta = {"offset": 64, "dtype": np.dtype(dtype),
            "dtype_name": str(np.dtype(dtype)),
            "det_y": dy, "det_x": dx, "scan_y": sy, "scan_x": sx}
    monkeypatch.setattr(dm4_io, "read_dm4_metadata", lambda _: meta)

    actual = []
    reader = dm4_io.extract_4d_data

    def spy(*args, **kwargs):
        data = reader(*args, **kwargs)
        actual.append(data.copy())
        return data

    monkeypatch.setattr(dm4_io, "extract_4d_data", spy)
    monkeypatch.setenv("STEM4D_DM4", str(src))
    monkeypatch.setenv("STEM4D_OUTPUT", str(tmp_path / "export"))
    monkeypatch.delenv("STEM4D_DATA", raising=False)

    vd.main()

    n = min(4, sy, sx)
    y, x = (sy - n) // 2, (sx - n) // 2
    assert len(actual) == 1, "the datacube must be read exactly once, via the real reader"
    np.testing.assert_array_equal(actual[0], values[y:y + n, x:x + n].astype(np.float32))
    png = tmp_path / "export" / "dtype_verification.png"
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    # the source file is untouched
    assert src.read_bytes() == b"\0" * 64 + values.transpose(2, 3, 0, 1).copy().tobytes()


@pytest.mark.parametrize("n_bytes", [1, 3, 5, 7, 81, 162])
def test_diagnostic_handles_blocks_that_are_not_itemsize_multiples(vd, n_bytes,
                                                                   capsys):
    """Regression: np.frombuffer raised ValueError on odd-length blocks.

    Each interpretation may consume only a whole number of its own itemsize, and
    a too-short block is skipped with an explanation.
    """
    vd._diagnostic_interpretations(bytes(range(n_bytes)))
    out = capsys.readouterr().out
    assert "Testing data type interpretations" in out
    for name in ("<i2 (LE int16)", "<f4 (LE float32)", ">u2 (BE uint16)"):
        assert name in out
    if n_bytes < 2:
        assert "skipped" in out
    assert "Traceback" not in out


def test_diagnostic_does_not_change_the_real_read_length(vd, tmp_path, monkeypatch):
    """The diagnostic prefix must not shorten the authoritative datacube."""
    shape = (3, 3, 3, 3)                     # 162 bytes, not 4-aligned
    values = (np.arange(int(np.prod(shape))).reshape(shape) % 121 + 1).astype("<i2")
    src = tmp_path / "odd.dm4"
    src.write_bytes(b"\0" * 64 + values.transpose(2, 3, 0, 1).copy().tobytes())
    meta = {"offset": 64, "dtype": np.dtype("<i2"), "dtype_name": "int16",
            "det_y": 3, "det_x": 3, "scan_y": 3, "scan_x": 3}
    monkeypatch.setattr(dm4_io, "read_dm4_metadata", lambda _: meta)
    monkeypatch.setenv("STEM4D_DM4", str(src))
    monkeypatch.setenv("STEM4D_OUTPUT", str(tmp_path / "out"))

    vd.main()
    produced = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert produced == ["dtype_verification.png"]


def test_missing_output_config_is_reported_before_building_a_figure(vd, monkeypatch,
                                                                   tmp_path):
    """A config error must not leave a matplotlib figure behind."""
    shape = (2, 2, 4, 4)
    values = _make_values(*shape)
    src = tmp_path / "f.dm4"
    meta = _write_cube(src, values, np.float32)
    monkeypatch.setattr(dm4_io, "read_dm4_metadata", lambda p: meta)
    monkeypatch.setenv("STEM4D_DM4", str(src))
    monkeypatch.delenv("STEM4D_OUTPUT", raising=False)
    monkeypatch.delenv("STEM4D_DATA", raising=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())
    with pytest.raises(SystemExit):
        vd.main()
    assert set(plt.get_fignums()) == before, "a failed run leaked a figure"


def test_output_dir_falls_back_to_stem4d_data(vd, tmp_path, monkeypatch):
    monkeypatch.delenv("STEM4D_OUTPUT", raising=False)
    base = tmp_path / "dataroot"
    monkeypatch.setenv("STEM4D_DATA", str(base))
    assert vd._output_dir() == str(base / "analysis" / "results" / "v2")


def test_output_dir_is_needed_even_with_output_set(vd, tmp_path, monkeypatch):
    """Regression: the old default expression referenced BASE and always evaluated.

    With STEM4D_OUTPUT set, resolution must succeed without any BASE-ish name.
    (Behavioural check -- asserting on the source text would also forbid the
    main checkout's legitimate ``_MAIN_BASE_DEFAULT`` fallback.)
    """
    monkeypatch.setenv("STEM4D_OUTPUT", str(tmp_path / "ok"))
    monkeypatch.delenv("STEM4D_DATA", raising=False)
    assert vd._output_dir() == str(tmp_path / "ok")


def test_missing_output_config_gives_an_actionable_error(vd, monkeypatch):
    monkeypatch.delenv("STEM4D_OUTPUT", raising=False)
    monkeypatch.delenv("STEM4D_DATA", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        vd._output_dir()
    message = str(excinfo.value)
    assert "STEM4D_OUTPUT" in message and "STEM4D_DATA" in message
    assert "本机默认路径" in message


def test_missing_dm4_env_gives_usage(vd, monkeypatch):
    monkeypatch.delenv("STEM4D_DM4", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        vd.main()
    assert "STEM4D_DM4" in str(excinfo.value)


def test_diagnostic_section_still_compares_interpretations(vd, tmp_path, monkeypatch,
                                                           capsys):
    """The multi-interpretation comparison must be kept, labelled as diagnostic."""
    values = _make_values(2, 2, 4, 4)
    path = tmp_path / "d.dm4"
    meta = _write_cube(path, values, np.float32)
    monkeypatch.setattr(dm4_io, "read_dm4_metadata", lambda p: meta)
    monkeypatch.setenv("STEM4D_DM4", str(path))
    monkeypatch.setenv("STEM4D_OUTPUT", str(tmp_path / "out"))

    vd.main()
    printed = capsys.readouterr().out
    assert "Testing data type interpretations (diagnostic only)" in printed
    for name in ("<i2 (LE int16)", "<f4 (LE float32)", ">u2 (BE uint16)"):
        assert name in printed
