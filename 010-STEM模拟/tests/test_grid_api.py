# -*- coding: utf-8 -*-
"""R8 回归：``build_slice_transmissions`` 的 gpts 兼容与矩形网格（2026-10-03）。

此前公开函数只接受标量网格，``gpts=(24, 40)`` 会在 ``int(tuple)`` 处抛
TypeError；STEM 探针扫描需要的矩形网格只能走私有入口。本文件钉住：

* ``gpts=None``（按 sampling 自动）、标量、``(ny, nx)`` 二元组三种形式都可用；
* 二元组的维度顺序是 **(ny, nx)**，返回的切片 shape 与之一致；
* 非法输入（长度不是 2、非正、非整数）给出清晰的 ValueError，而不是
  TypeError 或静默回退到标量语义；
* 矩形网格与私有兼容别名结果一致。

只跑网格解析与透射函数构造，不做完整 STEM 扫描（那属于 verify_physics.py）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stem_sim import Structure                                    # noqa: E402
from stem_sim.microscope import Microscope                        # noqa: E402
from stem_sim.multislice import (                                 # noqa: E402
    _build_slice_transmissions_shape,
    _grid_shape,
    build_slice_transmissions,
)


def _structure() -> Structure:
    """小盒子：x/y 周期不等，使矩形网格有实际意义。"""
    return Structure(["Si"], np.array([[2.0, 2.0, 1.0]]), np.diag([8.0, 10.0, 2.0]))


def _build(gpts, sampling=0.25):
    return build_slice_transmissions(
        _structure(), Microscope(voltage_kv=200), gpts=gpts, sampling=sampling,
        padding=0, slice_thickness=1,
    )


def test_grid_shape_none_uses_sampling():
    ny, nx = _grid_shape(8.0, 10.0, 0.25, None)
    assert (ny, nx) == (40, 32), "(ny, nx) 顺序：先 y（Ly）后 x（Lx）"


def test_grid_shape_scalar_is_square():
    assert _grid_shape(8.0, 10.0, 0.25, 32) == (32, 32)


def test_grid_shape_tuple_is_ny_nx():
    assert _grid_shape(8.0, 10.0, 0.25, (24, 40)) == (24, 40)


def test_public_build_accepts_none_scalar_and_tuple():
    """三种 gpts 形式都必须可用（回归：tuple 曾抛 TypeError）。"""
    for gpts in (None, 32, (24, 40)):
        transmissions, sampling, extent = _build(gpts)
        assert transmissions, f"gpts={gpts} 未产生切片"
        assert all(np.isfinite(t).all() for t in transmissions), gpts
        assert extent == (8.0, 10.0, 2.0), gpts
        assert all(s > 0 for s in sampling), gpts


def test_tuple_grid_produces_exact_shape():
    transmissions, sampling, _extent = _build((24, 40))
    assert [t.shape for t in transmissions] == [(24, 40)] * len(transmissions)
    # 采样按各轴周期与点数推出：sx = lx/nx, sy = ly/ny
    assert sampling == pytest.approx((8.0 / 40, 10.0 / 24))


def test_none_grid_matches_auto_sampling():
    """auto 网格由 sampling 推出，两轴采样应一致。"""
    transmissions, sampling, _extent = _build(None)
    assert [t.shape for t in transmissions] == [(40, 32)] * len(transmissions)
    assert sampling[0] == pytest.approx(sampling[1])


def test_private_alias_matches_public():
    """私有兼容别名必须与公开函数逐位一致。"""
    public = _build((24, 40))
    alias = _build_slice_transmissions_shape(
        _structure(), Microscope(voltage_kv=200), gpts=(24, 40), sampling=0.25,
        padding=0, slice_thickness=1,
    )
    assert [t.shape for t in public[0]] == [t.shape for t in alias[0]]
    for a, b in zip(public[0], alias[0]):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("bad", [(24,), (24, 40, 16), [1, 2, 3]])
def test_bad_tuple_length_rejected(bad):
    with pytest.raises(ValueError, match="二元组"):
        _grid_shape(8.0, 10.0, 0.25, bad)


@pytest.mark.parametrize("bad", [(0, 40), (24, 0), (-1, 40), (24, -5)])
def test_non_positive_grid_rejected(bad):
    with pytest.raises(ValueError, match="正整数"):
        _grid_shape(8.0, 10.0, 0.25, bad)


@pytest.mark.parametrize("bad", ["a", object(), [1.5], {"n": 4}])
def test_non_numeric_grid_rejected(bad):
    """非数值分量必须拒绝（``None`` 是合法的"自动"取值，不在此列）。"""
    with pytest.raises(ValueError):
        _grid_shape(8.0, 10.0, 0.25, bad)


def test_non_positive_scalar_rejected():
    with pytest.raises(ValueError, match="正整数"):
        _grid_shape(8.0, 10.0, 0.25, 0)


# --- 非法数值：此前 int() 会静默转换（回归 round15） ------------------------
@pytest.mark.parametrize("bad", [24.5, (24.5, 40), (24, 40.5), 0.5])
def test_fractional_grid_rejected_not_truncated(bad):
    """小数必须拒绝，不得静默截断（``int(24.5)`` 曾是 24）。"""
    with pytest.raises(ValueError, match="整数"):
        _grid_shape(8.0, 10.0, 0.25, bad)


@pytest.mark.parametrize("bad", [True, False, (True, 40), (24, False)])
def test_bool_grid_rejected(bad):
    """布尔值必须拒绝（``True`` 曾是 1）。"""
    with pytest.raises(ValueError, match="布尔"):
        _grid_shape(8.0, 10.0, 0.25, bad)


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), (-float("inf"), 40),
                                (24, float("nan"))])
def test_non_finite_grid_rejected(bad):
    with pytest.raises(ValueError):
        _grid_shape(8.0, 10.0, 0.25, bad)


@pytest.mark.parametrize("good,expected", [
    (32, (32, 32)),
    (32.0, (32, 32)),          # 整数值浮点：已声明接受
    ((24, 40), (24, 40)),
    ((24.0, 40.0), (24, 40)),
])
def test_integer_valued_forms_accepted(good, expected):
    assert _grid_shape(8.0, 10.0, 0.25, good) == expected


def test_numpy_integer_grid_accepted():
    """numpy 整数类型必须接受（GUI 与扫描参数常来自 numpy）。"""
    assert _grid_shape(8.0, 10.0, 0.25, np.int32(32)) == (32, 32)
    assert _grid_shape(8.0, 10.0, 0.25, np.int64(64)) == (64, 64)
    assert _grid_shape(8.0, 10.0, 0.25, (np.int32(24), np.int64(40))) == (24, 40)
