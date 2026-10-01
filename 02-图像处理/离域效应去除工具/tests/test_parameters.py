# -*- coding: utf-8 -*-
"""parameters 模块的单元测试。"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parameters import MAX_PIXELS, Parameters, finite_number  # noqa: E402


def test_finite_number_rejects_bool_nan_and_out_of_range():
    with pytest.raises(ValueError):          # bool 是 int 子类，必须显式排除
        finite_number(True, "参数", 0, 1)
    for bad in (float("nan"), float("inf"), "abc", None):
        with pytest.raises(ValueError):
            finite_number(bad, "参数", 0, 1)
    with pytest.raises(ValueError):
        finite_number(1.5, "参数", 0, 1)
    assert finite_number("0.5", "参数", 0, 1) == 0.5   # 数字字符串可解析


def test_parameters_validates_all_fields_and_ordering():
    for field, value in (("fmin", 0.0), ("fmax", 0.7), ("dilate", -1.0),
                         ("feather", 41.0), ("strength", 2.0), ("band_feather", -0.1)):
        with pytest.raises(ValueError):
            Parameters(**{field: value})
    with pytest.raises(ValueError):          # 下限不得 >= 上限
        Parameters(fmin=0.09, fmax=0.09)
    p = Parameters(fmin=0.05, fmax=0.10)
    assert p.fmin == 0.05 and p.fmax == 0.10


def test_parameters_roundtrip_and_defaults_merge():
    base = Parameters(strength=0.7)
    back = Parameters.from_dict(base.to_dict())
    assert back == base
    merged = Parameters.from_dict({"fmin": 0.04}, defaults=base)
    assert merged.fmin == 0.04 and merged.strength == 0.7   # 未给的字段沿用默认
    assert Parameters.from_dict({}) == Parameters()


def test_max_pixels_caps_resource_usage():
    assert MAX_PIXELS == 4096 * 4096
    import numpy as np
    import deloc_core as dc
    with pytest.raises(ValueError, match="资源上限"):
        dc._shape((4097, 4097))
    assert dc._shape((4096, 4096)) == (4096, 4096)
    with pytest.raises(ValueError):
        dc._shape((100, -1))
    with pytest.raises(ValueError):
        dc._shape((100, 100, 3))
