# -*- coding: utf-8 -*-
"""roi_model 的单元测试。"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roi_model import RoiSet, RoiShape  # noqa: E402


# --------------------------------------------------------------------------
# RoiShape
# --------------------------------------------------------------------------
def test_shape_rejects_unknown_kind():
    with pytest.raises(ValueError):
        RoiShape(kind="triangle", points=[(0, 0), (1, 0), (1, 1)])


def test_shape_points_are_coerced_to_float():
    s = RoiShape(kind="polygon", points=[(0, 1), (2, 3), (4, 5)])
    assert all(isinstance(v, float) for p in s.points for v in p)


def test_polygon_area_of_known_square():
    s = RoiShape(kind="polygon", points=[(0, 0), (10, 0), (10, 10), (0, 10)])
    assert s.area() == pytest.approx(100.0)


def test_polygon_area_is_orientation_independent():
    cw = RoiShape(kind="polygon", points=[(0, 0), (0, 10), (10, 10), (10, 0)])
    ccw = RoiShape(kind="polygon", points=[(0, 0), (10, 0), (10, 10), (0, 10)])
    assert cw.area() == pytest.approx(ccw.area())


def test_rect_area():
    s = RoiShape(kind="rect", points=[(5, 5), (25, 15)])
    assert s.area() == pytest.approx(20 * 10)


def test_ellipse_area_is_pi_ab():
    s = RoiShape(kind="ellipse", points=[(0, 0), (20, 10)])
    assert s.area() == pytest.approx(np.pi * 10 * 5)


def test_two_point_rect_is_valid_but_degenerate_polygon_is_not():
    assert RoiShape(kind="rect", points=[(0, 0), (10, 10)]).is_valid()
    assert not RoiShape(kind="polygon", points=[(0, 0), (10, 10)]).is_valid()
    assert not RoiShape(kind="polygon", points=[(0, 0), (1, 0), (1, 1)]).is_valid()


def test_shape_json_round_trip():
    s = RoiShape(kind="polygon", points=[(1.5, 2.5), (3, 4), (5, 6)], subtract=True)
    back = RoiShape.from_dict(s.to_dict())
    assert back.kind == s.kind
    assert back.subtract is True
    assert back.points == pytest.approx(s.points)


# --------------------------------------------------------------------------
# RoiSet 编辑
# --------------------------------------------------------------------------
def _square(x0=20, y0=20, x1=80, y1=80, subtract=False):
    return RoiShape(kind="rect", points=[(x0, y0), (x1, y1)], subtract=subtract)


def test_add_rejects_degenerate_shape():
    roi = RoiSet()
    assert roi.add(RoiShape(kind="polygon", points=[(0, 0), (1, 1)])) is False
    assert len(roi) == 0


def test_add_pop_clear():
    roi = RoiSet()
    assert roi.add(_square())
    assert roi.add(_square(30, 30, 50, 50, subtract=True))
    assert len(roi) == 2
    assert roi.pop()
    assert len(roi) == 1
    roi.clear()
    assert len(roi) == 0


def test_undo_redo():
    roi = RoiSet()
    roi.add(_square())
    roi.add(_square(0, 0, 10, 10))
    assert len(roi) == 2
    assert roi.undo() and len(roi) == 1
    assert roi.undo() and len(roi) == 0
    assert not roi.undo()
    assert roi.redo() and len(roi) == 1
    assert roi.redo() and len(roi) == 2
    assert not roi.redo()


def test_new_edit_clears_redo():
    roi = RoiSet()
    roi.add(_square())
    roi.undo()
    assert roi.can_redo
    roi.add(_square(0, 0, 30, 30))
    assert not roi.can_redo


def test_can_undo_can_redo_flags():
    roi = RoiSet()
    assert not roi.can_undo and not roi.can_redo
    roi.add(_square())
    assert roi.can_undo and not roi.can_redo


def test_summary_counts_both_kinds():
    roi = RoiSet()
    roi.add(_square())
    roi.add(_square(30, 30, 50, 50, subtract=True))
    assert "1" in roi.summary() and "1" in roi.summary()


# --------------------------------------------------------------------------
# 栅格化
# --------------------------------------------------------------------------
def test_empty_roi_gives_all_zero_mask():
    roi = RoiSet()
    m = roi.rasterize((64, 64))
    assert m.shape == (64, 64)
    assert m.max() == 0.0


def test_rect_mask_inside_outside():
    roi = RoiSet()
    roi.add(_square(20, 20, 80, 80))
    m = roi.rasterize((100, 100), dilate=0.0, feather=1.0)
    assert m.dtype == np.float32
    assert m.min() >= 0.0 and m.max() <= 1.0
    assert m[50, 50] > 0.95        # 正中
    assert m[5, 5] < 0.05          # 远角
    assert (m > 0.5).sum() == pytest.approx(60 * 60, rel=0.05)


def test_dilate_expands_the_keep_region():
    roi = RoiSet()
    roi.add(_square(30, 30, 70, 70))
    tight = roi.rasterize((100, 100), dilate=0.0, feather=1.0)
    wide = roi.rasterize((100, 100), dilate=15.0, feather=1.0)
    assert wide[20, 50] > 0.9          # 膨胀后盖住了原 ROI 外 10px 处
    assert tight[20, 50] < 0.1
    assert wide.sum() > tight.sum()


def test_dilate_does_not_change_edge_shape():
    """膨胀只是平移等值面，不应改变边界轮廓。"""
    roi = RoiSet()
    roi.add(_square(30, 30, 70, 70))
    a = roi.rasterize((100, 100), dilate=0.0, feather=1.0)
    b = roi.rasterize((100, 100), dilate=10.0, feather=1.0)
    # 把 a 的边界往外推 10px 应当大致得到 b 的边界
    assert abs(float(b.sum()) - float(a.sum())) > 0
    assert b[40, 40] > 0.9 and a[40, 40] > 0.9


def test_feather_produces_intermediate_values():
    roi = RoiSet()
    roi.add(_square(30, 30, 70, 70))
    hard = roi.rasterize((100, 100), dilate=0.0, feather=0.0)
    soft = roi.rasterize((100, 100), dilate=0.0, feather=8.0)
    assert set(np.unique(hard)).issubset({0.0, 1.0})
    assert ((soft > 0.05) & (soft < 0.95)).sum() > 100   # 有过渡带


def test_subtract_mode_carves_a_hole():
    roi = RoiSet()
    roi.add(_square(20, 20, 80, 80))
    roi.add(_square(40, 40, 60, 60, subtract=True))
    m = roi.rasterize((100, 100), dilate=0.0, feather=1.0)
    assert m[50, 50] < 0.05      # 洞心被挖掉
    assert m[30, 50] > 0.95      # 洞外仍在
    assert m[5, 5] < 0.05        # ROI 外仍是 0


def test_polygon_mask_matches_its_area():
    roi = RoiSet()
    roi.add(RoiShape(kind="polygon",
                     points=[(0, 0), (60, 0), (60, 60), (0, 60)]))
    m = roi.rasterize((64, 64), dilate=0.0, feather=0.0)
    # PIL 的多边形栅格化是端点闭区间，60x60 的矩形会落成 61x61 个像素
    assert (m > 0.5).sum() == pytest.approx(61 * 61, rel=0.02)


# --------------------------------------------------------------------------
# 距离场缓存
# --------------------------------------------------------------------------
def test_signed_distance_is_cached_until_edit():
    roi = RoiSet()
    roi.add(_square(20, 20, 80, 80))
    sd1 = roi.signed_distance((100, 100))
    sd2 = roi.signed_distance((100, 100))
    assert sd1 is sd2                      # 命中缓存

    roi.add(_square(10, 10, 30, 30))
    sd3 = roi.signed_distance((100, 100))
    assert sd3 is not sd1                  # 编辑后失效


def test_signed_distance_sign_convention():
    roi = RoiSet()
    roi.add(_square(30, 30, 70, 70))
    sd = roi.signed_distance((100, 100))
    assert sd[50, 50] > 0          # 内部为正
    assert sd[5, 5] < 0            # 外部为负


def test_soft_from_signed_matches_rasterize():
    roi = RoiSet()
    roi.add(_square(30, 30, 70, 70))
    sd = roi.signed_distance((100, 100))
    a = RoiSet.soft_from_signed(sd, dilate=7.0, feather=5.0)
    b = roi.rasterize((100, 100), dilate=7.0, feather=5.0)
    assert np.allclose(a, b)


def test_signed_distance_empty_is_very_negative():
    roi = RoiSet()
    sd = roi.signed_distance((32, 32))
    assert sd.max() < 0
    assert roi.soft_from_signed(sd, dilate=50.0, feather=4.0).max() < 1e-6


# --------------------------------------------------------------------------
# 序列化
# --------------------------------------------------------------------------
def test_json_round_trip(tmp_path):
    roi = RoiSet()
    roi.add(_square(20, 20, 80, 80))
    roi.add(RoiShape(kind="polygon", points=[(5, 5), (20, 5), (20, 20)],
                     subtract=True))
    path = tmp_path / "roi.json"
    roi.save_json(str(path), fmin=0.06, fmax=0.09, dilate=6.0)

    back, extra = RoiSet.load_json(str(path))
    assert len(back) == 2
    assert extra["fmin"] == pytest.approx(0.06)
    assert back.shapes[1].subtract is True
    assert np.allclose(back.rasterize((100, 100), 0.0, 1.0),
                       roi.rasterize((100, 100), 0.0, 1.0))


def test_load_json_skips_corrupt_entries(tmp_path):
    data = {
        "shapes": [
            {"kind": "rect", "points": [[0, 0], [10, 10]], "subtract": False},
            {"kind": "hexagon", "points": [[0, 0], [1, 1]]},        # 非法 kind
            {"points": [[0, 0], [1, 1]]},                          # 缺 kind
        ],
        "fmin": 0.05,
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    roi, extra = RoiSet.load_json(str(path))
    assert len(roi) == 1              # 只保留合法的那个
    assert extra["fmin"] == pytest.approx(0.05)


def test_replace_all():
    roi = RoiSet()
    roi.add(_square())
    roi.replace_all([_square(0, 0, 10, 10)])
    assert len(roi) == 1
    assert roi.undo() and len(roi) == 1   # 撤销回替换前的状态


def test_has_keep_distinguishes_subtract_only():
    """只有挖除区时掩膜必然全零，GUI 靠 has_keep 拦住"把整幅图晶格删掉"的保存。"""
    roi = RoiSet()
    assert not roi.has_keep                       # 空集

    roi.add(_square(0, 0, 40, 40))
    assert roi.has_keep

    roi.add(_square(50, 50, 60, 60, subtract=True))
    assert roi.has_keep                           # 有保留区就算

    only_cut = RoiSet()
    only_cut.add(_square(0, 0, 40, 40, subtract=True))
    assert not only_cut.has_keep
    assert only_cut.rasterize((100, 100), 0.0, 4.0).max() == 0.0


# --------------------------------------------------------------------------
# 坐标有限性与载入反馈（P2 回归）
# --------------------------------------------------------------------------
def test_shape_rejects_nonfinite_points():
    """Python 的 json 会原样解析 NaN 字面量：NaN 坐标会让 PIL 栅格化行为未定义。"""
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            RoiShape(kind="polygon",
                     points=[(bad, 0.0), (1.0, 0.0), (1.0, 1.0)])


def test_load_json_skips_nonfinite_entries_and_reports_count(tmp_path):
    data = {
        "shapes": [
            {"kind": "rect", "points": [[0, 0], [10, 10]]},
            {"kind": "polygon", "points": [[float("nan"), 0], [5, 0], [5, 5]]},
        ],
    }
    path = tmp_path / "nan.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    roi, _ = RoiSet.load_json(str(path))
    assert len(roi) == 1
    assert roi.skipped_on_load == 1


def test_fresh_roiset_has_no_skips():
    assert RoiSet().skipped_on_load == 0
