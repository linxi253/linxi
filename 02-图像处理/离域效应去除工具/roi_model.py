# -*- coding: utf-8 -*-
"""ROI 模型 —— 手绘晶体区域，栅格化成软掩膜。

坐标约定
--------
所有点存成 ``(x, y)``，其中 ``x = 列号``、``y = 行号``，与原图像素坐标
一致（与 matplotlib 在 ``extent=[0, W, H, 0]`` 下的 data 坐标也一致）。
这样预览用的降采样不会影响 ROI 精度。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw

__all__ = ["RoiShape", "RoiSet"]

Point = Tuple[float, float]

#: 小于这个像素面积的形状视为误点，直接丢弃
MIN_AREA_PX = 4.0

KINDS = ("polygon", "freehand", "rect", "ellipse")


@dataclass
class RoiShape:
    """一个 ROI 形状。``subtract=True`` 表示这个形状从保留区里挖掉。"""

    kind: str
    points: List[Point] = field(default_factory=list)
    subtract: bool = False

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"未知的 ROI 类型: {self.kind!r}（可选 {KINDS}）")
        pts = []
        for x, y in self.points:
            fx, fy = float(x), float(y)
            # Python 的 json 会原样解析 NaN 字面量：带 NaN 的坐标会让
            # bbox() 变 NaN，PIL 栅格化的行为未定义，必须在入口拦住。
            if not (math.isfinite(fx) and math.isfinite(fy)):
                raise ValueError(f"ROI 坐标必须是有限数，收到 ({x!r}, {y!r})")
            pts.append((fx, fy))
        self.points = pts

    # -- 几何辅助 ---------------------------------------------------------
    def bbox(self) -> Tuple[float, float, float, float]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)

    def area(self) -> float:
        """多边形面积（鞋带公式）；矩形/椭圆由其外接框推算。"""
        if self.kind in ("rect", "ellipse") and len(self.points) >= 2:
            x0, y0, x1, y1 = self.bbox()
            a = abs(x1 - x0) * abs(y1 - y0)
            return float(np.pi * a / 4.0) if self.kind == "ellipse" else float(a)
        if len(self.points) < 3:
            return 0.0
        pts = np.asarray(self.points, dtype=np.float64)
        x, y = pts[:, 0], pts[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)

    def is_valid(self) -> bool:
        if self.kind in ("rect", "ellipse"):
            return len(self.points) >= 2 and self.area() >= MIN_AREA_PX
        return len(self.points) >= 3 and self.area() >= MIN_AREA_PX

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "subtract": bool(self.subtract),
            "points": [[round(float(x), 2), round(float(y), 2)] for x, y in self.points],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoiShape":
        return cls(
            kind=str(data["kind"]),
            points=[(float(p[0]), float(p[1])) for p in data["points"]],
            subtract=bool(data.get("subtract", False)),
        )

    def draw(self, draw: ImageDraw.ImageDraw, value: int = 1) -> None:
        """把形状画进 PIL 的 8bit 掩膜图。"""
        if self.kind == "ellipse":
            x0, y0, x1, y1 = self.bbox()
            draw.ellipse([x0, y0, x1, y1], fill=value)
            return
        if self.kind == "rect":
            x0, y0, x1, y1 = self.bbox()
            draw.rectangle([x0, y0, x1, y1], fill=value)
            return
        if len(self.points) >= 3:
            draw.polygon(self.points, fill=value)
        elif len(self.points) == 2:
            # 只有两个点：当成一段细线，避免直接丢弃
            draw.line(self.points, fill=value, width=1)


class RoiSet:
    """一组 ROI 形状，负责栅格化与撤销/重做。"""

    def __init__(self) -> None:
        self.shapes: List[RoiShape] = []
        self._undo: List[List[RoiShape]] = []
        self._redo: List[List[RoiShape]] = []
        # 有符号距离变换只依赖形状，与 dilate/feather 无关，所以缓存下来：
        # 拖动"膨胀/羽化"滑块时不必重算距离变换，预览才能跟手。
        self._sd_cache: Dict[Tuple[int, int], np.ndarray] = {}
        # 上次 load_json 跳过了多少个损坏形状，供 GUI 提示用户
        self.skipped_on_load = 0

    # -- 编辑 -------------------------------------------------------------
    def _snapshot(self) -> List[RoiShape]:
        return [
            RoiShape(s.kind, list(s.points), s.subtract) for s in self.shapes
        ]

    def _push_undo(self) -> None:
        self._undo.append(self._snapshot())
        if len(self._undo) > 100:
            self._undo.pop(0)
        self._redo.clear()
        self._sd_cache.clear()

    def add(self, shape: RoiShape) -> bool:
        """加入一个形状；非法形状返回 False。"""
        if not shape.is_valid():
            return False
        self._push_undo()
        self.shapes.append(shape)
        return True

    def pop(self) -> bool:
        """移除最后一个形状（配合未闭合的临时绘制）。"""
        if not self.shapes:
            return False
        self._push_undo()
        self.shapes.pop()
        return True

    def clear(self) -> None:
        if not self.shapes:
            return
        self._push_undo()
        self.shapes.clear()

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self.shapes = self._undo.pop()
        self._sd_cache.clear()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self.shapes = self._redo.pop()
        self._sd_cache.clear()
        return True

    def replace_all(self, shapes: Sequence[RoiShape]) -> None:
        self._push_undo()
        self.shapes = list(shapes)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def has_keep(self) -> bool:
        """是否至少有一个「保留」形状（即没勾"挖除"的形状）。

        只画挖除区时 rasterize_binary() 全零 -> 距离场取 -1e6 -> 软掩膜全 0，
        于是整幅图都被当成"晶体外"、晶格带被全部移除。GUI 用这个属性拦保存。
        """
        return any(not s.subtract for s in self.shapes)

    def __len__(self) -> int:
        return len(self.shapes)

    # -- 栅格化 -----------------------------------------------------------
    def rasterize_binary(self, shape_hw: Tuple[int, int]) -> np.ndarray:
        """硬边 0/1 掩膜：所有"加"形状取并集，再减去所有"减"形状。"""
        h, w = shape_hw
        keep = Image.new("L", (w, h), 0)
        cut = Image.new("L", (w, h), 0)
        dk = ImageDraw.Draw(keep)
        dc = ImageDraw.Draw(cut)
        for shape in self.shapes:
            shape.draw(dc if shape.subtract else dk, value=1)
        keep_a = np.asarray(keep, dtype=bool)
        cut_a = np.asarray(cut, dtype=bool)
        return keep_a & ~cut_a

    def signed_distance(self, shape_hw: Tuple[int, int]) -> np.ndarray:
        """有符号距离场：ROI 内部为正、外部为负，单位像素。带缓存。"""
        key = (int(shape_hw[0]), int(shape_hw[1]))
        cached = self._sd_cache.get(key)
        if cached is not None:
            return cached

        mask = self.rasterize_binary(key)
        if mask.any():
            sd = ndimage.distance_transform_edt(mask) - ndimage.distance_transform_edt(~mask)
            sd = sd.astype(np.float32)
        else:
            # 没画任何东西 -> 保留区为空。用一个足够负的常数，
            # 保证 soft_from_signed() 在任何 dilate 取值下都返回 0。
            sd = np.full(key, -1.0e6, dtype=np.float32)

        self._sd_cache.clear()  # 只保留当前尺寸，避免大图堆积
        self._sd_cache[key] = sd
        return sd

    @staticmethod
    def soft_from_signed(sd: np.ndarray, dilate: float, feather: float) -> np.ndarray:
        """把有符号距离场转成 0..1 软掩膜。dilate/feather 变化时只需重跑这一步。"""
        signed = sd + float(dilate)
        f = float(feather)
        if f <= 1e-6:
            return (signed > 0.0).astype(np.float32)
        # logistic 过渡；约 6f 宽度内走完 0 -> 1
        soft = 1.0 / (1.0 + np.exp(-np.clip(signed / (f / 3.0), -30.0, 30.0)))
        # 过渡带之外直接饱和到 0 / 1：远离晶体处不残留 1e-27 这种无意义的
        # 泄漏值，也让"没画 ROI"时掩膜严格全零。
        soft = np.where(soft < 1e-6, 0.0, soft)
        soft = np.where(soft > 1.0 - 1e-6, 1.0, soft)
        return soft.astype(np.float32)

    def rasterize(
        self, shape_hw: Tuple[int, int], dilate: float = 0.0, feather: float = 8.0
    ) -> np.ndarray:
        """生成 0..1 的软掩膜。

        dilate > 0 把保留区**向外扩张**，用来覆盖晶体外那圈虚影带。
        feather 是边缘过渡尺度（像素），越大过渡越柔和。
        """
        return self.soft_from_signed(self.signed_distance(shape_hw), dilate, feather)

    # -- 序列化 -----------------------------------------------------------
    def to_dict(self, **extra: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {"shapes": [s.to_dict() for s in self.shapes]}
        out.update(extra)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoiSet":
        roi = cls()
        for item in data.get("shapes", []):
            try:
                roi.shapes.append(RoiShape.from_dict(item))
            except (KeyError, ValueError, TypeError, IndexError):
                roi.skipped_on_load += 1  # 跳过损坏条目，不让整个文件读取失败
        return roi

    def save_json(self, path: str, **extra: Any) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(**extra), fh, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: str) -> Tuple["RoiSet", Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        extra = {k: v for k, v in data.items() if k != "shapes"}
        return cls.from_dict(data), extra

    def summary(self) -> str:
        n_add = sum(1 for s in self.shapes if not s.subtract)
        n_sub = len(self.shapes) - n_add
        return f"{n_add} 个保留区 / {n_sub} 个挖除区"
