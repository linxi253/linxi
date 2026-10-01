"""Pixel-center / YOLO edge-coordinate conversion shared by all backends."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def points_to_yolo(points, width: int, height: int) -> np.ndarray:
    return (np.asarray(points, dtype=np.float64).reshape(-1, 2)+.5)/[width, height]


def yolo_to_points(centers, width: int, height: int) -> np.ndarray:
    return np.asarray(centers, dtype=np.float64).reshape(-1, 2)*[width, height]-.5


@dataclass(frozen=True)
class Letterbox:
    source_hw: tuple[int, int]
    target_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    padding_xy: tuple[int, int]

    @classmethod
    def create(cls, source_hw, target_hw):
        h, w = (int(v) for v in source_hw)
        th, tw = (int(v) for v in target_hw)
        if min(h, w, th, tw) <= 0:
            raise ValueError("image dimensions must be positive")
        ratio = min(th/h, tw/w)
        rh, rw = max(1, round(h*ratio)), max(1, round(w*ratio))
        return cls((h, w), (th, tw), (rh, rw), ((tw-rw)//2, (th-rh)//2))

    @property
    def scale_xy(self):
        return np.asarray([self.resized_hw[1]/self.source_hw[1],
                           self.resized_hw[0]/self.source_hw[0]])

    def forward_points(self, points):
        return (np.asarray(points, dtype=np.float64)+.5)*self.scale_xy+self.padding_xy

    def inverse_centers(self, centers):
        return (np.asarray(centers, dtype=np.float64)-self.padding_xy)/self.scale_xy-.5
