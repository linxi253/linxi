from __future__ import annotations

import unittest

from hrtem_filter.geometry import clamp_drag_rect


class ClampDragRectTests(unittest.TestCase):
    """GUI 拖拽矩形的纯函数：起点越界必须与终点一样被钳制。

    matplotlib 坐标区带边距，按下点可以落在图像之外（负坐标或越界）；
    不钳制会生成非法 ROI，直到预览时才以原始错误暴露。
    """

    def test_start_outside_image_is_clamped(self) -> None:
        # shape 为 (行, 列) = (100, 80)；x 属于 [0, 79]，y 属于 [0, 99]。
        self.assertEqual(clamp_drag_rect((100, 80), (-5, -3), (50, 40)), (0, 0, 51, 41))

    def test_end_outside_image_is_clamped(self) -> None:
        self.assertEqual(clamp_drag_rect((100, 80), (10, 10), (200, 200)), (10, 10, 80, 100))

    def test_start_beyond_far_edge_is_clamped(self) -> None:
        self.assertEqual(clamp_drag_rect((100, 80), (120, 90), (50, 40)), (50, 40, 80, 91))

    def test_reversed_drag_direction_is_normalised(self) -> None:
        self.assertEqual(clamp_drag_rect((100, 80), (60, 50), (10, 5)), (10, 5, 61, 51))

    def test_single_pixel_drag_is_valid_right_open_rect(self) -> None:
        self.assertEqual(clamp_drag_rect((100, 80), (7, 9), (7, 9)), (7, 9, 8, 10))


if __name__ == "__main__":
    unittest.main()
