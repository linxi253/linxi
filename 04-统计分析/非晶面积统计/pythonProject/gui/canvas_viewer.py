# -*- coding: utf-8 -*-
"""
画布图像显示组件

功能：
- 自适应缩放显示图像
- 鼠标滚轮缩放
- 鼠标拖拽平移
- 图像居中显示
- 标题叠加显示

修复了原版中缩放功能未绑定、resize 无防抖等问题。
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional
import numpy as np
import cv2
from PIL import Image, ImageTk

from constants import (
    DARK_CANVAS_BG,
    DEFAULT_CANVAS_WIDTH,
    DEFAULT_CANVAS_HEIGHT,
    MAX_ZOOM,
    MIN_ZOOM,
    ZOOM_FACTOR,
    RESIZE_DEBOUNCE_MS,
)
from core.segmentation import to_uint8


class CanvasViewer(ttk.Frame):
    """图像画布查看器，支持缩放和平移"""

    def __init__(self, parent, width=DEFAULT_CANVAS_WIDTH, height=DEFAULT_CANVAS_HEIGHT):
        super().__init__(parent)

        self._width = width
        self._height = height
        self._scale = 1.0
        self._image_tk = None  # 保持引用防止 GC
        self._current_image = None  # 当前显示的原始图像 (numpy)
        self._uint16_shift = None   # uint16 → uint8 的数据集级移位数
        self._has_image = False     # 是否已显示过图像（keep_view 首帧除外）
        self._title = ""
        self._disp_w = 0  # 最近一次渲染的显示尺寸（拖拽钳位用）
        self._disp_h = 0

        # 平移状态
        self._pan_start_x = 0
        self._pan_start_y = 0
        self._resize_timer = None  # resize 防抖

        self._build_ui()
        self._bind_events()

    def _build_ui(self):
        """构建画布和滚动条"""
        # 垂直滚动条
        v_scroll = ttk.Scrollbar(self, orient=tk.VERTICAL)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 水平滚动条
        h_scroll = ttk.Scrollbar(self, orient=tk.HORIZONTAL)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        # 画布
        self.canvas = tk.Canvas(
            self,
            width=self._width,
            height=self._height,
            bg=DARK_CANVAS_BG,
            cursor="fleur",
            yscrollcommand=v_scroll.set,
            xscrollcommand=h_scroll.set,
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        v_scroll.config(command=self.canvas.yview)
        h_scroll.config(command=self.canvas.xview)

    def _bind_events(self):
        """绑定交互事件"""
        # 鼠标滚轮缩放
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)  # Windows
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)    # Linux 上滚
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)    # Linux 下滚

        # 鼠标拖拽平移
        self.canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<B1-Motion>", self._on_pan_move)

        # 画布尺寸变化
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_canvas_configure(self, event):
        """画布尺寸变化时更新（带防抖）"""
        self._width = event.width
        self._height = event.height
        if self._current_image is not None:
            if self._resize_timer:
                self.after_cancel(self._resize_timer)
            self._resize_timer = self.after(RESIZE_DEBOUNCE_MS, self._render_image)

    def _on_mouse_wheel(self, event):
        """鼠标滚轮缩放"""
        if self._current_image is None:
            return

        # 判断滚动方向
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            self._scale = min(MAX_ZOOM, self._scale * ZOOM_FACTOR)
        elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            self._scale = max(MIN_ZOOM, self._scale / ZOOM_FACTOR)

        self._render_image()

    def _on_pan_start(self, event):
        """开始拖拽"""
        self._pan_start_x = event.x
        self._pan_start_y = event.y
        self.canvas.config(cursor="grabbing")

    def _on_pan_move(self, event):
        """拖拽平移（仅移动图像，不移动标题/指示器；钳位防止图像被推出可视区）"""
        dx = event.x - self._pan_start_x
        dy = event.y - self._pan_start_y
        self._pan_start_x = event.x
        self._pan_start_y = event.y
        # 只移动图像对象，不移动标题和缩放指示器
        self.canvas.move("img_obj", dx, dy)

        # 钳位：图像左上角坐标限制在 [min(0, 画布-图宽), max(0, 画布-图宽)]，
        # 图像必须始终与可视区有重叠，避免拖出后无法找回
        if self._disp_w > 0 and self._disp_h > 0:
            coords = self.canvas.coords("img_obj")
            if coords:
                x_lo, x_hi = sorted((self._width - self._disp_w, 0))
                y_lo, y_hi = sorted((self._height - self._disp_h, 0))
                nx = min(max(coords[0], x_lo), x_hi)
                ny = min(max(coords[1], y_lo), y_hi)
                self.canvas.coords("img_obj", nx, ny)

    def display_image(self, image: np.ndarray, title: str = "",
                      uint16_shift: Optional[int] = None,
                      keep_view: bool = False):
        """
        显示图像（公开接口）

        Args:
            image: numpy 图像数组（灰度或 BGR）
            title: 显示标题
            uint16_shift: uint16 输入的数据集级移位数（与分割路径一致）
            keep_view: True 时保留当前缩放比例（用于切片导航逐帧对比）
        """
        self._current_image = image
        self._uint16_shift = uint16_shift
        self._title = title

        if not (keep_view and self._has_image):
            # 自适应缩放：首次显示或明确要求时重新计算 fit 比例
            if len(image.shape) == 3:
                h, w = image.shape[:2]
            else:
                h, w = image.shape
            self._scale = min(self._width / w, self._height / h, 1.0)
        self._has_image = True

        self._render_image()

    def _render_image(self):
        """渲染当前图像到画布"""
        if self._current_image is None:
            return

        try:
            # 统一走 to_uint8：uint8 原样、uint16 按数据集级 shift、float 含
            # NaN/Inf 按背景处理（与分割路径同一映射；保留彩色通道供标注图显示）
            image = to_uint8(self._current_image, uint16_shift=self._uint16_shift)

            # 获取原始尺寸
            if len(image.shape) == 3:
                h, w = image.shape[:2]
            else:
                h, w = image.shape

            # 计算显示尺寸
            new_w = max(1, int(w * self._scale))
            new_h = max(1, int(h * self._scale))
            self._disp_w, self._disp_h = new_w, new_h

            # 缩放图像
            interpolation = cv2.INTER_AREA if self._scale < 1.0 else cv2.INTER_LINEAR
            img_resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

            # 转 RGB
            if len(img_resized.shape) == 2:
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            # 转 PIL → Tkinter
            pil_image = Image.fromarray(img_rgb)
            self._image_tk = ImageTk.PhotoImage(pil_image)

            # 清除画布并绘制
            self.canvas.delete("all")
            self.canvas.config(scrollregion=(0, 0, new_w, new_h))

            # 居中显示
            x_offset = max(0, (self._width - new_w) // 2)
            y_offset = max(0, (self._height - new_h) // 2)

            self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self._image_tk,
                                     tags="img_obj")

            # 标题文字
            if self._title:
                self.canvas.create_text(
                    10, 10, text=self._title, anchor=tk.NW,
                    fill='white', font=('Arial', 10, 'bold'),
                    tags="title"
                )

            # 缩放比例指示
            zoom_text = f"{self._scale * 100:.0f}%"
            self.canvas.create_text(
                self._width - 10, 10, text=zoom_text, anchor=tk.NE,
                fill='#aaaaaa', font=('Arial', 9),
                tags="zoom_indicator"
            )

        except Exception:
            pass  # 渲染失败时静默处理

    def reset_view(self):
        """重置视图（恢复自适应缩放；保留数据集级 uint16 位移）。

        2026-09-28 审查修正：此前重置/窗口 resize 后 uint16_shift 被丢回
        None（单图自动判定），12-bit 数据集中较暗切片的显示亮度会与
        分割路径不一致地跳变。
        """
        if self._current_image is not None:
            self.display_image(
                self._current_image, self._title,
                uint16_shift=self._uint16_shift)

    def zoom_in(self):
        """放大"""
        self._scale = min(MAX_ZOOM, self._scale * ZOOM_FACTOR)
        self._render_image()

    def zoom_out(self):
        """缩小"""
        self._scale = max(MIN_ZOOM, self._scale / ZOOM_FACTOR)
        self._render_image()

    def clear(self):
        """清空画布"""
        self.canvas.delete("all")
        self._current_image = None
        self._image_tk = None
