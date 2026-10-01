"""
HRTEM/STEM 图像增强 - 图像预览画布组件

使用 PPM 格式在 tkinter Canvas 中显示 numpy 数组，
无需 PIL/Pillow 依赖。

支持:
- 自动缩放适应画布大小
- 灰度/彩色图像显示
- 像素信息显示（鼠标悬停）
"""

import base64
import contextlib
import logging
import tkinter as tk

import cv2
import numpy as np

from contrast import uint16_to_display_uint8

logger = logging.getLogger(__name__)

RESIZE_DEBOUNCE_MS = 100


class PreviewCanvas(tk.Frame):
    """
    图像预览画布。

    将 numpy 数组转换为 PPM 格式后通过 tkinter PhotoImage 显示，
    完全不依赖 PIL/Pillow。

    用法:
        canvas = PreviewCanvas(parent)
        canvas.update_image(numpy_array)
    """

    def __init__(self, parent, bg_color="#1e1e1e", **kwargs):
        super().__init__(parent, **kwargs)

        self.bg_color = bg_color
        self._photo = None
        self._image_id = None
        self._current_array = None
        self._display_array = None
        self._display_range = None
        self._range_text = ""
        self._display_scale = 1.0
        self._resizing = (
            False  # 防止 _on_resize → update_image → Configure 事件循环递归
        )
        self._resize_after_id = None  # 保存 resize 防抖回调 id，clear/destroy 时取消

        # 画布
        self.canvas = tk.Canvas(
            self, bg=bg_color, highlightthickness=0, cursor="crosshair"
        )
        self.canvas.pack(fill="both", expand=True)

        # 信息标签（显示鼠标位置像素值）
        self.info_label = tk.Label(
            self,
            text="",
            anchor="w",
            bg="#333333",
            fg="#cccccc",
            font=("Consolas", 9),
            padx=5,
        )
        self.info_label.pack(fill="x", side="bottom")

        # 绑定鼠标事件
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.canvas.bind("<Configure>", self._on_resize)

    def update_image(
        self,
        image_array: np.ndarray,
        display_array: np.ndarray = None,
        display_range: tuple[float, float] | None = None,
    ):
        """
        更新预览图像。

        画布保存传入数组的只读视图而非副本（单帧可能达数百 MB）；
        调用方在下一次 update_image/clear 之前不得修改数组内容。

        Parameters
        ----------
        image_array : np.ndarray
            图像数据（2D 灰度或 3D 彩色，任意位深）
        """
        image_array = np.asarray(image_array)
        if image_array.ndim not in (2, 3) or image_array.size == 0:
            raise ValueError(f"无法预览 shape={image_array.shape} 的图像")
        self._current_array = self._readonly_view(image_array)
        self._display_array = (
            None if display_array is None else self._readonly_view(display_array)
        )
        if (
            self._display_array is not None
            and self._display_array.shape[:2] != image_array.shape[:2]
        ):
            raise ValueError("显示数组尺寸必须与原始图像一致")
        self._display_range = display_range
        # 显式标注当前显示窗口来源：同一画布在打开时用帧内自适应窗口、
        # 首次预览后切换为全堆栈窗口，隐性切换会让用户误以为图像变了。
        if display_range is None:
            self._range_text = "窗口: 帧内自适应 0.5–99.5%"
        else:
            self._range_text = (
                f"窗口: [{display_range[0]:.4g}, {display_range[1]:.4g}]"
            )

        # 等待画布完成布局
        self.canvas.update_idletasks()
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w < 10 or canvas_h < 10:
            canvas_w, canvas_h = 400, 300

        # 计算缩放比例
        h, w = image_array.shape[:2]
        scale = min(canvas_w / w, canvas_h / h, 1.0)
        self._display_scale = scale
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        # 先缩小再转换为 uint8，避免为超大原图分配整幅显示副本。
        display_source = (
            image_array if self._display_array is None else self._display_array
        )
        if scale < 1.0:
            display_source = cv2.resize(
                display_source, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
        display_img = self._to_display_uint8(display_source, display_range)

        # 转换为 PPM 格式
        ppm_data = self._array_to_ppm(display_img)

        # 创建 PhotoImage
        self._photo = tk.PhotoImage(data=base64.b64encode(ppm_data))

        # 绘制到画布
        if self._image_id is not None:
            self.canvas.delete(self._image_id)

        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2
        self._image_id = self.canvas.create_image(x, y, anchor="nw", image=self._photo)

        # 更新信息
        self.info_label.config(
            text=f"尺寸: {w}x{h} | dtype: {image_array.dtype} | "
            f"显示比例: {scale:.2f}x | {self._range_text}"
        )

    @staticmethod
    def _readonly_view(array: np.ndarray) -> np.ndarray:
        """Return a read-only view so canvas-held data cannot be written to."""
        view = np.asarray(array).view()
        view.flags.writeable = False
        return view

    @staticmethod
    def _to_display_uint8(
        image_array: np.ndarray,
        display_range: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """将任意位深的图像转换为 uint8 用于显示。"""
        if image_array.dtype == np.uint8 and display_range is None:
            return image_array

        if display_range is None:
            vmin = float(np.percentile(image_array, 0.5))
            vmax = float(np.percentile(image_array, 99.5))
        else:
            vmin, vmax = (float(display_range[0]), float(display_range[1]))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax < vmin:
                raise ValueError("显示强度范围无效")
            if image_array.dtype == np.uint16 and vmin <= 0.0 and vmax >= 65535.0:
                return uint16_to_display_uint8(image_array)

        if vmax - vmin < 1e-10:
            return np.zeros_like(image_array, dtype=np.uint8)

        scaled = (np.asarray(image_array, dtype=np.float64) - vmin) / (vmax - vmin)
        return np.rint(np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)

    @staticmethod
    def _array_to_ppm(image_uint8: np.ndarray) -> bytes:
        """
        将 uint8 numpy 数组转换为 PPM (P6) 格式字节。

        Parameters
        ----------
        image_uint8 : np.ndarray
            uint8 图像（2D 灰度或 3D BGR/RGB）

        Returns
        -------
        bytes
            PPM 格式数据
        """
        if len(image_uint8.shape) == 2:
            # 灰度转 RGB
            rgb = np.stack([image_uint8] * 3, axis=-1)
        elif image_uint8.shape[2] == 3:
            rgb = image_uint8
        elif image_uint8.shape[2] == 4:
            rgb = image_uint8[:, :, :3]
        else:
            rgb = np.stack([image_uint8[:, :, 0]] * 3, axis=-1)

        h, w = rgb.shape[:2]
        header = f"P6\n{w} {h}\n255\n".encode("ascii")
        # 确保 C 连续
        img_bytes = np.ascontiguousarray(rgb).tobytes()
        return header + img_bytes

    def _on_mouse_move(self, event):
        """鼠标移动时显示像素信息。"""
        if self._current_array is None:
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        h, w = self._current_array.shape[:2]
        scale = self._display_scale

        if scale <= 0:
            return

        # 画布坐标转图像坐标
        offset_x = (canvas_w - int(w * scale)) // 2
        offset_y = (canvas_h - int(h * scale)) // 2

        img_x = int((event.x - offset_x) / scale)
        img_y = int((event.y - offset_y) / scale)

        if 0 <= img_x < w and 0 <= img_y < h:
            pixel_val = self._current_array[img_y, img_x]
            if np.isscalar(pixel_val):
                val_str = f"值={pixel_val}"
            else:
                val_str = f"值={tuple(pixel_val)}"

            self.info_label.config(
                text=f"{self._range_text} | 坐标: ({img_x}, {img_y}) | {val_str} | "
                f"尺寸: {w}x{h} | dtype: {self._current_array.dtype}"
            )

    def _cancel_resize_after(self):
        """取消尚未执行的 resize 防抖回调。"""
        if self._resize_after_id is not None:
            with contextlib.suppress(tk.TclError):
                self.after_cancel(self._resize_after_id)
            self._resize_after_id = None

    def _on_resize(self, event):
        """画布大小改变时重新渲染（带防抖保护，避免无限递归）。"""
        if self._current_array is None or self._resizing:
            return
        self._resizing = True
        self._cancel_resize_after()

        def _do_update():
            self._resize_after_id = None
            try:
                if self._current_array is not None:
                    self.update_image(
                        self._current_array,
                        self._display_array,
                        self._display_range,
                    )
            finally:
                # update_image 内的 update_idletasks 会同步派发 Configure；
                # 渲染期间保持守卫为 True 抑制重入，且异常时也保证复位。
                self._resizing = False

        # 延迟 100ms 执行，合并连续 resize 事件（如窗口拖拽调整大小时）
        self._resize_after_id = self.after(RESIZE_DEBOUNCE_MS, _do_update)

    def clear(self):
        """清空画布。"""
        self._cancel_resize_after()
        self._resizing = False
        if self._image_id is not None:
            self.canvas.delete(self._image_id)
            self._image_id = None
        self._photo = None
        self._current_array = None
        self._display_array = None
        self._display_range = None
        self._range_text = ""
        self.info_label.config(text="")

    def destroy(self):
        """销毁控件前取消未执行的 after 回调。"""
        self._cancel_resize_after()
        super().destroy()
