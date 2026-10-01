import unittest
from unittest import mock

import numpy as np

from preview_canvas import RESIZE_DEBOUNCE_MS, PreviewCanvas


class PreviewCanvasTests(unittest.TestCase):
    def test_full_range_uint16_uses_integer_16_to_8_mapping(self):
        image = np.array([[0, 257, 65535]], dtype=np.uint16)
        display = PreviewCanvas._to_display_uint8(image, (0.0, 65535.0))
        np.testing.assert_array_equal(
            display,
            np.array([[0, 1, 255]], dtype=np.uint8),
        )

    def test_explicit_uint16_display_range_is_linear(self):
        image = np.array([[0, 32768, 65535]], dtype=np.uint16)
        display = PreviewCanvas._to_display_uint8(image, (0.0, 65535.0))
        np.testing.assert_array_equal(
            display,
            np.array([[0, 128, 255]], dtype=np.uint8),
        )

    def test_invalid_display_range_is_rejected(self):
        with self.assertRaises(ValueError):
            PreviewCanvas._to_display_uint8(
                np.ones((2, 2), dtype=np.uint16),
                (2.0, 1.0),
            )


class PreviewCanvasDebounceTests(unittest.TestCase):
    def _bare_canvas(self):
        canvas = object.__new__(PreviewCanvas)
        canvas._image_id = None
        canvas._photo = None
        canvas._current_array = None
        canvas._display_array = None
        canvas._display_range = None
        canvas._resizing = False
        canvas._resize_after_id = None
        canvas.info_label = mock.Mock()
        canvas.canvas = mock.Mock()
        canvas.after_cancel = mock.Mock()
        return canvas

    def test_resize_saves_after_id_and_uses_debounce_delay(self):
        canvas = self._bare_canvas()
        canvas._current_array = np.ones((8, 8), dtype=np.uint8)
        canvas.after = mock.Mock(return_value="after-1")
        canvas._on_resize(mock.Mock())
        canvas.after.assert_called_once_with(RESIZE_DEBOUNCE_MS, mock.ANY)
        self.assertEqual(canvas._resize_after_id, "after-1")
        self.assertTrue(canvas._resizing)

    def test_resize_replaces_pending_after_id(self):
        canvas = self._bare_canvas()
        canvas._current_array = np.ones((8, 8), dtype=np.uint8)
        canvas._resize_after_id = "old-after"
        canvas.after = mock.Mock(return_value="new-after")
        canvas._on_resize(mock.Mock())
        canvas.after_cancel.assert_called_once_with("old-after")
        self.assertEqual(canvas._resize_after_id, "new-after")

    def test_clear_cancels_pending_resize_after(self):
        canvas = self._bare_canvas()
        canvas._resize_after_id = "after-clear"
        canvas.clear()
        canvas.after_cancel.assert_called_once_with("after-clear")
        self.assertIsNone(canvas._resize_after_id)

    def test_destroy_cancels_pending_resize_after(self):
        canvas = self._bare_canvas()
        canvas._resize_after_id = "after-destroy"
        with mock.patch("preview_canvas.tk.Frame.destroy") as super_destroy:
            canvas.destroy()
        canvas.after_cancel.assert_called_once_with("after-destroy")
        super_destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
