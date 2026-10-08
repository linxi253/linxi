# -*- coding: utf-8 -*-
import queue
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from drift_core import DetectionResult
from drift_correction import (
    DriftCorrectionApp,
    _package_health_check,
    _preview_contrast_range,
    _representative_preview_frame,
)

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



class DriftCorrectionPreviewTests(unittest.TestCase):
    def test_package_health_check_runs_real_nonzero_correction(self):
        _package_health_check()

    def test_representative_preview_uses_largest_2d_drift(self):
        self.assertEqual(
            _representative_preview_frame(
                np.array([0.0, 5.0, 3.0]), np.array([0.0, 1.0, 8.0])),
            2,
        )

    def test_detection_completion_moves_preview_off_zero_shift_reference(self):
        app = DriftCorrectionApp.__new__(DriftCorrectionApp)
        app._closing = False
        app.document_id = 7
        app.current_filepath = "source.tif"
        app.frames = [np.zeros((8, 8), np.uint8) for _ in range(3)]
        app.current_frame_idx = 0
        app._corr_cache_key = (0, 0.0, 0.0, id(app.frames))
        app._task_start_time = time.time()
        app._set_busy = mock.Mock()
        app.status_var = mock.Mock()
        app.progress = {}
        app.time_label = mock.Mock()
        app.frame_slider = mock.Mock()
        app._update_preview = mock.Mock()
        app._update_curve = mock.Mock()
        result = DetectionResult(
            shifts_x=np.array([0.0, 2.0, 12.0], np.float32),
            shifts_y=np.array([0.0, -1.0, -3.0], np.float32),
            pair_quality=[], key_indices=[0, 1, 2], normalization=(0.0, 1.0),
        )

        app._on_detect_complete(result, document_id=7, source_path="source.tif")

        self.assertEqual(app.current_frame_idx, 2)
        app.frame_slider.set.assert_called_once_with(2)
        self.assertIsNone(app._corr_cache_key)
        app._update_preview.assert_called_once_with()
        app._update_curve.assert_called_once_with()

    @mock.patch("drift_correction.messagebox.showwarning")
    def test_save_rejects_loaded_but_undetected_zero_shifts(self, showwarning):
        app = DriftCorrectionApp.__new__(DriftCorrectionApp)
        app._busy = False
        app.frames = [np.zeros((8, 8), np.uint8) for _ in range(3)]
        # 回归旧故障：即使旧状态残留出一组零数组，也不能把它当检测结果保存。
        app.shifts = np.zeros(3, np.float32)
        app.shifts_y = np.zeros(3, np.float32)
        app.detection_result = None
        app.manual_mode = mock.Mock()
        app.manual_mode.get.return_value = False
        app.manual_dirty = False
        app._start_worker = mock.Mock()

        app._execute_correction()

        showwarning.assert_called_once()
        app._start_worker.assert_not_called()

    @mock.patch("drift_correction.messagebox.askyesno", return_value=True)
    def test_batch_does_not_reuse_undetected_zero_shift_cache(self, _askyesno):
        app = DriftCorrectionApp.__new__(DriftCorrectionApp)
        app._busy = False
        app.file_list = ["source.tif"]
        app.input_folder = mock.Mock()
        app.input_folder.get.return_value = ""  # 留空以跳过输出目录防护分支
        app.output_folder = mock.Mock()
        app.output_folder.get.return_value = "output"
        app.worker = None
        app._get_sift_params = mock.Mock(return_value={})
        app._get_skip_interval = mock.Mock(return_value=1)
        app.current_filepath = "source.tif"
        app.frames = [np.zeros((8, 8), np.uint8) for _ in range(3)]
        app.shifts = np.zeros(3, np.float32)
        app.shifts_y = np.zeros(3, np.float32)
        app.detection_result = None
        app.meta = {"n_frames": 3, "shape": (8, 8), "dtype": np.dtype("uint8")}
        app._start_worker = mock.Mock()

        app._batch_process()

        kwargs = app._start_worker.call_args.kwargs["kwargs"]
        self.assertEqual(kwargs["precomputed"], {})

    def _bare_app(self) -> DriftCorrectionApp:
        """构造绕过 __init__ 的最小 app，供无窗口状态测试使用。"""
        return DriftCorrectionApp.__new__(DriftCorrectionApp)

    def test_ui_queue_survives_callback_exception(self):
        app = self._bare_app()
        app._closing = False
        app._ui_queue = queue.Queue()
        executed = []
        app._ui_queue.put((mock.Mock(side_effect=RuntimeError("boom")), ()))
        app._ui_queue.put((lambda: executed.append(True), ()))
        app.root = mock.Mock()

        app._drain_ui_queue()

        self.assertEqual(executed, [True])
        app.root.after.assert_called_once_with(30, app._drain_ui_queue)

    def test_slider_change_clamps_on_single_frame(self):
        app = self._bare_app()
        app.frames = [np.zeros((8, 8), np.uint8)]
        app.current_frame_idx = 0
        app._update_preview = mock.Mock()
        app._update_curve = mock.Mock()

        app._on_slider_change(1.0)

        self.assertEqual(app.current_frame_idx, 0)
        app._update_preview.assert_not_called()

    def test_image_click_without_frames_is_guarded(self):
        app = self._bare_app()
        app.frames = []
        app.marker_mode = mock.Mock()
        app.marker_mode.get.return_value = True
        shared_ax = object()
        app.ax_prev = app.ax_curr = app.ax_corr = shared_ax
        event = mock.Mock(inaxes=shared_ax, xdata=5.0, ydata=5.0)

        app._on_image_click(event)  # 不应因 self.frames[0] 抛 IndexError

    def test_preview_contrast_range_uses_robust_percentiles(self):
        frame = np.full((128, 128), 1200.0, np.float32)
        frame[:76, :] = 800.0   # 约 60% 低值区，让 P1 落在真实数据上
        frame[0, 0] = 60000.0   # 单个热像素：min/max 归一会让预览整体发黑
        low, high = _preview_contrast_range([frame])
        self.assertLess(high, 2000.0)   # min/max 会得到 60000
        self.assertGreater(high, 1000.0)
        self.assertLess(low, 1000.0)
        self.assertGreater(low, 500.0)

    def test_preview_contrast_range_flat_fallback(self):
        self.assertEqual(_preview_contrast_range([np.full((8, 8), 5, np.uint8)]), (5.0, 6.0))
        self.assertEqual(_preview_contrast_range([np.full((8, 8), np.nan)]), (0.0, 1.0))

    @mock.patch("drift_correction.messagebox.showwarning")
    def test_get_sift_params_delegates_range_validation_to_core(self, showwarning):
        app = self._bare_app()

        def var(value):
            v = mock.Mock()
            v.get.return_value = value
            return v

        app.param_nfeatures = var("5000")
        app.param_noctave_layers = var("3")
        app.param_contrast_threshold = var("0.04")
        app.param_edge_threshold = var("10")
        app.param_sigma = var("1.6")
        # GUI 旧规则只要求 (0, 1)，会放过 0.05；核心要求 [0.1, 1)。
        app.param_ratio_threshold = var("0.05")
        app.param_ransac_threshold = var("3.0")
        app.param_min_matches = var("4")
        app.param_min_inliers = var("6")
        app.param_min_inlier_ratio = var("0.1")
        app.param_max_pair_residual = var("3.0")
        app.param_min_valid_pair_ratio = var("0.8")
        app.param_max_interpolation_gap = var("2")

        self.assertIsNone(app._get_sift_params())
        showwarning.assert_called_once()
        self.assertIn("Ratio", showwarning.call_args[0][1])

    @mock.patch("drift_correction.messagebox.showwarning")
    def test_get_skip_interval_rejects_nonpositive(self, showwarning):
        app = self._bare_app()
        app.param_skip_interval = mock.Mock()
        app.param_skip_interval.get.return_value = "0"
        app.frames = [np.zeros((8, 8), np.uint8) for _ in range(5)]

        self.assertIsNone(app._get_skip_interval())
        showwarning.assert_called_once()

    def _batch_app(self) -> DriftCorrectionApp:
        app = self._bare_app()
        app._busy = False
        app.file_list = ["a.tif", "b.tif"]
        app.input_folder = mock.Mock()
        app.input_folder.get.return_value = ""  # 跳过输出目录防护分支
        app.output_folder = mock.Mock()
        app.output_folder.get.return_value = "output"
        app.worker = None
        app._get_sift_params = mock.Mock(return_value={})
        app._get_skip_interval = mock.Mock(return_value=1)
        app.current_filepath = None
        app.frames = []
        app._start_worker = mock.Mock()
        return app

    @mock.patch("drift_correction.messagebox.askyesno", return_value=True)
    @mock.patch("drift_correction.TiffIO.inspect_stack")
    def test_batch_axis_confirmation_passes_flag(self, inspect_mock, _askyesno):
        app = self._batch_app()
        inspect_mock.side_effect = [
            {"confirm_required": ["含 Z 轴"], "axes": "ZYX"},
            {"confirm_required": [], "axes": "TYX"},
        ]

        app._batch_process()

        _askyesno.assert_called_with(mock.ANY, mock.ANY)  # 至少含轴确认询问
        kwargs = app._start_worker.call_args.kwargs["kwargs"]
        self.assertTrue(kwargs["allow_ambiguous_axes"])

    @mock.patch("drift_correction.messagebox.askyesno",
                side_effect=[True, False])  # 数量确认继续；轴确认拒绝
    @mock.patch("drift_correction.TiffIO.inspect_stack")
    def test_batch_axis_declined_keeps_flag_false(self, inspect_mock, _askyesno):
        app = self._batch_app()
        inspect_mock.side_effect = [
            {"confirm_required": ["含 Z 轴"], "axes": "ZYX"},
            {"confirm_required": [], "axes": "TYX"},
        ]

        app._batch_process()

        kwargs = app._start_worker.call_args.kwargs["kwargs"]
        self.assertFalse(kwargs["allow_ambiguous_axes"])
        # 批处理仍会启动：含 C/Z 的文件由核心层逐个拒绝并记录，其余照常
        app._start_worker.assert_called_once()

    # ---- v7.3：两段式加载、内存确认、预览平移、配置持久化 ----

    def _inspect_app(self):
        app = self._bare_app()
        app._closing = False
        app._set_busy = mock.Mock()
        app.status_var = mock.Mock()
        app._start_worker = mock.Mock()
        return app

    def test_inspect_complete_small_stack_starts_read_with_meta(self):
        """常规堆栈：元数据检视通过后直接进入像素读取，并复用 meta。"""
        app = self._inspect_app()
        meta = {"confirm_required": [], "warnings": [], "estimated_bytes": 4096}
        with mock.patch("drift_correction.total_physical_memory", return_value=None):
            app._on_inspect_complete(meta, "a.tif", 0)
        app._start_worker.assert_called_once()
        kwargs = app._start_worker.call_args.kwargs
        self.assertEqual(kwargs["target_func"].__name__, "read_stack")
        self.assertEqual(kwargs["kwargs"], {"meta": meta})

    @mock.patch("drift_correction.messagebox.askyesno", return_value=False)
    @mock.patch("drift_correction.total_physical_memory", return_value=16 * 1024 ** 3)
    def test_inspect_complete_large_stack_declined(self, _mem, _ask):
        app = self._inspect_app()
        # 解码体积 100% 于物理内存，必须要求确认；拒绝后不得开始读取
        meta = {"confirm_required": [], "warnings": [], "estimated_bytes": 16 * 1024 ** 3}
        app._on_inspect_complete(meta, "a.tif", 0)
        app._start_worker.assert_not_called()

    def test_pan_motion_moves_center_and_clamps(self):
        app = self._bare_app()
        app.frames = [np.zeros((100, 100), np.uint8)]
        app.zoom_level = 1.0
        app._pan_cx = None
        app._pan_cy = None
        app._pan_state = None
        app.marker_mode = mock.Mock()
        app.marker_mode.get.return_value = False
        ax = mock.Mock()
        ax.bbox = mock.Mock(width=200, height=200)
        app.ax_prev = None
        app.ax_curr = ax
        app.ax_corr = object()
        press = mock.Mock(inaxes=ax, button=1, xdata=50.0, ydata=50.0, x=200, y=150)
        app.canvas = mock.Mock()
        app._on_image_press(press)
        self.assertIsNotNone(app._pan_state)
        # 光标右移 10 屏幕 px；缩放 1.0、bbox 宽 200px → 0.5 数据px/屏幕px
        # 抓图语义：图像跟随光标 → 视野中心左移 5 数据px
        motion = mock.Mock(x=210, y=150)
        app._apply_zoom_to_axes = mock.Mock()
        app._on_image_motion(motion)
        self.assertAlmostEqual(app._pan_cx, 45.0)
        self.assertAlmostEqual(app._pan_cy, 50.0)
        # 大幅拖拽后中心被钳制在图像范围内
        motion_far = mock.Mock(x=2000, y=150)
        app._on_image_motion(motion_far)
        self.assertEqual(app._pan_cx, 0.0)
        app._on_image_release(mock.Mock())
        self.assertIsNone(app._pan_state)

    def test_config_save_restore_roundtrip(self):
        def var(value):
            holder = mock.Mock()
            holder.get.return_value = value
            return holder

        app = self._bare_app()
        app.input_folder = var("Z:\\不存在目录")
        app.output_folder = var("Z:\\不存在输出")
        app.file_type = var("*.tif")
        app.manual_step = var("0.5")
        app.param_ratio_threshold = var("0.92")
        app.param_skip_interval = var("2")
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Path(tmpdir) / "drift_config.json"
            with mock.patch("drift_correction._config_path", return_value=cfg):
                app._save_config()
                self.assertTrue(cfg.is_file())
                restored = self._bare_app()
                restored.input_folder = var("")
                restored.output_folder = var("")
                restored.file_type = var("*.tif")
                restored.manual_step = var("0.5")
                restored.param_ratio_threshold = var("0.5")
                restored.param_skip_interval = var("1")
                restored._load_file_list = mock.Mock()
                restored._restore_config()
            # 不存在的目录不恢复（也不触发文件列表加载），参数按名恢复
            restored._load_file_list.assert_not_called()
            restored.param_ratio_threshold.set.assert_called_once_with("0.92")
            restored.param_skip_interval.set.assert_called_once_with("2")
        with mock.patch("drift_correction._config_path",
                        return_value=Path("Z:/definitely/missing/config.json")):
            empty = self._bare_app()
            empty._load_file_list = mock.Mock()
            empty._restore_config()  # 配置缺失时静默返回
            empty._load_file_list.assert_not_called()


if __name__ == "__main__":
    unittest.main()
