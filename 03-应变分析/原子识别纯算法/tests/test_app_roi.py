# -*- coding: utf-8 -*-
"""选区 GUI 状态、快捷键守卫与后台逐帧识别的集成测试。"""
from __future__ import annotations

import csv
import json
import tempfile
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from atomic_app import AtomicRecognitionApp, _csv_safe_text, _intensity_fingerprint
from atomic_core import (
    AtomRecord,
    AtomicToolError,
    CalibrationResult,
    DetectionParams,
    DetectionRegion,
    IntensityParams,
    MarkerRule,
    TiffStackInfo,
    load_tiff_stack,
    measure_integrated_intensities,
)
from tests.tk_session import get_tk_session

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



def _gaussian_stack() -> np.ndarray:
    yy, xx = np.mgrid[:64, :64]
    centers = [(16.0, 16.0), (48.0, 16.0), (16.0, 48.0), (48.0, 48.0)]
    frames = []
    for frame_index in range(3):
        image = np.zeros((64, 64), dtype=np.float32)
        for atom_index, (x, y) in enumerate(centers):
            amplitude = 100.0 + atom_index * 15.0 + frame_index
            image += amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * 1.35**2))
        frames.append(image)
    return np.stack(frames)


class RegionWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.root, cls.app = get_tk_session()
        except tk.TclError as error:
            raise unittest.SkipTest(f"Tk 不可用：{error}") from error

    def setUp(self) -> None:
        self.app.stack = _gaussian_stack()
        # 统一走应用自己的重置入口，保证全部逐帧列表长度同步
        # （此前手写重置漏掉 frame_edited 等导致共享会话下 IndexError）。
        self.app._reset_frame_state(int(self.app.stack.shape[0]))

    def test_export_csv_writes_qc_columns_and_metadata(self) -> None:
        self.app.frame_records[0] = [AtomRecord(1, 16.0, 16.0)]
        # 帧 1 有原子但强度参数未记录 -> 溯源不一致必须被元数据标记。
        self.app.frame_records[1] = [AtomRecord(2, 20.0, 20.0)]
        self.app.frame_intensity_params[0] = IntensityParams(2.5, 3.0, 4.0, True)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "out.csv")
            # 导出完成弹窗会在轮询中触发；补丁必须覆盖 busy 等待循环全程。
            with patch("atomic_app.filedialog.asksaveasfilename", return_value=path), \
                    patch("atomic_app.messagebox.showinfo"):
                self.app.export_csv()
                deadline = time.monotonic() + 8.0
                while self.app._busy and time.monotonic() < deadline:
                    self.root.update()
                    time.sleep(0.01)
                self.root.update()
            self.assertFalse(self.app._busy, "CSV 导出未在测试时限内完成")
            self.assertTrue(Path(path).exists())
            self.assertFalse(Path(str(path) + ".part").exists(), "临时文件必须在成功后被替换")
            with open(path, encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0][9], "孔径截断")
            self.assertEqual(rows[0][10], "邻居污染")
            self.assertEqual(rows[0][11], "最近邻距离(px)")
            self.assertEqual(rows[0][12], "来源")
            self.assertEqual(rows[0][13], "本帧强度参数")
            self.assertEqual(rows[1][9], "否")
            self.assertEqual(rows[1][10], "否")
            self.assertEqual(rows[1][11], "")  # 单原子无最近邻
            self.assertEqual(rows[1][12], "自动")
            self.assertEqual(
                rows[1][13], "aperture=2.5;bg_inner=3;bg_outer=4;bright=1"
            )
            metadata_path = Path(path).with_suffix(".metadata.json")
            self.assertTrue(metadata_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["tool_version"], "1.3")
            self.assertIn("exported_at", metadata)
            self.assertIn("per_frame_detection_params", metadata)
            self.assertIn("manually_edited_frames", metadata)
            self.assertIn("truncation_note", metadata)
            self.assertIn("neighbor_contamination_note", metadata)
            self.assertEqual(
                metadata["per_frame_intensity_params"],
                [
                    {"aperture_radius": 2.5, "background_inner_radius": 3.0,
                     "background_outer_radius": 4.0, "bright": True},
                    None, None,
                ],
            )
            self.assertFalse(metadata["intensity_params_consistent_across_frames"])
            self.assertEqual(metadata["track_max_missed_frames"], 8)

    def test_scope_keeps_frame_and_stack_regions_separate(self) -> None:
        frame_region = DetectionRegion("rectangle", ((0.0, 0.0), (31.0, 63.0)))
        stack_region = DetectionRegion("polygon", ((0.0, 0.0), (31.0, 0.0), (31.0, 63.0), (0.0, 63.0)))

        self.app.selection_scope_var.set("current")
        self.app._set_active_region(frame_region)
        self.assertEqual(self.app.frame_regions[0], frame_region)
        self.assertIs(self.app._region_for_detection(0, False), frame_region)
        self.assertIsNone(self.app._region_for_detection(0, True))

        self.app.selection_scope_var.set("all")
        self.app._set_active_region(stack_region)
        self.assertEqual(self.app.stack_region, stack_region)
        self.assertIs(self.app._region_for_detection(0, False), stack_region)
        self.assertIs(self.app._region_for_detection(2, True), stack_region)

    def test_rectangle_and_polygon_mouse_workflows_store_regions(self) -> None:
        self.app.selection_scope_var.set("current")
        self.app.start_rectangle_selection()
        press = SimpleNamespace(
            inaxes=self.app.axes, xdata=5.0, ydata=6.0, button=1, dblclick=False
        )
        release = SimpleNamespace(
            inaxes=self.app.axes, xdata=30.0, ydata=40.0, button=1, dblclick=False
        )
        self.app._on_image_click(press)
        self.app._on_image_release(release)
        self.assertEqual(self.app.frame_regions[0].kind, "rectangle")
        self.assertEqual(self.app.frame_regions[0].vertices, ((5.0, 6.0), (30.0, 40.0)))

        self.app.start_polygon_selection()
        for x, y in [(8.0, 8.0), (35.0, 9.0), (24.0, 42.0)]:
            self.app._on_image_click(
                SimpleNamespace(
                    inaxes=self.app.axes,
                    xdata=x,
                    ydata=y,
                    button=1,
                    dblclick=False,
                )
            )
        self.app.finish_polygon_selection()
        self.assertEqual(self.app.frame_regions[0].kind, "polygon")
        self.assertEqual(len(self.app.frame_regions[0].vertices), 3)

    def test_stack_region_is_applied_to_every_frame_and_ids_remain_stable(self) -> None:
        self.app.detection_params = DetectionParams(
            sigma=0.8,
            min_distance=10.0,
            window=5,
            threshold=0.6,
            bright=True,
            method="com",
        )
        self.app.selection_scope_var.set("all")
        self.app.stack_region = DetectionRegion("rectangle", ((0.0, 0.0), (31.0, 63.0)))
        self.app.start_detection(all_frames=True)

        deadline = time.monotonic() + 8.0
        while self.app._busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.root.update()

        self.assertFalse(self.app._busy, "后台识别未在测试时限内完成")
        self.assertEqual([len(records) for records in self.app.frame_records], [2, 2, 2])
        id_sets = [{record.atom_id for record in records} for records in self.app.frame_records]
        self.assertEqual(id_sets, [{1, 2}, {1, 2}, {1, 2}])
        self.assertTrue(all(record.x <= 31.0 for records in self.app.frame_records for record in records))

    def test_complete_parameter_dialog_applies_calibration_prefill(self) -> None:
        suggested_detection = DetectionParams(0.55, 9.5, 7, 0.42, True, "gaussian")
        suggested_intensity = IntensityParams(2.8, 3.8, 5.1, True)
        calibration = CalibrationResult(
            sample_count=3,
            nearest_spacing=12.5,
            median_fwhm=4.2,
            peak_min=0.18,
            peak_max=0.54,
            detection_params=suggested_detection,
            intensity_params=suggested_intensity,
            match_distance=5.0,
        )

        def invoke_apply_button() -> None:
            def descendants(widget):
                for child in widget.winfo_children():
                    yield child
                    yield from descendants(child)

            button = next(
                widget
                for widget in descendants(self.root)
                if isinstance(widget, ttk.Button) and widget.cget("text") == "仅应用参数"
            )
            button.invoke()

        self.root.after(80, invoke_apply_button)
        action = self.app.show_detection_params(
            initial_detection=suggested_detection,
            initial_intensity=suggested_intensity,
            initial_match_distance=5.0,
            calibration_result=calibration,
        )
        self.assertEqual(action, "saved")
        self.assertEqual(self.app.detection_params, suggested_detection)
        self.assertEqual(self.app.intensity_params, suggested_intensity)
        self.assertEqual(self.app.match_distance, 5.0)

    def test_finish_calibration_opens_parameter_dialog_before_detection(self) -> None:
        suggested_detection = DetectionParams(0.6, 9.0, 5, 0.35, True, "com")
        suggested_intensity = IntensityParams(2.6, 3.5, 4.8, True)
        calibration = CalibrationResult(
            sample_count=2,
            nearest_spacing=12.0,
            median_fwhm=3.8,
            peak_min=0.2,
            peak_max=0.4,
            detection_params=suggested_detection,
            intensity_params=suggested_intensity,
            match_distance=4.5,
        )
        self.app.current_mode = "calibrate"
        self.app._calibration_points = [(16.0, 16.0), (48.0, 16.0)]
        with patch("atomic_app.calibrate_from_points", return_value=calibration), \
                patch.object(self.app, "show_detection_params", return_value="detect") as show_dialog, \
                patch.object(self.app, "start_detection") as start_detection:
            self.app.finish_calibration()

        show_dialog.assert_called_once_with(
            initial_detection=suggested_detection,
            initial_intensity=suggested_intensity,
            initial_match_distance=4.5,
            calibration_result=calibration,
        )
        start_detection.assert_called_once_with(False)
        self.assertEqual(self.app.current_mode, "add")
        self.assertEqual(self.app._calibration_points, [])

    def test_complete_parameter_validation_rejects_invalid_intensity_radii(self) -> None:
        with self.assertRaises(ValueError):
            self.app._validated_parameter_bundle(
                sigma="0.8",
                min_distance="8",
                window="5",
                threshold="",
                bright=True,
                method="com",
                aperture_radius="3.0",
                background_inner_radius="2.0",
                background_outer_radius="4.0",
                match_distance="4.0",
            )


class IntensitySourceOfTruthTests(unittest.TestCase):
    """强度参数以右侧卡片输入框为唯一真相源：手动增删前必须解析当前输入值。"""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.root, cls.app = get_tk_session()
        except tk.TclError as error:
            raise unittest.SkipTest(f"Tk 不可用：{error}") from error

    def setUp(self) -> None:
        self.app.stack = _gaussian_stack()
        self.app._reset_frame_state(int(self.app.stack.shape[0]))
        self.app._busy = False

    def test_add_point_uses_current_intensity_fields(self) -> None:
        self.app.aperture_var.set("4.0")
        self.app.bg_inner_var.set("5.0")
        self.app.bg_outer_var.set("6.5")
        with patch("atomic_app.refine_point_on_work", return_value=(16.0, 16.0)):
            self.app._add_point(16.0, 16.0)
        records = self.app.frame_records[0]
        self.assertEqual(len(records), 1)
        expected = measure_integrated_intensities(
            self.app.stack[0], [(16.0, 16.0)], IntensityParams(4.0, 5.0, 6.5, True)
        )[0]
        self.assertAlmostEqual(records[0].integrated_intensity, expected, places=6)
        self.assertEqual(self.app.intensity_params.aperture_radius, 4.0)

    def test_add_point_aborts_when_intensity_fields_invalid(self) -> None:
        self.app.aperture_var.set("abc")
        with patch("atomic_app.refine_point_on_work") as refine, \
                patch("atomic_app.messagebox.showerror") as showerror:
            self.app._add_point(16.0, 16.0)
        refine.assert_not_called()  # 参数解析必须发生在亚像素精炼之前
        showerror.assert_called_once()
        self.assertEqual(len(self.app.frame_records[0]), 0)

    def test_delete_nearest_uses_current_intensity_fields(self) -> None:
        self.app.aperture_var.set("4.0")
        self.app.bg_inner_var.set("5.0")
        self.app.bg_outer_var.set("6.5")
        self.app.frame_records[0] = [AtomRecord(1, 16.0, 16.0)]
        with patch("atomic_app.messagebox.showerror"):
            self.app._delete_nearest(16.5, 16.0)
        self.assertEqual(len(self.app.frame_records[0]), 0)
        self.assertEqual(self.app.intensity_params.aperture_radius, 4.0)


class ShortcutGuardTests(unittest.TestCase):
    """快捷键守卫：焦点判定逻辑与绑定接线分开测，避免依赖窗口管理器的真实焦点。"""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.root, cls.app = get_tk_session()
        except tk.TclError as error:
            raise unittest.SkipTest(f"Tk 不可用：{error}") from error

    def setUp(self) -> None:
        self.app.stack = _gaussian_stack()
        self.app._reset_frame_state(int(self.app.stack.shape[0]))
        self.app._busy = False

    def test_typing_focused_detects_text_widgets(self) -> None:
        entry = ttk.Entry(self.app.right_inner)
        canvas_widget = self.app.canvas.get_tk_widget()
        with patch.object(self.app.root, "focus_get", return_value=entry):
            self.assertTrue(self.app._typing_focused())
        with patch.object(self.app.root, "focus_get", return_value=canvas_widget):
            self.assertFalse(self.app._typing_focused())
        with patch.object(self.app.root, "focus_get", return_value=None):
            self.assertFalse(self.app._typing_focused())
        entry.destroy()

    def test_c_key_does_not_start_calibration_while_typing(self) -> None:
        with patch.object(self.app, "_typing_focused", return_value=True), \
                patch.object(self.app, "start_calibration") as start_calibration:
            self.app._shortcut_calibrate()
        start_calibration.assert_not_called()
        self.assertEqual(self.app.current_mode, "add")

    def test_c_key_starts_calibration_when_not_typing(self) -> None:
        with patch.object(self.app, "_typing_focused", return_value=False), \
                patch.object(self.app, "start_calibration") as start_calibration:
            self.app._shortcut_calibrate()
        start_calibration.assert_called_once_with()

    def test_ctrl_z_does_not_revert_atoms_while_typing(self) -> None:
        self.app.frame_records[0] = [AtomRecord(1, 5.0, 5.0)]
        self.app._save_undo_snapshot()
        with patch.object(self.app, "_typing_focused", return_value=True):
            self.app._shortcut_undo()
        self.assertEqual(len(self.app.frame_records[0]), 1, "输入框聚焦时 Ctrl+Z 不得回滚原子数据")
        self.assertEqual(len(self.app.undo_stack), 1)

    def test_ctrl_z_reverts_atoms_when_not_typing(self) -> None:
        self.app._save_undo_snapshot()  # 添加前的空状态快照
        self.app.frame_records[0] = [AtomRecord(1, 5.0, 5.0)]
        with patch.object(self.app, "_typing_focused", return_value=False):
            self.app._shortcut_undo()
        self.assertEqual(len(self.app.frame_records[0]), 0)
        self.assertEqual(len(self.app.undo_stack), 0)

    def test_mode_shortcuts_ignored_while_typing(self) -> None:
        self.app.mode_var.set("delete")
        with patch.object(self.app, "_typing_focused", return_value=True):
            self.app._shortcut_mode("add")
        self.assertEqual(self.app.mode_var.get(), "delete")
        self.assertEqual(self.app.current_mode, "add")

    def test_mode_shortcut_applies_when_not_typing(self) -> None:
        with patch.object(self.app, "_typing_focused", return_value=False):
            self.app._shortcut_mode("delete")
        self.assertEqual(self.app.current_mode, "delete")

    def test_shortcut_sequences_are_bound(self) -> None:
        for sequence in ("<KeyPress-a>", "<KeyPress-d>", "<KeyPress-c>", "<Control-z>", "<Escape>", "<Return>"):
            self.assertTrue(self.app.root.bind(sequence), f"{sequence} 应绑定到 root")
            self.assertIn(sequence, self.app._shortcut_handlers)

    def test_bound_handler_respects_guard(self) -> None:
        with patch.object(self.app, "_typing_focused", return_value=True), \
                patch.object(self.app, "start_calibration") as start_calibration:
            self.app._shortcut_handlers["<KeyPress-c>"](None)
        start_calibration.assert_not_called()

        with patch.object(self.app, "_typing_focused", return_value=False), \
                patch.object(self.app, "start_calibration") as start_calibration:
            self.app._shortcut_handlers["<KeyPress-c>"](None)
        start_calibration.assert_called_once_with()

        with patch.object(self.app, "_typing_focused", return_value=True), \
                patch.object(self.app, "undo") as undo_mock:
            self.app._shortcut_handlers["<Control-z>"](None)
        undo_mock.assert_not_called()


class SessionRoundtripTests(unittest.TestCase):
    """会话保存/加载：人工校对成果必须可以完整往返。"""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.root, cls.app = get_tk_session()
        except tk.TclError as error:
            raise unittest.SkipTest(f"Tk 不可用：{error}") from error

    def setUp(self) -> None:
        import tifffile

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.tiff_path = Path(self.directory.name) / "stack.tif"
        tifffile.imwrite(
            self.tiff_path,
            _gaussian_stack(),
            metadata={"axes": "TYX"},
            photometric="minisblack",
        )
        loaded, info = load_tiff_stack(self.tiff_path)
        self.app._install_stack(loaded, info)
        self.app._busy = False

    def tearDown(self) -> None:
        # 换回内存数组，释放对临时 TIFF 的内存映射，
        # 否则 Windows 下句柄未关闭导致临时目录清理失败。
        memory_stack = np.zeros((1, 8, 8), dtype=np.float32)
        memory_info = TiffStackInfo(
            path="<memory>",
            axes="TYX",
            source_shape=(1, 8, 8),
            stack_shape=(1, 8, 8),
            dtype="float32",
            memory_mapped=False,
            source=None,
        )
        self.app._install_stack(memory_stack, memory_info)

    def _populate(self) -> None:
        app = self.app
        app.detection_params = DetectionParams(0.7, 9.0, 5, 0.55, True, "com")
        app.intensity_params = IntensityParams(2.6, 3.4, 4.6, True)
        app.match_distance = 4.2
        app.rules = [
            MarkerRule("弱强度", 5.0, 25.0, "#ff9800", "diamond", True),
            MarkerRule("全部", 0.0, 100.0, "#00ff66", "circle", True),
        ]
        app.frame_regions[0] = DetectionRegion("rectangle", ((2.0, 2.0), (30.0, 60.0)))
        app.frame_intensity_params[0] = IntensityParams(2.6, 3.4, 4.6, True)
        app.frame_processed[0] = True
        app.frame_edited[0] = True
        app.frame_records[0] = [
            AtomRecord(1, 16.0, 16.0, 12.5, 40.0, source="manual"),
            AtomRecord(2, 48.0, 16.0, 20.0, 90.0),
        ]

    def test_save_and_reload_session_restores_full_state(self) -> None:
        app = self.app
        self._populate()
        session_path = str(Path(self.directory.name) / "s.session.json")
        with patch("atomic_app.filedialog.asksaveasfilename", return_value=session_path):
            self.assertTrue(app.save_session())
        data = json.loads(Path(session_path).read_text(encoding="utf-8"))
        self.assertEqual(data["format"], "atomic-recognition-session-1")
        self.assertEqual(data["source_tiff"], str(self.tiff_path))
        self.assertFalse(Path(str(session_path) + ".part").exists())

        with patch("atomic_app.messagebox.askyesno", return_value=True), \
                patch("atomic_app.filedialog.askopenfilename", return_value=session_path):
            app.open_session()

        self.assertEqual(app.detection_params, DetectionParams(0.7, 9.0, 5, 0.55, True, "com"))
        self.assertEqual(app.intensity_params, IntensityParams(2.6, 3.4, 4.6, True))
        self.assertEqual(app.match_distance, 4.2)
        self.assertEqual([rule.name for rule in app.rules], ["弱强度", "全部"])
        self.assertEqual(app.rules[0].marker, "diamond")
        self.assertEqual(app.frame_regions[0], DetectionRegion("rectangle", ((2.0, 2.0), (30.0, 60.0))))
        self.assertTrue(app.frame_edited[0])
        self.assertTrue(app.frame_processed[0])
        self.assertEqual([record.atom_id for record in app.frame_records[0]], [1, 2])
        self.assertEqual(app.frame_records[0][0].source, "manual")
        self.assertAlmostEqual(app.frame_records[0][0].integrated_intensity, 12.5)
        self.assertEqual(app.frame_intensity_params[0], IntensityParams(2.6, 3.4, 4.6, True))

    def test_session_with_wrong_stack_shape_is_rejected(self) -> None:
        app = self.app
        self._populate()
        session_path = Path(self.directory.name) / "bad.session.json"
        with patch("atomic_app.filedialog.asksaveasfilename", return_value=str(session_path)):
            app.save_session()
        payload = json.loads(session_path.read_text(encoding="utf-8"))
        payload["stack_shape"] = [7, 7, 7]
        session_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        with patch("atomic_app.messagebox.askyesno", return_value=True), \
                patch("atomic_app.filedialog.askopenfilename", return_value=str(session_path)), \
                patch("atomic_app.messagebox.showerror") as showerror:
            app.open_session()
        showerror.assert_called_once()
        # 原状态不被破坏
        self.assertEqual([record.atom_id for record in app.frame_records[0]], [1, 2])


class UndoSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.root, cls.app = get_tk_session()
        except tk.TclError as error:
            raise unittest.SkipTest(f"Tk 不可用：{error}") from error

    def setUp(self) -> None:
        self.app.stack = _gaussian_stack()
        self.app._reset_frame_state(int(self.app.stack.shape[0]))
        self.app._busy = False

    def test_undo_restores_edited_flag(self) -> None:
        self.assertFalse(self.app.frame_edited[0])
        self.app.frame_records[0] = [AtomRecord(1, 5.0, 5.0)]
        self.app._save_undo_snapshot()
        self.app.frame_edited[0] = True
        self.app.frame_records[0] = [AtomRecord(1, 5.0, 5.0), AtomRecord(2, 8.0, 6.0)]
        self.app.undo()
        self.assertFalse(self.app.frame_edited[0], "撤销必须一并回滚 frame_edited 溯源标志")
        self.assertEqual(len(self.app.frame_records[0]), 1)

    def test_single_frame_detection_keeps_other_frame_undo_history(self) -> None:
        self.app.current_frame = 1
        self.app.frame_records[1] = [AtomRecord(7, 20.0, 20.0)]
        self.app._save_undo_snapshot()
        self.assertEqual(len(self.app.undo_stack), 1)
        self.app.current_frame = 0
        self.app.frame_scale.set(0)
        self.app.detection_params = DetectionParams(0.8, 10.0, 5, 0.6, True, "com")
        self.app.start_detection(all_frames=False)
        deadline = time.monotonic() + 8.0
        while self.app._busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.root.update()
        self.assertFalse(self.app._busy, "后台识别未在测试时限内完成")
        self.assertEqual(len(self.app.undo_stack), 1, "单帧识别不得清空其他帧的撤销历史")
        self.assertEqual(self.app.undo_stack[0][0], 1)

    def test_metadata_flags_inconsistent_intensity_params(self) -> None:
        self.app.frame_records[0] = [AtomRecord(1, 5.0, 5.0)]
        self.app.frame_intensity_params[0] = IntensityParams(2.0, 3.0, 4.0, True)
        metadata = self.app._build_export_metadata()
        self.assertTrue(metadata["intensity_params_consistent_across_frames"])
        # 帧 1 参数不同：溯源不一致必须被标记。
        self.app.frame_records[1] = [AtomRecord(2, 8.0, 8.0)]
        self.app.frame_intensity_params[1] = IntensityParams(3.0, 4.0, 5.0, True)
        metadata = self.app._build_export_metadata()
        self.assertFalse(metadata["intensity_params_consistent_across_frames"])


class ExportPrecheckTests(unittest.TestCase):
    """导出前的写权限与磁盘空间预检（无需 Tk）。"""

    def test_missing_permission_raises_friendly_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "out.tif")
            with patch("atomic_app.os.access", return_value=False):
                with self.assertRaises(AtomicToolError) as context:
                    AtomicRecognitionApp._check_writable_target(path)
            self.assertIn("写入权限", str(context.exception))

    def test_insufficient_disk_space_is_rejected(self) -> None:
        class FakeUsage:
            free = 100

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "out.tif")
            with patch("atomic_app.shutil.disk_usage", return_value=FakeUsage()):
                with self.assertRaises(AtomicToolError) as context:
                    AtomicRecognitionApp._check_writable_target(path, estimated_bytes=1000)
            self.assertIn("磁盘空间不足", str(context.exception))


class CsvSafetyTests(unittest.TestCase):
    """CSV 公式注入防护与强度参数指纹（无需 Tk）。"""

    def test_formula_prefixes_are_escaped(self) -> None:
        for text in ("=cmd()", "+1", "-2", "@x", "\tTAB"):
            self.assertTrue(_csv_safe_text(text).startswith("'"), text)
        self.assertEqual(_csv_safe_text("正常文本"), "正常文本")

    def test_intensity_fingerprint(self) -> None:
        self.assertEqual(_intensity_fingerprint(None), "未记录")
        self.assertEqual(
            _intensity_fingerprint(IntensityParams(2.5, 3.0, 4.0, True)),
            "aperture=2.5;bg_inner=3;bg_outer=4;bright=1",
        )


if __name__ == "__main__":
    unittest.main()
