from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest import mock

import numpy as np
import tifffile

from hrtem_filter.params import (
    FilterParams,
    OutputEncoding,
    ParameterError,
    SaveOptions,
)
from hrtem_filter.pipeline import (
    InputFileChangedError,
    ProcessingCancelled,
    StackProcessor,
)
from hrtem_filter.tiff_io import (
    AtomicTiffWriter,
    TiffFormatError,
    TiffMetadata,
    UnsafeOutputPathError,
    inspect_tiff,
)


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.source = self.directory / "source.tif"
        data = (np.arange(3 * 32 * 64, dtype=np.uint16).reshape(3, 32, 64) % 4096)
        tifffile.imwrite(
            self.source,
            data,
            photometric="minisblack",
            resolution=(2.5, 2.5),
            resolutionunit="CENTIMETER",
            description="source calibration",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_float_output_preserves_stack_shape_and_calibration(self) -> None:
        output = self.directory / "filtered.tif"
        report = StackProcessor().process_tiff(self.source, output)
        self.assertEqual(report.frame_count, 3)
        self.assertTrue(report.provenance_path.is_file())
        provenance = json.loads(report.provenance_path.read_text(encoding="utf-8"))
        self.assertEqual(provenance["source_tiff_metadata"]["description"], "source calibration")
        self.assertEqual(provenance["source_tiff_metadata"]["resolution"], [2.5, 2.5])
        with tifffile.TiffFile(output) as handle:
            self.assertEqual(len(handle.pages), 3)
            self.assertEqual(handle.pages[0].shape, (32, 64))
            self.assertEqual(handle.pages[0].dtype, np.dtype("float32"))
            self.assertEqual(handle.pages[0].tags["XResolution"].value, (5, 2))

    def test_uint8_export_uses_one_stack_wide_range(self) -> None:
        output = self.directory / "display.tif"
        report = StackProcessor().process_tiff(
            self.source, output, save_options=SaveOptions(OutputEncoding.UINT8_DISPLAY)
        )
        self.assertIsNotNone(report.display_range)
        with tifffile.TiffFile(output) as handle:
            self.assertEqual(handle.pages[0].dtype, np.dtype("uint8"))

    def test_source_dtype_clip_export_preserves_integer_type(self) -> None:
        output = self.directory / "clip.tif"
        report = StackProcessor().process_tiff(
            self.source, output, save_options=SaveOptions(OutputEncoding.SOURCE_DTYPE_CLIP)
        )
        self.assertEqual(report.output_dtype, np.dtype("uint16"))
        with tifffile.TiffFile(output) as handle:
            self.assertEqual(handle.pages[0].dtype, np.dtype("uint16"))

    def test_display_range_requires_uint8_encoding(self) -> None:
        with self.assertRaises(ParameterError):
            SaveOptions(OutputEncoding.FLOAT32, display_range=(0.0, 1.0)).validated()

    def test_same_input_output_is_rejected(self) -> None:
        with self.assertRaises(UnsafeOutputPathError):
            StackProcessor().process_tiff(self.source, self.source)

    def test_cancel_never_publishes_partial_target(self) -> None:
        output = self.directory / "cancelled.tif"
        cancel = Event()

        def progress(phase: str, current: int, total: int) -> None:
            if phase == "write" and current == 1:
                cancel.set()

        with self.assertRaises(ProcessingCancelled):
            StackProcessor().process_tiff(self.source, output, cancel_event=cancel, progress=progress)
        self.assertFalse(output.exists())
        self.assertFalse(os.path.exists(str(output) + ".processing.json"))

    def test_provenance_failure_never_publishes_output(self) -> None:
        """Regression: a provenance write failure must not leave a TIFF without its record."""
        output = self.directory / "provenance_fail.tif"
        with (
            mock.patch(
                "hrtem_filter.pipeline.write_json_atomically", side_effect=OSError("disk full")
            ),
            self.assertRaises(OSError),
        ):
            StackProcessor().process_tiff(self.source, output)
        self.assertFalse(output.exists())
        self.assertFalse((self.directory / "provenance_fail.tif.processing.json").exists())

    def test_sample_axis_is_not_misread_as_a_stack(self) -> None:
        colour_like = self.directory / "samples.tif"
        tifffile.imwrite(colour_like, np.zeros((3, 16, 16), dtype=np.uint8), photometric="rgb")
        with self.assertRaises(TiffFormatError):
            inspect_tiff(colour_like)

    def test_input_file_change_before_write_aborts(self) -> None:
        """写阶段前必须复核输入文件 stat；不一致则中止且不发布输出。"""
        output = self.directory / "changed.tif"

        def progress(phase: str, current: int, total: int) -> None:
            if phase == "scan" and current == 1:
                with open(self.source, "ab") as handle:
                    handle.write(b"\0")

        with self.assertRaises(InputFileChangedError):
            StackProcessor().process_tiff(
                self.source,
                output,
                save_options=SaveOptions(OutputEncoding.UINT8_DISPLAY),
                progress=progress,
            )
        self.assertFalse(output.exists())
        self.assertFalse(os.path.exists(str(output) + ".processing.json"))

    def test_input_change_during_write_aborts(self) -> None:
        """写循环期间输入被修改也必须中止；已有的 TIFF 与 provenance 均保持原样。"""
        output = self.directory / "during_write.tif"
        StackProcessor().process_tiff(self.source, output)
        provenance_path = output.with_suffix(output.suffix + ".processing.json")
        tiff_bytes = output.read_bytes()
        json_bytes = provenance_path.read_bytes()

        def progress(phase: str, current: int, total: int) -> None:
            if phase == "write" and current == 2:
                with open(self.source, "ab") as handle:
                    handle.write(b"\0")

        with self.assertRaises(InputFileChangedError):
            StackProcessor().process_tiff(self.source, output, progress=progress)
        self.assertEqual(output.read_bytes(), tiff_bytes)
        self.assertEqual(provenance_path.read_bytes(), json_bytes)

    def test_failed_rerun_restores_previous_provenance(self) -> None:
        """Regression: commit 失败的重跑必须恢复旧 provenance，不能让旧 TIFF 配上新记录。"""
        output = self.directory / "rerun.tif"
        StackProcessor().process_tiff(self.source, output)
        provenance_path = output.with_suffix(output.suffix + ".processing.json")
        first_json = provenance_path.read_bytes()
        first_tiff = output.read_bytes()
        with (
            mock.patch(
                "hrtem_filter.tiff_io.AtomicTiffWriter.commit", side_effect=OSError("replace failed")
            ),
            self.assertRaises(OSError),
        ):
            StackProcessor().process_tiff(self.source, output, params=FilterParams(bw_ro=0.10))
        self.assertEqual(provenance_path.read_bytes(), first_json)
        self.assertEqual(output.read_bytes(), first_tiff)

    def test_json_write_failure_keeps_previous_provenance(self) -> None:
        """JSON 写入失败时旧 provenance 必须原样保留（写 JSON 本身是原子替换）。"""
        output = self.directory / "jsonfail.tif"
        StackProcessor().process_tiff(self.source, output)
        provenance_path = output.with_suffix(output.suffix + ".processing.json")
        first_json = provenance_path.read_bytes()
        with (
            mock.patch(
                "hrtem_filter.pipeline.write_json_atomically", side_effect=OSError("disk full")
            ),
            self.assertRaises(OSError),
        ):
            StackProcessor().process_tiff(self.source, output, params=FilterParams(bw_ro=0.10))
        self.assertEqual(provenance_path.read_bytes(), first_json)

    def test_palette_tiff_is_rejected(self) -> None:
        """Regression: palette（索引色）页面必须被拒绝，不能当作灰度索引滤波。"""
        palette = self.directory / "palette.tif"
        tifffile.imwrite(
            palette, np.arange(256, dtype=np.uint8).reshape(16, 16), photometric="palette"
        )
        with self.assertRaises(TiffFormatError):
            inspect_tiff(palette)

    def test_miniswhite_grayscale_is_accepted(self) -> None:
        """photometric=0（WhiteIsZero）同样是合法灰度，不能被误拒。"""
        grey = self.directory / "miniswhite.tif"
        tifffile.imwrite(grey, np.zeros((16, 16), dtype=np.uint8), photometric="miniswhite")
        info = inspect_tiff(grey)
        self.assertEqual(info.frame_shape, (16, 16))

    def test_atomic_commit_failure_cleans_temporary_file(self) -> None:
        """os.replace 失败时 commit 必须清理临时 TIFF 并保留原始异常。"""
        output = self.directory / "commit_fail.tif"
        writer = AtomicTiffWriter(
            output,
            frame_shape=(8, 8),
            expected_frames=1,
            metadata=TiffMetadata(None, None, None, None),
            provenance_hint="hint",
            output_dtype=np.dtype(np.float32),
        )
        writer.write(np.zeros((8, 8), dtype=np.float32))
        temp_path = writer.temporary_path
        self.assertTrue(temp_path.exists())
        with (
            mock.patch("hrtem_filter.tiff_io.os.replace", side_effect=OSError("mock replace failure")),
            self.assertRaises(OSError),
        ):
            writer.commit()
        self.assertFalse(temp_path.exists())
        self.assertFalse(output.exists())

    def test_preview_resets_save_cancel_event(self) -> None:
        """保存后进入预览时，cancel 不能复用保存的取消事件误报“正在取消”。

        GUI 模块导入需要 Tkinter；这里直接测试 gui.py 使用的纯状态辅助函数，
        保持与生产代码路径一致，同时避免在无 Tk 的 CI 环境中实例化 GUI。
        """
        from threading import Event

        from hrtem_filter.cancel_state import (
            cancel_status_text,
            reset_cancel_event_for_preview,
        )

        stale_event = Event()
        stale_event.set()
        # preview() 开始时会调用该函数，因此保存流程遗留的 event 必须被清掉。
        self.assertIsNone(reset_cancel_event_for_preview(stale_event))
        self.assertIsNone(reset_cancel_event_for_preview(None))
        # cancel() 根据当前 event 是否存在选择提示语。
        self.assertEqual(
            cancel_status_text(None),
            "预览无法中断，正在等待当前 FFT 完成…",
        )
        self.assertEqual(
            cancel_status_text(Event()),
            "正在取消；当前帧完成后将删除临时输出…",
        )

    def test_resolution_is_written_to_every_page(self) -> None:
        """分辨率是逐页标签：只写第一页会让逐页读取标定的工具取不到后续页。"""
        output = self.directory / "allpages.tif"
        StackProcessor().process_tiff(self.source, output)
        with tifffile.TiffFile(output) as handle:
            self.assertEqual(len(handle.pages), 3)
            for page in handle.pages:
                self.assertEqual(page.tags["XResolution"].value, (5, 2))

    def test_provenance_records_entry_workers_and_optional_hash(self) -> None:
        """provenance 必须记录入口、FFT 线程数与平台；可选 sha256 与文件一致。"""
        output = self.directory / "meta.tif"
        report = StackProcessor().process_tiff(
            self.source, output, entry="cli-test", input_hash=True
        )
        provenance = json.loads(report.provenance_path.read_text(encoding="utf-8"))
        self.assertEqual(provenance["entry"], "cli-test")
        self.assertEqual(provenance["runtime"]["fft_workers"], 1)
        self.assertIn("platform", provenance["runtime"])
        self.assertIn("python", provenance["runtime"])
        digest = provenance["input"]["integrity"]["sha256"]
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(digest, hashlib.sha256(self.source.read_bytes()).hexdigest())

    def test_provenance_omits_hash_unless_requested(self) -> None:
        output = self.directory / "nohash.tif"
        report = StackProcessor().process_tiff(self.source, output)
        provenance = json.loads(report.provenance_path.read_text(encoding="utf-8"))
        self.assertNotIn("sha256", provenance["input"]["integrity"])

    def test_unwritable_output_fails_before_display_scan(self) -> None:
        """目标目录不可写时必须在 uint8 预扫描之前失败，不能白算数分钟。"""
        blocker = self.directory / "blocker.tif"
        blocker.write_bytes(b"not a directory")
        output = blocker / "out.tif"
        progress = mock.Mock()
        with self.assertRaises(OSError):
            StackProcessor().process_tiff(
                self.source,
                output,
                save_options=SaveOptions(OutputEncoding.UINT8_DISPLAY),
                progress=progress,
            )
        progress.assert_not_called()


if __name__ == "__main__":
    unittest.main()
