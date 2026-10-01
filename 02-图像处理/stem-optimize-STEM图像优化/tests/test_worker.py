import hashlib
import json
import logging
import queue
import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import tifffile

from contrast import DEFAULT_HIGH_PERCENTILE, DEFAULT_LOW_PERCENTILE
from errors import OperationCancelled
from tiff_handler import TiffStackReader, TiffStackWriter
from worker import ProcessingWorker, stale_temporary_files


def _params(**overrides):
    values = {
        "k_factor": 1.2,
        "blend_ratio": 0.5,
        "delta": 3.0,
        "clahe_clip": 1.2,
        "output_bit_depth": 16,
        "apply_clahe": False,
    }
    values.update(overrides)
    return values


def _run_worker(input_path, output_path, params=None, cancel=False):
    messages = queue.Queue()
    event = threading.Event()
    if cancel:
        event.set()
    worker = ProcessingWorker(
        str(input_path),
        str(output_path),
        params or _params(),
        messages,
        event,
    )
    worker.run()
    result = []
    while not messages.empty():
        result.append(messages.get())
    return result


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self._previous_logging_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        self.temp = tempfile.TemporaryDirectory(prefix="stem_v21_tests_")
        self.directory = Path(self.temp.name)
        self.stack = np.stack(
            [
                np.arange(32 * 48, dtype=np.uint16).reshape(32, 48),
                np.arange(32 * 48, dtype=np.uint16).reshape(32, 48) + 1000,
                np.arange(32 * 48, dtype=np.uint16).reshape(32, 48) + 2000,
            ]
        )
        self.input_path = self.directory / "input.tif"
        tifffile.imwrite(
            self.input_path,
            self.stack,
            photometric="minisblack",
            resolution=(2.5, 3.5),
            resolutionunit="CENTIMETER",
        )

    def tearDown(self):
        self.temp.cleanup()
        logging.disable(self._previous_logging_disable)

    def test_writer_reopen_resets_frame_count(self):
        output = self.directory / "reused_writer.tif"
        frame_a = np.full((8, 8), 7, dtype=np.uint16)
        frame_b = np.full((8, 8), 9, dtype=np.uint16)
        writer = TiffStackWriter(
            output,
            stack_shape=(2, 8, 8),
            dtype=np.uint16,
        )
        writer.open()
        writer.write_stack([frame_a, frame_b])
        writer.close()
        self.assertEqual(writer.frame_count, 2)

        # 复用同一 writer 再次打开同一路径；若不重置 _frame_count，
        # write_stack 会从上一轮的 2 开始计数并在结束时误报帧数错误。
        writer.open()
        writer.write_stack([frame_a, frame_b])
        writer.close()
        self.assertEqual(writer.frame_count, 2)
        with tifffile.TiffFile(output) as tif:
            self.assertEqual(len(tif.pages), 2)
            self.assertEqual(tif.pages[0].shape, (8, 8))

    def test_same_input_and_output_is_rejected_without_modification(self):
        before = hashlib.sha256(self.input_path.read_bytes()).hexdigest()
        messages = _run_worker(self.input_path, self.input_path)
        after = hashlib.sha256(self.input_path.read_bytes()).hexdigest()
        self.assertEqual(before, after)
        self.assertIn("error", [message[0] for message in messages])

    def test_cancel_does_not_touch_existing_output(self):
        output = self.directory / "existing.tif"
        tifffile.imwrite(output, np.full((8, 8), 123, dtype=np.uint16))
        before = hashlib.sha256(output.read_bytes()).hexdigest()
        messages = _run_worker(self.input_path, output, cancel=True)
        after = hashlib.sha256(output.read_bytes()).hexdigest()
        self.assertEqual(before, after)
        self.assertIn("cancelled", [message[0] for message in messages])

    def test_mid_processing_cancel_does_not_publish_partial_output(self):
        output = self.directory / "existing.tif"
        tifffile.imwrite(output, np.full((8, 8), 123, dtype=np.uint16))
        before = hashlib.sha256(output.read_bytes()).hexdigest()
        messages = queue.Queue()
        event = threading.Event()

        class CancellingQueue:
            def put(self, message):
                messages.put(message)
                if message[0] == "progress" and message[2] == 1:
                    event.set()

        worker = ProcessingWorker(
            str(self.input_path),
            str(output),
            _params(),
            CancellingQueue(),
            event,
        )
        worker.run()
        after = hashlib.sha256(output.read_bytes()).hexdigest()
        kinds = []
        while not messages.empty():
            kinds.append(messages.get()[0])
        self.assertEqual(before, after)
        self.assertIn("cancelled", kinds)
        self.assertEqual(list(self.directory.glob(".*.partial*")), [])

    def test_invalid_parameters_do_not_replace_existing_output(self):
        output = self.directory / "existing.tif"
        tifffile.imwrite(output, np.full((8, 8), 123, dtype=np.uint16))
        before = hashlib.sha256(output.read_bytes()).hexdigest()
        messages = _run_worker(self.input_path, output, _params(k_factor=None))
        after = hashlib.sha256(output.read_bytes()).hexdigest()
        self.assertEqual(before, after)
        self.assertIn("error", [message[0] for message in messages])

    def test_low_disk_space_is_rejected_before_temporary_output(self):
        output = self.directory / "existing.tif"
        tifffile.imwrite(output, np.full((8, 8), 123, dtype=np.uint16))
        before = hashlib.sha256(output.read_bytes()).hexdigest()
        with mock.patch(
            "worker.shutil.disk_usage",
            return_value=mock.Mock(free=1),
        ):
            messages = _run_worker(self.input_path, output)
        self.assertIn("error", [message[0] for message in messages])
        self.assertEqual(before, hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertEqual(list(self.directory.glob(".*.partial*")), [])

    def test_frame_failure_aborts_transaction_without_zero_fallback(self):
        output = self.directory / "existing.tif"
        sidecar = Path(str(output) + ".stem.json")
        tifffile.imwrite(output, np.full((8, 8), 123, dtype=np.uint16))
        sidecar.write_text('{"old": true}\n', encoding="utf-8")
        before_image = hashlib.sha256(output.read_bytes()).hexdigest()
        before_sidecar = hashlib.sha256(sidecar.read_bytes()).hexdigest()

        import worker as worker_module

        real_process = worker_module.process_frame
        call_count = 0

        def fail_second(frame, params, global_range):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("synthetic frame failure")
            return real_process(frame, params, global_range)

        with mock.patch("worker.process_frame", side_effect=fail_second):
            messages = _run_worker(self.input_path, output)

        self.assertIn("error", [message[0] for message in messages])
        self.assertEqual(before_image, hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertEqual(
            before_sidecar, hashlib.sha256(sidecar.read_bytes()).hexdigest()
        )
        self.assertEqual(list(self.directory.glob(".*.partial*")), [])

    def test_input_hash_change_aborts_before_publication(self):
        output = self.directory / "existing.tif"
        tifffile.imwrite(output, np.full((8, 8), 321, dtype=np.uint16))
        before = hashlib.sha256(output.read_bytes()).hexdigest()

        import worker as worker_module

        real_sha256 = worker_module._sha256
        calls = 0

        def changed_hash(path, cancel_event=None):
            nonlocal calls
            calls += 1
            digest = real_sha256(path, cancel_event)
            return "0" * 64 if calls == 2 else digest

        with mock.patch("worker._sha256", side_effect=changed_hash):
            messages = _run_worker(self.input_path, output)

        self.assertIn("error", [message[0] for message in messages])
        self.assertEqual(before, hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertEqual(list(self.directory.glob(".*.partial*")), [])

    def test_sidecar_publication_failure_keeps_new_image_and_old_sidecar(self):
        output = self.directory / "existing.tif"
        sidecar = Path(str(output) + ".stem.json")
        tifffile.imwrite(output, np.full((8, 8), 456, dtype=np.uint16))
        sidecar.write_text('{"old": true}\n', encoding="utf-8")

        import worker as worker_module

        real_replace = worker_module.os.replace

        def fail_new_sidecar(source, destination):
            if str(source).endswith(".partial.json"):
                raise PermissionError("synthetic sidecar publication failure")
            return real_replace(source, destination)

        with mock.patch("worker.os.replace", side_effect=fail_new_sidecar):
            messages = _run_worker(self.input_path, output)

        kinds = [message[0] for message in messages]
        self.assertIn("error", kinds)
        self.assertNotIn("done", kinds)
        error_text = next(message[1] for message in messages if message[0] == "error")
        self.assertIn("sidecar", error_text)
        self.assertIn("已发布", error_text)
        # 原子直换：新 TIFF 已在目标路径且是有效的新输出（3 帧而非旧的单帧）。
        with tifffile.TiffFile(output) as tif:
            self.assertEqual(tif.series[0].shape, self.stack.shape)
            self.assertEqual(tif.series[0].dtype, np.dtype(np.uint16))
        # 失败的原子替换不触碰目标：旧 sidecar 内容原样保留。
        self.assertEqual(sidecar.read_text(encoding="utf-8"), '{"old": true}\n')
        self.assertEqual(list(self.directory.glob("*.backup")), [])
        self.assertEqual(list(self.directory.glob(".*.partial*")), [])

    def test_sidecar_publication_retry_succeeds_after_transient_failure(self):
        output = self.directory / "retry.tif"
        sidecar = Path(str(output) + ".stem.json")

        import worker as worker_module

        real_replace = worker_module.os.replace
        attempts = {"sidecar": 0}

        def fail_first_sidecar(source, destination):
            if str(source).endswith(".partial.json"):
                attempts["sidecar"] += 1
                if attempts["sidecar"] == 1:
                    raise PermissionError("synthetic transient failure")
            return real_replace(source, destination)

        with mock.patch("worker.os.replace", side_effect=fail_first_sidecar):
            messages = _run_worker(self.input_path, output)

        self.assertIn("done", [message[0] for message in messages])
        self.assertEqual(attempts["sidecar"], 2)
        self.assertTrue(sidecar.exists())
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        self.assertEqual(provenance["status"], "complete")

    def test_success_is_atomic_and_writes_provenance(self):
        output = self.directory / "result.tif"
        messages = _run_worker(self.input_path, output)
        self.assertIn("done", [message[0] for message in messages])
        with tifffile.TiffFile(output) as tif:
            self.assertEqual(len(tif.pages), 3)
            self.assertEqual(tif.pages[0].shape, (32, 48))
            self.assertEqual(tif.pages[0].dtype, np.dtype(np.uint16))
            self.assertFalse(tif.is_bigtiff)
            self.assertAlmostEqual(
                float(tif.pages[0].tags["XResolution"].value[0])
                / float(tif.pages[0].tags["XResolution"].value[1]),
                2.5,
            )
            self.assertEqual(
                tif.pages[0].tags["ResolutionUnit"].value.name, "CENTIMETER"
            )

        sidecar = Path(str(output) + ".stem.json")
        self.assertTrue(sidecar.exists())
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        self.assertEqual(provenance["output"]["frame_count"], 3)
        self.assertEqual(provenance["output"]["dtype"], "uint16")
        self.assertEqual(provenance["status"], "complete")
        self.assertEqual(provenance["input"]["name"], self.input_path.name)
        self.assertEqual(
            provenance["processing"]["global_range"]["low_percentile"],
            DEFAULT_LOW_PERCENTILE,
        )
        self.assertEqual(
            provenance["processing"]["global_range"]["high_percentile"],
            DEFAULT_HIGH_PERCENTILE,
        )
        self.assertNotIn("path", provenance["input"])
        self.assertNotIn("path", provenance["output"])
        self.assertEqual(
            provenance["input"]["calibration"]["resolution_unit"],
            "CENTIMETER",
        )
        self.assertEqual(
            provenance["input"]["sha256"],
            hashlib.sha256(self.input_path.read_bytes()).hexdigest(),
        )

    def test_ome_metadata_is_rebuilt_for_new_dtype(self):
        input_path = self.directory / "input.ome.tif"
        output_path = self.directory / "output.ome.tif"
        tifffile.imwrite(
            input_path,
            self.stack,
            ome=True,
            metadata={
                "axes": "TYX",
                "PhysicalSizeX": 0.2,
                "PhysicalSizeXUnit": "nm",
                "PhysicalSizeY": 0.3,
                "PhysicalSizeYUnit": "nm",
            },
        )
        messages = _run_worker(input_path, output_path, _params(output_bit_depth=8))
        self.assertIn("done", [message[0] for message in messages])
        with tifffile.TiffFile(output_path) as tif:
            self.assertTrue(tif.is_ome)
            self.assertEqual(tif.series[0].axes, "TYX")
            self.assertEqual(tif.series[0].dtype, np.dtype(np.uint8))
            self.assertIn('Type="uint8"', tif.ome_metadata)
            self.assertNotIn('Type="uint16"', tif.ome_metadata)
            self.assertIn('PhysicalSizeX="0.2"', tif.ome_metadata)

    def test_imagej_metadata_and_axes_are_preserved(self):
        input_path = self.directory / "input_imagej.tif"
        output_path = self.directory / "output_imagej.tif"
        tifffile.imwrite(
            input_path,
            self.stack,
            imagej=True,
            metadata={"axes": "TYX", "unit": "nm", "finterval": 0.25},
        )
        messages = _run_worker(input_path, output_path)
        self.assertIn("done", [message[0] for message in messages])
        with tifffile.TiffFile(output_path) as tif:
            self.assertTrue(tif.is_imagej)
            self.assertEqual(tif.series[0].shape, self.stack.shape)
            self.assertEqual(tif.series[0].axes, "TYX")
            self.assertEqual(tif.imagej_metadata["unit"], "nm")
            self.assertAlmostEqual(tif.imagej_metadata["finterval"], 0.25)

    def test_lzw_compressed_input_is_supported(self):
        input_path = self.directory / "compressed_lzw.tif"
        output_path = self.directory / "compressed_out.tif"
        tifffile.imwrite(
            input_path,
            self.stack,
            photometric="minisblack",
            compression="lzw",
        )
        messages = _run_worker(input_path, output_path)
        self.assertIn("done", [message[0] for message in messages])
        with tifffile.TiffFile(output_path) as tif:
            self.assertEqual(tif.series[0].shape, self.stack.shape)

    def test_generic_separate_page_stack_is_consolidated_safely(self):
        input_path = self.directory / "separate_pages.tif"
        output_path = self.directory / "separate_pages_out.tif"
        with tifffile.TiffWriter(input_path) as writer:
            for frame in self.stack:
                writer.write(frame, photometric="minisblack", contiguous=False)
        with TiffStackReader(input_path) as reader:
            self.assertEqual(reader.num_frames, 3)
            self.assertEqual(reader.stack_shape, self.stack.shape)
        messages = _run_worker(input_path, output_path)
        self.assertIn("done", [message[0] for message in messages])
        with tifffile.TiffFile(output_path) as tif:
            self.assertEqual(len(tif.series), 1)
            self.assertEqual(tif.series[0].shape, self.stack.shape)

    def test_four_dimensional_ome_axes_are_preserved(self):
        input_path = self.directory / "four_d.ome.tif"
        output_path = self.directory / "four_d_out.ome.tif"
        data = np.arange(2 * 2 * 16 * 16, dtype=np.uint16).reshape(2, 2, 16, 16)
        tifffile.imwrite(
            input_path,
            data,
            ome=True,
            metadata={"axes": "TZYX", "PhysicalSizeX": 0.4},
        )
        messages = _run_worker(input_path, output_path)
        self.assertIn("done", [message[0] for message in messages])
        with tifffile.TiffFile(output_path) as tif:
            self.assertEqual(tif.series[0].shape, (2, 2, 16, 16))
            self.assertEqual(tif.series[0].axes, "TZYX")
            self.assertEqual(tif.series[0].dtype, np.dtype(np.uint16))

    def test_float32_stack_is_supported_and_nonfinite_is_rejected(self):
        float_input = self.directory / "float.tif"
        float_output = self.directory / "float_out.tif"
        tifffile.imwrite(
            float_input,
            self.stack.astype(np.float32) / 100.0,
            photometric="minisblack",
        )
        messages = _run_worker(float_input, float_output)
        self.assertIn("done", [message[0] for message in messages])

        bad_input = self.directory / "nonfinite.tif"
        bad = self.stack.astype(np.float32)
        bad[1, 0, 0] = np.nan
        tifffile.imwrite(bad_input, bad, photometric="minisblack")
        with self.assertRaises(ValueError), TiffStackReader(bad_input) as reader:
            reader.read_frame(1)

    def test_rgb_input_is_rejected_at_open(self):
        rgb_path = self.directory / "rgb.tif"
        tifffile.imwrite(
            rgb_path, np.zeros((32, 32, 3), dtype=np.uint8), photometric="rgb"
        )
        with self.assertRaises(ValueError):
            TiffStackReader(rgb_path)

    def test_no_partial_files_remain_after_failure(self):
        output = self.directory / "result.tif"
        _run_worker(self.input_path, output, _params(delta=0))
        partials = list(self.directory.glob(".*.partial*"))
        self.assertEqual(partials, [])
        self.assertFalse(output.exists())

    def test_progress_reports_all_frames_before_done(self):
        output = self.directory / "progress.tif"
        messages = _run_worker(self.input_path, output)
        progress = [message for message in messages if message[0] == "progress"]
        self.assertEqual(len(progress), 3)
        self.assertEqual(progress[-1][1:], (100, 3, 3))

    def test_plain_tiff_image_description_is_preserved(self):
        input_path = self.directory / "desc.tif"
        output_path = self.directory / "desc_out.tif"
        tifffile.imwrite(
            input_path,
            self.stack,
            photometric="minisblack",
            description="my calibration description",
        )
        messages = _run_worker(input_path, output_path)
        self.assertIn("done", [message[0] for message in messages])
        with tifffile.TiffFile(output_path) as tif:
            self.assertEqual(
                tif.pages[0].tags["ImageDescription"].value,
                "my calibration description",
            )

    def test_radial_grid_cache_is_released_after_processing(self):
        import filters

        output = self.directory / "cache.tif"
        messages = _run_worker(self.input_path, output)
        self.assertIn("done", [message[0] for message in messages])
        self.assertEqual(filters.radial_cache_info().currsize, 0)

    def test_custom_range_percentiles_are_used_and_recorded(self):
        output = self.directory / "percentiles.tif"
        messages = queue.Queue()
        worker = ProcessingWorker(
            str(self.input_path),
            str(output),
            _params(),
            messages,
            threading.Event(),
            range_percentiles=(1.0, 99.0),
        )
        worker.run()
        kinds = []
        while not messages.empty():
            kinds.append(messages.get()[0])
        self.assertIn("done", kinds)
        sidecar = Path(str(output) + ".stem.json")
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        global_range = provenance["processing"]["global_range"]
        self.assertEqual(global_range["low_percentile"], 1.0)
        self.assertEqual(global_range["high_percentile"], 99.0)

    def test_invalid_range_percentiles_are_rejected_before_output(self):
        output = self.directory / "bad_percentiles.tif"
        messages = queue.Queue()
        worker = ProcessingWorker(
            str(self.input_path),
            str(output),
            _params(),
            messages,
            threading.Event(),
            range_percentiles=(99.0, 1.0),
        )
        worker.run()
        kinds = []
        while not messages.empty():
            kinds.append(messages.get()[0])
        self.assertIn("error", kinds)
        self.assertNotIn("done", kinds)
        self.assertFalse(output.exists())

    def test_cached_range_records_its_percentiles(self):
        output = self.directory / "cached_range.tif"
        messages = queue.Queue()
        worker = ProcessingWorker(
            str(self.input_path),
            str(output),
            _params(),
            messages,
            threading.Event(),
            global_range=(10.0, 3000.0),
            range_percentiles=(2.0, 98.0),
        )
        worker.run()
        kinds = []
        while not messages.empty():
            kinds.append(messages.get()[0])
        self.assertIn("done", kinds)
        sidecar = Path(str(output) + ".stem.json")
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        global_range = provenance["processing"]["global_range"]
        self.assertEqual(global_range["low_percentile"], 2.0)
        self.assertEqual(global_range["high_percentile"], 98.0)
        self.assertEqual(global_range["vmin"], 10.0)
        self.assertEqual(global_range["vmax"], 3000.0)

    def test_stale_temporary_file_discovery_targets_one_output(self):
        target = self.directory / "result.tif"
        stale_tif = self.directory / ".result.tif.abc.partial.tif"
        stale_json = self.directory / ".result.tif.def.partial.json"
        stale_backup = self.directory / ".result.tif.ghi.backup"
        stale_probe = self.directory / ".result.tif.jkl.probe"
        unrelated = self.directory / ".other.tif.xyz.partial.tif"
        for path in (stale_tif, stale_json, stale_backup, stale_probe, unrelated):
            path.write_bytes(b"stale")
        found = stale_temporary_files(str(target))
        self.assertIn(str(stale_tif), found)
        self.assertIn(str(stale_json), found)
        self.assertIn(str(stale_backup), found)
        self.assertIn(str(stale_probe), found)
        self.assertNotIn(str(unrelated), found)

    def test_input_changed_since_preview_is_rejected(self):
        # 导出必须对照"打开时"的签名：确认对话框停留期间输入被外部
        # 修改时，导出内容不得偏离用户预览过的数据。
        output = self.directory / "existing.tif"
        tifffile.imwrite(output, np.full((8, 8), 123, dtype=np.uint16))
        before = hashlib.sha256(output.read_bytes()).hexdigest()
        messages = queue.Queue()
        worker = ProcessingWorker(
            str(self.input_path),
            str(output),
            _params(),
            messages,
            threading.Event(),
            expected_signature=(0, 1, 2, 3),
        )
        worker.run()
        kinds = []
        while not messages.empty():
            kinds.append(messages.get()[0])
        self.assertIn("error", kinds)
        self.assertNotIn("done", kinds)
        self.assertEqual(before, hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertEqual(list(self.directory.glob(".*.partial*")), [])

    def test_unwritable_output_directory_is_rejected_early(self):
        output = self.directory / "existing.tif"
        tifffile.imwrite(output, np.full((8, 8), 123, dtype=np.uint16))
        before = hashlib.sha256(output.read_bytes()).hexdigest()
        with mock.patch(
            "worker.tempfile.mkstemp", side_effect=PermissionError("denied")
        ):
            messages = _run_worker(self.input_path, output)
        self.assertIn("error", [message[0] for message in messages])
        error_text = next(
            message[1] for message in messages if message[0] == "error"
        )
        self.assertIn("不可写", error_text)
        self.assertEqual(before, hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertEqual(list(self.directory.glob(".*.partial*")), [])

    def test_provenance_records_environment_and_timing(self):
        output = self.directory / "prov.tif"
        messages = _run_worker(self.input_path, output)
        self.assertIn("done", [message[0] for message in messages])
        provenance = json.loads(
            Path(str(output) + ".stem.json").read_text(encoding="utf-8")
        )
        libraries = provenance["application"]["libraries"]
        self.assertEqual(libraries["numpy"], np.__version__)
        self.assertEqual(libraries["opencv"], cv2.__version__)
        self.assertEqual(provenance["processing"]["clahe_tile_grid"], [8, 8])
        self.assertGreaterEqual(
            provenance["processing"]["duration_seconds"], 0.0
        )
        # started_utc 必须是合法的 ISO-8601 时间戳
        parsed = datetime.fromisoformat(provenance["processing"]["started_utc"])
        self.assertIsNotNone(parsed.tzinfo)

    def test_open_cancellation_interrupts_page_validation(self):
        event = threading.Event()
        event.set()
        with self.assertRaises(OperationCancelled):
            TiffStackReader(self.input_path, cancel_event=event)
        # 取消路径必须关闭已打开的句柄：随后仍可正常读取同一文件。
        with TiffStackReader(self.input_path) as reader:
            self.assertEqual(reader.num_frames, 3)


if __name__ == "__main__":
    unittest.main()
