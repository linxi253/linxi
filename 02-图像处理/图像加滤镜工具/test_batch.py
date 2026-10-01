"""Regression tests for complete ImageJ stacks and the GUI batch workflow."""

import json
from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

from image_filters import from_float, process_image, to_float
from main import FilterApp
from tif_io import ProcessingCancelled, TifDocument, safe_output_path
from version import __version__


def write_stack(path, frames=7):
    data = np.random.default_rng(261).integers(0, 256, (frames, 24, 31), dtype=np.uint8)
    tifffile.imwrite(path, data, imagej=True, metadata={'axes': 'ZYX', 'unit': 'nm'},
                     resolution=(5, 5))
    return data


class StackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_imagej_261_frames_filtered_and_metadata_preserved(self):
        source, output = self.folder / 'Aligned 261 of 261.tif', self.folder / 'filtered.tif'
        data = write_stack(source, 261)
        params = {'exposure': 20, 'gaussian_blur': 12, 'usm': 15}
        progress = []
        doc = TifDocument(source)
        # No full-series decoding is allowed in the stack processing path.
        with patch.object(tifffile.TiffPageSeries, 'asarray', side_effect=AssertionError('full stack read')):
            doc.save_processed(output, params, lambda n, total: progress.append((n, total)))
        self.assertEqual(progress, [(n, 261) for n in range(1, 262)])
        with tifffile.TiffFile(output) as result:
            self.assertTrue(result.is_imagej)
            self.assertEqual(result.series[0].shape, data.shape)
            self.assertEqual(result.series[0].axes, 'ZYX')
            self.assertEqual(result.imagej_metadata['slices'], 261)
            self.assertEqual(result.imagej_metadata['unit'], 'nm')
            self.assertEqual(result.pages[0].tags['XResolution'].value, (5, 1))
            for index, page in enumerate(result.pages):
                expected = from_float(process_image(to_float(data[index]), params), data.dtype)
                np.testing.assert_array_equal(page.asarray(), expected)
        manifest = json.loads(Path(f'{output}.filter.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['tiff']['frames'], 261)
        self.assertEqual(manifest['application']['version'], __version__)

    def test_cancel_mid_stack_leaves_no_output_or_temporary_files(self):
        source = self.folder / 'input.tif'
        data = write_stack(source)
        output_dir = self.folder / 'out'
        output_dir.mkdir()
        event = threading.Event()

        def cancel_after_three(current, total):
            if current == 3:
                event.set()

        with self.assertRaises(ProcessingCancelled):
            TifDocument(source).save_processed(output_dir / 'cancelled.tif', {'exposure': 20},
                                               cancel_after_three, event)
        self.assertEqual(list(output_dir.iterdir()), [])
        np.testing.assert_array_equal(tifffile.imread(source), data)

    def test_imagej_binary_metadata_roundtrip_in_both_byte_orders(self):
        data = np.arange(5 * 16 * 17, dtype=np.uint16).reshape(5, 16, 17)
        metadata = {'axes': 'ZYX', 'Info': 'TEM acquisition / 原始采集信息',
                    'Labels': [f'frame {n}' for n in range(5)],
                    'Ranges': (0.0, 65535.0), 'Properties': {'sample': 'Au'}}
        for byteorder, name in (('>', 'big'), ('<', 'little')):
            with self.subTest(byteorder=byteorder):
                source, output = self.folder / f'{name}.tif', self.folder / f'{name}_out.tif'
                tifffile.imwrite(source, data, imagej=True, byteorder=byteorder, metadata=metadata)
                TifDocument(source).save_processed(output, {'exposure': 20})
                with tifffile.TiffFile(source) as original, tifffile.TiffFile(output) as result:
                    self.assertEqual(result.byteorder, byteorder)
                    self.assertEqual(result.imagej_metadata, original.imagej_metadata)
                    for index, page in enumerate(result.pages):
                        expected = from_float(process_image(to_float(data[index]), {'exposure': 20}),
                                              data.dtype)
                        np.testing.assert_array_equal(page.asarray(), expected)

    def test_existing_manifest_is_preserved_and_name_is_skipped(self):
        source = self.folder / 'input.tif'
        write_stack(source)
        out_dir = self.folder / 'out'
        out_dir.mkdir()
        output = out_dir / source.name
        manifest = Path(f'{output}.filter.json')
        manifest.write_text('existing manifest', encoding='utf-8')
        self.assertNotEqual(safe_output_path(out_dir, source), str(output))
        with self.assertRaises(FileExistsError):
            TifDocument(source).save_processed(output)
        self.assertEqual(manifest.read_text(encoding='utf-8'), 'existing manifest')


class BatchGuiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            raise unittest.SkipTest(f'无可用显示环境，跳过 GUI 测试：{exc}')
        self.root.withdraw()
        self.app = FilterApp(self.root)
        self.addCleanup(self.close_app)
        for method in ('showinfo', 'showerror', 'showwarning', 'askyesno'):
            mock = patch(f'main.messagebox.{method}', return_value=True)
            mock.start()
            self.addCleanup(mock.stop)

    def close_app(self):
        self.app.cancel_processing()
        if self.app._task_thread is not None:
            self.app._task_thread.join(timeout=10)
        self.app.close()
        self.root.update()

    def wait_for_task(self):
        deadline = time.monotonic() + 15
        while self.app._busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertFalse(self.app._busy, 'Batch did not finish')
        self.assertEqual(str(self.app.btn_cancel['state']), 'disabled')

    def wait_for_preview(self, expected, timeout=15):
        """帧加载已异步化：轮询直到预览帧内容与期望一致。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.root.update()
            frame = self.app.preview_frame
            if frame is not None and frame.shape == expected.shape \
                    and np.array_equal(frame, expected):
                return
            time.sleep(0.01)
        self.fail(f'预览帧未在 {timeout}s 内加载到期望内容')

    def test_multiselect_jump_and_save_entire_current_stack(self):
        paths = [self.folder / 'first.tif', self.folder / 'second.tif']
        data = write_stack(paths[0], 261)
        write_stack(paths[1])
        with patch('main.filedialog.askopenfilenames', return_value=tuple(map(str, paths))):
            self.app.open_file()
        self.assertEqual(self.app.file_list, list(map(str, paths)))
        self.assertIn('全部帧', str(self.app.btn_save['text']))
        self.app.frame_number.set('261')
        self.app.jump_to_frame()
        self.wait_for_preview(to_float(data[-1]))
        self.assertEqual(self.app.frame_index, 260)
        self.app.frame_number.set('262')
        self.app.jump_to_frame()
        self.assertEqual(self.app.frame_index, 260)
        self.app.output_dir = str(self.folder / 'out')
        self.app.save_current()
        self.assertTrue(self.app._busy)
        self.assertEqual(str(self.app.frame_entry['state']), 'disabled')
        self.wait_for_task()
        np.testing.assert_array_equal(tifffile.imread(Path(self.app.output_dir) / paths[0].name), data)
        self.assertFalse((Path(self.app.output_dir) / paths[1].name).exists())
        self.assertEqual(self.app._completed_frames, 261)

    def test_batch_handles_duplicate_names_and_continues_after_bad_file(self):
        sources = []
        for name in ('a', 'b'):
            parent = self.folder / name
            parent.mkdir()
            source = parent / 'same.tif'
            write_stack(source)
            sources.append(str(source))
        bad = self.folder / 'bad.tif'
        bad.write_bytes(b'not a TIFF')
        self.app.file_list = [sources[0], str(bad), sources[1]]
        self.app.output_dir = str(self.folder / 'out')
        self.app.sliders['exposure'].set_value(20)
        self.app.batch_process()
        self.wait_for_task()
        outputs = sorted(Path(self.app.output_dir).glob('*.tif'))
        self.assertEqual(len(outputs), 2)
        self.assertEqual(self.app._completed_frames, 14)
        self.assertIn('失败 1', self.app.status_var.get())
        for output in outputs:
            self.assertEqual(TifDocument(output).num_frames, 7)
            expected = from_float(process_image(to_float(tifffile.imread(sources[0])[0]),
                                                 {'exposure': 20}), np.uint8)
            np.testing.assert_array_equal(tifffile.imread(output)[0], expected)

    def test_cancel_button_stops_active_stack_and_remaining_files(self):
        source = self.folder / 'input.tif'
        write_stack(source)
        self.app.file_list = [str(source)]
        self.app.output_dir = str(self.folder / 'out')
        started = threading.Event()
        original = TifDocument._iter_raw_frames

        def wait_until_cancelled(doc, event):
            started.set()
            if not event.wait(5):
                raise AssertionError('Cancel button did not signal worker')
            yield from original(doc, event)

        with patch.object(TifDocument, '_iter_raw_frames', wait_until_cancelled):
            self.app.batch_process()
            self.assertTrue(started.wait(5))
            self.app.cancel_processing()
            self.wait_for_task()
        self.assertIn('已取消', self.app.status_var.get())
        self.assertEqual(list(Path(self.app.output_dir).iterdir()), [])

    def test_load_params_from_manifest_roundtrip(self):
        """参数清单回读：保存→打乱→载入，滑块应恢复保存时的值。"""
        source = self.folder / 'input.tif'
        write_stack(source)
        self.app.file_list = [str(source)]
        self.app._load_current()
        self.app.output_dir = str(self.folder / 'out')
        self.app.sliders['exposure'].set_value(20)
        self.app.sliders['contrast'].set_value(15)
        self.app.save_current()
        self.wait_for_task()
        manifest = Path(self.app.output_dir) / f'{source.name}.filter.json'
        self.assertTrue(manifest.exists(), '保存后应生成参数清单')

        self.app.sliders['exposure'].set_value(-50)
        self.app.sliders['contrast'].set_value(0)
        with patch('main.filedialog.askopenfilename', return_value=str(manifest)):
            self.app.load_params()
        self.assertEqual(self.app.sliders['exposure'].get_value(), 20)
        self.assertEqual(self.app.sliders['contrast'].get_value(), 15)
        self.assertIn('已载入参数', self.app.status_var.get())

    def test_load_params_rejects_invalid_manifest(self):
        """范围越界/结构错误的清单必须被拒绝，且不改动当前参数。"""
        bad = self.folder / 'bad.filter.json'
        bad.write_text('{"schema_version": 2, "filters": {"exposure": 999}}',
                       encoding='utf-8')
        self.app.sliders['exposure'].set_value(7)
        with patch('main.filedialog.askopenfilename', return_value=str(bad)):
            self.app.load_params()
        self.assertEqual(self.app.sliders['exposure'].get_value(), 7, '无效清单不得改动参数')

        not_manifest = self.folder / 'plain.json'
        not_manifest.write_text('{"hello": "world"}', encoding='utf-8')
        with patch('main.filedialog.askopenfilename', return_value=str(not_manifest)):
            self.app.load_params()
        self.assertEqual(self.app.sliders['exposure'].get_value(), 7)

    def test_grayscale_disables_color_sliders(self):
        """灰度文件加载后，色彩类滑块禁用、影调滑块保持可用。"""
        source = self.folder / 'gray.tif'
        tifffile.imwrite(source, np.zeros((8, 8), np.uint8))
        self.app.file_list = [str(source)]
        self.app._load_current()
        self.assertFalse(self.app._color_supported)
        self.assertEqual(str(self.app.sliders['exposure'].slider['state']), 'normal')
        self.assertEqual(str(self.app.sliders['temperature'].slider['state']), 'disabled')

        rgb = self.folder / 'rgb.tif'
        tifffile.imwrite(rgb, np.zeros((8, 8, 3), np.uint8))
        self.app.file_list = [str(rgb)]
        self.app._load_current()
        self.assertTrue(self.app._color_supported)
        self.assertEqual(str(self.app.sliders['temperature'].slider['state']), 'normal')


if __name__ == '__main__':
    unittest.main(verbosity=2)
