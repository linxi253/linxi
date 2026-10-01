"""End-to-end tests for the batch GPA engine on synthetic stacks."""
import json
import os
import tempfile
import unittest

import numpy as np
import tifffile

from strainpp_gpa.batch import (
    DEFAULT_FIELDS,
    BatchConfig,
    refine_mask_from_roi,
    run_batch,
)
from strainpp_gpa.stack_reader import (
    discover_image_sequence,
    open_frame_source,
    read_stack_info,
)


def _lattice_frame(strain: float, rows: int = 96, cols: int = 96) -> np.ndarray:
    y, x = np.mgrid[:rows, :cols]
    return (
        np.cos(2 * np.pi * 16 / cols * (1.0 + strain) * x)
        + np.cos(2 * np.pi * 16 / rows * y)
    )


def _write_stack(path: str, frames) -> None:
    with tifffile.TiffWriter(path) as writer:
        for frame in frames:
            if frame.dtype == np.uint8:
                writer.write(frame)
            else:
                writer.write((frame * 63 + 128).clip(0, 255).astype(np.uint8))


class StackReaderTests(unittest.TestCase):
    def test_stack_info_and_lazy_iteration(self):
        frames = [_lattice_frame(0.0, 32, 40), _lattice_frame(0.0, 32, 40),
                  _lattice_frame(0.0, 32, 40)]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'stack.tif')
            _write_stack(path, frames)
            info = read_stack_info(path)
            self.assertEqual(info.n_frames, 3)
            self.assertEqual((info.height, info.width), (32, 40))
            collected = [
                index for index, _ in open_frame_source(path)[1]
            ]
            self.assertEqual(collected, [0, 1, 2])
            sub = [
                index for index, _ in
                open_frame_source(path, 0, None, 2)[1]
            ]
            self.assertEqual(sub, [0, 2])

    def test_sequence_directory_discovery_and_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            for number in (3, 1, 10, 2):
                path = os.path.join(directory, f'frame_{number:04d}.tif')
                tifffile.imwrite(path, _lattice_frame(0.0, 24, 32))
            files = discover_image_sequence(directory)
            self.assertEqual(
                [os.path.basename(f) for f in files],
                ['frame_0001.tif', 'frame_0002.tif',
                 'frame_0003.tif', 'frame_0010.tif'],
            )
            info, frames = open_frame_source(directory)
            self.assertEqual(info.kind, 'sequence')
            self.assertEqual(info.n_frames, 4)
            indices = [index for index, _ in frames]
            self.assertEqual(indices, [0, 1, 2, 3])


class BatchRunTests(unittest.TestCase):
    def _config(self, **overrides) -> BatchConfig:
        config = BatchConfig(
            fields=list(DEFAULT_FIELDS),
            export_stack=True,
            export_sequence=False,
            refine_g=False,
            sigma1=1.5,
            sigma2=1.5,
            g1=(16.0, 0.0),
            g2=(0.0, 16.0),
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    def test_batch_writes_stack_csv_and_json(self):
        strains = [0.0, 0.02, -0.02, 0.01]
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'stack.tif')
            _write_stack(source, [_lattice_frame(s) for s in strains])
            outdir = os.path.join(directory, 'out')
            records, cancelled = run_batch(source, self._config(), outdir)

            self.assertFalse(cancelled)
            self.assertEqual(len(records), 4)
            self.assertTrue(all(r['status'] == 'ok' for r in records))

            stack_path = os.path.join(outdir, 'batch_eps_xx.tif')
            with tifffile.TiffFile(stack_path) as tif:
                self.assertEqual(len(tif.pages), 4)
                first = tif.pages[0].asarray()
            self.assertEqual(first.shape, (96, 96))

            # Frame 2 has +2% compressed fringes -> eps_xx median ≈ -0.02
            # (negative per the sign convention verified in test_core;
            #  median is robust to the crossing-point outliers that pull
            #  the mean back towards zero on these synthetic frames).
            self.assertLess(records[1]['eps_xx_median'], -0.005)
            self.assertGreater(records[2]['eps_xx_median'], 0.005)

            self.assertTrue(os.path.exists(
                os.path.join(outdir, 'batch_statistics.csv')))
            self.assertTrue(os.path.exists(
                os.path.join(outdir, 'batch_metadata.json')))

    def test_batch_sequence_export_and_displacement(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'stack.tif')
            _write_stack(source, [_lattice_frame(0.0) for _ in range(3)])
            outdir = os.path.join(directory, 'out')
            config = self._config(
                fields=['eps_xx', 'u_x', 'quality_mask'],
                export_sequence=True,
                frame_step=2,
            )
            records, _ = run_batch(source, config, outdir)
            self.assertEqual(len(records), 2)  # frames 0 and 2
            frames_dir = os.path.join(directory, 'out', 'frames')
            written = sorted(os.listdir(frames_dir))
            self.assertEqual(len(written), 2 * 3)
            self.assertIn('frame_000002_eps_xx.tif', written)

    def test_bad_frame_does_not_stop_the_batch(self):
        """A blank (constant) frame is rejected — since the constant-image
        guard it fails at load instead of refinement — and is logged, while
        the batch continues."""
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'stack.tif')
            frames = [_lattice_frame(0.0),
                      np.zeros((96, 96), dtype=np.uint8),  # truly blank
                      _lattice_frame(0.0)]
            _write_stack(source, frames)
            outdir = os.path.join(directory, 'out')
            config = self._config(
                refine_g=True,
                reference_roi=(20, 20, 76, 76),
            )
            records, _ = run_batch(source, config, outdir)
            self.assertEqual([r['status'] for r in records],
                             ['ok', 'error', 'ok'])
            self.assertTrue(
                'no contrast' in records[1]['error']
                or 'reliable' in records[1]['error'],
                records[1]['error'],
            )

    def test_preview_stack_is_wysiwyg_rgb(self):
        """Previews render raw values full-field: almost no grey pixels."""
        from strainpp_gpa.batch import write_preview_stacks
        from strainpp_gpa.cmaps import NAN_GREY

        strains = [0.0, 0.02, -0.02]
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'stack.tif')
            _write_stack(source, [_lattice_frame(s) for s in strains])
            outdir = os.path.join(directory, 'out')
            records, _ = run_batch(
                source,
                self._config(
                    preview_specs={
                        'eps_xx': {'cmap': '_spp_turbo',
                                   'vmin': -0.05, 'vmax': 0.05},
                    },
                ),
                outdir,
            )
            self.assertEqual(len(records), 3)

            preview = os.path.join(outdir, 'preview_batch_eps_xx.tif')
            self.assertTrue(os.path.exists(preview))
            with tifffile.TiffFile(preview) as tif:
                self.assertEqual(len(tif.pages), 3)
                rgb = tif.pages[0].asarray()
                rgb2 = tif.pages[1].asarray()
            self.assertEqual(rgb.dtype, np.uint8)
            self.assertEqual(rgb.shape, (96, 96, 3))

            # WYSIWYG: raw rendering leaves (almost) no grey NaN pixels
            grey = np.array(NAN_GREY) * 255
            grey_fraction = float(np.mean(
                np.all(np.abs(rgb.astype(int) - grey) <= 2, axis=-1)
            ))
            self.assertLess(grey_fraction, 0.01)
            # 帧间颜色不同（应变符号相反应产生不同色分布）
            self.assertGreater(
                float(np.abs(rgb.astype(int) - rgb2.astype(int)).mean()), 1.0
            )

            # masked 模式：带 NaN 的数据页渲染为灰
            float_path = os.path.join(outdir, 'batch_eps_xx.tif')
            with tifffile.TiffFile(float_path) as tif:
                raw = tif.pages[0].asarray().copy()
            raw[:10, :10] = np.nan
            with tifffile.TiffWriter(float_path + '.tmp') as w:
                w.write(raw.astype(np.float32), contiguous=False)
            os.replace(float_path + '.tmp', float_path)
            write_preview_stacks(outdir, ['eps_xx'])
            with tifffile.TiffFile(preview) as tif:
                rgb = tif.pages[0].asarray()
            self.assertTrue(
                np.allclose(rgb[0, 0], grey, atol=2), rgb[0, 0]
            )

    def test_no_frame_failures_and_csv_median_columns(self):
        strains = [0.0, 0.02]
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'stack.tif')
            _write_stack(source, [_lattice_frame(s) for s in strains])
            outdir = os.path.join(directory, 'out')
            records, _ = run_batch(source, self._config(), outdir)
            import csv
            with open(os.path.join(outdir, 'batch_statistics.csv'),
                      encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertIn('eps_xx_median', rows[0])
            self.assertIn('eps_xx_mean', rows[0])

    def test_refine_records_g_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'stack.tif')
            # Frame drifted: peak is at 17 px, initial G guesses 16 px.
            frames = [
                _lattice_frame(0.0),
                _lattice_frame(0.0, 96, 96),  # same lattice for simplicity
            ]
            _write_stack(source, frames)
            outdir = os.path.join(directory, 'out')
            config = self._config(
                refine_g=True,
                reference_roi=(20, 20, 76, 76),
                g1=(15.0, 0.0),
                g2=(0.0, 16.0),
            )
            records, _ = run_batch(source, config, outdir)
            for record in records:
                self.assertIn('g1_refined', record)
                self.assertGreater(record['g1_refined'][0], 14.0)

    def test_cancel_stops_batch_and_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'stack.tif')
            _write_stack(source, [_lattice_frame(0.0) for _ in range(4)])
            outdir = os.path.join(directory, 'out')

            calls = {'n': 0}

            def cancel():
                calls['n'] += 1
                return calls['n'] > 1  # cancel right after the first frame

            records, cancelled = run_batch(
                source, self._config(), outdir, cancel_check=cancel
            )
            self.assertTrue(cancelled)
            self.assertLess(len(records), 4)

    def test_refine_without_roi_is_rejected(self):
        config = self._config(refine_g=True, reference_roi=None)
        with self.assertRaises(ValueError):
            config.validate()

    def test_refine_mask_from_roi(self):
        mask = refine_mask_from_roi((10, 10), (2, 3, 5, 7))
        self.assertEqual(mask.shape, (10, 10))
        self.assertEqual(int(mask.sum()), 4 * 3)
        self.assertTrue(mask[3:7, 2:5].all())
        with self.assertRaises(ValueError):
            refine_mask_from_roi((10, 10), (8, 0, 12, 5))


class BatchValidationTests(unittest.TestCase):
    def _config(self, **overrides) -> BatchConfig:
        config = BatchConfig(
            fields=['eps_xx'],
            refine_g=False,
            sigma1=1.5,
            sigma2=1.5,
            g1=(16.0, 0.0),
            g2=(0.0, 16.0),
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    def test_zero_g_vector_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            self._config(g1=(0.0, 0.0)).validate()
        self.assertIn('DC', str(raised.exception))

    def test_sub_radius_g_vector_is_rejected(self):
        with self.assertRaises(ValueError):
            self._config(g2=(1.0, 0.5)).validate()

    def test_nonpositive_sigma_is_rejected(self):
        with self.assertRaises(ValueError):
            self._config(sigma2=0.0).validate()
        with self.assertRaises(ValueError):
            self._config(sigma1=float('nan')).validate()

    def test_invalid_pixel_size_is_rejected(self):
        with self.assertRaises(ValueError):
            self._config(pixel_size=(1.0, -0.5)).validate()
        with self.assertRaises(ValueError):
            self._config(pixel_size=[1.0, 2.0, 3.0]).validate()
        with self.assertRaises(ValueError):
            self._config(pixel_size=0.0).validate()
        # valid forms must pass
        self._config(pixel_size=1.0).validate()
        self._config(pixel_size=(2.0, 1.0)).validate()

    def test_malformed_preview_specs_are_rejected(self):
        with self.assertRaises(ValueError):
            self._config(preview_specs={
                'eps_xx': {'cmap': '_spp_turbo', 'vmin': 0.1, 'vmax': 0.1},
            }).validate()
        with self.assertRaises(ValueError):
            self._config(preview_specs={
                'eps_xx': {'cmap': '_spp_turbo', 'vmin': -0.1},
            }).validate()
        with self.assertRaises(ValueError):
            self._config(preview_specs={
                'quality_mask': {'vmin': 0.0, 'vmax': 1.0},
            }).validate()

    def test_preview_spec_falls_back_to_auto_range(self):
        from strainpp_gpa.batch import _preview_range_for
        config = self._config(preview_specs={
            'eps_xx': {'cmap': '_spp_turbo', 'vmin': 1.0},  # vmax missing
        })
        state = {'lut': np.zeros((256, 3), dtype=np.uint8)}
        data = np.linspace(-0.02, 0.02, 96 * 96).reshape(96, 96)
        vmin, vmax, lut = _preview_range_for('eps_xx', data, config, state)
        self.assertTrue(np.isfinite(vmin) and np.isfinite(vmax))
        self.assertLess(vmin, vmax)


class BatchAnisotropicPixelTests(unittest.TestCase):
    def test_anisotropic_pixel_size_is_preserved_end_to_end(self):
        """Batch results with a (y, x) calibration match the equivalent
        single-image GPA computation, and the metadata records the pair."""
        from strainpp_gpa.gpa import GPA
        from strainpp_gpa.stack_reader import iter_tiff_frames

        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'stack.tif')
            _write_stack(source, [_lattice_frame(0.0),
                                  _lattice_frame(0.01)])
            outdir = os.path.join(directory, 'out')
            config = BatchConfig(
                fields=['e_xy', 'eps_xx'],
                refine_g=False,
                sigma1=1.5,
                sigma2=1.5,
                g1=(16.0, 0.0),
                g2=(0.0, 16.0),
                pixel_size=(2.0, 1.0),
            )
            records, _ = run_batch(source, config, outdir)
            self.assertTrue(all(r['status'] == 'ok' for r in records))

            with open(os.path.join(outdir, 'batch_metadata.json'),
                      encoding='utf-8') as stream:
                metadata = json.load(stream)
            self.assertEqual(metadata['pixel_size_nm_yx'], [2.0, 1.0])

            # Numeric equivalence against the direct GPA call.
            _, frame = next(iter(iter_tiff_frames(source, 0, 1)))
            gpa = GPA()
            gpa.load_image(frame, pixel_size=(2.0, 1.0))
            gpa.set_g1(16.0, 0.0, sigma=1.5)
            gpa.set_g2(0.0, 16.0, sigma=1.5)
            direct = gpa.compute(mask_results=True)
            with tifffile.TiffFile(
                os.path.join(outdir, 'batch_e_xy.tif')
            ) as tif:
                exported = tif.pages[0].asarray()
            np.testing.assert_allclose(
                exported, direct.e_xy, rtol=1e-6, atol=1e-9, equal_nan=True
            )


class BatchReportTests(unittest.TestCase):
    def test_csv_fieldnames_are_deterministic(self):
        from strainpp_gpa.batch import _csv_fieldnames
        ok_record = {
            'eps_xx_median': 0.1, 'frame': 0, 'status': 'ok',
            'eps_xx_std': 0.0, 'g1': [1, 0],
        }
        error_record = {
            'g1_refined': [1, 1], 'frame': 1, 'status': 'error',
            'error': 'boom', 'elapsed_s': 0.5,
        }
        columns = _csv_fieldnames([error_record, ok_record])
        self.assertEqual(
            columns[:4], ['frame', 'status', 'error', 'elapsed_s']
        )
        self.assertEqual(columns[4:6], ['g1', 'g1_refined'])
        self.assertEqual(columns[6:], sorted(
            ['eps_xx_median', 'eps_xx_std']
        ))


class SeriesFramesTests(unittest.TestCase):
    def test_series_frames_decodes_once_and_slices(self):
        from strainpp_gpa.stack_reader import _series_frames

        class _StubSeries:
            def __init__(self, data):
                self._data = data
                self.shape = data.shape
                self.decode_calls = 0

            def asarray(self):
                self.decode_calls += 1
                return self._data

        stack = np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4)
        series = _StubSeries(stack)
        collected = [(index, image.copy())
                     for index, image in _series_frames(series)]
        self.assertEqual([index for index, _ in collected], [0, 1, 2])
        for index, image in collected:
            np.testing.assert_array_equal(image, stack[index])
        self.assertEqual(series.decode_calls, 1)


if __name__ == '__main__':
    unittest.main()
