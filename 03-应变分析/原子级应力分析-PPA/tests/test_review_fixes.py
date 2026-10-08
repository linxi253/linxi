"""Regression counterexamples from the 2026-10-08 engineering/science audit."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import tifffile
import ppa
import ppa_stats as stats
from ppa_core import compute_cst_strain, compute_local_peak_pair_strain, load_analysis_image
from ppa_core.interpolation import interpolate_fields
from ppa_core.project_store import save_project, load_project, image_identity, ProjectValidationError
from ppa_core.validation import validate_project_data
from ppa_core.export_store import CsvExport


class ScientificCounterexamples(unittest.TestCase):
    def test_accumulated_affine_displacement_keeps_local_topology(self):
        indices = np.array([(n, m) for n in range(12) for m in range(100)])
        reference = indices*10.
        theta = .06
        matrices = [np.array([[1., .02], [0., 1.]]), np.diag([1.03, .98]),
                    np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])]
        for matrix in matrices:
            for origin in ([0., 0.], [30., 40.]):
                with self.subTest(matrix=matrix, origin=origin):
                    deformed = reference @ matrix.T + [.1, .2]
                    assigned = ppa.assign_unique_lattice_indices(deformed, origin, [10, 0], [0, 10])
                    result = compute_local_peak_pair_strain(
                        assigned.lattice_indices, assigned.lattice_indices*10., deformed,
                        neighbor_map=assigned.neighbor_map, reference_basis=np.eye(2)*10,
                        unresolved_mask=assigned.topology_unresolved_mask)
                    self.assertEqual(len(result.small_xx), len(indices))
                    self.assertTrue(result.quality_mask.all())
                    expected = .5*(matrix+matrix.T)-np.eye(2)
                    np.testing.assert_allclose(result.small_xy, expected[0, 1], atol=2e-13)
                    np.testing.assert_allclose(result.small_xx, expected[0, 0], atol=2e-13)
                    np.testing.assert_allclose(result.green_xx, .5*(matrix.T@matrix-np.eye(2))[0, 0], atol=2e-13)

    def test_vacancy_and_ambiguous_sites_are_not_removed(self):
        reference = np.array([(x, y) for x in range(6) for y in range(6)], float)*10
        deformed = np.delete(reference, 14, axis=0)
        deformed = np.vstack((deformed, deformed[0]+[.01, .01]))
        assigned = ppa.assign_unique_lattice_indices(deformed, [0, 0], [10, 0], [0, 10])
        result = compute_local_peak_pair_strain(
            assigned.lattice_indices, assigned.lattice_indices*10., deformed,
            neighbor_map=assigned.neighbor_map, reference_basis=np.eye(2)*10,
            unresolved_mask=assigned.topology_unresolved_mask)
        self.assertEqual(len(result.quality_mask), len(deformed))
        self.assertTrue((~result.quality_mask).any())
        self.assertTrue(np.isnan(result.small_xx[~result.quality_mask]).all())
        self.assertTrue(all(result.invalid_reasons[~result.quality_mask]))

    def test_reflection_and_collapse_invalid_for_both_algorithms(self):
        indices = np.array([(n, m) for n in range(5) for m in range(5)])
        reference = indices.astype(float)
        for matrix in (np.diag([-1., 1.]), np.diag([0., 1.]), np.diag([1e-12, 1.])):
            for method in ('local', 'cst'):
                result = (compute_local_peak_pair_strain(indices, reference, reference@matrix.T)
                          if method == 'local' else compute_cst_strain(reference, reference@matrix.T))
                self.assertFalse(result.quality_mask.any())
                self.assertTrue(np.isnan(result.polar_rotation).all())
                self.assertTrue(np.isnan(result.equivalent_green).all())
                self.assertTrue(all(result.invalid_reasons))

    def test_interpolation_preserves_holes_and_nonnegative_equivalent(self):
        points = np.array([(x, y) for x in (5, 15, 25, 35, 45) for y in (5, 15, 25, 35, 45)], float)
        values = np.random.default_rng(0).uniform(0, .02, len(points))
        grid = interpolate_fields(points, {'eq': values}, (-.5, 50.5, -.5, 50.5), 103)['eq']
        self.assertGreaterEqual(np.nanmin(grid), 0.)
        values[12] = np.nan
        grid = interpolate_fields(points, {'eq': values}, (-.5, 50.5, -.5, 50.5), 103)['eq']
        self.assertTrue(np.isnan(grid[51, 51]))
        self.assertTrue(np.isnan(grid[0, 0]))

    def test_interpolation_samples_actual_imshow_centers(self):
        points = np.array([[-.5, -.5], [4095.5, -.5], [-.5, 4095.5], [4095.5, 4095.5]])
        grid = interpolate_fields(points, {'x': points[:, 0]}, (-.5, 4095.5, -.5, 4095.5), 200)['x']
        centers = -.5 + (np.arange(200)+.5)*4096/200
        np.testing.assert_allclose(grid[100], centers, atol=1e-10)

    def test_scalar_and_batch_border_com_agree(self):
        app = ppa.AtomMarkerApp.__new__(ppa.AtomMarkerApp)
        app.centroid_method = 'com'
        y, x = np.mgrid[:21, :21]
        image = np.exp(-((x-.4)**2+(y-10.3)**2)/2)
        scalar = app._refine_centroid(image, 0, 10, window=5)
        bx, by = app._refine_centroids_batch(image, np.array([[10, 0]]*100), window=5)
        np.testing.assert_allclose(bx, scalar[0], atol=1e-12)
        np.testing.assert_allclose(by, scalar[1], atol=1e-12)

    def test_com_chunking_matches_single_peak_with_large_window(self):
        app = ppa.AtomMarkerApp.__new__(ppa.AtomMarkerApp)
        image = np.random.default_rng(7).random((130, 130))
        candidates = np.tile([[60, 60]], (500, 1))
        bx, by = app._refine_centroids_batch(image, candidates, window=101)
        x, y = app._refine_centroids_batch(image, candidates[:1], window=101)
        np.testing.assert_allclose(bx, x[0]); np.testing.assert_allclose(by, y[0])

    def test_compression_dominated_is_not_shear(self):
        mode, _ = stats.compute_deformation_mode(np.array([.01, .01, .01]), np.array([-.1, -.01, -.003]))
        self.assertEqual(mode.tolist(), [5, 1, 2])


class PersistenceCounterexamples(unittest.TestCase):
    def test_tiff_width_three_stack_uses_axes_and_requested_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'stack.tif'
            source = np.arange(72, dtype=np.uint16).reshape(2, 12, 3)
            tifffile.imwrite(path, source, photometric='minisblack')
            loaded = load_analysis_image(path, frame_index=1)
            self.assertEqual(loaded.pixels.shape, (12, 3))
            self.assertEqual((loaded.frame_count, loaded.frame_index), (2, 1))
            self.assertEqual(loaded.identity['frame_index'], 1)
            with self.assertRaises(ValueError):
                load_analysis_image(path)

    def test_project_rejects_replaced_source_and_retains_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            image, project = Path(directory)/'image.tif', Path(directory)/'project.json'
            original = np.arange(1024, dtype=np.uint16).reshape(32, 32)
            tifffile.imwrite(image, original)
            identity = load_analysis_image(image).identity
            save_project(project, {'points': [[5, 5]]}, image, expected_identity=identity)
            saved = project.read_bytes()
            tifffile.imwrite(image, original[::-1])
            with self.assertRaises(ProjectValidationError):
                save_project(project, {'points': [[6, 6]]}, image, expected_identity=identity)
            self.assertEqual(project.read_bytes(), saved)

    def test_finite_type_and_resource_bounds(self):
        for payload in ({'points': [[float('nan'), 2]]}, {'points': [[True, 2]]}, {'points': [[1]]},
                        {'detect_params': {'window': 1000001}}, {'detect_params': {'window': 4}},
                        {'preprocess': {'params': {'cycles': float('inf')}}},
                        {'analysis': {'equivalent_strain_coefficient': 'nan'}},
                        {'colorbar_manual': True, 'colorbar_vmin': 'nan'},
                        {'analysis': {'lattice_max_condition': 'nan'}},
                        {'reference': {'origin': [0, 0], 'a_vec': [10, 0], 'b_vec': [0, 10],
                                       'estimation': {'method': 'two-chain-average', 'a_point_count': 'nan'}}}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_project_data(payload)
        with self.assertRaises(ValueError):
            validate_project_data({'points': [[100, 2]]}, image_shape=(32, 32))

    def test_csv_metadata_failure_preserves_old_pair_and_checksum_detects_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'data.csv'
            first = CsvExport(path)
            first.csv_path.write_text('编号,x,y\n1,5,5\n', encoding='utf-8-sig')
            first.commit({'coordinate_convention': 'image-display'})
            original_csv, original_meta = path.read_bytes(), path.with_suffix('.metadata.json').read_bytes()
            second = CsvExport(path)
            second.csv_path.write_text('编号,x,y\n1,6,6\n', encoding='utf-8-sig')
            import os
            replace = os.replace
            def fail_metadata(source, target):
                if Path(target) == path.with_suffix('.metadata.json') and str(source).endswith('.tmp'):
                    raise PermissionError('controlled metadata write failure')
                return replace(source, target)
            with mock.patch('ppa_core.export_store.os.replace', side_effect=fail_metadata), self.assertRaises(OSError):
                second.commit({})
            self.assertEqual(path.read_bytes(), original_csv)
            self.assertEqual(path.with_suffix('.metadata.json').read_bytes(), original_meta)
            stats.load_atoms_csv(path)
            path.write_text('编号,x,y\n1,7,7\n', encoding='utf-8-sig')
            with self.assertRaises(ValueError):
                stats.load_atoms_csv(path)

    def test_source_consistency_requires_image_frame_algorithm_and_reference(self):
        identity = {'sha256': 'a'*64, 'size_bytes': 1, 'frame_index': 0}
        meta = {'image_identity': identity, 'algorithm_id': 'local',
                'reference_lattice': {'a': [10, 0]}, 'equivalent_strain_coefficient': 4/9}
        data = stats.PPAData()
        data.commit_source('displacement', meta)
        data.validate_source('strain', meta)
        for bad in ({**meta, 'algorithm_id': 'cst'}, {**meta, 'reference_lattice': {'a': [11, 0]}},
                    {**meta, 'image_identity': {**identity, 'frame_index': 1}},
                    {**meta, 'image_identity': {**identity, 'sha256': 'b'*64}}, {}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                data.validate_source('strain', bad)
        self.assertEqual(data.sources['displacement'], meta)

    def test_dataset_output_reuse_is_rejected_without_modifying_samples(self):
        from atom_detector.dataset.prepare_dataset import prepare_dataset
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            images, labels, output = base/'images', base/'labels', base/'dataset'
            images.mkdir(); labels.mkdir()
            for i in range(8):
                tifffile.imwrite(images/f'{i}.tif', np.ones((8, 8), dtype=np.uint8))
                (labels/f'{i}.txt').write_text('0 0.5 0.5 0.1 0.1', encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                prepare_dataset(images, labels, output, train_ratio=.5, val_ratio=.25, test_ratio=.25, seed=1)
            before = sorted(str(p.relative_to(output)) for p in output.rglob('*') if p.is_file())
            with self.assertRaises(ValueError):
                prepare_dataset(images, labels, output, train_ratio=.5, val_ratio=.25, test_ratio=.25, seed=2)
            self.assertEqual(before, sorted(str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()))

    def test_stats_rejects_mismatch_before_arrays_change_and_clears_old_optional_fields(self):
        app = stats.PPAStatsApp.__new__(stats.PPAStatsApp)
        app.data, app.status, app._update_status = stats.PPAData(), mock.Mock(), mock.Mock()
        identity = {'sha256': 'a'*64, 'size_bytes': 1, 'frame_index': None}
        meta = {'image_identity': identity, 'algorithm_id': 'peak-pairs-local-topology-v3',
                'reference_lattice': {'a_vec': [10, 0]}, 'equivalent_strain_coefficient': 4/9,
                'coordinate_convention': 'image-display', 'element_geometry': 'sites'}
        app.data.commit_source('displacement', meta)
        app.data.strain_xx = np.array([123.])
        app.data.strain_gl = {'gl_xx': np.array([456.])}
        app.data.tri_edge = np.array([True])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'strain.csv'
            def export(metadata):
                pending = CsvExport(path)
                pending.csv_path.write_text('编号,x,y,xx,yy,xy,eq,rot\n1,5,5,.01,.02,.03,.04,0\n', encoding='utf-8-sig')
                pending.commit(metadata)
            export({**meta, 'image_identity': {**identity, 'frame_index': 1}})
            with mock.patch.object(stats.filedialog, 'askopenfilename', return_value=str(path)), mock.patch.object(stats.messagebox, 'showerror') as error:
                app._load_strain()
                self.assertTrue(error.called)
                np.testing.assert_equal(app.data.strain_xx, [123.])
                np.testing.assert_equal(app.data.strain_gl['gl_xx'], [456.])
                export(meta)
                app._load_strain()
                np.testing.assert_equal(app.data.strain_xx, [.01])
                self.assertEqual(app.data.strain_gl, {})
                self.assertIsNone(app.data.tri_edge)
                self.assertEqual(app.data.filter_policy['retained'], 1)


class GuiCounterexamples(unittest.TestCase):
    def setUp(self):
        try:
            self.root = ppa.tk.Tk()
        except ppa.tk.TclError as error:
            self.skipTest(str(error))
        self.root.withdraw()
        self.app = ppa.AtomMarkerApp(self.root)
        self.dialogs = contextlib.ExitStack()
        for name in ('showerror', 'showwarning', 'showinfo'):
            self.dialogs.enter_context(mock.patch.object(ppa.messagebox, name))
        self.root.update_idletasks()

    def tearDown(self):
        self.dialogs.close()
        self.root.update_idletasks()
        self.root.destroy()

    def configure(self):
        self.app.image = np.zeros((90, 90))
        self.app.points = [(10.+10*n, 10.+10*m) for n in range(5) for m in range(5)]
        self.app.reference_vecs = (np.array([10., 0]), np.array([0., 10.]))
        self.app.ref_origin = np.array([10., 10.])

    def analyze(self, commit=True):
        with mock.patch.object(ppa.threading, 'Thread') as thread:
            self.app.run_ppa_analysis()
            self.assertTrue(thread.called)
            thread.call_args.kwargs['target']()
        if commit:
            self.app._poll_worker_queue()

    def test_reset_discards_queued_old_analysis(self):
        self.configure(); self.analyze(commit=False)
        generation = self.app._job_generation
        with mock.patch.object(ppa.messagebox, 'askyesno', return_value=True):
            self.app.reset_all()
        self.app._poll_worker_queue()
        self.assertGreater(self.app._job_generation, generation)
        self.assertIsNone(self.app.strain_xx)
        self.assertEqual(self.app.points, [])

    def test_provenance_tracks_effective_reference_and_controls_invalidate(self):
        self.configure()
        self.app.reference_vecs = (np.array([10.1, 0]), np.array([0., 10.1]))
        self.app.ref_region = (0., 0., 80., 80.)
        self.analyze()
        meta = self.app._result_metadata()
        np.testing.assert_allclose(meta['reference_lattice']['a_vec'], [10, 0], atol=1e-10)
        self.assertTrue(meta['reference_lattice']['refinement_applied'])
        self.app.von_mises_coeff = .9  # old array provenance is immutable even for API callers
        self.assertEqual(self.app._result_metadata()['equivalent_strain_coefficient'], 4/9)
        self.app._clear_ref_region()
        self.assertIsNone(self.app.strain_xx)
        self.analyze()
        self.app.analysis_method_var.set('兼容：晶格-CST（三角形）')
        self.app._on_analysis_method_changed()
        self.assertIsNone(self.app.strain_xx)

    def test_invalid_project_keeps_previous_session(self):
        with tempfile.TemporaryDirectory() as directory:
            image, project = Path(directory)/'image.tif', Path(directory)/'project.json'
            tifffile.imwrite(image, np.arange(1024, dtype=np.uint16).reshape(32, 32))
            save_project(project, {'points': [[5, 5]]}, image)
            data = json.loads(project.read_text(encoding='utf-8'))
            for bad_points in ([[100, 2]], [[1]], [['nan', 2]]):
                data['points'] = bad_points
                project.write_text(json.dumps(data), encoding='utf-8')
                self.app.points, self.app.image_path = [(20., 20.)], 'previous.tif'
                generation = self.app._job_generation
                with mock.patch.object(ppa.filedialog, 'askopenfilename', return_value=str(project)):
                    self.app.load_project()
                self.assertEqual(self.app.points, [(20., 20.)])
                self.assertEqual(self.app.image_path, 'previous.tif')
                self.assertEqual(self.app._job_generation, generation)

    def test_project_roundtrip_retains_coefficient_precision_and_condition_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            image, project = Path(directory)/'image.tif', Path(directory)/'project.json'
            tifffile.imwrite(image, np.arange(8100, dtype=np.uint16).reshape(90, 90))
            self.app._load_image_path(str(image))
            self.configure()
            self.app.von_mises_coeff = 4/9
            self.app.lattice_max_condition = 50.
            with mock.patch.object(ppa.filedialog, 'asksaveasfilename', return_value=str(project)):
                self.app.save_project()
            self.app.lattice_max_condition, self.app.von_mises_coeff = 30., .9
            with mock.patch.object(ppa.filedialog, 'askopenfilename', return_value=str(project)):
                self.app.load_project()
            self.assertEqual(self.app.lattice_max_condition, 50.)
            self.assertEqual(self.app.von_mises_coeff, 4/9)
            self.assertEqual(float(self.app.vm_var.get()), 4/9)

    def test_replacement_can_be_undone_without_deleting_new_atoms(self):
        self.app.points = [(5., 5.)]
        self.app.undo_stack = [('add', (0, (5., 5.)))]
        with mock.patch.object(ppa.messagebox, 'askyesnocancel', return_value=False):
            self.app._finish_auto_detect([(20., 20.), (30., 30.)])
        self.app.undo_last()
        self.assertEqual(self.app.points, [(5., 5.)])
        self.app.undo_last()
        self.assertEqual(self.app.points, [])

    def test_detection_uses_pixel_roi_and_method_snapshot(self):
        app = self.app
        app.image = np.zeros((50, 50))
        app.processed_image = np.zeros((50, 50)); app.processed_image[10, 10] = 1
        app.use_preprocessed.set(True)
        app.detect_roi = (0, 0, 20, 20)
        with mock.patch.object(ppa.threading, 'Thread') as thread:
            app._detect_peaks(8., 0., 5, None, True)
            app.processed_image[:] = 0; app.processed_image[30, 30] = 1
            app.detect_roi, app.centroid_method = (20, 20, 45, 45), 'gaussian'
            thread.call_args.kwargs['target']()
        kind, _, payload = app._worker_queue.get_nowait()
        self.assertEqual(kind, 'detect_done')
        self.assertEqual(payload[0], [(10., 10.)])

    def test_child_close_cancels_poll_and_canvas_idle(self):
        child = ppa.tk.Toplevel(self.root); child.withdraw()
        app = ppa.AtomMarkerApp(child)
        self.root.update_idletasks()
        app._ensure_worker_polling(); app.canvas.draw_idle()
        timers = [app._poll_after_id, app.canvas._idle_draw_id]
        generation = app._job_generation
        child.destroy()
        pending = self.root.tk.call('after', 'info')
        self.assertFalse(any(timer in pending for timer in timers))
        self.assertTrue(app._closed)
        self.assertGreater(app._job_generation, generation)
        self.root.update()

    def test_export_failure_never_reports_success(self):
        self.configure(); self.analyze()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'strain.csv'
            path.with_suffix('.metadata.json').mkdir()
            with mock.patch.object(ppa.filedialog, 'asksaveasfilename', return_value=str(path)), mock.patch.object(self.app.status, 'config') as status:
                self.app.save_strain_csv()
                self.assertFalse(status.called)
                self.assertFalse(path.exists())
            path = Path(directory)/'blocked.png'; path.mkdir()
            with mock.patch.object(ppa.filedialog, 'asksaveasfilename', return_value=str(path)), mock.patch.object(self.app.status, 'config') as status:
                self.app.save_current_view()
                self.assertFalse(status.called)

    def test_stat_histogram_uses_selected_weights(self):
        app = stats.PPAStatsApp.__new__(stats.PPAStatsApp)
        app.data = stats.PPAData()
        for name in ('strain_xx', 'strain_yy', 'strain_xy', 'strain_eq', 'rotation'):
            setattr(app.data, name, np.array([0., 1.]))
        app.data.element_area = np.array([99., 1.])
        app.fig = ppa.Figure(); app.canvas = mock.Mock()
        app.weighting_var = ppa.tk.StringVar(self.root, value='面积加权')
        app._plot_strain_histograms()
        self.assertIn('0.01000', app.fig.axes[0].get_title())
        patches = app.fig.axes[0].patches
        self.assertAlmostEqual(patches[0].get_height()/patches[-1].get_height(), 99.)
        app.weighting_var.set('元素等权')
        app._plot_strain_histograms()
        self.assertIn('0.50000', app.fig.axes[0].get_title())


if __name__ == '__main__':
    unittest.main()
