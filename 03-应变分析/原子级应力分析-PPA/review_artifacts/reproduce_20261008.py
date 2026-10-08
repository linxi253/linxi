"""Bounded probes for the 2026-10-08 working-tree review.

Run with the repository's .venv/Scripts/python.exe. A confirmed probe means
the defect was observed, not fixed. Uses temporary synthetic files, withdrawn
Tk windows and mocked dialogs. Never deserializes a real model or changes
application source. Writes only evidence_20261008.json beside this script.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ppa
import ppa_stats as stats
from ppa_core import compute_cst_strain, compute_local_peak_pair_strain, load_analysis_image
from ppa_core.project_store import image_identity, load_project, save_project


@contextlib.contextmanager
def app_context():
    root = ppa.tk.Tk()
    root.withdraw()
    app = ppa.AtomMarkerApp(root)
    with contextlib.ExitStack() as stack:
        for name in ('showerror', 'showwarning', 'showinfo'):
            stack.enter_context(mock.patch.object(ppa.messagebox, name))
        try:
            yield app
        finally:
            # Probe cleanup. The production app has no corresponding close hook.
            root.update_idletasks()
            if app._poll_after_id is not None:
                root.after_cancel(app._poll_after_id)
                app._poll_after_id = None
            root.destroy()


def configure_lattice(app, size=5, spacing=10.):
    app.image = np.zeros((90, 90))
    app.points = [(10.+spacing*n, 10.+spacing*m) for n in range(size) for m in range(size)]
    app.reference_vecs = (np.array([spacing, 0.]), np.array([0., spacing]))
    app.ref_origin = np.array([10., 10.])


def analyze_sync(app, commit=True):
    with mock.patch.object(ppa.threading, 'Thread') as worker:
        app.run_ppa_analysis()
        assert worker.called
        worker.call_args.kwargs['target']()
    if commit:
        app._poll_worker_queue()


def probe_alias():
    idx = np.array([(n, m) for n in range(12) for m in range(100)])
    X = idx * 10.
    x = X @ np.array([[1., .02], [0., 1.]]).T + [.1, 0.]
    assignment = ppa.assign_unique_lattice_indices(x, [0, 0], [10, 0], [0, 10])
    result = compute_local_peak_pair_strain(assignment.lattice_indices, assignment.lattice_indices*10., x)
    control = compute_local_peak_pair_strain(idx, X, x)
    bad = np.abs(result.small_xy - .01) > .001
    assert assignment.n_conflicts == 0 and bad.any()
    assert np.max(np.abs(control.small_xy - .01)) < 1e-12
    return dict(rows=len(idx), conflicts=assignment.n_conflicts, expected_xy=.01,
                observed_range=[np.nanmin(result.small_xy), np.nanmax(result.small_xy)],
                wrong_rows=int(bad.sum()), wrong_grade_A=int(np.sum(bad & (result.site_quality == 'A-symmetric'))),
                low_confidence_rows=int(assignment.low_confidence_mask.sum()),
                control_max_error=float(np.max(np.abs(control.small_xy - .01))))


def probe_nonphysical_F():
    idx = np.array([(n, m) for n in range(5) for m in range(5)])
    X = idx.astype(float)
    cases = []
    for F in [np.diag([-1., 1.]), np.diag([0., 1.])]:
        for method in ('local', 'cst'):
            r = compute_local_peak_pair_strain(idx, X, X@F.T) if method == 'local' else compute_cst_strain(X, X@F.T)
            assert r.quality_mask.all()
            cases.append(dict(method=method, det_F=float(np.linalg.det(F)), valid_rows=int(r.quality_mask.sum()),
                              rows=len(r.quality_mask), max_gl_equivalent=float(np.max(r.equivalent_green))))
    return cases


def probe_reset_stale_result():
    with app_context() as app:
        configure_lattice(app)
        analyze_sync(app, commit=False)
        before = app._job_generation
        with mock.patch.object(ppa.messagebox, 'askyesno', return_value=True):
            app.reset_all()
        after = app._job_generation
        app._poll_worker_queue()
        assert before == after and len(app.strain_xx) == 25 and not app.points
        return dict(generation_before=before, generation_after=after, points=len(app.points),
                    image_is_none=app.image is None, stale_strain_rows=len(app.strain_xx))


def probe_result_metadata():
    with tempfile.TemporaryDirectory() as td, app_context() as app:
        configure_lattice(app)
        app.points = [(x*1.01, y) for x, y in app.points]
        analyze_sync(app)
        path = Path(td)/'strain.csv'
        with mock.patch.object(ppa.filedialog, 'asksaveasfilename', return_value=str(path)):
            app.save_strain_csv()
            csv_before = path.read_bytes()
            meta_before = json.loads(path.with_suffix('.metadata.json').read_text(encoding='utf-8'))
            generation = app._job_generation
            app.analysis_method_var.set('兼容：晶格-CST（三角形）')
            app._on_analysis_method_changed()
            app.vm_var.set('0.9')
            app.save_strain_csv()
            meta_after = json.loads(path.with_suffix('.metadata.json').read_text(encoding='utf-8'))
        assert csv_before == path.read_bytes() and meta_after['algorithm_id'] == 'lattice-cst-legacy'
        assert app._job_generation == generation and meta_after['element_geometry'] == 'sites'
        return dict(csv_unchanged=True, generation_unchanged=True,
                    algorithm_before=meta_before['algorithm_id'], algorithm_after=meta_after['algorithm_id'],
                    coefficient_before=meta_before['equivalent_strain_coefficient'],
                    coefficient_after=meta_after['equivalent_strain_coefficient'], geometry=meta_after['element_geometry'])


def probe_refined_reference():
    with app_context() as app:
        configure_lattice(app, size=6)
        app.reference_vecs = (np.array([10.1, 0]), np.array([0, 10.1]))
        app.ref_region = (0., 0., 80., 80.)
        analyze_sync(app)
        effective_a = app.ideal_grid[6] - app.ideal_grid[0]
        exported_a = app._reference_lattice_metadata()['a_vec']
        assert np.linalg.norm(effective_a - exported_a) > .09
        generation = app._job_generation
        app._clear_ref_region()
        assert generation == app._job_generation and app.strain_xx is not None
        return dict(effective_a=effective_a.tolist(), exported_a=exported_a, refined_atoms=36,
                    clearing_region_keeps_result=True, clearing_region_keeps_generation=True)


def probe_image_replaced_before_save():
    with tempfile.TemporaryDirectory() as td, app_context() as app:
        image = Path(td)/'image.tif'
        project = Path(td)/'project.json'
        original = np.arange(1024, dtype=np.uint16).reshape(32, 32)
        tifffile.imwrite(image, original)
        app._load_image_path(str(image))
        old_identity = image_identity(image)
        app.points = [(5., 5.)]
        tifffile.imwrite(image, np.flipud(original))
        with mock.patch.object(ppa.filedialog, 'asksaveasfilename', return_value=str(project)):
            app.save_project()
        saved = load_project(project)
        new_identity = image_identity(image)
        assert saved['image']['sha256'] == new_identity['sha256'] != old_identity['sha256']
        return dict(replacement_hash_saved=True, later_identity_validation_passes=True,
                    loaded_pixels_match_replacement=bool(np.array_equal(app.image, load_analysis_image(image).pixels)))


def probe_project_validation():
    cases = {}
    with tempfile.TemporaryDirectory() as td, app_context() as app:
        image = Path(td)/'image.tif'
        project = Path(td)/'project.json'
        tifffile.imwrite(image, np.arange(1024, dtype=np.uint16).reshape(32, 32))
        for name, points in [('nan_point', [['nan', 2.]]), ('malformed_point', [[1.]]), ('out_of_bounds', [[100., 2.]])]:
            save_project(project, {'points': points}, image)
            app.points = [(20., 20.)]
            app.image_path = 'previous-image.tif'
            error = None
            with mock.patch.object(ppa.filedialog, 'askopenfilename', return_value=str(project)):
                try:
                    app.load_project()
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
            cases[name] = dict(exception=error, previous_image_replaced=app.image_path != 'previous-image.tif',
                              previous_points_lost=app.points != [(20., 20.)],
                              nan_accepted=bool(app.points and np.isnan(app.points[0][0])))
        assert cases['nan_point']['nan_accepted'] and cases['malformed_point']['exception']
        assert cases['out_of_bounds']['previous_points_lost']
        save_project(project, {'points': [[float('nan'), 2.]]}, image)
        strict_error = None
        def reject_constant(token):
            raise ValueError(token)
        try:
            json.loads(project.read_text(encoding='utf-8'), parse_constant=reject_constant)
        except ValueError as exc:
            strict_error = str(exc)
        assert strict_error == 'NaN'
        cases['strict_json_rejects_saved_file'] = strict_error
    return cases


def probe_sidecar_failure():
    with tempfile.TemporaryDirectory() as td, app_context() as app:
        configure_lattice(app)
        analyze_sync(app)
        csv_path = Path(td)/'strain.csv'
        csv_path.with_suffix('.metadata.json').mkdir()
        with mock.patch.object(ppa.filedialog, 'asksaveasfilename', return_value=str(csv_path)), \
                mock.patch.object(ppa.messagebox, 'showerror') as error, mock.patch.object(app.status, 'config') as status:
            app.save_strain_csv()
            success = '已保存' in status.call_args.kwargs['text']
        assert csv_path.is_file() and success and not error.called
        return dict(csv_written=True, metadata_write_failed=True, success_reported=True, error_reported=False)


def probe_tiff_axes():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)/'width3.tif'
        tifffile.imwrite(path, np.arange(72, dtype=np.uint16).reshape(2, 12, 3), photometric='minisblack')
        with tifffile.TiffFile(path) as tf:
            axes = tf.series[0].axes
        r = load_analysis_image(path, frame_index=1)
        assert r.pixels.shape == (2, 12) and r.frame_count == 1 and r.frame_index is None
        return dict(tiff_axes=axes, source_shape=[2, 12, 3], expected_selected_shape=[12, 3],
                    actual_shape=list(r.pixels.shape), requested_frame=1, saved_frame=r.frame_index, frame_count=r.frame_count)


def probe_border_com():
    app = ppa.AtomMarkerApp.__new__(ppa.AtomMarkerApp)
    app.centroid_method = 'com'
    y, x = np.mgrid[:21, :21]
    image = np.exp(-((x-.4)**2+(y-10.3)**2)/2)
    scalar = app._refine_centroid(image, 0, 10, window=5)
    bx, by = app._refine_centroids_batch(image, np.array([[10, 0]]), window=5)
    difference = float(np.linalg.norm(np.array(scalar)-[bx[0], by[0]]))
    assert difference > .7
    return dict(true_center=[.4, 10.3], scalar=list(scalar), batch=[bx[0], by[0]], difference_px=difference)


def fields_for(points, values):
    return {'tri_centroids': points, **{key: values for key in
        ('strain_xx', 'strain_yy', 'strain_xy', 'strain_eq', 'rotation',
         'strain_gl_xx', 'strain_gl_yy', 'strain_gl_xy', 'strain_gl_eq')}}


def probe_interpolation():
    points = np.array([(a, b) for a in (5, 15, 25, 35, 45) for b in (5, 15, 25, 35, 45)], float)
    values = np.random.default_rng(0).uniform(0, .02, len(points))
    grids, _, _ = ppa._interpolate_strain_grids(fields_for(points, values), (51, 51))
    negative = float(np.nanmin(grids['eq']))
    assert values.min() > 0 and negative < 0
    # A missing interior measurement is also assigned a finite interpolated value.
    values[:] = .01
    values[12] = np.nan
    grids, _, _ = ppa._interpolate_strain_grids(fields_for(points, values), (51, 51), grid_size=103)
    assert np.isfinite(grids['xx'][51, 51])
    return dict(nonnegative_sample_min=float(np.random.default_rng(0).uniform(0, .02, len(points)).min()),
                negative_equivalent_grid_min=negative, missing_site_coordinate=[25, 25],
                grid_value_at_missing_site=float(grids['xx'][51, 51]))


def probe_grid_registration():
    side = 4096
    points = np.array([[-.5, -.5], [side-.5, -.5], [-.5, side-.5], [side-.5, side-.5]])
    grids, _, extent = ppa._interpolate_strain_grids(fields_for(points, points[:, 0]), (side, side), grid_size=200)
    displayed_centers = extent[0] + (np.arange(200)+.5)*(extent[1]-extent[0])/200
    mismatch = float(np.max(np.abs(grids['xx'][100] - displayed_centers)))
    assert mismatch > 10
    return dict(image_side_px=side, grid_size=200, maximum_sample_vs_display_center_offset_px=mismatch)


def probe_stats_provenance():
    app = stats.PPAStatsApp.__new__(stats.PPAStatsApp)
    app.data = stats.PPAData()
    app.data.algorithm_id = ppa.LOCAL_PPA_ALGORITHM_ID
    app.data.displacements = np.array([[1., 0.]])
    app.status = mock.Mock()
    app._update_status = mock.Mock()
    incoming = dict(n=3, centroids=np.array([[0., 0.], [1., 0.], [0., 1.]]),
                    strain_xx=np.zeros(3), strain_yy=np.zeros(3), strain_xy=np.zeros(3),
                    strain_eq=np.zeros(3), rotation=np.zeros(3), algorithm_id='lattice-cst-legacy')
    with mock.patch.object(stats.filedialog, 'askopenfilename', return_value='synthetic.csv'), \
            mock.patch.object(stats, 'load_strain_csv', return_value=incoming), \
            mock.patch.object(stats.messagebox, 'showwarning') as warning:
        app._load_strain()
    assert app.data.n_tri == 3 and warning.called and app.data.algorithm_id == ppa.LOCAL_PPA_ALGORITHM_ID
    return dict(mismatched_strain_committed=True, old_displacement_retained=True,
                warning_shown=True, stored_algorithm=app.data.algorithm_id, incoming_algorithm='lattice-cst-legacy')


def probe_histogram_weighting():
    app = stats.PPAStatsApp.__new__(stats.PPAStatsApp)
    app.data = stats.PPAData()
    app.data.strain_xx = np.array([0., 1.])
    for key in ('strain_yy', 'strain_xy', 'strain_eq', 'rotation'):
        setattr(app.data, key, np.array([0., 1.]))
    app.data.element_area = np.array([99., 1.])
    app.fig = ppa.Figure()
    app.canvas = mock.Mock()
    app._plot_strain_histograms()
    title = app.fig.axes[0].get_title()
    weighted, _ = stats.weighted_mean_std(app.data.strain_xx, app.data.element_area)
    assert '0.50000' in title and weighted == .01
    return dict(panel_mean=weighted, plot_mean=.5, plot_title=title, area_weights=[99, 1])


def probe_mode_classification():
    mode, _ = stats.compute_deformation_mode(np.array([.01]), np.array([-.1]))
    assert int(mode[0]) == 1
    return dict(principal_strains=[.01, -.1], trace=-.09, returned_mode=int(mode[0]), label='shear-dominated')


def probe_undo_after_replace():
    with app_context() as app:
        app.points = [(5., 5.)]
        app.undo_stack = [('add', (0, (5., 5.)))]
        with mock.patch.object(ppa.messagebox, 'askyesnocancel', return_value=False):
            app._finish_auto_detect([(20., 20.), (30., 30.)])
        app.undo_last()
        assert app.points == [(30., 30.)]
        return dict(replacement_points=[[20, 20], [30, 30]], points_after_undo=[list(p) for p in app.points],
                    old_action_removed_new_atom=True)


def probe_optional_dataset_rerun():
    from atom_detector.dataset.prepare_dataset import prepare_dataset
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        images, labels, output = base/'images', base/'labels', base/'dataset'
        images.mkdir(); labels.mkdir()
        for i in range(8):
            tifffile.imwrite(images/f'{i}.tif', np.ones((8, 8), dtype=np.uint8))
            (labels/f'{i}.txt').write_text('0 0.5 0.5 0.1 0.1', encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_dataset(images, labels, output, train_ratio=.5, val_ratio=.25, test_ratio=.25, seed=1)
            prepare_dataset(images, labels, output, train_ratio=.5, val_ratio=.25, test_ratio=.25, seed=2)
        splits = {s: {p.stem for p in (output/'images'/s).glob('*.tif')} for s in ('train', 'val', 'test')}
        overlap = splits['train'] & splits['test']
        assert overlap
        return dict(train_test_duplicate_stems=sorted(overlap), split_file_counts={s: len(v) for s, v in splits.items()})


def probe_bounded_performance():
    cases = []
    for size in (50, 100):
        idx = np.array([(n, m) for n in range(size) for m in range(size)])
        points = idx * 10.
        points[-1] = points[-2] + [.1, 0.]
        start = time.perf_counter()
        ppa.assign_unique_lattice_indices(points, [0, 0], [10, 0], [0, 10])
        cases.append(dict(rows=len(idx), seconds=time.perf_counter()-start))
    return dict(lattice_matching_on_gui_thread=cases,
                com_10000_atoms_window_101_single_array_bytes=10000*101*101*8,
                note='COM also allocates percentile/background and roi_sub temporaries; timing is local only.')


def probe_detection_input_snapshot():
    with app_context() as app:
        app.image = np.zeros((50, 50))
        app.processed_image = np.zeros((50, 50))
        app.processed_image[10, 10] = 1.
        app.use_preprocessed.set(True)
        with mock.patch.object(ppa.threading, 'Thread') as worker:
            app._detect_peaks(8., 0., 5, None, True)
            generation = app._job_generation
            replacement = np.zeros((50, 50))
            replacement[30, 30] = 1.
            app.processed_image = replacement
            worker.call_args.kwargs['target']()
        kind, returned_generation, payload = app._worker_queue.get_nowait()
        points, _ = payload
        assert kind == 'detect_done' and points == [(30., 30.)] and generation == returned_generation
        return dict(submitted_peak=[10, 10], returned_peak=list(points[0]), generation_unchanged=True)


def probe_close_callback():
    host = ppa.tk.Tk()
    host.withdraw()
    child = ppa.tk.Toplevel(host)
    child.withdraw()
    app = ppa.AtomMarkerApp(child)
    host.update_idletasks()
    app._ensure_worker_polling()
    poll_id = app._poll_after_id
    child.destroy()
    pending = host.tk.call('after', 'info')
    orphaned = poll_id in pending
    for timer in pending:
        # Child commands were destroyed already, so cancel raw Tcl timers.
        host.tk.call('after', 'cancel', timer)
    host.destroy()
    assert orphaned
    return dict(poll_callback_pending_after_window_destroy=True)


def probe_png_failed_success_status():
    with tempfile.TemporaryDirectory() as td, app_context() as app:
        app.image = np.zeros((10, 10))
        path = Path(td)/'blocked.png'
        path.mkdir()
        with mock.patch.object(ppa.filedialog, 'asksaveasfilename', return_value=str(path)), \
                mock.patch.object(ppa.messagebox, 'showerror') as error, mock.patch.object(app.status, 'config') as status:
            app.save_current_view()
            success = '已保存' in status.call_args.kwargs['text']
        assert success and error.called and not path.is_file()
        return dict(file_written=False, error_shown=True, success_status_shown=True)


PROBES = [probe_alias, probe_nonphysical_F, probe_reset_stale_result, probe_result_metadata,
          probe_refined_reference, probe_image_replaced_before_save, probe_project_validation,
          probe_sidecar_failure, probe_tiff_axes, probe_border_com, probe_interpolation,
          probe_grid_registration, probe_stats_provenance, probe_histogram_weighting,
          probe_mode_classification, probe_undo_after_replace, probe_optional_dataset_rerun,
          probe_bounded_performance, probe_detection_input_snapshot, probe_close_callback,
          probe_png_failed_success_status]


def main():
    evidence = {}
    failed = []
    for probe in PROBES:
        try:
            evidence[probe.__name__] = {'observed': probe()}
            print(f'{probe.__name__}: observed')
        except Exception as exc:
            evidence[probe.__name__] = {'probe_error': f'{type(exc).__name__}: {exc}'}
            failed.append(probe.__name__)
            print(f'{probe.__name__}: probe error: {exc}')
    files = ['ppa.py', 'ppa_stats.py', 'ppa_core/strain.py', 'ppa_core/project_store.py',
             'ppa_core/image_io.py', 'ppa_core/refine.py', 'pyproject.toml', 'requirements.lock.txt',
             'atom_detector/dataset/prepare_dataset.py', 'atom_detector/infer/detector.py',
             'atom_detector/model_security.py']
    output = {'review_date': '2026-10-08', 'interpreter': sys.executable,
              'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
              'note': 'Observed means reproduced in this working tree, not fixed. GUI dialogs and thread dispatch are mocked; calculations and state mutations are real.',
              'probes': evidence, 'probe_errors': failed}
    target = Path(__file__).with_name('evidence_20261008.json')
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Evidence: {target}')
    return bool(failed)


if __name__ == '__main__':
    raise SystemExit(main())
