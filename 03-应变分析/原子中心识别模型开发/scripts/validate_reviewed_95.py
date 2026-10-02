"""Reviewed-95 concentrated validation: practical paths, repeatability, tolerance diagnostics.

Every prediction comes from the real inference CLI entry (`python -m atom_center.cli predict`,
the command scripts/atom-center.ps1 forwards to) run in its own process, or from the CPU PyTorch
reference runner; metrics are then computed from those saved coordinates. No answer is moved, no
tolerance is widened and no ROI is shrunk: the original exported ROI is used as frozen.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from atom_center.geometry import clip_roi                                  # noqa: E402
from atom_center.image_io import load_image                                # noqa: E402
from atom_center.metrics import match_points                               # noqa: E402
from atom_center.model_manifest import sha256_file, verify_model_bundle    # noqa: E402
from atom_center.source_identity import verify_source                      # noqa: E402
from atom_center.storage import write_json                                 # noqa: E402
from atom_center.training import checkpoint_for_run                        # noqa: E402
from evaluate_reviewed_test_answers import inside, score                   # noqa: E402

RUN = ROOT / 'runs/reviewed-95-validation-20260910'
RAW = RUN / 'raw_outputs'
REVIEW = ROOT / 'runs/reviewed-test-20260910'
MANIFEST = ROOT / 'runs/target-adaptation-20260910/target_adaptation_onnx/model_manifest.json'
FINETUNE = ROOT / 'runs/target-adaptation-20260910/finetune100'
PYTHON = ROOT / '.runtime/python310/python.exe'
TOLERANCES = (0.5, 1.0, 2.0)
PASS_RADIUS = 2.0
PASS_THRESHOLD = 0.95
GRID_ROWS, GRID_COLS = 2, 3
SMALL_SAMPLE = 20


def run_cli(arguments, output_path, label):
    """One real CLI invocation in its own process; returns elapsed seconds."""
    command = [str(PYTHON), '-X', 'utf8', '-m', 'atom_center.cli', 'predict',
               '--manifest', str(MANIFEST), *arguments, '--output', str(output_path)]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, encoding='utf-8')
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(f'{label} CLI failed ({completed.returncode}): {completed.stderr[-800:]}')
    return elapsed, ' '.join(command)


def run_torch(arguments, output_path, checkpoint):
    command = [str(PYTHON), '-X', 'utf8', str(ROOT / 'scripts/torch_predict_reference.py'),
               '--manifest', str(MANIFEST), '--checkpoint', str(checkpoint), *arguments,
               '--output', str(output_path)]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, encoding='utf-8')
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(f'torch reference failed ({completed.returncode}): {completed.stderr[-800:]}')
    return elapsed, ' '.join(command)


def load_points(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    return np.asarray(payload['points_xy'], dtype=float).reshape(-1, 2), payload


def compare_points(first, second, radius=0.01):
    """One-to-one comparison of two prediction sets at a sub-pixel radius."""
    matches = match_points(first, second, max_distance_px=radius)
    distances = matches.distances_px
    return {'first_points': int(len(first)), 'second_points': int(len(second)),
            'matched': int(matches.true_positives),
            'unmatched_first': int(matches.false_positives),
            'unmatched_second': int(matches.false_negatives),
            'max_delta_px': float(distances.max()) if len(distances) else None,
            'identical': bool(matches.false_positives == 0 and matches.false_negatives == 0
                              and (not len(distances) or distances.max() <= radius))}


def tolerance_metrics(prediction, truth):
    rows = []
    for radius in TOLERANCES:
        metrics = score(prediction, truth, radius)
        rows.append({'radius_px': radius, 'tp': metrics['tp'], 'fp': metrics['fp'], 'fn': metrics['fn'],
                     'precision': metrics['precision'], 'recall': metrics['recall'], 'f1': metrics['f1'],
                     'rmse_px': metrics['rmse_px'], 'p95_px': metrics['p95_px'],
                     'passes_precision_95': bool(metrics['precision'] >= PASS_THRESHOLD),
                     'passes_recall_95': bool(metrics['recall'] >= PASS_THRESHOLD),
                     'passes_both_95': bool(metrics['precision'] >= PASS_THRESHOLD
                                            and metrics['recall'] >= PASS_THRESHOLD)})
    return rows


def local_grid(prediction, truth, roi):
    """Fixed 2x3 grid over the frozen ROI, scored with the already computed full-image output."""
    x0, y0, x1, y1 = roi
    x_edges = np.linspace(x0, x1, GRID_COLS + 1)
    y_edges = np.linspace(y0, y1, GRID_ROWS + 1)
    rows = []
    for row in range(GRID_ROWS):
        for column in range(GRID_COLS):
            bx0, bx1 = float(x_edges[column]), float(x_edges[column + 1])
            by0, by1 = float(y_edges[row]), float(y_edges[row + 1])
            in_cell = lambda points: ((points[:, 0] >= bx0) & (points[:, 0] < bx1)
                                      & (points[:, 1] >= by0) & (points[:, 1] < by1))
            gt = truth[in_cell(truth)]
            pred = prediction[in_cell(prediction)]
            metrics = score(pred, gt, PASS_RADIUS)
            rows.append({'row': row, 'column': column, 'bounds_xyxy': [bx0, by0, bx1, by1],
                         'gt': int(len(gt)), 'predictions': int(len(pred)),
                         'tp': metrics['tp'], 'fp': metrics['fp'], 'fn': metrics['fn'],
                         'precision': metrics['precision'], 'recall': metrics['recall'],
                         'rmse_px': metrics['rmse_px'],
                         'small_sample': bool(len(gt) < SMALL_SAMPLE),
                         'zero_ground_truth_with_false_positives': bool(len(gt) == 0 and metrics['fp'] > 0)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-torch', action='store_true')
    args = parser.parse_args()

    RUN.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    bundle, graph = verify_model_bundle(MANIFEST)
    _, checkpoint_path, checkpoint_sha = checkpoint_for_run(FINETUNE, 'last.pt')
    report = {
        'task_id': 'reviewed-95-validation-20260910',
        'purpose': 'concentrated 95% acceptance validation on the reviewed 04/05 answers',
        'user_goal': 'each reviewed image at least 95% precision on the 2 px one-to-one criterion, '
                     'with recall held at or above 95%, plus localisation diagnostics',
        'primary_criterion': {'radius_px': PASS_RADIUS, 'precision_min': PASS_THRESHOLD,
                              'recall_min': PASS_THRESHOLD,
                              'note': '0.5 px and 1 px are diagnostics only and are never used to claim a pass'},
        'model': {'manifest': str(MANIFEST.relative_to(ROOT).as_posix()),
                  'model_id': bundle.model_id, 'version': bundle.version, 'provider': bundle.provider,
                  'model_sha256': bundle.model_sha256, 'graph_sha256': sha256_file(graph),
                  'inference': dict(bundle.inference), 'refinement': dict(bundle.refinement),
                  'checkpoint': str(checkpoint_path.relative_to(ROOT).as_posix()),
                  'checkpoint_sha256': checkpoint_sha},
        'entry_point': {'command': f'{PYTHON.name} -X utf8 -m atom_center.cli predict '
                                   f'(the command scripts/atom-center.ps1 forwards to)',
                        'note': 'pwsh is not installed on this host, so the wrapper was replicated by its exact '
                                'forwarded command; the CLI uses the deployment loader verify_model_bundle + OnnxBackend',
                        'loader_verification': 'every run re-verified the bundle SHA before inference'},
        'targets': {},
    }
    for number in (4, 5):
        record = json.loads((REVIEW / f'{number:02d}_comparison.json').read_text(encoding='utf-8-sig'))
        image_path, csv_path = Path(record['source_image']), Path(record['csv_file'])
        verify_source(image_path, record['source_identity'])
        csv_sha = sha256_file(csv_path)
        assert csv_sha == record['csv_sha256'], 'answer CSV changed since the frozen record'
        import tifffile
        with tifffile.TiffFile(image_path) as handle:
            page, series = handle.pages[0], handle.series[0]
            container = {'pages': len(handle.pages), 'series_shape': list(series.shape),
                         'series_dtype': str(series.dtype), 'series_axes': series.axes,
                         'photometric': str(page.photometric), 'bits_per_sample': int(page.bitspersample),
                         'samples_per_pixel': int(page.samplesperpixel), 'compression': str(page.compression)}
        truth_all = np.asarray(record['all_csv_truth_xy'], dtype=float)
        roi_float = np.asarray(record['roi_xyxy_inclusive'], dtype=float)
        bounds = clip_roi(roi_float, tuple(record['source_identity']['shape']))
        roi_int = [bounds.x0, bounds.y0, bounds.x1, bounds.y1]
        truth_roi = truth_all[inside(truth_all, roi_float)]
        entry = {'number': number, 'source_image': str(image_path), 'answer_csv': str(csv_path),
                 'image_sha256': sha256_file(image_path), 'csv_sha256': csv_sha,
                 'answer_sha_matches_frozen_record': True,
                 'file_size': image_path.stat().st_size,
                 'container': container,
                 'decoded_plane': {'shape': record['source_identity']['shape'],
                                   'dtype': record['source_identity']['dtype']},
                 'roi_xyxy_inclusive': roi_float.tolist(), 'roi_integer_cli_bounds': roi_int,
                 'roi_integer_matches_pipeline_clip': True,
                 'reference_count_all_csv': int(len(truth_all)),
                 'reference_count_in_roi': int(len(truth_roi)),
                 'paths': {}}

        plan = {
            'full_image': ([], truth_all, 'full_image_full_csv'),
            'full_image_on_roi': ([], truth_roi, 'full_image_original_roi'),
            'roi_only': (['--roi', *[str(value) for value in roi_int]], truth_roi, 'roi_only_inference'),
        }
        for path_name, (extra, truth_for_scope, scope_key) in plan.items():
            run1 = RAW / f'{number:02d}_{path_name}_run1.json'
            run2 = RAW / f'{number:02d}_{path_name}_run2.json'
            seconds1, command1 = run_cli(['--image', str(image_path), *extra], run1, f'{number}-{path_name}-1')
            seconds2, _ = run_cli(['--image', str(image_path), *extra], run2, f'{number}-{path_name}-2')
            first, payload1 = load_points(run1)
            second, _ = load_points(run2)
            repeat = compare_points(first, second)
            path_entry = {'command': command1, 'seconds_run1': round(seconds1, 3),
                          'seconds_run2': round(seconds2, 3),
                          'prediction_count': int(len(first)),
                          'coordinate_space': 'full original image coordinates',
                          'inference_metadata': {'tile_count': payload1['metadata']['tile_count'],
                                                 'roi_xyxy': payload1['metadata']['roi_xyxy'],
                                                 'refinement_method': payload1['metadata']['refinement_method'],
                                                 'duplicates_removed_after_refinement':
                                                     payload1['metadata']['duplicates_removed_after_refinement']},
                          'provider': payload1['provider'], 'model_sha256': payload1['model_sha256'],
                          'repeat_two_processes': repeat,
                          'tolerance_metrics': tolerance_metrics(first, truth_for_scope),
                          'scored_against': scope_key,
                          'reference_count_scored': int(len(truth_for_scope))}
            if path_name == 'full_image':
                path_entry['tolerance_metrics_full_csv'] = path_entry['tolerance_metrics']
                # Frozen convention from the accepted baseline: restrict both sides to the original ROI.
                prediction_in_roi = first[inside(first, roi_float)] if len(first) else first
                path_entry['prediction_count_in_roi'] = int(len(prediction_in_roi))
                path_entry['tolerance_metrics_on_original_roi'] = tolerance_metrics(prediction_in_roi, truth_roi)
                path_entry['local_grid_on_original_roi'] = local_grid(first, truth_roi, roi_int)
                path_entry['points_xy'] = first.tolist()
                path_entry['confidences'] = payload1['confidences']
            if path_name == 'roi_only':
                path_entry['points_inside_roi'] = int(inside(first, roi_float).sum())
            entry['paths'][path_name] = path_entry

        if not args.skip_torch:
            torch_out = RAW / f'{number:02d}_full_image_torch.json'
            torch_seconds, torch_command = run_torch(['--image', str(image_path)], torch_out, checkpoint_path)
            torch_points, torch_payload = load_points(torch_out)
            onnx_points, _ = load_points(RAW / f'{number:02d}_full_image_run1.json')
            entry['paths']['full_image']['torch_reference'] = {
                'command': torch_command, 'seconds': round(torch_seconds, 3),
                'provider': torch_payload['provider'], 'prediction_count': int(len(torch_points)),
                'checkpoint_sha256': torch_payload['checkpoint_sha256'],
                'agreement_with_onnx': compare_points(torch_points, onnx_points)}
        report['targets'][str(number)] = entry

    # 95% verdict per image and scope, never averaged across images.
    verdicts, all_pass = {}, True
    for number in ('4', '5'):
        entry = report['targets'][number]
        rows = entry['paths']['full_image']['tolerance_metrics_full_csv']
        roi_rows = entry['paths']['full_image']['tolerance_metrics_on_original_roi']
        roi_only_rows = entry['paths']['roi_only']['tolerance_metrics']
        pick = lambda rows, radius: next(row for row in rows if row['radius_px'] == radius)
        scopes = {'full_image_full_csv': pick(rows, PASS_RADIUS),
                  'full_image_on_original_roi': pick(roi_rows, PASS_RADIUS),
                  'roi_only_inference': pick(roi_only_rows, PASS_RADIUS)}
        image_pass = all(row['passes_both_95'] for row in scopes.values())
        all_pass = all_pass and image_pass
        verdicts[number] = {'scopes': scopes, 'passes_image_95': bool(image_pass)}
    report['verdict_95'] = {'pass_threshold': PASS_THRESHOLD, 'radius_px': PASS_RADIUS,
                            'per_image': verdicts, 'all_images_pass': bool(all_pass),
                            'statement': 'pass requires precision >= 0.95 and recall >= 0.95 for every '
                                         'scope of each image separately; the two images are never averaged'}
    write_json(RUN / 'validation.json', report)
    for number in ('4', '5'):
        verdict = verdicts[number]
        print(f"target {number}: passes_image_95={verdict['passes_image_95']}", flush=True)
        for scope, row in verdict['scopes'].items():
            print(f"   {scope}: P={row['precision']:.4f} R={row['recall']:.4f} "
                  f"TP{row['tp']} FP{row['fp']} FN{row['fn']} rmse={row['rmse_px']:.4f}", flush=True)
    print('all_images_pass =', all_pass, flush=True)
    print('wrote', RUN / 'validation.json')


if __name__ == '__main__':
    main()
