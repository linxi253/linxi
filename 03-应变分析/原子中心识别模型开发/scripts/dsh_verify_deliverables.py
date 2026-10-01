"""DSH step 4 support: determinism repeat check, figure audit and source-integrity check.

Read-only with respect to every input: it opens the acceptance records, the drawn PNGs,
the run manifest and the user's answer files, and writes one JSON verdict.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from atom_center.storage import content_digest  # noqa: E402
BASE = ROOT / 'runs/target-adaptation-20260910'
ACC = BASE / 'dsh_acceptance'
RUN = BASE / 'finetune100'
REVIEW = ROOT / 'runs/reviewed-test-20260910'
COLORS = {'green': (38, 231, 163), 'red': (255, 94, 100), 'yellow': (255, 206, 84),
          'blue': (98, 185, 255)}
METRIC_KEYS = ('tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'rmse_px', 'p95_px')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def compact(metrics):
    return {k: metrics[k] for k in METRIC_KEYS}


def color_counts(path):
    array = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
    flat = array.reshape(-1, 3).astype(np.int32)
    counts = {}
    for name, rgb in COLORS.items():
        counts[name] = int(np.all(flat == np.array(rgb, dtype=np.int32), axis=1).sum())
    return array.shape, counts


def main():
    run1, run2 = load(BASE / 'dsh_selection/acceptance_run1.json'), load(ACC / 'acceptance.json')
    repeat = {
        'parity_identical': run1['parity'] == run2['parity'],
        'targets_identical': run1['targets'] == run2['targets'],
        'regression_identical': (run1['old_regression_onnx_deployed_config']
                                 == run2['old_regression_onnx_deployed_config']),
        'input_hashes_identical': (run1['input_hashes_before'] == run2['input_hashes_before']
                                   and run1['input_hashes_after'] == run2['input_hashes_after']),
        'model_and_checkpoint_identical': (run1['model_sha256'] == run2['model_sha256']
                                           and run1['checkpoint_sha256'] == run2['checkpoint_sha256']),
        'selection_metrics_identical': (run1['torch_selection_reference']
                                        == run2['torch_selection_reference']),
    }
    repeat['all_identical'] = all(repeat.values())

    # Torch selection record vs ONNX acceptance: same points, so the 2 px counts must agree.
    selection = load(BASE / 'dsh_selection/comparison.json')
    chosen = [r for r in selection['records']
              if r['kind'] == 'candidate' and r['label'] == 'last' and r['conf'] == 0.1][0]
    cross = []
    for row, torch_row in zip(run2['targets_detailed'], chosen['targets']):
        cross.append({
            'number': row['number'],
            'torch_reference_tp_fp_fn': [torch_row['full_image'][k] for k in ('tp', 'fp', 'fn')],
            'onnx_tp_fp_fn': [row['full_image'][k] for k in ('tp', 'fp', 'fn')],
            'full_image_counts_match': all(row['full_image'][k] == torch_row['full_image'][k]
                                           for k in ('tp', 'fp', 'fn')),
            'same_roi_counts_match': all(row['same_original_roi'][k] == torch_row['same_original_roi'][k]
                                         for k in ('tp', 'fp', 'fn')),
            'onnx_rmse_px': row['full_image']['rmse_px'],
            'torch_rmse_px': torch_row['full_image']['rmse_px'],
            'rmse_abs_delta_px': abs(row['full_image']['rmse_px'] - torch_row['full_image']['rmse_px']),
            'onnx_metrics': compact(row['full_image']),
            'onnx_same_roi_metrics': compact(row['same_original_roi']),
            'onnx_roi_only_metrics': compact(row['roi_only_inference']),
            'csv_sha256': row['csv_sha256'],
            'image_sha256': row['image_sha256'],
        })
    cross_ok = all(c['full_image_counts_match'] and c['same_roi_counts_match']
                   and c['rmse_abs_delta_px'] < 1e-3 for c in cross)

    # Figure audit: annotation colours must be present in numbers consistent with the metrics.
    figures = {}
    for number, row in zip((4, 5), run2['targets_detailed']):
        coordinates = load(ACC / f'{number:02d}_coordinates.json')
        points = np.asarray(coordinates['points_xy'], dtype=float)
        truth = np.asarray(coordinates['truth_xy'], dtype=float)
        shape_recorded = load(REVIEW / f'{number:02d}_comparison.json')['source_identity']['shape']
        height, width = int(shape_recorded[0]), int(shape_recorded[1])
        metrics = coordinates['metrics_full_image']
        assert metrics['tp'] == row['full_image']['tp']
        assert len(points) == row['prediction_count'] and len(truth) == row['reference_count_all_csv']
        entry = {'points': int(len(points)), 'truth': int(len(truth)),
                 'image_shape': [height, width],
                 'points_inside_image': bool(((points[:, 0] >= 0) & (points[:, 0] < width)
                                              & (points[:, 1] >= 0) & (points[:, 1] < height)).all()),
                 'coordinate_file_metrics_match_acceptance': True}
        for name, expected in (('predictions_green', {'green': len(points)}),
                               ('comparison', {'green': metrics['tp'], 'red': metrics['fn'],
                                               'yellow': metrics['fp']})):
            shape, counts = color_counts(ACC / f'{number:02d}_{name}.png')
            entry[f'{name}_shape'] = list(shape)
            entry[f'{name}_color_pixels'] = counts
            entry[f'{name}_colors_consistent'] = all(
                counts[colour] >= 6 * expected[colour] for colour in expected)
        figures[str(number)] = entry

    # Source integrity: the user's TIFF and answer CSV in 原子标注 must still match the frozen
    # identity recorded when the answers were reviewed, and the run manifest must be untouched.
    identity = {}
    for number in (4, 5):
        truth = load(REVIEW / f'{number:02d}_comparison.json')
        identity[str(number)] = {
            'source_image': truth['source_image'],
            'image_sha256_matches_recorded': digest(truth['source_image']) == truth['source_identity']['file_sha256'],
            'csv_file': truth['csv_file'],
            'csv_sha256_matches_recorded': digest(truth['csv_file']) == truth['csv_sha256'],
        }
    manifest = load(RUN / 'run_manifest.json')
    payload = {k: v for k, v in manifest.items() if k != 'content_sha256'}
    manifest_ok = content_digest(payload) == manifest['content_sha256']
    deployment = load(BASE / 'target_adaptation_onnx/deployment_config.json')

    verdict = {
        'purpose': 'dsh_step4_support_checks',
        'repeat_runs': 'dsh_selection/acceptance_run1.json vs dsh_acceptance/acceptance.json',
        'repeat_check': repeat,
        'torch_vs_onnx_cross_check': cross, 'torch_vs_onnx_counts_agree': cross_ok,
        'figures': figures,
        'source_integrity': identity,
        'run_manifest_content_hash_valid': manifest_ok,
        'run_manifest_content_sha256': manifest['content_sha256'],
        'onnx_model_sha256': deployment['model_sha256'],
        'checkpoint_sha256': deployment['checkpoint_sha256'],
        'acceptance_passed': load(ACC / 'acceptance_summary.json')['passed'],
        'passed': bool(repeat['all_identical'] and cross_ok and manifest_ok
                       and all(v['image_sha256_matches_recorded'] and v['csv_sha256_matches_recorded']
                               for v in identity.values())
                       and all(v[f'{n}_colors_consistent'] for v in figures.values()
                               for n in ('predictions_green', 'comparison'))),
    }
    out = ACC / 'repeat_and_figure_check.json'
    out.write_text(json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({k: verdict[k] for k in ('passed', 'torch_vs_onnx_counts_agree',
                                              'run_manifest_content_hash_valid', 'acceptance_passed')}, indent=1))
    print(json.dumps(repeat, indent=1))
    for number in ('4', '5'):
        print(number, json.dumps(figures[number], ensure_ascii=False))
    print('wrote', out)


main()
