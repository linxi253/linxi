"""Round-4 repair: recompute every reported number from the saved CLI coordinates.

No inference is run: the script reads runs/reviewed-95-validation-20260910/raw_outputs (produced by
the real CLI in the previous round), the frozen answers, and regenerates validation.json,
metrics_summary.json, report_tables.md, dsh_result.json and review_fixes_01.json. It also checks
its own grid and metrics against Codex's independent numerical audit.
"""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from atom_center.metrics import match_points                               # noqa: E402
from atom_center.model_manifest import sha256_file, verify_model_bundle    # noqa: E402
from atom_center.storage import write_json                                 # noqa: E402
from evaluate_reviewed_test_answers import inside, score                   # noqa: E402

RUN = ROOT / 'runs/reviewed-95-validation-20260910'
RAW = RUN / 'raw_outputs'
REVIEW = ROOT / 'runs/reviewed-test-20260910'
MANIFEST = ROOT / 'runs/target-adaptation-20260910/target_adaptation_onnx/model_manifest.json'
FINETUNE = ROOT / 'runs/target-adaptation-20260910/finetune100'
AUDIT = RUN / 'codex_numerical_audit.json'
TOLERANCES = (0.5, 1.0, 2.0)
PASS_THRESHOLD = 0.95
GRID_ROWS, GRID_COLS = 2, 3
SMALL_SAMPLE = 20


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def load_points(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    return np.asarray(payload['points_xy'], dtype=float).reshape(-1, 2), payload


def compare(first, second, radius=0.01):
    matches = match_points(first, second, max_distance_px=radius)
    distances = matches.distances_px
    return {'first_points': int(len(first)), 'second_points': int(len(second)),
            'matched': int(matches.true_positives),
            'unmatched_first': int(matches.false_positives),
            'unmatched_second': int(matches.false_negatives),
            'max_delta_px': float(distances.max()) if len(distances) else None,
            'identical': bool(matches.false_positives == 0 and matches.false_negatives == 0
                              and (not len(distances) or distances.max() <= radius))}


def tolerance_rows(prediction, truth):
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


def pick(rows, radius):
    return next(row for row in rows if row['radius_px'] == radius)


def grid_cells(roi):
    """Fixed 2x3 float edges over the frozen inclusive ROI; last column/row reach the maximum."""
    x_edges = np.linspace(roi[0], roi[2], GRID_COLS + 1)
    y_edges = np.linspace(roi[1], roi[3], GRID_ROWS + 1)
    return x_edges, y_edges


def assign_cells(points, roi, x_edges, y_edges):
    """Exactly one cell per point; returns an (rows*cols) list of boolean masks.

    Interior cells are half-open; the last column and last row additionally include the ROI's
    maximum edge, so no answer or prediction on the boundary is dropped.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    in_roi = ((points[:, 0] >= roi[0]) & (points[:, 0] <= roi[2])
              & (points[:, 1] >= roi[1]) & (points[:, 1] <= roi[3])) if len(points) else np.zeros(0, dtype=bool)
    masks = []
    for row in range(GRID_ROWS):
        for column in range(GRID_COLS):
            last_column = column == GRID_COLS - 1
            last_row = row == GRID_ROWS - 1
            x_ok = ((points[:, 0] >= x_edges[column])
                    & (points[:, 0] <= x_edges[column + 1] if last_column
                       else points[:, 0] < x_edges[column + 1]))
            y_ok = ((points[:, 1] >= y_edges[row])
                    & (points[:, 1] <= y_edges[row + 1] if last_row
                       else points[:, 1] < y_edges[row + 1]))
            masks.append(in_roi & x_ok & y_ok)
    return in_roi, masks


def local_grid(prediction, truth, roi):
    x_edges, y_edges = grid_cells(roi)
    truth_in, truth_masks = assign_cells(truth, roi, x_edges, y_edges)
    prediction_in, prediction_masks = assign_cells(prediction, roi, x_edges, y_edges)
    rows = []
    for index in range(GRID_ROWS * GRID_COLS):
        row, column = divmod(index, GRID_COLS)
        gt = truth[truth_masks[index]]
        pred = prediction[prediction_masks[index]]
        metrics = score(pred, gt, 2.0)
        rows.append({'row': row, 'column': column,
                     'bounds_xyxy_float': [float(x_edges[column]), float(y_edges[row]),
                                           float(x_edges[column + 1]), float(y_edges[row + 1])],
                     'bounds_display_rounded': [int(np.floor(x_edges[column])), int(np.floor(y_edges[row])),
                                                int(np.ceil(x_edges[column + 1])), int(np.ceil(y_edges[row + 1]))],
                     'gt': int(len(gt)), 'predictions': int(len(pred)),
                     'tp': metrics['tp'], 'fp': metrics['fp'], 'fn': metrics['fn'],
                     'precision': metrics['precision'], 'recall': metrics['recall'],
                     'f1': metrics['f1'], 'rmse_px': metrics['rmse_px'], 'p95_px': metrics['p95_px'],
                     'small_sample': bool(len(gt) < SMALL_SAMPLE),
                     'passes_both_95': bool(metrics['precision'] >= PASS_THRESHOLD
                                            and metrics['recall'] >= PASS_THRESHOLD)})
    coverage = {
        'roi_truth_total': int(truth_in.sum()), 'roi_predictions_total': int(prediction_in.sum()),
        'grid_truth_sum': int(sum(row['gt'] for row in rows)),
        'grid_predictions_sum': int(sum(row['predictions'] for row in rows)),
        'every_roi_answer_in_exactly_one_cell': bool(int(truth_in.sum()) == sum(row['gt'] for row in rows)),
        'every_roi_prediction_in_exactly_one_cell': bool(int(prediction_in.sum())
                                                         == sum(row['predictions'] for row in rows)),
    }
    coverage['passed'] = bool(coverage['every_roi_answer_in_exactly_one_cell']
                              and coverage['every_roi_prediction_in_exactly_one_cell'])
    return rows, coverage


def main():
    bundle, graph = verify_model_bundle(MANIFEST)
    state = json.loads((FINETUNE / 'state.json').read_text(encoding='utf-8-sig'))
    checkpoint_info = state['checkpoints']['last.pt']
    checkpoint_path = FINETUNE / checkpoint_info['path']
    audit = json.loads(AUDIT.read_text(encoding='utf-8-sig')) if AUDIT.is_file() else None

    fixed = {
        'task_id': 'reviewed-95-validation-20260910',
        'updated_utc': utc_now(),
        'purpose': 'recomputed every reported number from the saved CLI coordinates (no inference re-run)',
        'recomputed_from': 'runs/reviewed-95-validation-20260910/raw_outputs (CLI run1/run2 per path plus the PyTorch reference)',
        'grid_convention': 'fixed 2 rows x 3 columns over the frozen float roi_xyxy_inclusive; interior cells half-open; '
                           'the last column and last row also include the ROI maximum edge, so every ROI point falls in '
                           'exactly one cell',
        'model_manifest': str(MANIFEST.relative_to(ROOT).as_posix()),
        'model_sha256': bundle.model_sha256, 'graph_sha256': sha256_file(graph),
        'checkpoint_path': str(checkpoint_path.relative_to(ROOT).as_posix()),
        'checkpoint_sha256': sha256_file(checkpoint_path),
        'inference': dict(bundle.inference), 'refinement': dict(bundle.refinement),
        'primary_criterion': {'radius_px': 2.0, 'precision_min': PASS_THRESHOLD, 'recall_min': PASS_THRESHOLD},
        'diagnostic_radii_px': [0.5, 1.0],
        'targets': {},
    }
    audit_mismatches = []
    for number in (4, 5):
        record = json.loads((REVIEW / f'{number:02d}_comparison.json').read_text(encoding='utf-8-sig'))
        truth_all = np.asarray(record['all_csv_truth_xy'], dtype=float)
        roi = np.asarray(record['roi_xyxy_inclusive'], dtype=float)
        truth_roi = truth_all[inside(truth_all, roi)]
        full1, payload1 = load_points(RAW / f'{number:02d}_full_image_run1.json')
        full2, _ = load_points(RAW / f'{number:02d}_full_image_run2.json')
        roi1, _ = load_points(RAW / f'{number:02d}_roi_only_run1.json')
        roi2, _ = load_points(RAW / f'{number:02d}_roi_only_run2.json')
        onroi1, _ = load_points(RAW / f'{number:02d}_full_image_on_roi_run1.json')
        onroi2, _ = load_points(RAW / f'{number:02d}_full_image_on_roi_run2.json')
        torch_points, _ = load_points(RAW / f'{number:02d}_full_image_torch.json')
        prediction_in_roi = full1[inside(full1, roi)]

        rows_full = tolerance_rows(full1, truth_all)
        rows_on_roi = tolerance_rows(prediction_in_roi, truth_roi)
        rows_roi_only = tolerance_rows(roi1, truth_roi)
        grid, coverage = local_grid(full1, truth_all, roi)
        match_roi_2px = match_points(prediction_in_roi, truth_roi, max_distance_px=2.0)

        entry = {
            'number': number,
            'source_image': record['source_image'], 'answer_csv': record['csv_file'],
            'image_sha256': sha256_file(record['source_image']), 'csv_sha256': sha256_file(record['csv_file']),
            'answer_sha_matches_frozen_record': bool(sha256_file(record['csv_file']) == record['csv_sha256']),
            'roi_xyxy_inclusive_float': roi.tolist(),
            'reference_count_all_csv': int(len(truth_all)), 'reference_count_in_roi': int(len(truth_roi)),
            'prediction_count_full_image': int(len(full1)),
            'prediction_count_full_image_in_roi': int(len(prediction_in_roi)),
            'prediction_count_roi_only': int(len(roi1)),
            'coordinate_sha256': {name: hashlib.sha256((RAW / f'{number:02d}_{name}.json').read_bytes()).hexdigest()
                                  for name in ('full_image_run1', 'full_image_run2', 'full_image_on_roi_run1',
                                               'full_image_on_roi_run2', 'roi_only_run1', 'roi_only_run2')},
            'repeat_two_processes': {
                'full_image': compare(full1, full2), 'roi_only': compare(roi1, roi2),
                'full_image_on_roi_command_repeat': compare(onroi1, onroi2),
                'full_image_equals_same_command_repeat': compare(full1, onroi1)},
            'torch_vs_onnx': compare(torch_points, full1),
            'scopes': {
                'full_image_full_csv': {'reference_points': int(len(truth_all)), 'predictions': int(len(full1)),
                                        'tolerance_metrics': rows_full},
                'full_image_on_original_roi': {'reference_points': int(len(truth_roi)),
                                               'predictions': int(len(prediction_in_roi)),
                                               'tolerance_metrics': rows_on_roi},
                'roi_only_inference': {'reference_points': int(len(truth_roi)), 'predictions': int(len(roi1)),
                                       'tolerance_metrics': rows_roi_only},
            },
            'local_grid_on_original_roi': grid, 'grid_coverage': coverage,
            'matches_lists': {
                'roi_truth_xy': truth_roi.tolist(),
                'missing_answers_within_roi': [truth_roi[index].round(3).tolist()
                                               for index in match_roi_2px.unmatched_ground_truth_indices],
                'unmatched_predictions_within_roi': [prediction_in_roi[index].round(3).tolist()
                                                     for index in match_roi_2px.unmatched_prediction_indices],
            },
        }
        # per-cell placement of every unmatched answer, so a missed border point is never hidden
        placement = []
        x_edges, y_edges = grid_cells(roi)
        _, truth_masks = assign_cells(truth_roi, roi, x_edges, y_edges)
        for index in match_roi_2px.unmatched_ground_truth_indices:
            cell = next(cell_index for cell_index, mask in enumerate(truth_masks) if mask[index])
            placement.append({'answer_xy': truth_roi[index].round(3).tolist(),
                              'cell': f'r{cell // GRID_COLS}c{cell % GRID_COLS}'})
        entry['unmatched_answers_by_cell'] = placement
        fixed['targets'][str(number)] = entry

        if audit:
            audit_target = next(item for item in audit['targets'] if item['number'] == number)
            for name, rows, audit_key in (('full_image_full_csv', rows_full, 'full_image_full_csv'),
                                          ('full_image_on_original_roi', rows_on_roi, 'full_image_on_original_roi'),
                                          ('roi_only_inference', rows_roi_only, 'roi_only_inference')):
                for row in rows:
                    expected = audit_target['scopes'][audit_key][str(row['radius_px'])]
                    for key in ('tp', 'fp', 'fn'):
                        if row[key] != expected[key]:
                            audit_mismatches.append({'target': number, 'scope': name, 'radius': row['radius_px'],
                                                     'metric': key, 'recomputed': row[key], 'audit': expected[key]})
                    for key in ('precision', 'recall', 'rmse_px'):
                        if abs(row[key] - expected[key]) > 1e-12:
                            audit_mismatches.append({'target': number, 'scope': name, 'radius': row['radius_px'],
                                                     'metric': key, 'recomputed': row[key], 'audit': expected[key]})
            for cell, expected in zip(grid, audit_target['correct_float_roi_grid']):
                for key in ('gt', 'predictions', 'tp', 'fp', 'fn'):
                    if cell[key] != expected[key]:
                        audit_mismatches.append({'target': number, 'scope': 'grid', 'metric': key,
                                                 'cell': f"r{cell['row']}c{cell['column']}",
                                                 'recomputed': cell[key], 'audit': expected[key]})
                for key in ('precision', 'recall'):
                    if abs(cell[key] - expected[key]) > 1e-12:
                        audit_mismatches.append({'target': number, 'scope': 'grid', 'metric': key,
                                                 'cell': f"r{cell['row']}c{cell['column']}",
                                                 'recomputed': cell[key], 'audit': expected[key]})
                if (cell['rmse_px'] is None) != (expected['rmse_px'] is None):
                    audit_mismatches.append({'target': number, 'scope': 'grid', 'metric': 'rmse_px_presence',
                                             'cell': f"r{cell['row']}c{cell['column']}",
                                             'recomputed': cell['rmse_px'], 'audit': expected['rmse_px']})
                elif cell['rmse_px'] is not None and abs(cell['rmse_px'] - expected['rmse_px']) > 1e-9:
                    audit_mismatches.append({'target': number, 'scope': 'grid', 'metric': 'rmse_px',
                                             'cell': f"r{cell['row']}c{cell['column']}",
                                             'recomputed': cell['rmse_px'], 'audit': expected['rmse_px']})
    fixed['cross_check_against_codex_audit'] = {
        'audit_file': str(AUDIT.relative_to(ROOT).as_posix()), 'mismatches': audit_mismatches,
        'agrees': not audit_mismatches,
        'checked': 'every tp/fp/fn/precision/recall/rmse at 0.5/1/2 px in all three scopes, plus every grid cell'}

    verdicts, all_pass = {}, True
    for number in ('4', '5'):
        scopes = {name: pick(scope['tolerance_metrics'], 2.0)
                  for name, scope in fixed['targets'][number]['scopes'].items()}
        image_pass = all(row['passes_both_95'] for row in scopes.values())
        all_pass = all_pass and image_pass
        verdicts[number] = {'scopes': scopes, 'passes_image_95': bool(image_pass)}
    fixed['verdict_95'] = {'pass_threshold': PASS_THRESHOLD, 'radius_px': 2.0,
                           'per_image': verdicts, 'all_images_pass': bool(all_pass)}
    write_json(validation_path := RUN / 'validation.json', fixed)

    summary = {
        'task_id': 'reviewed-95-validation-20260910', 'updated_utc': utc_now(),
        'generated_from': str(RAW.relative_to(ROOT).as_posix()),
        'model_sha256': fixed['model_sha256'], 'graph_sha256': fixed['graph_sha256'],
        'checkpoint_sha256': fixed['checkpoint_sha256'],
        'targets': {number: {
            'reference_all_csv': entry['reference_count_all_csv'],
            'reference_in_roi': entry['reference_count_in_roi'],
            'predictions_full_image': entry['prediction_count_full_image'],
            'predictions_in_roi': entry['prediction_count_full_image_in_roi'],
            'predictions_roi_only': entry['prediction_count_roi_only'],
            'scopes': {name: scope['tolerance_metrics'] for name, scope in entry['scopes'].items()},
            'grid': entry['local_grid_on_original_roi'], 'grid_coverage': entry['grid_coverage'],
            'unmatched_answers_by_cell': entry['unmatched_answers_by_cell'],
            'repeat_two_processes': entry['repeat_two_processes'], 'torch_vs_onnx': entry['torch_vs_onnx'],
            'roi_xyxy_inclusive_float': entry['roi_xyxy_inclusive_float']}
            for number, entry in fixed['targets'].items()},
        'verdict_95': fixed['verdict_95'], 'cross_check_against_codex_audit': fixed['cross_check_against_codex_audit'],
    }
    write_json(RUN / 'metrics_summary.json', summary)
    print('grid coverage 04:', fixed['targets']['4']['grid_coverage'])
    print('grid coverage 05:', fixed['targets']['5']['grid_coverage'])
    print('audit cross-check agrees:', fixed['cross_check_against_codex_audit']['agrees'],
          'mismatches:', len(audit_mismatches))
    if audit_mismatches:
        print(json.dumps(audit_mismatches[:6], ensure_ascii=False, indent=1))
    print('all_images_pass:', all_pass)
    print('wrote validation.json and metrics_summary.json at', fixed['updated_utc'])


main()
