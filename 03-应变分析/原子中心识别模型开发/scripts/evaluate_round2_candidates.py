"""Round 2 step 3c: evaluate the pre-declared localisation candidates against the baseline.

Candidates (each with a stated reason, no reference answers inside any inference path):
  C1 gaussian11 + per-image fit gate   - a fit that explains the window poorly may have converged
                                         on a neighbour; keep the coarse detection there instead
  C2 gaussian window 7                 - a narrower window limits contamination from a nearby
                                         shoulder on these 20.4 px lattices
  C3 gaussian window 7 + fit gate      - the two combined
Selection rule from the plan: improve the worse target first (full-image F1), then the same-ROI
F1, then the count of >2 px localisation offsets, then RMSE; floors and old-set guardrails veto.
"""
from pathlib import Path
from dataclasses import replace
import json
import sys
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
from atom_center.backends import onnx_pipeline                            # noqa: E402
from atom_center.evaluation import evaluate_dataset                       # noqa: E402
from atom_center.image_io import load_image                               # noqa: E402
from atom_center.storage import write_json                                # noqa: E402
from evaluate_reviewed_test_answers import score, inside                  # noqa: E402

RUN = ROOT / 'runs/training-update-20260910'
REVIEW = ROOT / 'runs/reviewed-test-20260910'
MANIFEST = ROOT / 'runs/target-adaptation-20260910/target_adaptation_onnx/model_manifest.json'
DATASET = ROOT / 'data/processed/real_workflow_20260909_v2'
# Audit 35: the acceptance floors and regression guardrails are derived from
# unpublished experiment results, so they are read from configs/local/ (gitignored)
# on the machine that owns the data, never hardcoded in the public repository.
LOCAL_RECORDS = ROOT / 'configs/local/experiment_records.json'
REQUIRED_GUARDRAIL_KEYS = ('val_f1_min', 'train_f1_min',
                           'min_region_recall_baseline', 'min_region_recall_drop_limit')
METRIC_KEYS = ('tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'rmse_px', 'p95_px')


def round2_thresholds():
    path = LOCAL_RECORDS
    if not path.is_file():
        raise SystemExit(f'missing local experiment records: {path}; copy '
                         'configs/local/experiment_records.example.json and fill in the '
                         'evaluate_round2_candidates section on the machine that owns '
                         'the data (audit 35: unpublished metrics stay out of the repository)')
    payload = json.loads(path.read_text(encoding='utf-8-sig'))
    section = payload.get('evaluate_round2_candidates') or {}
    try:
        floors = {int(number): dict(counts) for number, counts in section['floors'].items()}
        guardrails = dict(section['guardrails'])
    except (KeyError, TypeError, AttributeError, ValueError):
        raise SystemExit(f'{path} lacks a valid evaluate_round2_candidates '
                         'section (floors keyed by image number plus guardrails)') from None
    missing = [key for key in REQUIRED_GUARDRAIL_KEYS if key not in guardrails]
    if missing:
        raise SystemExit(f'{path} guardrails lack required keys: {missing}')
    return floors, guardrails


def build(manifest, window=None, gate=None):
    pipeline = onnx_pipeline(manifest)
    changes = {}
    if window is not None:
        changes['refinement_window'] = window
    if gate is not None:
        changes['refinement_fit_gate_quantile'] = gate
    if changes:
        pipeline.config = replace(pipeline.config, **changes)
    return pipeline


def target_row(pipeline, number):
    record = json.loads((REVIEW / f'{number:02d}_comparison.json').read_text(encoding='utf-8-sig'))
    raw = np.asarray(load_image(record['source_image'], normalize=False,
                                preserve_dtype=True).image, dtype=np.float64)
    truth = np.asarray(record['all_csv_truth_xy'], dtype=float)
    roi = np.asarray(record['roi_xyxy_inclusive'], dtype=float)
    result = pipeline.detect(raw)
    points = result.points
    distance = cKDTree(points).query(truth)[0]
    full = score(points, truth, 2.)
    same_roi = score(points[inside(points, roi)], truth[inside(truth, roi)], 2.)
    return {
        'number': number, 'prediction_count': int(len(points)),
        'full_image': {key: full[key] for key in METRIC_KEYS},
        'same_original_roi': {key: same_roi[key] for key in METRIC_KEYS},
        'offsets_2_to_6px': int(((distance > 2.) & (distance <= 6.)).sum()),
        'misses_beyond_6px': int((distance > 6.).sum()),
        'fit_gate': result.metadata.get('fit_gate'),
        'duplicates_removed_after_refinement': result.metadata['duplicates_removed_after_refinement'],
    }


def regression(pipeline):
    output = {}
    for split in ('train', 'val'):
        result = evaluate_dataset(pipeline, DATASET, split=split, max_distance_px=2.)
        output[split] = {key: result[key] for key in
                         ('tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'rmse_px', 'p95_px')}
        output[split]['min_region_recall'] = min(
            region['tp'] / (region['tp'] + region['fn']) if region['tp'] + region['fn'] else 1.
            for region in result['regions'])
        output[split]['worst_regions'] = sorted(
            ({'group': region.get('group'), 'tp': region['tp'], 'fn': region['fn'],
              'recall': region['tp'] / (region['tp'] + region['fn']) if region['tp'] + region['fn'] else 1.}
             for region in result['regions']), key=lambda row: row['recall'])[:5]
    return output


def main():
    FLOORS, GUARDRAILS = round2_thresholds()
    candidates = [
        ('baseline_gaussian11_conf0.1', 'round-1 accepted configuration, no gate', build(MANIFEST)),
        ('C1_gaussian11_fitgate', 'keep the coarse detection where the gaussian fit residual is in the '
                                  "image's own top decile", build(MANIFEST, gate=0.9)),
        ('C2_gaussian7', 'narrower 7 px window limits nearby-shoulder contamination', build(MANIFEST, window=7)),
        ('C3_gaussian7_fitgate', 'narrower window plus the fit gate',
         build(MANIFEST, window=7, gate=0.9)),
    ]
    records = []
    for label, reason, pipeline in candidates:
        record = {'label': label, 'reason': reason,
                  'window': pipeline.config.refinement_window,
                  'fit_gate_quantile': pipeline.config.refinement_fit_gate_quantile,
                  'model_sha256': pipeline.backend.model_sha256,
                  'targets': [target_row(pipeline, number) for number in (4, 5)]}
        record['old_regression'] = regression(pipeline)
        record['worst_target_f1_full_image'] = min(row['full_image']['f1'] for row in record['targets'])
        record['worst_target_f1_same_roi'] = min(row['same_original_roi']['f1'] for row in record['targets'])
        record['total_offsets_2_to_6px'] = sum(row['offsets_2_to_6px'] for row in record['targets'])
        record['worst_target_rmse_px'] = max(row['full_image']['rmse_px'] for row in record['targets'])
        floors = all(all(row['same_original_roi'][key] >= FLOORS[row['number']][key]
                         if key in ('tp',) else
                         row['same_original_roi'][key] <= FLOORS[row['number']][key]
                         for key in FLOORS[row['number']])
                     for row in record['targets'])
        record['floors_met'] = bool(floors)
        record['guardrails'] = {
            'val_f1_ok': record['old_regression']['val']['f1'] >= GUARDRAILS['val_f1_min'],
            'train_f1_ok': record['old_regression']['train']['f1'] >= GUARDRAILS['train_f1_min'],
            'val_region_ok': (record['old_regression']['val']['min_region_recall']
                              >= GUARDRAILS['min_region_recall_baseline']['val']
                              - GUARDRAILS['min_region_recall_drop_limit']),
            'train_region_ok': (record['old_regression']['train']['min_region_recall']
                                >= GUARDRAILS['min_region_recall_baseline']['train']
                                - GUARDRAILS['min_region_recall_drop_limit']),
        }
        record['eligible'] = bool(record['floors_met'] and all(record['guardrails'].values()))
        records.append(record)
        print(f"[{label}] worst_full_f1={record['worst_target_f1_full_image']:.5f} "
              f"worst_roi_f1={record['worst_target_f1_same_roi']:.5f} "
              f"offsets={record['total_offsets_2_to_6px']} worst_rmse={record['worst_target_rmse_px']:.4f} "
              f"val_f1={record['old_regression']['val']['f1']:.5f} train_f1={record['old_regression']['train']['f1']:.5f} "
              f"val_region={record['old_regression']['val']['min_region_recall']:.4f} "
              f"train_region={record['old_regression']['train']['min_region_recall']:.4f} "
              f"floors={record['floors_met']} eligible={record['eligible']}", flush=True)
        for row in record['targets']:
            print(f"    target {row['number']} full {row['full_image']['tp']}/{row['full_image']['fp']}/"
                  f"{row['full_image']['fn']} rmse {row['full_image']['rmse_px']:.4f} | roi "
                  f"{row['same_original_roi']['tp']}/{row['same_original_roi']['fp']}/"
                  f"{row['same_original_roi']['fn']} rmse {row['same_original_roi']['rmse_px']:.4f} | "
                  f"offsets2-6 {row['offsets_2_to_6px']} misses>6 {row['misses_beyond_6px']} "
                  f"gate={row['fit_gate']}", flush=True)
        del pipeline
    eligible = [record for record in records if record['eligible']]
    baseline = [record for record in records if record['label'].startswith('baseline')][0]
    ranked = sorted(eligible, key=lambda record: (record['worst_target_f1_full_image'],
                                                  record['worst_target_f1_same_roi'],
                                                  -record['total_offsets_2_to_6px'],
                                                  -record['worst_target_rmse_px']), reverse=True)
    better = [record for record in ranked
              if record['worst_target_f1_full_image'] > baseline['worst_target_f1_full_image'] + 1e-12]
    selection = {'selected': better[0]['label'] if better else 'baseline_gaussian11_conf0.1',
                 'reason': ('improves the worse target image while meeting floors and guardrails'
                            if better else 'no candidate improved the worse target image; baseline retained'),
                 'candidate_labels_better_than_baseline': [record['label'] for record in better],
                 'ranked_eligible': [record['label'] for record in ranked]}
    write_json(RUN / 'localization_candidates.json',
               {'purpose': 'bounded localisation candidates for the remaining 04/05 errors',
                'model_manifest': str(MANIFEST.resolve()), 'model_sha256': records[0]['model_sha256'],
                'metric_radius_px': 2.0, 'floors': {str(k): v for k, v in FLOORS.items()},
                'guardrails': GUARDRAILS, 'records': records, 'selection': selection})
    print('SELECTION', json.dumps(selection, ensure_ascii=False), flush=True)
    print('wrote', RUN / 'localization_candidates.json', flush=True)


main()
