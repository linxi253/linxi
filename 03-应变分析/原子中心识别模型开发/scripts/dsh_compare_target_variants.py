"""DSH step 1: compare two adaptation checkpoints under the deployment candidate config.

Config under test: refinement gaussian / window 11 / max_shift_px 4, conf in {0.25, 0.10}.
Every row reports full-image and original-ROI 2 px one-to-one metrics on the two
authorized target images (labels were used for training: fit, not blind test), plus
old frozen train/val regression metrics under the same config.

No files outside the DSH selection directory are written; source images/labels are read only.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import gc
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from atom_center.backends import TorchBackend, onnx_pipeline          # noqa: E402
from atom_center.configuration import pipeline_config                  # noqa: E402
from atom_center.evaluation import evaluate_dataset                    # noqa: E402
from atom_center.image_io import load_image                            # noqa: E402
from atom_center.model_manifest import sha256_file                     # noqa: E402
from atom_center.pipeline import DetectionPipeline                     # noqa: E402
from atom_center.source_identity import verify_source                  # noqa: E402
from atom_center.storage import read_json, write_json                  # noqa: E402
from atom_center.training import read_run                              # noqa: E402
from evaluate_reviewed_test_answers import score, inside               # noqa: E402

BASE = ROOT / 'runs/target-adaptation-20260910'
RUN = BASE / 'finetune100'
OUT = BASE / 'dsh_selection'
DATASET = ROOT / 'data/processed/real_workflow_20260909_v2'
OLD_ONNX = ROOT / 'runs/generalization-20260910/adaptive_blob-onnx/model_manifest.json'

REFINEMENT = {'enabled': True, 'method': 'gaussian', 'window_size': 11,
              'polarity': 'bright', 'max_shift_px': 4.0, 'merge_sigma': None}
METRIC_KEYS = ('tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'rmse_px', 'p95_px',
               'match_distance_px')


def compact(metric):
    return {key: metric[key] for key in METRIC_KEYS if key in metric}


def deployment_config(run_config, conf):
    """Candidate deployment config: gaussian11 refinement on top of the run's inference."""
    inference = {**run_config['inference'], 'conf': conf}
    refinement = dict(REFINEMENT)
    pipeline = replace(pipeline_config(run_config), refine=True,
                       refinement_method=refinement['method'],
                       refinement_window=refinement['window_size'],
                       refinement_polarity=refinement['polarity'],
                       refinement_max_shift_px=refinement['max_shift_px'],
                       merge_after_refinement=True, adaptive_merge_sigma=None)
    return inference, refinement, pipeline


def torch_pipeline(checkpoint, contract, inference, pipeline):
    digest = sha256_file(checkpoint)
    backend = TorchBackend(checkpoint, expected_sha256=digest, contract=contract,
                           inference=inference, device='cpu')
    return DetectionPipeline(backend, pipeline), digest


def target_rows(pipeline, include_roi_only=True):
    rows = []
    for number in (4, 5):
        truth = read_json(ROOT / f'runs/reviewed-test-20260910/{number:02d}_comparison.json')
        verify_source(Path(truth['source_image']), truth['source_identity'])
        assert sha256_file(Path(truth['csv_file'])) == truth['csv_sha256']
        raw = load_image(truth['source_image'], normalize=False, preserve_dtype=True).image
        g = np.asarray(truth['all_csv_truth_xy'])
        roi = truth['roi_xyxy_inclusive']
        result = pipeline.detect(raw)
        points = result.points
        row = {'number': number, 'reference_count': len(g), 'prediction_count': len(points),
               'full_image': compact(score(points, g, 2.)),
               'same_original_roi': compact(score(points[inside(points, roi)], g[inside(g, roi)], 2.)),
               'role': 'supervised_adaptation_fit', 'independent_test': False}
        if include_roi_only:
            roi_result = pipeline.detect(raw, roi=roi)
            row['roi_only_inference'] = compact(
                score(roi_result.points[inside(roi_result.points, roi)], g[inside(g, roi)], 2.))
        rows.append(row)
    return rows


def regression(pipeline):
    out = {}
    for split in ('train', 'val'):
        result = evaluate_dataset(pipeline, DATASET, split=split, max_distance_px=2.)
        out[split] = {key: result[key] for key in
                      ('tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'rmse_px', 'p95_px')}
        out[split]['min_region_recall'] = min(
            region['tp'] / (region['tp'] + region['fn']) if region['tp'] + region['fn'] else 1.
            for region in result['regions'])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--checkpoints', nargs='*', default=None,
                        help='override checkpoint paths (labels like name=path)')
    parser.add_argument('--baseline', action='store_true', default=True)
    args = parser.parse_args()

    _, manifest, state = read_run(RUN)
    assert state['status'] == 'completed'
    run_config = manifest['config']
    contract = manifest['contract']
    candidates = [('epoch90', RUN / 'point_validation/candidate_0090_cada71028a83.pt'),
                  ('last', RUN / 'training/weights/last.pt')]
    if args.checkpoints:
        candidates = [(item.split('=', 1)[0], Path(item.split('=', 1)[1])) for item in args.checkpoints]

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []

    for conf in (0.25, 0.10):
        inference, refinement, pipeline_config_value = deployment_config(run_config, conf)
        for label, checkpoint in candidates:
            pipeline, digest = torch_pipeline(checkpoint, contract, inference, pipeline_config_value)
            record = {'kind': 'candidate', 'label': label, 'checkpoint': str(checkpoint),
                      'checkpoint_sha256': digest, 'device': 'cpu-pytorch', 'conf': conf,
                      'inference': inference, 'refinement': refinement}
            record['targets'] = target_rows(pipeline)
            record['old_regression'] = regression(pipeline)
            worst_full = min(t['full_image']['f1'] for t in record['targets'])
            worst_roi = min(t['same_original_roi']['f1'] for t in record['targets'])
            record['worst_target_f1_full_image'] = worst_full
            record['worst_target_f1_same_roi'] = worst_roi
            records.append(record)
            print(f"[candidate] {label} conf={conf} worst_full_f1={worst_full:.4f} "
                  f"worst_roi_f1={worst_roi:.4f} targets="
                  + json.dumps([{k: t['full_image'][k] for k in ('tp', 'fp', 'fn')} for t in record['targets']])
                  + f" val_f1={record['old_regression']['val']['f1']:.4f}", flush=True)
            del pipeline
            gc.collect()

        if args.baseline:
            baseline = onnx_pipeline(OLD_ONNX)
            baseline.config = replace(baseline.config,
                                      refine=True, refinement_method='gaussian', refinement_window=11,
                                      refinement_polarity='bright', refinement_max_shift_px=4.0,
                                      merge_after_refinement=True, adaptive_merge_sigma=None)
            baseline.backend.inference['conf'] = conf
            record = {'kind': 'old_model_reference',
                      'label': 'adaptive_blob-onnx (previous accepted deployment)',
                      'checkpoint': str(OLD_ONNX), 'checkpoint_sha256': baseline.backend.model_sha256,
                      'device': 'cpu-onnxruntime', 'conf': conf,
                      'refinement': dict(REFINEMENT)}
            record['old_regression'] = regression(baseline)
            record['old_regression_own_config'] = {
                split: {key: read_json(ROOT / f'runs/generalization-20260910/baseline_com11_{split}.json')[key]
                        for key in ('tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'rmse_px')}
                for split in ('train', 'val')}
            records.append(record)
            print(f"[old-model] conf={conf} val_f1(new config)={record['old_regression']['val']['f1']:.4f} "
                  f"train_f1={record['old_regression']['train']['f1']:.4f}", flush=True)
            del baseline
            gc.collect()

    ranking = sorted((r for r in records if r['kind'] == 'candidate'),
                     key=lambda r: (r['worst_target_f1_full_image'], r['worst_target_f1_same_roi'],
                                    r['old_regression']['val']['f1']), reverse=True)
    write_json(output / 'comparison.json',
               {'purpose': 'checkpoint_and_threshold_selection_for_target_adaptation_deployment',
                'rule': 'maximise the worse of the two target images (full-image 2px F1); '
                        'tie-break by same-ROI F1 then old validation F1',
                'labels_used_in_training': True, 'independent_test': False,
                'metric_radius_px': 2.0, 'device': 'cpu-pytorch for candidates',
                'records': records,
                'ranking': [{'label': r['label'], 'conf': r['conf'],
                             'worst_target_f1_full_image': r['worst_target_f1_full_image'],
                             'worst_target_f1_same_roi': r['worst_target_f1_same_roi'],
                             'checkpoint_sha256': r['checkpoint_sha256']} for r in ranking]})
    print('SELECTED', json.dumps({'label': ranking[0]['label'], 'conf': ranking[0]['conf'],
                                  'sha256': ranking[0]['checkpoint_sha256']}), flush=True)
    print('wrote', output / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
