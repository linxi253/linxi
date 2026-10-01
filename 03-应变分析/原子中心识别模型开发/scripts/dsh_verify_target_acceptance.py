"""DSH steps 3-4: CPU acceptance of the exported ONNX bundle on the two authorized targets.

Checks: input image/CSV hashes unchanged, CPU-PyTorch vs CPU-ONNX point parity (0.01 px),
final 2 px one-to-one metrics recomputed from the ONNX output (full image, same original
ROI, ROI-only diagnostic), old frozen train/val regression under the deployed config, and
visual deliverables (green prediction-only circles + missed/false comparison + coordinates).
"""
from pathlib import Path
from dataclasses import replace
import argparse
import gc
import json
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from atom_center.backends import TorchBackend, onnx_pipeline              # noqa: E402
from atom_center.configuration import pipeline_config                      # noqa: E402
from atom_center.evaluation import evaluate_dataset                        # noqa: E402
from atom_center.image_io import load_image, normalize_percentile           # noqa: E402
from atom_center.metrics import match_points                               # noqa: E402
from atom_center.model_manifest import sha256_file, verify_model_bundle    # noqa: E402
from atom_center.pipeline import DetectionPipeline                         # noqa: E402
from atom_center.source_identity import verify_source                      # noqa: E402
from atom_center.storage import read_json, write_json                      # noqa: E402
from atom_center.training import checkpoint_for_run, read_run              # noqa: E402
from evaluate_reviewed_test_answers import score, inside, draw_overlay     # noqa: E402

BASE = ROOT / 'runs/target-adaptation-20260910'
RUN = BASE / 'finetune100'
DATASET = ROOT / 'data/processed/real_workflow_20260909_v2'
REVIEW = ROOT / 'runs/reviewed-test-20260910'
METRIC_KEYS = ('tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'rmse_px', 'p95_px',
               'match_distance_px')


def compact(metric, keys=METRIC_KEYS):
    return {key: metric[key] for key in keys if key in metric}


def pipeline_from_manifest(manifest, checkpoint_path):
    bundle, graph = verify_model_bundle(manifest)
    inf, ref = dict(bundle.inference), dict(bundle.refinement)
    config = replace(pipeline_config({'inference': {**inf, 'merge_after_refinement':
                                                    inf.get('merge_after_refinement', False)},
                                      'refinement': ref}),
                     refine=ref['enabled'], refinement_method=ref['method'],
                     refinement_window=ref['window_size'], refinement_polarity=ref['polarity'],
                     refinement_max_shift_px=ref['max_shift_px'],
                     merge_after_refinement=inf.get('merge_after_refinement', False),
                     adaptive_merge_sigma=ref.get('merge_sigma'))
    torch_backend = TorchBackend(checkpoint_path, expected_sha256=sha256_file(checkpoint_path),
                                 contract=dict(bundle.input), inference=inf, device='cpu')
    return (onnx_pipeline(manifest), DetectionPipeline(torch_backend, config),
            bundle, graph, inf, ref)


def draw_predictions(raw, prediction, title, target, radius=4):
    base = Image.fromarray(np.rint(normalize_percentile(raw) * 255).astype('uint8')).convert('RGB')
    draw = ImageDraw.Draw(base)
    for x, y in prediction:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline='#26e7a3', width=2)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
    panel = Image.new('RGB', (max(raw.shape[1], 900), raw.shape[0] + 100), '#101827')
    panel.paste(base, (0, 100))
    draw = ImageDraw.Draw(panel)
    for index, line in enumerate((title,
                                  f'绿色圆圈 = 模型全部预测点，共 {len(prediction)} 个',
                                  '该图未叠加人工答案；此模型训练时使用过本图答案，属拟合表现，不是独立测试')):
        draw.text((12, 10 + index * 29), line, font=font, fill='white')
    panel.save(target)
    return panel


def target_case(pipeline, number):
    truth = read_json(REVIEW / f'{number:02d}_comparison.json')
    verify_source(Path(truth['source_image']), truth['source_identity'])
    csv_sha = sha256_file(Path(truth['csv_file']))
    assert csv_sha == truth['csv_sha256'], 'answer CSV changed'
    raw = load_image(truth['source_image'], normalize=False, preserve_dtype=True).image
    g = np.asarray(truth['all_csv_truth_xy'], dtype=float)
    roi = truth['roi_xyxy_inclusive']
    return truth, raw, g, np.asarray(roi, dtype=float), csv_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=BASE / 'target_adaptation_onnx/model_manifest.json')
    parser.add_argument('--checkpoint-name', default='last.pt')
    parser.add_argument('--output', type=Path, default=BASE / 'dsh_acceptance')
    parser.add_argument('--skip-regression', action='store_true')
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _, run_manifest, state = read_run(RUN)
    _, checkpoint_path, checkpoint_sha = checkpoint_for_run(RUN, args.checkpoint_name)
    onnx_pipe, torch_pipe, bundle, graph, inference, refinement = pipeline_from_manifest(
        args.manifest, checkpoint_path)
    assert checkpoint_sha == bundle.metrics['source_checkpoint_sha256'], 'bundle/checkpoint mismatch'

    rows, parity_rows, hashes_before = [], [], {}
    for number in (4, 5):
        truth, raw, g, roi, csv_sha = target_case(onnx_pipe, number)
        hashes_before[number] = {'image': sha256_file(Path(truth['source_image'])),
                                 'csv': csv_sha,
                                 'plane_sha256_recorded': truth['source_identity']['plane_sha256'],
                                 'file_size_recorded': truth['source_identity']['file_size']}
        assert hashes_before[number]['image'] == truth['source_identity']['file_sha256']
        t_points = torch_pipe.detect(raw).points
        o_result = onnx_pipe.detect(raw)
        o_points = o_result.points
        parity = match_points(t_points, o_points, max_distance_px=.01)
        parity_row = {'number': number, 'torch_points': int(len(t_points)),
                      'onnx_points': int(len(o_points)),
                      'matched': int(parity.true_positives),
                      'unmatched_torch': int(parity.false_positives),
                      'unmatched_onnx': int(parity.false_negatives),
                      'max_delta_px': float(parity.distances_px.max()) if len(parity.distances_px) else None,
                      'passed': bool(parity.false_positives == 0 and parity.false_negatives == 0
                                     and (not len(parity.distances_px) or parity.distances_px.max() <= .01))}
        parity_rows.append(parity_row)
        assert parity_row['passed'], f'CPU torch/ONNX parity failed for image {number:02d}'

        full = score(o_points, g, 2.)
        same_roi = score(o_points[inside(o_points, roi)], g[inside(g, roi)], 2.)
        roi_only = onnx_pipe.detect(raw, roi=roi)
        roi_only_metrics = score(roi_only.points[inside(roi_only.points, roi)], g[inside(g, roi)], 2.)
        stale = read_json(BASE / f'gaussian11_preview/{number:02d}_prediction.json')
        row = {'number': number, 'source_image': truth['source_image'], 'csv_file': truth['csv_file'],
               'csv_sha256': csv_sha, 'image_sha256': hashes_before[number]['image'],
               'model_sha256': o_result.model_sha256, 'checkpoint_sha256': checkpoint_sha,
               'reference_count_all_csv': int(len(g)), 'reference_count_in_roi': int(inside(g, roi).sum()),
               'prediction_count': int(len(o_points)), 'roi_xyxy_inclusive': roi.tolist(),
               'full_image': compact(full), 'same_original_roi': compact(same_roi),
               'roi_only_inference': compact(roi_only_metrics),
               'role': 'supervised_adaptation_fit', 'independent_test': False,
               'previous_preview_agrees_on_full_image_metrics':
                   compact(full) == compact(stale['full_image'])}
        rows.append(row)
        write_json(output / f'{number:02d}_coordinates.json',
                   {'number': number, 'source_image': truth['source_image'],
                    'image_sha256': hashes_before[number]['image'], 'csv_sha256': csv_sha,
                    'model_sha256': o_result.model_sha256, 'checkpoint_sha256': checkpoint_sha,
                    'roi_xyxy_inclusive': roi.tolist(),
                    'points_xy': o_points.tolist(), 'confidences': o_result.confidences.tolist(),
                    'truth_xy': g.tolist(),
                    'matched_indices_prediction_truth': full['matched_indices'],
                    'unmatched_prediction_indices': full['unmatched_prediction_indices'],
                    'unmatched_truth_indices': full['unmatched_truth_indices'],
                    'metrics_full_image': compact(full), 'metrics_same_original_roi': compact(same_roi),
                    'note': 'labels were used in supervised adaptation training; fit not blind test'})
        draw_predictions(raw, o_points,
                         f'开发图像 {number:02d} · 自适应模型预测（人工答案参与训练后的拟合）',
                         output / f'{number:02d}_predictions_green.png')
        draw_overlay(raw, g, o_points, [0, 0, raw.shape[1] - 1, raw.shape[0] - 1], full,
                     f'开发图像 {number:02d} · ONNX 实际输出与人工答案对照（拟合表现）',
                     output / f'{number:02d}_comparison.png')
        print('target', number, json.dumps({'parity': parity_row, 'full': compact(full),
                                            'same_roi': compact(same_roi)}), flush=True)
        del raw
        gc.collect()

    regression = None
    if not args.skip_regression:
        regression = {}
        for split in ('train', 'val'):
            result = evaluate_dataset(onnx_pipe, DATASET, split=split, max_distance_px=2.)
            regression[split] = {key: result[key] for key in
                                 ('tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'rmse_px', 'p95_px')}
            regression[split]['min_region_recall'] = min(
                r['tp'] / (r['tp'] + r['fn']) if r['tp'] + r['fn'] else 1. for r in result['regions'])
            print('regression', split, json.dumps(regression[split]), flush=True)

    selection = read_json(BASE / 'dsh_selection/comparison.json')
    final_rows = []
    for row in rows:
        final_rows.append({'number': row['number'], 'full_image': row['full_image'],
                           'same_original_roi': row['same_original_roi'],
                           'roi_only_inference': row['roi_only_inference'],
                           'prediction_count': row['prediction_count']})
    hashes_after = {number: {'image': sha256_file(Path(read_json(REVIEW / f'{number:02d}_comparison.json')['source_image'])),
                             'csv': sha256_file(Path(read_json(REVIEW / f'{number:02d}_comparison.json')['csv_file']))}
                    for number in (4, 5)}
    inputs_unchanged = all(hashes_before[n]['image'] == hashes_after[n]['image']
                           and hashes_before[n]['csv'] == hashes_after[n]['csv'] for n in (4, 5))
    summary = {'purpose': 'cpu_onnx_acceptance_of_target_adaptation_export',
               'scientific_acceptance': False, 'role': 'supervised_adaptation_fit',
               'independent_test': False, 'labels_used_in_training': True,
               'metric_radius_px': 2.0, 'parity_radius_px': 0.01,
               'model_manifest': str(args.manifest.resolve()), 'model_sha256': bundle.model_sha256,
               'graph_sha256': bundle.metrics['graph_sha256'],
               'checkpoint': str(checkpoint_path), 'checkpoint_sha256': checkpoint_sha,
               'inference': inference, 'refinement': refinement,
               'torch_device': 'cpu', 'onnx_providers': onnx_pipe.backend.session.get_providers(),
               'parity': parity_rows,
               'targets': final_rows, 'targets_detailed': rows,
               'old_regression_onnx_deployed_config': regression,
               'input_hashes_before': {str(k): v for k, v in hashes_before.items()},
               'input_hashes_after': {str(k): v for k, v in hashes_after.items()},
               'inputs_unchanged': inputs_unchanged,
               'torch_selection_reference': [
                   {'label': r['label'], 'conf': r['conf'],
                    'worst_target_f1_full_image': r['worst_target_f1_full_image'],
                    'old_regression': r['old_regression']}
                   for r in selection['records'] if r['kind'] == 'candidate']}
    write_json(output / 'acceptance.json', summary)
    write_json(output / 'acceptance_summary.json',
               {'passed': bool(inputs_unchanged and all(p['passed'] for p in parity_rows)),
                'parity_passed': all(p['passed'] for p in parity_rows),
                'inputs_unchanged': inputs_unchanged,
                'targets': final_rows, 'old_regression': regression,
                'model_sha256': bundle.model_sha256, 'checkpoint_sha256': checkpoint_sha})
    print('ACCEPTANCE', json.dumps({'passed': inputs_unchanged and all(p['passed'] for p in parity_rows),
                                    'inputs_unchanged': inputs_unchanged}), flush=True)
    print('wrote', output / 'acceptance.json', flush=True)


if __name__ == '__main__':
    main()
