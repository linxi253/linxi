"""Round 2 step 3b: does the centre-fit quality separate good from bad final positions?

Runs the frozen pipeline and the same network without refinement, then joins every final point
with its pre-refinement candidate, the fit diagnostics and (analysis only) the human answer
distance. Inference reads no answers; the answer join happens afterwards for description.
"""
from pathlib import Path
from dataclasses import replace
import json
import sys
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from atom_center.backends import onnx_pipeline                            # noqa: E402
from atom_center.image_io import load_image                               # noqa: E402
from atom_center.storage import write_json                                # noqa: E402
from evaluate_reviewed_test_answers import score, inside                  # noqa: E402

RUN = ROOT / 'runs/training-update-20260910'
REVIEW = ROOT / 'runs/reviewed-test-20260910'
MANIFEST = ROOT / 'runs/target-adaptation-20260910/target_adaptation_onnx/model_manifest.json'


def main():
    refined_pipeline = onnx_pipeline(MANIFEST)
    unrefined_pipeline = onnx_pipeline(MANIFEST)
    unrefined_pipeline.config = replace(unrefined_pipeline.config, refine=False)
    report = {'purpose': 'quantify whether fit diagnostics identify harmful refinement',
              'model_sha256': refined_pipeline.backend.model_sha256, 'targets': {}}
    for number in (4, 5):
        record = json.loads((REVIEW / f'{number:02d}_comparison.json').read_text(encoding='utf-8-sig'))
        raw = np.asarray(load_image(record['source_image'], normalize=False,
                                    preserve_dtype=True).image, dtype=np.float64)
        truth = np.asarray(record['all_csv_truth_xy'], dtype=float)
        roi = np.asarray(record['roi_xyxy_inclusive'], dtype=float)
        final = refined_pipeline.detect(raw)
        rough = unrefined_pipeline.detect(raw)
        quality = final.metadata['point_quality']
        pre_index = cKDTree(rough.points).query(final.points)[0]
        pre_index = cKDTree(rough.points).query(final.points)[1]
        pre_distance = np.linalg.norm(rough.points[pre_index] - final.points, axis=1)
        truth_distance_final = cKDTree(truth).query(final.points)[0]
        truth_distance_pre = cKDTree(truth).query(rough.points[pre_index])[0]
        rmse = np.array([entry.get('normalized_rmse') if entry.get('normalized_rmse') is not None else np.nan
                         for entry in quality], dtype=float)
        rows = [{'index': int(index), 'final_xy': final.points[index].round(4).tolist(),
                 'pre_xy': rough.points[pre_index[index]].round(4).tolist(),
                 'refinement_shift_px': round(float(pre_distance[index]), 4),
                 'status': quality[index].get('status'),
                 'normalized_rmse': None if np.isnan(rmse[index]) else round(float(rmse[index]), 5),
                 'answer_distance_final_px': round(float(truth_distance_final[index]), 4),
                 'answer_distance_pre_px': round(float(truth_distance_pre[index]), 4),
                 'inside_roi': bool(inside(final.points[index].reshape(1, 2), roi)[0])}
                for index in range(len(final.points))]
        moved_out = [row for row in rows if row['answer_distance_pre_px'] <= 2. < row['answer_distance_final_px']]
        moved_in = [row for row in rows if row['answer_distance_final_px'] <= 2. < row['answer_distance_pre_px']]
        finite = rmse[np.isfinite(rmse)]
        report['targets'][str(number)] = {
            'point_count': len(rows),
            'fit_rmse_quantiles': {name: (round(float(value), 5) if np.isfinite(value) else None)
                                   for name, value in (('min', finite.min() if finite.size else np.nan),
                                                       ('p25', np.percentile(finite, 25) if finite.size else np.nan),
                                                       ('median', np.median(finite) if finite.size else np.nan),
                                                       ('p75', np.percentile(finite, 75) if finite.size else np.nan),
                                                       ('p90', np.percentile(finite, 90) if finite.size else np.nan),
                                                       ('max', finite.max() if finite.size else np.nan))},
            'refined_by_quality': {
                'median_rmse_when_final_within_2px': round(float(np.nanmedian(
                    [row['normalized_rmse'] for row in rows
                     if row['answer_distance_final_px'] <= 2. and row['normalized_rmse'] is not None])), 5),
                'median_rmse_when_final_outside_2px': (round(float(np.nanmedian(
                    [row['normalized_rmse'] for row in rows
                     if row['answer_distance_final_px'] > 2. and row['normalized_rmse'] is not None])), 5)
                    if any(row['answer_distance_final_px'] > 2. and row['normalized_rmse'] is not None
                           for row in rows) else None)},
            'refinement_moved_out_of_tolerance': moved_out,
            'refinement_moved_into_tolerance': moved_in,
            'worst_points': sorted(rows, key=lambda row: -row['answer_distance_final_px'])[:25],
        }
        print(number, json.dumps({'points': len(rows), 'moved_out': len(moved_out), 'moved_in': len(moved_in),
                                  'rmse_median': report['targets'][str(number)]['fit_rmse_quantiles']['median'],
                                  'rmse_p90': report['targets'][str(number)]['fit_rmse_quantiles']['p90'],
                                  'rmse_matched_median': report['targets'][str(number)]['refined_by_quality']['median_rmse_when_final_within_2px'],
                                  'rmse_unmatched_median': report['targets'][str(number)]['refined_by_quality']['median_rmse_when_final_outside_2px']}), flush=True)
        for row in moved_out:
            print('   moved_out', row['final_xy'], 'shift', row['refinement_shift_px'],
                  'rmse', row['normalized_rmse'], 'status', row['status'],
                  'd_pre', row['answer_distance_pre_px'], 'd_final', row['answer_distance_final_px'], flush=True)
    write_json(RUN / 'refinement_quality.json', report)
    print('wrote', RUN / 'refinement_quality.json')


main()
