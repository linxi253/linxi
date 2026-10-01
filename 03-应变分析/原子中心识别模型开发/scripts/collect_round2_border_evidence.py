"""Round 2 step 3d: border evidence for the answers the frozen network never proposes.

For each unmatched answer this reports the highest raw network score inside a small radius, which
separates "the network produced a weak, cut-off detection" from "the network produced nothing".
Analysis only: no threshold is changed and no configuration is selected here.
"""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from atom_center.backends import onnx_pipeline                            # noqa: E402
from atom_center.image_io import load_image                               # noqa: E402
from atom_center.storage import write_json                                # noqa: E402

RUN = ROOT / 'runs/training-update-20260910'
REVIEW = ROOT / 'runs/reviewed-test-20260910'
MANIFEST = ROOT / 'runs/target-adaptation-20260910/target_adaptation_onnx/model_manifest.json'


def main():
    pipeline = onnx_pipeline(MANIFEST)
    backend = pipeline.backend
    config = pipeline.config
    report = {'purpose': 'describe why some answers are never proposed by the frozen network',
              'model_sha256': backend.model_sha256, 'frozen_conf': backend.inference['conf'],
              'targets': {}}
    for number in (4, 5):
        record = json.loads((REVIEW / f'{number:02d}_comparison.json').read_text(encoding='utf-8-sig'))
        raw = np.asarray(load_image(record['source_image'], normalize=False,
                                    preserve_dtype=True).image, dtype=np.float64)
        truth = np.asarray(record['all_csv_truth_xy'], dtype=float)
        result = pipeline.detect(raw)
        distance = cKDTree(result.points).query(truth)[0]
        unmatched = np.flatnonzero(distance > 2.)

        # Collect every raw network candidate before thresholding, tile by tile.
        from atom_center.geometry import generate_tiles
        candidates = []
        for tile in generate_tiles(raw.shape, tile_size=config.tile_size, overlap=config.tile_overlap):
            tensor, transform = backend.prepare(raw[tile.slices])
            output = np.asarray(backend.raw(tensor))
            rows = output[0].T.astype(np.float64)
            points = transform.inverse_centers(rows[:, :2]) + [tile.x0, tile.y0]
            keep = (rows[:, 4] > 0.) & (rows[:, 2] > 0) & (rows[:, 3] > 0)
            candidates.append(np.concatenate([points[keep], rows[keep, 4][:, None]], axis=1))
        candidates = np.concatenate(candidates) if candidates else np.empty((0, 3))
        tree = cKDTree(candidates[:, :2])

        rows = []
        for index in unmatched:
            point = truth[index]
            near = tree.query_ball_point(point, 6.)
            best = float(candidates[near, 2].max()) if near else 0.
            best_anywhere = tree.query(point, k=1)
            rows.append({
                'answer_xy': point.round(2).tolist(),
                'nearest_final_prediction_px': round(float(distance[index]), 2),
                'max_network_score_within_6px': round(best, 6),
                'above_frozen_conf_within_6px': bool(best >= backend.inference['conf']),
                'nearest_raw_candidate_px': round(float(best_anywhere[0]), 2),
                'nearest_raw_candidate_score': round(float(candidates[best_anywhere[1], 2]), 6),
                'at_image_border_20px': bool(point[0] < 20 or point[0] > raw.shape[1] - 20
                                             or point[1] < 20 or point[1] > raw.shape[0] - 20),
                'local_max_offset_px': None})
        report['targets'][str(number)] = {
            'image_shape_hw': list(raw.shape), 'unmatched_answers': len(rows),
            'answers_with_weak_but_present_candidate': sum(1 for row in rows
                                                           if 0 < row['max_network_score_within_6px']
                                                           < backend.inference['conf']),
            'answers_with_no_candidate_above_zero': sum(1 for row in rows
                                                        if row['max_network_score_within_6px'] == 0),
            'rows': rows}
        print(number, json.dumps({key: report['targets'][str(number)][key] for key in
                                  ('unmatched_answers', 'answers_with_weak_but_present_candidate',
                                   'answers_with_no_candidate_above_zero')}), flush=True)
        for row in rows[:6]:
            print('   ', row, flush=True)
    write_json(RUN / 'border_evidence.json', report)
    print('wrote', RUN / 'border_evidence.json')


main()
