"""Compact readout of the DSH selection comparison (no model runs)."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / 'runs/target-adaptation-20260910'
KEYS = ('tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'rmse_px', 'p95_px', 'min_region_recall')


def compact(metric):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metric.items() if k in KEYS}


def main():
    data = json.loads((BASE / 'dsh_selection/comparison.json').read_text(encoding='utf-8'))
    print('rule:', data['rule'])
    for record in data['records']:
        print('=====', record['kind'], record['label'], 'conf=', record['conf'],
              'sha=', record['checkpoint_sha256'][:12])
        for target in record.get('targets', []):
            print('  target', target['number'],
                  'full', json.dumps(compact(target['full_image'])),
                  '| sameROI', json.dumps(compact(target['same_original_roi'])),
                  '| roiOnly', json.dumps(compact(target.get('roi_only_inference', {}))))
        if record.get('old_regression'):
            for split in ('train', 'val'):
                print(f'  old_{split}', json.dumps(compact(record['old_regression'][split])))
        if record.get('worst_target_f1_full_image') is not None:
            print('  worst_full', round(record['worst_target_f1_full_image'], 5),
                  'worst_roi', round(record['worst_target_f1_same_roi'], 5))
    print('ranking:')
    for row in data['ranking']:
        print(' ', json.dumps(row))


if __name__ == '__main__':
    main()
