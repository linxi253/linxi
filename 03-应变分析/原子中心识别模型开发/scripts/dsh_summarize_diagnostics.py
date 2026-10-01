"""Compact readout of existing target-adaptation diagnostics (no model runs)."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / 'runs/target-adaptation-20260910'
KEYS = ('tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'rmse_px', 'p95_px',
        'match_distance_px', 'reference_count', 'prediction_count')


def compact(metric):
    return {k: metric[k] for k in KEYS if k in metric}


def main():
    for name in ('refinement_diagnostic.json', 'threshold_diagnostic.json'):
        d = json.loads((BASE / name).read_text(encoding='utf-8'))
        print(f'===== {name} method={d.get("method")} checkpoint={Path(d["checkpoint"]).name} sha={d["sha256"][:12]}')
        for row in d['rows']:
            tag = {k: v for k, v in row.items() if k not in ('full_image', 'same_original_roi', 'targets', 'row')}
            print(' ROW', json.dumps(tag, ensure_ascii=False))
            for key in ('targets',):
                for t in row.get(key, []):
                    print('   target', json.dumps({k: v for k, v in t.items() if not isinstance(v, (list, dict))}, ensure_ascii=False))
                    for scope in ('full_image', 'same_original_roi'):
                        print(f'     {scope}', json.dumps(compact(t[scope]), ensure_ascii=False))
            for key in ('full_image', 'same_original_roi'):
                if key in row:
                    print(f'   {key}', json.dumps(compact(row[key]), ensure_ascii=False))
            if isinstance(row.get('row'), dict):
                print('   saved_row', json.dumps(row['row'], ensure_ascii=False)[:400])


if __name__ == '__main__':
    main()
