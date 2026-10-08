"""Compact readout of the round-3 boundary candidate measurement (no inference)."""

from pathlib import Path
import json

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


RUN = Path(__file__).resolve().parents[1] / 'runs/boundary-optimization-20260910'
KEYS = ('tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'rmse_px')


def compact(metrics):
    return {key: (round(metrics[key], 6) if isinstance(metrics[key], float) else metrics[key])
            for key in KEYS}


def main():
    data = json.loads((RUN / 'boundary_candidates.json').read_text(encoding='utf-8-sig'))
    print('model', data['model_sha256'], 'padding', data['padding_px'], 'conf', data['confidence'])
    for number in sorted(data['targets']):
        target = data['targets'][number]
        print(f"===== target {number} shape={target['image_shape']} "
              f"references={target['reference_count_all_csv']}")
        for row in target['candidates']:
            print(f"  [{row['mode']}] final={row['final_points']} added={row['points_added_vs_baseline']} "
                  f"(matched {row['added_points_matched_within_2px']}, FP {row['added_points_false_positives']}) "
                  f"lost interior={row['baseline_points_lost_interior']} "
                  f"zone={row['baseline_points_lost_in_boundary_zone']}")
            print('      full base', compact(row['baseline_full_image']))
            print('      full cand', compact(row['candidate_full_image']))
            print('      roi  base', compact(row['baseline_same_original_roi']))
            print('      roi  cand', compact(row['candidate_same_original_roi']))
            for point in row['added_points'][:10]:
                print("        added", point)
            for point in row['baseline_points_lost'][:6]:
                print("        lost ", point)
    print('===== decisions')
    for mode, decision in data['decisions'].items():
        print(f"  {mode}: passes={decision['passes_targets']} improved={decision['improved_images']} "
              f"regressed={decision['regressed_images']}")
        for key, value in decision['checks'].items():
            print(f"    {key}: {value}")


main()
