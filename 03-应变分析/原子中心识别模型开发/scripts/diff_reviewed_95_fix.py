"""Round-4 repair: machine diff of every reported metric between the pre-fix and fixed results."""

from pathlib import Path
import json
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from atom_center.storage import write_json                                # noqa: E402

RUN = ROOT / 'runs/reviewed-95-validation-20260910'
KEYS = ('tp', 'fp', 'fn', 'precision', 'recall', 'rmse_px')
SCOPES = ('full_image_full_csv', 'full_image_on_original_roi', 'roi_only_inference')


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def normalise(payload):
    """Flatten either the old hand-written shape or the new generated shape to one mapping."""
    flat = {}
    tolerance = payload.get('tolerance_diagnostics', {})
    for number in ('04', '05'):
        for scope in SCOPES:
            for label, entry in tolerance.items():
                head, _, tail = label.partition('_')
                if head.zfill(2) != number or tail != scope:
                    continue
                rows = entry if isinstance(entry, list) else [dict(value, radius_px=key)
                                                              for key, value in entry.items()]
                for row in rows:
                    radius = row.get('radius_px')
                    radius = float(str(radius).replace('px', '')) if radius is not None else None
                    if radius is None:
                        continue
                    for key in KEYS:
                        if key in row:
                            flat[f'{number}.{scope}.{radius:g}.{key}'] = row[key]
    verdict = payload.get('verdict', {}).get('per_image', {})
    for number, scopes in verdict.items():
        scopes = scopes if isinstance(scopes, dict) else {}
        for scope, row in scopes.items():
            if scope not in SCOPES or not isinstance(row, dict):
                continue
            for key in ('tp', 'fp', 'fn', 'precision', 'recall', 'p95_px'):
                if key in row:
                    flat[f'{int(number):02d}.{scope}.2.{key}'] = row[key]
    for number in ('04', '05'):
        grid = payload.get('local_grid_2x3', {}).get(number)
        if isinstance(grid, list):
            for cell in grid:
                label = cell.get('cell') or f'r{cell["row"]}c{cell["column"]}'
                for key in ('gt', 'predictions', 'tp', 'fp', 'fn', 'precision', 'recall', 'rmse_px'):
                    if key in cell:
                        flat[f'{number}.grid.{label}.{key}'] = cell[key]
    return flat


def main():
    before = json.loads((RUN / 'dsh_result_prefix01_snapshot.json').read_text(encoding='utf-8-sig'))
    after = json.loads((RUN / 'dsh_result.json').read_text(encoding='utf-8-sig'))
    old, new = normalise(before), normalise(after)
    changes = []
    for key in sorted(set(old) | set(new)):
        a, b = old.get(key), new.get(key)
        same = (a == b) if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) \
            else abs(a - b) <= 1e-15
        if not same:
            changes.append({'metric': key, 'before_fix': a, 'after_fix': b})
    record = json.loads((RUN / 'review_fixes_01.json').read_text(encoding='utf-8-sig'))
    record['updated_utc'] = utc_now()
    record['numeric_diff_against_pre_fix_result'] = {
        'method': 'the pre-fix dsh_result.json was snapshotted to dsh_result_prefix01_snapshot.json before '
                  'regeneration; both files are flattened to <image>.<scope>.<radius>.<metric> and compared',
        'compared_metrics': len(old | new), 'differences': len(changes), 'changes': changes}
    fix_two = next(item for item in record['fixes'] if item['id'] == 2)
    fix_two['verification']['recomputed_vs_previous_differences'] = len(changes)
    fix_two['other_corrected_values'] = changes
    fix_one = next(item for item in record['fixes'] if item['id'] == 1)
    fix_one['verification']['grid_gt_before_fix'] = {number: sum(cell.get('gt', 0) for cell in
                                                                 before['local_grid_2x3'][number])
                                                     for number in ('04', '05')}
    fix_one['verification']['grid_gt_after_fix'] = {number: sum(cell['gt'] for cell in
                                                                after['local_grid_2x3'][number])
                                                    for number in ('04', '05')}
    write_json(RUN / 'review_fixes_01.json', record)
    print('compared metrics:', len(old | new), 'differences:', len(changes))
    for change in changes:
        print('  ', change['metric'], change['before_fix'], '->', change['after_fix'])
    print('grid GT before:', fix_one['verification']['grid_gt_before_fix'],
          'after:', fix_one['verification']['grid_gt_after_fix'])
    print('updated', record['updated_utc'])


main()
