"""Round 3: overlay figure showing where the boundary candidates land, from the saved measurement.

Baseline points are drawn in red, added boundary-zone points in cyan. No inference is re-run.
"""
from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from atom_center.image_io import load_image, normalize_percentile           # noqa: E402
from atom_center.storage import write_json                                  # noqa: E402

RUN = ROOT / 'runs/boundary-optimization-20260910'
REVIEW = ROOT / 'runs/reviewed-test-20260910'
FONT = 'C:/Windows/Fonts/msyh.ttc'


def main():
    data = json.loads((RUN / 'boundary_candidates.json').read_text(encoding='utf-8-sig'))
    panels = []
    for number in ('4', '5'):
        record = json.loads((REVIEW / f'{int(number):02d}_comparison.json').read_text(encoding='utf-8-sig'))
        raw = np.asarray(load_image(record['source_image'], normalize=False,
                                    preserve_dtype=True).image, dtype=np.float64)
        base_image = Image.fromarray(np.rint(normalize_percentile(raw)*255).astype('uint8')).convert('RGB')
        for row in data['targets'][number]['candidates']:
            picture = base_image.copy()
            draw = ImageDraw.Draw(picture)
            for point in row['added_points']:
                x, y = point['xy']
                draw.ellipse((x-5, y-5, x+5, y+5), outline='#00e5ff', width=2)
            font = ImageFont.truetype(FONT, 20)
            canvas = Image.new('RGB', (max(picture.width, 980), picture.height + 120), '#101827')
            canvas.paste(picture, (0, 120))
            painter = ImageDraw.Draw(canvas)
            full = row['candidate_full_image']
            baseline = row['baseline_full_image']
            painter.text((12, 10), f"图 {number} · 边界延拓 {row['mode']} · 32 px", font=font, fill='white')
            painter.text((12, 38), f"青圈=新增候选 {row['points_added_vs_baseline']} 个"
                                   f"（匹配 {row['added_points_matched_within_2px']} / 误检 {row['added_points_false_positives']}）"
                                   f"；被替换的基线点 {row['baseline_points_lost_in_boundary_zone']} 个（均在边界带内）",
                         font=font, fill='#cbd5e1')
            painter.text((12, 66), f"整图基线 {baseline['tp']}/{baseline['fp']}/{baseline['fn']} "
                                   f"RMSE {baseline['rmse_px']:.4f} → 候选 {full['tp']}/{full['fp']}/{full['fn']} "
                                   f"RMSE {full['rmse_px']:.4f}", font=font,
                         fill='#ff8a8a' if full['f1'] < baseline['f1'] else '#7ee787')
            painter.text((12, 92), '青圈与既有预测位置几乎重合：延拓把同一边界原子重新检出并顶替了原输出',
                         font=font, fill='#94a3b8')
            panels.append(canvas)
    width = max(panel.width for panel in panels)
    sheet = Image.new('RGB', (width, sum(panel.height for panel in panels) + 16 * (len(panels)-1)), '#090e18')
    offset = 0
    for panel in panels:
        sheet.paste(panel, (0, offset))
        offset += panel.height + 16
    sheet.save(RUN / 'boundary_failure_overlay.png')
    write_json(RUN / 'boundary_failure_overlay.json',
               {'figure': str(RUN / 'boundary_failure_overlay.png'),
                'legend': {'cyan': 'added boundary-zone candidates', 'note': 'baseline points are the image content itself'},
                'panels': [{'target': number, 'mode': row['mode'],
                            'added': row['points_added_vs_baseline'],
                            'matched': row['added_points_matched_within_2px'],
                            'false_positives': row['added_points_false_positives'],
                            'baseline_points_replaced_in_zone': row['baseline_points_lost_in_boundary_zone']}
                           for number in ('4', '5')
                           for row in data['targets'][number]['candidates']]})
    print('wrote', RUN / 'boundary_failure_overlay.png')


main()
