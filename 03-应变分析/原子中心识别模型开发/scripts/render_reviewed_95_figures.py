"""Reviewed-95 figures (repair 01): green prediction circles, FP/FN comparison, diagnostics.

Reads only the saved CLI coordinates and the recomputed validation record. Repair changes:
the ROI crop now includes the full inclusive ROI so the two 04 border misses are visible, the local
grid panel shows the corrected float-bounds grid with its coverage check, and the overview stacks
both comparison panels at equal readable width.
"""
from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
from atom_center.image_io import load_image, normalize_percentile           # noqa: E402
from atom_center.metrics import match_points                               # noqa: E402
from atom_center.storage import write_json                                 # noqa: E402
from evaluate_reviewed_test_answers import inside                          # noqa: E402

RUN = ROOT / 'runs/reviewed-95-validation-20260910'
FIG = RUN / 'figures'
RAW = RUN / 'raw_outputs'
REVIEW = ROOT / 'runs/reviewed-test-20260910'
FONT_PATH = 'C:/Windows/Fonts/msyh.ttc'


def load_points(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    return np.asarray(payload['points_xy'], dtype=float).reshape(-1, 2)


def wrap(draw, text, xy, font, fill, max_width):
    x, y = xy
    line = ''
    for char in text:
        if line and draw.textlength(line + char, font=font) > max_width:
            draw.text((x, y), line, font=font, fill=fill)
            line, y = char, y + font.size + 6
        else:
            line += char
    draw.text((x, y), line, font=font, fill=fill)
    return y + font.size + 6


def display_crop(roi, shape):
    """Crop that covers the whole inclusive float ROI (the previous version stopped one pixel short)."""
    height, width = shape
    left = max(0, int(np.floor(roi[0])))
    top = max(0, int(np.floor(roi[1])))
    right = min(width, int(np.ceil(roi[2])) + 1)
    bottom = min(height, int(np.ceil(roi[3])) + 1)
    return left, top, right, bottom


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    validation = json.loads((RUN / 'validation.json').read_text(encoding='utf-8-sig'))
    summary = json.loads((RUN / 'metrics_summary.json').read_text(encoding='utf-8-sig'))
    records, comparison_panels = {}, []
    for number in ('4', '5'):
        key = f'{int(number):02d}'
        entry = validation['targets'][number]
        record = json.loads((REVIEW / f'{key}_comparison.json').read_text(encoding='utf-8-sig'))
        raw = np.asarray(load_image(record['source_image'], normalize=False,
                                    preserve_dtype=True).image, dtype=np.float64)
        truth_all = np.asarray(record['all_csv_truth_xy'], dtype=float)
        roi = np.asarray(entry['roi_xyxy_inclusive_float'], dtype=float)
        points = load_points(RAW / f'{key}_full_image_run1.json')
        distances = match_points(points, truth_all, max_distance_px=2.).distances_px
        rows = entry['scopes']['full_image_full_csv']['tolerance_metrics']
        roi_rows = entry['scopes']['full_image_on_original_roi']['tolerance_metrics']
        roi_only_rows = entry['scopes']['roi_only_inference']['tolerance_metrics']
        get = lambda table, radius: next(row for row in table if row['radius_px'] == radius)
        full, roi_row = get(rows, 2.0), get(roi_rows, 2.0)
        grid = entry['local_grid_on_original_roi']
        coverage = entry['grid_coverage']
        font = ImageFont.truetype(FONT_PATH, 21)
        small = ImageFont.truetype(FONT_PATH, 18)

        picture = Image.fromarray(np.rint(normalize_percentile(raw) * 255).astype('uint8')).convert('RGB')
        draw = ImageDraw.Draw(picture)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline='#26e7a3', width=2)
        canvas = Image.new('RGB', (max(picture.width, 1180), picture.height + 150), '#101827')
        canvas.paste(picture, ((max(picture.width, 1180) - picture.width) // 2, 150))
        painter = ImageDraw.Draw(canvas)
        y = wrap(painter, f'图 {number} · 部署 ONNX 实际输出 · 纯预测绿圈（未叠加人工答案）', (12, 10), font, 'white', 1150)
        y = wrap(painter, '阈值 0.1 · gaussian11/max_shift4 · 预测 '
                          f'{entry["prediction_count_full_image"]} 点 · 人工答案参与过训练，这是拟合表现而非独立测试',
                 (12, y), small, '#cbd5e1', 1150)
        wrap(painter, f'95% 主口径（2px，整张 CSV）：TP{full["tp"]} FP{full["fp"]} FN{full["fn"]} '
                      f'P={full["precision"]:.4%} R={full["recall"]:.4%} RMSE={full["rmse_px"]:.6f}px',
             (12, y), small, '#7ee787' if full['passes_both_95'] else '#ff8a8a', 1150)
        canvas.save(FIG / f'{number}_predictions_green.png')

        crop = display_crop(roi, raw.shape)
        picture = Image.fromarray(np.rint(normalize_percentile(raw) * 255).astype('uint8')
                                  ).convert('RGB').crop(crop)
        draw = ImageDraw.Draw(picture)
        offset = np.array([crop[0], crop[1]])
        pred_roi = points[inside(points, roi)]
        truth_roi = truth_all[inside(truth_all, roi)]
        roi_matches = match_points(pred_roi, truth_roi, max_distance_px=2.)
        for index in roi_matches.unmatched_ground_truth_indices:
            x, y = truth_roi[index] - offset
            draw.line((x - 5, y, x + 5, y), fill='#ff5e64', width=3)
            draw.line((x, y - 5, x, y + 5), fill='#ff5e64', width=3)
        for index in roi_matches.unmatched_prediction_indices:
            x, y = pred_roi[index] - offset
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline='#ffce54', width=3)
        for p_index, _ in roi_matches.matched_indices:
            x, y = pred_roi[p_index] - offset
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline='#26e7a3', width=2)
        width = max(picture.width, 1180)
        canvas = Image.new('RGB', (width, picture.height + 200), '#101827')
        canvas.paste(picture, ((width - picture.width) // 2, 200))
        painter = ImageDraw.Draw(canvas)
        y = wrap(painter, f'图 {number} · FP/FN 对照（整图推理，按冻结原选区评分，已含全部边界答案）',
                 (12, 10), font, 'white', width - 24)
        y = wrap(painter, f'原选区浮点边界 x[{roi[0]:.3f},{roi[2]:.3f}] y[{roi[1]:.3f},{roi[3]:.3f}] · '
                          f'人工点 {len(truth_roi)} · 预测 {len(pred_roi)} · TP{roi_row["tp"]} FP{roi_row["fp"]} '
                          f'FN{roi_row["fn"]}', (12, y), small, '#cbd5e1', width - 24)
        y = wrap(painter, f'P={roi_row["precision"]:.4%} R={roi_row["recall"]:.4%} '
                          f'RMSE={roi_row["rmse_px"]:.6f}px · 显示裁剪按浮点边界向外取整最多 1px（仅显示用，'
                          f'评测范围以上面的浮点边界为准）', (12, y), small,
                 '#7ee787' if roi_row['passes_both_95'] else '#ff8a8a', width - 24)
        wrap(painter, '绿圈=匹配 红十字=漏检(人工点) 黄圈=误检(预测)；答案为人工校对且未做任何移动',
             (12, y), small, '#94a3b8', width - 24)
        canvas.save(FIG / f'{number}_fp_fn_comparison.png')
        comparison_panels.append(canvas)

        width, height = 1420, 680
        canvas = Image.new('RGB', (width, height), '#101827')
        painter = ImageDraw.Draw(canvas)
        painter.text((12, 10), f'图 {number} · 定位偏差与 2×3 局部网格（修正版：浮点边界，边界答案不再丢失）',
                     font=font, fill='white')
        counts, edges = np.histogram(distances, bins=np.arange(0, 2.2, 0.2))
        chart_top, chart_bottom, chart_left, chart_width = 100, 260, 60, 430
        painter.rectangle((chart_left, chart_top, chart_left + chart_width, chart_bottom), outline='#334155')
        peak = max(int(counts.max()), 1)
        for index, count in enumerate(counts):
            x0 = chart_left + index * chart_width / len(counts)
            x1 = chart_left + (index + 1) * chart_width / len(counts)
            top = chart_bottom - (chart_bottom - chart_top) * count / peak
            painter.rectangle((x0 + 1, top, x1 - 1, chart_bottom), fill='#38bdf8')
            painter.text((x0 + 2, chart_bottom + 4), f'{edges[index]:.1f}', font=small, fill='#94a3b8')
        painter.text((chart_left, chart_top - 26), f'匹配点距离直方图（0–2px，共 {len(distances)} 对）',
                     font=small, fill='#cbd5e1')
        painter.text((chart_left, chart_bottom + 30), 'px', font=small, fill='#94a3b8')
        for index, (label, row) in enumerate((
                ('2px（主口径）', get(rows, 2.0)), ('1px（诊断）', get(rows, 1.0)), ('0.5px（诊断）', get(rows, 0.5)))):
            colour = '#7ee787' if row['passes_both_95'] else '#ff8a8a'
            painter.text((530, 96 + index * 34),
                         f'{label}：整张CSV P {row["precision"]:.4%} / R {row["recall"]:.4%} · '
                         f'TP{row["tp"]} FP{row["fp"]} FN{row["fn"]} · RMSE {row["rmse_px"]:.6f}px',
                         font=small, fill=colour)
        painter.text((530, 204), f'原选区 2px：P {roi_row["precision"]:.4%} / R {roi_row["recall"]:.4%} · '
                                 f'RMSE {roi_row["rmse_px"]:.6f}px', font=small, fill='#7ee787')
        painter.text((530, 234), f'仅选区推理 2px：P {get(roi_only_rows, 2.0)["precision"]:.4%} / '
                                 f'R {get(roi_only_rows, 2.0)["recall"]:.4%} · '
                                 f'RMSE {get(roi_only_rows, 2.0)["rmse_px"]:.6f}px', font=small, fill='#7ee787')
        painter.text((530, 272), f"网格覆盖：答案 {coverage['roi_truth_total']}→{coverage['grid_truth_sum']}，"
                                 f"预测 {coverage['roi_predictions_total']}→{coverage['grid_predictions_sum']}，"
                                 f"每点恰好一格：{coverage['passed']}", font=small, fill='#e2e8f0')
        painter.text((530, 302), '0.5/1px 只作诊断，不用于宣称亚像素精度通过', font=small, fill='#94a3b8')
        painter.text((530, 330), '格内漏检按实际坐标列出，边界点不再被丢弃', font=small, fill='#94a3b8')
        painter.text((12, 340), '2×3 局部网格（浮点边界；最后一列/最后一行包含 ROI 最大边界）', font=font, fill='white')
        for index, cell in enumerate(grid):
            row, column = divmod(index, 3)
            x = 40 + column * 460
            y = 382 + row * 148
            painter.rectangle((x, y, x + 430, y + 130), outline='#334155')
            bounds = cell['bounds_xyxy_float']
            painter.text((x + 10, y + 6), f"r{cell['row']}c{cell['column']}  x[{bounds[0]:.2f},{bounds[2]:.2f}] "
                                          f"y[{bounds[1]:.2f},{bounds[3]:.2f}]", font=small, fill='#94a3b8')
            flag = '（小样本）' if cell['small_sample'] else ''
            painter.text((x + 10, y + 32), f"GT {cell['gt']} · 预测 {cell['predictions']} · TP {cell['tp']} "
                                          f"FP {cell['fp']} FN {cell['fn']}{flag}", font=small, fill='#e2e8f0')
            colour = '#7ee787' if cell['passes_both_95'] else '#ff8a8a'
            rmse = '—' if cell['rmse_px'] is None else f"{cell['rmse_px']:.4f}px"
            painter.text((x + 10, y + 58), f"P {cell['precision']:.4%} · R {cell['recall']:.4%} · RMSE {rmse}",
                         font=small, fill=colour)
            missed = [item for item in entry['unmatched_answers_by_cell']
                      if item['cell'] == f"r{cell['row']}c{cell['column']}"]
            if missed:
                summary_text = '、'.join(f"({item['answer_xy'][0]:.1f},{item['answer_xy'][1]:.1f})"
                                         for item in missed[:3])
                more = f" 等 {len(missed)} 个" if len(missed) > 3 else ''
                wrap(painter, f"格内漏检：{summary_text}{more}", (x + 10, y + 84), small, '#ff8a8a', 410)
        canvas.save(FIG / f'{number}_localisation_diagnostics.png')
        records[number] = {'prediction_count': int(len(points)),
                           'matched_distance_median_px': float(np.median(distances)) if len(distances) else None,
                           'matched_distance_p95_px': float(np.percentile(distances, 95)) if len(distances) else None,
                           'grid_truth_sum': coverage['grid_truth_sum'],
                           'grid_coverage_passed': coverage['passed'],
                           'figures': [f'{number}_predictions_green.png', f'{number}_fp_fn_comparison.png',
                                       f'{number}_localisation_diagnostics.png']}

    panel_width = 1180
    scaled = []
    for panel in comparison_panels:
        ratio = panel_width / panel.width
        scaled.append(panel.resize((panel_width, int(round(panel.height * ratio))), Image.LANCZOS))
    overview = Image.new('RGB', (panel_width + 40, 150 + sum(p.height for p in scaled) + 20 * len(scaled)), '#090e18')
    painter = ImageDraw.Draw(overview)
    painter.text((20, 14), '人工校对组 04/05 · 95% 验收对照图集（2px 主口径，ONNX 实际坐标）',
                 font=ImageFont.truetype(FONT_PATH, 30), fill='white')
    painter.text((20, 56), '两图答案均参与过监督适配训练：以下为拟合表现，不是独立测试，也不是应变科学精度验收',
                 font=ImageFont.truetype(FONT_PATH, 21), fill='#cbd5e1')
    painter.text((20, 88), f'模型 {validation["model_sha256"]}', font=ImageFont.truetype(FONT_PATH, 18), fill='#94a3b8')
    painter.text((20, 114), '每张面板等宽显示；原选区浮点边界见图内标题，显示裁剪最多向外取整 1px（仅显示用）',
                 font=ImageFont.truetype(FONT_PATH, 18), fill='#94a3b8')
    offset = 150
    for panel in scaled:
        overview.paste(panel, (20, offset))
        offset += panel.height + 20
    overview.save(FIG / 'overview.png')
    write_json(FIG / 'figures.json', {
        'purpose': 'reviewed-95 validation figures (repair 01)', 'updated_utc': summary['updated_utc'],
        'model_sha256': validation['model_sha256'],
        'repair_notes': ['the FP/FN crop now covers the full inclusive ROI so the two 04 border misses are visible',
                         'the grid panel shows the corrected float-bounds grid with its coverage check',
                         'the overview stacks both panels at equal readable width'],
        'note': 'answers were used in supervised adaptation training; fit not independent test',
        'per_target': records})
    print(json.dumps(records, ensure_ascii=False))
    print('wrote', FIG / 'figures.json')


main()
