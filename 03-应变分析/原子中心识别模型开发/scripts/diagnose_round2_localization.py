"""Round 2 step 3a: locate where the remaining 04/05 errors are produced.

Compares the frozen round-1 pipeline with the same network without refinement, so a residual
2-5 px error can be attributed to detection or to the centre fit. Adds local-window intensity
diagnostics (background, contrast, saturation, peak splitting) and a zoomed figure for review.
No answer coordinates enter any inference function; the analysis section only reads them to
describe the errors afterwards.
"""
from pathlib import Path
from dataclasses import replace
import json
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from atom_center.backends import onnx_pipeline                            # noqa: E402
from atom_center.image_io import load_image, normalize_percentile         # noqa: E402
from atom_center.metrics import match_points                              # noqa: E402
from atom_center.storage import write_json                                # noqa: E402
from evaluate_reviewed_test_answers import score, inside                  # noqa: E402

RUN = ROOT / 'runs/training-update-20260910'
REVIEW = ROOT / 'runs/reviewed-test-20260910'
MANIFEST = ROOT / 'runs/target-adaptation-20260910/target_adaptation_onnx/model_manifest.json'
FONT = 'C:/Windows/Fonts/msyh.ttc'
HALF = 5


def local_window(array, x, y, half=HALF):
    """Unmodified local window statistics around a reported centre."""
    cx, cy = int(round(float(x))), int(round(float(y)))
    height, width = array.shape
    clipped = (cx - half < 0 or cy - half < 0 or cx + half >= width or cy + half >= height)
    x0, y0 = max(0, cx - half), max(0, cy - half)
    patch = array[y0:min(height, cy + half + 1), x0:min(width, cx + half + 1)]
    values = patch.astype(np.float64)
    ring = np.concatenate([values[0], values[-1], values[:, 0], values[:, -1]])
    peak = float(values.max())
    background = float(np.median(ring))
    peak_y, peak_x = np.unravel_index(int(np.argmax(values)), values.shape)
    # A second, separated maximum indicates a split or doubled peak inside the window.
    masked = values.copy()
    masked[max(0, peak_y - 2):peak_y + 3, max(0, peak_x - 2):peak_x + 3] = -np.inf
    second = float(masked.max()) if np.isfinite(masked).any() else float('nan')
    return {'clipped_by_image_border': bool(clipped),
            'peak': peak, 'background_median': background, 'contrast': peak - background,
            'peak_offset_px': [float(peak_x - half), float(peak_y - half)],
            'second_peak_ratio': (float(second / peak) if peak else None),
            'saturated_in_window': int((values >= peak).sum())}


def panel(array, truth, points_before, points_after, title, target, half=26):
    """Zoom on one truth point: network candidate (yellow), refined (green), answer (red cross)."""
    cx, cy = int(round(float(truth[0]))), int(round(float(truth[1])))
    height, width = array.shape
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(width, cx + half + 1), min(height, cy + half + 1)
    crop = array[y0:y1, x0:x1].astype(np.float64)
    crop = (crop - crop.min()) / max(crop.ptp(), 1e-9)
    scale = 6
    picture = Image.fromarray((crop * 255).astype(np.uint8)).convert('RGB')
    picture = picture.resize((picture.width * scale, picture.height * scale), Image.NEAREST)
    draw = ImageDraw.Draw(picture)

    def mark(point, colour, radius):
        px, py = (float(point[0]) - x0) * scale, (float(point[1]) - y0) * scale
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=colour, width=2)

    for point in points_before:
        mark(point, '#ffd166', 5)
    for point in points_after:
        mark(point, '#26e7a3', 9)
    mark(truth, '#ff5e64', 14)
    font = ImageFont.truetype(FONT, 17)
    canvas = Image.new('RGB', (max(picture.width, 360), picture.height + 52), '#101827')
    canvas.paste(picture, (0, 52))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), title, font=font, fill='white')
    draw.text((8, 27), '红=人工答案 绿=最终输出 黄=精修前网络候选', font=font, fill='#cbd5e1')
    canvas.save(target)
    return canvas


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    figure_dir = RUN / 'localization_diagnosis'
    figure_dir.mkdir(parents=True, exist_ok=True)
    refined_pipeline = onnx_pipeline(MANIFEST)
    unrefined_pipeline = onnx_pipeline(MANIFEST)
    unrefined_pipeline.config = replace(unrefined_pipeline.config, refine=False)
    report = {'purpose': 'attribute the remaining 04/05 residual errors to detection or centre fit',
              'model_manifest': str(MANIFEST.resolve()),
              'model_sha256': refined_pipeline.backend.model_sha256,
              'refined_config': {'refine': True, 'method': refined_pipeline.config.refinement_method,
                                 'window': refined_pipeline.config.refinement_window,
                                 'max_shift_px': refined_pipeline.config.refinement_max_shift_px,
                                 'merge_distance_px': refined_pipeline.config.merge_distance_px},
              'targets': {}}

    for number in (4, 5):
        truth_record = json.loads((REVIEW / f'{number:02d}_comparison.json').read_text(encoding='utf-8-sig'))
        raw = np.asarray(load_image(truth_record['source_image'], normalize=False,
                                    preserve_dtype=True).image, dtype=np.float64)
        truth = np.asarray(truth_record['all_csv_truth_xy'], dtype=float)
        roi = np.asarray(truth_record['roi_xyxy_inclusive'], dtype=float)
        refined = refined_pipeline.detect(raw)
        unrefined = unrefined_pipeline.detect(raw)
        after, before = refined.points, unrefined.points
        quality = refined.metadata['point_quality']
        metrics_after = score(after, truth, 2.)
        metrics_before = score(before, truth, 2.)
        unmatched_truth = metrics_after['unmatched_truth_indices']
        unmatched_prediction = set(metrics_after['unmatched_prediction_indices'])
        before_tree, after_tree = cKDTree(before), cKDTree(after)
        statuses = {}
        for entry in quality:
            statuses[str(entry.get('status'))] = statuses.get(str(entry.get('status')), 0) + 1

        problems = []
        for index in unmatched_truth:
            point = truth[index]
            distance_after, index_after = after_tree.query(point)
            distance_before, index_before = before_tree.query(point)
            problems.append({
                'truth_xy': point.round(3).tolist(),
                'nearest_final_xy': after[index_after].round(3).tolist(),
                'nearest_final_distance_px': round(float(distance_after), 3),
                'nearest_final_is_unmatched': bool(index_after in unmatched_prediction),
                'nearest_final_status': quality[index_after].get('status'),
                'nearest_final_shift_px': quality[index_after].get('shift_px'),
                'nearest_final_proposed_shift_px': quality[index_after].get('proposed_shift_px'),
                'nearest_pre_refinement_xy': before[index_before].round(3).tolist(),
                'nearest_pre_refinement_distance_px': round(float(distance_before), 3),
                'nearest_pre_refinement_within_2px': bool(distance_before <= 2.),
                'window': local_window(raw, point[0], point[1]),
                'inside_roi': bool(inside(point.reshape(1, 2), roi)[0]),
            })
        # Every point that the network candidate already placed within 2 px but the final output did not.
        moved_away = [row for row in problems
                      if row['nearest_pre_refinement_within_2px'] and row['nearest_final_distance_px'] > 2.]
        report['targets'][str(number)] = {
            'prediction_count_final': int(len(after)), 'prediction_count_pre_refinement': int(len(before)),
            'metrics_final_full_image': {key: metrics_after[key] for key in
                                         ('tp', 'fp', 'fn', 'precision', 'recall', 'rmse_px', 'p95_px')},
            'metrics_pre_refinement_full_image': {key: metrics_before[key] for key in
                                                  ('tp', 'fp', 'fn', 'precision', 'recall', 'rmse_px', 'p95_px')},
            'refinement_status_counts': statuses,
            'duplicates_removed_after_refinement': refined.metadata['duplicates_removed_after_refinement'],
            'outside_roi_after_refinement': refined.metadata['outside_roi_after_refinement'],
            'tile_count': refined.metadata['tile_count'],
            'unmatched_answers': problems,
            'errors_introduced_by_refinement': moved_away,
            'errors_in_detection': [row for row in problems if not row['nearest_pre_refinement_within_2px']],
        }
        panels = []
        for row in problems:
            truth_xy = np.asarray(row['truth_xy'])
            nearby_before = before[cKDTree(before).query_ball_point(truth_xy, 12.)]
            nearby_after = after[cKDTree(after).query_ball_point(truth_xy, 12.)]
            panels.append(panel(raw, truth_xy, nearby_before, nearby_after,
                                f'图 {number:02d} · 答案 {truth_xy[0]:.1f},{truth_xy[1]:.1f} · '
                                f'精修前 {row["nearest_pre_refinement_distance_px"]:.2f}px → '
                                f'最终 {row["nearest_final_distance_px"]:.2f}px',
                                figure_dir / f'{number:02d}_residual_{len(panels) + 1:02d}.png'))
        if panels:
            cols = min(4, len(panels))
            rows = (len(panels) + cols - 1) // cols
            width = max(p.width for p in panels)
            height = max(p.height for p in panels)
            sheet = Image.new('RGB', (cols * width, rows * height), '#090e18')
            for index, item in enumerate(panels):
                sheet.paste(item, ((index % cols) * width, (index // cols) * height))
            sheet.save(figure_dir / f'{number:02d}_all_residuals.png')
        print(number, json.dumps({'final': {k: report['targets'][str(number)]['metrics_final_full_image'][k]
                                            for k in ('tp', 'fp', 'fn')},
                                  'pre_refinement': {k: report['targets'][str(number)]['metrics_pre_refinement_full_image'][k]
                                                     for k in ('tp', 'fp', 'fn')},
                                  'errors_introduced_by_refinement': len(moved_away),
                                  'errors_in_detection': len(report['targets'][str(number)]['errors_in_detection']),
                                  'statuses': statuses}), flush=True)
    write_json(RUN / 'localization_diagnosis.json', report)
    print('wrote', RUN / 'localization_diagnosis.json')


main()
