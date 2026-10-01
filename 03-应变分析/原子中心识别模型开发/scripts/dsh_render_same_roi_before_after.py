"""DSH step 4 extra: same-original-ROI before/after figures for targets 04 and 05.

Reads the ONNX acceptance coordinates (already verified against the ONNX graph hash) and the
previous model's frozen predictions, so no new inference is run. Both panels are cropped to the
original exported ROI so the two models are visually and numerically comparable on the same area.
"""
from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from atom_center.image_io import load_image                                   # noqa: E402
from atom_center.storage import write_json                                    # noqa: E402
from evaluate_reviewed_test_answers import score, inside, draw_overlay        # noqa: E402

BASE = ROOT / 'runs/target-adaptation-20260910'
ACC = BASE / 'dsh_acceptance'
REVIEW = ROOT / 'runs/reviewed-test-20260910'
METRIC_KEYS = ('tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'rmse_px', 'p95_px')


def panel(raw, box, truth, prediction, metrics, title, target):
    """One ROI-cropped comparison panel with its own header and metrics."""
    x0, y0, x1, y1 = box
    crop = np.asarray(raw[y0:y1 + 1, x0:x1 + 1], dtype=float)
    local_truth, local_prediction = truth - [x0, y0], prediction - [x0, y0]
    return draw_overlay(crop, local_truth, local_prediction, [0, 0, x1 - x0, y1 - y0],
                        metrics, title, target)


def main():
    rows = {}
    for number in (4, 5):
        coordinates = json.loads((ACC / f'{number:02d}_coordinates.json').read_text(encoding='utf-8-sig'))
        previous = json.loads((REVIEW / f'{number:02d}_comparison.json').read_text(encoding='utf-8-sig'))
        raw = np.asarray(load_image(previous['source_image'], normalize=False,
                                    preserve_dtype=True).image, dtype=float)
        roi = np.asarray(coordinates['roi_xyxy_inclusive'], dtype=float)
        box = (int(np.floor(roi[0])), int(np.floor(roi[1])),
               int(np.ceil(roi[2])), int(np.ceil(roi[3])))
        truth_all = np.asarray(coordinates['truth_xy'], dtype=float)
        truth = truth_all[inside(truth_all, roi)]
        panels, records = [], []
        for key, label, prediction_all in (
                ('before', '上一轮已验收模型（答案提供前的固定预测）',
                 np.asarray(previous['frozen_prediction_xy'], dtype=float)),
                ('after', '本次自适应 ONNX 模型（答案已参与训练）',
                 np.asarray(coordinates['points_xy'], dtype=float))):
            prediction = prediction_all[inside(prediction_all, roi)]
            metrics = score(prediction, truth, 2.)
            records.append({'key': key, 'label': label, 'prediction_count_in_roi': int(len(prediction)),
                            'metrics': {k: metrics[k] for k in METRIC_KEYS}})
            panels.append(panel(raw, box, truth, prediction, metrics,
                                f'开发图像 {number:02d} · {label} · 原选区 {box[2]-box[0]}x{box[3]-box[1]} px',
                                ACC / f'{number:02d}_roi_panel_{key}.png'))
        stack = Image.new('RGB', (max(p.width for p in panels), sum(p.height for p in panels) + 16), '#101827')
        stack.paste(panels[0], (0, 0))
        stack.paste(panels[1], (0, panels[0].height + 16))
        stack.save(ACC / f'{number:02d}_same_roi_before_after.png')
        rows[str(number)] = {'number': number, 'roi_xyxy_inclusive': roi.tolist(),
                             'reference_points_in_roi': int(len(truth)),
                             'before': records[0], 'after': records[1],
                             'figure': str(ACC / f'{number:02d}_same_roi_before_after.png'),
                             'note': 'same frozen ROI and same 2 px one-to-one matching for both rows; '
                                     'the adapted model used these two images\' answers in training, so only '
                                     'the before row is an independent measurement'}
        print(number, json.dumps(rows[str(number)], ensure_ascii=False)[:400], flush=True)
    write_json(ACC / 'same_roi_before_after.json', rows)
    print('wrote', ACC / 'same_roi_before_after.json')


main()
