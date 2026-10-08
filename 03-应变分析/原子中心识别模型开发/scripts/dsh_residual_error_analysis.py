"""DSH step 4 extra: characterize the residual 2 px errors of the accepted ONNX bundle.

Uses only the acceptance coordinate JSON (ONNX actual output) and the frozen answer files.
"""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.spatial import cKDTree

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from atom_center.storage import write_json                                  # noqa: E402

ACC = ROOT / 'runs/target-adaptation-20260910/dsh_acceptance'
REVIEW = ROOT / 'runs/reviewed-test-20260910'


def main():
    out = {}
    for number in (4, 5):
        data = json.loads((ACC / f'{number:02d}_coordinates.json').read_text(encoding='utf-8-sig'))
        prediction = np.asarray(data['points_xy'], dtype=float)
        truth = np.asarray(data['truth_xy'], dtype=float)
        confidences = np.asarray(data['confidences'], dtype=float)
        fn_index = np.asarray(data['unmatched_truth_indices'], dtype=int)
        fp_index = np.asarray(data['unmatched_prediction_indices'], dtype=int)
        meta = json.loads((REVIEW / f'{number:02d}_comparison.json').read_text(encoding='utf-8-sig'))
        height, width = meta['source_identity']['shape']
        roi = np.asarray(data['roi_xyxy_inclusive'], dtype=float)
        spacing = float(np.median(cKDTree(truth).query(truth, k=2)[0][:, 1]))
        to_prediction = cKDTree(prediction).query(truth[fn_index])[0] if len(fn_index) else np.empty(0)
        to_answer = cKDTree(truth).query(prediction[fp_index])[0] if len(fp_index) else np.empty(0)

        def flags(points):
            inside = ((points[:, 0] >= roi[0]) & (points[:, 0] <= roi[2])
                      & (points[:, 1] >= roi[1]) & (points[:, 1] <= roi[3])) if len(points) \
                else np.empty(0, dtype=bool)
            return {'inside_roi': int(inside.sum()), 'outside_roi': int((~inside).sum()),
                    'within_20px_of_image_border': int(((points[:, 0] < 20) | (points[:, 0] > width - 20)
                                                        | (points[:, 1] < 20) | (points[:, 1] > height - 20)).sum())
                    if len(points) else 0}

        def group_signature(points):
            """Describe the >6 px class: one row, evenly spaced, outside the checked ROI?"""
            if len(points) < 2:
                return None
            x_sorted = np.sort(points[:, 0])
            return {'same_row_within_1px_y': bool(np.ptp(points[:, 1]) <= 1.0),
                    'median_x_spacing_px': round(float(np.median(np.diff(x_sorted))), 2),
                    'y_median': round(float(np.median(points[:, 1])), 2)}

        out[str(number)] = {
            'image_shape_hw': [height, width], 'roi_xyxy_inclusive': roi.tolist(),
            'lattice_spacing_median_px': round(spacing, 3),
            'fn': {'count': int(len(fn_index)),
                   'positions_xy': truth[fn_index].round(2).tolist(),
                   'distance_to_nearest_prediction_px': to_prediction.round(2).tolist(),
                   'offset_class_le_6px': int((to_prediction <= 6).sum()),
                   'missed_class_gt_6px': int((to_prediction > 6).sum()),
                   'missed_class_signature': group_signature(truth[fn_index][to_prediction > 6]),
                   **flags(truth[fn_index])},
            'fp': {'count': int(len(fp_index)),
                   'positions_xy': prediction[fp_index].round(2).tolist(),
                   'distance_to_nearest_answer_px': to_answer.round(2).tolist(),
                   'offset_class_le_6px': int((to_answer <= 6).sum()),
                   'spurious_class_gt_6px': int((to_answer > 6).sum()),
                   'spurious_class_signature': group_signature(prediction[fp_index][to_answer > 6]),
                   'confidence_min': round(float(confidences[fp_index].min()), 4) if len(fp_index) else None,
                   'confidence_median': round(float(np.median(confidences[fp_index])), 4) if len(fp_index) else None,
                   'confidence_max': round(float(confidences[fp_index].max()), 4) if len(fp_index) else None,
                   **flags(prediction[fp_index])},
        }
        print(number, json.dumps(out[str(number)], ensure_ascii=False), flush=True)
    out['note'] = ('2 px one-to-one matching on the ONNX actual output. An offset pair (one FN and one FP '
                   '2-6 px apart) means the column was found but its centre sits just outside the 2 px '
                   'radius; a >6 px nearest neighbour is about one lattice spacing, i.e. a missing '
                   'detection or an extra detection at a lattice site.')
    write_json(ACC / 'residual_error_analysis.json', out)


main()
