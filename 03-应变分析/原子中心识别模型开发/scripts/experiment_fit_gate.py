"""Round-2 rejected experiment: the centre-fit gate, kept out of the production API.

Candidate C1 ("revert to the coarse candidate where the gaussian fit residual is in the image's
own top decile") was measured, did not meet the round-2 floors and was NOT adopted. Its logic is
kept here so the measurement can be reproduced without leaving any gate code or configuration
option on the production path.

The gate is applied by temporarily wrapping ``atom_center.pipeline.refine_with_diagnostics``, so
the untouched production ``DetectionPipeline.detect`` flow runs with the gate inserted at exactly
the same point the experiment used. Nothing in ``src/atom_center`` references this module.

Usage:
    python -X utf8 scripts/experiment_fit_gate.py            # reproduce the recorded C1 numbers
"""
from contextlib import contextmanager
from pathlib import Path
import json
import sys

import numpy as np

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

import atom_center.pipeline as pipeline_module                            # noqa: E402
from atom_center.backends import onnx_pipeline                            # noqa: E402
from atom_center.image_io import load_image                               # noqa: E402
from atom_center.refinement import RefinementResult                       # noqa: E402
from atom_center.storage import write_json                                # noqa: E402
from evaluate_reviewed_test_answers import score, inside                  # noqa: E402

RUN = ROOT / 'runs/training-update-20260910'
REVIEW = ROOT / 'runs/reviewed-test-20260910'
MANIFEST = ROOT / 'runs/target-adaptation-20260910/target_adaptation_onnx/model_manifest.json'
DEFAULT_QUANTILE = 0.9


def apply_fit_gate(points, coarse_points, diagnostics, quantile):
    """Revert points whose fit residual sits above this image's own residual quantile.

    ``points`` are the refined centres, ``coarse_points`` the network candidates they came from.
    The quantile comes from the image's own diagnostics, never from reference answers.
    """
    points = np.asarray(points, dtype=float)
    coarse_points = np.asarray(coarse_points, dtype=float)
    if not len(points):
        return points, list(diagnostics)
    residuals = np.asarray([entry.get("normalized_rmse") if entry.get("normalized_rmse") is not None
                            else np.nan for entry in diagnostics], dtype=float)
    usable = np.isfinite(residuals) & np.asarray(
        [entry.get("status") == "refined" for entry in diagnostics], dtype=bool)
    if usable.sum() < 2:
        return points, list(diagnostics)
    threshold = float(np.quantile(residuals[usable], quantile))
    output = points.copy()
    updated = list(diagnostics)
    reverted = 0
    for index in np.flatnonzero(usable & (residuals > threshold)):
        output[index] = coarse_points[index]
        updated[index] = {**diagnostics[index], "status": "fit_gate_reverted",
                          "gate_threshold": threshold, "reverted_to_candidate": True}
        reverted += 1
    return output, updated


@contextmanager
def fit_gate_enabled(quantile=DEFAULT_QUANTILE):
    """Apply the gate inside the unchanged production detect flow, then restore it."""
    original = pipeline_module.refine_with_diagnostics

    def gated(image, points, **kwargs):
        result = original(image, points, **kwargs)
        new_points, new_diagnostics = apply_fit_gate(result.points, points, result.diagnostics, quantile)
        return RefinementResult(new_points, tuple(new_diagnostics))

    pipeline_module.refine_with_diagnostics = gated
    try:
        yield
    finally:
        pipeline_module.refine_with_diagnostics = original


def measure(number, quantile=None):
    """Return the 2 px metrics for one reviewed target with or without the gate."""
    record = json.loads((REVIEW / f'{number:02d}_comparison.json').read_text(encoding='utf-8-sig'))
    raw = np.asarray(load_image(record['source_image'], normalize=False,
                                preserve_dtype=True).image, dtype=np.float64)
    truth = np.asarray(record['all_csv_truth_xy'], dtype=float)
    roi = np.asarray(record['roi_xyxy_inclusive'], dtype=float)
    if quantile is None:
        result = onnx_pipeline(MANIFEST).detect(raw)
    else:
        with fit_gate_enabled(quantile):
            result = onnx_pipeline(MANIFEST).detect(raw)
    points = result.points
    full = score(points, truth, 2.)
    same_roi = score(points[inside(points, roi)], truth[inside(truth, roi)], 2.)
    reverted = sum(1 for entry in result.metadata['point_quality']
                   if entry.get('status') == 'fit_gate_reverted')
    keys = ('tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'rmse_px')
    return {'number': number, 'quantile': quantile, 'prediction_count': int(len(points)),
            'reverted_points': reverted,
            'full_image': {key: full[key] for key in keys},
            'same_original_roi': {key: same_roi[key] for key in keys}}


def main():
    rows = []
    for number in (4, 5):
        for quantile in (None, DEFAULT_QUANTILE):
            row = measure(number, quantile)
            rows.append(row)
            label = 'baseline' if quantile is None else f'gate q={quantile}'
            print(f"{number} {label}: full {row['full_image']['tp']}/{row['full_image']['fp']}/"
                  f"{row['full_image']['fn']} rmse {row['full_image']['rmse_px']:.4f} | roi "
                  f"{row['same_original_roi']['tp']}/{row['same_original_roi']['fp']}/"
                  f"{row['same_original_roi']['fn']} | reverted {row['reverted_points']}", flush=True)
    write_json(RUN / 'experiment_fit_gate_reproduction.json',
               {'purpose': 'reproduce the rejected C1 measurement after removing it from src',
                'component': 'scripts/experiment_fit_gate.py', 'quantile': DEFAULT_QUANTILE,
                'production_src_references_this_module': False,
                'model_sha256': onnx_pipeline(MANIFEST).backend.model_sha256,
                'recorded_earlier': {'C1_gaussian11_fitgate': {'target_4_same_roi': '189/0/2',
                                                               'target_5_full_image': '1132/26/20',
                                                               'reverted_points_05': 116}},
                'rows': rows})
    print('wrote', RUN / 'experiment_fit_gate_reproduction.json')


if __name__ == '__main__':
    main()
