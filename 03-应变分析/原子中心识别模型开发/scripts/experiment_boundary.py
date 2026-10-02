"""Round 3 boundary experiment: two fixed 32 px extensions for network candidate generation.

Hypothesis: the frozen detector sees constant grey padding beyond the image border, so atoms cut
by the border produce weak responses. Providing a general boundary continuation may recover them.

Both candidates keep every existing baseline prediction and only add candidates that come from the
extended inference and land within 32 px of the true image border. Padding is fixed at 32 px with
mode 'edge' (replicate) or 'reflect' (mirror); nothing here looks at file names, answers or error
tables. The extension is used for candidate generation only: the padded-image candidates are
mapped back by subtracting the padding, candidates outside the original image are discarded, and
refinement then runs on the unmodified original image with the unchanged gaussian11/max_shift4
settings. Synthetic border pixels never enter refinement, metrics or any truth.

Everything stays in this round-specific module; no production configuration is touched.
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
sys.path.insert(0, str(ROOT / 'scripts'))
from atom_center.backends import onnx_pipeline                            # noqa: E402
from atom_center.geometry import merge_close_points, point_nms_indices    # noqa: E402
from atom_center.image_io import load_image                               # noqa: E402
from atom_center.interfaces import CandidateSet                           # noqa: E402
from atom_center.refinement import refine_with_diagnostics                # noqa: E402
from atom_center.storage import write_json                                # noqa: E402
from evaluate_reviewed_test_answers import score, inside                  # noqa: E402

RUN = ROOT / 'runs/boundary-optimization-20260910'
REVIEW = ROOT / 'runs/reviewed-test-20260910'
MANIFEST = ROOT / 'runs/target-adaptation-20260910/target_adaptation_onnx/model_manifest.json'
FROZEN_MODEL_SHA256 = '15222438748086ba56fb334926588656fb8057a3b056cdb41f3bd198d910c267'
PADDING_PX = 32
MODES = ('edge', 'reflect')
METRIC_KEYS = ('tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'rmse_px', 'p95_px')
FLOORS = {'04': {'no_new_false_positives': True},
          '05': {'same_roi_fp_max': 16, 'same_roi_fn_max': 18}}


def padded(image, width=PADDING_PX, mode='edge'):
    """Boundary continuation used only to give the network context beyond the image."""
    if mode not in MODES:
        raise ValueError(f'unsupported extension mode: {mode}')
    return np.pad(np.asarray(image, dtype=np.float64), width, mode=mode)


def boundary_zone(points, shape, width=PADDING_PX):
    """True for points within `width` px of the true image border (not the ROI or tile edge)."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(points):
        return np.zeros(0, dtype=bool)
    height, width_px = shape
    return ((points[:, 0] < width) | (points[:, 0] >= width_px - width)
            | (points[:, 1] < width) | (points[:, 1] >= height - width))


def extended_candidates(pipeline, image, mode, width=PADDING_PX):
    """Network candidates from the padded image, mapped back into original-image coordinates."""
    candidates, tile_count, diagnostics = pipeline._predict_region(padded(image, width, mode))
    points = candidates.points - width
    height, width_px = image.shape
    keep = ((points[:, 0] >= 0) & (points[:, 0] < width_px)
            & (points[:, 1] >= 0) & (points[:, 1] < height)) if len(points) else np.zeros(0, dtype=bool)
    return CandidateSet(points[keep], candidates.confidences[keep]), tile_count, diagnostics, int((~keep).sum())


def baseline_candidates(pipeline, image):
    candidates, tile_count, diagnostics = pipeline._predict_region(image)
    return candidates, tile_count, diagnostics


def finish_like_production(pipeline, image, candidates):
    """Reproduce the tail of DetectionPipeline.detect for a given candidate set.

    The pipeline's own refinement, inside mask and post-refinement NMS are reused unchanged, so a
    candidate set equal to the baseline's reproduces the production output exactly.
    """
    config = pipeline.config
    if config.min_atom_distance_px is not None:
        raise ValueError('this experiment assumes the frozen config without min_atom_distance_px')
    points = np.asarray(candidates.points, dtype=float)
    quality = []
    if config.refine and len(points):
        refined = refine_with_diagnostics(image, points, window_size=config.refinement_window,
                                          method=config.refinement_method,
                                          polarity=config.refinement_polarity,
                                          max_shift_px=config.refinement_max_shift_px)
        points, quality = refined.points, list(refined.diagnostics)
    height, width_px = image.shape
    inside_image = ((points[:, 0] >= 0) & (points[:, 0] < width_px)
                    & (points[:, 1] >= 0) & (points[:, 1] < height)) if len(points) else np.zeros(0, dtype=bool)
    final_points, final_scores = points[inside_image], candidates.confidences[inside_image]
    final_quality = [entry for entry, keep in zip(quality, inside_image) if keep]
    removed = 0
    if config.refine and config.merge_after_refinement and len(final_points):
        keep = point_nms_indices(final_points, final_scores, min_distance=config.merge_distance_px)
        removed = len(final_points) - len(keep)
        final_points, final_scores = final_points[keep], final_scores[keep]
        if final_quality:
            final_quality = [final_quality[i] for i in keep]
    return {'points': final_points, 'confidences': final_scores, 'quality': final_quality,
            'duplicates_removed_after_refinement': int(removed)}


def load_target(number):
    record = json.loads((REVIEW / f'{number:02d}_comparison.json').read_text(encoding='utf-8-sig'))
    image = np.asarray(load_image(record['source_image'], normalize=False,
                                  preserve_dtype=True).image, dtype=np.float64)
    return record, image


def metrics(points, truth, roi):
    inside_roi = inside(points, roi) if len(points) else np.zeros(0, dtype=bool)
    truth_inside = inside(truth, roi)
    full = score(points, truth, 2.)
    same_roi = score(points[inside_roi], truth[truth_inside], 2.)
    return ({key: full[key] for key in METRIC_KEYS},
            {key: same_roi[key] for key in METRIC_KEYS})


def run_candidate(number, image, truth, roi, mode, pipeline_factory=onnx_pipeline, width=PADDING_PX):
    """Baseline vs boundary-supplemented detection for one image and one extension mode."""
    pipeline = pipeline_factory(MANIFEST)
    baseline_final = pipeline.detect(image)
    baseline_candidate_set, baseline_tiles, _ = baseline_candidates(pipeline, image)
    reproduction = finish_like_production(pipeline, image, baseline_candidate_set)
    reproduction_exact = bool(np.array_equal(reproduction['points'], baseline_final.points)
                              and np.array_equal(reproduction['confidences'], baseline_final.confidences))

    extended, extended_tiles, extended_diagnostics, dropped_outside = extended_candidates(pipeline, image, mode, width)
    zone = boundary_zone(extended.points, image.shape, width)
    zone_points, zone_confidences = extended.points[zone], extended.confidences[zone]

    combined_points = np.concatenate([baseline_candidate_set.points, zone_points]) if len(zone_points) \
        else baseline_candidate_set.points
    combined_confidences = np.concatenate([baseline_candidate_set.confidences, zone_confidences]) \
        if len(zone_points) else baseline_candidate_set.confidences
    merged = merge_close_points(combined_points, combined_confidences,
                                min_distance=pipeline.config.merge_distance_px)
    supplemented = finish_like_production(pipeline, image, merged)

    baseline_tree = cKDTree(baseline_final.points) if len(baseline_final.points) else None
    final = supplemented['points']
    added = (np.asarray([baseline_tree.query(point)[0] for point in final]) > 1e-9
             if baseline_tree is not None and len(final) else np.ones(len(final), dtype=bool))
    final_tree = cKDTree(final) if len(final) else None
    lost = ([int((final_tree.query(point)[0] > 1e-9)) for point in baseline_final.points]
            if final_tree is not None else [1] * len(baseline_final.points))
    interior = ~boundary_zone(baseline_final.points, image.shape, width)
    interior_lost = int(sum(lost[index] for index in np.flatnonzero(interior)))
    zone_lost = int(sum(lost[index] for index in np.flatnonzero(~interior)))

    baseline_full, baseline_roi = metrics(baseline_final.points, truth, roi)
    candidate_full, candidate_roi = metrics(final, truth, roi)
    truth_tree = cKDTree(truth)
    added_indices = np.flatnonzero(added) if len(final) else np.zeros(0, dtype=int)
    added_rows = [{'xy': final[index].round(3).tolist(),
                   'confidence': round(float(supplemented['confidences'][index]), 6),
                   'nearest_reference_px': round(float(truth_tree.query(final[index])[0]), 3),
                   'matched_within_2px': bool(truth_tree.query(final[index])[0] <= 2.)}
                  for index in added_indices]
    lost_rows = [{'xy': baseline_final.points[index].round(3).tolist(),
                  'confidence': round(float(baseline_final.confidences[index]), 6),
                  'in_boundary_zone': bool(boundary_zone(baseline_final.points[index].reshape(1, 2),
                                                         image.shape, width)[0])}
                 for index, value in enumerate(lost) if value]
    return {
        'number': number, 'mode': mode, 'padding_px': width,
        'baseline_points': int(len(baseline_final.points)),
        'baseline_candidates': int(len(baseline_candidate_set.points)),
        'baseline_tiles': int(baseline_tiles),
        'production_reproduction_exact': reproduction_exact,
        'extended_candidates_total': int(len(extended.points)),
        'extended_candidates_outside_image_dropped': dropped_outside,
        'extended_tiles': int(extended_tiles),
        'extended_candidates_in_boundary_zone': int(len(zone_points)),
        'candidates_merged_total': int(len(merged.points)),
        'final_points': int(len(final)),
        'points_added_vs_baseline': int(len(added_indices)),
        'added_points': added_rows,
        'added_points_matched_within_2px': int(sum(row['matched_within_2px'] for row in added_rows)),
        'added_points_false_positives': int(sum(not row['matched_within_2px'] for row in added_rows)),
        'baseline_points_lost': lost_rows,
        'baseline_points_lost_interior': interior_lost,
        'baseline_points_lost_in_boundary_zone': zone_lost,
        'interior_unchanged': interior_lost == 0,
        'duplicates_removed_after_refinement': supplemented['duplicates_removed_after_refinement'],
        'baseline_full_image': baseline_full, 'candidate_full_image': candidate_full,
        'baseline_same_original_roi': baseline_roi, 'candidate_same_original_roi': candidate_roi,
        'extended_tile_diagnostics': extended_diagnostics,
    }


def evaluate(pipeline_factory=onnx_pipeline):
    report = {'task_id': 'boundary-optimization-20260910',
              'purpose': 'two fixed 32 px boundary extensions for network candidate generation only',
              'model_manifest': str(MANIFEST.resolve()),
              'model_sha256': pipeline_factory(MANIFEST).backend.model_sha256,
              'frozen_model_sha256_confirmed': pipeline_factory(MANIFEST).backend.model_sha256 == FROZEN_MODEL_SHA256,
              'padding_px': PADDING_PX, 'modes': list(MODES), 'confidence': 0.1,
              'refinement': 'unchanged gaussian window 11, bright, max_shift 4 px, merge after refinement 3 px',
              'rule': 'keep every baseline prediction; add extended-inference candidates only within '
                      '32 px of the true image border; dedupe with the existing merge rule',
              'floors': FLOORS, 'targets': {}}
    for number in (4, 5):
        record, image = load_target(number)
        truth = np.asarray(record['all_csv_truth_xy'], dtype=float)
        roi = np.asarray(record['roi_xyxy_inclusive'], dtype=float)
        rows = []
        for mode in MODES:
            row = run_candidate(number, image, truth, roi, mode, pipeline_factory)
            rows.append(row)
            print(f"target {number} mode {mode}: final {row['final_points']} (+{row['points_added_vs_baseline']}) "
                  f"full {row['candidate_full_image']['tp']}/{row['candidate_full_image']['fp']}/"
                  f"{row['candidate_full_image']['fn']} roi {row['candidate_same_original_roi']['tp']}/"
                  f"{row['candidate_same_original_roi']['fp']}/{row['candidate_same_original_roi']['fn']} "
                  f"interior_unchanged={row['interior_unchanged']} repro={row['production_reproduction_exact']}",
                  flush=True)
        report['targets'][str(number)] = {'image_shape': list(image.shape), 'roi_xyxy_inclusive': roi.tolist(),
                                          'reference_count_all_csv': int(len(truth)), 'candidates': rows}
    return report


def verdict(report):
    """Apply the plan's acceptance rule to each mode across both images."""
    decisions = {}
    for mode in MODES:
        rows = {number: next(row for row in report['targets'][number]['candidates'] if row['mode'] == mode)
                for number in ('4', '5')}
        improved, regressed = [], []
        for number, row in rows.items():
            delta = row['candidate_full_image']['f1'] - row['baseline_full_image']['f1']
            if delta > 1e-12:
                improved.append(number)
            elif delta < -1e-12:
                regressed.append(number)
        four = rows['4']
        five = rows['5']
        checks = {
            'at_least_one_image_improved': bool(improved),
            'no_image_regressed': not regressed,
            'zero_four_adds_no_false_positive': four['candidate_full_image']['fp'] <= four['baseline_full_image']['fp'],
            'zero_five_roi_within_floors': (five['candidate_same_original_roi']['fp'] <= FLOORS['05']['same_roi_fp_max']
                                            and five['candidate_same_original_roi']['fn'] <= FLOORS['05']['same_roi_fn_max']),
            'interior_output_unchanged': all(row['interior_unchanged'] for row in rows.values()),
            'production_flow_reproduced': all(row['production_reproduction_exact'] for row in rows.values()),
            'matched_localisation_not_worse': all(
                row['candidate_full_image']['rmse_px'] <= row['baseline_full_image']['rmse_px'] + 0.02
                for row in rows.values()),
        }
        decisions[mode] = {'improved_images': improved, 'regressed_images': regressed,
                           'checks': checks, 'passes_targets': all(checks.values()),
                           'rows': {number: {'baseline_full_image': rows[number]['baseline_full_image'],
                                             'candidate_full_image': rows[number]['candidate_full_image'],
                                             'baseline_same_original_roi': rows[number]['baseline_same_original_roi'],
                                             'candidate_same_original_roi': rows[number]['candidate_same_original_roi'],
                                             'points_added_vs_baseline': rows[number]['points_added_vs_baseline'],
                                             'added_points_matched_within_2px': rows[number]['added_points_matched_within_2px'],
                                             'added_points_false_positives': rows[number]['added_points_false_positives'],
                                             'baseline_points_lost_interior': rows[number]['baseline_points_lost_interior'],
                                             'baseline_points_lost_in_boundary_zone': rows[number]['baseline_points_lost_in_boundary_zone']}
                                    for number in ('4', '5')}}
    return decisions


if __name__ == '__main__':
    result = evaluate()
    result['decisions'] = verdict(result)
    write_json(RUN / 'boundary_candidates.json', result)
    for mode, decision in result['decisions'].items():
        print(mode, 'passes_targets=', decision['passes_targets'], decision['checks'], flush=True)
    print('wrote', RUN / 'boundary_candidates.json')
