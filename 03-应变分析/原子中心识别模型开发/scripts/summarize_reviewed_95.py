"""Compact readout of the reviewed-95 validation record (no inference)."""

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


RUN = Path(__file__).resolve().parents[1] / 'runs/reviewed-95-validation-20260910'


def main():
    data = json.loads((RUN / 'validation.json').read_text(encoding='utf-8-sig'))
    model = data['model']
    print('model', model['model_id'], model['model_sha256'][:16])
    print('graph', model['graph_sha256'][:16], 'checkpoint', model['checkpoint_sha256'][:16])
    print('inference', json.dumps(model['inference'], ensure_ascii=False))
    print('refinement', json.dumps(model['refinement'], ensure_ascii=False))
    for number in ('4', '5'):
        entry = data['targets'][number]
        container = entry['container']
        print(f"==== target {number} image={Path(entry['source_image']).name}")
        print(f"   container: shape {container['series_shape']} dtype {container['series_dtype']} "
              f"axes {container['series_axes']} photometric {container['photometric']} "
              f"bits {container['bits_per_sample']} spp {container['samples_per_pixel']} "
              f"compression {container['compression']} pages {container['pages']}")
        print(f"   decoded plane {entry['decoded_plane']}")
        print(f"   answers all={entry['reference_count_all_csv']} roi={entry['reference_count_in_roi']} "
              f"csv_sha_ok={entry['answer_sha_matches_frozen_record']} image_sha={entry['image_sha256'][:16]}")
        print(f"   roi float {[round(v, 3) for v in entry['roi_xyxy_inclusive']]} -> cli int {entry['roi_integer_cli_bounds']}")
        for name in ('full_image', 'roi_only'):
            path = entry['paths'][name]
            repeat = path['repeat_two_processes']
            print(f"   [{name}] points={path['prediction_count']} provider={path['provider']}")
            print(f"      repeat: matched={repeat['matched']} unmatched={repeat['unmatched_first']}/"
                  f"{repeat['unmatched_second']} maxdelta={repeat['max_delta_px']} identical={repeat['identical']}")
            print(f"      scored against {path['scored_against']} ({path['reference_count_scored']} points)")
            for row in path['tolerance_metrics']:
                print(f"        {row['radius_px']}px: P={row['precision']:.4f} R={row['recall']:.4f} "
                      f"F1={row['f1']:.4f} TP{row['tp']} FP{row['fp']} FN{row['fn']} "
                      f"rmse={row['rmse_px'] if row['rmse_px'] is None else round(row['rmse_px'], 4)} "
                      f"p95={row['p95_px'] if row['p95_px'] is None else round(row['p95_px'], 4)} "
                      f"pass95={row['passes_both_95']}")
        full = entry['paths']['full_image']
        print(f"   [full_image on original ROI] predictions_in_roi={full['prediction_count_in_roi']}")
        for row in full['tolerance_metrics_on_original_roi']:
            print(f"        {row['radius_px']}px: P={row['precision']:.4f} R={row['recall']:.4f} "
                  f"TP{row['tp']} FP{row['fp']} FN{row['fn']} "
                  f"rmse={row['rmse_px'] if row['rmse_px'] is None else round(row['rmse_px'], 4)} "
                  f"pass95={row['passes_both_95']}")
        torch_entry = full.get('torch_reference')
        if torch_entry:
            agreement = torch_entry['agreement_with_onnx']
            print(f"   [torch vs onnx] torch points={torch_entry['prediction_count']} "
                  f"matched={agreement['matched']} unmatched={agreement['unmatched_first']}/"
                  f"{agreement['unmatched_second']} maxdelta={agreement['max_delta_px']} "
                  f"identical={agreement['identical']}")
        print('   local grid (2x3):')
        for cell in full['local_grid_on_original_roi']:
            print(f"      r{cell['row']}c{cell['column']} bounds={[round(v,1) for v in cell['bounds_xyxy']]} "
                  f"GT={cell['gt']} pred={cell['predictions']} TP={cell['tp']} FP={cell['fp']} FN={cell['fn']} "
                  f"P={cell['precision']:.4f} R={cell['recall']:.4f} small={cell['small_sample']} "
                  f"zero_gt_fp={cell['zero_ground_truth_with_false_positives']}")
    print('==== verdict')
    print(json.dumps(data['verdict_95'], ensure_ascii=False, indent=1)[:1500])


main()
