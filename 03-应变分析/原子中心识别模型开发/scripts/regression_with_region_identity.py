"""Round-2 review fix 4: old train/val regression with stable per-region identity.

Re-runs only the necessary evaluation with the frozen round-1 ONNX bundle and records every region
with its dataset record_key, isolation group, ROI index and source image path, instead of a bare
group label. Writes runs/training-update-20260910/regression_region_identity.json.
"""
from pathlib import Path
import sys

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from atom_center.backends import onnx_pipeline                            # noqa: E402
from atom_center.evaluation import evaluate_dataset                       # noqa: E402
from atom_center.storage import read_json, write_json                     # noqa: E402

RUN = ROOT / 'runs/training-update-20260910'
MANIFEST = ROOT / 'runs/target-adaptation-20260910/target_adaptation_onnx/model_manifest.json'
DATASET = ROOT / 'data/processed/real_workflow_20260909_v2'


def main():
    pipeline = onnx_pipeline(MANIFEST)
    manifest = read_json(DATASET / 'dataset_manifest.json')
    sources = {source['record_key']: source for source in manifest['sources']}
    report = {'purpose': 'old frozen train/val regression with stable region identity',
              'dataset': str(DATASET.relative_to(ROOT).as_posix()),
              'dataset_content_sha256': manifest['content_sha256'],
              'model_manifest': str(MANIFEST.relative_to(ROOT).as_posix()),
              'model_sha256': pipeline.backend.model_sha256,
              'config': {'conf': pipeline.backend.inference['conf'],
                         'iou': pipeline.backend.inference['iou'],
                         'max_det': pipeline.backend.inference['max_det'],
                         'refinement_method': pipeline.config.refinement_method,
                         'refinement_window': pipeline.config.refinement_window,
                         'refinement_max_shift_px': pipeline.config.refinement_max_shift_px},
              'metric_radius_px': 2.0, 'splits': {}}
    for split in ('train', 'val'):
        result = evaluate_dataset(pipeline, DATASET, split=split, max_distance_px=2.)
        regions = []
        for row in result['regions']:
            source = sources.get(row['record_key'], {})
            regions.append({
                'record_key': row['record_key'], 'isolation_group': row['isolation_group'],
                'roi_index': row['roi_index'], 'split': split,
                'source_image': source.get('source_image'),
                'raw_in_dataset': source.get('raw'),
                'label_file': source.get('label_file'),
                'roi_xyxy': (read_json(DATASET / 'truth' / f"{row['record_key']}.json")['regions_xyxy']
                             [row['roi_index']] if (DATASET / 'truth' / f"{row['record_key']}.json").is_file()
                             else None),
                'tp': row['tp'], 'fp': row['fp'], 'fn': row['fn'],
                'precision': row['tp'] / (row['tp'] + row['fp']) if row['tp'] + row['fp'] else 1.0,
                'recall': row['tp'] / (row['tp'] + row['fn']) if row['tp'] + row['fn'] else 1.0,
                'rmse_px': row['rmse_px']})
        regions.sort(key=lambda row: row['recall'])
        report['splits'][split] = {
            'tp': result['tp'], 'fp': result['fp'], 'fn': result['fn'],
            'precision': result['precision'], 'recall': result['recall'], 'f1': result['f1'],
            'rmse_px': result['rmse_px'], 'p95_px': result['p95_px'], 'bias_xy_px': result['bias_xy_px'],
            'region_count': len(regions),
            'min_region_recall': min(row['recall'] for row in regions),
            'regions_by_recall': regions}
        print(split, f"tp={result['tp']} fp={result['fp']} fn={result['fn']} "
                     f"f1={result['f1']:.6f} rmse={result['rmse_px']:.4f} regions={len(regions)} "
                     f"min_region_recall={report['splits'][split]['min_region_recall']:.4f}", flush=True)
        for row in regions[:4]:
            print(f"    {row['record_key']} roi{row['roi_index']} {Path(row['source_image']).name if row['source_image'] else '?'} "
                  f"tp={row['tp']} fn={row['fn']} recall={row['recall']:.4f} rmse={row['rmse_px']}", flush=True)
    write_json(RUN / 'regression_region_identity.json', report)
    print('wrote', RUN / 'regression_region_identity.json')


main()
