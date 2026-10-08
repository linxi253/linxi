"""DSH step 2: export the selected adaptation checkpoint as a standalone ONNX bundle.

The graph comes from the existing verified ``deployment.export_run`` path (same run, same
registered checkpoint name), so no training-run manifest is touched. A separate manifest
variant then records the *actual* deployment configuration (gaussian/window11/max_shift 4
plus the selected confidence), marked development / target_adaptation /
scientific_acceptance=false.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import json
import shutil
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

from atom_center.deployment import export_run                       # noqa: E402
from atom_center.model_manifest import verify_model_bundle, sha256_file  # noqa: E402
from atom_center.storage import read_json, write_json               # noqa: E402
from atom_center.training import checkpoint_for_run, read_run, software_versions  # noqa: E402

BASE = ROOT / 'runs/target-adaptation-20260910'
RUN = BASE / 'finetune100'
OLD_DATASET = ROOT / 'data/processed/real_workflow_20260909_v2'
REFINEMENT = {'enabled': True, 'method': 'gaussian', 'window_size': 11,
              'polarity': 'bright', 'max_shift_px': 4.0, 'merge_sigma': None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-name', default='last.pt',
                        help='registered checkpoint name in the run state (e.g. last.pt, best_points.pt)')
    parser.add_argument('--conf', type=float, required=True)
    parser.add_argument('--output', type=Path, default=BASE / 'target_adaptation_onnx')
    parser.add_argument('--staging', type=Path, default=BASE / 'dsh_selection/onnx_export_run_config')
    parser.add_argument('--selection-record', type=Path,
                        default=BASE / 'dsh_selection/comparison.json')
    args = parser.parse_args()

    run, manifest, state = read_run(RUN)
    assert state['status'] == 'completed'
    training, checkpoint_path, digest = checkpoint_for_run(RUN, args.checkpoint_name)
    inference = {**manifest['config']['inference'], 'conf': args.conf}

    # 1) Reuse the verified exporter on the already-registered checkpoint (run config).
    export_run(RUN, args.staging, checkpoint=args.checkpoint_name)
    original, graph = verify_model_bundle(args.staging / 'model_manifest.json')
    graph_sha = sha256_file(graph)
    assert original.model_sha256 == graph_sha, 'bundle graph hash mismatch'
    validation = read_json(args.staging / 'onnx_validation.json')
    assert validation['passed'], 'exporter parity check did not pass'

    # 2) Publish the deployment bundle: identical graph, authoritative config variant.
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    target_graph = output / graph.name
    shutil.copy2(graph, target_graph)
    assert sha256_file(target_graph) == graph_sha
    metrics = {
        'purpose': 'development',
        'training_purpose': 'target_adaptation',
        'scientific_acceptance': False,
        'software_equivalence_passed': True,
        'source_checkpoint_name': args.checkpoint_name,
        'source_checkpoint_sha256': digest,
        'source_run_manifest_content_sha256': manifest['content_sha256'],
        'old_frozen_dataset_manifest': str(OLD_DATASET / 'dataset_manifest.json'),
        'old_frozen_dataset_content_sha256': read_json(
            OLD_DATASET / 'dataset_manifest.json')['content_sha256'],
        'adaptation_dataset_manifest': str(BASE / 'finetune100/dataset_manifest.snapshot.json'),
        'adaptation_dataset_content_sha256': read_json(
            BASE / 'finetune100/dataset_manifest.snapshot.json')['content_sha256'],
        'selection_rule': 'maximise the worse of the two target images (full-image 2 px F1); '
                          'tie-break by same-ROI F1 then old frozen validation F1',
        'evaluation_radius_px': 2.0,
        'torch_onnx_parity_radius_px': 0.01,
        'labels_used_in_training': True,
        'independent_test': False,
        'role': 'supervised_adaptation_fit_to_user_reviewed_04_05',
        'variant': 'gaussian11_maxshift4',
        'refinement_contract': REFINEMENT,
        'inference_contract': inference,
        'export_manifest': str(args.staging / 'model_manifest.json'),
        'export_validation': str(args.staging / 'onnx_validation.json'),
        'selection_record': str(args.selection_record),
        'graph_sha256': graph_sha,
        'nonempty_validation_cases': validation['nonempty_cases'],
    }
    variant = replace(original, model_id=f'atom-center-target-adaptation-20260910-gaussian11-conf{args.conf:g}',
                      version='development-0.3.0', provider='onnxruntime-cpu-yolov8',
                      inference=inference, refinement=REFINEMENT, metrics=metrics,
                      software=software_versions(), git_commit=manifest['code']['git_commit'])
    variant.write(output / 'model_manifest.json')
    published, published_graph = verify_model_bundle(output / 'model_manifest.json')
    assert sha256_file(published_graph) == graph_sha
    write_json(output / 'deployment_config.json',
               {'model_id': published.model_id, 'model_manifest': str(output / 'model_manifest.json'),
                'model_file': published.model_file, 'model_sha256': published.model_sha256,
                'input': dict(published.input), 'inference': dict(published.inference),
                'refinement': dict(published.refinement), 'metrics': dict(published.metrics),
                'checkpoint': str(checkpoint_path), 'checkpoint_sha256': digest,
                'graph_sha256': graph_sha, 'source_export_manifest': str(args.staging / 'model_manifest.json')})
    print(json.dumps({'bundle': str(output / 'model_manifest.json'), 'model_sha256': published.model_sha256,
                      'checkpoint': str(checkpoint_path), 'checkpoint_sha256': digest,
                      'conf': args.conf, 'graph_sha256': graph_sha}, indent=1), flush=True)


if __name__ == '__main__':
    main()
