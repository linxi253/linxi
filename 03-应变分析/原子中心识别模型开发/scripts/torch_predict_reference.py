"""CPU PyTorch reference prediction for the reviewed-95 validation round.

Runs the same manifest configuration as the deployed ONNX pipeline but with the PyTorch
checkpoint, in its own process, and writes the same JSON shape the CLI produces so the two can be
compared directly.
"""
from pathlib import Path
import argparse
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
from atom_center.backends import TorchBackend                             # noqa: E402
from atom_center.configuration import pipeline_config                     # noqa: E402
from atom_center.model_manifest import sha256_file, verify_model_bundle   # noqa: E402
from atom_center.pipeline import DetectionPipeline                        # noqa: E402
from atom_center.source_identity import observe_source                    # noqa: E402
from atom_center.image_io import load_image                               # noqa: E402
from atom_center.storage import write_json                                # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--series', type=int, default=0)
    parser.add_argument('--frame', type=int)
    parser.add_argument('--roi', type=int, nargs=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    bundle, graph = verify_model_bundle(args.manifest)
    digest = sha256_file(args.checkpoint)
    if digest != bundle.metrics.get('source_checkpoint_sha256'):
        raise ValueError('checkpoint does not match the manifest checkpoint hash')
    inference, refinement = dict(bundle.inference), dict(bundle.refinement)
    config = pipeline_config({'inference': inference, 'refinement': refinement})
    backend = TorchBackend(args.checkpoint, expected_sha256=digest, contract=dict(bundle.input),
                           inference=inference, device='cpu')
    pipeline = DetectionPipeline(backend, config)
    identity = observe_source(args.image, series_index=args.series, frame_index=args.frame)
    raw = np.asarray(load_image(args.image, normalize=False, series_index=args.series,
                                frame_index=args.frame).image, dtype=np.float64)
    result = pipeline.detect(raw, roi=args.roi)
    payload = {'points_xy': result.points.tolist(), 'confidences': result.confidences.tolist(),
               'provider': result.provider, 'model_sha256': result.model_sha256,
               'checkpoint_sha256': digest, 'graph_sha256': sha256_file(graph),
               'image_identity': identity, 'metadata': dict(result.metadata),
               'torch_device': 'cpu'}
    write_json(args.output, payload)
    print(json.dumps({'output': str(args.output), 'points': len(payload['points_xy']),
                      'provider': payload['provider']}))


if __name__ == '__main__':
    main()
