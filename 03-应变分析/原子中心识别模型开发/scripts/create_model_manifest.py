"""Create release metadata beside an already exported model file."""

from __future__ import annotations

import argparse

from atom_center.model_manifest import ModelManifest

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--modality", choices=("haadf_stem", "hrtem"), required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--output", default="model_manifest.json")
    args = parser.parse_args()
    manifest = ModelManifest.create(
        args.model,
        model_id=args.model_id,
        modality=args.modality,
        version=args.version,
        provider=args.provider,
        dataset_manifest_path=args.dataset_manifest,
    )
    output = manifest.write(args.output)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
