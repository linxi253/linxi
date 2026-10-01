"""Create release metadata beside an already exported model file."""

from __future__ import annotations

import argparse

from atom_center.model_manifest import ModelManifest


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
