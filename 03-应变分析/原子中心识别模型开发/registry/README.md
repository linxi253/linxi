# Model registry

Each released model occupies `registry/<modality>/<version>/` and contains:

- `model.onnx` (excluded from Git by default)
- `model_manifest.json`
- `sha256.txt`
- `config.yaml`
- `benchmark_report.json`

The manifest digest must match the model before a provider is allowed to load it.
