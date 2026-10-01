# Phase 1 acceptance record

Phase 1 establishes the shared runtime and contracts. It does not approve any trained model for quantitative PPA use.

## Required checks

- [x] Independent Python 3.10 virtual environment under `.venv/`.
- [x] Exact top-level package versions and a CUDA-specific PyTorch source.
- [x] Stable `(x, y)` `DetectionResult` and patch-backend contracts.
- [x] Explicit multi-frame TIFF selection; no silent stack averaging.
- [x] Deterministic grouped train/validation/test splitting.
- [x] ROI and tiled-inference coordinate restoration.
- [x] Bright/dark/automatic-polarity sub-pixel refinement.
- [x] One-to-one point matching and lattice-relative thresholds.
- [x] Model manifest and SHA-256 bundle verification.
- [x] Model-free TIFF-to-metrics smoke workflow.
- [x] CUDA PyTorch smoke check on RTX 4070 Ti SUPER (CUDA 12.8).
- [x] PyTorch-to-ONNX Runtime numerical parity.
- [x] Complete locked environment snapshot in `requirements/lock-win-cu128.txt`.

## Validation result

- Date: 2026-08-31
- Tests: 26 passed
- Line/branch coverage: 82% combined
- CUDA tensor smoke: passed on `cuda:0`
- ONNX Runtime providers: `CPUExecutionProvider`, `AzureExecutionProvider`
- Synthetic TIFF pipeline: precision/recall/F1 = 1.0; localization RMSE = 0.1732 px
- Dependency integrity: `pip check` reports no broken requirements

`onnxruntime==1.23.2` failed to initialize on this Windows host, which also has a different system-level `onnxruntime.dll`. The project therefore pins the separately verified CPU runtime `onnxruntime==1.20.1`; no system DLL was modified.

## Coordinate convention

- Points are `(x, y)` in pixels.
- Image origin is the top-left pixel center.
- Arrays are indexed as `image[y, x]`.
- ROIs are half-open `(x0, y0, x1, y1)` bounds.
- Patch backends return patch-local coordinates; the pipeline returns image-global coordinates.

## Promotion rule

A future model can enter `registry/` only with an ONNX file, `model_manifest.json`, matching SHA-256, frozen preprocessing/refinement configuration, dataset-manifest digest, and blind-test report.
