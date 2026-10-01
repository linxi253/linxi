"""Report the exact local training runtime and GPU visibility."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib import metadata


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="return a non-zero exit code when PyTorch cannot use CUDA",
    )
    args = parser.parse_args()

    report: dict[str, object] = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in (
                "atom-center",
                "numpy",
                "scipy",
                "tifffile",
                "torch",
                "torchvision",
                "ultralytics",
                "onnx",
                "onnxruntime",
                "opencv-python",
                "pytest",
            )
        },
    }

    cuda_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda: dict[str, object] = {
            "available": cuda_available,
            "torch_cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        }
        if cuda_available:
            device = torch.cuda.get_device_properties(0)
            cuda.update(
                {
                    "device_0": device.name,
                    "memory_gib": round(device.total_memory / 1024**3, 2),
                    "compute_capability": f"{device.major}.{device.minor}",
                }
            )
        report["cuda"] = cuda
    except Exception as exc:
        report["cuda"] = {"available": False, "error": str(exc)}

    try:
        import onnxruntime

        report["onnxruntime_providers"] = onnxruntime.get_available_providers()
    except Exception as exc:
        report["onnxruntime_providers"] = {"error": str(exc)}

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if args.require_gpu and not cuda_available else 0


if __name__ == "__main__":
    raise SystemExit(main())
