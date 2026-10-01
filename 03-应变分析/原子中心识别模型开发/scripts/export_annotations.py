"""Export reviewed point annotations as auditable YOLO training artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from atom_center.annotations import AnnotationProject, export_yolo_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, help="annotation project JSON")
    parser.add_argument("output", type=Path, help="new output directory")
    parser.add_argument("--box-size", type=float, default=8.0)
    parser.add_argument("--low", type=float, default=1.0)
    parser.add_argument("--high", type=float, default=99.0)
    args = parser.parse_args()

    project = AnnotationProject.load(args.project)
    summary = export_yolo_dataset(
        project,
        args.output,
        box_size_px=args.box_size,
        percentile_low=args.low,
        percentile_high=args.high,
    )
    print(
        f"exported {summary.source_images} source planes, "
        f"{summary.derived_images} derived images and {summary.points} points "
        f"to {summary.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
