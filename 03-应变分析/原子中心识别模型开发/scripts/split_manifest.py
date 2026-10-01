"""Create a deterministic, group-safe train/val/test manifest."""

from __future__ import annotations

import argparse
import json

from atom_center.splitting import split_manifest_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="source CSV manifest")
    parser.add_argument("output", help="output CSV manifest with a split column")
    parser.add_argument("--group-key", default="acquisition_id")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    args = parser.parse_args()
    result = split_manifest_csv(
        args.input,
        args.output,
        group_key=args.group_key,
        ratios={"train": args.train, "val": args.val, "test": args.test},
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "record_counts": result.record_counts,
                "group_counts": result.group_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
