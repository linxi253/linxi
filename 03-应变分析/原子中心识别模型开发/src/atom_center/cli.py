"""Single entry point for annotation audits, frozen datasets, training, and inference."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import yaml
from .storage import read_json, write_json, clean_json

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



def main(argv=None):
    parser = argparse.ArgumentParser(prog="atom-center")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "build"):
        p = sub.add_parser(name)
        p.add_argument("projects", nargs="+")
        p.add_argument("--output", required=True)
        p.add_argument("--selections", help="JSON map: plane_sha256 to selected record_key")
        p.add_argument("--confirm-current-sources", action="store_true",
                       help="Record explicit review of legacy/current file correspondence")
        if name == "build":
            p.add_argument("--tile-size", type=int, default=640)
            p.add_argument("--box-size", type=float, default=8.)
            p.add_argument("--box-policy", choices=["fixed", "spacing"], default="fixed")
            p.add_argument("--low", type=float, default=1.)
            p.add_argument("--high", type=float, default=99.)
            p.add_argument("--seed", type=int, default=20260831)
            p.add_argument("--development", action="store_true")
            p.add_argument("--workflow-only", action="store_true",
                           help="Train/val only; defer missing legacy provenance, retain all source groups and audit failures; no scientific acceptance")
    p = sub.add_parser("migrate")
    p.add_argument("project")
    p.add_argument("--output", required=True)
    for name in ("check", "train"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True)
        p.add_argument("--common")
        p.add_argument("--modality")
        p.add_argument("--set", action="append", default=[], metavar="KEY=YAML_VALUE")
        p.add_argument("--init-sha256")
        if name == "train":
            p.add_argument("--run", required=True)
            p.add_argument("--stop-after-epochs", type=int)
    p = sub.add_parser("resume")
    p.add_argument("--run", required=True)
    p.add_argument("--data", help="Only for a relocated, byte-identical dataset")
    for name in ("export", "validate"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--checkpoint", choices=["best.pt", "best_points.pt", "last.pt"], default="best.pt")
        if name == "export":
            p.add_argument("--output", required=True)
        else:
            p.add_argument("--manifest", required=True)
            p.add_argument("--output", required=True)
    for name in ("predict", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--output", required=True)
        if name == "predict":
            p.add_argument("--image", required=True)
            p.add_argument("--series", type=int, default=0)
            p.add_argument("--frame", type=int)
            p.add_argument("--roi", type=int, nargs=4)
        else:
            p.add_argument("--data", required=True)
            p.add_argument("--split", choices=["train", "val", "test"], default="val")
            p.add_argument("--match-distance", type=float, default=2.)
    args = parser.parse_args(argv)
    try:
        result = _dispatch(args)
        print(json.dumps(clean_json(result), ensure_ascii=False, indent=2))
        return 0 if not isinstance(result, dict) or result.get("ok", result.get("passed", True)) else 2
    except (ValueError, RuntimeError, OSError, ImportError, KeyError, TypeError) as error:
        print(f"atom-center: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


def _dispatch(args):
    if args.command in ("audit", "build", "migrate"):
        from .data_workflow import audit_projects, build_dataset, migrate_project
        if args.command == "migrate":
            return str(migrate_project(args.project, args.output))
        options = {"selections": read_json(args.selections) if args.selections else None,
                   "confirm_current_sources": args.confirm_current_sources}
        if args.command == "audit":
            report = audit_projects(args.projects, **options)
            write_json(args.output, report)
            return {k: v for k, v in report.items() if k != "records"}
        return str(build_dataset(args.projects, args.output, tile_size=args.tile_size,
            box_size_px=args.box_size, box_policy=args.box_policy, percentiles=(args.low, args.high), seed=args.seed,
            development=args.development, workflow_only=args.workflow_only, **options))
    if args.command in ("check", "train"):
        from .configuration import load_config
        from .training import preflight, train_run
        overrides = {}
        for entry in args.set:
            key, separator, value = entry.partition("=")
            if not separator:
                raise ValueError("--set requires KEY=VALUE")
            overrides[key] = yaml.safe_load(value)
        config = load_config(args.common, args.modality, overrides)
        if args.command == "check":
            data, _, initialization = preflight(args.data, config, initialization_sha256=args.init_sha256)
            return {"ok": True, "data_sha256": data["content_sha256"], "config": config,
                    "initialization": initialization}
        return train_run(args.data, args.run, config, initialization_sha256=args.init_sha256,
                         stop_after_epochs=args.stop_after_epochs)
    if args.command == "resume":
        from .training import resume_run
        return resume_run(args.run, data=args.data)
    if args.command in ("export", "validate"):
        from .deployment import export_run, compare_backends
        if args.command == "export":
            return str(export_run(args.run, args.output, checkpoint=args.checkpoint))
        report = compare_backends(args.run, args.manifest, checkpoint=args.checkpoint)
        write_json(args.output, report)
        return report
    from .backends import onnx_pipeline
    provider = onnx_pipeline(args.manifest)
    if args.command == "predict":
        from .image_io import load_image
        from .source_identity import observe_source
        identity = observe_source(args.image, series_index=args.series, frame_index=args.frame)
        raw = load_image(args.image, normalize=False, series_index=args.series, frame_index=args.frame).image
        result = provider.detect(raw, roi=args.roi)
        payload = {"points_xy": result.points.tolist(), "confidences": result.confidences.tolist(),
                   "provider": result.provider, "model_sha256": result.model_sha256,
                   "image_identity": identity, "metadata": dict(result.metadata)}
    else:
        from .evaluation import evaluate_dataset
        payload = evaluate_dataset(provider, args.data, split=args.split, max_distance_px=args.match_distance)
    write_json(args.output, payload)
    return {"output": str(Path(args.output).resolve()), "points": len(payload.get("points_xy", [])),
            **{k: payload[k] for k in ("tp", "fp", "fn", "rmse_px") if k in payload}}


if __name__ == "__main__":
    raise SystemExit(main())
