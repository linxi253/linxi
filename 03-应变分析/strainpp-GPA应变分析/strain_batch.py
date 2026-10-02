#!/usr/bin/env python3
"""
Batch Geometric Phase Analysis over an image stack or frame sequence.

Processes every frame of a multi-page TIFF (e.g. a Fiji "Substack" export)
or a folder of numbered TIFFs with the same G-vectors, and streams the
selected strain fields to a multi-page result stack and/or numbered
single-frame files. The first frame's G-vectors are the initial guess;
every frame can be re-refined against a homogeneous reference region to
follow specimen drift during in-situ experiments.

Examples:
    python strain_batch.py Substack.tif --g1 -20.5 -63.3 --g2 -77.2 -20.7 \\
        --sigma1 11 --sigma2 13 --refine-roi 100 100 200 200 -o batch_out

    python strain_batch.py stack.tif --g1 30 0 --g2 0 30 --fields eps_xx,omega_xy \\
        --frame-step 2 --sequence -o out
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_curdir = os.path.dirname(os.path.abspath(__file__))
if _curdir not in sys.path:
    sys.path.insert(0, _curdir)

from strainpp_gpa.batch import (
    BATCH_FIELDS,
    DEFAULT_FIELDS,
    BatchConfig,
    run_batch,
)
from strainpp_gpa.stack_reader import read_stack_info


def _fields_arg(text: str) -> list:
    names = [part.strip() for part in text.split(',') if part.strip()]
    unknown = [name for name in names if name not in BATCH_FIELDS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown fields: {unknown}. Choose from: {sorted(BATCH_FIELDS)}"
        )
    if not names:
        raise argparse.ArgumentTypeError("At least one field is required.")
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='strainpp-gpa-batch',
        description='Batch GPA strain analysis over TIFF stacks / sequences.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Fields: " + ', '.join(sorted(BATCH_FIELDS)),
    )
    parser.add_argument('source', help='Multi-page TIFF stack or frame folder')
    parser.add_argument('-o', '--output', default='./batch_output',
                        help='Output directory (default: ./batch_output)')
    parser.add_argument('--g1', type=float, nargs=2, metavar=('GX', 'GY'),
                        help='First g-vector (initial guess)')
    parser.add_argument('--g2', type=float, nargs=2, metavar=('GX', 'GY'),
                        help='Second g-vector (initial guess)')
    parser.add_argument('--sigma1', type=float, default=5.0)
    parser.add_argument('--sigma2', type=float, default=5.0)
    parser.add_argument('--pixel-size', type=str, default=None,
                        metavar='NM[,NM]',
                        help='nm/pixel: one isotropic value or two values '
                             'Y,X (comma-separated); default: ImageJ '
                             'calibration, else 1.0')
    parser.add_argument('--hann', action='store_true')
    parser.add_argument('--rotation', type=float, default=0.0,
                        help='Coordinate rotation in degrees')
    parser.add_argument('--refine-roi', type=int, nargs=4, default=None,
                        metavar=('X1', 'Y1', 'X2', 'Y2'),
                        help='Homogeneous reference region for per-frame '
                             'G refinement (recommended for in-situ stacks)')
    parser.add_argument('--no-refine', action='store_true',
                        help='Keep the first-frame G-vectors for every frame')
    parser.add_argument('--fields', type=_fields_arg,
                        default=','.join(DEFAULT_FIELDS),
                        help='Comma-separated export fields '
                             '(default: ' + ','.join(DEFAULT_FIELDS) + ')')
    parser.add_argument('--frame-start', type=int, default=0)
    parser.add_argument('--frame-stop', type=int, default=None,
                        help='Exclusive last frame (default: all)')
    parser.add_argument('--frame-step', type=int, default=1)
    parser.add_argument('--sequence', action='store_true',
                        help='Also export numbered single-frame TIFFs')
    parser.add_argument('--sequence-dirname', default='frames')
    parser.add_argument('--no-preview', action='store_true',
                        help='Skip 8-bit colour preview stacks')
    parser.add_argument('--list-only', action='store_true',
                        help='Print stack info and exit')
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    info = read_stack_info(args.source)
    print(
        f"Stack: {args.source}\n"
        f"  frames: {info.n_frames}  size: {info.width}x{info.height}  "
        f"dtype: {info.dtype}  kind: {info.kind}"
    )
    if info.pixel_size is not None:
        print(f"  ImageJ calibration: {info.pixel_size:g} nm/pixel")
    else:
        print("  No ImageJ calibration found; using 1.0 nm/pixel.")
    if info.fps:
        print(f"  Source frame rate: {info.fps:g} fps")
    if args.list_only:
        return
    if args.g1 is None or args.g2 is None:
        parser.error('--g1 and --g2 are required for batch processing.')

    pixel_size = info.pixel_size
    if args.pixel_size is not None:
        text = args.pixel_size.strip().replace('，', ',')
        parts = [part.strip() for part in text.split(',') if part.strip()]
        try:
            if len(parts) == 1:
                pixel_size = float(parts[0])
            elif len(parts) == 2:
                pixel_size = (float(parts[0]), float(parts[1]))
            else:
                parser.error(
                    '--pixel-size accepts one value or two values in Y,X order'
                )
        except ValueError:
            parser.error('--pixel-size values must be numbers')
    if pixel_size is None:
        pixel_size = 1.0

    roi = args.refine_roi
    if roi is not None:
        if not (roi[0] < roi[2] and roi[1] < roi[3]):
            parser.error('--refine-roi must satisfy X1 < X2 and Y1 < Y2.')
        roi = tuple(roi)
    refine = not args.no_refine
    if refine and roi is None:
        print("\nWarning: --refine-roi not given; per-frame G refinement is "
              "disabled (fixed first-frame G-vectors).")

    config = BatchConfig(
        fields=list(args.fields),
        export_stack=True,
        export_sequence=args.sequence,
        sequence_dirname=args.sequence_dirname,
        preview_stack=not args.no_preview,
        refine_g=refine and roi is not None,
        reference_roi=roi if refine else None,
        frame_start=args.frame_start,
        frame_stop=args.frame_stop,
        frame_step=args.frame_step,
        pixel_size=pixel_size,
        use_hann=args.hann,
        rotation_degrees=args.rotation,
        sigma1=args.sigma1,
        sigma2=args.sigma2,
        g1=(args.g1[0], args.g1[1]),
        g2=(args.g2[0], args.g2[1]),
    )

    print(f"\nBatch fields: {', '.join(config.fields)}")
    print(f"Frames: start={config.frame_start} "
          f"stop={config.frame_stop} step={config.frame_step}")
    t0 = time.time()

    def progress(done, total, message):
        bar = '#' * int(30 * done / max(total, 1))
        print(f"\r  [{bar:<30}] {done}/{total}  {message:<28}",
              end='', flush=True)

    records, cancelled = run_batch(
        args.source, config, args.output, progress_cb=progress
    )
    elapsed = time.time() - t0
    failed = sum(1 for record in records if record['status'] != 'ok')
    print(
        f"\n\nDone in {elapsed:.1f}s — processed {len(records)} frames, "
        f"{failed} failed" + (", CANCELLED" if cancelled else "") + "."
    )
    print(f"Results in: {os.path.abspath(args.output)}")
    with open(os.path.join(args.output, 'batch_metadata.json'),
              encoding='utf-8') as stream:
        summary = json.load(stream)
    print(f"Stack outputs: {summary['export_stack']}, "
          f"sequence: {summary['export_sequence']}")
    if cancelled:
        sys.exit(130)


if __name__ == '__main__':
    main()
