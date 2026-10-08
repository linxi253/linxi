#!/usr/bin/env python3
"""
Strain Analysis - Command-line entry point for Geometric Phase Analysis (GPA).

Performs nanoscale strain measurement from HRTEM images using the
Geometric Phase Analysis algorithm.

Usage:
    # Basic analysis with two g-vectors
    python strain_analysis.py image.tif --g1 45.2 -30.1 --g2 -45.2 -30.1

    # With output directory and mask sizes
    python strain_analysis.py image.dm3 --g1 50 28 --sigma1 8.0 --g2 -50 28 --sigma2 8.0 -o results/

    # Auto-detect mask radius and print g-vector estimate
    python strain_analysis.py image.tif --estimate-g

    # With Hann window for reduced edge artifacts
    python strain_analysis.py image.tif --g1 45 -30 --g2 -45 -30 --hann

    # With coordinate rotation
    python strain_analysis.py image.tif --g1 45 -30 --g2 -45 -30 --rotation 15.0

Input formats:
    - TIFF (.tif, .tiff) - grayscale, 8/16/32-bit integer or float
    - DM3 (.dm3) - Gatan DigitalMicrograph version 3
    - DM4 (.dm4) - Gatan DigitalMicrograph version 4

Output:
    - Strain tensor components: eps_xx, eps_xy, eps_yy
    - Distortion tensor components: e_xx, e_xy, e_yx, e_yy
    - Rotation: omega_xy
    - Dilatation (volumetric strain): trace(e)
    - Displacement field: u_x, u_y (optional)

Reference:
    Hytch, M. J., Snoeck, E. & Kilaas, R.
    "Quantitative measurement of displacement and strain fields from HREM micrographs."
    Ultramicroscopy 74, 131-146 (1998).
"""

import argparse
import datetime as _datetime
import json
import os
import platform
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


# The CLI lives inside the package; make the project root importable so
# direct execution (``python strainpp_gpa/cli.py``) still resolves
# ``strainpp_gpa`` from a source checkout.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from strainpp_gpa.gpa import GPA
from strainpp_gpa.dm_reader import read_dm_file, read_dm_file_simple, is_dm_file, read_tiff
from strainpp_gpa.metadata import (
    dependency_versions as _dependency_versions,
    json_pixel_size as _json_pixel_size,
    project_version as _project_version,
)
from strainpp_gpa.utils import detect_bragg_peaks


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        prog='strainpp-gpa',
        description='Geometric Phase Analysis (GPA) for strain measurement from HRTEM images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.tif --g1 45.2 -30.1 --g2 -45.2 -30.1
  %(prog)s image.dm3 --g1 50 28 --sigma1 8.0 --g2 -50 28 --sigma2 8.0 -o results/
  %(prog)s image.tif --estimate-g
  %(prog)s image.tif --g1 45 -30 --g2 -45 -30 --hann --rotation 15.0
        """,
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {_project_version()} (GPL-3.0-or-later)',
    )

    # Input/output
    parser.add_argument('input', type=str,
                        help='Input image file (TIFF, DM3, or DM4)')
    parser.add_argument('-o', '--output', type=str, default='./strain_output',
                        help='Output directory (default: ./strain_output)')
    parser.add_argument(
        '--pixel-size', type=str, default=None, metavar='NM[,NM]',
        help='Pixel size in nm: one isotropic value or two values Y,X '
             '(comma-separated; single argument so it may appear before '
             'the input path)',
    )

    # G-vectors
    parser.add_argument('--g1', type=float, nargs=2, metavar=('GX', 'GY'),
                        help='First g-vector (x, y) in pixel coordinates')
    parser.add_argument('--sigma1', type=float, default=5.0,
                        help='Gaussian mask sigma for g1 (default: 5.0)')
    parser.add_argument('--g2', type=float, nargs=2, metavar=('GX', 'GY'),
                        help='Second g-vector (x, y) in pixel coordinates')
    parser.add_argument('--sigma2', type=float, default=5.0,
                        help='Gaussian mask sigma for g2 (default: 5.0)')

    # Options
    parser.add_argument('--hann', action='store_true',
                        help='Apply Hann window to reduce FFT edge artifacts')
    parser.add_argument('--rotation', type=float, default=0.0,
                        help='Coordinate rotation angle in degrees (default: 0)')
    parser.add_argument('--estimate-g', action='store_true',
                        help='Estimate Bragg peak radius and print g-vector info, then exit')
    parser.add_argument(
        '--reference-roi',
        type=int,
        nargs=4,
        metavar=('X1', 'Y1', 'X2', 'Y2'),
        help='Homogeneous reference rectangle for iterative G refinement',
    )
    parser.add_argument(
        '--crop',
        type=int,
        nargs=4,
        metavar=('X1', 'Y1', 'X2', 'Y2'),
        help='Crop the input in memory before FFT (useful for scale bars)',
    )

    # Export options
    parser.add_argument('--export-all', action='store_true',
                        help='Export all fields (displacement, distortion, strain, rotation, dilatation)')
    parser.add_argument('--format', type=str, default='tiff',
                        choices=['tiff', 'npy', 'txt', 'all'],
                        help='Export format (default: tiff)')
    parser.add_argument('--no-displacement', action='store_true',
                        help='Skip displacement field computation (faster)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow overwriting existing output files')

    args = parser.parse_args()
    if args.pixel_size is not None:
        text = args.pixel_size.strip().replace('，', ',')
        parts = [part.strip() for part in text.split(',') if part.strip()]
        try:
            if len(parts) == 1:
                args.pixel_size = float(parts[0])
            elif len(parts) == 2:
                args.pixel_size = tuple(float(part) for part in parts)
            else:
                parser.error(
                    '--pixel-size accepts one value or two values in Y,X order'
                )
        except ValueError:
            parser.error('--pixel-size values must be numbers')

    # Resolve the output directory and check for existing files up front, so
    # a forgotten --overwrite fails before any expensive computation.
    basename = os.path.splitext(os.path.basename(args.input))[0]
    try:
        os.makedirs(args.output, exist_ok=True)
    except OSError as error:
        print(f"\nError: cannot create output directory {args.output}: {error}")
        sys.exit(1)
    existing = [
        name for name in os.listdir(args.output)
        if name.startswith(f"{basename}_")
    ]
    if existing and not args.overwrite:
        print(
            f"\nError: output files for this input already exist in "
            f"{args.output}. Choose another directory or pass --overwrite."
        )
        sys.exit(1)

    # =========================================================================
    # Load Image
    # =========================================================================
    print(f"Strain++ GPA Analysis {_project_version()} (GPL-3.0-or-later)")
    print("No warranty; see LICENSE for complete terms.")
    print("============================================================")
    print(f"Loading: {args.input}")

    ext = os.path.splitext(args.input)[1].lower()
    pixel_size = args.pixel_size

    try:
        if ext in ('.dm3', '.dm4') or is_dm_file(args.input):
            try:
                image, metadata = read_dm_file(args.input)
            except Exception as primary_error:
                print(
                    "Primary DM reader failed, trying simplified reader: "
                    f"{primary_error}"
                )
                image, metadata = read_dm_file_simple(args.input)

            if 'pixel_size' in metadata and args.pixel_size is None:
                pixel_size = metadata['pixel_size']
                print(f"  Pixel size from verified metadata: {pixel_size} nm")
            elif 'pixel_size_candidate' in metadata and args.pixel_size is None:
                print(
                    "  Warning: an unverified calibration candidate "
                    f"({metadata['pixel_size_candidate']}) was found but not "
                    "applied because its unit is unknown."
                )
        elif ext in ('.tif', '.tiff'):
            image = read_tiff(args.input)
        else:
            print(f"Unsupported format: {ext}. Trying TIFF reader...")
            image = read_tiff(args.input)
    except FileNotFoundError:
        print(f"\nError: File not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError loading image: {e}")
        sys.exit(1)

    if pixel_size is None:
        pixel_size = 1.0
        print("  Warning: no verified calibration found; using 1.0 nm/pixel.")

    # Validate image
    if image.ndim != 2:
        print(f"\nError: Image must be 2D grayscale, got {image.ndim}D array.")
        sys.exit(1)
    if image.shape[0] < 4 or image.shape[1] < 4:
        print(f"\nError: Image too small ({image.shape[1]}x{image.shape[0]}). Minimum 4x4.")
        sys.exit(1)
    if not np.all(np.isfinite(image)):
        print("\nError: image contains NaN or infinite values.")
        sys.exit(1)

    source_shape = image.shape
    crop_record = None
    if args.crop is not None:
        x1, y1, x2, y2 = args.crop
        source_rows, source_cols = source_shape
        if not (
            0 <= x1 < x2 <= source_cols
            and 0 <= y1 < y2 <= source_rows
        ):
            print(
                f"\nError: crop must satisfy 0 <= X1 < X2 <= {source_cols} "
                f"and 0 <= Y1 < Y2 <= {source_rows}."
            )
            sys.exit(1)
        if x2 - x1 < 4 or y2 - y1 < 4:
            print("\nError: cropped image must be at least 4 x 4 pixels.")
            sys.exit(1)
        image = np.ascontiguousarray(image[y1:y2, x1:x2])
        crop_record = [x1, y1, x2, y2]
        print(f"  In-memory crop (X1 Y1 X2 Y2): {crop_record}")

    M, N = image.shape
    print(f"  Image size: {N} x {M} pixels")
    print(f"  Pixel size: {pixel_size} nm")
    print(f"  Value range: [{image.min():.2f}, {image.max():.2f}]")

    # =========================================================================
    # Initialize GPA
    # =========================================================================
    gpa = GPA()
    gpa.load_image(image, pixel_size=pixel_size, use_hann=args.hann)
    if args.hann:
        print("  Hann window: applied")

    # =========================================================================
    # Estimate g-vector (optional)
    # =========================================================================
    if args.estimate_g:
        try:
            radius = gpa.estimate_mask_radius()
        except ValueError as exc:
            print(f"\nError: reliable Bragg peaks could not be detected: {exc}")
            sys.exit(1)
        # Find peaks in power spectrum for user reference
        ps = gpa.get_power_spectrum()
        # forward_fft returns a DC-centred spectrum; no additional shift needed.
        ps_centered = ps
        cy, cx = M // 2, N // 2

        # Use vectorized peak detection (fast)
        peaks = detect_bragg_peaks(ps_centered, radius, n_peaks=10)
        if not peaks:
            print("\nError: no reliable Bragg peak candidates were found.")
            sys.exit(1)
        representative_radius = float(
            np.median([peak[3] for peak in peaks[:6]])
        )
        # The original Strain++ suggests sigma = |G| / 6 (r / (2*3)).
        suggested_sigma = max(1.5, representative_radius / 6.0)
        print("\n  Shortest strong peak radius: "
              f"{radius:.1f} pixels")
        print(
            f"  Candidate-family radius for sigma: "
            f"{representative_radius:.1f} pixels"
        )
        print(
            f"  Mask sigma range: "
            f"{representative_radius / 12.0:.1f} (smooth) to "
            f"{representative_radius / 4.0:.1f} (detailed)"
        )
        print(
            f"  Suggested starting sigma (as original Strain++): "
            f"{suggested_sigma:.1f}"
        )

        print(f"\n  Top Bragg peak candidates (x, y, intensity, radius):")
        for i, (gx, gy, pv, pr) in enumerate(peaks[:10]):
            print(f"    {i+1}. ({gx:6.1f}, {gy:6.1f})  radius={pr:.1f}  intensity={pv:.3f}")

        if not args.g1:
            print("\n  Use --g1 <x> <y> --g2 <x> <y> to set g-vectors from the candidates above.")
            return

    # =========================================================================
    # Set G-Vectors
    # =========================================================================
    if args.g1 is None:
        print("\nError: --g1 is required (unless using --estimate-g).")
        sys.exit(1)

    # Validate parameters
    if (
        not np.isfinite(args.sigma1)
        or not np.isfinite(args.sigma2)
        or args.sigma1 <= 0
        or args.sigma2 <= 0
    ):
        print("\nError: sigma values must be finite and positive.")
        sys.exit(1)
    if not np.all(np.isfinite(pixel_size)) or np.any(np.asarray(pixel_size) <= 0):
        print("\nError: pixel size must be positive.")
        sys.exit(1)

    g1x, g1y = args.g1
    try:
        print(f"\nG-vector 1: ({g1x:.2f}, {g1y:.2f}), sigma={args.sigma1}")
        gpa.set_g1(g1x, g1y, sigma=args.sigma1)

        if args.g2 is not None:
            g2x, g2y = args.g2
            print(f"G-vector 2: ({g2x:.2f}, {g2y:.2f}), sigma={args.sigma2}")
            gpa.set_g2(g2x, g2y, sigma=args.sigma2)
        else:
            print("\nWarning: Only g1 set. Full strain tensor requires two non-colinear g-vectors.")
            print("Use --g2 <x> <y> to set the second g-vector.")
    except (ValueError, RuntimeError) as error:
        print(f"\nError setting G-vectors: {error}")
        sys.exit(1)

    refinement_record = None
    if args.reference_roi is not None:
        if args.g2 is None:
            print("\nError: --reference-roi requires both --g1 and --g2.")
            sys.exit(1)
        x1, y1, x2, y2 = args.reference_roi
        if not (0 <= x1 < x2 <= N and 0 <= y1 < y2 <= M):
            print(
                f"\nError: reference ROI must satisfy "
                f"0 <= X1 < X2 <= {N} and 0 <= Y1 < Y2 <= {M}."
            )
            sys.exit(1)
        reference_mask = np.zeros(image.shape, dtype=bool)
        reference_mask[y1:y2, x1:x2] = True
        try:
            corrections1 = gpa.phase1.refine_iterative(reference_mask)
            corrections2 = gpa.phase2.refine_iterative(reference_mask)
        except ValueError as error:
            print(f"\nError refining G-vectors: {error}")
            sys.exit(1)
        refinement_record = {
            'roi_xyxy': [x1, y1, x2, y2],
            'g1_corrections': [list(step) for step in corrections1],
            'g2_corrections': [list(step) for step in corrections2],
        }
        print(
            f"Refined G1: ({gpa.g1[0]:.4f}, {gpa.g1[1]:.4f}); "
            f"G2: ({gpa.g2[0]:.4f}, {gpa.g2[1]:.4f})"
        )

    # Rotation
    if not np.isfinite(args.rotation):
        print("\nError: rotation must be finite.")
        sys.exit(1)
    if args.rotation != 0.0:
        theta = np.radians(args.rotation)
        gpa.set_rotation(theta)
        print(f"Coordinate rotation: {args.rotation}°")

    # =========================================================================
    # Compute
    # =========================================================================
    print("\nComputing GPA...")
    t0 = time.time()

    if args.g2 is not None:
        try:
            result = gpa.compute(include_displacement=not args.no_displacement)
        except (ValueError, RuntimeError) as error:
            print(f"\nError computing GPA: {error}")
            sys.exit(1)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.2f}s")

        # Print statistics
        print(f"\nResults Summary:")
        print(f"{'Field':<18} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
        print(f"{'-'*58}")
        for name in ['eps_xx', 'eps_xy', 'eps_yy', 'omega_xy', 'dilatation']:
            field = getattr(result, name)
            if field is not None:
                print(
                    f"{name:<18} {np.nanmin(field):>10.6f} "
                    f"{np.nanmax(field):>10.6f} "
                    f"{np.nanmean(field):>10.6f} "
                    f"{np.nanstd(field):>10.6f}"
                )
        valid_fraction = float(np.mean(result.quality_mask))
        finite_strain = np.stack([
            result.eps_xx, result.eps_xy, result.eps_yy
        ])
        finite_strain = finite_strain[np.isfinite(finite_strain)]
        extreme_fraction = (
            float(np.mean(np.abs(finite_strain) > 0.5))
            if finite_strain.size else 0.0
        )
        print(f"\n  Reliable phase pixels: {valid_fraction:.2%}")
        print(f"  G-matrix condition number: {result.g_condition_number:.3f}")
        if extreme_fraction > 0:
            print(
                f"  Warning: {extreme_fraction:.3%} of valid strain values "
                "have |strain| > 50%. Inspect the selected peaks, sigma, "
                "phase defects, and reference region."
            )
    else:
        # Single g-vector: just compute phase
        print("  Single g-vector: phase map only.")
        result = None
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.2f}s")

    # =========================================================================
    # Export
    # =========================================================================
    print(f"\nExporting to: {args.output}")

    try:
        # Save power spectrum for reference
        ps = gpa.get_power_spectrum()
        _save_field(ps, args.output, f'{basename}_power_spectrum', args.format)

        # Save phases, quality-masked to NaN outside the reliable-pixel mask so
        # they share the same semantics as the strain fields.
        phase1_out = gpa.phase1.wrapped_phase
        phase2_out = gpa.phase2.wrapped_phase if args.g2 is not None else None
        quality_mask = (
            result.quality_mask if result is not None
            else gpa.phase1.quality_mask()
        )
        if phase1_out is not None:
            _save_field(
                np.where(quality_mask, phase1_out, np.nan),
                args.output,
                f'{basename}_phase1',
                args.format,
            )
        if phase2_out is not None:
            _save_field(
                np.where(quality_mask, phase2_out, np.nan),
                args.output,
                f'{basename}_phase2',
                args.format,
            )

        if result is not None:
            fields_to_save = {
                'eps_xx': result.eps_xx,
                'eps_xy': result.eps_xy,
                'eps_yy': result.eps_yy,
                'omega_xy': result.omega_xy,
                'dilatation': result.dilatation,
                'quality_mask': result.quality_mask,
            }

            if args.export_all:
                fields_to_save.update({
                    'e_xx': result.e_xx,
                    'e_xy': result.e_xy,
                    'e_yx': result.e_yx,
                    'e_yy': result.e_yy,
                })
                if not args.no_displacement and result.u_x is not None:
                    fields_to_save.update({
                        'u_x': result.u_x,
                        'u_y': result.u_y,
                    })

            for name, field in fields_to_save.items():
                if field is not None:
                    _save_field(field, args.output, f'{basename}_{name}', args.format)

        # Save human-readable metadata.
        with open(
            os.path.join(args.output, f'{basename}_info.txt'),
            'w',
            encoding='utf-8',
            newline='\n',
        ) as f:
            f.write("Strain++ GPA Analysis\n")
            f.write("=====================\n")
            f.write(f"Input: {args.input}\n")
            f.write(f"Image size: {N} x {M} pixels\n")
            f.write(
                f"Source image size: {source_shape[1]} x {source_shape[0]} pixels\n"
            )
            if crop_record:
                f.write(f"Analysis crop (X1 Y1 X2 Y2): {crop_record}\n")
            f.write(f"Pixel size: {pixel_size} nm\n")
            f.write(f"G1: ({gpa.g1[0]:.4f}, {gpa.g1[1]:.4f}), sigma={args.sigma1}\n")
            if args.g2:
                f.write(f"G2: ({gpa.g2[0]:.4f}, {gpa.g2[1]:.4f}), sigma={args.sigma2}\n")
            if args.reference_roi:
                f.write(f"Reference ROI (X1 Y1 X2 Y2): {args.reference_roi}\n")
            if args.rotation:
                f.write(f"Rotation: {args.rotation} degrees\n")
            f.write(f"Hann window: {args.hann}\n")
            f.write(f"Analysis time: {elapsed:.2f}s\n")
            if result is not None:
                f.write(f"Valid fraction: {np.mean(result.quality_mask):.8f}\n")
                f.write(f"Extreme strain fraction (abs > 0.5): {extreme_fraction:.8f}\n")
                f.write(f"G condition number: {result.g_condition_number:.8f}\n")
            f.write("Strain unit: dimensionless (0.01 = 1%)\n")
            f.write("Coordinate convention: +x right, +y down (image coordinates)\n")
            f.write(
                "Rotation sign: mathematically positive (from +x towards +y);\n"
                "              with +y down on screen a positive angle appears clockwise\n"
            )

        metadata_record = {
            'software': 'Strain++ GPA',
            'software_version': _project_version(),
            'created_utc': _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
            'input': os.path.abspath(args.input),
            'source_image_shape_yx': [
                int(source_shape[0]), int(source_shape[1])
            ],
            'image_shape_yx': [int(M), int(N)],
            'analysis_crop_xyxy': crop_record,
            'pixel_size_nm_yx': _json_pixel_size(pixel_size),
            'g1_fft_pixels': [float(gpa.g1[0]), float(gpa.g1[1])],
            'g2_fft_pixels': (
                [float(gpa.g2[0]), float(gpa.g2[1])]
                if args.g2 is not None else None
            ),
            'sigma1_fft_pixels': float(args.sigma1),
            'sigma2_fft_pixels': float(args.sigma2),
            'rotation_degrees': float(args.rotation),
            'hann_window': bool(args.hann),
            'reference_roi_xyxy': (
                refinement_record['roi_xyxy'] if refinement_record else None
            ),
            'g1_refinement_steps': (
                refinement_record['g1_corrections'] if refinement_record else None
            ),
            'g2_refinement_steps': (
                refinement_record['g2_corrections'] if refinement_record else None
            ),
            'phase_fields_quality_masked': True,
            'coordinate_convention': '+x right, +y down',
            'positive_rotation': (
                'mathematically positive (from +x towards +y); with the '
                'image convention +y down this appears clockwise on screen'
            ),
            'array_indexing': '[y, x]',
            'strain_unit': 'dimensionless; 0.01 = 1%',
            'displacement_unit': 'nm',
            'invalid_pixels': 'NaN',
            'phase_field_semantics': (
                'phase1/phase2 are wrapped geometric phases P_g1/P_g2 '
                'in radians; masked pixels are NaN'
            ),
            'valid_fraction': (
                float(np.mean(result.quality_mask)) if result is not None else None
            ),
            'extreme_strain_fraction_abs_gt_0_5': (
                extreme_fraction if result is not None else None
            ),
            'g_condition_number': (
                result.g_condition_number if result is not None else None
            ),
            'python': platform.python_version(),
            'dependency_versions': _dependency_versions(),
        }
        with open(
            os.path.join(args.output, f'{basename}_metadata.json'),
            'w',
            encoding='utf-8',
            newline='\n',
        ) as f:
            json.dump(metadata_record, f, ensure_ascii=False, indent=2)
            f.write('\n')
    except (OSError, ValueError) as error:
        print(f"\nError writing output files: {error}")
        sys.exit(1)

    print(f"  Done! Files saved to {args.output}/")


def _save_field(data: np.ndarray, directory: str, name: str, fmt: str):
    """Save a 2D field to file(s)."""
    if data is None:
        return

    data = np.asarray(data)

    if fmt in ('tiff', 'all'):
        try:
            import tifffile
            tifffile.imwrite(
                os.path.join(directory, f'{name}.tif'),
                data.astype(np.float32),
                metadata={
                    'axes': 'YX',
                    # Key matches the GUI/batch exports so one parser covers all.
                    'invalid_pixels': 'NaN marks unreliable phase pixels',
                },
            )
        except ImportError:
            # Fallback: save as numpy binary
            np.save(os.path.join(directory, f'{name}.npy'), data)
            print("  (tifffile not available, saved as .npy)")

    if fmt in ('npy', 'all'):
        np.save(os.path.join(directory, f'{name}.npy'), data)

    if fmt in ('txt', 'all'):
        if data.size > 4_000_000:
            print(
                f"  Warning: {name}.txt holds {data.size} values (hundreds of "
                "MB of text); prefer tiff/npy for full-resolution fields."
            )
        np.savetxt(os.path.join(directory, f'{name}.txt'), data, fmt='%.8e')


if __name__ == '__main__':
    main()
