"""
Batch GPA processing over image stacks and frame sequences.

The engine is GUI-agnostic: it consumes a frame iterator, runs the full
per-frame GPA workflow (first-frame G as initial guess, optional per-frame
reference-region refinement), and streams every requested field to either a
multi-page result stack or numbered single-frame files. Per-frame statistics
and G-drift records are collected for time-series analysis.

Unreliable pixels are NaN in all numeric exports and excluded from the
statistics, independent of what the GUI currently displays.
"""

import datetime as _datetime
import json
import os
import platform
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from .cmaps import NAN_GREY
from .gpa import GPA
from .metadata import json_pixel_size
from .phase import ComputationCancelled
from .stack_reader import StackInfo

__all__ = [
    'BATCH_FIELDS', 'DEFAULT_FIELDS', 'BatchConfig', 'run_batch',
    'result_field', 'refine_mask_from_roi', 'write_preview_stacks',
]

# Selectable export fields (subset of GPAOutput attribute names).
BATCH_FIELDS: Dict[str, str] = {
    'eps_xx': '应变 ε_xx',
    'eps_yy': '应变 ε_yy',
    'eps_xy': '应变 ε_xy',
    'omega_xy': '旋转 ω_xy',
    'dilatation': '膨胀率 Δ',
    'e_xx': '畸变 e_xx',
    'e_yy': '畸变 e_yy',
    'e_xy': '畸变 e_xy',
    'e_yx': '畸变 e_yx',
    'u_x': '位移 u_x (nm)',
    'u_y': '位移 u_y (nm)',
    'phase1': '相位 P_g1 (rad)',
    'phase2': '相位 P_g2 (rad)',
    'quality_mask': '可靠像素掩膜',
}
DEFAULT_FIELDS = ['eps_xx', 'eps_yy', 'eps_xy', 'omega_xy', 'dilatation']


def result_field(result, name: str) -> Optional[np.ndarray]:
    """Numeric export view of a result field: quality-masked, mask as float32."""
    data = getattr(result, name, None)
    if data is None:
        return None
    if name == 'quality_mask':
        return data.astype(np.float32)
    return np.where(result.quality_mask, data, np.nan)


def refine_mask_from_roi(image_shape: Tuple[int, int],
                         roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Boolean reference mask from an inclusive-left/exclusive-right ROI
    ``(left, top, right, bottom)`` in image coordinates."""
    left, top, right, bottom = roi
    rows, cols = image_shape
    if not (0 <= left < right <= cols and 0 <= top < bottom <= rows):
        raise ValueError(
            f"Reference ROI {roi} does not fit image of shape {image_shape}."
        )
    mask = np.zeros(image_shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


@dataclass
class BatchConfig:
    """Parameters for a batch run; geometry/colour options are per-stack."""

    fields: List[str] = field(default_factory=lambda: list(DEFAULT_FIELDS))
    export_stack: bool = True
    export_sequence: bool = False
    sequence_dirname: str = 'frames'
    preview_stack: bool = True
    # WYSIWYG preview specification, snapshotted from the GUI's first-frame
    # display: {field: {'cmap': name, 'vmin': float, 'vmax': float}}.
    # Missing fields fall back to a robust estimate on the raw first frame.
    preview_specs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # When True, previews apply the quality mask (NaN -> grey) to mirror a
    # masked GUI view; when False, every pixel is coloured like the
    # original Strain++ full-field display.
    preview_masked: bool = False
    refine_g: bool = True
    reference_roi: Optional[Tuple[int, int, int, int]] = None
    frame_start: int = 0
    frame_stop: Optional[int] = None
    frame_step: int = 1
    # First-frame analysis parameters (initial guess for every frame).
    # pixel_size follows GPA.load_image semantics: a positive scalar for
    # square pixels, or a (y_size, x_size) nm pair for anisotropic pixels.
    pixel_size: Union[float, Tuple[float, float]] = 1.0
    use_hann: bool = False
    rotation_degrees: float = 0.0
    sigma1: float = 5.0
    sigma2: float = 5.0
    g1: Tuple[float, float] = (0.0, 0.0)
    g2: Tuple[float, float] = (0.0, 0.0)

    def validate(self) -> None:
        """Fail fast on parameters that would otherwise fail every frame."""
        unknown = [name for name in self.fields if name not in BATCH_FIELDS]
        if unknown:
            raise ValueError(f"Unknown export fields: {unknown}")
        if not self.fields:
            raise ValueError("No export fields selected.")
        if self.frame_step < 1:
            raise ValueError("frame_step must be >= 1.")
        if self.refine_g and self.reference_roi is None:
            raise ValueError(
                "Per-frame G refinement requires a reference ROI; "
                "disable refinement or select a homogeneous region."
            )
        for name, vector in (('g1', self.g1), ('g2', self.g2)):
            values = np.asarray(vector, dtype=float).ravel()
            if values.shape != (2,) or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{name} must be a finite (gx, gy) pair, got {vector!r}."
                )
            if float(np.hypot(values[0], values[1])) < 2.0:
                raise ValueError(
                    f"{name} {tuple(values)} is at or near the DC component "
                    "(radius < 2 FFT pixels); pick a real Bragg peak."
                )
        for sigma_name in ('sigma1', 'sigma2'):
            sigma_value = float(getattr(self, sigma_name))
            if not np.isfinite(sigma_value) or sigma_value <= 0:
                raise ValueError(
                    f"{sigma_name} must be finite and positive, "
                    f"got {sigma_value}."
                )
        sizes = np.asarray(self.pixel_size, dtype=float)
        if sizes.ndim == 0:
            sizes_ok = bool(np.isfinite(sizes) and sizes > 0)
        elif sizes.shape == (2,):
            sizes_ok = bool(
                np.all(np.isfinite(sizes)) and np.all(sizes > 0)
            )
        else:
            sizes_ok = False
        if not sizes_ok:
            raise ValueError(
                "pixel_size must be a positive scalar or a positive "
                f"(y, x) pair, got {self.pixel_size!r}."
            )
        for field_name, spec in (self.preview_specs or {}).items():
            if field_name not in BATCH_FIELDS or field_name == 'quality_mask':
                raise ValueError(
                    f"preview_specs names a non-previewable field: "
                    f"{field_name!r}."
                )
            if not isinstance(spec, dict):
                raise ValueError(
                    f"preview_specs[{field_name!r}] must be a dict, "
                    f"got {type(spec).__name__}."
                )
            vmin, vmax = spec.get('vmin'), spec.get('vmax')
            if vmin is None or vmax is None:
                raise ValueError(
                    f"preview_specs[{field_name!r}] needs both 'vmin' "
                    "and 'vmax'."
                )
            try:
                vmin, vmax = float(vmin), float(vmax)
            except (TypeError, ValueError):
                raise ValueError(
                    f"preview_specs[{field_name!r}] vmin/vmax must be "
                    "numbers."
                ) from None
            if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin >= vmax:
                raise ValueError(
                    f"preview_specs[{field_name!r}] requires finite "
                    f"vmin < vmax, got ({vmin}, {vmax})."
                )


def run_batch(source: str, config: BatchConfig, outdir: str,
              progress_cb: Optional[Callable[[int, int, str], None]] = None,
              cancel_check: Optional[Callable[[], bool]] = None
              ) -> Tuple[List[dict], bool]:
    """Run GPA on every frame of ``source`` and stream results to ``outdir``.

    Returns ``(records, cancelled)`` where ``records`` holds one statistics
    dict per processed frame (also written as CSV/JSON side files). Raises
    :class:`ComputationCancelled` when ``cancel_check`` fires.
    """
    from .stack_reader import open_frame_source

    config.validate()
    os.makedirs(outdir, exist_ok=True)

    info = read_info_only(source)
    total = _planned_frames(info, config)
    sequence_dir = None
    if config.export_sequence:
        sequence_dir = os.path.join(outdir, config.sequence_dirname)
        os.makedirs(sequence_dir, exist_ok=True)

    writers = {}          # field -> float32 (quality-masked) TiffWriter
    preview_writers = {}  # field -> uint8 RGB preview TiffWriter
    preview_state = {}    # per-run state: lut + per-field display ranges
    records: List[dict] = []
    cancelled = False
    t_start = time.time()

    def _progress(done: int, message: str):
        if progress_cb is not None:
            progress_cb(done, total, message)

    try:
        info, frames = open_frame_source(
            source, config.frame_start, config.frame_stop, config.frame_step
        )
        for done, (frame_index, image) in enumerate(frames):
            if cancel_check is not None and cancel_check():
                cancelled = True
                raise ComputationCancelled(
                    f"Batch cancelled at frame {frame_index}."
                )
            record = _process_frame(
                frame_index, image, config, writers, sequence_dir, outdir,
                preview_writers, preview_state,
            )
            records.append(record)
            elapsed = time.time() - t_start
            eta = elapsed / max(done + 1, 1) * (total - done - 1)
            _progress(
                done + 1,
                f"帧 {frame_index + 1}/{info.n_frames} 完成，"
                f"剩余约 {eta:.0f} s",
            )
    except ComputationCancelled:
        cancelled = True
    finally:
        for writer in writers.values():
            writer.close()
        for writer in preview_writers.values():
            writer.close()

    _write_reports(records, config, info, outdir, cancelled, t_start, total)
    return records, cancelled


def read_info_only(source: str) -> StackInfo:
    from .stack_reader import open_frame_source, read_stack_info
    if os.path.isdir(source):
        info, _ = open_frame_source(source)
        return info
    return read_stack_info(source)


def _planned_frames(info: StackInfo, config: BatchConfig) -> int:
    begin = max(0, min(info.n_frames, config.frame_start))
    end = info.n_frames if config.frame_stop is None else \
        max(begin, min(info.n_frames, config.frame_stop))
    return len(range(begin, end, config.frame_step))


def _process_frame(frame_index: int, image: np.ndarray, config: BatchConfig,
                   writers: Dict[str, 'object'], sequence_dir: Optional[str],
                   outdir: str, preview_writers: Optional[Dict] = None,
                   preview_state: Optional[Dict] = None) -> dict:
    """Run GPA on one frame and append its fields to the open writers.

    Previews are WYSIWYG: every field is rendered from the *raw* (unmasked)
    values using the colour map and value range the user sees in the GUI
    (snapshotted into ``config.preview_specs``), matching the original
    Strain++ full-field display. NaN pixels (masked views, failed frames)
    render light grey.

    A frame's pages are fully materialised before anything is written, so a
    compute failure cannot leave the per-field export stacks misaligned.
    """
    t0 = time.time()
    record: dict = {
        'frame': frame_index,
        'status': 'ok',
        'error': '',
        'g1': list(config.g1),
        'g2': list(config.g2),
    }
    try:
        gpa = GPA()
        gpa.load_image(image, pixel_size=config.pixel_size,
                       use_hann=config.use_hann)
        gpa.set_g1(config.g1[0], config.g1[1], sigma=config.sigma1)
        gpa.set_g2(config.g2[0], config.g2[1], sigma=config.sigma2)

        if config.refine_g and config.reference_roi is not None:
            mask = refine_mask_from_roi(image.shape, config.reference_roi)
            corrections1 = gpa.phase1.refine_iterative(mask)
            corrections2 = gpa.phase2.refine_iterative(mask)
            record['g1_refined'] = [gpa.phase1.gx, gpa.phase1.gy]
            record['g2_refined'] = [gpa.phase2.gx, gpa.phase2.gy]
            record['g1_corrections'] = [list(step) for step in corrections1]
            record['g2_corrections'] = [list(step) for step in corrections2]

        if config.rotation_degrees != 0.0:
            gpa.set_rotation(np.radians(config.rotation_degrees))

        result = gpa.compute(
            include_displacement=needs_displacement(config),
            mask_results=False,
        )
        record['valid_fraction'] = float(np.mean(result.quality_mask))
        record['g_condition_number'] = (
            float(result.g_condition_number)
            if result.g_condition_number is not None else None
        )
        for name in ('eps_xx', 'eps_yy', 'eps_xy', 'omega_xy', 'dilatation'):
            if name in config.fields:
                data = np.where(result.quality_mask,
                                getattr(result, name), np.nan)
                # median is the robust central value for GPA fields: phase
                # singularities and crossings leave heavy-tailed outliers
                # that a mean would chase (mean kept for comparison).
                record[f'{name}_median'] = float(np.nanmedian(data))
                record[f'{name}_mean'] = float(np.nanmean(data))
                record[f'{name}_std'] = float(np.nanstd(data))

        do_preview = bool(
            getattr(config, 'preview_stack', False)
            and preview_writers is not None
            and preview_state is not None
        )
        if do_preview and 'lut' not in preview_state:
            preview_state['lut'] = _build_lut_table()
            preview_state['grey'] = np.array(NAN_GREY, dtype=np.float64) * 255.0

        pages: List[Tuple[str, np.ndarray]] = []
        previews: List[Tuple[str, np.ndarray]] = []
        for name in config.fields:
            data = result_field(result, name)
            if data is None:
                continue
            pages.append((name, data.astype(np.float32, copy=False)))
            if do_preview and name != 'quality_mask':
                raw = getattr(result, name)
                preview_data = (
                    np.where(result.quality_mask, raw, np.nan)
                    if config.preview_masked
                    else raw
                )
                if preview_data is None:
                    continue
                vmin, vmax, lut = _preview_range_for(
                    name, preview_data, config, preview_state
                )
                previews.append((name, _render_preview_page(
                    np.asarray(preview_data, dtype=np.float64),
                    vmin, vmax, lut, preview_state['grey'],
                )))

        for name, page in pages:
            if config.export_stack:
                writer = writers.get(name)
                if writer is None:
                    import tifffile
                    writer = tifffile.TiffWriter(
                        os.path.join(outdir, f'batch_{name}.tif'),
                        bigtiff=True,
                    )
                    writers[name] = writer
                writer.write(page, contiguous=False)
            if config.export_sequence and sequence_dir is not None:
                import tifffile
                tifffile.imwrite(
                    os.path.join(
                        sequence_dir,
                        f'frame_{frame_index:06d}_{name}.tif',
                    ),
                    page,
                    metadata={'axes': 'YX', 'invalid_pixels': 'NaN'},
                )
        for name, rgb in previews:
            pwriter = preview_writers.get(name)
            if pwriter is None:
                import tifffile
                pwriter = tifffile.TiffWriter(
                    os.path.join(outdir, f'preview_batch_{name}.tif'),
                    bigtiff=True,
                )
                preview_writers[name] = pwriter
            pwriter.write(rgb, contiguous=False)
    except ComputationCancelled:
        raise
    except Exception as error:  # a bad frame must not kill the batch
        record['status'] = 'error'
        record['error'] = str(error)
    record['elapsed_s'] = round(time.time() - t0, 3)
    return record


def _build_lut_table(cmap_name: Optional[str] = None) -> np.ndarray:
    """256x3 uint8 lookup table for a colour map (default: Strain++ Turbo)."""
    from .cmaps import get_cmap
    cm = None
    if cmap_name:
        try:
            import matplotlib as mpl
            cm = mpl.colormaps[cmap_name]
        except (KeyError, ValueError, TypeError):
            cm = None
    if cm is None:
        cm = get_cmap()
    return (np.clip(cm(np.linspace(0.0, 1.0, 256))[:, :3], 0.0, 1.0)
            * 255.0).astype(np.uint8)


def _preview_range_for(name: str, data: np.ndarray, config: BatchConfig,
                       preview_state: Dict) -> Tuple[float, float, np.ndarray]:
    """Resolve (vmin, vmax, lut) for a field: GUI snapshot first, then a
    robust median/MAD estimate on the frame's raw values.

    A malformed snapshot (missing/inverted/non-finite limits) falls back to
    the robust estimate instead of failing the frame; :meth:`BatchConfig.validate`
    rejects such specs up front for API callers.
    """
    spec = config.preview_specs.get(name) if config.preview_specs else None
    if isinstance(spec, dict):
        try:
            vmin = float(spec['vmin'])
            vmax = float(spec['vmax'])
        except (KeyError, TypeError, ValueError):
            vmin = vmax = None
        if (
            vmin is not None
            and np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax
        ):
            luts = preview_state.setdefault('luts', {})
            key = spec.get('cmap')
            if key not in luts:
                luts[key] = _build_lut_table(key)
            return vmin, vmax, luts[key]
    ranges = preview_state.setdefault('auto_ranges', {})
    if name not in ranges:
        values = np.asarray(data, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            ranges[name] = (0.0, 1.0)
        else:
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            half = 4.0 * 1.4826 * mad
            if not np.isfinite(half) or half <= 0:
                half = float(np.max(np.abs(values))) or 1.0
            ranges[name] = (median - half, median + half)
    return ranges[name][0], ranges[name][1], preview_state['lut']


def _render_preview_page(data: np.ndarray, vmin: float, vmax: float,
                         lut: np.ndarray, grey: np.ndarray) -> np.ndarray:
    """Map one float field to an RGB uint8 page; NaN pixels become grey."""
    safe = np.nan_to_num(
        (data - vmin) / max(vmax - vmin, 1e-12), nan=0.0, posinf=1.0,
        neginf=0.0,
    )
    indices = np.clip(
        np.round(safe * 255.0), 0, 255
    ).astype(np.intp)
    rgb = lut[indices]
    rgb[~np.isfinite(data)] = grey.astype(np.uint8)
    return rgb


def needs_displacement(config: BatchConfig) -> bool:
    return bool({'u_x', 'u_y'} & set(config.fields))


def write_preview_stacks(outdir: str, fields: List[str]) -> List[str]:
    """Render 8-bit colour preview stacks next to the float exports.

    Each ``batch_<field>.tif`` (float32, NaN-masked) gets a companion
    ``preview_batch_<field>.tif``: uint8 RGB pages coloured with the
    original Strain++ Turbo ring. The colour scale is fixed across the
    whole stack using a robust median/MAD estimate of all finite values,
    so frames are directly comparable frame-by-frame; NaN pixels become
    light grey. These previews are for viewing only — the float TIFFs
    remain the quantitative data.
    """
    from .cmaps import NAN_GREY, get_cmap

    turbo = get_cmap()
    lut = (np.clip(turbo(np.linspace(0.0, 1.0, 256))[:, :3], 0.0, 1.0)
           * 255.0).astype(np.uint8)
    grey = np.array(NAN_GREY, dtype=np.float64) * 255.0

    written = []
    for name in fields:
        if name == 'quality_mask':
            continue  # mask preview would be binary; float export suffices
        path = os.path.join(outdir, f'batch_{name}.tif')
        if not os.path.exists(path):
            continue

        import tifffile
        values = []
        with tifffile.TiffFile(path) as tif:
            n_pages = len(tif.pages)
            for page in tif.pages:
                data = page.asarray()
                finite = data[np.isfinite(data)]
                if finite.size:
                    values.append(finite)
        if not values:
            continue

        all_finite = np.concatenate(values)
        median = float(np.median(all_finite))
        mad = float(np.median(np.abs(all_finite - median)))
        half_range = 4.0 * 1.4826 * mad
        if not np.isfinite(half_range) or half_range <= 0:
            half_range = float(np.max(np.abs(all_finite))) or 1.0
        vmin, vmax = median - half_range, median + half_range

        preview_path = os.path.join(outdir, f'preview_batch_{name}.tif')
        with tifffile.TiffFile(path) as tif, \
                tifffile.TiffWriter(preview_path) as writer:
            for page in tif.pages:
                data = page.asarray().astype(np.float64)
                normalized = (data - vmin) / (vmax - vmin)
                # NaN -> index 0 first (undefined int cast otherwise); the
                # grey overwrite below paints those pixels correctly.
                safe = np.nan_to_num(normalized, nan=0.0, posinf=1.0,
                                     neginf=0.0)
                indices = np.clip(
                    np.round(safe * 255.0), 0, 255
                ).astype(np.intp)
                rgb = lut[indices]
                rgb[~np.isfinite(data)] = grey.astype(np.uint8)
                writer.write(rgb, contiguous=False)
        written.append(preview_path)
    return written


# Stable leading columns of batch_statistics.csv; everything after them is
# appended in sorted order so failed frames cannot shift the layout between
# runs.
_FIXED_CSV_COLUMNS = (
    'frame', 'status', 'error', 'elapsed_s',
    'g1', 'g2', 'g1_refined', 'g2_refined',
    'g1_corrections', 'g2_corrections',
    'valid_fraction', 'g_condition_number',
)


def _csv_fieldnames(records: List[dict]) -> List[str]:
    present = {key for record in records for key in record}
    ordered = [name for name in _FIXED_CSV_COLUMNS if name in present]
    ordered.extend(sorted(present.difference(_FIXED_CSV_COLUMNS)))
    return ordered


def _write_reports(records: List[dict], config: BatchConfig, info: StackInfo,
                   outdir: str, cancelled: bool, t_start: float,
                   planned: int) -> None:
    """Write per-frame statistics CSV and the batch provenance JSON."""
    import csv

    csv_path = os.path.join(outdir, 'batch_statistics.csv')
    columns = _csv_fieldnames(records)
    with open(csv_path, 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)

    metadata = {
        'software': 'Strain++ GPA batch',
        'created_utc': _datetime.datetime.now(
            _datetime.timezone.utc
        ).isoformat(),
        'source': os.path.abspath(info.source_path),
        'source_kind': info.kind,
        'stack_frames_total': info.n_frames,
        'stack_shape_yx': [info.height, info.width],
        # The calibration actually used for the analysis (scalar → square
        # pixels, pair → anisotropic), not just the source-file calibration.
        'pixel_size_nm_yx': json_pixel_size(config.pixel_size),
        'source_calibration_nm': info.pixel_size,
        'source_fps': info.fps,
        'frame_range': [config.frame_start,
                        config.frame_stop,
                        config.frame_step],
        'fields': list(config.fields),
        'export_stack': config.export_stack,
        'export_sequence': config.export_sequence,
        'preview_stack': config.preview_stack,
        'refine_g_per_frame': config.refine_g,
        'reference_roi': (
            list(config.reference_roi) if config.reference_roi else None
        ),
        'sigma1_px': config.sigma1,
        'sigma2_px': config.sigma2,
        'g1_initial': list(config.g1),
        'g2_initial': list(config.g2),
        'rotation_degrees': config.rotation_degrees,
        'hann_window': config.use_hann,
        'frames_requested': planned,
        'frames_processed': len(records),
        'frames_failed': sum(
            1 for record in records if record['status'] != 'ok'
        ),
        'cancelled': cancelled,
        'elapsed_s': round(time.time() - t_start, 2),
        'python': platform.python_version(),
    }
    with open(
        os.path.join(outdir, 'batch_metadata.json'),
        'w',
        encoding='utf-8',
        newline='\n',
    ) as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
