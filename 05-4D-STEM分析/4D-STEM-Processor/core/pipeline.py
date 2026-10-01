"""pipeline.py - Shared per-file dataset preparation.

Both the GUI (stem_processor_gui.py) and the CLI batch script
(processing/batch_dpc.py) use :func:`prepare_dataset`, so the extraction /
dimension-fix / preprocessing / beam-centre logic exists exactly once.
"""
import os

import numpy as np

try:
    from core import dm4_io
    from core.dpc_core import find_alpha_from_radial
except ImportError:  # pragma: no cover - direct (non-package) execution
    import dm4_io
    from dpc_core import find_alpha_from_radial


def prepare_dataset(dm4_path, scan_crop=128, log=None, name=None):
    """Read, extract, dimension-fix and preprocess one DM4 dataset.

    Parameters
    ----------
    dm4_path : str - path to the DM4 file
    scan_crop : int - centre crop size in scan pixels
    log : callable, optional - progress sink (e.g. the GUI logger or print)
    name : str, optional - output basename override (the GUI uses it to
        disambiguate DM4 files sharing a basename across subfolders)

    Returns
    -------
    dict with keys: meta, data (scan_y, scan_x, det_y, det_x float32),
    center, alpha, name, scale_nm, dim_info.
    Raises RuntimeError (via dm4_io) when the file cannot be read.
    """
    def _log(msg):
        if log is not None:
            log(msg)

    if name is None:
        name = os.path.splitext(os.path.basename(dm4_path))[0]
        name = name.replace(' ', '_').replace('(', '').replace(')', '')

    _log(f"Reading: {os.path.basename(dm4_path)}")
    meta = dm4_io.read_dm4_metadata(dm4_path)
    _log(f"  Scan: {meta['scan_y']}x{meta['scan_x']}, "
         f"Det: {meta['det_y']}x{meta['det_x']}, "
         f"dtype: {meta['dtype_name']}")

    _log("  Extracting data...")
    data = dm4_io.extract_4d_data(dm4_path, meta, scan_crop)
    data, dim_info = dm4_io.fix_dimensions(data)
    if dim_info.get('swapped'):
        _log(f"  Dimensions FIXED -> {data.shape}")
    elif 'not decisively' in str(dim_info.get('reason', '')):
        _log("  Dimension heuristic ambiguous - kept DM4 header layout "
             f"(scores: scan-first={dim_info.get('score_scan_first'):.3f}, "
             f"det-first={dim_info.get('score_det_first'):.3f})")
    data = dm4_io.preprocess(data)
    scan_y, scan_x, det_y, det_x = data.shape
    _log(f"  Shape: {data.shape}, range: [{data.min():.1f}, {data.max():.1f}]")

    # Beam centre: positive-only CoM of the average diffraction pattern.
    avg_dp = np.maximum(np.mean(data, axis=(0, 1)), 0)
    total = np.sum(avg_dp)
    if total > 0:
        yy, xx = np.mgrid[0:det_y, 0:det_x]
        center = (float(np.sum(yy * avg_dp) / total),
                  float(np.sum(xx * avg_dp) / total))
    else:
        center = (det_y / 2.0, det_x / 2.0)
    alpha, _, _ = find_alpha_from_radial(data, center)
    _log(f"  Center: ({center[0]:.1f}, {center[1]:.1f}), Alpha: {alpha}")

    scale_nm = dm4_io.scan_step_nm(meta)
    scales = meta.get('scales', [])
    units = meta.get('units', [])
    has_nm_scale = any('nm' in str(u).lower() and 0 < float(s) < 1e4
                       for s, u in zip(scales, units))
    if not has_nm_scale:
        # scan_step_nm silently falls back to 1.0; the wrong calibration is
        # more dangerous than a loud warning, since it feeds the
        # charge-density/divergence scaling.
        _log("  WARNING: no nm scan step in the DM4 metadata - charge "
             "density scaling assumes 1.0 nm/px")

    return {
        'meta': meta,
        'data': data,
        'center': center,
        'alpha': alpha,
        'name': name,
        'scale_nm': scale_nm,
        'dim_info': dim_info,
    }
