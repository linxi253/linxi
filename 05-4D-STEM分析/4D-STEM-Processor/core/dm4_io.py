"""dm4_io.py - Unified DM4 reading utilities.

This module is the single source of truth for:
1. Reading DM4 metadata via ncempy
2. Mapping DM4 data-type codes to numpy dtypes
3. Extracting (and cropping) 4D-STEM datacubes
4. Verifying/fixing the scan/detector dimension order
5. Light preprocessing (percentile clipping)

The old approach guessed the dtype from the file size (int16 vs float32).
That silently corrupted uint8/uint16/int32/float64 datasets. Here the dtype
comes from the DM4 header itself, exactly like ``extract_au_correct.py``
originally did.
"""
import os
import numpy as np

try:
    from ncempy.io import dm as _ncempy_dm
    NCEMPY_AVAILABLE = True
except Exception:  # pragma: no cover - depends on environment
    _ncempy_dm = None
    NCEMPY_AVAILABLE = False


# DM4 image dataType codes -> numpy base types.
# Source: ncempy.io.dm._DM2NPDataTypes (the mapping ncempy applies to the
# image data body) / the GatanDataType enum in Gatan's dm4io.h.
# This is NOT ncempy's tag-level encoded-type table (_EncodedTypeDTypes,
# which only serves small binary tag values such as "2: SHORT"): using that
# table for image data silently decodes float32 (code 2) as int16,
# complex64 (3) as int32, uint16 (10) as uint8 and float64 (12) as uint64.
# ncempy only supports little-endian files, so the prefix is always '<'.
DM4_DTYPES = {
    1: np.int16,
    2: np.float32,
    3: np.complex64,
    6: np.uint8,
    7: np.int32,
    9: np.int8,
    10: np.uint16,
    11: np.uint32,
    12: np.float64,
    13: np.complex128,
}

DM4_DTYPE_NAMES = {np.int16: 'int16', np.int32: 'int32', np.uint16: 'uint16',
                   np.uint32: 'uint32', np.float32: 'float32',
                   np.float64: 'float64', np.uint8: 'uint8', np.int8: 'int8',
                   np.complex64: 'complex64',
                   np.complex128: 'complex128'}


def ncempy_available():
    """Return True if the ncempy dependency can be imported."""
    return NCEMPY_AVAILABLE


def numpy_dtype_from_code(code):
    """Map a DM4 image dataType code to a little-endian numpy dtype."""
    base = DM4_DTYPES.get(int(code))
    if base is None:
        raise RuntimeError(
            f"Unsupported DM4 data-type code: {code}. Supported codes: "
            f"{sorted(DM4_DTYPES)}")
    return np.dtype('<' + np.dtype(base).str[1:])


def declared_data_bytes(f, index):
    """Byte count the DM tag tree declares for an image data block.

    ncempy parses every binary tag array header (including the image
    ``Data`` blob) with the tag-level encoded-type sizes and records the
    declared byte count as ``<path>.ImageData.Data.arraySize``, exposed as
    the ``fileDM.dataSize`` list (parallel to ``dataOffset``/``dataType``).
    This is an independent statement from the file itself: the number of
    bytes the tag tree reserved for the data block.

    Returns None when the declaration is unavailable, in which case the
    caller falls back to the file-size bound check only.
    """
    sizes = getattr(f, 'dataSize', None)
    if sizes is None:
        return None
    try:
        value = sizes[index]
    except (TypeError, IndexError, KeyError):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def dtype_display_name(dtype):
    """Short display name for a numpy dtype (e.g. '<i2' -> 'int16')."""
    return DM4_DTYPE_NAMES.get(np.dtype(dtype).type, str(np.dtype(dtype)))


def read_dm4_metadata(dm4_path):
    """Read DM4 file metadata and locate the 4D-STEM object.

    Returns
    -------
    dict with keys: numObjects, objects, 4d_index, scan_x, scan_y, det_x,
    det_y, offset, dtype, dtype_name, scales, units.

    Raises RuntimeError with an actionable message if the file cannot be
    read or contains no 4D-STEM object.
    """
    if not NCEMPY_AVAILABLE:
        raise RuntimeError(
            "ncempy is not available. Install it with: "
            "pip install ncempy")
    if not os.path.isfile(dm4_path):
        raise RuntimeError(f"File not found: {dm4_path}")

    try:
        f = _ncempy_dm.fileDM(dm4_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read DM4 metadata from "
                           f"{dm4_path}: {exc}") from exc

    try:
        info = {'numObjects': int(f.numObjects), 'objects': []}
        obj_4d = None

        for i in range(int(f.numObjects)):
            obj = {
                'ndim': int(f.dataShape[i]),
                'dtype_code': int(f.dataType[i]),
                'xSize': int(f.xSize[i]),
                'ySize': int(f.ySize[i]),
                'zSize': int(f.zSize[i]),
                'zSize2': int(f.zSize2[i]),
            }
            info['objects'].append(obj)
            if obj['ndim'] == 4 or (obj['zSize'] > 1 and obj['zSize2'] > 1):
                obj_4d = i

        if obj_4d is None:
            raise RuntimeError(
                f"No 4D-STEM object found in {os.path.basename(dm4_path)} "
                f"(objects: {info['objects']})")

        obj = info['objects'][obj_4d]
        scan_x, scan_y = obj['xSize'], obj['ySize']
        det_x, det_y = obj['zSize'], obj['zSize2']
        dtype = numpy_dtype_from_code(obj['dtype_code'])
        offset = int(f.dataOffset[obj_4d])

        # Scale/units are stored per dimension in file order
        # (x, y, z, z2). Convert defensively.
        scales = [float(s) for s in f.scale]
        units = [str(u) for u in f.scaleUnit]

        info.update({
            '4d_index': obj_4d,
            'scan_x': scan_x,
            'scan_y': scan_y,
            'det_x': det_x,
            'det_y': det_y,
            'offset': offset,
            'dtype': dtype,
            'dtype_name': dtype_display_name(dtype),
            'scales': scales,
            'units': units,
        })

        # Sanity checks before anything memmaps the data block:
        #
        # 1. Code -> dtype -> itemsize must agree with the byte count the
        #    DM tag tree itself declares for the Data blob (element count
        #    times the encoded-type item size written in the tag header,
        #    recorded by ncempy as ImageData.Data.arraySize). A drifted
        #    dataType table passes the file-size bound below whenever its
        #    itemsize is too small or equal (e.g. float32 declared as 4
        #    bytes/element but decoded as int16 = 2 bytes/element); this
        #    exact-equality check turns that silent corruption into a loud
        #    failure.
        # 2. The header offset + expected byte count must not exceed the
        #    file size. A mismatch means the dtype/dimensions are
        #    inconsistent and the data would be garbage.
        n_elements = scan_y * scan_x * det_y * det_x
        expected_bytes = n_elements * dtype.itemsize
        declared_bytes = declared_data_bytes(f, obj_4d)
        if declared_bytes is not None and declared_bytes != expected_bytes:
            raise RuntimeError(
                f"DM4 dataType code {obj['dtype_code']} maps to {dtype} "
                f"({dtype.itemsize} bytes/element), implying "
                f"{expected_bytes} bytes of data, but the tag tree declares "
                f"{declared_bytes} bytes in {os.path.basename(dm4_path)}. "
                f"The data-type mapping is inconsistent with the file; "
                f"refusing to decode garbage.")
        expected_size = offset + expected_bytes
        actual_size = os.path.getsize(dm4_path)
        if expected_size > actual_size:
            raise RuntimeError(
                f"Dimension/dtype mismatch for {os.path.basename(dm4_path)}: "
                f"header implies {expected_size} bytes but file is only "
                f"{actual_size} bytes")
        info['expected_size'] = expected_size
        return info
    finally:
        try:
            f.fid.close()
        except Exception:
            pass


def extract_4d_data(dm4_path, meta, scan_crop=128):
    """Extract a center-cropped 4D-STEM datacube from a DM4 file.

    DM4 4D-STEM data is laid out in C order as
    (det_y, det_x, scan_y, scan_x); the returned array is
    (scan_y, scan_x, det_y, det_x) float32.

    A single materializing copy is made (native dtype -> float32 in the
    transposed order); the intermediate native-dtype copy of the old
    implementation doubled peak memory for large detectors.
    """
    scan_y, scan_x = meta['scan_y'], meta['scan_x']
    det_y, det_x = meta['det_y'], meta['det_x']
    dtype = meta['dtype']
    offset = meta['offset']

    actual_crop = min(int(scan_crop), scan_y, scan_x)
    sy0 = (scan_y - actual_crop) // 2
    sx0 = (scan_x - actual_crop) // 2

    data = np.memmap(dm4_path, dtype=dtype, mode='r', offset=offset,
                     shape=(det_y, det_x, scan_y, scan_x))
    return (data[:, :, sy0:sy0 + actual_crop, sx0:sx0 + actual_crop]
            .transpose(2, 3, 0, 1)
            .astype(np.float32))


def fix_dimensions(data, method='auto'):
    """Verify and fix the (scan, scan, det, det) dimension order.

    Wraps ``core.dimension_utils``. Returns ``(data, info_dict)``.
    """
    try:
        from core.dimension_utils import fix_dimensions as _fix
    except ImportError:
        from dimension_utils import fix_dimensions as _fix
    return _fix(data, method=method)


def preprocess(data, pmin=1, pmax=None, dtype=np.float32):
    """Percentile-clip and convert a datacube to float32.

    Only the *lower* percentile is applied by default: it trims the most
    negative outliers of background-subtracted data. An upper clip must be
    requested explicitly (``pmax=99``) because a global upper percentile
    destroys the bright-field disk on well-sampled detectors: when the disk
    covers less than ``100 - pmax`` percent of the cube (e.g. a 3 px disk on
    a 512 px detector), every disk pixel is clipped to the background noise
    ceiling and the CoM shift information is lost.

    The clip+cast produces exactly one copy (clip output is cast in place
    when the target dtype already matches).
    """
    p1 = np.percentile(data, pmin)
    if pmax is None:
        out = np.clip(data, p1, None)
    else:
        out = np.clip(data, p1, np.percentile(data, pmax))
    if out.dtype == np.dtype(dtype):
        return out
    return out.astype(dtype, copy=False)


def scan_step_nm(meta):
    """Best-effort scan step size in nm from DM4 scale metadata."""
    scales = meta.get('scales', [])
    units = meta.get('units', [])
    for s, u in zip(scales, units):
        if 'nm' in str(u).lower() and 0 < s < 1e4:
            return float(s)
    return 1.0
