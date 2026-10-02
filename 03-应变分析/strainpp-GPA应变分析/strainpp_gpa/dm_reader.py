"""
Reader for Gatan DigitalMicrograph DM3 and DM4 files.

DM3 (version 3) and DM4 (version 4) are tagged binary file formats used by
Gatan's DigitalMicrograph software for electron microscopy data.

Format overview:
- DM3 header: version(4B, BE) + file_size(4B, BE) + byte_order(4B, BE) = 12 bytes
- DM4 header: version(4B, BE) + file_size(8B, BE) + byte_order(4B, BE) = 16 bytes
- After the header, the root tag group starts immediately.
- Tag group: is_sorted(1B) + is_open(1B) + n_tags(4B DM3 / 8B DM4)
- Tag entry: type(1B: 0x14=group, 0x15=data) + label_len(2B) + label
- Tag data: "%%%%" delimiter + info_array_length(4B) + info_array + data

Data type codes (two distinct code spaces, do not mix them up):
    Tag-tree encoded types (info arrays of binary tags, incl. the header
    of the image "Data" blob; see DM_TAG_ENCODED_TYPE_MAP):
        2=int16, 3=int32, 4=uint16, 5=uint32, 6=float32, 7=float64,
        8=bool, 9=char, 10=octet, 11=uint64, 12=uint64
    Image dataType codes (value of the ImageData/DataType tag; see
    DM_TYPE_MAP): 1=int16, 2=float32, 3=complex64, 6=uint8, 7=int32,
    9=int8, 10=uint16, 11=uint32, 12=float64, 13=complex128

Reference:
    Based on the DM3/DM4 format specification documented by the
    ncempy, hyperspy, and pyDM open-source projects.
"""

import struct
import os
import mmap
import numpy as np
from typing import Tuple, Dict, Any, Optional, BinaryIO, List

__all__ = [
    'DM3Error', 'DM_TYPE_MAP', 'DM_TAG_ENCODED_TYPE_MAP', 'is_dm_file',
    'read_dm_file', 'read_dm_file_simple', 'read_tiff',
]


class DM3Error(Exception):
    """Error reading DM3/DM4 file."""
    pass


# DM 图像 dataType 码表：ImageList/.../ImageData/DataType 标量值的语义。
# Source: ncempy.io.dm._DM2NPDataTypes / Gatan dm4io.h GatanDataType 枚举。
# 注意：这不是 tag 树的编码类型表（见下方 DM_TAG_ENCODED_TYPE_MAP）。
# 把两者混用——用编码类型表解释图像 dataType（float32 码 2 当 int16、
# uint16 码 10 当 uint8）——会静默解码出错误数值；反之用本表解释 tag
# info 数组的类型码也会按错误的 itemsize 读取。
DM_TYPE_MAP = {
    1: np.dtype('int16'),
    2: np.dtype('float32'),
    3: np.dtype('complex64'),
    6: np.dtype('uint8'),
    7: np.dtype('int32'),
    9: np.dtype('int8'),
    10: np.dtype('uint16'),
    11: np.dtype('uint32'),
    12: np.dtype('float64'),
    13: np.dtype('complex128'),
}

# DM tag 树「编码类型表」：二进制 tag info 数组中简单类型/数组元素类型的
# 语义（SHORT=2, LONG=3, USHORT=4, ULONG=5, FLOAT=6, DOUBLE=7, BOOLEAN=8,
# CHAR=9, OCTET=10, UINT64=12）。Source: ncempy.io.dm 的 _EncodedTypeDTypes
# 与 _encodedTypeSizes——ncempy 正是用它计算并跳过包括图像 Data 块在内的
# 所有二进制 tag 数组。本文件的 tag 解析（_read_simple_type / _read_array /
# _read_complex_array / read_dm_file_simple 的原始扫描）只应使用这张表；
# 图像 dataType 码一律用上方 DM_TYPE_MAP。
DM_TAG_ENCODED_TYPE_MAP = {
    2: np.dtype('int16'),
    3: np.dtype('int32'),
    4: np.dtype('uint16'),
    5: np.dtype('uint32'),
    6: np.dtype('float32'),
    7: np.dtype('float64'),
    8: np.dtype('uint8'),
    9: np.dtype('uint8'),
    10: np.dtype('uint8'),
    11: np.dtype('uint64'),
    12: np.dtype('uint64'),
}


def is_dm_file(filepath: str) -> bool:
    """
    Check if a file appears to be a DM3 or DM4 file by reading its magic number.

    Parameters
    ----------
    filepath : str
        File path.

    Returns
    -------
    bool
    """
    try:
        with open(filepath, 'rb') as f:
            magic = f.read(4)
            version = struct.unpack('>I', magic)[0]
            return version in (3, 4)
    except Exception:
        return False


def read_dm_file(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read a DM3 or DM4 file and return the image data and metadata.

    Parameters
    ----------
    filepath : str
        Path to the DM3/DM4 file.

    Returns
    -------
    (image, metadata) : (np.ndarray, dict)
        image : 2D numpy array of the image data.
        metadata : dict with keys like 'pixel_size', 'pixel_unit', 'shape', etc.

    Raises
    ------
    DM3Error
        If the file cannot be read or has unsupported features.
    """
    # ncempy is the maintained reference implementation used by openNCEM. It
    # correctly handles thumbnails, nested tag encodings, stacks, and both DM
    # versions. If ncempy is unavailable or rejects the file, fall back to the
    # bounded in-tree parser before giving up.
    try:
        from ncempy.io import dm
    except ImportError:
        dm = None
    ncempy_error: Optional[Exception] = None

    if dm is not None:
        try:
            parsed = dm.dmReader(filepath, dSetNum=0, on_memory=True)
            return _convert_ncempy_result(parsed)
        except Exception as error:
            ncempy_error = error

    try:
        with open(filepath, 'rb') as f:
            reader = _DMReader(f)
            return reader.read()
    except DM3Error as builtin_error:
        if ncempy_error is not None:
            raise DM3Error(
                "Both ncempy and the built-in parser failed to read the DM "
                f"file. ncempy: {ncempy_error}; built-in: {builtin_error}"
            ) from builtin_error
        raise


def _convert_ncempy_result(parsed: Dict[str, Any]):
    """Normalize ncempy's dataset dictionary to this project's 2D contract."""
    if 'data' not in parsed:
        raise DM3Error("ncempy returned no image dataset.")
    source = np.asarray(parsed['data'])
    source_shape = source.shape
    if np.iscomplexobj(source):
        raise DM3Error(
            "The selected DM dataset is complex-valued; select a real-space "
            "intensity image instead."
        )
    if source.ndim < 2:
        raise DM3Error(f"DM dataset is not an image (shape={source_shape}).")
    while source.ndim > 2:
        source = source[0]
    image = source.astype(np.float64, copy=False)

    metadata: Dict[str, Any] = {
        'shape': image.shape,
        'source_shape': source_shape,
        'reader': 'ncempy',
        'calibration_verified': False,
    }
    raw_sizes = parsed.get('pixelSize')
    raw_units = parsed.get('pixelUnit')
    sizes = [] if raw_sizes is None else list(np.asarray(raw_sizes).ravel())
    units = [] if raw_units is None else list(np.asarray(raw_units).ravel())
    if len(sizes) >= 2 and len(units) >= 2:
        y_size, x_size = map(float, sizes[-2:])
        y_unit, x_unit = map(str, units[-2:])
        y_nm = _pixel_size_to_nm(y_size, y_unit)
        x_nm = _pixel_size_to_nm(x_size, x_unit)
        metadata['pixel_size_original'] = (y_size, x_size)
        metadata['pixel_unit'] = (y_unit, x_unit)
        if y_nm is not None and x_nm is not None:
            metadata['pixel_size_y_nm'] = y_nm
            metadata['pixel_size_x_nm'] = x_nm
            metadata['pixel_size'] = (
                float(x_nm) if np.isclose(y_nm, x_nm)
                else (float(y_nm), float(x_nm))
            )
            metadata['calibration_verified'] = True
    return image, metadata


class _DMReader:
    """
    Internal DM3/DM4 file reader implementing the correct tag-based format.

    The reader parses the tag tree to find:
    - ImageData (dimensions + pixel data)
    - Calibrations (pixel size / scale / units)
    """

    def __init__(self, f: BinaryIO):
        self.f = f
        self.version = 0
        self.byte_order = '<'  # data byte order (little-endian typical)
        self._image_data: Optional[np.ndarray] = None
        self._metadata: Dict[str, Any] = {}

        # Internal state for collecting image info from tags
        self._image_dimensions: List[int] = []
        self._image_dtype_code: Optional[int] = None
        self._image_data_raw: Optional[bytes] = None
        self._calibrations: List[Dict[str, Any]] = []

        # Track tag path for contextual parsing
        self._tag_path: List[str] = []
        self._all_images: List[Tuple[np.ndarray, List[int]]] = []

    def read(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Read the complete file."""
        self._read_header()
        self._read_tag_group()

        # Select the best image (largest 2D array)
        if self._all_images:
            # Prefer 2D images
            images_2d = [(img, dims) for img, dims in self._all_images if img.ndim == 2]
            if images_2d:
                best = max(images_2d, key=lambda x: x[0].size)
            else:
                best = max(self._all_images, key=lambda x: x[0].size)
            image = best[0]
            # Unify with the ncempy reference path (_convert_ncempy_result):
            # a declared multi-frame stack is cut to its first frame, so
            # read_dm_file returns the same 2D image whether or not ncempy
            # is installed.
            while image.ndim > 2:
                image = image[0]
            self._image_data = np.ascontiguousarray(image)
            self._metadata['shape'] = self._image_data.shape

        if self._image_data is None:
            raise DM3Error("No image data found in DM file.")
        if self._image_data.ndim != 2:
            # Mirror the ncempy path (_convert_ncempy_result), which rejects
            # sub-2D datasets instead of returning a flattened array the
            # analysis pipeline cannot use (audit ticket 71: the two paths
            # must agree for the same file).
            raise DM3Error(
                "DM dataset is not a 2D image "
                f"(shape={self._image_data.shape}, no usable dimension tags)."
            )

        # Extract pixel size from calibrations
        if self._calibrations:
            valid_calibrations = [
                c for c in self._calibrations
                if 'scale' in c and np.isfinite(c['scale']) and c['scale'] > 0
            ]
            if valid_calibrations:
                x_cal = valid_calibrations[0]
                y_cal = valid_calibrations[1] if len(valid_calibrations) > 1 else x_cal
                x_nm = _pixel_size_to_nm(x_cal['scale'], x_cal.get('units', ''))
                y_nm = _pixel_size_to_nm(y_cal['scale'], y_cal.get('units', ''))
                self._metadata['pixel_size_original'] = (
                    y_cal['scale'], x_cal['scale']
                )
                self._metadata['pixel_unit'] = (
                    y_cal.get('units', ''), x_cal.get('units', '')
                )
                if x_nm is not None and y_nm is not None:
                    self._metadata['pixel_size_x_nm'] = x_nm
                    self._metadata['pixel_size_y_nm'] = y_nm
                    self._metadata['pixel_size'] = (
                        float(x_nm) if np.isclose(x_nm, y_nm)
                        else (float(y_nm), float(x_nm))
                    )

        return self._image_data, self._metadata

    def _read_header(self):
        """Read the file header (version, size, byte order)."""
        # Version: always big-endian 4 bytes
        version_bytes = self.f.read(4)
        if len(version_bytes) < 4:
            raise DM3Error("File too small to be a DM3/DM4 file.")
        self.version = struct.unpack('>I', version_bytes)[0]
        if self.version not in (3, 4):
            raise DM3Error(f"Unknown DM version: {self.version}")

        # File size
        if self.version == 3:
            size_bytes = self.f.read(4)  # 4-byte file size (not always reliable)
        else:
            size_bytes = self.f.read(8)  # 8-byte file size for DM4
        if len(size_bytes) != (4 if self.version == 3 else 8):
            raise DM3Error(f"Unexpected end of DM file while reading DM{self.version} size header.")

        # Byte order indicator (always stored big-endian)
        # 1 = little-endian data, 0 = big-endian data
        bo_bytes = self.f.read(4)
        if len(bo_bytes) < 4:
            raise DM3Error("Unexpected end of DM file while reading byte-order indicator.")
        bo = struct.unpack('>I', bo_bytes)[0]
        if bo == 1:
            self.byte_order = '<'
        elif bo == 0:
            self.byte_order = '>'
        else:
            raise DM3Error(f"Invalid DM byte-order indicator: {bo}")

    def _read_tag_group(self):
        """
        Read a tag group.

        Structure:
            is_sorted: 1 byte
            is_open: 1 byte
            n_tags: 4 bytes (DM3) or 8 bytes (DM4)
            Then n_tags tag entries follow.
        """
        is_sorted = self._read_bytes(1)
        if len(is_sorted) == 0:
            return
        is_open = self._read_bytes(1)
        if len(is_open) == 0:
            return

        # Tag-tree structural integers are always big-endian.
        if self.version == 3:
            n_tags = self._unpack('I', 4)
        else:
            n_tags = self._unpack('Q', 8)

        # Sanity check to avoid infinite loops on corrupted files
        if n_tags > 100000:
            raise DM3Error(f"Unreasonable tag count: {n_tags}")

        for _ in range(n_tags):
            self._read_tag_entry()

    def _read_tag_entry(self):
        """
        Read a single tag entry (either a group or data).

        Structure:
            type: 1 byte (0x14 = group, 0x15 = data)
            label_length: 2 bytes (DM3) or 8 bytes (DM4)
            label: label_length bytes (ASCII/Latin-1)
            Then: nested group (if type=0x14) or data (if type=0x15)
        """
        type_byte = self._read_bytes(1)
        if len(type_byte) == 0:
            return
        tag_type = type_byte[0]

        # Read label
        # Both DM3 and DM4 use a 16-bit label length.
        label_len = self._unpack('H', 2)

        label = ''
        if label_len >= 10000:
            raise DM3Error(f"Unreasonable DM tag label length: {label_len}")
        if label_len > 0:
            label_bytes = self._read_bytes(label_len)
            label = label_bytes.decode('latin-1', errors='replace')

        # DM4 stores the total encoded tag size after the label.
        if self.version == 4:
            self._read_bytes(8)

        if tag_type == 0x14:  # Tag group
            self._tag_path.append(label)
            self._read_tag_group()
            self._tag_path.pop()
        elif tag_type == 0x15:  # Tag data
            self._read_tag_data(label)
        else:
            raise DM3Error(f"Unknown DM tag type: 0x{tag_type:02x}")

    def _read_tag_data(self, label: str):
        """
        Read tag data entry.

        Structure after label:
            delimiter: 4 bytes "%%%%"
            info_array_length: 4 bytes (number of entries in info array)
            info_array: info_array_length * 4 bytes
            data: variable (depends on type info)
        """
        # Read and verify delimiter
        delim = self._read_bytes(4)
        if delim != b'%%%%':
            raise DM3Error(f"Invalid DM data delimiter: {delim!r}")

        # Info array
        int_fmt, int_size = ('I', 4) if self.version == 3 else ('Q', 8)
        info_len = self._unpack(int_fmt, int_size)
        if info_len == 0 or info_len > 1000:
            raise DM3Error(f"Invalid DM encoded type length: {info_len}")

        info_array = []
        for _ in range(info_len):
            info_array.append(self._unpack(int_fmt, int_size))

        # Parse type info and read data
        data = self._parse_type_and_read_data(info_array)

        # Process the data based on tag path context
        if data is not None:
            self._process_tag_data(label, data)

    def _parse_type_and_read_data(self, info_array: List[int]) -> Any:
        """
        Parse the info_array to determine data type and read the data.

        Encoding rules:
        - info_len == 1: simple type, info_array = [type_code]
        - info_len == 3 and info_array[0] == 20: array
            info_array = [20, element_type_code, array_length]
        - info_len > 1 and info_array[0] == 15: struct
            info_array = [15, 0, n_fields, field1_name, field1_type, ...]
        - info_len > 3 and info_array[0] == 20: array of struct
            info_array = [20, 15, 0, n_fields, ...types..., array_length]
        """
        if len(info_array) == 0:
            return None

        if len(info_array) == 1:
            # Simple scalar type
            type_code = info_array[0]
            return self._read_simple_type(type_code)

        if info_array[0] == 20:
            # Array type
            if len(info_array) == 3:
                # Simple array: [20, element_type, length]
                elem_type = info_array[1]
                arr_len = info_array[2]
                return self._read_array(elem_type, arr_len)
            else:
                # Array of complex type (struct array)
                # Last element is array length
                arr_len = info_array[-1]
                # The middle part describes the element type
                elem_info = info_array[1:-1]
                return self._read_complex_array(elem_info, arr_len)

        if info_array[0] == 15:
            # Struct type: [15, 0, n_fields, name1, type1, name2, type2, ...]
            return self._read_struct(info_array)

        # Unknown encoding - try to skip gracefully
        return None

    def _read_simple_type(self, type_code: int) -> Any:
        """Read a single value of the given type code."""
        if type_code not in DM_TAG_ENCODED_TYPE_MAP:
            return None
        dtype = DM_TAG_ENCODED_TYPE_MAP[type_code]
        raw = self._read_bytes(dtype.itemsize)
        if len(raw) < dtype.itemsize:
            return None
        val = np.frombuffer(raw, dtype=dtype.newbyteorder(self.byte_order))[0]
        return val

    def _read_array(self, elem_type_code: int, length: int) -> Optional[np.ndarray]:
        """Read an array of simple type."""
        if elem_type_code not in DM_TAG_ENCODED_TYPE_MAP:
            return None
        if length <= 0 or length > 250_000_000:
            return None

        dtype = DM_TAG_ENCODED_TYPE_MAP[elem_type_code]
        data_size = length * dtype.itemsize
        remaining = self._remaining_bytes()
        if data_size > remaining:
            raise DM3Error(
                f"DM array declares {data_size} bytes but only {remaining} remain."
            )
        raw = self._read_bytes(data_size)
        if len(raw) < data_size:
            return None

        arr = np.frombuffer(raw, dtype=dtype.newbyteorder(self.byte_order))
        return arr.copy()  # copy to own the data

    def _read_complex_array(self, elem_info: List[int], arr_len: int) -> Optional[np.ndarray]:
        """Read an array of complex (struct) elements."""
        if arr_len <= 0 or arr_len > 10_000_000:
            return None

        # Determine element size from struct info
        if len(elem_info) >= 3 and elem_info[0] == 15:
            # Struct: [15, 0, n_fields, name1, type1, ...]
            n_fields = elem_info[2]
            field_types = []
            idx = 3
            for _ in range(n_fields):
                if idx + 1 < len(elem_info):
                    # field_name = elem_info[idx]  # (encoded as int, not useful)
                    field_type = elem_info[idx + 1]
                    field_types.append(field_type)
                    idx += 2

            # Calculate element size
            elem_size = 0
            for ft in field_types:
                if ft in DM_TAG_ENCODED_TYPE_MAP:
                    elem_size += DM_TAG_ENCODED_TYPE_MAP[ft].itemsize
                else:
                    return None  # Unknown field type

            if elem_size == 0:
                return None

            # Read all data
            total_size = arr_len * elem_size
            raw = self._read_bytes(total_size)
            if len(raw) < total_size:
                return None

            # Build structured dtype
            dt_list = []
            for i, ft in enumerate(field_types):
                dt_list.append((f'f{i}', DM_TAG_ENCODED_TYPE_MAP[ft].newbyteorder(self.byte_order)))
            struct_dtype = np.dtype(dt_list)

            arr = np.frombuffer(raw, dtype=struct_dtype, count=arr_len)
            # Return first field as the primary data (common for image data)
            if len(field_types) > 0:
                return np.ascontiguousarray(arr['f0']).astype(np.float64)
            return None

        return None

    def _read_struct(self, info_array: List[int]) -> Optional[Dict]:
        """Read a struct value."""
        if len(info_array) < 3:
            return None
        n_fields = info_array[2]
        result = {}
        idx = 3
        for i in range(n_fields):
            if idx + 1 >= len(info_array):
                break
            # field_name_code = info_array[idx]
            field_type = info_array[idx + 1]
            idx += 2
            val = self._read_simple_type(field_type)
            result[f'field_{i}'] = val
        return result

    def _process_tag_data(self, label: str, data: Any):
        """
        Process parsed tag data based on its label and path context.

        Looks for:
        - Image data (large arrays in ImageData context)
        - Calibration scale/units
        """
        path_str = '/'.join(self._tag_path).lower()
        label_lower = label.lower()

        # Detect image data: large 1D arrays that could be image pixels
        if isinstance(data, np.ndarray) and data.ndim == 1:
            # Check if we're in an image data context
            if 'image' in path_str or 'data' in path_str:
                # Store as potential image data
                if data.size > 100:  # Ignore small arrays (metadata)
                    shape = self._shape_from_known_dimensions(data.size)
                    if shape is None:
                        # A declared multi-frame stack cannot be matched with
                        # a dimension pair; use the full declared shape so
                        # read() can cut the first frame like the ncempy
                        # reference path does.
                        shape = self._shape_from_declared_dimensions(data.size)
                    if shape is not None:
                        image = data.reshape(shape)
                        self._all_images.append((image, list(shape)))
                    else:
                        self._all_images.append((data, [data.size]))

        # Detect calibration scale
        if label_lower == 'scale' and isinstance(data, (int, float, np.integer, np.floating)):
            # Store scale in order of appearance
            self._calibrations.append({'scale': float(data)})

        # Detect calibration units
        if label_lower == 'units' and isinstance(data, np.ndarray):
            # String data stored as uint16 array (UTF-16); honour the file's
            # declared byte order rather than assuming little-endian.
            encoding = 'utf-16-le' if self.byte_order == '<' else 'utf-16-be'
            try:
                units_str = data.astype(np.uint16).tobytes().decode(
                    encoding
                ).strip('\x00')
            except Exception:
                try:
                    units_str = data.astype(np.uint8).tobytes().decode('latin-1').strip('\x00')
                except Exception:
                    units_str = ''
            if self._calibrations:
                self._calibrations[-1]['units'] = units_str
            else:
                self._calibrations.append({'units': units_str})

        # Detect dimensions
        if label_lower in ('dimension', 'dimensions') or 'dimension' in path_str:
            if isinstance(data, (int, float, np.integer, np.floating)):
                val = int(data)
                if val > 0:
                    self._image_dimensions.append(val)

    def _count_dim_calibrations(self) -> int:
        """Count how many dimension calibrations we have."""
        return len([c for c in self._calibrations if 'scale' in c])

    # ==========================================================================
    # Low-level I/O helpers
    # ==========================================================================

    def _read_bytes(self, n: int) -> bytes:
        """Read n bytes from the file."""
        if n <= 0:
            return b''
        data = self.f.read(n)
        if len(data) != n:
            raise DM3Error(f"Unexpected end of DM file while reading {n} bytes.")
        return data

    def _unpack(self, fmt: str, size: int) -> int:
        """Read and unpack a tag-tree structural integer (always big-endian).

        In the DM3/DM4 specification the tag tree itself is always stored in
        big-endian order; the header byte-order flag applies only to the data
        payloads read by ``_read_simple_type``/``_read_array``.
        """
        raw = self._read_bytes(size)
        return struct.unpack('>' + fmt, raw)[0]

    def _remaining_bytes(self) -> int:
        current = self.f.tell()
        self.f.seek(0, os.SEEK_END)
        end = self.f.tell()
        self.f.seek(current)
        return max(0, end - current)

    def _shape_from_known_dimensions(self, total: int) -> Optional[Tuple[int, int]]:
        """Return (rows, columns) from the most recent compatible DM dimensions."""
        dims = [int(v) for v in self._image_dimensions if int(v) > 0]
        for end in range(len(dims), 1, -1):
            width, height = dims[end - 2], dims[end - 1]
            if width * height == total:
                return height, width
        return None

    def _shape_from_declared_dimensions(
            self, total: int) -> Optional[Tuple[int, ...]]:
        """Return the full (>=3 dim) declared shape, DM order reversed.

        ``_shape_from_known_dimensions`` only matches dimension pairs, so a
        declared 3D stack used to fall through to the flat 1D payload and
        read() returned a different shape than the ncempy path (audit
        ticket 71).  Matching the full declared shape lets read() reshape
        the stack and slice its first frame.
        """
        dims = [int(v) for v in self._image_dimensions if int(v) > 0]
        for length in range(len(dims), 2, -1):
            window = dims[len(dims) - length:]
            if int(np.prod(window)) == total:
                return tuple(reversed(window))
        return None


def _read_preceding_label(data, found: int, version: int) -> Optional[str]:
    """Recover the tag label preceding a ``%%%%`` delimiter, if trusted.

    Every tag entry places its label immediately before the tag-data
    delimiter: ``[0x15][label_len:u16][label][DM4: total size:u64] %%%%``.
    A delimiter not preceded by such a well-formed structure is almost
    certainly a ``0x25252525`` byte sequence inside binary payload data (a
    phantom delimiter) and must not be interpreted as a tag.  Returns the
    printable-ASCII label (``''`` for legal unnamed tags) or ``None`` when
    the position is untrusted.
    """
    label_end = found - 8 if version == 4 else found
    if label_end - 3 < 0:
        return None
    # 标签长度字段在标签之前（[0x15][len:2][label]），位置依赖未知的
    # label_len，只能枚举：len 字段值 == L 且前置类型字节 0x15 同时成立。
    for label_len in range(0, 65):
        len_pos = label_end - label_len - 2
        if len_pos < 1:
            break
        if struct.unpack('>H', data[len_pos:len_pos + 2])[0] != label_len:
            continue
        if data[len_pos - 1] != 0x15:
            continue
        raw = data[len_pos + 2:label_end]
        if len(raw) != label_len:
            continue
        if any(b < 0x20 or b > 0x7E for b in raw):
            continue
        return raw.decode('ascii')
    return None


def read_dm_file_simple(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fallback DM3/DM4 reader that walks the raw ``%%%%`` tag-data delimiters.

    Only used when both ncempy and the structural parser reject a file (e.g.
    non-standard or corrupted tag structures).  The scan validates the
    tag-entry structure preceding every delimiter (type byte 0x15 plus a
    length-prefixed printable label) so phantom ``0x25252525`` sequences
    inside binary payload data are ignored instead of hijacking the stream.

    The main image block is selected from the file's own metadata, never by
    picking the largest array:

    - a block is a candidate only for a ``Data``-labelled array whose
      declared ``Dimensions`` metadata holds exactly two positive values
      whose product equals the array length; the returned shape comes from
      that declaration (row-major ``(height, width)``), never from
      factorising the element count, so 3D stacks cannot be reshaped into an
      invented 2D shape;
    - when several candidates survive, a ``DataType`` code from
      ``DM_TYPE_MAP`` (real numeric image) disambiguates against
      palette/compressed thumbnail types;
    - if the main image is still not uniquely identified, ``DM3Error`` is
      raised instead of silently guessing.

    Returns
    -------
    (image, metadata) : (np.ndarray, dict)
    """
    file_size = os.path.getsize(filepath)
    if file_size < 16:
        raise DM3Error("File too small to be a DM3/DM4 file.")

    with open(filepath, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
            version = struct.unpack('>I', data[0:4])[0]
            if version not in (3, 4):
                raise DM3Error(f"Unknown DM version: {version}")

            byte_order_offset = 8 if version == 3 else 12
            bo = struct.unpack(
                '>I', data[byte_order_offset:byte_order_offset + 4]
            )[0]
            endian = '<' if bo == 1 else '>'
            encoded_fmt, encoded_size = ('I', 4) if version == 3 else ('Q', 8)

            # Descriptors only while scanning; the selected block is copied
            # after the metadata-based selection has uniquely identified it.
            image_blocks: List[Dict[str, Any]] = []
            pending_dims: List[int] = []
            pending_dtype_code: Optional[int] = None
            delimiter = b'%%%%'
            idx = 0
            while idx < file_size - (4 + encoded_size * 2):
                found = data.find(delimiter, idx)
                if found < 0:
                    break
                label = _read_preceding_label(data, found, version)
                idx = found + 4

                if label is None:
                    # Phantom delimiter inside binary payload data: never
                    # interpret the bytes that follow as an info array.
                    idx = found + 1
                    continue

                if idx + encoded_size > file_size:
                    break
                n_vals = struct.unpack(
                    '>' + encoded_fmt, data[idx:idx + encoded_size]
                )[0]
                idx += encoded_size
                info_bytes = n_vals * encoded_size
                if n_vals < 1 or n_vals > 64 or idx + info_bytes > file_size:
                    idx = found + 1
                    continue

                info_array = [
                    struct.unpack(
                        '>' + encoded_fmt,
                        data[idx + j * encoded_size:idx + (j + 1) * encoded_size],
                    )[0]
                    for j in range(n_vals)
                ]
                idx += info_bytes
                lowered = label.strip().lower()

                if n_vals == 1 and info_array[0] in DM_TAG_ENCODED_TYPE_MAP:
                    # Scalar tag: Dimensions/DataType carry the metadata used
                    # to identify and shape the main image block.
                    scalar_dtype = DM_TAG_ENCODED_TYPE_MAP[info_array[0]]
                    if idx + scalar_dtype.itemsize <= file_size:
                        value = int(np.frombuffer(
                            data,
                            dtype=scalar_dtype.newbyteorder(endian),
                            count=1,
                            offset=idx,
                        )[0])
                        idx += scalar_dtype.itemsize
                        if lowered == 'dimensions' and value > 0 \
                                and len(pending_dims) < 8:
                            pending_dims.append(value)
                        elif lowered == 'datatype':
                            pending_dtype_code = value
                elif (n_vals == 3 and info_array[0] == 20
                        and info_array[1] in DM_TAG_ENCODED_TYPE_MAP):
                    elem_type, arr_len = info_array[1], int(info_array[2])
                    dtype = DM_TAG_ENCODED_TYPE_MAP[elem_type]
                    total_bytes = arr_len * dtype.itemsize
                    fits = arr_len > 0 and idx + total_bytes <= file_size
                    if lowered == 'data':
                        if arr_len > 100 and fits:
                            image_blocks.append({
                                'arr_len': arr_len,
                                'offset': idx,
                                'dtype': dtype.newbyteorder(endian),
                                'dims': tuple(pending_dims),
                                'dtype_code': pending_dtype_code,
                            })
                        if fits:
                            idx += total_bytes
                        pending_dims = []
                        pending_dtype_code = None
                    elif lowered == 'dimensions' and 1 <= arr_len <= 8 and fits:
                        # Array-form Dimensions: {20, elem_type, n_values}.
                        values = np.frombuffer(
                            data,
                            dtype=dtype.newbyteorder(endian),
                            count=arr_len,
                            offset=idx,
                        )
                        idx += total_bytes
                        for value in values:
                            value = int(value)
                            if value > 0 and len(pending_dims) < 8:
                                pending_dims.append(value)
                    elif fits:
                        # Any other labelled array: skip its declared payload
                        # so the scan resumes after it instead of inside it.
                        idx += total_bytes

            if not image_blocks:
                raise DM3Error("No plausible 2D image data found in DM file.")

            # Verify each candidate against the declared dimension metadata.
            verified = []
            for block in image_blocks:
                dims = block['dims']
                if len(dims) == 2 and dims[0] > 0 and dims[1] > 0 \
                        and dims[0] * dims[1] == block['arr_len']:
                    block['shape'] = (dims[1], dims[0])  # (height, width)
                    verified.append(block)

            if not verified:
                for block in image_blocks:
                    dims = block['dims']
                    if len(dims) > 2 and 0 not in dims \
                            and int(np.prod(dims)) == block['arr_len']:
                        raise DM3Error(
                            "DM file declares a "
                            f"{len(dims)}-dimensional dataset (dimensions="
                            f"{list(dims)}), not a 2D image; the heuristic "
                            "fallback reader cannot reshape it without "
                            "inventing a shape."
                        )
                raise DM3Error("No plausible 2D image data found in DM file.")

            # DataType metadata disambiguation: a real numeric image type
            # wins over palette/compressed thumbnail types (e.g. 23).
            if len(verified) > 1:
                numeric = [
                    block for block in verified
                    if block['dtype_code'] in DM_TYPE_MAP
                ]
                if len(numeric) == 1:
                    verified = numeric
            if len(verified) > 1:
                raise DM3Error(
                    f"Found {len(verified)} plausible image data blocks "
                    f"(dims/dtype: "
                    f"{[(block['dims'], block['dtype_code']) for block in verified]}"
                    "); cannot identify the main image without guessing."
                )

            block = verified[0]
            count = block['arr_len']
            largest = np.frombuffer(
                data,
                dtype=block['dtype'],
                count=count,
                offset=block['offset'],
            ).astype(np.float64, copy=True).reshape(block['shape'])
            metadata = {
                'shape': largest.shape,
                'reader': 'heuristic',
                'calibration_verified': False,
            }

            # A heuristic scale is retained as a candidate only; without a
            # verified Units tag it must not silently be labelled as nanometres.
            _extract_pixel_size_heuristic(data, endian, metadata)

    return largest, metadata


def _find_dm_unit_hint(data: bytes, start: int, end: int) -> Optional[str]:
    """Return a recognised DM calibration unit near a Scale label.

    Both the built-in parser's Latin-1 string data and ncempy's UTF-16
    DigitalMicrograph strings are common in the wild, so the search accepts a
    small explicit set of encodings for nm and Å.  Unknown units return None.
    """
    if start < 0:
        start = 0
    if end > len(data):
        end = len(data)
    window = data[start:end]
    for token in (
        b'n\x00m\x00', b'n\x00m', b'nm', b'nanometer', b'nanometre',
        b'\xc5\x00', b'\xc3\x85', b'angstrom', b'angstroms',
    ):
        if token in window:
            return 'nm' if b'nm' in token or b'n\x00m' in token else 'angstrom'
    return None


def _extract_pixel_size_heuristic(data, endian: str, metadata: dict):
    """
    Heuristic extraction of pixel size from DM file binary data.

    This is intentionally strict because the DM tag stream contains many
    arbitrary float64 values.  A Scale label is only used when:

    1. it is followed within a small window by a tag-data delimiter (``%%%%``)
       and a simple encoded type entry, so the value is positioned like a real
       DigitalMicrograph calibration tag; and
    2. a recognised unit token (nm/Å) appears near the label.

    Candidates are stored as ``pixel_size_candidate`` and never promoted to
    ``pixel_size`` unless a calibrated unit has been verified by the caller.
    """
    data_len = len(data)
    idx = 0
    scale_values: List[Tuple[float, Optional[str]]] = []

    while idx < data_len - 30:
        label_len = 0
        found = data.find(b'S\x00c\x00a\x00l\x00e\x00', idx)  # UTF-16LE "Scale"
        if found >= 0:
            label_len = 10
            idx = found
        else:
            found = data.find(b'Scale', max(0, idx))
            if found < 0:
                break
            label_len = 5
            idx = found

        label_end = idx + label_len
        # Position check: a real tag-data entry starts with a delimiter shortly
        # after the label.  Without it this Scale match is a false positive
        # (e.g. an image title or a metadata string containing "Scale").
        delimiter_rel = data.find(b'%%%%', label_end, min(data_len, label_end + 64))
        if delimiter_rel < 0:
            idx += max(1, label_len)
            continue

        # Unit check: only accept Scale values whose unit is present and
        # recognised in the surrounding bytes.
        unit = _find_dm_unit_hint(data, idx, min(data_len, idx + 160))
        if unit is None:
            idx += max(1, label_len)
            continue

        # Look for a float64 value at the expected simple-type position after
        # the label.  The encoded type length + one type code (float64 = 7 in
        # DM3, still 7 in DM4) precedes the payload; accept a tight window.
        for offset in range(8, 56):
            pos = idx + offset
            if pos + 8 > data_len:
                break
            try:
                val = struct.unpack(endian + 'd', data[pos:pos + 8])[0]
            except Exception:
                continue
            # Reasonable pixel size: a positive finite value in [0.0001, 100] nm.
            if 0.0001 < val < 100.0 and np.isfinite(val):
                scale_values.append((float(val), unit))
                break
        idx += max(1, label_len)

    if scale_values:
        value, unit = scale_values[0]
        metadata['pixel_size_candidate'] = value
        metadata['pixel_size_candidate_unit'] = unit


def _pixel_size_to_nm(value: float, unit: str) -> Optional[float]:
    """Convert common DigitalMicrograph calibration units to nanometres."""
    if not np.isfinite(value) or value <= 0:
        return None
    normalized = str(unit or '').strip().lower().replace('μ', 'µ')
    factors = {
        'nm': 1.0,
        'nanometer': 1.0,
        'nanometers': 1.0,
        'nanometre': 1.0,
        'nanometres': 1.0,
        'å': 0.1,
        'a': 0.1,
        'angstrom': 0.1,
        'angstroms': 0.1,
        'pm': 0.001,
        'µm': 1000.0,
        'um': 1000.0,
        'micrometer': 1000.0,
        'micrometre': 1000.0,
        'm': 1.0e9,
    }
    factor = factors.get(normalized)
    return None if factor is None else float(value) * factor


def read_tiff(filepath: str) -> np.ndarray:
    """
    Read a grayscale TIFF image.

    Uses tifffile if available, otherwise falls back to PIL.

    Parameters
    ----------
    filepath : str
        Path to TIFF file.

    Returns
    -------
    np.ndarray (M, N)
        Grayscale image as float64.
    """
    photometric = None
    axes = ''
    try:
        import tifffile
        with tifffile.TiffFile(filepath) as tif:
            if not tif.series:
                raise ValueError("TIFF file contains no image series.")
            series = tif.series[0]
            axes = getattr(series, 'axes', '') or ''
            img = series.asarray()
            try:
                photometric = tif.pages[0].photometric.name.upper()
            except Exception:
                photometric = None
    except ImportError:
        try:
            from PIL import Image
            img = np.array(Image.open(filepath))
        except ImportError:
            raise ImportError(
                "Either 'tifffile' or 'PIL' (Pillow) is required to read TIFF files."
            )

    img = np.asarray(img)

    # Reduce time/depth/page dimensions by selecting the first image. A sample
    # axis or RGB photometric tag is handled separately as colour data.
    is_rgb = (
        photometric == 'RGB'
        or axes.endswith('S')
        or (photometric is None and img.ndim == 3 and img.shape[-1] in (3, 4))
    )
    while img.ndim > 3:
        img = img[0]
        axes = axes[1:] if axes else ''

    if img.ndim == 3:
        if is_rgb:
            if img.shape[-1] in (3, 4):
                rgb = img[..., :3].astype(np.float64)
            elif img.shape[0] in (3, 4):
                rgb = np.moveaxis(img[:3], 0, -1).astype(np.float64)
            else:
                raise ValueError(f"Unsupported RGB TIFF shape: {img.shape}.")
            # ITU-R BT.709 luminance coefficients.
            img = (
                0.2126 * rgb[..., 0]
                + 0.7152 * rgb[..., 1]
                + 0.0722 * rgb[..., 2]
            )
        else:
            img = img[0]

    if img.ndim != 2:
        raise ValueError(f"Unsupported TIFF shape: {img.shape}. Expected a 2D image.")
    return img.astype(np.float64, copy=False)
