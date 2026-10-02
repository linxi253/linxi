import importlib.util
import io
import os
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


try:
    import tifffile
except ImportError:  # pragma: no cover - exercised on minimal test installs
    tifffile = None

from dm_reader import (
    DM3Error,
    _DMReader,
    _convert_ncempy_result,
    _extract_pixel_size_heuristic,
    read_dm_file,
    read_dm_file_simple,
    read_tiff,
)


def _build_minimal_dm3(width=10, height=16):
    """Build a minimal DM3 stream the built-in parser can read.

    Layout: header + root tag group containing one "ImageData" group with
    two scalar "Dimensions" tags (width, height) and one float32 "Data" array.
    """
    parts = []
    # Header: version, file size (unused), byte order (1 = little-endian).
    parts.append(struct.pack('>III', 3, 0, 1))
    # Root tag group.
    parts.append(b'\x01\x01')
    parts.append(struct.pack('>I', 1))
    group_label = b'ImageData'
    parts.append(b'\x14')
    parts.append(struct.pack('>H', len(group_label)))
    parts.append(group_label)
    # Nested ImageData group with three tags.
    parts.append(b'\x01\x01')
    parts.append(struct.pack('>I', 3))
    for value in (width, height):
        dim_label = b'Dimensions'
        parts.append(b'\x15')
        parts.append(struct.pack('>H', len(dim_label)))
        parts.append(dim_label)
        parts.append(b'%%%%')
        parts.append(struct.pack('>I', 1))
        parts.append(struct.pack('>I', 3))  # int32
        parts.append(struct.pack('<i', value))
    data_label = b'Data'
    parts.append(b'\x15')
    parts.append(struct.pack('>H', len(data_label)))
    parts.append(data_label)
    parts.append(b'%%%%')
    parts.append(struct.pack('>I', 3))
    parts.append(struct.pack('>III', 20, 6, width * height))  # float32 array
    parts.append(np.arange(width * height, dtype=np.float32).tobytes())
    return b''.join(parts)


def _build_minimal_dm4(width=10, height=16):
    """Build a minimal DM4 stream exercising the 64-bit tag-tree encoding.

    Differences from DM3: 8-byte file-size header field, 64-bit tag counts,
    and an 8-byte total-tag-size field after every tag label.
    """
    parts = []
    # Header: version, 8-byte file size (unused), byte order (1 = little-endian).
    parts.append(struct.pack('>IQI', 4, 0, 1))
    # Root tag group with a 64-bit tag count.
    parts.append(b'\x01\x01')
    parts.append(struct.pack('>Q', 1))
    group_label = b'ImageData'
    parts.append(b'\x14')
    parts.append(struct.pack('>H', len(group_label)))
    parts.append(group_label)
    parts.append(struct.pack('>Q', 0))  # total encoded tag size (skipped)
    # Nested ImageData group with three tags.
    parts.append(b'\x01\x01')
    parts.append(struct.pack('>Q', 3))
    for value in (width, height):
        dim_label = b'Dimensions'
        parts.append(b'\x15')
        parts.append(struct.pack('>H', len(dim_label)))
        parts.append(dim_label)
        parts.append(struct.pack('>Q', 0))
        parts.append(b'%%%%')
        parts.append(struct.pack('>Q', 1))
        parts.append(struct.pack('>Q', 3))  # int32
        parts.append(struct.pack('<i', value))
    data_label = b'Data'
    parts.append(b'\x15')
    parts.append(struct.pack('>H', len(data_label)))
    parts.append(data_label)
    parts.append(struct.pack('>Q', 0))
    parts.append(b'%%%%')
    parts.append(struct.pack('>Q', 3))
    parts.append(struct.pack('>QQQ', 20, 6, width * height))  # float32 array
    parts.append(np.arange(width * height, dtype=np.float32).tobytes())
    return b''.join(parts)


@unittest.skipUnless(tifffile is not None, "tifffile is not installed")
class TIFFTests(unittest.TestCase):
    def test_multipage_stack_reads_first_page(self):
        stack = np.arange(3 * 8 * 10, dtype=np.uint16).reshape(3, 8, 10)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "stack.tif")
            tifffile.imwrite(path, stack, photometric="minisblack")
            image = read_tiff(path)
        np.testing.assert_array_equal(image, stack[0].astype(np.float64))

    def test_rgb_is_converted_to_luminance(self):
        rgb = np.zeros((5, 7, 3), dtype=np.uint8)
        rgb[..., 0] = 100
        rgb[..., 1] = 50
        rgb[..., 2] = 10
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "rgb.tif")
            tifffile.imwrite(path, rgb, photometric="rgb")
            image = read_tiff(path)
        expected = 0.2126 * 100 + 0.7152 * 50 + 0.0722 * 10
        np.testing.assert_allclose(image, expected, atol=1e-12)


class DMReaderTests(unittest.TestCase):
    def test_dm3_and_dm4_tag_group_counts_use_version_sized_integers(self):
        fixtures = [
            struct.pack('>III', 3, 18, 1) + b'\x01\x01' + struct.pack('>I', 0),
            struct.pack('>IQI', 4, 26, 1) + b'\x01\x01' + struct.pack('>Q', 0),
        ]
        for payload in fixtures:
            version = struct.unpack('>I', payload[:4])[0]
            with self.subTest(version=version):
                stream = io.BytesIO(payload)
                reader = _DMReader(stream)
                reader._read_header()
                reader._read_tag_group()
                self.assertEqual(stream.tell(), len(payload))

    def test_dm4_minimal_fixture_reads_image_and_shape(self):
        payload = _build_minimal_dm4()
        reader = _DMReader(io.BytesIO(payload))
        image, metadata = reader.read()
        self.assertEqual(image.shape, (16, 10))
        self.assertEqual(metadata['shape'], (16, 10))
        np.testing.assert_array_equal(
            image,
            np.arange(160, dtype=np.float32).reshape(16, 10),
        )

    def test_ncempy_stack_is_first_sliced_and_calibration_is_normalized(self):
        source = np.arange(2 * 4 * 5).reshape(2, 4, 5)
        image, metadata = _convert_ncempy_result({
            'data': source,
            'pixelSize': [1.0, 2.0, 3.0],
            'pixelUnit': ['frame', 'Å', 'nm'],
        })
        np.testing.assert_array_equal(image, source[0])
        self.assertEqual(metadata['pixel_size'], (0.2, 3.0))
        self.assertTrue(metadata['calibration_verified'])


class DMScaleHeuristicTests(unittest.TestCase):
    def test_scale_without_unit_or_delimiter_is_ignored(self):
        """A bare 'Scale' string plus an arbitrary double must not be trusted."""
        payload = (
            b'Some text Scale more text\x00'
            + struct.pack('<d', 0.025)
            + b'\x00\x00\x00\x00\x00\x00\x00\x00'
        )
        metadata: dict = {}
        _extract_pixel_size_heuristic(payload, '<', metadata)
        self.assertNotIn('pixel_size_candidate', metadata)

    def test_scale_with_delimiter_and_nm_unit_is_detected(self):
        """A realistic Scale tag pattern with a delimiter and nm unit is kept."""
        payload = (
            b'Scale%%%%'
            + struct.pack('<I', 1)
            + struct.pack('<I', 7)
            + struct.pack('<d', 0.025)
            + b'Unitsn\x00m\x00'
        )
        metadata: dict = {}
        _extract_pixel_size_heuristic(payload, '<', metadata)
        self.assertIn('pixel_size_candidate', metadata)
        self.assertEqual(metadata['pixel_size_candidate'], 0.025)
        self.assertEqual(metadata['pixel_size_candidate_unit'], 'nm')


def _dm4_group(label: bytes, inner: bytes) -> bytes:
    return (
        b'\x14' + struct.pack('>H', len(label)) + label
        + struct.pack('>Q', 0) + inner
    )


def _dm4_tag_open(label: bytes, info: list) -> bytes:
    return (
        b'\x15' + struct.pack('>H', len(label)) + label + struct.pack('>Q', 0)
        + b'%%%%' + struct.pack('>Q', len(info))
        + b''.join(struct.pack('>Q', value) for value in info)
    )


def _dm4_image_group(dims, dtype_code, encoded_type, arr_len, payload):
    parts = [b'\x01\x01', struct.pack('>Q', len(dims) + 2)]
    for dim in dims:
        parts.append(_dm4_tag_open(b'Dimensions', [3]) + struct.pack('<i', dim))
    parts.append(_dm4_tag_open(b'DataType', [3]) + struct.pack('<i', dtype_code))
    parts.append(_dm4_tag_open(b'Data', [20, encoded_type, arr_len]) + payload)
    return _dm4_group(b'ImageData', b''.join(parts))


def _dm4_file(*entries):
    return (
        struct.pack('>IQI', 4, 0, 1) + b'\x01\x01' + struct.pack('>Q', len(entries))
        + b''.join(
            _dm4_group(b'', b'\x01\x01' + struct.pack('>Q', len(entry)) + entry)
            for entry in entries
        )
    )


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1
    return True


class ReadDmFileSimpleTests(unittest.TestCase):
    """Heuristic fallback must pick the main image from file metadata."""

    def _write(self, payload: bytes) -> str:
        handle, path = tempfile.mkstemp(suffix='.dm4')
        with os.fdopen(handle, 'wb') as stream:
            stream.write(payload)
        self.addCleanup(os.remove, path)
        return path

    def test_minimal_dm3_and_dm4_single_image(self):
        expected = np.arange(160, dtype=np.float32).reshape(16, 10)
        for builder in (_build_minimal_dm3, _build_minimal_dm4):
            with self.subTest(builder=builder.__name__):
                with tempfile.TemporaryDirectory() as directory:
                    path = os.path.join(directory, f'{builder.__name__}.dm')
                    with open(path, 'wb') as stream:
                        stream.write(builder())
                    image, metadata = read_dm_file_simple(path)
                self.assertEqual(image.shape, (16, 10))
                np.testing.assert_array_equal(image, expected)
                self.assertEqual(metadata['reader'], 'heuristic')

    def test_main_image_wins_over_palette_thumbnail_with_phantom_delimiter(self):
        """A phantom %%%% inside ImageTags must not hijack the block choice.

        Regression for the audit finding: the old scan treated any {20, type,
        len} info array after a raw delimiter as a candidate and jumped over
        its claimed payload, so one planted delimiter swallowed the main
        image's Data tag and the thumbnail was silently returned.
        """
        main = np.arange(256 * 256, dtype=np.float32) * 0.5 + 7.25
        thumb = np.arange(64 * 64, dtype=np.uint8) % 251
        junk1, junk2 = b'A' * 16, b'B' * 32

        blob_core = (
            junk1 + b'%%%%' + struct.pack('>Q', 3)
            + struct.pack('>QQQ', 20, 12, 0) + junk2  # array length planted below
        )
        blob_tag = _dm4_tag_open(b'Blob', [15, 0, 1, 7]) + blob_core
        image_tags = _dm4_group(
            b'ImageTags', b'\x01\x01' + struct.pack('>Q', 1) + blob_tag
        )

        entry_thumb = (
            _dm4_image_group((64, 64), 23, 8, 4096, thumb.tobytes()) + image_tags
        )
        entry_main = _dm4_image_group(
            (256, 256), 2, 6, 65536, main.tobytes()
        )
        placeholder = _dm4_file(entry_thumb, entry_main)
        phantom_data_start = placeholder.find(junk1) + len(junk1) + 4 + 8 + 24
        claim = (len(placeholder) - phantom_data_start - 16) // 8
        while not _is_prime(claim):
            claim -= 1
        planted = bytearray(placeholder)
        planted[phantom_data_start - 8:phantom_data_start] = struct.pack('>Q', claim)
        self.assertGreater(
            phantom_data_start + 8 * claim,
            len(planted) - main.nbytes,
            'planted array must skip past the main Data info array',
        )

        image, _ = read_dm_file_simple(self._write(bytes(planted)))
        self.assertEqual(image.shape, (256, 256))
        np.testing.assert_array_equal(image, main.reshape(256, 256))

    def test_declared_3d_stack_raises_instead_of_fabricating_shape(self):
        """A 4x256x256 stack must raise, not return an invented (512, 512)."""
        stack = np.arange(4 * 256 * 256, dtype=np.float32).reshape(4, 256, 256)
        payload = _dm4_file(
            _dm4_image_group((256, 256, 4), 2, 6, stack.size, stack.tobytes())
        )
        with self.assertRaises(DM3Error) as context:
            read_dm_file_simple(self._write(payload))
        self.assertIn('3-dimensional', str(context.exception))

    def test_ambiguous_numeric_blocks_raise_instead_of_taking_max(self):
        """Two numeric Data blocks with declared dims must raise, not max()."""
        main = np.arange(128 * 128, dtype=np.uint16)
        thumb = np.arange(64 * 64, dtype=np.uint16) + 5000
        payload = _dm4_file(
            _dm4_image_group((64, 64), 10, 4, 4096, thumb.tobytes()),
            _dm4_image_group((128, 128), 10, 4, 16384, main.tobytes()),
        )
        with self.assertRaises(DM3Error) as context:
            read_dm_file_simple(self._write(payload))
        self.assertIn('plausible image data blocks', str(context.exception))


_HAS_NCEMPY = importlib.util.find_spec('ncempy') is not None


@unittest.skipUnless(_HAS_NCEMPY, "ncempy is not installed")
class DMReaderFallbackTests(unittest.TestCase):
    def test_ncempy_failure_falls_back_to_builtin_parser(self):
        payload = _build_minimal_dm3()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'minimal.dm3')
            with open(path, 'wb') as stream:
                stream.write(payload)
            with mock.patch(
                'ncempy.io.dm.dmReader',
                side_effect=RuntimeError('ncempy exploded'),
            ):
                image, metadata = read_dm_file(path)
        self.assertEqual(image.shape, (16, 10))
        self.assertEqual(metadata['shape'], (16, 10))
        np.testing.assert_array_equal(
            image,
            np.arange(160, dtype=np.float32).reshape(16, 10),
        )

    def test_both_parsers_failing_raises_dm3_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'garbage.dm3')
            with open(path, 'wb') as stream:
                stream.write(b'not a dm file at all')
            with mock.patch(
                'ncempy.io.dm.dmReader',
                side_effect=RuntimeError('ncempy exploded'),
            ):
                with self.assertRaises(DM3Error):
                    read_dm_file(path)


if __name__ == "__main__":
    unittest.main()
