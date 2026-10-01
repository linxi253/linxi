# -*- coding: utf-8 -*-
"""TIFF 读写分支测试（回归 2026-09-06：数据集级位深扫描 + 切片数口径；
2026-09-28：单页彩色切片口径 + 位深扫描失败上报）。"""

import numpy as np
import pytest
import tifffile

from io_utils.tiff_handler import (
    load_tiff_slices,
    get_tiff_info,
    get_slice,
    get_slice_count,
    scan_uint16_max_shift,
)


def _write(path, arr):
    tifffile.imwrite(str(path), arr)


class TestLoadTiffSlices:
    def test_multipage_gray_yields_per_page(self, tmp_path):
        p = tmp_path / 'stack.tif'
        _write(p, np.random.randint(0, 255, (3, 16, 16), dtype=np.uint8))
        slices = list(load_tiff_slices(str(p)))
        assert len(slices) == 3
        assert all(s.shape == (16, 16) for s in slices)

    def test_single_gray_page_yields_once(self, tmp_path):
        p = tmp_path / 'single.tif'
        _write(p, np.random.randint(0, 255, (16, 16), dtype=np.uint8))
        slices = list(load_tiff_slices(str(p)))
        assert len(slices) == 1 and slices[0].shape == (16, 16)

    def test_single_rgb_page_yields_once(self, tmp_path):
        p = tmp_path / 'rgb.tif'
        _write(p, np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8))
        slices = list(load_tiff_slices(str(p)))
        assert len(slices) == 1
        assert slices[0].shape == (16, 16, 3)

    def test_uint16_multipage(self, tmp_path):
        p = tmp_path / 'u16.tif'
        _write(p, np.random.randint(0, 65535, (2, 16, 16), dtype=np.uint16))
        slices = list(load_tiff_slices(str(p)))
        assert len(slices) == 2
        assert slices[0].dtype == np.uint16


class TestColorSliceSemantics:
    """回归 2026-09-28：单页彩色 (H,W,3) 曾被当作 H 个"切片"（每片是一行
    (W,3) 条带），预览路径对垃圾数据出结果。"""

    def test_single_color_image_is_one_slice(self):
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        assert get_slice_count(rgb) == 1
        assert get_slice(rgb, 0).shape == (100, 200, 3)
        with pytest.raises(IndexError):
            get_slice(rgb, 1)

    def test_gray_stack_still_indexable(self):
        stack = np.zeros((5, 16, 16), dtype=np.uint8)
        assert get_slice_count(stack) == 5
        assert get_slice(stack, 4).shape == (16, 16)

    def test_multipage_color_stack_indexable(self):
        stack = np.zeros((3, 16, 16, 3), dtype=np.uint8)
        assert get_slice_count(stack) == 3
        assert get_slice(stack, 2).shape == (16, 16, 3)


class TestGetTiffInfo:
    def test_num_slices_matches_generator(self, tmp_path):
        """get_tiff_info 的切片数口径必须与 load_tiff_slices 实际产出一致。"""
        p = tmp_path / 'stack.tif'
        _write(p, np.random.randint(0, 255, (4, 16, 16), dtype=np.uint8))
        info = get_tiff_info(str(p))
        assert info['num_slices'] == len(list(load_tiff_slices(str(p)))) == 4

    def test_info_fields(self, tmp_path):
        p = tmp_path / 'single.tif'
        _write(p, np.zeros((16, 16), dtype=np.uint8))
        info = get_tiff_info(str(p))
        assert info['num_slices'] == 1
        assert info['dtype'] == 'uint8'
        assert info['file_size_mb'] > 0


class TestScanUint16MaxShift:
    def test_no_uint16_returns_none(self, tmp_path):
        _write(tmp_path / 'a.tif', np.zeros((8, 8), dtype=np.uint8))
        shift, failed = scan_uint16_max_shift(['a.tif'], str(tmp_path))
        assert shift is None
        assert failed == []

    def test_dataset_level_shift(self, tmp_path):
        """暗文件与亮文件并存时，按全局 max 统一位深（回归 2026-09-06 P1）。"""
        _write(tmp_path / 'dark.tif', np.full((8, 8), 3000, dtype=np.uint16))
        _write(tmp_path / 'bright.tif', np.full((8, 8), 60000, dtype=np.uint16))
        shift, failed = scan_uint16_max_shift(['dark.tif', 'bright.tif'], str(tmp_path))
        assert shift == 8  # 数据集 max=60000 → 右移 8 位
        assert failed == []

    def test_missing_file_skipped(self, tmp_path):
        _write(tmp_path / 'a.tif', np.full((8, 8), 4095, dtype=np.uint16))
        shift, failed = scan_uint16_max_shift(['a.tif', 'ghost.tif'], str(tmp_path))
        assert shift == 4
        assert failed == ['ghost.tif']  # 2026-09-28：失败文件如实上报

    def test_corrupt_file_skipped(self, tmp_path):
        bad = tmp_path / 'bad.tif'
        bad.write_bytes(b'not a tiff')
        _write(tmp_path / 'a.tif', np.full((8, 8), 4095, dtype=np.uint16))
        shift, failed = scan_uint16_max_shift(['bad.tif', 'a.tif'], str(tmp_path))
        assert shift == 4
        assert failed == ['bad.tif']
