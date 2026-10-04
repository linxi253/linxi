# -*- coding: utf-8 -*-
"""非晶面积统计核心纯函数测试（不导入 GUI/Tk）。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.analysis import (  # noqa: E402
    calculate_radial_growth_rate,
    calculate_growth_statistics,
    calculate_temporal_shape_stability,
    mean_effective_growth_rate,
    effective_growth_rows_mask,
    assess_shape_uniformity,
    generate_summary_text,
)
from core.segmentation import (  # noqa: E402
    ensure_uint8_gray,
    uint16_to_uint8,
    resolve_uint16_shift,
    has_non_finite,
    segment_threshold,
    segment_image,
    SegmentationParams,
)
from core.measurement import (  # noqa: E402
    measure_crystal_properties,
    measure_regions,
    compute_full_measurement,
    create_annotated_image,
)
from io_utils.tiff_handler import _natural_key  # noqa: E402


def _make_df(files=("a.tif",)):
    rows = []
    for f in files:
        for sl in (1, 2, 3):
            rows.append({
                '文件名': f,
                '切片': sl,
                '晶体区域实际面积(nm²)': float(100 * sl),
                '晶体区域周长(nm)': float(10 * sl),
                '晶体区域数量': 1,
                '晶体区域面积比例': 0.5,
                '非晶区域面积比例': 0.5,
                '形状因子_平均值': 0.8,
                '形状因子_标准差': 0.01,
            })
    return pd.DataFrame(rows)


class TestRadialGrowthRate:
    def test_returns_series_aligned_to_original_index(self):
        # 故意打乱行序，确保返回值按原索引对齐
        df = _make_df(["a.tif"])
        df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
        rates = calculate_radial_growth_rate(df, time_interval=2.0)
        assert isinstance(rates, pd.Series)
        assert list(rates.index) == list(df.index)
        assert len(rates) == len(df)
        # 返回序列与 df 行序一致：切片1 的速率等于同文件切片2 的速率。
        s1_idx = df[(df['文件名'] == 'a.tif') & (df['切片'] == 1)].index[0]
        s2_idx = df[(df['文件名'] == 'a.tif') & (df['切片'] == 2)].index[0]
        assert rates.loc[s1_idx] == pytest.approx(rates.loc[s2_idx])

    def test_grouped_by_file_not_cross_file(self):
        df = _make_df(["a.tif", "b.tif"])
        rates = calculate_radial_growth_rate(df, time_interval=1.0)
        assert isinstance(rates, pd.Series)
        assert list(rates.index) == list(df.index)
        # 每个文件第一个切片用第二个切片值填充，不应跨文件跳跃
        a_idx = df[(df['文件名'] == 'a.tif') & (df['切片'] == 1)].index[0]
        a2_idx = df[(df['文件名'] == 'a.tif') & (df['切片'] == 2)].index[0]
        assert rates.loc[a_idx] == pytest.approx(rates.loc[a2_idx])


class TestGrowthStatistics:
    def test_single_file(self):
        df = _make_df(["a.tif"])
        stats = calculate_growth_statistics(df)
        assert stats
        per_file = stats['per_file']
        assert len(per_file) == 1
        assert per_file[0]['文件名'] == 'a.tif'
        assert per_file[0]['初始面积(nm²)'] == pytest.approx(100.0)
        assert per_file[0]['最终面积(nm²)'] == pytest.approx(300.0)
        assert per_file[0]['面积增长率(%)'] == pytest.approx(200.0)

    def test_multi_file_overall_descriptive_only(self):
        df = _make_df(["a.tif", "b.tif"])
        stats = calculate_growth_statistics(df)
        assert stats
        assert len(stats['per_file']) == 2
        overall = stats['overall']
        assert overall['文件数'] == 2
        assert '面积增长率(%)_平均' in overall
        assert '周长增长率(%)_标准差' in overall

    def test_insufficient_data(self):
        df = _make_df(["a.tif"]).iloc[:1]
        assert calculate_growth_statistics(df) == {}


class TestSummaryText:
    def test_summary_mentions_per_file(self):
        df = _make_df(["a.tif", "b.tif"])
        text = generate_summary_text(df, time_interval=1.0)
        assert '按文件分组' in text
        assert 'a.tif' in text


class TestUint8Normalization:
    def test_uint8_unchanged(self):
        arr = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        out = ensure_uint8_gray(arr)
        assert out.dtype == np.uint8
        assert np.array_equal(out, arr)

    def test_uint16_16bit_range_uses_shift8(self):
        # 满量程 16-bit 数据（max > 4095）：右移 8 位，与旧版 /256 等价
        arr = np.array([[256, 512], [1024, 65535]], dtype=np.uint16)
        out = ensure_uint8_gray(arr)
        assert out.dtype == np.uint8
        assert out[0, 0] == 1
        assert out[0, 1] == 2
        assert out[1, 0] == 4
        assert out[1, 1] == 255

    def test_uint16_12bit_not_compressed_to_noise(self):
        # 回归 2026-09-05 P1：12-bit 相机数据（0-4095）旧版固定 /256 会被
        # 压到 0-15，固定阈值 128 / Otsu 全部失效。新行为按 4 位右移。
        arr = np.array([[256, 512], [1024, 4095]], dtype=np.uint16)
        out = ensure_uint8_gray(arr)
        assert out[0, 0] == 16
        assert out[0, 1] == 32
        assert out[1, 0] == 64
        assert out[1, 1] == 255
        # 12-bit 数据经阈值 128 仍能分出前景（旧版整图压到 0-15、零检出）
        params = SegmentationParams(method='threshold', threshold=128)
        mask = segment_threshold(out, params)
        assert mask[1, 1]          # 255
        assert not mask[1, 0]      # 64 < 128
        assert not mask[0, 0]      # 16 < 128

    def test_uint16_to_uint8_low_range_kept(self):
        # max ≤ 255 的 uint16 数据不缩放
        arr = np.array([[10, 200]], dtype=np.uint16)
        out = uint16_to_uint8(arr)
        assert np.array_equal(out, np.array([[10, 200]], dtype=np.uint8))


class TestSegmentationAndMeasurement:
    def test_segment_threshold(self):
        img = np.zeros((20, 20), dtype=np.uint8)
        img[5:15, 5:15] = 200
        params = SegmentationParams(method='threshold', threshold=128)
        mask = segment_threshold(img, params)
        assert mask.shape == (20, 20)
        assert mask.sum() == 100

    def test_measure_square_perimeter_and_area(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        area, perimeter, sf, num, factors = measure_crystal_properties(mask, pixel_size_nm=1.0)
        assert area == pytest.approx(100.0)
        assert perimeter > 0
        assert 0.0 <= sf <= 1.0
        assert num == 1
        assert len(factors) == 1

    def test_region_count_excludes_subpixel_noise(self):
        # 回归 2026-09-05 P2：区域数量曾按全部连通域计数（含 1-2px 噪斑），
        # 与周长/形状因子口径不一致。现统一只统计 ≥3 像素的区域。
        mask = np.zeros((40, 40), dtype=bool)
        mask[2:6, 2:6] = True      # 4×4 区域
        mask[20:24, 20:24] = True  # 4×4 区域
        mask[0, 30] = True         # 1px 噪声
        mask[35, 10:12] = True     # 2px 噪声
        *_, num, factors = measure_crystal_properties(mask, pixel_size_nm=1.0)
        assert num == 2
        assert len(factors) == 2

    def test_dark_foreground_inverts_mask(self):
        # 回归 2026-09-05 P2：TEM 明场衍射衬度下晶体更暗，支持反相
        img = np.full((20, 20), 200, dtype=np.uint8)
        img[5:15, 5:15] = 30  # 暗区 = 晶体
        bright_params = SegmentationParams(method='threshold', threshold=128)
        dark_params = SegmentationParams(method='threshold', threshold=128,
                                         foreground='dark')
        mask_bright, _, info_bright = segment_image(img, bright_params)
        mask_dark, _, info_dark = segment_image(img, dark_params)
        assert mask_bright.sum() == 300      # 亮区为晶体：取周围背景（400-100）
        assert mask_dark.sum() == 100        # 暗区为晶体：取中央暗块
        assert not (mask_bright & mask_dark).any()
        assert info_dark['foreground'] == 'dark'

    def test_invalid_foreground_falls_back_to_bright(self):
        params = SegmentationParams(method='threshold', threshold=128, foreground='side')
        params.validate()
        assert params.foreground == 'bright'

    def test_pixel_calibrated_recorded_in_export(self):
        img = np.zeros((20, 20), dtype=np.uint8)
        img[5:15, 5:15] = 200
        mask = img >= 128
        result = compute_full_measurement(img, mask, ~mask, pixel_size_nm=1.0,
                                          pixel_calibrated=False)
        assert result.to_dict()['像素尺寸标定'] == '未标定(按1.0nm/px假定)'
        result_cal = compute_full_measurement(img, mask, ~mask, pixel_size_nm=0.5,
                                              pixel_calibrated=True)
        assert result_cal.to_dict()['像素尺寸标定'] == '已标定'


class TestNaturalKey:
    def test_file2_before_file10(self):
        files = ['file10.tif', 'file2.tif', 'file1.tif']
        assert sorted(files, key=_natural_key) == ['file1.tif', 'file2.tif', 'file10.tif']

    def test_mixed_digit_prefix_names_are_type_safe(self):
        # re.split(r'(\d+)') 的结构保证：偶数下标恒为 str、奇数下标恒为 int，
        # int/str 永不在同一下标相遇，混排不会触发 TypeError。
        files = ['slice.tif', 'slice1.tif', '1a.tif', 'a1b.tif']
        sorted(files, key=_natural_key)  # 不抛异常即通过


class TestAnnotatedImageBlend:
    """回归 2026-09-06 P1：全零 overlay 混色曾把掩膜外像素整体压暗 35-40%。"""

    def test_blend_preserves_unmasked_brightness(self):
        gray = np.full((10, 10), 200, dtype=np.uint8)
        cryst = np.zeros((10, 10), dtype=bool)
        cryst[:5, :] = True  # 上半晶体，下半非晶
        amorph = ~cryst

        out = create_annotated_image(gray, cryst, amorph)

        # 晶体: 0.4·(0,255,0) + 0.6·200 = (120, 222, 120) BGR
        # 非晶: 0.35·(0,0,255) + 0.65·200 = (130, 130, 219) BGR
        np.testing.assert_allclose(out[2, 2], [120, 222, 120], atol=1)
        np.testing.assert_allclose(out[8, 8], [130, 130, 219], atol=1)

    def test_blend_accepts_uint16_with_dataset_shift(self):
        gray = np.full((8, 8), 2048, dtype=np.uint16)
        cryst = np.zeros((8, 8), dtype=bool)
        cryst[0, 0] = True
        out = create_annotated_image(gray, cryst, ~cryst, uint16_shift=4)
        # 2048 >> 4 = 128 灰度底图，标注只改变色调不改变整体亮度口径
        assert out.shape == (8, 8, 3)
        assert out.dtype == np.uint8


class TestFloatNonFinite:
    """回归 2026-09-06 P1：含 NaN 的 float TIFF 曾静默产出全零分割。"""

    def test_nan_float_image_still_segments(self):
        img = np.full((20, 20), 100.0, dtype=np.float32)
        img[5:15, 5:15] = 250.0
        img[0, 0] = np.nan
        params = SegmentationParams(method='threshold', threshold=128)
        mask, _, info = segment_image(img, params)
        assert int(mask.sum()) == 100
        assert info['invalid_input'] is True

    def test_all_nan_returns_empty_mask_with_flag(self):
        img = np.full((10, 10), np.nan, dtype=np.float32)
        params = SegmentationParams(method='threshold', threshold=128)
        mask, _, info = segment_image(img, params)
        assert int(mask.sum()) == 0
        assert info['invalid_input'] is True

    def test_finite_image_not_flagged(self):
        img = np.full((10, 10), 50.0, dtype=np.float32)
        img[2:8, 2:8] = 200.0
        params = SegmentationParams(method='threshold', threshold=128)
        _, _, info = segment_image(img, params)
        assert info['invalid_input'] is False

    def test_has_non_finite_dtype_kinds(self):
        assert has_non_finite(np.array([1.0, np.nan])) is True
        assert has_non_finite(np.array([1.0, np.inf])) is True
        assert has_non_finite(np.array([1.0, 2.0])) is False
        assert has_non_finite(np.array([1, 2], dtype=np.uint16)) is False


class TestUint16DatasetShift:
    """回归 2026-09-06 P1：位深判定以数据集为单位，明暗切片映射一致。"""

    def test_dataset_shift_applies_uniformly(self):
        dark = np.array([[3000]], dtype=np.uint16)
        bright = np.array([[60000, 3000]], dtype=np.uint16)
        shift = resolve_uint16_shift(60000)  # 数据集 max → 8 位
        assert shift == 8
        assert uint16_to_uint8(dark, shift)[0, 0] == uint16_to_uint8(bright, shift)[0, 1]

    def test_auto_shift_backward_compatible(self):
        # 不传 shift 时按单图 max 自动选择（R2 起：满足"顶端 > 阈值"的最大位移）
        arr = np.array([[3000]], dtype=np.uint16)
        assert resolve_uint16_shift(3000) == 4
        assert uint16_to_uint8(arr)[0, 0] == 3000 >> 4

    def test_resolve_shift_invariant_reaches_threshold(self):
        """回归 2026-10-03 R2：shift 必须让量程顶端映射到固定阈值之上。

        此前本用例断言的是**分档表**（256→4、4096→6、16384→8）。那张表在档位
        **下沿**是错的：256>>4=16、4096>>6=64、16384>>8=64、33023>>8=128，全部
        ≤ 默认阈值 128，于是固定阈值分割静默产出全空掩膜（面积 0、非晶 100%）。
        断言分档数字等于把缺陷固化下来，因此改为断言**不变量本身**：
        对任意量程，映射后的最大值必须严格大于阈值。
        """
        for peak in (255, 256, 257, 258, 1024, 4095, 4096, 8191, 16383, 16384,
                     33023, 65535):
            shift = resolve_uint16_shift(peak)
            assert 0 <= shift <= 8, peak
            mapped = min(peak >> shift, 255)
            if peak <= 255:
                assert shift == 0, peak
                continue
            assert mapped > 128, (
                f"量程 {peak} 经 shift={shift} 映射到 {mapped}，"
                "不超过默认阈值 128：固定阈值会静默全灭"
            )

    def test_every_uint16_range_is_reachable(self):
        """穷举 256..65535：不存在被压到阈值以下的量程（R2 全量反例）。"""
        unreachable = []
        for peak in range(256, 65536):
            shift = resolve_uint16_shift(peak)
            if min(peak >> shift, 255) <= 128:
                unreachable.append((peak, shift))
        assert unreachable == [], f"仍有 {len(unreachable)} 个量程不可达，例如 {unreachable[:5]}"

    def test_shift_never_wraps_bright_to_black(self):
        """共享 shift 时亮像素不得回绕变黑（R2 clip 反例）。

        ``astype(np.uint8)`` 是回绕语义：shift=0 时 256 会变成 0。修复前
        ``[0, 256, 511, 65535]`` 映射为 ``[0, 0, 255, 255]``，亮像素变黑。
        """
        raw = np.array([0, 256, 511, 65535], dtype=np.uint16)
        mapped = uint16_to_uint8(raw, 0)
        assert mapped.tolist() == [0, 255, 255, 255]

    def test_mapping_is_monotonic_within_one_shift(self):
        """同一 shift 下映射必须单调不减，否则同一数据集内比较失真。"""
        raw = np.arange(0, 65536, dtype=np.uint16)
        for shift in (0, 1, 4, 6, 8):
            mapped = uint16_to_uint8(raw, shift).astype(np.int32)
            assert np.all(np.diff(mapped) >= 0), f"shift={shift} 映射非单调"

    def test_shared_shift_keeps_frames_comparable(self):
        """同一数据集共用一个 shift：相同灰度必须映射到同一值（R2 跨帧一致性）。"""
        shift = resolve_uint16_shift(65535)
        dark = np.array([[3000]], dtype=np.uint16)
        bright = np.array([[60000, 3000]], dtype=np.uint16)
        assert uint16_to_uint8(dark, shift)[0, 0] == uint16_to_uint8(bright, shift)[0, 1]

    def test_bright_and_dark_polarity_both_usable(self):
        """亮/暗两种前景极性在同一映射下都必须能分出前景（R2 极性）。"""
        raw = np.array([[0, 65535], [0, 65535]], dtype=np.uint16)
        gray = uint16_to_uint8(raw, resolve_uint16_shift(65535))
        bright = segment_threshold(gray, SegmentationParams(foreground='bright'))
        dark = segment_threshold(gray, SegmentationParams(foreground='dark'))
        assert int(bright.sum()) > 0
        assert int(dark.sum()) > 0

    def test_14bit_data_survives_threshold_128(self):
        # 14-bit 量程顶端映射到 ~255，固定阈值 128 不再全灭
        arr = np.array([1000, 4000, 8000, 16000], dtype=np.uint16)
        mapped = uint16_to_uint8(arr, resolve_uint16_shift(16000))
        assert int(mapped.max()) >= 250
        params = SegmentationParams(method='threshold', threshold=128)
        mask, _, _ = segment_image(arr.reshape(1, 4), params,
                                   uint16_shift=resolve_uint16_shift(16000))
        assert int(mask.sum()) > 0

    def test_threshold_is_strictly_greater(self):
        """固定阈值判据是 `gray > threshold`（不是 >=），文档口径必须一致。"""
        gray = np.array([[128, 129]], dtype=np.uint8)
        mask = segment_threshold(gray, SegmentationParams(threshold=128))
        assert mask.tolist() == [[False, True]]


class TestRegionRecords:
    """回归 2026-09-06 P2：新增逐区域测量，聚合值须与逐区域记录一致。"""

    def _mask(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[2:6, 2:6] = True     # 4×4
        mask[20:24, 20:24] = True  # 4×4
        mask[0, 30] = True        # 1px 噪声，不计
        return mask

    def test_regions_match_aggregates(self):
        mask = self._mask()
        regions = measure_regions(mask, pixel_size_nm=0.5)
        assert len(regions) == 2
        # ndimage.label 按光栅序编号：噪斑在 row 0 时为 label 1，两方块为 2/3
        assert len({r['区域ID'] for r in regions}) == 2

        area, perim, sf, num, factors = measure_crystal_properties(mask, pixel_size_nm=0.5)
        assert num == len(regions)
        assert perim == pytest.approx(sum(r['周长(nm)'] for r in regions))
        assert sorted(factors) == sorted(r['形状因子'] for r in regions)

    def test_region_area_is_pixel_count(self):
        """回归 2026-09-28：区域面积曾用 cv2.contourArea（多边形口径），
        10×10 方块记 81px 而聚合记 100px，逐区域面积系统性偏低。"""
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True   # 10×10 方块
        regions = measure_regions(mask, pixel_size_nm=1.0)
        assert len(regions) == 1
        assert regions[0]['面积(px)'] == 100.0
        assert regions[0]['面积(nm²)'] == 100.0

    def test_three_pixel_line_region_counted(self):
        """回归 2026-09-28：3px 直线区域（contourArea=0）曾被静默丢弃，
        与"≥3px 计入区域数量"的文档口径不符。计入后形状因子为 0（线状）。"""
        mask = np.zeros((20, 20), dtype=bool)
        mask[5, 5:8] = True   # 3px 水平线
        regions = measure_regions(mask, pixel_size_nm=1.0)
        assert len(regions) == 1
        assert regions[0]['面积(px)'] == 3.0
        assert regions[0]['形状因子'] == pytest.approx(0.0)

    def test_region_record_fields(self):
        regions = measure_regions(self._mask(), pixel_size_nm=1.0)
        r = regions[0]
        for key in ('区域ID', '面积(px)', '周长(px)', '面积(nm²)', '周长(nm)',
                    '形状因子', '质心X(px)', '质心Y(px)',
                    '外接框X(px)', '外接框Y(px)', '外接框宽(px)', '外接框高(px)'):
            assert key in r
        assert 0.0 <= r['形状因子'] <= 1.0

    def test_compute_full_measurement_carries_regions(self):
        img = np.zeros((20, 20), dtype=np.uint8)
        img[5:15, 5:15] = 200
        mask = img >= 128
        result = compute_full_measurement(img, mask, ~mask, pixel_size_nm=1.0)
        assert len(result.regions) == result.num_regions == 1


class TestShapeUniformityAndTemporal:
    """回归 2026-09-06 P2：'形状稳定性'正名为区域形状一致性，并新增时序 CV。"""

    def test_uniformity_wording(self):
        df = pd.DataFrame({u'形状因子_标准差': [0.02, 0.03]})
        assert assess_shape_uniformity(df).startswith('区域形状')

    def test_temporal_cv_constant_series_is_zero(self):
        df = pd.DataFrame({
            '文件名': ['a.tif'] * 3,
            '切片': [1, 2, 3],
            '形状因子_平均值': [0.8, 0.8, 0.8],
        })
        stats = calculate_temporal_shape_stability(df)
        assert stats['overall']['CV平均'] == pytest.approx(0.0)

    def test_temporal_cv_detects_drift(self):
        df = pd.DataFrame({
            '文件名': ['a.tif'] * 3,
            '切片': [1, 2, 3],
            '形状因子_平均值': [0.2, 0.5, 0.8],
        })
        stats = calculate_temporal_shape_stability(df)
        assert stats['overall']['CV平均'] > 0.3


class TestMeanEffectiveGrowthRate:
    """回归 2026-09-06 P3：速率均值不再混入每文件首行与单切片文件的 0 值。"""

    def test_excludes_first_row_of_each_file(self):
        rows = []
        for f in ('a.tif', 'b.tif'):
            for sl, (area, perim) in enumerate([(100, 10), (200, 20), (300, 30)], 1):
                rows.append({'文件名': f, '切片': sl,
                             '晶体区域实际面积(nm²)': float(area),
                             '晶体区域周长(nm)': float(perim)})
        df = pd.DataFrame(rows)
        rates = calculate_radial_growth_rate(df, time_interval=1.0)
        # a/b 各文件: 切片2 速率 = 100/10 = 10；切片3 = 100/20 = 5；首行=10(填充)
        assert mean_effective_growth_rate(df, rates) == pytest.approx((10.0 + 5.0) / 2)
        # 旧口径（全行均值）确实更高，证明修正生效
        assert np.mean(rates) == pytest.approx((10 + 10 + 5) / 3)


class TestEffectiveGrowthRowsMask:
    """回归 2026-09-28：有效差分行掩码提取为公共函数（均值/最值共用口径）。"""

    def test_mask_excludes_first_row_per_file_and_single_slice_files(self):
        rows = []
        for f, slices in (('a.tif', 3), ('b.tif', 3), ('c.tif', 1)):
            for sl in range(1, slices + 1):
                rows.append({'文件名': f, '切片': sl})
        df = pd.DataFrame(rows)
        mask = effective_growth_rows_mask(df)
        assert mask.sum() == 4  # a/b 各 2 行有效，c 单切片 0 行
        a1 = df[(df['文件名'] == 'a.tif') & (df['切片'] == 1)].index[0]
        c1 = df[(df['文件名'] == 'c.tif') & (df['切片'] == 1)].index[0]
        assert not bool(mask.loc[a1])   # 每文件首行（回填值）不算有效
        assert not bool(mask.loc[c1])   # 单切片文件无差分行

    def test_empty_df_mask(self):
        df = pd.DataFrame(columns=['文件名', '切片'])
        assert effective_growth_rows_mask(df).sum() == 0


class TestAdaptiveBlocksizeClamp:
    """回归 2026-09-28：块大小钳位到不超过图像短边，避免 cv2 异常。"""

    def test_large_blocksize_on_small_image(self):
        img = np.zeros((8, 8), dtype=np.uint8)
        img[2:6, 2:6] = 200
        params = SegmentationParams(method='adaptive',
                                     adaptive_blocksize=999, adaptive_c=2)
        params.validate()
        mask, _, _ = segment_image(img, params)
        assert mask.shape == img.shape  # 不抛异常即通过
