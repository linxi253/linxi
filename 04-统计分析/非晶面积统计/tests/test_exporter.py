# -*- coding: utf-8 -*-
"""导出模块测试（回归 2026-09-06：参数表补全 + 逐区域导出 + 形状一致性正名）。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from constants import (
    COL_FILENAME,
    COL_SLICE,
    COL_CRYST_AREA,
    COL_GROWTH_RATE,
)
from core.segmentation import SegmentationParams
from core.measurement import compute_full_measurement
from io_utils.exporter import export_results


def _build_df_and_regions(n_slices=2):
    """合成一个文件 n 个切片的测量结果，返回 (df, region_rows)。"""
    rows = []
    region_rows = []
    for si in range(1, n_slices + 1):
        img = np.zeros((30, 30), dtype=np.uint8)
        img[4:14, 4:14] = 200          # 10×10 晶体块
        img[16 + si:20 + si, 16:20] = 200  # 随切片变化的第二块
        mask = img >= 128
        result = compute_full_measurement(
            img, mask, ~mask, pixel_size_nm=1.0,
            filename='a.tif', slice_index=si)
        rows.append(result.to_dict())
        for r in result.regions:
            row = {COL_FILENAME: 'a.tif', COL_SLICE: si}
            row.update(r)
            region_rows.append(row)
    return pd.DataFrame(rows), region_rows


@pytest.fixture()
def canny_params():
    return SegmentationParams(
        method='canny', canny_low=40, canny_high=120, canny_min_area=77)


class TestExportResults:
    def test_exports_csv_region_csv_and_excel(self, tmp_path, canny_params):
        df, region_rows = _build_df_and_regions()
        csv_path, excel_path, region_csv_path = export_results(
            df, str(tmp_path), time_interval=2.0, pixel_size_nm=1.0,
            seg_params=canny_params, pixel_calibrated=False,
            region_rows=region_rows)

        assert csv_path and excel_path and region_csv_path
        assert Path(csv_path).exists() and Path(excel_path).exists()
        assert Path(region_csv_path).name.startswith('region_results_')

    def test_params_sheet_contains_all_method_params(self, tmp_path, canny_params):
        df, region_rows = _build_df_and_regions()
        _, excel_path, _ = export_results(
            df, str(tmp_path), seg_params=canny_params, region_rows=region_rows)

        params_df = pd.read_excel(excel_path, sheet_name='实验参数')
        kv = dict(zip(params_df['参数名'], params_df['参数值'].astype(str)))

        assert kv['分割方法'].startswith('Canny边缘')
        assert kv['Canny低阈值 (仅canny)'] == '40'
        assert kv['Canny高阈值 (仅canny)'] == '120'
        assert kv['Canny最小轮廓面积 (仅canny)'] == '77'
        assert kv['最小区域像素数 (所有方法)'] == '3'
        assert kv['自适应块大小 (仅adaptive)'] == '—'
        assert kv['分水岭形态学核 (仅watershed)'] == '—'
        assert '真空' in kv['统计口径警告'] and '非晶区域' in kv['统计口径警告']

    def test_shape_sheet_has_uniformity_and_temporal_cv(self, tmp_path, canny_params):
        df, region_rows = _build_df_and_regions()
        _, excel_path, _ = export_results(
            df, str(tmp_path), seg_params=canny_params, region_rows=region_rows)

        shape_df = pd.read_excel(excel_path, sheet_name='形状因子分析')
        items = set(shape_df['统计项'])
        assert '区域形状一致性' in items
        assert '形状因子时序稳定性(平均CV)' in items
        assert '形状稳定性评估' not in items  # 旧措辞不再出现

    def test_region_sheet_and_csv_rows_match(self, tmp_path, canny_params):
        df, region_rows = _build_df_and_regions()
        csv_path, excel_path, region_csv_path = export_results(
            df, str(tmp_path), seg_params=canny_params, region_rows=region_rows)

        region_df = pd.read_csv(region_csv_path, encoding='utf-8-sig')
        assert len(region_df) == len(region_rows) > 0
        assert COL_FILENAME in region_df.columns and COL_SLICE in region_df.columns
        assert '形状因子' in region_df.columns and '面积(nm²)' in region_df.columns

        # 行数受控时 Excel 附逐区域 Sheet
        excel_region = pd.read_excel(excel_path, sheet_name='逐区域数据')
        assert len(excel_region) == len(region_rows)

    def test_main_csv_keeps_schema_columns(self, tmp_path, canny_params):
        df, region_rows = _build_df_and_regions()
        csv_path, _, _ = export_results(
            df, str(tmp_path), seg_params=canny_params, region_rows=region_rows)

        out = pd.read_csv(csv_path, encoding='utf-8-sig')
        for col in (COL_FILENAME, COL_SLICE, COL_CRYST_AREA, COL_GROWTH_RATE,
                    '像素尺寸标定'):
            assert col in out.columns

    def test_no_region_rows_skips_region_outputs(self, tmp_path, canny_params):
        df, _ = _build_df_and_regions()
        csv_path, excel_path, region_csv_path = export_results(
            df, str(tmp_path), seg_params=canny_params, region_rows=None)
        assert region_csv_path is None
        assert Path(csv_path).exists()

        with pd.ExcelFile(excel_path) as xls:
            assert '逐区域数据' not in xls.sheet_names


class TestGrowthSheetRates:
    """回归 2026-09-28：生长分析表的速率最值与均值统一"仅有效差分行"口径，
    且逐文件行的速率列不再恒空。"""

    @staticmethod
    def _make_two_file_df():
        rows = []
        # a.tif: 3 切片，有效速率 = [10, 5]（首行为回填值 10）
        for sl, (area, perim) in enumerate([(100, 10), (200, 20), (300, 30)], 1):
            rows.append({'文件名': 'a.tif', '切片': sl,
                         '晶体区域实际面积(nm²)': float(area),
                         '晶体区域周长(nm)': float(perim),
                         '形状因子_平均值': 0.8, '形状因子_标准差': 0.01,
                         '形状因子_最小值': 0.7, '形状因子_最大值': 0.9})
        # b.tif: 单切片，无差分行（旧口径的最小值被其 0 值拉到 0）
        rows.append({'文件名': 'b.tif', '切片': 1,
                     '晶体区域实际面积(nm²)': 50.0,
                     '晶体区域周长(nm)': 5.0,
                     '形状因子_平均值': 0.8, '形状因子_标准差': 0.01,
                     '形状因子_最小值': 0.7, '形状因子_最大值': 0.9})
        return pd.DataFrame(rows)

    def test_per_file_rates_populated(self, tmp_path):
        df = self._make_two_file_df()
        _, excel_path, _ = export_results(df, str(tmp_path), time_interval=1.0)
        growth = pd.read_excel(excel_path, sheet_name='生长分析')

        # 单切片文件（b.tif）本就不进生长分析表（无首尾可比）
        assert 'b.tif' not in set(growth['文件名'])

        a_row = growth[growth['文件名'] == 'a.tif'].iloc[0]
        assert a_row['平均径向生长速率(nm/单位时间)'] == '7.5000'
        assert float(a_row['最大径向生长速率(nm/单位时间)']) == pytest.approx(10.0)
        assert float(a_row['最小径向生长速率(nm/单位时间)']) == pytest.approx(5.0)

    def test_overall_min_max_exclude_single_slice_and_backfill(self, tmp_path):
        df = self._make_two_file_df()
        _, excel_path, _ = export_results(df, str(tmp_path), time_interval=1.0)
        growth = pd.read_excel(excel_path, sheet_name='生长分析')

        overall = growth[growth['文件名'].str.startswith('整体汇总')].iloc[0]
        # 有效速率 {10, 5}：min=5、max=10；旧口径会把 b.tif 的 0 值行算进 min
        assert float(overall['最小径向生长速率(nm/单位时间)']) == pytest.approx(5.0)
        assert float(overall['最大径向生长速率(nm/单位时间)']) == pytest.approx(10.0)
        assert '7.5000' in str(overall['平均径向生长速率(nm/单位时间)'])

    def test_params_sheet_documents_growth_backfill(self, tmp_path):
        df = self._make_two_file_df()
        _, excel_path, _ = export_results(df, str(tmp_path))
        params_df = pd.read_excel(excel_path, sheet_name='实验参数')
        kv = dict(zip(params_df['参数名'], params_df['参数值'].astype(str)))
        assert '回填' in kv['生长速率口径说明']
