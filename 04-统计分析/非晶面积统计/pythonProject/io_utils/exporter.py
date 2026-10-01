# -*- coding: utf-8 -*-
"""
数据导出模块

负责将分析结果导出为 CSV 和 Excel 文件：
- CSV：UTF-8-SIG 编码，兼容 Excel 中文显示
- Excel：多 Sheet 结构（原始数据 + 形状因子分析 + 生长分析 + 实验参数
  + 逐区域数据）
- 逐区域数据另存独立 CSV（行数可能很大，Excel 仅在行数受控时附 Sheet）
- 统一时间戳确保同批输出文件名一致
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple

from constants import (
    APP_VERSION,
    METHOD_NAMES_CN,
    FOREGROUND_BRIGHT,
    FOREGROUND_DARK,
    MIN_REGION_PIXELS,
    REGION_EXCEL_MAX_ROWS,
    COL_FILENAME,
    COL_GROWTH_RATE,
    COL_SF_MEAN,
    COL_SF_STD,
    COL_SF_MIN,
    COL_SF_MAX,
)
from core.segmentation import SegmentationParams
from core.analysis import (
    calculate_radial_growth_rate,
    calculate_area_perimeter_correlation,
    calculate_growth_statistics,
    calculate_temporal_shape_stability,
    assess_shape_uniformity,
    mean_effective_growth_rate,
    effective_growth_rows_mask,
)


def export_results(
    df: pd.DataFrame,
    output_folder: str,
    time_interval: float = 1.0,
    pixel_size_nm: float = 1.0,
    seg_params: Optional[SegmentationParams] = None,
    pixel_calibrated: bool = False,
    region_rows: Optional[List[dict]] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    导出分析结果为 CSV、逐区域 CSV 和 Excel 文件

    Args:
        df: 测量结果 DataFrame
        output_folder: 输出目录
        time_interval: 时间间隔
        pixel_size_nm: 像素尺寸（未标定时为假定值 1.0）
        seg_params: 分割参数对象（全部方法参数写入"实验参数"表，保证可复现）
        pixel_calibrated: 像素尺寸是否经过显微镜标定
        region_rows: 逐区域记录（含 文件名/切片 前缀），None/空 则不导出

    Returns:
        (csv_path, excel_path, region_csv_path)，region_csv_path 无逐区域
        数据时为 None；失败时抛出 IOError
    """
    if seg_params is None:
        seg_params = SegmentationParams()

    os.makedirs(output_folder, exist_ok=True)

    # 统一时间戳，确保同批输出文件名一致
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 计算径向生长速率
    df_export = df.copy()
    growth_rates = calculate_radial_growth_rate(df, time_interval)
    df_export[COL_GROWTH_RATE] = growth_rates.values

    # === 主结果 CSV ===
    csv_filename = f"analysis_results_{timestamp}.csv"
    csv_path = os.path.join(output_folder, csv_filename)
    df_export.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # === 逐区域 CSV ===
    region_csv_path = None
    if region_rows:
        region_csv_path = os.path.join(output_folder, f"region_results_{timestamp}.csv")
        pd.DataFrame(region_rows).to_csv(region_csv_path, index=False, encoding='utf-8-sig')

    # === Excel（多 Sheet） ===
    excel_filename = f"analysis_results_{timestamp}.xlsx"
    excel_path = os.path.join(output_folder, excel_filename)

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: 原始数据
        df_export.to_excel(writer, sheet_name='原始数据', index=False)

        # Sheet 2: 实验参数
        params_data = _build_params_sheet(
            seg_params, pixel_size_nm, time_interval, len(df), pixel_calibrated
        )
        pd.DataFrame(params_data).to_excel(writer, sheet_name='实验参数', index=False)

        # Sheet 3: 形状因子分析
        shape_data = _build_shape_analysis_sheet(df)
        pd.DataFrame(shape_data).to_excel(writer, sheet_name='形状因子分析', index=False)

        # Sheet 4: 生长分析（仅多切片时）
        if len(df) > 1:
            growth_rows = _build_growth_analysis_sheet(df, growth_rates)
            if growth_rows:
                pd.DataFrame(growth_rows).to_excel(writer, sheet_name='生长分析', index=False)

        # Sheet 5: 逐区域数据（行数受控时才写入 Excel，超出见 CSV）
        if region_rows:
            if len(region_rows) <= REGION_EXCEL_MAX_ROWS:
                pd.DataFrame(region_rows).to_excel(
                    writer, sheet_name='逐区域数据', index=False)
            else:
                note = pd.DataFrame({
                    '说明': [f"逐区域数据共 {len(region_rows)} 行，超出 Excel 写入上限"
                             f"（{REGION_EXCEL_MAX_ROWS}），请查看 region_results_*.csv"]
                })
                note.to_excel(writer, sheet_name='逐区域数据', index=False)

    return csv_path, excel_path, region_csv_path


def _build_params_sheet(
    seg_params: SegmentationParams,
    pixel_size_nm: float,
    time_interval: float,
    total_slices: int,
    pixel_calibrated: bool,
) -> dict:
    """构建实验参数 Sheet 数据。

    记录全部分割方法参数（当前未选中的方法以 '—' 占位），保证批量结果
    可完整复现（回归 2026-09-06 P2：此前 adaptive/canny/watershed 参数与
    最小区域像素数不落盘）。
    """
    method = seg_params.method

    if pixel_calibrated:
        calibration_note = "已标定：面积/周长单位为 nm/nm²，可作绝对物理量引用"
    else:
        calibration_note = ("未标定：按假定值 1.0 nm/px 计算，面积/周长的实际单位"
                            "应读作 px²/px，请勿作为绝对物理量引用；"
                            "请在显微镜标定后重新分析")

    param_names = [
        '像素尺寸 (nm/像素)',
        '像素尺寸标定',
        '分割方法',
        '前景极性',
        '固定阈值 (仅threshold)',
        'Otsu阈值 (仅otsu)',
        '自适应块大小 (仅adaptive)',
        '自适应C值 (仅adaptive)',
        'Canny低阈值 (仅canny)',
        'Canny高阈值 (仅canny)',
        'Canny最小轮廓面积 (仅canny)',
        '分水岭形态学核 (仅watershed)',
        '最小区域像素数 (所有方法)',
        '时间间隔',
        '总切片数',
        '统计口径警告',
        '生长速率口径说明',
        '分析时间',
        '工具版本',
    ]
    param_values = [
        f"{pixel_size_nm}",
        calibration_note,
        f"{METHOD_NAMES_CN.get(method, method)} ({method})",
        "亮区为晶体（HAADF/暗场默认）" if seg_params.foreground == FOREGROUND_BRIGHT else "暗区为晶体（TEM 明场衍射衬度）",
        f"{seg_params.threshold}" if method == "threshold" else "—",
        "自动(逐切片计算)" if method == "otsu" else "—",
        f"{seg_params.adaptive_blocksize}" if method == "adaptive" else "—",
        f"{seg_params.adaptive_c}" if method == "adaptive" else "—",
        f"{seg_params.canny_low}" if method == "canny" else "—",
        f"{seg_params.canny_high}" if method == "canny" else "—",
        f"{seg_params.canny_min_area}" if method == "canny" else "—",
        f"{seg_params.watershed_kernel}" if method == "watershed" else "—",
        f"{MIN_REGION_PIXELS}",
        f"{time_interval}",
        f"{total_slices}",
        ("\"非晶区域\"为\"晶体掩膜\"的灰度补集，包含真空、支持膜和孔洞，"
         "未做背景剔除；请勿将\"非晶面积\"直接作为非晶相的物理面积引用。"
         "所有分割均为灰度阈值判据，不代表晶体学/非晶结构的直接鉴定。"),
        ("\"径向生长速率\"列中每个文件的首行为下一行速率的回填值（非实测差分），"
         "生长分析表中的均值/最大/最小只统计每文件第 2 行起的有效差分行。"),
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        f"v{APP_VERSION}",
    ]
    return {'参数名': param_names, '参数值': param_values}


def _build_shape_analysis_sheet(df: pd.DataFrame) -> dict:
    """构建形状因子分析 Sheet 数据"""
    correlation, area_perim_ratio = calculate_area_perimeter_correlation(df)
    uniformity = assess_shape_uniformity(df)
    temporal = calculate_temporal_shape_stability(df)
    cv_mean = temporal['overall']['CV平均'] if temporal else None

    return {
        '统计项': [
            '平均形状因子',
            '形状因子标准差(均值)',
            '最小形状因子',
            '最大形状因子',
            '面积-周长相关系数',
            '面积/周长平均比值',
            '区域形状一致性',
            '形状因子时序稳定性(平均CV)',
        ],
        '数值': [
            f"{df[COL_SF_MEAN].mean():.4f}",
            f"{df[COL_SF_STD].mean():.4f}",
            f"{df[COL_SF_MIN].min():.4f}",
            f"{df[COL_SF_MAX].max():.4f}",
            f"{correlation:.4f}",
            f"{area_perim_ratio:.4f}",
            uniformity,
            f"{cv_mean:.4f}" if cv_mean is not None else "—",
        ],
        '说明': [
            "越接近1形状越圆（完美圆=1.0）",
            "值越小形状越一致",
            "最小圆度值",
            "最大圆度值",
            "1=完全正相关，-1=完全负相关，0=无线性相关；多文件时为池化相关",
            "圆形≈r/2，反映等效半径",
            "切片内区域间形状离散度（非时序指标）",
            "各文件形状因子均值跨切片的变异系数，越小形状随时间越稳定",
        ],
    }


def _build_growth_analysis_sheet(df: pd.DataFrame, growth_rates) -> list:
    """构建生长分析 Sheet 数据。

    多文件时按文件名分组逐文件输出；最后一行给出整体描述性统计。
    径向生长速率的均值/最大/最小**统一只统计有效差分行**（每文件首行
    为回填值、单切片文件无差分，均不纳入；2026-09-28 审查修正：此前
    均值排除了这些行而最值没有，同一张表两个口径自相矛盾，且逐文件
    行的速率列恒为空）。
    """

    growth_stats = calculate_growth_statistics(df)

    if not growth_stats:
        return []

    # 有效差分行的速率（有限值），按文件分组供逐文件行与整体行共用
    effective = effective_growth_rows_mask(df)
    finite_mask = effective & np.isfinite(growth_rates.values)
    finite_rates = growth_rates[finite_mask]

    per_file_rates = {}
    for fname, group in df.groupby(COL_FILENAME, sort=False):
        sel = finite_mask.loc[group.index]
        # 先限定到组内行再做布尔选择（布尔 Series 索引须与被选对象一致）
        per_file_rates[fname] = growth_rates.loc[group.index][sel]

    def _rate_summary(series) -> Tuple[str, str, str]:
        if len(series) == 0:
            return "—", "—", "—"
        return (
            f"{float(np.mean(series)):.4f}",
            f"{float(np.max(series)):.4f}",
            f"{float(np.min(series)):.4f}",
        )

    rows = []
    empty_rates = growth_rates.iloc[:0]
    for row in growth_stats.get('per_file', []):
        file_rates = per_file_rates.get(row['文件名'])
        if file_rates is None:
            file_rates = empty_rates
        mean_s, max_s, min_s = _rate_summary(file_rates)
        rows.append({
            '文件名': row['文件名'],
            '切片数': row['切片数'],
            '初始面积(nm²)': f"{row['初始面积(nm²)']:,.2f}",
            '最终面积(nm²)': f"{row['最终面积(nm²)']:,.2f}",
            '面积增长率(%)': f"{row['面积增长率(%)']:.2f}",
            '初始周长(nm)': f"{row['初始周长(nm)']:,.2f}",
            '最终周长(nm)': f"{row['最终周长(nm)']:,.2f}",
            '周长增长率(%)': f"{row['周长增长率(%)']:.2f}",
            '平均径向生长速率(nm/单位时间)': mean_s,
            '最大径向生长速率(nm/单位时间)': max_s,
            '最小径向生长速率(nm/单位时间)': min_s,
        })

    overall = growth_stats.get('overall', {})
    if overall:
        mean_s, max_s, min_s = _rate_summary(finite_rates)
        rows.append({
            '文件名': f"整体汇总 (n={overall.get('文件数', 0)})",
            '切片数': "",
            '初始面积(nm²)': "",
            '最终面积(nm²)': "",
            '面积增长率(%)': (
                f"均值 {overall['面积增长率(%)_平均']:.2f} "
                f"± {overall['面积增长率(%)_标准差']:.2f} "
                f"[{overall['面积增长率(%)_最小']:.2f}, {overall['面积增长率(%)_最大']:.2f}]"
            ),
            '初始周长(nm)': "",
            '最终周长(nm)': "",
            '周长增长率(%)': (
                f"均值 {overall['周长增长率(%)_平均']:.2f} "
                f"± {overall['周长增长率(%)_标准差']:.2f} "
                f"[{overall['周长增长率(%)_最小']:.2f}, {overall['周长增长率(%)_最大']:.2f}]"
            ),
            '平均径向生长速率(nm/单位时间)': (
                f"{mean_s} (仅有效差分行)" if mean_s != "—" else "—"
            ),
            '最大径向生长速率(nm/单位时间)': max_s,
            '最小径向生长速率(nm/单位时间)': min_s,
        })

    return rows
