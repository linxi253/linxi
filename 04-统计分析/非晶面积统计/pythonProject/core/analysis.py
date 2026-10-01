# -*- coding: utf-8 -*-
"""
数据分析模块

对批量测量结果进行高级分析：
- 径向生长速率计算
- 面积-周长相关性分析
- 形状一致性 / 形状时序稳定性
- 生长模式判断
"""

import numpy as np
import pandas as pd
from typing import Tuple

from constants import (
    COL_FILENAME,
    COL_SLICE,
    COL_CRYST_AREA,
    COL_CRYST_PERIM,
    COL_N_REGIONS,
    COL_CRYST_RATIO,
    COL_AMORPH_RATIO,
    COL_SF_MEAN,
    COL_SF_STD,
    SHAPE_UNIFORMITY_TIERS,
    SHAPE_UNIFORMITY_WORST,
)


def calculate_radial_growth_rate(df: pd.DataFrame, time_interval: float) -> pd.Series:
    """
    计算径向生长速率

    物理意义：晶体前沿单位时间内的径向推进距离
    公式：v = (ΔA / Δt) / P
    其中 ΔA 为面积变化量，Δt 为时间间隔，P 为当前周长

    对多文件数据按 ``文件名`` 分组、组内按 ``切片`` 排序后计算；
    返回的 ``pd.Series`` 使用原始 df 索引对齐，导出时不会发生行序错位。

    Args:
        df: 包含 '切片', '晶体区域实际面积(nm²)', '晶体区域周长(nm)' 列的 DataFrame
        time_interval: 相邻切片间的时间间隔

    Returns:
        径向生长速率 Series (nm/单位时间)，索引与 df.index 一致
    """
    if len(df) < 2 or time_interval <= 0:
        return pd.Series(np.zeros(len(df)), index=df.index)

    result = pd.Series(np.zeros(len(df)), index=df.index)

    # 多文件数据按文件名分组；组内按切片排序。
    group_col = COL_FILENAME if COL_FILENAME in df.columns else None
    if group_col is not None:
        groups = df.groupby(group_col, sort=False)
    else:
        groups = [('', df)]

    for fname, group in groups:
        g = group.sort_values(COL_SLICE)
        if len(g) < 2:
            continue

        areas = g[COL_CRYST_AREA].values
        perimeters = g[COL_CRYST_PERIM].values
        growth = np.zeros(len(g))

        for i in range(1, len(g)):
            if perimeters[i - 1] > 0:
                area_change = areas[i] - areas[i - 1]
                growth[i] = (area_change / time_interval) / perimeters[i - 1]

        # 每个文件组内第一个点用第二个点的值填充
        growth[0] = growth[1]
        result.loc[g.index] = growth

    return result


def effective_growth_rows_mask(df: pd.DataFrame) -> pd.Series:
    """
    标记"有效差分行"：每文件按切片排序后第 2 行起。

    每文件首行的速率是回填值（growth[0] = growth[1]），单切片文件无差分，
    二者都不应纳入速率均值/最值统计。2026-09-28 审查修正：此前只有均值
    排除了这些行，Excel"最大/最小径向生长速率"仍混入回填值，同一张表
    两个口径自相矛盾；现提取为公共函数供均值与最值共用。

    Args:
        df: 测量结果 DataFrame

    Returns:
        与 df.index 对齐的布尔 Series
    """
    mask = pd.Series(False, index=df.index)
    if len(df) == 0:
        return mask

    group_col = COL_FILENAME if COL_FILENAME in df.columns else None
    groups = df.groupby(group_col, sort=False) if group_col else [('', df)]
    for _, group in groups:
        g = group.sort_values(COL_SLICE)
        if len(g) >= 2:
            mask.loc[g.index[1:]] = True
    return mask


def mean_effective_growth_rate(df: pd.DataFrame, growth_rates: pd.Series) -> float:
    """
    计算有效差分行（每文件按切片排序后第 2 行起）的径向生长速率均值。

    每文件首行与单切片文件的速率恒为 0，直接对全部行取均值会系统性
    偏低（回归 2026-09-06 P3 口径修正）。

    Args:
        df: 与 growth_rates 索引一致的测量结果 DataFrame
        growth_rates: calculate_radial_growth_rate 的返回值

    Returns:
        有效速率均值；无有效行时返回 0.0
    """
    if len(df) == 0 or len(growth_rates) != len(df):
        return 0.0

    valid = growth_rates[effective_growth_rows_mask(df)]
    return float(np.mean(valid)) if len(valid) else 0.0


def calculate_area_perimeter_correlation(df: pd.DataFrame) -> Tuple[float, float]:
    """
    计算面积与周长的相关系数及面积/周长比值

    相关系数接近 1 表示面积和周长同步增长（形状稳定）；
    偏离 1 表示形状在演化过程中发生变化。

    注意：多文件时所有行混算（池化相关），文件间尺寸差异会主导结果，
    解读时应结合"按文件分组"的生长分析表。

    Args:
        df: 测量结果 DataFrame

    Returns:
        (correlation, area_perimeter_ratio)
    """
    if len(df) < 2:
        return 0.0, 0.0

    # 常数序列时 corr 内部除零产生 RuntimeWarning，结果 NaN 由下文处理
    with np.errstate(invalid='ignore', divide='ignore'):
        correlation = df[COL_CRYST_AREA].corr(df[COL_CRYST_PERIM])

    # 处理 NaN（当数据为常数时 corr 返回 NaN）
    if np.isnan(correlation):
        correlation = 0.0

    mean_perim = df[COL_CRYST_PERIM].mean()
    area_perimeter_ratio = df[COL_CRYST_AREA].mean() / mean_perim if mean_perim > 0 else 0.0

    return float(correlation), float(area_perimeter_ratio)


def calculate_growth_statistics(df: pd.DataFrame) -> dict:
    """
    计算生长统计摘要（按文件分组逐文件计算）。

    - 单文件：返回该文件的首尾面积/周长与增长率。
    - 多文件：返回 ``per_file``（每个文件一行）和 ``overall``
      （各文件增长率的描述性统计：平均/标准差/最小/最大/文件数）。
    - 不会把不同文件的首尾切片拼在一起做伪增长。

    Args:
        df: 测量结果 DataFrame（包含 '文件名' 列）

    Returns:
        统计量字典；数据不足时返回空 dict。
    """
    required = {COL_FILENAME, COL_SLICE, COL_CRYST_AREA, COL_CRYST_PERIM}

    if len(df) < 2 or not required.issubset(df.columns):
        return {}

    per_file = []
    for fname, group in df.groupby(COL_FILENAME, sort=False):
        g = group.sort_values(COL_SLICE)
        if len(g) < 2:
            continue

        initial_area = float(g[COL_CRYST_AREA].iloc[0])
        final_area = float(g[COL_CRYST_AREA].iloc[-1])
        initial_perim = float(g[COL_CRYST_PERIM].iloc[0])
        final_perim = float(g[COL_CRYST_PERIM].iloc[-1])

        area_growth_pct = (
            ((final_area - initial_area) / initial_area * 100)
            if initial_area > 0 else 0.0
        )
        perim_growth_pct = (
            ((final_perim - initial_perim) / initial_perim * 100)
            if initial_perim > 0 else 0.0
        )

        per_file.append({
            '文件名': fname,
            '切片数': int(len(g)),
            '初始面积(nm²)': initial_area,
            '最终面积(nm²)': final_area,
            '面积增长率(%)': area_growth_pct,
            '初始周长(nm)': initial_perim,
            '最终周长(nm)': final_perim,
            '周长增长率(%)': perim_growth_pct,
        })

    if not per_file:
        return {}

    area_rates = np.array([p['面积增长率(%)'] for p in per_file], dtype=np.float64)
    perim_rates = np.array([p['周长增长率(%)'] for p in per_file], dtype=np.float64)

    overall = {
        '文件数': len(per_file),
        '面积增长率(%)_平均': float(np.mean(area_rates)),
        '面积增长率(%)_标准差': float(np.std(area_rates)),
        '面积增长率(%)_最小': float(np.min(area_rates)),
        '面积增长率(%)_最大': float(np.max(area_rates)),
        '周长增长率(%)_平均': float(np.mean(perim_rates)),
        '周长增长率(%)_标准差': float(np.std(perim_rates)),
        '周长增长率(%)_最小': float(np.min(perim_rates)),
        '周长增长率(%)_最大': float(np.max(perim_rates)),
    }

    return {'per_file': per_file, 'overall': overall}


def assess_shape_uniformity(df: pd.DataFrame) -> str:
    """
    评估区域形状一致性（切片内区域间形状离散度）

    根据"形状因子_标准差"（切片内各连通区域形状因子的离散程度）的均值判断：
    - < 0.05: 区域形状高度一致
    - < 0.10: 区域形状较一致
    - < 0.20: 区域形状较分散
    - >= 0.20: 区域形状显著分散

    注意：这是**切片内**区域间离散度指标，不代表形状随时间（切片序列）
    的演化；后者见 calculate_temporal_shape_stability。
    （回归 2026-09-06 P2：原"形状稳定性"命名名不副实，已正名。）

    Args:
        df: 测量结果 DataFrame

    Returns:
        区域形状一致性评估文本
    """
    if COL_SF_STD not in df.columns or len(df) == 0:
        return "数据不足"

    mean_std = df[COL_SF_STD].mean()

    for threshold, text in SHAPE_UNIFORMITY_TIERS:
        if mean_std < threshold:
            return text
    return SHAPE_UNIFORMITY_WORST


def calculate_temporal_shape_stability(df: pd.DataFrame) -> dict:
    """
    计算形状因子均值的跨切片变异系数（时序形状稳定性）

    assess_shape_uniformity 衡量切片**内部**区域间的形状离散度；
    本函数衡量各文件"形状因子_平均值"随切片（时间序列）的漂移：
    CV = 组内标准差 / 组内均值，越小表示形状随时间越稳定。

    Args:
        df: 测量结果 DataFrame

    Returns:
        {'per_file': [{'文件名', '形状因子均值', '标准差', '变异系数CV'}, ...],
         'overall': {'文件数', 'CV平均', 'CV标准差'}}
        数据不足时返回空 dict。
    """
    if COL_SF_MEAN not in df.columns or len(df) == 0:
        return {}

    per_file = []
    for fname, group in df.groupby(COL_FILENAME, sort=False):
        g = group.sort_values(COL_SLICE)
        vals = g[COL_SF_MEAN].astype(float).values
        if len(vals) < 2:
            continue
        m = float(np.mean(vals))
        s = float(np.std(vals))
        cv = (s / m) if m > 0 else 0.0
        per_file.append({'文件名': fname, '形状因子均值': m, '标准差': s, '变异系数CV': cv})

    if not per_file:
        return {}

    cvs = np.array([p['变异系数CV'] for p in per_file], dtype=np.float64)
    return {
        'per_file': per_file,
        'overall': {
            '文件数': len(per_file),
            'CV平均': float(np.mean(cvs)),
            'CV标准差': float(np.std(cvs)),
        },
    }


def generate_summary_text(df: pd.DataFrame, time_interval: float = 1.0) -> str:
    """
    生成批量处理结果的文本摘要

    Args:
        df: 完整测量结果 DataFrame
        time_interval: 时间间隔

    Returns:
        格式化的摘要文本
    """
    if len(df) == 0:
        return "无数据"

    lines = [
        "═══ 批量处理结果摘要 ═══",
        "",
        f"总切片数: {len(df)}",
        f"晶体区域总数: {df[COL_N_REGIONS].sum():,}",
        f"平均晶体比例: {df[COL_CRYST_RATIO].mean():.2%}",
        f"平均非晶比例: {df[COL_AMORPH_RATIO].mean():.2%}",
        f"平均晶体面积: {df[COL_CRYST_AREA].mean():,.2f} nm²",
        f"平均晶体周长: {df[COL_CRYST_PERIM].mean():,.2f} nm",
        f"平均形状因子: {df[COL_SF_MEAN].mean():.4f}",
        f"区域形状一致性: {assess_shape_uniformity(df)}",
    ]

    # 径向生长速率
    if len(df) > 1:
        growth_rates = calculate_radial_growth_rate(df, time_interval)
        effective_mean = mean_effective_growth_rate(df, growth_rates)
        lines.append(f"平均径向生长速率: {effective_mean:.4f} nm/单位时间（仅有效差分行）")

        # 形状时序稳定性
        temporal = calculate_temporal_shape_stability(df)
        if temporal:
            ov = temporal['overall']
            lines.append(f"形状时序稳定性: 平均CV {ov['CV平均']:.4f}（越小越稳定）")

        # 生长统计：多文件按文件名分组逐文件计算，整体只给描述性统计。
        growth_stats = calculate_growth_statistics(df)
        if growth_stats:
            lines.append("")
            lines.append("─── 生长分析（按文件分组） ───")
            per_file = growth_stats.get('per_file', [])
            for row in per_file:
                lines.append(
                    f"  {row['文件名']}: 面积 {row['初始面积(nm²)']:,.2f} → "
                    f"{row['最终面积(nm²)']:,.2f} nm² "
                    f"({row['面积增长率(%)']:+.2f}%)，"
                    f"周长 {row['初始周长(nm)']:,.2f} → "
                    f"{row['最终周长(nm)']:,.2f} nm "
                    f"({row['周长增长率(%)']:+.2f}%)"
                )
            overall = growth_stats.get('overall')
            if overall:
                lines.append(
                    f"  整体汇总: {overall['文件数']} 个文件，"
                    f"面积增长率均值 {overall['面积增长率(%)_平均']:+.2f}% "
                    f"(SD={overall['面积增长率(%)_标准差']:.2f})，"
                    f"周长增长率均值 {overall['周长增长率(%)_平均']:+.2f}% "
                    f"(SD={overall['周长增长率(%)_标准差']:.2f})"
                )

    return "\n".join(lines)
