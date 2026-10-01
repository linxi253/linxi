# -*- coding: utf-8 -*-
"""
测量计算模块

负责对分割后的掩膜进行定量测量：
- 面积统计（像素数 → 物理面积 nm²）
- 周长计算（OpenCV 轮廓 + scikit-image 交叉验证）
- 形状因子 (Circularity = 4πA/P²)
- 连通区域计数
- 生成彩色标注图像
"""

import numpy as np
import cv2
from scipy import ndimage
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from constants import (
    COLOR_CRYSTALLINE,
    COLOR_AMORPHOUS,
    AMORPHOUS_OVERLAY_ALPHA,
    CRYSTALLINE_OVERLAY_ALPHA,
    MIN_REGION_PIXELS,
    COL_FILENAME,
    COL_SLICE,
    COL_TOTAL_PIXELS,
    COL_CRYST_PIXELS,
    COL_AMORPH_PIXELS,
    COL_CRYST_AREA,
    COL_AMORPH_AREA,
    COL_CRYST_PERIM,
    COL_N_REGIONS,
    COL_SF_MEAN,
    COL_SF_STD,
    COL_SF_MIN,
    COL_SF_MAX,
    COL_CRYST_RATIO,
    COL_AMORPH_RATIO,
    COL_PIXEL_CALIB,
    COL_OTSU,
)
from core.segmentation import ensure_uint8_gray


@dataclass
class MeasurementResult:
    """单张切片的完整测量结果"""
    filename: str = ""
    slice_index: int = 0

    # 像素级统计
    total_pixels: int = 0
    crystalline_pixels: int = 0
    amorphous_pixels: int = 0

    # 物理尺寸统计
    crystalline_area_nm2: float = 0.0
    amorphous_area_nm2: float = 0.0
    crystalline_perimeter_nm: float = 0.0

    # 区域统计
    num_regions: int = 0

    # 形状因子统计
    shape_factor_mean: float = 0.0
    shape_factor_std: float = 0.0
    shape_factor_min: float = 0.0
    shape_factor_max: float = 0.0

    # 面积比例
    crystalline_ratio: float = 0.0
    amorphous_ratio: float = 0.0

    # 附加信息
    otsu_threshold: Optional[int] = None
    pixel_calibrated: bool = False   # 像素尺寸是否经过显微镜标定

    # 逐区域记录（measure_regions 输出，供逐区域导出；聚合统计不使用）
    regions: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转换为字典格式（用于 DataFrame）"""
        result = {
            COL_FILENAME: self.filename,
            COL_SLICE: self.slice_index,
            COL_TOTAL_PIXELS: self.total_pixels,
            COL_CRYST_PIXELS: self.crystalline_pixels,
            COL_AMORPH_PIXELS: self.amorphous_pixels,
            COL_CRYST_AREA: round(self.crystalline_area_nm2, 4),
            COL_AMORPH_AREA: round(self.amorphous_area_nm2, 4),
            COL_CRYST_PERIM: round(self.crystalline_perimeter_nm, 4),
            COL_N_REGIONS: self.num_regions,
            COL_SF_MEAN: round(self.shape_factor_mean, 6),
            COL_SF_STD: round(self.shape_factor_std, 6),
            COL_SF_MIN: round(self.shape_factor_min, 6),
            COL_SF_MAX: round(self.shape_factor_max, 6),
            COL_CRYST_RATIO: round(self.crystalline_ratio, 6),
            COL_AMORPH_RATIO: round(self.amorphous_ratio, 6),
            # 面积/周长数值按 pixel_size 换算；未标定时 pixel_size 恒为假定值
            # 1.0，实际单位应读作 px/px²（见导出参数 Sheet 的口径警告）。
            COL_PIXEL_CALIB: '已标定' if self.pixel_calibrated else '未标定(按1.0nm/px假定)',
        }
        if self.otsu_threshold is not None:
            result[COL_OTSU] = self.otsu_threshold
        return result


def measure_regions(
    crystalline_mask: np.ndarray,
    pixel_size_nm: float = 1.0,
) -> List[dict]:
    """
    逐连通区域测量（单次遍历，供聚合统计与逐区域导出共用）

    与聚合统计同一口径：只统计像素数 ≥ MIN_REGION_PIXELS 的连通区域。
    用 ndimage.find_objects 一次性取各区域外接框，逐区域只在 bbox 内
    提取轮廓——避免 (labeled == region_id) 的整图扫描（回归 2026-09-06
    性能项：4K 图像数千区域时原写法为 O(N_区域 × H×W)）。

    口径说明（2026-09-28 审查修正）：
    - 面积用**像素计数**（与聚合面积同口径；此前用 cv2.contourArea 的
      多边形面积，10×10 方块会记成 81px 而聚合记 100px，逐区域面积
      系统性偏低约半个周长量级）。
    - 周长/形状因子用**轮廓多边形**（arcLength 与 contourArea 互相自洽；
      若面积改像素计数而周长仍用多边形，形状因子会系统性偏高）。因此
      形状因子 = 4π·contourArea / arcLength²，与面积列口径不同，属预期。
    - 3 像素的线状区域（contourArea=0）按 ≥3px 口径计入，形状因子为 0
      （此前被 contourArea<=0 的守卫静默丢弃，与文档口径不符）。

    Returns:
        区域记录列表，字段：区域ID、面积(px/nm²)、周长(px/nm)、形状因子、
        质心(px)、外接框(px)。
    """
    labeled_mask, num_labels = ndimage.label(crystalline_mask)
    if num_labels == 0:
        return []

    sizes = np.bincount(labeled_mask.ravel())
    bbox_slices = ndimage.find_objects(labeled_mask)
    area_per_pixel_nm2 = pixel_size_nm ** 2

    regions = []
    for region_id in range(1, num_labels + 1):
        sl = bbox_slices[region_id - 1]
        if sl is None or sizes[region_id] < MIN_REGION_PIXELS:
            continue

        # 仅在 bbox 内重建该区域的掩膜
        region = (labeled_mask[sl] == region_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        y0, x0 = sl[0].start, sl[1].start
        for contour in contours:
            perimeter_px = cv2.arcLength(contour, True)
            poly_area_px = cv2.contourArea(contour)
            if perimeter_px <= 0:
                continue

            # 形状因子 = 4πA / P²（A、P 均取轮廓多边形口径，自洽；
            # 理论上 ≤ 1，数值误差可能略超）
            sf = float(np.clip((4.0 * np.pi * poly_area_px) / (perimeter_px ** 2), 0.0, 1.0))

            # 面积列用像素计数，与聚合统计同口径
            area_px = int(sizes[region_id])

            m = cv2.moments(contour)
            cx = m['m10'] / m['m00'] + x0 if m['m00'] > 0 else 0.0
            cy = m['m01'] / m['m00'] + y0 if m['m00'] > 0 else 0.0
            bx, by, bw, bh = cv2.boundingRect(contour)

            regions.append({
                '区域ID': region_id,
                '面积(px)': float(area_px),
                '周长(px)': float(perimeter_px),
                '面积(nm²)': float(area_px * area_per_pixel_nm2),
                '周长(nm)': float(perimeter_px * pixel_size_nm),
                '形状因子': sf,
                '质心X(px)': float(cx),
                '质心Y(px)': float(cy),
                '外接框X(px)': int(bx + x0),
                '外接框Y(px)': int(by + y0),
                '外接框宽(px)': int(bw),
                '外接框高(px)': int(bh),
            })
    return regions


def measure_crystal_properties(
    crystalline_mask: np.ndarray,
    pixel_size_nm: float = 1.0
) -> Tuple[float, float, float, int, List[float]]:
    """
    测量晶体区域的面积、周长和形状因子（兼容入口，委托 compute_full_measurement）

    2026-09-28 审查修正：此前本函数与 compute_full_measurement 各自维护一份
    聚合逻辑，口径容易漂移；现统一走同一条代码路径（旧签名保留，测试与
    潜在外部调用不受影响）。

    区域数量、周长与形状因子使用同一口径：只统计像素数 ≥ MIN_REGION_PIXELS
    的连通区域。注意：total_area 按全部前景像素计数（含 <3px 噪斑），与
    周长/形状因子的区域口径不同——面积求完整、区域统计求抗噪。

    Args:
        crystalline_mask: 晶体区域布尔掩膜 (H, W)
        pixel_size_nm: 像素尺寸 (nm/像素)；未标定时为假定值 1.0，
            面积/周长的实际单位应读作 px/px

    Returns:
        (total_area_nm2, total_perimeter_nm, avg_shape_factor, num_regions, shape_factors)
    """
    result = compute_full_measurement(
        crystalline_mask, crystalline_mask, ~crystalline_mask,
        pixel_size_nm=pixel_size_nm,
    )
    shape_factors = [r['形状因子'] for r in result.regions]
    return (
        result.crystalline_area_nm2,
        result.crystalline_perimeter_nm,
        result.shape_factor_mean,
        result.num_regions,
        shape_factors,
    )


def compute_full_measurement(
    image: np.ndarray,
    crystalline_mask: np.ndarray,
    amorphous_mask: np.ndarray,
    pixel_size_nm: float,
    filename: str = "",
    slice_index: int = 0,
    otsu_threshold: Optional[int] = None,
    pixel_calibrated: bool = False,
) -> MeasurementResult:
    """
    对单张图像执行完整的定量测量

    Args:
        image: 原始灰度图像（用于获取总像素数）
        crystalline_mask: 晶体区域掩膜
        amorphous_mask: 非晶区域掩膜（晶体掩膜的补集，含真空/支持膜/孔洞）
        pixel_size_nm: 像素尺寸（未标定时为假定值 1.0）
        filename: 文件名
        slice_index: 切片序号（从1开始）
        otsu_threshold: Otsu 阈值（如果使用 Otsu 方法）
        pixel_calibrated: 像素尺寸是否经过显微镜标定

    Returns:
        MeasurementResult 完整测量结果
    """
    # 总像素数只需要图像前两维（此前为拿 size 做了整图 cvtColor，
    # 每切片白付一次全图颜色转换，2026-09-28 审查性能项）
    total_pixels = int(image.shape[0]) * int(image.shape[1])
    area_per_pixel_nm2 = pixel_size_nm ** 2

    # 逐区域测量（单次遍历）：聚合统计与逐区域导出共用同一份结果
    regions = measure_regions(crystalline_mask, pixel_size_nm)

    # 像素级统计
    cryst_pixels = int(np.sum(crystalline_mask))
    amorph_pixels = int(np.sum(amorphous_mask))

    # 晶体区域聚合（measure_crystal_properties 委托本函数，口径单一来源）
    cryst_area_nm2 = cryst_pixels * area_per_pixel_nm2
    cryst_perim_px = sum(r['周长(px)'] for r in regions)
    cryst_perim_nm = cryst_perim_px * pixel_size_nm
    shape_factors = [r['形状因子'] for r in regions]
    avg_sf = float(np.mean(shape_factors)) if shape_factors else 0.0
    num_regions = len(regions)

    # 非晶面积
    amorph_area_nm2 = amorph_pixels * area_per_pixel_nm2

    # 面积比例
    cryst_ratio = cryst_pixels / total_pixels if total_pixels > 0 else 0.0
    amorph_ratio = amorph_pixels / total_pixels if total_pixels > 0 else 0.0

    # 形状因子统计
    sf_std = float(np.std(shape_factors)) if shape_factors else 0.0
    sf_min = float(np.min(shape_factors)) if shape_factors else 0.0
    sf_max = float(np.max(shape_factors)) if shape_factors else 0.0

    return MeasurementResult(
        filename=filename,
        slice_index=slice_index,
        total_pixels=total_pixels,
        crystalline_pixels=cryst_pixels,
        amorphous_pixels=amorph_pixels,
        crystalline_area_nm2=cryst_area_nm2,
        amorphous_area_nm2=amorph_area_nm2,
        crystalline_perimeter_nm=cryst_perim_nm,
        num_regions=num_regions,
        shape_factor_mean=avg_sf,
        shape_factor_std=sf_std,
        shape_factor_min=sf_min,
        shape_factor_max=sf_max,
        crystalline_ratio=cryst_ratio,
        amorphous_ratio=amorph_ratio,
        otsu_threshold=otsu_threshold,
        pixel_calibrated=pixel_calibrated,
        regions=regions,
    )


def create_annotated_image(
    gray_image: np.ndarray,
    crystalline_mask: np.ndarray,
    amorphous_mask: np.ndarray,
    uint16_shift: Optional[int] = None,
) -> np.ndarray:
    """
    生成彩色标注图像

    在原始灰度图上叠加半透明颜色标注：
    - 晶体区域：绿色叠加
    - 非晶区域：红色叠加

    混色以"原图的着色副本"为对象（overlay = annotated.copy() 后着色再混合），
    非掩膜像素保持原始亮度。禁止用全零 overlay 混色：那会把掩膜外像素整体
    乘以 (1-alpha) 压暗（回归 2026-09-06 P1）。

    Args:
        gray_image: 灰度图（uint8/uint16/float 均可，内部统一转 uint8）
        crystalline_mask: 晶体区域掩膜
        amorphous_mask: 非晶区域掩膜
        uint16_shift: uint16 → uint8 的显式右移位数；None 按单图自动

    Returns:
        BGR 彩色标注图像 (uint8)
    """
    gray_image = ensure_uint8_gray(gray_image, uint16_shift=uint16_shift)

    # 转为 BGR 三通道
    annotated = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # 两个图层各自以"原图着色副本"为混色对象，逐像素选择所属图层
    cryst_layer = annotated.copy()
    amorph_layer = annotated.copy()
    if np.any(crystalline_mask):
        cryst_layer[crystalline_mask] = COLOR_CRYSTALLINE
    if np.any(amorphous_mask):
        amorph_layer[amorphous_mask] = COLOR_AMORPHOUS

    blend_cryst = cv2.addWeighted(
        cryst_layer, CRYSTALLINE_OVERLAY_ALPHA, annotated, 1 - CRYSTALLINE_OVERLAY_ALPHA, 0)
    blend_amorph = cv2.addWeighted(
        amorph_layer, AMORPHOUS_OVERLAY_ALPHA, annotated, 1 - AMORPHOUS_OVERLAY_ALPHA, 0)

    return np.where(crystalline_mask[:, :, None], blend_cryst, blend_amorph)
