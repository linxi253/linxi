# -*- coding: utf-8 -*-
"""
图像分割算法模块

提供多种图像分割方法用于区分电镜图像中的晶体区域与非晶区域：
- 固定阈值分割
- Otsu 自动阈值分割（新增）
- 自适应阈值分割
- Canny 边缘检测 + 轮廓填充
- 分水岭算法

每种方法在 8 位灰度图上以"亮区为前景"计算二值掩膜；params.foreground
可选 "dark" 将掩膜反相（TEM 明场衍射衬度下晶体区域常更暗）。

重要口径说明：所有方法本质都是**灰度阈值分割**。"crystalline_mask" 的
物理含义是"高/低衬度区"，并非严格的晶体学判定；"amorphous_mask" 是
晶体掩膜的补集，**包含真空、支持膜和孔洞**，请勿把"非晶面积"直接作为
非晶相的物理面积引用（HAADF Z 衬度下近似成立，TEM 明场不成立）。
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple

from constants import (
    DEFAULT_THRESHOLD,
    DEFAULT_ADAPTIVE_BLOCKSIZE,
    DEFAULT_ADAPTIVE_C,
    DEFAULT_CANNY_LOW,
    DEFAULT_CANNY_HIGH,
    DEFAULT_WATERSHED_KERNEL,
    DEFAULT_CANNY_MIN_AREA,
    FOREGROUND_BRIGHT,
    FOREGROUND_DARK,
)


@dataclass
class SegmentationParams:
    """分割参数数据类，集中管理所有分割方法的参数"""
    method: str = "threshold"

    # 固定阈值参数
    threshold: int = DEFAULT_THRESHOLD

    # 自适应阈值参数
    adaptive_blocksize: int = DEFAULT_ADAPTIVE_BLOCKSIZE
    adaptive_c: int = DEFAULT_ADAPTIVE_C

    # Canny 参数
    canny_low: int = DEFAULT_CANNY_LOW
    canny_high: int = DEFAULT_CANNY_HIGH
    canny_min_area: int = DEFAULT_CANNY_MIN_AREA

    # 分水岭参数
    watershed_kernel: int = DEFAULT_WATERSHED_KERNEL

    # 前景极性：bright=亮区为晶体（默认），dark=暗区为晶体（掩膜反相）
    foreground: str = FOREGROUND_BRIGHT

    def validate(self):
        """验证参数合法性，自动修正非法值"""
        self.threshold = max(0, min(255, self.threshold))
        # 自适应阈值块大小必须为奇数且 >= 3
        if self.adaptive_blocksize < 3:
            self.adaptive_blocksize = 3
        if self.adaptive_blocksize % 2 == 0:
            self.adaptive_blocksize += 1
        # Canny 阈值
        self.canny_low = max(0, min(255, self.canny_low))
        self.canny_high = max(self.canny_low + 1, min(255, self.canny_high))
        # 分水岭核大小
        self.watershed_kernel = max(1, min(21, self.watershed_kernel))
        # 前景极性
        if self.foreground not in (FOREGROUND_BRIGHT, FOREGROUND_DARK):
            self.foreground = FOREGROUND_BRIGHT


def resolve_uint16_shift(img_max: float) -> int:
    """按数据实际量程选择 uint16 → uint8 的右移位数。

    12-bit 相机（满量程 0-4095）用固定 /256 会被压到 0-15，导致固定阈值
    与 Otsu 全部失效。按量程分档右移 0/4/6/8 位，每档把该量程顶端映射到
    ~255，固定阈值始终有可用动态范围。位深判定应以"数据集"为单位
    （同一批数据共用一个 shift），而不是逐图判定——逐图判定会让同一堆栈中
    明暗不同的切片使用不同映射，破坏跨切片可比性（回归 2026-09-06 P1；
    2026-09-28 补 14-bit 档：0-16383 数据曾被 >>8 压到 0-63，阈值 128
    静默全灭）。

    分档保持粗粒度（而非按 max 逐位计算）：同为 12-bit 相机的明暗两次
    采集仍落在同一档，跨数据集的固定阈值语义尽量稳定。

    Args:
        img_max: 该数据集（目录/堆栈）中观测到的最大灰度值

    Returns:
        右移位数（0 / 4 / 6 / 8）
    """
    if img_max <= 255:
        return 0
    if img_max <= 4095:
        return 4
    if img_max <= 16383:
        return 6
    return 8


def uint16_to_uint8(image: np.ndarray, shift: Optional[int] = None) -> np.ndarray:
    """uint16 → uint8 按给定移位缩放，而非固定 /256。

    Args:
        image: uint16 灰度图
        shift: 右移位数。None 时按单图最大值自动选择（仅适合独立单图场景；
            批量/堆栈应先用 resolve_uint16_shift 在数据集级确定 shift 再传入）。

    Returns:
        uint8 灰度图
    """
    if shift is None:
        shift = resolve_uint16_shift(float(np.max(image)))
    if shift == 0:
        return image.astype(np.uint8)
    return np.right_shift(image, shift).astype(np.uint8)


def has_non_finite(image: np.ndarray) -> bool:
    """检测图像是否含 NaN/Inf（仅 float 类型可能；整型恒为有限值）。"""
    if image.dtype.kind == 'f':
        return not bool(np.isfinite(image).all())
    return False


def to_uint8(image: np.ndarray, uint16_shift: Optional[int] = None) -> np.ndarray:
    """仅做 dtype 归一（不改通道数），供显示/标注等需要保留彩色的场景。

    - uint8 原样返回；uint16 按位移缩放（显式 shift 优先，否则按单图自动）
    - float：NaN/Inf 按背景(0)处理后线性归一化到 [0, 255]
    """
    if image.dtype == np.uint8:
        return image
    if image.dtype == np.uint16:
        return uint16_to_uint8(image, uint16_shift)

    # float 等类型：线性归一化到 [0, 255]。
    # NaN/Inf 会传播 min/max 使整图归零，必须先按背景处理（回归 2026-09-06 P1：
    # 含 NaN 的 float TIFF 曾静默产出全零分割）。逐图 min/max 会影响批量
    # 可比性，float 输入建议先转成固定位深。
    if image.dtype.kind == 'f' and not np.isfinite(image).all():
        finite = np.isfinite(image)
        work = np.where(finite, image, 0.0).astype(np.float64)
        img_min = float(work.min())
        img_max = float(work.max())
        if img_max > img_min:
            work = (work - img_min) / (img_max - img_min) * 255.0
        return work.astype(np.uint8)

    img_min = float(image.min())
    img_max = float(image.max())
    if img_max > img_min:
        return ((image.astype(np.float64) - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
    return np.zeros_like(image, dtype=np.uint8)


def ensure_uint8_gray(image: np.ndarray, uint16_shift: Optional[int] = None) -> np.ndarray:
    """
    将输入图像转换为 8 位灰度图

    处理逻辑：
    1. 如果是 3 通道彩色图 → 转灰度
    2. 之后走 to_uint8 完成 dtype 归一

    Args:
        image: 输入图像（任意通道数和数据类型）
        uint16_shift: uint16 专用移位数（数据集级统一），None 按单图自动

    Returns:
        8 位灰度图像 (dtype=uint8, shape=(H, W))
    """
    # 多通道转灰度
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            image = image[:, :, 0]  # 取第一个通道

    return to_uint8(image, uint16_shift)


def segment_threshold(gray: np.ndarray, params: SegmentationParams) -> np.ndarray:
    """
    固定阈值分割

    原理：像素值 >= threshold 的区域判定为晶体区域。
    适用于衬度均匀、晶体与非晶区域灰度差异明显的图像。

    Args:
        gray: 8位灰度图
        params: 分割参数

    Returns:
        晶体区域布尔掩膜
    """
    _, binary = cv2.threshold(gray, params.threshold, 255, cv2.THRESH_BINARY)
    return binary == 255


def segment_otsu(gray: np.ndarray, params: SegmentationParams) -> Tuple[np.ndarray, int]:
    """
    Otsu 自动阈值分割（新增方法）

    原理：自动寻找使类间方差最大的阈值，将图像分为前景/背景两类。
    适用于批量处理时无需手动调参的场景。

    Args:
        gray: 8位灰度图
        params: 分割参数（此方法不使用手动阈值）

    Returns:
        (晶体区域布尔掩膜, 自动计算的阈值)
    """
    # 先做轻微高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    otsu_thresh, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary == 255, int(otsu_thresh)


def segment_adaptive(gray: np.ndarray, params: SegmentationParams) -> np.ndarray:
    """
    自适应阈值分割

    原理：对每个像素根据其邻域计算局部阈值，适应光照/衬度不均匀的图像。
    使用高斯加权方式计算邻域均值。

    块大小在运行时钳位到不超过图像短边（cv2 要求 blockSize ≤ min(H, W)，
    否则抛异常；validate() 无法感知图像尺寸，只能在此处收口）。

    Args:
        gray: 8位灰度图
        params: 分割参数

    Returns:
        晶体区域布尔掩膜
    """
    blocksize = params.adaptive_blocksize
    # 钳位到不超过短边的最大奇数，避免小图 + 大块大小的 cv2 异常
    max_odd = min(gray.shape[0], gray.shape[1])
    if max_odd % 2 == 0:
        max_odd -= 1
    blocksize = max(3, min(blocksize, max_odd))
    c = params.adaptive_c

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blocksize, c
    )
    return binary == 255


def segment_canny(gray: np.ndarray, params: SegmentationParams) -> np.ndarray:
    """
    Canny 边缘检测 + 轮廓填充分割

    原理：
    1. Canny 检测边缘
    2. 膨胀闭合边缘间隙
    3. 查找轮廓并填充面积大于阈值的区域

    适用于晶体区域边界清晰、内部衬度均匀的图像。

    Args:
        gray: 8位灰度图
        params: 分割参数

    Returns:
        晶体区域布尔掩膜
    """
    # 预处理：轻微模糊减少噪声边缘
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Canny 边缘检测
    edges = cv2.Canny(blurred, params.canny_low, params.canny_high)

    # 膨胀闭合边缘间隙
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # 查找并填充轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > params.canny_min_area:
            cv2.drawContours(mask, [contour], -1, 255, -1)

    return mask == 255


def segment_watershed(gray: np.ndarray, params: SegmentationParams) -> np.ndarray:
    """
    分水岭算法分割

    原理：
    1. Otsu 二值化获取前景
    2. 形态学开运算去噪
    3. 距离变换确定确信前景
    4. 分水岭算法分割粘连区域

    适用于晶体颗粒相互粘连、需要分离的场景。

    Args:
        gray: 8位灰度图
        params: 分割参数

    Returns:
        晶体区域布尔掩膜
    """
    # 高斯模糊减噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu 二值化（注意：这里用 THRESH_BINARY 而非 INV，亮区为前景）
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学开运算去除小噪点
    kernel_size = params.watershed_kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 确定背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 距离变换确定确信前景
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dist_max = dist_transform.max()
    if dist_max == 0:
        # 图像无前景区域，直接返回 Otsu 结果
        return thresh == 255
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_max, 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # 未知区域
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 连通域标记
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 分水岭
    img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)

    # 标记 > 1 的区域为晶体（排除背景=1 和边界=-1）
    return markers > 1


def segment_image(
    image: np.ndarray,
    params: SegmentationParams,
    uint16_shift: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
    """
    统一分割入口

    根据 params.method 选择对应的分割算法，返回晶体掩膜和附加信息。
    params.foreground == "dark" 时掩膜整体反相（暗区为前景/晶体）。

    Args:
        image: 输入图像（任意格式）
        params: 分割参数
        uint16_shift: uint16 输入的数据集级移位数（None 按单图自动）。
            批量/堆栈应先在数据集级 resolve_uint16_shift 后传入，
            保证所有切片使用同一映射。

    Returns:
        (crystalline_mask, amorphous_mask, info_dict)
        - crystalline_mask: 前景（"晶体"）区域布尔掩膜，失败时为 None。
          注意这是灰度阈值意义上的"高衬度区"，非严格晶体学判定。
        - amorphous_mask: crystalline_mask 的补集，**包含真空、支持膜和
          孔洞**，请勿直接作为非晶相的物理面积引用。
        - info_dict: 附加信息（Otsu 阈值、前景极性、8 位灰度图、
          invalid_input 标记等）
    """
    params.validate()

    # 输入健全性标记：float 图含 NaN/Inf 时归一化会按背景处理，调用方应告警
    invalid_input = has_non_finite(image)

    # 非 uint8/uint16 输入（int/float）只能按单图 min/max 归一化，
    # 跨切片映射不一致——标记出来由调用方告警（2026-09-28 审查项）
    per_image_norm = image.dtype not in (np.uint8, np.uint16)

    # 转换为 8 位灰度图
    gray = ensure_uint8_gray(image, uint16_shift=uint16_shift)

    info = {
        "method": params.method,
        "gray_image": gray,
        "foreground": params.foreground,
        "invalid_input": invalid_input,
        "per_image_normalization": per_image_norm,
    }

    method = params.method
    crystalline_mask = None

    if method == "threshold":
        crystalline_mask = segment_threshold(gray, params)

    elif method == "otsu":
        crystalline_mask, otsu_value = segment_otsu(gray, params)
        info["otsu_threshold"] = otsu_value

    elif method == "adaptive":
        crystalline_mask = segment_adaptive(gray, params)

    elif method == "canny":
        crystalline_mask = segment_canny(gray, params)

    elif method == "watershed":
        crystalline_mask = segment_watershed(gray, params)

    else:
        raise ValueError(f"不支持的分割方法: {method}")

    if crystalline_mask is None:
        return None, None, info

    # 暗场/明场极性：所有方法都以"亮区为前景"计算，暗区极性时整体反相
    if params.foreground == FOREGROUND_DARK:
        crystalline_mask = ~crystalline_mask

    # 注意：amorphous_mask 是补集口径，含真空/支持膜/孔洞（见模块 docstring）
    amorphous_mask = ~crystalline_mask
    return crystalline_mask, amorphous_mask, info
