# -*- coding: utf-8 -*-
"""
统计面积核心纯函数模块（无 GUI/Tk 依赖）。

从 `统计面积.py` 中抽出的可测试逻辑：
- 多边形面积（鞋带公式）
- 顶点 clamp
- 自相交检测（线段相交法）
- RGB → uint8 灰度
- TIF 数据维度规范化（结合 TiffFile 元数据）
- 面积换算
"""

from typing import Iterable, List, Optional, Sequence, Tuple

import math

import numpy as np
import re

# 常用彩色轴标记：tifffile 用 C 或 S 表示 samples/color 轴。
COLOR_AXES = ("C", "S")


def natural_sort_key(path) -> tuple:
    """文件名自然排序键：数字段按数值比较，其余按字符串比较。

    "Frame2" 应排在 "Frame10" 之前（字典序会得到相反结果）。
    返回 (0, int) / (1, str) 对的元组，避免 int 与 str 直接比较报错。
    """
    name = str(path)
    key = []
    for part in re.split(r"(\d+)", name):
        if not part:
            continue
        if part.isdigit():
            key.append((0, int(part), ""))
        else:
            key.append((1, 0, part))
    return tuple(key)


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    """鞋带公式计算多边形面积（顶点可为 x/y 序列）。"""
    pts = [(float(p[0]), float(p[1])) for p in points]
    n = len(pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def polygon_perimeter(points: Sequence[Sequence[float]]) -> float:
    """多边形周长（含闭合边）：相邻顶点欧氏距离之和。"""
    pts = [(float(p[0]), float(p[1])) for p in points]
    n = len(pts)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def clamp_point(x: float, y: float, w: float, h: float) -> Tuple[float, float]:
    """把事件坐标 clamp 到图像像素范围 [0, w] × [0, h]。"""
    return min(max(float(x), 0.0), float(w)), min(max(float(y), 0.0), float(h))


def flip_polygon_y(points: Sequence[Sequence[float]], height: float) -> List[Tuple[float, float]]:
    """把点集 y 坐标在“左上原点”与“左下原点”之间翻转（y' = height - y）。

    用于 v2 项目文件（底部原点存储）到 v3（左上原点/行号）的迁移；
    翻转两次应还原原值。
    """
    height = float(height)
    return [(float(p[0]), height - float(p[1])) for p in points]


_EPS = 1e-12


def _orient(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _on_segment(p, q, r) -> bool:
    """共线前提下，判断点 q 是否落在线段 pr 的包围范围内。"""
    return (min(p[0], r[0]) - _EPS <= q[0] <= max(p[0], r[0]) + _EPS
            and min(p[1], r[1]) - _EPS <= q[1] <= max(p[1], r[1]) + _EPS)


def _segments_intersect(p1, p2, p3, p4) -> bool:
    """判断线段 p1-p2 与 p3-p4 是否相交。

    覆盖三类情形（鞋带公式对这些形状的面积意义都有问题）：
    - 严格交叉（蝴蝶形）
    - 端点/顶点落在另一条线段上（压边）
    - 共线重叠（折返尖刺）
    注意：共享端点也判为相交，因此多边形层面的相邻边必须由
    polygon_self_intersects 显式跳过。
    """
    d1 = _orient(p3, p4, p1)
    d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3)
    d4 = _orient(p1, p2, p4)

    if ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4)):
        return True
    if abs(d1) <= _EPS and _on_segment(p3, p1, p4):
        return True
    if abs(d2) <= _EPS and _on_segment(p3, p2, p4):
        return True
    if abs(d3) <= _EPS and _on_segment(p1, p3, p2):
        return True
    if abs(d4) <= _EPS and _on_segment(p1, p4, p2):
        return True
    return False


def polygon_self_intersects(points: Sequence[Sequence[float]]) -> bool:
    """简单多边形自相交检测。

    对每条非相邻边对做线段相交测试（含共线重叠与压边）；
    相邻边（共享端点）显式跳过。
    """
    pts = [(float(p[0]), float(p[1])) for p in points]
    n = len(pts)
    if n < 4:
        return False
    for i in range(n):
        a1, a2 = pts[i], pts[(i + 1) % n]
        for j in range(i + 1, n):
            # 跳过相邻边：j==i+1（顺序相邻）或 (j+1)%n==i（首尾相邻）
            if j == (i + 1) % n or (j + 1) % n == i:
                continue
            b1, b2 = pts[j], pts[(j + 1) % n]
            if _segments_intersect(a1, a2, b1, b2):
                return True
    return False


def rgb_to_gray_uint8(rgb: np.ndarray) -> np.ndarray:
    """把 RGB(A) 图像转成 uint8 灰度（避免生成 float64 大堆叠）。

    缩放规则：
    - 整型输入：按 dtype 最大值线性缩放到 [0, 255]（uint16 不会 mod-256 截断），
      有符号整型先截掉负值。
    - 浮点输入：全局 max <= 1.0 视为 [0,1] 归一化并 ×255；否则假定 [0,255] 直接 clip。
    - NaN → 0，+Inf → 255，-Inf → 0。
    """
    rgb = np.asarray(rgb)
    if rgb.ndim < 3 or rgb.shape[-1] not in (3, 4):
        raise ValueError(f"rgb_to_gray_uint8 需要 (..., H, W, 3/4)，得到 {rgb.shape}")

    channels = rgb[..., :3].astype(np.float64)
    weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)
    gray = channels @ weights

    if np.issubdtype(rgb.dtype, np.integer):
        gray = np.nan_to_num(gray, nan=0.0, posinf=255.0, neginf=0.0)
        info = np.iinfo(rgb.dtype)
        gray = gray * (255.0 / float(info.max))
        return np.clip(gray, 0.0, 255.0).astype(np.uint8)

    # 浮点输入：范围判断只看有限值——若把 ±Inf/NaN 先折算再取 max，
    # 单个 +Inf（→255）会把整张 [0,1] 图误判成 [0,255]，全部压成近 0。
    finite_vals = gray[np.isfinite(gray)]
    finite_max = float(finite_vals.max()) if finite_vals.size else 0.0
    if finite_max <= 1.0:
        gray = gray * 255.0
    gray = np.nan_to_num(gray, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(gray, 0.0, 255.0).astype(np.uint8)


def convert_area_to_units(pixel_area: float, pixel_to_unit_ratio: float,
                          calibrated: bool) -> float:
    """根据校准标志换算物理面积；未校准时返回像素面积。"""
    if not calibrated:
        return float(pixel_area)
    ratio = float(pixel_to_unit_ratio)
    if ratio <= 0:
        return float(pixel_area)
    return float(pixel_area) / (ratio ** 2)


def ruler_ratio(length_px, actual_length) -> float:
    """由标尺长度计算“像素/实际单位”比例。

    Raises
    ------
    ValueError
        输入非数字，或标尺像素长度/实际长度 <= 0（两点重合或输入非法）。
    """
    try:
        length_px = float(length_px)
        actual_length = float(actual_length)
    except (TypeError, ValueError):
        raise ValueError("请输入有效的数字")
    if length_px <= 0:
        raise ValueError(f"标尺像素长度必须为正数，得到 {length_px:.2f}")
    if actual_length <= 0:
        raise ValueError(f"实际长度必须为正数，得到 {actual_length:.2f}")
    return length_px / actual_length


def normalize_tiff_data(arr: np.ndarray, axes: str = "") -> Tuple[np.ndarray, str]:
    """统一 TIF 数据为 (T, H, W) 灰度堆叠。

    Parameters
    ----------
    arr:
        tif.asarray() 得到的数组。
    axes:
        tifffile 报告的轴名，如 ``"YX"``、``"TYX"``、``"YXC"``、``"TYXC"``。

    Returns
    -------
    (stack, note)
        stack: (T, H, W) 灰度堆叠，单张会扩展为 (1, H, W)
        note: 规范化说明文字

    Raises
    ------
    ValueError
        无法识别维度时抛出。
    """
    arr = np.asarray(arr)
    axes = (axes or "").strip()
    has_color_axis = any(c in axes for c in COLOR_AXES)
    has_t_axis = "T" in axes

    if arr.ndim == 2:
        return arr[np.newaxis, ...], "单张灰度"

    if arr.ndim == 3:
        # 单张彩色 (H, W, 3/4) —— 通道在末轴
        if has_color_axis and arr.shape[-1] in (3, 4):
            return rgb_to_gray_uint8(arr)[np.newaxis, ...], "单张彩色，已转灰度"
        # 单张平面彩色 (C, H, W) —— 通道前置（axes 以 C/S 开头）。
        # 修复前会被误判成 3 帧灰度堆叠（R/G/B 三个平面当 3 帧）。
        if has_color_axis and axes[:1] in COLOR_AXES:
            if arr.shape[0] in (3, 4):
                return (rgb_to_gray_uint8(np.moveaxis(arr, 0, -1))[np.newaxis, ...],
                        "单张彩色(平面格式)，已转灰度")
            raise ValueError(
                f"不支持的通道前置形状: {arr.shape}（axes={axes!r}，仅支持 3/4 通道）")
        # 灰度堆叠 (T, H, W)。若没有轴信息且最后一轴为 3/4，也按堆叠处理，
        # 避免把 W=3/4 的灰度堆叠误判为彩色。
        return arr, "灰度堆叠"

    if arr.ndim == 4:
        if arr.shape[-1] in (3, 4) and (has_color_axis or not has_t_axis):
            # (T, H, W, C) 彩色堆叠 → 逐帧转 uint8 灰度后 stack
            gray_frames = [rgb_to_gray_uint8(arr[i]) for i in range(arr.shape[0])]
            return np.stack(gray_frames), "彩色堆叠，已逐帧转灰度"
        if arr.shape[-1] == 1:
            return arr[..., 0], "4D 单通道，已压缩"
        # (T, C, H, W) 通道前置
        if arr.shape[1] in (3, 4) and arr.shape[0] > arr.shape[1]:
            gray_frames = [
                rgb_to_gray_uint8(np.moveaxis(arr[i], 0, -1))
                for i in range(arr.shape[0])
            ]
            return np.stack(gray_frames), "4D 通道前置彩色堆叠，已转灰度"
        if arr.shape[1] == 1:
            return arr[:, 0, ...], "4D 单通道，已压缩"

    raise ValueError(f"无法识别的 TIF 数据形状: {arr.shape}（axes={axes!r}）")


def group_series_by_root(measurements: dict, polygon_roots: dict) -> dict:
    """把逐帧测量值按多边形 lineage（root id）分组，供“面积变化”追踪图使用。

    Parameters
    ----------
    measurements:
        {frame_idx: {poly_id: {'pixel_area': float} 或 旧版纯数值}}。
    polygon_roots:
        {poly_id: root_id}；缺失的 id 视为自己的 root（与旧版项目兼容）。

    Returns
    -------
    {root_id: [(frame_idx, pixel_area, poly_id), ...]}，组内按帧号升序；
    poly_id 供调用方取该副本的名称/颜色。
    """
    series = {}
    for frame_idx, frame_data in measurements.items():
        for poly_id, measurement in frame_data.items():
            if isinstance(measurement, dict):
                area = measurement.get('pixel_area', 0)
            else:
                area = float(measurement) if measurement else 0
            root = polygon_roots.get(poly_id, poly_id)
            series.setdefault(root, []).append((frame_idx, area, poly_id))
    for root in series:
        series[root].sort(key=lambda item: item[0])
    return series


def sanitize_project_polygons(raw: dict, total_frames: int) -> Tuple[dict, int]:
    """清洗项目文件中的多边形数据，防御手工编辑或损坏的 JSON。

    规则：
    - 帧索引必须是 [0, total_frames) 内的整数，越界/非法整帧丢弃；
    - 每个多边形至少 3 个有限数值点对，否则丢弃；
    - 点对写法为 [x, y]，非数值或 NaN/Inf 丢弃。

    Returns
    -------
    (polygons, dropped)
        polygons: {frame_idx: {poly_id: [(x, y), ...]}}
        dropped: 被丢弃的多边形总数
    """
    cleaned: dict = {}
    dropped = 0
    for frame_str, frame_polys in (raw or {}).items():
        n_here = len(frame_polys) if isinstance(frame_polys, dict) else 1
        try:
            frame_idx = int(frame_str)
        except (TypeError, ValueError):
            dropped += n_here
            continue
        if not (0 <= frame_idx < total_frames):
            dropped += n_here
            continue
        if not isinstance(frame_polys, dict):
            dropped += 1
            continue
        for poly_str, points in frame_polys.items():
            try:
                poly_id = int(poly_str)
                pts = []
                for p in points:
                    x, y = float(p[0]), float(p[1])
                    if not (math.isfinite(x) and math.isfinite(y)):
                        raise ValueError("非有限坐标")
                    pts.append((x, y))
            except (TypeError, ValueError, IndexError):
                dropped += 1
                continue
            if len(pts) < 3:
                dropped += 1
                continue
            cleaned.setdefault(frame_idx, {})[poly_id] = pts
    return cleaned, dropped


def rebuild_measurements(polygons: dict, pixel_to_unit_ratio: float,
                         calibrated: bool) -> dict:
    """从多边形顶点重建全部测量值（顶点是唯一数据源，避免缓存失真）。

    Returns
    -------
    {frame_idx: {poly_id: {'pixel_area': float, 'unit_area': float}}}
    """
    measurements = {}
    for frame_idx, frame_polys in polygons.items():
        frame_m = {}
        for poly_id, points in frame_polys.items():
            pixel_area = polygon_area(points)
            frame_m[poly_id] = {
                'pixel_area': pixel_area,
                'unit_area': convert_area_to_units(
                    pixel_area, pixel_to_unit_ratio, calibrated),
            }
        if frame_m:
            measurements[frame_idx] = frame_m
    return measurements


def count_out_of_bounds_points(polygons: dict, width: float, height: float) -> int:
    """统计落在 [0,width]×[0,height] 之外的多边形顶点数（供加载时告警）。

    数据仍保留不丢弃，仅提示用户坐标可能与当前图像不匹配。
    """
    count = 0
    for frame_polys in (polygons or {}).values():
        for points in frame_polys.values():
            for x, y in points:
                if x < 0 or x > width or y < 0 or y > height:
                    count += 1
    return count


def sanitize_ruler_points(raw, width: Optional[float] = None,
                          height: Optional[float] = None) -> List[Tuple[float, float]]:
    """清洗项目文件中的标尺两点数据。

    返回合法的 2 点列表；点数不是 2、坐标非有限数值时返回 []。
    给出 width/height 时把越界点 clamp 到图像边界（与点击时的行为一致）。
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return []
    pts = []
    for p in raw:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            return []
        if not (math.isfinite(x) and math.isfinite(y)):
            return []
        pts.append((x, y))
    if width is not None and height is not None:
        pts = [clamp_point(x, y, float(width), float(height)) for x, y in pts]
    return pts


def coerce_positive_float(value) -> Optional[float]:
    """把项目文件中的数值字段安全转为正有限 float。

    非法输入（None/非数字/NaN/±Inf/<=0）返回 None，
    调用方据此回退到默认值并给出告警。
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number
