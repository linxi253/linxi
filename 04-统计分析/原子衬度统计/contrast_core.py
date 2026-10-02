# -*- coding: utf-8 -*-
"""
TIF 衬度分析核心算法模块（无 GUI 依赖）。

从主程序拆分出来，便于不导入 Tk/matplotlib 即可进行单元测试。

包含：
- 版本常量 __version__（全项目唯一版本来源）
- 算法注册表 ALGORITHMS
- compute_contrast：单帧 ROI 衬度计算
- compute_contrast_stack / compute_contrast_stack_ex：整堆栈向量化衬度计算
- normalize_tiff_array / normalize_tiff_array_ex：TIF 数组维度规范化
- probe_tiff：读取 TIFF 头部信息（布局、压缩、采集元数据、启发式像素尺寸）
- open_tiff_stack / open_tiff_stack_ex：打开 TIF 文件（memmap 优先，失败回退内存）
- page_datetimes：读取每页 DateTime 标签（时间序列 CSV 导出用）
- pixel_size_nm：把启发式解析到的像素尺寸换算为 nm（ROI 物理面积估算用）
- estimate_display_clim：采样估计全堆栈显示灰度范围（仅显示，不参与计算）
- clamp_roi_to_image：ROI 顶点 clamp（floor/ceil 取整）
"""

__version__ = "2.5.0"

import os
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import tifffile

# 算法枚举：(显示名, method_key)。索引由 GUI 下拉框维护。
ALGORITHMS: List[Tuple[str, str]] = [
    ("标准差/平均强度", "std_over_mean"),
    ("Michelson衬度", "michelson"),
    ("方差/平均强度", "var_over_mean"),
    ("稳健Michelson(1%-99%分位)", "michelson_robust"),
]

_ALGORITHM_KEYS: Tuple[str, ...] = tuple(key for _, key in ALGORITHMS)

# 颜色轴标记：tifffile 用 "C"（chunky）或 "S"（planar samples）表示彩色轴。
_COLOR_AXES = "CS"


def algorithm_names() -> List[str]:
    """返回算法显示名列表（GUI 下拉框使用）。"""
    return [name for name, _ in ALGORITHMS]


def method_name_of(method_key: str) -> str:
    """由 key 反查显示名；未知 key 返回 key 本身。"""
    for name, key in ALGORITHMS:
        if key == method_key:
            return name
    return method_key


def _check_method_key(method_key: str) -> None:
    """校验算法 key；必须在任何提前返回之前调用。

    否则「空数组 + 非法 key」会静默返回 NaN，与文档约定（未知 key 抛
    ValueError）不一致。
    """
    if method_key not in _ALGORITHM_KEYS:
        raise ValueError(
            f"未知的衬度算法 key: {method_key!r}，请使用 {list(_ALGORITHM_KEYS)!r}")


def _as_float_array(arr, what: str = "输入") -> np.ndarray:
    """转 float64；复数额度直接报错而不是静默丢弃虚部。"""
    a = np.asarray(arr)
    if np.iscomplexobj(a):
        raise ValueError(
            f"不支持复数像素质地（dtype={a.dtype}）；请先取实部/模后再计算衬度。")
    return a.astype(np.float64, copy=False)


def compute_contrast(roi_data: np.ndarray, method_key: str) -> float:
    """按 method_key 计算单帧 ROI 衬度。

    Parameters
    ----------
    roi_data:
        任意形状的 ROI 像素数组（会被展平统计）。
    method_key:
        ``ALGORITHMS`` 中的 key，如 ``"std_over_mean"``。

    Returns
    -------
    float
        衬度值；空数组或分母为零时返回 ``np.nan``。

    Raises
    ------
    ValueError
        当 ``method_key`` 不是已注册算法，或输入为复数数组时抛出。

    Notes
    -----
    四种定义都假定强度非负（``min >= 0``）。输入含负值时 Michelson 类
    可能返回 > 1 或负值，``std/mean`` 可能为负——这在数学上无意义，
    调用方（GUI）会给出显式警告。

    ``michelson_robust`` 的 1%/99% 分位只有在像素数足够（约 ≥100）时
    才能排除单像素离群；小 ROI（如 3×3=9 像素）下 p99 仍含约 92%
    权重的最大值，结果与极值版几乎相同（调用方据此提示用户）。
    """
    _check_method_key(method_key)
    arr = _as_float_array(roi_data)
    if arr.size == 0:
        return float("nan")

    # errstate：含 ±Inf 的帧在 mean/std 中触发 invalid 运算警告
    # （如 Inf-Inf），结果本身按 NaN 传播是预期行为，无需告警。
    with np.errstate(invalid="ignore"):
        m = float(np.mean(arr))
        s = float(np.std(arr))

        if method_key == "std_over_mean":
            return s / m if m != 0 else float("nan")
        if method_key == "michelson":
            v_max = float(np.max(arr))
            v_min = float(np.min(arr))
            denom = v_max + v_min
            return (v_max - v_min) / denom if denom != 0 else float("nan")
        if method_key == "var_over_mean":
            return (s ** 2) / m if m != 0 else float("nan")
        # michelson_robust：用 1%/99% 分位替代全局极值，抑制单像素离群
        # （热像素/宇宙射线）的影响。
        p1, p99 = np.percentile(arr, [1.0, 99.0])
        denom = float(p99) + float(p1)
        return (float(p99) - float(p1)) / denom if denom != 0 else float("nan")


def compute_contrast_stack_ex(stack: np.ndarray,
                              method_key: str) -> Tuple[np.ndarray, np.ndarray,
                                                        np.ndarray, np.ndarray,
                                                        np.ndarray]:
    """整段 (F, H, W) 堆栈逐帧向量化计算衬度，并返回每帧强度统计。

    Returns
    -------
    (contrast, mean, std, vmin, vmax)
        五个形状 ``(F,)`` 的 float64 数组。空 ROI 或分母为零的帧，衬度
        为 NaN；空 ROI 的 mean/std/min/max 同样为 NaN。
        每帧均值/极值随衬度一并导出，便于判断「衬度变化」是结构变化
        还是束流/厚度变化，也便于发现热像素与饱和。

    Notes
    -----
    含 ±Inf 像素的帧在 std/divide 中触发 invalid 运算警告（如
    Inf-Inf、Inf/Inf），衬度按 NaN 传播是预期行为，这里统一抑制，
    由调用方（GUI）统计非有限帧数并给出显式警告。
    """
    _check_method_key(method_key)
    arr = _as_float_array(stack)
    if arr.ndim != 3:
        raise ValueError(f"compute_contrast_stack 需要 (F, H, W) 输入，得到 {arr.shape}")

    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    mean = np.full(n, np.nan, dtype=np.float64)
    std = np.full(n, np.nan, dtype=np.float64)
    vmin = np.full(n, np.nan, dtype=np.float64)
    vmax = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or arr.shape[1] == 0 or arr.shape[2] == 0:
        return out, mean, std, vmin, vmax

    with np.errstate(invalid="ignore"):
        arr.mean(axis=(1, 2), dtype=np.float64, out=mean)
        arr.std(axis=(1, 2), dtype=np.float64, out=std)
        arr.min(axis=(1, 2), out=vmin)
        arr.max(axis=(1, 2), out=vmax)

        if method_key == "std_over_mean":
            np.divide(std, mean, out=out, where=mean != 0)
        elif method_key == "var_over_mean":
            np.divide(std ** 2, mean, out=out, where=mean != 0)
        elif method_key == "michelson":
            denom = vmax + vmin
            np.divide(vmax - vmin, denom, out=out, where=denom != 0)
        else:  # michelson_robust
            # numpy >= 1.22 支持分位数的 tuple axis，一次调用处理整段堆栈。
            p1, p99 = np.percentile(arr, [1.0, 99.0], axis=(1, 2))
            denom = p99 + p1
            np.divide(p99 - p1, denom, out=out, where=denom != 0)
    return out, mean, std, vmin, vmax


def compute_contrast_stack(stack: np.ndarray, method_key: str) -> np.ndarray:
    """按 method_key 对整段 (F, H, W) 堆栈逐帧向量化计算衬度。

    与对每帧调用 :func:`compute_contrast` 数值等价；空 ROI 或分母为零
    的帧返回 NaN。需要每帧强度统计时用 :func:`compute_contrast_stack_ex`。
    """
    return compute_contrast_stack_ex(stack, method_key)[0]


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """将 (..., H, W, 3/4) RGB(A) 转为灰度，保持原始 dtype 数值范围。

    对于 uint8/uint16 等整型，使用整数权重加权后四舍五入（np.rint）
    并 clip 回原 dtype 范围，纯白 255 映射到 255；对于浮点，直接加权
    求和并 clip 到 [0, +inf) 后转回原 dtype（不设上界，保持浮点动态范围）。
    """
    rgb = np.asarray(rgb)
    if rgb.ndim < 3 or rgb.shape[-1] not in (3, 4):
        raise ValueError(f"rgb_to_gray 需要 (..., H, W, 3/4) 输入，得到 {rgb.shape}")

    channels = rgb[..., :3].astype(np.float64)
    weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)
    weighted = channels @ weights

    dtype = rgb.dtype
    if np.issubdtype(dtype, np.integer):
        gray = np.rint(weighted)
        info = np.iinfo(dtype)
        return np.clip(gray, 0, info.max).astype(dtype)
    return np.clip(weighted, 0.0, None).astype(dtype)


def _color_axis_position(shape: Tuple[int, ...], axes: str) -> Optional[int]:
    """按 axes 判断颜色轴位置；无颜色轴或长度不匹配时返回 None。

    仅当该轴的实际长度为 3/4 时才认定为颜色轴，避免 axes 字符串与实际
    形状不符时误判。
    """
    if not axes or len(axes) != len(shape):
        return None
    for pos, ch in enumerate(axes):
        if ch in _COLOR_AXES and shape[pos] in (3, 4):
            return pos
    return None


def normalize_tiff_array_ex(arr: np.ndarray, axes: Optional[str] = None,
                            force_frames: bool = False
                            ) -> Tuple[np.ndarray, List[str]]:
    """统一 TIF 数组维度为 (T, H, W) 灰度堆叠，并返回处理说明。

    处理规则：
    - ``(H, W)`` 单张灰度 -> ``(1, H, W)``
    - ``(H, W, C=3/4)`` 单张彩色 -> 转灰度 -> ``(1, H, W)``
    - ``(C/S, H, W)`` planar 彩色单页 -> 转置后转灰度 -> ``(1, H, W)``
    - ``(T, H, W)`` 灰度堆叠 -> 原样
    - ``(T, H, W, C=3/4)`` / ``(T, C, H, W)`` 彩色堆叠 -> 逐帧转灰度
    - 其他维度 -> 抛出 ValueError（信息中含 axes，便于定位）

    ``axes`` 为 tifffile 报告的轴名字符串（如 ``"TYX"``、``"YXC"``、
    ``"SYX"``）。提供 axes 时优先以 axes 判断颜色/堆叠；未提供 axes 时
    3D 一律按灰度堆叠处理，避免把 ``W=3/4`` 的窄条灰度堆叠误判为彩色。

    ``force_frames=True`` 时，把 planar 彩色单页（``axes[0]`` 为 C/S 且
    长度为 3/4）当作**多帧堆叠**读取。部分软件（含 tifffile 自身在
    3 帧/4 帧输入时的默认行为）会把多帧写成 planar RGB(A)，
    这时按颜色解析会静默合并帧，需要用户显式覆盖。

    Returns
    -------
    (array, notes)
        notes 为面向用户的中文说明列表（GUI 直接逐行显示）。
    """
    arr = np.asarray(arr)
    axes = (axes or "").strip()
    notes: List[str] = []

    if arr.ndim == 2:
        return arr[np.newaxis, ...], notes

    if arr.ndim == 3:
        pos = _color_axis_position(arr.shape, axes)
        if force_frames and pos == 0:
            notes.append(
                f"已按『多帧解析』把 {arr.shape[0]} 个平面当作 {arr.shape[0]} 帧读取"
                f"（axes={axes or '未知'}）")
            return arr, notes
        if pos is not None:
            planes = arr.shape[pos]
            if pos == arr.ndim - 1:
                gray = rgb_to_gray(arr)
                notes.append(f"检测到单张彩色图（{planes} 通道）→ 已按 RGB 权重转灰度，得到 1 帧")
            else:
                gray = rgb_to_gray(np.moveaxis(arr, pos, -1))
                notes.append(
                    f"检测到 planar 彩色单页（axes={axes or '未知'}，{planes} 个平面）"
                    f"→ 已按 RGB 权重转灰度，得到 1 帧")
                if planes == 4:
                    notes.append("平面 4（alpha）已忽略；若这 4 个平面实为 4 帧，请勾选『多帧解析』")
                elif planes == 3:
                    notes.append("若这 3 个平面实为 3 帧，请勾选『多帧解析』")
            return gray[np.newaxis, ...], notes
        # 无颜色轴（或 axes 缺失）：按灰度堆叠处理
        if "Z" in axes:
            notes.append(
                f"检测到 Z 轴（axes={axes}）：Z 切片将按『帧』序处理；"
                f"若这是层析/系列切片而非时间序列，请留意结果口径")
        return arr, notes

    if arr.ndim == 4:
        pos = _color_axis_position(arr.shape, axes)
        if pos == 3:
            if arr.shape[3] == 1:
                notes.append("彩色轴长度为 1 → 已按单通道灰度堆叠读取")
                return arr[..., 0], notes
            notes.append(
                f"检测到彩色堆叠（axes={axes}）→ 逐帧按 RGB 权重转灰度，得到 {arr.shape[0]} 帧")
            return np.stack([rgb_to_gray(arr[i]) for i in range(arr.shape[0])]), notes
        if pos == 1:
            notes.append(
                f"检测到 planar 彩色堆叠（axes={axes}）→ 逐帧按 RGB 权重转灰度，"
                f"得到 {arr.shape[0]} 帧")
            return np.stack([rgb_to_gray(np.moveaxis(arr[i], 0, -1))
                             for i in range(arr.shape[0])]), notes
        if pos is not None:
            raise ValueError(
                f"无法识别的 4D TIF 数据形状: {arr.shape}（axes={axes}，颜色轴位于第 {pos} 轴）；"
                f"预期为 (T, H, W, C) 或 (T, C, H, W)。")
        if "Z" in axes:
            raise ValueError(
                f"检测到三维层析数据（axes={axes}，形状 {arr.shape}）；本工具只处理二维"
                f"图像堆叠，请先导出为 (T, Y, X) 后再分析。")
        # axes 未提供颜色轴：保留形状启发式（(T, H, W, 1) 与 (T, C, H, W)）
        if arr.shape[3] == 1:
            return arr[..., 0], notes
        if arr.shape[3] in (3, 4):
            notes.append(
                f"未提供可用 axes，按 (T, H, W, {arr.shape[3]}) 彩色堆叠猜测 → 逐帧转灰度，"
                f"得到 {arr.shape[0]} 帧")
            return np.stack([rgb_to_gray(arr[i]) for i in range(arr.shape[0])]), notes
        if arr.shape[1] in (1, 3, 4):
            if arr.shape[1] == 1:
                notes.append("未提供可用 axes，按 (T, 1, H, W) 单通道堆叠猜测")
                return arr[:, 0, ...], notes
            notes.append(
                f"未提供可用 axes，按 (T, {arr.shape[1]}, H, W) planar 彩色堆叠猜测 → "
                f"逐帧转灰度，得到 {arr.shape[0]} 帧")
            return np.stack([rgb_to_gray(np.moveaxis(arr[i], 0, -1))
                             for i in range(arr.shape[0])]), notes
        raise ValueError(
            f"无法识别的 4D TIF 数据形状: {arr.shape}（axes={axes or '未知'}）；"
            f"预期为 (T, H, W, 3/4) 彩色堆叠或 (T, H, W, 1)。")

    raise ValueError(
        f"不支持的 TIF 数据维度: {arr.ndim}D，形状 {arr.shape}（axes={axes or '未知'}）；"
        f"仅支持单张 2D、单张彩色 3D、灰度堆叠 3D 或彩色堆叠 4D。")


def normalize_tiff_array(arr: np.ndarray,
                         axes: Optional[str] = None,
                         force_frames: bool = False) -> np.ndarray:
    """统一 TIF 数组维度为 (T, H, W) 灰度堆叠；返回数组（丢弃处理说明）。

    规则详见 :func:`normalize_tiff_array_ex`。
    """
    return normalize_tiff_array_ex(arr, axes=axes, force_frames=force_frames)[0]


# ==================== TIFF 头部探测 ====================

@dataclass(frozen=True)
class TiffProbe:
    """TIFF 头部信息（不含像素数据），用于加载前评估与元数据记录。"""

    axes: str = ""
    shape: Tuple[int, ...] = ()
    dtype_str: str = ""
    nbytes: int = 0
    n_pages: int = 0
    compression: str = "未知"
    is_compressed: bool = False
    photometric: str = ""
    planarconfig: str = ""
    samplesperpixel: int = 0
    image_description: str = ""
    software: str = ""
    datetime: str = ""
    x_resolution: str = ""
    resolution_unit: str = ""
    pixel_size_text: str = ""
    file_size: int = 0
    file_mtime: str = ""

    @property
    def shape_text(self) -> str:
        return "x".join(str(v) for v in self.shape) if self.shape else "未知"

    @property
    def size_text(self) -> str:
        if self.nbytes <= 0:
            return "未知"
        for unit, div in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
            if self.nbytes >= div:
                return f"{self.nbytes / div:.2f} {unit}"
        return f"{self.nbytes} B"


_PIXEL_SIZE_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"(?i)pixel\s*(?:size|width|scale|spacing)\s*[:=]?\s*"
               r"([0-9]+(?:\.[0-9]+)?)\s*(nm|um|µm|pm|å|angstrom|[aA])\b"),
    re.compile(r"(?i)([0-9]+(?:\.[0-9]+)?)\s*(nm|um|µm|pm|å|angstrom|[aA])\s*"
               r"(?:/|per\s+)\s*(?:px|pixel|pixels|点)"),
    re.compile(r"(?i)\bscale\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(nm|um|µm|pm|å|angstrom)\b"),
)

_UNIT_ALIASES = {
    "um": "µm",
    "µm": "µm",
    "nm": "nm",
    "pm": "pm",
    "å": "Å",
    "angstrom": "Å",
    "a": "Å",
}

# 各长度单位到 nm 的换算系数（key 取小写）
_UNIT_TO_NM = {
    "nm": 1.0,
    "um": 1e3,
    "µm": 1e3,
    "pm": 1e-3,
    "å": 0.1,
    "angstrom": 0.1,
    "a": 0.1,
}


def pixel_size_from_description(text: str) -> str:
    """从 TIFF ImageDescription 中启发式提取像素尺寸，失败返回空串。

    这是**启发式**解析：不同厂商（Gatan/FEI/JEOL）描述格式差异很大，
    结果必须由用户核对。原始描述串会一并写入 CSV 头部，便于人工确认。
    """
    if not text:
        return ""
    for pattern in _PIXEL_SIZE_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        value = m.group(1)
        unit = _UNIT_ALIASES.get(m.group(2).lower(), m.group(2))
        return f"{value} {unit}/px"
    return ""


def pixel_size_nm(text: str) -> Optional[float]:
    """把 ``pixel_size_from_description`` 的输出（或原始描述串）换算为 nm/px。

    换算失败（无匹配/非法数值）返回 None；供 GUI 估算 ROI 物理面积，
    结果必须标注「基于启发式像素尺寸」并由用户核对。
    """
    if not text:
        return None
    for pattern in _PIXEL_SIZE_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        try:
            value = float(m.group(1))
        except ValueError:
            return None
        factor = _UNIT_TO_NM.get(m.group(2).lower())
        if factor is None or not np.isfinite(value) or value <= 0:
            return None
        return value * factor
    return None


def _compression_name(page) -> Tuple[str, bool]:
    """返回 (压缩方式名, 是否压缩)。

    压缩方式记录在页（TiffPage）上而不是 TiffFile 上，必须从首页读取。
    """
    if page is None:
        return "未知", False
    raw = getattr(page, "compression", None)
    if raw is None:
        return "无压缩", False
    name = getattr(raw, "name", None) or str(raw).split(".")[-1]
    name = str(name).upper()
    if name in ("NONE", "UNCOMPRESSED", "1", ""):
        return "无压缩", False
    return name, True


def _resolution_text(value) -> str:
    """把 TIFF 分辨率标签整理成文本；默认值（<=1）返回空串以免噪声。"""
    if value in ("", None):
        return ""
    try:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            den = float(value[1]) or 1.0
            val = float(value[0]) / den
        else:
            val = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(val) or val <= 1.0:
        return ""
    return f"{val:g}"


def probe_tiff(path: str) -> TiffProbe:
    """读取 TIFF 头部信息（元数据/布局/体积估计），不解码像素数据。

    Raises
    ------
    tifffile.TiffFileError
        文件不是 TIFF 时抛出（调用方需单独捕获，它不继承 ValueError）。
    OSError
        文件不存在或不可读。
    """
    file_size = 0
    file_mtime = ""
    try:
        st = os.stat(path)
        file_size = st.st_size
        file_mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
    except OSError:
        pass

    with tifffile.TiffFile(path) as tf:
        n_pages = 0
        try:
            n_pages = len(tf.pages)
        except Exception:
            n_pages = 0

        axes, shape, dtype_str, nbytes = "", (), "", 0
        try:
            if len(tf.series) > 0:
                s0 = tf.series[0]
                axes = getattr(s0, "axes", "") or ""
                shape = tuple(int(v) for v in s0.shape)
                dtype_str = str(np.dtype(s0.dtype))
                nbytes = int(np.prod(shape)) * int(np.dtype(s0.dtype).itemsize) if shape else 0
        except Exception:
            if n_pages:
                try:
                    shape = tuple(int(v) for v in tf.pages[0].shape)
                    dtype_str = str(tf.pages[0].dtype)
                    nbytes = int(np.prod(shape)) * int(tf.pages[0].dtype.itemsize)
                except Exception:
                    pass

        page = None
        try:
            if n_pages:
                page = tf.pages[0]
        except Exception:
            page = None

        def _tag(name: str, default=""):
            if page is None:
                return default
            try:
                return page.tags[name].value
            except Exception:
                return default

        compression, is_compressed = _compression_name(page)

        photometric = ""
        planarconfig = ""
        samplesperpixel = 0
        if page is not None:
            try:
                photometric = str(page.photometric).split(".")[-1]
            except Exception:
                photometric = ""
            try:
                planarconfig = str(page.planarconfig).split(".")[-1]
            except Exception:
                planarconfig = ""
            try:
                samplesperpixel = int(page.samplesperpixel or 0)
            except Exception:
                samplesperpixel = 0

        desc = str(_tag("ImageDescription", "") or "")
        software = str(_tag("Software", "") or "")
        dt = str(_tag("DateTime", "") or "")
        xres = _resolution_text(_tag("XResolution", ""))
        unit = str(_tag("ResolutionUnit", "") or "").split(".")[-1]

    return TiffProbe(
        axes=axes, shape=shape, dtype_str=dtype_str, nbytes=nbytes, n_pages=n_pages,
        compression=compression, is_compressed=is_compressed,
        photometric=photometric, planarconfig=planarconfig,
        samplesperpixel=samplesperpixel,
        image_description=desc, software=software, datetime=dt,
        x_resolution=xres, resolution_unit=unit if xres else "",
        pixel_size_text=pixel_size_from_description(desc),
        file_size=file_size, file_mtime=file_mtime,
    )


def format_metadata_lines(probe: Optional[TiffProbe], prefix: str = "# ") -> List[str]:
    """把探测到的采集元数据整理成 CSV 头部注释行。"""
    if probe is None:
        return []
    lines = [
        f"{prefix}TIFF 布局: axes={probe.axes or '未知'}, 页数={probe.n_pages}, "
        f"photometric={probe.photometric or '未知'}, planar={probe.planarconfig or '未知'}, "
        f"通道数={probe.samplesperpixel or 1}, 压缩={probe.compression}",
    ]
    if probe.software:
        lines.append(f"{prefix}采集软件: {probe.software}")
    if probe.datetime:
        lines.append(f"{prefix}采集时间(标签): {probe.datetime}")
    if probe.x_resolution or probe.resolution_unit:
        lines.append(f"{prefix}TIFF 分辨率标签: {probe.x_resolution or '未知'} "
                     f"{probe.resolution_unit or ''}（通常为打印分辨率，非物理标尺）")
    if probe.pixel_size_text:
        lines.append(f"{prefix}像素尺寸(启发式解析, 请核对): {probe.pixel_size_text}")
    if probe.image_description:
        desc = " ".join(probe.image_description.split())
        if len(desc) > 500:
            desc = desc[:500] + "…"
        lines.append(f"{prefix}ImageDescription: {desc}")
    if probe.file_size:
        lines.append(f"{prefix}源文件: {probe.file_size} 字节, 修改时间 {probe.file_mtime}")
    return lines


def page_datetimes(path: str, max_pages: int = 100_000) -> List[str]:
    """读取每页 DateTime 标签（tag 306），返回长度等于页数的字符串列表。

    任一页无标签则对应位置为空串；全部为空或读取失败时返回 ``[]``
    （调用方据此跳过 CSV 的逐帧时间列）。只解析 IFD 头部，不解码像素，
    供后台加载线程调用，把真实采集时刻随帧号一并导出。
    """
    try:
        with tifffile.TiffFile(path) as tf:
            out: List[str] = []
            for i, page in enumerate(tf.pages):
                if i >= max_pages:
                    break
                value = ""
                try:
                    tag = page.tags.get("DateTime")
                    if tag is not None and tag.value:
                        value = str(tag.value)
                except Exception:
                    value = ""
                out.append(value)
    except Exception:
        return []
    return out if any(out) else []


# ==================== 文件打开 ====================

def open_tiff_stack_ex(path: str, probe: Optional[TiffProbe] = None,
                       force_frames: bool = False
                       ) -> Tuple[np.ndarray, str, List[str]]:
    """打开 TIF 文件，返回 (规范化为 (T,H,W) 的数组, 加载方式, 处理说明)。

    优先尝试 tifffile.memmap 建立内存映射，大堆栈可流式访问；压缩
    TIFF（tifffile 抛 ValueError）回退为 tifffile.imread 整卷读入内存。
    其余 memmap 失败形态（网络盘/超大文件/非常规布局引发的 OSError、
    OverflowError 等）同样回退内存读取，避免落入「未知错误」。本项目
    对数组只做只读切片，两种加载方式返回的数组行为一致，仅加载方式
    说明不同。

    ``probe`` 可复用 :func:`probe_tiff` 的结果，避免重复解析头部。
    维度规范化规则同 :func:`normalize_tiff_array_ex`。
    """
    axes = probe.axes if probe is not None else ""
    if probe is None:
        try:
            axes = probe_tiff(path).axes
        except Exception:
            axes = ""

    try:
        raw = tifffile.memmap(path)
        mode = "内存映射"
    except ValueError:
        # 压缩 TIFF（LZW/Deflate/JPEG 等）不支持 memmap，整卷读入内存。
        raw = tifffile.imread(path)
        mode = "整卷载入（压缩 TIFF，已回退为内存读取）"
    except Exception:
        # 其余 memmap 失败形态（网络盘/非常规布局等）也回退内存读取。
        raw = tifffile.imread(path)
        mode = "整卷载入（memmap 不可用，已回退为内存读取）"

    try:
        arr, notes = normalize_tiff_array_ex(raw, axes=axes, force_frames=force_frames)
    except ValueError:
        del raw
        raise
    return arr, mode, notes


def open_tiff_stack(path: str, probe: Optional[TiffProbe] = None,
                    force_frames: bool = False) -> Tuple[np.ndarray, str]:
    """同 :func:`open_tiff_stack_ex`，但只返回 (数组, 加载方式)。"""
    arr, mode, _ = open_tiff_stack_ex(path, probe=probe, force_frames=force_frames)
    return arr, mode


# 显示灰度范围采样上限（仅影响显示，不参与衬度计算）
DISPLAY_CLIM_MAX_SAMPLES = 2_000_000
DISPLAY_CLIM_MAX_FRAMES = 64


def estimate_display_clim(arr: np.ndarray,
                          max_frames: int = DISPLAY_CLIM_MAX_FRAMES,
                          max_samples: int = DISPLAY_CLIM_MAX_SAMPLES
                          ) -> Tuple[float, float]:
    """采样估计全堆栈灰度范围（仅用于显示对比度，不参与衬度计算）。

    帧与像素均按步长采样，采样点总数受 ``max_samples`` 约束；NaN/Inf
    像素被忽略（先过滤非有限值，避免全 NaN 堆栈触发 ``All-NaN slice``
    RuntimeWarning）；退化（min==max）时展开半级，避免 matplotlib 警告。
    """
    if arr is None or arr.ndim != 3 or arr.size == 0:
        return (0.0, 1.0)
    n_frames, h, w = arr.shape
    step_f = max(1, n_frames // max(1, max_frames))
    n_sampled_frames = len(range(0, n_frames, step_f))
    per_frame = max(1, max_samples // max(1, n_sampled_frames))
    step = max(1, int(np.ceil(np.sqrt(max(1, h * w) / per_frame))))
    try:
        sub = np.asarray(arr[::step_f, ::step, ::step], dtype=np.float64)
    except Exception:
        return (0.0, 1.0)
    if sub.size == 0:
        return (0.0, 1.0)
    finite = sub[np.isfinite(sub)]
    if finite.size == 0:
        return (0.0, 1.0)
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmin >= vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    return (vmin, vmax)


def clamp_roi_to_image(x1: float, y1: float, x2: float, y2: float,
                       h: int, w: int) -> Tuple[int, int, int, int]:
    """把 ROI 顶点排序并 clamp 到图像坐标范围内。

    图像坐标约定与 imshow 一致：x 范围 [0, w]，y 范围 [0, h]，像素中心
    位于整数坐标。取整规则为**下界 floor、上界 ceil**，即 ROI 覆盖所有
    被矩形触及的像素；返回 ``(xmin, ymin, xmax, ymax)``，其中 xmax/ymax
    为开区间端点（对应 numpy 切片 ``img[ymin:ymax, xmin:xmax]``）。

    例：拖拽 (10.9, 10.9) -> (20.1, 20.1) 得到 (10, 10, 21, 21)，包含像素
    10 与 20（两者都被矩形部分覆盖）。
    """
    lo_x, hi_x = sorted([float(x1), float(x2)])
    lo_y, hi_y = sorted([float(y1), float(y2)])

    xmin = int(np.floor(lo_x))
    ymin = int(np.floor(lo_y))
    xmax = int(np.ceil(hi_x))
    ymax = int(np.ceil(hi_y))

    xmin = max(0, min(xmin, w))
    xmax = max(0, min(xmax, w))
    ymin = max(0, min(ymin, h))
    ymax = max(0, min(ymax, h))

    return xmin, ymin, xmax, ymax
