# -*- coding: utf-8 -*-
"""TIF 读写 —— 单帧为主，保位深、保 ImageJ 元数据、绝不覆盖输入。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tifffile

import export_io

try:  # imagecodecs 提供 LZW / JPEG / JPEG2000 等压缩编解码
    import imagecodecs  # noqa: F401

    HAS_IMAGECODECS = True
except ImportError:  # pragma: no cover - 环境相关
    HAS_IMAGECODECS = False

__all__ = ["ImageInfo", "inspect", "load", "save", "preview_clip",
           "make_output_path", "HAS_IMAGECODECS"]

#: 这些 dtype 有明确的整数饱和范围，保存时要裁剪
_INT_RANGE = {
    np.dtype(np.uint8): (0.0, 255.0),
    np.dtype(np.int8): (-128.0, 127.0),
    np.dtype(np.uint16): (0.0, 65535.0),
    np.dtype(np.int16): (-32768.0, 32767.0),
    np.dtype(np.uint32): (0.0, 4294967295.0),
    np.dtype(np.int32): (-2147483648.0, 2147483647.0),
}

#: 浮点源没有"整数量程"可裁：选"保持原位深"时也必须保持浮点，
#: 否则会被静默降成 uint8/uint16 并大面积裁剪。
_FLOAT_DTYPES = (np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64))

#: ImageJ 描述里值得带下去的标定字段（images/frames 等与单帧无关的键不带）
_IMAGEJ_KEEP_KEYS = ("unit", "spacing", "fps", "loop", "mode", "finterval")


@dataclass
class ImageInfo:
    path: str
    shape: Tuple[int, int]
    dtype: np.dtype
    n_pages: int
    description: str = ""
    is_imagej: bool = False
    #: 像素标定（X/Y 方向每单位多少像素）与单位（1=none, 2=inch, 3=cm）
    resolution: Optional[Tuple[float, float]] = None
    resolutionunit: Optional[int] = None
    #: 从源文件 ImageJ 描述里挑出来的标定字段，写回时原样带上
    imagej_metadata: Dict[str, Any] = field(default_factory=dict)
    #: 灰度语义（审查项09）：True 表示源是 MINISWHITE（0=白）。
    #: 像素值原样搬运，但输出必须保留同一标签，否则按标签语义显示的
    #: 阅读器会把图看成反相。
    miniswhite: bool = False

    @property
    def is_stack(self) -> bool:
        return self.n_pages > 1

    @property
    def bits(self) -> int:
        return int(self.dtype.itemsize * 8)

    def summary(self) -> str:
        base = f"{self.shape[1]}×{self.shape[0]}  {self.dtype}"
        if self.n_pages > 1:
            return f"{base}  共 {self.n_pages} 帧"
        return base


def _as_2d(arr: np.ndarray, *, planar_rgb: bool = False) -> np.ndarray:
    """把 (1,H,W) / (H,W,1) / RGB 之类的输入压成二维灰度。

    ``planar_rgb=True`` 表示输入是平面分离彩色 (S, H, W)：采样轴在**首维**。
    旧实现在末维分支里 ``a[..., 0]`` 会静默切掉宽度方向，得到 (S, W) 的
    错误图像（审查项08）。
    """
    a = np.asarray(arr)
    if planar_rgb and a.ndim == 3:
        a = np.moveaxis(a, 0, -1)          # (S,H,W) -> (H,W,S)
    if a.ndim == 3:
        if a.shape[0] == 1:
            a = a[0]
        elif a.shape[-1] == 1:
            a = a[..., 0]
        elif a.shape[-1] in (3, 4):
            # 按亮度权重合并，保持原来的整数 dtype
            rgb = a[..., :3].astype(np.float32)
            gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            if np.issubdtype(a.dtype, np.integer):
                info = np.iinfo(a.dtype)
                gray = np.clip(gray, info.min, info.max)
            a = gray.astype(a.dtype)
        else:
            raise ValueError(
                f"无法解释的图像形状 {np.shape(arr)}：三维输入只支持 "
                "(1,H,W)、(H,W,1) 与 RGB/RGBA（3/4 采样）。"
            )
    if a.ndim != 2:
        raise ValueError(f"无法解释的图像形状: {np.shape(arr)}")
    return a


def _read_resolution(page) -> Tuple[Optional[Tuple[float, float]], Optional[int]]:
    """读 IFD 里的像素标定，返回 ((x, y), unit)。缺失或非法时对应项为 None。"""
    def _one(tag_name: str) -> Optional[float]:
        tag = page.tags.get(tag_name)
        if tag is None:
            return None
        value = tag.value
        try:
            if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2:
                num, den = float(value[0]), float(value[1])
                return num / den if den else None
            return float(value)
        except (TypeError, ValueError, IndexError):
            return None

    x, y = _one("XResolution"), _one("YResolution")
    unit_tag = page.tags.get("ResolutionUnit")
    unit: Optional[int] = None
    if unit_tag is not None:
        try:
            unit = int(unit_tag.value)
        except (TypeError, ValueError):
            unit = None
    if x is None or y is None or x <= 0 or y <= 0:
        return None, unit
    return (x, y), unit


def _photometric_code(page: Any, default: int = 1) -> int:
    """取 photometric 的整数值。

    tifffile 返回的是 ``PHOTOMETRIC`` 枚举，MINISWHITE 的值为 **0**；
    写成 ``int(x) or default`` 会把合法零值替换成默认值。
    """
    value = getattr(page, "photometric", None)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _series_layout(tif) -> Tuple[str, int, bool]:
    """判定首个 series 的布局，返回 (kind, frame_count, is_planar_rgb)。

    kind 取值：
      ``yx``   —— 单帧二维灰度
      ``tyx``  —— 多帧栈（沿第一维取帧）
      ``yxs``  —— 单帧彩色（末维是 RGB/RGBA 采样）
      ``syx``  —— 平面分离彩色（首维是采样，误当帧数会毁数据）
    其余布局（多 series、四维体数据等）抛 ValueError，交由调用方给
    可理解的提示，而不是靠维数猜（审查项08）。
    """
    if len(tif.series) != 1:
        raise ValueError(
            f"该 TIFF 含 {len(tif.series)} 个 series，v1 只支持单 series 文件；"
            "请先在 Fiji 里拆分成单序列 TIFF。"
        )
    series = tif.series[0]
    axes = str(series.axes or "")
    shape = tuple(int(v) for v in series.shape)
    photometric = _photometric_code(tif.pages[0])

    if len(axes) != len(shape):        # axes 不可信时退回维数推断
        if len(shape) == 2:
            axes = "YX"
        elif len(shape) == 3:
            axes = "YXS" if photometric in (2, 6) else "TYX"
        else:
            raise ValueError(
                f"无法解释的 TIFF 维度 shape={shape}；v1 支持 YX / TYX / YXS / SYX。"
            )

    if axes == "YX":
        return "yx", 1, False
    if axes == "YXS":
        return "yxs", 1, False
    if axes == "SYX":
        return "syx", 1, True
    # T/Z/C 是明确的帧/深度/通道轴；Q 是 tifffile 对"未指定轴"的标准写法，
    # 普通多页栈（我们自己的录屏导出）就报 QYX，必须当作帧栈接受。
    if axes in ("TYX", "ZYX", "CYX", "QYX"):
        return "tyx", shape[0], False
    # 其余多轴组合（如 TYXS、TZCYX）明确拒绝而不是静默取切片
    raise ValueError(
        f"不支持的 TIFF 布局 axes={axes} shape={shape}；"
        "v1 支持 YX / TYX / YXS / SYX，请先在 Fiji 里拆分。"
    )


def inspect(path: str) -> ImageInfo:
    """只读文件头，取回尺寸/位深/帧数/标定元数据，不载入像素。"""
    with tifffile.TiffFile(path) as tif:
        kind, n, _planar = _series_layout(tif)
        page = tif.pages[0]
        if kind in ("yxs", "syx"):
            # 彩色：空间尺寸取后两轴
            series_shape = tuple(int(v) for v in tif.series[0].shape)
            if kind == "yxs":
                shape = (series_shape[0], series_shape[1])
            else:
                shape = (series_shape[1], series_shape[2])
        else:
            shape = tuple(int(v) for v in page.shape[:2])
        dtype = np.dtype(page.dtype)
        desc = str(page.description or "")
        is_ij = bool(getattr(tif, "is_imagej", False))
        resolution, resolutionunit = _read_resolution(page)
        # ImageJ 的 unit/spacing 等标定字段写在描述里，写回时必须带上，
        # 否则处理后的图在 Fiji 里量出来的物理尺度是错的。
        raw_ij = getattr(tif, "imagej_metadata", None) or {}
        imagej_metadata = {k: raw_ij[k] for k in _IMAGEJ_KEEP_KEYS if k in raw_ij}
        # 审查项09：MINISWHITE（0）的像素值语义与 MINISBLACK（1）相反，
        # 输出必须保留同一标签，否则按标签显示的阅读器看到的是反相图。
        # 注意不能写 ``int(...) or 1``：MINISWHITE 的枚举值就是 0，是合法
        # 取值却被 ``or`` 当成缺失而替换掉，导致该标签永远检测不到。
        miniswhite = _photometric_code(page) == 0
    return ImageInfo(path=path, shape=(shape[0], shape[1]), dtype=dtype,
                     n_pages=n, description=desc, is_imagej=is_ij,
                     resolution=resolution, resolutionunit=resolutionunit,
                     imagej_metadata=imagej_metadata, miniswhite=miniswhite)


def load(path: str, frame: int = 0) -> Tuple[np.ndarray, ImageInfo]:
    """载入一帧。frame 越界时抛 IndexError。

    帧号在入口按判定出的布局校验（审查项08）：单页 RGB 曾被误判成
    10 帧、选第 5 帧报"越界"，而平面分离 RGB (3,H,W) 曾被静默切成
    (3,W) 的错误图像。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    info = inspect(path)
    frame = int(frame)
    if not 0 <= frame < info.n_pages:
        raise IndexError(f"帧号 {frame} 超出范围（共 {info.n_pages} 帧，有效 0~{info.n_pages - 1}）")

    with tifffile.TiffFile(path) as tif:
        kind, _n, planar = _series_layout(tif)
        try:
            if kind == "tyx":
                arr = tif.series[0].asarray(key=frame)
            elif kind == "syx":
                arr = tif.series[0].asarray()      # (S, H, W) -> _as_2d 合并
            else:
                arr = tif.pages[0].asarray()       # yx / yxs 单页
        except (IndexError, KeyError) as exc:
            raise IndexError(
                f"帧号 {frame} 超出范围（共 {info.n_pages} 帧）"
            ) from exc

    img = _as_2d(arr, planar_rgb=(kind == "syx"))
    if img.shape != info.shape:
        info.shape = img.shape
    if img.dtype != info.dtype:
        info.dtype = img.dtype
    return np.ascontiguousarray(img), info


def make_output_path(path: str, suffix: str = "_deloc", ext: Optional[str] = None) -> str:
    """在源文件旁生成不冲突的输出路径，绝不返回源文件本身。"""
    if not suffix:
        raise ValueError("suffix 不能为空，否则会返回源文件本身")
    root, old_ext = os.path.splitext(path)
    ext = ext or old_ext or ".tif"
    candidate = f"{root}{suffix}{ext}"
    i = 2
    while os.path.exists(candidate) and os.path.abspath(candidate) != os.path.abspath(path):
        candidate = f"{root}{suffix}_{i}{ext}"
        i += 1
    return candidate


def _prepare_output(img: np.ndarray) -> np.ndarray:
    """写出前的输入校验。

    NaN/Inf 裁到整数位深会静默变成 0（或极值），属于无声的数据损坏，
    所以这里直接拒绝，而不是写出一张看起来正常的错图。
    """
    arr = np.asarray(img)
    if arr.size == 0:
        raise ValueError("拒绝写出空图像")
    if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
        raise ValueError("输出图像含 NaN/Inf，拒绝写盘")
    return arr


def _plan_output(
    arr: np.ndarray, source: Optional[ImageInfo], as_float32: bool
) -> Tuple[np.dtype, Optional[Tuple[float, float]]]:
    """决定写出的 dtype 及其整数饱和范围（范围 None 表示不需要裁剪）。"""
    if as_float32:
        return np.dtype(np.float32), None
    dtype = np.dtype(source.dtype if source is not None else arr.dtype)
    if dtype in _FLOAT_DTYPES:
        # 浮点源没有整数量程：保持浮点，别静默降位深
        return np.dtype(np.float32), None
    if dtype in _INT_RANGE:
        return dtype, _INT_RANGE[dtype]
    # bool / complex / int64 等罕见输入：退到装得下的整数位深
    fallback = np.dtype(np.uint8) if float(arr.max()) <= 255.0 else np.dtype(np.uint16)
    return fallback, _INT_RANGE[fallback]


def _assert_representable(arr: np.ndarray, dtype: np.dtype) -> None:
    """转换后复查可表示性（审查项02残留）。

    有限的 float64（如 1e40）在 astype(np.float32) 后会变成 Inf：写前
    校验在原数组上做，拦不住这种"转换即损坏"，报告 clipped_fraction=0
    却写出不可用的数据。
    """
    # errstate 抑制溢出警告：这里的溢出正是要被检出的情况，不是异常。
    with np.errstate(over="ignore", invalid="ignore"):
        converted = arr.astype(dtype)
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"数值超出 {dtype} 可表示范围（转换后为 Inf），拒绝写盘")


def preview_clip(
    img: np.ndarray, source: Optional[ImageInfo] = None, as_float32: bool = False
) -> Dict[str, Any]:
    """**不写盘**，只算出将要写出的 dtype 与裁剪比例，供 GUI 提前提示用户。

    与 save() 共用 _plan_output 与可表示性检查，所以这里提示的结论与实际
    落盘结果一致。
    """
    arr = _prepare_output(img)
    dtype, rng = _plan_output(arr, source, as_float32)
    if rng is None:
        _assert_representable(arr, dtype)
    src_dtype = np.dtype(source.dtype if source is not None else arr.dtype)
    clipped = 0.0
    if rng is not None:
        lo, hi = rng
        clipped = (int(np.count_nonzero(arr < lo))
                   + int(np.count_nonzero(arr > hi))) / float(arr.size)
    return {
        "dtype": str(dtype),
        "source_dtype": str(src_dtype),
        "dtype_changed": bool(dtype != src_dtype),
        "clipped_fraction": clipped,
    }


def save(
    path: str,
    img: np.ndarray,
    source: Optional[ImageInfo] = None,
    as_float32: bool = False,
    guard_path: Optional[str] = None,
) -> Dict[str, Any]:
    """写出结果。

    返回一份写盘报告，含是否发生裁剪、裁剪像素比例等，供 GUI 提示用户。
    想**在写盘前**就知道会不会裁剪，用 preview_clip()。

    源保护与原子替换统一走 export_io：normcase+realpath+samefile 三重
    判定拦住大小写变体（Windows 上 SourceCase.tif 与 sourcecase.tif 是
    同一个文件），临时文件在同目录内落盘后原子提交。
    """
    arr = _prepare_output(img)
    dtype, rng = _plan_output(arr, source, as_float32)
    src_dtype = np.dtype(source.dtype if source is not None else arr.dtype)
    report: Dict[str, Any] = {
        "path": path,
        "dtype": str(dtype),
        "source_dtype": str(src_dtype),
        "dtype_changed": bool(dtype != src_dtype),
        "clipped_fraction": 0.0,
    }

    if rng is None:
        _assert_representable(arr, dtype)
        out = arr.astype(dtype)
    else:
        lo, hi = rng
        below = int(np.count_nonzero(arr < lo))
        above = int(np.count_nonzero(arr > hi))
        report["clipped_fraction"] = (below + above) / float(arr.size)
        # 先四舍五入再裁剪：astype 是向零截断，100.7 会被存成 100，
        # 所有整数输出系统性偏低约半个灰阶，对定量分析不可接受。
        out = np.clip(np.rint(arr), lo, hi).astype(dtype)

    kwargs: Dict[str, Any] = {}
    if source is not None:
        # 标定元数据必须跟着结果走：丢了 XResolution/unit，处理后的图在
        # Fiji 里量晶格间距就会得到错的物理尺度。
        if source.resolution is not None:
            kwargs["resolution"] = tuple(float(v) for v in source.resolution)
        if source.resolutionunit in (1, 2, 3):
            kwargs["resolutionunit"] = int(source.resolutionunit)
        # 审查项09：灰度语义随结果保留。MINISWHITE 的 0=白，输出若标成
        # MINISBLACK，按标签语义显示的阅读器会把图看成反相。
        kwargs["photometric"] = "miniswhite" if source.miniswhite else "minisblack"
        if source.is_imagej:
            # imagej=True 对 float32 同样有效（tifffile 写 mode=grayscale），
            # 所以两条写出路径都能保住 unit/spacing。
            kwargs["imagej"] = True
            meta = dict(source.imagej_metadata)
            meta.setdefault("axes", "YX")
            kwargs["metadata"] = meta

    # 原子导出：先写同目录临时文件再替换，磁盘满/中途被杀不留半个 TIFF；
    # 提交前二次校验目标不是源文件。
    with export_io.atomic_path(path, guard_path) as tmp:
        tifffile.imwrite(tmp, out, **kwargs)
    report["shape"] = tuple(int(v) for v in out.shape)
    return report
