# -*- coding: utf-8 -*-
"""安全、按帧读写 TIFF。

设计原则：宁可明确拒绝当前 GUI 无法无损表达的 TIFF，也绝不把其错误地
解释后再静默写出。支持一个 TIFF series 内的灰度、RGB/RGBA 图像及堆栈，
并按页流式处理，避免把整套堆栈复制到内存中。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Callable, Iterator

import numpy as np
import tifffile
from tifffile import COMPRESSION, PHOTOMETRIC
from version import APP_NAME, __version__

from image_filters import (
    SUPPORTED_INTEGER_DTYPES,
    default_params,
    from_float,
    is_default_params,
    process_image,
    to_float,
    validate_params,
)


class UnsupportedTiffError(ValueError):
    """输入结构可读取，但不在本工具可安全处理的范围内。"""


class ProcessingCancelled(RuntimeError):
    """由用户取消的处理任务。"""


ProgressCallback = Callable[[int, int], None]

_MANIFEST_SCHEMA_VERSION = 2

_LOSSY_COMPRESSION = {
    'OJPEG', 'JPEG', 'JPEG_99', 'ALT_JPEG', 'JPEG_LOSSY', 'JPEG_2000',
    'JPEG2000', 'JPEG_2000_LOSSY', 'JPEGXR', 'JPEGXR_NDPI', 'WEBP',
    'WEBP_DEPRECATED', 'JPEGXL', 'JPEGXL_DNG', 'JETRAW',
}
_COPY_TAG_CODES = {
    269, 271, 272, 285, 315, 33432,  # 文档、设备、页名、作者、版权
    33550, 33922, 34735, 34736, 34737,  # GeoTIFF
    42112, 42113,  # GDAL metadata / nodata
}
_HANDLED_TAG_CODES = {
    254, 256, 257, 258, 259, 262, 270, 273, 274, 277, 278, 279,
    282, 283, 284, 286, 287, 296, 305, 306, 317, 320, 322, 323,
    324, 325, 330, 338, 339, 34675, 50838, 50839,
}
# 子 IFD 指针（EXIF/GPS/Interop）：值是文件内偏移，复制到新文件必然失效，
# tifffile 写出端也会拒绝，这里显式跳过。
_POINTER_TAG_CODES = {34665, 34853, 40965}


def _as_enum(value, enum_type):
    """把标签值规范化为 tifffile 枚举成员。

    部分 writer（如 FEI 电镜）会省略 Compression 标签，tifffile 此时返回
    原始 ``int`` 而非枚举成员；按规范默认值所属枚举转换。无法识别时返回
    ``None``，由调用方决定是否拒绝。
    """
    if value is None or hasattr(value, 'name'):
        return value
    try:
        return enum_type(int(value))
    except (TypeError, ValueError):
        return None


def _serializable_tag_value(tag) -> bool:
    """判断标签值能否被 tifffile 忠实回写。

    ASCII（dtype=2）标签若被 tifffile 解析成 dict（如 FEI_HELIOS 专有元
    数据）或含非 7-bit 字符，写出端会直接抛错或产生变体编码；这类值一律
    视为不可序列化，由调用方跳过并靠清单记录。
    """
    if int(tag.dtype) != 2:
        return True
    value = tag.value
    if isinstance(value, bytes):
        return True
    if not isinstance(value, str):
        return False
    try:
        value.encode('ascii')
    except UnicodeEncodeError:
        return False
    return True


def paths_equivalent(path_a: str | os.PathLike[str], path_b: str | os.PathLike[str]) -> bool:
    """在 Windows 大小写、链接和既有文件条件下正确比较两个路径。"""
    a = os.fspath(path_a)
    b = os.fspath(path_b)
    try:
        if os.path.exists(a) and os.path.exists(b):
            return os.path.samefile(a, b)
    except OSError:
        pass
    return os.path.normcase(os.path.realpath(os.path.abspath(a))) == os.path.normcase(
        os.path.realpath(os.path.abspath(b))
    )


def file_sha256(path: str | os.PathLike[str], chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """流式计算文件 SHA-256；大堆栈也不会整体载入内存。"""
    digest = hashlib.sha256()
    with open(os.fspath(path), 'rb') as stream:
        for block in iter(lambda: stream.read(chunk_bytes), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_output_path(output_dir: str | os.PathLike[str], src_path: str | os.PathLike[str]) -> str:
    """返回永不覆盖输入或既有文件的输出路径。"""
    out_dir = os.path.abspath(os.fspath(output_dir))
    source = os.path.abspath(os.fspath(src_path))
    stem, suffix = os.path.splitext(os.path.basename(source))
    candidate = os.path.join(out_dir, os.path.basename(source))
    if (paths_equivalent(candidate, source) or os.path.exists(candidate)
            or os.path.exists(f'{candidate}.filter.json')):
        candidate = os.path.join(out_dir, f'{stem}_processed{suffix}')
    number = 1
    while (os.path.exists(candidate) or paths_equivalent(candidate, source)
           or os.path.exists(f'{candidate}.filter.json')):
        candidate = os.path.join(out_dir, f'{stem}_processed_{number}{suffix}')
        number += 1
    return candidate


def _natural_key(name: str) -> list[object]:
    return [int(part) if part.isdigit() else part.casefold()
            for part in re.split(r'(\d+)', name)]


def list_tif_files(folder: str | os.PathLike[str]) -> list[str]:
    """列出当前目录的 TIFF 文件，采用自然排序且不跟随目录项。"""
    root = os.fspath(folder)
    entries = [entry for entry in os.scandir(root)
               if entry.is_file() and entry.name.lower().endswith(('.tif', '.tiff'))]
    return [entry.path for entry in sorted(entries, key=lambda item: _natural_key(item.name))]


def is_tif_file(path: str | os.PathLike[str]) -> bool:
    return os.fspath(path).lower().endswith(('.tif', '.tiff'))


def display_levels(frame_float: np.ndarray) -> tuple[float, float]:
    """为显示计算稳健黑白点；仅影响预览，绝不影响保存数值。"""
    values = np.asarray(frame_float)
    if values.ndim == 3:
        values = values[:, :, :3]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, (0.5, 99.5))
    if high - low < 1e-8:
        low, high = float(finite.min()), float(finite.max())
    if high - low < 1e-8:
        return 0.0, 1.0
    return float(low), float(high)


def to_display(frame_float: np.ndarray, levels: tuple[float, float] | None = None) -> np.ndarray:
    """把处理空间浮点帧映射为 uint8 预览。"""
    low, high = display_levels(frame_float) if levels is None else levels
    arr = (np.asarray(frame_float) - low) / (high - low)
    return (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


class TifDocument:
    """一个可按帧读取与写出的、单 series TIFF 文档。

    支持的像素类型为 uint8、uint16、uint32；支持灰度、RGB、RGBA，以及
    任意数量的前导帧维度（如 Z/T/Q）。帧序沿用 tifffile series 的页序
    （等价于 axes 声明 shape 的 C 序展开），本类不单独重排。不支持
    的格式会在加载阶段失败，防止错误输出覆盖原始科研数据。
    MINISWHITE（0=白）图像在读取时按亮度反转，使预览与滤镜方向与视觉
    一致，写出时再反转并保留原 photometric。
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = os.path.abspath(os.fspath(path))
        self.name = os.path.basename(self.path)
        self._read_header()

    def _read_header(self) -> None:
        try:
            with tifffile.TiffFile(self.path) as tif:
                if len(tif.series) != 1:
                    raise UnsupportedTiffError(
                        f'检测到 {len(tif.series)} 个 TIFF series。为避免丢失缩略图或附加图像，'
                        '当前版本拒绝处理多 series 文件。'
                    )
                series = tif.series[0]
                if not series.pages:
                    raise UnsupportedTiffError('TIFF 不含可读取页面。')
                self.axes = str(series.axes)
                self.shape = tuple(int(v) for v in series.shape)
                self.dtype = np.dtype(series.dtype)
                self.is_bigtiff = bool(tif.is_bigtiff)
                self.byteorder = tif.byteorder
                self._first_page = series.pages[0]
                self._configure_layout(series)
                self._capture_write_options(self._first_page)
        except tifffile.TiffFileError as exc:
            raise UnsupportedTiffError(f'无法读取 TIFF：{exc}') from exc

    def _configure_layout(self, series: tifffile.TiffPageSeries) -> None:
        if self.dtype.type not in SUPPORTED_INTEGER_DTYPES:
            raise UnsupportedTiffError(
                f'不支持 dtype={self.dtype}。为保证数值不被隐式归一化或截断，'
                '当前仅支持 uint8、uint16、uint32。'
            )
        if len(self.axes) != len(self.shape) or self.axes.count('Y') != 1 or self.axes.count('X') != 1:
            raise UnsupportedTiffError(f'无法识别空间轴：axes={self.axes}, shape={self.shape}')

        s_positions = [i for i, axis in enumerate(self.axes) if axis == 'S']
        c_positions = [i for i, axis in enumerate(self.axes) if axis == 'C']
        if len(s_positions) > 1 or len(c_positions) > 1:
            raise UnsupportedTiffError(f'无法识别通道布局：axes={self.axes}')
        if c_positions and not s_positions:
            # 通道按页存储（如 TCZYX）：单页不含全部样本，无法按帧流式处理。
            raise UnsupportedTiffError(
                f'通道按页存储的 TIFF（axes={self.axes}）暂不支持；'
                '请先转换为页内通道（YXS）布局。'
            )
        # OME-TIFF 的 RGB 图像 axes 形如 CYXS：C 是 series 级分页通道维，
        # S 才是页内样本数。通道轴取 S，C 自然归入前导帧维逐帧处理。
        self.channel_axis = s_positions[0] if s_positions else None
        self.channel_count = self.shape[self.channel_axis] if self.channel_axis is not None else 1
        if self.channel_count not in (1, 3, 4):
            raise UnsupportedTiffError(
                f'仅支持灰度、RGB 或 RGBA，实际通道数为 {self.channel_count}。'
            )
        if self.channel_axis is not None and self.channel_count == 1:
            raise UnsupportedTiffError('带单通道 C/S 轴的 TIFF 暂不支持；请先转换为 YX 灰度 TIFF。')

        self.y_axis = self.axes.index('Y')
        self.x_axis = self.axes.index('X')
        structural_axes = {self.y_axis, self.x_axis}
        if self.channel_axis is not None:
            structural_axes.add(self.channel_axis)
        self.leading_indices = [i for i in range(len(self.axes)) if i not in structural_axes]
        self.leading_shape = tuple(self.shape[i] for i in self.leading_indices)
        self.leading_axes = ''.join(self.axes[i] for i in self.leading_indices)
        self.num_frames = math.prod(self.leading_shape) if self.leading_shape else 1
        self.is_stack = self.num_frames > 1
        self.frame_axes = ''.join(self.axes[i] for i in range(len(self.axes)) if i in structural_axes)
        self.frame_shape = tuple(self.shape[i] for i in range(len(self.axes)) if i in structural_axes)
        self._display_order = [self.frame_axes.index('Y'), self.frame_axes.index('X')]
        if self.channel_axis is not None:
            self._display_order.append(self.frame_axes.index(self.axes[self.channel_axis]))
        self._inverse_display_order = tuple(np.argsort(self._display_order))
        page_count = len(series.pages)
        if page_count != self.num_frames:
            raise UnsupportedTiffError(
                f'该 TIFF 的 page 数 ({page_count}) 与帧数 ({self.num_frames}) 不一致，'
                '不能安全按帧流式处理。'
            )
        first_page_shape = tuple(int(v) for v in self._first_page.shape)
        if first_page_shape != self.frame_shape:
            # 只读标签不读像素，零开销；把布局不符提前到加载期报错，
            # 避免读到中段才发现每页内容与 axes 声明对不上。
            raise UnsupportedTiffError(
                f'首页 shape={first_page_shape} 与由 axes={self.axes} 推导的帧 '
                f'{self.frame_shape} 不一致，不能安全按帧流式处理。'
            )

        photometric = _as_enum(self._first_page.photometric, PHOTOMETRIC)
        compression = _as_enum(self._first_page.compression, COMPRESSION)
        if photometric is None:
            raise UnsupportedTiffError(
                f'无法识别 photometric={self._first_page.photometric!r}，拒绝处理。')
        if compression is None:
            raise UnsupportedTiffError(
                f'无法识别 compression={self._first_page.compression!r}，拒绝处理。')
        photometric_name = photometric.name
        compression_name = compression.name
        if self.channel_axis is None and photometric_name not in {'MINISBLACK', 'MINISWHITE'}:
            raise UnsupportedTiffError(f'不支持灰度 photometric={photometric_name}。')
        if self.channel_axis is not None and photometric_name != 'RGB':
            raise UnsupportedTiffError(f'不支持彩色 photometric={photometric_name}。')
        if compression_name in _LOSSY_COMPRESSION:
            raise UnsupportedTiffError(
                f'为避免再次有损编码，当前拒绝 compression={compression_name} 的 TIFF。'
            )

    def _capture_write_options(self, page: tifffile.TiffPage) -> None:
        photometric = _as_enum(page.photometric, PHOTOMETRIC)
        compression = _as_enum(page.compression, COMPRESSION)
        # 布局校验已确保枚举可识别；此处兜底保留原值以便诊断。
        self.photometric = page.photometric if photometric is None else photometric
        self.compression = page.compression if compression is None else compression
        self.planarconfig = page.planarconfig
        self.extrasamples = tuple(page.extrasamples or ())
        self.predictor = page.predictor
        self.bitspersample = page.bitspersample
        self.rowsperstrip = page.rowsperstrip
        self.tile = (page.tilelength, page.tilewidth) if page.is_tiled else None
        self.resolution = self._resolution_from_page(page)
        self.resolutionunit = page.resolutionunit
        self.description = page.description or None
        self.datetime = page.tags.get(306).value if page.tags.get(306) else None
        self.software = page.tags.get(305).value if page.tags.get(305) else None
        icc_tag = page.tags.get(34675)
        self.iccprofile = icc_tag.value if icc_tag else None
        self.extratags = self._copyable_extratags(page)
        # tifffile decodes IJMetadata (a BYTE tag) into a dict. It cannot be
        # passed directly to struct.pack as ordinary BYTE extratag values.
        # Rebuild the payload and its byte counts together in the output's
        # byte order, preserving ImageJ Info, frame Labels, LUTs, and ROIs.
        imagej_tag = page.tags.get(50839)
        if imagej_tag is not None:
            if not isinstance(imagej_tag.value, dict):
                raise UnsupportedTiffError('无法解析 ImageJ 二进制元数据，拒绝丢失后保存。')
            self.extratags.extend(tifffile.imagej_metadata_tag(imagej_tag.value, self.byteorder))
        self.original_tag_codes = sorted(int(tag.code) for tag in page.tags.values())

    @staticmethod
    def _resolution_from_page(page: tifffile.TiffPage) -> tuple[tuple[int, int], tuple[int, int]] | None:
        x_tag = page.tags.get(282)
        y_tag = page.tags.get(283)
        if x_tag is None or y_tag is None:
            return None
        return x_tag.value, y_tag.value

    @staticmethod
    def _copyable_extratags(page: tifffile.TiffPage) -> list[tuple[int, object, int, object, bool]]:
        extras: list[tuple[int, object, int, object, bool]] = []
        for tag in page.tags.values():
            if tag.code in _HANDLED_TAG_CODES or tag.code in _POINTER_TAG_CODES:
                continue
            if tag.code not in _COPY_TAG_CODES and tag.code < 32768:
                continue
            if not _serializable_tag_value(tag):
                continue
            try:
                extras.append((int(tag.code), tag.dtype, int(tag.count), tag.value, False))
            except (TypeError, ValueError):
                # 不能安全序列化的标签不会伪装为已保留；清单会记录原始标签列表。
                continue
        return extras

    def _read_raw_frame(self, index: int) -> np.ndarray:
        if not 0 <= index < self.num_frames:
            raise IndexError(f'帧索引超出范围: {index}')
        with tifffile.TiffFile(self.path) as tif:
            page = tif.series[0].pages[index]
            self._validate_page(page, index)
            frame = page.asarray()
        self._validate_frame(frame, index)
        return frame

    def _iter_raw_frames(self, cancel_event: Event | None = None) -> Iterator[np.ndarray]:
        """在单个 TiffFile 上下文中流式读取全部帧，避免每帧重复打开文件。"""
        with tifffile.TiffFile(self.path) as tif:
            pages = tif.series[0].pages
            for index in range(self.num_frames):
                if cancel_event is not None and cancel_event.is_set():
                    raise ProcessingCancelled('处理已取消。')
                page = pages[index]
                self._validate_page(page, index)
                frame = page.asarray()
                self._validate_frame(frame, index)
                yield frame

    def _validate_page(self, page: tifffile.TiffPage, index: int) -> None:
        # OME 等 series 的非首页是 TiffFrame 轻量对象，标签属性在 keyframe 上。
        page = getattr(page, 'keyframe', page)
        page_photometric = _as_enum(page.photometric, PHOTOMETRIC)
        page_compression = _as_enum(page.compression, COMPRESSION)
        if page_photometric != self.photometric or page_compression != self.compression:
            raise UnsupportedTiffError(
                f'第 {index + 1} 页的 photometric/compression 与首页不一致，拒绝静默改写。'
            )
        if page.bitspersample != self.bitspersample:
            raise UnsupportedTiffError(
                f'第 {index + 1} 页的 bitspersample 与首页不一致，拒绝静默改写。'
            )
        if page.planarconfig != self.planarconfig:
            raise UnsupportedTiffError(
                f'第 {index + 1} 页的 planarconfig 与首页不一致，拒绝静默改写。'
            )
        if tuple(page.extrasamples or ()) != self.extrasamples:
            raise UnsupportedTiffError(
                f'第 {index + 1} 页的 extrasamples 与首页不一致，拒绝静默改写。'
            )

    def _validate_frame(self, frame: np.ndarray, index: int) -> None:
        if tuple(frame.shape) != self.frame_shape:
            raise UnsupportedTiffError(
                f'第 {index + 1} 帧 shape={frame.shape} 与预期 {self.frame_shape} 不一致。'
            )
        if frame.dtype != self.dtype:
            raise UnsupportedTiffError(f'第 {index + 1} 帧 dtype={frame.dtype} 与文件不一致。')

    def _to_processing_frame(self, raw_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        oriented = np.transpose(raw_frame, self._display_order)
        if self.channel_axis is None:
            work = to_float(oriented)
            if self.photometric.name == 'MINISWHITE':
                # 0=白：处理空间统一按亮度（大=亮），显示与滤镜方向才符合直觉。
                work = 1.0 - work
            return work, None
        rgb = oriented[:, :, :3]
        alpha = oriented[:, :, 3].copy() if self.channel_count == 4 else None
        return to_float(rgb), alpha

    def _from_processing_frame(self, processed: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
        if self.photometric.name == 'MINISWHITE':
            processed = 1.0 - processed
        native = from_float(processed, self.dtype)
        if self.channel_count == 4:
            if alpha is None:
                raise RuntimeError('RGBA 图像缺少 Alpha 通道。')
            native = np.dstack([native, alpha])
        return np.transpose(native, self._inverse_display_order)

    def get_frame_float(self, index: int) -> np.ndarray:
        frame, _ = self._to_processing_frame(self._read_raw_frame(index))
        return frame

    def peak_bytes(self) -> int:
        """单帧处理链路的粗略内存峰值估算（字节）。

        保存链路：原始帧 + 处理空间浮点副本 + 滤镜中间量 + 量化输出；
        预览链路：整帧浮点 + 抗混叠模糊副本 + 降采样结果。取两者较大者。
        """
        pixels = int(np.prod(self.frame_shape))
        itemsize = self.dtype.itemsize
        work = 8 if self.dtype == np.uint32 else 4
        return max(
            pixels * (itemsize + work * 3 + itemsize),   # 保存
            pixels * work * 3 + work,                    # 预览（整帧载入是主要压力）
        )

    def get_display_levels(self, index: int) -> tuple[float, float]:
        return display_levels(self.get_frame_float(index))

    def get_original_display(self, index: int, levels: tuple[float, float] | None = None) -> np.ndarray:
        return to_display(self.get_frame_float(index), levels)

    def get_processed_display(
        self, index: int, params: dict[str, float], levels: tuple[float, float] | None = None
    ) -> np.ndarray:
        return to_display(process_image(self.get_frame_float(index), params), levels)

    def _write_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            'photometric': self.photometric,
            'planarconfig': self.planarconfig,
            'compression': self.compression,
            'metadata': None,
            'description': self.description,
            'resolution': self.resolution,
            'resolutionunit': self.resolutionunit,
            'datetime': self.datetime,
            'software': self.software,
            'extratags': self.extratags or None,
        }
        if self.extrasamples:
            kwargs['extrasamples'] = self.extrasamples
        predictor_name = getattr(self.predictor, 'name', 'NONE' if self.predictor in (0, 1, None) else str(self.predictor))
        if predictor_name != 'NONE':
            kwargs['predictor'] = self.predictor
        if self.bitspersample:
            kwargs['bitspersample'] = self.bitspersample
        if self.iccprofile:
            kwargs['iccprofile'] = self.iccprofile
        if self.tile is not None:
            kwargs['tile'] = self.tile
        elif self.rowsperstrip:
            kwargs['rowsperstrip'] = self.rowsperstrip
        return kwargs

    def _processed_frames(
        self,
        params: dict[str, float],
        progress_callback: ProgressCallback | None,
        cancel_event: Event | None,
    ) -> Iterator[np.ndarray]:
        no_changes = is_default_params(params)
        for index, raw in enumerate(self._iter_raw_frames(cancel_event)):
            if no_changes:
                output = raw
            else:
                frame, alpha = self._to_processing_frame(raw)
                output = self._from_processing_frame(process_image(frame, params), alpha)
            if progress_callback is not None:
                progress_callback(index + 1, self.num_frames)
            yield output

    def save_processed(
        self,
        out_path: str | os.PathLike[str],
        params: dict[str, float] | None = None,
        progress_callback: ProgressCallback | None = None,
        cancel_event: Event | None = None,
        write_manifest: bool = True,
    ) -> tuple[str, str | None]:
        """流式处理并原子写出；绝不允许目标与源文件等价。"""
        values = validate_params(params)
        started = time.monotonic()
        out = os.path.abspath(os.fspath(out_path))
        if paths_equivalent(out, self.path):
            raise ValueError('拒绝覆盖输入文件。请选择独立输出目录。')
        out_dir = os.path.dirname(out)
        if not out_dir:
            raise ValueError('输出路径必须包含目录。')
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(out):
            raise FileExistsError(f'输出文件已存在，拒绝覆盖：{out}')
        if write_manifest and os.path.exists(f'{out}.filter.json'):
            raise FileExistsError(f'参数清单已存在，拒绝覆盖：{out}.filter.json')

        suffix = Path(out).suffix or '.tif'
        descriptor, temporary = tempfile.mkstemp(prefix=f'.{Path(out).stem}.', suffix=suffix, dir=out_dir)
        os.close(descriptor)
        manifest_path = f'{out}.filter.json' if write_manifest else None
        manifest_temporary: str | None = None
        try:
            with tifffile.TiffWriter(temporary, bigtiff=self.is_bigtiff,
                                     byteorder=self.byteorder) as writer:
                # TiffWriter 的 generator API 面向单页内的编码块，而不是“每次 yield
                # 一帧”。逐页 write 才能真正流式处理大型堆栈。
                options = self._write_kwargs()
                for index, frame in enumerate(
                    self._processed_frames(values, progress_callback, cancel_event)
                ):
                    page_options = dict(options)
                    # ImageDescription 是系列级信息；重复写入会让某些阅读器误判为
                    # 多 series。自定义描述保留在首页，完整 axes 同时写入 sidecar。
                    if index:
                        page_options['description'] = None
                        page_options['extratags'] = None
                    writer.write(frame, **page_options)
            if cancel_event is not None and cancel_event.is_set():
                raise ProcessingCancelled('处理已取消。')
            # 清单先写成临时文件，再与 TIFF 一起提交：避免“TIFF 已落盘而清单失败”
            # 造成半成品被误报为成功，或重跑被已存在的输出卡住。
            if manifest_path is not None:
                # TIFF 此刻仍是临时文件（尚未 replace）：体积以临时文件为准，
                # 内容与最终输出逐字节一致。
                try:
                    output_size: int | None = os.stat(temporary).st_size
                except OSError:
                    output_size = None
                descriptor, manifest_temporary = tempfile.mkstemp(
                    prefix=f'.{Path(manifest_path).name}.', suffix='.tmp', dir=out_dir)
                with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as handle:
                    json.dump(
                        self._manifest_payload(out, values, time.monotonic() - started,
                                               output_size),
                        handle, ensure_ascii=False, indent=2)
                    handle.write('\n')
            os.replace(temporary, out)
            temporary = None
            if manifest_temporary is not None:
                try:
                    os.replace(manifest_temporary, manifest_path)
                    manifest_temporary = None
                except Exception:
                    # 清单提交失败时回滚刚写出的 TIFF，避免留下不完整的一对产物。
                    try:
                        os.unlink(out)
                    except OSError:
                        pass
                    raise
        except Exception:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
            if manifest_temporary is not None:
                try:
                    os.unlink(manifest_temporary)
                except FileNotFoundError:
                    pass
            raise

        return out, manifest_path

    def _manifest_payload(
        self, output_path: str, params: dict[str, float], duration_seconds: float,
        output_size_bytes: int | None = None,
    ) -> dict:
        source_stat = os.stat(self.path)
        # 首页中既不会被写出端重建、也未复制进输出的标签：这是元数据
        # 丢失的显式明细，不再要求使用者对两个码表做集合差。
        copied_codes = {tag[0] for tag in self.extratags}
        skipped_codes = sorted(
            set(self.original_tag_codes) - _HANDLED_TAG_CODES - copied_codes)
        return {
            'schema_version': _MANIFEST_SCHEMA_VERSION,
            'application': {'name': APP_NAME, 'version': __version__},
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'duration_seconds': round(max(duration_seconds, 0.0), 3),
            'source': {
                'path': self.path,
                'size_bytes': source_stat.st_size,
                'modified_utc': datetime.fromtimestamp(source_stat.st_mtime, timezone.utc).isoformat(),
                'sha256': file_sha256(self.path),
            },
            'output_path': output_path,
            'output_size_bytes': output_size_bytes,
            'filters': params,
            'tiff': {
                'shape': self.shape,
                'axes': self.axes,
                'dtype': self.dtype.name,
                'frames': self.num_frames,
                'photometric': self.photometric.name,
                'compression': self.compression.name,
                'original_tag_codes': self.original_tag_codes,
                'copied_extra_tag_codes': sorted(copied_codes),
                'skipped_tag_codes': skipped_codes,
            },
            'integrity_note': (
                '默认参数时像素按原始 dtype/布局逐帧转写；元数据按支持项保留，'
                '每页级标签仅保留首页副本，skipped_tag_codes 列出未复制的标签。'
                '滤镜结果经 float64 rint 量化回原始 dtype，单像素与浮点期望值'
                '最多相差 1 个最低位。source.sha256 可验证源文件自处理以来未被'
                '替换。请保留原始采集文件，输出为处理派生文件。'
            ),
        }
