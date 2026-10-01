# -*- coding: utf-8 -*-
"""
TIFF 文件处理模块

负责 TIFF 堆栈文件的读取和写入：
- 支持单张和多页（堆栈）TIFF
- 大文件逐片读取（避免内存溢出）
- 标注图像保存
- 文件列表扫描
"""

import os
import re

import numpy as np
import tifffile
from typing import List, Optional, Tuple, Generator

from constants import SUPPORTED_EXTENSIONS, ANNOTATED_SUBDIR
from core.segmentation import resolve_uint16_shift


def _is_color_tail(shape) -> bool:
    """末维为 3/4 通道的单张彩色图（(H,W,3/4)）判定。

    与 load_tiff_slices/get_tiff_info 的 is_single_color 口径一致：
    2026-09-28 审查修正——预览路径的 get_slice_count 曾把 (H,W,3) 的
    单页 RGB 当作 H 个"切片"，每张实为一行 (W,3) 条带，预览/分析
    当前切片全部对垃圾数据出结果。
    注意 (N,H,W=3) 的病态灰度堆栈在数组层面无法与彩色区分，此时
    get_slice 会显式报 IndexError（宁可失败不可静默错数据）。
    """
    return len(shape) == 3 and shape[2] in (3, 4)


def scan_tiff_files(folder: str) -> List[str]:
    """
    扫描文件夹中的所有 TIFF 文件

    Args:
        folder: 目标文件夹路径

    Returns:
        排序后的 TIFF 文件名列表

    Raises:
        FileNotFoundError: 文件夹不存在
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"文件夹不存在: {folder}")

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(SUPPORTED_EXTENSIONS) and os.path.isfile(os.path.join(folder, f))
    ]
    files.sort(key=_natural_key)
    return files


def _natural_key(text: str):
    """自然排序键：把文件名中的数字段转为整数，实现 file2 < file10。"""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', text)]


def load_tiff_stack(file_path: str) -> np.ndarray:
    """
    加载 TIFF 文件（完整堆栈）

    注意：对于超大文件（>1GB），建议使用 load_tiff_slices 逐片读取。

    Args:
        file_path: TIFF 文件完整路径

    Returns:
        图像数组 (H, W) 或堆栈 (N, H, W)

    Raises:
        FileNotFoundError: 文件不存在
        IOError: 读取失败
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    try:
        stack = tifffile.imread(file_path)
        return stack
    except Exception as e:
        raise IOError(f"读取 TIFF 文件失败 [{file_path}]: {e}")


def load_tiff_slices(file_path: str) -> Generator[np.ndarray, None, None]:
    """
    逐片读取 TIFF 堆栈（生成器，节省内存）

    适用于超大文件，避免一次性加载到内存。
    - 多页 TIFF：每页 yield 一次。
    - 单页 3D 灰度堆叠：使用 tifffile.memmap 逐帧 yield，不整卷读入。
    - 单页 RGB：直接 yield 该页（由 segmentation 统一转灰度）。

    Args:
        file_path: TIFF 文件完整路径

    Yields:
        单张切片图像 (H, W) 或单页 RGB (H, W, 3/4)
    """
    with tifffile.TiffFile(file_path) as tif:
        if not tif.series:
            return

        series = tif.series[0]
        axes = getattr(series, "axes", "") or ""
        series_shape = tuple(getattr(series, "shape", ()))
        pages = tif.pages

        # 单页 3D：排除掉 (H,W,3/4) 单张彩色后，按灰度堆叠逐帧切片。
        is_single_color = (
            len(series_shape) == 3
            and series_shape[-1] in (3, 4)
            and ("C" in axes or "S" in axes)
        )
        if len(pages) == 1 and len(series_shape) == 3 and not is_single_color:
            page = pages[0]
            if page.ndim == 3 and page.shape[0] > 1:
                # 单页 3D 灰度堆叠：逐帧切片，避免 pages[0].asarray() 读入全量。
                try:
                    stack = tifffile.memmap(file_path)
                except Exception:
                    stack = page.asarray()
                try:
                    for i in range(stack.shape[0]):
                        yield stack[i]
                finally:
                    # 提前 break（用户停止批量）也能尽快释放 memmap 句柄；
                    # 已 yield 的切片视图会继续持有缓冲区，属预期。
                    del stack
                return

        for page in pages:
            yield page.asarray()


def scan_uint16_max_shift(files: List[str], folder: str) -> Tuple[Optional[int], List[str]]:
    """
    扫描一批 TIFF 文件，确定数据集级统一的 uint16 → uint8 右移位数。

    位深判定必须以"数据集"为单位（回归 2026-09-06 P1）：逐图按各自 max
    判定会让同一堆栈中明暗不同的切片使用不同映射，固定阈值跨切片语义
    漂移。这里对目录内所有 uint16 文件取全局最大值后统一解析。

    仅对头文件 dtype 为 uint16 的页做解码（uint8 文件零解码开销）。
    单个文件读取失败时跳过但**如实上报**（2026-09-28 审查修正：旧
    docstring 声称"返回 -1 计数由调用方告警"，实际既不计数也不告警，
    损坏文件被静默跳过）。

    Args:
        files: 文件名列表（相对 folder）
        folder: 输入目录

    Returns:
        (统一右移位数, 扫描失败的文件名列表)；
        目录中无 uint16 数据时移位数为 None（失败列表仍可能非空）。
    """
    global_max = 0.0
    has_uint16 = False
    failed: List[str] = []

    for filename in files:
        file_path = os.path.join(folder, filename)
        try:
            with tifffile.TiffFile(file_path) as tif:
                for page in tif.pages:
                    if page.dtype == np.uint16:
                        has_uint16 = True
                        global_max = max(global_max, float(page.asarray().max()))
        except Exception:
            # 单文件损坏不阻断位深扫描；该文件在正式处理时会再报错
            failed.append(filename)

    if not has_uint16:
        return None, failed
    return resolve_uint16_shift(global_max), failed


def get_tiff_info(file_path: str) -> dict:
    """
    获取 TIFF 文件基本信息（不加载像素数据）

    切片数按与 load_tiff_slices 相同的口径解析：
    - 多页 TIFF → 页数
    - 单页 3D 灰度堆叠（排除单张彩色）→ 第一维长度
    - 单页 2D/单张彩色 → 1

    Args:
        file_path: TIFF 文件路径

    Returns:
        包含 num_slices, slice_shape, dtype, file_size_mb 的字典
    """
    with tifffile.TiffFile(file_path) as tif:
        if not tif.series:
            return {
                'num_slices': 0,
                'slice_shape': (0, 0),
                'dtype': 'unknown',
                'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
            }

        series = tif.series[0]
        series_shape = tuple(getattr(series, "shape", ()))
        axes = getattr(series, "axes", "") or ""

        is_single_color = (
            len(series_shape) == 3
            and series_shape[-1] in (3, 4)
            and ("C" in axes or "S" in axes)
        )
        if len(tif.pages) == 1 and len(series_shape) == 3 and not is_single_color:
            num_slices = int(series_shape[0])
            slice_shape = series_shape[1:]
        else:
            num_slices = len(tif.pages)
            first = tif.pages[0] if tif.pages else None
            slice_shape = first.shape if first is not None else (0, 0)

        dtype_str = str(getattr(series, "dtype", "unknown"))

    return {
        'num_slices': num_slices,
        'slice_shape': slice_shape,
        'dtype': dtype_str,
        'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
    }


def get_slice_count(stack: np.ndarray) -> int:
    """
    获取堆栈切片数量

    口径与 load_tiff_slices 一致：
    - 2D 灰度 / 单张彩色 (H,W,3/4) → 1
    - 多页堆栈 (N,H,W) 或 (N,H,W,3/4) → N

    Args:
        stack: 图像数组

    Returns:
        切片数量（单张图像返回 1）
    """
    if stack.ndim <= 2:
        return 1
    if _is_color_tail(stack.shape):
        return 1
    return int(stack.shape[0])


def get_slice(stack: np.ndarray, index: int) -> np.ndarray:
    """
    获取堆栈中指定索引的切片

    Args:
        stack: 图像数组 (H,W)、单张彩色 (H,W,3/4)、堆栈 (N,H,W) 或
            多页彩色 (N,H,W,3/4)
        index: 切片索引（0-based）

    Returns:
        单张切片
    """
    if stack.ndim <= 2 or _is_color_tail(stack.shape):
        if index != 0:
            raise IndexError(f"单张图像只有切片 0，请求索引: {index}")
        return stack

    if 0 <= index < stack.shape[0]:
        return stack[index]
    raise IndexError(f"切片索引越界: {index}, 范围 [0, {stack.shape[0] - 1}]")


def save_annotated_image(
    image: np.ndarray,
    output_folder: str,
    filename: str,
) -> str:
    """
    保存标注图像为 TIFF 文件

    自动创建 annotated_tif 子目录。

    Args:
        image: 标注图像 (uint8 BGR 或灰度)
        output_folder: 输出根目录
        filename: 输出文件名（不含扩展名）

    Returns:
        保存的完整文件路径
    """
    annotated_dir = os.path.join(output_folder, ANNOTATED_SUBDIR)
    os.makedirs(annotated_dir, exist_ok=True)

    # 确保是 uint8
    if image.dtype != np.uint8:
        img_min, img_max = float(image.min()), float(image.max())
        if img_max > img_min:
            image = ((image.astype(np.float64) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # 清理文件名中的非法字符
    safe_name = "".join(c if c.isalnum() or c in '._- ' else '_' for c in filename)
    filepath = os.path.join(annotated_dir, f"{safe_name}.tif")

    tifffile.imwrite(filepath, image)
    return filepath
