"""
Readers for image stacks and frame sequences.

Supports multi-page TIFF stacks (e.g. Fiji/ImageJ "Substack" exports) and
directories of numbered single-frame TIFFs. Frames are yielded lazily so a
thousand-frame stack never needs to fit in memory.

ImageJ calibration metadata (unit/spacing) is surfaced when present so batch
processing can pick up the physical pixel size without manual input.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import numpy as np

__all__ = [
    'StackInfo',
    'read_stack_info',
    'iter_tiff_frames',
    'discover_image_sequence',
    'iter_sequence_frames',
    'open_frame_source',
]


@dataclass
class StackInfo:
    """Metadata for a stack or sequence without loading the pixel data."""

    n_frames: int
    height: int
    width: int
    dtype: str
    pixel_size: Optional[float] = None  # nm per pixel when calibration exists
    fps: Optional[float] = None
    source_path: str = ''
    frame_indices: List[int] = field(default_factory=list)
    kind: str = 'stack'  # 'stack' or 'sequence'


def read_stack_info(filepath: str) -> StackInfo:
    """Read metadata of a multi-page TIFF stack (ImageJ/Fiji exports).

    Plain multi-page TIFFs store one series per page, so the frame count is
    taken from the page count; ImageJ/OME files carry a single 3D series and
    are handled through the same page/shape logic.
    """
    import tifffile

    with tifffile.TiffFile(filepath) as tif:
        if not tif.pages:
            raise ValueError(f"TIFF file contains no pages: {filepath}")
        if len(tif.pages) == 1 and len(tif.series) == 1 \
                and len(tif.series[0].shape) >= 3:
            series = tif.series[0]
            n_frames = int(series.shape[0])
            height, width = int(series.shape[-2]), int(series.shape[-1])
            dtype = str(series.dtype)
        else:
            first = tif.pages[0]
            n_frames = len(tif.pages)
            height, width = int(first.shape[-2]), int(first.shape[-1])
            dtype = str(first.dtype)
        if n_frames < 2:
            raise ValueError(
                f"Not a multi-frame stack (frames={n_frames}); "
                "use the normal 打开图像 workflow for single frames."
            )
        metadata = tif.imagej_metadata or {}

    pixel_size = None
    spacing = metadata.get('spacing')
    unit = str(metadata.get('unit', '')).lower()
    if spacing and np.isfinite(float(spacing)) and float(spacing) > 0:
        spacing = float(spacing)
        factors = {'nm': 1.0, 'um': 1000.0, 'µm': 1000.0,
                   'mm': 1.0e6, 'angstrom': 0.1, 'å': 0.1, 'cm': 1.0e7,
                   'm': 1.0e9, 'micron': 1000.0, 'microns': 1000.0}
        pixel_size = spacing * factors.get(unit, 1.0) if unit not in ('', 'pixel') else None

    fps = metadata.get('fps')
    fps = float(fps) if fps else None

    return StackInfo(
        n_frames=n_frames,
        height=height,
        width=width,
        dtype=dtype,
        pixel_size=pixel_size,
        fps=fps,
        source_path=filepath,
        kind='stack',
    )


def _series_frames(series) -> Iterator[Tuple[int, np.ndarray]]:
    """Decode a single-page multi-dimensional TIFF series exactly once.

    Single-page 3D series (some OME exports) cannot be read plane-by-plane
    through ``pages``, so the whole series is decoded up front and frames are
    then yielded as cheap views — decoding per frame inside the loop would be
    O(n²) time and would peak at the full stack per iteration.
    """
    stack = series.asarray()
    for index in range(int(series.shape[0])):
        yield index, stack[index]


def iter_tiff_frames(filepath: str, start: int = 0, stop: Optional[int] = None,
                     step: int = 1) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(frame_index, image)`` lazily from a multi-page TIFF.

    ``frame_index`` is the position inside the full stack (0-based) so the
    caller can always report the true source frame even with sub-sampling.
    """
    import tifffile

    if step < 1:
        raise ValueError(f"frame step must be >= 1, got {step}")
    with tifffile.TiffFile(filepath) as tif:
        if not tif.pages:
            raise ValueError(f"TIFF file contains no pages: {filepath}")
        single_series_3d = len(tif.pages) == 1 and len(tif.series) == 1 \
            and len(tif.series[0].shape) >= 3
        n_frames = (
            int(tif.series[0].shape[0]) if single_series_3d else len(tif.pages)
        )
        begin = max(0, min(n_frames, start))
        end = n_frames if stop is None else max(begin, min(n_frames, stop))
        if single_series_3d:
            for index, image in _series_frames(tif.series[0]):
                if index < begin:
                    continue
                if index >= end:
                    break
                if (index - begin) % step:
                    continue
                if image.ndim > 2:  # page with extra sample axis -> first component
                    image = image[0]
                yield index, np.asarray(image, dtype=np.float64)
            return
        for index in range(begin, end, step):
            image = tif.pages[index].asarray()
            if image.ndim > 2:  # page with extra sample axis -> first component
                image = image[0]
            yield index, np.asarray(image, dtype=np.float64)


def discover_image_sequence(directory: str) -> List[str]:
    """Return the sorted TIFF files of a numbered frame sequence."""
    files = [
        name for name in os.listdir(directory)
        if name.lower().endswith(('.tif', '.tiff'))
    ]
    def _number(name: str):
        match = re.findall(r'\d+', name)
        return (int(match[-1]) if match else 0, name)
    return [os.path.join(directory, name) for name in sorted(files, key=_number)]


def iter_sequence_frames(directory: str, start: int = 0,
                         stop: Optional[int] = None,
                         step: int = 1) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(sequence_position, image)`` from a folder of numbered TIFFs."""
    from .dm_reader import read_tiff

    files = discover_image_sequence(directory)
    if not files:
        raise ValueError(f"No TIFF frames found in: {directory}")
    begin = max(0, min(len(files), start))
    end = len(files) if stop is None else max(begin, min(len(files), stop))
    for position in range(begin, end, step):
        yield position, np.asarray(read_tiff(files[position]), dtype=np.float64)


def open_frame_source(source: str, start: int = 0, stop: Optional[int] = None,
                      step: int = 1) -> Tuple[StackInfo, Iterator[Tuple[int, np.ndarray]]]:
    """Open a stack file or a frame-sequence directory.

    Returns ``(info, frames)`` where ``info`` describes the source and
    ``frames`` is a single-pass iterator of ``(frame_index, image)``.
    Call this function again to iterate a second time.
    """
    if os.path.isdir(source):
        files = discover_image_sequence(source)
        if not files:
            raise ValueError(f"No TIFF frames found in: {source}")
        from .dm_reader import read_tiff
        first = np.asarray(read_tiff(files[0]))
        info = StackInfo(
            n_frames=len(files),
            height=first.shape[0],
            width=first.shape[1],
            dtype=str(first.dtype),
            source_path=os.path.abspath(source),
            frame_indices=list(range(len(files))),
            kind='sequence',
        )

        def factory():
            return iter_sequence_frames(source, start, stop, step)

        return info, factory()

    info = read_stack_info(source)
    info.frame_indices = list(range(0, info.n_frames, step))
    return info, iter_tiff_frames(source, start, stop, step)
