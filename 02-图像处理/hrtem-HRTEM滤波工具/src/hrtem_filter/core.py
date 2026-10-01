"""Headless numerical implementation of the Kilaas/Mitchell filter workflow."""

from __future__ import annotations

import threading
import numbers
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import fft, ndimage

from .geometry import PaddingInfo, Roi, extract_and_pad
from .params import FilterParams


@dataclass(frozen=True)
class FilterResult:
    """A cropped scientific result plus optional diagnostics in padded FFT space."""

    outputs: Mapping[str, np.ndarray]
    primary_key: str
    padding: PaddingInfo
    roi: Roi | None
    diagnostics: Mapping[str, np.ndarray]

    @property
    def primary(self) -> np.ndarray:
        return self.outputs[self.primary_key]


class HRTEMFilter:
    """Numerical filter independent of files, Tk, Matplotlib, and OpenCV.

    ``fast_radial_bin`` preserves the prior Python implementation's fast
    integer-annulus average.  ``dm_compatible`` uses 512 angular samples and
    bilinear interpolation, matching the sampling strategy of the bundled DM
    v4 reference more closely.  Both modes are intentionally explicit so
    future golden-image validation can select the default based on evidence.

    Instances are safe to share across threads: the radial-grid cache is
    lock-protected, and ``process_image`` itself holds no mutable state.
    """

    def __init__(self, *, radial_cache_bytes: int = 64 * 1024 * 1024, workers: int = 1) -> None:
        self._workers = 1
        self.workers = workers
        self._radial_cache: OrderedDict[tuple[int, int], tuple[np.ndarray, np.ndarray, int]] = OrderedDict()
        self._radial_cache_bytes = 0
        self._radial_cache_limit = radial_cache_bytes
        self._radial_cache_lock = threading.Lock()

    @property
    def workers(self) -> int:
        """FFT 线程数（scipy pocketfft 每次调用读取，可随时调整）。"""
        return self._workers

    @workers.setter
    def workers(self, value: int) -> None:
        if value < 1:
            raise ValueError(f"FFT workers 必须至少为 1，当前为 {value}")
        self._workers = int(value)

    @staticmethod
    def estimate_peak_bytes(
        shape: tuple[int, int],
        dtype: np.dtype | str | None = None,
        output_count: int = 1,
    ) -> int:
        """Conservative peak-memory estimate for one or more selected outputs.

        The baseline ``128`` bytes per padded pixel is the v5 float32,
        single-output estimate (spatial arrays + complex64 FFTs + masks +
        transient rotational-average arrays).  For float64/complex128 work the
        working set roughly doubles, so we scale by ``itemsize // 4``; multiple
        outputs (e.g. ``include_all_outputs=True`` or diagnostics) scale the
        estimate by ``output_count``.  This is intentionally a warning estimate
        rather than a promise: SciPy allocators and platform allocator overhead
        vary.
        """
        size = 1 << (max(shape) - 1).bit_length()
        itemsize = np.dtype(dtype).itemsize if dtype is not None else np.dtype(np.float32).itemsize
        count = max(1, int(output_count))
        bytes_per_pixel = 128 * max(1, itemsize // 4) * count
        return size * size * bytes_per_pixel

    @staticmethod
    def butterworth_filter(size: int, order: int, zero_radius: float) -> np.ndarray:
        if size < 4:
            raise ValueError("Butterworth 尺寸至少为 4")
        if size % 2:
            raise ValueError(
                f"Butterworth 尺寸必须为偶数（FFT 对称网格），当前为 {size}；"
                "矩形图像请通过 process_image 自动填充到二次幂偶数尺寸"
            )
        if order < 1:
            raise ValueError("Butterworth 阶数至少为 1")
        if zero_radius <= 0:
            raise ValueError("Butterworth 截止半径必须为正数")
        y, x = np.ogrid[-size // 2 : size // 2, -size // 2 : size // 2]
        radius = np.hypot(x, y, dtype=np.float32)
        # At zero_radius the amplitude is 1/sqrt(2), i.e. half power (-3 dB).
        half_power_constant = np.sqrt(2.0, dtype=np.float32) - np.float32(1.0)
        return (1.0 / (1.0 + half_power_constant * (radius / zero_radius) ** (2 * order))).astype(np.float32)

    @staticmethod
    def create_crosshair_image(
        size: int, crosshair_width: int, hole_radius: int, zero_radius: float
    ) -> np.ndarray:
        # 基础几何防御（DM 参数范围校验在 params.validated()，这里只挡
        # 绕过 FilterParams 直接调用静态方法的退化输入）。
        if isinstance(crosshair_width, bool) or not isinstance(crosshair_width, numbers.Integral) \
                or crosshair_width < 1:
            raise ValueError(f"crosshair_width 必须是 >=1 的整数，收到 {crosshair_width!r}")
        if crosshair_width >= size:
            raise ValueError(
                f"crosshair_width ({crosshair_width}) 必须小于 size ({size})，"
                "否则十字线会把整个频域涂黑"
            )
        if isinstance(hole_radius, bool) or not isinstance(hole_radius, numbers.Integral) \
                or hole_radius < 0:
            raise ValueError(f"hole_radius 必须是 >=0 的整数，收到 {hole_radius!r}")
        if hole_radius > size // 2:
            raise ValueError(
                f"hole_radius ({hole_radius}) 超过半幅（size//2 = {size // 2}），"
                "十字掩膜退化为纯 Butterworth 低通"
            )
        crosshair = np.ones((size, size), dtype=np.float32)
        center = size // 2
        half_width = crosshair_width // 2
        crosshair[center - half_width : center + half_width, :] = 0.0
        crosshair[:, center - half_width : center + half_width] = 0.0
        y, x = np.ogrid[-center : size - center, -center : size - center]
        crosshair[x * x + y * y <= hole_radius * hole_radius] = 1.0
        envelope = HRTEMFilter.butterworth_filter(size, 5, zero_radius)
        return crosshair * envelope + (1.0 - envelope)

    @staticmethod
    def crosshair_half_extent(size: int, crosshair_bw_ro: float) -> float:
        """十字线有效长度：包络幅值降至 0.5 处的半径（px）。

        掩膜为 ``envelope*cross + (1-envelope)``，十字遮挡量随包络衰减，
        此半径之外十字基本不再起作用。界面用它提示"小尺寸 ROI 上 STEM
        十字掩膜会退化为空操作"（十字长度按图像尺寸比例缩放，而中心孔
        半径是绝对像素）。
        """
        # 包络 = 1/(1+C*(r/r0)^10)（order 5），=0.5 当 (r/r0)^10 = 1/C。
        return crosshair_bw_ro * size * (1.0 / (np.sqrt(2.0) - 1.0)) ** 0.1

    def _fast_radial_grid(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        with self._radial_cache_lock:
            cached = self._radial_cache.get(shape)
            if cached is not None:
                self._radial_cache.move_to_end(shape)
                return cached[0], cached[1]
            h, w = shape
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            max_radius = int(np.hypot(max(cx, w - cx - 1), max(cy, h - cy - 1)))
            dtype = np.uint16 if max_radius <= np.iinfo(np.uint16).max else np.uint32
            radii = np.hypot(x - cx, y - cy).astype(dtype)
            counts = np.bincount(radii.ravel()).astype(np.int32, copy=False)
            byte_count = radii.nbytes + counts.nbytes
            # 单个网格超过预算时仍保留为唯一缓存项：大图（如 8192²）正是
            # 重复旋转平均最多的场景（cycles × frames），静默禁用缓存会把
            # 网格重算成本放大到不可接受。总内存上界 = max(预算, 单个网格)。
            while self._radial_cache and (
                byte_count > self._radial_cache_limit
                or self._radial_cache_bytes + byte_count > self._radial_cache_limit
            ):
                _, (_, _, evicted_bytes) = self._radial_cache.popitem(last=False)
                self._radial_cache_bytes -= evicted_bytes
            self._radial_cache[shape] = (radii, counts, byte_count)
            self._radial_cache_bytes += byte_count
            return radii, counts

    def _rotational_average_fast(self, image: np.ndarray) -> np.ndarray:
        radii, counts = self._fast_radial_grid(image.shape)
        sums = np.bincount(radii.ravel(), weights=image.ravel(), minlength=counts.size)
        profile = (sums / np.maximum(counts, 1)).astype(np.float32)
        return profile[radii]

    @staticmethod
    def _rotational_average_dm(image: np.ndarray, samples: int = 512) -> np.ndarray:
        h, w = image.shape
        cy, cx = h // 2, w // 2
        max_radius = min(h, w) // 2
        radii = np.arange(max_radius, dtype=np.float32)
        angles = np.linspace(0, 2 * np.pi, samples, endpoint=False, dtype=np.float32)
        yy = cy + radii[:, None] * np.cos(angles)[None, :]
        xx = cx + radii[:, None] * np.sin(angles)[None, :]
        samples_image = ndimage.map_coordinates(image, [yy.ravel(), xx.ravel()], order=1, mode="nearest")
        profile = samples_image.reshape(max_radius, samples).mean(axis=1)
        y, x = np.ogrid[:h, :w]
        distance = np.hypot(x - cx, y - cy)
        return np.interp(distance, radii, profile, left=profile[0], right=profile[-1]).astype(np.float32)

    def rotational_average(self, image: np.ndarray, method: str) -> np.ndarray:
        if method == "fast_radial_bin":
            return self._rotational_average_fast(image)
        if method == "dm_compatible":
            return self._rotational_average_dm(image)
        raise ValueError(f"未知旋转平均方法: {method}")

    def process_image(
        self,
        image: np.ndarray,
        params: FilterParams | None = None,
        *,
        roi: Roi | tuple[int, int, int, int] | None = None,
        include_diagnostics: bool = False,
        include_all_outputs: bool = False,
    ) -> FilterResult:
        params = (params or FilterParams()).validated()
        padded, padding, checked_roi = extract_and_pad(image, roi)
        outputs, diagnostics = self._process_padded(
            padded,
            params,
            include_diagnostics=include_diagnostics,
            include_all_outputs=include_all_outputs,
        )
        cropped = {key: padding.crop(value).astype(np.float32, copy=False) for key, value in outputs.items()}
        return FilterResult(cropped, params.primary_key, padding, checked_roi, diagnostics)

    def _process_padded(
        self,
        image: np.ndarray,
        params: FilterParams,
        *,
        include_diagnostics: bool,
        include_all_outputs: bool,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        size = image.shape[0]
        if image.shape != (size, size):
            raise ValueError(
                f"_process_padded 只接受已填充为正方形的图像，实际为 {image.shape}；"
                "矩形输入请通过 process_image/geometry.extract_and_pad 处理，"
                "它们会自动把矩形图像填充到二次幂偶数尺寸"
            )
        clean_fft = fft.fftshift(fft.fft2(image, workers=self._workers))
        diagnostics: dict[str, np.ndarray] = {}
        if include_diagnostics:
            diagnostics["fft"] = clean_fft

        wanted = {params.primary_output}
        if include_all_outputs and params.delta != 0:
            # "All outputs" means every output family regardless of the
            # primary; delta == 0 is an explicit pure-Butterworth mode, so
            # include_all_outputs is intentionally ignored there.
            wanted.update({"wiener", "absf", "butterworth"})

        # STEM crosshair applies to every requested output, matching the v4
        # behaviour where the pure-Butterworth path also multiplied it in.
        crosshair = None
        if params.stem_filter:
            crosshair = self.create_crosshair_image(
                size,
                params.crosshair_width,
                params.crosshair_hole_radius,
                params.crosshair_bw_ro * size,
            )

        outputs: dict[str, np.ndarray] = {}
        if "butterworth" in wanted:
            bw = self.butterworth_filter(size, params.bw_order, (size / 2) * params.bw_ro)
            filtered_fft = clean_fft * bw
            key = "butterworth_filtered"
            if crosshair is not None:
                filtered_fft = filtered_fft * crosshair
                # 与 wiener/absf 保持一致：应用了十字掩膜的输出必须带 stem_
                # 前缀，下游才能仅凭键名判断掩膜是否生效。
                key = f"stem_{key}"
            outputs[key] = np.real(fft.ifft2(fft.ifftshift(filtered_fft))).astype(np.float32)
            if include_diagnostics:
                diagnostics["butterworth_mask"] = bw
            if wanted == {"butterworth"}:
                if include_diagnostics and crosshair is not None:
                    diagnostics["stem_crosshair_mask"] = crosshair
                return outputs, diagnostics

        edge_trimmer = self.butterworth_filter(size, 12, 0.4 * size)
        fft_for_background = fft.fftshift(fft.fft2(image * edge_trimmer, workers=self._workers))
        magnitude = np.abs(fft_for_background).astype(np.float32)
        smoothed = ndimage.median_filter(magnitude, size=3).astype(np.float32, copy=False)
        binary_mask = np.zeros((size, size), dtype=bool)
        y, x = np.ogrid[-size // 2 : size // 2, -size // 2 : size // 2]
        outer_mask = (x * x + y * y) > (size // 2) ** 2
        binary_sum = 0.0
        threshold = 1
        counter = 0
        eps = np.float32(1e-12)
        while binary_sum < params.delta and counter < params.cycles:
            rotational = self.rotational_average(magnitude, params.rotation_method)
            change = ((smoothed - rotational) / np.maximum(smoothed, eps)) * 100.0
            np.maximum(change, 0.0, out=change)
            binary_mask |= change > (100 - threshold)
            binary_mask[outer_mask] = False
            binary_sum = float(binary_mask.mean() * 100.0)
            magnitude[binary_mask] = rotational[binary_mask]
            counter += 1
            threshold += params.step

        background = self.rotational_average(magnitude, params.rotation_method)
        whole_magnitude = np.abs(clean_fft).astype(np.float32)
        absf_mask = (whole_magnitude - background) / np.maximum(whole_magnitude, eps)
        np.maximum(absf_mask, 0.0, out=absf_mask)

        if params.low_freq_percent > 0:
            radius = max(1, round((size / 2) * (params.low_freq_percent / 100.0)))
            yy, xx = np.ogrid[-size // 2 : size // 2, -size // 2 : size // 2]
            distance = np.hypot(xx, yy)
            center = distance < radius
            absf_mask[center] = ((1.0 - distance[center] / radius) * 0.1 + 0.05).astype(np.float32)

        wiener_mask = absf_mask * ((whole_magnitude + background) / np.maximum(whole_magnitude, eps))
        np.maximum(wiener_mask, 0.0, out=wiener_mask)
        if params.apply_butterworth:
            envelope = self.butterworth_filter(size, params.bw_order, (size / 2) * params.bw_ro)
            absf_mask *= envelope
            wiener_mask *= envelope
        else:
            envelope = None

        masks = {"wiener": wiener_mask, "absf": absf_mask}
        for mode in wanted:
            if mode == "butterworth":
                continue
            filtered_fft = clean_fft * masks[mode]
            key = f"{mode}_filtered"
            if params.stem_filter:
                filtered_fft *= crosshair
                key = f"stem_{key}"
            outputs[key] = np.real(fft.ifft2(fft.ifftshift(filtered_fft))).astype(np.float32)

        if include_diagnostics:
            diagnostics.update(
                {
                    "background_magnitude": background,
                    "wiener_mask": wiener_mask,
                    "absf_mask": absf_mask,
                    "binary_mask": binary_mask.astype(np.uint8),
                }
            )
            if envelope is not None:
                diagnostics["butterworth_envelope"] = envelope
            if crosshair is not None:
                diagnostics["stem_crosshair_mask"] = crosshair
        return outputs, diagnostics
