r"""Compatibility launcher and legacy API adapter for the v5 package.

The legacy v4 implementation was archived on the original developer machine as
``08-历史版本\hrtem-HRTEM滤波工具-v4-2026-07-29``; that archive is not
distributed with this repository.

Two entry styles are supported:

1. Run ``python butter.py`` to open the v5 graphical interface.
2. Import the v4-compatible facade from downstream tools::

       from butter import HRTEMFilter

       proc = HRTEMFilter()
       proc.params['delta'] = 5
       results = proc.process_image(image)   # dict, same keys as v4

   This keeps ppa.py's preprocessing panel working against the v5 core.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

from hrtem_filter.core import HRTEMFilter as _CoreFilter
from hrtem_filter.params import FilterParams


class HRTEMFilter:
    """v4-compatible dict-based facade over the v5 numerical core.

    Preserves the old contract relied upon by downstream callers (ppa.py):

    - ``self.params`` is a plain mutable dict with the v4 key names.
    - ``process_image(image)`` returns a dict whose keys match v4:
      ``original``, ``fft``, ``wiener_filtered``, ``absf_filtered``,
      ``butterworth_filtered`` (and ``stem_*`` variants).
    - Input already padded to a 2^n square passes through unchanged, so the
      caller's own crop-back logic keeps working.
    """

    def __init__(self) -> None:
        self.default_params = {
            'step': 2,
            'delta': 5,
            'cycles': 99,
            'bw_order': 4,
            'bw_ro': 0.3,
            'apply_bw_filter': True,
            'apply_wiener': True,
            'apply_absf': True,
            'show_fft': True,
            'stem_filter': False,
            'show_crosshair_fft': True,
            'crosshair_width': 4,
            'crosshair_hole_radius': 4,
            'crosshair_bw_ro': 0.03,
            'display_size': 400,
            'low_freq_percent': 0,
        }
        self.params = self.default_params.copy()
        self._core = _CoreFilter()

    # -- v4 static helpers kept as thin proxies -------------------------
    @staticmethod
    def butterworth_filter(img_size: int, order: int, zero_radius: float) -> np.ndarray:
        return _CoreFilter.butterworth_filter(img_size, order, zero_radius)

    def rotational_average(self, image: np.ndarray) -> np.ndarray:
        return self._core.rotational_average(np.asarray(image, dtype=np.float32),
                                             "fast_radial_bin")

    # -- parameter translation ------------------------------------------
    def _to_filter_params(self) -> FilterParams:
        """Translate the v4 dict into a validated v5 FilterParams.

        注意兼容性差异：这里用 ``int()``/``float()`` 转换参数，非整数浮点
        （如 step=2.5）会被**静默截断**，与 v5 核心 API"显式拒绝"的行为
        不同。这是 v4 向后兼容所要求的；新代码请直接使用
        ``hrtem_filter.FilterParams``。
        """
        p = self.params
        if float(p.get('delta', 5)) == 0:
            primary = 'butterworth'
        elif p.get('apply_wiener', True):
            primary = 'wiener'
        elif p.get('apply_absf', False):
            primary = 'absf'
        else:
            primary = 'butterworth'
        return FilterParams(
            step=int(p.get('step', 2)),
            delta=float(p.get('delta', 5)),
            cycles=int(p.get('cycles', 99)),
            bw_order=int(p.get('bw_order', 4)),
            bw_ro=float(p.get('bw_ro', 0.3)),
            apply_butterworth=bool(p.get('apply_bw_filter', True)),
            primary_output=primary,
            stem_filter=bool(p.get('stem_filter', False)),
            crosshair_width=int(p.get('crosshair_width', 4)),
            crosshair_hole_radius=int(p.get('crosshair_hole_radius', 4)),
            crosshair_bw_ro=float(p.get('crosshair_bw_ro', 0.03)),
            low_freq_percent=float(p.get('low_freq_percent', 0.0)),
        ).validated()

    # -- v4 processing entry point ---------------------------------------
    def process_image(self, image: np.ndarray, use_roi: bool = False,
                      roi_coords=None) -> dict:
        """Run the v5 core and re-shape the result into the v4 dict format."""
        params = self._to_filter_params()

        roi = None
        if use_roi and roi_coords:
            top, left, bottom, right = roi_coords
            roi = (top, left, bottom, right)

        # v4 returned every enabled output; ask the core for both when the
        # caller enabled Wiener and ABSF together.
        want_all = (params.delta != 0
                    and bool(self.params.get('apply_wiener', True))
                    and bool(self.params.get('apply_absf', False)))

        result = self._core.process_image(
            np.asarray(image),
            params,
            roi=roi,
            include_diagnostics=True,
            include_all_outputs=want_all,
        )

        img32 = np.asarray(image).astype(np.float32)
        if use_roi and roi_coords:
            # The core has already validated integer ROI coordinates, so
            # slicing here is safe and reports errors through its API.
            top, left, bottom, right = roi_coords
            img32 = img32[top:bottom, left:right]

        legacy: dict = {'original': img32}
        fft_image = result.diagnostics.get('fft')
        if fft_image is not None:
            legacy['fft'] = fft_image
        legacy.update(result.outputs)
        # v4 把应用了十字掩膜的 BW 输出也放在 "butterworth_filtered" 键下；
        # v5 为前缀一致性改名为 stem_butterworth_filtered，这里映射回 v4
        # 键名，保证 ppa.py 等下游调用方不受影响。
        if 'stem_butterworth_filtered' in legacy:
            legacy['butterworth_filtered'] = legacy.pop('stem_butterworth_filtered')
        return legacy


def main() -> int:
    # GUI imports (Tk/Matplotlib) stay out of module import time so that
    # ``from butter import HRTEMFilter`` works in headless environments.
    from hrtem_filter.gui import main as gui_main

    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
