"""二维高斯亚像素精炼的规范实现 (审计工单27 消重)。

ppa.py 与 原子识别纯算法/atomic_core.py 此前各自维护一份逐行复制的
二维高斯拟合实现 (初值、bounds、maxfev=500 完全相同)。本模块收敛
PPA 侧的实现; atomic_core 侧属跨项目边界, 由后续工单合并到同一真源。

数值口径与两份历史拷贝保持一致:
  - ROI 裁剪不足 3px 或扣除背景后全零 -> 拟合不成立;
  - 背景扣除: ROI 5% 分位数, 负值截零;
  - 初值: ROI 内峰值位置 (而非固定几何中心), sigma 初值 1.0,
    振幅取峰值高度, 常数底 0;
  - bounds: 中心限制在 ROI 内, sigma ∈ [0.3, 5], 振幅 >= 0, 底不设限;
  - maxfev=500;
  - 拟合中心偏离初始位置超过窗口半径视为误收敛, 按失败处理。
"""

import numpy as np
from scipy.optimize import curve_fit

__all__ = ["gaussian_refine_point"]


def gaussian_refine_point(image, x, y, window):
    """二维高斯精炼单个位置。

    Parameters
    ----------
    image : np.ndarray (H, W)
    x, y : float
        初始位置 (col, row), 允许亚像素。
    window : int
        局部窗口大小 (奇数)。

    Returns
    -------
    (cx, cy, fitted) : tuple
        ``fitted=False`` 表示拟合未成功 (窗口不足、ROI 全零、curve_fit
        异常或中心偏出窗口半径), 由调用方回退 COM 并提示。
    """
    hw = window // 2
    H, W = image.shape
    xi, yi = int(round(x)), int(round(y))
    y0 = max(0, yi - hw)
    y1 = min(H, yi + hw + 1)
    x0 = max(0, xi - hw)
    x1 = min(W, xi + hw + 1)
    if y1 - y0 < 3 or x1 - x0 < 3:
        return float(x), float(y), False

    roi = image[y0:y1, x0:x1].astype(np.float64)
    bg = np.percentile(roi, 5)
    roi = np.maximum(roi - bg, 0)
    if roi.max() <= 0:
        return float(x), float(y), False

    try:
        def gauss2d(xy, xo, yo, sx, sy, amplitude, offset):
            xv, yv = xy
            return (amplitude * np.exp(-((xv - xo) ** 2 / (2 * max(sx, 0.3) ** 2)
                                         + (yv - yo) ** 2 / (2 * max(sy, 0.3) ** 2)))
                    + offset)

        ys, xs = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
        xdata = np.vstack((xs.ravel(), ys.ravel()))
        ydata = roi.ravel()
        # 初值: ROI 内峰值位置 (靠边窗口不再以几何中心为初值)
        pk_r, pk_c = np.unravel_index(np.argmax(roi), roi.shape)
        p0 = [float(pk_c), float(pk_r), 1.0, 1.0, float(roi.max()), 0]
        bounds = ([0, 0, 0.3, 0.3, 0, -np.inf],
                  [roi.shape[1] - 1, roi.shape[0] - 1, 5, 5, np.inf, np.inf])
        popt, _ = curve_fit(gauss2d, xdata, ydata, p0=p0,
                            bounds=bounds, maxfev=500)
        cx = x0 + popt[0]
        cy = y0 + popt[1]
        # 校验: 偏离初始位置不超过窗口半径, 防止误收敛到邻近峰
        if abs(cx - x) <= hw and abs(cy - y) <= hw:
            return float(cx), float(cy), True
    except Exception:
        pass
    return float(x), float(y), False
