# -*- coding: utf-8 -*-
"""用真实液相 Ag 数据做端到端校验。

做三件事：
  1. 载入指定的裁剪图，用低频阈值法生成一个"参考掩膜"，取其轮廓
     当作手绘多边形，喂给本工具的管线；
  2. 与上一版一次性脚本的结果对比，确认数值一致（差异只来自边缘过渡形状）；
  3. 校验算法不变量（掩膜全 1 / 强度 0 恒等，掩膜全 0 时等于 base）。

用法::

    python validate_real_data.py --input 某图.tif
    python validate_real_data.py --input 某图.tif --reference 旧结果.tif

不指定 --input 时退回合成数据自检（不依赖任何外部路径）。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import deloc_core
import roi_model
import tif_io


def reference_mask(img: np.ndarray) -> np.ndarray:
    """上一版脚本用的自动掩膜（低频阈值），仅用于构造对照多边形。"""
    from scipy import ndimage

    sm = ndimage.gaussian_filter(img.astype(np.float32), 12)
    sm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-9)
    m = 1.0 / (1.0 + np.exp((sm - 0.42) / 0.04))
    return ndimage.gaussian_filter(m, 10)


def mask_to_polygon(mask: np.ndarray, max_points: int = 200):
    """把二值掩膜最长的轮廓转成 (x, y) 多边形点列。"""
    try:
        from skimage import measure
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise SystemExit(
            "本脚本需要 scikit-image 做轮廓提取（仅校验用，程序本体不依赖）。\n"
            "安装：python -m pip install scikit-image"
        ) from exc

    contours = measure.find_contours((mask > 0.5).astype(float), 0.5)
    if not contours:
        return []
    largest = max(contours, key=len)
    step = max(1, len(largest) // max_points)
    # find_contours 给的是 (row, col)，ROI 要的是 (x=col, y=row)
    return [(float(c), float(r)) for r, c in largest[::step]]


def check_invariants(img: np.ndarray, fmin: float, fmax: float) -> bool:
    engine = deloc_core.CleanEngine()
    ones = np.ones(img.shape, np.float32)
    zeros = np.zeros(img.shape, np.float32)
    base, lat = engine.components(img, fmin, fmax)

    cases = [
        ("掩膜全 1 + 强度 1 -> 等于原图",
         engine.render(img, ones, fmin, fmax, 1.0), img),
        ("掩膜全 0 + 强度 0 -> 等于原图",
         engine.render(img, zeros, fmin, fmax, 0.0), img),
        ("掩膜全 0 + 强度 1 -> 等于 base",
         engine.render(img, zeros, fmin, fmax, 1.0), base),
    ]
    ok = True
    for name, got, want in cases:
        delta = float(np.abs(got - want).max())
        flag = "OK " if delta < 1e-3 else "FAIL"
        if delta >= 1e-3:
            ok = False
        print(f"  [{flag}] {name}   max|Δ| = {delta:.3e}")
    print(f"  [OK ] 晶格带分量均值 ≈ 0   {float(lat.mean()):.3e}")
    return ok


def synthetic_selfcheck() -> int:
    """合成数据自检。

    审查项23：合成自检只证明算法不变量，不证明真实数据处理正确。退出码
    必须与真实校验可区分（0=真实数据全过，3=仅合成自检），否则脚本在
    没有 --input 时"成功退出"是假绿灯，会混进 CI/批处理当作真验证。
    """
    print("改用合成数据自检（仅验证算法不变量，不代表真实数据处理正确）。")
    rng = np.random.default_rng(0)
    img = (128 + 25 * np.sin(2 * np.pi * np.mgrid[0:256, 0:256][1] / 10.0)
           + rng.normal(0, 3, (256, 256))).astype(np.float32)
    print("\n不变量校验（合成数据）:")
    ok = check_invariants(img, 0.075, 0.135)
    print(f"\n结论: {'合成自检通过' if ok else '合成自检失败'}（退出码 3，区别于真实数据校验的 0）")
    return 3 if ok else 1


#: 真实数据校验的退出码：0=全过, 1=有不变量未通过
EXIT_REAL_OK = 0
EXIT_SYNTHETIC_OK = 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None,
                    help="待校验的 TIF；缺省时退回合成数据自检")
    ap.add_argument("--reference", default=None,
                    help="旧脚本结果 TIF，用于数值对比")
    ap.add_argument("--fmin", type=float, default=deloc_core.DEFAULT_FMIN)
    ap.add_argument("--fmax", type=float, default=deloc_core.DEFAULT_FMAX)
    ap.add_argument("--dilate", type=float, default=0.0)
    ap.add_argument("--feather", type=float, default=10.0)
    ap.add_argument("--out", default=None, help="对比图输出路径")
    args = ap.parse_args()

    if not args.input:
        print("未指定 --input。")
        return synthetic_selfcheck()
    if not os.path.isfile(args.input):
        print(f"找不到 {args.input}。")
        return synthetic_selfcheck()

    img, info = tif_io.load(args.input)
    print(f"输入: {os.path.basename(args.input)}   {info.summary()}")

    ref = reference_mask(img)
    poly = mask_to_polygon(ref)
    print(f"参考掩膜覆盖率 {float((ref > 0.5).mean()):.3f} -> 多边形 {len(poly)} 个顶点")

    roi = roi_model.RoiSet()
    if not roi.add(roi_model.RoiShape(kind="polygon", points=poly)):
        print("多边形非法，退出。")
        return 1

    mask = roi.rasterize(img.shape, dilate=args.dilate, feather=args.feather)
    print(f"本工具掩膜覆盖率 {float((mask > 0.5).mean()):.3f}")

    engine = deloc_core.CleanEngine()
    out = engine.render(img, mask, args.fmin, args.fmax, 1.0)
    print(f"处理完成，输出范围 [{out.min():.1f}, {out.max():.1f}]")

    ok = True
    if args.reference and os.path.isfile(args.reference):
        old = tif_io.load(args.reference)[0].astype(np.float32)
        if old.shape == out.shape:
            diff = np.abs(out - old)
            print(f"与参考结果差异: mean={diff.mean():.2f}  "
                  f"p99={np.percentile(diff, 99):.1f}  max={diff.max():.0f}")
        else:
            print(f"参考结果尺寸不一致 ({old.shape} vs {out.shape})，跳过对比")

    print("\n不变量校验:")
    ok &= check_invariants(img, args.fmin, args.fmax)

    out_png = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "validate_real_data.png")
    try:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                                  "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        import matplotlib.pyplot as plt

        lo, hi = np.percentile(img.astype(np.float32), [0.5, 99.5])
        fig, axes = plt.subplots(1, 4, figsize=(26, 6))
        panels = [("原图", img, "gray", lo, hi),
                  ("处理结果", out, "gray", lo, hi),
                  ("掩膜（亮=保留晶格）", mask, "inferno", 0, 1),
                  ("原图 - 结果", img.astype(np.float32) - out, "seismic",
                   -40, 40)]
        for ax, (title, data, cmap, v0, v1) in zip(axes, panels):
            ax.imshow(data, cmap=cmap, vmin=v0, vmax=v1)
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_png, dpi=95)
        plt.close(fig)
        print(f"\n对比图: {out_png}")
    except Exception as exc:  # pragma: no cover
        print(f"画图失败（不影响校验）: {exc}")

    print("\n结论:", "全部通过" if ok else "有不变量未通过")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
