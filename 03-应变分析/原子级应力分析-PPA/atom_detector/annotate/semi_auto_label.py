"""
半自动标注工具 — 利用 PPA 传统算法生成预标注
============================================
对文件夹中的 HRTEM/STEM 图像批量执行原子检测，
将检测结果保存为 JSON 预标注文件，供 label_review.py 人工修正。

用法:
    python semi_auto_label.py --input ./raw_images --output ./raw_labels
    python semi_auto_label.py --input ./raw_images --sigma 1.0 --min_dist 6
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# 添加父目录到路径以导入 PPA 检测算法
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scipy.ndimage import maximum_filter, gaussian_filter


def background_threshold(img_filt, k_sigma=6.0):
    """背景自适应阈值: 中位数 + k_sigma·σ_MAD (σ_MAD = 1.4826·MAD)。

    旧实现取全图 80% 分位数, 阈值与"原子占空比"耦合: HAADF 中强度近似
    正比于 Z², 暗 (低 Z) 原子列天然落在分布底部, 被系统性漏检; 该偏差经
    "预标注→人工确认→训练"链路放大。背景统计只随噪声尺度变化, 与视场
    内原子数量无关。
    """
    background = float(np.median(img_filt))
    mad = float(np.median(np.abs(img_filt - background)))
    return background + k_sigma * 1.4826 * mad


def detect_atoms_traditional(image, sigma=0.8, min_dist=8, window=5, threshold=None):
    """
    传统原子检测算法（复用 PPA 核心逻辑）

    Parameters
    ----------
    image : np.ndarray (H, W) float64, 0-1 范围
    sigma : float 高斯预滤波 sigma
    min_dist : float 最小原子间距
    window : int 质心精炼窗口 (必须为 >=3 的奇数)
    threshold : float or None
        强度阈值。None = 背景自适应 (中位数 + 6·σ_MAD);
        (0, 1) 内的值为百分位数小数; >=1 无意义 (图像已归一化) 会被拒绝。

    Returns
    -------
    points : list of (x, y) 亚像素坐标
    """
    if window % 2 == 0 or window < 3:
        raise ValueError(f"window 必须是 >=3 的奇数 (COM 窗口中心对称), 收到 {window}")
    if threshold is not None:
        if not np.isfinite(threshold) or not (0.0 < threshold < 1.0):
            raise ValueError(
                f"threshold 只接受 (0,1) 内的百分位数或 None; 图像已归一化到 [0,1], "
                f">=1 的绝对阈值选不出任何像素。收到 {threshold}")

    img = image.copy()
    h_img, w_img = img.shape

    # 1) 高斯预滤波
    if sigma > 0:
        img_filt = gaussian_filter(img, sigma=sigma)
    else:
        img_filt = img

    # 2) 自适应阈值: 背景统计 (中位数 + 6σ_MAD), 见 background_threshold 文档
    if threshold is None:
        threshold_abs = background_threshold(img_filt)
    else:
        threshold_abs = np.percentile(img_filt, threshold * 100)

    # 3) 局部最大值检测
    max_filt = maximum_filter(img_filt, size=3)
    peaks_mask = (img_filt >= max_filt - 1e-12) & (img_filt > threshold_abs)

    coords = np.argwhere(peaks_mask)  # (row, col)
    if len(coords) == 0:
        return []

    # 4) 按强度排序
    vals = img_filt[peaks_mask]
    idx_sort = np.argsort(-vals)
    coords = coords[idx_sort]

    # 5) 非极大值抑制
    selected = []
    for c in coords:
        if all(np.linalg.norm(c - s) > min_dist for s in selected):
            selected.append(c)
    selected = np.array(selected)

    if len(selected) == 0:
        return []

    # 6) 亚像素质心修正 (COM)
    hw = window // 2
    padded = np.pad(img, hw, mode='reflect')
    points = []
    ys_grid, xs_grid = np.mgrid[0:window, 0:window]

    for c in selected:
        r, col = int(c[0]), int(c[1])
        pr, pc = r + hw, col + hw
        roi = padded[pr - hw:pr + hw + 1, pc - hw:pc + hw + 1]

        bg = np.percentile(roi, 5)
        roi_sub = np.maximum(roi - bg, 0)
        total = roi_sub.sum()

        if total > 0:
            cx = pc - 2 * hw + (xs_grid * roi_sub).sum() / total
            cy = pr - 2 * hw + (ys_grid * roi_sub).sum() / total
        else:
            cx, cy = float(col), float(r)

        points.append((float(cx), float(cy)))

    return points


def load_image(path):
    """加载图像并归一化到 0-1"""
    import tifffile
    try:
        img = tifffile.imread(str(path))
    except Exception:
        import matplotlib.pyplot as plt
        img = plt.imread(str(path))

    if img.ndim == 3:
        if img.shape[2] >= 3:
            img = img[..., :3].mean(axis=2)
        else:
            img = img[..., 0]

    img = img.astype(np.float64)
    lo, hi = np.percentile(img, [1, 99])
    if hi > lo:
        img = np.clip((img - lo) / (hi - lo), 0, 1)
    return img


def process_folder(input_dir, output_dir, sigma=0.8, min_dist=8, window=5, threshold=None):
    """批量处理文件夹中的图像"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 支持的图像格式
    extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}
    image_files = sorted([
        f for f in input_path.iterdir()
        if f.suffix.lower() in extensions
    ])

    if not image_files:
        print(f"[错误] 在 {input_dir} 中未找到图像文件")
        return

    print(f"找到 {len(image_files)} 张图像，开始预标注...")
    print(f"参数: sigma={sigma}, min_dist={min_dist}, window={window}, threshold={threshold}")
    print("-" * 60)

    total_atoms = 0
    for i, img_file in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] 处理: {img_file.name} ...", end=" ")

        try:
            img = load_image(img_file)
            points = detect_atoms_traditional(img, sigma, min_dist, window, threshold)

            # 保存预标注 JSON
            label_data = {
                'image_file': img_file.name,
                'image_path': str(img_file.resolve()),
                'image_shape': [img.shape[0], img.shape[1]],  # (H, W)
                'points': [{'x': p[0], 'y': p[1], 'confidence': 1.0} for p in points],
                'params': {
                    'sigma': sigma,
                    'min_dist': min_dist,
                    'window': window,
                    'threshold': threshold,
                },
                'reviewed': False,
            }

            label_file = output_path / f"{img_file.stem}.json"
            with open(label_file, 'w', encoding='utf-8') as f:
                json.dump(label_data, f, ensure_ascii=False, indent=2)

            total_atoms += len(points)
            print(f"检测到 {len(points)} 个原子 -> {label_file.name}")

        except Exception as e:
            print(f"失败: {e}")

    print("-" * 60)
    print(f"完成! 共处理 {len(image_files)} 张图像，检测到 {total_atoms} 个原子")
    print(f"预标注已保存到: {output_path.resolve()}")
    print(f"\n下一步: 运行 label_review.py 进行人工审核修正")


def main():
    parser = argparse.ArgumentParser(description="半自动原子标注 — 预标注生成")
    parser.add_argument("--input", "-i", required=True, help="输入图像文件夹路径")
    parser.add_argument("--output", "-o", default="./raw_labels", help="输出标注文件夹路径")
    parser.add_argument("--sigma", type=float, default=0.8, help="高斯滤波 sigma")
    parser.add_argument("--min_dist", type=float, default=8, help="最小原子间距")
    parser.add_argument("--window", type=int, default=5, help="质心窗口大小 (奇数)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="强度阈值 (None=背景自适应, 或 0~1 之间的百分位数)")
    args = parser.parse_args()

    if args.window % 2 == 0 or args.window < 3:
        parser.error(f"--window 必须是 >=3 的奇数, 收到 {args.window}")
    if args.threshold is not None and not (0.0 < args.threshold < 1.0):
        parser.error("--threshold 只接受 (0,1) 内的百分位数 (图像已归一化到 [0,1])")

    process_folder(args.input, args.output, args.sigma, args.min_dist,
                   args.window, args.threshold)


if __name__ == "__main__":
    main()
