"""
数据增强 — 针对 HRTEM/STEM 原子图像的增强策略
=============================================
原子图像有周期性晶格结构，增强策略需要保持物理合理性:
  - 安全: 90/180/270度旋转、翻转、亮度/对比度抖动、高斯噪声
  - 谨慎: 小角度旋转(+-5度)、轻微缩放(0.9-1.1)
  - 禁用: 大角度旋转、Mosaic、MixUp、Copy-Paste

本模块提供离线增强（扩充数据集）和在线增强参数配置。

用法:
    python augment.py --input ./datasets/atom_v1/images/train --labels ./datasets/atom_v1/labels/train --factor 3
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from scipy.ndimage import rotate, zoom, gaussian_filter
import shutil


def augment_image_and_labels(img, labels, augment_type):
    """
    对单张图像及其 YOLO 标签执行增强

    Parameters
    ----------
    img : np.ndarray (H, W) float64
    labels : list of [class_id, cx, cy, w, h] 归一化坐标
    augment_type : str 增强类型

    Returns
    -------
    img_aug : np.ndarray
    labels_aug : list
    """
    h, w = img.shape
    labels_aug = [list(l) for l in labels]  # deep copy

    if augment_type == 'flip_h':
        img_aug = np.fliplr(img).copy()
        for l in labels_aug:
            l[1] = 1.0 - l[1]  # cx -> 1-cx

    elif augment_type == 'flip_v':
        img_aug = np.flipud(img).copy()
        for l in labels_aug:
            l[2] = 1.0 - l[2]  # cy -> 1-cy

    elif augment_type == 'rot90':
        img_aug = np.rot90(img, 1).copy()
        for l in labels_aug:
            cx, cy = l[1], l[2]
            l[1] = cy
            l[2] = 1.0 - cx
            l[3], l[4] = l[4], l[3]  # w, h 交换

    elif augment_type == 'rot180':
        img_aug = np.rot90(img, 2).copy()
        for l in labels_aug:
            l[1] = 1.0 - l[1]
            l[2] = 1.0 - l[2]

    elif augment_type == 'rot270':
        img_aug = np.rot90(img, 3).copy()
        for l in labels_aug:
            cx, cy = l[1], l[2]
            l[1] = 1.0 - cy
            l[2] = cx
            l[3], l[4] = l[4], l[3]

    elif augment_type == 'brightness_up':
        factor = np.random.uniform(1.1, 1.3)
        img_aug = np.clip(img * factor, 0, 1)

    elif augment_type == 'brightness_down':
        factor = np.random.uniform(0.7, 0.9)
        img_aug = np.clip(img * factor, 0, 1)

    elif augment_type == 'contrast':
        mean = img.mean()
        factor = np.random.uniform(0.8, 1.2)
        img_aug = np.clip((img - mean) * factor + mean, 0, 1)

    elif augment_type == 'noise':
        sigma = np.random.uniform(0.01, 0.03)
        noise = np.random.normal(0, sigma, img.shape)
        img_aug = np.clip(img + noise, 0, 1)

    elif augment_type == 'blur':
        sigma = np.random.uniform(0.3, 0.8)
        img_aug = gaussian_filter(img, sigma=sigma)

    elif augment_type == 'small_rot':
        angle = np.random.uniform(-5, 5)
        img_aug = rotate(img, angle, reshape=False, mode='reflect')
        # 旋转标签坐标
        # scipy.ndimage.rotate 围绕 ((N-1)/2, (N-1)/2) 像素中心旋转,
        # 标签变换必须用同一中心; 此前误用 w/2, h/2 在图像边缘引入
        # ~0.71·θ(rad)·离心距 的系统性亚像素偏置 (5° 时约 0.05px)。
        cx_img, cy_img = (w - 1) / 2, (h - 1) / 2
        rad = np.radians(-angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        for l in labels_aug:
            px = l[1] * w - cx_img
            py = l[2] * h - cy_img
            new_px = px * cos_a - py * sin_a
            new_py = px * sin_a + py * cos_a
            l[1] = (new_px + cx_img) / w
            l[2] = (new_py + cy_img) / h
        # 过滤出界标签
        labels_aug = [l for l in labels_aug
                      if 0 < l[1] < 1 and 0 < l[2] < 1]

    elif augment_type == 'scale':
        s = np.random.uniform(0.9, 1.1)
        img_aug = zoom(img, s, mode='reflect')
        # 裁剪或填充回原尺寸
        if s > 1:
            sh = (img_aug.shape[0] - h) // 2
            sw = (img_aug.shape[1] - w) // 2
            img_aug = img_aug[sh:sh + h, sw:sw + w]
        else:
            pad_h = (h - img_aug.shape[0]) // 2
            pad_w = (w - img_aug.shape[1]) // 2
            img_aug = np.pad(img_aug, ((pad_h, h - pad_h - img_aug.shape[0]),
                                       (pad_w, w - pad_w - img_aug.shape[1])),
                             mode='reflect')
        # 调整标签: zoom(s) 后内容放大 s 倍, 相对中心的坐标与框尺寸均 ×s
        # (此前误写为 /s, 经合成图实测验证: 标注与原子位置系统性错位)
        for l in labels_aug:
            l[1] = (l[1] - 0.5) * s + 0.5
            l[2] = (l[2] - 0.5) * s + 0.5
            l[3] = min(l[3] * s, 1.0)
            l[4] = min(l[4] * s, 1.0)
        labels_aug = [l for l in labels_aug
                      if 0 < l[1] < 1 and 0 < l[2] < 1]

    else:
        img_aug = img.copy()

    return img_aug, labels_aug


def load_image(path):
    """加载图像"""
    try:
        import tifffile
        img = tifffile.imread(str(path))
    except Exception:
        import matplotlib.pyplot as plt
        img = plt.imread(str(path))

    if img.ndim == 3:
        img = img[..., :3].mean(axis=2) if img.shape[2] >= 3 else img[..., 0]
    img = img.astype(np.float64)
    lo, hi = np.percentile(img, [1, 99])
    if hi > lo:
        img = np.clip((img - lo) / (hi - lo), 0, 1)
    return img


def save_image(img, path):
    """保存图像"""
    path = Path(path)
    if path.suffix.lower() in ('.tif', '.tiff'):
        import tifffile
        tifffile.imwrite(str(path), (img * 65535).astype(np.uint16))
    else:
        import matplotlib.pyplot as plt
        plt.imsave(str(path), img, cmap='gray')


def augment_dataset(image_dir, label_dir, output_image_dir=None, output_label_dir=None,
                    factor=3, seed=42):
    """
    离线数据增强

    Parameters
    ----------
    image_dir : str 训练图像目录
    label_dir : str 训练标签目录
    output_image_dir : str 输出图像目录 (默认同输入)
    output_label_dir : str 输出标签目录 (默认同输入)
    factor : int 每张图增强几倍
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    out_img_dir = Path(output_image_dir) if output_image_dir else image_dir
    out_lbl_dir = Path(output_label_dir) if output_label_dir else label_dir
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 可用增强类型
    augment_types = [
        'flip_h', 'flip_v', 'rot90', 'rot180', 'rot270',
        'brightness_up', 'brightness_down', 'contrast',
        'noise', 'blur', 'small_rot', 'scale'
    ]

    np.random.seed(seed)

    extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}
    # 排除已有增强副本 (_aug_ 前缀): 重跑增强不得二次增强, 否则
    # 复合变换的标签误差会累积且输出不可复现。
    image_files = sorted([f for f in image_dir.iterdir()
                          if f.suffix.lower() in extensions and '_aug_' not in f.stem])

    print(f"原始图像: {len(image_files)} 张")
    print(f"增强倍数: {factor}")
    print(f"预计输出: ~{len(image_files) * (factor + 1)} 张")

    total_generated = 0
    for img_file in image_files:
        label_file = label_dir / f"{img_file.stem}.txt"
        if not label_file.exists():
            continue

        # 读取标签
        lines = label_file.read_text().strip().split('\n')
        labels = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                labels.append([int(parts[0])] + [float(x) for x in parts[1:]])

        if not labels:
            continue

        img = load_image(img_file)

        # 随机选择 factor 种增强
        chosen = np.random.choice(augment_types, size=min(factor, len(augment_types)),
                                  replace=False)

        for aug_type in chosen:
            try:
                img_aug, labels_aug = augment_image_and_labels(img, labels, aug_type)

                if not labels_aug:
                    continue

                # 保存
                stem = f"{img_file.stem}_aug_{aug_type}"
                out_img_path = out_img_dir / f"{stem}{img_file.suffix}"
                out_lbl_path = out_lbl_dir / f"{stem}.txt"

                save_image(img_aug, out_img_path)

                lbl_lines = []
                for l in labels_aug:
                    lbl_lines.append(f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}")
                out_lbl_path.write_text('\n'.join(lbl_lines))

                total_generated += 1
            except Exception as e:
                print(f"  [警告] {img_file.name} + {aug_type} 失败: {e}")

    print(f"\n增强完成! 生成 {total_generated} 张新图像")
    print(f"输出目录: {out_img_dir.resolve()}")


def get_yolo_augment_config():
    """
    返回 YOLOv8 训练时的在线增强配置
    (用于 hyp.yaml 或 train() 参数)
    """
    return {
        # 安全增强
        'fliplr': 0.5,       # 水平翻转
        'flipud': 0.5,       # 垂直翻转
        'degrees': 5.0,      # 小角度旋转 (最大5度)
        'scale': 0.1,        # 缩放范围 +-10%
        'brightness': 0.2,   # 亮度抖动
        'contrast': 0.2,     # 对比度抖动

        # 禁用不适合原子图的增强
        'mosaic': 0.0,       # Mosaic 破坏周期性
        'mixup': 0.0,        # MixUp 不适合
        'copy_paste': 0.0,   # Copy-Paste 不适合
        'shear': 0.0,        # 剪切破坏晶格
        'perspective': 0.0,  # 透视变换不适合
        'hsv_h': 0.0,        # 灰度图无需色相
        'hsv_s': 0.0,        # 灰度图无需饱和度
        'hsv_v': 0.0,        # 用 brightness 代替
    }


def main():
    parser = argparse.ArgumentParser(description="HRTEM 原子图像数据增强")
    parser.add_argument("--input", "-i", required=True, help="训练图像目录")
    parser.add_argument("--labels", "-l", required=True, help="训练标签目录")
    parser.add_argument("--output_images", default=None, help="输出图像目录")
    parser.add_argument("--output_labels", default=None, help="输出标签目录")
    parser.add_argument("--factor", "-f", type=int, default=3, help="增强倍数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    augment_dataset(args.input, args.labels, args.output_images, args.output_labels,
                    args.factor, args.seed)


if __name__ == "__main__":
    main()
