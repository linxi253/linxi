"""
数据集准备 — 将标注转换为 YOLOv8 训练格式
========================================
功能:
  1. 收集已审核的图像 + YOLO 标签
  2. 按比例划分 train/val/test
  3. 生成 data.yaml 配置文件
  4. 统计原子尺寸分布

用法:
    python prepare_dataset.py --images ./raw_images --labels ./yolo_labels --output ./datasets/atom_v1
    python prepare_dataset.py --images ./raw_images --labels ./yolo_labels --output ./datasets/atom_v1 --imgsz 1024
"""
import os
import sys
import shutil
import argparse
import random
import numpy as np
from pathlib import Path
import yaml

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



def _source_stem(stem):
    """返回增强副本的原图 stem: 'img001_aug_rot90' → 'img001'。

    数据集划分必须按原图分组 —— 同一原图的多个增强变体若分别落入
    train 与 test, 模型对传统算法的对比指标会因泄漏而虚高。
    """
    return stem.split('_aug_')[0]


def prepare_dataset(image_dir, label_dir, output_dir, imgsz=640,
                    train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    准备 YOLOv8 格式数据集

    Parameters
    ----------
    image_dir : str 图像文件夹
    label_dir : str YOLO 标签文件夹 (.txt)
    output_dir : str 输出数据集目录
    imgsz : int 目标图像尺寸
    train_ratio, val_ratio, test_ratio : float 划分比例 (三者之和必须为 1)
    seed : int 随机种子

    划分按"原图分组"进行: 同一原图的 `_aug_` 增强变体永远进入同一
    子集, 防止 train/test 泄漏。
    """
    if not np.isfinite([train_ratio, val_ratio, test_ratio]).all():
        raise ValueError("划分比例必须为有限数值")
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio 必须等于 1, 当前为 {ratio_sum}")
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("划分比例不能为负")

    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)

    # A second split must not retain samples from an earlier train/test split.
    # Refuse reuse without deleting any existing datasets.
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("输出目录必须为空。请为本次划分选择新的目录，避免旧样本跨 train/test 泄漏。")

    # 创建目录结构
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 收集有效的图像-标签对
    extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}
    pairs = []

    for label_file in sorted(label_dir.glob("*.txt")):
        stem = label_file.stem
        # 查找对应图像
        img_file = None
        for ext in extensions:
            candidate = image_dir / f"{stem}{ext}"
            if candidate.exists():
                img_file = candidate
                break

        if img_file is None:
            print(f"[警告] 标签 {label_file.name} 无对应图像，跳过")
            continue

        # 验证标签格式
        try:
            lines = label_file.read_text().strip().split('\n')
            valid_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    # 框心必须在图内, 框尺寸必须为正且不越界 (>1 的框是坏标注)
                    if (0 <= cx <= 1 and 0 <= cy <= 1
                            and 0 < w <= 1 and 0 < h <= 1):
                        valid_lines.append(line.strip())
            if valid_lines:
                pairs.append((img_file, label_file, valid_lines))
        except Exception as e:
            print(f"[警告] 标签 {label_file.name} 格式错误: {e}")

    if not pairs:
        print("[错误] 未找到有效的图像-标签对!")
        return

    print(f"找到 {len(pairs)} 个有效图像-标签对")

    # 按"原图 stem"分组后随机划分: 同一原图 (含其 _aug_ 增强变体)
    # 必须整体进入同一子集, 否则增强副本会造成 train/test 泄漏。
    groups = {}
    for pair in pairs:
        groups.setdefault(_source_stem(pair[0].stem), []).append(pair)
    group_keys = sorted(groups)

    random.seed(seed)
    random.shuffle(group_keys)

    n_groups = len(group_keys)
    n_train = int(n_groups * train_ratio)
    n_val = int(n_groups * val_ratio)
    # test 显式取剩余, 避免"比例之和≠1 时 test 份额静默漂移"
    n_test = n_groups - n_train - n_val
    if n_test < 0:
        raise ValueError("比例配置使 test 份额为负, 请检查比例之和为 1")

    splits = {
        'train': [p for k in group_keys[:n_train] for p in groups[k]],
        'val': [p for k in group_keys[n_train:n_train + n_val] for p in groups[k]],
        'test': [p for k in group_keys[n_train + n_val:] for p in groups[k]],
    }
    print(f"按原图分组划分: {n_groups} 组 (含增强变体), "
          f"train/val/test 组数 = {n_train}/{n_val}/{n_test}")

    # 复制文件并统计
    stats = {'train': 0, 'val': 0, 'test': 0}
    all_box_sizes = []

    for split_name, split_pairs in splits.items():
        for img_file, label_file, valid_lines in split_pairs:
            # 复制图像
            dst_img = output_dir / 'images' / split_name / img_file.name
            shutil.copy2(img_file, dst_img)

            # 写入标签
            dst_label = output_dir / 'labels' / split_name / f"{img_file.stem}.txt"
            dst_label.write_text('\n'.join(valid_lines))

            stats[split_name] += 1

            # 收集框尺寸
            for line in valid_lines:
                parts = line.split()
                w, h = float(parts[3]), float(parts[4])
                all_box_sizes.append((w, h))

    # 生成 data.yaml
    data_yaml = {
        'path': str(output_dir.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {0: 'atom'},
    }

    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    # 打印统计
    print("\n" + "=" * 50)
    print("数据集准备完成!")
    print("=" * 50)
    print(f"  训练集: {stats['train']} 张")
    print(f"  验证集: {stats['val']} 张")
    print(f"  测试集: {stats['test']} 张")
    print(f"  总计:   {sum(stats.values())} 张")
    print(f"\n  输出目录: {output_dir.resolve()}")
    print(f"  配置文件: {yaml_path.resolve()}")

    if all_box_sizes:
        sizes = np.array(all_box_sizes)
        print(f"\n  原子框尺寸统计 (归一化):")
        print(f"    宽度:  mean={sizes[:,0].mean():.4f}, std={sizes[:,0].std():.4f}, "
              f"min={sizes[:,0].min():.4f}, max={sizes[:,0].max():.4f}")
        print(f"    高度:  mean={sizes[:,1].mean():.4f}, std={sizes[:,1].std():.4f}, "
              f"min={sizes[:,1].min():.4f}, max={sizes[:,1].max():.4f}")
        print(f"    总标注原子数: {len(all_box_sizes)}")

        # 估算像素尺寸
        print(f"\n  按 imgsz={imgsz} 换算为像素:")
        print(f"    宽度:  mean={sizes[:,0].mean()*imgsz:.1f}px, "
              f"range=[{sizes[:,0].min()*imgsz:.1f}, {sizes[:,0].max()*imgsz:.1f}]px")
        print(f"    高度:  mean={sizes[:,1].mean()*imgsz:.1f}px, "
              f"range=[{sizes[:,1].min()*imgsz:.1f}, {sizes[:,1].max()*imgsz:.1f}]px")

    print(f"\n  下一步: python train/train.py --data {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="准备 YOLOv8 原子检测数据集")
    parser.add_argument("--images", "-i", required=True, help="图像文件夹")
    parser.add_argument("--labels", "-l", required=True, help="YOLO 标签文件夹")
    parser.add_argument("--output", "-o", default="./datasets/atom_v1", help="输出目录")
    parser.add_argument("--imgsz", type=int, default=640, help="目标图像尺寸")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataset(args.images, args.labels, args.output, args.imgsz,
                    args.train_ratio, args.val_ratio, args.test_ratio, args.seed)


if __name__ == "__main__":
    main()
