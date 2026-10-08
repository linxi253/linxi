"""
评测脚本 — 对比 YOLOv8 模型与传统算法的原子检测性能
==================================================
指标:
  - 检测率 (Recall): 真实原子被检出的比例
  - 误检率 (FPR): 非原子位置被误报的比例
  - 定位精度: 检测中心与真实中心的 RMSE (像素)
  - 推理速度: FPS / 每张图耗时

用法:
    python benchmark.py --model ../models/best.pt --test_data ../datasets/atom_v1 --test_split test
    python benchmark.py --model ../models/best.pt --test_data ../datasets/atom_v1 --compare_traditional
"""
import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# 添加路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def load_yolo_labels(label_file):
    """加载 YOLO 格式标签，返回归一化坐标 (cx, cy)。

    注意: 返回值是归一化坐标而非像素坐标; match_points 内部再乘以
    图像尺寸转换为像素。畸形行 (非 5 列) 被静默跳过。
    """
    lines = Path(label_file).read_text().strip().split('\n')
    points = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            cx, cy = float(parts[1]), float(parts[2])
            points.append((cx, cy))  # 归一化坐标
    return points


def resolve_eval_dirs(test_data, test_split="test"):
    """Return ``(images_dir, labels_dir)`` for a YOLO-format dataset.

    This is intentionally a plain tuple of two directories instead of the
    previous dynamic ``__truediv__`` object.  ``benchmark_yolo`` and
    ``benchmark_traditional`` both accept either a dataset root (which is
    joined with ``images``/``labels``) or an explicit ``(images, labels)``
    tuple, so callers can pass the two paths directly.
    """
    root = Path(test_data)
    img_dir = root / "images" / test_split
    lbl_dir = root / "labels" / test_split
    if img_dir.exists():
        return img_dir, lbl_dir
    # Simple structure: test_data/images + test_data/labels.
    return root / "images", root / "labels"


def match_points(pred_points, gt_points, img_shape, match_dist=5.0):
    """
    匹配预测点和真实点

    Parameters
    ----------
    pred_points : list of (x, y) 像素坐标
    gt_points : list of (cx, cy) 归一化坐标
    img_shape : (H, W)
    match_dist : float 匹配距离阈值 (像素)

    Returns
    -------
    tp : int 真阳性 (正确检测)
    fp : int 假阳性 (误检)
    fn : int 假阴性 (漏检)
    matched_errors : list of float 匹配对的定位误差
    """
    h, w = img_shape

    # 将 GT 归一化坐标转为像素坐标
    gt_px = [(cx * w, cy * h) for cx, cy in gt_points]

    if not pred_points or not gt_px:
        return 0, len(pred_points), len(gt_px), []

    pred = np.array(pred_points)
    gt = np.array(gt_px)

    # 计算距离矩阵
    # pred: (N, 2), gt: (M, 2)
    dist_matrix = np.sqrt(
        ((pred[:, None, 0] - gt[None, :, 0]) ** 2 +
         (pred[:, None, 1] - gt[None, :, 1]) ** 2)
    )

    # 贪心匹配
    matched_gt = set()
    matched_pred = set()
    matched_errors = []

    # 按距离排序所有 (pred_idx, gt_idx) 对
    pairs = []
    for i in range(len(pred)):
        for j in range(len(gt)):
            if dist_matrix[i, j] < match_dist:
                pairs.append((dist_matrix[i, j], i, j))
    pairs.sort()

    for dist, pi, gj in pairs:
        if pi in matched_pred or gj in matched_gt:
            # 跳过已被更近配对占用的 pred/gt (竞争匹配是常态: 一个 FP
            # 落在某 GT 的 match_dist 内, 或两个预测争抢同一 GT)。
            # 此前误写为 raise 且引用了未定义的 label_file → NameError,
            # 任何有噪声的真实数据都会让评测崩溃。
            continue
        matched_pred.add(pi)
        matched_gt.add(gj)
        matched_errors.append(dist)

    tp = len(matched_pred)
    fp = len(pred) - tp
    fn = len(gt) - len(matched_gt)

    return tp, fp, fn, matched_errors


def _ensure_yolo_dependencies():
    """YOLO 评测链路依赖仓库外的 atom_center 包; 缺失时给出明确错误。

    此前的失败模式是运行到一半抛 ModuleNotFoundError, 用户难以定位。
    """
    try:
        import atom_center  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "YOLO 评测链路依赖外部的 atom_center 包 (\"原子中心识别模型开发\"项目),\n"
            "当前环境未安装。请在该项目环境中运行, 或先只用传统算法评测。\n"
            f"原始错误: {exc}")


def benchmark_yolo(model_path, test_dir, match_dist=5.0, conf=0.5, imgsz=640,
                   max_det=3000, allowlist_path=None, device="cpu"):
    """评测 YOLOv8 模型。

    ``test_dir`` may be a dataset root or a ``(images_dir, labels_dir)`` tuple.
    """
    _ensure_yolo_dependencies()
    from atom_detector.infer.detector import AtomDetector

    if isinstance(test_dir, (tuple, list)):
        img_dir, lbl_dir = Path(test_dir[0]), Path(test_dir[1])
    else:
        test_dir = Path(test_dir)
        img_dir = test_dir / "images"
        lbl_dir = test_dir / "labels"

    if not img_dir.is_dir():
        print(f"[错误] 图像目录不存在: {img_dir}")
        return None

    # 收集测试图像
    extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}
    test_images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in extensions])

    if not test_images:
        print(f"[错误] 在 {img_dir} 中未找到测试图像")
        return None

    print(f"\n{'='*60}")
    print(f"  YOLOv8 原子检测评测")
    print(f"{'='*60}")
    print(f"  模型: {model_path}")
    print(f"  测试集: {len(test_images)} 张图像")
    print(f"  匹配距离: {match_dist} px")
    print(f"  置信度: {conf}")
    print(f"{'='*60}\n")

    # 加载模型
    detector = AtomDetector(str(model_path), conf=conf, imgsz=imgsz, refine=True,
                            max_det=max_det, allowlist_path=allowlist_path, device=device)

    total_tp, total_fp, total_fn = 0, 0, 0
    all_errors = []
    total_time = 0
    total_atoms_gt = 0

    for i, img_file in enumerate(test_images):
        # 加载标签
        label_file = lbl_dir / f"{img_file.stem}.txt"
        if not label_file.exists():
            raise ValueError(f"Missing label: {label_file}")

        gt_points = load_yolo_labels(label_file)

        # 加载图像
        img = detector._load_image_file(img_file)
        img_shape = img.shape

        # 推理计时
        t0 = time.perf_counter()
        pred_points, confidences = detector.detect(img)
        t1 = time.perf_counter()
        total_time += (t1 - t0)

        # 匹配
        tp, fp, fn, errors = match_points(pred_points, gt_points, img_shape, match_dist)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_errors.extend(errors)
        total_atoms_gt += len(gt_points)

        if (i + 1) % 10 == 0 or i == len(test_images) - 1:
            # 打印累计指标而非单张值, 反映整体进度
            print(f"  [{i+1}/{len(test_images)}] "
                  f"累计 TP={total_tp} FP={total_fp} FN={total_fn} "
                  f"耗时={total_time:.3f}s")

    # 计算指标
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    rmse = np.sqrt(np.mean(np.array(all_errors) ** 2)) if all_errors else float('nan')
    mae = np.mean(all_errors) if all_errors else float('nan')
    avg_time = total_time / len(test_images) if test_images else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0

    results = {
        'method': 'YOLOv8',
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'rmse_px': rmse,
        'mae_px': mae,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_gt': total_atoms_gt,
        'avg_time_s': avg_time,
        'fps': fps,
    }

    print(f"\n{'─'*60}")
    print(f"  YOLOv8 评测结果")
    print(f"{'─'*60}")
    print(f"  精确率 (Precision):  {precision:.4f} ({precision*100:.1f}%)")
    print(f"  检测率 (Recall):     {recall:.4f} ({recall*100:.1f}%)")
    print(f"  F1 分数:             {f1:.4f}")
    print(f"  定位 RMSE:           {rmse:.3f} px")
    print(f"  定位 MAE:            {mae:.3f} px")
    print(f"  真阳性 / 假阳性 / 漏检: {total_tp} / {total_fp} / {total_fn}")
    print(f"  平均推理时间:        {avg_time:.3f} s/图")
    print(f"  推理速度:            {fps:.1f} FPS")
    print(f"{'─'*60}")

    return results


def benchmark_traditional(test_dir, match_dist=5.0, sigma=0.8, min_dist=8):
    """评测传统 PPA 算法。

    ``test_dir`` may be a dataset root or a ``(images_dir, labels_dir)`` tuple.
    """
    from atom_detector.annotate.semi_auto_label import detect_atoms_traditional, load_image

    if isinstance(test_dir, (tuple, list)):
        img_dir, lbl_dir = Path(test_dir[0]), Path(test_dir[1])
    else:
        test_dir = Path(test_dir)
        img_dir = test_dir / "images"
        lbl_dir = test_dir / "labels"

    if not img_dir.is_dir():
        print(f"[错误] 图像目录不存在: {img_dir}")
        return None

    extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}
    test_images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in extensions])

    if not test_images:
        print(f"[错误] 在 {img_dir} 中未找到测试图像")
        return None

    print(f"\n{'='*60}")
    print(f"  传统算法 (Gaussian + NMS + COM) 评测")
    print(f"{'='*60}")
    print(f"  参数: sigma={sigma}, min_dist={min_dist}")
    print(f"  测试集: {len(test_images)} 张图像")
    print(f"{'='*60}\n")

    total_tp, total_fp, total_fn = 0, 0, 0
    all_errors = []
    total_time = 0
    total_atoms_gt = 0

    for i, img_file in enumerate(test_images):
        label_file = lbl_dir / f"{img_file.stem}.txt"
        if not label_file.exists():
            raise ValueError(f"Missing label: {label_file}")

        gt_points = load_yolo_labels(label_file)

        img = load_image(img_file)
        img_shape = img.shape

        t0 = time.perf_counter()
        pred_points = detect_atoms_traditional(img, sigma=sigma, min_dist=min_dist)
        t1 = time.perf_counter()
        total_time += (t1 - t0)

        tp, fp, fn, errors = match_points(pred_points, gt_points, img_shape, match_dist)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_errors.extend(errors)
        total_atoms_gt += len(gt_points)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    rmse = np.sqrt(np.mean(np.array(all_errors) ** 2)) if all_errors else float('nan')
    mae = np.mean(all_errors) if all_errors else float('nan')
    avg_time = total_time / len(test_images) if test_images else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0

    results = {
        'method': 'Traditional',
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'rmse_px': rmse,
        'mae_px': mae,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_gt': total_atoms_gt,
        'avg_time_s': avg_time,
        'fps': fps,
    }

    print(f"\n{'─'*60}")
    print(f"  传统算法评测结果")
    print(f"{'─'*60}")
    print(f"  精确率 (Precision):  {precision:.4f} ({precision*100:.1f}%)")
    print(f"  检测率 (Recall):     {recall:.4f} ({recall*100:.1f}%)")
    print(f"  F1 分数:             {f1:.4f}")
    print(f"  定位 RMSE:           {rmse:.3f} px")
    print(f"  定位 MAE:            {mae:.3f} px")
    print(f"  真阳性 / 假阳性 / 漏检: {total_tp} / {total_fp} / {total_fn}")
    print(f"  平均推理时间:        {avg_time:.3f} s/图")
    print(f"  推理速度:            {fps:.1f} FPS")
    print(f"{'─'*60}")

    return results


def print_comparison(yolo_results, trad_results):
    """打印对比表格"""
    if not yolo_results or not trad_results:
        return

    print(f"\n{'='*60}")
    print(f"  对比总结")
    print(f"{'='*60}")
    print(f"{'指标':<20} {'YOLOv8':<15} {'传统算法':<15} {'提升':<10}")
    print(f"{'─'*60}")

    metrics = [
        ('精确率', 'precision', '%'),
        ('检测率', 'recall', '%'),
        ('F1 分数', 'f1', ''),
        ('定位 RMSE (px)', 'rmse_px', 'px'),
        ('定位 MAE (px)', 'mae_px', 'px'),
        ('推理时间 (s)', 'avg_time_s', 's'),
        ('FPS', 'fps', ''),
    ]

    for name, key, unit in metrics:
        y_val = yolo_results[key]
        t_val = trad_results[key]

        if unit == '%':
            y_str = f"{y_val*100:.1f}%"
            t_str = f"{t_val*100:.1f}%"
            diff = (y_val - t_val) * 100
            diff_str = f"{diff:+.1f}%"
        elif unit == 'px':
            y_str = f"{y_val:.3f}"
            t_str = f"{t_val:.3f}"
            diff = y_val - t_val
            diff_str = f"{diff:+.3f}"
        elif unit == 's':
            y_str = f"{y_val:.3f}"
            t_str = f"{t_val:.3f}"
            diff = y_val - t_val
            diff_str = f"{diff:+.3f}"
        else:
            y_str = f"{y_val:.4f}"
            t_str = f"{t_val:.4f}"
            diff = y_val - t_val
            diff_str = f"{diff:+.4f}"

        print(f"  {name:<18} {y_str:<15} {t_str:<15} {diff_str:<10}")

    print(f"{'─'*60}")
    print(f"  注: RMSE/MAE/时间 越小越好; 其余越大越好")


def main():
    parser = argparse.ArgumentParser(description="原子检测模型评测")
    parser.add_argument("--model", "-m", required=True, help="YOLOv8 模型路径")
    parser.add_argument("--test_data", "-d", required=True, help="测试数据集目录")
    parser.add_argument("--test_split", default="test", help="测试集子目录名")
    parser.add_argument("--match_dist", type=float, default=5.0, help="匹配距离阈值 (px)")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理图像尺寸")
    parser.add_argument("--max_det", type=int, default=3000)
    parser.add_argument("--model_allowlist", help="已确认模型来源的 SHA-256 登记文件")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compare_traditional", action="store_true",
                        help="同时评测传统算法进行对比")
    parser.add_argument("--sigma", type=float, default=0.8, help="传统算法 sigma")
    parser.add_argument("--min_dist", type=float, default=8, help="传统算法最小间距")
    args = parser.parse_args()

    eval_dirs = resolve_eval_dirs(args.test_data, args.test_split)
    yolo_results = benchmark_yolo(args.model, eval_dirs, args.match_dist, args.conf, args.imgsz,
                                  args.max_det, args.model_allowlist, args.device)
    trad_results = None
    if args.compare_traditional:
        trad_results = benchmark_traditional(eval_dirs, args.match_dist, args.sigma, args.min_dist)

    if yolo_results and trad_results:
        print_comparison(yolo_results, trad_results)


if __name__ == "__main__":
    main()
