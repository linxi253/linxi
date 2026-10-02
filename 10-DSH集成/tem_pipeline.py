# -*- coding: utf-8 -*-
"""TEM 原位视频流水线驱动：视频 → TIFF 提取 → 漂移矫正 → 审计/曲线/预览/汇总。

独立于 DeepSeek Harness 运行。DSH 侧的模型工具只是转调本脚本。

复用的既有工具（只读引用，不修改）：
  - 视频切片工具  video_extractor  (<仓库根>\\01-视频与数据提取\\视频切片工具)
  - 漂移矫正 v7    drift_core      (<仓库根>\\02-图像处理\\drift-correction-v7)

进程协议（stdout 每行一个 JSON 对象，供上层消费）：
  {"type":"progress", ...}   阶段进度
  {"type":"info",     ...}   阶段信息/警告
  {"type":"summary",  ...}   最终成功汇总（最后一行）
  {"type":"error",    ...}   失败信息（退出码非 0）

输出目录结构（<output-root>/<视频名>/）：
  01_raw_<视频名>_stack.ome.tif        原始提取堆栈（OME-TIFF, TYX）
  01_raw_<视频名>_frames.csv           逐帧清单（含时间轴，output_file 已改写为最终路径）
  02_corrected_<视频名>.tif            漂移矫正结果 TIFF（裁剪到共同有效区）
  02_drift_shifts.csv                  每帧位移 dx/dy/模长（带时间轴）
  02_drift_curve.png                   漂移曲线图（人工复核用）
  02_audit.json                        检测质量/参数/插值/输出审计
  03_summary.json                      流水线汇总（与 stdout 的 summary 相同内容）
  03_preview_montage.png               矫正后帧的缩略拼图（6 帧抽样）
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

DRIVER_DIR = Path(__file__).resolve().parent
SUITE_ROOT = DRIVER_DIR.parent
DEFAULT_EXTRACTOR_DIR = SUITE_ROOT / "01-视频与数据提取" / "视频切片工具"
DEFAULT_DRIFT_DIR = SUITE_ROOT / "02-图像处理" / "drift-correction-v7"

# 默认堆栈解码后内存上限（GiB）；0 表示不限制
DEFAULT_MAX_STACK_GIB = 24.0
# 预览拼图边长上限
PREVIEW_TILE_MAX = 320


def emit(payload_type: str, **fields) -> None:
    print(json.dumps({"type": payload_type, **fields}, ensure_ascii=False, allow_nan=False), flush=True)


def force_utf8_stdio() -> None:
    """被外部进程捕获 stdout 时确保按 UTF-8 输出，避免中文路径乱码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def emit_info(message: str, **fields) -> None:
    emit("info", message=message, **fields)


def emit_progress(stage: str, **fields) -> None:
    emit("progress", stage=stage, **fields)


# ---------------------------------------------------------------------------
# 环境与工具加载
# ---------------------------------------------------------------------------

def load_tools(extractor_dir: Path, drift_dir: Path):
    extractor_dir = Path(extractor_dir)
    drift_dir = Path(drift_dir)
    if not (extractor_dir / "video_extractor" / "__init__.py").is_file():
        raise FileNotFoundError(f"未找到视频切片工具包目录: {extractor_dir}")
    if not (drift_dir / "drift_core.py").is_file():
        raise FileNotFoundError(f"未找到漂移矫正核心: {drift_dir}")
    for item in (extractor_dir, drift_dir):
        if str(item) not in sys.path:
            sys.path.insert(0, str(item))


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def available_memory_bytes() -> float:
    """Windows 下返回物理可用内存；其他平台返回 None。"""
    if os.name != "nt":
        return float("inf")
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return float(status.ullAvailPhys)
    except Exception:
        pass
    return float("inf")


def stage_video(video: Path, stage_in: Path) -> Path:
    """把单个视频放入 staging 目录（优先硬链接，跨卷回退复制）。"""
    stage_in.mkdir(parents=True, exist_ok=True)
    target = stage_in / video.name
    if target.exists():
        target.unlink()
    try:
        os.link(str(video), str(target))
        return target
    except OSError:
        shutil.copy2(str(video), str(target))
        return target


def rewrite_manifest_output_file(manifest_path: Path, old_prefix: str, new_prefix: str) -> None:
    """帧清单里的 output_file 原指向 staging 路径，改写为最终路径。"""
    if not manifest_path.is_file():
        return
    rows = []
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            value = row.get("output_file", "")
            if value and old_prefix and value.startswith(old_prefix):
                row["output_file"] = new_prefix + value[len(old_prefix):]
            rows.append(row)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_time_axis(frames_csv: Path, frame_count: int) -> list[float]:
    """从帧清单读取 scheduled_time_s；缺失时用帧号占位。"""
    times: list[float] = []
    if frames_csv.is_file():
        try:
            with frames_csv.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    value = row.get("scheduled_time_s", "")
                    if value not in ("", None):
                        times.append(float(value))
        except Exception:
            times = []
    if len(times) != frame_count and times:
        # 时间轴与帧数不匹配时弃用，避免错位
        times = []
    return times or [float(i) for i in range(frame_count)]


# ---------------------------------------------------------------------------
# 流程 1：TIFF 提取（单视频 staging）
# ---------------------------------------------------------------------------

def run_extraction(video: Path, output_root: Path, args) -> dict:
    from video_extractor.models import (  # 延迟导入：依赖 sys.path
        BitDepth,
        ColorMode,
        Compression,
        ExtractOptions,
        Sampling,
        SamplingMode,
    )
    from video_extractor.runner import JobRunner

    stage_root = output_root / ".stage" / f"job_{uuid.uuid4().hex[:12]}"
    stage_in = stage_root / "in"
    stage_out = stage_root / "out"
    staged = None
    results = None  # try 前置初始化：finally 及后续判断不得引用未绑定变量
    try:
        staged = stage_video(video, stage_in)
        options = ExtractOptions(
            sampling=Sampling(SamplingMode(args.sampling), args.sampling_value),
            # 漂移核心仅接受二维灰度；强制灰度并使用源色彩矩阵换算。
            color_mode=ColorMode.GRAYSCALE,
            bit_depth=BitDepth(args.bit_depth),
            compression=Compression(args.compression),
        )
        options.validate()
        runner = JobRunner(options, None)

        def report(event: dict) -> None:
            kind = event.get("kind", "")
            if kind in ("state", "output_plan"):
                emit_progress("extract", event=kind, path=event.get("path"),
                              state=event.get("state"), frames=event.get("frames"))
            elif kind in ("progress", "counting_progress"):
                emit_progress("extract", event=kind, path=event.get("path"),
                              current=event.get("current"), total=event.get("total"))
            elif kind == "video_finished":
                result = event.get("result", {})
                emit_progress("extract", event=kind, path=event.get("path"),
                              state=result.get("state"), frames=result.get("frame_count"))
            elif kind in ("info", "warning"):
                emit_info(event.get("message", ""), severity=kind, path=event.get("path"))

        emit_progress("extract", event="batch_started")
        results = runner.run_batch(stage_in, stage_out, report)
    finally:
        if staged is not None:
            try:
                staged.unlink(missing_ok=True)
            except OSError:
                pass
        # 只有当提取本身未完成（无输出可移动）时才随手清理 staging；
        # 成功时产物还在 stage 里，由调用方移出后再清理。
        if results is None or any(item.state.value != "completed" for item in results):
            shutil.rmtree(stage_root, ignore_errors=True)
            try:
                stage_root.parent.rmdir()
            except OSError:
                pass

    if results is None or any(item.state.value != "completed" for item in results):
        failed = next((item for item in results if item.state.value == "failed"), results[0] if results else None)
        raise RuntimeError(
            "视频提取失败: " + (failed.error if failed and failed.error else "无结果")
        )
    result = results[0]
    raw_tif = Path(result.output_path)
    # 清单按“视频名_frames.csv”命名（与 TIFF 名不同），以视频 stem 为准，并带 glob 兜底
    manifest = raw_tif.parent / f"{video.stem}_frames.csv"
    if not manifest.is_file():
        candidates = sorted(raw_tif.parent.glob(f"{video.stem}*_frames.csv"))
        manifest = candidates[0] if candidates else manifest
    if not raw_tif.is_file():
        raise RuntimeError(f"提取完成但未找到输出文件: {raw_tif}")
    return {"stage_root": stage_root, "output": raw_tif, "manifest": manifest,
            "result": result, "info": result.video_info or {}, "spec": result.frame_spec or {}}


def parse_crop_rect(value: str, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """解析裁剪区域 "x0,y0,x1,y1"（0~1 分数坐标，相对整帧）。

    例：左侧三分之一 = "0,0,0.3333,1"。返回像素切片 (x0,y0,x1,y1)。
    """
    parts = [item.strip() for item in str(value).split(",")]
    if len(parts) != 4:
        raise ValueError(f"crop-rect 需要 4 个分数值 x0,y0,x1,y1（0~1），收到: {value!r}")
    try:
        fx0, fy0, fx1, fy1 = (float(item) for item in parts)
    except ValueError as exc:
        raise ValueError(f"crop-rect 含非数值: {value!r}") from exc
    height, width = shape
    for fraction in (fx0, fy0, fx1, fy1):
        if not (0.0 <= fraction <= 1.0):
            raise ValueError(f"crop-rect 分量必须在 0~1 之间: {value!r}")
    x0 = int(round(fx0 * width))
    y0 = int(round(fy0 * height))
    x1 = int(round(fx1 * width))
    y1 = int(round(fy1 * height))
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"crop-rect 区域非法（需 x0<x1, y0<y1）: {value!r}")
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# 流程 2：漂移矫正 + 审计
# ---------------------------------------------------------------------------

def run_highlevel_pipeline(video: Path, output_root: Path, args) -> dict:
    from video_extractor.runner import VIDEO_EXTENSIONS

    video = Path(video).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video}")
    if video.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"不支持的视频格式 {video.suffix}，支持: {', '.join(sorted(VIDEO_EXTENSIONS))}")

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    extraction = run_extraction(video, output_root, args)
    raw_tif = extraction["output"]
    manifest = extraction["manifest"]
    result = extraction["result"]

    # 移动提取产物到最终目录
    stem = video.stem
    final_dir = output_root / stem
    if final_dir.exists() and any(final_dir.iterdir()):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        final_dir = output_root / f"{stem}_{stamp}"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_tif = final_dir / f"01_raw_{stem}_stack.ome.tif"
    final_manifest = final_dir / f"01_raw_{stem}_frames.csv"
    stage_prefix = str(raw_tif.parent)
    os.replace(str(raw_tif), str(final_tif))
    if manifest.is_file():
        os.replace(str(manifest), str(final_manifest))
    rewrite_manifest_output_file(final_manifest, stage_prefix, str(final_dir))
    # 产物已移出，清理本任务 staging
    shutil.rmtree(extraction["stage_root"], ignore_errors=True)
    try:
        extraction["stage_root"].parent.rmdir()
    except OSError:
        pass

    emit_info(f"提取完成：{result.frame_count} 帧 → {final_tif}")

    # ---- 漂移矫正 ----
    from drift_core import (
        DEFAULT_SIFT_PARAMS,
        DetectionQualityError,
        DriftCorrector,
        DriftDetector,
        TiffIO,
        correct_and_save,
    )

    inspect = TiffIO.inspect_stack(str(final_tif))
    estimated_bytes = int(inspect.get("estimated_bytes") or 0)
    limit_bytes = args.max_stack_gib * 1024**3 if args.max_stack_gib > 0 else None
    if limit_bytes and estimated_bytes > limit_bytes:
        raise RuntimeError(
            f"堆栈解码后约需 {estimated_bytes / 1024**3:.2f} GiB，超过上限 "
            f"{args.max_stack_gib:.1f} GiB（--max-stack-gib 可调整；建议降低采样帧率）"
        )
    if estimated_bytes > available_memory_bytes() * 0.7:
        raise RuntimeError(
            f"堆栈解码后约需 {estimated_bytes / 1024**3:.2f} GiB，接近可用内存的 70%；"
            "建议降低采样帧率或增大可用内存"
        )
    emit_progress("correct", event="read", frames=inspect["n_frames"], size=list(inspect["shape"]))
    frames, meta = TiffIO.read_stack(str(final_tif), max_bytes=None)

    # 可选的区域裁剪：例如界面录制只需 STEM 图像区域（左侧 1/3）
    crop_rect = None
    original_n_frames = len(frames)
    if args.crop_rect:
        crop_rect = parse_crop_rect(args.crop_rect, frames[0].shape[:2])
        x0, y0, x1, y1 = crop_rect
        frames = [frame[y0:y1, x0:x1] for frame in frames]
        meta["shape"] = (frames[0].shape[0], frames[0].shape[1])
        meta["size"] = (frames[0].shape[1], frames[0].shape[0])
        emit_info(f"已按 crop-rect 截取 x[{x0}:{x1}] y[{y0}:{y1}] → 形状 {frames[0].shape}")

    # 剔除首尾空白/无纹理帧（采集开始/结束的黑帧或渐隐帧会导致首尾帧对无法匹配）
    trim_leading = 0
    trim_trailing = 0
    if args.trim_blank_edges and len(frames) > 2:
        import cv2

        # 与漂移检测同源的 SIFT 判定：无特征点的帧不参与配准
        detector = cv2.SIFT_create(nfeatures=5000, nOctaveLayers=3,
                                   contrastThreshold=0.04, edgeThreshold=10.0, sigma=1.6)

        def _featureless(frame) -> bool:
            if float(np.std(frame)) < args.blank_std_threshold:
                return True
            return not detector.detect(np.ascontiguousarray(frame), None)

        max_trim = max(2, int(len(frames) * 0.20))  # 保守上限：最多剔除 20% 边缘帧
        lead = 0
        while lead < len(frames) and lead < max_trim and _featureless(frames[lead]):
            lead += 1
        tail = len(frames)
        while tail > lead and (len(frames) - tail) < max_trim and _featureless(frames[tail - 1]):
            tail -= 1
        if lead or tail < len(frames):
            trim_leading = lead
            trim_trailing = len(frames) - tail
            emit_info(
                f"已剔除首尾空白/无纹理帧：开头 {trim_leading} 帧，结尾 {trim_trailing} 帧，"
                f"保留 {tail - lead} 帧"
            )
            frames = frames[lead:tail]
            meta["n_frames"] = len(frames)

    # 组合自定义 SIFT/质量门控参数（仅包含显式提供的项；核心会做范围校验）
    sift_params: dict = {}
    for key, flag in (
        ("nfeatures", "sift_nfeatures"),
        ("noctave_layers", "sift_noctave_layers"),
        ("contrast_threshold", "sift_contrast_threshold"),
        ("edge_threshold", "sift_edge_threshold"),
        ("sigma", "sift_sigma"),
        ("ratio_threshold", "sift_ratio_threshold"),
        ("ransac_threshold", "sift_ransac_threshold"),
        ("min_matches", "sift_min_matches"),
        ("min_inliers", "sift_min_inliers"),
        ("min_inlier_ratio", "sift_min_inlier_ratio"),
        ("max_pair_residual", "sift_max_pair_residual"),
        ("min_valid_pair_ratio", "sift_min_valid_pair_ratio"),
        ("max_interpolation_gap", "sift_max_interpolation_gap"),
    ):
        value = getattr(args, flag, None)
        if value is not None:
            sift_params[key] = value

    try:
        detection = DriftDetector.detect_drift(
            frames,
            progress_callback=lambda ratio: emit_progress("correct", event="detect", ratio=round(float(ratio), 3)),
            skip_interval=args.skip_interval,
            params=sift_params,
        )
        detection_tier = "strict"
        detection_params = sift_params
    except DetectionQualityError as first_error:
        # 自动两档：严格档未通过 → 用放宽档重试（与 GUI 放宽阈值行为一致），并在审计中标注
        if not args.relax_on_fail:
            raise
        RELAXED = {
            "ratio_threshold": 0.8,
            "min_matches": 3,
            "min_inliers": 3,
            "min_valid_pair_ratio": 0.7,
            "max_interpolation_gap": 15,
        }
        relaxed_params = {key: value for key, value in RELAXED.items() if key not in sift_params}
        relaxed_params = {**sift_params, **relaxed_params}
        emit_info(f"严格档检测未通过（{first_error}），已自动放宽 SIFT 参数重试: {relaxed_params}")
        try:
            detection = DriftDetector.detect_drift(
                frames,
                progress_callback=lambda ratio: emit_progress("correct", event="detect", ratio=round(float(ratio), 3)),
                skip_interval=args.skip_interval,
                params=relaxed_params,
            )
            detection_tier = "relaxed"
            detection_params = relaxed_params
        except DetectionQualityError as error:
            # 记录诊断：为什么失败、用了什么参数、建议怎么放宽
            diagnostic = {
                "schema_version": 1,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "status": "detection_failed",
                "error": str(error),
                "video": str(video),
                "raw_tiff": str(final_tif),
                "frames_csv": str(final_manifest),
                "n_frames": len(frames),
                "shape": list(frames[0].shape),
                "dtype": str(frames[0].dtype),
                "requested_params": {**DEFAULT_SIFT_PARAMS, **relaxed_params},
                "advice": (
                    "检测未通过质量门控（工具设计为拒绝不可靠结果，不生成伪装零漂移）。"
                    "可尝试：增大 --sift-max-interpolation-gap（允许更多插值段）、"
                    "降低 --sift-min-valid-pair-ratio（放宽有效帧对比例要求）、"
                    "提高 --sift-ratio-threshold（放宽 Lowe ratio）、"
                    "提高 --sift-max-pair-residual 与 --sift-ransac-threshold（容忍更大位移）、"
                    "或使用 --skip-interval 更大的跳帧间隔；放宽后请人工复核漂移曲线。"
                ),
            }
            diagnostic_path = final_dir / "02_detection_failed.json"
            diagnostic_path.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            raise RuntimeError(
                f"漂移检测未通过质量门控（严格档与放宽档均失败）：{error}\n"
                f"诊断已写入 {diagnostic_path}\n"
                "提示：可进一步放宽 SIFT 质量参数后重试（见诊断文件建议）；对周期性晶格/快速运动数据尤其需要人工复核。"
            ) from error
    shifts_x, shifts_y = detection.shifts_x, detection.shifts_y

    corrected_tif = final_dir / f"02_corrected_{stem}.tif"
    written = correct_and_save(
        frames, shifts_x, shifts_y, meta, str(corrected_tif),
        progress_callback=lambda ratio: emit_progress(
            "correct", event="write", ratio=round(float(ratio), 3),
        ),
        overwrite=False,
        crop_mode=args.crop_mode,
    )
    emit_info(f"矫正完成：{written} 帧 → {corrected_tif}")

    # ---- 位移 CSV ----
    full_times = read_time_axis(final_manifest, original_n_frames)
    times = full_times[trim_leading:trim_leading + len(frames)] if (trim_leading or trim_trailing) else full_times
    shifts_csv = final_dir / "02_drift_shifts.csv"
    with shifts_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "time_s", "dx_px", "dy_px", "magnitude_px"])
        for index, (dx, dy) in enumerate(zip(shifts_x, shifts_y)):
            writer.writerow([index, f"{times[index]:.6f}", f"{float(dx):.4f}",
                             f"{float(dy):.4f}", f"{float(math.hypot(dx, dy)):.4f}"])

    # ---- 漂移曲线 ----
    curve_png = final_dir / "02_drift_curve.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = times if len(times) == len(frames) else list(range(len(frames)))
        mags = [float(math.hypot(dx, dy)) for dx, dy in zip(shifts_x, shifts_y)]
        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
        ax.plot(x, list(map(float, shifts_x)), label="dx (px)", linewidth=1.2)
        ax.plot(x, list(map(float, shifts_y)), label="dy (px)", linewidth=1.2)
        ax.plot(x, mags, label="|d| (px)", linewidth=1.4, alpha=0.75)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("shift (px)")
        ax.set_title(f"Drift curve - {stem}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(curve_png)
        plt.close(fig)
    except Exception as exc:
        emit_info(f"漂移曲线生成失败（不影响结果）: {exc}", severity="warning")
        curve_png = None

    # ---- 预览拼图（矫正后 6 帧抽样） ----
    montage_png = final_dir / "03_preview_montage.png"
    try:
        indices = [int(round(i * (len(frames) - 1) / 5)) for i in range(6)] if len(frames) > 1 else [0]
        tiles = []
        for index in indices:
            corrected = DriftCorrector.correct_single_frame(frames[index], float(shifts_x[index]),
                                                            float(shifts_y[index]))
            tile = cv2_downscale(corrected)
            tiles.append(tile)
        h = max(t.shape[0] for t in tiles)
        row = []
        for tile in tiles:
            padded = np.zeros((h, min(tile.shape[1], PREVIEW_TILE_MAX)), dtype=tile.dtype)
            padded[:tile.shape[0], :tile.shape[1]] = tile
            row.append(padded)
        grid = np.concatenate(row, axis=1)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(14, 3), dpi=120)
        ax.imshow(grid, cmap="gray", aspect="equal")
        ax.set_title(f"Corrected preview - {stem} (frames {indices[0]}-{indices[-1]})", fontsize=9)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(montage_png)
        plt.close(fig)
    except Exception as exc:
        emit_info(f"预览拼图生成失败（不影响结果）: {exc}", severity="warning")
        montage_png = None

    # ---- 审计 JSON ----
    audit = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tool_versions": {
            "video_extractor": "4.1.1",
            "drift_core": getattr(sys.modules.get("drift_core"), "__version__", "7.0.3"),
        },
        "video": {
            "path": str(video),
            **{k: extraction["info"].get(k) for k in
               ("width", "height", "duration_s", "average_fps", "nominal_fps", "frame_count",
                "codec", "pixel_format", "color_family", "bits_per_sample", "is_variable_fps")
               if extraction["info"].get(k) is not None},
        },
        "extraction": {
            "sampling_mode": args.sampling,
            "sampling_value": args.sampling_value,
            "color_mode": "grayscale",
            "bit_depth": args.bit_depth,
            "compression": args.compression,
            "frame_count": result.frame_count,
            "estimated_bytes": result.estimated_bytes,
            "tiff_container": result.tiff_container,
            "raw_tiff": str(final_tif),
            "frames_csv": str(final_manifest),
            "crop_rect": args.crop_rect,
        },
        "detection": {
            "skip_interval": args.skip_interval,
            "tier": detection_tier,
            "n_frames": len(frames),
            "original_n_frames": original_n_frames,
            "blank_trimmed": {"leading": trim_leading, "trailing": trim_trailing},
            "shape": list(frames[0].shape),
            "valid_pair_ratio": detection.valid_pair_ratio,
            "matched_pairs": detection.matched_pairs,
            "total_pairs": detection.total_pairs,
            "interpolated_pairs": detection.interpolated_pairs,
            "warnings": list(detection.warnings),
            "params": {**DEFAULT_SIFT_PARAMS, **detection_params},
            "pairs": [
                {
                    "source_index": q.source_index, "target_index": q.target_index,
                    "keypoints_source": q.keypoints_source, "keypoints_target": q.keypoints_target,
                    "good_matches": q.good_matches, "inliers": q.inliers,
                    "inlier_ratio": q.inlier_ratio,
                    "median_residual": None if not math.isfinite(q.median_residual) else q.median_residual,
                    "dx": q.dx, "dy": q.dy, "status": q.status, "message": q.message,
                }
                for q in detection.pair_quality
            ],
        },
        "correction": {
            "output": str(corrected_tif),
            "written_frames": written,
            "crop_mode": args.crop_mode,
            "max_abs_shift_x": float(max(abs(float(v)) for v in shifts_x)),
            "max_abs_shift_y": float(max(abs(float(v)) for v in shifts_y)),
            "max_abs_shift_magnitude": float(max(math.hypot(float(a), float(b)) for a, b in zip(shifts_x, shifts_y))),
            "drift_shifts_csv": str(shifts_csv),
            "drift_curve_png": str(curve_png) if curve_png else None,
        },
        "preview_montage_png": str(montage_png) if montage_png else None,
        "outputs": {
            "raw_tiff": str(final_tif),
            "frames_csv": str(final_manifest),
            "corrected_tiff": str(corrected_tif),
            "drift_shifts_csv": str(shifts_csv),
            "drift_curve_png": str(curve_png) if curve_png else None,
            "audit_json": "",
            "summary_json": "",
            "preview_montage_png": str(montage_png) if montage_png else None,
        },
        "elapsed_seconds": round(time.time() - started, 2),
        "warnings": ((["严格档检测未通过，已自动放宽 SIFT 参数并成功检测（见检测 tier/params）"]
                      if detection_tier == "relaxed" else [])
                     + list(detection.warnings) + extraction.get("warnings", [])),
    }
    audit_path = final_dir / "02_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    audit["outputs"]["audit_json"] = str(audit_path)

    summary_path = final_dir / "03_summary.json"
    # summary 保持紧凑：逐帧对明细只在 audit 中，不重复携带（避免巨大载荷）
    detection_summary = {key: value for key, value in audit["detection"].items() if key != "pairs"}
    summary = {
        "status": "completed",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "video": str(video),
        "output_dir": str(final_dir),
        "extraction": audit["extraction"],
        "detection": detection_summary,
        "correction": audit["correction"],
        "outputs": audit["outputs"],
        "warnings": audit["warnings"],
        "elapsed_seconds": audit["elapsed_seconds"],
        "result_paths": [str(p) for p in (final_tif, corrected_tif, shifts_csv,
                                          curve_png, audit_path, summary_path, montage_png) if p],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    audit["outputs"]["summary_json"] = str(summary_path)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def cv2_downscale(frame):
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    scale = min(1.0, PREVIEW_TILE_MAX / max(height, width))
    if scale >= 1.0:
        return np.ascontiguousarray(frame)
    resized = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))),
                         interpolation=cv2.INTER_AREA)
    return resized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TEM 视频 → TIFF 提取 → 漂移矫正 → 审计/曲线/汇总")
    parser.add_argument("--video", required=True, help="原位视频文件绝对路径")
    parser.add_argument("--output-root", required=True, help="输出根目录（每次调用时指定）")
    parser.add_argument("--sampling", choices=["all", "target_fps", "interval"], default="target_fps")
    parser.add_argument("--sampling-value", type=float, default=5.0, help="目标帧率(fps)或间隔(秒)")
    parser.add_argument("--skip-interval", type=int, default=1, help="漂移检测跳帧间隔（1=逐帧对）")
    parser.add_argument("--sift-nfeatures", type=int, default=None,
                        help="SIFT 特征点数（默认 5000；0=全部）")
    parser.add_argument("--sift-noctave-layers", type=int, default=None, help="SIFT Octave 层数（默认 3）")
    parser.add_argument("--sift-contrast-threshold", type=float, default=None, help="SIFT 对比度阈值（默认 0.04）")
    parser.add_argument("--sift-edge-threshold", type=float, default=None, help="SIFT 边缘阈值（默认 10）")
    parser.add_argument("--sift-sigma", type=float, default=None, help="SIFT Sigma（默认 1.6）")
    parser.add_argument("--sift-ratio-threshold", type=float, default=None,
                        help="Lowe ratio 阈值（默认 0.75；越大越宽松）")
    parser.add_argument("--sift-ransac-threshold", type=float, default=None,
                        help="RANSAC 阈值（默认 3.0；越大越容忍位移）")
    parser.add_argument("--sift-min-matches", type=int, default=None,
                        help="最少匹配点（默认 4；>=3）")
    parser.add_argument("--sift-min-inliers", type=int, default=None,
                        help="最少内点（默认 6；>=3）")
    parser.add_argument("--sift-min-inlier-ratio", type=float, default=None,
                        help="内点比例下限（默认 0.10）")
    parser.add_argument("--sift-max-pair-residual", type=float, default=None,
                        help="帧对内点中位残差上限 px（默认 3.0）")
    parser.add_argument("--sift-min-valid-pair-ratio", type=float, default=None,
                        help="可靠帧对比例下限（默认 0.80；困难数据可降低到 0.4-0.5）")
    parser.add_argument("--sift-max-interpolation-gap", type=int, default=None,
                        help="允许的连续失败帧对插值长度（默认 2；可放宽到 5-10）")
    parser.add_argument("--relax-on-fail", action="store_true", default=True,
                        help="严格档检测失败时自动用放宽档重试（ratio 0.8, min_matches/min_inliers 3, "
                             "最小有效帧对比例 0.7, 最大插值 15）")
    parser.add_argument("--no-relax-on-fail", dest="relax_on_fail", action="store_false",
                        help="关闭自动放宽档，严格档失败即报错")
    parser.add_argument("--trim-blank-edges", action="store_true", default=True,
                        help="检测前剔除首尾空白帧（std 低于阈值）")
    parser.add_argument("--no-trim-blank-edges", dest="trim_blank_edges", action="store_false",
                        help="关闭首尾空白帧剔除")
    parser.add_argument("--blank-std-threshold", type=float, default=2.0,
                        help="空白帧判定阈值（帧标准差，默认 2.0）")
    parser.add_argument("--bit-depth", choices=["source", "8", "16"], default="source")
    parser.add_argument("--compression", choices=["none", "deflate"], default="none")
    parser.add_argument("--crop-mode", choices=["crop", "keep"], default="crop",
                        help="crop=裁剪至共同有效区域；keep=保持原尺寸")
    parser.add_argument("--crop-rect", default=None,
                        help="提取后按区域裁剪再检测/矫正，4 个 0~1 分数坐标 x0,y0,x1,y1；"
                             "如界面录制只取 STEM 图像：0,0,0.3333,1（左侧三分之一）")
    parser.add_argument("--max-stack-gib", type=float, default=DEFAULT_MAX_STACK_GIB,
                        help="堆栈解码内存上限 GiB（0=不限制）")
    parser.add_argument("--extractor-dir", default=str(DEFAULT_EXTRACTOR_DIR))
    parser.add_argument("--drift-dir", default=str(DEFAULT_DRIFT_DIR))
    return parser


def main(argv=None) -> int:
    force_utf8_stdio()
    args = build_parser().parse_args(argv)
    try:
        load_tools(args.extractor_dir, args.drift_dir)
        summary = run_highlevel_pipeline(Path(args.video), Path(args.output_root), args)
        emit("summary", **summary)
        return 0
    except Exception as error:
        emit("error", message=str(error), traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
