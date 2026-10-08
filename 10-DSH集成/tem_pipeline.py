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
  01_raw_<视频名>_stack.tif            原始提取堆栈（ImageJ 兼容普通 TIFF，TYX）
  01_raw_<视频名>_frames.csv           逐帧清单（含时间轴，output_file 已改写为最终路径）
  02_corrected_<视频名>.tif            漂移矫正结果 TIFF（裁剪到共同有效区）
  02_drift_shifts.csv                  每帧位移 dx/dy/模长（带时间轴）
  02_drift_curve.png                   漂移曲线图（人工复核用）
  02_audit.json                        检测质量/参数/插值/输出审计
  03_summary.json                      流水线汇总（与 stdout 的 summary 相同内容）
  03_preview_montage.png               矫正后帧的缩略拼图（6 帧抽样）

格式说明：提取堆栈由 video_extractor 写出，是 **ImageJ 兼容的未压缩普通 TIFF**
（writers.py 明确拒绝 OME 并要求 is_imagej），因此文件名不带 .ome.tif 后缀。
"""

from __future__ import annotations

import argparse
import contextlib
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


# JSON 协议输出通道：**每次 main() 调用时**绑定到当时的 sys.stdout。
# 不在模块 import 时固定 —— 否则在 A 流 import、在 B 流调用 main 时，事件会
# 跑到 A 流（回归 2026-10-03 R17）。
_JSON_OUT = None


def _clean_json_value(value):
    """递归清洗非有限值并把 numpy 标量/数组转成原生类型。

    ``json.dumps(allow_nan=False)`` 遇到 NaN/Inf 会抛 ValueError，而进度/质量
    指标（valid_pair_ratio、max_abs_shift 等）在退化数据上可能非有限。只清洗
    顶层不够：detection/summary 里还有嵌套 dict/list，numpy 标量也不是原生
    类型。若不彻底清洗，emit() 会在**已写出部分产物**后抛异常，用户既拿不到
    summary 也拿不到干净的 error。

    语义（如实记录）：**数据中不确定的数值转为 JSON null**，即 NaN/Inf 一律
    变成 null，调用方无法区分"确实是 NaN"与"缺失"。选择 null 而非字符串，
    是为了让 ``allow_nan=False`` 的严格 JSON 仍然可解析；需要区分时看
    ``warnings``/``status`` 字段。

    覆盖 0 维数组（``np.array(nan)``）：``.tolist()`` 返回标量而非列表，
    直接迭代会 TypeError，因此先判断维数。
    """
    if isinstance(value, dict):
        return {key: _clean_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _clean_json_value(value.item())
        return [_clean_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):          # numpy 标量 → 原生
        value = value.item()
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    return value


def clean_json_payload(payload: dict) -> dict:
    """清洗并返回一份可安全 ``json.dumps(..., allow_nan=False)`` 的副本。

    emit() 之外，audit/summary 落盘也必须走这里：那里同样用 allow_nan=False，
    指标含 NaN/Inf 时会直接抛异常并丢掉整份报告（回归 2026-10-03 R17）。
    """
    return _clean_json_value(payload)


def _json_channel() -> object:
    """当前应写入 JSON 的流；未显式绑定时用当时的 sys.stdout。"""
    return _JSON_OUT if _JSON_OUT is not None else sys.stdout


def emit(payload_type: str, **fields) -> None:
    payload = _clean_json_value({"type": payload_type, **fields})
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False),
          file=_json_channel(), flush=True)


@contextlib.contextmanager
def json_output_channel(stream=None):
    """把 JSON 通道绑定到本此调用当前的 stdout，并保证退出时还原。

    契约（与 :func:`third_party_output_on_stderr` 一致，不再保留"永久
    divert"那种隐式全局状态）：

    * 进入时记录调用方的 stdout 与旧的 ``_JSON_OUT``；
    * 作用域内 ``emit`` 写入该 stdout；
    * 期间上游第三方代码的 print 由
      :func:`third_party_output_on_stderr` 单独导到 stderr，
      **不会**污染 JSON 通道；
    * 退出时（成功或异常）恢复 ``sys.stdout`` 与 ``_JSON_OUT``，因此同一进程
      内连续两次 ``main()`` 可以把事件分别送到各自的字符串流。
    """
    global _JSON_OUT
    saved_stdout = sys.stdout
    saved_channel = _JSON_OUT
    _JSON_OUT = sys.stdout if stream is None else stream
    try:
        yield _JSON_OUT
    finally:
        sys.stdout = saved_stdout
        _JSON_OUT = saved_channel


@contextlib.contextmanager
def third_party_output_on_stderr():
    """在调用上游第三方代码期间把它们的 stdout 导向 stderr。

    作用域内结束后**恢复**原来的 sys.stdout —— 早期实现永久重绑定，重复调用
    或同进程内其他消费者会被污染。
    """
    saved = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = saved


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


def rewrite_manifest_output_file(manifest_path: Path, old_file: str, new_file: str,
                                 original_input: Path | None = None,
                                 staged_input: Path | None = None) -> None:
    """把清单里的路径从 staging 改写为**可长期追溯**的最终位置。

    * ``output_file``：按**整个文件路径**精确映射到最终 TIFF。上游每行写的是
      当时真实的输出文件（staging 下的 ``<stem>.tif``），本流水线会重命名，
      只替换目录前缀会留下指向不存在名字的路径，因此不做前缀后门
      （``clip.tif.extra`` 这类同前缀但不同文件的值必须保持不变）。
    * ``input_path``：上游记录的是 staging 里的**硬链接/副本**，而该链接在提取
      返回前就被删除，用户拿到的清单会指向已消失的路径。这里改回用户原始视频
      路径；``input_sha256`` 本来就是解码前对输入算的，内容与原始文件一致，
      因此"原始路径 + 原始哈希"才是可追溯的组合。

    读取端用 ``utf-8-sig`` 兼容带 BOM 的清单。
    """
    if not manifest_path.is_file():
        return
    rows = []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            value = row.get("output_file", "")
            if value and old_file and os.path.normcase(value) == os.path.normcase(old_file):
                row["output_file"] = new_file
            if (original_input is not None and staged_input is not None
                    and "input_path" in row):
                current = row.get("input_path", "")
                if current and os.path.normcase(current) == os.path.normcase(str(staged_input)):
                    row["input_path"] = str(original_input)
            rows.append(row)
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_time_axis(frames_csv: Path, frame_count: int) -> list[float]:
    """从帧清单读取 scheduled_time_s；缺失时用帧号占位。

    上游清单以 ``utf-8-sig``（带 BOM）写出，读取端必须用同一编码，否则首列键名
    会带上 BOM 前缀而读不到值。
    """
    times: list[float] = []
    if frames_csv.is_file():
        try:
            with frames_csv.open("r", newline="", encoding="utf-8-sig") as handle:
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


def build_sampling(mode: str, value: float | None):
    """把 CLI 的 sampling 参数适配成 video_extractor 的 Sampling。

    ``SamplingMode.ALL`` **不接受**采样值（传了会 validate 失败），而 CLI 的
    ``--sampling-value`` 有默认 5.0；因此 all 模式下必须显式传 ``value=None``，
    不能把默认值一并塞进去（回归 2026-10-03 R11）。
    """
    from video_extractor.models import Sampling, SamplingMode

    sampling_mode = SamplingMode(mode)
    if sampling_mode is SamplingMode.ALL:
        if value not in (None, 0, 0.0):
            emit_info(
                f"sampling=all 会提取全部帧，已忽略 --sampling-value={value:g}"
            )
        return Sampling(mode=sampling_mode, value=None)
    if value is None:
        raise ValueError(f"sampling={mode} 需要 --sampling-value 指定正数")
    return Sampling(mode=sampling_mode, value=float(value))


def build_extract_options(args) -> "object":
    """按 video_extractor 现有 API 组装 ExtractOptions。

    该 dataclass **没有** compression 字段：上游视频工具固定写未压缩的
    ImageJ TIFF（见 writers.py 的 is_imagej/is_ome 校验），压缩语义属于本流水线
    自己的最终校正 TIFF 写入（见 write_corrected_stack），因此不在这里传递，
    也不为适配器新增无用的 Compression 枚举（回归 2026-10-03 R11）。
    """
    from video_extractor.models import BitDepth, ColorMode, ExtractOptions

    options = ExtractOptions(
        sampling=build_sampling(args.sampling, args.sampling_value),
        # 漂移核心仅接受二维灰度；强制灰度并使用源色彩矩阵换算。
        color_mode=ColorMode.GRAYSCALE,
        bit_depth=BitDepth(args.bit_depth),
    )
    options.validate()
    return options


def write_corrected_stack(frames, shifts_x, shifts_y, meta, output_path: Path, args):
    """写出最终校正 TIFF，并在此处应用 ``--compression``。

    ``--compression`` 的语义只作用在本流水线自己的产物上：
      none    → 未压缩（默认，与历史行为一致）
      deflate → zlib 无损压缩（只改容器编码，不改像素）

    实现直接复用 drift_core 的安全 writer（``correct_and_save`` →
    ``TiffIO.write_stack``），因此同源拒绝、已有目标拒绝、磁盘与 BigTIFF
    策略、resolution/时间等 ImageJ 元数据、crop/n_frames/shape/dtype 校验、
    临时文件清理后发布、以及"矫正生效"防空转校验全部照旧生效。
    早期版本另写了一套 tifffile 直写，绕开了上述全部保护（缺目标存在检查、
    缺同源检查、失败留 final、丢元数据、固定 bigtiff=False），已删除。
    """
    compression = getattr(args, "compression", "none")
    if compression not in ("none", "deflate"):
        raise ValueError(f"不支持的 compression: {compression!r}（可选 none/deflate）")
    from drift_core import correct_and_save

    return correct_and_save(
        frames, shifts_x, shifts_y, meta, str(output_path),
        progress_callback=lambda ratio: emit_progress(
            "correct", event="write", ratio=round(float(ratio), 3),
        ),
        overwrite=False,
        crop_mode=args.crop_mode,
        compression=compression,
    )


# ---------------------------------------------------------------------------
# 流程 1：TIFF 提取（单视频 staging）
# ---------------------------------------------------------------------------

def _discard_stage(stage_root: Path) -> None:
    """只清理**本次**调用的 staging 目录，绝不动其它 job 或原始输入。"""
    shutil.rmtree(stage_root, ignore_errors=True)
    try:
        stage_root.parent.rmdir()      # .stage/ 为空时一并收掉
    except OSError:
        pass


def run_extraction(video: Path, output_root: Path, args) -> dict:
    """提取单视频到 staging，成功后把 staging 所有权交给调用方。

    边界契约（回归 2026-10-03 R17）：``run_batch`` 返回 ``None``、空列表、
    状态非 completed、状态 completed 但 ``output_path`` 缺失/文件不存在 ——
    这些情况一律**清理本次 stage** 并抛出清晰的 ``RuntimeError``，不留
    ``.stage/job_*`` 残留，也不触碰其它 job 或原始输入。
    """
    from video_extractor.runner import JobRunner

    stage_root = output_root / ".stage" / f"job_{uuid.uuid4().hex[:12]}"
    stage_in = stage_root / "in"
    stage_out = stage_root / "out"
    staged = None
    results = None  # try 前置初始化：finally 及后续判断不得引用未绑定变量
    owned_stage = True             # 成功后交由调用方清理，置 False
    try:
        staged = stage_video(video, stage_in)
        options = build_extract_options(args)
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
        with third_party_output_on_stderr():
            results = runner.run_batch(stage_in, stage_out, report)

        # ---- 统一校验：None / 空 / 非 completed / 输出缺失 都归为失败 ----
        if not results:
            raise RuntimeError(
                f"视频提取未产生任何结果（run_batch 返回 "
                f"{'None' if results is None else '空列表'}）: {video.name}"
            )
        failed = next((item for item in results
                       if item.state.value in ("failed", "cancelled")), None)
        if failed is not None:
            detail = failed.error or f"状态 {failed.state.value}"
            raise RuntimeError(f"视频提取失败（{failed.state.value}）: {detail}")
        pending = [item for item in results if item.state.value != "completed"]
        if pending:
            raise RuntimeError(
                f"视频提取未完成，状态: {[item.state.value for item in pending]}"
            )
        result = results[0]
        output_value = getattr(result, "output_path", None)
        if not output_value:
            raise RuntimeError("提取报告完成但未提供 output_path，结果不可用")
        raw_tif = Path(output_value)
        if not raw_tif.is_file():
            raise RuntimeError(f"提取完成但输出文件不存在: {raw_tif}")

        # 清单按“视频名_frames.csv”命名（与 TIFF 名不同），以视频 stem 为准，并带 glob 兜底
        manifest = raw_tif.parent / f"{video.stem}_frames.csv"
        if not manifest.is_file():
            candidates = sorted(raw_tif.parent.glob(f"{video.stem}*_frames.csv"))
            manifest = candidates[0] if candidates else manifest
        owned_stage = False        # 所有权移交调用方
        return {"stage_root": stage_root, "output": raw_tif, "manifest": manifest,
                "result": result, "info": result.video_info or {},
                "spec": result.frame_spec or {},
                "staged_input": staged,
                "warnings": list(getattr(result, "warnings", []) or [])}
    finally:
        if staged is not None:
            try:
                staged.unlink(missing_ok=True)
            except OSError:
                pass
        # 只有本次提取未成功（无输出可移动）时才清理 staging；
        # 成功时产物还在 stage 里，由调用方移出后再清理。
        if owned_stage:
            _discard_stage(stage_root)


def parse_crop_fractions(value: str) -> tuple[float, float, float, float]:
    """解析裁剪区域 "x0,y0,x1,y1" 的**分数**部分（0~1），不涉及像素尺寸。

    与 :func:`parse_crop_rect` 分开：校验阶段拿不到帧尺寸，用虚构 shape 去
    换算像素会把合法窄裁剪（如 0,0,0.3333,1 或 0,0,0.5,1 在 1×1 下都会
    round 成空范围）误判为非法。这里只检查有限性、0~1 范围与严格
    x0<x1 / y0<y1；像素级合法性在解码后用真实尺寸再校验。
    """
    parts = [item.strip() for item in str(value).split(",")]
    if len(parts) != 4:
        raise ValueError(f"crop-rect 需要 4 个分数值 x0,y0,x1,y1（0~1），收到: {value!r}")
    try:
        fx0, fy0, fx1, fy1 = (float(item) for item in parts)
    except ValueError as exc:
        raise ValueError(f"crop-rect 含非数值: {value!r}") from exc
    for fraction in (fx0, fy0, fx1, fy1):
        if not math.isfinite(fraction):
            raise ValueError(f"crop-rect 分量必须是有限数: {value!r}")
        if not (0.0 <= fraction <= 1.0):
            raise ValueError(f"crop-rect 分量必须在 0~1 之间: {value!r}")
    if not (fx0 < fx1 and fy0 < fy1):
        raise ValueError(f"crop-rect 需要 x0<x1 且 y0<y1: {value!r}")
    return fx0, fy0, fx1, fy1


def parse_crop_rect(value: str, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """解析裁剪区域 "x0,y0,x1,y1"（0~1 分数坐标，相对整帧）。

    例：左侧三分之一 = "0,0,0.3333,1"。返回像素切片 (x0,y0,x1,y1)。
    分数校验见 :func:`parse_crop_fractions`；这里额外用**真实**帧尺寸
    校验换算出的像素范围非空（解码后才能做）。
    """
    fx0, fy0, fx1, fy1 = parse_crop_fractions(value)
    height, width = shape
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

def _reserve_final_dir(output_root: Path, stem: str) -> Path:
    """为本次运行保留一个**不会复用**的产物目录并创建它。

    命名规则：``<stem>`` 不存在则直接用；已存在则依次尝试
    ``<stem>_<时间戳>``、``<stem>_<时间戳>_<序号>``，直到找到一个**不存在**
    的路径（``mkdir`` 不带 ``exist_ok``，由文件系统保证原子性）。

    早期实现用秒级时间戳且同秒内会落回同一个目录，再被 ``os.replace`` 直接
    覆盖既有产物（回归 2026-10-03 R17）。这里既不递归删除既有产物，也不复用
    已存在的目录，因此同一秒内重复运行不会破坏上一次的结果。
    """
    for attempt in range(1000):
        if attempt == 0:
            candidate = output_root / stem
        elif attempt == 1:
            candidate = output_root / f"{stem}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        else:
            candidate = output_root / (
                f"{stem}_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{attempt}"
            )
        try:
            candidate.mkdir(parents=True)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"无法为 {stem} 保留唯一产物目录（尝试 1000 次仍冲突）")


def run_highlevel_pipeline(video: Path, output_root: Path, args) -> dict:
    from video_extractor.runner import VIDEO_EXTENSIONS

    video = Path(video).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video}")
    if video.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"不支持的视频格式 {video.suffix}，支持: {', '.join(sorted(VIDEO_EXTENSIONS))}")

    output_root = Path(output_root).expanduser().resolve()
    # 先把"目标就是输入文件"这种明显自毁的用法拒掉，再去建目录 ——
    # 否则 mkdir 会先对输入文件报 FileExistsError（错误信息指向文件系统而非用法）
    if output_root == video:
        raise ValueError(f"--output-root 不能就是输入视频文件本身: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    extraction = run_extraction(video, output_root, args)
    raw_tif = extraction["output"]
    manifest = extraction["manifest"]
    result = extraction["result"]
    staged_link = extraction.get("staged_input")
    # 实际生效的采样值：all 模式不消费采样值，因此为 None（见 build_sampling）
    actual_sampling_value = None if args.sampling == "all" else args.sampling_value

    # 移动提取产物到最终目录
    stem = video.stem
    final_dir = output_root / stem
    final_dir = _reserve_final_dir(output_root, stem)
    final_tif = final_dir / f"01_raw_{stem}_stack.tif"
    final_manifest = final_dir / f"01_raw_{stem}_frames.csv"
    if final_tif.exists() or final_manifest.exists():
        # 保留目录仍撞名（同一秒内并发/重复调用）：绝不 os.replace 覆盖既有产物
        raise FileExistsError(
            f"目标产物已存在，拒绝覆盖：{final_tif if final_tif.exists() else final_manifest}\n"
            "请稍后重试或改用不同的 --output-root。"
        )
    os.replace(str(raw_tif), str(final_tif))
    if manifest.is_file():
        os.replace(str(manifest), str(final_manifest))
    # output_file 逐行精确指向最终 TIFF；input_path 改回用户原始视频路径
    # （staging 里的硬链接马上就删，不能留在清单里当追溯依据）。
    rewrite_manifest_output_file(final_manifest, str(raw_tif), str(final_tif),
                                original_input=video, staged_input=staged_link)
    # 产物已移出，清理本任务 staging
    _discard_stage(extraction["stage_root"])

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

    # 上游可能"报错但退出码 0"（丢帧、旋转元数据导致像素重排、输入被替换），
    # 此时 state 仍为 completed。仅凭 state 判定会把不可信数据当成功。这里核对
    # TIFF 实际页数与上游报告的帧数，不一致就记入告警（并在 summary 降级状态）。
    extraction_warnings = list(extraction.get("warnings", []))
    actual_frames = int(inspect.get("n_frames") or 0)
    reported_frames = int(result.frame_count or 0)
    if actual_frames and reported_frames and actual_frames != reported_frames:
        extraction_warnings.append(
            f"提取帧数与输出不一致：上游报告 {reported_frames} 帧，"
            f"TIFF 实际 {actual_frames} 页（可能丢帧，请复核）"
        )
    if not actual_frames:
        extraction_warnings.append("输出 TIFF 无可读页，数据不可信")
    if extraction_warnings:
        for message in extraction_warnings:
            emit_info(message, severity="warning")

    frames, meta = TiffIO.read_stack(str(final_tif), max_bytes=None)

    # 帧数与解码结果复查：空结果不得进入后续索引
    if not frames:
        raise RuntimeError(f"输出 TIFF 解码后没有任何帧: {final_tif}")

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
            diagnostic_path.write_text(json.dumps(clean_json_payload(diagnostic), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            raise RuntimeError(
                f"漂移检测未通过质量门控（严格档与放宽档均失败）：{error}\n"
                f"诊断已写入 {diagnostic_path}\n"
                "提示：可进一步放宽 SIFT 质量参数后重试（见诊断文件建议）；对周期性晶格/快速运动数据尤其需要人工复核。"
            ) from error
    shifts_x, shifts_y = detection.shifts_x, detection.shifts_y

    corrected_tif = final_dir / f"02_corrected_{stem}.tif"
    # --compression 作用在本流水线自己的最终校正 TIFF 上（上游提取固定未压缩）
    written = write_corrected_stack(
        frames, shifts_x, shifts_y, meta, corrected_tif, args,
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
            # 从模块读取真实版本，避免与上游 __init__.py 漂移
            # （此前硬编码 "4.1.1"，而 __init__.py 已是 4.4）
            "video_extractor": getattr(
                sys.modules.get("video_extractor"), "__version__", "unknown"
            ),
            "drift_core": getattr(sys.modules.get("drift_core"), "__version__", "unknown"),
        },
        "video": {
            "path": str(video),
            **{k: extraction["info"].get(k) for k in
               ("width", "height", "duration_s", "average_fps", "nominal_fps", "frame_count",
                "codec", "pixel_format", "color_family", "bits_per_sample", "is_variable_fps")
               if extraction["info"].get(k) is not None},
        },
        "extraction": {
            # 如实记录**实际**生效的值，而不是 CLI 请求值：
            #   * all 模式下采样值不被消费 → value=None（请求值另见 requested_*）
            #   * 提取堆栈由上游固定未压缩写出 → compression 恒为 "none"，
            #     最终产物的编码在 correction.compression 里
            "sampling_mode": args.sampling,
            "sampling_value": actual_sampling_value,
            "requested_sampling_value": args.sampling_value,
            "color_mode": "grayscale",
            "bit_depth": args.bit_depth,
            "compression": "none",
            "requested_compression": args.compression,
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
            # 最终产物的真实编码（none/deflate），与 extraction.compression 区分开
            "compression": args.compression,
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
                     + list(detection.warnings) + extraction_warnings),
    }
    audit_path = final_dir / "02_audit.json"
    audit_path.write_text(json.dumps(clean_json_payload(audit), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    audit["outputs"]["audit_json"] = str(audit_path)

    summary_path = final_dir / "03_summary.json"
    # summary 保持紧凑：逐帧对明细只在 audit 中，不重复携带（避免巨大载荷）
    detection_summary = {key: value for key, value in audit["detection"].items() if key != "pairs"}
    # 质量状态如实传递：有告警就不是干净的 completed，不能把质量不足标成正常。
    warnings = audit["warnings"]
    quality_status = "completed_with_warnings" if warnings else "completed"
    summary = {
        "status": quality_status,
        "warnings_count": len(warnings),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "video": str(video),
        "output_dir": str(final_dir),
        "extraction": audit["extraction"],
        "detection": detection_summary,
        "correction": audit["correction"],
        "outputs": audit["outputs"],
        "warnings": warnings,
        "elapsed_seconds": audit["elapsed_seconds"],
        "result_paths": [str(p) for p in (final_tif, corrected_tif, shifts_csv,
                                          curve_png, audit_path, summary_path, montage_png) if p],
    }
    summary_path.write_text(json.dumps(clean_json_payload(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    audit["status"] = quality_status
    audit["outputs"]["summary_json"] = str(summary_path)
    audit_path.write_text(json.dumps(clean_json_payload(audit), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
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


class _ProtocolArgumentParser(argparse.ArgumentParser):
    """argparse 失败也走 JSON 协议：末行 error + 退出码非 0。

    ``--help`` 正常打印帮助并退出 0（argparse 自己的行为，不拦截）。
    其他解析错误（非法 enum、非数字、缺必填）不再只往 stderr 写 usage 后
    以退出码 2 结束 —— 上层按"stdout 每行 JSON、末行 summary/error"消费时
    会拿不到任何可解析的失败原因（回归 2026-10-03 R17）。
    """

    def error(self, message):        # noqa: D102 - argparse 钩子
        raise ValueError(f"参数解析失败：{message}")


def build_parser() -> argparse.ArgumentParser:
    parser = _ProtocolArgumentParser(description="TEM 视频 → TIFF 提取 → 漂移矫正 → 审计/曲线/汇总")
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


def validate_args(args) -> None:
    """在昂贵的提取之前拒绝非法输入。

    采样值、压缩、比特深、裁剪框等都在这里先校验：等到解码/写盘之后再失败，
    既浪费几分钟，也会留下半成品目录。

    失败时抛 ``ValueError``（由 :func:`main` 转成 stdout 的 ``error`` 事件并以
    退出码 1 结束），**不用** ``parser.error`` —— 后者只写 stderr 且退出码 2，
    上层按"stdout 每行 JSON、末行为 error"消费时拿不到失败原因。
    """
    problems: list[str] = []
    if args.sampling != "all" and args.sampling_value is not None:
        if not math.isfinite(args.sampling_value) or args.sampling_value <= 0:
            problems.append(
                f"--sampling-value 必须是正的有限数，得到 {args.sampling_value!r}"
            )
    if args.sampling == "all":
        pass  # all 模式忽略 sampling-value（由 build_sampling 提示）
    if args.compression not in ("none", "deflate"):
        problems.append(f"--compression 只能是 none/deflate，得到 {args.compression!r}")
    if args.bit_depth not in ("source", "8", "16"):
        problems.append(f"--bit-depth 只能是 source/8/16，得到 {args.bit_depth!r}")
    if args.crop_mode not in ("crop", "keep"):
        problems.append(f"--crop-mode 只能是 crop/keep，得到 {args.crop_mode!r}")
    if not math.isfinite(args.max_stack_gib) or args.max_stack_gib < 0:
        problems.append(f"--max-stack-gib 必须是非负有限数，得到 {args.max_stack_gib!r}")
    if args.skip_interval < 1:
        problems.append(f"--skip-interval 必须 >= 1，得到 {args.skip_interval}")
    if args.crop_rect:
        # 只做与尺寸无关的分数校验；像素范围在解码后用真实 shape 再核对
        try:
            parse_crop_fractions(args.crop_rect)
        except ValueError as exc:
            problems.append(f"--crop-rect 非法: {exc}")
    if problems:
        raise ValueError("参数校验失败：" + "；".join(problems))


def main(argv=None) -> int:
    """命令行入口。JSON 通道按**本次调用**的 stdout 绑定。

    在 ``json_output_channel`` 作用域内：进入时记录并接管当时的 ``sys.stdout``
    与旧 ``_JSON_OUT``，退出时（成功/失败/异常）一并还原。因此同一进程里先在
    A 流 import、再在 B 流调用 main，事件会写到 B 流，而不是 import 时的 A 流。
    """
    force_utf8_stdio()
    with json_output_channel() as channel:
        try:
            args = build_parser().parse_args(argv)
        except SystemExit as exit_signal:
            # --help / --version：argparse 已按正常帮助输出，保持退出码语义
            if exit_signal.code in (0, None):
                return 0
            raise
        except ValueError as parse_error:
            emit("error", message=str(parse_error))
            return 1
        try:
            validate_args(args)
            load_tools(args.extractor_dir, args.drift_dir)
            # 上游 import 会打 banner：把它们的 stdout 让给 stderr，保住 JSON 通道
            with third_party_output_on_stderr():
                summary = run_highlevel_pipeline(
                    Path(args.video), Path(args.output_root), args)
            emit("summary", **summary)
            return 0
        except Exception as error:
            emit("error", message=str(error), traceback=traceback.format_exc())
            return 1
        finally:
            if channel is not None:
                with contextlib.suppress(Exception):
                    channel.flush()


if __name__ == "__main__":
    raise SystemExit(main())
