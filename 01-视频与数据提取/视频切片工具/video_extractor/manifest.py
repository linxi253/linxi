from __future__ import annotations

import csv
import os
from pathlib import Path
import threading

from .models import ExtractOptions, VideoInfo
from .sampling import output_time_seconds
from .utils import atomic_write_json


def write_frame_manifest(
    path: Path,
    video_info: VideoInfo,
    options: ExtractOptions,
    output_file: Path,
    frame_count: int,
    input_sha256: str,
    *,
    scheduled_times: list[float] | None = None,
    source_pts: list[float] | None = None,
    timeline_source: str = "estimated",
    time_is_estimated: bool = False,
    stream_start_time_s: float | None = None,
    cancelled: threading.Event | None = None,
) -> None:
    """写入逐帧 CSV 清单。

    input_sha256 由调用方在解码前计算并传入，避免：
    1. 处理完成后再全量读取输入文件（大文件的第三次 IO）；
    2. 处理期间文件被替换导致哈希与实际解码内容不符。

    时间轴口径（timeline_source）：
    - measured: scheduled_time_s 与 source_pts_s 来自 count pass 中 showinfo
      实测的源帧 PTS，scheduled_time_s 已按首帧归零；
    - model: 目标帧率采样，scheduled_time_s 是采样栅格 k/fps（精确的模型值），
      source_pts_s 留空（输出帧与源帧的映射不是一一对应）；
    - estimated: PTS 捕获不可用时的回退，按平均帧率均分估算，
      time_is_estimated=True。

    stream_start_time_s 为选中视频流首帧 PTS（未归零的原始媒体时间），
    用于把归零时间轴映射回容器时间。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig：含中文路径的清单在 Excel 中直接打开不乱码（与其他工具一致）。
    # 读取端（含 10-DSH集成 流水线）必须用 utf-8-sig，否则首列键名会带 BOM 前缀。
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "output_index", "output_file", "scheduled_time_s", "source_pts_s",
                "time_is_estimated", "timeline_source", "stream_start_time_s",
                "input_path", "input_sha256",
                "source_average_fps", "source_nominal_fps", "source_pixel_format",
                "source_color_space", "source_color_range", "source_rotation_degrees",
                "sampling_mode", "sampling_value",
            ],
        )
        writer.writeheader()
        rows_per_cancel_check = 8192
        for index in range(frame_count):
            if cancelled is not None and index % rows_per_cancel_check == 0 and cancelled.is_set():
                raise InterruptedError("清单写入已取消")
            if scheduled_times is not None and index < len(scheduled_times):
                scheduled = scheduled_times[index]
            else:
                scheduled = output_time_seconds(index, options.sampling, video_info)
            if source_pts is not None and index < len(source_pts):
                source_pts_value = source_pts[index]
            else:
                source_pts_value = None
            writer.writerow(
                {
                    "output_index": index,
                    "output_file": str(output_file),
                    "scheduled_time_s": scheduled,
                    "source_pts_s": source_pts_value,
                    "time_is_estimated": time_is_estimated,
                    "timeline_source": timeline_source,
                    "stream_start_time_s": stream_start_time_s,
                    "input_path": str(video_info.path),
                    "input_sha256": input_sha256,
                    "source_average_fps": video_info.average_fps,
                    "source_nominal_fps": video_info.nominal_fps,
                    "source_pixel_format": video_info.pixel_format,
                    "source_color_space": video_info.color_space,
                    "source_color_range": video_info.color_range,
                    "source_rotation_degrees": video_info.rotation_degrees,
                    "sampling_mode": options.sampling.mode.value,
                    "sampling_value": options.sampling.value,
                }
            )
        # 清单与输出在 runner 中作为一对提交；清单本身也必须落盘，
        # 否则断电时会出现"输出完好但清单损坏"的半提交状态
        handle.flush()
        os.fsync(handle.fileno())


def write_job_manifest(path: Path, payload: dict) -> None:
    atomic_write_json(path, payload)
