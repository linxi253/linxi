from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

from .decoder import FrameDecoder, resolve_frame_spec
from . import __version__
from .ffmpeg import FFmpegError, FFmpegManager, FFmpegPaths, probe_first_frame_pts, probe_video
from .manifest import write_frame_manifest, write_job_manifest
from .models import (
    ExtractOptions,
    JobState,
    SamplingMode,
    VideoResult,
)
from .sampling import (
    effective_fps,
    estimate_frame_count,
    median_interval,
    timeline_is_uniform,
)
from .utils import ensure_same_filesystem_stage, file_sha256, fsync_directory, output_stem
from .writers import (
    StackWriter,
    estimate_raw_bytes,
    needs_truncate,
    remove_partial,
)


VIDEO_EXTENSIONS = {
    ".264", ".265", ".3g2", ".3gp", ".asf", ".av1", ".avi", ".dav",
    ".divx", ".dv", ".f4v", ".flv", ".h264", ".h265", ".hevc", ".ivf",
    ".m2t", ".m2ts", ".m4v", ".mjpeg", ".mjpg", ".mkv", ".mov", ".mp4",
    ".mpeg", ".mpg", ".mts", ".mxf", ".ogm", ".ogv", ".qt", ".rm", ".rmvb",
    ".ts", ".vob", ".webm", ".wmv", ".y4m",
}

# 只清理超过该时长的 .partial 目录：新目录可能属于并行运行的其他任务
STALE_STAGE_MAX_AGE_SECONDS = 24 * 3600
# 活跃任务的 stage 目录内会定期 touch 心跳文件；清理时看到"足够新"的心跳即跳过。
STAGE_HEARTBEAT_NAME = ".heartbeat"
STAGE_HEARTBEAT_MAX_AGE_SECONDS = 120.0
# 进度事件最小发射间隔：CLI 模式下每个事件是一次 stderr 写行，
# 高帧率视频逐帧发事件会让控制台成为吞吐瓶颈
PROGRESS_EMIT_INTERVAL_SECONDS = 0.1
# 磁盘安全保留量：保留 2 GiB 与 5% 可用空间中的较大者，但不超过 32 GiB——
# 无上限的百分比在大容量阵列上（如空闲 8 TiB 时保留 400 GiB）会误拦合法任务
DISK_RESERVE_MIN_BYTES = 2 * 1024**3
DISK_RESERVE_MAX_BYTES = 32 * 1024**3
# 孤儿输出隔离区：放在 .partial 之外，cleanup_stale_stages 不会碰它，
# 被隔离的数据永远不会被自动删除，由用户决定去留
ORPHAN_DIR_NAME = ".orphans"

EventCallback = Callable[[dict], None]


def find_videos(folder: Path, exclude: Path | None = None) -> list[Path]:
    """扫描输入目录中的视频；跳过隐藏目录、.partial 和 exclude 指向的目录。"""
    found: list[Path] = []
    errors: list[str] = []
    exclude_resolved = exclude.resolve() if exclude is not None else None
    for root, dirs, files in os.walk(folder, onerror=lambda error: errors.append(str(error))):
        dirs[:] = [
            name for name in dirs
            if not name.startswith(".")
            and (
                exclude_resolved is None
                or (Path(root) / name).resolve() != exclude_resolved
            )
        ]
        for name in files:
            path = Path(root) / name
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                found.append(path)
    if errors:
        raise OSError("扫描输入目录失败: " + "；".join(errors))
    return sorted(found, key=lambda item: str(item).casefold())


def cleanup_stale_stages(
    output_root: Path,
    max_age_seconds: float = STALE_STAGE_MAX_AGE_SECONDS,
    heartbeat_max_age_seconds: float = STAGE_HEARTBEAT_MAX_AGE_SECONDS,
) -> int:
    """清理崩溃残留的 .partial 暂存目录，返回清理数量。

    只清理目录 mtime 超过 max_age_seconds 的目录；如果目录内存在
    足够新的心跳文件（STAGE_HEARTBEAT_NAME），说明它属于活跃任务，
    即使目录已超过 24h 也跳过，避免并行任务被误删。
    只处理 .partial；.orphans 隔离区是持久数据，永不自动清理。
    """
    partial_root = output_root / ".partial"
    if not partial_root.is_dir():
        return 0
    now = time.time()
    stale: list[Path] = []
    for item in partial_root.iterdir():
        if not item.is_dir():
            continue
        try:
            heartbeat = item / STAGE_HEARTBEAT_NAME
            if heartbeat.is_file():
                heartbeat_age = now - heartbeat.stat().st_mtime
                if heartbeat_age < heartbeat_max_age_seconds:
                    continue
            age = now - item.stat().st_mtime
        except OSError:
            continue
        if age >= max_age_seconds:
            stale.append(item)
    for item in stale:
        shutil.rmtree(item, ignore_errors=True)
    return len(stale)


def output_paths_for(
    input_root: Path,
    output_root: Path,
    video: Path,
) -> tuple[Path, Path, Path]:
    """Compute final output path, per-video CSV manifest path and relative parent."""
    parent, stem = output_stem(input_root, video)
    # 后缀刻意不用 .ome.tif：ImageJ/Bio-Formats 会按 OME-XML 解析
    final = output_root / parent / f"{stem}_stack.tif"
    manifest = output_root / parent / f"{stem}_frames.csv"
    return final, manifest, parent


def detect_orphan_outputs(
    input_root: Path,
    output_root: Path,
    videos: Iterable[Path],
) -> list[tuple[Path, Path]]:
    """Return (final, manifest) pairs where the final output exists but the CSV is missing.

    这类孤儿是双文件提交（先 os.replace 最终输出，再 os.replace 清单）在两步之间
    崩溃时产生的。若不处理，下次运行会因"输出已存在"拒绝覆盖且清单缺失，形成死锁。
    """
    orphans: list[tuple[Path, Path]] = []
    for video in videos:
        final, manifest, _parent = output_paths_for(input_root, output_root, video)
        if final.exists() and not manifest.exists():
            orphans.append((final, manifest))
    return orphans


def file_identity(path: Path) -> tuple[int, int]:
    """输入文件的身份指纹（大小 + 纳秒级 mtime）。

    哈希、计数、解码各自独立打开同一路径；正在录制/同步/被替换的文件会
    让溯源哈希与实际解码内容脱节。阶段间校验身份可以把窗口内的变更
    变成显式失败。注意：同尺寸同 mtime 的替换仍可能逃过检测——stat
    校验是缓解手段，不是强一致保证。
    """
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns)


def verify_file_identity(path: Path, expected: tuple[int, int], stage: str) -> None:
    current = file_identity(path)
    if current != expected:
        raise OSError(
            f"输入文件在{stage}期间被修改（大小或修改时间已变化），"
            f"为保证溯源与解码内容一致已中止: {path}"
        )


class JobRunner:
    def __init__(
        self,
        options: ExtractOptions,
        ffmpeg_custom_path: str | None = None,
        *,
        allow_unsafe_ffmpeg: bool = False,
    ):
        options.validate()
        self.options = options
        self.ffmpeg_paths: FFmpegPaths = FFmpegManager(
            ffmpeg_custom_path, allow_unsafe=allow_unsafe_ffmpeg
        ).resolve()
        self.cancelled = threading.Event()
        self._current_process: subprocess.Popen[bytes] | None = None
        self._process_lock = threading.Lock()

    def cancel(self) -> None:
        self.cancelled.set()
        with self._process_lock:
            process = self._current_process
        if process and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass

    def _event(self, callback: EventCallback | None, **payload: object) -> None:
        if callback:
            callback(payload)

    def _track_process(self, process: subprocess.Popen[bytes]) -> None:
        with self._process_lock:
            self._current_process = process

    def _release_process(self) -> None:
        with self._process_lock:
            self._current_process = None

    def _output_paths(self, input_root: Path, output_root: Path, video: Path) -> tuple[Path, Path, Path]:
        return output_paths_for(input_root, output_root, video)

    @staticmethod
    def _check_space(path: Path, minimum_bytes: int | None) -> None:
        if minimum_bytes is None:
            return
        free = shutil.disk_usage(path).free
        reserve = min(
            max(DISK_RESERVE_MIN_BYTES, int(free * 0.05)),
            DISK_RESERVE_MAX_BYTES,
        )
        if free - reserve < minimum_bytes:
            raise OSError(
                f"磁盘空间不足：任务预计需要 {minimum_bytes / 1024**3:.2f} GiB，"
                f"当前可用 {free / 1024**3:.2f} GiB，"
                f"其中 {reserve / 1024**3:.2f} GiB 为安全保留量"
                f"（2 GiB 与 5% 可用空间取大者，上限 32 GiB）"
            )

    def _check_space_for_frames(self, output_root: Path, frame_count: int | None, spec) -> None:
        required = estimate_raw_bytes(frame_count, spec)
        self._check_space(output_root, required)

    @staticmethod
    def _touch_heartbeat(stage_root: Path) -> None:
        (stage_root / STAGE_HEARTBEAT_NAME).touch(exist_ok=True)

    @staticmethod
    def _heartbeat_loop(stage_root: Path, stop_event: threading.Event) -> None:
        while not stop_event.wait(30):
            try:
                JobRunner._touch_heartbeat(stage_root)
            except OSError:
                pass

    def _quarantine_orphan(self, output_root: Path, final: Path) -> Path:
        """把孤儿输出移入持久隔离区（不会被自动清理，由用户决定去留）。"""
        orphan_dir = output_root / ORPHAN_DIR_NAME / f"orphan_{uuid.uuid4().hex[:12]}"
        orphan_dir.mkdir(parents=True, exist_ok=True)
        destination = orphan_dir / final.name
        shutil.move(str(final), str(destination))
        return destination

    def run_batch(self, input_root: Path, output_root: Path, callback: EventCallback | None = None) -> list[VideoResult]:
        input_root = input_root.resolve()
        output_root = output_root.resolve()
        if not input_root.is_dir():
            raise ValueError("输入目录不存在")
        if output_root.exists() and not output_root.is_dir():
            raise ValueError(f"输出路径已存在但不是目录: {output_root}")
        output_root.mkdir(parents=True, exist_ok=True)
        # 先清理 stale .partial，再扫描输入；即使输入目录为空也执行清理
        stale_count = cleanup_stale_stages(output_root)
        videos = find_videos(input_root, exclude=output_root)
        if not videos:
            raise ValueError("输入目录中没有支持的视频")
        orphans = detect_orphan_outputs(input_root, output_root, videos)
        job_id = uuid.uuid4().hex
        stage_root = ensure_same_filesystem_stage(output_root, job_id)
        try:
            self._touch_heartbeat(stage_root)
        except OSError:
            pass
        results: list[VideoResult] = []
        started = time.time()
        self._event(callback, kind="batch_started", total=len(videos), job_id=job_id)
        if stale_count:
            self._event(
                callback, kind="info",
                message=f"已清理 {stale_count} 个上次运行残留的暂存目录（.partial）",
            )
        for orphan_final, orphan_manifest in orphans:
            destination = self._quarantine_orphan(output_root, orphan_final)
            self._event(
                callback, kind="warning", path=str(orphan_final),
                message=(
                    f"检测到孤立输出（缺少逐帧清单 {orphan_manifest.name}），"
                    f"已移入持久隔离区 {ORPHAN_DIR_NAME}/（不会被自动清理，"
                    f"确认无需保留后可手动删除）: {destination}；本次将重新生成输出"
                ),
            )
        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, args=(stage_root, heartbeat_stop),
            name="stage-heartbeat", daemon=True,
        )
        heartbeat_thread.start()
        try:
            for index, video in enumerate(videos, start=1):
                if self.cancelled.is_set():
                    break
                self._event(callback, kind="video_started", index=index, total=len(videos), path=str(video))
                result = self._run_one(
                    input_root, output_root, stage_root, video, callback,
                    batch_index=index, batch_total=len(videos),
                )
                results.append(result)
                self._event(callback, kind="video_finished", index=index, total=len(videos), result=result.as_dict())
            if self.cancelled.is_set():
                for video in videos[len(results):]:
                    results.append(VideoResult(input_path=str(video), state=JobState.CANCELLED, error="批任务已取消"))
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1)
            self._release_process()
            shutil.rmtree(stage_root, ignore_errors=True)
            partial_parent = stage_root.parent
            try:
                partial_parent.rmdir()
            except OSError:
                pass
            manifest_payload = {
                "schema_version": 4,
                "tool_version": __version__,
                "job_id": job_id,
                "started_at_epoch": started,
                "finished_at_epoch": time.time(),
                "ffmpeg": str(self.ffmpeg_paths.ffmpeg),
                "ffprobe": str(self.ffmpeg_paths.ffprobe),
                "ffmpeg_version": self.ffmpeg_paths.version,
                "options": asdict(self.options),
                "results": [item.as_dict() for item in results],
            }
            try:
                write_job_manifest(output_root / "manifests" / f"job_{job_id}.json", manifest_payload)
            except OSError as error:
                # 清单写入失败不应掩盖已完成的视频结果
                self._event(
                    callback, kind="warning",
                    message=f"写入任务清单失败（视频结果不受影响）: {error}",
                )
        return results

    def _run_one(
        self,
        input_root: Path,
        output_root: Path,
        stage_root: Path,
        video: Path,
        callback: EventCallback | None,
        *,
        batch_index: int = 1,
        batch_total: int = 1,
    ) -> VideoResult:
        result = VideoResult(input_path=str(video), state=JobState.PROBING)
        warnings: list[str] = []
        stage_target: Path | None = None
        stack_writer: StackWriter | None = None

        def warn(message: str) -> None:
            warnings.append(message)
            self._event(callback, kind="warning", path=str(video), message=message)

        try:
            # probe 改为 Popen 并纳入 _track_process；cancel() 会直接终止 ffprobe。
            info = probe_video(
                self.ffmpeg_paths.ffprobe, video, on_started=self._track_process
            )
            self._release_process()
            if self.cancelled.is_set():
                result.state = JobState.CANCELLED
                result.error = "用户取消"
                return result
            if info.stream_count > 1:
                warn(
                    f"{video.name} 包含 {info.stream_count} 条视频流，"
                    f"使用全局索引 {info.stream_index} 的流"
                )
            if info.rotation_degrees not in (None, 0):
                warn(
                    f"{video.name} 容器声明了 {info.rotation_degrees:g}° 显示旋转，"
                    "输出保持编码方向的像素矩阵（未应用旋转）；如需旋转后的视图请在下游处理"
                )
            if info.bits_source != "declared" and info.bits_per_sample > 8:
                warn(
                    f"{video.name} 未声明可靠的位深元数据，"
                    f"已按像素格式推断为 {info.bits_per_sample} 位"
                )
            if self.options.lenient_decode:
                warn(
                    f"{video.name} 以宽松解码模式处理：损坏帧可能被 FFmpeg 跳过，"
                    "输出帧数可能少于源帧数且缺帧位置不可知"
                )
            heuristic_vfr = info.is_variable_fps
            if (
                self.options.sampling.mode is not SamplingMode.ALL
                and heuristic_vfr
            ):
                # fps 滤镜在源帧间隔大于目标间隔的空档会复制前帧；
                # 输出时间轴本身是精确的采样间隔，但并非每帧都来自独立采样
                warn(
                    f"{video.name} 为可变帧率视频：按目标帧率采样时，"
                    "源帧间隔大于目标间隔的空档会复制前帧"
                )
            spec = resolve_frame_spec(info, self.options)
            if spec.dtype == "uint8" and info.bits_per_sample > 8:
                warn(
                    f"{video.name} 源为 {info.bits_per_sample} 位，"
                    "将降采样为 8 位输出（损失高位精度）"
                )
            if spec.dtype == "uint16" and 8 < info.bits_per_sample < 16:
                warn(
                    f"{video.name} 源为 {info.bits_per_sample} 位，输出 16 位时"
                    f"FFmpeg 会把有效码值线性重映射到 16 位满量程"
                    f"（例如 10 位源的 1023 对应 65535）；数值比例保持但非原始码值"
                )
            # 拒绝帧率上采样：fps 滤镜在目标帧率高于源帧率时会复制帧，
            # 对定量时间序列分析这是虚假的时间分辨率
            target_fps = self.options.sampling.target_fps
            source_fps = info.average_fps or info.nominal_fps
            if target_fps and source_fps and target_fps > source_fps * 1.001:
                raise ValueError(
                    f"目标帧率 {target_fps:g} fps 高于源视频帧率 {source_fps:g} fps，"
                    "会通过复制帧制造虚假时间分辨率；请降低目标帧率或使用全部帧模式"
                )
            final, frame_manifest, parent = self._output_paths(input_root, output_root, video)
            if final.exists() or frame_manifest.exists():
                existing = final.name if final.exists() else frame_manifest.name
                raise FileExistsError(
                    f"输出已存在，拒绝覆盖: {existing}；如需重新生成请先移动或删除"
                    f" {final.parent} 下的对应文件"
                )
            estimated_frames = estimate_frame_count(self.options.sampling, info)
            estimated_bytes = estimate_raw_bytes(estimated_frames, spec)
            # 用估计帧数先做一次磁盘空间检查，能在昂贵的 count pass 前拦下
            # 明显不足的情况；count pass 后还会用真实帧数复查。
            self._check_space_for_frames(output_root, estimated_frames, spec)
            result.video_info = info.as_dict()
            result.frame_spec = asdict(spec)
            result.estimated_bytes = estimated_bytes
            # 解码前计算输入哈希：处理后再算可能因文件被替换导致溯源失真，
            # 且避免在全部帧写完后额外全量读取一次输入文件。
            # 哈希与后续各阶段之间用 stat 指纹校验身份（file_identity）。
            self._event(callback, kind="state", path=str(video), state="hashing")
            identity = file_identity(video)
            input_sha256 = file_sha256(video, cancelled=self.cancelled)
            result.input_sha256 = input_sha256
            verify_file_identity(video, identity, "哈希计算")
            if self.cancelled.is_set():
                result.state = JobState.CANCELLED
                result.error = "用户取消"
                return result
            # 首帧 PTS：fps 滤镜的 start_time 必须锚定在流的真实起点，
            # 否则晚于容器起始的视频流（如前置音轨的 MP4）会在首帧前被复制填充帧。
            # 注意换算：ffmpeg 解码时默认把 format.start_time 从时间轴减去，
            # 而 ffprobe 的帧 PTS 保留原值，两者相减才是滤镜时间轴的起点
            stream_start = probe_first_frame_pts(
                self.ffmpeg_paths.ffprobe, video, info.video_stream_ordinal
            )
            filter_start = (
                max(0.0, stream_start - (info.format_start_time or 0.0))
                if stream_start is not None else 0.0
            )
            start_time = filter_start
            decoder = FrameDecoder(self.ffmpeg_paths.ffmpeg)
            eff_fps = effective_fps(self.options.sampling, info)
            stage_target = stage_root / parent / final.name

            # 先做 count pass：堆栈需要真实 T 轴长度，且写盘前必须按真实体积复查空间。
            # 该遍同时通过 showinfo 捕获源帧 PTS，用于实测时间轴与可变帧率检测。
            result.state = JobState.COUNTING
            self._event(callback, kind="state", path=str(video), state=result.state.value)
            last_count_emit = 0.0

            def on_count_progress(current: int) -> None:
                nonlocal last_count_emit
                now = time.monotonic()
                if now - last_count_emit >= PROGRESS_EMIT_INTERVAL_SECONDS:
                    last_count_emit = now
                    self._event(
                        callback, kind="counting_progress", path=str(video), current=current,
                        total_hint=estimated_frames, index=batch_index, batch_total=batch_total,
                    )

            count = decoder.count_frames(
                video, self.options, spec, self.cancelled, self._track_process,
                on_progress=on_count_progress,
                start_time=start_time, lenient=self.options.lenient_decode,
            )
            self._release_process()
            if self.cancelled.is_set():
                result.state = JobState.CANCELLED
                result.error = "用户取消"
                return result
            expected = count.frames
            source_pts = count.pts_seconds
            if expected <= 0:
                raise FFmpegError("视频不包含可输出的帧")
            verify_file_identity(video, identity, "帧统计")
            self._check_space_for_frames(output_root, expected, spec)
            result.estimated_bytes = estimate_raw_bytes(expected, spec)
            truncated = needs_truncate(expected, spec)
            result.tiff_container = "ImageJ-TIFF"

            # 时间轴口径：count pass 实测的源帧 PTS 优先；
            # 捕获不完整时回退到采样模型/平均帧率估算。
            # showinfo 记录的是 ffmpeg 滤镜时间轴（已减去 format.start_time），
            # 写入 CSV 前加回容器起始时间还原为原始媒体时间
            format_offset = info.format_start_time or 0.0
            measured_usable = (
                self.options.sampling.mode is SamplingMode.ALL
                and len(source_pts) == expected
                and expected > 0
            )
            scheduled_times: list[float] | None = None
            csv_source_pts: list[float] | None = None
            if measured_usable:
                timeline_source = "measured"
                scheduled_times = [value - source_pts[0] for value in source_pts]
                csv_source_pts = [value + format_offset for value in source_pts]
                measured_interval = median_interval(source_pts)
                if (
                    timeline_is_uniform(source_pts)
                    and measured_interval is not None
                    and measured_interval > 0
                ):
                    interval: float | None = measured_interval
                else:
                    # timeline_is_uniform 对少于 3 帧恒判均匀，两帧 PTS 相同
                    # （重复帧/单帧循环容器）时 median 间隔为 0：时间轴不可用，
                    # 与可变帧率一致地不写固定帧间隔，避免写 TIFF 元数据时除零
                    interval = None
                    warn(
                        f"{video.name} 实测帧间隔不均匀或退化（可变帧率/重复 PTS）："
                        "TIFF 不写入固定帧间隔，逐帧精确时间见 CSV 的 source_pts_s 列"
                    )
            elif self.options.sampling.mode is SamplingMode.ALL:
                # PTS 捕获不可用：回退为按平均帧率估算并明确标记
                timeline_source = "estimated"
                interval = None if not eff_fps else 1.0 / eff_fps
            else:
                timeline_source = "model"
                interval = 1.0 / target_fps if target_fps else None
            # 实测可变帧率补告警：启发式（流级帧率比较）漏检时，
            # 采样模式的复制帧风险仍然存在，必须让用户知道
            if (
                self.options.sampling.mode is not SamplingMode.ALL
                and len(source_pts) >= 3
                and not heuristic_vfr
                and not timeline_is_uniform(source_pts)
            ):
                warn(
                    f"{video.name} 实测源帧间隔不均匀（可变帧率），"
                    "按目标帧率采样时空档会复制前帧"
                )
            time_is_estimated = timeline_source == "estimated"
            if time_is_estimated:
                warn(
                    f"{video.name} 帧时间未能实测，CSV 中的时间为按平均帧率的估算值"
                )
            result.timeline_source = timeline_source

            self._event(
                callback,
                kind="output_plan",
                path=str(video),
                frames=expected,
                estimated_bytes=result.estimated_bytes,
                container=result.tiff_container,
                truncated=truncated,
            )

            result.state = JobState.WRITING
            self._event(callback, kind="state", path=str(video), state=result.state.value)
            stack_writer = StackWriter(
                stage_target, spec, expected, interval,
            )
            stack_writer.open()
            written = 0
            last_emit = 0.0
            for frame in decoder.iter_frames(
                video, self.options, spec, self.cancelled, self._track_process,
                start_time=start_time, lenient=self.options.lenient_decode,
            ):
                stack_writer.write(written, frame)
                written += 1
                now = time.monotonic()
                if now - last_emit >= PROGRESS_EMIT_INTERVAL_SECONDS:
                    last_emit = now
                    self._event(
                        callback, kind="progress", path=str(video), current=written,
                        total=expected, index=batch_index, batch_total=batch_total,
                    )
            # 末尾无条件补发一次，保证进度以真实总帧数收尾
            self._event(
                callback, kind="progress", path=str(video), current=written,
                total=expected, index=batch_index, batch_total=batch_total,
            )
            self._release_process()
            if self.cancelled.is_set():
                stack_writer.abort()
                remove_partial(stage_target)
                result.state = JobState.CANCELLED
                result.error = "用户取消"
                return result
            result.state = JobState.VALIDATING
            self._event(callback, kind="state", path=str(video), state=result.state.value)
            stack_writer.close(written)
            # 清单先写入暂存区，再与输出一起提交；清单提交失败时回滚输出，
            # 避免出现"输出已提交但无清单、重跑又被拒绝覆盖"的死锁状态
            stage_manifest = stage_root / parent / frame_manifest.name
            write_frame_manifest(
                stage_manifest, info, self.options, final, written,
                input_sha256,
                scheduled_times=scheduled_times,
                source_pts=csv_source_pts,
                timeline_source=timeline_source,
                time_is_estimated=time_is_estimated,
                stream_start_time_s=stream_start,
                cancelled=self.cancelled,
            )
            # 提交区前的最后一道取消与身份检查：进入双文件提交后就不再中断
            if self.cancelled.is_set():
                stack_writer.abort()
                remove_partial(stage_target)
                result.state = JobState.CANCELLED
                result.error = "用户取消"
                return result
            verify_file_identity(video, identity, "解码")
            final.parent.mkdir(parents=True, exist_ok=True)
            # 收窄覆盖竞态窗口：exists 检查与 os.replace 之间原本隔着完整的
            # 哈希+计数+解码过程，另一个实例可能在期间发布同名输出。
            # 提交区内复查把窗口从小时级压到毫秒级；彻底消除需要跨进程锁
            if final.exists() or frame_manifest.exists():
                raise FileExistsError(f"提交时发现输出已被其他任务创建，拒绝覆盖: {final}")
            os.replace(stage_target, final)
            fsync_directory(final.parent)
            try:
                os.replace(stage_manifest, frame_manifest)
                fsync_directory(frame_manifest.parent)
            except OSError:
                remove_partial(final)
                raise
            result.state = JobState.COMPLETED
            result.output_path = str(final)
            result.frame_manifest_path = str(frame_manifest)
            try:
                result.output_bytes = final.stat().st_size
            except OSError:
                result.output_bytes = None
            result.frame_count = written
            return result
        except Exception as error:
            # 清理路径自身也可能失败（如磁盘满导致 abort 的 flush 再抛异常）；
            # 二次异常绝不能掩盖原始错误、更不能逃出本视频的隔离边界终止整批
            cleanup_errors: list[str] = []
            if stack_writer is not None:
                try:
                    stack_writer.abort()
                except Exception as cleanup_error:  # noqa: BLE001
                    cleanup_errors.append(f"writer 清理失败: {cleanup_error}")
            if stage_target is not None:
                try:
                    remove_partial(stage_target)
                except OSError as cleanup_error:
                    cleanup_errors.append(f"暂存文件清理失败: {cleanup_error}")
            result.state = JobState.CANCELLED if self.cancelled.is_set() else JobState.FAILED
            message = "用户取消" if self.cancelled.is_set() else str(error)
            if cleanup_errors:
                message = f"{message}（{'；'.join(cleanup_errors)}）"
            result.error = message
            result.traceback = traceback.format_exc()
            return result
        finally:
            result.warnings = list(warnings)
            self._release_process()
