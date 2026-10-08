# -*- coding: utf-8 -*-
"""真实完整链测试：合成视频 → CLI → 提取 → 漂移矫正 → 审计（不 mock 业务链）。

与单元测试分层：``test_pipeline_regressions.py`` 是快速单元/边界测试；
本文件通过**真实子进程 CLI**跑端到端，因此慢但覆盖面真实。上游 FFmpeg 由
``video_extractor.ffmpeg`` 自行解析（仓库内 tools/ffmpeg 或 CI 提供的 FFmpeg）。

**失败与跳过的界线**：本套真实全链**要求**可用的 FFmpeg，因此**任何** resolver 错误
都是**测试失败**，不 skip。上游 ``FFmpegError`` 同时覆盖两类情况——完全没有候选资产，
以及"资产存在但损坏"（随包 SHA-256 与 PROVENANCE.md 不符、``ffprobe -version`` 失败、
版本低于下限等）——两者在这里都 fail：把第二类 skip 掉会让供应链/资产回归伪装成
"这台机器没资产"，而在显式下载并校验资产的 CI 上更会把 7 项端到端全部变成 skip 通过的绿。
解析到可执行文件后编码非 0 同样是失败，见 :func:`classify_encode`（只有 ``ok``/``failed``，
没有 skip 结果）；其回归在 ``test_pipeline_regressions.py``，直接驱动真实 ``_ffmpeg_exe``
与真实夹具编码分支。

覆盖：sampling=all/none、all/deflate、target_fps 与 interval 取不同合法值产生
不同帧数、有效 crop（窄裁剪）、JSON 协议、输入哈希不变、位移接近期望、
manifest 回读（output_file 指向最终文件、input_path 指向原始视频）。

所有产物写 ``tmp_path``，不触碰实验数据。
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

DRIVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DRIVER_DIR))

import tem_pipeline as pipeline  # noqa: E402

pytestmark = pytest.mark.slow


def _ffmpeg_exe() -> Path:
    """The FFmpeg this chain must use -- resolved by the same code as the CLI.

    The real chain **requires** FFmpeg, so any resolver error is a failure, not a
    skip. ``FFmpegError`` covers two very different situations:

    * no candidate asset at all (nothing to test with), and
    * an asset was found but is damaged/mismatched (bundled SHA-256 differs from
      PROVENANCE.md, ``ffprobe -version`` fails, version below the floor, ...).

    Skipping on the second kind would hide a real supply-chain/asset regression
    behind an "environment has no FFmpeg" message -- and in CI, which explicitly
    downloads and verifies the asset, it would turn all seven end-to-end cases
    green-with-skips. So neither kind skips here; both fail with the resolver's
    own message, which names the actual cause.
    """
    sys.path.insert(0, str(pipeline.DEFAULT_EXTRACTOR_DIR))
    from video_extractor.ffmpeg import FFmpegError, FFmpegManager

    try:
        return Path(FFmpegManager(None).resolve().ffmpeg)
    except FFmpegError as exc:
        pytest.fail(
            "真实全链需要可用的 FFmpeg，但解析失败（这不是 skip：CI 会显式提供并校验"
            f"该资产）。请在该环境放置 FFmpeg 8.0.3+ 或修复资产。resolver error: {exc}")


def _encode(ffmpeg: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([str(ffmpeg), *args], capture_output=True, text=True,
                          timeout=120)


def classify_encode(result: subprocess.CompletedProcess) -> str:
    """Classify an encode performed by an *already resolved* executable.

    Only ``ok``/``failed`` exist here: once a real FFmpeg has been resolved, a
    non-zero exit is a regression (bad arguments, bad build, API change) and must
    fail. There is deliberately no "skipped" outcome -- that was the defect, and
    it let a broken asset or a parameter regression pass as "no FFmpeg here".
    """
    return "ok" if result.returncode == 0 else "failed"


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory) -> Path:
    """8 帧 256×256 灰度 FFV1 mkv，第 i 帧相对首帧平移 (i, i//2)。"""
    root = tmp_path_factory.mktemp("chain")
    rng = np.random.default_rng(284)
    base = np.zeros((256, 256), dtype=np.uint8) + 25
    import cv2

    for _ in range(180):
        x, y = map(int, rng.integers(16, 240, 2))
        cv2.circle(base, (x, y), int(rng.integers(2, 6)), int(rng.integers(70, 250)), -1)
    base = cv2.GaussianBlur(base, (3, 3), 0.6)
    frames = np.stack([
        cv2.warpAffine(base, np.float32([[1, 0, i], [0, 1, i // 2]]), (256, 256),
                       borderValue=25)
        for i in range(8)
    ])
    raw = root / "frames.raw"
    raw.write_bytes(frames.tobytes())
    video = root / "synthetic-drift.mkv"
    ffmpeg = _ffmpeg_exe()          # fails (never skips) if FFmpeg is unusable
    encode = _encode(ffmpeg, [
        "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "gray", "-video_size", "256x256",
        "-framerate", "5", "-i", str(raw), "-c:v", "ffv1", str(video)])
    if classify_encode(encode) == "failed":
        pytest.fail(f"已解析到 FFmpeg（{ffmpeg}）但合成视频编码失败 "
                    f"(exit {encode.returncode}): {(encode.stderr or '')[:400]}")
    assert video.is_file() and video.stat().st_size > 0
    return video


def run_chain(video: Path, out_root: Path, *extra: str) -> dict:
    """跑一次真实 CLI，返回解析后的结果（含逐行事件与产物回读）。"""
    cmd = [sys.executable, "-X", "utf8", str(DRIVER_DIR / "tem_pipeline.py"),
           "--video", str(video), "--output-root", str(out_root),
           "--no-relax-on-fail", "--no-trim-blank-edges", *extra]
    before = hashlib.sha256(video.read_bytes()).hexdigest()
    proc = subprocess.run(cmd, cwd=str(DRIVER_DIR), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)
    result: dict = {"exit": proc.returncode, "stderr": proc.stderr}
    events, invalid = [], []
    for index, line in enumerate(proc.stdout.splitlines(), 1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(
                line, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s))))
        except (json.JSONDecodeError, ValueError):
            invalid.append({"line": index, "text": line[:200]})
    result["events"] = events
    result["non_json_stdout"] = invalid
    result["last_event"] = events[-1] if events else None
    result["input_unchanged"] = hashlib.sha256(video.read_bytes()).hexdigest() == before
    return result


def read_products(out_root: Path) -> dict:
    """回读产物：TIFF 页数/编码、manifest 映射、位移 CSV。"""
    import tifffile

    dirs = [d for d in out_root.iterdir() if d.is_dir() and d.name != ".stage"]
    assert len(dirs) == 1, f"应只有一个产物目录，实际 {[d.name for d in dirs]}"
    final = dirs[0]
    info: dict = {"dir": final}
    for name, key in (("01_raw", "raw"), ("02_corrected", "corrected")):
        matches = list(final.glob(f"{name}_*.tif"))
        assert matches, f"缺少 {name} 产物"
        with tifffile.TiffFile(matches[0]) as tif:
            info[key] = {
                "path": matches[0], "pages": len(tif.pages),
                "shape": list(tif.series[0].shape), "axes": tif.series[0].axes,
                "compression": str(tif.pages[0].compression.name),
                "is_ome": tif.is_ome, "array": tif.asarray(),
            }
    manifests = list(final.glob("01_raw_*_frames.csv"))
    assert manifests, "缺少逐帧清单"
    raw_bytes = manifests[0].read_bytes()
    with manifests[0].open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        info["manifest_rows"] = list(reader)
        info["manifest_fields"] = reader.fieldnames
    info["manifest_has_bom"] = raw_bytes.startswith(b"\xef\xbb\xbf")
    shifts = list(final.glob("02_drift_shifts.csv"))
    if shifts:
        with shifts[0].open(encoding="utf-8-sig", newline="") as handle:
            info["shifts"] = list(csv.DictReader(handle))
    info["summary"] = json.loads((final / "03_summary.json").read_text(encoding="utf-8"))
    info["audit"] = json.loads((final / "02_audit.json").read_text(encoding="utf-8"))
    return info


# ---------------------------------------------------------------------------
# 完整链：编码 × 采样
# ---------------------------------------------------------------------------
def test_chain_all_none(synthetic_video, tmp_path):
    """sampling=all, compression=none：8 帧全取，未压缩。"""
    res = run_chain(synthetic_video, tmp_path / "out", "--sampling", "all",
                    "--compression", "none")
    assert res["exit"] == 0, res["last_event"]
    assert res["non_json_stdout"] == [], res["non_json_stdout"]
    assert res["last_event"]["type"] == "summary"
    assert res["input_unchanged"], "输入视频被改动"

    info = read_products(tmp_path / "out")
    assert info["raw"]["pages"] == 8
    assert info["raw"]["compression"] == "NONE"
    assert info["corrected"]["compression"] == "NONE"
    assert info["raw"]["is_ome"] is False and info["corrected"]["is_ome"] is False
    assert info["raw"]["axes"] == "TYX"
    assert info["manifest_has_bom"]
    assert info["manifest_fields"][0] == "output_index", "BOM 未正确处理"
    # 提取堆栈写入的是真实 none；最终产物 none
    assert info["audit"]["extraction"]["compression"] == "none"
    assert info["audit"]["extraction"]["sampling_value"] is None, (
        "all 模式实际采样值应为 None")
    assert info["audit"]["correction"]["compression"] == "none"


def test_chain_all_deflate(synthetic_video, tmp_path):
    """sampling=all, compression=deflate：页数/像素与 none 一致，仅编码不同。"""
    none_root, deflate_root = tmp_path / "none", tmp_path / "deflate"
    run_chain(synthetic_video, none_root, "--sampling", "all", "--compression", "none")
    res = run_chain(synthetic_video, deflate_root, "--sampling", "all",
                    "--compression", "deflate")
    assert res["exit"] == 0, res["last_event"]
    assert res["last_event"]["type"] == "summary"

    a = read_products(none_root)
    b = read_products(deflate_root)
    assert b["corrected"]["compression"] == "DEFLATE"
    assert b["corrected"]["pages"] == a["corrected"]["pages"]
    assert b["corrected"]["shape"] == a["corrected"]["shape"]
    assert np.array_equal(b["corrected"]["array"], a["corrected"]["array"]), (
        "压缩改变像素值")
    # deflate 确实更小（无损压缩生效），且元数据仍记录真实编码
    assert b["corrected"]["path"].stat().st_size < a["corrected"]["path"].stat().st_size
    assert b["audit"]["correction"]["compression"] == "deflate"
    assert b["audit"]["extraction"]["compression"] == "none", "提取堆栈实际仍是未压缩"


def test_chain_sampling_modes_yield_different_frame_counts(synthetic_video, tmp_path):
    """target_fps 与 interval 取不同合法值 → 帧数不同且与采样语义一致。"""
    all_res = run_chain(synthetic_video, tmp_path / "all", "--sampling", "all")
    fps_res = run_chain(synthetic_video, tmp_path / "fps", "--sampling", "target_fps",
                        "--sampling-value", "2.5")
    itv_res = run_chain(synthetic_video, tmp_path / "itv", "--sampling", "interval",
                        "--sampling-value", "0.3")
    for res in (all_res, fps_res, itv_res):
        assert res["exit"] == 0, res["last_event"]

    counts = {
        "all": read_products(tmp_path / "all")["raw"]["pages"],
        "target_fps=2.5": read_products(tmp_path / "fps")["raw"]["pages"],
        "interval=0.3": read_products(tmp_path / "itv")["raw"]["pages"],
    }
    assert counts["all"] == 8
    # 8 帧 @5fps = 1.6s；2.5fps → 4 帧；0.3s 间隔 → 5 帧
    assert counts["target_fps=2.5"] == 4, counts
    assert counts["interval=0.3"] == 5, counts
    assert len(set(counts.values())) == 3, f"三种采样应产生不同帧数: {counts}"
    # 实际生效的采样值如实记录
    fps_audit = read_products(tmp_path / "fps")["audit"]
    assert fps_audit["extraction"]["sampling_mode"] == "target_fps"
    assert fps_audit["extraction"]["sampling_value"] == 2.5


def test_chain_with_narrow_crop_uses_real_dimensions(synthetic_video, tmp_path):
    """有效窄裁剪（文档用例）必须真跑通，输出尺寸反映裁剪与共同有效区。"""
    res = run_chain(synthetic_video, tmp_path / "crop", "--sampling", "all",
                    "--crop-rect", "0,0,0.3333,1")
    assert res["exit"] == 0, res["last_event"]
    assert res["last_event"]["type"] == "summary"

    info = read_products(tmp_path / "crop")
    assert info["raw"]["pages"] == 8
    # 原始提取堆栈保持完整帧
    assert info["raw"]["shape"] == [8, 256, 256]
    assert info["audit"]["extraction"]["crop_rect"] == "0,0,0.3333,1"
    # 裁剪先按分数作用在完整帧（宽 round(0.3333*256)=85），检测阶段形状为 (256, 85)
    assert info["audit"]["detection"]["shape"] == [256, 85], (
        info["audit"]["detection"]["shape"])
    # 最终产物再裁到共同有效区：高<256、宽<85（位移最大 7px/3px）
    height, width = info["corrected"]["shape"][1], info["corrected"]["shape"][2]
    assert height < 256 and width < 85, info["corrected"]["shape"]
    assert height >= 240 and width >= 70, f"裁剪量异常: {info['corrected']['shape']}"


def test_chain_manifest_and_shift_readback(synthetic_video, tmp_path):
    """清单/位移回读：output_file 指向最终文件，input_path 仍可追溯。"""
    res = run_chain(synthetic_video, tmp_path / "out", "--sampling", "all")
    assert res["exit"] == 0, res["last_event"]
    info = read_products(tmp_path / "out")
    rows = info["manifest_rows"]
    assert len(rows) == 8

    outputs = {row["output_file"] for row in rows}
    assert len(outputs) == 1, f"output_file 应精确指向同一最终文件: {outputs}"
    final_raw = info["raw"]["path"]
    assert Path(outputs.pop()) == final_raw
    assert final_raw.is_file()
    assert ".stage" not in str(final_raw), "清单不得指向已删除的 staging"
    # 输入可追溯：指向用户原始视频，且哈希为原始输入的哈希
    inputs = {row["input_path"] for row in rows}
    assert inputs == {str(synthetic_video)}, f"input_path 应指向原始视频: {inputs}"
    assert rows[0]["input_sha256"] == hashlib.sha256(
        synthetic_video.read_bytes()).hexdigest()
    assert rows[0]["timeline_source"] in ("measured", "model", "estimated")

    # 位移接近期望：第 i 帧平移 (i, i//2)
    shifts = info["shifts"]
    assert len(shifts) == 8
    max_err = max(
        max(abs(float(r["dx_px"]) - int(r["frame_index"])),
            abs(float(r["dy_px"]) - int(r["frame_index"]) // 2))
        for r in shifts
    )
    assert max_err <= 0.5, f"位移偏离期望 {max_err} px"
    # 误差减少：矫正后帧间差异显著小于提取堆栈
    corrected = info["corrected"]["array"].astype(float)
    inner = corrected[:, 16:-16, 16:-16]
    assert float(np.mean((inner[1:] - inner[0]) ** 2)) < 1.0


def test_chain_rejects_illegal_crop_before_extraction(synthetic_video, tmp_path):
    """非法裁剪必须在昂贵提取前报错（协议 error + 非 0 退出）。"""
    res = run_chain(synthetic_video, tmp_path / "bad", "--crop-rect", "0,0,2,1")
    assert res["exit"] != 0
    assert res["last_event"]["type"] == "error"
    assert "crop-rect" in res["last_event"]["message"]
    assert not (tmp_path / "bad").exists() or not any(
        d.name != ".stage" for d in (tmp_path / "bad").iterdir())


def test_chain_duplicate_run_does_not_destroy_previous(synthetic_video, tmp_path):
    """同一秒内连续两次运行：前一次产物必须保持不变。"""
    root = tmp_path / "repeat"
    run_chain(synthetic_video, root, "--sampling", "all")
    first = read_products(root)
    first_raw = first["raw"]["path"]
    first_sha = hashlib.sha256(first_raw.read_bytes()).hexdigest()

    run_chain(synthetic_video, root, "--sampling", "all")
    dirs = sorted(d.name for d in root.iterdir() if d.is_dir() and d.name != ".stage")
    assert len(dirs) == 2, f"第二次运行应新建目录而不是复用: {dirs}"
    assert first_raw.is_file(), "前一次产物被覆盖/删除"
    assert hashlib.sha256(first_raw.read_bytes()).hexdigest() == first_sha
