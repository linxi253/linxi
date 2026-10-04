# -*- coding: utf-8 -*-
"""10-DSH集成 流水线回归测试。

覆盖本轮修复的真实失配/协议/失败路径（不是 mock 整条业务链的部分另见
``test_pipeline_end_to_end.py``）：

* R11 API 失配：``ExtractOptions`` 没有 compression；``Sampling(ALL)`` 必须
  ``value=None`` 而 CLI 默认 5.0 —— 用真实 ``video_extractor.models`` 校验。
* 压缩语义：``none``/``deflate`` 作用在本流水线最终校正 TIFF 上，不静默丢弃。
* JSON 协议：stdout 每行严格 JSON，嵌套非有限值与 numpy 标量都被清洗。
* 清单：BOM 用 ``utf-8-sig`` 读；``output_file`` 精确指向最终 TIFF。
* 输入校验：非法采样/压缩/裁剪在昂贵提取前清晰报错。
* 产物格式：提取堆栈是 ImageJ 普通 TIFF（不是 OME）。

运行：``.venv\\Scripts\\python -X utf8 -m pytest tests -q``
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

DRIVER_DIR = Path(__file__).resolve().parent.parent
SUITE_ROOT = DRIVER_DIR.parent
sys.path.insert(0, str(DRIVER_DIR))

import tem_pipeline as pipeline  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _tools_loaded():
    pipeline.load_tools(pipeline.DEFAULT_EXTRACTOR_DIR, pipeline.DEFAULT_DRIFT_DIR)


def make_args(**overrides):
    """构造与 build_parser() 同形状的参数对象。"""
    args = pipeline.build_parser().parse_args(
        ["--video", str(DRIVER_DIR / "nonexistent.mkv"),
         "--output-root", str(DRIVER_DIR / "out")]
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


# ---------------------------------------------------------------------------
# R11：API 失配
# ---------------------------------------------------------------------------
def test_build_extract_options_matches_real_api():
    """必须按 video_extractor 现有 API 组装，且不传 compression 字段。"""
    from video_extractor.models import ExtractOptions

    options = pipeline.build_extract_options(make_args(sampling="all"))
    assert isinstance(options, ExtractOptions)
    assert not hasattr(options, "compression"), (
        "ExtractOptions 没有 compression 字段，不应存在 Compression 适配"
    )
    assert not hasattr(pipeline, "Compression") or "Compression" not in dir(pipeline)


def test_sampling_all_drops_default_value():
    """sampling=all 必须传 value=None，否则 Sampling.validate 拒绝。"""
    sampling = pipeline.build_sampling("all", 5.0)      # CLI 默认 5.0
    assert sampling.value is None
    sampling.validate()                                 # 不抛异常即为正确适配


def test_sampling_target_fps_and_interval_keep_value():
    fps = pipeline.build_sampling("target_fps", 5.0)
    assert fps.value == 5.0 and fps.target_fps == 5.0
    interval = pipeline.build_sampling("interval", 0.2)
    assert interval.value == 0.2 and interval.target_fps == pytest.approx(5.0)
    for mode in ("target_fps", "interval"):
        pipeline.build_sampling(mode, 1.0).validate()


def test_sampling_modes_require_value_when_not_all():
    with pytest.raises(ValueError, match="sampling-value"):
        pipeline.build_sampling("target_fps", None)


def test_no_compression_import_left_in_pipeline():
    """源码里不得再 import/调用 Compression（会 ImportError）。"""
    source = (DRIVER_DIR / "tem_pipeline.py").read_text(encoding="utf-8")
    assert "import Compression" not in source
    assert "Compression(" not in source
    assert "from video_extractor.models import (\n        BitDepth,\n        ColorMode,\n        Compression" not in source


# ---------------------------------------------------------------------------
# 压缩语义：作用在最终校正 TIFF 上
# ---------------------------------------------------------------------------
def test_write_corrected_stack_applies_none_and_deflate(tmp_path):
    """none=未压缩、deflate=Deflate，且两者像素值一致（不改变科学数据）。"""
    import tifffile

    from drift_core import DriftCorrector

    frames = [np.full((32, 32), 100 + i, dtype=np.uint8) for i in range(3)]
    shifts_x = [0.0, 1.0, 2.0]
    shifts_y = [0.0, 0.0, 0.0]
    meta = {"dtype": np.dtype(np.uint8), "n_frames": 3, "shape": (32, 32)}

    results = {}
    for mode in ("none", "deflate"):
        target = tmp_path / f"corrected_{mode}.tif"
        args = make_args(compression=mode, crop_mode="keep")
        written = pipeline.write_corrected_stack(
            frames, shifts_x, shifts_y, dict(meta), target, args)
        assert written == 3
        with tifffile.TiffFile(target) as tif:
            results[mode] = (str(tif.pages[0].compression.name), tif.asarray())

    assert results["none"][0] == "NONE"
    assert results["deflate"][0] == "DEFLATE"
    # 期望的校正结果由漂移核心决定；两种压缩必须一致
    expected = [DriftCorrector.correct_single_frame(f, float(x), float(y))
                for f, x, y in zip(frames, shifts_x, shifts_y)]
    assert np.array_equal(results["none"][1], np.asarray(expected))
    assert np.array_equal(results["deflate"][1], results["none"][1])


def test_write_corrected_stack_rejects_unknown_compression(tmp_path):
    args = make_args(compression="lzma", crop_mode="keep")
    with pytest.raises(ValueError, match="compression"):
        pipeline.write_corrected_stack(
            [np.zeros((8, 8), np.uint8)], [0.0], [0.0],
            {"dtype": np.dtype(np.uint8), "n_frames": 1, "shape": (8, 8)},
            tmp_path / "x.tif", args)


@pytest.mark.parametrize("mode", ["none", "deflate"])
def test_write_corrected_stack_keeps_core_safety(mode, tmp_path):
    """两种编码都必须保留 drift_core 的全部保护（回归 R17）。

    早期 deflate 分支另写了一套 tifffile 直写：覆盖已有目标、覆盖同源、
    失败留下 final —— 三条保护全部失效。
    """
    frames = [np.arange(64, dtype=np.uint8).reshape(8, 8)] * 2
    meta = {"dtype": frames[0].dtype, "shape": (8, 8), "n_frames": 2}
    args = make_args(compression=mode, crop_mode="keep")

    # 1) 已有目标：拒绝且字节不变
    existing = tmp_path / f"{mode}-existing.tif"
    existing.write_bytes(b"SYNTHETIC_DO_NOT_OVERWRITE")
    sha = hashlib.sha256(existing.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        pipeline.write_corrected_stack(frames, [0, 0], [0, 0], dict(meta),
                                       existing, args)
    assert hashlib.sha256(existing.read_bytes()).hexdigest() == sha

    # 2) 同源：拒绝且字节不变
    same = tmp_path / f"{mode}-same.tif"
    same.write_bytes(b"SYNTHETIC_SOURCE")
    sha = hashlib.sha256(same.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="不能覆盖输入"):
        pipeline.write_corrected_stack(frames, [0, 0], [0, 0],
                                       {**meta, "source_path": str(same)}, same, args)
    assert hashlib.sha256(same.read_bytes()).hexdigest() == sha

    # 3) 第 2 页写失败：不得留下 final，也不得留下 .part 残留
    import tifffile

    target = tmp_path / f"{mode}-partial.tif"
    original_write = tifffile.TiffWriter.write
    calls = [0]

    def fail_second(self, *pos, **kw):
        calls[0] += 1
        if calls[0] == 2:
            raise OSError("SYNTHETIC_WRITE_FAILURE")
        return original_write(self, *pos, **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tifffile.TiffWriter, "write", fail_second)
        with pytest.raises(OSError, match="SYNTHETIC_WRITE_FAILURE"):
            pipeline.write_corrected_stack(frames, [0, 0], [0, 0], dict(meta),
                                           target, args)
    assert not target.exists(), "失败后不得留下最终文件"
    assert not list(tmp_path.glob("*.part.tif")), "临时文件未清理"


@pytest.mark.parametrize("mode", ["none", "deflate"])
def test_write_corrected_stack_preserves_metadata_and_crop(mode, tmp_path):
    """两种编码都必须保留 ImageJ 元数据、resolution 与共同有效区裁剪。"""
    import tifffile

    rng = np.random.default_rng(7)
    frames = [rng.integers(0, 255, (32, 32), dtype=np.uint8) for _ in range(3)]
    meta = {"dtype": np.dtype(np.uint8), "shape": (32, 32), "n_frames": 3,
            "imagej": True,
            "source_path": str(tmp_path / "source.tif"),
            "resolution": (25.0, 25.0), "resolutionunit": "CENTIMETER"}
    target = tmp_path / f"meta-{mode}.tif"
    written = pipeline.write_corrected_stack(
        frames, [0.0, 2.0, 4.0], [0.0, 1.0, 2.0], meta, target,
        make_args(compression=mode, crop_mode="crop"))
    assert written == 3

    with tifffile.TiffFile(target) as tif:
        imagej = tif.imagej_metadata or {}
        assert imagej.get("frames") == 3
        assert imagej.get("ImageJ") == 1.53, "ImageJ 版本标记缺失"
        assert tif.series[0].axes == "TYX"
        page = tif.pages[0]
        # 共同有效区由 drift_core 计算（位移 x 最大 4px、y 最大 2px）
        assert tuple(page.shape) == (30, 28), tuple(page.shape)
        assert page.tags.get("XResolution") is not None, "resolution 丢失"
    # 时间轴/ImageJ 描述对两种编码都保留
    assert b"ImageJ=1.53" in target.read_bytes()


# ---------------------------------------------------------------------------
# JSON 协议
# ---------------------------------------------------------------------------
def test_clean_json_value_handles_nested_and_numpy():
    """嵌套非有限值转 None，numpy 标量转原生（否则 allow_nan=False 会抛）。"""
    payload = {
        "ratio": float("nan"),
        "nested": {"inf": float("inf"), "list": [1.0, float("-inf"), 3.0]},
        "np_int": np.int64(7),
        "np_float": np.float32(1.5),
        "np_bool": np.bool_(True),
        "np_array": np.array([1, 2, 3]),
        "path": Path("a/b"),
        "ok": 1.0,
    }
    cleaned = pipeline._clean_json_value(payload)
    assert cleaned["ratio"] is None
    assert cleaned["nested"]["inf"] is None
    assert cleaned["nested"]["list"] == [1.0, None, 3.0]
    assert cleaned["np_int"] == 7 and isinstance(cleaned["np_int"], int)
    assert isinstance(cleaned["np_float"], float)
    assert isinstance(cleaned["np_bool"], bool)
    assert isinstance(cleaned["np_array"], list)
    assert cleaned["path"] == str(Path("a/b"))
    # 严格 JSON 必须能序列化
    json.loads(json.dumps(cleaned, ensure_ascii=False, allow_nan=False),
               parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def test_emit_writes_strict_json_on_stdout(capsys):
    """stdout 每行一个 JSON；非有限值不得让 emit 抛异常。"""
    pipeline.emit("info", message="x", ratio=float("nan"),
                  nested={"v": float("inf")}, npv=np.int64(3))
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    payload = json.loads(out[0], parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    assert payload["type"] == "info"
    assert payload["ratio"] is None and payload["nested"]["v"] is None
    assert payload["npv"] == 3


def test_clean_json_value_handles_zero_dim_array():
    """0 维数组（np.array(nan)/np.array(1.0)）不得抛 TypeError（回归 R17）。"""
    assert pipeline._clean_json_value({"v": np.array(float("nan"))}) == {"v": None}
    assert pipeline._clean_json_value({"v": np.array(3.0)}) == {"v": 3.0}
    assert pipeline._clean_json_value({"v": np.array(3)}) == {"v": 3}
    text = json.dumps(pipeline.clean_json_payload(
        {"a": np.array([[1.0, float("nan")]]), "b": np.array(float("inf"))}),
        allow_nan=False)
    assert json.loads(text) == {"a": [[1.0, None]], "b": None}


def test_two_main_calls_route_to_their_own_streams():
    """同一进程内两次 main 调用必须各自写入当时的流（回归 R17）。

    模块 import 时固定通道会让第二次调用的事件跑到第一次的流里；这里两个
    StringIO 分别收到各自的 error，互不串流。
    """
    first, second = io.StringIO(), io.StringIO()
    original = sys.stdout
    with contextlib.redirect_stdout(first):
        code_a = pipeline.main(["--video", "absent.mkv", "--output-root", "out",
                                "--sampling-value", "0", "--crop-mode", "bad"])
    first_text = first.getvalue()
    with contextlib.redirect_stdout(second):
        code_b = pipeline.main(["--video", "absent.mkv", "--output-root", "out",
                                "--sampling", "all", "--bit-depth", "12"])
    second_text = second.getvalue()

    assert sys.stdout is original, "调用后必须恢复 stdout"
    assert code_a == 1 and code_b == 1
    assert first_text.strip() and second_text.strip()
    msg_a = json.loads([l for l in first_text.splitlines() if l.strip()][-1])
    msg_b = json.loads([l for l in second_text.splitlines() if l.strip()][-1])
    assert msg_a["type"] == "error" and msg_b["type"] == "error"
    # 关键：两次调用的内容必须不同，且各自只出现在自己的流里
    assert msg_a["message"] != msg_b["message"]
    assert msg_a["message"] not in second_text
    assert msg_b["message"] not in first_text
    assert pipeline._JSON_OUT is None or pipeline._JSON_OUT is sys.stdout


def test_main_error_and_summary_go_to_call_stream(tmp_path):
    """错误与成功都写到本次调用的流；成功后通道还原。"""
    stream = io.StringIO()
    video = tmp_path / "absent.mkv"
    with contextlib.redirect_stdout(stream):
        code = pipeline.main(["--video", str(video), "--output-root",
                              str(tmp_path / "out")])
    assert code == 1
    lines = [ln for ln in stream.getvalue().splitlines() if ln.strip()]
    assert json.loads(lines[-1])["type"] == "error"
    assert pipeline._JSON_OUT is None or pipeline._JSON_OUT is sys.stdout


def test_json_output_channel_restores_even_on_exception():
    """通道作用域异常退出也必须还原 stdout 与旧通道。"""
    saved = pipeline._JSON_OUT
    stream = io.StringIO()
    with pytest.raises(RuntimeError):
        with pipeline.json_output_channel(stream):
            assert pipeline._json_channel() is stream
            raise RuntimeError("boom")
    assert pipeline._JSON_OUT is saved
    assert pipeline._json_channel() is not stream


def test_third_party_output_restores_stdout():
    """上游输出导 stderr 后必须恢复原 stdout（不能永久重绑定）。"""
    original = sys.stdout
    with pipeline.third_party_output_on_stderr():
        assert sys.stdout is sys.stderr
        print("upstream banner")
    assert sys.stdout is original
    # 异常路径也必须恢复
    with pytest.raises(RuntimeError):
        with pipeline.third_party_output_on_stderr():
            raise RuntimeError("boom")
    assert sys.stdout is original


# ---------------------------------------------------------------------------
# 清单：BOM + output_file 精确映射
# ---------------------------------------------------------------------------
def test_rewrite_manifest_output_file_exact_target(tmp_path):
    """output_file 必须逐行指向最终 TIFF；不得有目录前缀后门。"""
    manifest = tmp_path / "frames.csv"
    stage_tif = str(tmp_path / ".stage" / "job1" / "out" / "clip.tif")
    final_tif = str(tmp_path / "clip" / "01_raw_clip_stack.tif")
    with manifest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["output_index", "output_file"])
        writer.writeheader()
        writer.writerow({"output_index": 0, "output_file": stage_tif})
        writer.writerow({"output_index": 1, "output_file": stage_tif})
        # 同前缀但不同文件：必须原样保留（回归 R17 的前缀后门）
        writer.writerow({"output_index": 2, "output_file": stage_tif + ".extra"})

    pipeline.rewrite_manifest_output_file(manifest, stage_tif, final_tif)

    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames
    assert fields[0] == "output_index", "BOM 未正确处理会让首列键名带前缀"
    assert [r["output_file"] for r in rows[:2]] == [final_tif] * 2
    assert rows[2]["output_file"] == stage_tif + ".extra", (
        "同前缀但不同文件的值被前缀后门误改")
    assert not any("_stage" in r["output_file"] for r in rows[:2])


def test_rewrite_manifest_repairs_dead_staging_input_path(tmp_path):
    """input_path 必须改回用户原始视频，不能留下马上被删的 staging 链接。"""
    manifest = tmp_path / "frames.csv"
    staged = str(tmp_path / ".stage" / "job1" / "in" / "clip.mkv")
    original = tmp_path / "originals" / "clip.mkv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["output_index", "output_file",
                                                    "input_path", "input_sha256"])
        writer.writeheader()
        writer.writerow({"output_index": 0, "output_file": "x", "input_path": staged,
                         "input_sha256": "abc"})

    pipeline.rewrite_manifest_output_file(manifest, "old", "new",
                                          original_input=original, staged_input=Path(staged))

    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle))
    assert row["input_path"] == str(original)
    assert ".stage" not in row["input_path"]
    assert row["input_sha256"] == "abc", "哈希不应被改写（本来就取自原始输入）"


def test_rewrite_manifest_is_tolerant_of_missing_file(tmp_path):
    pipeline.rewrite_manifest_output_file(tmp_path / "absent.csv", "a", "b")


def test_read_time_axis_handles_bom_manifest(tmp_path):
    """上游以 utf-8-sig 写清单；读取端用 utf-8 会读不到 scheduled_time_s。"""
    manifest = tmp_path / "frames.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["output_index", "scheduled_time_s"])
        writer.writeheader()
        for index, value in enumerate([0.0, 0.5, 1.0]):
            writer.writerow({"output_index": index, "scheduled_time_s": value})
    assert pipeline.read_time_axis(manifest, 3) == [0.0, 0.5, 1.0]


def test_video_manifest_roundtrips_with_pipeline_reader(tmp_path):
    """上游真实写出的清单必须能被流水线读回（编码契约的两端一致）。"""
    from video_extractor.manifest import write_frame_manifest
    from video_extractor.models import ExtractOptions, VideoInfo

    info = VideoInfo(path=tmp_path / "v.mkv", width=64, height=64, duration_s=1.0,
                     average_fps=5.0, nominal_fps=5.0, frame_count=3,
                     pixel_format="gray", bits_per_sample=8, color_family="gray",
                     codec="ffv1", is_variable_fps=False)
    manifest = tmp_path / "v_frames.csv"
    write_frame_manifest(manifest, info, ExtractOptions(), tmp_path / "out.tif",
                         3, "ab" * 32)
    raw = manifest.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "清单应以 BOM 写出（便于 Excel 打开）"
    # 采样模式为默认 ALL：output_time_seconds 按平均帧率 5fps → 0/0.2/0.4
    assert pipeline.read_time_axis(manifest, 3) == [0.0, 0.2, 0.4]


# ---------------------------------------------------------------------------
# 输入校验：昂贵提取之前
# ---------------------------------------------------------------------------
def test_validate_args_rejects_bad_inputs():
    """非法采样/裁剪等必须在昂贵提取前以 ValueError 报错。

    不用 ``parser.error``（退出码 2、只写 stderr）：上层按"stdout 每行 JSON、
    末行为 error"消费，必须拿到可解析的失败原因。
    """
    parser = pipeline.build_parser()

    def rejected(argv_extra):
        args = parser.parse_args(
            ["--video", "v.mkv", "--output-root", "o", *argv_extra])
        with pytest.raises(ValueError) as exc:
            pipeline.validate_args(args)
        return str(exc.value)

    assert "sampling-value" in rejected(
        ["--sampling", "target_fps", "--sampling-value", "0"])
    assert "sampling-value" in rejected(
        ["--sampling", "target_fps", "--sampling-value", "nan"])
    assert "sampling-value" in rejected(
        ["--sampling", "target_fps", "--sampling-value", "inf"])
    assert "skip-interval" in rejected(["--skip-interval", "0"])
    assert "max-stack-gib" in rejected(["--max-stack-gib", "-1"])
    assert "crop-rect" in rejected(["--crop-rect", "0,0,2,1"])
    assert "crop-rect" in rejected(["--crop-rect", "0,0,1"])


def test_validate_args_ignores_sampling_value_in_all_mode():
    """all 模式不消费 sampling-value，其默认 5.0 不得被判非法。"""
    parser = pipeline.build_parser()
    args = parser.parse_args(["--video", "v.mkv", "--output-root", "o",
                              "--sampling", "all", "--sampling-value", "0"])
    pipeline.validate_args(args)      # 不抛异常


def test_argparse_choices_reject_unknown_enum_values():
    """choices 约束必须拦住 compression/crop-mode/bit-depth 的非法值。

    解析失败走协议 error（``ValueError`` → main 转成末行 error + 退出码 1），
    不再只写 stderr 后以 2 退出 —— 上层要能从 stdout 拿到可解析的失败原因。
    """
    parser = pipeline.build_parser()
    bad = {"--compression": "lzma", "--crop-mode": "bad", "--bit-depth": "12"}
    for flag, value in bad.items():
        with pytest.raises(ValueError) as exc:
            parser.parse_args(["--video", "v.mkv", "--output-root", "o",
                               flag, value])
        assert "参数解析失败" in str(exc.value)


def test_help_still_exits_zero_with_normal_help():
    """--help 必须保持 argparse 的正常帮助行为（退出码 0）。"""
    parser = pipeline.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_validate_args_accepts_defaults():
    parser = pipeline.build_parser()
    args = parser.parse_args(["--video", "v.mkv", "--output-root", "o"])
    pipeline.validate_args(args)     # 默认值必须合法


def test_parse_crop_rect_rejects_invalid():
    assert pipeline.parse_crop_rect("0,0,0.5,1", (100, 200)) == (0, 0, 100, 100)
    for bad in ("0,0,1", "0,0,1,1,1", "a,b,c,d", "0,0,0.5,1.5", "0.5,0,0.5,1"):
        with pytest.raises(ValueError):
            pipeline.parse_crop_rect(bad, (100, 200))


def test_validate_args_accepts_narrow_crops_that_a_1x1_shape_would_reject():
    """文档用例与半宽裁剪必须通过校验（回归 R17）。

    早期实现用 ``parse_crop_rect(value, (1, 1))`` 做校验：1×1 下任何 <1 的
    分数都会 round 成 0，于是 0,0,0.3333,1 与 0,0,0.5,1 被误判为非法，
    而真实 256×256 帧完全合法。
    """
    parser = pipeline.build_parser()
    for rect, expected in (("0,0,0.3333,1", (0, 0, 85, 256)),
                           ("0,0,0.5,1", (0, 0, 128, 256)),
                           ("0.2,0.2,0.8,0.8", (51, 51, 205, 205))):
        args = parser.parse_args(["--video", "v.mkv", "--output-root", "o",
                                  "--crop-rect", rect])
        pipeline.validate_args(args)                       # 不抛异常
        assert pipeline.parse_crop_rect(rect, (256, 256)) == expected


def test_validate_args_rejects_illegal_crop_fractions_before_extraction():
    """分数层非法（越界/倒置/非有限/形状错）必须在提取前拒绝。"""
    parser = pipeline.build_parser()
    for rect in ("0,0,2,1", "0,0,1", "0.5,0,0.5,1", "0,0,1,nan", "0,0,1,inf",
                 "1,0,0,1", "0,1,1,0"):
        args = parser.parse_args(["--video", "v.mkv", "--output-root", "o",
                                  "--crop-rect", rect])
        with pytest.raises(ValueError, match="crop-rect"):
            pipeline.validate_args(args)


def test_parse_crop_fractions_is_dimension_free():
    """分数校验不依赖帧尺寸，因此不会把合法窄裁剪判死。"""
    assert pipeline.parse_crop_fractions("0,0,0.3333,1") == (0.0, 0.0, 0.3333, 1.0)
    assert pipeline.parse_crop_fractions("0,0,0.004,1")[2] == 0.004


def test_output_root_may_be_input_directory(tmp_path):
    """output-root 指向输入视频所在目录是**合法**用法（产物进子目录，不覆盖输入）。

    早期实现无依据地禁止该用法（回归 2026-10-03 R17 更正）。真正的保护在
    最终文件层与唯一新建产物目录层，见
    :func:`test_reserve_final_dir_never_reuses_existing`。
    """
    video = tmp_path / "clip.mkv"
    video.write_bytes(b"x")
    args = make_args()
    reserved = pipeline._reserve_final_dir(tmp_path, video.stem)
    assert reserved.parent == tmp_path and reserved.name == "clip"
    assert video.read_bytes() == b"x", "保留目录不得影响输入"
    # 目录名与视频同前缀但不同路径：不会覆盖输入
    assert reserved != video


def test_reserve_final_dir_never_reuses_existing(tmp_path):
    """同一秒内重复保留必须得到不同目录，绝不复用已存在产物目录。"""
    first = pipeline._reserve_final_dir(tmp_path, "clip")
    second = pipeline._reserve_final_dir(tmp_path, "clip")
    third = pipeline._reserve_final_dir(tmp_path, "clip")
    assert len({first, second, third}) == 3, (first, second, third)
    for path in (first, second, third):
        assert path.is_dir()


def test_output_root_equal_to_video_file_rejected(tmp_path):
    """output-root 就是输入文件本身时必须立刻拒绝。"""
    video = tmp_path / "clip.mkv"
    video.write_bytes(b"x")
    with pytest.raises(ValueError, match="输入视频文件本身"):
        pipeline.run_highlevel_pipeline(video, video, make_args())
    assert video.read_bytes() == b"x"


def test_highlevel_rejects_unsupported_extension(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="不支持的视频格式"):
        pipeline.run_highlevel_pipeline(bad, tmp_path / "out", make_args())


def test_highlevel_rejects_missing_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.run_highlevel_pipeline(tmp_path / "absent.mkv", tmp_path / "out",
                                        make_args())


# ---------------------------------------------------------------------------
# 格式：提取堆栈是 ImageJ 普通 TIFF，不是 OME
# ---------------------------------------------------------------------------
def test_extraction_output_name_is_not_ome():
    """产物名不得声称 OME：上游明确拒绝 OME 并要求 is_imagej。"""
    source = (DRIVER_DIR / "tem_pipeline.py").read_text(encoding="utf-8")
    assert "stack.ome.tif" not in source
    assert "01_raw_{stem}_stack.tif" in source


def test_writers_reject_ome_and_require_imagej():
    """上游契约：写出校验拒绝 OME、要求 ImageJ（本流水线的格式依据）。"""
    from video_extractor.writers import OutputValidationError, _validate_tiff
    import tifffile

    import inspect
    source = inspect.getsource(_validate_tiff)
    assert "is_ome" in source and "is_imagej" in source
    assert issubclass(OutputValidationError, ValueError)


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# S1c/R26：FFmpeg 解析与编码失败分类
#
# 这些用例调用**真实**的 _ffmpeg_exe() 与**真实**的 synthetic_video 编码分支，
# 只把解析器/编码进程换成受控实现，而不是测试一个与真实路径脱节的分类函数。
# 覆盖：资产损坏（bundle 哈希不符 / ffprobe 版本失败）、完全没有资源、
# CI 明确要求提供资源、已解析到可执行文件后编码非 0 / 编码成功。
# ---------------------------------------------------------------------------
def _e2e():
    import importlib.util
    import sys as _sys
    from pathlib import Path as _Path

    path = _Path(__file__).resolve().parent / "test_pipeline_end_to_end.py"
    spec = importlib.util.spec_from_file_location("e2e_under_test", path)
    module = importlib.util.module_from_spec(spec)
    _sys.modules["e2e_under_test"] = module
    spec.loader.exec_module(module)
    return module


class _RaisingResolver:
    """Stand-in FFmpegManager whose resolve() raises a given FFmpegError."""

    def __init__(self, message: str):
        self._message = message

    def resolve(self):
        from video_extractor.ffmpeg import FFmpegError

        raise FFmpegError(self._message)


DAMAGED_CASES = [
    "PROVENANCE sha256 mismatch for bundled ffmpeg.exe",
    "ffprobe found but -version exited 1",
]


@pytest.mark.parametrize("message", DAMAGED_CASES)
def test_damaged_asset_fails_the_real_resolver_helper(message, monkeypatch):
    """A *found but damaged* asset must FAIL, never skip.

    round26 showed both of these ending as Skipped with ci true/false alike.
    """
    e2e = _e2e()
    monkeypatch.setitem(sys.modules, "video_extractor", None) if False else None
    sys.path.insert(0, str(e2e.pipeline.DEFAULT_EXTRACTOR_DIR))
    import video_extractor.ffmpeg as upstream

    monkeypatch.setattr(upstream, "FFmpegManager",
                        lambda *a, **k: _RaisingResolver(message))
    with pytest.raises(BaseException) as excinfo:
        e2e._ffmpeg_exe()
    assert excinfo.type is not pytest.skip.Exception, \
        f"damaged asset must not skip: {message}"
    assert "解析失败" in str(excinfo.value)
    assert message in str(excinfo.value)


@pytest.mark.parametrize("ci", [True, False])
def test_no_asset_also_fails_not_skips(monkeypatch, ci):
    """With no candidate asset at all the chain still fails (never skips).

    The end-to-end suite requires FFmpeg by contract; a silent skip there is
    what allowed a damaged asset to look like an absent one.
    """
    if ci:
        monkeypatch.setenv("CI", "true")
    e2e = _e2e()
    sys.path.insert(0, str(e2e.pipeline.DEFAULT_EXTRACTOR_DIR))
    import video_extractor.ffmpeg as upstream

    monkeypatch.setattr(upstream, "FFmpegManager",
                        lambda *a, **k: _RaisingResolver("no FFmpeg candidate found"))
    with pytest.raises(BaseException) as excinfo:
        e2e._ffmpeg_exe()
    assert excinfo.type is not pytest.skip.Exception
    assert "no FFmpeg candidate found" in str(excinfo.value)


def test_working_resolver_returns_the_executable(monkeypatch):
    """The success path still returns the resolved path."""
    e2e = _e2e()
    sys.path.insert(0, str(e2e.pipeline.DEFAULT_EXTRACTOR_DIR))
    import video_extractor.ffmpeg as upstream

    class _Ok:
        def resolve(self):
            from pathlib import Path as _Path

            return type("P", (), {"ffmpeg": _Path("X:/ffmpeg.exe"),
                                  "ffprobe": _Path("X:/ffprobe.exe")})()

    monkeypatch.setattr(upstream, "FFmpegManager", lambda *a, **k: _Ok())
    assert e2e._ffmpeg_exe() == Path("X:/ffmpeg.exe")


@pytest.mark.parametrize("code,expected", [(0, "ok"), (1, "failed"), (127, "failed"),
                                           (2, "failed")])
def test_encode_classification_has_no_skip_outcome(code, expected):
    import subprocess

    e2e = _e2e()
    result = subprocess.CompletedProcess([], code, "boom")
    assert e2e.classify_encode(result) == expected


def test_fixture_encoding_failure_fails_the_real_fixture(monkeypatch, tmp_path_factory):
    """Drive the REAL synthetic_video fixture with a failing encoder.

    This is the actual code path a damaged/参数回归 would take, not a detached
    classifier: the fixture must raise a failure, not skip.
    """
    import subprocess

    e2e = _e2e()
    sys.path.insert(0, str(e2e.pipeline.DEFAULT_EXTRACTOR_DIR))
    import video_extractor.ffmpeg as upstream

    class _Ok:
        def resolve(self):
            from pathlib import Path as _Path

            return type("P", (), {"ffmpeg": _Path(sys.executable),
                                  "ffprobe": _Path(sys.executable)})()

    monkeypatch.setattr(upstream, "FFmpegManager", lambda *a, **k: _Ok())
    monkeypatch.setattr(
        e2e, "_encode",
        lambda *a, **k: subprocess.CompletedProcess([], 1, "", "Unknown encoder 'ffv1'"))
    with pytest.raises(BaseException) as excinfo:
        e2e.synthetic_video.__wrapped__(tmp_path_factory)
    assert excinfo.type is not pytest.skip.Exception, "must fail, not skip"
    assert "编码失败" in str(excinfo.value)
    assert "Unknown encoder" in str(excinfo.value)


def test_fixture_encoding_success_returns_the_video(monkeypatch, tmp_path_factory):
    """Same real fixture, successful encoder -> a real file is produced."""
    import subprocess

    e2e = _e2e()
    sys.path.insert(0, str(e2e.pipeline.DEFAULT_EXTRACTOR_DIR))
    import video_extractor.ffmpeg as upstream

    class _Ok:
        def resolve(self):
            from pathlib import Path as _Path

            return type("P", (), {"ffmpeg": _Path(sys.executable),
                                  "ffprobe": _Path(sys.executable)})()

    monkeypatch.setattr(upstream, "FFmpegManager", lambda *a, **k: _Ok())
    captured = {}

    def fake_encode(ffmpeg, args):
        target = Path(args[-1])
        target.write_bytes(b"fake-mkv")
        captured["args"] = args
        return subprocess.CompletedProcess([], 0, "")

    monkeypatch.setattr(e2e, "_encode", fake_encode)
    video = e2e.synthetic_video.__wrapped__(tmp_path_factory)
    assert video.is_file() and video.stat().st_size > 0
    assert "-c:v" in captured["args"] and "ffv1" in captured["args"]
