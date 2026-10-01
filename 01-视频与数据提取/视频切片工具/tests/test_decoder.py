from pathlib import Path

import pytest

from video_extractor.decoder import (
    FrameDecoder,
    grayscale_color_filter,
    rgb_color_filter,
    resolve_frame_spec,
)
from video_extractor.models import (
    BitDepth,
    ColorMode,
    ExtractOptions,
    Sampling,
    SamplingMode,
    VideoInfo,
)


def _info(bits: int = 8, stream_index: int = 0) -> VideoInfo:
    return VideoInfo(
        path=Path("sample.mp4"), width=8, height=6, duration_s=2.0,
        average_fps=10.0, nominal_fps=10.0, frame_count=20, pixel_format="yuv420p",
        bits_per_sample=bits, color_family="rgb", codec="h264", is_variable_fps=False,
        stream_index=stream_index,
    )


def test_upscaling_8bit_source_to_16bit_is_rejected() -> None:
    """8 位源选择 16 位输出只是位复制放大，必须拒绝以防伪装数据精度。"""
    with pytest.raises(ValueError, match="位复制"):
        resolve_frame_spec(_info(bits=8), ExtractOptions(bit_depth=BitDepth.UINT16))


def test_rgb_16bit_is_rejected_for_imagej_format() -> None:
    """ImageJ 彩色 TIFF 仅支持 8 位/通道，16 位彩色组合必须显式拒绝。"""
    with pytest.raises(ValueError, match="彩色"):
        resolve_frame_spec(
            _info(bits=16),
            ExtractOptions(bit_depth=BitDepth.UINT16, color_mode=ColorMode.COLOR),
        )


def test_gray_16bit_source_is_accepted() -> None:
    """灰度 16 位输出 ImageJ 原生支持，仍可用。"""
    spec = resolve_frame_spec(
        _info(bits=16),
        ExtractOptions(bit_depth=BitDepth.SOURCE, color_mode=ColorMode.GRAYSCALE),
    )
    assert spec.dtype == "uint16"
    assert spec.channels == 1


def test_high_bit_source_keeps_16bit() -> None:
    spec = resolve_frame_spec(
        _info(bits=10),
        ExtractOptions(bit_depth=BitDepth.UINT16, color_mode=ColorMode.GRAYSCALE),
    )
    assert spec.dtype == "uint16"
    assert spec.ffmpeg_pixel_format == "gray16le"


def test_source_depth_resolves_without_error() -> None:
    spec = resolve_frame_spec(_info(bits=8), ExtractOptions(bit_depth=BitDepth.SOURCE))
    assert spec.dtype == "uint8"


def test_stream_index_propagates_to_spec() -> None:
    """probe 选中的流索引必须传递给解码器，保证两者指向同一流。"""
    spec = resolve_frame_spec(_info(stream_index=2), ExtractOptions())
    assert spec.stream_index == 2


def test_grayscale_color_filter_maps_probe_metadata() -> None:
    filter_text = grayscale_color_filter("bt709", "tv")
    assert "in_color_matrix=1" in filter_text
    assert "out_color_matrix=1" in filter_text
    assert "in_range=1" in filter_text
    assert "out_range=1" in filter_text
    assert grayscale_color_filter(None, None) is None
    assert grayscale_color_filter("bt2020nc", "pc") is not None


def test_rgb_color_filter_declares_input_side_only() -> None:
    """RGB 输出没有输出侧矩阵概念，滤镜只声明输入侧，避免错误的输出范围换算。"""
    filter_text = rgb_color_filter("bt709", "tv")
    assert "scale=in_color_matrix=1" in filter_text
    assert "in_range=1" in filter_text
    assert "out_color_matrix" not in filter_text
    assert "out_range" not in filter_text
    assert rgb_color_filter(None, None) is None


def test_rgb_chain_declares_matrix_for_yuv_source() -> None:
    """YUV 源转 RGB 必须显式声明源矩阵/范围，否则 swscale 按分辨率猜测。"""
    info = _info()
    info = VideoInfo(
        **{**info.__dict__, "color_space": "bt709", "color_range": "tv"}
    )
    options = ExtractOptions(color_mode=ColorMode.COLOR)
    spec = resolve_frame_spec(info, options)
    assert spec.source_pixel_format == "yuv420p"
    filters = FrameDecoder._decode_filters(options, spec)
    assert any(f.startswith("scale=in_color_matrix=1") for f in filters)
    assert "format=rgb24" in filters


def test_rgb_chain_skips_matrix_for_rgb_source_and_unknown_metadata() -> None:
    """RGB 源或元数据缺失时保持旧行为（输出端 -pix_fmt 自动转换），
    但尾部尺寸保险始终存在。"""
    options = ExtractOptions(color_mode=ColorMode.COLOR)
    spec = resolve_frame_spec(
        VideoInfo(
            **{**_info().__dict__, "pixel_format": "bgr24",
               "color_space": None, "color_range": None}
        ),
        options,
    )
    assert spec.source_pixel_format == "bgr24"
    assert FrameDecoder._decode_filters(options, spec) == ["scale=8:6"]


def test_rgb_chain_sampling_filter_still_first() -> None:
    """采样滤镜必须在最前，保证计数与解码链一致。"""
    options = ExtractOptions(
        sampling=Sampling(SamplingMode.TARGET_FPS, 10.0),
        color_mode=ColorMode.COLOR,
    )
    spec = resolve_frame_spec(
        VideoInfo(**{**_info().__dict__, "color_space": "bt709", "color_range": "tv"}),
        options,
    )
    filters = FrameDecoder._decode_filters(options, spec)
    assert filters[0].startswith("fps=")
    assert "scale=in_color_matrix=1" in filters[1]
    assert filters[2] == "format=rgb24"


def test_decode_filters_end_with_scale_pin() -> None:
    """解码链尾部必须是 scale=W:H 尺寸保险：编码流中途改变分辨率时，
    rawvideo 字节流会按新尺寸输出，读取端按旧规格 reshape 即静默转置。"""
    options = ExtractOptions(color_mode=ColorMode.GRAYSCALE)
    spec = resolve_frame_spec(_info(), options)
    filters = FrameDecoder._decode_filters(options, spec)
    assert filters[-1] == "scale=8:6"


def test_decode_filters_pass_start_time_to_fps_filter() -> None:
    """start_time 必须进入 fps 滤镜参数，供延迟起始流复用。"""
    options = ExtractOptions(sampling=Sampling(SamplingMode.TARGET_FPS, 5.0))
    spec = resolve_frame_spec(_info(), options)
    filters = FrameDecoder._decode_filters(options, spec, start_time=1.5)
    assert filters[0].startswith("fps=fps=5:round=near:start_time=1.5")


def test_decode_filters_are_frame_preserving() -> None:
    """计数链一致性的根基：解码链中除 fps 外全部为帧保持滤镜。

    若未来新增会增删帧的滤镜（如 select/decimate），必须同步修改
    _count_filters，否则 count pass 与实际解码帧数会静默不一致。
    """
    options = ExtractOptions(
        sampling=Sampling(SamplingMode.TARGET_FPS, 5.0),
        color_mode=ColorMode.GRAYSCALE,
    )
    spec = resolve_frame_spec(
        VideoInfo(**{**_info().__dict__, "color_space": "bt709", "color_range": "tv"}),
        options,
    )
    frame_dropping = {"select", "decimate", "framestep", "fps", "drop", "trim", "loop"}
    for text in FrameDecoder._decode_filters(options, spec):
        name = text.split("=", 1)[0]
        if name in frame_dropping:
            assert name == "fps", f"解码链新增了帧数相关滤镜 {name}，需同步 _count_filters"


def test_count_filters_are_light_and_capture_source_pts() -> None:
    """计数链只有 showinfo + fps：省去色彩换算，且 showinfo 在 fps 之前
    保证捕获的是源帧 PTS（供实测可变帧率检测）。"""
    options = ExtractOptions(sampling=Sampling(SamplingMode.TARGET_FPS, 5.0))
    assert FrameDecoder._count_filters(options, 0.0) == ["showinfo", "fps=fps=5:round=near:start_time=0"]
    options_all = ExtractOptions()
    assert FrameDecoder._count_filters(options_all, 2.0) == ["showinfo"]


def test_input_args_disable_autorotate_and_enable_strict_decode() -> None:
    """-noautorotate 保证输出与 probe 的编码宽高一致；-xerror 让解码错误
    显式失败而不是静默跳帧后退出码仍为 0。"""
    from video_extractor.models import FrameSpec

    spec = FrameSpec(8, 6, 1, "uint8", "gray", "gray")
    decoder = FrameDecoder(Path("ffmpeg"))
    args = decoder._input_args(Path("v.mp4"), spec, lenient=False)
    assert "-noautorotate" in args
    assert "-xerror" in args
    lenient_args = decoder._input_args(Path("v.mp4"), spec, lenient=True)
    assert "-noautorotate" in lenient_args
    assert "-xerror" not in lenient_args
