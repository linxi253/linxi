# -*- coding: utf-8 -*-
"""核心模块测试脚本：验证滤镜算法与 TIF 读写的正确性。"""

import json
import os
import struct
import tempfile
import numpy as np
import tifffile

from image_filters import (
    to_float, from_float, process_image, default_params, FILTER_PARAMS,
    FILTER_FUNCS,
)
from tif_io import TifDocument, UnsupportedTiffError, file_sha256, list_tif_files

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_images')


def make_test_images():
    """生成测试用 TIF 图像。"""
    os.makedirs(TEST_DIR, exist_ok=True)
    # 清理历史测试产物，避免残留的 out_* 干扰输入目录统计
    for name in os.listdir(TEST_DIR):
        if name.startswith('out_') and name.lower().endswith(('.tif', '.tiff')):
            try:
                os.remove(os.path.join(TEST_DIR, name))
            except OSError:
                pass
    rng = np.random.default_rng(42)

    # 1. RGB 8bit 单页
    rgb8 = (rng.random((200, 300, 3)) * 255).astype(np.uint8)
    tifffile.imwrite(os.path.join(TEST_DIR, 'rgb8.tif'), rgb8)

    # 2. 灰度 8bit 单页
    gray8 = (rng.random((200, 300)) * 255).astype(np.uint8)
    tifffile.imwrite(os.path.join(TEST_DIR, 'gray8.tif'), gray8)

    # 3. RGB 16bit 单页
    rgb16 = (rng.random((150, 200, 3)) * 65535).astype(np.uint16)
    tifffile.imwrite(os.path.join(TEST_DIR, 'rgb16.tif'), rgb16)

    # 4. 灰度堆叠 (5 页)
    stack_gray = (rng.random((5, 100, 120)) * 255).astype(np.uint8)
    tifffile.imwrite(os.path.join(TEST_DIR, 'stack_gray.tif'), stack_gray)

    # 5. RGB 堆叠 (3 页)
    stack_rgb = (rng.random((3, 80, 100, 3)) * 255).astype(np.uint8)
    tifffile.imwrite(os.path.join(TEST_DIR, 'stack_rgb.tif'), stack_rgb)

    print(f"已生成 5 个测试图像于: {TEST_DIR}")


def test_dtype_conversion():
    """测试数据类型转换。"""
    a8 = np.array([[0, 128, 255]], dtype=np.uint8)
    f = to_float(a8)
    assert f.dtype == np.float32
    assert abs(f[0, 0] - 0.0) < 1e-6 and abs(f[0, 2] - 1.0) < 1e-6
    back = from_float(f, np.uint8)
    assert np.array_equal(back, a8), f"{back} != {a8}"

    a16 = np.array([[0, 32768, 65535]], dtype=np.uint16)
    f16 = to_float(a16)
    back16 = from_float(f16, np.uint16)
    assert np.all(np.abs(back16.astype(int) - a16.astype(int)) <= 1)
    print("[OK] 数据类型转换")


def test_each_filter():
    """逐个测试每个滤镜函数不崩溃且输出范围正确。"""
    rng = np.random.default_rng(0)
    rgb = rng.random((64, 64, 3)).astype(np.float32)
    gray = rng.random((64, 64)).astype(np.float32)

    for key, cn, en, mn, mx, default, step in FILTER_PARAMS:
        func = FILTER_FUNCS[key]
        for img, label in [(rgb, 'RGB'), (gray, 'Gray')]:
            # 测试最小值、中间值、最大值
            for val in (mn, (mn + mx) / 2, mx):
                out = func(img, val)
                assert out.shape == img.shape, f"{key} {label} shape mismatch"
                assert np.all(np.isfinite(out)), f"{key} {label} non-finite at {val}"
                # 管线输出应始终被裁剪到 [0, 1]
                params = default_params()
                params[key] = val
                piped = process_image(img, params)
                assert piped.shape == img.shape, f"{key} {label} pipeline shape mismatch"
                assert np.all(np.isfinite(piped)), f"{key} {label} pipeline non-finite"
                assert piped.min() >= 0.0 and piped.max() <= 1.0, \
                    f"{key} {label} pipeline out of range"
        print(f"[OK] {cn} ({en})")


def test_pipeline():
    """测试完整处理管线。"""
    rng = np.random.default_rng(1)
    rgb = rng.random((100, 100, 3)).astype(np.float32)
    gray = rng.random((100, 100)).astype(np.float32)

    # 默认参数应输出与输入相同
    params = default_params()
    out = process_image(rgb, params)
    assert np.allclose(out, rgb, atol=1e-5), "默认参数应不改变图像"

    # 全部参数拉满
    full = {key: mx for key, _, _, _, mx, _, _ in
            [(k, c, e, mn, mx, d, s) for k, c, e, mn, mx, d, s in FILTER_PARAMS]}
    out2 = process_image(rgb, full)
    assert out2.shape == rgb.shape
    assert np.all(np.isfinite(out2))
    assert out2.min() >= 0.0 and out2.max() <= 1.0

    out3 = process_image(gray, full)
    assert out3.shape == gray.shape
    print("[OK] 完整处理管线")


def test_tif_io():
    """测试 TIF 读写。"""
    files = list_tif_files(TEST_DIR)
    basenames = {os.path.basename(f) for f in files}
    expected = {'rgb8.tif', 'gray8.tif', 'rgb16.tif', 'stack_gray.tif', 'stack_rgb.tif'}
    assert expected.issubset(basenames), f"缺少输入图像：{expected - basenames}"

    # RGB 单页
    doc = TifDocument(os.path.join(TEST_DIR, 'rgb8.tif'))
    assert doc.num_frames == 1 and not doc.is_stack
    assert doc.dtype == np.uint8

    # 灰度单页
    doc = TifDocument(os.path.join(TEST_DIR, 'gray8.tif'))
    assert doc.num_frames == 1 and not doc.is_stack

    # 16bit
    doc = TifDocument(os.path.join(TEST_DIR, 'rgb16.tif'))
    assert doc.dtype == np.uint16

    # 灰度堆叠
    doc = TifDocument(os.path.join(TEST_DIR, 'stack_gray.tif'))
    assert doc.is_stack and doc.num_frames == 5, f"堆叠帧数错误: {doc.num_frames}"

    # RGB 堆叠
    doc = TifDocument(os.path.join(TEST_DIR, 'stack_rgb.tif'))
    assert doc.is_stack and doc.num_frames == 3

    print("[OK] TIF 读取（单页/堆叠/RGB/灰度/16bit）")


def test_save_processed():
    """测试处理后保存，验证结构与 dtype 保持。"""
    params = default_params()
    params['contrast'] = 20
    params['saturation'] = 30

    with tempfile.TemporaryDirectory(prefix='tif_test_') as tmp:
        # 堆叠保存
        doc = TifDocument(os.path.join(TEST_DIR, 'stack_gray.tif'))
        out_path = os.path.join(tmp, 'out_stack.tif')
        doc.save_processed(out_path, params)
        re = tifffile.imread(out_path)
        assert re.shape == (5, 100, 120), f"堆叠结构未保持: {re.shape}"
        assert re.dtype == np.uint8
        assert os.path.exists(out_path + '.filter.json'), '缺少参数清单'

        # 单页 RGB 保存
        doc = TifDocument(os.path.join(TEST_DIR, 'rgb16.tif'))
        out_path = os.path.join(tmp, 'out_rgb16.tif')
        doc.save_processed(out_path, params)
        re = tifffile.imread(out_path)
        assert re.shape == (150, 200, 3) and re.dtype == np.uint16

    print("[OK] 处理后保存（结构/数据类型保持）")


def test_uint32_and_extra_channels():
    """uint32 精确往返；RGBA 输入时模糊类滤镜不得污染 alpha。"""
    a32 = np.array([[0, 1, 2**32 - 1]], dtype=np.uint32)
    f = to_float(a32)
    assert f.dtype == np.float64
    assert np.array_equal(from_float(f, np.uint32), a32)

    rng = np.random.default_rng(2)
    rgba = rng.random((48, 64, 4)).astype(np.float32)
    alpha = rgba[:, :, 3].copy()
    params = default_params()
    params['gaussian_blur'] = 100
    params['usm'] = 100
    params['clarity'] = 100
    out = process_image(rgba, params)
    assert out.shape == rgba.shape
    assert np.array_equal(out[:, :, 3], alpha), 'alpha 通道不应被模糊类滤镜修改'
    assert out.min() >= 0.0 and out.max() <= 1.0
    print("[OK] uint32 往返与 RGBA 通道保护")


def test_rgba_all_filters_preserve_alpha():
    """process_image 对任意滤镜都必须原样保留 alpha 通道。"""
    rng = np.random.default_rng(4)
    rgba = rng.random((40, 56, 4)).astype(np.float32)
    alpha = rgba[:, :, 3].copy()
    for key, cn, en, mn, mx, dft, _ in FILTER_PARAMS:
        for val in (mn, mx):
            params = default_params()
            params[key] = val
            out = process_image(rgba, params)
            assert out.shape == rgba.shape, f"{key}@{val} shape mismatch"
            assert np.all(np.isfinite(out)), f"{key}@{val} 输出非有限"
            assert np.array_equal(out[:, :, 3], alpha), f"{key}@{val} 修改了 alpha 通道"
    print("[OK] RGBA 全滤镜 alpha 通道保护")


def test_process_image_rejects_integer_input():
    """非浮点输入必须显式拒绝，而不是静默当作 [0,1] 处理。"""
    img = np.zeros((8, 8), dtype=np.uint8)
    rejected = False
    try:
        process_image(img, default_params())
    except ValueError:
        rejected = True
    assert rejected, 'process_image 应拒绝整型输入（请先 to_float）'
    print("[OK] 整型输入拒绝")


def test_miniswhite_display_and_filter_semantics():
    """MINISWHITE（0=白）的预览与滤镜方向必须与视觉一致。"""
    with tempfile.TemporaryDirectory(prefix='tif_miniswhite_') as tmp:
        gray = np.array([[0, 64, 128, 255]], dtype=np.uint8)
        src = os.path.join(tmp, 'mw.tif')
        tifffile.imwrite(src, gray, photometric='miniswhite')
        doc = TifDocument(src)

        # 处理空间按亮度反转：raw 0(白) -> 1.0，raw 255(黑) -> 0.0
        f = doc.get_frame_float(0)
        expected = np.array([1.0, 1.0 - 64 / 255.0, 1.0 - 128 / 255.0, 0.0])
        assert np.allclose(f[0], expected, atol=1e-6), f'处理空间未按亮度反转: {f[0]}'
        disp = doc.get_original_display(0)
        assert int(disp[0, 0]) == 255 and int(disp[0, 3]) == 0, 'MINISWHITE 预览应为正片'

        # 曝光正值使中间调变亮：MINISWHITE 数值应减小
        out_path = os.path.join(tmp, 'mw_out.tif')
        doc.save_processed(out_path, {'exposure': 100})
        re = tifffile.imread(out_path)
        assert re.dtype == np.uint8
        assert int(re[0, 0]) == 0 and int(re[0, 3]) == 255, '纯白/纯黑应保持'
        assert int(re[0, 2]) < int(gray[0, 2]), '曝光正值应使中间调变亮'
        with tifffile.TiffFile(out_path) as tif:
            assert tif.pages[0].photometric.name == 'MINISWHITE', '应保留原 photometric'
    print("[OK] MINISWHITE 显示/滤镜方向")


def test_rgba_and_miniswhite_roundtrip():
    """RGBA 与 MINISWHITE 文件保存后数值保持。"""
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory(prefix='tif_roundtrip_') as tmp:
        rgba = (rng.random((30, 40, 4)) * 255).astype(np.uint8)
        src = os.path.join(tmp, 'rgba.tif')
        tifffile.imwrite(src, rgba)
        doc = TifDocument(src)
        assert doc.channel_count == 4
        out_path = os.path.join(tmp, 'rgba_out.tif')
        doc.save_processed(out_path, {'contrast': 20})
        re = tifffile.imread(out_path)
        assert re.shape == rgba.shape and re.dtype == np.uint8
        assert np.array_equal(re[:, :, 3], rgba[:, :, 3]), 'alpha 应原样保留'
        assert not np.array_equal(re[:, :, :3], rgba[:, :, :3]), 'RGB 应已应用对比度'

        gray = (rng.random((24, 32)) * 255).astype(np.uint8)
        src = os.path.join(tmp, 'miniswhite.tif')
        tifffile.imwrite(src, gray, photometric='miniswhite')
        doc = TifDocument(src)
        assert doc.photometric.name == 'MINISWHITE'
        out_path = os.path.join(tmp, 'miniswhite_out.tif')
        doc.save_processed(out_path, default_params())
        re = tifffile.imread(out_path)
        assert np.array_equal(re, gray) and re.dtype == np.uint8
    print("[OK] RGBA / MINISWHITE 读写往返")


# ----- FEI 电镜等非规范 writer 的兼容性（回归：20260907 真实数据） -----

def _rename_tag(src_path, dst_path, old_code, new_code):
    """把 TIFF IFD 里的标签 old_code 改名为 new_code（字节手术，值不变）。"""
    data = bytearray(open(src_path, 'rb').read())
    byte_order = '>' if data[0:2] == b'MM' else '<'
    ifd_offset = struct.unpack(byte_order + 'I', data[4:8])[0]
    entry_count = struct.unpack(byte_order + 'H', data[ifd_offset:ifd_offset + 2])[0]
    for i in range(entry_count):
        entry = ifd_offset + 2 + 12 * i
        tag_id = struct.unpack(byte_order + 'H', data[entry:entry + 2])[0]
        if tag_id == old_code:
            struct.pack_into(byte_order + 'H', data, entry, new_code)
            open(dst_path, 'wb').write(bytes(data))
            return
    raise AssertionError(f'fixture 构造失败：未找到标签 {old_code}')


class _FakeTag:
    def __init__(self, code, dtype, value, count=1):
        self.code = code
        self.dtype = dtype
        self.value = value
        self.count = count


class _FakePage:
    def __init__(self, tags):
        self.tags = {tag.code: tag for tag in tags}


def test_missing_compression_tag_loads():
    """缺失 Compression 标签的 TIFF（tifffile 返回原始 int）必须按 NONE 处理。"""
    rng = np.random.default_rng(5)
    rgb = (rng.random((24, 32, 3)) * 255).astype(np.uint8)
    with tempfile.TemporaryDirectory(prefix='tif_nocomp_') as tmp:
        src = os.path.join(tmp, 'src.tif')
        tifffile.imwrite(src, rgb)
        stripped = os.path.join(tmp, 'nocomp.tif')
        _rename_tag(src, stripped, 259, 59999)
        with tifffile.TiffFile(stripped) as tif:
            assert isinstance(tif.pages[0].compression, int), 'fixture 未复现原始 int 场景'
        doc = TifDocument(stripped)
        assert doc.compression.name == 'NONE'
        assert doc.channel_count == 3
        out_path = os.path.join(tmp, 'out.tif')
        doc.save_processed(out_path, default_params())
        assert np.array_equal(tifffile.imread(out_path), rgb), '恒等往返必须逐像素一致'
        re_doc = TifDocument(out_path)
        assert re_doc.compression.name == 'NONE'
    print("[OK] 缺失 Compression 标签（原始 int 规范化为 NONE）")


def test_unknown_compression_rejected_with_clear_error():
    """无法映射到枚举的 compression 值必须给出明确报错而非 AttributeError。"""
    rng = np.random.default_rng(7)
    gray = (rng.random((16, 16)) * 255).astype(np.uint8)
    with tempfile.TemporaryDirectory(prefix='tif_badcomp_') as tmp:
        src = os.path.join(tmp, 'src.tif')
        tifffile.imwrite(src, gray)
        # 把 Compression 值就地改成 COMPRESSION 枚举中不存在的 12345
        data = bytearray(open(src, 'rb').read())
        byte_order = '>' if data[0:2] == b'MM' else '<'
        ifd_offset = struct.unpack(byte_order + 'I', data[4:8])[0]
        entry_count = struct.unpack(byte_order + 'H', data[ifd_offset:ifd_offset + 2])[0]
        patched = False
        for i in range(entry_count):
            entry = ifd_offset + 2 + 12 * i
            if struct.unpack(byte_order + 'H', data[entry:entry + 2])[0] == 259:
                struct.pack_into(byte_order + 'H', data, entry + 8, 12345)
                patched = True
                break
        assert patched, 'fixture 构造失败：未找到 Compression 标签'
        broken = os.path.join(tmp, 'broken.tif')
        open(broken, 'wb').write(bytes(data))
        try:
            TifDocument(broken)
        except UnsupportedTiffError as exc:
            assert 'compression' in str(exc).lower(), f'报错应指明 compression：{exc}'
        else:
            raise AssertionError('未知 compression 值应被明确拒绝')
    print("[OK] 未知 compression 值明确拒绝")


def test_unserializable_extra_tags_dropped_not_fatal():
    """ASCII 标签被解析成 dict（FEI_HELIOS）或含非 7-bit 字符时：跳过并成功保存。"""
    rng = np.random.default_rng(6)
    gray = (rng.random((20, 24)) * 255).astype(np.uint8)
    with tempfile.TemporaryDirectory(prefix='tif_feitag_') as tmp:
        src = os.path.join(tmp, 'fei.tif')
        fei_payload = b"{'Beam': {'PixelWidth': 8.941161e-12}}\x00"
        micron_payload = '像素尺寸 1.5 µm'.encode('utf-8') + b'\x00'
        tifffile.imwrite(src, gray, extratags=[
            (34682, 2, len(fei_payload), fei_payload, True),
            (60001, 2, len(micron_payload), micron_payload, True),
        ])
        doc = TifDocument(src)
        copied = {tag[0] for tag in doc.extratags}
        assert 34682 not in copied and 60001 not in copied, '不可序列化标签必须跳过'
        out_path = os.path.join(tmp, 'out.tif')
        doc.save_processed(out_path, default_params())
        assert np.array_equal(tifffile.imread(out_path), gray)
        with tifffile.TiffFile(out_path) as tif:
            out_codes = {tag.code for tag in tif.pages[0].tags.values()}
        assert 34682 not in out_codes and 60001 not in out_codes
        with open(out_path + '.filter.json', encoding='utf-8') as handle:
            manifest = json.load(handle)
        assert 34682 in manifest['tiff']['original_tag_codes'], '原始标签清单应如实记录'
        assert 60001 in manifest['tiff']['original_tag_codes']
        assert 34682 not in manifest['tiff']['copied_extra_tag_codes']
    print("[OK] 不可序列化的 ASCII 标签跳过（FEI_HELIOS / 非 7-bit）")


def test_pointer_extra_tags_skipped():
    """子 IFD 指针标签（EXIF/GPS/Interop）不得复制进输出。"""
    tags = [
        _FakeTag(34665, 4, 16),   # EXIF IFD 指针
        _FakeTag(34853, 4, 16),   # GPS IFD 指针
        _FakeTag(40965, 4, 16),   # Interop IFD 指针
        _FakeTag(700, 2, 'xmp'),  # XMP：<32768 且不在白名单，按既有规则跳过
        _FakeTag(33432, 2, '(c)'),  # Copyright：在白名单，应保留
    ]
    copied = {tag[0] for tag in TifDocument._copyable_extratags(_FakePage(tags))}
    assert copied == {33432}, f'指针标签应被跳过、白名单标签应保留：{copied}'
    print("[OK] 子 IFD 指针标签跳过")


def test_ome_rgb_stack_supported():
    """OME-TIFF 的 RGB 图像（series 轴 CYXS）应可加载、恒等往返与带滤镜保存。"""
    rng = np.random.default_rng(8)
    data = (rng.random((2, 20, 24, 3)) * 255).astype(np.uint8)
    with tempfile.TemporaryDirectory(prefix='tif_ome_') as tmp:
        src = os.path.join(tmp, 'ome_rgb.tif')
        tifffile.imwrite(src, data, ome=True, photometric='rgb')
        with tifffile.TiffFile(src) as tif:
            assert tif.series[0].axes == 'CYXS', f'fixture 应为 CYXS：{tif.series[0].axes}'
        doc = TifDocument(src)
        assert doc.num_frames == 2 and doc.channel_count == 3
        assert doc.get_frame_float(0).shape == (20, 24, 3)
        out_path = os.path.join(tmp, 'out.tif')
        doc.save_processed(out_path, default_params())
        re = tifffile.imread(out_path)
        assert re.shape == data.shape and re.dtype == data.dtype
        assert np.array_equal(re, data), '恒等往返必须逐像素一致'
        out2 = os.path.join(tmp, 'out2.tif')
        doc.save_processed(out2, {'exposure': 30})
        assert not np.array_equal(tifffile.imread(out2), data), '滤镜应已生效'
    print("[OK] OME RGB（CYXS 轴）加载/往返/保存")


def test_channel_stored_as_pages_rejected_early():
    """通道按页存储（CYX 等无 S 轴布局）必须在加载期拒绝并说明原因。"""
    with tempfile.TemporaryDirectory(prefix='tif_cyx_') as tmp:
        src = os.path.join(tmp, 'cyx.tif')
        tifffile.imwrite(src, np.zeros((2, 16, 16), np.uint8),
                         ome=True, metadata={'axes': 'CYX'})
        with tifffile.TiffFile(src) as tif:
            assert tif.series[0].axes == 'CYX'
        try:
            TifDocument(src)
        except UnsupportedTiffError as exc:
            assert '通道按页存储' in str(exc), f'报错应说明原因：{exc}'
        else:
            raise AssertionError('通道分页存储应被拒绝')
    print("[OK] 通道按页存储（CYX）加载期明确拒绝")


def test_manifest_v2_traceability():
    """清单 v2 必须包含源 SHA-256、跳过标签明细、时长和输出大小。"""
    rng = np.random.default_rng(9)
    gray = (rng.random((20, 24)) * 255).astype(np.uint8)
    with tempfile.TemporaryDirectory(prefix='tif_manifest_') as tmp:
        src = os.path.join(tmp, 'src.tif')
        # 34682（FEI 风格 dict 值）不可序列化 → 应显式出现在 skipped_tag_codes
        payload = b"{'Beam': {'PixelWidth': 1e-11}}\x00"
        tifffile.imwrite(src, gray, extratags=[(34682, 2, len(payload), payload, True)])
        doc = TifDocument(src)
        out_path = os.path.join(tmp, 'out.tif')
        doc.save_processed(out_path, {'exposure': 10})
        with open(out_path + '.filter.json', encoding='utf-8') as handle:
            manifest = json.load(handle)
        assert manifest['schema_version'] == 2, '清单应为 schema v2'
        assert manifest['source']['sha256'] == file_sha256(src), '源文件哈希必须可核对'
        assert manifest['duration_seconds'] >= 0.0
        assert manifest['output_size_bytes'] == os.path.getsize(out_path)
        assert 34682 in manifest['tiff']['skipped_tag_codes'], '被跳过标签必须显式列出'
        assert 34682 in manifest['tiff']['original_tag_codes']
        assert 34682 not in manifest['tiff']['copied_extra_tag_codes']
        # skipped = 原始标签 - 已复制标签 - 写出端重建的结构标签
        handled_sample = {254, 256, 257, 258, 259, 262, 273, 277, 278, 279, 296, 338, 339}
        assert not (set(manifest['tiff']['skipped_tag_codes'])
                    & set(manifest['tiff']['copied_extra_tag_codes']))
        assert not (set(manifest['tiff']['skipped_tag_codes']) & handled_sample)
    print("[OK] 清单 v2 溯源字段（sha256/skipped/duration/size）")


def test_peak_bytes_estimation():
    """内存峰值估算与帧尺寸/位深一致，供 GUI 大图提示使用。"""
    doc16 = TifDocument(os.path.join(TEST_DIR, 'rgb16.tif'))
    pixels = 150 * 200 * 3
    assert doc16.peak_bytes() == max(pixels * (2 + 4 * 3 + 2), pixels * 4 * 3 + 4)
    doc8 = TifDocument(os.path.join(TEST_DIR, 'gray8.tif'))
    assert doc8.peak_bytes() > 200 * 300 * 1  # 至少覆盖原始+浮点副本
    print("[OK] 单帧内存峰值估算")


if __name__ == '__main__':
    make_test_images()
    test_dtype_conversion()
    test_each_filter()
    test_pipeline()
    test_tif_io()
    test_save_processed()
    test_uint32_and_extra_channels()
    test_rgba_all_filters_preserve_alpha()
    test_process_image_rejects_integer_input()
    test_miniswhite_display_and_filter_semantics()
    test_rgba_and_miniswhite_roundtrip()
    test_missing_compression_tag_loads()
    test_unknown_compression_rejected_with_clear_error()
    test_unserializable_extra_tags_dropped_not_fatal()
    test_pointer_extra_tags_skipped()
    test_ome_rgb_stack_supported()
    test_channel_stored_as_pages_rejected_early()
    test_manifest_v2_traceability()
    test_peak_bytes_estimation()
    print("\n所有测试通过！")
