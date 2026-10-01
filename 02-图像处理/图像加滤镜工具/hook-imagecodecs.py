# -*- coding: utf-8 -*-
"""PyInstaller hook：按需收集 imagecodecs 的 TIFF 编解码扩展模块。

imagecodecs 通过 ``__getattr__`` 惰性加载各编解码扩展（如 ``_imcd``、
``_tiff``），静态依赖分析看不到它们。这里从已安装版本的注册表读取
模块映射，只收集本工具无损 TIFF 读写需要的扩展，避免打包全部编解码器。
"""

_WANTED_CODEC_PREFIXES = (
    'lzw', 'packbits', 'deflate', 'gzip', 'zlib', 'lzma', 'zstd',
    'png', 'lerc', 'tiff', 'dicomrle', 'delta', 'byteshuffle',
    'bitshuffle', 'xor', 'floatpred', 'packints', 'bitorder',
)


def _wanted_extension_names():
    try:
        import imagecodecs.imagecodecs as _registry
    except Exception as exc:
        raise RuntimeError(
            '无法导入 imagecodecs.imagecodecs 注册表，PyInstaller 无法确定要收集的'
            f'TIFF 编解码扩展。请检查 imagecodecs 是否安装完好：{exc!r}'
        ) from exc
    # _MODULES 是 imagecodecs 的私有注册表；一旦未来版本移除或改名，必须在
    # 构建期报错，而不是静默产出缺少 LZW/PackBits 解码器的 exe。
    modules = getattr(_registry, '_MODULES', None)
    if not isinstance(modules, dict) or not modules:
        raise RuntimeError(
            'imagecodecs.imagecodecs._MODULES 注册表不可用（可能已被上游移除）。'
            'hook-imagecodecs.py 依赖该私有 API，请对照已安装的 imagecodecs '
            '版本更新本 hook。'
        )
    wanted = set()
    for module_name, attributes in modules.items():
        for attr in attributes:
            lowered = str(attr).lower()
            if any(lowered.startswith(prefix) for prefix in _WANTED_CODEC_PREFIXES):
                wanted.add(module_name)
                break
    if not wanted:
        raise RuntimeError(
            'imagecodecs 注册表中未匹配到任何所需编解码前缀'
            f'{_WANTED_CODEC_PREFIXES}；请更新 hook-imagecodecs.py 的前缀列表。'
        )
    return wanted


# 扩展模块（.pyd）只能通过 hiddenimports 让 PyInstaller 作为二进制收集；
# 直接列出的模块名会随版本变化，因此从已安装注册表动态解析。
hiddenimports = [
    f'imagecodecs.{name}' for name in sorted(_wanted_extension_names())
]

if __name__ == '__main__':  # python hook-imagecodecs.py 可手动自检
    print('\n'.join(hiddenimports))
