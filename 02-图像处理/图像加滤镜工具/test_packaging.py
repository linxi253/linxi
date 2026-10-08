# -*- coding: utf-8 -*-
"""打包 hook 测试：验证 imagecodecs 编解码器收集逻辑的可用性与失败显式化。

hook 文件名含连字符无法常规导入，这里通过 importlib 按路径加载。
"""

import importlib.util
import os
import sys
import types

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


HOOK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hook-imagecodecs.py')

results = []


def load_hook():
    spec = importlib.util.spec_from_file_location('hook_imagecodecs', HOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hiddenimports_cover_lossless_codecs():
    hook = load_hook()
    assert hook.hiddenimports, 'hiddenimports 不应为空'
    joined = ' '.join(hook.hiddenimports)
    registry = hook._wanted_extension_names()
    # LZW / PackBits / DICOMRLE 等编解码实现位于 _imcd 扩展；deflate/zlib 在 _deflate/_zlib
    assert '_imcd' in registry, f'LZW/PackBits 所在的 _imcd 扩展应被匹配：{registry}'
    assert '_deflate' in registry, f'deflate 扩展应被匹配：{registry}'
    assert 'imagecodecs._imcd' in hook.hiddenimports, '_imcd 扩展必须进入 hiddenimports'
    assert joined, 'hiddenimports 非空'
    results.append(('hiddenimports 覆盖无损编解码（lzw/packbits）', True, ''))


def test_missing_registry_raises_instead_of_silent():
    """注册表缺失时必须抛 RuntimeError，而不是静默产出缺解码器的 exe。"""
    import builtins
    hook = load_hook()
    real_import = builtins.__import__

    def poisoned_import(name, *args, **kwargs):
        if name.startswith('imagecodecs'):
            raise ImportError(f'simulated offline import failure: {name}')
        return real_import(name, *args, **kwargs)

    builtins.__import__ = poisoned_import
    try:
        hook._wanted_extension_names()
    except RuntimeError:
        results.append(('imagecodecs 导入失败 → RuntimeError', True, ''))
    except Exception as exc:  # noqa: BLE001
        results.append(('imagecodecs 导入失败 → RuntimeError', False, f'错误类型不对: {exc!r}'))
    else:
        results.append(('imagecodecs 导入失败 → RuntimeError', False, '未抛出异常'))
    finally:
        builtins.__import__ = real_import

    # 注册表结构变化（_MODULES 被移除）也必须显式报错。
    # 注意 `import a.b as c` 优先从父包属性解析子模块，需同时替换两处。
    import imagecodecs as _top_package
    fake = types.ModuleType('imagecodecs.imagecodecs')
    saved_module = sys.modules.get('imagecodecs.imagecodecs')
    saved_attr = getattr(_top_package, 'imagecodecs', None)
    sys.modules['imagecodecs.imagecodecs'] = fake
    _top_package.imagecodecs = fake
    try:
        hook._wanted_extension_names()
    except RuntimeError:
        results.append(('_MODULES 缺失 → RuntimeError', True, ''))
    except Exception as exc:  # noqa: BLE001
        results.append(('_MODULES 缺失 → RuntimeError', False, f'错误类型不对: {exc!r}'))
    else:
        results.append(('_MODULES 缺失 → RuntimeError', False, '未抛出异常'))
    finally:
        if saved_module is None:
            sys.modules.pop('imagecodecs.imagecodecs', None)
        else:
            sys.modules['imagecodecs.imagecodecs'] = saved_module
        if saved_attr is None:
            del _top_package.imagecodecs
        else:
            _top_package.imagecodecs = saved_attr


if __name__ == '__main__':
    test_hiddenimports_cover_lossless_codecs()
    test_missing_registry_raises_instead_of_silent()
    print('\n===== 打包 hook 测试结果 =====')
    all_ok = True
    for desc, ok, err in results:
        print(f'[{"通过" if ok else "失败"}] {desc}' + (f' -> {err}' if err else ''))
        all_ok &= ok
    print('全部通过！' if all_ok else '存在失败项！')
    sys.exit(0 if all_ok else 1)
