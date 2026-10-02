# -*- coding: utf-8 -*-
"""Strain++ GPA 图形界面启动入口。

源码用户可直接 ``python run.py``；安装用户建议使用
``strainpp-gpa-gui`` console script。两者最终都启动同一个 GUI。
"""

from __future__ import annotations

import os
import sys

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


def _check_dependencies() -> None:
    """启动前只检查纯 TIFF/GUI 所必需的依赖。

    ncempy 是 DM3/DM4 的可选增强解析器，在用户实际打开 DM 文件时才懒检查；
    未安装 ncempy 时内置解析器仍会尝试读取 DM 文件。
    """
    missing = []
    for package_name, import_name in [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("tifffile", "tifffile"),
        ("ttkbootstrap", "ttkbootstrap"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)

    if not missing:
        return

    print("=" * 56)
    print("缺少以下 Python 依赖：")
    for package_name in missing:
        print(f"  - {package_name}")
    print("\n请在项目目录运行：python -m pip install -r requirements.txt")
    print("=" * 56)
    if getattr(sys, "stdin", None) and sys.stdin.isatty():
        input("按回车键退出……")
    raise SystemExit(1)


def _run_packaged_smoke_test() -> None:
    """供发布验收使用：验证打包后的科学计算与 DM 读取模块可导入。"""
    import numpy as np
    import tifffile  # noqa: F401

    try:
        from ncempy.io import dm  # noqa: F401
    except ImportError:
        # DM 冒烟测试需要 ncempy；缺失时给出明确失败原因而不是模糊的
        # import 错误。发布构建应安装 requirements.txt 后重新运行。
        raise SystemExit(
            "SMOKE TEST SKIPPED/FAILED: ncempy is required for the packaged "
            "DM smoke test.  Install it with: python -m pip install ncempy"
        )

    from strainpp_gpa.gpa import GPA

    yy, xx = np.indices((32, 32), dtype=np.float64)
    image = np.cos(2.0 * np.pi * 4.0 * xx / 32.0)
    image += np.cos(2.0 * np.pi * 5.0 * yy / 32.0)

    analyzer = GPA()
    analyzer.load_image(image)
    analyzer.set_g1(4.0, 0.0, sigma=1.0)
    analyzer.set_g2(0.0, 5.0, sigma=1.0)
    result = analyzer.compute(include_displacement=False)
    if not np.isfinite(result.eps_xx).any() or not np.isfinite(result.eps_yy).any():
        raise RuntimeError("Packaged GPA smoke test produced no finite strain values.")


def main() -> None:
    _check_dependencies()
    if os.environ.get("STRAINPP_SMOKE_TEST") == "1":
        _run_packaged_smoke_test()
        return

    from strain_gui import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()
