"""冒烟测试 —— 逐个加载全部内嵌工具，验证 ToolHost 代理机制是否有效。

在隐藏的根窗口中真实创建每个工具的 GUI 部件树，只要能构造成功即视为通过。
不显示窗口、不进入事件循环，因此可在命令行批量执行。

用法::

    python tests/smoke_test.py            # 测试全部
    python tests/smoke_test.py drift_correct hrtem_filter   # 只测指定工具
"""

from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import tkinter as tk  # noqa: E402
from tkinter import ttk  # noqa: E402

from temsuite.hostframe import ToolHost  # noqa: E402
from temsuite.loader import LOADER  # noqa: E402
from temsuite.mplbackend import backend_locked, force_headless_backend  # noqa: E402
from temsuite.registry import TOOLS, ToolSpec  # noqa: E402
from temsuite.tkpatch import tk_root_redirected, ttk_theme_frozen  # noqa: E402

# 用于检测工具是否破坏了主窗口样式数据库的哨兵样式
SENTINEL_STYLE = "Suite.Treeview"
SENTINEL_ROWHEIGHT = 44


def instantiate(spec: ToolSpec, host: ToolHost) -> object:
    # 与 app.SuiteApp._instantiate 保持一致：实例化期间同样锁定 matplotlib 后端，
    # 防止在 __init__ 内切换 TkAgg 的工具（如 HRTEM 模拟）创建第二 Tcl 解释器。
    with ttk_theme_frozen(), backend_locked():
        if spec.load_mode == "file":
            module = LOADER.load_file(
                spec.tool_id, spec.project_dir, spec.entry, extra_paths=spec.extra_paths
            )
        else:
            module = LOADER.load_module(
                spec.tool_id,
                spec.project_dir,
                spec.entry,
                extra_paths=spec.extra_paths,
                preload=spec.preload,
            )
        factory = getattr(module, spec.factory)
        if spec.root_mode == "patch":
            with tk_root_redirected(host):
                return factory()
        return factory(host)


def main(argv: list[str]) -> int:
    wanted = set(argv)
    targets = [
        s for s in TOOLS if s.run_mode == "embed" and s.available and (not wanted or s.tool_id in wanted)
    ]
    if not targets:
        print("没有匹配的工具")
        return 2

    force_headless_backend()

    try:
        import ttkbootstrap as ttkb

        root = ttkb.Window(themename="cosmo")
    except ImportError:
        root = tk.Tk()
    root.withdraw()

    style = ttk.Style()
    initial_theme = style.theme_use()
    style.configure(SENTINEL_STYLE, rowheight=SENTINEL_ROWHEIGHT)

    notebook = ttk.Notebook(root)
    notebook.pack()

    passed: list[str] = []
    failed: list[tuple[str, BaseException]] = []
    style_broken: list[str] = []
    hosts: list[ToolHost] = []

    for spec in targets:
        print(f"--- {spec.tool_id}  ({spec.name})")
        host = ToolHost(notebook, tool_id=spec.tool_id)
        notebook.add(host, text=spec.tab_label)
        hosts.append(host)
        try:
            instantiate(spec, host)
            root.update_idletasks()
        except BaseException as exc:  # noqa: BLE001 - 需要收集所有失败
            failed.append((spec.tool_id, exc))
            print(f"    FAIL  {type(exc).__name__}: {exc}")
            continue

        n_children = len(host.winfo_children())
        passed.append(spec.tool_id)

        # 检查工具是否破坏了主窗口样式与主题
        rowheight = str(style.lookup(SENTINEL_STYLE, "rowheight"))
        theme = style.theme_use()
        if rowheight != str(SENTINEL_ROWHEIGHT) or theme != initial_theme:
            style_broken.append(spec.tool_id)
            print(
                f"    PASS  已创建 {n_children} 个子部件  [样式受损! rowheight={rowheight!r} theme={theme}]"
            )
        else:
            print(f"    PASS  已创建 {n_children} 个子部件  [样式完好]")

    print()
    print("=" * 68)
    print(f"通过 {len(passed)} / 失败 {len(failed)}  （共 {len(targets)}）")
    print(f"样式数据库受损的工具: {len(style_broken)}" + (f" -> {style_broken}" if style_broken else ""))
    print(f"最终 ttk 主题: {style.theme_use()}  （初始 {initial_theme}）")
    if failed:
        print()
        for tool_id, exc in failed:
            print("-" * 68)
            print(f"[{tool_id}]")
            print("".join(traceback.format_exception(exc))[-1800:])

    # 走正式清理流程：让各工具取消自己的 after 轮询，
    # 否则销毁根窗口后残留回调会触发 Tcl 的 bgerror。
    for host in hosts:
        host.run_close_callbacks()
    root.update()
    root.destroy()
    return 1 if (failed or style_broken) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
