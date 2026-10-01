"""内置自检 —— 在打包后的 exe 内验证全部工具能否正常加载。

打包环境与开发环境有本质差异：工具源码位于**外部磁盘目录**，
而其依赖的第三方库来自 exe 内部的 ``_MEIPASS``。这条混合导入路径
必须实际跑一遍才能确认可用。

由于 exe 以 windowed 模式构建（无控制台），自检结果写入文本报告，
并用对话框显示摘要。

用法::

    TEM Suite.exe --self-test
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path

from .hostframe import ToolHost
from .loader import LOADER
from .mplbackend import backend_locked, force_headless_backend
from .registry import TOOLS, WORKSPACE_ROOT, ToolSpec
from .tkpatch import tk_root_redirected, ttk_theme_frozen

SENTINEL_STYLE = "Suite.Treeview"
SENTINEL_ROWHEIGHT = 44


def _report_path() -> Path:
    """报告输出位置：打包后放在 exe 同目录，开发时放在包的上层目录。"""
    base = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parents[1]
    return base / "selftest_report.txt"


def _instantiate(spec: ToolSpec, host: ToolHost) -> object:
    # 与 app.SuiteApp._instantiate 保持一致：实例化期间锁定 matplotlib 后端
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


def run_self_test() -> int:
    """逐个加载所有内嵌工具，把结果写入报告并弹窗显示摘要。"""
    import tkinter as tk
    from tkinter import messagebox, ttk

    force_headless_backend()

    lines: list[str] = [
        "TEM Suite 自检报告",
        f"时间: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"运行模式: {'打包 exe' if getattr(sys, 'frozen', False) else '源码'}",
        f"可执行文件: {sys.executable}",
        f"工作区: {WORKSPACE_ROOT}",
        "=" * 70,
        "",
    ]

    try:
        import ttkbootstrap as ttkb

        root = ttkb.Window(themename="cosmo")
    except ImportError:
        root = tk.Tk()
        lines.append("[提示] ttkbootstrap 不可用，已退回原生 ttk\n")
    root.withdraw()

    style = ttk.Style()
    initial_theme = style.theme_use()
    style.configure(SENTINEL_STYLE, rowheight=SENTINEL_ROWHEIGHT)

    notebook = ttk.Notebook(root)
    notebook.pack()

    targets = [s for s in TOOLS if s.run_mode == "embed"]
    passed: list[str] = []
    failed: list[str] = []
    style_broken: list[str] = []
    hosts: list[ToolHost] = []

    for spec in targets:
        if not spec.available:
            failed.append(spec.tool_id)
            lines.append(f"[缺失] {spec.name}")
            lines.append(f"       项目目录不存在: {spec.project_dir}\n")
            continue

        host = ToolHost(notebook, tool_id=spec.tool_id)
        notebook.add(host, text=spec.tab_label)
        hosts.append(host)
        try:
            _instantiate(spec, host)
            root.update_idletasks()
        except BaseException as exc:  # noqa: BLE001 - 自检需收集全部失败
            failed.append(spec.tool_id)
            lines.append(f"[失败] {spec.name}  ({spec.tool_id})")
            lines.append(f"       {type(exc).__name__}: {exc}")
            lines.append(traceback.format_exc())
            lines.append("")
            continue

        passed.append(spec.tool_id)
        rowheight = str(style.lookup(SENTINEL_STYLE, "rowheight"))
        theme = style.theme_use()
        note = ""
        if rowheight != str(SENTINEL_ROWHEIGHT) or theme != initial_theme:
            style_broken.append(spec.tool_id)
            note = f"  [样式受损 rowheight={rowheight!r} theme={theme}]"
        lines.append(f"[通过] {spec.name}  —— 已创建 {len(host.winfo_children())} 个子部件{note}")

    summary = (
        f"通过 {len(passed)} / 失败 {len(failed)}  （共 {len(targets)}）\n"
        f"样式受损工具数: {len(style_broken)}\n"
        f"ttk 主题: {style.theme_use()}（初始 {initial_theme}）"
    )
    lines += ["", "=" * 70, summary]

    for host in hosts:
        host.run_close_callbacks()
    root.update()

    report = _report_path()
    try:
        report.write_text("\n".join(lines), encoding="utf-8")
        saved = f"\n\n报告已保存至:\n{report}"
    except OSError as exc:
        saved = f"\n\n报告写入失败: {exc}"

    root.deiconify()
    root.withdraw()
    if failed or style_broken:
        messagebox.showerror("自检发现问题", summary + saved)
    else:
        messagebox.showinfo("自检通过", summary + saved)
    root.destroy()
    return 1 if (failed or style_broken) else 0
