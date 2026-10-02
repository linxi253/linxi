"""TEM Suite 主窗口 —— 工具树 + 标签页工作区。

布局
----
    +----------------+--------------------------------------+
    |  工具树         |  Notebook 工作区                      |
    |  按流程分组      |  每个已打开工具占一个标签页             |
    |                |                                      |
    +----------------+--------------------------------------+
    |  状态栏                                                |
    +-------------------------------------------------------+
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tkinter as tk
import traceback
from contextlib import suppress
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox, ttk

from .hostframe import ToolHost
from .loader import LOADER
from .mplbackend import backend_locked, force_headless_backend
from .registry import CATEGORIES, TOOLS, ToolSpec, tools_in_category
from .tkpatch import tk_root_redirected, ttk_theme_frozen

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


logger = logging.getLogger(__name__)

APP_TITLE = "TEM Suite —— 电镜数据分析工具集"


class SuiteApp:
    """集成软件主窗口。"""

    def __init__(self, root: tk.Misc) -> None:
        self.root = root

        # 高 DPI 适配：4K + 200% 缩放下 Tk 工作在真实分辨率，
        # 所有硬编码像素值都必须按此系数换算，否则面板过窄、文字被横向截断。
        self.scale = max(1.0, root.winfo_fpixels("1i") / 96.0)

        self.root.title(APP_TITLE)
        self.root.geometry(f"{self.px(1500)}x{self.px(920)}")
        self.root.minsize(self.px(1100), self.px(700))

        # tool_id -> (ToolHost, 工具实例)
        self._open_tools: dict[str, tuple[ToolHost, object]] = {}
        # Treeview item id -> tool_id
        self._item_to_tool: dict[str, str] = {}
        # 保持子进程引用，避免被回收
        self._subprocesses: list[subprocess.Popen] = []

        self._build_ui()
        self._populate_tree()

    def px(self, logical: int) -> int:
        """把逻辑像素换算为当前 DPI 下的物理像素。"""
        return round(logical * self.scale)

    def _apply_suite_styles(self) -> None:
        """（重新）应用主窗口自有样式。

        工具可能通过切换 ttk 主题清空样式数据库。虽然加载期间已冻结主题切换，
        这里仍在每次加载后重新套用一次作为兜底 —— 样式配置是幂等操作，代价极低。
        """
        style = ttk.Style()
        line_space = tkfont.nametofont("TkDefaultFont").metrics("linespace")
        style.configure("Suite.Treeview", rowheight=line_space + self.px(10))

    # ------------------------------------------------------------------
    # 界面构建
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=6)
        outer.pack(fill=tk.BOTH, expand=True)

        paned = ttk.PanedWindow(outer, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # ---- 左侧：工具树 ----
        left = ttk.Frame(paned, width=self.px(260))
        paned.add(left, weight=0)

        ttk.Label(left, text="分析工具", font=("微软雅黑", 11, "bold")).pack(anchor=tk.W, padx=4, pady=(0, 4))

        tree_wrap = ttk.Frame(left)
        tree_wrap.pack(fill=tk.BOTH, expand=True)

        self._apply_suite_styles()

        self.tree = ttk.Treeview(tree_wrap, show="tree", selectmode="browse", style="Suite.Treeview")
        vsb = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.bind("<Double-1>", self._on_tree_activate)
        self.tree.bind("<Return>", self._on_tree_activate)

        # 工具说明面板
        desc_frame = ttk.LabelFrame(left, text="说明", padding=6)
        desc_frame.pack(fill=tk.X, pady=(6, 0))
        self.desc_var = tk.StringVar(value="双击工具名称即可打开。")
        ttk.Label(
            desc_frame,
            textvariable=self.desc_var,
            wraplength=self.px(230),
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

        self.open_btn = ttk.Button(left, text="打开选中工具", command=self._open_selected)
        self.open_btn.pack(fill=tk.X, pady=(6, 0))

        # ---- 右侧：工作区 ----
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.notebook.bind("<Button-2>", self._on_middle_click)  # 中键关闭标签页

        self._welcome = self._build_welcome(self.notebook)
        self.notebook.add(self._welcome, text="  起始页  ")

        # ---- 底部：状态栏 ----
        status = ttk.Frame(outer)
        status.pack(fill=tk.X, pady=(4, 0))
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(status, textvariable=self.status_var, anchor=tk.W).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(status, text="关闭当前标签页", command=self._close_current_tab).pack(side=tk.RIGHT)

    def _build_welcome(self, master: tk.Misc) -> ttk.Frame:
        frame = ttk.Frame(master, padding=24)
        ttk.Label(frame, text=APP_TITLE, font=("微软雅黑", 18, "bold")).pack(anchor=tk.W)
        ttk.Label(
            frame,
            text="从左侧工具树中双击任一工具即可在标签页中打开。",
            font=("微软雅黑", 10),
        ).pack(anchor=tk.W, pady=(8, 16))

        info = ttk.LabelFrame(frame, text="推荐处理流程", padding=12)
        info.pack(anchor=tk.W, fill=tk.X)
        steps = (
            "1. 数据提取 —— 从视频提取 TIFF 图像堆栈",
            "2. 图像处理 —— 漂移矫正、滤波降噪、衬度优化",
            "3. EELS谱学 —— 边缘价态、复散射校正、参考谱拟合",
            "4. 应变分析 —— GPA / PPA / 原子识别 / 原子标注",
            "5. 4D-STEM  —— DPC、应变映射、取向映射",
            "6. 定量统计 —— 面积、衬度、晶体占比、演化趋势",
            "7. 模拟仿真 —— HRTEM 多层法模拟与衍射分析",
        )
        for text in steps:
            ttk.Label(info, text=text, font=("微软雅黑", 10)).pack(anchor=tk.W, pady=2)

        avail = sum(1 for spec in TOOLS if spec.available)
        ttk.Label(
            frame,
            text=f"已注册工具 {len(TOOLS)} 个，其中 {avail} 个就绪。",
            foreground="gray",
        ).pack(anchor=tk.W, pady=(16, 0))
        return frame

    # ------------------------------------------------------------------
    def _populate_tree(self) -> None:
        for category, label in CATEGORIES:
            specs = tools_in_category(category)
            if not specs:
                continue
            parent = self.tree.insert("", tk.END, text=label, open=True)
            for spec in specs:
                suffix = "" if spec.available else "  (缺失)"
                item = self.tree.insert(parent, tk.END, text=f"{spec.name}{suffix}")
                self._item_to_tool[item] = spec.tool_id

    # ------------------------------------------------------------------
    # 事件处理
    # ------------------------------------------------------------------
    def _selected_spec(self) -> ToolSpec | None:
        sel = self.tree.selection()
        if not sel:
            return None
        tool_id = self._item_to_tool.get(sel[0])
        if tool_id is None:
            return None
        from .registry import get_tool

        return get_tool(tool_id)

    def _on_tree_select(self, _event: object = None) -> None:
        spec = self._selected_spec()
        if spec is None:
            self.desc_var.set("双击工具名称即可打开。")
            return
        text = spec.description
        if not spec.available:
            text += f"\n\n[项目目录缺失]\n{spec.project_dir}"
        elif spec.run_mode == "subprocess":
            text += "\n\n[以独立窗口运行]"
        self.desc_var.set(text)

    def _on_tree_activate(self, _event: object = None) -> None:
        self._open_selected()

    def _open_selected(self) -> None:
        spec = self._selected_spec()
        if spec is not None:
            self.open_tool(spec)

    def _on_middle_click(self, event: tk.Event) -> None:
        try:
            index = self.notebook.index(f"@{event.x},{event.y}")
        except tk.TclError:
            return
        self._close_tab_at(index)

    # ------------------------------------------------------------------
    # 打开工具
    # ------------------------------------------------------------------
    def open_tool(self, spec: ToolSpec) -> None:
        if not spec.available:
            messagebox.showerror(
                "工具不可用",
                f"未找到 {spec.name} 的项目目录：\n{spec.project_dir}",
                parent=self.root,
            )
            return

        if spec.run_mode == "subprocess":
            self._launch_subprocess(spec)
            return

        # 已打开则直接切换
        existing = self._open_tools.get(spec.tool_id)
        if existing is not None:
            self.notebook.select(existing[0])
            self.status_var.set(f"{spec.name} 已在标签页中打开")
            return

        self.status_var.set(f"正在加载 {spec.name} ...")
        self.root.update_idletasks()

        host = ToolHost(
            self.notebook,
            on_title_change=lambda text, s=spec: self._on_tool_title(s, text),
            tool_id=spec.tool_id,
        )
        self.notebook.add(host, text=f" {spec.tab_label} ")
        self.notebook.select(host)

        try:
            instance = self._instantiate(spec, host)
        except BaseException as exc:  # noqa: BLE001 - 任何失败都要显示给用户而非崩溃
            logger.exception("加载工具 %s 失败", spec.tool_id)
            self._show_load_error(host, spec, exc)
            self.status_var.set(f"{spec.name} 加载失败")
            return

        self._open_tools[spec.tool_id] = (host, instance)
        # 兜底：若工具仍以某种方式改动了样式数据库，此处恢复主窗口样式
        self._apply_suite_styles()
        self.status_var.set(f"{spec.name} 已就绪")

    def _instantiate(self, spec: ToolSpec, host: ToolHost) -> object:
        """加载工具模块并实例化其 GUI 类。

        全程冻结 ttk 主题：部分工具会在 ``__init__`` 里调用
        ``theme_use()`` 切换全局主题，导致主窗口样式被清空。

        实例化同样纳入 matplotlib 后端锁定：个别工具（如 HRTEM 模拟）不在模块
        顶层、而是在 ``__init__`` 内执行 ``matplotlib.use('TkAgg')``，若不屏蔽，
        其 ``plt.subplots()`` 会创建第二个 Tcl 解释器并使进程崩溃。
        """
        with ttk_theme_frozen(), backend_locked():
            if spec.load_mode == "file":
                module = LOADER.load_file(
                    spec.tool_id,
                    spec.project_dir,
                    spec.entry,
                    extra_paths=spec.extra_paths,
                )
            else:
                module = LOADER.load_module(
                    spec.tool_id,
                    spec.project_dir,
                    spec.entry,
                    extra_paths=spec.extra_paths,
                    preload=spec.preload,
                )

            try:
                factory = getattr(module, spec.factory)
            except AttributeError:
                raise AttributeError(f"模块 {spec.entry} 中不存在 {spec.factory}，请检查注册表配置") from None

            if spec.root_mode == "patch":
                # 工具内部自建 tk.Tk()，拦截构造让它拿到 host
                with tk_root_redirected(host):
                    return factory()
            return factory(host)

    def _show_load_error(self, host: ToolHost, spec: ToolSpec, exc: BaseException) -> None:
        """在标签页内展示错误详情，便于排查。"""
        for child in host.winfo_children():
            child.destroy()

        wrap = ttk.Frame(host, padding=16)
        wrap.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            wrap, text=f"{spec.name} 加载失败", font=("微软雅黑", 13, "bold"), foreground="#c0392b"
        ).pack(anchor=tk.W)
        ttk.Label(wrap, text=f"{type(exc).__name__}: {exc}", wraplength=900, justify=tk.LEFT).pack(
            anchor=tk.W, pady=(6, 10)
        )
        ttk.Label(wrap, text=f"项目目录：{spec.project_dir}", foreground="gray").pack(anchor=tk.W)

        box = ttk.Frame(wrap)
        box.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        text = tk.Text(box, wrap=tk.NONE, height=20, font=("Consolas", 9))
        ysb = ttk.Scrollbar(box, orient=tk.VERTICAL, command=text.yview)
        xsb = ttk.Scrollbar(box, orient=tk.HORIZONTAL, command=text.xview)
        text.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)
        xsb.pack(side=tk.BOTTOM, fill=tk.X)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text.insert("1.0", "".join(traceback.format_exception(exc)))
        text.configure(state=tk.DISABLED)

    def _on_tool_title(self, spec: ToolSpec, text: str) -> None:
        """工具调用 root.title() 时，把完整标题显示到状态栏。"""
        if text:
            self.status_var.set(f"{spec.name} · {text}")

    # ------------------------------------------------------------------
    def _launch_subprocess(self, spec: ToolSpec) -> None:
        env = os.environ.copy()
        path_prepend = os.pathsep.join(str(p) for p in spec.extra_paths)
        if path_prepend:
            # 子进程工具可能采用 src 布局，把其源码目录交给 Python 的导入系统
            env["PYTHONPATH"] = (
                f"{path_prepend}{os.pathsep}{env['PYTHONPATH']}"
                if env.get("PYTHONPATH")
                else path_prepend
            )

        # 优先使用项目自带的解释器：各子进程工具的依赖锁定在它们自己的 venv
        # 里（如原子标注工具锁定 numpy 1.26.4，套件锁为 numpy 2.2.6），复用
        # 套件解释器会打破其版本契约。缺失时才回退到套件解释器并告警。
        python_exe = spec.resolve_python_exe()
        if python_exe is not None:
            suite_runtime = False
        else:
            python_exe = Path(sys.executable)
            suite_runtime = True
            logger.warning(
                "%s 未找到项目自带解释器（%s），回退到套件解释器/运行时 %s —— "
                "子进程依赖版本可能与该工具自身锁定的版本不一致",
                spec.name,
                spec.project_dir / ".venv",
                python_exe,
            )

        if spec.subprocess_module:
            # 包内使用相对导入的模块无法按脚本路径执行，以 python -m 方式启动
            if suite_runtime and getattr(sys, "frozen", False):
                cmd = [sys.executable, "--run-module", spec.subprocess_module]
            else:
                cmd = [str(python_exe), "-m", spec.subprocess_module, *spec.subprocess_args]
        else:
            script = spec.project_dir / spec.entry
            if not script.is_file():
                messagebox.showerror("无法启动", f"未找到脚本：\n{script}", parent=self.root)
                return

            # 回退到套件本体时，打包后 sys.executable 是 TEM Suite.exe 本身
            # 而非 python，因此以 --run-script 参数让 exe 的另一个实例代为
            # 执行目标脚本；项目自带解释器则可直接运行脚本。
            if suite_runtime and getattr(sys, "frozen", False):
                cmd = [sys.executable, "--run-script", str(script)]
            else:
                cmd = [str(python_exe), str(script), *spec.subprocess_args]

        try:
            proc = subprocess.Popen(  # noqa: S603 - 路径来自内部注册表
                cmd, cwd=str(spec.project_dir), env=env
            )
        except OSError as exc:
            messagebox.showerror("启动失败", str(exc), parent=self.root)
            return
        self._subprocesses.append(proc)
        self.status_var.set(f"{spec.name} 已在独立窗口中启动")

    # ------------------------------------------------------------------
    # 关闭标签页
    # ------------------------------------------------------------------
    def _close_current_tab(self) -> None:
        try:
            index = self.notebook.index(self.notebook.select())
        except tk.TclError:
            return
        self._close_tab_at(index)

    def _close_tab_at(self, index: int) -> None:
        try:
            widget_name = self.notebook.tabs()[index]
        except IndexError:
            return
        widget = self.root.nametowidget(widget_name)
        if widget is self._welcome:
            return  # 起始页不可关闭

        tool_id = next((tid for tid, (host, _) in self._open_tools.items() if host is widget), None)
        if tool_id is not None:
            host, _ = self._open_tools.pop(tool_id)
            host.run_close_callbacks()

        self.notebook.forget(index)
        if isinstance(widget, tk.Widget):
            widget.destroy()
        self.status_var.set("标签页已关闭")

    # ------------------------------------------------------------------
    def on_app_close(self) -> None:
        """主窗口关闭前，让各工具执行自身清理。"""
        for host, _ in self._open_tools.values():
            host.run_close_callbacks()
        self.root.destroy()


def build_root() -> tk.Misc:
    """创建根窗口。优先使用 ttkbootstrap 主题，缺失时退回原生 ttk。"""
    try:
        import ttkbootstrap as ttkb
    except ImportError:
        root = tk.Tk()
    else:
        root = ttkb.Window(themename="cosmo")

    with suppress(tk.TclError):
        root.option_add("*Font", "微软雅黑 10")
    return root


def _run_external_script(script_path: str) -> int:
    """脚本执行模式：``TEM Suite.exe --run-script <path>``。

    供打包后的子进程工具使用 —— exe 内含完整的 Python 运行时与依赖，
    以另一实例代为执行外部脚本，行为等同于 ``python <path>``。
    不应用任何整合层 patch，忠实还原脚本独立运行的环境。
    """
    import runpy

    script = Path(script_path).resolve()
    if not script.is_file():
        from tkinter import messagebox

        messagebox.showerror("脚本不存在", str(script))
        return 2

    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script)]
    runpy.run_path(str(script), run_name="__main__")
    return 0


def _run_external_module(module_name: str) -> int:
    """模块执行模式：``TEM Suite.exe --run-module <pkg.module>``。

    供以 ``python -m`` 方式启动的子进程工具使用 —— 包内使用相对导入的模块
    无法按脚本路径执行，必须在包上下文中运行。工具若采用 src 布局，
    依据注册表把其源码目录加入 ``sys.path`` 后再执行。
    """
    import runpy

    spec = next((s for s in TOOLS if s.subprocess_module == module_name), None)
    if spec is not None:
        for path in spec.extra_paths:
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

    sys.argv = [module_name]
    runpy.run_module(module_name, run_name="__main__", alter_sys=True)
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    argv = sys.argv[1:]
    if len(argv) >= 2 and argv[0] == "--run-script":
        return _run_external_script(argv[1])
    if len(argv) >= 2 and argv[0] == "--run-module":
        return _run_external_module(argv[1])

    # 必须在加载任何工具之前固定 matplotlib 后端，
    # 否则工具的 TkAgg + plt.subplots() 会创建第二个 Tcl 解释器并使进程崩溃。
    force_headless_backend()

    # 高 DPI 支持（Windows）
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception as exc:  # noqa: BLE001 - 合理吞除：DPI 感知设置失败
            # 仅退化为标准 DPI 显示（旧系统/已设置过都会失败），不应阻断启动；
            # 类型随 Windows 版本而异无法收窄，留 debug 痕迹防真问题被掩盖。
            logger.debug("SetProcessDpiAwareness 失败（忽略，按标准 DPI 运行）: %s", exc)

    if "--self-test" in sys.argv[1:]:
        from .selftest import run_self_test

        return run_self_test()

    root = build_root()
    app = SuiteApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_app_close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
