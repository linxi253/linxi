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
import tempfile
import time
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

# 程序级关闭时等待各工具完成后台清理的上限（秒）。超时只终止子进程并销毁
# 主窗口，不用 os._exit 硬杀 —— 那会截断仍在写出的文件。
_CLOSE_WAIT_TIMEOUT_S = 30.0


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
        # 子进程输出重定向到的临时文件（不能用 PIPE，见 _launch_subprocess）
        self._subprocess_logs: dict[subprocess.Popen, object] = {}
        # 程序级关闭：首次进入时冻结的等待集合与起始时刻（见 on_app_close）
        self._closing_hosts: list[ToolHost] | None = None
        self._close_wait_started: float = 0.0
        # 唯一轮询 after id：保证任一时刻至多一条关闭轮询链
        self._close_after_id: str | None = None
        # 关闭处理器抛异常的工具（记录并在状态栏说明，不因此强退）
        self._close_handler_errors: list[str] = []

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
        # 显式绑定到本窗口的解释器：不依赖 tkinter 的隐式默认 root。
        # 测试中 disposable_root 会清空 tk._default_root，此时隐式构造会另起一个
        # Tcl 解释器（可能直接 TclError），而且样式也不会落到本窗口上。
        style = ttk.Style(master=self.root)
        line_space = tkfont.nametofont("TkDefaultFont", root=self.root).metrics(
            "linespace")
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
        self.desc_var = tk.StringVar(master=self.root, value="双击工具名称即可打开。")
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
        # 每工具菜单随激活页恢复；后台加载不夺前台菜单（回归 2026-10-03 R3）。
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self._welcome = self._build_welcome(self.notebook)
        self.notebook.add(self._welcome, text="  起始页  ")

        # ---- 底部：状态栏 ----
        status = ttk.Frame(outer)
        status.pack(fill=tk.X, pady=(4, 0))
        self.status_var = tk.StringVar(master=self.root, value="就绪")
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
    # 菜单栏：随激活标签页恢复
    # ------------------------------------------------------------------
    def _on_tab_changed(self, _event: object = None) -> None:
        """切换标签页时恢复该页自己的菜单栏。

        工具通过 ``root.config(menu=...)`` 声明菜单；菜单是**顶层窗口级**状态，
        不随 Notebook 页切换。若不在这里恢复，切页后菜单栏会停留在上一个工具
        （回归 2026-10-03 R3）。起始页/无菜单工具/已关闭页都必须清空菜单，
        且不能留下已销毁的 Tcl 菜单。
        """
        self._apply_active_menu()

    def _apply_active_menu(self) -> None:
        try:
            current = self.root.nametowidget(self.notebook.select())
        except (tk.TclError, KeyError):
            current = None
        menu: tk.Menu | None = None
        if isinstance(current, ToolHost) and current.winfo_exists():
            menu = current.menu
            # 菜单对象本身可能已被工具销毁，挂上去会报 TclError
            if menu is not None:
                with suppress(tk.TclError):
                    if not menu.winfo_exists():
                        menu = None
        with suppress(tk.TclError):
            self.root.configure(menu=menu if menu is not None else "")

    def _on_tool_menu_change(self, host: ToolHost, menu: tk.Menu | None) -> None:
        """工具声明菜单时调用：记录在本页上，仅当本页激活才挂到顶层。"""
        host.menu = menu
        if host is not None and host.winfo_exists() and host is self._current_host():
            with suppress(tk.TclError):
                self.root.configure(menu=menu if menu is not None else "")

    def _current_host(self) -> ToolHost | None:
        try:
            widget = self.root.nametowidget(self.notebook.select())
        except (tk.TclError, KeyError):
            return None
        return widget if isinstance(widget, ToolHost) else None

    # ------------------------------------------------------------------
    # 打开工具
    # ------------------------------------------------------------------
    def open_tool(self, spec: ToolSpec) -> None:
        # 退出流程进行中：不允许再打开新工具，否则新页不在待关闭集合里，
        # 会出现"窗口正要关闭却又有新任务进来"。（回归 2026-10-03 R3）
        if self._closing_hosts is not None:
            self.status_var.set("正在退出，已忽略打开工具的请求")
            return

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
            on_close_request=self._on_tool_close_request,
            on_menu_change=self._on_tool_menu_change,
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
        # 退出流程进行中：不允许再启动新子进程，否则它不在关闭协调范围内。
        if self._closing_hosts is not None:
            self.status_var.set("正在退出，已忽略启动子进程工具的请求")
            return

        env = os.environ.copy()
        path_prepend = os.pathsep.join(str(p) for p in spec.extra_paths)
        if path_prepend:
            # 子进程工具可能采用 src 布局，把其源码目录交给 Python 的导入系统
            env["PYTHONPATH"] = (
                f"{path_prepend}{os.pathsep}{env['PYTHONPATH']}"
                if env.get("PYTHONPATH")
                else path_prepend
            )

        # 各子进程工具的依赖锁定在它们自己的 venv 里（如原子标注工具锁定
        # numpy 1.26.4，套件锁为 numpy 2.2.6）。源码模式下**必须**用工具自己的
        # 解释器：缺失时明确报环境缺失并停止启动，绝不回退到套件 Python ——
        # 那会打破依赖隔离契约，告警不等于授权回退（回归 2026-10-03 R3）。
        python_exe = spec.resolve_python_exe()
        if python_exe is None and not getattr(sys, "frozen", False):
            expected = spec.project_dir / ".venv"
            logger.error(
                "%s 缺少项目自带解释器（%s），已停止启动以避免破坏依赖隔离",
                spec.name, expected,
            )
            messagebox.showerror(
                "环境缺失",
                f"{spec.name} 未找到项目自己的虚拟环境解释器：\n{expected}\n\n"
                "为避免用套件解释器运行而破坏该工具的依赖隔离，已停止启动。\n"
                "请先按该项目的 requirements.lock.txt 建立 .venv 后重试。",
                parent=self.root,
            )
            return

        if spec.subprocess_module:
            # 包内使用相对导入的模块无法按脚本路径执行，以 python -m 方式启动
            if python_exe is None:
                # 冻结 EXE 的打包运行时：sys.executable 是 TEM Suite.exe 本身，
                # 由它以 --run-module 代为执行（保留既有打包支持）。
                cmd = [sys.executable, "--run-module", spec.subprocess_module]
            else:
                cmd = [str(python_exe), "-m", spec.subprocess_module, *spec.subprocess_args]
        else:
            script = spec.project_dir / spec.entry
            if not script.is_file():
                messagebox.showerror("无法启动", f"未找到脚本：\n{script}", parent=self.root)
                return

            if python_exe is None:
                # 同上：冻结 EXE 以 --run-script 让另一个实例代为执行外部脚本
                cmd = [sys.executable, "--run-script", str(script)]
            else:
                cmd = [str(python_exe), str(script), *spec.subprocess_args]

        log_file = None
        try:
            # 输出重定向到临时文件而不是 PIPE：PIPE 需要有人持续读取，若只在
            # poll() 结束后才 read()，子进程写满管道缓冲区（Windows 约 4-64 KB）
            # 就会阻塞在 write 上永不退出（回归 2026-10-03 R3）。
            log_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
            proc = subprocess.Popen(  # noqa: S603 - 路径来自内部注册表
                cmd,
                cwd=str(spec.project_dir),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            # 启动失败必须关掉已创建的句柄，否则泄漏文件描述符
            if log_file is not None:
                with suppress(Exception):
                    log_file.close()
            messagebox.showerror("启动失败", str(exc), parent=self.root)
            return
        self._reap_subprocesses()
        self._subprocesses.append(proc)
        self._subprocess_logs[proc] = log_file
        self.status_var.set(f"{spec.name} 已在独立窗口中启动")
        # 打包后子工具 console=False，崩溃时无任何可见信息，必须轮询回报
        self.root.after(500, self._watch_subprocess, proc, spec)

    def _read_subprocess_tail(self, proc: subprocess.Popen, limit: int = 4000) -> str:
        """读取该子进程日志**末尾**最多 ``limit`` 个字符（进程退出后调用）。

        文件方式不需要运行期 drain：子进程直接写文件，没有管道缓冲区上限，
        不会因为没人读而卡住。运行期反复 ``seek(0)+read`` 会重复读取越来
        越多的内容、并使共享文件偏移互相干扰（回归 2026-10-03 R3），因此
        只在进程**已退出**后读一次尾部。
        """
        handle = self._subprocess_logs.get(proc)
        if handle is None:
            return ""
        with suppress(Exception):
            handle.flush()
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit), os.SEEK_SET)
            return handle.read() or ""
        return ""

    def _close_subprocess_log(self, proc: subprocess.Popen) -> None:
        handle = self._subprocess_logs.pop(proc, None)
        if handle is not None:
            with suppress(Exception):
                handle.close()

    # ------------------------------------------------------------------
    def _reap_subprocesses(self) -> None:
        """只从列表里移除已退出的子进程引用。

        **不关闭日志句柄**：另一个仍在 ``_watch_subprocess`` 轮询中的子进程
        可能与其并行退出，提前关掉它的句柄会让后一个的错误输出丢失
        （回归 2026-10-03 R3）。句柄统一由 ``_watch_subprocess`` 在读完 tail
        之后关闭。
        """
        self._subprocesses = [proc for proc in self._subprocesses if proc.poll() is None]

    def _watch_subprocess(self, proc: subprocess.Popen, spec: ToolSpec) -> None:
        """轮询子进程：非零退出时把捕获的输出尾部显示给用户。

        打包后子工具以 ``console=False`` 运行，启动即崩时用户完全无感
        （回归 2026-09-29 P0）。输出收在临时文件里：运行期**不读**（文件方式
        无需 drain），退出后只读有上限的 tail。
        """
        if proc.poll() is None:
            self.root.after(500, self._watch_subprocess, proc, spec)
            return

        # 先读 tail，再关本进程的句柄（_reap_subprocesses 不关别人的句柄）
        output = self._read_subprocess_tail(proc)
        self._reap_subprocesses()
        self._close_subprocess_log(proc)
        if proc.returncode:
            detail = output.strip() if output.strip() else "（无输出）"
            logger.error("子进程工具 %s 退出码 %s", spec.tool_id, proc.returncode)
            self.status_var.set(f"{spec.name} 启动失败（退出码 {proc.returncode}）")
            messagebox.showerror(
                "工具启动失败",
                f"{spec.name} 已退出（退出码 {proc.returncode}）。\n\n{detail}",
                parent=self.root,
            )
        else:
            logger.info("子进程工具 %s 正常退出", spec.tool_id)

    def _on_tool_close_request(self, host: ToolHost) -> None:
        """工具主动请求关闭（``root.destroy()`` / ``quit()``）。

        这是**明确**的"关掉我"信号：无论工具是同步销毁、还是延迟清理后再销毁，
        都直接摘除标签页，且不再重跑其关闭回调（否则会二次弹确认框、二次写配置）。
        """
        self._forget_tool(host)
        self._remove_tab(host)
        self.status_var.set("标签页已关闭")
        self._apply_active_menu()

    def _close_current_tab(self) -> None:
        """关闭当前标签页（底部按钮）。"""
        try:
            index = self.notebook.index(self.notebook.select())
        except tk.TclError:
            return
        self._close_tab_at(index)

    def _forget_tool(self, host: ToolHost) -> None:
        """按部件（而非索引）清理 ``_open_tools`` 登记。"""
        for tool_id, (registered, _instance) in list(self._open_tools.items()):
            if registered is host:
                self._open_tools.pop(tool_id, None)
                return

    def _close_tab_at(self, index: int) -> None:
        """关闭指定索引的标签页（用户点标签页关闭按钮 / 中键点击）。

        绝不能用 ``forget(index)`` 收尾：工具接受关闭时会执行
        ``self.root.destroy()``，ToolHost 是 Notebook 的子部件，Notebook 会自动
        移除该页，此时 ``index`` 已过期，``forget(index)`` 会误删**后一个**标签页；
        单页时则直接抛 "Slave index out of bounds"（回归 2026-09-29 P0，本轮
        真实 Tk 复现：同步 destroy 后邻页 tab 消失）。一律按**部件**摘除。

        关闭决策交给工具自己的 ``WM_DELETE_WINDOW`` 处理器 —— 与点它自己的关闭
        按钮行为完全一致：处理器可能弹确认框并**拒绝**关闭（此时标签页与登记
        都必须原样保留），也可能先取消后台任务、延迟若干毫秒后再销毁（避免截断
        正在写出的文件）。未注册处理器的工具直接关闭。
        """
        try:
            widget_name = self.notebook.tabs()[index]
        except IndexError:
            return
        widget = self.root.nametowidget(widget_name)
        if widget is self._welcome:
            return  # 起始页不可关闭

        if not isinstance(widget, ToolHost):
            self._remove_tab(widget)
            self.status_var.set("标签页已关闭")
            self._apply_active_menu()
            return

        if widget.close_in_progress:
            # 处理器同步销毁了自己，destroy() 重入到本函数：直接摘除
            self._forget_tool(widget)
            self._remove_tab(widget)
            self.status_var.set("标签页已关闭")
            self._apply_active_menu()
            return

        handler = widget._close_callbacks.get("WM_DELETE_WINDOW")
        if handler is None:
            self._forget_tool(widget)
            self._remove_tab(widget)
            self.status_var.set("标签页已关闭")
            self._apply_active_menu()
            return

        widget.close_in_progress = True
        try:
            handler()
        except Exception:  # noqa: BLE001 - 记录根因并保留页，不静默吞掉
            logger.exception("工具 %s 的关闭处理器抛出异常", widget._tool_id)
            _exc = traceback.format_exc(limit=1).strip().splitlines()[-1]
            self.status_var.set(f"关闭失败：{_exc}")
            messagebox.showerror(
                "关闭失败",
                f"工具 {widget._tool_id} 的关闭处理器抛出异常，已保留该标签页：\n"
                f"{_exc}\n\n请先处理该问题再关闭。",
                parent=self.root,
            )
            return
        finally:
            widget.close_in_progress = False

        if not widget.winfo_exists():
            # 处理器同步销毁了部件：登记与标签页随之清理
            self._forget_tool(widget)
            self.status_var.set("标签页已关闭")
            self._apply_active_menu()
            return
        # 部件仍存在：用户在确认框里选了取消（保留登记与标签页），或工具正在
        # 延迟收尾（稍后会 destroy() → _on_tool_close_request 完成关页）。
        self.status_var.set("已取消关闭")

    def _remove_tab(self, widget: tk.Misc) -> None:
        """按部件摘除并销毁标签页，避免使用过期索引。"""
        with suppress(tk.TclError):
            self.notebook.forget(widget)
        if isinstance(widget, ToolHost):
            with suppress(tk.TclError):
                widget._destroy_now()
        elif isinstance(widget, tk.Widget):
            with suppress(tk.TclError):
                widget.destroy()

    # ------------------------------------------------------------------
    # 程序级关闭：用户入口与串行轮询分离
    #
    # 用户可能连续点两次关闭按钮，或短时间内排队多个关闭请求。若入口本身也
    # 负责泵事件循环并重新调度，第二次请求就会在 root.update() 里**重入**，
    # 形成两条 after 链：一条先超时清空 _closing_hosts，另一条随即开始新一轮
    # 关闭并**再次**调用工具处理器（真实复现：close_handler_calls=2、
    # closing_state_cleared=false，回归 2026-10-03 R3）。
    #
    # 因此：
    #   * on_app_close()   —— 用户入口。已在关闭中就直接 return，绝不重入。
    #   * _poll_app_close() —— 唯一轮询体，由**单个** _close_after_id 串行调度，
    #                          自身不调用 root.update()（把控制权交回 mainloop）。
    #   * 完成/中止时取消并清空自己的 after_id；过期回调不再启动新关闭。
    # ------------------------------------------------------------------
    def on_app_close(self) -> None:
        """用户请求关闭主窗口（关闭按钮 / 协议回调）。

        每次**显式**的退出尝试只在第一轮调用一次各工具的
        ``WM_DELETE_WINDOW`` 处理器；随后由 :meth:`_poll_app_close` 串行轮询，
        直到待关闭集合中的工具全部完成，才真正关闭主窗口。

        超时或处理器异常会**中止本次退出**并保留窗口（不终止子进程、不销毁
        root），把原因告知用户。中止后状态被清空，用户下一次显式关闭视为一次
        **新的退出尝试**，会重新询问尚未关闭的工具 —— 这是有意为之。
        """
        if self._closing_hosts is not None:
            # 已在关闭流程中：忽略重复/排队请求，避免第二条 after 链重入
            return
        self._begin_app_close()

    def _begin_app_close(self) -> None:
        """开始一次新的退出尝试：冻结集合、跑一轮处理器、启动唯一轮询。"""
        self._closing_hosts = [host for host, _ in self._open_tools.values()]
        self._close_wait_started = time.monotonic()
        self._close_handler_errors.clear()
        self.status_var.set("正在等待工具完成后台清理…")

        for host in self._closing_hosts:
            if not self._host_alive(host):
                continue
            if host._close_callbacks.get("WM_DELETE_WINDOW") is None:
                # 无关闭处理器：没有谁会异步销毁它，直接按关页流程摘除，
                # 否则会一直等到超时。
                self._forget_tool(host)
                self._remove_tab(host)
                continue
            try:
                host.run_close_callbacks()
            except Exception as exc:  # noqa: BLE001 - 记录根因并保留页
                logger.exception("工具 %s 的关闭处理器抛出异常", host._tool_id)
                self._close_handler_errors.append(f"{host._tool_id}: {exc}")

        if self._all_hosts_closed():
            self._finish_app_close()
            return
        self._schedule_close_poll()

    def _schedule_close_poll(self) -> None:
        """（重新）安排唯一轮询；同一时刻至多存在一个待执行的轮询。"""
        if self._close_after_id is not None:
            with suppress(tk.TclError):
                self.root.after_cancel(self._close_after_id)
        with suppress(tk.TclError):
            self._close_after_id = self.root.after(50, self._poll_app_close)

    def _cancel_close_poll(self) -> None:
        """取消待执行的轮询，防止过期回调再次启动关闭。"""
        if self._close_after_id is not None:
            with suppress(tk.TclError):
                self.root.after_cancel(self._close_after_id)
            self._close_after_id = None

    def _poll_app_close(self) -> None:
        """唯一轮询体：检查进度，完成则收尾，超时则中止。

        自身**不**调用 ``root.update()`` —— 在 ``after`` 回调里再泵事件循环会
        让排队的关闭请求重入本流程。控制权交回 mainloop 即可让工具的
        ``after(...)`` 收尾正常执行。
        """
        self._close_after_id = None
        if self._closing_hosts is None:
            return                     # 本轮已结束：过期回调不启动新关闭

        if self._all_hosts_closed():
            self._finish_app_close()
            return

        if time.monotonic() - self._close_wait_started > _CLOSE_WAIT_TIMEOUT_S:
            self._abort_app_close()
            return

        self._schedule_close_poll()

    def _abort_app_close(self) -> None:
        """中止本次退出：保留窗口与登记，说明原因，允许安全重试。"""
        self._cancel_close_poll()
        pending = [
            host._tool_id for host in (self._closing_hosts or [])
            if self._host_alive(host)
        ]
        logger.warning("等待工具清理超时，已中止本次退出；仍在收尾: %s", pending)
        self.status_var.set(
            f"仍有 {len(pending)} 个工具在收尾，已取消退出：{'、'.join(pending)}"
        )
        message = (
            f"仍有 {len(pending)} 个工具未完成后台清理：{'、'.join(pending)}\n\n"
            "为避免截断正在写出的文件，已取消本次退出，主窗口保持打开。\n"
            "请等待其任务结束后再次关闭。"
        )
        if self._close_handler_errors:
            message += (
                "\n\n以下工具的关闭处理器抛出异常，需先处理：\n  "
                + "\n  ".join(self._close_handler_errors)
            )
        messagebox.showerror("退出已取消", message, parent=self.root)
        self._closing_hosts = None          # 允许安全重试
        self._close_handler_errors.clear()

    def _all_hosts_closed(self) -> bool:
        """待关闭集合中的工具是否都已不存在。"""
        assert self._closing_hosts is not None
        return not any(self._host_alive(host) for host in self._closing_hosts)

    def _finish_app_close(self) -> None:
        """所有工具都已完成清理：终止仍在运行的子进程并关闭主窗口。"""
        self._cancel_close_poll()
        for proc in list(self._subprocesses):
            if proc.poll() is None:
                with suppress(OSError):
                    proc.terminate()
            self._close_subprocess_log(proc)
        self._subprocesses.clear()
        self._closing_hosts = None
        self._close_handler_errors.clear()
        with suppress(tk.TclError):
            self.root.destroy()

    @staticmethod
    def _host_alive(host: ToolHost) -> bool:
        try:
            return bool(host.winfo_exists())
        except tk.TclError:
            return False


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
