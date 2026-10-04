# TEM Suite 源码与验证导航

设计背景见 [README](README.md)，修改约定见 [AGENTS.md](AGENTS.md)。以下路径相对本项目；命令从全整合目录执行。

## 运行关系

[run.py](run.py) → [temsuite/app.py](temsuite/app.py) 的 main / SuiteApp → 根据 [registry.py](temsuite/registry.py) 的 ToolSpec 选择内嵌或子进程。

- 内嵌：ProjectLoader 加载工具源码 → ToolHost 提供 Tk 接口 → 工具 GUI 工厂创建标签页内容。
- 子进程：根据 subprocess_module 或脚本 entry 启动；冻结 EXE 分别经 --run-module / --run-script 转发。
- 工作区定位：TEMSUITE_WORKSPACE 优先，否则从源码/EXE 位置向上查找标志目录。
- [TEMSuite.spec](TEMSuite.spec) 打包整合层和依赖，不把所有独立工具源码封入 EXE。已有根目录 TEM Suite.exe 仍依赖工作区源码；本轮没有重打包。

## 按任务定位

| 修改目标 | 文件与关键符号 |
|---|---|
| 增减工具、路径和入口 | [registry.py](temsuite/registry.py)：ToolSpec、TOOLS、_detect_workspace_root |
| 标签页、错误恢复和关闭 | [app.py](temsuite/app.py)：SuiteApp.open_tool、_instantiate、_close_tab_at、_rollback_global_state、on_app_close |
| 同名模块与路径隔离 | [loader.py](temsuite/loader.py)：ProjectLoader.load_module、load_file、_scan_top_level_names |
| Tk 代理、按页键绑定 | [hostframe.py](temsuite/hostframe.py)：ToolHost、run_close_callbacks |
| 临时 Tk 构造 / ttk 主题保护 | [tkpatch.py](temsuite/tkpatch.py)：tk_root_redirected、ttk_theme_frozen |
| matplotlib 后端 | [mplbackend.py](temsuite/mplbackend.py)：force_headless_backend、backend_locked |
| 子进程及冻结模式 | app.py：_launch_subprocess、_watch_subprocess、_run_external_script、_run_external_module |
| 源码冒烟 | [tests/smoke_test.py](tests/smoke_test.py)：instantiate、main、_check_toolhost_invariants |
| 成品自检 | [selftest.py](temsuite/selftest.py)：run_self_test；会写 selftest_report.txt 并弹窗 |
| 源码版本 | [pyproject.toml](pyproject.toml) 的 project.version 与 [__init__.py](temsuite/__init__.py) 的 __version__ |

ToolSpec 中 relative_dir、entry、factory、load_mode、root_mode、run_mode、src_layout、preload 是入口契约。首批两个工具为 eels_edge_analyzer → eels_edge_analyzer.gui.EELSEdgeAnalyzerApp，以及 ppa_strain → ppa.AtomMarkerApp；真实定义以 registry.py 为准。

## 容易误改的约束

- 各项目的 main、core、utils 等顶层名可能冲突；普通包和无 __init__.py 的命名空间包都需识别。延迟导入还可能需要 preload。
- 不能把 sys.modules 中当前名字的归属等同于已创建对象的归属；重命名、重载与跨副本 isinstance 都需要谨慎验证。
- 本集成设计要求单一主 Tk 事件循环。构造重定向是受控的一次性适配，不能全程替换 Tk。
- matplotlib 先解析 Agg 后端，再在加载/实例化阶段锁定；绘图通过显式 FigureCanvasTkAgg 嵌入。仅删除 matplotlib.use 调用可能破坏该顺序。
- ttk 主题与样式是进程共享状态。构造失败也需恢复；样式测试是正确性检查，不只是外观测试。
- ToolHost 将键绑定转发到真实顶层窗口并按活动标签页隔离。关闭时清理绑定、after 轮询和工具任务。
- 子进程模式不能一律改成 sys.executable script.py：冻结后 sys.executable 指向 TEM Suite.exe。
- 各工具源码的延迟 import 不一定会被 PyInstaller 静态发现，hiddenimports 与资源收集必须按真实依赖维护。

## 环境与验证

- **解释器**：`.venv/Scripts/python.exe`（本项目独立 venv；清单登记 Python >=3.10，本机母本 3.10）。
- **安装锁**：`requirements.lock.txt`；按锁安装，不要用 `pip install -r requirements.txt` 代替。
- **建环境**：参考仓库根 [环境隔离说明](../环境隔离说明.md) 与 [.tools/projects.json](../.tools/projects.json)；常规 `provision-envs.py` 只读锁、`verify-envs.py` 只验证不安装。
- **导入自检**：`numpy`、`scipy`、`matplotlib`、`pandas`、`tifffile`、`PIL`、`skimage`、`seaborn`、`openpyxl`、`imagecodecs`、`defusedxml`、`ttkbootstrap`、`ncempy`、`tkinterdnd2`、`cv2`。
- **注意**：tests/test_integration_review.py 是集成回归 pytest 套件；tests/smoke_test.py 是 14 工具构造冒烟脚本（直接改 stdout，不能与 pytest 同进程收集，故分开声明）。

从**本项目目录**执行：

~~~powershell
# pytest
& ./.venv/Scripts/python.exe -m pytest tests/test_integration_review.py -q -rs
# smoke_test
& ./.venv/Scripts/python.exe tests/smoke_test.py
~~~

旧报告不代表当前通过；运行后按实际结果记录 pass / skip / 未运行。
