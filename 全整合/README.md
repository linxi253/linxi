# TEM Suite —— 电镜数据分析工具集成软件

把 16 个原本独立的电镜分析工具整合进单一窗口（14 个内嵌 + 2 个独立子进程），
按「工具树 + 标签页」布局组织，遵循电镜数据的实际处理流程分组。

## 核心设计：不修改任何原项目

每个被整合的工具都是独立的 git 仓库且在持续演进。整合层**不改动它们一行代码**，
而是通过运行期适配把它们「借」进主窗口。这样原项目可以继续独立开发，
整合层自动获得上游改动，不存在代码分叉与同步负担。

工具源码通过 `sys.path` 按需引用，不复制。

## 启动

按仓库的[环境隔离约定](../环境隔离说明.md)，**必须使用项目自己的虚拟环境**，不要用 PATH 上的 `python`：

```powershell
cd 全整合
python -m venv .venv
.venv\Scripts\python -X utf8 -m pip install -r requirements.lock.txt
.venv\Scripts\python run.py
```

`requirements.txt` 是宽松的上游依赖范围；`requirements.lock.txt` 是锁定的精确版本，复现请用后者。

工具项目的存放位置默认为本目录的父目录（仓库根 `AIforTEM`），
可用环境变量 `TEMSUITE_WORKSPACE` 覆盖。

## 已整合的工具

| 分组 | 工具 | 接入方式 |
| --- | --- | --- |
| 1 · 数据提取 | 视频帧提取 | 内嵌（拦截 Tk 构造） |
| 2 · 图像处理 | TIFF 漂移矫正 | 内嵌 |
| | HRTEM/STEM 滤波 | 内嵌（src 布局） |
| | STEM 图像优化 | 内嵌（拦截 Tk 构造） |
| | TIF 滤镜工具 | 内嵌 |
| 3 · EELS谱学 | EELS边缘价态分析 | 内嵌（src 布局） |
| 4 · 应变分析 | Strain++ GPA | 内嵌 |
| | PPA 原子级应力 | 内嵌 |
| | 原子识别与强度分析 | 内嵌 |
| | 原子标注工具 | 独立子进程（python -m） |
| 5 · 4D-STEM | 4D-STEM 处理 | 内嵌 |
| 6 · 定量统计 | 原子衬度统计 | 内嵌（按文件路径加载） |
| | 晶体/非晶区域统计 | 内嵌 |
| | TIFF 面积测量 | 内嵌（拦截 Tk 构造 + 文件路径加载） |
| | 特征区域演化分析 | 独立子进程 |
| 7 · 模拟仿真 | HRTEM 高分辨模拟 | 内嵌（tem_sim 引擎） |

本表不重复维护版本号（曾与下方表格失同步）；各工具当前版本以下方「版本对齐」表为唯一来源。

## 版本对齐（2026-09-28）

各工具保持独立演进，整合层按以下路径在运行时引用最新源码。
版本号取自各工具的代码常量（`__version__` / `pyproject.toml`）。

| 工具 | 当前版本 | 引用路径 |
| --- | --- | --- |
| 视频帧提取 | 4.4 | `01-视频与数据提取/视频切片工具` |
| TIFF 漂移矫正 | 7.3.0（v5.2 + v6.1 合并） | `02-图像处理/drift-correction-v7` |
| HRTEM/STEM 滤波 | 5.1.0 | `02-图像处理/hrtem-HRTEM滤波工具` |
| STEM 图像优化 | 2.1.0 | `02-图像处理/stem-optimize-STEM图像优化` |
| TIF 滤镜工具 | 1.2.0 | `02-图像处理/图像加滤镜工具` |
| EELS 边缘价态分析 | 0.2.0 | `05-EELS分析/EELS边缘价态分析工具` |
| Strain++ GPA | 1.4.1（GPL-3.0） | `03-应变分析/strainpp-GPA应变分析` |
| PPA 原子级应力 | 3.4.0 | `03-应变分析/原子级应力分析-PPA` |
| 原子识别与强度分析 | 1.3.0 | `03-应变分析/原子识别纯算法` |
| 原子标注工具 | 0.3.0 | `03-应变分析/原子中心识别模型开发` |
| 4D-STEM 处理 | 2.2.0 | `05-4D-STEM分析/4D-STEM-Processor` |
| 原子衬度统计 | 2.5.0 | `04-统计分析/原子衬度统计` |
| 晶体/非晶区域统计 | 5.3 | `04-统计分析/非晶面积统计` |
| TIFF 面积测量 | —（未维护版本常量） | `04-统计分析/统计面积` |
| 特征区域演化分析 | 2026.09.2 | `04-统计分析/特征区域演化分析` |
| HRTEM 高分辨模拟 | 1.2.0（tem_sim 引擎） | `09-HRTEM模拟` |

> 目前**未**接入 TEM Suite：`010-STEM模拟`（0.1.0，由独立 `run.bat` 启动）与
> `02-图像处理/离域效应去除工具`（1.1.0）——两者均未在 `temsuite/registry.py` 中注册。

旧版项目不再作为整合入口；其源码归档在开发机的 `08-历史版本` 目录，该归档未随本仓库分发（仓库内不存在此目录）。

## 整合过程中解决的四个真实冲突

### 1. 一个进程只能有一个 Tk 根窗口

14 个内嵌工具中有 11 个的 GUI 类接收 `root` 参数，3 个在 `__init__` 内部自建 `tk.Tk()`。

`hostframe.ToolHost` 继承 `ttk.Frame`（可作标签页内容），同时模拟 Tk 根窗口独有的接口。
统计各工具对 root 的调用共涉及 17 个方法，其中 9 个是 `ttk.Frame` 自带的可直接透传，
只有 8 类需要模拟（`title` 转为更新标签页文字，`geometry`/`minsize`/`mainloop` 忽略，
`protocol` 转为标签页关闭回调等）。

对自建根窗口的 3 个工具，`tkpatch.tk_root_redirected` 在实例化期间临时把 `tkinter.Tk`
替换为返回 `ToolHost` 的工厂。采用**一次性语义**：仅首次调用交出 host，
后续调用放行给真实 `Tk` —— 否则 matplotlib 的 TkAgg 后端会误把标签页容器当顶层窗口。

### 2. 顶层模块名冲突

多个项目存在同名顶层模块，若把项目目录一并加入 `sys.path`，
先加载者会永久占据 `sys.modules`，后加载者拿到别人的模块：

- `main.py` —— STEM 图像优化 / TIF 滤镜工具 / 晶体非晶统计，三方冲突
- `core/` —— 晶体非晶统计（常规包）/ 4D-STEM（**命名空间包**，无 `__init__.py`），双方冲突
- `pipeline.py`、`filters.py`、`utils.py`、`version.py`、`constants.py` 等常见名

`loader.ProjectLoader` 按需加载，并在加载前把其他工具占用的同名模块从 `sys.modules`
暂存归档。该策略成立的依据是：模块对象一旦绑定到已创建的类与函数上，
即便从 `sys.modules` 移除也不影响既有对象运行，只有运行期新执行的 import 才受影响。

注意扫描顶层名时必须同时识别常规包与命名空间包 —— 4D-STEM 的 `core/` 没有
`__init__.py`，早期漏判导致晶体非晶统计加载失败。

### 3. matplotlib 后端引发进程崩溃

半数工具在模块顶层执行 `matplotlib.use('TkAgg')`。而 4D-STEM 处理与 TIFF 面积测量
使用 `plt.subplots()`（pyplot 全局接口），TkAgg 后端会为每个 pyplot figure 创建
`tk.Tk(className="matplotlib")` —— 即**第二个 Tcl 解释器**。两个解释器各有独立事件循环，
共存会导致进程整体崩溃（实测连续打开 4 个工具后进程退出）。

`mplbackend` 把全局后端固定为 `Agg`，并在**加载与实例化期间**屏蔽其 `matplotlib.use()` 调用
（部分工具如 HRTEM 模拟不在模块顶层、而是在 `__init__` 内切换后端）。
安全依据：

1. 所有工具的图形都通过 `FigureCanvasTkAgg(fig, master)` **显式嵌入**自己的部件，
   该类不依赖全局后端；
2. 已逐一核查，**没有任何工具使用 `plt.show()`**，因此 Agg 的非交互特性不影响功能；
3. `plt.subplots()` 在 Agg 下只创建 Figure 对象，不再产生多余的 Tcl 解释器。

一个隐蔽的坑：pyplot 的后端是**惰性解析**的，模块导入并不会加载具体后端，
首次 `plt.*` 调用才触发 `switch_backend()`。若该首次解析发生在后端锁定的窗口内
（`switch_backend` 被屏蔽），pyplot 将永远拿不到后端模块，`plt.subplots()` 直接崩溃。
因此 `force_headless_backend()` 在无锁状态下立即把 pyplot 解析到 Agg，
之后的屏蔽只拦「换后端」、不影响「用后端」。

附带收益：特征区域演化分析本身就要求 Agg，固定后端后它不再与其他工具冲突。

### 4. 工具切换全局 ttk 主题

TIF 滤镜工具在 `FilterApp.__init__` 中执行 `ttk.Style().theme_use('clam')`。
ttk 主题是**进程级全局资源**，切换它会清空所有自定义样式配置
（表现为工具树的 `rowheight` 失效、中文行距不足而上下重叠），
并把其余所有工具的外观一并改成 clam。

两层防护：加载期间用 `tkpatch.ttk_theme_frozen` 屏蔽主题切换（无参查询调用照常放行）；
每次加载后再调用 `_apply_suite_styles()` 重新套用主窗口样式作为兜底。

## 依赖冲突的收敛

- **opencv-python vs opencv-python-headless**：两者占用同一 `cv2` 命名空间无法共存。
  STEM 图像优化声明 headless，其余声明完整版。统一取完整版（功能超集）。
- **numpy / matplotlib / tifffile**：各项目声明的区间存在交集，收敛为单一版本。
- 详见 `requirements.txt` 中的说明。

## 高 DPI 适配

在 4K + 200% 缩放环境下，启用 DPI 感知后 Tk 工作在真实 3840×2160 分辨率，
`tk scaling` 为 2.67，字体实际行距达 35px。所有硬编码像素值都必须经 `SuiteApp.px()`
按 `winfo_fpixels('1i') / 96` 换算，否则面板过窄、文字被横向截断。
行高则直接取字体度量 `linespace` 而非硬编码。

## 测试

```bash
python tests/smoke_test.py                     # 全部工具
python tests/smoke_test.py drift_correct       # 指定工具
```

冒烟测试在隐藏根窗口中真实构造每个工具的部件树，并检查两项不变量：

- 工具能否成功实例化；
- 加载后主窗口的**样式数据库与 ttk 主题是否仍完好**（防止第 4 类冲突回归）。

当前状态：14 个内嵌工具全部通过，样式受损数 0。

## 已知限制

- **运行期延迟导入**：若某工具在运行期间（例如点击按钮后）才首次 import 自己项目内的
  模块，且该模块名已被其他工具占用，会加载出第二份副本，跨副本的 `isinstance`
  判断可能失效。经检查各项目均在文件头部完成导入，暂未触发。
  如遇此类问题，为该工具补充 `ToolSpec.preload` 清单即可。
- **子进程工具**：特征区域演化分析（无 GUI 类的批处理，顶层 `matplotlib.use('Agg')`）
  与原子标注工具（启动对话框 → 销毁根窗口 → 重建主窗口的多根窗口设计，无法内嵌）
  以独立子进程运行，不共享主窗口的标签页与状态栏。
  后者经 `ToolSpec.subprocess_module` 以 `python -m atom_center.annotator_gui` 启动
  （包内相对导入无法按脚本路径执行）。
  子进程解释器优先取项目自带的 `.venv`（`ToolSpec.resolve_python_exe()`，亦接受
  显式 `ToolSpec.python_exe`）——各工具的依赖锁定在自己的 venv 里（如原子标注工具
  numpy 1.26.4，套件为 2.2.6），复用套件解释器会打破其版本契约。
  **源码模式下项目解释器缺失时，套件明确报「环境缺失」并停止启动该工具，绝不回退到
  套件解释器**（告警不等于授权回退，回归 2026-10-03 R3）。仅**冻结 EXE** 的打包运行时
  没有项目解释器，此时由 exe 自身的 `--run-module` / `--run-script` 模式代为执行。

## 未整合的工作区内容

以下工作区内容未纳入本 GUI 套件：或属资料 / 无界面组件，或为尚未在
`temsuite/registry.py` 中注册的独立工具（由各自入口单独启动）：

- `01-视频与数据提取/XRD计算` —— XRD 数据处理教学录屏，无可执行代码；
- `02-图像处理/离域效应去除工具`（1.1.0）—— 独立 GUI 工具，未在
  `temsuite/registry.py` 中注册，由其自身入口单独启动；
- `03-应变分析/原子标注` —— 标注任务的输出文件夹，非程序；
- `03-应变分析/原子中心识别模型开发` 的训练/评测代码 —— 仅其中的
  AtomCenterAnnotator 标注 GUI 被整合（见上表）；
- `010-STEM模拟`（0.1.0）—— 独立 GUI 工具，未在 `temsuite/registry.py`
  中注册，由 `run.bat` 单独启动；
- `10-DSH集成/tem_pipeline.py` —— 面向 DeepSeek Harness 的命令行流水线
  （视频 → TIFF 提取 → 漂移矫正），需要命令行参数、无图形界面，
  适合脚本自动化而非 GUI 集成；
- `06-独立脚本` —— 零散脚本。（旧版归档 `08-历史版本` 在开发机上，未随本仓库分发。）
- **跨工具数据流转**尚未实现。当前各工具通过文件交换数据；
  若要做「漂移矫正 → 滤波 → 应变分析」的内存级流水线，需要为工具补充数据注入接口，
  这会涉及修改原项目，需另行评估。

## 目录结构

```
全整合/
├── temsuite/
│   ├── app.py            主窗口：工具树 + 标签页工作区
│   ├── registry.py       工具注册表（元数据与接入方式）
│   ├── hostframe.py      Tk 根窗口代理（ToolHost）
│   ├── loader.py         隔离式项目加载器
│   ├── tkpatch.py        Tk 构造拦截 + ttk 主题冻结
│   └── mplbackend.py     matplotlib 后端治理
├── tests/smoke_test.py   冒烟测试（含样式完整性检查）
├── requirements.txt      统一依赖清单
└── run.py                开发模式入口
```

<!-- README-QUICKREF:BEGIN 本区块为手工维护，需与版本来源（pyproject.toml / 代码 __version__，见「版本来源」行）保持一致 -->

---

## 速查

| 项 | 内容 |
|---|---|
| 当前版本 | **1.1.0** |
| 版本来源 | `pyproject.toml` |
| 入口 | `run.py` |
| 依赖锁定 | `requirements.lock.txt` |
| 许可证 | MIT（本体代码）；**发行 exe 内含 GPL-3.0 的 ncempy**，且运行期以同进程方式加载 GPL-3.0 的 strainpp，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) |

**从源码运行**——必须使用本工具自己的虚拟环境，不要用 PATH 上的 `python`：
各工具依赖版本互不相同，共用解释器会互相污染。

```powershell
cd <本工具目录>
python -m venv .venv
.venv\Scripts\python -X utf8 -m pip install -r requirements.lock.txt
.venv\Scripts\python run.py
```

**未纳入版本控制**：`.venv/`、`build/`、`dist/`、`release/`、`__pycache__/`、
测试数据与运行输出。具体规则见本目录 `.gitignore`。

<!-- README-QUICKREF:END -->
