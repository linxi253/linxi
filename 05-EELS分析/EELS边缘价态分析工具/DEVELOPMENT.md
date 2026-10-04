# EELS 源码与验证导航

本页面向开发；功能使用方法仍在 [README](README.md)，工作约定在 [AGENTS.md](AGENTS.md)。以下路径相对本项目，命令从项目根目录执行。

## 入口与版本

| 场景 | 入口或依据 |
|---|---|
| 独立 GUI | [run.py](run.py) → [gui.py](src/eels_edge_analyzer/gui.py) 的 EELSEdgeAnalyzerApp / main |
| CLI | [__main__.py](src/eels_edge_analyzer/__main__.py) → [cli.py](src/eels_edge_analyzer/cli.py) 的 main |
| TEM Suite | tool_id 为 eels_edge_analyzer；entry 为 eels_edge_analyzer.gui；factory 为 EELSEdgeAnalyzerApp；src_layout 为 src |
| 包版本 | [pyproject.toml](pyproject.toml) 的 project.version 与 [__init__.py](src/eels_edge_analyzer/__init__.py) 的 __version__ |
| 打包入口 | [EELS_Edge_Analyzer.spec](EELS_Edge_Analyzer.spec)，显式收集内置 Cu L2,3 预设 |

README 中的 v1/v1.1/v1.2 是功能说明与修订记录，不等同于包版本。已有 EXE 是否对应当前源码需要单独核实；导航校验不认证成品版本。

## 按任务定位

| 修改目标 | 首先阅读 | 关键符号 / 责任 |
|---|---|---|
| 参数、维度、输入约束 | [models.py](src/eels_edge_analyzer/models.py) | AnalysisConfig、validate_paired_energy_axes；dataclass 及配置校验 |
| 预设与历史配置 | [presets.py](src/eels_edge_analyzer/presets.py)、[cu_l23.json](src/eels_edge_analyzer/presets/cu_l23.json) | load_preset、config_from_saved；JSON 参数解析 |
| DM3/DM4 与原始输入身份 | [dm4io.py](src/eels_edge_analyzer/dm4io.py) | 对象检查、数据装载、HashWatch |
| 参考谱 | [references.py](src/eels_edge_analyzer/references.py) | 参考文件读取、能量范围与参考矩阵准备 |
| 配准、表面、去卷积 | [processing.py](src/eels_edge_analyzer/processing.py) | trace_surface、inward_distance_nm、fourier_ratio_deconvolution |
| MLLS、bootstrap、注入恢复 | [fitting.py](src/eels_edge_analyzer/fitting.py) | fit_all_cu_models、paired_bootstrap_difference、injection_recovery |
| 分析顺序、质量结论 | [pipeline.py](src/eels_edge_analyzer/pipeline.py) | run_analysis；组合处理、拟合、敏感性及完整性检查 |
| 文件、图表和报告 | [reporting.py](src/eels_edge_analyzer/reporting.py) | output_directory_state、export_artifacts；输出字段和覆盖策略 |
| CLI 参数与返回码 | [cli.py](src/eels_edge_analyzer/cli.py) | _config_overrides、_build_config、main |
| GUI 配置与后台生命周期 | [gui.py](src/eels_edge_analyzer/gui.py) | _make_config、_optional_positive_float、_worker、_poll_messages、_on_close |

主调用链：GUI/CLI 组装 AnalysisConfig → run_analysis 返回 AnalysisArtifacts → export_artifacts 写结果。定位导出问题不必先通读整个 GUI；定位数值问题先查对应处理模块与核心测试。

## 行为约束

- 低损与高损 SI 使用 energy × y × x；除空间配对外还校验能量通道数及色散。不要把"尺寸一致"当作物理配对充分条件。
- 配对校验由 [models.py](src/eels_edge_analyzer/models.py) 的 `validate_paired_energy_axes` 承担：一维、通道数一致、长度 ≥ 2、全部有限、各自均匀、严格递增、两轴色散在容差内一致。**不比较能量原点**（低损含 ZLP、高损从边开始，起点本就不同），**不做重采样**。
- 容差：基础相对 1e-3·step；再按 float32 坐标量子 `eps32·max|E|` 补偿，且补偿项受硬上限约束（均匀性 5%·step、色散 2e-3·maxstep）。超过上限即报"坐标精度不足"，不放大容差。这是**有限支持边界**：不保证每条高分辨率 float32 轴都被支持，例如 900 eV 原点、0.001 eV/ch 的轴会被诚实拒绝；对应地，3%–20% 的真实色散失配不会被放过。被拒时请从原始能量标定重新生成等间距坐标，**不要**为绕开检查而改能量原点（拟合窗口按绝对 eV 定义）。
- `processing.fourier_ratio_deconvolution*` 的 `high_energy_ev` 是**可选**键字：传入才做配对校验；不传时**只保证低损轴合法**，不得声称已检查配对。`pipeline.run_analysis` 与敏感性分析路径都已传入。
- 能量轴使用 eV，距样品表面分层使用 nm。表面方向、边界近似和空间标尺转换影响分层结果。
- 一般参数优先级为默认值 < 预设 < 显式 CLI 覆盖；`--config` 从既往 analysis_config.json 恢复配置，再应用显式覆盖。**所有字段在单次 `AnalysisConfig(...)` 构造中合并**（`_build_config`），不得对同一字段既显式传参又用 `**overrides` 展开，否则会 `got multiple values for keyword argument`。
- 保留随机种子、移动块重采样及配对差 bootstrap；更改平滑、归一化、能量平移或去卷积都可能改变模型比较。
- 结果是投影谱权重；质量告警、参考谱相似性及不可辨识说明是结果的一部分。
- 原始文件在计算与导出阶段受到完整性检查；已有结果默认拒绝覆盖。export_artifacts 逐文件写入，当前没有整个输出目录的事务保证，异常可能留下部分输出。
- GUI 关闭会请求取消并等待工作线程；保留该流程以避免导出截断。CLI 常规错误返回 2，KeyboardInterrupt 返回 130。
- GUI 的沿表面分段宽度输入框**默认留空**：非空时优先于预设，留空时使用预设值；取消预设且留空由 `AnalysisConfig.validate()` 报错。NaN/Inf/0/负数必须在 `_make_config` 阶段就被拒绝。

## 环境与命令

环境约定为项目内 .venv，依赖快照见 [requirements.lock.txt](requirements.lock.txt)，直接依赖见 [requirements.txt](requirements.txt)。检查环境是否存在不等同于验证它可用。环境按工作区约定从本机母本重建（见公开仓库根 [环境隔离说明.md](../../环境隔离说明.md)），不要因普通源码运行而重写锁文件。

~~~powershell
# 独立 GUI；run.py 已将 src 加入本进程导入路径
& ./.venv/Scripts/python.exe run.py

# 现有测试自己设置 src 路径；无需 editable 安装
& ./.venv/Scripts/python.exe -m pytest -q -rs
& ./.venv/Scripts/python.exe -m pytest -q -rs
~~~

pytest 是测试前置条件，运行锁并不保证包含它。环境缺失时记录未运行；需要建立测试环境时单独安排，保留运行锁。

无安装方式检查 CLI 帮助：

~~~powershell
Push-Location src
try { & ../.venv/Scripts/python.exe -m eels_edge_analyzer --help }
finally { Pop-Location }
~~~

实际分析会写输出目录；从 src 运行时输入、参考谱、输出路径应使用绝对路径，避免工作目录改变含义。

## 验证范围

| 测试 | 验证对象与限制 |
|---|---|
| [test_core.py](tests/test_core.py) | 合成数据的标尺、表面、拟合、分析链及输入保护 |
| [test_presets.py](tests/test_presets.py) | 预设解析与参数处理 |
| [test_review_fixes.py](tests/test_review_fixes.py) | CLI（含 `--config` 真实复跑）、配对能量轴校验、边界检查等修复的回归 |
| [test_gui.py](tests/test_gui.py) | GUI 配置（含沿表面分段宽度优先级与非法值）、覆盖确认及状态；需要 Tk，无显示或导入失败可能跳过 |

更改集成入口后，在全整合目录用其解释器运行：

~~~powershell
& ./.venv/Scripts/python.exe tests/smoke_test.py eels_edge_analyzer
~~~

核对输出确实执行了 eels_edge_analyzer。构造成功不能代替真实数据分析验证。

## 本公开仓库的布局差异

本项目使用独立的 `.venv`。公开版是**统一的 AIforTEM 仓库**（各工具为子目录），工作区根没有该导航脚本，因此：

- 在**公开仓库根**执行 git status/diff，而不是在本项目目录；
- 环境约定见公开仓库根 [环境隔离说明.md](../../环境隔离说明.md)；公开版 `.tools` 只有环境隔离与 CI 工具（如 `check-env-isolation.py`），没有导航检查器。

主源码同步本目录时，请保留主源码侧的导航命令，不要把它当作公开版可执行命令。
