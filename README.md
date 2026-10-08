# AIforTEM 电镜数据处理工具集

[English](README_EN.md) | 中文

本仓库汇总了研究生期间开发、整理和打包的 TEM、HRTEM、STEM、4D-STEM 图像处理、EELS 谱学与定量分析工具源码。
各工具原本相互独立（各自维护 git 仓库），这里按原有目录结构统一收录，方便一处查阅与克隆。

> **本仓库只收录源码与文档。** 可执行文件（`.exe`）、模型权重（`.pt`/`.onnx`）、电镜原始数据（`.tif`/`.dm3`/`.dm4`）
> 和 PyInstaller 构建产物体积过大，均未入库，详见 [.gitignore](.gitignore)。需要成品程序请按各工具目录的 README 自行打包。

## 工具一览

版本号取自各工具的代码常量（`__version__` / `pyproject.toml`），最后核对于 2026-09-28。

| 分类 | 工具 | 版本 | 源码目录 | 入口 |
|---|---|---|---|---|
| 视频与数据提取 | TEM 视频帧提取 | 4.4 | `01-视频与数据提取/视频切片工具` | `run_app.py` |
| 视频与数据提取 | TEM 视频流水线（切片+漂移矫正） | — | `10-DSH集成` | `tem_pipeline.py` |
| 图像处理 | TIFF 漂移矫正 | 7.3.0 | `02-图像处理/drift-correction-v7` | `drift_correction.py` |
| 图像处理 | HRTEM/STEM 滤波 | 5.1.0 | `02-图像处理/hrtem-HRTEM滤波工具` | `hrtem.py`、`cli_launcher.py` |
| 图像处理 | HRTEM/STEM 图像增强 | 2.1.0 | `02-图像处理/stem-optimize-STEM图像优化` | `main.py` |
| 图像处理 | TIF 图像滤镜 | 1.2.0 | `02-图像处理/图像加滤镜工具` | `main.py` |
| 图像处理 | 离域效应去除工具 | 1.1.0 | `02-图像处理/离域效应去除工具` | `main.py` |
| 应变分析 | Strain++ GPA ⚠️ GPL | 1.4.1 | `03-应变分析/strainpp-GPA应变分析` | `run.py` |
| 应变分析 | PPA 原子位移与应变分析 | 3.4.0 | `03-应变分析/原子级应力分析-PPA` | `ppa.py` |
| 应变分析 | 原子识别与强度分析 | 1.3.0 | `03-应变分析/原子识别纯算法` | `atomic_app.py` |
| 应变分析 | 原子中心识别模型（训练/推理工程） | 0.3.0 | `03-应变分析/原子中心识别模型开发` | `src/atom_center` |
| 统计分析 | TIF 原子衬度统计 | 2.5.0 | `04-统计分析/原子衬度统计` | `tif图像衬度分析工具.py` |
| 统计分析 | HAADF-STEM 特征区域演化 | 2026.09.2 | `04-统计分析/特征区域演化分析` | `应力面积统计.py` |
| 统计分析 | TIFF 面积测量 | — | `04-统计分析/统计面积` | `统计面积.py` |
| 统计分析 | 晶体/非晶区域统计 | 5.3 | `04-统计分析/非晶面积统计/pythonProject` | `main.py` |
| 4D-STEM | 4D-STEM Processor | 2.2.0 | `05-4D-STEM分析/4D-STEM-Processor` | `stem_processor_gui.py` |
| EELS 谱学 | EELS 边缘价态分析 | 0.2.0 | `05-EELS分析/EELS边缘价态分析工具` | `run.py` |
| HRTEM 模拟 | HRTEM 高分辨模拟工具（tem_sim 引擎） | 1.2.0 | `09-HRTEM模拟` | `run.bat` |
| STEM 模拟 | STEM-HAADF 模拟工具（stem_sim 引擎，冻结声子多层法） | 0.1.0 | `010-STEM模拟` | `run.bat` |
| 集成环境 | TEM Suite（集成 GUI） | 1.1.0 | `全整合` | `run.py` |

`—` 表示该工具未在代码中维护版本常量。

## 目录说明

- [01-视频与数据提取](01-视频与数据提取/README.md)：视频转 ImageJ TIFF、FFmpeg 与 XRD 操作录屏（后两者为本地资料，未入库）。
- [02-图像处理](02-图像处理/README.md)：漂移矫正、HRTEM/STEM 增强、离域效应去除及通用 TIF 滤镜。
- [03-应变分析](03-应变分析/README.md)：GPA、PPA 应变分析，原子识别与原子中心识别模型工程。
- [04-统计分析](04-统计分析/README.md)：衬度、区域演化、手工面积及晶体/非晶统计。
- [05-4D-STEM分析](05-4D-STEM分析/README.md)：DM4 数据的 DPC、SSB、取向、应变和叠层成像。
- [05-EELS分析](05-EELS分析/README.md)：Dual-EELS 边缘价态定量分析（MLLS 拟合、复散射校正、bootstrap）。
- [06-独立脚本](06-独立脚本/README.md)：未封装成项目的去噪和安装脚本。
- [09-HRTEM模拟](09-HRTEM模拟/README.md)：CIF 导入、带轴/电镜参数/厚度/取向可控的 HRTEM 多层法模拟 GUI。
- [010-STEM模拟](010-STEM模拟/README.md)：CIF 导入、带轴/厚度/探针/环形探测器可控的 HAADF/ADF/BF/ABF STEM 模拟 GUI。
- [10-DSH集成](10-DSH集成/README.md)：DeepSeek Harness 会话内的 TEM 视频流水线。
- [全整合](全整合/README.md)：TEM Suite 集成层源码（工具树 + 标签页）。

> 编号说明：`05-` 有两个目录（4D-STEM 与 EELS）系历史遗留，为避免破坏各工具内部的相对引用，未做重排。

## 通用运行方式

**从源码运行时，必须使用项目自己的虚拟环境**，不要用 PATH 上的 `python`：
各工具的依赖版本互不相同（例如 numpy 1.26 与 2.2 并存、Python 3.10 与 3.12 并存），
共用解释器会让工具读到别的项目的版本，`pip install` 还会污染全局环境、连带破坏其它工具。

每个项目首次准备环境（只需一次）：

```powershell
cd <项目目录>
python -m venv .venv
.venv\Scripts\python -X utf8 -m pip install -r requirements.lock.txt
```

之后用项目内解释器运行：

```powershell
.venv\Scripts\python <入口脚本.py>
```

`09-HRTEM模拟`、`010-STEM模拟`、`原子识别纯算法` 提供 `run.bat`，双击即可。

带 `.spec` 文件的项目可用 PyInstaller 在**项目内构建环境**中重新打包：

```powershell
.venv\Scripts\python -m pip install pyinstaller
.venv\Scripts\python -m PyInstaller <项目.spec>
```

> 完整的隔离约定与各工具的环境清单见 [环境隔离说明](环境隔离说明.md)。
> 随时运行 `python .tools\check-env-isolation.py` 检查约定是否被破坏。

具体入口和例外情况以各子目录的 README 为准。

## 数据与结果注意事项

- 处理实验原始数据前先复制一份样例；输出目录不要与输入目录相同，避免覆盖源文件。
- TIFF 堆栈可能非常大，应预留足够内存和磁盘空间。
- 8 位与 16 位 TIFF 的强度范围不同。定量分析前确认工具是否保留原始位深。
- GPA、PPA、DPC、SSB、MLLS 等结果依赖采集条件、标定和参数选择，软件输出不能替代物理有效性验证。

## 许可证与来源

本仓库为 **MIT**（见 [LICENSE](LICENSE)），适用于其中的原创代码。

**例外：`03-应变分析/strainpp-GPA应变分析/` 采用 GPL-3.0-or-later**（派生自
[JJPPeters/Strainpp](https://github.com/JJPPeters/Strainpp)，该目录自带 `LICENSE`）。
GPL-3.0 具传染性：若把它与其余代码打包成**单一作品**分发，整体授权需按 GPL-3.0
重新审视；仅在同一仓库中并列存放、各自独立分发则不受影响。

逐工具的许可证、第三方依赖、以及**未收录的第三方材料**清单见
[NOTICE.md](NOTICE.md)。

## 引用与贡献

- **引用**：见 [CITATION.cff](CITATION.cff)。引用本工具集时，请一并记录你实际使用的
  具体子工具、版本或 commit，以及相关上游方法的出处。
- **贡献**：见 [CONTRIBUTING.md](CONTRIBUTING.md)（先选子项目并读其 README；
  每个项目使用独立环境与实际锁文件；原始实验数据只读且不提交机密/大数据）。

## README 覆盖范围

README 以“独立项目或资料集合”为单位撰写，并列出其中的关键文件。生成物与外部依赖不逐层创建 README。
