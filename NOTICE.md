# 许可证范围与例外

根目录 [LICENSE](LICENSE) 为 **MIT**，适用于本仓库中的原创代码。
本文件说明它的适用范围与例外，避免与第三方许可证混淆。

## 例外：GPL-3.0-or-later

**`03-应变分析/strainpp-GPA应变分析/` 不采用 MIT，而是 GPL-3.0-or-later。**

该工具派生自 [JJPPeters/Strainpp](https://github.com/JJPPeters/Strainpp)，
目录内自带 `LICENSE` 与 `pyproject.toml` 中的 `license` 声明。

GPL-3.0 具传染性，需要注意两种情形：

| 情形 | 影响 |
|---|---|
| 在同一仓库中**并列存放**、各自独立分发 | 不受影响，MIT 与 GPL 组件可共存 |
| 把它与其余代码打包成**单一作品**再分发 | 整体授权需按 GPL-3.0 重新审视 |

## 例外：随发行物分发的 GPL 第三方组件

以下第三方组件采用 GPL 系许可证，**已随本项目代码一起被 PyInstaller 打进
对外发行的单文件 exe**。分发这些 exe 时，这些组件仍受其原许可证约束，
不因本项目声明 MIT 而改变：

| 组件 | 许可证 | 被哪些发行物打包 | 说明 |
|---|---|---|---|
| `ncempy`（DM3/DM4 读取） | **GPL-3.0-or-later** | `05-EELS分析/EELS边缘价态分析工具`、`全整合` | 两者的 `.spec` 均含 `collect_data_files("ncempy")` 与 ncempy 隐藏导入 |
| `ultralytics`（YOLO 训练/标注） | **AGPL-3.0** | 否——**不进发行 exe** | 仅用于 `03-应变分析/原子中心识别模型开发` 的训练与标注链路；发行推理走 onnxruntime |

`ncempy` 的 GPL-3.0 义务详见
[`05-EELS分析/EELS边缘价态分析工具/THIRD_PARTY_NOTICES.md`](05-EELS分析/EELS边缘价态分析工具/THIRD_PARTY_NOTICES.md)
与 [`全整合/THIRD_PARTY_NOTICES.md`](全整合/THIRD_PARTY_NOTICES.md)。
若不想承担 GPL 义务，可改为让 ncempy 只作运行期依赖、不进 exe
（该项目已把 ncempy 调用集中在单一模块内，替换面可控）。

## 各工具许可证

| 工具 / 目录 | 许可证 | 说明 |
|---|---|---|
| 根目录及其余工具 | MIT | 本项目原创 |
| `03-应变分析/strainpp-GPA应变分析` | **GPL-3.0-or-later** | 派生自 JJPPeters/Strainpp |
| `01-视频与数据提取/视频切片工具/tools/ffmpeg` | LGPLv3 | BtbN FFmpeg-Builds 共享构建，见该目录 `THIRD_PARTY_NOTICES.md` |
| `02-图像处理/hrtem-HRTEM滤波工具` | MIT（本体） | 算法参考 Kilaas, *J. Microscopy* 190 (1998) 45–51 |
| `02-图像处理/stem-optimize-STEM图像优化` | MIT | 见该目录 `NOTICE.md` |
| `03-应变分析/原子中心识别模型开发` | MIT（本体） | 训练侧依赖 `ultralytics`（AGPL-3.0），打包依赖声明见 `packaging/THIRD_PARTY_NOTICES.txt` |
| `05-EELS分析/EELS边缘价态分析工具` | MIT（本体） | **发行 exe 内含 GPL-3.0 的 ncempy**，见该目录 `THIRD_PARTY_NOTICES.md` |
| `全整合`（TEM Suite） | MIT（本体） | **发行 exe 内含 GPL-3.0 的 ncempy**，且运行期以同进程方式加载 GPL-3.0 的 strainpp，见该目录 `THIRD_PARTY_NOTICES.md` |

### `全整合`（TEM Suite）的模块加载方式

TEM Suite 在运行期**不复制**各工具代码，按 `registry.py` 中每个工具的
`run_mode` 分别处理，两种方式在著作权意义上并不等同：

| 加载方式 | 说明 | 涉及 GPL 时的含义 |
|---|---|---|
| `embed`（默认） | 在同一进程内 `import` 并 `preload` 目标工具的模块 | 与 TEM Suite 构成**同一程序**，GPL-3.0 的传染性判定需按「单一作品」处理 |
| `subprocess` | 用 `sys.executable -m <module>` 另起进程 | 相对独立，但若一并打包进同一 exe，仍属同一分发物 |

**`03-应变分析/strainpp-GPA应变分析`（GPL-3.0-or-later）走的是 `embed` 路径**
（`registry.py` 中该条目带 `preload=("gpa","phase","utils","dm_reader","strain_analysis")`），
即其模块被 `import` 进 TEM Suite 的解释器。因此若要把它与 TEM Suite 一起
打包成单一可执行文件对外分发，请先确认 GPL 合规性；
若要放弃传染性，可把该条目改为 `run_mode="subprocess"`。

## 第三方数据文件

以下数据文件随本仓库分发，来源与许可状态如下：

| 文件 | 位置 | 来源 | 许可状态 |
|---|---|---|---|
| `peng_high.json` | `09-HRTEM模拟/tem_sim/data/`、`010-STEM模拟/stem_sim/data/` | [abTEM](https://github.com/abTEM/abtem) 仓库的同名文件（Peng 1999 参数化） | **GPL-3.0**（abTEM 项目许可证）。两处副本均逐值等同上游文件 |
| `gauss3.txt` | 同上两处 | 提取自 SimulaTEM | **许可是否允许再分发未经确认**，按参考资料对待 |
| `cif/` 下各 `.cif` | `09-HRTEM模拟/cif/`、`010-STEM模拟/cif/` | 由 pymatgen 依据公开晶体学数据生成 | 本项目生成，随本仓库按 MIT 分发 |

上述 abTEM 派生数据文件未在其它地方登记。若需彻底规避，
可改用 Peng 1999 原文参数自行生成，但注意 `gauss3.txt` 的绝对标度相对
Peng 标准系统性偏大（约 1.81–1.92×），换表涉及重新标定。

## 第三方 Python 包

各工具依赖的第三方 Python 包保留各自上游许可证，分发前请分别核对。
本仓库实际使用的主要第三方包包括：

`numpy`、`scipy`、`matplotlib`、`tifffile`、`imagecodecs`、`defusedxml`、
`opencv-python`（`cv2`）、`Pillow`、`scikit-image`、`ttkbootstrap`、`pandas`、
`seaborn`、`ase`、`pyfftw`、`tkinterdnd2`、`PyYAML`、`ncempy`、`torch`、
`onnx`、`onnxruntime`、`ultralytics`、`abtem`、`imageio-ffmpeg`、`pytest`。

> 注：本仓库**不使用** PyQt/PySide。全部 GUI 基于标准库 `tkinter`
> （部分工具使用 `ttkbootstrap` 主题）。个别 PyInstaller `.spec` 的
> `excludes` 列表中出现 `PyQt5`/`PySide6` 字样，那是**排除**声明，不是依赖。

## 未收录的第三方材料

以下材料因未获再分发授权，**不在本仓库中**：

| 材料 | 作者 |
|---|---|
| DigitalMicrograph 脚本 `HRTEM Filter.s`、`去噪音插件.s`/`.txt` | D. R. G. Mitchell |
| ImageJ 插件 `Radial_Profile_Angle.jar` | Philippe Carl |
| 选区电子衍射教程 PDF | 浙江大学 王勇 |

它们仍可从各自原始来源获取。详见 `02-图像处理/hrtem-HRTEM滤波工具/legacy/README.md`
与该工具根目录的 `THIRD_PARTY_NOTICES.md`。

`02-图像处理/hrtem-HRTEM滤波工具/legacy/imagej/FFT for image stack.ijm`
无作者与版权声明、内容为通用 ImageJ 批处理命令，其来源未经确认，
按参考资料对待，不作为可再分发组件。

## 已移除的未授权材料

`09-HRTEM模拟/cif/Fe_bcc.cif` 原为 ICSD 数据库导出文件
（`data_53802-ICSD`，含 FIZ Karlsruhe / 美国商务部 "All rights reserved" 声明），
随 MIT 仓库公开分发缺乏依据，**已替换**为按公开晶体学数据重新生成的等价文件
（与同目录其余 `.cif` 一致，均为 pymatgen 风格输出）。
