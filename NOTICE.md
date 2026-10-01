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

`全整合/`（TEM Suite）在运行期以子进程/按需加载方式调用各工具，不复制其代码；
若要把它与 strainpp 一起打成单一可执行文件对外分发，请先确认 GPL 合规性。

## 各工具许可证

| 工具 / 目录 | 许可证 | 说明 |
|---|---|---|
| 根目录及其余工具 | MIT | 本项目原创 |
| `03-应变分析/strainpp-GPA应变分析` | **GPL-3.0-or-later** | 派生自 JJPPeters/Strainpp |
| `01-视频与数据提取/视频切片工具/tools/ffmpeg` | LGPLv3 | BtbN FFmpeg-Builds 共享构建，见该目录 `THIRD_PARTY_NOTICES.md` |
| `02-图像处理/hrtem-HRTEM滤波工具` | MIT（本体） | 算法参考 Kilaas, *J. Microscopy* 190 (1998) 45–51 |
| `02-图像处理/stem-optimize-STEM图像优化` | MIT | 见 `NOTICE.md` |
| `03-应变分析/原子中心识别模型开发` | MIT | 打包依赖声明见 `packaging/THIRD_PARTY_NOTICES.txt` |

各工具依赖的第三方 Python 包（PySide/PyQt、numpy、tifffile、scikit-image、matplotlib 等）
保留各自上游许可证，分发前请分别核对。

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
