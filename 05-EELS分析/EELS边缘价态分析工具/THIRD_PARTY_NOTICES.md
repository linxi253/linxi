# 第三方组件声明

本工具（EELS 边缘价态分析工具）**本体代码采用 MIT**（见仓库根 [LICENSE](../../LICENSE)）。

但本工具**对外发行的单文件 exe 中打包了 GPL-3.0 的第三方组件**，
发行时该组件仍受其原许可证约束，不因本项目声明 MIT 而改变。

## ncempy（GPL-3.0-or-later）

| 项 | 内容 |
|---|---|
| 包名 / 版本 | `ncempy` 1.16（`requirements.lock.txt` 中钉死） |
| 上游 | https://github.com/ercius/openNCEM |
| 许可证 | **GNU General Public License v3.0 or later** |
| 用途 | 读取 Gatan DM3/DM4 文件（`src/eels_edge_analyzer/dm4io.py` 集中封装其调用） |
| 打包方式 | `EELS_Edge_Analyzer.spec` 中 `datas = collect_data_files("ncempy")`，并把 `ncempy`/`ncempy.io`/`ncempy.io.dm` 列入 `hiddenimports`；单文件模式下随 `a.binaries`/`a.datas` 一并打进 exe |

**分发该 exe 时的 GPL-3.0 义务**（GPL-3.0 §4–§6）：

1. 随发行物提供或书面承诺提供 **ncempy 的完整对应源码**（可指向上述上游仓库的对应版本）；
2. 随发行物附上 **GPL-3.0 许可证全文**；
3. 若对 ncempy 做过修改，须在修改文件中保留修改声明与日期（本仓库未修改 ncempy）；
4. 不得对本发行物施加比 GPL-3.0 更严格的附加限制。

**规避方式（可选）**：`ncempy` 只用于 DM 读取，且调用已集中在
`src/eels_edge_analyzer/dm4io.py` 一个模块内。若不想承担上述义务，
可改为让 ncempy 仅作运行期依赖、不进 exe（例如改用非 GPL 的 DM 读取实现，
或在 spec 中改为从外部环境加载而非打包）。

## 其它第三方 Python 包

`numpy`、`scipy`、`matplotlib`、`tifffile`、`imagecodecs` 等依赖采用
BSD/MIT/PSF 类宽松许可证，保留各自上游许可证；完整清单见该工具的
`requirements.lock.txt`。
