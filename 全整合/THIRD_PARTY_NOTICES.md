# 第三方组件声明（TEM Suite）

本工具（TEM Suite 集成层）**本体代码采用 MIT**（见仓库根 [LICENSE](../LICENSE)）。
但它的发行物与运行方式涉及两个 **GPL-3.0** 组件，发行前请先读本节。

## 1. ncempy（GPL-3.0-or-later）— 被打包进发行 exe

| 项 | 内容 |
|---|---|
| 包名 | `ncempy`（版本见 `requirements.lock.txt`） |
| 上游 | https://github.com/ercius/openNCEM |
| 许可证 | **GNU General Public License v3.0 or later** |
| 打包方式 | `TEMSuite.spec` 的 `hiddenimports` 含 `ncempy`、`ncempy.io`，随单文件 exe 一并分发 |

分发该 exe 时须履行 GPL-3.0 §4–§6 义务（提供对应源码、附许可证全文等），
详见 [`05-EELS分析/EELS边缘价态分析工具/THIRD_PARTY_NOTICES.md`](../05-EELS分析/EELS边缘价态分析工具/THIRD_PARTY_NOTICES.md)。

## 2. strainpp（GPL-3.0-or-later）— 以同进程方式加载

`registry.py` 中 `03-应变分析/strainpp-GPA应变分析` 这一条目**未指定 `run_mode`**，
因此走默认的 `embed` 模式，并带有

```python
preload=("gpa", "phase", "utils", "dm_reader", "strain_analysis")
```

即 **strainpp 的模块被 `import` 进 TEM Suite 自己的解释器**，与 TEM Suite
构成同一程序的组成部分——这与「另起进程调用」在著作权意义上并不等同。

### 各工具的实际加载方式（按 `registry.py`）

| 加载方式 | 工具 |
|---|---|
| `subprocess`（`sys.executable -m <module>`，另起进程） | `03-应变分析/原子中心识别模型开发`、`04-统计分析/特征区域演化分析` |
| `embed`（同一进程内 `import` + `preload`） | 其余全部工具，**含 GPL-3.0 的 `03-应变分析/strainpp-GPA应变分析`** |

> 仓库根 `NOTICE.md` 早期版本把调用方式笼统描述为「以子进程/按需加载方式调用」，
> 实际 18 个工具中只有 2 个走子进程。本文件按代码实际行为更正。

### 含义

把 TEM Suite 与 strainpp 一起打包成**单一可执行文件**对外分发时，
应比照「单一作品」处理 GPL-3.0 的传染性，不能沿用「不复制其代码故不受影响」的表述。

若要消除传染性，可把该条目的 `run_mode` 改为 `"subprocess"`——`registry.py`
已支持该模式（`app.py` 中会用 `sys.executable -m` 另起进程加载）。
注意即便改为子进程，若两者仍被打进同一个 exe，依然属于同一分发物。

## 其它第三方 Python 包

`numpy`、`scipy`、`matplotlib`、`tifffile`、`opencv-python`、`ttkbootstrap`、
`ase`、`pyfftw` 等依赖保留各自上游许可证；完整清单见 `requirements.lock.txt`。
