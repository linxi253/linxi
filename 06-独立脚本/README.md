# 独立脚本

这里保存尚未整理为独立项目的历史排障脚本和算法原稿（一次性脚本残留，**不保证可用**），运行前必须先读源码检查。

## 文件说明

### `sqye.py`

批量读取指定文件夹中的 `.tif` 图像，构造 sinc 形式的点扩散函数（PSF），使用 `skimage.restoration.wiener` 进行 Wiener 反卷积并输出 8 位 TIFF。

依赖（**在项目内 venv 中安装**，遵循仓库「一个工具一个 venv」的隔离约定，不要装进 Miniconda base 或 PATH 上的全局 Python）：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install numpy scipy scikit-image
```

运行时同样使用该 venv 的解释器（`.venv\Scripts\python sqye.py`）。运行前先设置环境变量 `SQYE_INPUT_DIR`、`SQYE_OUTPUT_DIR`（输入/输出文件夹路径，缺失即报错退出并打印用法），必要时修改文件顶部的 `saveImage` 和 `nsr`。`saveImage` 非 0 即保存结果。输出被直接转换为 `uint8`，不适合要求保留 16 位定量强度的数据。

### `下载.txt`

原文件名是 `下载.py`，但内容只有 `pip install numpy`，是命令记录而不是有效 Python 程序，
无法通过语法检查。为避免拖挂 CI 的语法检查，已改名为 `下载.txt`，内容不变：

```powershell
python -m pip install numpy
```

不要执行 `python 下载.txt`；上面这行只是当年的命令记录。如需装包，请按上文先建 `.venv`，再用 `.venv\Scripts\python -m pip install ...` 安装。

> 本目录原有的两个第三方滤波脚本原稿（D. R. G. Mitchell 的 HRTEM Filter v4.0 及其 Python 移植版）
> 因版权归属第三方，未收录进本仓库；当前维护实现见 `02-图像处理/hrtem-HRTEM滤波工具/butter.py`。
