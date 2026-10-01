# 独立脚本

这里保存尚未整理为独立项目的脚本和算法原稿。它们包含硬编码路径或不完整命令，运行前必须检查。

## 文件说明

### `sqye.py`

批量读取指定文件夹中的 `.tif` 图像，构造 sinc 形式的点扩散函数（PSF），使用 `skimage.restoration.wiener` 进行 Wiener 反卷积并输出 8 位 TIFF。

依赖：

```powershell
python -m pip install numpy scipy scikit-image
```

运行前修改文件顶部的 `input_folder`、`output_folder`、`saveImage` 和 `nsr`。当前 `saveImage = 5` 仍会被视为真值；若只想控制开关，建议改为 `0` 或 `1`。输出被直接转换为 `uint8`，不适合要求保留 16 位定量强度的数据。

### `下载.txt`

原文件名是 `下载.py`，但内容只有 `pip install numpy`，是命令记录而不是有效 Python 程序，
无法通过语法检查。为避免拖挂 CI 的语法检查，已改名为 `下载.txt`，内容不变：

```powershell
python -m pip install numpy
```

不要执行 `python 下载.txt`；如需装包，请在命令行里直接运行上面这行。

> 本目录原有的两个第三方滤波脚本原稿（D. R. G. Mitchell 的 HRTEM Filter v4.0 及其 Python 移植版）
> 因版权归属第三方，未收录进本仓库；当前维护实现见 `02-图像处理/hrtem-HRTEM滤波工具/butter.py`。
