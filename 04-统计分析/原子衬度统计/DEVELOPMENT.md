# 原子衬度统计 源码与验证导航

TIFF 上多个矩形 ROI 的逐帧衬度与完整曲线导出。

**状态：现有实现。** 公开仓库内的工具子项目，使用独立 `.venv`。 工作约定见 [AGENTS.md](AGENTS.md)，下面命令从本项目目录执行。

## 入口与版本

- 入口文件：[tif图像衬度分析工具.py](tif图像衬度分析工具.py)。
- 版本来源：[contrast_core.py](contrast_core.py) 的 __version__
- 锁文件：[requirements.lock.txt](requirements.lock.txt)
- 构建配置：[衬度分析v2.spec](衬度分析v2.spec)

## 按任务定位

| 任务 | 文件 | 关键符号 |
|---|---|---|
| GUI、加载代次与后台分析 | [tif图像衬度分析工具.py](tif图像衬度分析工具.py) | TIFContrastAnalyzer |
| 衬度、布局与像素计算 | [contrast_core.py](contrast_core.py) | compute_contrast、normalize_tiff_array |
| 桌面交互回归 | [smoke_test_gui.py](smoke_test_gui.py) | pump |

## 保持的约束

- ROI 下界 floor、上界 ceil，末端为开区间；坐标和真实像素数按同一口径计算。
- 锁定显示范围不改变原始强度；图上超长曲线抽稀，但 CSV 必须保留完整数据。
- 压缩 TIFF 从 memmap 回退整卷读取时保持内存提示；RGB/平面堆叠解释模式与标定不能静默改变。
- 分析结果使用启动时参数快照，取消、失败回滚与 ROI 状态同步保持。

## 环境与验证

- **解释器**：`.venv/Scripts/python.exe`（本项目独立 venv；清单登记 Python >=3.10，本机母本 3.10）。
- **安装锁**：`requirements.lock.txt`；按锁安装，不要用 `pip install -r requirements.txt` 代替。
- **建环境**：参考仓库根 [环境隔离说明](../../环境隔离说明.md) 与 [.tools/projects.json](../../.tools/projects.json)；常规 `provision-envs.py` 只读锁、`verify-envs.py` 只验证不安装。
- **导入自检**：`numpy`、`matplotlib`、`tifffile`、`ttkbootstrap`、`PIL`。

从**本项目目录**执行：

~~~powershell
# pytest
& ./.venv/Scripts/python.exe -m pytest -q -rs
~~~

旧报告不代表当前通过；运行后按实际结果记录 pass / skip / 未运行。
