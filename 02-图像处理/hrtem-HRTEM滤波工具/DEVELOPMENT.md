# hrtem-HRTEM滤波工具 源码与验证导航

HRTEM/STEM 频域滤波与有溯源的定量 TIFF 输出。

**状态：现有实现。** 公开仓库内的工具子项目，使用独立 `.venv`。 工作约定见 [AGENTS.md](AGENTS.md)，下面命令从本项目目录执行。

## 入口与版本

- 入口文件：[butter.py](butter.py)、[hrtem.py](hrtem.py)、[cli_launcher.py](cli_launcher.py)。
- 版本来源：[pyproject.toml](pyproject.toml) 的 project.version；[src/hrtem_filter/_version.py](src/hrtem_filter/_version.py) 的 __version__
- 锁文件：[requirements.lock.txt](requirements.lock.txt)
- 构建配置：[HRTEM_Filter_GUI.spec](HRTEM_Filter_GUI.spec)、[HRTEM_Filter_CLI.spec](HRTEM_Filter_CLI.spec)

## 按任务定位

| 任务 | 文件 | 关键符号 |
|---|---|---|
| 兼容门面与外部工具复用入口 | [butter.py](butter.py) | HRTEMFilter、main |
| 滤波算法与结果 | [src/hrtem_filter/core.py](src/hrtem_filter/core.py) | HRTEMFilter、FilterResult |
| 参数和输出编码 | [src/hrtem_filter/params.py](src/hrtem_filter/params.py) | FilterParams、SaveOptions |
| ROI、补边及裁剪 | [src/hrtem_filter/geometry.py](src/hrtem_filter/geometry.py) | Roi、validate_roi、extract_and_pad |
| 堆栈处理、取消与输入保护 | [src/hrtem_filter/pipeline.py](src/hrtem_filter/pipeline.py) | StackProcessor |
| 格式、标定与安全路径 | [src/hrtem_filter/tiff_io.py](src/hrtem_filter/tiff_io.py) | inspect_tiff |
| 参数记录与恢复 | [src/hrtem_filter/provenance.py](src/hrtem_filter/provenance.py) | build_provenance、write_json_atomically |
| GUI | [src/hrtem_filter/gui.py](src/hrtem_filter/gui.py) | HRTEMFilterGUI |

## 保持的约束

- 默认 float32 定量输出不逐帧归一化；uint8-display 是显示用途，不能用其强度做定量比较。
- 矩形图像/ROI 补边后裁回原始尺寸，TIFF 标定逐页保留。
- 输入保护、取消和临时文件发布保持；TIFF 与 JSON 的两文件发布仍有 README 所述硬崩溃窗口，不能宣称目录级事务。
- butter.py 是 PPA 等工具的兼容入口，不能当作重复代码直接删除。

## 环境与验证

- **解释器**：`.venv/Scripts/python.exe`（本项目独立 venv；清单登记 Python `>=3.10,<3.15`）。
- **安装锁**：`requirements.lock.txt`；按锁安装，不要用 `pip install -r requirements.txt` 代替。
- **建环境**：参考仓库根 [环境隔离说明](../../环境隔离说明.md) 与 [.tools/projects.json](../../.tools/projects.json)；常规 `provision-envs.py` 只读锁、`verify-envs.py` 只验证不安装。
- **导入自检**：`numpy`、`scipy`、`tifffile`、`imagecodecs`、`matplotlib`。

从**本项目目录**执行：

~~~powershell
# pytest
& ./.venv/Scripts/python.exe -m pytest -q -rs
~~~

旧报告不代表当前通过；运行后按实际结果记录 pass / skip / 未运行。
