# 离域效应去除工具 源码与验证导航

按手绘真实晶体边界抑制晶体外离域条纹的单图工具。

**状态：现有实现。** 公开仓库内的工具子项目，使用独立 `.venv`。 工作约定见 [AGENTS.md](AGENTS.md)，下面命令从本项目目录执行。

## 入口与版本

- 入口文件：[main.py](main.py)。
- 版本来源：[version.py](version.py) 的 __version__
- 锁文件：[requirements.lock.txt](requirements.lock.txt)、[requirements-dev.lock.txt](requirements-dev.lock.txt)
- 构建配置：[DelocCleaner.spec](DelocCleaner.spec)、[build.bat](build.bat)

## 按任务定位

| 任务 | 文件 | 关键符号 |
|---|---|---|
| GUI 与错误呈现 | [main.py](main.py) | DelocApp、main |
| 频带与预览/处理算法 | [deloc_core.py](deloc_core.py) | radial_frequency、preview_resample |
| 数值参数校验 | [parameters.py](parameters.py) | Parameters、finite_number |
| 保留/挖除区域模型 | [roi_model.py](roi_model.py) | RoiShape、RoiSet |
| TIFF 检查、加载和命名 | [tif_io.py](tif_io.py) | inspect、load、make_output_path |
| 源文件保护和安全写入 | [export_io.py](export_io.py) | protect_source、atomic_path、write_json |

## 保持的约束

- 掩膜表示保留晶格的区域，至少存在一块保留区；未圈到的晶体会按外部处理。
- 频带单位为 cyc/px，换数据集应重新选择；预览缩放与全分辨率频率口径不能混淆。
- 保持源文件保护、位深与标定元数据；抑制虚影不能被描述为恢复真实结构或定量反演。

## 环境与验证

- **解释器**：`.venv/Scripts/python.exe`（本项目独立 venv；清单登记 Python >=3.10，本机母本 3.10）。
- **安装锁**：`requirements.lock.txt`；按锁安装，不要用 `pip install -r requirements.txt` 代替。
- **建环境**：参考仓库根 [环境隔离说明](../../环境隔离说明.md) 与 [.tools/projects.json](../../.tools/projects.json)；常规 `provision-envs.py` 只读锁、`verify-envs.py` 只验证不安装。
- **导入自检**：`numpy`、`scipy`、`matplotlib`、`tifffile`、`PIL`、`imagecodecs`。

从**本项目目录**执行：

~~~powershell
# pytest
& ./.venv/Scripts/python.exe -m pytest -q -rs
~~~

旧报告不代表当前通过；运行后按实际结果记录 pass / skip / 未运行。
