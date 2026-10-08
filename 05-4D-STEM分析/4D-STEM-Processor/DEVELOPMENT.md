# 4D-STEM-Processor 源码与验证导航

DM4 四维扫描数据的 DPC、SSB、应变/取向、峰对与 ePIE。

**状态：现有实现。** 公开仓库内的工具子项目，使用独立 `.venv`。 工作约定见 [AGENTS.md](AGENTS.md)，下面命令从本项目目录执行。

## 入口与版本

- 入口文件：[stem_processor_gui.py](stem_processor_gui.py)。
- 版本来源：[core/__init__.py](core/__init__.py) 的 __version__
- 锁文件：[requirements.lock.txt](requirements.lock.txt)
- 构建配置：[4D-STEM_Processor.spec](4D-STEM_Processor.spec)

## 按任务定位

| 任务 | 文件 | 关键符号 |
|---|---|---|
| 界面、任务和结果图 | [stem_processor_gui.py](stem_processor_gui.py) | STEMProcessorApp |
| DM4 头部、dtype 与数据提取 | [core/dm4_io.py](core/dm4_io.py) | read_dm4_metadata、extract_4d_data |
| 扫描/衍射维度核实 | [core/dimension_utils.py](core/dimension_utils.py) | verify_dimensions、fix_dimensions |
| 共享预处理 | [core/pipeline.py](core/pipeline.py) | prepare_dataset |
| CoM、DPC/iDPC | [core/dpc_core.py](core/dpc_core.py) | compute_com_robust、idpc_reconstruct |
| 单边带重建 | [core/ssb_core.py](core/ssb_core.py) | ssb_reconstruct |
| 衍射盘和应变 | [core/strain_mapping.py](core/strain_mapping.py) | detect_bragg_disks、strain_from_gradient |
| 晶体及倒易格 | [core/orientation_mapping.py](core/orientation_mapping.py) | lattice_matrix、reciprocal_matrix |
| ePIE | [core/ptychography.py](core/ptychography.py) | epie_reconstruct |

## 保持的约束

- 扫描维与衍射维、DM4 dtype、中心/会聚盘半径及单位要一起核实，不能靠数组尺寸猜物理轴。
- 不同科学模块的假设与开关分开，预览重建不等于全部模块均经真实数据验证。
- 取消、输出目录避让、参数及质量信息保留；processing/diagnostics 是专项脚本，含样例约定，不能代替默认 GUI 管线。
- core 顶层名容易与其他工具冲突；TEM Suite 的 preload/加载器必须与实际模块对应。

## 环境与验证

- **解释器**：`.venv/Scripts/python.exe`（本项目独立 venv；清单登记 Python >=3.10，本机母本 3.10）。
- **安装锁**：`requirements.resolved-win-py310.lock.txt`；按锁安装，不要用 `pip install -r requirements.txt` 代替。
- **建环境**：参考仓库根 [环境隔离说明](../../环境隔离说明.md) 与 [.tools/projects.json](../../.tools/projects.json)；常规 `provision-envs.py` 只读锁、`verify-envs.py` 只验证不安装。
- **导入自检**：`numpy`、`scipy`、`matplotlib`、`ncempy`。

从**本项目目录**执行：

~~~powershell
# pytest
& ./.venv/Scripts/python.exe -m pytest -q -rs
~~~

旧报告不代表当前通过；运行后按实际结果记录 pass / skip / 未运行。
