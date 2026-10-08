# strainpp-GPA应变分析 源码与验证导航

Bragg 峰与参考区约束下的 GPA 位移、应变和批量分析。

**状态：现有实现。** 公开仓库内的工具子项目，使用独立 `.venv`。 工作约定见 [AGENTS.md](AGENTS.md)，下面命令从本项目目录执行。

## 入口与版本

- 入口文件：[run.py](run.py)、[strain_analysis.py](strain_analysis.py)、[strain_batch.py](strain_batch.py)。
- 版本来源：[_version.py](_version.py) 的 __version__
- 锁文件：[requirements-lock.txt](requirements-lock.txt)
- 构建配置：[Strain++GPA.spec](Strain++GPA.spec)

## 按任务定位

| 任务 | 文件 | 关键符号 |
|---|---|---|
| GUI 启动与依赖检查 | [run.py](run.py) | main |
| GUI、参考区与批处理对话框 | [strain_gui.py](strain_gui.py) | StrainGUI、BatchSettingsDialog |
| GPA 数值管线 | [strainpp_gpa/gpa.py](strainpp_gpa/gpa.py) | GPA、GPAOutput |
| 几何相位与取消 | [strainpp_gpa/phase.py](strainpp_gpa/phase.py) | Phase、ComputationCancelled |
| 逐帧批处理 | [strainpp_gpa/batch.py](strainpp_gpa/batch.py) | BatchConfig、run_batch |
| 堆栈/序列读取 | [strainpp_gpa/stack_reader.py](strainpp_gpa/stack_reader.py) | read_stack_info、open_frame_source |
| DM 格式与标定 | [strainpp_gpa/dm_reader.py](strainpp_gpa/dm_reader.py) | read_dm_file |
| 单图 CLI | [strain_analysis.py](strain_analysis.py) | main |
| 批处理 CLI | [strain_batch.py](strain_batch.py) | main |

## 保持的约束

- Bragg 峰、掩膜和参考区定义物理参考态，软件不能自动保证无应变区域。
- 质量掩膜和 NaN 用于定量输出；浮点 TIFF/NPY 与 PNG 预览分开，不能用显示灰度替代应变值。
- 批量帧顺序、标定、输出前缀防覆盖与元数据保留；取消的旧任务不得覆盖新结果。
- 顶层 gpa.py/phase.py 等是兼容门面，核心在 strainpp_gpa；调整导入前核查外部调用。

## 环境与验证

- **解释器**：`.venv/Scripts/python.exe`（本项目独立 venv；清单登记 Python >=3.10，本机母本 3.10）。
- **安装锁**：`requirements-lock.txt`；按锁安装，不要用 `pip install -r requirements.txt` 代替。
- **建环境**：参考仓库根 [环境隔离说明](../../环境隔离说明.md) 与 [.tools/projects.json](../../.tools/projects.json)；常规 `provision-envs.py` 只读锁、`verify-envs.py` 只验证不安装。
- **导入自检**：`numpy`、`scipy`、`matplotlib`、`tifffile`、`ttkbootstrap`、`ncempy`。

从**本项目目录**执行：

~~~powershell
# pytest
& ./.venv/Scripts/python.exe -m pytest -q -rs
~~~

旧报告不代表当前通过；运行后按实际结果记录 pass / skip / 未运行。
