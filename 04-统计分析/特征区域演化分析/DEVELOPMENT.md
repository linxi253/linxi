# 特征区域演化分析 源码与验证导航

HAADF-STEM 时间序列的背景归一化、分割与面积/强度演化分析。

**状态：现有实现。** 公开仓库内的工具子项目，使用独立 `.venv`。 工作约定见 [AGENTS.md](AGENTS.md)，下面命令从本项目目录执行。

## 入口与版本

- 入口文件：[应力面积统计.py](应力面积统计.py)。
- 版本来源：未登记统一静态版本源；按 README、源码和构建记录分别核实，不能以旧 EXE 文件名推断。
- 锁文件：[requirements.lock.txt](requirements.lock.txt)
- 构建配置：[特征区域演化分析.spec](特征区域演化分析.spec)

## 按任务定位

| 任务 | 文件 | 关键符号 |
|---|---|---|
| 配置、命令行、分析与图表 | [应力面积统计.py](应力面积统计.py) | AnalysisConfig |
| 统计、平滑及无效值处理 | [stress_core.py](stress_core.py) | safe_corrcoef、safe_cov、safe_divide |

## 保持的约束

- 文件名含“应力”，但输出是特征区域面积/归一化柱强度及统计，不能直接当作材料应力。
- 逐帧 Otsu 与 shared-threshold 是不同分析口径；背景归一化、形态学参数与帧间隔必须进入记录。
- 输入 SHA-256、非空输出目录保护、坏像素/无效帧与不确定度字段属于结果契约。

## 环境与验证

- **解释器**：`.venv/Scripts/python.exe`（本项目独立 venv；清单登记 Python >=3.10，本机母本 3.10）。
- **安装锁**：`requirements.lock.txt`；按锁安装，不要用 `pip install -r requirements.txt` 代替。
- **建环境**：参考仓库根 [环境隔离说明](../../环境隔离说明.md) 与 [.tools/projects.json](../../.tools/projects.json)；常规 `provision-envs.py` 只读锁、`verify-envs.py` 只验证不安装。
- **导入自检**：`numpy`、`scipy`、`matplotlib`、`tifffile`、`seaborn`、`skimage`。

从**本项目目录**执行：

~~~powershell
# pytest
& ./.venv/Scripts/python.exe -m pytest -q -rs
~~~

旧报告不代表当前通过；运行后按实际结果记录 pass / skip / 未运行。
