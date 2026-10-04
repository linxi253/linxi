# 09-HRTEM模拟 源码与验证导航

基于 tem_sim 多层法的 HRTEM 模拟、显示及系列导出。

**状态：现有实现。** 公开仓库内的工具子项目，使用独立 `.venv`。 工作约定见 [AGENTS.md](AGENTS.md)，下面命令从本项目目录执行。

## 入口与版本

- 入口文件：[hrtem_tool/main.py](hrtem_tool/main.py)、[run.bat](run.bat)。
- 版本来源：[hrtem_tool/__init__.py](hrtem_tool/__init__.py) 的 __version__
- 锁文件：[requirements.lock.txt](requirements.lock.txt)
- 构建配置：本批未登记打包配置，不推断已有成品与源码一致。

## 按任务定位

| 任务 | 文件 | 关键符号 |
|---|---|---|
| 启动 | [hrtem_tool/main.py](hrtem_tool/main.py) | main |
| GUI | [hrtem_tool/gui.py](hrtem_tool/gui.py) | HRTEMApp |
| 模拟参数 | [hrtem_tool/params.py](hrtem_tool/params.py) | SimParams |
| 结构与模拟调度 | [hrtem_tool/sim_core.py](hrtem_tool/sim_core.py) | load_structure |
| 散射因子表 | [tem_sim/scattering.py](tem_sim/scattering.py) | load_peng、load_gauss3 |
| 多层传播 | [tem_sim/multislice.py](tem_sim/multislice.py) | ExitWave |
| 成像光学 | [tem_sim/imaging.py](tem_sim/imaging.py) | aberration_phase |
| 物理数组与显示导出 | [hrtem_tool/export.py](hrtem_tool/export.py) | export_npy、export_tiff16 |
| 系列计算 | [hrtem_tool/series.py](hrtem_tool/series.py) | run_series |

## 保持的约束

- 当前默认 Peng 标准表与 gauss3 legacy 复现口径不同。旧 NCC 验收不能作为当前默认物理标度的证明。
- 波长、相互作用常数、势相位标度、离焦和球差单位是科学接口，不因合并相似代码改变。
- 显示归一化/PNG/16 位显示导出与原始物理强度 NPY 分开；系列参数与元数据保留。

## 环境与验证

- **解释器**：`.venv/Scripts/python.exe`（本项目独立 venv；清单登记 Python >=3.10，本机母本 3.10）。
- **安装锁**：`requirements.lock.txt`；按锁安装，不要用 `pip install -r requirements.txt` 代替。
- **建环境**：参考仓库根 [环境隔离说明](../环境隔离说明.md) 与 [.tools/projects.json](../.tools/projects.json)；常规 `provision-envs.py` 只读锁、`verify-envs.py` 只验证不安装。
- **导入自检**：`numpy`、`scipy`、`matplotlib`、`PIL`、`tifffile`、`ase`、`ttkbootstrap`。

从**本项目目录**执行：

~~~powershell
# pytest
& ./.venv/Scripts/python.exe -m pytest -q -rs
# verify_physics
& ./.venv/Scripts/python.exe tests/verify_physics.py
~~~

旧报告不代表当前通过；运行后按实际结果记录 pass / skip / 未运行。
