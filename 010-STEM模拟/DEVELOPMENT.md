# 010-STEM模拟 源码与验证导航

冻结声子多层法的 STEM 探针扫描与环形探测器强度。

**状态：现有实现。** 本目录未发现 Git 仓库；工作区根目录也不是仓库，修改前保留文件基线。 工作约定见 [AGENTS.md](AGENTS.md)，下面命令从本项目目录执行。

## 入口与版本

- 入口文件：[stem_tool/main.py](stem_tool/main.py)、[run.bat](run.bat)。
- 版本来源：[stem_sim/__init__.py](stem_sim/__init__.py) 的 __version__
- 锁文件：[requirements.lock.txt](requirements.lock.txt)
- 构建配置：本批未登记打包配置，不推断已有成品与源码一致。

## 按任务定位

| 任务 | 文件 | 关键符号 |
|---|---|---|
| 启动 | [stem_tool/main.py](stem_tool/main.py) | main |
| GUI | [stem_tool/gui.py](stem_tool/gui.py) | STEMApp |
| 参数 | [stem_tool/params.py](stem_tool/params.py) | StemParams |
| 模拟计划与内存预算 | [stem_tool/sim_core.py](stem_tool/sim_core.py) | SimPlan、plan |
| 探针与光学 | [stem_sim/probe.py](stem_sim/probe.py) | StemOptics、aberration_chi |
| Debye 热位移 | [stem_sim/phonons.py](stem_sim/phonons.py) | debye_waller_B、rms_displacement |
| 环形探测器 | [stem_sim/detectors.py](stem_sim/detectors.py) | RingDetector、detector_signal |
| 扫描与批次预算 | [stem_sim/scan.py](stem_sim/scan.py) | ScanGeometry、plan_batches |
| 物理/显示输出 | [stem_tool/export.py](stem_tool/export.py) | export_npy、export_tiff16 |

## 保持的约束

- 冻结声子对强度求平均，不对波函数平均；随机性和参数必须可复现。
- 探针归一化、探测器角度/采样自洽、Cs/离焦单位与相位绝对标度一起验证。
- 探测器收集比例的物理强度与显示归一化分开；NPY 与参数 JSON 是定量复现依据。
- CPU 进程数、FFT 线程数及内存预算联合考虑，不能把性能重排当作无行为改动。

## 环境与验证

- **解释器**：`.venv/Scripts/python.exe`（本项目独立 venv；清单登记 Python >=3.10，本机母本 3.10）。
- **安装锁**：`requirements.resolved-win-py310.lock.txt`；按锁安装，不要用 `pip install -r requirements.txt` 代替。
- **建环境**：参考仓库根 [环境隔离说明](../环境隔离说明.md) 与 [.tools/projects.json](../.tools/projects.json)；常规 `provision-envs.py` 只读锁、`verify-envs.py` 只验证不安装。
- **导入自检**：`numpy`、`scipy`、`matplotlib`、`PIL`、`tifffile`、`ase`、`ttkbootstrap`、`pyfftw`。
- **注意**：tests/ 同时有 pytest 套件（含真实 Au 端到端）与 verify_physics.py 物理自检；test_e2e_au.py 已改为真 fixture（原 fixture 'pl' not found 已修），两个入口都要跑。

从**本项目目录**执行：

~~~powershell
# pytest
& ./.venv/Scripts/python.exe -m pytest tests -q -rs
# verify_physics
& ./.venv/Scripts/python.exe tests/verify_physics.py
~~~

旧报告不代表当前通过；运行后按实际结果记录 pass / skip / 未运行。
