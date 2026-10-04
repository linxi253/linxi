# 010-STEM模拟 开发约定

冻结声子多层法的 STEM 探针扫描与环形探测器强度。 **状态：现有实现。** 本目录未发现 Git 仓库；工作区根目录也不是仓库，修改前保留文件基线。

按任务查看 [源码与验证导航](DEVELOPMENT.md)，使用与科学说明见 [README](README.md)。

## 修改边界

- 保留已有修改与未跟踪文件；可读性整理只补导航，不改算法、路径、输出或依赖。
- 使用 .venv/Scripts/python.exe；锁文件与测试前置条件见开发说明，缺失时记录未验证，不回退全局 Python。
- 原始数据、发布成品、权重及历史验证材料保持原样。只读导航无需安装、重打包或执行业务模块。

## 关键行为

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
