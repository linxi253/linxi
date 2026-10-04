# 09-HRTEM模拟 开发约定

基于 tem_sim 多层法的 HRTEM 模拟、显示及系列导出。 **状态：现有实现。** 公开仓库内的工具子项目，使用独立 `.venv`。

按任务查看 [源码与验证导航](DEVELOPMENT.md)，使用与科学说明见 [README](README.md)。

## 修改边界

- 保留已有修改与未跟踪文件；可读性整理只补导航，不改算法、路径、输出或依赖。
- 使用 .venv/Scripts/python.exe；锁文件与测试前置条件见开发说明，缺失时记录未验证，不回退全局 Python。
- 原始数据、发布成品、权重及历史验证材料保持原样。只读导航无需安装、重打包或执行业务模块。

## 关键行为

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
