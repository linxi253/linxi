# PPA 开发约定

这是公开仓库内的工具子项目，使用独立 `.venv`。主 GUI 为 [ppa.py](ppa.py)，统计 GUI 为 [ppa_stats.py](ppa_stats.py)，数值与存储模块在 ppa_core。按功能定位见 [DEVELOPMENT.md](DEVELOPMENT.md)，科学定义见 [README.md](README.md)。

## 工作边界

- 保留当前工作区已有修改及未跟踪文件。可读性整理优先增加导航，不按文件大小直接拆分 ppa.py。
- 使用 .venv/Scripts/python.exe 和 [requirements.lock.txt](requirements.lock.txt)。环境缺失时说明未验证，不调用全局环境运行项目。
- 现有测试基于 unittest；不要为了执行这些测试额外引入 pytest。
- .spec 名字沿用历史命名。源码、构建配置与旧 EXE 的版本应分别核实，不以文件名推断当前版本。

## 必须保留的行为

- 工具计算相对位移、应变与旋转，不能直接称为应力结果。
- 数值核心使用物理坐标 y 向上，GUI 使用图像坐标 y 向下；显示变换与 CSV 物理定义分开处理。
- 张量剪应变与工程剪应变相差两倍；小应变、Green–Lagrange 应变、旋转和等效应变定义不得在整理时改变。
- 默认局部 Peak Pairs 与兼容晶格-CST 的结果位置不同，不能混合统计；几何不足保留 NaN 和位移，质量标记不能变成静默删除原子。
- 保留项目文件的原图身份校验、旧版本导入规则、CSV 元数据及 BOM 兼容。
- 取消、点表修改后的任务失效和关闭回调属于正确性约束，不能作为无用状态清理。
- TEM Suite 调用 ppa.AtomMarkerApp(root)。可选滤波器通过 PPA_HRTEM_FILTER_DIR 或工作区定位；稳定版不加载深度模型。

## 按改动选择验证

- 核心计算/存储：tests/test_core.py。
- 阈值、导出与历史修复：tests/test_fix_regressions.py。
- 点表、后台流程和界面：tests/test_gui_workflow.py，需要 Tk 显示环境。
- 改入口后加 TEM Suite 的 ppa_strain 冒烟检查，具体命令见开发说明。
- 纯文档只检查导航和差异；环境缺失、GUI 跳过与功能通过分别记录。

## 环境与验证

- **解释器**：`.venv/Scripts/python.exe`（本项目独立 venv；清单登记 Python >=3.10，本机母本 3.10）。
- **安装锁**：`requirements.lock.txt`；按锁安装，不要用 `pip install -r requirements.txt` 代替。
- **建环境**：参考仓库根 [环境隔离说明](../../环境隔离说明.md) 与 [.tools/projects.json](../../.tools/projects.json)；常规 `provision-envs.py` 只读锁、`verify-envs.py` 只验证不安装。
- **导入自检**：`numpy`、`scipy`、`matplotlib`、`tifffile`、`PIL`。
- **注意**：tests/ 用标准库 unittest（原独立 92 项证据即 -m unittest discover -s tests -v），这是唯一登记入口：同一批测试不再按 pytest 重复跑一遍，也不要为它安装 pytest。

从**本项目目录**执行：

~~~powershell
# unittest
& ./.venv/Scripts/python.exe -m unittest discover -s tests -v
~~~

旧报告不代表当前通过；运行后按实际结果记录 pass / skip / 未运行。
