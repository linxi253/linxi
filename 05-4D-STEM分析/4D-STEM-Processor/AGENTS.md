# 4D-STEM-Processor 开发约定

DM4 四维扫描数据的 DPC、SSB、应变/取向、峰对与 ePIE。 **状态：现有实现。** 公开仓库内的工具子项目，使用独立 `.venv`。

按任务查看 [源码与验证导航](DEVELOPMENT.md)，使用与科学说明见 [README](README.md)。

## 修改边界

- 保留已有修改与未跟踪文件；可读性整理只补导航，不改算法、路径、输出或依赖。
- 使用 .venv/Scripts/python.exe；锁文件与测试前置条件见开发说明，缺失时记录未验证，不回退全局 Python。
- 原始数据、发布成品、权重及历史验证材料保持原样。只读导航无需安装、重打包或执行业务模块。

## 关键行为

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
