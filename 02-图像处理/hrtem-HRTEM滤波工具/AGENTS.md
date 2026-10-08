# hrtem-HRTEM滤波工具 开发约定

HRTEM/STEM 频域滤波与有溯源的定量 TIFF 输出。 **状态：现有实现。** 公开仓库内的工具子项目，使用独立 `.venv`。

按任务查看 [源码与验证导航](DEVELOPMENT.md)，使用与科学说明见 [README](README.md)。

## 修改边界

- 保留已有修改与未跟踪文件；可读性整理只补导航，不改算法、路径、输出或依赖。
- 使用 .venv/Scripts/python.exe；锁文件与测试前置条件见开发说明，缺失时记录未验证，不回退全局 Python。
- 原始数据、发布成品、权重及历史验证材料保持原样。只读导航无需安装、重打包或执行业务模块。

## 关键行为

- 默认 float32 定量输出不逐帧归一化；uint8-display 是显示用途，不能用其强度做定量比较。
- 矩形图像/ROI 补边后裁回原始尺寸，TIFF 标定逐页保留。
- 输入保护、取消和临时文件发布保持；TIFF 与 JSON 的两文件发布仍有 README 所述硬崩溃窗口，不能宣称目录级事务。
- butter.py 是 PPA 等工具的兼容入口，不能当作重复代码直接删除。

## 环境与验证

- **解释器**：`.venv/Scripts/python.exe`（本项目独立 venv；清单登记 Python 3.10–3.15，本机母本 3.10）。
- **安装锁**：`requirements.lock.txt`；按锁安装，不要用 `pip install -r requirements.txt` 代替。
- **建环境**：参考仓库根 [环境隔离说明](../../环境隔离说明.md) 与 [.tools/projects.json](../../.tools/projects.json)；常规 `provision-envs.py` 只读锁、`verify-envs.py` 只验证不安装。
- **导入自检**：`numpy`、`scipy`、`tifffile`、`imagecodecs`、`matplotlib`。

从**本项目目录**执行：

~~~powershell
# pytest
& ./.venv/Scripts/python.exe -m pytest -q -rs
~~~

旧报告不代表当前通过；运行后按实际结果记录 pass / skip / 未运行。
