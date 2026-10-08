# 原子中心识别模型开发 开发约定

原子中心模型的数据治理、标注、训练、评估及部署；正式科研模型仍需真实数据验收。 **状态：现有实现。** 公开仓库内的工具子项目，使用独立 `.venv`。

按任务查看 [源码与验证导航](DEVELOPMENT.md)，使用与科学说明见 [README](README.md)。

## 修改边界

- 保留已有修改与未跟踪文件；可读性整理只补导航，不改算法、路径、输出或依赖。
- 使用 .venv/Scripts/python.exe；锁文件与测试前置条件见开发说明，缺失时记录未验证，不回退全局 Python。
- 原始数据、发布成品、权重及历史验证材料保持原样。只读导航无需安装、重打包或执行业务模块。

## 关键行为

- 标注坐标、ROI 完整性、草稿/审核状态和原图身份是训练数据契约；自动保存与来源校验不能省略。
- 按原图/实验来源分组再做增强，避免同一视野跨训练/验证/测试泄漏。
- 模型 manifest、SHA-256、像素坐标变换、点匹配与 PyTorch/ONNX 对齐必须成套验证。
- 训练 runs、发布权重和真实标注数据有复现用途；软件/合成数据验收不等于真实科研模型验收。
- TEM Suite 仅以子进程启动标注 GUI；PPA 稳定版的深度模型接入仍是另一项工作。

## 环境与验证

- **解释器**：`.venv/Scripts/python.exe`（本项目独立 venv；清单登记 Python 3.10–3.11，本机母本 3.10）。
- **安装锁**：`requirements/inference-win-py310.lock`；按锁安装，不要用 `pip install -r requirements.txt` 代替。
- **建环境**：参考仓库根 [环境隔离说明](../../环境隔离说明.md) 与 [.tools/projects.json](../../.tools/projects.json)；常规 `provision-envs.py` 只读锁、`verify-envs.py` 只验证不安装。
- **可编辑安装**：`pip install -e . --no-deps`（包 `atom_center`，源码在 `src/`）。
- **导入自检**：`numpy`、`yaml`、`PIL`、`scipy`、`tifffile`。
- **注意**：requires-python >=3.10,<3.11，母本固定 3.10。用 inference-win-py310.lock（推理依赖，不含 torch 训练栈）+ editable --no-deps；训练锁 training-win-py310.lock / lock-win-cu128.txt 本轮不安装。packaged_smoke 依赖未收录的字体资产，按存在性跳过。

从**本项目目录**执行：

~~~powershell
# pytest
& ./.venv/Scripts/python.exe -m pytest -q -rs
~~~

旧报告不代表当前通过；运行后按实际结果记录 pass / skip / 未运行。
