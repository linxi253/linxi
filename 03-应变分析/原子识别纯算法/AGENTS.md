# 原子识别纯算法 开发约定

传统原子识别、逐帧人工校对、积分强度与跨帧 ID。 **状态：现有实现。** 公开仓库内的工具子项目，使用独立 `.venv`。

按任务查看 [源码与验证导航](DEVELOPMENT.md)，使用与科学说明见 [README](README.md)。

## 修改边界

- 保留已有修改与未跟踪文件；可读性整理只补导航，不改算法、路径、输出或依赖。
- 使用 .venv/Scripts/python.exe；锁文件与测试前置条件见开发说明，缺失时记录未验证，不回退全局 Python。
- 原始数据、发布成品、权重及历史验证材料保持原样。只读导航无需安装、重打包或执行业务模块。

## 关键行为

- 1%–99% 显示归一化只服务预览/检测，积分强度使用原始像素。
- ROI 限制自动候选；手工修改、每帧/公共选区、撤销与溯源标志必须保持一致。
- 跨帧 ID 规则、背景环、孔径截断与邻居污染 QC 影响科研解释，不能作为显示细节省略。
- 会话依赖原 TIFF，CSV 与 metadata.json 配套；取消和临时文件发布不得破坏旧输出。

## 环境与验证

- **解释器**：`.venv/Scripts/python.exe`（本项目独立 venv；清单登记 Python >=3.10，本机母本 3.10）。
- **安装锁**：`requirements.lock.txt`；按锁安装，不要用 `pip install -r requirements.txt` 代替。
- **建环境**：参考仓库根 [环境隔离说明](../../环境隔离说明.md) 与 [.tools/projects.json](../../.tools/projects.json)；常规 `provision-envs.py` 只读锁、`verify-envs.py` 只验证不安装。
- **导入自检**：`numpy`、`scipy`、`matplotlib`、`tifffile`、`PIL`。

从**本项目目录**执行：

~~~powershell
# pytest
& ./.venv/Scripts/python.exe -m pytest -q -rs
~~~

旧报告不代表当前通过；运行后按实际结果记录 pass / skip / 未运行。
