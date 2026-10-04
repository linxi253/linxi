# 视频切片工具 开发约定

视频解码、采样及 ImageJ 兼容 TIFF 堆栈导出。 **状态：现有实现。** 公开仓库内的工具子项目，使用独立 `.venv`。

按任务查看 [源码与验证导航](DEVELOPMENT.md)，使用与科学说明见 [README](README.md)。

## 修改边界

- 保留已有修改与未跟踪文件；可读性整理只补导航，不改算法、路径、输出或依赖。
- 使用 .venv/Scripts/python.exe；锁文件与测试前置条件见开发说明，缺失时记录未验证，不回退全局 Python。
- 原始数据、发布成品、权重及历史验证材料保持原样。只读导航无需安装、重打包或执行业务模块。

## 关键行为

- 默认输出普通 ImageJ TIFF，不能为了大文件方便静默改成 BigTIFF；保持既有容量/截断策略与告警。
- 保持像素编码方向，不自动套用视频容器旋转元数据；旋转告警、帧时间轴和采样规则都要保留。
- 严格解码、既有输出拒绝覆盖、暂存区发布及输入身份复查共同保护实验数据。
- 当前源码说明的输出格式不同于某些旧流水线中的 OME-TIFF 描述，集成时核实真实 API/输出，不能只按后缀推断。

## 环境与验证

- **解释器**：`.venv/Scripts/python.exe`（本项目独立 venv；清单登记 Python 3.11–3.14，本机母本 3.12）。
- **安装锁**：`requirements.resolved-win-py312.lock.txt`；按锁安装，不要用 `pip install -r requirements.txt` 代替。
- **建环境**：参考仓库根 [环境隔离说明](../../环境隔离说明.md) 与 [.tools/projects.json](../../.tools/projects.json)；常规 `provision-envs.py` 只读锁、`verify-envs.py` 只验证不安装。
- **导入自检**：`numpy`、`tifffile`、`imageio_ffmpeg`。
- **注意**：requires-python >=3.11,<3.14，必须 3.12 母本；CI 另需 FFmpeg 8.x（由 CI 步骤提供）。

从**本项目目录**执行：

~~~powershell
# pytest
& ./.venv/Scripts/python.exe -m pytest -q -rs
~~~

旧报告不代表当前通过；运行后按实际结果记录 pass / skip / 未运行。
