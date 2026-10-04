# 原子识别纯算法 源码与验证导航

传统原子识别、逐帧人工校对、积分强度与跨帧 ID。

**状态：现有实现。** 公开仓库内的工具子项目，使用独立 `.venv`。 工作约定见 [AGENTS.md](AGENTS.md)，下面命令从本项目目录执行。

## 入口与版本

- 入口文件：[atomic_app.py](atomic_app.py)。
- 版本来源：[pyproject.toml](pyproject.toml) 的 project.version
- 锁文件：[requirements.lock.txt](requirements.lock.txt)
- 构建配置：[原子识别与强度分析工具.spec](原子识别与强度分析工具.spec)

## 按任务定位

| 任务 | 文件 | 关键符号 |
|---|---|---|
| GUI、后台任务、会话与导出 | [atomic_app.py](atomic_app.py) | AtomicRecognitionApp、main |
| 检测、原始强度积分与关联 | [atomic_core.py](atomic_core.py) | DetectionParams、IntensityParams、detect_atoms、measure_integrated_intensities、link_frame_points、render_marked_frame |

## 保持的约束

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
