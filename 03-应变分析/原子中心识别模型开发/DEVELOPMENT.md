# 原子中心识别模型开发 源码与验证导航

原子中心模型的数据治理、标注、训练、评估及部署；正式科研模型仍需真实数据验收。

**状态：现有实现。** 公开仓库内的工具子项目，使用独立 `.venv`。 工作约定见 [AGENTS.md](AGENTS.md)，下面命令从本项目目录执行。

## 入口与版本

- 入口文件：[src/atom_center/annotator_gui.py](src/atom_center/annotator_gui.py)、[src/atom_center/cli.py](src/atom_center/cli.py)、[packaging/annotator_entry.py](packaging/annotator_entry.py)。
- 版本来源：[pyproject.toml](pyproject.toml) 的 project.version；[src/atom_center/__init__.py](src/atom_center/__init__.py) 的 __version__
- 锁文件：[requirements/training-win-py310.lock](requirements/training-win-py310.lock)、[requirements/inference-win-py310.lock](requirements/inference-win-py310.lock)
- 构建配置：本批未登记打包配置，不推断已有成品与源码一致。

## 按任务定位

| 任务 | 文件 | 关键符号 |
|---|---|---|
| 标注 GUI | [src/atom_center/annotator_gui.py](src/atom_center/annotator_gui.py) | main |
| 训练/数据/评估命令分发 | [src/atom_center/cli.py](src/atom_center/cli.py) | main |
| 数据迁移、审计与冻结 | [src/atom_center/data_workflow.py](src/atom_center/data_workflow.py) | audit_projects、migrate_project |
| 按来源分组划分 | [src/atom_center/splitting.py](src/atom_center/splitting.py) | grouped_split_records |
| 模型身份与发布契约 | [src/atom_center/model_manifest.py](src/atom_center/model_manifest.py) | ModelManifest、verify_model_bundle |
| 分块检测管线 | [src/atom_center/pipeline.py](src/atom_center/pipeline.py) | DetectionPipeline、PipelineConfig |
| PyTorch/ONNX 实际后端 | [src/atom_center/backends.py](src/atom_center/backends.py) | TorchBackend、OnnxBackend |
| 点匹配指标 | [src/atom_center/metrics.py](src/atom_center/metrics.py) | PointMetrics、PointMatchResult |
| 训练溯源、预检查和运行锁 | [src/atom_center/training.py](src/atom_center/training.py) | preflight、run_lock |
| 导出与后端对齐 | [src/atom_center/deployment.py](src/atom_center/deployment.py) | compare_backends、export_run |

## 保持的约束

- 标注坐标、ROI 完整性、草稿/审核状态和原图身份是训练数据契约；自动保存与来源校验不能省略。
- 按原图/实验来源分组再做增强，避免同一视野跨训练/验证/测试泄漏。
- 模型 manifest、SHA-256、像素坐标变换、点匹配与 PyTorch/ONNX 对齐必须成套验证。
- 训练 runs、发布权重和真实标注数据有复现用途；软件/合成数据验收不等于真实科研模型验收。
- TEM Suite 仅以子进程启动标注 GUI；PPA 稳定版的深度模型接入仍是另一项工作。

## 环境与验证

- **解释器**：`.venv/Scripts/python.exe`（本项目独立 venv；清单登记 Python `>=3.10,<3.11`）。
- **安装锁**：`requirements/inference-win-py310.lock`；按锁安装，不要用 `pip install -r requirements.txt` 代替。
- **建环境**：参考仓库根 [环境隔离说明](../../环境隔离说明.md) 与 [.tools/projects.json](../../.tools/projects.json)；常规 `provision-envs.py` 只读锁、`verify-envs.py` 只验证不安装。
- **可编辑安装**：由 `.tools/provision-envs.py` 按清单的 `editable` 执行（`-e . --no-deps`，包 `atom_center`，源码在 `src/`），并做 `.pth` 规范化；若需手工复核，请用**本项目解释器** `.venv/Scripts/python.exe -m pip install -e . --no-deps` 后确认 `.pth` 编码与 `atom_center` 导入正常。
- **导入自检**：`numpy`、`yaml`、`PIL`、`scipy`、`tifffile`。
- **注意**：requires-python >=3.10,<3.11，母本固定 3.10。用 inference-win-py310.lock（推理依赖，不含 torch 训练栈）+ editable --no-deps；训练锁 training-win-py310.lock / lock-win-cu128.txt 本轮不安装。packaged_smoke 依赖未收录的字体资产，按存在性跳过。

从**本项目目录**执行：

~~~powershell
# pytest
& ./.venv/Scripts/python.exe -m pytest -q -rs
~~~

旧报告不代表当前通过；运行后按实际结果记录 pass / skip / 未运行。
