# 原子中心识别模型开发

HAADF-STEM 与 HRTEM 原子中心模型的独立开发仓库。训练代码、数据治理和评测在这里完成；PPA 稳定版只消费经过验证、带 manifest 和 SHA-256 的发布模型。

## 当前状态

已完成 M1～M3 的软件建设和合成数据验收，包括数据审计/迁移、独立训练运行、断点续训、真实 PyTorch/ONNX 后端。正式科研模型仍需真实数据验收，PPA GUI 接入属于后续 M4。

操作入口见 [训练与推理使用说明](docs/training-workflow.md)；验收证据见 `reports/M123_DELIVERY_2026-09-06.md`（运行训练/评测后生成的本地产物，未入库）。

现有功能：

- 明确的 `(x, y)` 坐标约定和统一 `DetectionResult`；
- 二维、RGB 与多帧 TIFF 的显式读取规则；
- 按样品/采集批次分组的数据划分；
- ROI、切片坐标恢复和跨切片点合并；
- 明/暗原子均可用的 COM/二维高斯亚像素精修；
- 匈牙利匹配定位指标及应变噪声估算；
- 模型 manifest、SHA-256 校验和发布目录约束；
- 实际 YOLOv8 的训练、续训、导出、PyTorch/ONNX Runtime 一致性检查；
- HAADF-STEM/HRTEM 独立标注项目、递归图像发现和多帧 TIFF 拆帧；
- 精确浮点 `(x, y)` 点、完整标注 ROI、草稿/审核状态和原子化自动保存；
- 采集元数据清单，以及从已审核 ROI 派生可回溯 YOLO 数据的导出器。

## 环境

PowerShell：

```powershell
.\scripts\setup_environment.ps1
.\.venv\Scripts\Activate.ps1
python scripts\check_environment.py
pytest
python scripts\smoke_test.py
```

训练依赖锁定在 `requirements/training-win-py310.lock`，纯 CPU 推理依赖锁定在 `requirements/inference-win-py310.lock`。PyTorch 使用 CUDA 12.8 官方 wheel；部署使用已实测的 ONNX Runtime 1.20.1。

本机原 Miniconda Python 的 ONNX 原生校验器会崩溃；安装脚本准备了项目内官方 CPython 3.10.11 运行时。训练/导出统一运行 `scripts/atom-center.ps1`，标注 GUI 继续使用现有 `.venv` 启动入口。

**版本隔离边界（审计 42）：** 本子项目钉死 CPython 3.10（`requires-python >=3.10,<3.11`）与 numpy 1.26.4 / opencv 4.11；`pyproject.toml` 与 `requirements/` 下的三份锁文件一致，仓库内其余工具则使用 numpy 2.2.6。这是有意的隔离边界，不是遗漏：两组精确钉死的版本不能并入同一份统一依赖，跨进程/跨工具调用必然存在版本落差。若确需统一到 numpy 2.x，必须先解除 `<3.11` 约束，再重新验证训练、ONNX 导出与纯 CPU 推理三条链路，并同步更新 `pyproject.toml` 与全部锁文件。

## 数据原则

1. 原始图像只读保存于 `data/raw/<modality>/`。
2. 原始标签是精确中心点，YOLO 框只由中心点派生。
3. 按 `acquisition_id` 或更严格的实验批次分组，再进行裁剪和增强。
4. HAADF-STEM 与 HRTEM 使用独立数据、配置和权重，但共享代码和评测接口。
5. 盲测集不得使用仅由模拟生成的数据代替真实实验数据。

## 常用命令

```powershell
# 运行核心测试
.\.venv\Scripts\python.exe -m pytest

# 包含 PyTorch -> ONNX -> ONNX Runtime 一致性检查
.\.venv\Scripts\python.exe -m pytest -m ml

# 检查环境/GPU
.\.venv\Scripts\python.exe scripts\check_environment.py

# 无权重端到端冒烟流程
.\.venv\Scripts\python.exe scripts\smoke_test.py

# 启动手工标注工具（也可双击 scripts 下的两个 annotate_*.cmd）
.\scripts\start_annotation.ps1 -Modality haadf_stem
.\scripts\start_annotation.ps1 -Modality hrtem

# 构建无需 Python 的 Windows x64 便携版和单文件版
.\scripts\build_windows_annotator.ps1
```

数据采集数量、目录规范、标注操作和审核规则见 [第二阶段指南](docs/phase-2-data-collection-and-annotation.md)。已审核数据按 [训练流程](docs/training-workflow.md) 审计和冻结后，即可启动独立基线训练。

## Windows 便携发行版

打包脚本先构建并自检 PyInstaller one-folder 便携版，再构建 one-file 备用版；最终 ZIP、单文件 EXE 和 SHA-256 清单写入 `release/`。便携版支持 Windows 10/11 x64，不要求目标电脑安装 Python 或使用管理员权限。发行与标注任务分发方法见 [Windows 便携版说明](docs/windows-portable-release.md)。

注意：打包版本资源 `packaging/windows_version_info.txt` 是手写副本，可能滞后于
`pyproject.toml`；发布前需核对同步（当前资源记录 0.2.4，落后于单一来源的 0.3.0）。

标注人员无需使用命令行或寻找项目 JSON：启动后选择负责人发送的整个任务文件夹即可。启动页会按系统显示缩放自动调整，主界面右侧参数区在小屏幕上可滚动。

<!-- README-QUICKREF:BEGIN 本区块为手工维护，需与版本来源（pyproject.toml / 代码 __version__，见「版本来源」行）保持一致 -->

---

## 速查

| 项 | 内容 |
|---|---|
| 当前版本 | **0.3.0** |
| 版本来源 | `pyproject.toml` |
| 入口 | 训练/推理入口见 `docs/training-workflow.md`；Python 包位于 `src/atom_center` |
| 依赖锁定 | `requirements/*.lock`（训练/推理两套；另有 CUDA 完整快照 `requirements/lock-win-cu128.txt`） |
| 许可证 | MIT |

**从源码运行**——必须使用本工具自己的虚拟环境，不要用 PATH 上的 `python`：
各工具依赖版本互不相同，共用解释器会互相污染。

```powershell
cd <本工具目录>
python -m venv .venv
# 纯 CPU 推理装推理锁；训练/导出的 training 锁由 scripts\setup_environment.ps1 安装（两者版本契约不同，勿混用）
.venv\Scripts\python -X utf8 -m pip install -r requirements\inference-win-py310.lock
见本目录 README 的入口说明
```

**未纳入版本控制**：`.venv/`、`build/`、`dist/`、`release/`、`__pycache__/`、
测试数据与运行输出。具体规则见本目录 `.gitignore`。

<!-- README-QUICKREF:END -->
