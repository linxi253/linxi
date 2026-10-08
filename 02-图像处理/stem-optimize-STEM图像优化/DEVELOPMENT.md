# stem-optimize-STEM图像优化 源码与验证导航

局部纹理调制的自适应频域降噪与显示增强。

**状态：现有实现。** 公开仓库内的工具子项目，使用独立 `.venv`。 工作约定见 [AGENTS.md](AGENTS.md)，下面命令从本项目目录执行。

## 入口与版本

- 入口文件：[main.py](main.py)。
- 版本来源：[pyproject.toml](pyproject.toml) 的 project.version；[version.py](version.py) 的 APP_VERSION
- 锁文件：[requirements-win-py310.lock](requirements-win-py310.lock)
- 构建配置：[stem-optimize-v2.1.spec](stem-optimize-v2.1.spec)

## 按任务定位

| 任务 | 文件 | 关键符号 |
|---|---|---|
| 启动与日志 | [main.py](main.py) | main、setup_logging |
| GUI | [main_window.py](main_window.py) | MainWindow |
| 单帧处理与参数校验 | [pipeline.py](pipeline.py) | validate_params、process_frame |
| 频域算法 | [filters.py](filters.py) | adaptive_spectral_filter |
| 灰度归一化与 CLAHE | [contrast.py](contrast.py) | normalize_to_uint16、apply_clahe |
| TIFF、输入签名与内存估算 | [tiff_handler.py](tiff_handler.py) | paths_refer_to_same_file、file_signature |
| 后台输出事务与溯源 | [worker.py](worker.py) | ProcessingWorker、stale_temporary_files |

## 保持的约束

- 该算法不是使用物理 PSF/CTF 的 Wiener 反卷积，不改变其科学解释。
- TIFF 完整校验后发布；.stem.json 随后发布，旁车发布失败时明确报告，不能将两者写成不可分割事务。
- 取消或任何一帧失败均不发布半成品；同路径/硬链接输入保护、全堆栈显示范围和标定保留。
- 16 位显示输出不意味着保持原始强度标定，CLAHE 等增强会改变灰度分布。

## 环境与验证

- **解释器**：`.venv/Scripts/python.exe`（本项目独立 venv；清单登记 Python >=3.10，本机母本 3.10）。
- **安装锁**：`requirements-win-py310.lock`；按锁安装，不要用 `pip install -r requirements.txt` 代替。
- **建环境**：参考仓库根 [环境隔离说明](../../环境隔离说明.md) 与 [.tools/projects.json](../../.tools/projects.json)；常规 `provision-envs.py` 只读锁、`verify-envs.py` 只验证不安装。
- **导入自检**：`numpy`、`scipy`、`cv2`、`tifffile`、`defusedxml`、`imagecodecs`。
- **注意**：锁名与头部均为 Windows x64 / CPython 3.10 且带 --require-hashes，因此母本固定 3.10，不允许 3.12 回退。

从**本项目目录**执行：

~~~powershell
# pytest
& ./.venv/Scripts/python.exe -m pytest -q -rs
~~~

旧报告不代表当前通过；运行后按实际结果记录 pass / skip / 未运行。
