# 视频切片工具 源码与验证导航

视频解码、采样及 ImageJ 兼容 TIFF 堆栈导出。

**状态：现有实现。** 公开仓库内的工具子项目，使用独立 `.venv`。 工作约定见 [AGENTS.md](AGENTS.md)，下面命令从本项目目录执行。

## 入口与版本

- 入口文件：[run_app.py](run_app.py)、[video_extractor/cli.py](video_extractor/cli.py)。
- 版本来源：[video_extractor/__init__.py](video_extractor/__init__.py) 的 __version__
- 锁文件：[requirements.lock](requirements.lock)
- 构建配置：[tem_video_extractor_v4.spec](tem_video_extractor_v4.spec)

## 按任务定位

| 任务 | 文件 | 关键符号 |
|---|---|---|
| GUI 与任务交互 | [video_extractor/ui.py](video_extractor/ui.py) | ExtractorApp |
| 选项、帧与任务数据模型 | [video_extractor/models.py](video_extractor/models.py) | ExtractOptions、FrameSpec |
| 输入身份、任务调度与发布 | [video_extractor/runner.py](video_extractor/runner.py) | JobRunner、verify_file_identity |
| FFmpeg 解码与帧布局 | [video_extractor/decoder.py](video_extractor/decoder.py) | FrameDecoder、resolve_frame_spec |
| 采样与时间轴 | [video_extractor/sampling.py](video_extractor/sampling.py) | timeline_is_uniform、output_time_seconds |
| TIFF 格式与输出校验 | [video_extractor/writers.py](video_extractor/writers.py) | StackWriter |
| 逐帧及任务清单 | [video_extractor/manifest.py](video_extractor/manifest.py) | write_frame_manifest、write_job_manifest |

## 保持的约束

- 默认输出普通 ImageJ TIFF，不能为了大文件方便静默改成 BigTIFF；保持既有容量/截断策略与告警。
- 保持像素编码方向，不自动套用视频容器旋转元数据；旋转告警、帧时间轴和采样规则都要保留。
- 严格解码、既有输出拒绝覆盖、暂存区发布及输入身份复查共同保护实验数据。
- 当前源码说明的输出格式不同于某些旧流水线中的 OME-TIFF 描述，集成时核实真实 API/输出，不能只按后缀推断。

## 环境与验证

- **解释器**：`.venv/Scripts/python.exe`（本项目独立 venv；清单登记 Python `>=3.11,<3.14`）。
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
