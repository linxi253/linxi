# 10-DSH集成 源码与验证导航

复用视频提取与漂移矫正源码的 DSH/命令行流水线。

**状态：跨项目适配脚本。** 本目录未发现 Git 仓库；工作区根目录也不是仓库，修改前保留文件基线。 工作约定见 [AGENTS.md](AGENTS.md)，下面命令从本项目目录执行。

## 入口与版本

- 入口文件：[tem_pipeline.py](tem_pipeline.py)。
- 版本来源：未登记统一静态版本源；按 README、源码和构建记录分别核实，不能以旧 EXE 文件名推断。
- 锁文件：未登记，不能宣称依赖环境已可复现。
- 构建配置：本批未登记打包配置，不推断已有成品与源码一致。

## 按任务定位

| 任务 | 文件 | 关键符号 |
|---|---|---|
| JSON 行协议、上游加载、提取与矫正 | [tem_pipeline.py](tem_pipeline.py) | load_tools、run_extraction、run_highlevel_pipeline、emit、main |

## 保持的约束

- stdout 每行一个 JSON 对象，summary/error 终止记录和失败返回码是上层工具协议，日志输出不要破坏它。
- 自动放宽漂移参数必须记入审计；质量不足、内存上限和输入保护不能因自动化而省略。
- 视频与漂移项目独立演进；旧文档的 OME-TIFF/环境说明需与当前上游实际 API 核实。动态 DSH 注册也不等于永久安装。

## 环境与验证

- **解释器**：`.venv/Scripts/python.exe`（本项目独立 venv；清单登记 Python `>=3.11,<3.13`）。
- **安装锁**：`requirements.lock.txt`；按锁安装，不要用 `pip install -r requirements.txt` 代替。
- **建环境**：参考仓库根 [环境隔离说明](../环境隔离说明.md) 与 [.tools/projects.json](../.tools/projects.json)；常规 `provision-envs.py` 只读锁、`verify-envs.py` 只验证不安装。
- **导入自检**：`numpy`、`cv2`、`tifffile`、`matplotlib`。
- **注意**：跨项目适配脚本。tests/ 含单元层与真实全链层（test_pipeline_end_to_end.py，合成 FFV1 视频经真实 CLI 跑通），入口 -m pytest tests -q；CI 第 20 项提供 FFmpeg。运行锁按两份上游锁定精确 pin，不含 pytest。

从**本项目目录**执行：

~~~powershell
# pytest
& ./.venv/Scripts/python.exe -m pytest tests -q -rs
~~~

旧报告不代表当前通过；运行后按实际结果记录 pass / skip / 未运行。
