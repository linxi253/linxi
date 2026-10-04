# 非晶面积统计 源码与验证导航

晶体/非晶分割、逐区域几何与时间序列生长统计。

**状态：现有实现。** 工作约定见 [AGENTS.md](AGENTS.md)，下面命令从本项目目录执行。

## 入口与版本

- 入口文件：[main.py](main.py)。
- 版本来源：[constants.py](constants.py) 的 APP_VERSION
- 锁文件：[requirements.lock.txt](requirements.lock.txt)
- 构建配置：[非晶面积统计工具.spec](非晶面积统计工具.spec)

## 按任务定位

| 任务 | 文件 | 关键符号 |
|---|---|---|
| 程序入口 | [main.py](main.py) | main |
| GUI 与批处理 | [gui/app.py](gui/app.py) | EMImageAnalyzerApp |
| 分割与位深转换 | [core/segmentation.py](core/segmentation.py) | SegmentationParams、resolve_uint16_shift、uint16_to_uint8、segment_threshold |
| 逐区域测量 | [core/measurement.py](core/measurement.py) | measure_regions、compute_full_measurement |
| 生长统计与有效帧筛选 | [core/analysis.py](core/analysis.py) | calculate_radial_growth_rate、effective_growth_rows_mask |
| TIFF、堆栈及序列 | [io_utils/tiff_handler.py](io_utils/tiff_handler.py) | load_tiff_stack、load_tiff_slices、scan_uint16_max_shift |
| 结果与工作表 | [io_utils/exporter.py](io_utils/exporter.py) | export_results |
| 界面配置持久化 | [io_utils/app_config.py](io_utils/app_config.py) | load_config、save_config |

## 保持的约束

- 源码根即本目录（公开版去掉了主源码的 `pythonProject` 层）。
- 亮/暗晶体极性、16 位灰度转换与分割方法共同决定面积；未标定状态不能伪装成实测 nm 标尺。
- 固定阈值是 `gray > threshold`（严格大于）；文档与注释不得写成 `>=`。
- `resolve_uint16_shift` 的不变量：对任意 `img_max > 255`，映射后的量程顶端必须 `> 128`（默认阈值），且移位结果 **clip 到 255** 而非 `astype(uint8)` 回绕。
- 位移以**数据集**为单位：同一数据集内映射单调、跨切片可比；不同数据集之间 shift 可能不同，固定阈值对应的原始强度因此不同，跨数据集比较需先确认量程档位。
- 不做逐帧自动伸缩；显式 shift 与 8-bit 输入行为保持不变，不新造标定含义。
- 逐区域面积、有效帧和生长速率定义保持一致；配置持久化与同 stem 输入的输出避让保留。

## 环境与验证

按工作区环境约定准备项目内环境（本机从 3.10 母本按 `requirements.lock.txt` 重建，见公开仓库根 [环境隔离说明.md](../../环境隔离说明.md)）；解释器路径存在不代表依赖或测试已验证。

~~~powershell
# 源码入口（依赖环境已准备后）
& ./.venv/Scripts/python.exe main.py
# 现有验证入口（按改动选择，不是每次都全跑）
& ./.venv/Scripts/python.exe -m pytest -q -rs
~~~

测试需要所选环境具备相应依赖；GUI/Tk 用例的跳过应单独报告，不计作交互功能通过。

现有测试文件：[tests/test_app_config.py](tests/test_app_config.py)、[tests/test_core.py](tests/test_core.py)、[tests/test_exporter.py](tests/test_exporter.py)、[tests/test_tiff_handler.py](tests/test_tiff_handler.py)

`tests/test_core.py` 的 `TestUint16DatasetShift` 覆盖位深不变量：穷举 256..65535 的默认阈值可达性、clip 不回绕、同一 shift 的单调性与跨帧一致性、亮/暗极性，以及"严格大于"的阈值语义。这些断言替换了原先固定分档数字（如 `4096 -> 6`）的写法——那种写法把缺陷固化成了期望值。

若使用 pytest，先核实测试环境包含 pytest；运行锁不保证含开发依赖。unittest 和独立自检按各自入口执行，不能统一替换成 pytest。已存在的历史通过数字不是本轮测试结果。

## 集成与相关资料

TEM Suite 中对应工具 ID 为 amorphous_area。注册表中的 entry/factory/src_layout/preload 是接口；修改后在全整合环境验证。内嵌构造冒烟不覆盖子进程和真实数据处理。

使用说明与既有设计背景见本项目 README。

## 本公开仓库的布局差异

本项目使用独立的 `.venv`。公开版是**统一的 AIforTEM 仓库**（各工具为子目录），源码根即本目录，工作区根没有该导航脚本，因此：

- 在**公开仓库根**执行 git status/diff，而不是在本项目目录；
- 环境约定见公开仓库根 [环境隔离说明.md](../../环境隔离说明.md)；公开版 `.tools` 只有环境隔离与 CI 工具（如 `check-env-isolation.py`），没有导航检查器。

主源码同步本目录时，请保留主源码侧的导航命令与 `pythonProject` 层，不要把它当作公开版可执行命令。
