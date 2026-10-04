# 图像加滤镜工具 源码与验证导航

灰度/RGB TIFF 单帧及堆栈的交互式滤镜与批量导出。

**状态：现有实现。** 公开仓库内的工具子项目，使用独立 `.venv`。 工作约定见 [AGENTS.md](AGENTS.md)，下面命令从本项目目录执行。

## 入口与版本

- 入口文件：[main.py](main.py)。
- 版本来源：[version.py](version.py) 的 __version__
- 锁文件：[requirements.lock.txt](requirements.lock.txt)、[requirements-dev.lock.txt](requirements-dev.lock.txt)
- 构建配置：[TIF_FilterTool.spec](TIF_FilterTool.spec)、[build.bat](build.bat)

## 按任务定位

| 任务 | 文件 | 关键符号 |
|---|---|---|
| 界面、预览与批量调度 | [main.py](main.py) | FilterApp、main |
| 位深转换与滤镜 | [image_filters.py](image_filters.py) | to_float、from_float、gaussian_blur、usm_sharpen |
| TIFF 布局、源路径保护与清单 | [tif_io.py](tif_io.py) | UnsupportedTiffError、ProcessingCancelled、safe_output_path |
| 测试、打包与校验和发布 | [build_release.py](build_release.py) | main |

## 保持的约束

- 保持帧顺序、位深、支持的元数据和 .filter.json 参数记录；非支持的通道布局明确拒绝。
- MINISWHITE、RGB/alpha 与灰度位深不能混用显示转换和定量处理。
- 取消批次只清理正在生成的输出，已完成文件保留；源目录冲突通过现有安全命名处理。
- build_release.py 会测试并生成 EXE，不是只读验证入口。

## 环境与验证

- **解释器**：`.venv/Scripts/python.exe`（本项目独立 venv；清单登记 Python >=3.10，本机母本 3.10）。
- **安装锁**：`requirements.lock.txt`；按锁安装，不要用 `pip install -r requirements.txt` 代替。
- **建环境**：参考仓库根 [环境隔离说明](../../环境隔离说明.md) 与 [.tools/projects.json](../../.tools/projects.json)；常规 `provision-envs.py` 只读锁、`verify-envs.py` 只验证不安装。
- **导入自检**：`numpy`、`scipy`、`PIL`、`tifffile`、`imagecodecs`。
- **注意**：测试文件在项目根（test_*.py），不在 tests/。另有 requirements-dev.lock.txt 属构建环境，不在运行契约内。

从**本项目目录**执行：

~~~powershell
# pytest
& ./.venv/Scripts/python.exe -m pytest -q -rs
~~~

旧报告不代表当前通过；运行后按实际结果记录 pass / skip / 未运行。
