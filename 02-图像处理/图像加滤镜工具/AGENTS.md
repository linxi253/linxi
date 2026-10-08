# 图像加滤镜工具 开发约定

灰度/RGB TIFF 单帧及堆栈的交互式滤镜与批量导出。 **状态：现有实现。** 公开仓库内的工具子项目，使用独立 `.venv`。

按任务查看 [源码与验证导航](DEVELOPMENT.md)，使用与科学说明见 [README](README.md)。

## 修改边界

- 保留已有修改与未跟踪文件；可读性整理只补导航，不改算法、路径、输出或依赖。
- 使用 .venv/Scripts/python.exe；锁文件与测试前置条件见开发说明，缺失时记录未验证，不回退全局 Python。
- 原始数据、发布成品、权重及历史验证材料保持原样。只读导航无需安装、重打包或执行业务模块。

## 关键行为

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
