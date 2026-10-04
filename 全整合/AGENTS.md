# TEM Suite 开发约定

这是公开仓库内的工具子项目，使用独立 `.venv`。，负责集成层。入口为 [run.py](run.py)，工具注册表为 [temsuite/registry.py](temsuite/registry.py)。按功能定位见 [DEVELOPMENT.md](DEVELOPMENT.md)，设计背景见 [README.md](README.md)。

## 工作边界

- 保留当前仓库已有修改；整合问题优先在适配层处理。工具自身问题回对应项目定位并单独验证。
- 使用本项目 .venv/Scripts/python.exe 和 [requirements.lock.txt](requirements.lock.txt)，不把各工具环境合并到全局 Python。
- 源码运行和打包运行都可能动态读取工作区内的工具源码；本项目 EXE 不是各独立工具 EXE 的简单启动器。
- TEMSUITE_WORKSPACE、相对目录、entry、factory、src_layout 和 preload 是集成接口，改名或迁移必须联动检查。

## 必须保留的行为

- 保留同名模块隔离与命名空间包识别；不能用永久叠加 sys.path 或批量删除 sys.modules 的方式替代加载器。
- Tk 根窗口重定向只在受控构造阶段生效；matplotlib 后端初始化、锁定与显式嵌入须保持顺序。
- ToolHost 的键绑定转发、按标签页隔离、关闭回调及后台任务退出流程必须保留。
- ttk 主题和主窗口样式是全局状态，工具加载失败也需要恢复。
- 子进程工具的脚本模式、模块模式和冻结 EXE 模式分别处理；保留返回码与失败提示。
- 打包依赖收集和运行期资源路径不能仅凭静态 import 简化。

## 按改动选择验证

- 使用 tests/smoke_test.py 的工具 ID 参数运行受影响的内嵌工具；变更公共加载/代理机制时扩大到全部内嵌工具。
- 该脚本只筛选当前 available 的内嵌工具，必须核对实际执行的工具 ID；退出码 0 不保证所有注册工具均被覆盖。
- 子进程、交互式按钮、长任务取消和成品 EXE 需要相应专项验证，不能用构造冒烟代替。
- --self-test 会写报告并弹窗，不是只读检查。既有报告仅是历史记录。
- 纯文档运行工作区导航检查即可；缺少环境时不安装依赖来证明文档改动安全。

## 环境与验证

- **解释器**：`.venv/Scripts/python.exe`（本项目独立 venv；清单登记 Python >=3.10，本机母本 3.10）。
- **安装锁**：`requirements.lock.txt`；按锁安装，不要用 `pip install -r requirements.txt` 代替。
- **建环境**：参考仓库根 [环境隔离说明](../环境隔离说明.md) 与 [.tools/projects.json](../.tools/projects.json)；常规 `provision-envs.py` 只读锁、`verify-envs.py` 只验证不安装。
- **导入自检**：`numpy`、`scipy`、`matplotlib`、`pandas`、`tifffile`、`PIL`、`skimage`、`seaborn`、`openpyxl`、`imagecodecs`、`defusedxml`、`ttkbootstrap`、`ncempy`、`tkinterdnd2`、`cv2`。
- **注意**：tests/test_integration_review.py 是集成回归 pytest 套件；tests/smoke_test.py 是 14 工具构造冒烟脚本（直接改 stdout，不能与 pytest 同进程收集，故分开声明）。

从**本项目目录**执行：

~~~powershell
# pytest
& ./.venv/Scripts/python.exe -m pytest tests/test_integration_review.py -q -rs
# smoke_test
& ./.venv/Scripts/python.exe tests/smoke_test.py
~~~

旧报告不代表当前通过；运行后按实际结果记录 pass / skip / 未运行。
