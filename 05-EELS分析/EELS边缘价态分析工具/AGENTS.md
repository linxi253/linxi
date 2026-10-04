# EELS 开发约定

公开版采用 src 布局，位于**统一的 AIforTEM 公开仓库**内（`05-EELS分析/EELS边缘价态分析工具`），使用本项目独立的 `.venv`。GUI 入口为 [run.py](run.py)，CLI 为 eels_edge_analyzer 包；职责与命令见 [DEVELOPMENT.md](DEVELOPMENT.md)，使用方法见 [README.md](README.md)。

## 工作边界

- 开始修改前在**公开仓库根**运行 git status/diff，保留已有代码修改。可读性整理优先修改文档。
- 使用 .venv/Scripts/python.exe 和 [requirements.lock.txt](requirements.lock.txt)；环境不存在时标记未验证。测试另外需要 pytest，不能假定运行锁包含它。
- 不为运行源码而默认执行 editable 安装：run.py 和现有测试已处理 src 路径；CLI 的无安装运行方法见开发说明。
- 项目依赖不装入全局 Python，不因整理导航升级依赖、改写锁或重打包。

## 必须保留的行为

- Spectrum Image 的能量轴与空间轴含义、低损/高损配对校验、eV 与 nm 单位保持一致；shape 一致不能代替能量轴校验。
- 配对校验比较的是**色散**（eV/channel）而非能量原点：低损含 ZLP、高损从吸收边附近开始，两者起点不同是合法输入。非均匀采样、非有限值与非正步长必须被拒绝；不做隐式重采样。
- 参数优先级、随机种子、移动块 bootstrap、参考谱归一化和模型比较均影响科学结果。
- 输出是投影谱权重，不能改称原子百分比或直接证明晶体相。
- 保留原始输入完整性检查和已有结果覆盖确认；不要将逐文件导出描述成目录级原子事务。
- GUI 的取消、关闭等待和后台消息处理必须保持；CLI 错误退出码和已保存配置复跑行为属于接口。
- 沿表面分段宽度由预设或显式输入提供，不设全局默认值：该值与样品/预设的距离分层绑定。
- TEM Suite 通过 eels_edge_analyzer.gui.EELSEdgeAnalyzerApp(root) 内嵌；更改构造入口时检查集成。

## 按改动选择验证

- 算法/参数：tests/test_core.py、tests/test_presets.py、tests/test_review_fixes.py。
- GUI：另跑 tests/test_gui.py，需要 Tk 显示环境；跳过不算通过。
- 集成入口：在 TEM Suite 环境运行针对 eels_edge_analyzer 的冒烟检查，命令见开发说明。
- 纯文档：核对文件差异与相对链接即可；不需要安装依赖或启动 GUI。
