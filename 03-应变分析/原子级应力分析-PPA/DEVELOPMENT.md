# PPA 源码与验证导航

功能与科学定义见 [README](README.md)，修改约定见 [AGENTS.md](AGENTS.md)。以下路径相对本项目，命令从项目根目录执行。

## 入口与版本

- 主 GUI：[ppa.py](ppa.py) 的 AtomMarkerApp；TEM Suite 的 ppa_strain 条目调用 ppa.AtomMarkerApp(root)。
- 统计 GUI：[ppa_stats.py](ppa_stats.py) 的 PPAStatsApp，读取主程序导出的 CSV 与元数据。
- 源码包版本：[pyproject.toml](pyproject.toml) 的 project.version。构建配置 [ppa-v2.spec](ppa-v2.spec)、[ppa_version_info.txt](ppa_version_info.txt) 和已有成品需单独核对；历史文件名不代表源码版本。
- 可选传统滤波器：ppa.py 的 _find_hrtem_filter_dir，通过 PPA_HRTEM_FILTER_DIR 或工作区路径寻找 butter.py。
- [atom_detector](atom_detector/README.md) 是可选子项目。稳定主程序不加载 .pt 模型；涉及该子项目时遵循其模型白名单约束。

## 大文件中的功能定位

ppa.py 同时包含纯辅助函数、GUI、状态管理和导出。先按符号定位相关段落；通过现有导航定位 GUI；新边界模块按数据契约组织，保留 AtomMarkerApp 集成接口。

| 修改目标 | 文件与符号 |
|---|---|
| 原子阈值 / 去重 | [ppa.py](ppa.py)：parse_detection_threshold、greedy_nms |
| 双方向选点与参考区 | ppa.py：infer_atom_chain_from_anchors、estimate_reference_vectors_from_chains、refine_lattice_basis_in_region |
| 晶格索引与空洞 | ppa.py：assign_unique_lattice_indices、find_lattice_holes |
| 分析调度 / 过期结果 | ppa.py：AtomMarkerApp.run_ppa_analysis、_poll_worker_queue、_clear_analysis_results |
| 云图与坐标显示 | ppa.py：strain_value_for_display、_strain_fields_from_result、_interpolate_strain_grids |
| 项目保存 / 加载 / 导出 | ppa.py：AtomMarkerApp.save_project、load_project、save_displacement_csv、save_strain_csv |
| 数值应变与参考格校验 | [ppa_core/strain.py](ppa_core/strain.py)：validate_reference_lattice、compute_local_peak_pair_strain、compute_cst_strain |
| 图像与帧选择 | [ppa_core/image_io.py](ppa_core/image_io.py)：load_analysis_image |
| 项目格式与原图身份 | [ppa_core/project_store.py](ppa_core/project_store.py)：image_identity、save_project、load_project |
| 局部邻居拓扑 | [ppa_core/lattice.py](ppa_core/lattice.py)：local_lattice_topology |
| 项目与参数边界 | [ppa_core/validation.py](ppa_core/validation.py)：validate_project_data、finite_number |
| CSV 与元数据提交 | [ppa_core/export_store.py](ppa_core/export_store.py)：CsvExport |
| 云图支持域和采样 | [ppa_core/interpolation.py](ppa_core/interpolation.py)：interpolate_fields |
| 审查反例回归 | [tests/test_review_fixes.py](tests/test_review_fixes.py) |
| 核心公开导入 | [ppa_core/__init__.py](ppa_core/__init__.py) |
| CSV 读入、坐标和统计 | [ppa_stats.py](ppa_stats.py)：load_displacement_csv、load_strain_csv、detect_physical_convention、compute_strain_gradient |

主流程：图像与确认原子点 → 参考晶格/索引 → 局部 PPA 或兼容 CST → 位移/应变结果 → 显示或带元数据的 CSV → ppa_stats。算法输出与渲染变换分开核查。

## 科学与数据约束

- 数值核心使用 x 向右、y 向上的物理坐标；GUI y 向下。strain_value_for_display 对剪应变和旋转实施显示变换，CSV 保持物理约定。
- 张量剪应变 ε_xy 与工程剪应变 γ_xy = 2 ε_xy 不等价。Green–Lagrange 应变、小应变、有限旋转和等效应变有各自定义，详见 README。
- 局部 PPA 按原子输出，兼容 CST 按有效三角形输出。几何不足的原子仍保留位移，应变用 NaN 表示；不能为简化数组而丢弃它们。
- 晶格索引一对一分配、质量标记和参考区精化共同约束计算；全图晶格精化与线性去趋势在稳定版禁用，避免吸收真实均匀应变。
- 编辑点表会使在途任务结果失效；线程代际与取消状态不是冗余变量。
- v2 项目文件绑定原图路径、大小、SHA-256 和帧号；旧 v1 的确认导入另有规则。
- 位移/应变 CSV 与同名 .metadata.json 配套；UTF-8 BOM、坐标约定、算法标识和可选列解析都是兼容接口。统计优先读取元数据，缺失时启发式判断存在局限。
- 本项目不直接计算材料应力。

## 环境与命令

使用项目内 .venv 和 [requirements.lock.txt](requirements.lock.txt)，测试基于标准库 unittest。恢复环境参阅工作区 [环境隔离说明](../../环境隔离说明.md)，源码可直接从项目目录运行，不必先 editable 安装。

~~~powershell
& ./.venv/Scripts/python.exe ppa.py
& ./.venv/Scripts/python.exe ppa_stats.py

# 核心测试模块（也会导入 GUI 模块及可选检测子项目的部分代码）
& ./.venv/Scripts/python.exe -m unittest tests.test_core -v

# 历史修复；部分用例涉及 GUI 或可选子项目
& ./.venv/Scripts/python.exe -m unittest tests.test_fix_regressions -v

# 界面与后台流程，需要 Tk 显示
& ./.venv/Scripts/python.exe -m unittest tests.test_gui_workflow -v

# 完整现有测试
& ./.venv/Scripts/python.exe -m unittest discover -s tests -v
~~~

运行锁与源码元数据分别核对。当前 Pillow 声明为 >=9，与锁定的 12.3.0 一致；测试成功不代表独立 CVE、实验标定或成品 EXE 验证。

## 验证范围

| 测试 | 关注点 |
|---|---|
| [test_core.py](tests/test_core.py) | 应变、图像/项目持久化、辅助函数与模型安全；会导入 ppa、ppa_stats 和 atom_detector 部分模块 |
| [test_fix_regressions.py](tests/test_fix_regressions.py) | 阈值、任务失效、CSV/BOM 与可选检测子项目回归 |
| [test_gui_workflow.py](tests/test_gui_workflow.py) | 保留确认原子、参考格流程、显示与后台任务；无 Tk 时部分测试跳过 |

更改主 GUI 构造入口后，在全整合目录执行：

~~~powershell
& ./.venv/Scripts/python.exe tests/smoke_test.py ppa_strain
~~~

核对实际执行了 ppa_strain。算法改动应比较固定合成晶格/参考态下的结果与质量字段；图形能打开不能代替科学回归。文档改动仅需工作区 [只读导航检查](../../.tools/check-codex-navigation.py) 和差异检查。

## 2026-10-08 修复验证

19 项审查问题的修复、验收映射和剩余范围见 [更新报告](review_artifacts/PPA_DEEP_REVIEW_2026-10-08.md)。历史缺陷探针不应在修复版本重跑并覆盖原始证据；执行 `-m unittest tests.test_review_fixes -v` 验证修复。
