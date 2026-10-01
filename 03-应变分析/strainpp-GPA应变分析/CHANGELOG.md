# Changelog

## 1.4.1 — 2026-09-28

深度代码审查修复（对照《电镜分析工具代码审查报告》与本仓库自查）。

- **修复批量链路丢弃各向异性像素标定（科学正确性）**：
  `BatchConfig.pixel_size` 现接受 `(y, x)` 二元组，GUI 批量对话框把第一帧
  的各向异性标定完整传入（此前仅取 Y 分量，各向异性输入下批量的
  e_xy/e_yx 与位移单位会和第一帧 GUI 结果不一致）；批量 CLI
  `--pixel-size` 同样支持 `Y,X`；`batch_metadata.json` 记录实际使用的
  `pixel_size_nm_yx`（另存 `source_calibration_nm` 溯源文件标定）。
- **修复 GUI 消息泵可被单条消息异常杀死的问题**：
  `_poll_worker_queue` 改为逐条消息隔离异常 + `finally` 中重注册
  `after` 轮询；批量完成/失败弹窗在对话框已关闭时回退到主窗口为父，
  不再抛 TclError 导致后台任务全部失联。
- **批量消息改用独立 token**：批量运行期间启动/取消其它任务（会使普通
  任务的 generation 失效）不再吞掉批量进度与完成消息，进度对话框能正常
  收尾；关闭批量对话框时明确询问"是否同时取消"，选择后台继续时状态栏
  提示。
- **修复单页 3D TIFF 序列的 O(n²) 解码**：`iter_tiff_frames` 对单页多维
  series 现在只解码一次再按帧切片（此前每帧都完整解码整个堆栈）。
- **GUI"打开连续图像"新增帧序列文件夹入口**（按钮弹出菜单二选一），
  与帮助文档宣称的能力一致。
- **BatchConfig.validate 补全 fail-fast 校验**：G 矢量非零且距 DC ≥ 2 px、
  σ 有限正数、pixel_size 标量或 (y,x) 正对、preview_specs 结构完整
  （vmin < vmax、字段可预览）；畸形预览快照在渲染层自动回退稳健自动色标，
  不再导致整帧失败。
- **批量导出帧对齐加固**：每帧所有数据页与预览页先全部物化再写盘，
  计算失败不会在各 `batch_*.tif` 间留下页数错位。
- **常数（无衬度）图像在 `GPA.load_image` 直接拒绝**；新增
  `estimate_peak_bytes` 内存预估，估算峰值内存超过 4 GiB 时在加载阶段
  警告（GUI 中以弹窗提示，CLI 打印警告）。
- **FFT 切换到 scipy.fft 并默认多核并行**（`workers=-1`；可用环境变量
  `STRAINPP_FFT_WORKERS` 覆盖），结果与 worker 数无关；大图 FFT 提速。
- `batch_statistics.csv` 列序固定（关键列在前，统计列按字典序），
  失败帧不再导致两次运行列序漂移。
- 元数据统一与语义澄清：CLI 数值 TIFF 的无效像素键改为
  `invalid_pixels`（与 GUI/批量一致）；`positive_rotation` 明确为
  "数学正方向（+x→+y）；屏幕 +y 向下时视觉为顺时针"。
- 抽取 `strainpp_gpa/metadata.py` 单源提供版本/依赖/像素尺寸 JSON 化
  （消除 GUI 与 CLI 的三处重复）；DM 内置解析器的单位字符串按文件字节序
  解码；功率谱点击边界（正好点在 extent 最右/最下边缘）不再产生越界 G；
  中文字体列表增加 Linux 回退（Noto Sans CJK SC / 文泉驿）；
  `unwrap_phase_2d` 文档化残差/枝切线处理策略。
- 测试 49 → 62 项：新增常数图像拒绝、内存警告、FFT worker 无关性、
  批量配置校验、各向异性批量端到端数值一致性、CSV 列序、
  `_series_frames` 单次解码等回归测试。

## 1.4.0 — 2026-09-06

- **新增连续图像批量处理**：
  - 工具栏"打开连续图像"支持多页堆栈 TIF（Fiji Substack 等）与编号帧
    序列文件夹；自动读取 ImageJ 像素标定（nm）与帧率；
  - 第一帧照常交互选 G/σ/参考区并计算，确认后一键批量处理全部帧；
  - 每帧以第一帧 G 为初值、可用参考区逐帧精修（跟随原位漂移），G 修正、
    有效比例与统计量（median/mean/std）写入 `batch_statistics.csv`；
  - 导出场可勾选（应变/旋转/膨胀/畸变/位移/相位/掩膜），默认输出多页
    结果堆栈（流式逐帧写入，内存 O(1) 帧），可选单帧序列文件夹；
  - 单帧失败不中断批处理，失败原因记录在 CSV；参数完整记录在
    `batch_metadata.json`；进度条 + 可取消；
  - 新增 CLI：`strainpp-gpa-batch stack.tif --g1 ... --g2 ... [--sequence]
    [--refine-roi X1 Y1 X2 Y2]`，`--list-only` 先查看帧数与标定。
- 新增模块 `strainpp_gpa/stack_reader.py`（堆栈/序列惰性读取）、
  `strainpp_gpa/batch.py`（批处理引擎，GUI 无关）、`strain_batch.py`
  （命令行入口）、`strainpp_gpa/cmaps.py`（原版配色公共定义）。
- **批量结果新增 8-bit 彩色预览堆栈**（`preview_batch_*.tif`，默认开启）：
  所见即所得——从每帧**原始值**流式渲染（满幅彩色、不套质量掩膜），配色
  与色标范围快照自界面第一帧的显示状态，与用户在 GUI 中所见完全一致；
  全栈色标固定、帧间可直接对比；NaN 像素浅灰（掩膜视图/失败帧）。float32
  的 `batch_*.tif` 是定量数据（在 ImageJ 中默认按真实 min/max 显示，需要
  Auto 对比度才能看到纹理），预览仅供查看，不用于定量。
- 批量统计以 median 为稳健中心值——条纹交叉点/奇点的长尾会把 mean
  拉向零（mean 一并保留供对比）。
- 新增 9 项测试：堆栈/序列读取、批量堆栈+序列+CSV/JSON、坏帧不中断、
  G 漂移记录、取消、ROI 校验、预览堆栈所见即所得校验、GUI 全链路批量
  冒烟（49 项测试通过）。

## 1.3.1 — 2026-09-06

- **默认场图配色改为原版 Strain++ 的 Turbo 色环**（9 个色标自上游
  `ColorBarPlot::CreateColorMaps` 精确移植）：色环为深紫→蓝→青→绿→黄绿→
  黄→橙→深红，0 应变位于中段黄绿色而非白色，因此背景微小噪声也会显示
  为满幅彩色纹理，与原版软件的观感一致。
- 原版全部五套配色注册为内置项并保留为可选项：Turbo（默认）、Polar
  （蓝→黑→红，按 QCustomPlot 官方文档；替换旧版误仿的蓝→白→红）、
  BlOr（深蓝→近白→橙，精确色标替换旧近似）、Thermal、Greyscale；
  matplotlib 其余配色（RdBu_r、viridis 等）全部保留在各自分组。
- NaN（质量掩膜排除的不可靠像素）在界面预览与导出 PNG 中改为浅灰，
  与色图的 0 值颜色明确区分——"无数据"与"零应变"不再混淆。
- 右键单图配色菜单与手动调节对话框的快速选择加入原版配色；
  功率谱配色下拉排除原版场图色环组。
- 说明：先前"应变图大片白色"是发散色图 RdBu_r 把 0 映射为白色的显示
  语义问题，调整色标范围无法改变；现默认色图从语义上解决。

## 1.3.0 — 2026-09-06

- 新增"显示全部像素"开关（配色区复选框）：勾选后所有像素按原始计算值
  上色，包括边缘和低振幅区的不可靠值，与原版 Strain++ 的整幅显示一致。
  分层设计——界面预览与 PNG 出图随开关；右侧统计摘要与浮点 TIFF/NPY
  数值导出**始终**套用质量掩膜，定量语义不受开关影响；元数据 JSON 记录
  `png_show_all_pixels` 供追溯。
- `GPA.compute()` 新增 `mask_results` 参数（默认 True 保持原语义）；
  GUI 计算改为保留原始值，质量掩膜统一移到显示层按需套用。
- 结果统计口径显式标注"掩膜内"；导出 info.txt 记录 PNG 显示范围。
- 新增测试：`mask_results=False` 原始值/掩膜一致性、GUI `_field_data`
  恒掩膜导出、块均值降采样（35/35 通过）。

## 1.2.0 — 2026-09-06

- 修复 VALIDATION.md 推荐命令中不存在的 `--sigma` 参数（应为
  `--sigma1`/`--sigma2`），该命令此前会直接被 argparse 拒绝。
- GUI：菜单"导出当前结果"不再绕过过期结果防护——修改 G 矢量后（
  `_result_stale`）结果场导出会被阻止并提示重新计算，与工具栏导出按钮
  行为一致，避免以同名文件静默导出与界面参数不符的数据。
- GUI：打开 DM3/DM4 时若既无可验证标定也无候选标定，现在会弹警告
  说明暂用 1 nm/px，而不是静默回退。
- GUI：高斯掩膜宽度拆分为 σ1/σ2 两个独立输入，与 CLI 的 `--sigma1`/
  `--sigma2` 能力对齐；自动检测后两者同步更新。
- 统一 CLI 与 GUI 导出的 `*_metadata.json` schema：GUI 改用与 CLI 相同的
  顶层键（`software_version`、`created_utc`、`input`、`pixel_size_nm_yx`、
  `g1/g2_fft_pixels`、`sigma1/sigma2_fft_pixels`、`reference_roi_xyxy`、
  `g1/g2_refinement_steps`、`dependency_versions` 等），CLI 的
  `reference_refinement` 拆分为对应三个扁平键，`dependencies` 更名为
  `dependency_versions`。
- GUI 预览降采样由等间隔抽点改为块均值（有限样本均值、全 NaN 块保持
  NaN），消除高频条纹在预览中的混叠伪影。
- `Phase.refine_iterative` 新增 `cancel_check` 协作取消参数，GUI 参考区
  精修现在可在迭代之间被取消；取消异常基类 `ComputationCancelled` 移入
  `strainpp_gpa.phase`，`GPAComputationCancelled` 改为其子类（对外
  `except RuntimeError`/原异常名均兼容）。
- CLI：输出目录创建与已有文件覆盖检查前移到计算之前，忘加
  `--overwrite` 时不再浪费一次完整计算；导出阶段包裹错误处理，
  写盘失败输出清晰错误并以状态码 1 退出；`--format txt` 对超大场
  （>4M 像素）打印体积警告。
- 移除死代码 `utils.fit_plane_svd` 与 `utils.power_spectrum`；修正
  `Phase.refine_g_vector` 文档（实际使用中位数梯度而非平面拟合）。
- 命名空间治理：`strainpp_gpa` 的 `__init__` 改为显式导入并定义完整
  `__all__`；`utils`/`phase`/`gpa`/`dm_reader` 增加 `__all__`，不再经
  `import *` 泄漏 `svd`、`fft2` 等第三方名称。
- `GPA.load_image` 对接近相等的像素尺寸改用 `np.isclose` 归一化为标量，
  与 DM 读取器的标定归一化行为一致；phase2 复用同一 FFT 缓冲时跳过
  重复的全数组有限性扫描（`Phase.set_fft` 新增 `check_finite` 参数）。
- GUI：工具栏/清除/配色逻辑不再访问 `GPA._fft` 私有属性，改用公共
  `power_spectrum` 属性；导出统计标签改用 ε/ω 记号（原误写为 e/w）；
  "关于"对话框移除硬编码的更新日期。
- PyInstaller 打包默认关闭 UPX（`upx=False`），降低杀软误报风险。
- 修复 PyInstaller 产物在本机 setuptools 84 环境下启动即崩的问题：
  新版 setuptools 的 `pkg_resources` 依赖外部 `jaraco.text`，而
  `pyi_rth_pkgres` 运行时钩子未收集该链路。本项目及全部运行时依赖
  均不使用 `pkg_resources`，spec 改为排除 `setuptools`/`pkg_resources`，
  `STRAINPP_SMOKE_TEST=1` 打包冒烟测试通过。
- 版本号 1.1.0 → 1.2.0（`_version.py`、exe 版本资源、README 标题），
  CHANGELOG 两个未发布小节并入本版本。
- 新增测试：DM4 内置解析器最小 fixture 端到端读取、`refine_iterative`
  协作取消、`GPAComputationCancelled` 继承关系、近等像素尺寸归一化。

### 随 1.2.0 一并发布的前期未发布变更（原 2026-07-31 条目）

- 包结构：`gpa`/`phase`/`utils`/`dm_reader` 的真实实现移入 `strainpp_gpa`
  包并使用相对导入；顶层同名文件仅保留为源码兼容 shim，不再安装到
  site-packages，避免注入通用模块名。
- `requirements.txt` 与 `pyproject.toml` 的依赖上界已统一；README 说明两种
  安装方式等价并推荐 `pip install -e .`。
- `run.py` 不再因缺少 `ncempy` 阻止纯 TIFF 用户启动；DM 打开时才懒检查。
- GUI 计算增加协作式取消：`GPA.compute()` 在 FFT/IFFT、相位解包裹和
  张量场计算等阶段之间轮询取消标志，旧任务结果不会覆盖新任务。
- 修复 `_show_field()` 更新分支过早 `return` 导致标题/轴标签/单位不刷新。
- 功率谱选 g 点增加有限值、图像边界与 DC 邻近校验；自动检测后同步像素
  尺寸输入框。
- DM Scale 启发式搜索增加分隔符与单位模式校验，降低误报；补误报防护测试。
- DM3/DM4 内置解析器补充文件头/字节序边界校验。
- GUI 全部导出的 `*_metadata.json` 增加 `dependency_versions` 和
  `phase_field_semantics`。
- CI：安装 `.[test]` extras，修复 `python -m build` 在干净 runner 上缺 `build`
  包而失败的问题。
- 版本号改为单一来源 `_version.py`（pyproject dynamic version、CLI/GUI/包
  `__init__` 统一引用），移除 `from __init__ import __version__` 的脆弱写法。
- DM 读取：ncempy 解析失败时回退到内置解析器，两者都失败才报错；新增回退
  回归测试。
- GUI：修改 G 矢量后结果标记为过期并禁用导出，避免导出旧参数结果；自动峰
  检测重建 GPA 后清空不一致的旧结果。
- 相位场（P_g1/P_g2）在 GUI 显示/导出与 CLI 导出中统一套用质量掩膜（掩膜外
  NaN），与应变场语义一致。
- CLI `--pixel-size` 改为单参数（`0.02` 或 `0.02,0.025`），消除 `nargs='+'`
  吞掉输入文件名的解析陷阱。
- `compute()`/`compute_strain_only()` 共享张量计算辅助方法，消除重复代码。
- `Phase` 新增公共 `compute()` 流程方法，GPA 不再直接调用私有方法。
- GUI“全部导出”移入后台线程，导出大图不再卡住界面。
- 新增端到端合成晶格回归测试（掩膜→相位→应变/位移符号与幅值、g 精修），
  并清理未使用导入、死代码和误导性注释。

## 1.1.0 — 2026-07-29

- Corrected wrapped-phase derivative sign, edge handling, and affine strain recovery.
- Corrected coordinate rotation and anisotropic-pixel tensor scaling.
- Added lazy phase unwrapping, reliable-pixel masks, G-vector conditioning checks,
  and robust Bragg peak/radius detection.
- Fixed odd-sized FFT centring and displacement reconstruction across phase wraps.
- Added iterative reference-region G refinement to the GUI and CLI.
- Added non-destructive in-memory cropping for excluding scale bars, labels, and
  damaged borders before FFT/GPA computation.
- Added robust TIFF stack/RGB handling and ncempy-backed DM3/DM4 loading.
- Moved GUI file I/O and GPA work off the Tk main thread and added safe cancellation.
- Added full-precision TIFF/NPY export, NaN preservation, JSON provenance, and
  overwrite confirmation.
- Added an installable Python package, console entry points, tests, CI, GPL license,
  reproducible dependency lock, and versioned PyInstaller build/signing support.
- Validated the repaired workflow against the supplied 2026-07-13 TEM images and
  added a packaged-runtime smoke test for the numerical and DM-loading paths.
