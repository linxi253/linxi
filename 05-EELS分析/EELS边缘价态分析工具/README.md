# EELS边缘价态分析工具 v1

用于 DM3/DM4 配对 Dual-EELS Spectrum Image 的样品物理边缘分析。v1 提供 Cu L2,3
的 Cu0/Cu1/Cu2 非负 MLLS、复散射校正、移动块 bootstrap、模型选择、参数敏感性和
Cu1 注入恢复检验。

该工具首先作为独立 Tkinter 程序开发，同时提供命令行入口；随后可由
AIforTEM 的 TEM Suite 以标签页方式内嵌。

## 适用输入

- 一个原始 DM3/DM4 文件，含 survey（可选）、低损 SI 和高损 SI；
- 低损与高损必须是已经配对的 Dual-EELS，形状均为 energy × y × x；
- 三个参考谱：Cu0、Cu1（通常为 Cu2O）和 Cu2（通常为 CuO），格式可为 DM3、DM4、
  CSV、TXT、DAT 或 MSA。

v1 会自动按能量范围推荐 Survey、低损和高损对象；若判断有歧义，可在 GUI 或命令行
指定对象编号。

## 启动

开发环境：

    python -m pip install -r requirements.txt
    python run.py

Windows 发行版：

    dist\EELS边缘价态分析工具.exe

重新生成发行版：

    python -m PyInstaller --noconfirm --clean EELS_Edge_Analyzer.spec

命令行：

    python -m eels_edge_analyzer input.dm4 --references-dir references --output-dir results

参数预设与拟合参数覆盖（优先级：程序默认 < 预设 < 显式命令行参数）：

    python -m eels_edge_analyzer input.dm4 --references-dir refs --preset my.json
    python -m eels_edge_analyzer input.dm4 --no-preset --fit-min-ev 930 --fit-max-ev 975 \
        --background-max-ev 930 --distance-bins "E1:0:2.2,Bulk:2.2"

参数预设 JSON 的可用键包括：`fit_min_ev`/`fit_max_ev`（或成对的
`fit_range_ev`）、`background_min_ev`/`background_max_ev`（或
`background_range_ev`）、`smoothing_ev`、`savgol_polyorder`、
`deconvolution_regularization`、`zlp_window_ev`、`low_loss_baseline_ev`、
`bootstrap_resamples`、`injection_fractions`、`distance_bins_nm`
（`[标签, 下限, 上限|null]` 列表）等；未知键会被拒绝。内置预设位于包内
`eels_edge_analyzer/presets/cu_l23.json`，GUI 中也可选择预设文件
（留空使用内置预设）。

只检查 DM4 对象而不运行：

    python -m eels_edge_analyzer input.dm4 --inspect

复现一次历史分析（用该次输出目录中的 analysis_config.json 重建全部参数；
显式 CLI 参数仍可覆盖）：

    python -m eels_edge_analyzer --config results/analysis_config.json

覆盖保护：输出目录已包含既往分析结果时默认拒绝写入，需显式允许：

    python -m eels_edge_analyzer input.dm4 --references-dir references --overwrite

快速试运行（100 次 bootstrap，关闭注入恢复；与 --bootstrap 互斥）：

    python -m eels_edge_analyzer input.dm4 --references-dir references --fast

大文件导出加速（processed_arrays.npz 不压缩）：

    python -m eels_edge_analyzer input.dm4 --references-dir references --npz-plain

## 处理链

1. 检查 DM4 内的对象、维度、能量轴和配对关系；
2. 将 Spectrum Image 注册回 survey，并自动寻找 top/bottom/left/right 的物理表面；
3. 校准各像素 ZLP，计算相对 t/lambda，并用配对低损谱做正则化 Fourier-ratio
   复散射校正；
4. 按距物理表面的 nm 分层，对每条平行表面的扫描线先平均；
5. 拟合幂律背景、平滑、能量对齐和面积归一化；
6. 比较 Cu0、Cu0+Cu1、Cu0+Cu2 和 Cu0+Cu1+Cu2 竞争模型；
7. 通过移动块 bootstrap、边界/背景/去卷积敏感性、沿表面分段和注入恢复评估证据；
   注入前统一样品基线与参考谱的正面积尺度，并对残差做移动块重采样；
8. 输出完整图表、CSV、JSON、NPZ、运行参数和原始文件哈希。

## 输出

每次运行都会写入一个独立结果目录，主要包括：

- edge_analysis_report.md：可直接阅读的中文报告；
- edge_analysis_summary.json：所有参数、质量控制与结论（含时间戳与依赖版本）；
- edge_profile.csv、model_comparison.csv：二次作图和统计数据；
- processed_arrays.npz：去卷积 SI、距离图、厚度图和参考矩阵；
- 10 张 PNG 图：边界、厚度、价态剖面、模型比较、敏感性、空间连续性、检出能力、
  参考谱对照（reference_spectra.png）和逐距离层拟合目检（edge_region_model_fits.png）；
- run_log.txt：运行时间、依赖版本与告警清单；
- input_integrity.json：原始输入未被修改的哈希证明。

## 科学解释边界

软件输出的是投影谱权重，不是原子百分比。它把结果区分为“无可靠信号”“谱学迹象但不能
独立识别”“支持混合谱学状态”三类。即使得到混合价态的 EELS 证据，也不能单独证明
晶体学 Cu0/Cu1 复合相；需要结合 O-K/EDS、HRTEM、纳米衍射或 4D-STEM。

## v1 已知范围

- 仅支持配对 Dual-EELS 的 DM3/DM4，且能量轴必须单调递增；
- 仅包含 Cu L2,3 的完整竞争模型；
- inward 距离为每列相对边界的垂直距离近似（弯曲表面曲率大处会偏差）；
- GUI 可选择表面方向与参考谱文件，但自由曲线编辑和任意元素端元组合将在
  后续版本加入；
- 参考谱的来源、处理条件和纯度必须由使用者核对。软件会报告参考谱相似性和
  条件数，但不会把相似参考谱误报为可独立辨识的价态。

## v1.2 修复与改进（2026-09 代码审查）

缺陷修复：

- GUI 输出目录为空时，`Path("")` 等价于当前目录，结果会静默写进启动目录；
  现在校验直接拒绝空输入/输出路径；
- 输出目录已包含既往结果时不再静默覆盖：CLI 需 `--overwrite`，GUI 弹确认框，
  `export_artifacts` 默认拒绝（`overwrite=True` 显式放行）；
- CLI 出错时输出一行友好错误与退出码 2（此前是 5 层 traceback），
  Ctrl+C 优雅退出（130）；
- 能量平移扫描现在以数据覆盖余量收窄范围，且区域谱处理拒绝越界平移——
  此前 `np.interp` 会静默端点外推，使预边噪声被估为 0、SNR 虚高为无穷大；
- 数据集编号可为负数、bootstrap 次数无上限的问题在校验中拦截；
- CLI 快速连换文件时对象检查竞态：GUI 忙时记下待办，空闲后检查最新文件。

科学稳健性：

- 样品区中位 t/λ 超过 1.0 时，summary、报告与 run log 均输出显式告警
  （`thickness_quality`），提示 Fourier-ratio 校正可靠性受限；
- auto 表面方向在真空-样品明暗对比不可信（如 BF 像极性相反）时，
  `surface_registration.auto_orientation_confident` 置 False 并在报告中告警；
- bootstrap 块宽新增基于沿表面积分强度自相关的建议值
  （`along_surface_block_width`），配置块宽偏小时在报告中提示不确定度可能被低估；
- 边缘-内部组分差 bootstrap 在两区域列数一致时改为共享沿表面块位置（真配对），
  `edge_minus_bulk_bootstrap_method` 记录所用方法；
- 报告"方法和质量控制"声明 Savitzky-Golay 平滑对 BIC 独立噪声假设的影响。

可复现性与输出：

- summary 新增 `application.timestamp_utc` 与 `environment`（Python/NumPy/SciPy/ncempy 版本）；
- 输出新增 run_log.txt、reference_spectra.png（参考谱对照）、
  edge_region_model_fits.png（逐距离层实验-拟合目检）；
- 新增 `--config results/analysis_config.json` 一键复跑历史分析；
- CSV 导出清洗以 `=`/`+`/`@` 等开头的字符串单元格（CSV 公式注入加固）；
- 导出前预检查输出目录写权限，无权限时报 PermissionError 而非中途失败。

性能：

- 敏感性分析的去卷积改为多正则化共享逐像素 FFT（正则化只改变 Wiener 比值），
  3 组正则化的去卷积开销约降为原来的 1/3，数值与逐组计算逐位一致。

内部质量：

- 预边/边窗口宽度、能量平移扫描范围、t/λ 告警阈值等魔法数字提升为命名常量；
- 删除重复的参考谱三元组校验；模型线色统一为单一常量表；
- 敏感性散点图用颜色区分正则化、marker 形状区分背景下限，两个维度同时可读；
- GUI 新增"使用参数预设"开关（等效 CLI `--no-preset`）。

## v1.1 修复与改进

- 依赖下界收紧到 numpy>=2.0（代码使用的 `np.trapezoid` 为 NumPy 2.0 API）；
- DM4 对象检查改为只解析头部标签树（不读数据体），多 GB 文件瞬时完成且
  不再冻结 GUI；关窗时会等待后台导出完成，避免写出截断的结果文件；
- 删除未被引用的死代码（旧版参考谱平移扫描、ZLP 对齐函数、无用的
  去卷积立方体缓存）；
- 新增科学性留痕：auto 方向候选诊断、空间标尺换算明细、t/λ 积分上限、
  距离分层像素覆盖统计、能量轴单调性校验；
- GUI/CLI 易用性：Bootstrap 输入解析、Cu0/Cu1/Cu2 单文件选择、参数预设、
  进度条不再倒退、CLI 友好错误提示；
- 输入校验补全（多项式阶数、平滑惩罚、ZLP/基线区间顺序、敏感性背景上限、
  输出路径占用检查）；
- 性能：敏感性分析仅对 E1 及边界邻域像素执行去卷积（大型 SI 下每次去卷积
  约 16 倍加速），原始文件哈希校验由 4 次全量读取降为 2 次（stat 快路径）。

<!-- README-QUICKREF:BEGIN 由 tools/gen-readme-block.py 生成，请勿手工编辑本区块 -->

---

## 速查

| 项 | 内容 |
|---|---|
| 当前版本 | **0.2.0** |
| 版本来源 | `pyproject.toml` |
| 入口 | `run.py` |
| 依赖锁定 | `requirements.lock.txt` |
| 许可证 | MIT（本体代码）；**发行 exe 内含 GPL-3.0 的 ncempy**，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) |

**从源码运行**——必须使用本工具自己的虚拟环境，不要用 PATH 上的 `python`：
各工具依赖版本互不相同，共用解释器会互相污染。

```powershell
cd <本工具目录>
python -m venv .venv
.venv\Scripts\python -X utf8 -m pip install -r requirements.lock.txt
.venv\Scripts\python run.py
```

**未纳入版本控制**：`.venv/`、`build/`、`dist/`、`release/`、`__pycache__/`、
测试数据与运行输出。具体规则见本目录 `.gitignore`。

<!-- README-QUICKREF:END -->
