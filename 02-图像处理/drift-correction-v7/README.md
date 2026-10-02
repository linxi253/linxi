# TIFF 漂移矫正工具 v7.3.0（ImageJ 参数对齐版）

面向二维灰度 TIFF 堆栈的平移漂移检测与矫正工具。v7 由 v5.2（安全版）与
v6.1（算法版）合并而来：既保留文件安全、TIFF 结构校验、配准质量检查和
审计报告，也引入纯平移估计与可配置边界模式。

## 合并来源

- `drift-correction-漂移矫正`（v5.2 安全版）：文件安全、`_corrected.tif` 默认输出、
  临时文件原子替换、结构校验、共同有效区域裁剪、JSON/CSV 审计报告、后台任务与测试。
- `drift-correction-v6`（v6.1）：纯平移估计、缺失段插值、
  `border_mode`/`border_value`、
  `nfeatures` 默认 5000、死代码清理。

两个旧版本（v5.2、v6）的源码归档在开发机的 `<仓库根>\08-历史版本`，该归档未随本仓库分发（仓库内不存在此目录）；仓库内仅保留合并后的 v7。

## 安全行为

- 输出默认命名为 `原名_corrected.tif`，绝不覆盖输入文件。
- 保存先写入同目录临时文件，重新验证页数、类型和堆栈结构后再原子替换输出。
- 取消、写入失败或程序关闭只会移除本次临时文件，不会删除已有输出。
- 默认裁剪到全部矫正帧的共同有效区域，避免零值黑边污染定量分析。
- 检测失败、纹理不足和匹配质量不足会显示为失败；不会伪装成零漂移。
- 尚未检测的文件不会生成全零位移输出；保存前还会检查非零矫正是否实际改变像素。
- 批处理会预检同名输出并询问是否覆盖；输出目录与输入目录相同时会明确提示
  二次矫正风险。
- 加载前只读元数据：C/Z 轴确认与内存预估都发生在整栈解码之前；解码体积
  超过物理内存 40% 时要求人工确认，批处理对超过 90% 的文件直接跳过并记录。
- 批处理生成 JSON 和 CSV 审计报告，记录工具版本、每个文件的耗时、源文件
  大小/修改时间、质量、位移范围、裁剪区域、生效校验比例、逐帧对质量明细
  （特征数/匹配数/内点/残差/状态）、使用的检测参数，以及被取消文件的处理状态。
- 保存与批处理同时写出逐帧位移表 `<原名>_corrected_shifts.csv`
  （frame、shift_x_px、shift_y_px、key_frame、segment_estimated），插值
  区段显式标记，可直接导入 Origin/Excel 复核漂移曲线。

## 支持的输入

v7 只接受单一 TIFF series、每页尺寸和 dtype 相同的二维灰度图像。

- 支持：uint8、uint16、int16、uint32、int32、float32、float64。
- 1-bit 输入会显式转换为 uint8 0/255。
- 拒绝：RGB/RGBA、调色板 TIFF、混合尺寸/类型页、多个 series。
- 序列轴只要包含通道（C）或层（Z）维度——无论是否同时存在 T 轴——加载时
  都会要求人工确认按页处理确为时间序列；实测 ImageJ TCYX/TZYX hyperstack
  会被当成数倍数量的"时间帧"，必须经过确认。TEM 相机把时间轴标注为
  slices/Z 的连续采集堆栈即属此类。批处理启动前会对这类文件统一询问一次：
  确认后按时间序列处理，拒绝则仅跳过这些文件（核心 API 可用
  `allow_ambiguous_axes=True` 跳过询问）。

复杂 OME-TIFF、真正的多通道数据和 z-stack 应先在采集软件中选择正确的时间序列后再导出。

## 算法

- SIFT 特征 + Lowe ratio 匹配；每个帧对只估计纯平移。
- 检测与匹配默认值与 Fiji "Linear Stack Alignment with SIFT" 对齐：
  initial sigma 1.6、steps/octave 3、closest/next closest ratio 0.92、
  RANSAC 对齐误差 25 px、最小内点比例 0.05。与 ImageJ 的刻意差异：
  特征点数默认上限 5000（设 0 可取消，大堆栈慎用）、保留 OpenCV 特有的
  对比度/边缘阈值、只输出平移（不含 ImageJ 默认 Rigid 模型的旋转）。
- 本工具的质量门控（最少内点、最大残差、有效帧对比例、最大插值间隙）
  默认放宽到非约束水平，与 ImageJ 的宽容度相当；两者失败处理的区别在于
  ImageJ 静默沿用上一帧模型，本工具按相邻可靠速度插值并在审计报告中警告。
  收紧阈值（如 RANSAC 3 px）可恢复严格门控与对称双峰冲突拒绝。
- 使用 RANSAC 仿射模型筛选空间一致的匹配内点，但最终位移只取内点平移量的
  中位数，不把缩放或旋转写入矫正。
- 跳帧匹配先换算成每帧速度；可靠帧对的真实阶跃和非匀速运动完整累计，
  只对质量门控失败的短缺失段按相邻可靠速度插值。
- 每个帧对记录特征数、匹配数、内点比例和中位残差，供质量门控与审计报告使用。
- 大规模特征匹配（≥512 描述子）使用 FLANN KD-tree 近似最近邻加速，
  小规模保持暴力匹配；FLANN 失败自动回退，行为总是确定。
- 矫正使用 OpenCV 线性平移（INTER_LINEAR），正确恢复原始 dtype；边界模式
  （`constant`/`reflect`/`nearest`）与填充值为核心 API 能力（`DriftCorrector`），
  GUI 固定使用默认零填充。

对低纹理、空白图、周期性晶格、强形变、旋转或多通道数据，自动配准可能不适用。
周期性 TEM 晶格尤其需要人工复核漂移曲线和结果图。

## 运行

```powershell
python -m pip install -r requirements.txt
python .\drift_correction.py
```

拖拽依赖为可选项；若 `tkinterdnd2` 或其原生库不可用，程序会自动回退至普通文件选择。

界面交互补充：普通模式下按住左键拖动预览可平移视野（配合滚轮缩放查看任意
区域，0/1:1 重置）；漂移曲线中橙色底纹为插值区段。输入/输出目录与全部参数
自动持久化到 `~/.drift_correction_v7.json`，重启后恢复。

## 建议流程

1. 选择输入目录和输出目录；建议输出到独立位置。
2. 加载 TIFF；若提示 C/Z 轴含义不明确，确认数据确实可按页视为时间序列；
   大堆栈会先给出内存预估确认。
3. 执行自动检测，检查可靠帧对比例、最大位移和任何插值警告（曲线中橙色
   底纹即插值区段）。
4. 在预览中手动检查配准（可缩放+拖拽平移）。需要定量分析时保留默认
   "共同有效区域裁剪"。
5. 保存当前文件或运行批处理；阅读 `<原名>_corrected_shifts.csv` 位移表与
   JSON/CSV 审计报告，复核出现警告的文件。

## 测试与构建

```powershell
python -m venv .venv-build
.\.venv-build\Scripts\python -m pip install -r requirements.lock -r requirements-build.lock
.\.venv-build\Scripts\python -m unittest discover -s tests -v
$env:PYTHONNOUSERSITE = "1"
.\.venv-build\Scripts\python -m PyInstaller .\TIFF漂移矫正工具v7.3.spec --clean --noconfirm
```

构建后必须运行测试，并确认打包程序可启动、可读写小型 TIFF、归档中包含
`drift_core` 和可选 tkdnd 文件。当前打包配置为 `TIFF漂移矫正工具v7.3.spec`
（产物 `dist\TIFF漂移矫正工具v7.3.exe`）；历史版本 spec 与 EXE（v7/v7.1/v7.2）
保留用于对照，**不要**用旧 spec 打包当前代码（版本资源与产物名均不匹配）。
请始终使用上述隔离环境构建，避免全局 Python/Conda 中的无关可选包被误收集进程序。

## 主要文件

- `drift_correction.py`：Tkinter/Matplotlib 图形界面及交互。
- `drift_core.py`：TIFF I/O、检测、矫正、批处理和后台任务核心。
- `tests/test_drift_core.py`：核心回归测试。
- `tests/test_drift_correction.py`：界面逻辑回归测试（无窗口）。
- `TIFF漂移矫正工具v7.3.spec`：当前版本 PyInstaller 打包配置；
  `v7.2.spec`、`v7.1.spec` 与 `v7.spec` 为历史版本打包配置，仅供对照。
- `requirements*.txt` / `requirements*.lock`：运行与构建依赖。

## 已知边界

- v7 面向二维单通道灰度堆栈（多页 TIFF）；通道容器会被拒绝，避免静默把通道当时间帧。
- 堆栈不设固定总大小上限；界面加载会在解码前按元数据给出内存预估确认
  （阈值为物理内存的 40%），批处理对超过物理内存 90% 的文件直接跳过并
  记录到审计报告。
- GUI 的加载与批处理通过后台线程执行；写出过程为流式，峰值内存明显降低，
  但检测阶段仍需整个堆栈驻留内存。
- 版本号单一来源为 `drift_core.__version__`；GUI 标题、pyproject、打包
  资源与审计报告均由其派生。

<!-- README-QUICKREF:BEGIN 由 tools/gen-readme-block.py 生成，请勿手工编辑本区块 -->

---

## 速查

| 项 | 内容 |
|---|---|
| 当前版本 | **7.3.0** |
| 版本来源 | `pyproject.toml` |
| 入口 | `drift_correction.py` |
| 依赖锁定 | `requirements.lock` |
| 许可证 | MIT |

**从源码运行**——必须使用本工具自己的虚拟环境，不要用 PATH 上的 `python`：
各工具依赖版本互不相同，共用解释器会互相污染。

```powershell
cd <本工具目录>
python -m venv .venv
.venv\Scripts\python -X utf8 -m pip install -r requirements.lock
.venv\Scripts\python drift_correction.py
```

**未纳入版本控制**：`.venv/`、`build/`、`dist/`、`release/`、`__pycache__/`、
测试数据与运行输出。具体规则见本目录 `.gitignore`。

<!-- README-QUICKREF:END -->
