# TIF 图像滤镜处理工具 v1.2.0

面向 TIFF 单图、RGB 图像和多页堆栈的交互式滤镜工具。可同时预览原图与结果，并对单文件或整个目录批量处理。

## 直接运行

本仓库不含构建产物（无 `dist/` 目录），历史上分发的 `dist/TIF_FilterTool.exe` 与
`dist/TIF_FilterTool-v1.2.0.exe`（同一版本，无版本文件名保留给已有快捷方式使用）均未随仓库分发。
需要 exe 请用项目内环境自行打包：

```powershell
.venv\Scripts\python -m pip install pyinstaller
.venv\Scripts\python -m PyInstaller TIF_FilterTool.spec
```

## 批量处理 TIFF 堆栈

1. 点击“打开文件（多选）”选择一个或多个 `.tif/.tiff`；也可“打开文件夹”加载该目录下的 TIFF（不递归子目录）。
2. 使用上一帧/下一帧，或输入帧号后点击“跳转帧”，查看堆栈中的任意帧并调整滤镜。预览只显示当前帧。
3. 选择“输出文件夹”。
4. 点击“处理当前堆栈全部帧”，将当前文件的**全部帧**按当前参数处理并写为一个完整 TIFF；点击“批量处理文件列表”，则对列表中每个文件执行同样操作。
5. 进度区显示文件序号、当前堆栈帧数和全部任务的总帧进度。“取消处理”会清理正在生成的堆栈，保留已经完成的文件。

键盘快捷键：`←/→` 切换帧，`PgUp/PgDn` 切换文件，`Ctrl+O` 打开文件，`Ctrl+Shift+O` 打开文件夹，`Ctrl+S` 处理保存当前文件。

每个输出保留帧顺序、尺寸、位深和支持的元数据，并生成 `.filter.json` 参数清单。
滤镜逐帧执行，不把整个堆栈载入内存，也不逐帧自动拉伸保存数值。
已有同名输出或参数清单会自动避让；某个文件失败时会记录错误并继续后续文件，
全部失败明细同时追加写入输出目录的 `TIF_FilterTool-errors.log`。

支持本次目标样例 `Aligned 261 of 261.tif` 的 ImageJ `ZYX`、261 帧、640×639、uint8 布局，
以及 `TYX` 的单通道 OME 时间堆栈。ImageJ 的二进制 Info、Labels 等标签会重新编码后保存，
同时保持输入字节序，修复含这类标签的 ImageJ 堆栈保存时的 `struct.error`。

## 参数复现（载入参数清单）

工具栏的“载入参数清单”可打开任意 `.filter.json`（含批处理输出目录中的清单），
把保存时的全部滤镜参数恢复到滑块上，直接用于新文件——这就是“保存清单便于复现”的
完整闭环。无效清单（未知参数、越界数值、结构错误）会被明确拒绝且不改动当前参数。

## 从源码运行

需要 Python 3.9 及以上（开发环境实测 3.10）。

```powershell
python -m pip install -r requirements.txt
python .\main.py
```

## 可调滤镜

共 16 项：高斯模糊、USM 锐化、白平衡、色温、色调、曝光、对比度、高光、阴影、白色、黑色、纹理、清晰度、去除薄雾、自然饱和度和饱和度。

除高斯模糊、USM 和白平衡外，多数参数范围为 `-100` 到 `100`。默认值为 0，表示不改变图像。
灰度文件加载后，色彩类滑块（白平衡/色温/色调/自然饱和度/饱和度）自动禁用。

滤镜按固定顺序应用，与界面分组顺序不同：白平衡与色彩 → 影调 → 质感 → 饱和度 → 高斯模糊 → USM 锐化（锐化固定收尾）。顺序定义见 `image_filters.py` 的 `PROCESS_ORDER`，全部强度系数集中为该文件顶部的命名常量。

## 处理空间与定量提示

滤镜在**存储值 [0,1] 空间**运算（8/16/32 位整型精确归一化），既不是线性光空间也不是光密度（OD）空间。对显微计数类数据请注意：

- 曝光在计数空间做乘法，对线性采集数据物理含义正确；
- 对比度、高光/阴影等滤镜的枢轴与边界常数沿袭摄影习惯（如中点 0.5），对背景接近 0 的图像行为可能不对称；
- 去雾与白平衡对"大面积黑背景 + 小样品"的图像可能失真；
- 保存清单（`*.filter.json`）记录全部参数与原始结构信息，便于复现。

## 输入与输出

- 输入：`.tif/.tiff`，支持灰度、RGB/RGBA、8/16/32 位和多页堆栈（单 series）。
- 输出：保持原文件结构与数据类型，写入用户选择的独立输出目录。选择输出目录时
  即探测写权限（网络共享/只读盘会立即提示），批处理启动前再次校验。
- 若输出目录与源目录冲突，程序会在文件名后追加 `_processed`，避免直接覆盖。
- LZW、PackBits 等常见无损压缩的读写依赖 `imagecodecs`，已包含在 `requirements.txt`。
- 目前只处理单 series TIFF；多页文件中若某页的 photometric/compression 与首页不一致，会明确拒绝而不是静默改写。
- 每次保存同时写出参数清单 `输出文件名.filter.json`（schema v2）：
  - `source.sha256`：源文件内容哈希，可验证源自处理以来未被替换；
  - `duration_seconds` / `output_size_bytes`：处理耗时与输出体积；
  - `tiff.skipped_tag_codes`：未复制进输出的标签明细（如无法安全回写的
    FEI 私有标签、子 IFD 指针），与 `original_tag_codes` / `copied_extra_tag_codes`
    一起构成完整的元数据去向记录。

## 已知限制

- 多 series TIFF（含缩略图/预览页）被拒绝，避免丢失附加图像。
- 通道按页存储的布局（OME 的 `CYX`、ImageJ 的 `TCZYX` 等无页内 S 轴写法）不支持，会明确报错；页内含 S 轴的 OME RGB（`CYXS`）正常支持。
- 灰度+alpha（samples=2）文件不支持。
- 无法忠实回写的私有元数据标签（如 FEI_HELIOS 的结构化字典）会从输出中省略，
  并记录在清单的 `skipped_tag_codes` 中；每页级标签（如 PageName）仅保留首页副本。
  源文件永远不被修改。
- 帧序沿用 tifffile series 的页序（等价于 axes 声明 shape 的 C 序展开），
  本工具不单独重排；非标准 writer 若页序非线性展开会被当作 C 序处理。
- 超大单帧（如 20000×20000 16 位）内存峰值可达数 GB（整帧浮点转换 + 滤镜
  中间量），加载时状态栏会给出估算提示；帧读取与预览计算均在后台线程执行，
  界面保持响应。堆栈文件按页流式处理不受影响。
- MINISWHITE（0=白）图像读取时按亮度反转处理，写出时反转回来并保留原 photometric。

## 主要文件

- `main.py`：Tkinter GUI、文件导航、后台预览和批处理。
- `image_filters.py`：滤镜实现、参数定义、处理管线与强度系数常量。
- `tif_io.py`：TIFF 读取、结构识别、保持数据类型的写出与参数清单（schema v2）。
- `test_core.py`：核心滤镜和 TIFF I/O 测试（含 FEI/OME 兼容性与清单溯源 fixture）。
- `test_packaging.py`：打包 hook 测试。
- `test_gui.py`：GUI/预览/滚轮的冒烟测试。
- `test_batch.py`：261 帧、ImageJ 元数据、多选/跳帧/取消、参数回读与批量错误处理回归测试。
- `conftest.py`：pytest 兼容层（会话级生成测试图像；官方入口仍为逐文件运行）。
- `validate_samples.py`：对用户指定的真实样例执行全栈处理、逐像素验证和源文件哈希核对。
- `version.py`：界面、参数清单和 EXE 属性共用的版本号。
- `test_images/`：自动生成的测试输入（不入库）。
- `requirements-dev.txt`：构建打包专用依赖。
- `build.bat`、`build_release.py`、`TIF_FilterTool.spec`：Windows 测试、打包与校验和生成。
- `dist/TIF_FilterTool.exe`：已打包程序（构建产物，未随仓库分发）。

## 验证与打包

```powershell
python -m pip install -r .\requirements-dev.txt
python -X utf8 .\build_release.py
```

构建先运行全部四组回归测试，再生成上述两个 EXE 和 `dist/SHA256SUMS.txt`。
也可运行 `build.bat`，它优先使用已有的 `.venv-build` 构建环境。
安装了 pytest 的环境也可以直接 `pytest`，`conftest.py` 会自动生成测试图像，
无显示环境自动跳过 GUI 用例。

大样例验收（可选，会在指定输出目录保留处理结果和 `validation.json`）：

```powershell
python -X utf8 .\validate_samples.py --output-dir .\release\validation "堆栈.tif" "样例文件夹"
```

验收使用曝光 20、对比度 10，对每一帧的所有像素核对期望滤镜结果，检查结构、
ImageJ 信息及源文件 SHA-256。这里只是固定的测试参数，不是推荐的实验处理参数。

核心测试的输出写入临时目录并在结束后自动清理。连续叠加多个增强滤镜可能放大噪声或改变定量强度。建议一次只调整少量参数，并保留参数清单。

<!-- README-QUICKREF:BEGIN 由 tools/gen-readme-block.py 生成，请勿手工编辑本区块 -->

---

## 速查

| 项 | 内容 |
|---|---|
| 当前版本 | **1.2.0** |
| 版本来源 | `version.py` |
| 入口 | `main.py` |
| 依赖锁定 | `requirements.lock.txt` |
| 许可证 | MIT |

**从源码运行**——必须使用本工具自己的虚拟环境，不要用 PATH 上的 `python`：
各工具依赖版本互不相同，共用解释器会互相污染。

```powershell
cd <本工具目录>
python -m venv .venv
.venv\Scripts\python -X utf8 -m pip install -r requirements.lock.txt
.venv\Scripts\python main.py
```

**未纳入版本控制**：`.venv/`、`build/`、`dist/`、`release/`、`__pycache__/`、
测试数据与运行输出。具体规则见本目录 `.gitignore`。

<!-- README-QUICKREF:END -->
