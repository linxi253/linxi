# HRTEM/STEM 图像增强工具 v2.1

面向二维灰度 HRTEM/STEM TIFF 单帧和堆栈的安全桌面处理工具。
核心算法是“局部纹理调制的自适应频域降噪”，不使用 PSF/CTF，因此不宣称
物理 Wiener 反卷积。

## v2.1 的数据安全保证

- 输入和输出同路径会被拒绝，包括大小写差异、相对路径和硬链接。
- 输出只写入同目录临时文件；全部帧完成、逐页解码校验并落盘同步后，用一次原子替换发布目标。任一时刻目标路径都是完整文件，替换失败时旧输出原样保留。
- TIFF 发布后 `.stem.json` 紧随其后原子发布（含重试）；若 sidecar 仍发布失败，旧 sidecar 不会被改动，程序会明确报告新 TIFF 已发布、sidecar 未更新。
- 取消、算法错误、磁盘写入错误和退出都不会发布半成品。
- 任一帧失败会终止整个事务，不会静默写原帧、零帧或错误转换的帧。
- uint8/uint16 使用每一帧的精确直方图估计全堆栈百分位范围。
- 预览和正式导出共用同一范围、同一处理函数。
- OME-TIFF 会按新 dtype、shape 和 axes 重建 OME-XML；ImageJ 元数据按安全字段重建。
- 保留 TIFF X/Y Resolution 和 ResolutionUnit。
- 每次成功输出生成 `<输出文件>.stem.json`，记录版本、参数、SHA-256、范围和裁剪率。
- 处理前后分别校验输入 SHA-256；处理中原文件变化会使整个事务作废。

## 支持范围

支持：

- 单系列 TIFF / BigTIFF / OME-TIFF / ImageJ TIFF；
- 每页二维、单通道、同尺寸、同 dtype；
- uint8、uint16及有限值整数/浮点灰度数据；
- 多页堆栈和 OME `T/Z/Q + YX` 轴；
- imagecodecs 提供的 LZW、Deflate、PackBits、JPEG、JPEG2000、Zstd等压缩。

主动拒绝：

- RGB/RGBA、多通道 `C/S` 轴；
- 多系列 OME/ImageJ；
- 帧尺寸或 dtype 不一致；
- NaN/Inf、复数、空图；
- 超过内存安全阈值的任务；
- 预计剩余磁盘空间不足以安全容纳未压缩临时输出的任务；
- 非 `.tif`/`.tiff` 输出。

## 运行

推荐直接运行经过测试的 v2.1 发布文件：

```text
release/STEM图像优化工具v2.1.exe
```

从源码运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements-win-py310.lock
.\.venv\Scripts\python.exe .\main.py
```

日志写入：

- 源码模式：项目目录 `stem_enhancer.log`；
- 打包模式：`%APPDATA%\STEM_Enhancer\stem_enhancer.log`。

日志最大 5 MiB，保留 3 个滚动备份。

## 安全使用流程

1. 打开二维灰度 TIFF；程序会先检查全部页面结构。
2. 调整 K 因子、最大混合比例和频谱背景平滑参数。
3. 预览时首次扫描全部帧，之后缓存同一强度范围。
4. 选择与输入不同的 `.tif`/`.tiff` 输出路径。
5. 开始处理；旧输出在新文件通过完整解码校验前保持不变。
6. 检查完成摘要和 `.stem.json` 中的裁剪率、参数与 SHA-256。

取消后临时文件会清理，既有输出不会改变。窗口退出时会等待当前 FFT 返回并安全
清理事务，不会强制杀死写入线程。

为保证范围一致性和可验证性，大堆栈会经历全栈范围扫描、处理、输出逐页解码以及
处理前后输入哈希等多个顺序读阶段。在网络盘上可能明显变慢；建议先复制到本地
SSD，并为未压缩临时输出预留充足空间。

## CLAHE 与定量分析

CLAHE 默认关闭。它会根据局部直方图非线性改变灰度；同一原始灰度在不同帧或区域
可能映射成不同输出，不能用于保持帧间定量可比性。

16-bit 只意味着保留更多离散灰阶，不代表保持原始强度标定。所有归一化、频域滤波
和 CLAHE 都会改变灰度分布。定量研究必须保留原始数据、关闭 CLAHE，并结合
`.stem.json` 检查映射范围和裁剪率。

## 测试

```powershell
python -m unittest discover -s tests -v
ruff check .
```

建议启用本地提交防线，每次 commit 自动执行静态检查与回归测试：

```powershell
# 在仓库根执行：core.hooksPath 的相对路径按仓库顶层解析，
# 必须写成带子项目前缀的路径，只写 .githooks 会指向不存在的 <仓库根>/.githooks
git config core.hooksPath "02-图像处理/stem-optimize-STEM图像优化/.githooks"
```

60 项测试覆盖同路径保护、预取消/处理中取消、发布回滚、处理中输入变化、
预览后签名比对、目录不可写预检、OME dtype/axes、四维 OME、ImageJ 元数据
校验、描述标签保留、LZW、分离页堆栈、分辨率标签、RGB 拒绝、浮点/NaN、
极端幅值尺度不变性、小图、常量堆栈、磁盘余量、恶意 XML 实体、预览映射、
uint16 显示转换、CLAHE 输出、溯源环境记录、残留临时文件发现、进度事件、
缓存上限和合成晶格降噪。

## 构建

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_release.ps1
```

构建脚本会创建隔离环境、安装带哈希运行时依赖、运行审计/静态检查/测试、执行
PyInstaller、运行成品内置 LZW/OME/事务自检，并生成
`release/SHA256SUMS.txt`。如果提供代码签名证书指纹：

```powershell
.\scripts\build_release.ps1 -CertificateThumbprint "<thumbprint>"
```

没有私钥证书时无法生成可信数字签名；脚本会明确把产物标记为 `NotSigned`。

## 项目结构

- `main.py`、`main_window.py`：入口、日志和 GUI 状态机。
- `pipeline.py`：预览/导出共享处理管线。
- `filters.py`：自适应频域降噪和有界半径缓存。
- `contrast.py`：全堆栈范围、线性位深映射、可选 CLAHE。
- `tiff_handler.py`：输入预检、元数据提取、流式 TIFF 写入和输出校验。
- `worker.py`：原子输出事务、取消、SHA-256和溯源。
- `tests/`：标准库 `unittest` 回归测试。
- `docs/`：算法、格式和发布说明。
- `tools/gen_pt_slab.py`：独立、参数化的 VASP slab 工具，不属于图像应用。
- `legacy/`：旧可执行程序、旧反编译资源和失效生成文档，仅供追溯。

详细变更见 `CHANGELOG.md`，安全策略见 `SECURITY.md`。

<!-- README-QUICKREF:BEGIN 由 tools/gen-readme-block.py 生成，请勿手工编辑本区块 -->

---

## 速查

| 项 | 内容 |
|---|---|
| 当前版本 | **2.1.0** |
| 版本来源 | `version.py` |
| 入口 | `main.py` |
| 依赖锁定 | `requirements-win-py310.lock` |
| 许可证 | MIT |

**从源码运行**——必须使用本工具自己的虚拟环境，不要用 PATH 上的 `python`：
各工具依赖版本互不相同，共用解释器会互相污染。

```powershell
cd <本工具目录>
python -m venv .venv
.venv\Scripts\python -X utf8 -m pip install -r requirements-win-py310.lock
.venv\Scripts\python main.py
```

**未纳入版本控制**：`.venv/`、`build/`、`dist/`、`release/`、`__pycache__/`、
测试数据与运行输出。具体规则见本目录 `.gitignore`。

<!-- README-QUICKREF:END -->
