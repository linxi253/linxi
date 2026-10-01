# v4.3 修复说明与验收记录

本版本按用户需求把输出格式从 OME-TIFF（超过 3.8 GiB 自动 BigTIFF）
整体切换为 **ImageJ 原生兼容的普通 TIFF**。动机：ImageJ 1.53c 打开
OME-TIFF 时按 OME XML 解析，而目标工作流要求普通 TIFF（II/MM 头、
魔数 42、非 BigTIFF）+ ImageJ 1.53c 描述元数据 + 全部帧不截断。
这是输出格式的破坏性变更，故版本号从 4.2 升到 4.3。

## 输出格式

- **普通 TIFF 结构，永不 BigTIFF**：小端 II 头、魔数 42；像素数据无压缩
  连续存放。
- **ImageJ 1.53c 描述元数据**：首页 IFD 的 ImageDescription 注入
  `ImageJ=1.53c`、`images=N`、`frames=N`、`hyperstack=true`、`loop=false`，
  时间校准写 `finterval=`（秒）与 `fps=`。ImageJ/Fiji 直接
  File → Open 得到带时间校准的 hyperstack。
- **任意大小、全部帧**：约 3.8 GiB 以内写完整多页 IFD 链（任何标准 TIFF
  读取器可读）；超过后自动切换单 IFD 截断式结构（只写首页 IFD、数据
  连续），ImageJ 按描述中的帧数以虚拟栈按需读取，几十 GiB 量级可用。
  这与 tifffile 的 ImageJ 大文件约定一致；>4 GiB 的经典 TIFF 本无标准
  读取器，因此截断式不损失兼容性。
- **文件名**从 `<视频名>_stack.ome.tif` 改为 `<视频名>_stack.tif`：
  `.ome.` 后缀会误导 ImageJ/Bio-Formats 按 OME 解析。

## 选项调整

- **位深默认 8 位**（此前默认"保持原始"）：匹配 ImageJ 工作流的目标
  规格。"保持原始"与 16 位（仅灰度）仍可手动选择。
- **移除压缩选项**：tifffile 明确 ImageJ 格式不支持压缩
  （"ImageJ hyperstacks do not support BigTIFF or compression"），
  Deflate 输出会生成 ImageJ 无法原生打开的文件，因此删除无压缩以外的
  全部压缩路径。GUI 下拉与 CLI `--compression` 一并移除。
- **分辨率不强制**：输出始终按源视频原生分辨率，无缩放、无校验。

## 随格式切换的科学性约束

- **16 位 + 彩色输出组合被显式拒绝**（ImageJ 彩色 TIFF 仅支持
  8 位/通道），错误信息引导改用 8 位彩色或 16 位灰度。
- **高位深源（10/12/16 位）按 8 位输出时发降采样告警**，损失精度不再
  静默发生。

## 其余改进（沿自 v4.2 的审查修复，此处不变）

目标帧率上采样拒绝、NaN/Inf 采样值校验、YUV→RGB 色彩矩阵声明、位深
元数据缺失按像素格式推断、VFR 目标帧率采样告警、提交前 fsync、原子
提交 + 孤儿回收、CLI stderr/stdout 分流、输入 SHA-256 溯源、逐帧 CSV
清单全部保留。

## 兼容性

- v4.1/v4.2 生成的 OME-TIFF 输出仍在磁盘上不受影响；新输出后缀不同，
  不会与旧输出发生覆盖冲突。
- job manifest `schema_version` 2→3：`options` 不再含 `compression` 键，
  `tiff_container` 恒为 `ImageJ-TIFF`；结果新增 `truncated` 事件标志。
- 依赖旧 `--compression` 参数的脚本需移除该参数。

## 验收记录

- 全量测试 66 项通过（新增 ImageJ 格式断言：`is_imagej`、版本串、
  finterval/fps、II 小端、页结构、截断阈值两侧行为、单帧 T 轴压缩、
  RGB 光度解释；删除 DEFLATE 与 BigTIFF 相关测试，DEFLATE 端到端改为
  彩色端到端）。
- 写入器以 `tifffile.memmap(..., imagej=True, bigtiff=False,
  truncate=<按需>)` 实现，写入后由 tifffile 读回校验容器类型、描述
  版本、页结构、T/Y/X/C 维度与 dtype。
- `scripts/build.ps1` 完整出包：锁定依赖安装 → 全量测试 → PyInstaller
  单文件构建 → SHA-256 记录 → `--headless-smoke` 冒烟。
