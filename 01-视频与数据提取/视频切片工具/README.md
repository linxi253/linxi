# TEM 视频转 TIFF 堆栈工具 v4.4

这个程序只做一件事：批量读取视频，并把每个视频写成一个 ImageJ 原生兼容的
TIFF 时间堆栈——普通 TIFF 结构（II 头、魔数 42、永不 BigTIFF），带
ImageJ 1.53c 风格描述元数据，ImageJ/Fiji 直接 File → Open 即得带时间校准
的超 Stack，无需 Bio-Formats 导入。

## 默认行为

- 输入目录中的每个视频对应一个 `*_stack.tif`。
- 默认提取全部帧、保持源颜色、输出 8 位。
- 可选按目标 FPS 或时间间隔采样、转灰度/彩色、16 位（仅灰度）。
- 输出按源分辨率原样写入，不做任何缩放或裁剪；像素矩阵保持编码方向
  （不应用容器旋转元数据，检测到旋转会告警并记录）。
- 默认严格解码：FFmpeg 的任何解码错误都会让该视频显式失败，而不是
  静默跳帧后标记成功；抢救性提取可用 `--lenient-decode`（结果带缺帧风险告警）。
- 写入前统计真实帧数（顺带实测逐帧 PTS）并检查磁盘空间，写入后检查
  T 轴、宽高、通道和 dtype。
- 正式输出先在同一磁盘的 `.partial` 中完成，校验通过后原子提交。
- 已存在的正式输出不会被覆盖，避免误删实验数据；提交前还会复查一次，
  防止处理期间被其他任务抢注同名输出。
- 像素数据无压缩连续存放：约 3.8 GiB 以内写完整多页 IFD 链，任何标准
  TIFF 读取器可读；超过后自动切换单 IFD 结构（首页 IFD + 连续数据）。
  ImageJ 1.53c 可以正常读取该结构（已用官方 1.53c 验证）；普通
  File → Open 会把整个堆栈载入内存，几十 GiB 量级的大文件请留意内存
  占用，必要时改用支持虚拟栈的读取方式（如 ImageJ 的 Import → Raw
  或 napari/tifffile 按页读取）。

常见视频容器（MP4、WMV、AVI、MOV、MKV、MPEG、MTS/M2TS、WebM、MXF、
VOB、RM/RMVB、裸 H.264/H.265 等）会被扫描。最终可否读取取决于随包
FFmpeg 是否包含对应解码器；损坏、加密或私有编码文件不能保证解码。

## 输出与追溯

每个视频会产生：

```text
<原文件名>__<相对路径短哈希>_stack.tif
<原文件名>__<相对路径短哈希>_frames.csv
```

逐帧 CSV 记录：输出索引、输出文件、帧时间（`scheduled_time_s`，按首帧
归零）、源帧真实呈现时间（`source_pts_s`，实测可用时填写）、时间轴口径
（`timeline_source`：measured=逐帧实测 / model=采样栅格 / estimated=估算）、
流起始时间（`stream_start_time_s`）、输入路径与 SHA-256、源帧率/像素格式/
色彩空间/旋转角、采样参数。

每批任务还会在 `manifests/` 下产生 JSON 清单，记录 FFmpeg 路径与版本、
采样选项（含宽松解码开关）、每个视频的输入 SHA-256、探测信息、输出像素
规格、实际帧数、实际输出大小、逐视频告警列表（与界面日志同源）和错误
信息。同名视频用相对路径哈希消歧。

上批崩溃残留的”有 TIFF 无 CSV”孤儿输出会被移入 `.orphans/` 持久隔离区
重新生成；隔离区不会被自动清理，确认无需保留后可手动删除。

ImageJ/Fiji 直接打开 `*_stack.tif` 即可；napari、tifffile 等同样兼容。

## 为什么发布包仍然不是几 MiB

视频解码能力主要来自 FFmpeg，而不是 Python 界面。v4.0 把约 100 MiB 的
静态 FFmpeg 和约 100 MiB 的静态 ffprobe 各带了一份重复的编解码代码。
v4.1 改为两个小程序共用七个 FFmpeg DLL，并移除了只为 PNG/JPEG 序列服务
的 Pillow。仍保留 ffprobe，是因为结构化 JSON 探测比解析 FFmpeg 控制台文本
可靠，尤其是中文路径、多个视频流、可变帧率和不同编码器元数据。

共享运行时本身约 137.5 MiB；来源、许可证和逐文件校验值见
[`tools/ffmpeg/PROVENANCE.md`](tools/ffmpeg/PROVENANCE.md)。单文件发布经压缩后
约 80 MiB，运行时会在 Windows 临时目录展开所需文件。

## 从源码运行

要求 Python 3.11–3.13。仓库已经准备好经校验的 FFmpeg 8.1 共享运行时时：

```powershell
python -m pip install -r requirements.lock
python -m pytest
python -m video_extractor.app
```

无界面批处理：

```powershell
python -m video_extractor.cli --input <输入目录> --output <输出目录>
python -m video_extractor.cli --input <输入目录> --output <输出目录> --mode target_fps --value 12 --color grayscale
python -m video_extractor.cli --input <输入目录> --output <输出目录> --lenient-decode
```

批处理模式下 stdout 只输出最终结果 JSON 数组，可直接管道给其他程序解析；
进度与告警事件以 JSON 行输出到 stderr。退出码：0=全部成功，1=有失败，
130=用户取消（与失败区分）。`--version` 查看版本号。

输出格式参数已移除，因为本工具永远生成一个 ImageJ 兼容的 TIFF 堆栈。

## 打包

```powershell
.\scripts\build.ps1
```

构建脚本会安装锁定依赖、按 `tools/ffmpeg/PROVENANCE.md` 逐文件校验
FFmpeg 运行时 SHA-256（不匹配即拒绝构建）、运行完整测试，并在缺少任一
共享库、测试失败或 PyInstaller 失败时立即停止。发布物是单个
`dist/TEMVideoExtractor-v<版本>.exe`（版本号自动取自
`video_extractor/__init__.py` 的 `__version__`），内含 Python 与 FFmpeg 运行库；
可以把这一个 EXE 复制到桌面或其他电脑运行，不需要随附 `_internal` 目录。首次启动
会先将运行库解压到 Windows 临时目录，因此会比目录式版本多等待几秒。

## 数据保真约束

- 目标 FPS 高于源帧率会被拒绝，因为 FFmpeg 会复制帧并制造虚假时间分辨率。
- 8 位源选择 16 位输出会被拒绝，因为位复制不会产生新的信号精度。
- 16 位彩色输出会被拒绝，因为 ImageJ 彩色 TIFF 仅支持 8 位/通道；
  需要 16 位精度请使用灰度模式。
- 高位深源按 8 位输出时会明确告警（降采样损失精度）；9–14 位源按 16 位
  输出时会告警数值被线性重映射到 16 位满量程（非原始码值）。
- 严格解码为默认：损坏帧导致任务失败而非静默丢弃；宽松模式会带告警。
- 容器带旋转元数据时输出保持编码方向并告警，rawvideo 字节流与探测宽高
  严格一致，不会发生旋转后的静默转置。
- 时间轴以实测为准：count pass 通过 showinfo 捕获逐帧 PTS，”全部帧”模式的
  CSV 写入真实 PTS；帧间隔不均匀（真可变帧率）时不写固定 finterval，
  避免给 ImageJ 一个不存在的等间隔时间轴。
- 可变帧率视频按目标帧率采样时提示源帧间隔空档会复制前帧；流级帧率
  检测漏检时，实测 PTS 会补发同样的告警。
- 视频流晚于容器起始（如前置音轨）时，采样起点锚定在流的真实首帧 PTS，
  不会在首帧之前复制填充帧。
- 输入文件在哈希/统计/解码阶段之间被替换或追加时任务显式失败
  （大小 + 纳秒级 mtime 指纹校验），保证溯源哈希与输出内容一致。
- 灰度转换与 YUV 源的彩色转换都会根据源 `color_space`/`color_range` 选择
  可识别的矩阵和范围，不依赖 swscale 的分辨率启发式。
- 位深元数据缺失或不可信（例如打包总位深）时按像素格式推断有效位深，
  并在日志与任务清单中明确告警。
- 中文及其他 Unicode 路径使用二进制管道接收 ffprobe 的 UTF-8 JSON，不依赖
  Windows 当前 ANSI 代码页。
- 解码链尾部固定 `scale=W:H`：编码流中途改变分辨率时强制回到探测规格，
  防止 rawvideo 字节流按错误宽高静默转置。
- 解码停滞超过 120 秒（无任何帧输出）自动终止并报错，损坏输入不会把
  任务永久挂起。

<!-- README-QUICKREF:BEGIN 由 tools/gen-readme-block.py 生成，请勿手工编辑本区块 -->

---

## 速查

| 项 | 内容 |
|---|---|
| 当前版本 | **4.4** |
| 版本来源 | `video_extractor/__init__.py` |
| 入口 | `run_app.py` |
| 依赖锁定 | `requirements.lock` |
| 许可证 | MIT |

**从源码运行**——必须使用本工具自己的虚拟环境，不要用 PATH 上的 `python`：
各工具依赖版本互不相同，共用解释器会互相污染。

```powershell
cd <本工具目录>
python -m venv .venv
.venv\Scripts\python -X utf8 -m pip install -r requirements.lock
.venv\Scripts\python run_app.py
```

**未纳入版本控制**：`.venv/`、`build/`、`dist/`、`release/`、`__pycache__/`、
测试数据与运行输出。具体规则见本目录 `.gitignore`。

<!-- README-QUICKREF:END -->
