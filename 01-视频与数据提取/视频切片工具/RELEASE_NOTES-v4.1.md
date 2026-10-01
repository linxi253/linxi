# v4.1 / v4.1.1 修复说明与验收记录

## v4.1.1 单文件启动修复

v4.1 最初采用 PyInstaller 目录式发布。主 EXE 只有约 5 MiB，必须与
`_internal/python312.dll` 一起移动；用户把 EXE 单独复制到桌面时会在启动前
报 `Failed to load Python DLL`。v4.1.1 改为真正的 one-file 发布，将 Python、
FFmpeg 和所有 DLL 嵌入同一个 EXE。单独复制或移动这个 EXE 仍可启动。构建后
将 EXE 复制到一个没有 `_internal` 的独立中文路径执行 `--headless-smoke`，
Python 与随包 FFmpeg 均成功加载，进程退出码为 0。

## 根因

1. Windows 下旧代码让 `subprocess` 按系统 ANSI 代码页解码 ffprobe 输出。
   ffprobe JSON 实际是 UTF-8；路径中出现中文时，后台解码线程会失败，随后
   `communicate()` 得到 `None`，最终才冒出误导性的
   `the JSON object must be str, bytes or bytearray, not NoneType`。
2. 程序同时支持 ImageJ TIFF、OME-TIFF、TIFF/PNG/JPEG 序列，界面把容器、
   压缩和预览格式的兼容矩阵交给用户处理。很多正常输入因此在写入前被格式
   限制拒绝，尤其是超过 3.8 GiB 的 ImageJ TIFF。
3. v4.0 同时携带约 100 MiB 的静态 `ffmpeg.exe` 和约 100 MiB 的静态
   `ffprobe.exe`，两者重复包含大部分编解码代码。
4. 原测试没有覆盖高 DPI 窗口布局、单帧 OME-TIFF 的 T 轴压缩，以及宽度恰好
   为 3/4 像素的灰度图光度解释。

## v4.1 的处理方式

- ffprobe 全程使用二进制管道，成功后严格按 UTF-8 JSON 解码，并为无输出、
  非 UTF-8、无效 JSON、超时和非零退出码提供明确错误。
- 删除所有序列图片写入分支和 Pillow 依赖；产品只输出一个 OME-TIFF 堆栈。
- 先统计真实帧数，小文件用普通 TIFF，超过经典 TIFF 安全范围时自动使用
  BigTIFF，不再把 3.8 GiB 当作任务失败条件。
- 明确写入 OME 时间轴、轴顺序和光度解释；写后核对 OME、容器、shape、axes
  和 dtype。单帧 `SizeT=1` 按 OME 语义校验，不受读取库压缩单例轴影响。
- 保留同盘暂存、原子提交、取消清理、磁盘空间预检、输入 SHA-256 和逐帧清单。
- 改用经 SHA-256 复核的 FFmpeg 8.1 LGPL 共享构建，两个程序共用七个 DLL；
  打包规则保证运行库只出现一次。
- 操作按钮固定在窗口底部，解决高 DPI 下“开始处理”被日志框挤出的问题。

## 测量结果

| 项目 | v4.0 | v4.1 | v4.1.1 |
|---|---:|---:|---:|
| 交付体积 | 266.72 MiB 目录 | 198.61 MiB 目录 | 80.71 MiB 单文件 |
| FFmpeg 运行库重复份数 | 2 份静态代码 | 1 套共享代码 | 1 套共享代码 |
| Pillow/PIL 文件 | 有 | 0 | 0 |
| 用户可选输出类型 | 5 | 1 | 1 |

v4.1 比 v4.0 减少 68.11 MiB，约 25.5%。剩余体积中约 137.5 MiB 是覆盖常见
视频解码器的 FFmpeg 共享运行时；再大幅缩小只能改成依赖系统 FFmpeg，或裁剪
解码器范围，这会牺牲“解压即用”或输入兼容性。

## 验收

- 自动化测试：49 项通过。
- 打包自检：退出码 0。
- 发布包中 FFmpeg/ffprobe/DLL：9 个文件，各 1 份。
- `test` 中真实 MP4：GUI 输出 6 帧灰度 OME-TIFF，shape
  `(6, 1948, 1920)`，轴 `TYX`，像素范围 0–255。
- `test` 中真实 WMV：GUI 输出 1 帧灰度 OME-TIFF，OME XML 为 `SizeT=1`，
  像素范围 0–255。
- 两个结果均带 64 位输入 SHA-256、逐帧 CSV 和 schema 2 任务清单；正式输出
  后无 `.partial` 残留。
- 全帧预检：MP4 为 1,051 帧、RGB 原始体积约 10.98 GiB；WMV 为 12,685
  帧、RGB 原始体积约 132.56 GiB；两者均选择 BigTIFF。

v4.1 发布文件：`dist/TEMVideoExtractor-v4.1/TEMVideoExtractor-v4.1.exe`

SHA-256：`a3747b9a5adc987d3cbfbb7acf246cb77eef13c2199d2bfe3cb32edadafe95ff`

v4.1.1 单文件：`dist/TEMVideoExtractor-v4.1.1.exe`

SHA-256：`1415cc4df02a182344e1068631faa3b4e14bf8b275f04555d2888927f8295ca5`
