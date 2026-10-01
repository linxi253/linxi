# 视频与数据提取

本分类包含视频转 TIFF 工具，以及随工具发布的 FFmpeg Windows 发行包。

## 子目录

- [视频切片工具](视频切片工具/README.md)：TEM 视频帧提取工具（v4.4）；采用显式 FFmpeg、时间轴采样、原子输出和任务清单。旧 v3.1 已归档。
- [视频切片工具/tools/ffmpeg](视频切片工具/tools/ffmpeg/PROVENANCE.md)：随工具发布的 FFmpeg 8.1.2 共享构建（BtbN FFmpeg-Builds），来源与校验值见 `PROVENANCE.md`。

## 未入库的本地资料

以下内容只存在于本地工作副本，未提交到本仓库：

- `ffmpeg-8.0.1-essentials_build/`：旧版 FFmpeg 8.0.1 归档备份（v4 工具要求 8.0.3+，实际随工具发布的是 8.1.2），以及同名 `.7z` 压缩包。第三方二进制发行包体积大，需要时请从 FFmpeg 官方或 BtbN 构建页获取。
- `XRD计算/`：XRD 数据整理与 Origin 绘图操作录屏（`.mp4`），属教学资料且体积较大。
