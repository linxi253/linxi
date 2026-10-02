# FFmpeg 运行时来源记录

- 版本：`n8.1.3-14-g330caae0c1-20261001`
- 下载并复核日期：2026-10-02
- 发布方：BtbN / FFmpeg-Builds
- 发布标签：`latest`（自动构建别名；此前钉住的 `autobuild-2026-08-21-13-40`
  已被上游删除，2026-10-02 实测下载链接与 GitHub API 均返回 404，故重钉到
  当时 latest 指向的构建）
- 构建变体：`win64-lgpl-shared-8.1`
- 发布页：https://github.com/BtbN/FFmpeg-Builds/releases/tag/latest
- 归档：`ffmpeg-n8.1-latest-win64-lgpl-shared-8.1.zip`
- 归档 SHA-256：`a8fbc540821511f2720763bb59aad8738cd2fc8b0e4677ea1795d4b6b6cc9b77`
- 上游 FFmpeg commit：`330caae0c1`

> 注意：`latest` 别名会随上游发布新构建而漂移。`.github/workflows/ci.yml`
> 的 FFmpeg 步骤对下载归档做强 SHA-256 校验，与本文件钉住的哈希不一致时
> 显式失败并提示重钉——届时请复核新构建、更新本文件（含逐文件哈希表与
> 本地 `tools/ffmpeg/` 二进制），保持三方一致。

归档解压后仅保留 `ffmpeg.exe`、`ffprobe.exe` 及二者实际依赖的七个共享库；
不包含播放器 `ffplay.exe`。

## 文件 SHA-256

| 文件 | SHA-256 |
|---|---|
| `ffmpeg.exe` | `1d93f786e551355d070f4622bf4cd0f1c990eb6769a1c1ab7b18cf0e638e9f97` |
| `ffprobe.exe` | `c55888785a631b4e72ec9a7970d9c62678a493d9650a2f92f47c1cf722eed569` |
| `avcodec-62.dll` | `e7f1183e43425b75a14a4a781f8d2cf29de9ac8238c890f464e7ddcd14633055` |
| `avdevice-62.dll` | `b7ff7864dd1dc7e82f26e0c5a65f54bad158eccde6c607005bdac5f6f7392f68` |
| `avfilter-11.dll` | `d854a2c4f7e7585f961fea1d3d55ea467afd858fd5d8ec89f58d7223279b1e74` |
| `avformat-62.dll` | `30062c7315ec600a97ee731dd05c2f3d8d701d60874541dc0338b8731523c4ff` |
| `avutil-60.dll` | `88befd1b2f4f07ec73617fa75cfc5e8175915f70a69e58115c1e1d4ee0035a1e` |
| `swresample-6.dll` | `8d71f620d188e18aadf931550c8867db8969be015221e7d3c705f72d6c192a87` |
| `swscale-9.dll` | `73175a6691158584aee3fd0dc93726fa66e428b41830e1a10adc0d022821bdf7` |
