# FFmpeg 运行时来源记录

- 版本：`n8.1.3-14-g330caae0c1-20261001`
- 下载并复核日期：2026-10-02
- 发布方：BtbN / FFmpeg-Builds
- 发布标签：`autobuild-2026-10-01-13-06`（**固定标签**）
  - 历史：初次钉住的 `autobuild-2026-08-21-13-40` 已被上游删除（2026-10-02 实测下载
    链接与 GitHub API 均返回 404）；2026-10-02 曾临时重钉到 `latest` 别名。
  - **2026-10-08 复核**：`latest` 别名已漂移（GitHub API 报出的归档 digest 与当时钉住
    的 `a8fbc540…` 不同），故改为钉住固定标签 `autobuild-2026-10-01-13-06` 下的同一构建。
    该固定标签发布于 2026-10-01T13:07:07Z，其归档 digest 与本文件当前钉住值一致。
- 构建变体：`win64-lgpl-shared-8.1`
- 发布页：https://github.com/BtbN/FFmpeg-Builds/releases/tag/autobuild-2026-10-01-13-06
- 归档：`ffmpeg-n8.1.3-14-g330caae0c1-win64-lgpl-shared-8.1.zip`
- 归档 SHA-256：`bf545d8fee9bb6957c1f3dea0f384bf64edead407d763326dbbd2de1b04768a4`
  - 2026-10-08 用固定标签归档复核：**归档整体 SHA-256 与 2026-10-02 记录的 `a8fbc540…` 不同**（打包/归档层面有差异），但**解包后 9 个运行时文件（`ffmpeg.exe`、`ffprobe.exe` 与 7 个 DLL）的 SHA-256 与已验收构建逐一相同**，见 `.review-tmp` 证据 `ffmpeg-tag-binary-check.json` 与 `ffmpeg-runtime-accepted.json`。
  - 归档中还含 `ffplay.exe`，本项目**明确不分发**该文件，故逐文件比对时它标记为不匹配属预期，不是 9 个运行时文件的差异。
- 上游 FFmpeg commit：`330caae0c1`

> 注意：本文件钉住的是**固定发布标签** `autobuild-2026-10-01-13-06`，不再依赖
> `latest` 别名。固定标签同样可能被上游删除：`.github/workflows/ci.yml`
> 的 FFmpeg 步骤对下载归档做强 SHA-256 校验，标签缺失或归档变化时显式失败，
> 提示先核实该标签/构建是否仍可获取——届时请复核来源、更新本文件
> （含逐文件哈希表与本地 `tools/ffmpeg/` 二进制），保持三方一致。

归档解压后仅保留 `ffmpeg.exe`、`ffprobe.exe` 及二者实际依赖的七个共享库；
不包含播放器 `ffplay.exe`。

## 文件 SHA-256

本节哈希用于核验自行下载并放入本目录的文件；运行时二进制因体积不入库
（被 `.gitignore` 排除），仓库内没有可供现行比对的副本，校验只能在下载后
按本表逐文件比对（构建脚本 `scripts/build.ps1` 会在构建时强制复验）。

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
