# FFmpeg 共享运行时

v4.1 使用经校验的 FFmpeg/ffprobe 8.1 共享运行时。与两份约 100 MiB 的静态
可执行文件相比，两个小型程序共用七个 `libav*`/`sw*` DLL，可显著减少重复体积，
同时继续使用 ffprobe JSON，不靠解析易变的控制台日志。

构建前必须存在：

```text
ffmpeg.exe
ffprobe.exe
avcodec-62.dll
avdevice-62.dll
avfilter-11.dll
avformat-62.dll
avutil-60.dll
swresample-6.dll
swscale-9.dll
LICENSE
LICENSE-LGPLv3.txt
PROVENANCE.md
THIRD_PARTY_NOTICES.md
```

替换运行时时必须重新核对归档和单文件 SHA-256、执行两个程序的 `-version`，
并完整运行测试。`tem_video_extractor_v4.spec` 和 `scripts/build.ps1` 会在共享库
不完整时拒绝构建。

