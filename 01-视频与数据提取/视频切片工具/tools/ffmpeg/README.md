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

## 本地跑集成/回归测试的前提

`tests/test_integration.py` 与 `tests/test_regression.py` 依赖本机可解析到的
真实 FFmpeg/ffprobe ≥8.0.3 二进制对（两个可执行文件需版本一致）。本地仓库
不含 `tools/ffmpeg/*.exe` 时，先把 FFmpeg 8.0.3+ 的 `ffmpeg.exe`、`ffprobe.exe`
放进本目录（或装好可被 PATH/imageio_ffmpeg 解析到的同版本对）再跑 pytest；
缺二进制时这两份用例会自动 SKIP（见 `tests/conftest.py`），不会报 ERROR。

