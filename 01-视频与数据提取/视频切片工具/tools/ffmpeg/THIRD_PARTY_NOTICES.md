# 第三方组件声明

本目录包含 BtbN 发布的 FFmpeg 8.1 LGPLv3 64 位共享构建。构建配置启用了
`--enable-shared --disable-static --enable-version3`，未启用 `--enable-gpl`。

- LGPLv3 附加条款：`LICENSE-LGPLv3.txt`
- GPLv3 基础条款：`LICENSE`
- 来源、构建变体和校验值：`PROVENANCE.md`
- 构建项目：https://github.com/BtbN/FFmpeg-Builds
- FFmpeg 源码：https://git.ffmpeg.org/ffmpeg.git

本程序通过独立子进程调用 FFmpeg/ffprobe。对外分发时必须同时保留许可证、
来源记录和可替换的共享库文件，并自行核对适用的 LGPLv3 及第三方许可证义务。

