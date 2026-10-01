# Distribution notice

Copyright (c) 2026 AIforTEM.

本目录（HRTEM/STEM TIFF 图像增强工具）采用 **MIT** 许可证，与仓库根
[LICENSE](../../LICENSE) 一致。许可证适用范围与例外见根
[NOTICE.md](../../NOTICE.md)。

本工具为离线程序，不联网、不执行外部命令。

## 发布产物

`scripts/build_release.ps1` 生成发布产物，并在 `release/SHA256SUMS.txt`
登记各产物校验和。未提供代码签名证书时，脚本会把产物明确标记为
`NotSigned`（不影响 MIT 授权，仅表示未做数字签名）。

安全策略与漏洞报告方式见 [SECURITY.md](SECURITY.md)。
