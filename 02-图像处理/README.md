# 图像处理

本分类收集 TIFF 堆栈漂移矫正、HRTEM/STEM 去噪增强、离域效应去除和通用图像滤镜工具。

## 当前项目

- [drift-correction-v7](drift-correction-v7/README.md)：漂移矫正 v7（合并版），
  由 v5.2 安全框架与 v6.1 纯平移算法合并而来，源码、测试与打包配置齐全。
- [hrtem-HRTEM滤波工具](hrtem-HRTEM滤波工具/README.md)：Kilaas/Mitchell HRTEM/STEM 滤波、DigitalMicrograph 脚本和 ImageJ 宏。
- [stem-optimize-STEM图像优化](stem-optimize-STEM图像优化/README.md)：模块化 HRTEM/STEM 图像增强工具 v2.1.0。
- [图像加滤镜工具](图像加滤镜工具/README.md)：16 项可组合调节的 TIF 图像滤镜工具。
- [离域效应去除工具](离域效应去除工具/README.md)：手绘 ROI 抑制液相 HRTEM 晶体边缘的
  离域条纹/白色虚影，支持多边形/手绘/矩形/椭圆与 FFT 选带，配套单元测试与打包配置。

## 版本选择

漂移矫正的 v5.2 与 v6 已合并为 v7，旧版目录归档在开发机的
`<仓库根>\08-历史版本\`（v5.2 目录名带 `-20260731` 后缀，v6 同名加日期）；
该归档未随本仓库分发（仓库内不存在此目录）。新数据处理请直接使用 v7。
