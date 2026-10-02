# HRTEM/STEM 滤波工具 v5

这是基于 Kilaas/Mitchell 频域流程的 HRTEM/STEM 滤波工具。v5 的重点是
**科研数据完整性、可重复处理和安全保存**，而不是把结果默认转成显示图片。

## 重要行为

- 默认输出为 `float32` TIFF，不进行逐帧归一化。
- 输入 TIFF 永远不能被当作输出覆盖。
- 堆栈写入先使用同目录临时文件；取消或错误不会发布正式输出。
- 每次成功处理会产生 `output.tif.processing.json`，记录参数、入口（GUI/CLI）、
  FFT 线程数、运行平台与输入完整性签名；`--input-hash` 可额外记录输入的
  SHA-256（归档级标识，需对输入做一次完整读取）。
- STEM 十字掩膜的预览、保存和输出文件名使用同一个结果键。
- 任意矩形灰度图或 ROI 都会处理后裁剪回原始尺寸。
- 输出 TIFF 的分辨率标签写入每一页，逐页读取标定的工具不会在后续页丢失像素标定。

`uint8-display` 仅用于展示。它使用整个堆栈统一的显示范围，不能用于定量
强度比较。

## 已知限制

- 记录先于 TIFF 发布（保证成功输出必有配对记录）。若进程在 JSON 已替换、
  TIFF 尚未替换之间被硬杀（断电/任务管理器），磁盘上会留下旧 TIFF 配新
  记录的组合；可捕获的失败会自动回滚，此窗口仅存在于不可恢复的硬崩溃。
- STEM 十字线长度按图像尺寸比例缩放（XH Ro 为尺寸比例），而中心孔半径是
  绝对像素。小尺寸 ROI（约 < 256 px）上十字线可能被中心孔吞掉而几乎不起
  作用；GUI 会在该情形下显式警告，全图处理不受影响。

## 安装

建议使用单独的 Python 3.10–3.14 虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`imagecodecs` 使 tifffile 能读取常见的 LZW、JPEG、JPEG2000 等压缩 TIFF。

## 图形界面

```powershell
python .\butter.py
```

或：

```powershell
python .\hrtem.py
```

打开文件后，程序只显示原图和处理规模，不会自动运行全图 FFT。请先框选 ROI、
确认输出模式，再点击“预览”或“处理并保存”。

界面辅助行为：

- 框选 ROI 后状态栏即时给出该 ROI 的峰值内存估计。
- 单次预览/处理的峰值内存估计超过约 4 GiB 时会先弹出确认对话框。
- 修改任何参数后，已显示的预览标题会标注“参数已修改，预览可能过期”。
- “FFT 线程”输入框对应 CLI 的 `--fft-workers`（大图/多帧可调高）。
- Delta=0 时“主输出/包络/STEM”控件一并禁用，反映纯 Butterworth 强制模式。

## 命令行

```powershell
python .\hrtem.py input.tif output.tif --mode wiener
python .\hrtem.py input.tif output.tif --mode wiener --stem
python .\hrtem.py input.tif display.tif --encoding uint8-display
python .\hrtem.py input.tif output.tif --roi 100 200 1124 1224
python .\hrtem.py input.tif output.tif --input-hash --fft-workers 4
python .\hrtem.py input.tif --inspect
```

可用模式为 `wiener`、`absf`、`butterworth`。当 `--delta 0` 时，程序明确切换为
纯 Butterworth 模式，不会静默导出原图。单帧峰值内存估计超过约 4 GiB 时 CLI
会先在 stderr 给出警告。

## 支持范围

v5 目前安全支持：

- 单帧灰度 TIFF。
- 每页均为同尺寸、同位深 `YX` 灰度图的多页堆栈。
- `uint8`、`uint16`、`int16` 和浮点灰度数据。

彩色 `CYX`/`SYX`、调色板（palette）页面、多样本页（如灰度+alpha）、单页多维数组、
尺寸或类型不一致的页面会被明确拒绝，避免把颜色/样本维度或色表索引误当作时间
或 Z 堆栈。后续扩展 OME-TIFF axes 时必须新增对应测试，不能依靠自动猜测。

## 算法模式

- `fast_radial_bin`：高效的整数径向分箱；这是 v5 的默认模式。
- `dm_compatible`：512 角度采样与双线性插值，更接近随附 DigitalMicrograph v4
  脚本的旋转平均采样方式。

两种模式的名称是刻意区分的；在获得更多 DM golden images 前，不能把二者都
宣称为“完全一致”。在一个 128×128 固定合成图上，v5 `fast_radial_bin` 与归档 v4
Python 实现的 Wiener 输出 NRMSE 为约 `1.41e-5`。

注意 Butterworth“阶数”沿用 DM 参考实现的约定，而非教科书定义：幅值在
`zero_radius` 处为 `1/√2`（半功率），但渐近衰减为 `r^(-2n)`——即本工具的
order 4 相当于教科书 Butterworth 的 order 8。与 DM 结果对齐时无需换算，
与其他软件对比时需要留意。

## 开发与测试

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -v
python -m py_compile butter.py hrtem.py src\hrtem_filter\*.py
```

测试覆盖矩形尺寸、ROI、STEM 输出选择、NaN 拒绝、8 位堆栈显示范围、源类型裁剪
导出、元数据、原子保存、取消、拖拽矩形钳制，以及 `fast_radial_bin` /
`dm_compatible` 两种模式的算法回归样本。lint 规则集在 `pyproject.toml` 的
`[tool.ruff.lint]` 中显式固定：

```powershell
python -m ruff check src tests butter.py hrtem.py cli_launcher.py
```

## 打包

当前只保留两个可维护的 PyInstaller 配置：

```powershell
pyinstaller .\HRTEM_Filter_GUI.spec
pyinstaller .\HRTEM_Filter_CLI.spec
```

发布前请在干净 Windows 环境中完成测试，并审查生成的依赖锁定文件和第三方许可。

## 历史版本与第三方文件

修复前的完整 42 文件快照归档在开发机的
`<仓库根>\08-历史版本\hrtem-HRTEM滤波工具-v4-2026-07-29`，
该归档未随本仓库分发（仓库内不存在此目录）。

DigitalMicrograph、ImageJ、PASAD 等历史材料位于本项目 `legacy/`。它们不属于
v5 Python 运行时，也不会被打包。详细说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

<!-- README-QUICKREF:BEGIN 由 tools/gen-readme-block.py 生成，请勿手工编辑本区块 -->

---

## 速查

| 项 | 内容 |
|---|---|
| 当前版本 | **5.1.0** |
| 版本来源 | `pyproject.toml` |
| 入口 | `hrtem.py`、`cli_launcher.py` |
| 依赖锁定 | `requirements.lock.txt` |
| 许可证 | MIT |

**从源码运行**——必须使用本工具自己的虚拟环境，不要用 PATH 上的 `python`：
各工具依赖版本互不相同，共用解释器会互相污染。

```powershell
cd <本工具目录>
python -m venv .venv
.venv\Scripts\python -X utf8 -m pip install -r requirements.lock.txt
.venv\Scripts\python hrtem.py
```

**未纳入版本控制**：`.venv/`、`build/`、`dist/`、`release/`、`__pycache__/`、
测试数据与运行输出。具体规则见本目录 `.gitignore`。

<!-- README-QUICKREF:END -->
