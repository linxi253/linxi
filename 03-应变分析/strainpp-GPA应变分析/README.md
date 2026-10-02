# Strain++ GPA 1.4.1

用于 HRTEM 图像几何相位分析（Geometric Phase Analysis, GPA）的 Python
实现，提供图形界面、命令行和 Python API。项目基于
[JJPPeters/Strainpp](https://github.com/JJPPeters/Strainpp)，算法参考：

> Hytch, M. J., Snoeck, E. & Kilaas, R. (1998). Quantitative measurement
> of displacement and strain fields from HREM micrographs.
> *Ultramicroscopy* 74, 131–146.

## 重要说明

- 本软件给出的是基于所选 Bragg 峰、掩膜和参考区的 GPA 结果，不会自动判定
  样品中哪一块“物理上无应变”。
- 定量结果应使用导出的浮点 TIFF/NPY；PNG 仅用于查看和汇报。
- 数值数组中的 `NaN` 表示 Bragg 复振幅不足、相位不可靠的像素，不应当作 0。
- 应变是无量纲量：`0.01 = 1%`。位移单位跟随像素尺寸，界面和 CLI 统一为 nm。
- 数组索引是 `[y, x]`，图像坐标 `+x` 向右、`+y` 向下；正旋转角按逆时针定义。

## 安装与启动

推荐使用 Python 3.10–3.12，并且推荐可编辑安装方式（`pip install -e .`）：

```powershell
python -m pip install -e .
strainpp-gpa-gui
```

`pip install -e .` 与 `python -m pip install -r requirements.txt` 安装的是同一组
核心依赖（`requirements.txt` 与 `pyproject.toml` 中的上界保持一致）。两者等价；
前者额外提供 `strainpp-gpa` / `strainpp-gpa-gui` 两个命令入口，适合常规使用。

也可直接运行源码（不安装 console script）：

```powershell
python -m pip install -r requirements.txt
python .\run.py
```

`ncempy` 仅在打开 DM3/DM4 文件时使用；未安装时会自动回退到内置 DM 解析器，
纯 TIFF 用户不受影响。打包发布（PyInstaller）时 DM 读取链路必须用
`STRAINPP_SMOKE_TEST=1` 做冒烟测试，确认 ncempy 的 h5py/hdf5plugin 依赖
已正确打入产物。

命令行入口：

```powershell
strainpp-gpa image.tif --g1 30 0 --g2 0 30 --sigma1 5 --sigma2 5
```

## 连续图像批量处理

工具栏"打开连续图像"支持多页堆栈 TIF（如 Fiji 的 Substack 导出）和编号
帧序列文件夹。第一帧照常选 G 并计算后，工具栏出现"批量处理全部帧"：

- 导出场可勾选（应变、旋转、膨胀、畸变、位移、相位、质量掩膜）；
- 默认输出两套堆栈（每场一个文件）：`batch_*.tif` 是 float32 定量数据
  （NaN = 不可靠像素；在 ImageJ 中默认按真实 min/max 显示所以看似全黑，
  请 Image → Adjust → B&C → Auto 后查看），`preview_batch_*.tif` 是
  8-bit 彩色预览（原版 Turbo 配色、全栈统一色标、NaN 浅灰），可直接
  查看和汇报，不用于定量；可选同时导出单帧序列文件夹；
- 建议框选参考区并开启"逐帧精修 G"，让 G 矢量跟随原位实验中的样品漂移；
  每帧的 G 修正、有效像素比例与统计量记录在 `batch_statistics.csv`，
  可直接用于绘制时间曲线；
- 单帧失败不会中断批处理，失败原因写入 CSV；批处理参数完整写入
  `batch_metadata.json`；
- 也可全程使用命令行：`strainpp-gpa-batch stack.tif --g1 30 0 --g2 0 30
  --refine-roi 100 100 300 300 -o batch_out`（`--list-only` 先查看帧数与
  ImageJ 标定）。

Windows 独立程序位于 `dist\Strain++GPA.exe`。它带有版本资源，但只有在构建时
提供代码签名证书才会被数字签名。版本资源 `version_info.txt` 为手写副本、可能滞后，
发布版本一律以 `_version.py`（单一来源，当前 1.4.1）为准。

## 图形界面工作流

1. 打开 TIFF、DM3 或 DM4。多页 TIFF 默认分析第一页；RGB TIFF 会按亮度转换。
2. 核对尺寸和像素标定。标量表示方形像素；非方形像素按 `Y,X` 输入，例如
   `0.02,0.025`。
3. 若图像已烧录标尺、文字或黑边，先点击“裁剪分析区”框选纯图像区域。裁剪只
   发生在内存中，不会修改源文件。
4. 在 FFT 功率谱上左键选 G1、右键选 G2，也可以直接在右侧输入相对 DC 的
   `(gx, gy)`。
5. 可点击“检测 Bragg 峰”显示候选峰和建议 Gaussian `sigma`。
6. 如有已知均匀无应变区域，点击“框选参考区”并在原始图像上拖框。计算时两个
   G 矢量会分别迭代修正；不确定时不要随意设置参考区。
7. 设置 Hann 窗、输出旋转角，点击“计算应变”。耗时操作在后台执行，可取消；
   取消后的旧任务不会覆盖新结果。
8. 检查“质量掩膜”、有效像素比例和 G 矩阵条件数，再解释应变图。默认只给
   可靠像素上色；勾选配色区的“显示全部像素”可按原版 Strain++ 风格整幅
   上色（含边缘和低幅值区的原始值）。该开关只影响界面预览与 PNG 出图，
   统计数字和浮点 TIFF/NPY 导出始终套用质量掩膜。场图默认配色为原版
   Strain++ 的 Turbo 色环（0 应变为黄绿色，背景噪声显示为满幅纹理），
   原版 Polar/BlOr/Thermal/Greyscale 及 matplotlib 其余配色均可在下拉
   框切换；NaN 不可靠像素显示为浅灰。
9. “导出当前”可选 PNG、浮点 TIFF 或 NPY；“全部导出”同时生成预览、原始数值、
   文本摘要和 JSON 可复现元数据。

### Bragg 峰选择建议

- 两个峰必须非共线，夹角以 60°–120° 为佳，至少大于 5°。
- 优先选择同一晶格的第一阶、强度高且半径相近的峰。
- 避开 DC 中心、FFT 边缘和被其他相/双衍射明显污染的峰。
- `sigma ≈ 峰距 DC / 6`，与原版 Strain++ 的自动值一致；可在 `/12` 到
  `/4` 间试验。扩大掩膜可提高空间分辨率并显现更丰富的纹理，但会混入
  更多频率；缩小掩膜则平滑、损失空间分辨率。
- 若 `3 × sigma` 已触及 DC，程序会警告；通常应减小 `sigma` 或换更远的峰。
- Hann 窗可减少非周期边缘造成的频谱泄漏，但会降低边缘复振幅，边缘区域常被
  质量掩膜排除。

## 命令行示例

先查看候选峰：

```powershell
strainpp-gpa image.tif --estimate-g
```

完整分析并导出所有场：

```powershell
strainpp-gpa image.tif `
  --g1 30 0 --g2 0 30 `
  --sigma1 4 --sigma2 4 `
  --pixel-size 0.02 `
  --hann --export-all --format all `
  --output results
```

非方形像素与参考区：

```powershell
strainpp-gpa image.dm4 `
  --crop 0 0 1024 920 `
  --g1 42 -18 --g2 17 39 `
  --pixel-size 0.020,0.025 `
  --reference-roi 100 120 300 320 `
  --output results
```

只计算应变、跳过全局相位解包裹和位移：

```powershell
strainpp-gpa image.tif --g1 30 0 --g2 0 30 --no-displacement
```

默认拒绝覆盖同名前缀的已有结果；确认需要覆盖时使用 `--overwrite`。

## Python API

```python
import numpy as np
from strainpp_gpa import GPA

image = np.asarray(...)  # 2D，必须全部为有限值
gpa = GPA()
gpa.load_image(image, pixel_size=0.02, use_hann=True)
gpa.set_g1(30, 0, sigma=4)
gpa.set_g2(0, 30, sigma=4)

# 可选：已知无应变参考区
reference = np.zeros(image.shape, dtype=bool)
reference[100:300, 120:320] = True
gpa.phase1.refine_iterative(reference)
gpa.phase2.refine_iterative(reference)

result = gpa.compute()
print(np.nanmedian(result.eps_xx))
print(result.g_condition_number)
print(result.quality_mask.mean())
```

`GPAOutput` 包含：

| 字段 | 含义 |
|---|---|
| `e_xx`, `e_xy`, `e_yx`, `e_yy` | 非对称畸变张量 `e = ∇u` |
| `eps_xx`, `eps_xy`, `eps_yy` | 对称应变张量 `ε = (e + eᵀ)/2` |
| `omega_xy` | 反对称旋转 `ω_xy = (e_xy - e_yx)/2` |
| `dilatation` | 迹 `e_xx + e_yy` |
| `u_x`, `u_y` | 连续相位重建的位移；可关闭 |
| `phase1`, `phase2` | 包裹几何相位 `P_g1`、`P_g2`（单位：rad）。GUI/CLI 导出的相位场与应变场一样套用质量掩膜，掩膜外为 `NaN` |
| `quality_mask` | 两个 Bragg 复振幅都可靠的像素 |
| `g_condition_number` | G 矩阵条件数，越接近 1 越稳定 |

核心关系为：

```text
P_g(r) = -2π g·u(r)
[u_x, u_y]ᵀ = -(1/2π) G_norm⁻¹ [P_g1, P_g2]ᵀ
```

其中 FFT 像素 G 矢量会按图像宽高归一化为 cycles/pixel。相位梯度采用
包裹安全的相邻主值差，不在图像首尾之间制造周期边界；位移则按需对相位做二维
连续解包裹。旋转输出采用 `E' = R E Rᵀ`，保持迹等张量不变量。

## 输入与标定

- TIFF：支持灰度整数/浮点、多页/堆栈和 RGB。堆栈选择第一页，不会把页维误作
  图像行。
- DM3/DM4：优先由 openNCEM 的 `ncempy` 读取并跳过缩略图。只有单位可识别时
  才自动转换为 nm；启发式后备读取的未知单位候选值不会被悄悄当作 nm。
- 输入中的 NaN/Inf 会被拒绝，避免把缺失数据静默变成“零应变”。
- 已烧录的标尺、文字、选区框和注释都属于像素内容，会产生额外 FFT 峰；务必
  使用 `--crop X1 Y1 X2 Y2` 或 GUI“裁剪分析区”排除。

对于 DM 堆栈，当前工作流分析第一个二维切片。若需要批量分析所有帧，应先显式
拆分并为每帧保留标定和采集条件。

## 输出与复现

- TIFF/NPY 保存完整浮点数组，保留 NaN。
- PNG 使用当前色图和稳健的 99.5% 显示范围，避免少量缺陷核压扁整体对比度；它
  只是可视化预览，极值仍完整保留在 TIFF/NPY。
- `*_metadata.json` 记录输入绝对路径、图像形状、像素尺寸、最终 G 矢量、sigma、
  旋转角、Hann、参考区、有效像素比例、条件数、软件及依赖版本；其中
  `phase_field_semantics` 字段说明相位场的单位与 NaN 语义。
- `*_info.txt` 提供便于人工查看的参数和 NaN 感知统计。

建议在论文或报告中同时记录：输入文件校验值、G1/G2、sigma、参考区、像素标定、
Hann、旋转角、有效像素比例及色标范围。

## 测试、构建与签名

```powershell
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
python -m build
python -m twine check dist\*
```

构建 Windows EXE：

```powershell
.\build.ps1
```

使用 PFX 证书签名：

```powershell
$env:STRAINPP_CERT_PATH = 'C:\secure\codesign.pfx'
$env:STRAINPP_CERT_PASSWORD = '...'
.\build.ps1
```

证书和密码不要提交到项目中。CI 配置位于 `.github\workflows\ci.yml`
（矩阵条目 `03 Strain++ GPA`）。

## 已知边界

- GPA 假设所选局部晶格频率可由两个非共线 G 矢量描述；强多相重叠、严重晶格
  缺陷、厚度/倾转对比或非线性扫描畸变可能使“应变”混入成像伪影。
- 简单二维相位解包裹不能保证跨越密集位错核得到唯一绝对位移；应变由包裹安全
  局部梯度计算，通常比绝对位移更稳健。
- 自动峰检测是候选工具，不替代用户对晶体学指数和衍射条件的判断。
- Windows SmartScreen 信誉与 Authenticode 签名需要发布者自己的有效证书；
  项目无法生成可信证书。

## 许可证

本修改版于 2026-07-29 明确标注修改，并依照上游要求以
GNU General Public License v3 或更高版本发布。完整条款见 [LICENSE](LICENSE)。
发布二进制时必须同时满足 GPL 对相应源代码的提供要求。

本软件不提供任何明示或默示担保。原始 Strain++ 作者及论文作者不对本修改版的
错误或结果解释负责。

<!-- README-QUICKREF:BEGIN 本区块为手工维护，需与版本来源（pyproject.toml / 代码 __version__，见「版本来源」行）保持一致 -->

---

## 速查

| 项 | 内容 |
|---|---|
| 当前版本 | **1.4.1** |
| 版本来源 | `_version.py` |
| 入口 | `run.py` |
| 依赖锁定 | `requirements-lock.txt` |
| 许可证 | **GPL-3.0-or-later**（派生自 [JJPPeters/Strainpp](https://github.com/JJPPeters/Strainpp)，见本目录 `LICENSE`） |

> **GPL-3.0 具传染性**：把它与其它代码打包成**单一作品**分发时，整体授权需按
> GPL-3.0 重新审视；仅在同一仓库中并列存放、各自独立分发则不受影响。

**从源码运行**——必须使用本工具自己的虚拟环境，不要用 PATH 上的 `python`：
各工具依赖版本互不相同，共用解释器会互相污染。

```powershell
cd <本工具目录>
python -m venv .venv
.venv\Scripts\python -X utf8 -m pip install -r requirements-lock.txt
.venv\Scripts\python run.py
```

**未纳入版本控制**：`.venv/`、`build/`、`dist/`、`release/`、`__pycache__/`、
测试数据与运行输出。具体规则见本目录 `.gitignore`。

<!-- README-QUICKREF:END -->
