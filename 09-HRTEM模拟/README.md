# 09-HRTEM 高分辨模拟工具

基于 **tem_sim** 引擎（SimulaTEM v1.3.2 Cowley–Moody 多层法的 Python 重构）的
HRTEM 图像模拟 GUI 工具。本工具的物理管线已通过 20260820 交付结果验收：
用 0820 最终参数重跑，与交付的物理强度数据 **NCC = 1.000000**（数值一致）；
S36 参考图红框风格复现 NCC = 0.996。

## 散射因子标度修正（2026-09 P0，务必阅读）

2026-09 深度审查发现 SimulaTEM `gauss3.txt` 的**绝对标度**系统性偏大：

- 氢原子第一性原理（Born 极限解析解）f_e(0) = 0.5292 Å，表值 1.0144 Å（**×1.92**）；
- 与 Peng 标准参数化（Peng, *Micron* 30, 625–648, 1999）逐元素比对，
  H→Pb 十元素偏大 **1.81–1.92×**（均值 1.87），且 g 衰减形状不符（缺窄 g 宽项）；
- 引擎等效平均内电位 V₀：Al 32.0 / Si 26.4 / Fe 53.0 V，实验公认
  13.4 / 11.9 / 22.7 V。

因此自本版本起：

1. **默认使用 Peng 标准表**（`tem_sim/data/peng_high.json`，经 f(0) 第一性原理
   与 Mott–Bethe 形状双重验证）；全部自检见 `tests/verify_physics.py`（新增
   第 8/9 项：散射标度、平均内电位）。
2. `gauss3.txt` 降级为 legacy：`multislice(..., table="gauss3")`，数值与修正前
   逐位一致，仅供复现 0820 交付口径（`python validate_reproduce.py --legacy`，
   NCC = 1.000000）。
3. **对既有结论的影响**：相位/衬度绝对强度变为原 ~1/1.87，衍射强度比改变；
   0820/S36 预设参数是 legacy 口径寻优结果，标准表下需重标定——
   (厚度, 离焦) 二维重扫 best NCC(实验) ≈ 0.35（t≈2.6 nm, df≈−30 nm），
   legacy 口径为 0.60；完整重标定需扩展至光阑/像散/极性等（**待办**）。
   各预设已标注 `status: needs-recalibration`。

## 相位标度修正（2026-10 P0，务必阅读）

2026-10 审计发现 `tem_sim/scattering.py::phase_kernels` 的相位核把**像素积分**
（量纲 rad·Å²）当逐像素相位（rad）用，少除一个像素面积 Δx·Δy——与
010-STEM模拟 的 `stem_sim` 同一缺陷（后者 2026-09 先行修复，本工具现已同步）：

- 默认采样 0.07 Å/px 下透射函数相位被整体压低 **×204**（1/0.07²），
  且随采样而变；
- 引擎实际只工作在弱相位物体（单散射）极限：图像形状仍近似正确的线性像
  （min-max 归一化显示与 NCC 等尺度无关指标看不出异常），但**绝对强度低约
  200 倍**、高角散射强度低约 4 个量级，厚度非线性/衬度反转/Pendellösung
  等动力学效应被抹掉；
- 长期未被发现的原因：自检的求和类断言（∫K = γλ·a、V0 = Σφ/(σ·V_cell)）
  与 NCC 验收都对整体标度不敏感——它们只依赖积分的和。

因此自本版本起：

1. 相位核改为连续相位在像素上的**平均值**：`K_i = ∫_pixel φ dA/(sx·sy)`，
   与 010-STEM模拟 的 `stem_sim/scattering.py` 逐行一致；
2. 判定性判据：单 Au 原子投影相位峰值收敛到解析值
   γλ·π·Σ(a_i/b_i) = **3.4713 rad**（≈π，重原子中心透射函数近 −1），
   且与采样无关。自检新增该绝对标度回归项（`tests/verify_physics.py`
   相位标度），并把原 DFT 比对基准（÷(sx·sy)）、V0 口径（⟨φ⟩/(σ·t)）
   与 Σφ 求和规则（γλ·f_e(0)/(Δx·Δy)）一并改为新口径；
3. **对既有结论的影响**：在 2026-09 散射因子标度修正之上，绝对强度口径
   再次改变；0820/S36 预设（已标 `status: needs-recalibration`）需在
   新口径下重新标定（**待办**），历史 NCC 数值仅对旧口径成立。

## 启动

```
run.bat
```
或
```
python hrtem_tool/main.py
```

依赖：numpy、scipy、matplotlib、Pillow、ase（读 CIF）、ttkbootstrap（界面主题，可选）。
运行环境为本项目独立的 `.venv`（Python 3.10），依赖锁定在 `requirements.lock.txt`。

> **不要使用 Miniconda base 环境运行本项目。** 本工作区多个工具依赖版本不同
> （例如 numpy 1.26 与 2.2 并存），共用解释器会让工具读到别的项目的依赖版本，
> 并且 `pip install` 会污染全局环境、连带破坏其它工具。

首次准备环境（只需一次）：

```bat
python -m venv .venv
.venv\Scripts\python -X utf8 -m pip install -r requirements.lock.txt
```

之后双击 `run.bat`，或用项目内解释器启动：

```bat
.venv\Scripts\python hrtem_tool\main.py
```

## 功能

| 面板 | 内容 |
|---|---|
| 结构与取向 | 导入 CIF/PDB/XYZ；带轴 [h k l] 选择（快捷按钮+自定义）；样品厚度；横向视场；面内旋转+水平镜像；带轴投影信息（投影周期 Lx/Ly/Lz、投影原胞原子数、实际厚度） |
| 电镜与模拟参数 | 电压、球差 Cs（mm，可负=校正器）、离焦（Scherzer 一键）、物镜光阑（最佳光阑一键）、离焦展宽、束发散、二重像散；采样/像素数/散射因子表/切片厚度；λ、Scherzer、点分辨率只读显示 |
| 模拟结果 | HRTEM 像（标尺+参数角标）、衍射花样（log）、CTF 曲线切换；对比度/极性/探测器模糊/旋转/镜像/输出采样实时联动（取向类从原始强度重渲染，无需重新模拟）；导出 PNG / 16-bit TIFF（参数写入 ImageDescription）/ float32 NPY / 参数 JSON |
| 批量系列 | 离焦系列、厚度系列（一次势场多厚度捕获）、厚度×离焦矩阵（S36 风格蒙太奇），输出 summary CSV |

## 参数约定（SimulaTEM 惯例，务必注意）

- **离焦符号：正 = 过焦，负 = 欠焦**（Scherzer 离焦为负值）。
- **球差 Cs**：界面输入 mm；1 mm = 1e7 Å；负值 = 球差校正器过校正（如 -0.8 mm）。
- **厚度**：超胞沿带轴按最小周期 Lz 向上取整，界面显示实际厚度
  （例：Fe3O4[110] Lz=11.94 Å，请求 3.0 nm → 实际 3.582 nm，与 0820 一致）。
- **面内旋转**：绕光轴逆时针；镜像 = 水平翻转（用于与实验图取向对齐）。
  单次模拟完成后旋转/镜像/输出采样可实时调整（从原始强度重渲染）。
- **散射因子表**：GUI「多层法」面板可选。默认 **Peng 1999 标准表**（物理标度）；
  `gauss3 legacy` 仅用于复现 SimulaTEM/0820 交付口径（标度偏大 ~1.87×）。
  所用表会写入 TIFF 描述串与参数 JSON。
- **对比度归一口径**：屏幕显示与导出 PNG 使用界面百分比（默认 0.5–99.5，
  随参数保存）；16-bit TIFF 固定 0.05–99.95（0820 交付口径）；float32 NPY
  为未归一的物理强度。三种口径均在文件元数据中注明。
- CTF：χ(k) = πλΔf k² + (π/2)Csλ³k⁴ + πλA k²cos2(θ−φ)，
  含时间（离焦展宽）/空间（束发散）相干包络。

## 预设（presets/，均来自已验证参数）

| 预设 | 条件 |
|---|---|
| Fe3O4 [110] 0820最终 | 200 kV, Cs=0.085 mm, df=-34 nm, 24 mrad, 像散120Å@145°, t≈3.58 nm, 91.8°+镜像（可接受结果全套参数） |
| Fe3O4 [110] S36红框 | 300 kV, Cs=1 mm, df=-20 nm, 5.5 mrad, t=25 nm, 120.25°+镜像, 极性反转 |
| FeO [111] 300kV校正 | 300 kV, Cs=-0.8 mm, df=+3 nm, 30 mrad |
| Fe3O4 [110] 300kV经典 | 300 kV, Cs=1 mm, df=-50 nm, 16 mrad |

## 目录结构

```
09-HRTEM模拟/
├── hrtem_tool/            # GUI 应用（Tk + matplotlib）
│   ├── main.py            # 入口
│   ├── gui.py             # 三栏主界面
│   ├── params.py          # 参数模型（nm/mm ↔ Å 转换）
│   ├── sim_core.py        # 模拟管线（GUI/命令行共用）
│   ├── series.py          # 批量系列 + 蒙太奇
│   ├── render.py          # 旋转/镜像/重采样/标尺/对比度
│   ├── export.py          # PNG/TIFF16/NPY/JSON 导出
│   └── worker.py          # 后台线程
├── tem_sim/               # 物理引擎（自 20260815/tem模拟 迁移并增强）
│   └── ...                # 多厚度捕获 multislice_series、zone_axis_info 为本工具新增
├── presets/               # 参数预设 JSON（可自建）
├── cif/                   # 示例结构：Fe3O4、FeO、α-Fe2O3、α-Fe
├── tests/                 # verify_physics.py 物理自检 + 输出
├── validate_reproduce.py  # 验收：复现 20260820 可接受结果
└── run.bat
```

## 验收与自检

```
python tests/verify_physics.py     # 波长/σ/散射因子/幺正性/Scherzer + 标度/相位标度/V0
python validate_reproduce.py       # 标准表（Peng）：端到端基线 + 重标定扫描
python validate_reproduce.py --legacy  # legacy（gauss3）：逐比特复现 0820 交付
```

`validate_reproduce.py` 需要 20260820 参考数据，默认路径
`D:\refdata\20260820`；可用 `--ref-root DIR` 或环境变量
`HRTEM_REF_ROOT` 覆盖。缺失时自动跳过对应项。

单元测试（无需参考数据）：`python -m pytest tests/ -v`
（物理自检 + 工具层校验/渲染/导出/惰性路径一致性）。

## 已知边界（引擎未实现，与 0815 工作记录一致）

- 声子/TDS、束倾斜、高阶像散（C32 等）未实现；
- 带轴重构要求公度取向（非公度会明确报错，可手工建超胞后用 PDB/XYZ 导入）；
- 采样过粗（>0.15 Å/px）或视场过小会影响衬度准确性，正式出图建议
  0.05–0.08 Å/px、切片 ≤2 Å（与 0820 验收口径一致）；
- 厚样品（≳10 nm）建议开启引擎级反混叠带限 `multislice(..., band_limit=True)`
  （2/3 奈奎斯特，Kirkland 惯例；默认关闭以保持结果口径，GUI 暂未暴露）；
- **出射波语义**：出射波在最后一片透射后多传播一片 dz（与 SimulaTEM 一致，
  等效附加 +dz 过焦；`multislice_series` 报告的捕获厚度含此 dz）；
- **取向渲染为中心裁剪**：输出视场 ≈ 输入 FOV × N/(N+16)（约 1–2%），
  与 0820 验收标定耦合，实验匹配的采样标度已包含该效应，勿单独修改；
- 非周期结构（PDB/XYZ 无晶胞）建议真空边距 ≥ 8 Å（`padding` 参数），
  不足时引擎会告警（宽尾相位核回绕风险）。

<!-- README-QUICKREF:BEGIN 由 tools/gen-readme-block.py 生成，请勿手工编辑本区块 -->

---

## 速查

| 项 | 内容 |
|---|---|
| 当前版本 | **1.2.0** |
| 版本来源 | `tem_sim/__init__.py` |
| 入口 | `run.bat` |
| 依赖锁定 | `requirements.lock.txt` |
| 许可证 | MIT（本体代码）；`tem_sim/data/peng_high.json` 取自 abTEM（GPL-3.0），见根 [NOTICE.md](../NOTICE.md) |

**从源码运行**——必须使用本工具自己的虚拟环境，不要用 PATH 上的 `python`：
各工具依赖版本互不相同，共用解释器会互相污染。

```powershell
cd <本工具目录>
python -m venv .venv
.venv\Scripts\python -X utf8 -m pip install -r requirements.lock.txt
.\run.bat
```

`run.bat` 检测到环境缺失时会打印重建指引，不会静默回退到 PATH 上的解释器。

**未纳入版本控制**：`.venv/`、`build/`、`dist/`、`release/`、`__pycache__/`、
测试数据与运行输出。具体规则见本目录 `.gitignore`。

<!-- README-QUICKREF:END -->
