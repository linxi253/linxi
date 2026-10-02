# 4D-STEM Processor

用于批量读取 DM4 四维扫描数据并生成 DPC/iDPC、SSB、应变、晶体取向、衍射峰对和 ePIE 叠层成像结果。当前已提供 Windows 可执行文件和一键 GUI。

## 当前快速开始

### 直接运行

```text
dist/4D-STEM_Processor.exe
```

1. 选择包含 `.dm4` 文件的输入文件夹；程序会递归搜索。
2. 选择独立输出文件夹。
3. 设置中心扫描裁剪尺寸，默认 `128 × 128`。
4. 选择附加分析模块并开始处理。
5. 每个 DM4 文件会在输出根目录下生成独立子目录。

### 从源码运行

建议 Python 3.10 或 3.11：

```powershell
python -m pip install -r requirements.txt
python .\stem_processor_gui.py
```

主 GUI 的基础流程会固定生成虚拟 BF/ADF/ABF、CoM/E 场/电荷密度和元数据；
`DPC/iDPC Phase`、`SSB Reconstruction` 复选框控制对应重建；应变、取向、
Peak Pairs 和 ePIE 复选框控制附加模块（应变/取向默认关闭——它们是逐像素
晶体学计算，在大扫描上耗时远超其余模块之和）。高级参数行可调 iDPC 正则化、
SSB 离焦（像差校正）与 ePIE 扫描步距（<1 为超分辨采样）。

## GUI 输出

常见输出包括：

- `*_dpc_phase.npy`、`*_com_x.npy`、`*_com_y.npy`、`*_com_magnitude.npy`；
- `*_ssb_phase.npy`、`*_ssb_amplitude.npy`、`*_ssb_object.npy`（复透射函数）；
- `*_bf.npy`、`*_adf.npy`、`*_abf.npy`、`*_bf_intensity.npy`；
- `*_charge_density.npy`；
- `*_dpc.png`、`*_ssb.png`、`*_comparison.png`、`*_efield.png`；
- `*_metadata.json`（含来源路径、文件大小、dm4 标尺/单位、预处理参数、
  正则化/离焦、维度判定分数与依据、耗时）；
- 输出根目录的 `summary.csv`：每个文件一行（含失败文件），记录
  状态/耗时/形状/alpha/相关系数/扫描步长/错误信息；同名 DM4 会自动
  加后缀避免输出目录互相覆盖；
- 启用附加模块时的应变（`*_strain.npz`，含应变分量 + E 场 + 匹配数 +
  拟合残差 + 参考斑点；`*_strain_metadata.json` 记录参考区语义）、
  取向（`*_orientation.npz` + 图）、峰对（npy + 图）或 ptychography
  （`*_ptychography_phase.npy` + 图 + json）。

## 当前项目文件

- `stem_processor_gui.py`：GUI 入口与批处理调度；数值计算全部委托给 `core/`。
- `core/dm4_io.py`：统一的 DM4 读取模块（元数据、dtype/字节序、裁剪提取、维度修复）。
- `core/dpc_core.py`：DPC/iDPC、虚拟探测器、向量化稳健 CoM。
- `core/ssb_core.py`：SSB 重建（支持取消与像差校正）。
- `core/dimension_utils.py`：扫描/探测器维度验证与修复。
- `core/strain_mapping.py`：基于 CoM 场的应变映射。
- `core/orientation_mapping.py`：衍射对称性和模板取向映射。
- `core/peak_pairs.py`：衍射峰对分析。
- `core/ptychography.py`：ePIE/WDD 叠层成像。
- `core/pipeline.py`：GUI 与批处理脚本共享的数据准备层。
- `processing/`：针对原 Au 数据集的提取、批处理和处理脚本。
- `diagnostics/`：数据类型、原始数据和结果质量诊断脚本。
- `tests/`：基于合成数据的自动化冒烟测试（`pytest tests`）。
- `requirements.txt`：运行时与开发依赖清单。
- `4D-STEM_Processor.spec`：PyInstaller 配置。
- `dist/4D-STEM_Processor.exe`：已打包程序。

## 重要限制

- GUI 会把每个数据集中心裁剪到设定扫描尺寸；这不是完整扫描区域分析。
- Stop 按钮现在可以协作式中断：数据提取后、CoM/SSB/ePIE 等耗时循环都会检查
  停止标志，通常在数秒内响应。
- ePIE 在 GUI 设定的整个（裁剪后）扫描区域上运行，迭代次数由 GUI 控制；
  大扫描区域耗时较长，请酌情调小迭代次数或裁剪尺寸。
- 输入数据类型和字节序直接取自 DM4 头部（不再按文件大小猜测）；维度顺序由
  ncempy 布局 + 启发式校验共同判断，仍建议核对 BF 图和元数据。
- 脚本路径可通过环境变量配置（见“数据路径配置”）；未设置时回退到原
  `D:\data\...` 路径。
- 当前 Au 数据存在背景扣除、负值、小探测器和低信背比问题；相关结果只能用于方法验证，不能直接作为可靠定量结论。

下面的原文档记录了 Au 数据处理方法、修复历史和已知数据质量问题。

## 目录

1. [概述](#概述)
2. [文件结构](#文件结构)
3. [运行环境](#运行环境)
4. [数据处理流程](#数据处理流程)
5. [发现的技术问题](#发现的技术问题)
6. [可修复问题与实验数据问题](#可修复问题与实验数据问题)
7. [脚本功能说明](#脚本功能说明)
8. [处理参数配置](#处理参数配置)
9. [重现处理过程](#重现处理过程)
10. [结果评估](#结果评估)

---

## 概述

本项目包含对 Au（金）样品 4D-STEM 数据的完整处理流程，包括：
- **DPC (Differential Phase Contrast)** 微分相位衬度分析
- **iDPC (integrated DPC)** 积分相位重建
- **SSB (Single Side Band)** 单边带叠层成像
- **虚拟探测器成像** (BF/ADF/ABF)
- **电场映射** 和电荷密度分析

### 数据概况

| 数据集 | 扫描尺寸 | 探测器尺寸 | 数据类型 | 状态 |
|--------|----------|------------|----------|------|
| Au_SI19 | 2048×2048 | 32×32 | float32 (dataType=2) | 有质量问题 |
| Au_SI20 | 2048×2048 | 31×32 | float32 (dataType=2) | 有质量问题 |
| Au_SI21 | 2048×2048 | 30×34 | float32 (dataType=2) | 有质量问题 |
| Cu foil (标准) | 45×45 | 512×512 | 待复核¹ | 正常 |

> ¹ 早期文档按错码表（ncempy 的 tag 编码类型表 `_EncodedTypeDTypes`）把 Cu foil
> 记为 uint8、Au 记为 int16。图像 dataType 码的正确语义见
> `core/dm4_io.py` 的 `DM4_DTYPES`（2=float32、10=uint16 等）；Au 的头部
> dataType=2 → float32，Cu foil 的原始 dataType 码仓库内无记录，需用
> DM4 头部重新确认。

---

## 文件结构

```
measure/
├── README.md                    # 本文档
├── stem_processor_gui.py        # 当前一键 GUI 入口
├── core/                        # 核心算法模块
│   ├── dpc_core.py             # DPC/iDPC 核心算法
│   ├── ssb_core.py             # SSB 叠层成像算法
│   ├── dimension_utils.py       # 维度验证/修复
│   ├── strain_mapping.py        # 应变映射
│   ├── orientation_mapping.py   # 取向映射
│   ├── peak_pairs.py            # 衍射峰对
│   ├── ptychography.py          # ePIE/WDD
│   └── pipeline.py              # 共享数据准备层
├── processing/                  # 处理脚本
│   ├── process_au_v3.py        # 主处理脚本（推荐）
│   ├── extract_au_correct.py   # 正确数据提取脚本
│   └── batch_dpc.py            # 批量DPC处理
├── diagnostics/                 # 诊断和验证脚本
│   ├── diagnose_au.py          # Au数据全面诊断
│   ├── check_raw_data.py       # 原始DM4数据检查
│   ├── check_cu_standard.py    # Cu标准数据验证
│   ├── analyze_correct_data.py # 正确提取数据分析
│   ├── verify_dtype.py         # 数据类型验证
│   └── visualize_dp.py         # 衍射图案可视化
└── results/                     # 处理结果输出目录
```

---

## 运行环境

### Python 环境

```bash
# 推荐 Python 3.10+
python --version

# 必需依赖
pip install numpy scipy matplotlib ncempy
```

### 依赖项

| 包名 | 版本 | 用途 |
|------|------|------|
| numpy | ≥1.24 | 数值计算 |
| scipy | ≥1.10 | 取向、峰值、插值和叠层成像辅助计算 |
| matplotlib | ≥3.7 | 可视化 |
| ncempy | ≥1.0 | DM4文件读取 |

### 硬件要求

- **内存**: ≥16 GB（处理完整2048×2048扫描时需要）
- **存储**: ≥50 GB（原始数据+中间结果）
- **CPU**: 多核处理器（支持并行计算）

---

## 数据处理流程

### 完整流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    4D-STEM 数据处理流程                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 数据提取 (extract_au_correct.py)                         │
│     ├── 读取 DM4 文件（使用 ncempy）                         │
│     ├── 识别数据类型（按 dataType 码映射，Au 为 float32）    │
│     ├── 提取 4D-STEM 数据立方体                              │
│     └── 裁剪感兴趣区域（128×128 扫描位置）                   │
│                                                             │
│  2. 数据预处理 (process_au_v3.py)                            │
│     ├── 负值处理：data_shifted = data - data.min()           │
│     ├── 计算质心 (CoM) 确定光束中心                          │
│     └── 估计 BF 盘半径 (alpha)                               │
│                                                             │
│  3. DPC 分析 (dpc_core.py)                                   │
│     ├── 计算 CoM 场 (CoM-Y, CoM-X)                          │
│     ├── iDPC 相位重建（傅里叶积分）                          │
│     └── 虚拟图像生成 (BF/ADF/ABF)                            │
│                                                             │
│  4. SSB 重建 (ssb_core.py)                                   │
│     ├── 扫描维度 FFT                                         │
│     ├── Q 空间遍历和单边带提取                               │
│     └── 逆 FFT 获得复透射函数                                │
│                                                             │
│  5. 后处理和分析                                             │
│     ├── DPC vs SSB 对比                                      │
│     ├── 电场映射 (Ex, Ey, |E|)                               │
│     └── 电荷密度计算 (∇·E)                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 详细步骤

#### 步骤 1: 数据提取

```python
# 使用 extract_au_correct.py
from ncempy.io import dm
import numpy as np

# 打开 DM4 文件
f = dm.fileDM('007_STEM SI.dm4')

# 识别 4D-STEM 对象（ndim=4）
# 数据类型: float32 (dataType=2；旧文档按错码表记为 int16)
# 字节序: little-endian (byte_order=1)

# 使用 memmap 读取数据
data = np.memmap(filepath, dtype='<f4', mode='r', offset=offset,
                 shape=(det_y, det_x, scan_y, scan_x))

# 转换维度顺序: (det_y, det_x, scan_y, scan_x) -> (scan_y, scan_x, det_y, det_x)
data = data.transpose(2, 3, 0, 1)
```

#### 步骤 2: 预处理

```python
# 负值处理（数据是背景扣除后的）
data_shifted = data - data.min()  # 使所有值为正

# 计算光束中心
com_result = compute_com_fast(data_shifted)
center = com_result['center']  # 例如: (15.5, 15.5)

# 估计 BF 盘半径
alpha, radial, counts = find_alpha_from_radial(data_shifted, center)
# 例如: alpha = 7 pixels
```

#### 步骤 3: DPC 分析

```python
# 运行 DPC pipeline
dpc_result = run_dpc_pipeline(
    data_shifted,
    alpha_pixels=alpha,
    center=center,
    regularization=1e-3
)

# 输出:
# - dpc_result['phase']: iDPC 相位
# - dpc_result['com_result']['com_y']: Y 方向 CoM
# - dpc_result['com_result']['com_x']: X 方向 CoM
# - dpc_result['virtual_images']['bf']: 明场像
# - dpc_result['virtual_images']['adf']: 环形暗场像
```

#### 步骤 4: SSB 重建

```python
# SSB 重建
ssb_result = ssb_reconstruct(
    data_shifted,
    alpha_pixels=alpha,
    center=center,
    regularization=1e-2
)

# 输出:
# - ssb_result['phase']: SSB 相位
# - ssb_result['amplitude']: SSB 振幅
# - ssb_result['complex_obj']: 复透射函数
```

---

## 发现的技术问题

### 问题 1: 数据类型错误（已修复；当时的 dtype 结论需再修正）

| 项目 | 详情 |
|------|------|
| **问题描述** | 原始提取使用 `>u2` (big-endian uint16)，字节序确属错误；当时把 dtype 修为 `<i2` (little-endian int16)，但那是按错码表（ncempy 的 tag 编码类型表 `_EncodedTypeDTypes`）得出的结论——Au 的 dataType=2 按图像 dataType 表（`core/dm4_io.py` 的 `DM4_DTYPES`）应为 `<f4` (little-endian float32) |
| **发现方法** | `check_raw_data.py` 对比不同数据类型解释的结果 |
| **影响** | 数据完全错误，无法进行任何分析；按 int16 解码 float32 字节数相同、能通过文件大小自检，错值静默进入后续定量分析 |
| **解决方案** | 使用 ncempy 正确读取 DM4 元数据确定 dataType 码，并按图像 dataType 表映射 dtype；`core/dm4_io.py` 现在还会在解码前校验「码→itemsize→与 tag 声明字节数一致」 |
| **修复状态** | ✅ 已修复 |

### 问题 2: 约 50% 负值像素（数据质量问题）

| 项目 | 详情 |
|------|------|
| **问题描述** | Au 数据有 48.8-49.7% 的像素为负值（该比例按当时的 int16 解码统计） |
| **发现方法** | `diagnose_au.py` 统计负值比例 |
| **原因** | 当时归因于采集时激进的背景扣除；但 float32 位型被按 int16 截断本就会产生约一半「负值」，该观察主要须先以正确的 float32 解码重新核实，再谈背景扣除 |
| **影响** | 物理意义丧失，CoM 计算需要正值 |
| **解决方案** | 偏移处理: `data_shifted = data - data.min()`（对正确解码后的数据重新评估） |
| **修复状态** | ⚠️ 可处理，但信息已丢失 |

### 问题 3: 探测器尺寸过小（硬件限制）

| 项目 | 详情 |
|------|------|
| **问题描述** | 探测器仅 32×32 像素（标准应为 256×256+） |
| **发现方法** | DM4 元数据读取 |
| **影响** | 角度分辨率极低，SSB 重建严重欠采样 |
| **解决方案** | 无法通过软件修复 |
| **修复状态** | ❌ 不可修复（硬件限制） |

### 问题 4: 信背比极低（数据质量问题）

| 项目 | 详情 |
|------|------|
| **问题描述** | 信背比仅 0.016 (1.6%) |
| **发现方法** | `diagnose_au.py` 计算信号/背景比值 |
| **影响** | 信号淹没在噪声中，定量分析不可靠 |
| **解决方案** | 无法通过软件修复 |
| **修复状态** | ❌ 不可修复（数据质量） |

### 问题 5: 无清晰 BF 盘（数据质量问题）

| 项目 | 详情 |
|------|------|
| **问题描述** | 中心/角落强度比仅 ~1.0（正常应 >1.5） |
| **发现方法** | `check_cu_standard.py` 对比标准数据 |
| **影响** | 无法准确确定 alpha 参数 |
| **解决方案** | 使用梯度法估计 alpha |
| **修复状态** | ⚠️ 可估计，但精度有限 |

### 问题 6: DPC-SSB 负相关（算法/数据问题）

| 项目 | 详情 |
|------|------|
| **问题描述** | DPC 和 SSB 相位相关系数为 -0.24（应为正相关） |
| **发现方法** | `process_au_v3.py` 计算相关系数 |
| **可能原因** | SSB 欠采样、相位约定差异、数据质量问题 |
| **影响** | 至少有一种方法结果不可靠 |
| **解决方案** | 需要进一步调查 |
| **修复状态** | ⚠️ 待调查 |

---

## 可修复问题与实验数据问题

### 可修复问题（软件/算法层面）

| 问题 | 修复方法 | 状态 |
|------|----------|------|
| 数据类型错误 | 使用 ncempy 正确读取 | ✅ 已修复 |
| 字节序错误 | 从 DM4 头读取 byte_order | ✅ 已修复 |
| 负值处理 | 偏移处理 (data - min) | ✅ 已修复 |
| CoM 中心计算 | 使用实际 CoM 而非几何中心 | ✅ 已修复 |
| Alpha 估计 | 梯度法 + 阈值法组合 | ✅ 已修复 |
| ADF mask 自适应 | 根据探测器尺寸调整 | ✅ 已修复 |
| 大数据集内存 | 分块处理 | ✅ 已修复 |

### 实验数据问题（不可修复）

| 问题 | 数值 | 影响 | 可修复性 |
|------|------|------|----------|
| 探测器尺寸过小 | 32×32 像素 | 角度分辨率极低 | ❌ 不可修复 |
| ~50% 负值像素 | 48.8-49.7% | 物理意义丧失 | ❌ 不可修复 |
| 总强度负值位置 | 41-47% 扫描位置 | CoM 不可靠 | ❌ 不可修复 |
| 信背比极低 | S/B = 0.016 | 信号淹没在噪声中 | ❌ 不可修复 |
| 无清晰 BF 盘 | 中心/角落比 ~1.0 | 无法确定 alpha | ❌ 不可修复 |
| 过度背景扣除 | 均值 ~0 | 真实信号丢失 | ❌ 不可修复 |

### 问题严重程度矩阵

```
                    可修复性
                    ↑
        高 │  数据类型错误        负值处理
           │  Alpha估计          SSB参数优化
           │
        中 │  DPC-SSB相关性      虚拟图像优化
           │
        低 │  小探测器适配       
           │
        无 │  50%负值            低信背比
           │  总强度负值         无BF盘
           │  探测器尺寸         过度背景扣除
           └────────────────────────────────→ 严重程度
              低        中        高        致命
```

---

## 脚本功能说明

### core/ - 核心算法模块

#### dm4_io.py

统一的 DM4 读取模块，是数据进管线的唯一入口：

- `read_dm4_metadata()`：用 ncempy 读取头部，按 DM4 图像 dataType 码映射
  numpy dtype（1=int16、2=float32、3=complex64、6=uint8、7=int32、9=int8、
  10=uint16、11=uint32、12=float64、13=complex128，来源为 ncempy dm.py 的
  `_DM2NPDataTypes` / Gatan dm4io.h 枚举），不再按文件大小猜测；映射结果
  还会与 tag 树为数据块声明的字节数精确比对，不一致即报错拒绝解码；
- `extract_4d_data()`：按 `(det_y, det_x, scan_y, scan_x)` 布局 memmap 裁剪
  并转置为 `(scan_y, scan_x, det_y, det_x)`；
- `fix_dimensions()`：委托 `dimension_utils` 做维度校验/修复；
- `preprocess()`：默认仅做下限 1% 百分位裁剪并转 float32（修剪背景扣除
  数据的极端负离群点）；不做上限裁剪——全局上限裁剪会把小盘径数据集的
  BF 盘削平、摧毁 CoM 信号，如确需上限裁剪可显式传参启用；
- `scan_step_nm()`：从 DM4 标尺读取扫描步长。

#### dpc_core.py (585 行)

DPC/iDPC 核心算法实现。

**主要函数:**

| 函数名 | 功能 | 参数 |
|--------|------|------|
| `compute_com_fast()` | 计算质心 (CoM) 场 | datacube, center, radius |
| `compute_com_robust()` | 向量化稳健 CoM（正值裁剪、离群点剔除、可取消） | datacube, center, radius, should_stop |
| `find_alpha_from_radial()` | 估计 BF 盘半径 | datacube, center |
| `idpc_reconstruct()` | iDPC 相位重建 | com_y, com_x, regularization |
| `compute_virtual_images()` | 生成虚拟图像 | datacube, alpha, center |
| `run_dpc_pipeline()` | 完整 DPC 流程 | datacube, alpha_pixels, center |
| `plot_dpc_results()` | 可视化 DPC 结果 | result, label |

**关键算法:**

```python
# CoM 计算
CoM_y(R) = Σ_K [I(K,R) × K_y] / Σ_K [I(K,R)]
CoM_x(R) = Σ_K [I(K,R) × K_x] / Σ_K [I(K,R)]

# iDPC 傅里叶积分
φ̃(q) = -i(q · CoM̃(q)) / (2π|q|² + ε)
```

#### ssb_core.py (528 行)

SSB 单边带叠层成像算法。

**主要函数:**

| 函数名 | 功能 | 参数 |
|--------|------|------|
| `ssb_reconstruct()` | SSB 主重建函数（支持 `should_stop` 取消、像差校正） | datacube, alpha_pixels, center, defocus_rad, Cs_rad |
| `_generate_q_indices()` | 生成 Q 空间索引 | scan_size, info_limit |
| `find_center_of_mass()` | 查找光束中心 | datacube |

**关键算法:**

```python
# SSB 重建流程
1. G(K,Q) = FFT_R[I(K,R)]  # 扫描维度 FFT
2. 对每个 Q 矢量:
   DO+(Q) = A(K)·A(K-Q)·[1-A(K+Q)]  # 双重叠区
   Ψ_s(Q) = Σ G(K,Q)·Γ*(K,Q) / Σ|Γ|²
3. ψ(r) = IFFT[Ψ_s(Q)]  # 逆 FFT
```

---

### processing/ - 处理脚本

#### process_au_v3.py (440 行) ⭐ 推荐

主处理脚本，使用正确提取的数据。

**功能:**
- 加载提取后的 Au 数据（历史 `*_correct.npy` 为按错码表以 int16 提取的产物，
  数值不可作定量依据；当前管线经 `core/dm4_io.py` 按 dataType=2 以 float32 解码）
- 预处理：偏移处理负值
- 运行 DPC 和 SSB 分析
- 生成对比图和电场映射
- 保存结果和元数据

**使用方法:**

```bash
python processing/process_au_v3.py
python processing/process_au_v3.py --data-dir D:\data --output-dir D:\out
```

**输出文件:**
- `*_dpc.png`: DPC 分析结果
- `*_v3_ssb.png`: SSB 重建结果
- `*_v3_comparison.png`: DPC vs SSB 对比
- `*_v3_efield.png`: 电场分析
- `*_v3_metadata.json`: 处理参数和结果元数据

#### extract_au_correct.py (173 行)

正确提取 Au 4D-STEM 数据。

**功能:**
- 使用 ncempy 读取 DM4 文件
- 识别正确的数据类型和字节序
- 提取 4D-STEM 数据立方体
- 裁剪感兴趣区域

**使用方法:**

```bash
python processing/extract_au_correct.py
python processing/extract_au_correct.py --base-dir D:\data --crop 128
python processing/extract_au_correct.py a.dm4 b.dm4 --out-dir out
```

**输出:**
- `Au_SI19_correct.npy`: 正确提取的 SI19 数据
- `Au_SI20_correct.npy`: 正确提取的 SI20 数据
- `Au_SI21_correct.npy`: 正确提取的 SI21 数据

#### batch_dpc.py (264 行)

批量 DPC 处理脚本。

**功能:**
- 批量处理多个数据集
- 自动估计 alpha 参数
- 生成汇总报告

**使用方法:**

```bash
python processing/batch_dpc.py --input-dir D:\dm4data --output-dir D:\results
```

新版 batch_dpc 不再依赖硬编码的偏移量、形状和 `>u2` 等错误 dtype 配置，
而是递归扫描输入目录并用 `core.dm4_io` 正确读取每个文件。

---

### diagnostics/ - 诊断脚本

#### diagnose_au.py (241 行)

Au 数据全面诊断。

**检查项目:**
- 基本统计（均值、标准差、范围）
- 饱和像素检测
- 平均衍射图案分析
- CoM 分析
- BF 盘估计
- 径向轮廓分析
- 扫描位置分析

**使用方法:**

```bash
python diagnostics/diagnose_au.py
```

#### check_raw_data.py (117 行)

原始 DM4 数据检查。

**功能:**
- 检查原始 DM4 文件结构
- 测试不同数据类型解释
- 验证数据偏移
- 与标准数据对比

#### check_cu_standard.py (176 行)

Cu 标准数据验证。

**功能:**
- 验证 Cu foil 标准数据质量
- 与 Au 数据对比
- 确认算法正确性

#### analyze_correct_data.py (233 行)

正确提取数据分析。

**功能:**
- 分析正确提取的数据
- 测试不同预处理方法
- 比较 CoM 计算结果

#### verify_dtype.py (166 行)

数据类型验证。

**功能:**
- 验证 DM4 数据类型
- 测试不同字节序
- 确认正确解释方式

#### visualize_dp.py (170 行)

衍射图案可视化。

**功能:**
- 可视化平均衍射图案
- 显示径向轮廓
- 比较不同处理方法

---

## 处理参数配置

### DPC 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `alpha_pixels` | 自动估计 | BF 盘半径（像素） |
| `center` | 自动计算 | 光束中心 (y, x) |
| `regularization` | 1e-3 | iDPC 正则化参数 |
| `rotation_deg` | 0 | 旋转角度（度） |

### SSB 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `alpha_pixels` | 自动估计 | BF 盘半径（像素） |
| `center` | 自动计算 | 光束中心 (y, x) |
| `regularization` | 1e-3 | SSB 正则化参数（相对最大权重） |
| `info_limit` | alpha×2 | 信息极限（像素） |

### 虚拟图像参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bf_radius` | alpha | BF 掩模半径 |
| `adf_inner` | alpha×1.2 | ADF 内半径 |
| `abf_inner` | alpha×0.4 | ABF 内半径 |
| `abf_outer` | alpha×0.9 | ABF 外半径 |

---

## 重现处理过程

### 快速开始

```bash
# 1. 进入 measure 目录
cd D:\data\4dSTEM\measure

# 2. 安装依赖
pip install numpy matplotlib ncempy

# 3. 提取数据（如果尚未提取）
python processing/extract_au_correct.py

# 4. 运行主处理
python processing/process_au_v3.py

# 5. 查看结果
# 结果保存在 results/ 目录
```

### 完整流程

```bash
# 步骤 1: 数据诊断
python diagnostics/check_raw_data.py
python diagnostics/diagnose_au.py
python diagnostics/check_cu_standard.py

# 步骤 2: 数据提取
python processing/extract_au_correct.py

# 步骤 3: 数据分析
python diagnostics/analyze_correct_data.py
python diagnostics/visualize_dp.py

# 步骤 4: 主处理
python processing/process_au_v3.py

# 步骤 5: 批量处理（可选）
python processing/batch_dpc.py
```

### 数据路径配置

所有脚本都支持命令行参数（见各脚本 `--help`），并可用环境变量配置
默认路径，未设置时回退到 `D:\data\...` 路径：

```powershell
# 原始 Au 数据集根目录
$env:STEM4D_DATA = 'D:\data\4dSTEM\20260707-Au'

# 标准数据目录（Cu foil / DPC Demo）
$env:STEM4D_STANDARD = 'D:\data\4dSTEM\standard\data'

# 分析数据与输出目录（默认分别为 <STEM4D_DATA>\analysis\data 与 ...\results\v2）
$env:STEM4D_ANALYSIS_DATA = 'D:\data'
$env:STEM4D_OUTPUT = 'D:\results'
```

---

## 结果评估

### 当前结果可靠性

| 分析类型 | 可靠性 | 说明 |
|----------|--------|------|
| Virtual BF/ADF/ABF | ⚠️ 低 | 数据质量问题 |
| CoM/DPC | ⚠️ 中低 | 趋势可能正确，但精度受限 |
| iDPC 相位 | ⚠️ 中低 | 定性可用，定量不可靠 |
| SSB 重建 | ❌ 不可靠 | 欠采样 + 数据质量问题 |
| 电场/电荷密度 | ⚠️ 中低 | 依赖 CoM 质量 |

### 与标准数据对比

| 指标 | Cu foil (标准) | Au SI19 | 评估 |
|------|----------------|---------|------|
| 数据类型 | 待复核（旧记录 uint8 系错码表） | float32 (dataType=2；旧记录 int16 系错码表) | dtype 须按 DM4 头部 dataType 码确认 |
| 负值像素 | 0% | 48.84% | ❌ 巨大差异 |
| 平均强度 | 50.2 | ~0 (原始) | Au 需偏移 |
| 中心/角落比 | 1.571 | ~1.0 | ❌ Au 无 BF 盘 |
| 探测器尺寸 | 512×512 | 32×32 | ❌ 256 倍差异 |
| 信背比 | 正常 | 0.016 | ❌ 极低 |

### 建议

1. **当前结果**: 仅供方法验证，不宜用于科学发表
2. **重新采集**: 使用更大探测器（≥128×128），避免过度背景扣除
3. **验证数据**: 用 DigitalMicrograph 打开原始文件检查
4. **算法验证**: 使用标准 Cu foil 数据验证算法正确性

---

## 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v1 | 2026-07-16 | 初始版本，使用错误数据类型 |
| v2 | 2026-07-19 | 修复数据类型，添加诊断脚本 |
| v3 | 2026-07-19 | 正确提取数据，偏移处理负值 |
| v4 | 2026-07-20 | **修复维度交换问题**，添加维度验证工具 |
| v2.0 重构 | 2026-07-31 | 统一 `core/dm4_io` 读取；GUI 接入 core 模块、复选框生效、单次读取、线程安全日志、协作式停止；修复 ePIE 窗口偏移、strain 断链导入、batch_dpc 死导入与错误 dtype；路径配置化；新增测试与 requirements；重新打包 exe |
| v2.1 修复 | 2026-09-06 | **P0 数值修复**：`preprocess` 默认不再做上限百分位裁剪（旧默认会把小盘径数据集的 BF 盘削平、摧毁 CoM 信号）；SSB 光阑/探针移位改为解析重建，消除 `np.roll` 环绕在欠采样探测器上制造的假 DO+ 重叠。**P1**：GUI 选项全部在主线程快照（修复 Tk 变量跨线程读取）；取消语义补全（CoM/SSB 取消时不再输出残缺结果）；Peak Pairs 接入 Stop 并向量化配对。**P2**：束心自动检测对非正总强度报错；维度 heuristic 对负值数据鲁棒；CoM 三套实现收敛为一套；新增 `core/pipeline` 共享数据准备层（GUI 与 batch_dpc 共用）；strain npz 增存 E 场/匹配数/拟合残差/参考斑点；README 与实现对齐。**P3**：GUI 关窗确认与全局异常弹窗、summary.csv 汇总、UPX 关闭、SSB 精度回归测试 |
| v2.2 修复 | 2026-09-28 | **P1**：修复 SSB 像差路径 `aperture_shifted` NameError（defocus/Cs 非 0 即崩，此前无测试覆盖）；内存链全面降载（提取单拷贝 float32、preprocess 单拷贝、虚拟探测器分块 einsum、SSB FFT 保留 complex64、ePIE 就地 float32 sqrt——128 裁剪×512 探测器场景峰值内存降约 4 倍）；维度自动交换加分数裕度闸（1.25×，弱证据保留头部布局），双分数与依据落盘 metadata。**P2**：SSB 奈奎斯特 bin 重复计数修复；iDPC 正则化/SSB 离焦/ePIE 步距进 GUI 高级参数；Bragg 盘检测大核走 FFT 相关（512² 探测器可用）；应变/取向细粒度进度进 GUI 日志；peak-pairs 阈值 mean+2σ 兜底；detect_rings 用实测束心；GUI 预检输出目录可写、同名 DM4 输出目录去重、应变/取向默认关闭、matplotlib 显式 Agg。**P3**：`*_ssb_object/_com_magnitude/_bf_intensity.npy` 落盘；metadata 增来源路径/标尺/预处理/维度分数/耗时；summary.csv 记录失败文件；README 删除不存在的 dm4_parser 条目并修正 SSB 正则化默认值；requirements 钉大版本 |

---

## 重要修复：维度交换问题 (v4)

### 问题描述

在DPC Demo标准数据处理中发现，**扫描维度和探测器维度被交换**，导致：
- BF图像显示软边圆形（探测器形状），而非周期性原子结构
- 原子位置和数量与原始数据不匹配
- DPC/SSB分析结果错误

### 问题原因

DM4文件中的数据布局为 `(det_y, det_x, scan_y, scan_x)`，但原始提取脚本错误地解释为 `(scan_y, scan_x, det_y, det_x)`。

### 修复方法

```python
# 错误（原始）
data = data.reshape(scan_y, scan_x, det_y, det_x)  # 维度交换

# 正确
data = data.reshape(det_y, det_x, scan_y, scan_x)
data = data.transpose(2, 3, 0, 1)  # 转换为 (scan_y, scan_x, det_y, det_x)
```

### 验证方法

使用 `core/dimension_utils.py` 中的工具：

```python
from dimension_utils import verify_dimensions, fix_dimensions

# 验证维度
result = verify_dimensions(data)
print(f"维度正确: {result['is_correct']}")
print(f"BF图像峰值数: {result['bf_peaks']}")  # 应>0表示有原子结构

# 自动修复
fixed_data, info = fix_dimensions(data, method='auto')
```

### 修复效果对比

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| BF图像形状 | 128×128 软边圆形 | 110×110 周期性结构 |
| BF图像峰值 | 0个 | 60个（原子柱） |
| Alpha估计 | 15像素（不合理） | 62像素（合理） |
| DPC相位范围 | [-0.4, 0.4] rad | [-12.8, 26.8] rad |
| SSB Q矢量 | ~300个 | 12,320个 |

### 新增文件

- `core/dimension_utils.py`: 维度验证和修复工具
- `standard/output/v2/`: 修正后的DPC Demo结果

---

## 参考文献

1. Ishikawa, K., et al. "Differential phase contrast STEM imaging." *Ultramicroscopy* (2016).
2. Pennycook, S. J., & Nellist, P. D. *Scanning Transmission Electron Microscopy*. Springer (2011).
3. Gatan DM4 File Format Documentation.

---

## 联系与支持

如有问题，请检查：
1. 运行环境是否满足要求
2. 数据路径是否正确配置
3. 依赖项是否完整安装

---

*文档生成时间: 2026-07-19*
*处理版本: v3*

<!-- README-QUICKREF:BEGIN 由 tools/gen-readme-block.py 生成，请勿手工编辑本区块 -->

---

## 速查

| 项 | 内容 |
|---|---|
| 当前版本 | **2.2.0** |
| 版本来源 | `stem_processor_gui.py` |
| 入口 | `stem_processor_gui.py` |
| 依赖锁定 | `requirements.lock.txt` |
| 许可证 | MIT |

**从源码运行**——必须使用本工具自己的虚拟环境，不要用 PATH 上的 `python`：
各工具依赖版本互不相同，共用解释器会互相污染。

```powershell
cd <本工具目录>
python -m venv .venv
.venv\Scripts\python -X utf8 -m pip install -r requirements.lock.txt
.venv\Scripts\python stem_processor_gui.py
```

**未纳入版本控制**：`.venv/`、`build/`、`dist/`、`release/`、`__pycache__/`、
测试数据与运行输出。具体规则见本目录 `.gitignore`。

<!-- README-QUICKREF:END -->
