# HAADF-STEM 特征区域演化分析工具

对 HAADF-STEM 多页 TIFF 时间序列进行背景归一化、Otsu 分割、形态学清洗，并统计特征区域面积和归一化柱强度随时间的变化。

## 直接运行

```text
dist/HAADF-STEM特征区域演化分析工具.exe
```

程序会弹出 TIFF 文件选择框，随后询问帧时间间隔与样品描述，并在输入文件旁创建带时间戳的 `NatureStyle_Analysis_*` 输出目录。

## 从源码运行

```powershell
python -m pip install -r requirements.txt
python .\应力面积统计.py
```

需要精确复现分析环境时，改用锁定版本安装：`python -m pip install -r requirements.lock.txt`。

也可通过命令行指定参数：

```powershell
python .\应力面积统计.py --file input.tif --dt 0.2 --title "Fe surface reconstruction"
python .\应力面积统计.py --file input.tif --dt 0.2 --shared-threshold
python .\应力面积统计.py --file input.tif --output D:\results --min-size 6 --dpi 300
```

- `--file`：输入 TIFF。
- `--dt`：帧间隔，默认 0.2 秒（必须为正的有限数）。
- `--title`：图表标题。
- `--shared-threshold`：全序列统一 Otsu 阈值（逐帧阈值中位数），消除阈值随帧漂移混入面积序列的噪声。
- `--output`：输出目录（默认在输入文件旁创建带时间戳的 `NatureStyle_Analysis_*` 目录）。目标目录已存在且非空时会拒绝写入以免覆盖旧结果。
- `--min-size`：形态学清洗保留的最小对象面积（像素），默认 4。
- `--dpi`：输出图像 dpi，默认 600（1–10000）。

## 处理与输出

处理流程：

1. 逐帧清洗非有限坏像素（NaN/inf 以有限像素中值替换；单帧坏点 >50% 判定输入损坏并终止）；
2. 用多帧边缘区域中值估计背景；
3. 按背景强度归一化，降低束流波动影响；
4. Otsu 自适应阈值分割（默认逐帧独立；`--shared-threshold` 用全序列统一阈值）；
5. 去除小对象并做闭运算；
6. 逐帧计算区域像素数、平均强度和强度置信区间（空间自相关折减有效样本量）；Otsu 失败的帧记为无效帧（数值为 NaN、`Frame_Valid=0`），分割成功但掩膜为空的帧面积=0、强度为 NaN（旧版把这类帧的强度记为 1.0，会污染统计）；
7. 生成 Fig1/Fig2 子图、组合图、分割质检图、补充图、CSV/JSON 和 Markdown 报告。

主要输出包括 `Fig1A`–`Fig1F`、`Fig2A`–`Fig2F`、两张组合图、`Segmentation_QA.png`（首/中/末帧原图 + 分割掩膜叠加，供目检分割质量）、补充图、`Publication_Data.csv`、`Statistical_Summary.md`、`Processing_Metadata.json`（机器可读元数据，含输入文件 SHA-256）和 `ANALYSIS_REPORT.md`。`Publication_Data.csv` 每帧一行，含时间、面积、归一化强度、实际使用的分割阈值、面积不确定度半宽、强度 95% CI 半宽、N_eff 与帧有效性标记；`Statistical_Summary.md` 记录全部处理参数（背景基准、形态学参数、阈值模式、坏像素统计）保证可复现。图像默认以 600 dpi 保存。

## 运行测试

```powershell
python -m pip install -r requirements.txt pytest
python -m pytest tests/ -q
```

## 主要文件

- `应力面积统计.py`：完整命令行/文件选择入口和分析逻辑。
- `requirements.txt`：运行依赖。
- `特征区域演化分析.spec`：PyInstaller 配置。
- `dist/HAADF-STEM特征区域演化分析工具.exe`：已打包程序。
- `应力图说明.txt`：图表和分析意图的补充说明。

## 限制

- 面积单位默认是像素数，代码没有读取像素标尺。
- 背景来自图像边缘；若边缘本身含强特征，归一化会偏差。
- 形态学参数默认 `min_size=4`，可用 `--min-size` 调整；逐帧 Otsu 阈值已随 CSV 导出，并可用 `--shared-threshold` 以全序列统一阈值复算对比。
- “突变点”“状态转移”等高级图仅是统计描述（Fig2F 异常评分为 mean+3SD 启发式，非显著性检验），不能在没有实验依据时直接解释为物理机制。
- 趋势 p 值未校正时间自相关，应视为描述性指标。
- 强度 95% CI（z·σ/√N_eff）是区域空间均值的代表性区间：混合了原子柱间真实物理不均匀性与测量噪声，未传播 I₀ 估计误差，且阈值选择对均值存在向高值的偏置。不要将其当作单帧测量噪声的置信区间做显著性比较。

<!-- README-QUICKREF:BEGIN 由 tools/gen-readme-block.py 生成，请勿手工编辑本区块 -->

---

## 速查

| 项 | 内容 |
|---|---|
| 当前版本 | **2026.09.2** |
| 版本来源 | `应力面积统计.py` |
| 入口 | `应力面积统计.py` |
| 依赖锁定 | `requirements.lock.txt` |
| 许可证 | MIT |

**从源码运行**——必须使用本工具自己的虚拟环境，不要用 PATH 上的 `python`：
各工具依赖版本互不相同，共用解释器会互相污染。

```powershell
cd <本工具目录>
python -m venv .venv
.venv\Scripts\python -X utf8 -m pip install -r requirements.lock.txt
.venv\Scripts\python 应力面积统计.py
```

**未纳入版本控制**：`.venv/`、`build/`、`dist/`、`release/`、`__pycache__/`、
测试数据与运行输出。具体规则见本目录 `.gitignore`。

<!-- README-QUICKREF:END -->
