# AIforTEM 仓库多维度并行审计报告

- **审计对象**：`D:\AIforTEM`（公开仓库 [linxi253/linxi](https://github.com/linxi253/linxi)，分支 `main`）
- **审计日期**：2026-10-01
- **审计方式**：六个维度**并行**独立审计 + 主审对全部 high 级发现逐条**独立复核**（只读，未安装依赖、未运行任何工具）
- **审计基线**：提交 `0df2ec42`（浅克隆，仅 1 次提交）

---

## 一、总体结论

**这是一个组织度明显高于同类个人科研仓库的 monorepo**：395 个 Python 文件、约 10.6 万行代码，全部通过语法解析（0 错误）；21 个子项目基本各自配有 README、`pyproject.toml` 与测试；已建立 CI（lint + 全库 compileall + 18 项 Windows 测试矩阵）；`全整合/temsuite` 是**真实的**运行时动态整合（16 个 ToolSpec + 同名模块归档），而非转发壳。

**最要紧的问题集中在三处，且都与"环境契约"有关：**

1. **CI 在中文 Windows 上装不上依赖** —— 锁文件是 UTF-8 无 BOM，CI 未加 `-X utf8`，pip 会按 locale 解码。实测 48 个 requirements/lock 文件中 **31 个在 cp1252 与 cp936 下都无法解码**，安装步骤会直接失败。
2. **环境供给脚本与文档同时失真** —— `.tools/provision-envs.py` 硬编码的母本解释器实测是 **Python 3.7.1**，而各项目要求 `>=3.10`；`环境隔离说明.md` 所描述的 4 个目录、全部 venv、py312 母本、清华 pip.ini **在本检出中一个都不存在**。
3. **跨项目版本契约被打破** —— `全整合` 用自身解释器启动 `atom_center` 子进程，使后者锁定的 `numpy==1.26.4 / opencv==4.11` 完全失效（实际跑在 `numpy==2.2.6 / opencv==5.0.0.93`）。

**安全底线是干净的**：全库无硬编码密钥、无个人信息、无 `pickle`、无 `shell=True`、无 `os.system`、无出网请求；XML 用 `defusedxml` 并配 XXE 回归测试，zip 解压有逃逸校验。但**模型加载这条边界形同虚设**：`atom_center` 有两处把 SHA-256 校验写成"自己和自己比"的同义反复，校验永远通过，随后文件被 YOLO 反序列化加载。

---

## 二、克隆与仓库状态核查（主审亲自完成）

| 检查项 | 结果 |
|---|---|
| 远端一致性 | ✅ 本地 `HEAD` = `origin/main` = GitHub `main` 树 SHA `0df2ec42…`，**克隆完整** |
| 全库语法扫描 | ✅ 395 个 `.py` 经 Python 3.12 `ast.parse`，**0 语法错误**（105,769 行） |
| 提交历史 | ⚠️ 浅克隆，仅 1 次提交（`AIforTEM 电镜数据处理工具集`），**无法审计历史泄露** |
| 体积卫生 | ✅ 无 `.exe/.dll/.pt/.onnx` 等大文件入库；最大跟踪文件为 `ppa.py`（269 KB）；`.git` 仅 1.79 MiB |
| 强制入库 | ✅ `git ls-files -ci` 为空，无"被 ignore 却已提交"的文件 |
| 工作区脏污 | ⚠️ 仅 2 项未跟踪：根目录 `nul` 与 `01-视频与数据提取/视频切片工具/nul` |
| 分支/tag | `main` 无 tag、无 Release |

### ⚠️ 关于 `nul`：一个被 PowerShell 放大、但确实有害的真实问题

`Get-ChildItem` 会报告约 **200 个** `nul`，但那是 PowerShell provider 的**幻影条目**；用 `.NET Directory.GetFiles` 原始枚举确认**真实存在的只有 2 个**（根目录 0 字节；视频切片工具目录 34 字节）。

两者均**未被跟踪**（不在 GitHub 上），也**未被 `.gitignore` 覆盖**，因此常驻 `git status`。34 字节那个的内容是一段 PowerShell 报错文本——典型的 `2>nul` 重定向误写产物，**时间戳 22:20，早于本次会话（22:53）**，属先前会话遗留。

**实际危害已复现**：`nul` 是 Windows 保留设备名，对仓库根做整库 ripgrep 会直接失败：

```
rg: D:\AIforTEM\nul: 函数不正确。 (os error 1)
rg: D:\AIforTEM\01-视频与数据提取\视频切片工具\nul: 函数不正确。 (os error 1)
```

同样地，`python .tools/check-env-isolation.py` 在默认 3.7.1 下会崩在 `nul` 上（`OSError: [WinError 1]`）。

---

## 三、各维度评价

| 维度 | 总体评价 |
|---|---|
| **结构与组织** | 组织度好：子项目结构统一，`strainpp` 根目录平铺 `.py` 经核实是 2–9 行的**兼容转发壳**而非重复实现，README 主动说明了 `05-` 双编号的历史原因。主要问题是"复制而非共享"：`09-HRTEM模拟` 与 `010-STEM模拟` 是同源副本。 |
| **代码质量** | "单文件工具成熟、跨工具复用为零"。中文 docstring 与"为什么这么做"的注释密度很高，核心算法普遍用 `with` 管理句柄；无 TODO/FIXME、无 `pdb`。但 TIFF/DM 读取有 5 份独立实现，静默吞异常 169 处，memmap 普遍不释放。 |
| **安全与合规** | 底线干净（无密钥/无个人信息/无 `shell=True`/无出网），`defusedxml`、zip 逃逸校验、ffmpeg PROVENANCE 哈希流程都到位。缺口在模型加载：白名单默认空 + 错误提示教用户自助 `--allow`（等价 TOFU），以及两处同义反复的哈希校验。 |
| **依赖与环境** | 思路清晰（一项目一环境一锁，`stem-optimize` 的锁还带 `--hash`），但三个 `.tools` 脚本与文档承诺不一致，跨项目版本契约存在真实冲突。**注：全仓源码中 grep 不到任何 PyQt/PySide 的 import，"双体系并存"不成立——GUI 一律是 tkinter/ttkbootstrap。** |
| **测试与 CI** | 基础设施超出同类仓库水平：18 个 `tests/` 目录共 **1168 个 `def test_`**，绝大多数是真断言，失败路径与数值不变量覆盖扎实。缺口在覆盖面：有测试却完全不在 CI 矩阵的项目，以及 CI 注释对 pyyaml 缺口的描述有误。 |
| **文档一致性** | 根 README 与 README_EN 的 **20 行工具一览表逐条核对全部通过**（版本号/目录/入口文件均真实存在，两份 README 逐格一致），19 个子项目版本常量与 CHANGELOG 对得上，GPL 声明链条完整。问题在 `环境隔离说明.md` 整体脱离现实，及若干子文档内部自相矛盾。 |

---

## 四、需要优先处理的问题

> 每条均经主审**独立复核**（✓ = 主审亲自复现）。

### 🔴 高优先级

**1. CI 安装依赖会在 Windows 上直接失败（编码问题）** ✓
- 位置：[`.github/workflows/ci.yml:116`](.github/workflows/ci.yml#L116)
- 证据：工作流中 grep `PYTHONUTF8|utf8|chcp` 为空；锁文件头为 UTF-8 无 BOM，pip 的 `auto_decode` 会退回 `locale.getpreferredencoding()`。**主审实测：48 个 requirements/lock 文件中，cp1252 与 cp936 下各有 31 个 `UnicodeDecodeError`。**
- 对照：[`环境隔离说明.md:113`](环境隔离说明.md#L113) 自己就写明"中文 Windows 上 pip 默认按 GBK 读取，`-X utf8` 是必需的"。
- 建议：CI 中改用 `python -X utf8 -m pip install`，或设置 `PYTHONUTF8=1` 环境变量。

**2. 环境供给脚本必然失败（母本解释器版本错误）** ✓
- 位置：[`.tools/provision-envs.py:26`](.tools/provision-envs.py#L26)
- 证据：`BASE_PY = Path(r"C:\ProgramData\Miniconda3\python.exe")`，**实测该解释器为 Python 3.7.1**，而各项目 `requires-python = ">=3.10"`、锁中钉 `numpy==2.2.6`（需 ≥3.10）。且该脚本自身用了 `Path.unlink(missing_ok=True)`（3.8+ 才有），**在 3.7 上根本跑不起来**。
- 文档却称 base 是 3.10.9（[`环境隔离说明.md:47`](环境隔离说明.md#L47)）。

**3. 全整合与 atom_center 的版本契约被打破** ✓
- 位置：[`全整合/temsuite/registry.py:317`](全整合/temsuite/registry.py#L317)、[`全整合/temsuite/app.py:357-359`](全整合/temsuite/app.py#L357-L359)
- 证据：`run_mode="subprocess"` + `cmd = [sys.executable, "-m", spec.subprocess_module]`，用的是**套件自己的解释器**。而 `atom_center` 锁 `numpy==1.26.4`/`opencv-python==4.11.0.86`，`全整合` 锁 `numpy==2.2.6`/`opencv-python==5.0.0.93`；套件锁中亦无 `torch/onnx/ultralytics/PyYAML`。
- 后果：标注工具实际运行在与其声明完全不同的依赖版本上，可能崩溃或产出错误结果。

**4. 有测试的项目被整体漏出 CI 矩阵** ✓
- 位置：[`.github/workflows/ci.yml:54-75`](.github/workflows/ci.yml#L54-L75)
- 证据：`03-应变分析/原子中心识别模型开发` 有 **14 个测试文件、77 个 `def test_`**，是**唯一**有测试却不在 18 项矩阵中的项目；ci.yml 中 grep `原子中心` 为空。且该项目在无 `pip install -e .` 的干净环境中连 `import atom_center` 都会 `ModuleNotFoundError`（无 `conftest.py`）。
- 另：`06-独立脚本`、`10-DSH集成` 无任何测试。

**5. 环境隔离文档整体脱离仓库现实** ✓
- 位置：[`环境隔离说明.md:3`](环境隔离说明.md#L3)、`:47`、`:61`、`:79-88`、`:138`、`:204`
- 证据：文档所述 `010-QSTEM模拟2`、`开发中`、`08-历史版本`、`.tools/re-tools` **Test-Path 全部为 False**；全仓 `pyvenv.cfg` 数量为 **0**；`py -0p` 无 py312；`pip config list` 为空（无清华镜像配置）；引用的整改证据目录 `.review-tmp/base-env-snapshot-…` 不存在且已被 `.gitignore` 忽略。
- 建议：该文档需按当前实际重写，或明确标注"描述的是作者本机环境，非仓库内容"。

**6. memmap 打开后从不释放，Windows 下锁死源文件**
- 位置：[`04-统计分析/统计面积/统计面积.py:543,620`](04-统计分析/统计面积/统计面积.py#L543)、[`04-统计分析/原子衬度统计/contrast_core.py:676`](04-统计分析/原子衬度统计/contrast_core.py#L676)
- 证据：`data = tifffile.memmap(file_path, mode='r')` 后直接 `self.tiff_stack = stack` 交给 GUI 长期持有，该文件内 `.close()`/`del` 命中数为 **0**。用户将无法重命名/覆盖/删除已打开的 TIFF。
- 反例：同仓库 `原子识别纯算法` 已实现 `close_tiff_stack()` 并在切换/退出时调用——说明团队知道该问题。
- 另：[`04-统计分析/特征区域演化分析/应力面积统计.py:615`](04-统计分析/特征区域演化分析/应力面积统计.py#L615) 用 `tiff.memmap(tif_path)` **未传 mode**（默认 `'r+'` 可写），而同仓库 `统计面积.py:537-539` 的注释明确记录过"可写映射反而会在 Windows 上把源文件锁死"。

**7. 模型完整性校验被写成同义反复，完全失效** ✓
- 位置：[`src/atom_center/point_selection.py:97`](03-应变分析/原子中心识别模型开发/src/atom_center/point_selection.py#L97)、[`scripts/dsh_verify_target_acceptance.py:56`](03-应变分析/原子中心识别模型开发/scripts/dsh_verify_target_acceptance.py#L56)
- 证据：`TorchBackend(checkpoint, expected_sha256=sha256_file(checkpoint), …)` —— **用刚算出的哈希作为期望值**，于是 `backends.py:82-84` 的 `if sha256_file(path) != expected_sha256: raise` 永远不触发，随后该 `.pt` 被 `YOLO(...)` 反序列化加载。
- 建议：期望哈希必须来自可信来源（模型清单/manifest），不能现场计算。

**8. 跨工具重复实现，数值核心被逐行复制**
- 位置：[`atomic_core.py:411-451`](03-应变分析/原子识别纯算法/atomic_core.py#L411-L451) 与 [`ppa.py:3835-3879`](03-应变分析/原子级应力分析-PPA/ppa.py#L3835-L3879)
- 证据：二维高斯亚像素精炼被逐行复制，连 `bounds` 数值与 `maxfev=500` 都完全相同；**两处都在拟合失败时 `except Exception: pass` 静默回退到整数像素坐标**，而该坐标直接决定后续应变/应力数值，用户收不到任何提示。
- 另：`dm4_io.py`/`dm4io.py`、两份 `tif_io.py`、两份 `tiff_handler.py` 各有独立实现，其中 `dm4_io.py:1-3` 自称 "the single source of truth"，但 EELS 侧另有 17 KB 的实现且互不 import。

### 🟡 中优先级

| # | 位置 | 问题 |
|---|---|---|
| 9 | [`nul`](nul)、[`01-视频与数据提取/视频切片工具/nul`](01-视频与数据提取/视频切片工具/nul) ✓ | Windows 保留名残留文件，破坏整库 ripgrep 与 `check-env-isolation.py`；应删除并加入 `.gitignore` |
| 10 | [`09-HRTEM模拟/tem_sim/microscope.py`](09-HRTEM模拟/tem_sim/microscope.py) ↔ [`010-STEM模拟/stem_sim/microscope.py`](010-STEM模拟/stem_sim/microscope.py) ✓ | **SHA-256 完全相同**（`A8C76D29…`，均 6792 B）；`gauss3.txt`、`peng_high.json` 亦 byte-identical；工具层 `worker.py` 92%、`render.py` 86% 行相同 → 长期双维护 |
| 11 | [`010-STEM模拟/requirements.txt:9`](010-STEM模拟/requirements.txt#L9) ✓ | 声明 `numpy>=1.26`，却调用 NumPy 2.0 才有的 `np.trapezoid`（[`phonons.py:111`](010-STEM模拟/stem_sim/phonons.py#L111)）→ 按声明下界安装会 `AttributeError`；EELS 项目对同一 API 正确要求了 `>=2.0` |
| 12 | [`ci.yml:63`](.github/workflows/ci.yml#L63) ✓ | 注释称"测试导入 yaml"，但 **PPA 的 tests/ 中 grep `yaml` 结果为 0**；真正需要 pyyaml 的是 `atom_detector`（`detector.py:91`、`prepare_dataset.py:21`），而主 `requirements.lock.txt` 始终没有它 |
| 13 | [`models/allowlist.json:3`](03-应变分析/原子级应力分析-PPA/atom_detector/models/allowlist.json#L3) ✓ | 白名单默认为**空数组**，错误提示直接教用户运行 `--allow` 自助登记哈希 → 实为 TOFU 而非来源校验 |
| 14 | [`scripts/diagnose_learning.py:80`](03-应变分析/原子中心识别模型开发/scripts/diagnose_learning.py#L80) | 显式 `torch.load(..., weights_only=False)`，主动关闭安全反序列化开关 |
| 15 | [`04-STEM-Processor/requirements.lock.txt:26`](05-4D-STEM分析/4D-STEM-Processor/requirements.lock.txt#L26) ✓ | `requirements.txt` 写 `pytest>=8.0,<9`，锁文件却钉 `pytest==9.1.1`；CI 按锁安装 → 用 9.x 跑声明 `<9` 的测试 |
| 16 | [`.tools/check-env-isolation.py:255`](.tools/check-env-isolation.py#L255) | 第 [5] 项只 `rglob("requirements.lock.txt")` 且只认 `.venv` → **8 个锁**（含 2 个带自动生成标记的 dev 锁）与 `.venv-build/.venv-run` 完全不被检查 |
| 17 | [`README.md:76`](README.md#L76) ✓ | 称 `原子识别纯算法` 提供 `run.bat`，该文件**不存在**，实际为 `启动原子识别工具.bat`（README_EN.md:83、环境隔离说明.md:101 同错） |
| 18 | [`03-应变分析/原子中心识别模型开发/README.md:91`](03-应变分析/原子中心识别模型开发/README.md#L91) | 速查块称"无锁定文件，使用 requirements.txt"，与同文件 :37 及实际（`requirements/` 下 6 个锁）矛盾 |
| 19 | [`02-图像处理/README.md:10`](02-图像处理/README.md#L10) | 把 stem-optimize 写成 "v2.0"，代码常量为 `2.1.0`，与根 README、全整合 README 矛盾 |
| 20 | [`全整合/README.md:187-198`](全整合/README.md#L187-L198) | "未整合的工作区内容"清单遗漏 `离域效应去除工具`，而同文件 :77-78 明确说它未接入 |
| 21 | [`.github/workflows/ci.yml:74`](.github/workflows/ci.yml#L74) | 010-STEM模拟 只跑 `verify_physics.py`；README 列为验收步骤的 `test_e2e_au.py`/`test_gui_smoke.py` 从未进 CI |
| 22 | [`09-HRTEM模拟/tests/test_tool.py:216`](09-HRTEM模拟/tests/test_tool.py#L216) | `@pytest.mark.skipif(os.name == "nt")` 而 CI 只在 `windows-latest` 跑 → 该用例 100% 被跳过 |
| 23 | 全仓库 | 静默吞异常 **169 处**（1 处裸 `except`，位于 [`check_raw_data.py:55`](05-4D-STEM分析/4D-STEM-Processor/diagnostics/check_raw_data.py#L55) ✓）；[`contrast_core.py:521-550`](04-统计分析/原子衬度统计/contrast_core.py#L521-L550) 会把元数据解析失败静默降级为"空文件" |
| 24 | [`label_review.py:474`](03-应变分析/原子级应力分析-PPA/atom_detector/annotate/label_review.py#L474) 等 3 处 | 文本模式 `open()` 未指定 `encoding`，中文 Windows 下会乱码或 `UnicodeEncodeError`（其余 90+ 处均已正确指定） |
| 25 | 全仓库 | **1061 处 `print`（119 个文件）**，仅 8 个文件做了 stdout UTF-8 保护；[`xcheck_abtem.py:82`](010-STEM模拟/tests/xcheck_abtem.py#L82) 实测在 cp936 下 `UnicodeEncodeError` |

### 🟢 低优先级 / 改进建议

- **超长函数**：15 个函数体 ≥200 行，最长 `strain_analysis.py:65-607`（543 行）；`ppa.py` 单文件 5602 行（占全库 5.3%）。
- **CI 与本地脚本脱节**：`.tools/verify-envs.py` 有 17 项 JOBS，与 CI 的 18 项矩阵**互不覆盖**（它独有 atom_center，CI 独有 图像加滤镜工具/非晶面积统计）；且 ci.yml **从未调用**这三个 `.tools` 脚本。
- **失效引用**：19 个子项目 README 的速查块都标注由 `tools/gen-readme-block.py` 生成，**该脚本不存在**；[`strainpp/README.md:245`](03-应变分析/strainpp-GPA应变分析/README.md#L245) 指向不存在的 `.github/workflows/test.yml` ✓；`08-历史版本` 被 15 处引用但**目录不存在** ✓。
- **NOTICE.md 不完整**：未登记已入库的 `peng_high.json`（代码注明取自 abTEM 仓库）。
- **签名密码暴露面**：[`strainpp/build.ps1:50-54`](03-应变分析/strainpp-GPA应变分析/build.ps1#L50-L54) 把 PFX 密码作为命令行参数传给 `signtool.exe`，可被同机进程经 WMI 读取。
- **规范不统一**：21 份 `.gitignore` 去重后 15 种；`010-STEM模拟/.gitignore` 第 33/36 行重复；`.gitattributes` 仅覆盖 4/21 项目且行尾策略互相矛盾。
- **两个零断言伪测试**：[`test_failure_paths.py:179`](04-统计分析/特征区域演化分析/tests/test_failure_paths.py#L179)、[`test_tool.py:45`](09-HRTEM模拟/tests/test_tool.py#L45)。
- **zip 解压边界**：[`audit_training_update_inputs.py:59-61`](03-应变分析/原子中心识别模型开发/scripts/audit_training_update_inputs.py#L59-L61) 未处理空成员名（`..` 与绝对路径穿越已正确拦截）。
- **`__pycache__` 未清理**：磁盘上有 389 个 `.pyc`（均未跟踪、已被 ignore），建议清理。
- **目录编号不自洽**：两个 `05-` 前缀、`010-` 用三位数（字典序会插到 `01-` 与 `02-` 之间）。

---

## 五、做得好的地方（同样需要指出）

- ✅ **零语法错误**：395 个文件、10.6 万行，Python 3.12 `ast` 全部通过，与 CI 的 `compileall` 口径一致。
- ✅ **无凭据泄露**：`sk-`/`ghp_`/`AKIA`/`xox`/`AIza`/`glpat_`/`hf_` 及私钥块正则**零命中**；无 `C:\Users\<用户名>`、邮箱、电话、学号。
- ✅ **无危险调用**：无 `pickle`、无 `shell=True`、无 `os.system`、无命令字符串拼接（全部列表参数）、无出网请求。
- ✅ **供应链意识**：FFmpeg 有 `PROVENANCE.md` + 哈希校验构建流程；XML 用 `defusedxml` 并配 XXE 回归测试；zip 解压有逃逸校验。
- ✅ **许可证链条完整**：`strainpp-GPA应变分析/LICENSE` 为 553 行 GPL-3.0 全文，`pyproject.toml` 标 `GPL-3.0-or-later`，NOTICE.md 与根 README 的传染性表述一致。
- ✅ **README 准确度高**：20 行工具一览表的版本号、源码目录、入口文件**逐条核对全部通过**，中英文两版逐格一致；19 个子项目版本常量与 CHANGELOG 对得上。
- ✅ **测试是真断言**：1168 个 `def test_`，`np.testing.assert_*`/`self.assert*` 为主，伪测试仅 2 个；失败路径与数值不变量覆盖扎实。
- ✅ **已建 CI**：lint（E9/F63/F7/F82）+ 全库 `compileall` + 18 项 Windows 矩阵，且 CI 注释里主动记录了已知缺口。
- ✅ **主动记录技术债**：`loader.py:26-30` 自述模块隔离的已知限制，`统计面积.py:537-539` 记录了 memmap 锁文件的踩坑原因——注释质量高于平均水平。

---

## 六、建议的后续动作（按优先级）

1. **立即修**：CI 加 `python -X utf8`（或 `PYTHONUTF8=1`）—— 这是当前 CI 能否跑通的前提。
2. **立即修**：删除两个 `nul` 文件并把 `nul` 加入 `.gitignore` —— 一次性解决 ripgrep 与自检脚本崩溃。
3. **修契约**：给 `atom_center` 子进程用**它自己的 venv 解释器**启动，或把 `全整合` 的 numpy/opencv 对齐到 `atom_center` 的锁定版本。
4. **补 CI**：把 `原子中心识别模型开发` 加进矩阵（需 `pip install -e .`），并修正 `Known gaps` 中关于 pyyaml 的错误描述。
5. **修校验**：让 `expected_sha256` 来自模型清单而非现场计算；`allowlist.json` 至少预置官方模型的哈希。
6. **修资源**：给 `统计面积`/`原子衬度统计` 的 memmap 补 `close` 路径（可复用 `原子识别纯算法` 的 `close_tiff_stack`），`应力面积统计.py:615` 补 `mode='r'`。
7. **重写文档**：`环境隔离说明.md` 按当前实际重写；修正 `run.bat`、`requirements.txt`、`v2.0`、`08-历史版本`、`gen-readme-block.py` 等失效引用。
8. **长期**：抽取共享的 TIFF/DM I/O 层与高斯精炼核心，消除 `09-HRTEM模拟`/`010-STEM模拟` 的双维护；给静默 `except` 至少补日志。

---

## 七、核查方法与覆盖范围

**本次做了什么**
- 六个维度**并行**独立审计（结构/质量/安全/依赖/测试文档），每个维度由独立子代理在只读约束下完成。
- 主审独立完成基线核查：克隆完整性（比对 GitHub API 树 SHA）、全库 AST 语法扫描、体积与跟踪状态统计、`nul` 幻影辨析（`.NET` 原始枚举 + `\\?\` 扩展路径读取）。
- **主审对全部 high 级发现逐条独立复现**（表中标 ✓ 者），包括亲自触发 ripgrep 失败、读取同义反复校验代码、比对文件哈希、复现锁文件解码失败。

**没做什么 / 局限**
- **未运行任何测试或工具**：本机未安装各工具依赖（tkinter 以外的 GUI 库、torch、numpy、tifffile 等），Python 为 3.7.1 + 3.12。所有结论均为静态审计与只读命令所得，**"测试是否全绿"无法核实**。
- **未审计提交历史**：浅克隆仅 1 次提交，无法判断历史中是否曾泄露凭据。
- **未验证算法科学正确性**：GPA 应变、多层法模拟等物理结果的对错不在本次范围。
- **未核实 CI 在 GitHub 上的实际运行结果**，也未联网核对 BtbN 发布页的归档摘要。
- **未检查二进制内容**：`ffmpeg.exe` 与 DLL 不在仓库内，仅凭 `PROVENANCE.md` 静态判读；389 个 `.pyc` 未做字符串扫描。
- **`nul` 条目无法按常规方式读取**（Windows 保留名），本次通过 `\\?\` 扩展路径绕过确认其大小与内容。
