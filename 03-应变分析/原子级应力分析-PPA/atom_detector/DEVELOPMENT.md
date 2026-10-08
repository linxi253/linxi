# atom_detector 源码与验证导航

PPA 内的可选深度原子检测与训练子项目，稳定版 PPA 主程序当前不加载其权重。

**状态：支持/维护目录。** 属于 03-应变分析/原子级应力分析-PPA 仓库中的子目录。 工作约定见 [AGENTS.md](AGENTS.md)，下面命令从本项目目录执行。

## 入口与版本

- 入口文件：[model_security.py](model_security.py)。
- 版本来源：未登记统一静态版本源；按 README、源码和构建记录分别核实，不能以旧 EXE 文件名推断。
- 锁文件：未登记，不能宣称依赖环境已可复现。
- 构建配置：本批未登记打包配置，不推断已有成品与源码一致。

## 按任务定位

| 任务 | 文件 | 关键符号 |
|---|---|---|
| 模型身份和白名单 | [model_security.py](model_security.py) | verify_model、load_allowlist |
| 模型推理 | [infer/detector.py](infer/detector.py) | AtomDetector、quick_detect |
| 数据划分 | [dataset/prepare_dataset.py](dataset/prepare_dataset.py) | prepare_dataset |
| 人工审核 | [annotate/label_review.py](annotate/label_review.py) | LabelReviewApp |
| 可选接入 | [integration/ppa_plugin.py](integration/ppa_plugin.py) | DLDetectDialog、check_model_available |

## 保持的约束

- 不能把此可选子项目与独立的“原子中心识别模型开发”仓库混为同一训练/发布流程。
- 模型加载前保持 models/allowlist.json 的 SHA-256 校验；没有权重和数据不能宣称可开箱推理。
- 按原始图像/实验来源划分数据，防止增强样本泄漏；检测框中心用于定量前仍需亚像素精修。
- 父项目 tests/test_core.py 与 test_fix_regressions.py 覆盖本子项目的部分回归，命令从 PPA 根目录执行，见上一级 DEVELOPMENT.md。

## 环境与验证

此支持目录没有单独登记的本地解释器/锁；按原文档和所属项目准备隔离环境，导航检查不执行生成、训练或模型加载。 参见 [工作区环境约定](../../../环境隔离说明.md)。以下命令是后续功能改动的验证入口，本轮没有执行它们。

当前未提供可直接照跑的统一业务命令。原有用法与前置条件见 README；导航维护不执行该目录中的业务/生成脚本。

未登记此目录独立测试入口。只读导航检查只证明文件/符号存在；实际工作需按用途制定验证。

现有测试文件：未登记独立测试文件；不能把导航检查通过当成功能验证。

若使用 pytest，先核实测试环境包含 pytest；运行锁不保证含开发依赖。unittest 和独立自检按各自入口执行，不能统一替换成 pytest。已存在的历史通过数字不是本轮测试结果。

## 集成与相关资料

当前 TEM Suite 注册表没有此目录条目；不为完善导航自动接入。

- [requirements.txt](requirements.txt)
- [config.yaml](config.yaml)
- [models/allowlist.json](models/allowlist.json)
- [../DEVELOPMENT.md](../DEVELOPMENT.md)
- [../tests/test_core.py](../tests/test_core.py)
- [../tests/test_fix_regressions.py](../tests/test_fix_regressions.py)

只改文档时运行 [工作区只读导航检查](../../../.tools/check-codex-navigation.py)，并核对源文件差异；检查脚本不会导入本项目。
