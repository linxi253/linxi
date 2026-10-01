# 训练、数据审计与 ONNX 推理使用说明

版本：0.3.0；2026-09-06。M1～M3 软件流程已验收；真实模型精度与 PPA GUI 联调尚未验收。

## 1. 本机入口

在“原子中心识别模型开发”目录运行 PowerShell。环境已准备好：

```powershell
.\scripts\atom-center.ps1 --help
```

此入口使用项目内官方 CPython 3.10.11，并共享项目 `.venv` 的依赖。已验证的组合为 PyTorch 2.7.1+cu128、Ultralytics 8.4.135、ONNX 1.19.1、ONNX Runtime 1.20.1。原 Miniconda Python 3.10.9 在 ONNX 原生校验时发生访问冲突；独立运行时解决了本次复现的问题，没有替换系统或 Miniconda DLL。

新机器使用 Python 3.10 创建环境，再执行：

```powershell
.\scripts\setup_environment.ps1
```

仅需要补装项目内 CLI 运行时时执行 `scripts/setup_cli_runtime.ps1`。官方 Python 压缩包的 SHA-256 固定在该脚本中。

标注 GUI 保持使用 `.venv` 的原启动方式：

```powershell
.\.venv\Scripts\atom-annotate.exe --project "<任务目录>/annotation_project.json"
```

项目内独立 CLI 运行时不包含 Tk，不用于启动标注 GUI。旧 EXE 不认识项目 schema v2；需要打开迁移副本时使用当前源码标注工具或之后重新打包的版本。

## 2. 先处理现有标注

**用户指定保留测试集（2026-09-09）：** `<仓库根>/03-应变分析/原子标注/测试集` 中的图像及其后续标签只用于测试。训练/验证项目列表必须显式排除此目录、相同图像副本及关联派生图；不可因为它位于“原子标注”下就递归纳入训练。文件身份及来源检查见 `reports/testset_reservation_20260909.json`，当前只有 6 张图像，没有坐标真值。CLI 不按文件夹名称自动识别测试用途，组装前须核对项目列表。

已为原来 5 个任务生成迁移副本：

`data/annotation_projects/migrated_20260906/<原任务名>/annotation_project.json`

迁移保留全部标签文件字节，新增原图文件/平面摘要、帧身份及迁移说明。新增摘要只能证明迁移时观察到的文件，不能证明标注当时的原始内容。

现有审计在 `reports/current_annotation_audit_m123.json`。四个任务的同一原图有不同标注版本；来源元数据也不够完整，目前不能直接导出正式训练集。

在副本中补充真实的 `sample_id`、`acquisition_id`，需要时填写 `parent_field_id` 或 `parent_source_sha256`。同一父视野、连续帧、重封装的同一图和同一采集组会关联隔离。像素标定使用 `pixel_size` 和 `pixel_size_unit`。父视野字段可在标签 JSON 的 metadata 中填写。

采集组采用 `(sample_id, acquisition_id)` 命名空间，需在全部任务中保持一致。若跨样品/批次仍有共同来源，应通过父原图 SHA-256 记录关联，或在正式审核前统一更严格的分组约定。

审计示例：

```powershell
$projects = @(Get-ChildItem data/annotation_projects/migrated_20260906 -Filter annotation_project.json -Recurse | Select-Object -ExpandProperty FullName)
.\scripts\atom-center.ps1 audit @projects --output reports/my_audit.json
```

若有重复标注冲突，查看报告每个 record 的原任务、点位及 record_key。负责人选定正式版本后，创建 selections.json，内容为“plane_sha256 → record_key”的 JSON 对象，再加 `--selections selections.json` 审计/组装。工具不会替负责人挑选不同人的标签。

确认旧标签确实对应当前图像后，才添加 `--confirm-current-sources`。它记录这项人工确认，原始身份限制仍保留在审计警告中，不会被改成历史已验证。

单个旧项目迁移命令：

```powershell
.\scripts\atom-center.ps1 migrate "<旧项目>/annotation_project.json" --output "<新的迁移目录>"
```

## 3. 冻结训练数据

2026-09-09 新增：若当前只需验证训练流程，旧标注尚缺采集编号/历史原图哈希，可显式使用 `build --workflow-only`。它仍阻止原图变化、标签错误、重复版本冲突和跨集合来源关联泄漏，但将来源资料缺失保留在 manifest 的 `deferred_provenance_errors` 中。数据用途为 `workflow_validation`，只生成按来源组隔离的 train/val，不生成 test，也不代表正式审计通过。至少需要两个可分离的来源组；同一个任务不能被逐图随机拆成假独立组。

```powershell
.\scripts\atom-center.ps1 build @projects --output data/processed/haadf_v001 --tile-size 640 --box-size 8 --seed 20260831 --selections selections.json --confirm-current-sources
```

按实际需要使用末尾两个确认/版本参数。审计失败会返回非零退出码并说明原因。

生成规则：

- 仅使用已审核、完整覆盖且不重叠的 ROI，确认无原子的背景 ROI 保留；未审核区域不作负样本。
- 在源图及关联来源层先划分 train/val/test，随后裁剪。正式 test 至少 2 个独立来源组，train/val 各至少 1 个；不够时拒绝划分。
- 每个原始切片独立做 1%～99% 强度归一化、8 位量化，再按固定尺寸 letterbox。训练 PNG 和推理共用同一预处理函数。
- 真值保存在完整原始强度 TIFF 和浮点点位 JSON 中。YOLO 标签是派生物，使用 +0.5 像素边界坐标约定；边缘框对称缩小，中心不移动。
- 输出含 data.yaml、dataset_manifest.json、raw、truth、images、labels、source_map.csv 和审计报告。受保护数据内容有摘要，来源和分组不依赖输出目录绝对位置。
- 同一输入项目、参数和种子重复生成会得到相同数据指纹。输出目录已存在时拒绝覆盖；内容变化需生成新版本。

分组只能使用已知图像身份和真实元数据，无法从不同像素内容自动判断所有未记录的相关视野。父视野和采集信息应在正式组装前核实。

`reviewed` 状态和格式检查不能自动证明 ROI 内没有漏标。正式长训练前需叠加原图复核：完整覆盖矩形内的目标必须全部标注；若只标注特定原子柱类型，各任务应使用同一目标定义。未完成区域不能当成负样本，应补标或从完整覆盖 ROI 中排除；嵌入标尺、文字也应避开。滤波后的提交图像会原样作为强度输入保存，工具不会恢复仪器原始数据；应记录处理步骤及父原图关联。

## 4. 配置与训练

**点位选择开发候选（2026-09-09）：** 已实现并实测 `best_points.pt` 选择，第 45 轮候选在原有验证集 F1=95.56%，实际运行清单使用 COM7 和最大位移 2.4 像素（原文 COM11 已更正）。模型清单为 `runs/point-selection-20260909/onnx-points/model_manifest.json`，配置为 `configs/experiments/haadf_point_selected_150.yaml`，报告见 `reports/POINT_SELECTION_2026-09-09.md`。不替代原来的科研验收条件。

**最新开发候选（2026-09-10）：** 已完成四组共 800 轮训练对照，采用间距训练框、关闭增强训练的第 135 轮权重及自适应亮斑精修。清单为 `runs/generalization-20260910/adaptive_blob-onnx/model_manifest.json`；训练集 P/R=92.95%/93.87%，验证集 P/R=95.73%/97.22%，匹配距离仍为原图 2 像素。困难图像及无人工真值的两个测试集尚未通过验收，仅用于预标注。详细指标、图像预览和独立 draft 复核项目见 `reports/GENERALIZATION_2026-09-10.md`。该清单需要当前源码中的 `adaptive_blob` 和精修后去重支持，旧版 0.3.0 安装包不能直接解释新增配置；PPA 默认候选未更换。

**可运行开发基线（2026-09-09 后续）：** 两组 150 轮完整训练侧对照已完成，当前保留 8 像素框、IoU=0.45、COM；原有验证集点位 F1=93.36%。可复现配置为 `configs/experiments/haadf_workflow_baseline_150.yaml`，新模型清单位于 `runs/box-comparison-20260909/box8/onnx/model_manifest.json`。详见 `reports/BOX_COMPARISON_2026-09-09.md`。这是开发基线，不是科研验收；测试集未用于本次对照。

**2026-09-09 排查发现：** 固定 8 原图像素框配合随机初始化，在部分真实样本上出现有效正样本权重过低；扩大框又会影响 NMS 对相邻原子的保留。默认值尚不能视为已验证正式方案。长训练前先核查网络输入上的目标尺度、正样本分配和无增强小样本过拟合，详见 `reports/LEARNING_DIAGNOSIS_2026-09-09.md`。3 轮/18 次优化更新仅证明流程执行，不证明模型可用。

配置优先级：公共 YAML → 模态 YAML → `--set key=value`。未知键在加载权重前报错。亮度/对比度属于独立 augmentation，不能写成 training.brightness/contrast。

```powershell
.\scripts\atom-center.ps1 check --data data/processed/haadf_v001 --common configs/common.yaml --modality configs/haadf_stem.yaml

.\scripts\atom-center.ps1 train --data data/processed/haadf_v001 --run runs/haadf_baseline_001 --common configs/common.yaml --modality configs/haadf_stem.yaml
```

默认 YOLOv8s 随机初始化、640 输入、batch=8、200 epochs、FP32、单 GPU 0。这是候选实验配置，batch 和 epoch 数应根据真实目标密度及验证集结果调整。也可设置 `--set training.device=cpu`。

使用已确认来源的本地预训练权重时：

```powershell
.\scripts\atom-center.ps1 train --data data/processed/haadf_v001 --run runs/haadf_pretrained_001 --common configs/common.yaml --modality configs/haadf_stem.yaml --set "training.model=C:/models/yolov8s.pt" --init-sha256 "<已核对的64位SHA-256>"
```

不自动下载初始化权重。运行目录保存数据指纹、最终配置、源码摘要、环境版本、初始化权重/实际模型张量摘要、best.pt/last.pt、可续训 resume.pt 及训练日志。每次实验使用独立目录。训练只使用 train 与 val；test 不进入模型选择。

max_det 默认 3000，并要求至少高于训练切片最大原子数的 1.25 倍加 32。验证、推理共用这个上限；触及上限会发出诊断。实际密度高于训练覆盖范围时仍需调整适用范围和配置。

若已知合理最小间距，可设置 `inference.min_atom_distance_px`，同时保证 merge_distance_px 小于其一半。精修在完整原始图上进行，再按 ROI 过滤；失败保留候选浮点坐标并输出质量字段。

## 5. 续训

### 原图点位检查点选择

新增可选 `point_validation` 配置，不作为 Ultralytics 参数传入：

```yaml
point_validation:
  enabled: true
  interval_epochs: 5
  match_distance_px: 2.0
```

开启后在第 1 轮、每隔指定轮数、最终/提前停止轮次以及主动暂停轮次上，加载刚保存的检查点，在冻结数据的 **val 原图完整 ROI** 上运行与部署相同的分块、坐标恢复和精修流程。所有指标使用原图像素，置信度和精修参数固定为该次运行配置；不会读取 test。它评测的是实际保存的 FP16 EMA 检查点重新加载为 FP32 后的输出，避免内存模型和发布权重不同。

选择顺序为 F1 最大、并列时召回率最大、再并列时匹配点 RMSE 最小；没有匹配点时不把 RMSE 当作零。选择只覆盖实际评测过的轮次。默认每 5 轮会漏过中间轮次的潜在最优点，可设为 1 增加评测频率和运行成本。

原 `best.pt` 继续按框 mAP 保存。`best_points.pt` 是登记在 state.json 中的**逻辑检查点名称**，实际指向 `point_validation/candidate_<epoch>_<sha>.pt`；每次提升保留独立候选，选择记录原子更新，避免中断时覆盖上一份最佳权重。逐轮评测、指标、文件 SHA-256 与历史保存在 `point_validation/selection.json` 及相邻 JSON 中。续训会核对选择合同和已选权重摘要，并保留此前最佳值。

示例配置：`configs/experiments/haadf_point_selected_150.yaml`。它使用 COM 11 像素窗口并将最大位移固定为 2.4 原图像素，是当前验证集上的开发候选，不能当作独立科研验收配置。

```powershell
.\scripts\atom-center.ps1 train --data data/processed/real_workflow_20260909_v2 --run runs/<新的运行目录> --common configs/experiments/haadf_point_selected_150.yaml --modality configs/haadf_stem.yaml
.\scripts\atom-center.ps1 export --run runs/<新的运行目录> --checkpoint best_points.pt --output runs/<新的运行目录>/onnx-points
```

点位选择不改变优化器、学习率或原有按框指标的提前停止机制；需自行留足训练轮数和 patience。若启用后数据中没有可用 val 区域，评测会报错，不会自动使用 train/test 代替。

```powershell
.\scripts\atom-center.ps1 resume --run runs/haadf_baseline_001
```

续训使用原运行记录，拒绝更换数据、配置、源码或记录的依赖版本。冻结目录中混入额外图像、标签或 .npy 图像缓存会被拒绝；标签缓存每次从已校验的标签文件重建。可用 `--data <新位置>` 指向迁移后内容相同的数据目录。完成的运行不能作为续训追加 epoch，应另开实验。

采用 Ultralytics 的 epoch 检查点语义，恢复 optimizer/EMA 和剩余 epoch；不承诺中断与连续运行逐比特相同。意外退出时只从最后完整检查点恢复。若进程被强制杀死留下 .running.lock，先确认对应进程已退出，再删除该运行目录内的锁文件。

开发验证可设置 `--stop-after-epochs 1`，要求总 epochs 大于 1。这个选项在完整保存检查点后暂停。

## 6. 导出、预测与评测

```powershell
.\scripts\atom-center.ps1 export --run runs/haadf_baseline_001 --output runs/haadf_baseline_001/onnx

.\scripts\atom-center.ps1 predict --manifest runs/haadf_baseline_001/onnx/model_manifest.json --image "<原始图.tif>" --output reports/prediction.json

.\scripts\atom-center.ps1 evaluate --manifest runs/haadf_baseline_001/onnx/model_manifest.json --data data/processed/haadf_v001 --split val --match-distance 2 --output reports/validation_points.json
```

多帧 TIFF 必须显式传 `--frame`，必要时传 `--series`；ROI 使用 `--roi x0 y0 x1 y1`，整数半开边界。

导出固定 NCHW、1×3×H×W、FP32、opset 17，无内置 NMS。两种后端共享解码/坐标恢复/NMS，置信度按六位小数处理，使 FP32 近似相同的分数排序稳定。阈值附近输出有独立诊断。输出为全图 (x,y)，左上像素中心 (0,0)，y 向下。

每次导出运行原生 ONNX 校验与六类实际网络 Torch/CPU ONNX 对照：空白、密集、弱信号、非方形、重叠切片和 ROI。比较失败不会发布目标目录。

预测 JSON 分开保存点位、模型置信度、逐点精修质量、图像/模型身份；不能把置信度塞入 PPA 的 point_size。评测使用冻结的原始点真值和完整强度图，包括背景误检；零匹配时 RMSE/p95/bias 为 null。

导出包标为 development、scientific_acceptance=false，保存在指定实验目录。它可供后续 M4 开发联调，尚不是 M5 科研正式发布包。验证集阈值和精修参数应在正式盲测前冻结。

## 7. 纯 CPU 推理安装

已使用“全新 venv + 锁文件 + wheel”实际安装、加载模型并预测。该环境没有 torch、ultralytics 或 onnx 导出库。

```powershell
python -m venv .venv-infer
.\.venv-infer\Scripts\python.exe -m pip install -r requirements/inference-win-py310.lock
.\.venv-infer\Scripts\python.exe -m pip install --no-deps "<atom_center-0.3.0-py3-none-any.whl>"
.\.venv-infer\Scripts\python.exe -m atom_center.cli predict --manifest "<model_manifest.json>" --image "<原始图.tif>" --output prediction.json
```

本次构建 wheel 位于 `runs/m123-validation/wheels/`。包仍含标注模块，但导入 backends/pipeline 不会导入训练框架或 GUI。部署时由后续 PPA 适配层管理已登记模型的信任边界；自填模型 manifest 并不构成正式放行。

## 8. 已完成的合成验收

`runs/m123-validation/experiment-acceptance`：8 个模拟来源组，6/1/1 划分，包含背景；YOLOv8n、128 输入、batch=2，启用 2 个数据加载进程；训练首轮后暂停，保留原运行并复制实验和数据目录，在副本中恢复到 3 轮。

`runs/m123-validation/onnx-acceptance`：真实网络 ONNX 及六场景一致性报告。短训练开发模型的定位能力尚未形成；不能将它用于科研中心点或应变结论。

重新生成模拟项目可使用 `scripts/make_synthetic_development.py`。组装模拟数据时必须加 `--development`，输出不得当作真实实验数据。纯几何精修的已知候选测试仍由 `scripts/smoke_test.py` 单独维护，二者证据不同。

旧 atom_detector 训练入口现在给出迁移命令并返回非零，旧评测目录/背景计数/点匹配已修复。PPA 主界面的后台检测、逐点元数据持久化和取消/状态切换处理留给 M4。
