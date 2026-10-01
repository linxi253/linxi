# PPA 深度学习原子检测模块

这是 PPA 的可选 YOLOv8 原子检测子项目，覆盖半自动标注、标签审核、数据集划分与增强、训练、推理评测以及 PPA 集成。

当前目录不包含原始图像、标注数据或训练好的 `best.pt`，因此不能开箱完成推理；需先准备数据或提供模型权重。

## 安装

建议在独立虚拟环境中安装：

```powershell
python -m pip install -r requirements.txt
```

默认训练配置针对 16 GB GPU 编写，也可在命令行把 `--device` 设为 `cpu`，但训练会很慢。

## 典型流程

在 `atom_detector` 目录执行：

```powershell
# 1. 传统算法生成预标注 JSON
python .\annotate\semi_auto_label.py --input .\raw_images --output .\raw_labels

# 2. 人工审核并导出 YOLO 标签
python .\annotate\label_review.py --labels .\raw_labels --images .\raw_images --export .\yolo_labels

# 3. 划分并缩放数据集
python .\dataset\prepare_dataset.py --images .\raw_images --labels .\yolo_labels --output .\datasets\atom_v1

# 4. 可选数据增强
python .\dataset\augment.py --input .\datasets\atom_v1\images\train --labels .\datasets\atom_v1\labels\train --factor 3

# 5. 训练
python .\train\train.py --data .\datasets\atom_v1\data.yaml

# 6. 评测
python .\infer\benchmark.py --model .\models\best.pt --test_data .\datasets\atom_v1 --compare_traditional
```

实际训练输出路径取决于 Ultralytics 配置；需要把选定权重复制或配置为 `models/best.pt` 才能使用默认推理路径。

## 模型安全（必须阅读）

PyTorch 的 `.pt` 权重在反序列化时可能执行任意代码，因此本模块只允许加载
`models/allowlist.json` 白名单中记录了 SHA-256 的模型，其他模型一律拒绝。
白名单默认是空的：

```powershell
# 1. 计算哈希并查看输出
python -m atom_detector.model_security .\models\best.pt

# 2. 确认模型来源与哈希后，将其加入白名单
python -m atom_detector.model_security .\models\best.pt --allow
```

不要加载来源不明、哈希未核对的 `.pt` 文件；即使加入白名单，也应当只
信任自己训练或从可信渠道分发的权重。`integration/ppa_plugin.py` 的
“DL 检测”对话框与 `infer/detector.py` 都会在加载前执行该校验。

## 目录说明

- `config.yaml`：数据路径、标注、训练和推理默认参数。
- `models/allowlist.json`：允许加载的模型 SHA-256 列表（默认空）。
- `model_security.py`：模型哈希计算、白名单校验与命令行工具。
- `annotate/semi_auto_label.py`：基于传统峰值检测生成预标注。
- `annotate/label_review.py`：图形化审核、增删标注并导出 YOLO 格式。
- `dataset/prepare_dataset.py`：训练/验证/测试划分和 `data.yaml` 生成。
- `dataset/augment.py`：适合原子图的小目标数据增强。
- `train/train.py`、`train/hyp.yaml`：YOLOv8 训练入口和超参数。
- `infer/detector.py`：模型推理和亚像素质心修正接口。
- `infer/benchmark.py`：检测指标及与传统方法对比。
- `integration/ppa_plugin.py`：从 PPA 调用模型的适配层。
- `requirements.txt`：PyTorch、Ultralytics、OpenCV 等依赖。

## 数据注意事项

- 训练、验证和测试必须按原始图像或实验批次划分，避免同一视野增强副本泄漏到不同集合。
- YOLO 检测框中心不是天然的亚像素原子坐标，定量 PPA 前仍需质心或二维高斯精修。
- 模型对显微镜模式、像素尺寸、原子衬度极性和噪声分布敏感，跨数据集使用前必须重新评测。
