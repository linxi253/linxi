# TEM 原位视频流水线（DeepSeek Harness 集成）

把原位 TEM 视频自动转换为"TIFF 提取 + 漂移矫正"产物。
只读复用你的两个既有工具（**未修改原代码**）：

- 视频切片工具：`<仓库根>\01-视频与数据提取\视频切片工具\video_extractor`
- 漂移矫正 v7：`<仓库根>\02-图像处理\drift-correction-v7\drift_core.py`

环境要求：Python 3.10-3.12（本机 Miniconda base：Python 3.10.9）+ numpy / opencv / tifffile / matplotlib，均已满足；ffmpeg 由切片工具自带（`tools\ffmpeg`），无需额外安装。

## 用法一：DSH 工具（推荐）

在 DeepSeek Harness 会话中，向助手提供视频路径与输出根目录即可，例如：

```
处理这个视频：D:\data\dshtest\input\xxx.wmv，输出到 D:\data\dshtest\output
```

助手的 `tem_video_pipeline` 工具会自动：

1. TIFF 帧提取（默认 5 fps；可指定全部帧/时间间隔）
2. 可选的区域裁剪（界面录制视频只取 STEM 图像：`crop_rect="0,0,0.3333,1"` 左侧三分之一）
3. 首尾空白/无纹理帧剔除（默认开启）
4. 漂移检测：**自动两档** —— 先按严格参数（有效帧对 ≥80%）；失败自动放宽（ratio 0.8 / min_matches 3 / min_inliers 3 / 有效比 0.7 / 插值 15）并在报告中标注
5. 漂移矫正（默认裁剪至共同有效区）+ 全套产物

**当前为会话内动态插件**（temvid-1）；重启后需重新定义，永久化版本见下方"永久化"。

## 用法二：命令行直接运行（无 DSH）

```powershell
python <仓库根>\10-DSH集成\tem_pipeline.py `
  --video D:\data\dshtest\input\xxx.wmv `
  --output-root D:\data\dshtest\output `
  --sampling target_fps --sampling-value 5 `
  --crop-rect 0,0,0.3333,1
```

stdout 每行一个 JSON：`progress` / `info` / `summary` / `error`（末行为 summary 或 error）。

## 输出目录结构

```
<output-root>/<视频名>/            （目录已存在且有产物时自动追加 _yyyyMMdd-HHmmss）
  01_raw_<视频名>_stack.ome.tif      原始提取堆栈（OME-TIFF，TYX）
  01_raw_<视频名>_frames.csv         逐帧清单（含时间轴）
  02_corrected_<视频名>.tif          漂移矫正结果 TIFF
  02_drift_shifts.csv                每帧位移 dx/dy/模长
  02_drift_curve.png                 漂移曲线图（人工复核用）
  02_audit.json                      完整审计（参数/逐帧对质量/插值/裁剪/剔除信息）
  03_summary.json                    汇总（与工具返回一致）
  03_preview_montage.png             矫正后 6 帧缩略拼图
```

## 关键参数

| 参数 | 作用 | 默认 |
|---|---|---|
| `sampling` / `sampling_value` | 采样方式：all / target_fps / interval | target_fps / 5 |
| `crop_rect` | 区域裁剪 `x0,y0,x1,y1`（0~1 分数） | 无（界面录制视频通常需要 `0,0,0.3333,1`） |
| `skip_interval` | 漂移检测跳帧间隔 | 1 |
| `crop_mode` | 矫正输出：crop（共同有效区）/ keep | crop |
| `sift_min_valid_pair_ratio` 等 | SIFT 质量门控参数（自动两档已内置） | 严格--放宽自动切换 |
| `max_stack_gib` | 堆栈解码内存上限 | 24 GiB |
| `trim_blank_edges` / `blank_std_threshold` | 首尾空白/无纹理帧剔除 | 开 / 2.0（+SIFT 判定） |

## 已知边界（与工具 v7 一致）

- 高倍率周期晶格 + 快速漂移视频（如 1200k HR mapping）：SIFT 匹配歧义，默认参数会拒绝；
  自动放宽档可处理（本案 0341：88% 可靠帧对），但**务必人工复核漂移曲线**。
- 界面录制视频必须 `crop_rect` 截取 STEM 图像区域，否则面板内容变化会污染配准。
- 大漂移（数十像素）时"共同有效区裁剪"会明显缩小图像，属预期行为。

## 永久化（P2）

动态插件重启即失。如需每次会话自带该工具，按 DSH 惯例：
新建 workspace 包 `packages/tem/tool-tem-pipeline`（仿 `packages/todo/tool-todo` 结构），
并复制 `standard`/`cordis` preset 为 `tem-analysis` 用户 preset，在其中添加该工具行 + 系统提示。
