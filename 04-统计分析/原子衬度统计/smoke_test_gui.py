# -*- coding: utf-8 -*-
"""GUI 冒烟测试：驱动 TIFContrastAnalyzer 验证 v2.5.0 关键修复项。

不进入 mainloop，用 root.update() 泵事件；文件/消息对话框全部打桩。

覆盖：
- 中文字体探测与图内缺字形告警
- 工具栏布局（进度条可见、复选框文字不截断）
- 加载（后台线程）/换文件清理/加载失败保持原状态
- planar 彩色单页的两种解析（默认转灰度 vs 多帧解析），切换时 ROI 保留
- 分析（分块、进度、逐帧统计、1 基 x 轴）、取消、负值域/非有限/小 ROI 警告
- CSV 导出（元数据头 / 逐帧统计列 / 跨帧统计尾部 / 时区时间戳 /
  逐帧 DateTime 列 / pandas 可读）
- ROI JSON 导入导出、显示范围锁定、均匀帧 clim、拖拽出界释放仍提交
"""
import importlib.util
import json
import os
import re
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
import tifffile
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as _mb
import tkinter.filedialog as _fd

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


TMP = tempfile.mkdtemp(prefix="contrast_smoke_")
STACK_TIF = os.path.join(TMP, "stack.tif")
SINGLE_TIF = os.path.join(TMP, "single.tif")
PLANAR4_TIF = os.path.join(TMP, "planar4.tif")
UNIFORM_TIF = os.path.join(TMP, "uniform.tif")
NEGATIVE_TIF = os.path.join(TMP, "negative.tif")
NAN_SINGLE_TIF = os.path.join(TMP, "nan_single.tif")
DT_TIF = os.path.join(TMP, "dt.tif")
BIG_TIF = os.path.join(TMP, "big.tif")
BAD_FILE = os.path.join(TMP, "not_a_tiff.dat")
OUT_CSV = os.path.join(TMP, "out.csv")
ROI_JSON = os.path.join(TMP, "rois.json")

# --- 生成测试数据 ---
rng = np.random.default_rng(42)
stack = (rng.integers(100, 2000, size=(8, 64, 80))).astype(np.uint16)
stack[:, 20:30, 30:40] += 8000      # 高亮区
stack[3, 15, 15] = 65535            # 热像素（位于 ROI1 内）
tifffile.imwrite(STACK_TIF, stack)
tifffile.imwrite(SINGLE_TIF, stack[0:1])
# (4,H,W) 会被 tifffile 写成 planar RGBA（其自身也会发 DeprecationWarning）
tifffile.imwrite(PLANAR4_TIF, rng.integers(100, 2000, (4, 64, 80)).astype(np.uint16))
tifffile.imwrite(UNIFORM_TIF, np.full((3, 32, 32), 500, dtype=np.uint16))
neg = rng.normal(50.0, 5.0, size=(6, 32, 32)).astype(np.float32)
neg[:, 0:8, 0:8] -= 200.0           # 明显的负值区
tifffile.imwrite(NEGATIVE_TIF, neg, photometric="minisblack")
# 单帧含 NaN（验证单帧路径的非有限警告）
nan_single = np.full((1, 32, 32), 100.0, dtype=np.float32)
nan_single[0, 8:16, 8:16] = np.nan
tifffile.imwrite(NAN_SINGLE_TIF, nan_single, photometric="minisblack")
# 单帧带 DateTime 标签（验证 CSV 时间列全链路；tifffile 的 contiguous
# 多页会复制首页标签，公开 API 无法写出每页不同的时间戳，逐页读取
# 逻辑由 tests/test_core.py::TestPageDatetimes 用多 series 文件覆盖）
_DT_STAMP = "2026:09:28 10:00:00"
tifffile.imwrite(DT_TIF, np.full((6, 6), 10, dtype=np.uint16),
                 photometric="minisblack",
                 extratags=[(306, 2, 19, _DT_STAMP, False)])
# 大堆叠（用于取消路径：需多分块才能稳定命中取消）
tifffile.imwrite(BIG_TIF, rng.integers(0, 4096, (300, 256, 256)).astype(np.uint16),
                 photometric="minisblack")
with open(BAD_FILE, "wb") as f:
    f.write(b"\x89PNG\r\n\x1a\n" + b"0" * 512)

# --- 导入 GUI 模块 ---
spec = importlib.util.spec_from_file_location(
    "gui", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "tif图像衬度分析工具.py"))
gui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gui)

# --- 打桩对话框 ---
shown_msgs = []
dialog_path = {"open": STACK_TIF, "save": OUT_CSV}
_fd.askopenfilename = lambda **kw: (ROI_JSON if "ROI" in str(kw.get("title", ""))
                                    else dialog_path["open"])
_fd.asksaveasfilename = lambda **kw: (ROI_JSON if "ROI" in str(kw.get("title", ""))
                                      else dialog_path["save"])
_mb.showinfo = lambda *a, **kw: shown_msgs.append(("info",) + a)
_mb.showerror = lambda *a, **kw: shown_msgs.append(("error",) + a)
_mb.showwarning = lambda *a, **kw: shown_msgs.append(("warn",) + a)
_mb.askyesno = lambda *a, **kw: (shown_msgs.append(("ask",) + a), True)[1]

checks = []


def check(name, cond, detail=""):
    checks.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'}  {name} {detail}")


import ttkbootstrap as ttkb
root = ttkb.Window(themename="darkly")
app = gui.TIFContrastAnalyzer(root)
root.deiconify()
root.update()


def pump(timeout=60.0):
    t0 = time.time()
    while app._busy() and time.time() - t0 < timeout:
        root.update()
        time.sleep(0.005)
    root.update()


def load(path):
    app._start_load(path)
    pump()
    root.update()


# 1. 中文字体与工具栏布局
check("字体: 已选中可用中文字体", bool(app.font_name), app.font_name)
check("布局: 进度条可见", app.progress.winfo_ismapped() and app.progress.winfo_width() > 50,
      f"w={app.progress.winfo_width()}")
check("布局: 多帧解析复选框文字未截断",
      app.chk_force_frames.winfo_width() >= app.chk_force_frames.winfo_reqwidth(),
      f"{app.chk_force_frames.winfo_width()}/{app.chk_force_frames.winfo_reqwidth()}")

# 2. 加载多帧堆叠
load(STACK_TIF)
check("加载: 形状 (8,64,80)", app.tif_memmap.shape == (8, 64, 80), str(app.tif_memmap.shape))
check("加载: 持久图像对象已创建", app._im_img is not None)
check("加载: 全堆栈显示范围已估计",
      app._stack_clim is not None and app._stack_clim[0] < app._stack_clim[1],
      str(app._stack_clim))
check("加载: 锁定 clim 生效", app._im_img.get_clim() == app._stack_clim)
check("加载: 头部探测结果已记录", app._probe is not None and app._probe.n_pages == 8)
check("加载: 结果区为占位提示", "等待分析" in app.ax_plot.get_title(), app.ax_plot.get_title())

# 3. 添加 2 个 ROI（含一次重复 _show_frame 验证 artist 不累积）
app.roi_list = [(10, 10, 30, 30), (40, 20, 60, 50)]
app._update_roi_listbox()
app._show_frame()
app._show_frame()
root.update()
n_rect = len(app.ax_img.patches)
check("ROI: artist 不累积(2 个矩形)", n_rect == 2, f"patches={n_rect}")

# 3b. 拖拽出画布释放：xdata=None 时用矩形几何提交（不再丢弃拖拽成果）
class _FakeEvent:
    xdata = None
    ydata = None

app.dragging_roi = True
app.roi_start = (5.2, 6.7)
app.temp_rect = gui.Rectangle((5.2, 6.7), 20.5, 10.3, linewidth=1.5,
                              edgecolor='#ff4444', facecolor='none', linestyle='--')
app.ax_img.add_patch(app.temp_rect)
app._on_mouse_release(_FakeEvent())
root.update()
added = app.roi_list[-1]
check("拖拽出界: 仍按矩形几何提交", added == (5, 6, 26, 17), str(added))
app.roi_list = [(10, 10, 30, 30), (40, 20, 60, 50)]
app._update_roi_listbox()
app._show_frame()
root.update()

# 4. 切帧保留缩放视图
app.ax_img.set_xlim(5.0, 30.0)
app.ax_img.set_ylim(60.0, 10.0)
app._show_frame()
root.update()
xlim = app.ax_img.get_xlim()
check("切帧: 缩放视图保留", abs(xlim[0] - 5.0) < 1e-9 and abs(xlim[1] - 30.0) < 1e-9, str(xlim))

# 5. 方向键焦点守卫（mock focus_get：本环境无法可靠移动 Tk 真实焦点）
app.current_frame = 0
_real_focus_get = root.focus_get
root.focus_get = lambda: app.jump_input  # 模拟 Entry 聚焦
app._step_frame(1)
time.sleep(0.15)
root.update()
check("方向键: Entry 聚焦时不切帧", app.current_frame == 0, f"frame={app.current_frame}")
root.focus_get = lambda: app.roi_listbox  # 模拟 ROI 列表聚焦
app._step_frame(1)
time.sleep(0.15)
root.update()
check("方向键: ROI 列表聚焦时不切帧", app.current_frame == 0, f"frame={app.current_frame}")
root.focus_get = lambda: None  # 模拟画布/其他控件聚焦
app._step_frame(1)
time.sleep(0.15)  # 等待滑动条 80ms 防抖
root.update()
check("方向键: 非输入控件聚焦时正常切帧", app.current_frame == 1, f"frame={app.current_frame}")
root.focus_get = _real_focus_get

# 6. 帧跳转 GO：滑动条与画面同步（回归 Scale.set 不触发 command 的问题）
app.jump_input.delete(0, tk.END)
app.jump_input.insert(0, "5")
app._jump_to_frame()
time.sleep(0.15)
root.update()
check("帧跳转: GO 后渲染第5帧", app.current_frame == 4 and app.frame_slider.get() == 5,
      f"frame={app.current_frame}, slider={app.frame_slider.get()}")

# 7. 四种算法逐个跑多帧分析（第 2 个起顺带验证分析期间导出/编辑被拒绝）
for i, (name, key) in enumerate(gui.ALGORITHMS):
    app.contrast_method.current(i)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        app._analyze_contrast()
        if i == 1:
            n_before = len(shown_msgs)
            app._export_data()
            app._clear_roi()
            root.update()
            export_msg = " ".join(str(m) for m in shown_msgs[n_before:])
            check("分析中: 导出被拒绝", "分析" in export_msg, export_msg[:60])
            check("分析中: 清空ROI被拒绝",
                  "暂不可" in app.status_var.get() and len(app.roi_list) == 2,
                  f"status={app.status_var.get()!r} rois={len(app.roi_list)}")
        pump()
    glyph = [str(x.message) for x in caught if "missing from font" in str(x.message)]
    if i == 0:
        check("绘图: 无缺字形告警", not glyph, f"{len(glyph)} 条")
    cd = app.contrast_data
    ok = (cd is not None and len(cd) == 2 and all(len(r) == 8 for r in cd)
          and not app._analysis_running)
    finite = all(np.isfinite(v) for r in cd for v in r) if ok else False
    check(f"分析[{key}]: 2ROI×8帧", ok)
    check(f"分析[{key}]: 数值有限", finite)

check("分析: 逐帧统计已保存",
      app.contrast_stats is not None
      and all(len(app.contrast_stats[k]) == 2 for k in ("mean", "sigma", "min", "max")),
      str({k: len(v) for k, v in (app.contrast_stats or {}).items()}))
check("分析: 曲线 x 轴为 1 基",
      list(app.ax_plot.lines[0].get_xdata()[:3]) == [1.0, 2.0, 3.0],
      str(app.ax_plot.lines[0].get_xdata()[:3]))
check("分析: 进度条走满", app.progress['value'] == app.progress['maximum'])
check("分析: 完成后按钮状态复位",
      str(app.btn_analyze['state']) != 'disabled' and str(app.btn_cancel['state']) == 'disabled'
      and str(app.btn_export['state']) != 'disabled')

# 热像素影响：极值 Michelson 高、稳健版低（热像素位于 ROI1 内）
app.contrast_method.current(1)  # michelson
app._analyze_contrast(); pump()
mich = np.array(app.contrast_data)
app.contrast_method.current(3)  # robust
app._analyze_contrast(); pump()
rob = np.array(app.contrast_data)
check("离群抑制: 极值Michelson≈1", mich[0].max() > 0.99, f"ROI1={mich[0].max():.3f}")
check("离群抑制: 稳健版显著更低", rob[0].max() < mich[0].max() - 0.05,
      f"ROI1 rob={rob[0].max():.3f}")

# 8. CSV 导出（元数据头 / 逐帧统计列 / 跨帧统计尾部）
app.contrast_method.current(0)
app._analyze_contrast(); pump()
app._export_data()
root.update()
raw = open(OUT_CSV, "rb").read().decode("utf-8-sig")
check("CSV: 头部注释块", raw.startswith("# 软件: TIF图像衬度分析工具 v"), raw.splitlines()[0])
check("CSV: 算法已记录", "标准差/平均强度 (std_over_mean)" in raw)
check("CSV: 导出时间已记录", "# 导出时间:" in raw)
_ts_line = [ln for ln in raw.splitlines() if ln.startswith("# 导出时间:")][0]
check("CSV: 导出时间带时区", re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [+-]\d{4}", _ts_line)
      is not None, _ts_line)
check("CSV: σ 口径注明(ddof=0)", "ddof=0" in raw)
check("CSV: TIFF 布局已记录", "# TIFF 布局: axes=" in raw)
check("CSV: ROI 坐标已记录", "ROI_1: (10,10)-(30,30) [20x20px]" in raw)
check("CSV: 取整规则已说明", "ROI 取整规则" in raw and "开区间端点" in raw)
check("CSV: 帧号说明", "帧号从 1 开始" in raw)
check("CSV: 尾部跨帧统计", "# ROI_1: 有效帧 8/8" in raw, raw.strip().splitlines()[-1][:60])
df = pd.read_csv(OUT_CSV, comment="#", index_col="Frame")
check("CSV: pandas 可读且帧号 1 基",
      list(df.index) == list(range(1, 9)) and list(df.columns)[:2] == ["ROI_1", "ROI_2"],
      str(list(df.columns)))
check("CSV: 含每帧均值/σ/极值列",
      {"ROI_1_mean", "ROI_2_mean", "ROI_1_sigma", "ROI_2_sigma",
       "ROI_1_min", "ROI_1_max"}.issubset(set(df.columns)))
check("CSV: 均值列与图像一致", abs(df["ROI_1_mean"].iloc[0]
                                - stack[0, 10:30, 10:30].mean()) < 1e-6)

# 8b. 分析完成后切换算法：结果不失效但给出明确提示
app.contrast_method.current(2)
app._on_method_changed()
check("算法切换: 提示结果属旧算法",
      "重新分析" in app.status_var.get()
      and app.contrast_data is not None, app.status_var.get())
app.contrast_method.current(0)
app._on_method_changed()

# 9. ROI JSON 往返
app._export_rois()
root.update()
check("ROI: JSON 已导出", os.path.exists(ROI_JSON)
      and json.load(open(ROI_JSON, encoding="utf-8"))["rois"] == [[10, 10, 30, 30], [40, 20, 60, 50]])
app.roi_list = []
app._update_roi_listbox()
app._import_rois()
root.update()
check("ROI: JSON 导入还原", app.roi_list == [(10, 10, 30, 30), (40, 20, 60, 50)],
      str(app.roi_list))

# 10. 加载失败：原状态保持不变
before = (len(app.roi_list), app.contrast_data is not None, app.total_frames,
          app.tif_memmap.shape)
dialog_path["open"] = BAD_FILE
app._load_tif()
pump()
after = (len(app.roi_list), app.contrast_data is not None, app.total_frames,
         app.tif_memmap.shape)
check("失败加载: 报『不是有效的 TIFF』",
      shown_msgs[-1][0] == "error" and "不是有效的 TIFF" in str(shown_msgs[-1][2]),
      str(shown_msgs[-1][1:]))
check("失败加载: 状态与结果未被破坏", before == after, f"{before} -> {after}")
check("失败加载: 可继续操作", not app._loading and str(app.btn_load['state']) != 'disabled')

# 11. 换文件：ROI 与结果区清空
load(SINGLE_TIF)
check("换文件: ROI 与结果已清空", app.roi_list == [] and app.contrast_data is None)
check("换文件: 结果区回到占位提示",
      "等待分析" in app.ax_plot.get_title() and not app.ax_plot.lines)
check("换文件: 单帧模式", app.total_frames == 1)

# 12. planar 彩色单页：默认 1 帧 / 多帧解析 4 帧（切换时 ROI 保留）
load(PLANAR4_TIF)
default_frames = app.total_frames
app.roi_list = [(4, 4, 20, 20)]
app._update_roi_listbox()
app._show_frame()
app._force_frames_var.set(True)
app._on_toggle_force_frames()
pump()
check("多帧解析: planar RGBA → 4 帧", app.total_frames == 4,
      f"默认={default_frames} 多帧解析={app.total_frames}")
check("多帧解析: ROI 保留（仅换解析方式不丢 ROI）",
      app.roi_list == [(4, 4, 20, 20)], str(app.roi_list))
check("多帧解析: 日志说明了解析方式",
      "多帧解析" in app.info_box.get("1.0", "end"))
app._force_frames_var.set(False)
app._on_toggle_force_frames()
pump()
check("多帧解析: 关闭后回到 1 帧", app.total_frames == 1, str(app.total_frames))
check("多帧解析: 关闭后 ROI 仍在", app.roi_list == [(4, 4, 20, 20)], str(app.roi_list))

# 13. 均匀帧 + 解锁显示范围（clim 不退化）
load(UNIFORM_TIF)
app._lock_display_var.set(False)
app._on_toggle_display_lock()
root.update()
c0 = app._im_img.get_clim()
check("解锁: 均匀帧 clim 不退化", c0[0] < c0[1], str(c0))
app._lock_display_var.set(True)
app._on_toggle_display_lock()
check("上锁: clim = 全堆栈范围", app._im_img.get_clim() == app._stack_clim)

# 14. 负值域警告
load(NEGATIVE_TIF)
app.roi_list = [(0, 0, 16, 16)]
app._update_roi_listbox()
app.contrast_method.current(1)
app._analyze_contrast()
pump()
check("科学性: 负强度给出显式警告", "负强度" in app.info_box.get("1.0", "end"))

# 15. 取消（大堆叠，等待首块完成后再取消，确保命中取消检查点）
load(BIG_TIF)
app.roi_list = [(0, 0, 256, 256)]
app._update_roi_listbox()
app.contrast_method.current(3)  # 稳健Michelson 最慢，保证多分块
app._analyze_contrast()
t0 = time.time()
while app.progress['value'] == 0 and time.time() - t0 < 30:
    root.update()
    time.sleep(0.005)
p_before = app.progress['value']
app._cancel_analysis()
pump()
check("取消: 状态复位且无结果",
      not app._analysis_running and app.contrast_data is None
      and app.progress['value'] == 0,
      f"取消前进度={p_before} 取消后进度={app.progress['value']}")

# 16. 复制信息面板
app._copy_info()
root.update()
try:
    copied = root.clipboard_get()
except tk.TclError:
    copied = ""
check("信息面板: 内容可复制", len(copied) > 0, f"{len(copied)} 字符")

# 17. 长序列曲线抽稀（临时降低阈值，避免生成超大测试数据）
load(STACK_TIF)
app.roi_list = [(10, 10, 30, 30)]
app._update_roi_listbox()
_orig_max_points = gui.PLOT_MAX_POINTS
gui.PLOT_MAX_POINTS = 4
try:
    app._analyze_contrast()
    pump()
    n_points = len(app.ax_plot.lines[0].get_xdata())
    # 8 帧 / 阈值 4 → 抽稀步长 2 → 4 个点（帧号 1,3,5,7）
    check("长序列: 曲线按阈值抽稀并标注",
          n_points == 4 and "抽稀 1/2" in app.ax_plot.get_title(),
          f"点数={n_points} 标题={app.ax_plot.get_title()}")
    check("长序列: 抽稀不影响导出数据完整", len(app.contrast_data[0]) == 8,
          f"{len(app.contrast_data[0])} 帧")
finally:
    gui.PLOT_MAX_POINTS = _orig_max_points

# 18. 小 ROI 稳健 Michelson 警告（3x3=9px < 100px 阈值）
app.roi_list = [(10, 10, 13, 13)]
app._update_roi_listbox()
app._show_frame()
app.contrast_method.current(3)
app._analyze_contrast()
pump()
check("科学性: 小ROI稳健分位给出警告",
      "无法有效排除" in app.info_box.get("1.0", "end"),
      "info 缺少小 ROI 提示")
app.roi_list = [(10, 10, 30, 30)]
app._update_roi_listbox()
app._show_frame()

# 19. 单帧含 NaN：单帧路径的非有限警告（与多帧路径对齐）
load(NAN_SINGLE_TIF)
app.roi_list = [(0, 0, 32, 32)]
app._update_roi_listbox()
app._show_frame()
app.contrast_method.current(0)
app._analyze_contrast()
root.update()
info_text = app.info_box.get("1.0", "end")
check("科学性: 单帧NaN给出非有限警告", "非有限" in info_text)
check("科学性: 单帧NaN衬度按NaN记录",
      app.contrast_data is not None and not np.isfinite(app.contrast_data[0][0]))

# 20. 逐页 DateTime 标签 → CSV 时间列
load(DT_TIF)
app.roi_list = [(0, 0, 6, 6)]
app._update_roi_listbox()
app._show_frame()
app.contrast_method.current(0)
app._analyze_contrast()
root.update()
DT_CSV = os.path.join(TMP, "dt_out.csv")
app.write_csv(DT_CSV)
dt_raw = open(DT_CSV, "rb").read().decode("utf-8-sig")
dt_df = pd.read_csv(DT_CSV, comment="#", index_col="Frame")
check("CSV: 逐帧 DateTime 列存在", "DateTime" in dt_df.columns, str(list(dt_df.columns)))
check("CSV: DateTime 值与页标签一致", list(dt_df["DateTime"]) == [_DT_STAMP],
      str(list(dt_df["DateTime"])))

root.destroy()
root.update()

n_fail = sum(1 for _, ok in checks if not ok)
print(f"\n==== 冒烟结果: {len(checks) - n_fail}/{len(checks)} 通过 ====")
sys.exit(1 if n_fail else 0)
