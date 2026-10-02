"""
标注审核/修正 GUI — 人工修正预标注并导出 YOLO 格式
=================================================
功能:
  - 加载图像 + 预标注点 (来自 semi_auto_label.py)
  - 左键点击添加漏检原子
  - 右键删除误检原子
  - 拖拽微调原子位置
  - Enter 保存并切换下一张
  - Ctrl+Z 撤销
  - 导出 YOLO 格式标签

用法:
    python label_review.py --labels ./raw_labels --images ./raw_images --export ./yolo_labels
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import matplotlib.patheffects as pe


class LabelReviewApp:
    """标注审核 GUI"""

    def __init__(self, root, label_dir, image_dir=None, export_dir=None,
                 atom_diameter=6.0, box_scale=1.2):
        self.root = root
        self.root.title("原子标注审核工具 — 左键添加 | 右键删除 | 拖拽移动 | Enter下一张")
        self.root.geometry("1400x850")

        self.label_dir = Path(label_dir)
        self.image_dir = Path(image_dir) if image_dir else None
        self.export_dir = Path(export_dir) if export_dir else self.label_dir / "yolo_export"
        self.atom_diameter = atom_diameter
        self.box_scale = box_scale

        # 加载所有标注文件
        self.label_files = sorted(self.label_dir.glob("*.json"))
        if not self.label_files:
            messagebox.showerror("错误", f"在 {label_dir} 中未找到 JSON 标注文件")
            self.root.destroy()
            return

        self.current_idx = 0
        self.points = []          # [(x, y), ...]
        self.image = None
        self.image_shape = (512, 512)
        self.undo_stack = []
        self._dragging_idx = None
        self._drag_original = None
        self._modified = False

        self._setup_ui()
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._load_current()

    def _setup_ui(self):
        # 主布局
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧: 图像显示
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(9, 8), dpi=100, facecolor='#1e1e1e')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

        # 右侧: 控制面板
        right_frame = ttk.Frame(main_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        right_frame.pack_propagate(False)

        # 文件信息
        info_frame = ttk.LabelFrame(right_frame, text="文件信息", padding=8)
        info_frame.pack(fill=tk.X, pady=(0, 5))
        self.lbl_filename = ttk.Label(info_frame, text="", font=('Consolas', 9))
        self.lbl_filename.pack(anchor=tk.W)
        self.lbl_progress = ttk.Label(info_frame, text="", font=('Consolas', 9))
        self.lbl_progress.pack(anchor=tk.W)
        self.lbl_atoms = ttk.Label(info_frame, text="", font=('Consolas', 9))
        self.lbl_atoms.pack(anchor=tk.W)

        # 参数调节
        param_frame = ttk.LabelFrame(right_frame, text="标注参数", padding=8)
        param_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(param_frame, text="原子直径 (px):").pack(anchor=tk.W)
        self.diam_var = tk.DoubleVar(value=self.atom_diameter)
        diam_scale = ttk.Scale(param_frame, from_=2, to=20, variable=self.diam_var,
                               orient=tk.HORIZONTAL, length=200,
                               command=self._on_diam_changed)
        diam_scale.pack(fill=tk.X)
        self.lbl_diam = ttk.Label(param_frame, text=f"{self.atom_diameter:.1f} px")
        self.lbl_diam.pack(anchor=tk.W)

        # 导航按钮
        nav_frame = ttk.LabelFrame(right_frame, text="导航", padding=8)
        nav_frame.pack(fill=tk.X, pady=(0, 5))

        btn_row1 = ttk.Frame(nav_frame)
        btn_row1.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row1, text="< 上一张 [←]", command=self._prev_image).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(btn_row1, text="下一张 [→] >", command=self._next_image).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        btn_row2 = ttk.Frame(nav_frame)
        btn_row2.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row2, text="保存 [Ctrl+S]", command=self._save_current).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(btn_row2, text="保存+下一张 [Enter]", command=self._save_and_next).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        # 编辑按钮
        edit_frame = ttk.LabelFrame(right_frame, text="编辑", padding=8)
        edit_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(edit_frame, text="撤销 [Ctrl+Z]", command=self._undo).pack(fill=tk.X, pady=1)
        ttk.Button(edit_frame, text="清除所有点", command=self._clear_all).pack(fill=tk.X, pady=1)

        # 导出
        export_frame = ttk.LabelFrame(right_frame, text="导出", padding=8)
        export_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(export_frame, text="导出当前为 YOLO 格式", command=self._export_yolo_current).pack(
            fill=tk.X, pady=1)
        ttk.Button(export_frame, text="批量导出全部为 YOLO", command=self._export_yolo_all).pack(
            fill=tk.X, pady=1)
        self.lbl_export_path = ttk.Label(export_frame, text=f"导出目录: {self.export_dir}",
                                         font=('', 7), foreground='gray')
        self.lbl_export_path.pack(anchor=tk.W, pady=(4, 0))

        # 状态栏
        self.status = ttk.Label(right_frame, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))

        # 已审核进度
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(right_frame, variable=self.progress_var,
                                            maximum=100)
        self.progress_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))

    def _bind_shortcuts(self):
        self.root.bind("<Return>", lambda e: self._save_and_next())
        self.root.bind("<Left>", lambda e: self._prev_image())
        self.root.bind("<Right>", lambda e: self._next_image())
        self.root.bind("<Control-z>", lambda e: self._undo())
        self.root.bind("<Control-s>", lambda e: self._save_current())
        self.root.bind("<Escape>", lambda e: self._on_close())

    def _load_current(self):
        """加载当前标注文件"""
        if self.current_idx >= len(self.label_files):
            self.current_idx = len(self.label_files) - 1
        if self.current_idx < 0:
            self.current_idx = 0

        label_file = self.label_files[self.current_idx]
        with open(label_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 只取前两个元素 (H, W): 旧项目可能存了 (H, W, C) 三元组, 直接解包会崩
        shape = data.get('image_shape') or [512, 512]
        if len(shape) < 2:
            raise ValueError(f"标注文件 {label_file.name} 的 image_shape 无效: {shape!r}")
        self.image_shape = (int(shape[0]), int(shape[1]))
        self.points = [(float(p['x']), float(p['y'])) for p in data.get('points', [])]
        self.undo_stack = []
        self._modified = False

        # 加载图像
        img_path = data.get('image_path', '')
        if self.image_dir:
            img_name = data.get('image_file', '')
            candidate = self.image_dir / img_name
            if candidate.exists():
                img_path = str(candidate)

        self.image = self._load_image_file(img_path)
        self._redraw()
        self._update_info()

    def _load_image_file(self, path):
        """加载图像"""
        try:
            import tifffile
            img = tifffile.imread(path)
        except Exception:
            try:
                img = plt.imread(path)
            except Exception:
                return np.zeros(self.image_shape)

        if img.ndim == 3:
            img = img[..., :3].mean(axis=2) if img.shape[2] >= 3 else img[..., 0]
        img = img.astype(np.float64)
        lo, hi = np.percentile(img, [1, 99])
        if hi > lo:
            img = np.clip((img - lo) / (hi - lo), 0, 1)
        return img

    def _redraw(self):
        """重绘图像和标注点"""
        self.ax.clear()
        if self.image is not None:
            self.ax.imshow(self.image, cmap='gray', origin='upper', aspect='equal')
        else:
            self.ax.set_xlim(0, self.image_shape[1])
            self.ax.set_ylim(self.image_shape[0], 0)

        # 画标注点
        radius = self.atom_diameter / 2.0
        for i, (x, y) in enumerate(self.points):
            c = Circle((x, y), radius=radius, fill=False,
                       edgecolor='#00ff88', linewidth=1.2, alpha=0.8)
            self.ax.add_patch(c)

        self.ax.set_title(f"原子数: {len(self.points)}  |  绿圈=标注  |  左键添加  右键删除",
                          color='white', fontsize=10)
        self.ax.tick_params(colors='gray', labelsize=7)
        self.canvas.draw_idle()

    def _update_info(self):
        """更新文件信息"""
        label_file = self.label_files[self.current_idx]
        with open(label_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.lbl_filename.config(text=f"文件: {data.get('image_file', 'N/A')}")
        self.lbl_progress.config(text=f"进度: {self.current_idx + 1} / {len(self.label_files)}")
        self.lbl_atoms.config(text=f"原子数: {len(self.points)}")

        # 更新进度条
        reviewed = sum(1 for f in self.label_files
                       if json.loads(f.read_text(encoding='utf-8')).get('reviewed', False))
        self.progress_var.set(reviewed / len(self.label_files) * 100)

    def _on_click(self, event):
        """鼠标点击"""
        if event.inaxes != self.ax or event.xdata is None:
            return

        x, y = event.xdata, event.ydata

        if event.button == 1:  # 左键: 添加或开始拖拽
            # 检查是否点击了已有点（拖拽模式）
            idx = self._find_nearest(x, y, threshold=self.atom_diameter)
            if idx is not None:
                self._dragging_idx = idx
                self._drag_original = self.points[idx]
            else:
                # 添加新点
                self.undo_stack.append(('add', len(self.points), (x, y)))
                self.points.append((x, y))
                self._modified = True
                self._redraw()
                self._update_info()

        elif event.button == 3:  # 右键: 删除最近点
            idx = self._find_nearest(x, y, threshold=self.atom_diameter * 1.5)
            if idx is not None:
                self.undo_stack.append(('delete', idx, self.points[idx]))
                del self.points[idx]
                self._modified = True
                self._redraw()
                self._update_info()

    def _on_release(self, event):
        """鼠标释放 — 完成拖拽"""
        if self._dragging_idx is not None and event.xdata is not None:
            old = self._drag_original
            new = (event.xdata, event.ydata)
            if abs(old[0] - new[0]) > 0.5 or abs(old[1] - new[1]) > 0.5:
                self.undo_stack.append(('move', self._dragging_idx, old))
                self.points[self._dragging_idx] = new
                self._modified = True
                self._redraw()
                self._update_info()
        self._dragging_idx = None
        self._drag_original = None

    def _on_motion(self, event):
        """鼠标移动 — 拖拽预览"""
        if self._dragging_idx is not None and event.xdata is not None:
            self.points[self._dragging_idx] = (event.xdata, event.ydata)
            self._redraw()

    def _on_diam_changed(self, val):
        self.atom_diameter = float(val)
        self.lbl_diam.config(text=f"{self.atom_diameter:.1f} px")
        self._redraw()

    def _find_nearest(self, x, y, threshold=10):
        """找最近的点索引"""
        if not self.points:
            return None
        pts = np.array(self.points)
        dists = np.sqrt((pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2)
        idx = np.argmin(dists)
        if dists[idx] < threshold:
            return int(idx)
        return None

    def _undo(self):
        """撤销"""
        if not self.undo_stack:
            return
        action, idx, data = self.undo_stack.pop()
        if action == 'add':
            if idx < len(self.points):
                del self.points[idx]
        elif action == 'delete':
            self.points.insert(idx, data)
        elif action == 'move':
            self.points[idx] = data
        elif action == 'clear':
            self.points = list(data)
        self._modified = True
        self._redraw()
        self._update_info()

    def _clear_all(self):
        """清除所有点"""
        if self.points:
            self.undo_stack.append(('clear', 0, list(self.points)))
            self.points = []
            self._modified = True
            self._redraw()
            self._update_info()

    def _on_close(self):
        """Prevent Escape/window close from silently discarding a drag or edit."""
        if not self._modified:
            self.root.destroy()
            return
        answer = messagebox.askyesnocancel("未保存修改", "当前标注尚未保存。是否保存后退出？")
        if answer is None:
            return
        if answer:
            self._save_current()
        self.root.destroy()

    def _save_current(self):
        """保存当前标注"""
        label_file = self.label_files[self.current_idx]
        with open(label_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        data['points'] = [{'x': p[0], 'y': p[1], 'confidence': 1.0} for p in self.points]
        data['reviewed'] = True
        data['atom_diameter'] = self.atom_diameter

        with open(label_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self._modified = False
        self.status.config(text=f"已保存: {label_file.name}")
        self._update_info()

    def _save_and_next(self):
        """保存并切换下一张"""
        self._save_current()
        self._next_image()

    def _next_image(self):
        """下一张"""
        if self._modified:
            self._save_current()
        if self.current_idx < len(self.label_files) - 1:
            self.current_idx += 1
            self._load_current()
        else:
            self.status.config(text="已是最后一张!")

    def _prev_image(self):
        """上一张"""
        if self._modified:
            self._save_current()
        if self.current_idx > 0:
            self.current_idx -= 1
            self._load_current()
        else:
            self.status.config(text="已是第一张!")

    def _export_yolo_current(self):
        """导出当前标注为 YOLO 格式"""
        self.export_dir.mkdir(parents=True, exist_ok=True)
        label_file = self.label_files[self.current_idx]
        stem = label_file.stem

        h, w = self.image_shape
        box_size = self.atom_diameter * self.box_scale

        yolo_lines = []
        dropped = 0
        for x, y in self.points:
            # YOLO 格式: class_id cx cy width height (归一化)
            cx = x / w
            cy = y / h
            bw = box_size / w
            bh = box_size / h
            # 出界点直接丢弃 (与 augment 管线一致); clamp 会生成贴边假框,
            # 让模型学到图像边界处的虚假目标
            if not (0.0 < cx < 1.0 and 0.0 < cy < 1.0):
                dropped += 1
                continue
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        export_file = self.export_dir / f"{stem}.txt"
        with open(export_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))

        drop_note = f", 丢弃 {dropped} 个出界点" if dropped else ""
        self.status.config(text=f"已导出 YOLO: {export_file.name} ({len(yolo_lines)} 个原子{drop_note})")

    def _export_yolo_all(self):
        """批量导出所有已审核标注为 YOLO 格式"""
        self.export_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        failed = []

        for label_file in self.label_files:
            # 单文件失败只记录并继续: 中途崩溃会让整批导出状态不明
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (OSError, ValueError) as exc:
                failed.append(f"{label_file.name}: {exc}")
                continue

            if not data.get('reviewed', False):
                continue

            shape = data.get('image_shape') or [512, 512]
            if len(shape) < 2:
                print(f"[警告] {label_file.name}: image_shape 无效 ({shape!r}), 跳过")
                continue
            h, w = int(shape[0]), int(shape[1])
            diam = data.get('atom_diameter', self.atom_diameter)
            box_size = diam * self.box_scale
            points = [(p['x'], p['y']) for p in data.get('points', [])]

            yolo_lines = []
            for x, y in points:
                cx = x / w
                cy = y / h
                bw = box_size / w
                bh = box_size / h
                # 出界点丢弃而非 clamp: clamp 会生成贴边假框 (与单文件导出一致)
                if not (0.0 < cx < 1.0 and 0.0 < cy < 1.0):
                    continue
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            export_file = self.export_dir / f"{label_file.stem}.txt"
            # encoding 显式化：内容目前是纯 ASCII，但按 locale 落盘不可移植。
            with open(export_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines))
            count += 1

        fail_note = f"\n{len(failed)} 个文件解析失败 (详见控制台)" if failed else ""
        for entry in failed:
            print(f"[警告] 批量导出跳过: {entry}")
        self.status.config(text=f"批量导出完成: {count} 个文件 -> {self.export_dir}")
        messagebox.showinfo("导出完成",
                            f"已导出 {count} 个 YOLO 标签文件到:\n{self.export_dir.resolve()}{fail_note}")


def main():
    parser = argparse.ArgumentParser(description="原子标注审核工具")
    parser.add_argument("--labels", "-l", required=True, help="预标注 JSON 文件夹")
    parser.add_argument("--images", "-i", default=None, help="图像文件夹 (可选)")
    parser.add_argument("--export", "-e", default=None, help="YOLO 导出目录")
    parser.add_argument("--diameter", "-d", type=float, default=6.0, help="原子直径 (px)")
    args = parser.parse_args()

    root = tk.Tk()
    app = LabelReviewApp(root, args.labels, args.images, args.export, args.diameter)
    root.mainloop()


if __name__ == "__main__":
    main()
