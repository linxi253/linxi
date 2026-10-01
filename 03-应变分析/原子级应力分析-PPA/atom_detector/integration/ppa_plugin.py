"""
PPA GUI 集成插件 — 将 YOLOv8 原子检测集成到 PPA 主程序
====================================================
在 PPA 的"自动识别原子点"对话框中新增"深度学习检测"选项。

集成方式:
  - 不修改 ppa.py 核心逻辑
  - 通过 ppa_plugin.py 提供回调函数
  - 在 ppa.py 中仅需添加一个按钮调用本模块

用法 (在 ppa.py 中):
    from atom_detector.integration.ppa_plugin import DLDetectDialog
    # 在 auto_detect_points() 中添加:
    DLDetectDialog(self).show()
"""
import os
import sys
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ..model_security import ModelVerificationError


class DLDetectDialog:
    """
    深度学习原子检测对话框

    Parameters
    ----------
    ppa_app : AtomMarkerApp PPA 主程序实例
    """

    def __init__(self, ppa_app):
        self.app = ppa_app
        self.root = ppa_app.root
        self.detector = None

        # 默认模型路径
        self.default_model = Path(__file__).resolve().parent.parent / "models" / "best.pt"

    def show(self):
        """显示深度学习检测对话框"""
        # 检查模型是否存在
        if not self.default_model.exists():
            # 尝试查找任何 .pt 文件
            models_dir = self.default_model.parent
            pt_files = list(models_dir.glob("*.pt")) if models_dir.exists() else []
            if pt_files:
                self.default_model = pt_files[0]
            else:
                messagebox.showwarning(
                    "模型未找到",
                    f"未找到训练好的模型文件。\n\n"
                    f"请先训练模型:\n"
                    f"  python atom_detector/train/train.py --data <data.yaml>\n\n"
                    f"或将模型文件放到:\n  {models_dir}"
                )
                return

        # 创建对话框
        self.dlg = tk.Toplevel(self.root)
        self.dlg.title("深度学习原子检测 (YOLOv8)")
        self.dlg.geometry("450x380")
        self.dlg.resizable(False, False)
        self.dlg.transient(self.root)
        self.dlg.grab_set()

        self._build_ui()

    def _build_ui(self):
        """构建对话框 UI"""
        main = ttk.Frame(self.dlg, padding=15)
        main.pack(fill=tk.BOTH, expand=True)

        # 标题
        ttk.Label(main, text="YOLOv8 原子检测",
                  font=('Microsoft YaHei', 14, 'bold')).pack(anchor=tk.W, pady=(0, 10))

        # 模型路径
        model_frame = ttk.LabelFrame(main, text="模型配置", padding=8)
        model_frame.pack(fill=tk.X, pady=(0, 8))

        path_row = ttk.Frame(model_frame)
        path_row.pack(fill=tk.X)
        self.model_var = tk.StringVar(value=str(self.default_model))
        ttk.Entry(path_row, textvariable=self.model_var, width=40).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_row, text="浏览...", command=self._browse_model).pack(
            side=tk.LEFT, padx=(5, 0))

        # 参数配置
        param_frame = ttk.LabelFrame(main, text="检测参数", padding=8)
        param_frame.pack(fill=tk.X, pady=(0, 8))

        # 置信度
        conf_row = ttk.Frame(param_frame)
        conf_row.pack(fill=tk.X, pady=3)
        ttk.Label(conf_row, text="置信度阈值:", width=12).pack(side=tk.LEFT)
        self.conf_var = tk.DoubleVar(value=0.5)
        conf_scale = ttk.Scale(conf_row, from_=0.1, to=0.95,
                               variable=self.conf_var, orient=tk.HORIZONTAL, length=200)
        conf_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.conf_label = ttk.Label(conf_row, text="0.50", width=5)
        self.conf_label.pack(side=tk.LEFT)
        self.conf_var.trace_add("write", lambda *a: self.conf_label.config(
            text=f"{self.conf_var.get():.2f}"))

        # 推理尺寸
        size_row = ttk.Frame(param_frame)
        size_row.pack(fill=tk.X, pady=3)
        ttk.Label(size_row, text="推理尺寸:", width=12).pack(side=tk.LEFT)
        self.imgsz_var = tk.IntVar(value=640)
        ttk.Spinbox(size_row, from_=320, to=1280, increment=320,
                    textvariable=self.imgsz_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(size_row, text="(原子<4px 建议用 1024)").pack(side=tk.LEFT)

        # 亚像素修正
        refine_row = ttk.Frame(param_frame)
        refine_row.pack(fill=tk.X, pady=3)
        self.refine_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(refine_row, text="启用亚像素质心修正 (COM)",
                        variable=self.refine_var).pack(side=tk.LEFT)

        # ROI 选项
        roi_row = ttk.Frame(param_frame)
        roi_row.pack(fill=tk.X, pady=3)
        self.use_roi_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(roi_row, text="仅检测当前 ROI 区域",
                        variable=self.use_roi_var).pack(side=tk.LEFT)

        # 按钮
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        self.detect_btn = ttk.Button(btn_frame, text="开始检测",
                                     command=self._run_detection)
        self.detect_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(btn_frame, text="取消", command=self.dlg.destroy).pack(side=tk.LEFT)

        # 状态
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(main, textvariable=self.status_var, foreground='gray').pack(
            anchor=tk.W, pady=(8, 0))

    def _browse_model(self):
        """浏览模型文件"""
        path = filedialog.askopenfilename(
            title="选择 YOLOv8 模型文件",
            filetypes=[("PyTorch 模型", "*.pt"), ("ONNX 模型", "*.onnx"), ("所有文件", "*.*")]
        )
        if path:
            self.model_var.set(path)

    def _run_detection(self):
        """执行检测"""
        model_path = self.model_var.get()
        if not Path(model_path).exists():
            messagebox.showerror("错误", f"模型文件不存在:\n{model_path}")
            return

        # 检查是否有图像
        if self.app.image is None:
            messagebox.showwarning("警告", "请先加载图像!")
            return

        self.detect_btn.config(state='disabled')
        self.status_var.set("正在加载模型...")
        self.dlg.update()

        try:
            # 导入并创建检测器
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
            from atom_detector.infer.detector import AtomDetector

            detector = AtomDetector(
                model_path=model_path,
                conf=self.conf_var.get(),
                imgsz=self.imgsz_var.get(),
                refine=self.refine_var.get(),
                refine_window=5,
            )

            self.status_var.set("正在检测原子...")
            self.dlg.update()

            # 获取 ROI (PPA 主程序属性为 detect_roi, 原 'roi' 属性不存在导致该功能失效)
            roi = None
            if self.use_roi_var.get():
                roi = getattr(self.app, 'detect_roi', None)

            # 执行检测
            points, confidences = detector.detect(self.app.image, roi=roi)

            if not points:
                messagebox.showinfo("提示", "未检测到原子，请尝试降低置信度阈值。")
                self.detect_btn.config(state='normal')
                self.status_var.set("未检测到原子")
                return

            # 与主程序 _finish_auto_detect 相同的三选一合并语义:
            # 是=追加(非破坏), 否=替换(破坏性, 明确标注), 取消=丢弃。
            choice = messagebox.askyesnocancel(
                "检测完成",
                f"检测到 {len(points)} 个原子\n"
                f"置信度范围: [{min(confidences):.2f}, {max(confidences):.2f}]\n"
                f"亚像素修正: {'已启用' if self.refine_var.get() else '未启用'}\n\n"
                f"如何处理检测结果？\n"
                f"  [是(Y)]   追加到当前列表 (保留已有 {len(self.app.points)} 个点)\n"
                f"  [否(N)]   替换现有所有点 (当前 {len(self.app.points)} 个点将被丢弃)\n"
                f"  [取消]    丢弃本次检测结果")
            if choice is None:
                self.status_var.set(f"已丢弃本次检测结果 ({len(points)} 个原子)")
                self.detect_btn.config(state='normal')
                return
            if choice:
                self.app.points.extend(points)
            else:
                self.app.points = points
            if hasattr(self.app, 'selected_point_idx'):
                self.app.selected_point_idx = None
            if hasattr(self.app, 'ref_select_indices'):
                self.app.ref_select_indices = []  # 替换点表后旧索引会错位
            if hasattr(self.app, '_reset_ref_multi_selection'):
                self.app._reset_ref_multi_selection()
            if hasattr(self.app, '_clear_analysis_results'):
                self.app._clear_analysis_results()
            if hasattr(self.app, 'refresh_display'):
                self.app.refresh_display()

            self.status_var.set(f"检测完成: {len(points)} 个原子")
            messagebox.showinfo(
                "检测完成",
                f"成功检测到 {len(points)} 个原子!\n\n"
                f"结果已载入 PPA，可继续进行应变分析。")

            self.dlg.destroy()

        except ModelVerificationError as e:
            messagebox.showerror(
                "模型未通过校验",
                f"{e}\n\n"
                f"只有在确认模型来源与哈希后，才能将其加入白名单:\n"
                f"  python -m atom_detector.model_security <model.pt> --allow")
            self.detect_btn.config(state='normal')
            self.status_var.set("模型校验失败")

        except ImportError as e:
            messagebox.showerror("导入错误",
                                 f"无法导入依赖:\n{e}\n\n"
                                 f"请确保已安装:\n"
                                 f"  pip install ultralytics torch")
            self.detect_btn.config(state='normal')
            self.status_var.set("导入失败")

        except Exception as e:
            messagebox.showerror("检测错误", f"检测过程出错:\n{e}")
            self.detect_btn.config(state='normal')
            self.status_var.set("检测失败")


def check_model_available():
    """检查模型是否可用 (供 PPA 判断是否显示 DL 按钮)"""
    model_path = Path(__file__).resolve().parent.parent / "models" / "best.pt"
    if model_path.exists():
        return True
    # 检查是否有任何 .pt 文件
    models_dir = model_path.parent
    if models_dir.exists():
        return bool(list(models_dir.glob("*.pt")))
    return False


def get_dl_detect_callback(ppa_app):
    """
    获取 DL 检测回调函数 (供 PPA 按钮绑定)

    用法 (在 ppa.py 中):
        try:
            from atom_detector.integration.ppa_plugin import get_dl_detect_callback, check_model_available
            if check_model_available():
                ttk.Button(..., text="DL 检测", command=get_dl_detect_callback(self))
        except ImportError:
            pass
    """
    def callback():
        DLDetectDialog(ppa_app).show()
    return callback
