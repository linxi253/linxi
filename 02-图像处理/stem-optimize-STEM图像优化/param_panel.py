"""
HRTEM/STEM 图像增强 - 参数调节面板组件

提供滑块和输入控件用于调整算法参数:
- 频谱正则 K 因子
- 混合比例（滤波/原图）
- 背景平滑 sigma
- CLAHE 截断阈值
- 输出位深选择（8-bit / 16-bit）
- CLAHE 开关
"""

import tkinter as tk
from tkinter import ttk

from contrast import DEFAULT_HIGH_PERCENTILE, DEFAULT_LOW_PERCENTILE
from pipeline import DEFAULT_PARAMS


class ParamPanel(tk.LabelFrame):
    """
    算法参数调节面板。

    用法:
        panel = ParamPanel(parent)
        params = panel.get_params()
        panel.set_params({'k_factor': 2.0})
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, text="算法参数", padx=8, pady=5, **kwargs)

        # 参数变量（与处理管线共用同一默认值来源）
        self.vars = {
            "k_factor": tk.DoubleVar(value=DEFAULT_PARAMS["k_factor"]),
            "blend_ratio": tk.DoubleVar(value=DEFAULT_PARAMS["blend_ratio"]),
            "delta": tk.DoubleVar(value=DEFAULT_PARAMS["delta"]),
            "clahe_clip": tk.DoubleVar(value=DEFAULT_PARAMS["clahe_clip"]),
            "output_bit_depth": tk.IntVar(value=DEFAULT_PARAMS["output_bit_depth"]),
            "apply_clahe": tk.BooleanVar(value=DEFAULT_PARAMS["apply_clahe"]),
        }
        # 范围分析百分位独立于帧处理参数，不进入 validate_params 白名单
        self.range_vars = {
            "low": tk.DoubleVar(value=DEFAULT_LOW_PERCENTILE),
            "high": tk.DoubleVar(value=DEFAULT_HIGH_PERCENTILE),
        }

        self._build_ui()

    def _build_ui(self):
        """构建参数界面。"""
        row = 0

        # --- 频谱正则 K 因子 ---
        tk.Label(self, text="频谱正则 K:", anchor="w").grid(
            row=row, column=0, sticky="w", padx=(0, 5), pady=3
        )
        k_frame = tk.Frame(self)
        k_frame.grid(row=row, column=1, sticky="ew", padx=2, pady=3)

        tk.Scale(
            k_frame,
            from_=0.1,
            to=5.0,
            resolution=0.1,
            orient="horizontal",
            variable=self.vars["k_factor"],
            length=160,
            showvalue=False,
        ).pack(side="left", fill="x", expand=True)
        tk.Label(
            k_frame, textvariable=self.vars["k_factor"], width=5, font=("Consolas", 9)
        ).pack(side="left", padx=(3, 0))

        row += 1

        # --- 混合比例 ---
        tk.Label(self, text="混合比例:", anchor="w").grid(
            row=row, column=0, sticky="w", padx=(0, 5), pady=3
        )
        br_frame = tk.Frame(self)
        br_frame.grid(row=row, column=1, sticky="ew", padx=2, pady=3)

        tk.Scale(
            br_frame,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient="horizontal",
            variable=self.vars["blend_ratio"],
            length=160,
            showvalue=False,
        ).pack(side="left", fill="x", expand=True)
        tk.Label(
            br_frame,
            textvariable=self.vars["blend_ratio"],
            width=5,
            font=("Consolas", 9),
        ).pack(side="left", padx=(3, 0))

        row += 1

        # --- 背景平滑 sigma ---
        tk.Label(self, text="背景平滑 σ:", anchor="w").grid(
            row=row, column=0, sticky="w", padx=(0, 5), pady=3
        )
        d_frame = tk.Frame(self)
        d_frame.grid(row=row, column=1, sticky="ew", padx=2, pady=3)

        tk.Scale(
            d_frame,
            from_=1.0,
            to=10.0,
            resolution=0.5,
            orient="horizontal",
            variable=self.vars["delta"],
            length=160,
            showvalue=False,
        ).pack(side="left", fill="x", expand=True)
        tk.Label(
            d_frame, textvariable=self.vars["delta"], width=5, font=("Consolas", 9)
        ).pack(side="left", padx=(3, 0))

        row += 1

        # --- CLAHE 截断 ---
        tk.Label(self, text="CLAHE 截断:", anchor="w").grid(
            row=row, column=0, sticky="w", padx=(0, 5), pady=3
        )
        c_frame = tk.Frame(self)
        c_frame.grid(row=row, column=1, sticky="ew", padx=2, pady=3)

        tk.Scale(
            c_frame,
            from_=0.5,
            to=5.0,
            resolution=0.1,
            orient="horizontal",
            variable=self.vars["clahe_clip"],
            length=160,
            showvalue=False,
        ).pack(side="left", fill="x", expand=True)
        tk.Label(
            c_frame, textvariable=self.vars["clahe_clip"], width=5, font=("Consolas", 9)
        ).pack(side="left", padx=(3, 0))

        row += 1

        # --- 范围百分位 ---
        tk.Label(self, text="范围百分位:", anchor="w").grid(
            row=row, column=0, sticky="w", padx=(0, 5), pady=3
        )
        p_frame = tk.Frame(self)
        p_frame.grid(row=row, column=1, sticky="w", padx=2, pady=3)
        tk.Label(p_frame, text="低").pack(side="left")
        tk.Spinbox(
            p_frame,
            from_=0.0,
            to=100.0,
            increment=0.5,
            width=6,
            textvariable=self.range_vars["low"],
        ).pack(side="left", padx=(2, 8))
        tk.Label(p_frame, text="高").pack(side="left")
        tk.Spinbox(
            p_frame,
            from_=0.0,
            to=100.0,
            increment=0.5,
            width=6,
            textvariable=self.range_vars["high"],
        ).pack(side="left", padx=(2, 0))

        row += 1

        # --- 输出位深 ---
        tk.Label(self, text="输出位深:", anchor="w").grid(
            row=row, column=0, sticky="w", padx=(0, 5), pady=3
        )
        bit_frame = tk.Frame(self)
        bit_frame.grid(row=row, column=1, sticky="w", padx=2, pady=3)

        tk.Radiobutton(
            bit_frame,
            text="8-bit",
            variable=self.vars["output_bit_depth"],
            value=8,
        ).pack(side="left")
        tk.Radiobutton(
            bit_frame,
            text="16-bit (推荐)",
            variable=self.vars["output_bit_depth"],
            value=16,
        ).pack(side="left", padx=(5, 0))

        row += 1

        # --- CLAHE 开关 ---
        self.clahe_check = tk.Checkbutton(
            self,
            text="应用 CLAHE（非线性，仅建议展示增强）",
            variable=self.vars["apply_clahe"],
        )
        self.clahe_check.grid(
            row=row, column=0, columnspan=2, sticky="w", padx=0, pady=3
        )

        row += 1

        # --- 参数说明 ---
        tk.Label(
            self,
            text="提示: K 因子越大滤波越保守；\n"
            "混合比例是局部纹理调制前的最大强度；\n"
            "范围百分位控制全堆栈映射窗口，改动后重新预览生效；\n"
            "16-bit 仅保留更多灰阶，不保证定量关系",
            justify="left",
            fg="#666666",
            font=("Arial", 8),
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(5, 0))

        # 配置列权重
        self.columnconfigure(1, weight=1)

    def get_params(self) -> dict:
        """获取当前参数字典。"""
        return {key: var.get() for key, var in self.vars.items()}

    def get_range_percentiles(self) -> tuple[float, float]:
        """获取全堆栈强度范围分析的低/高百分位。"""
        try:
            low = float(self.range_vars["low"].get())
            high = float(self.range_vars["high"].get())
        except (ValueError, tk.TclError) as exc:
            raise ValueError("范围百分位必须是数值") from exc
        if not (0.0 <= low < high <= 100.0):
            raise ValueError("范围百分位无效：需要 0 ≤ 低 < 高 ≤ 100")
        return low, high

    def set_enabled(self, enabled: bool):
        """Enable or disable every interactive control in the panel."""
        state = "normal" if enabled else "disabled"

        def apply_state(widget):
            for child in widget.winfo_children():
                if isinstance(
                    child,
                    (
                        tk.Scale,
                        tk.Spinbox,
                        tk.Checkbutton,
                        tk.Radiobutton,
                        ttk.Combobox,
                    ),
                ):
                    child.configure(state=state)
                apply_state(child)

        apply_state(self)

    def set_params(self, params: dict):
        """设置参数值。"""
        for key, value in params.items():
            if key in self.vars:
                self.vars[key].set(value)

    def reset_to_defaults(self):
        """重置为默认参数。"""
        self.set_params(dict(DEFAULT_PARAMS))
        self.range_vars["low"].set(DEFAULT_LOW_PERCENTILE)
        self.range_vars["high"].set(DEFAULT_HIGH_PERCENTILE)
