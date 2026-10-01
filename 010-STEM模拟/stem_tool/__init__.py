"""STEM（HAADF/ADF）模拟工具：CIF → 带轴超胞 → 冻结声子多层法扫描 → 图像。

模块划分（与 09-HRTEM模拟 的工具层同构，便于两工具交叉使用）：
    params.py    参数模型（界面单位 ↔ 引擎单位）
    sim_core.py  模拟管线 + 计划预估（GUI/命令行共用）
    render.py    取向渲染 / 对比度 / 标尺
    export.py    PNG / TIFF16 / NPY / JSON 导出
    series.py    厚度系列、离焦系列、蒙太奇
    worker.py    后台线程
    gui.py       Tk 三栏界面
"""

from .params import QUALITY_PRESETS, StemParams
from .sim_core import SimPlan, StemSimResult, load_structure, plan, run_simulation

__all__ = [
    "StemParams",
    "QUALITY_PRESETS",
    "SimPlan",
    "StemSimResult",
    "plan",
    "run_simulation",
    "load_structure",
]
