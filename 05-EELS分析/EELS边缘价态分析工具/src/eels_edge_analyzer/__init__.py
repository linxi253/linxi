"""EELS Edge Analyzer.

一套面向 STEM/EELS spectrum image 的、可复现的边缘价态分析工具。
核心算法与 Tkinter 界面分离，因此既可独立运行，也可被 TEM Suite 内嵌。
"""

from .models import AnalysisConfig, DistanceBin, ReferenceSpec

__all__ = ["AnalysisConfig", "DistanceBin", "ReferenceSpec"]
__version__ = "0.2.0"
