"""工具注册表 —— 描述 16 个被整合工具的加载方式与元数据。

分类顺序遵循电镜数据的实际处理流程：
    数据提取 -> 图像处理 -> EELS 谱学 -> 应变分析 / 4D-STEM -> 定量统计 -> 模拟仿真
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ----------------------------------------------------------------------
# 项目根定位
# ----------------------------------------------------------------------


def _looks_like_workspace(path: Path) -> bool:
    """判断目录是否为存放各工具项目的工作区。"""
    return (path / "01-视频与数据提取").is_dir() or (path / "02-图像处理").is_dir()


# 主源码开发树里部分项目的源码根比分类目录多一层（如非晶面积统计的
# pythonProject）。公开候选仓库已扁平化；Suite 需同时支持两种布局。
_NESTED_SOURCE_ROOT = "pythonProject"


def _detect_workspace_root() -> Path:
    """定位存放各工具项目的根目录（即 AIforTEM 目录）。

    打包后 exe 可能位于工作区内的任意层级（如 ``全整合/dist/``），
    因此从起点逐级向上查找标志性目录，而非假定固定层数。
    """
    env = os.environ.get("TEMSUITE_WORKSPACE")
    if env:
        candidate = Path(env).expanduser().resolve()
        if candidate.is_dir():
            return candidate

    if getattr(sys, "frozen", False):
        start = Path(sys.executable).resolve().parent
        # 若把工具源码一并打进了 exe，优先使用捆绑副本
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            bundled = Path(meipass) / "projects"
            if _looks_like_workspace(bundled):
                return bundled
    else:
        start = Path(__file__).resolve().parent

    for candidate in (start, *start.parents):
        if _looks_like_workspace(candidate):
            return candidate

    # 兜底：源码布局下 temsuite -> 全整合 -> AIforTEM
    return Path(__file__).resolve().parents[2]


WORKSPACE_ROOT = _detect_workspace_root()

LoadMode = Literal["module", "file"]
RootMode = Literal["inject", "patch"]
RunMode = Literal["embed", "subprocess"]


@dataclass(frozen=True)
class ToolSpec:
    """单个工具的接入描述。

    Attributes
    ----------
    tool_id:
        唯一标识，同时用作模块归属记账的键。
    name:
        界面显示名称。
    category:
        所属工作流阶段。
    relative_dir:
        项目目录，相对 :data:`WORKSPACE_ROOT`。
    entry:
        ``load_mode="module"`` 时为模块名；``load_mode="file"`` 时为文件名。
    factory:
        要实例化的 GUI 类名。``run_mode="subprocess"`` 时可为空。
    load_mode:
        ``module`` 走正常 import；``file`` 按路径加载（用于中文文件名）。
    root_mode:
        ``inject`` 表示 GUI 类接收 root 参数；``patch`` 表示其内部自建
        ``tk.Tk()``，需拦截构造函数。
    run_mode:
        ``embed`` 内嵌为标签页；``subprocess`` 以独立进程启动
        （用于与主进程存在不可调和冲突的工具）。
    src_layout:
        采用 src 布局时需额外加入 sys.path 的子目录。
    preload:
        需一并预加载的模块，规避运行期延迟导入产生模块副本。
    subprocess_args:
        ``run_mode="subprocess"`` 时的命令行参数。
    subprocess_module:
        ``run_mode="subprocess"`` 且工具需以 ``python -m <module>`` 方式启动时填写的
        模块名（包内使用相对导入的模块无法按脚本路径直接执行）。填写后
        ``entry`` 仅用于 :attr:`available` 校验文件存在，实际启动走本字段。
    python_exe:
        显式指定的子进程解释器路径。缺省时自动探测项目自带的 ``.venv``
        解释器（见 :meth:`resolve_python_exe`），两者都缺失才回退到套件
        自身解释器。
    """

    tool_id: str
    name: str
    category: str
    relative_dir: str
    entry: str
    factory: str = ""
    short_name: str = ""
    load_mode: LoadMode = "module"
    root_mode: RootMode = "inject"
    run_mode: RunMode = "embed"
    src_layout: tuple[str, ...] = ()
    preload: tuple[str, ...] = ()
    subprocess_args: tuple[str, ...] = ()
    subprocess_module: str = ""
    python_exe: Path | None = None
    description: str = ""
    notes: str = ""
    _cached_dir: list[Path] = field(default_factory=list, repr=False, compare=False)

    @property
    def tab_label(self) -> str:
        """标签页文字。十余个标签共享一行，故使用短名避免被挤压截断。"""
        return self.short_name or self.name

    @property
    def project_dir(self) -> Path:
        """项目根目录。

        公开候选仓库里各工具是**扁平**布局（``04-统计分析/非晶面积统计``），
        而主源码开发树里非晶面积统计的源码根多一层 ``pythonProject``
        （``04-统计分析/非晶面积统计/pythonProject``）。两者都已存在于实际
        工作区，Suite 必须在显式 ``TEMSUITE_WORKSPACE`` 下都能发现，否则同步时
        会被迫搬迁主源码目录（回归 2026-10-03 R3）。

        解析顺序：**扁平优先**，只有扁平布局的入口不存在时才回退到
        ``pythonProject`` 层。这样候选仓库行为完全不变，主源码树也能直接用。
        """
        base = (WORKSPACE_ROOT / self.relative_dir).resolve()
        if self._entry_exists(base):
            return base
        nested = base / _NESTED_SOURCE_ROOT
        if nested.is_dir() and self._entry_exists(nested):
            return nested
        return base

    def _entry_exists(self, root: Path) -> bool:
        """该根目录下是否能找到本工具的入口（不递归，只看本层）。"""
        if not root.is_dir():
            return False
        if self.load_mode == "file":
            return (root / self.entry).is_file()
        # module 模式：entry 的首段作为包/模块名查找
        head = self.entry.split(".")[0]
        return (root / f"{head}.py").is_file() or (root / head).is_dir()

    @property
    def extra_paths(self) -> tuple[Path, ...]:
        return tuple((self.project_dir / sub).resolve() for sub in self.src_layout)

    @property
    def available(self) -> bool:
        """项目目录与入口文件是否都存在。"""
        if not self.project_dir.is_dir():
            return False
        if self.load_mode == "file":
            return (self.project_dir / self.entry).is_file()
        return True

    def resolve_python_exe(self) -> Path | None:
        """解析 ``run_mode="subprocess"`` 工具应使用的解释器。

        优先级：显式 :attr:`python_exe` > 项目自带 ``.venv`` 中的解释器 >
        ``None``（调用方回退到套件自身解释器并告警）。

        各子进程工具的依赖版本锁定在它们自己的 venv 中（如原子标注工具锁定
        numpy 1.26.4，而套件锁为 numpy 2.2.6），复用套件解释器会打破该
        版本契约，因此必须优先使用项目解释器。
        """
        if self.python_exe is not None:
            return self.python_exe
        candidates = (
            self.project_dir / ".venv" / "Scripts" / "python.exe",
            self.project_dir / ".venv" / "bin" / "python",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None


# ----------------------------------------------------------------------
# 分类定义（决定界面中的分组顺序）
# ----------------------------------------------------------------------
CATEGORIES: tuple[tuple[str, str], ...] = (
    ("extract", "1 · 数据提取"),
    ("image", "2 · 图像处理"),
    ("eels", "3 · EELS谱学"),
    ("strain", "4 · 应变分析"),
    ("stem4d", "5 · 4D-STEM"),
    ("stats", "6 · 定量统计"),
    ("sim", "7 · 模拟仿真"),
)


# ----------------------------------------------------------------------
# 16 个工具
# ----------------------------------------------------------------------
TOOLS: tuple[ToolSpec, ...] = (
    # ---------------- 1 · 数据提取 ----------------
    ToolSpec(
        tool_id="video_extract",
        name="视频帧提取",
        short_name="视频提取",
        category="extract",
        relative_dir=r"01-视频与数据提取\视频切片工具",
        entry="video_extractor.ui",
        factory="ExtractorApp",
        root_mode="patch",
        preload=(
            "video_extractor.models",
            "video_extractor.runner",
            "video_extractor.writers",
            "video_extractor.decoder",
            "video_extractor.ffmpeg",
            "video_extractor.sampling",
            "video_extractor.manifest",
        ),
        description="视频逐帧提取为 ImageJ 兼容 TIFF 堆栈（未压缩、TYX），支持多种采样策略。",
    ),
    # ---------------- 2 · 图像处理 ----------------
    ToolSpec(
        tool_id="drift_correct",
        name="TIFF 漂移矫正 v7",
        short_name="漂移矫正",
        category="image",
        relative_dir=r"02-图像处理\drift-correction-v7",
        entry="drift_correction",
        factory="DriftCorrectionApp",
        preload=("drift_core",),
        description="v7 合并版：相位互相关 + 纯平移鲁棒中位数匹配的帧间漂移矫正，兼顾文件安全与算法质量。",
        notes="v7 由 v5.2（安全版）与 v6.1（算法版）合并而来；旧版归档（08-历史版本）在开发机上，未随本仓库分发。",
    ),
    ToolSpec(
        tool_id="hrtem_filter",
        name="HRTEM/STEM 滤波",
        short_name="HRTEM滤波",
        category="image",
        relative_dir=r"02-图像处理\hrtem-HRTEM滤波工具",
        entry="hrtem_filter.gui",
        factory="HRTEMFilterGUI",
        src_layout=("src",),
        preload=(
            "hrtem_filter.core",
            "hrtem_filter.params",
            "hrtem_filter.pipeline",
            "hrtem_filter.geometry",
            "hrtem_filter.tiff_io",
        ),
        description="Kilaas 旋转平均滤波，提供 Wiener / ABSF / Butterworth 多种增强。",
    ),
    ToolSpec(
        tool_id="stem_optimize",
        name="STEM 图像优化",
        short_name="图像优化",
        category="image",
        relative_dir=r"02-图像处理\stem-optimize-STEM图像优化",
        entry="main_window",
        factory="MainWindow",
        root_mode="patch",
        preload=(
            "pipeline",
            "filters",
            "contrast",
            "tiff_handler",
            "worker",
            "param_panel",
            "preview_canvas",
        ),
        description="STEM 图像增强与衬度优化，支持 ROI 选择与堆栈批处理。",
    ),
    ToolSpec(
        tool_id="image_filter",
        name="TIF 滤镜工具",
        short_name="滤镜",
        category="image",
        relative_dir=r"02-图像处理\图像加滤镜工具",
        entry="main",
        factory="FilterApp",
        preload=("image_filters", "tif_io"),
        description="批量空间域与频率域滤镜：高斯、中值、双边、FFT 带通。",
    ),
    # ---------------- 3 · EELS 谱学 ----------------
    ToolSpec(
        tool_id="eels_edge_analyzer",
        name="EELS边缘价态分析",
        short_name="EELS价态",
        category="eels",
        relative_dir=r"05-EELS分析\EELS边缘价态分析工具",
        entry="eels_edge_analyzer.gui",
        factory="EELSEdgeAnalyzerApp",
        src_layout=("src",),
        preload=(
            "eels_edge_analyzer.models",
            "eels_edge_analyzer.dm4io",
            "eels_edge_analyzer.processing",
            "eels_edge_analyzer.references",
            "eels_edge_analyzer.fitting",
            "eels_edge_analyzer.pipeline",
            "eels_edge_analyzer.reporting",
        ),
        description=(
            "配对 Dual-EELS 的样品边缘 Cu-L 价态剖面：自动数据对象识别、"
            "复散射校正、MLLS、bootstrap、模型选择和可追溯报告。"
        ),
        notes=(
            "v1 支持 DM3/DM4 配对 Dual-EELS 及 Cu0/Cu1/Cu2 参考谱。混合谱权重不是晶体学复合相的单独证据。"
        ),
    ),
    # ---------------- 4 · 应变分析 ----------------
    ToolSpec(
        tool_id="strain_gpa",
        name="GPA 应变分析",
        short_name="GPA",
        category="strain",
        relative_dir=r"03-应变分析\strainpp-GPA应变分析",
        entry="strain_gui",
        factory="StrainGUI",
        preload=("gpa", "phase", "utils", "dm_reader", "strain_analysis"),
        description="几何相位分析测量晶格应变场，支持 DM3/DM4/TIFF 输入。",
    ),
    ToolSpec(
        tool_id="ppa_strain",
        name="PPA 原子级应力",
        short_name="PPA",
        category="strain",
        relative_dir=r"03-应变分析\原子级应力分析-PPA",
        entry="ppa",
        factory="AtomMarkerApp",
        preload=("ppa_core", "ppa_stats"),
        description="原子柱定位与位移场分析，Delaunay 三角剖分计算局部应变张量。",
    ),
    ToolSpec(
        tool_id="atomic_recognition",
        name="原子识别与强度分析",
        short_name="原子识别",
        category="strain",
        relative_dir=r"03-应变分析\原子识别纯算法",
        entry="atomic_app",
        factory="AtomicRecognitionApp",
        preload=("atomic_core",),
        description=(
            "原子柱识别与逐帧人工校对，计算局部背景扣除的积分强度，"
            "支持跨帧稳定 ID 关联与标记图 / CSV 导出。"
        ),
        notes="自 PPA 独立出的传统识别工具（v1.1），只做识别与强度统计，不含位移/应变分析。",
    ),
    ToolSpec(
        tool_id="atom_annotator",
        name="原子标注工具",
        short_name="原子标注",
        category="strain",
        relative_dir=r"03-应变分析\原子中心识别模型开发",
        entry=r"src\atom_center\annotator_gui.py",
        load_mode="file",
        run_mode="subprocess",
        subprocess_module="atom_center.annotator_gui",
        src_layout=("src",),
        description=(
            "AtomCenterAnnotator：HAADF-STEM/HRTEM 原子中心点标注，"
            "支持 ROI 覆盖、草稿/审核流转、采集元数据清单与 YOLO 数据导出。"
        ),
        notes=(
            "GUI 流程需先经启动对话框创建/打开标注项目、销毁对话框根窗口后再建主窗口"
            "（多根窗口设计，无法内嵌），因此以独立子进程按 "
            "python -m atom_center.annotator_gui 启动；entry 仅供 available 校验，"
            "实际启动走 subprocess_module。"
        ),
    ),
    # ---------------- 5 · 4D-STEM ----------------
    ToolSpec(
        tool_id="stem4d",
        name="4D-STEM 处理",
        short_name="4D-STEM",
        category="stem4d",
        relative_dir=r"05-4D-STEM分析\4D-STEM-Processor",
        entry="stem_processor_gui",
        factory="STEMProcessorApp",
        preload=("core.dpc_core", "core.strain_mapping", "core.dimension_utils"),
        description="四维 STEM 数据处理：DPC、应变映射、取向映射、Ptychography。",
    ),
    # ---------------- 6 · 定量统计 ----------------
    ToolSpec(
        tool_id="contrast_stats",
        name="原子衬度统计",
        short_name="衬度统计",
        category="stats",
        relative_dir=r"04-统计分析\原子衬度统计",
        entry="tif图像衬度分析工具.py",
        factory="TIFContrastAnalyzer",
        load_mode="file",
        description="ROI 区域衬度统计，追踪原子柱强度随时间演化。",
    ),
    ToolSpec(
        tool_id="amorphous_area",
        name="晶体/非晶区域统计",
        short_name="晶非统计",
        category="stats",
        relative_dir=r"04-统计分析\非晶面积统计",
        entry="gui.app",
        factory="EMImageAnalyzerApp",
        preload=(
            "core.analysis",
            "core.segmentation",
            "core.measurement",
            "io_utils.tiff_handler",
            "constants",
        ),
        description="图像分割区分晶体与非晶区域，统计面积比例、周长与形状因子。",
    ),
    ToolSpec(
        tool_id="area_measure",
        name="TIFF 面积测量",
        short_name="面积测量",
        category="stats",
        relative_dir=r"04-统计分析\统计面积",
        entry="统计面积.py",
        factory="TiffStackViewer",
        load_mode="file",
        root_mode="patch",
        description="多边形 ROI 面积测量，支持标尺校准与像素到实际单位换算。",
    ),
    ToolSpec(
        tool_id="feature_evolution",
        name="特征区域演化分析",
        short_name="演化分析",
        category="stats",
        relative_dir=r"04-统计分析\特征区域演化分析",
        entry="应力面积统计.py",
        load_mode="file",
        run_mode="subprocess",
        description="HAADF-STEM 特征演化分析，输出 14 张期刊级图表与统计报告。",
        notes=(
            "交互式批处理脚本，**不是**纯 CLI：不带 --file 时它会自建 Tk 根窗口弹出"
            "文件选择框，再依次询问帧时间间隔与样品描述（取消任一输入即退出）。"
            "因此 Suite 以默认参数启动时用户会看到该工具自己的选择框，这是预期的"
            "交互路径；带 --file/--output/--dt 则可无人值守跑完整流程。"
            "模块顶层执行 matplotlib.use('Agg') 会全局覆盖其他工具依赖的 TkAgg 后端，"
            "故必须以独立子进程运行以彻底隔离。"
        ),
    ),
    # ---------------- 7 · 模拟仿真 ----------------
    ToolSpec(
        tool_id="hrtem_sim",
        name="HRTEM 高分辨模拟",
        short_name="HRTEM模拟",
        category="sim",
        relative_dir=r"09-HRTEM模拟",
        entry="hrtem_tool.gui",
        factory="HRTEMApp",
        preload=("tem_sim",),
        description=(
            "多层法 HRTEM 模拟（tem_sim 引擎）：CIF 导入，带轴/电镜参数/厚度/取向可控，"
            "输出 HRTEM 像、衍射花样与 CTF 曲线，支持离焦/厚度系列。"
        ),
        notes=(
            "gui.py 在 __init__ 阶段调用 matplotlib.use('TkAgg') 并经 plt.subplots() 建图，"
            "实例化期间已统一屏蔽后端切换，图像经 FigureCanvasTkAgg 显式嵌入。"
        ),
    ),
)


TOOLS_BY_ID: dict[str, ToolSpec] = {spec.tool_id: spec for spec in TOOLS}
CATEGORY_LABELS: dict[str, str] = dict(CATEGORIES)


def tools_in_category(category: str) -> list[ToolSpec]:
    return [spec for spec in TOOLS if spec.category == category]


def get_tool(tool_id: str) -> ToolSpec:
    try:
        return TOOLS_BY_ID[tool_id]
    except KeyError:
        raise KeyError(f"未注册的工具: {tool_id}") from None
