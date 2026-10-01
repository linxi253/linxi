"""隔离式项目加载器 —— 在同一进程内安全地加载多个独立项目。

为什么需要隔离
--------------
被整合的项目彼此独立开发，顶层模块名存在真实冲突：

    main.py         stem-optimize / 图像加滤镜工具 / 非晶面积统计  三方冲突
    core/           非晶面积统计 / 4D-STEM-Processor              双方冲突
    pipeline.py     stem-optimize
    filters.py      stem-optimize
    utils.py        strainpp-GPA
    version.py errors.py constants.py tiff_handler.py             常见名

若把所有项目目录一并塞进 ``sys.path``，先加载者会永久占据 ``sys.modules``，
后加载的项目会拿到别人的模块，引发难以排查的错误。

隔离策略
--------
``ProjectLoader`` 按需加载：用户点开某个工具时才加载它，并在加载前把其他工具
占用的同名模块从 ``sys.modules`` 暂存归档，使当前项目能加载到自己的版本。

这一策略成立的依据是：Python 模块对象一旦被绑定到已创建的类与函数上，即便随后
从 ``sys.modules`` 移除也不影响既有对象继续运行；只有「运行期新执行的 import
语句」才会受影响。经检查各项目均在文件头部完成导入，因此安全。

已知限制
--------
若某工具在运行期间（例如点击按钮后）才首次 import 自己项目内的模块，且该模块名
已被其他工具占用，则会加载出第二份副本。此时跨副本的 ``isinstance`` 判断可能失效。
如遇此类问题，应为该工具补充预加载模块清单（见 ``ToolSpec.preload``）。
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

from .mplbackend import backend_locked

logger = logging.getLogger(__name__)

# 这些目录明确不是可导入模块，不参与顶层名字记账。
# 注意不要把 tools/scripts/diagnostics/processing 之类列进来 ——
# 它们在部分项目中确实是真实的包（如 4D-STEM 的 processing/、diagnostics/）。
_SKIP_DIR_NAMES = frozenset(
    {
        "__pycache__",
        "build",
        "dist",
        "tests",
        "test",
        "legacy",
        "backups",
        "docs",
        "release",
        "wheelhouse",
        "output",
        "htmlcov",
        "site-packages",
    }
)


def _scan_top_level_names(project_dir: Path) -> set[str]:
    """列出项目根目录下会占用 ``sys.modules`` 顶层名字的模块与包。

    目录既可能是常规包（含 ``__init__.py``），也可能是**命名空间包**
    （无 ``__init__.py`` 但含 .py 文件，如 4D-STEM-Processor 的 ``core/``）。
    两者都会占用顶层模块名，必须一并纳入记账，否则冲突归档会失效。
    """
    names: set[str] = set()
    if not project_dir.is_dir():
        return names

    for entry in project_dir.iterdir():
        name = entry.name
        if name.startswith(".") or name.startswith("_"):
            continue
        if entry.is_file() and name.endswith(".py"):
            names.add(name[:-3])
        elif entry.is_dir() and name not in _SKIP_DIR_NAMES:
            if (entry / "__init__.py").is_file():
                names.add(name)
            elif any(child.suffix == ".py" for child in entry.iterdir() if child.is_file()):
                # 命名空间包
                names.add(name)
    return names


class ProjectLoader:
    """管理多个独立项目在同一进程内的加载与模块隔离。"""

    def __init__(self) -> None:
        # module_name -> 占用它的 tool_id
        self._owner: dict[str, str] = {}
        # tool_id -> {module_name: module}  被暂存归档的模块
        self._archive: dict[str, dict[str, ModuleType]] = {}
        # tool_id -> 该工具项目根目录下的顶层模块名集合
        self._claims: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    def _release_conflicts(self, tool_id: str, project_dir: Path) -> None:
        """把其他工具占用的同名模块归档，为当前工具腾出名字。"""
        wanted = _scan_top_level_names(project_dir)
        self._claims[tool_id] = wanted

        for name in wanted:
            owner = self._owner.get(name)
            if owner is None or owner == tool_id:
                continue
            # 该名字被别的工具占用，连同其子模块一起归档
            store = self._archive.setdefault(owner, {})
            for mod_name in [n for n in sys.modules if n == name or n.startswith(f"{name}.")]:
                store[mod_name] = sys.modules.pop(mod_name)
            logger.debug("模块名 %r 由 %s 归档，让位给 %s", name, owner, tool_id)

        # 若当前工具此前被归档过，恢复它自己的模块
        restored = self._archive.pop(tool_id, None)
        if restored:
            sys.modules.update(restored)
            logger.debug("恢复 %s 的 %d 个归档模块", tool_id, len(restored))

        for name in wanted:
            self._owner[name] = tool_id

    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def _project_on_path(self, project_dir: Path, extra_paths: tuple[Path, ...] = ()) -> Iterator[None]:
        """临时把项目目录置于 ``sys.path`` 最前，退出时精确还原。"""
        injected = [str(project_dir), *(str(p) for p in extra_paths)]
        for p in reversed(injected):
            sys.path.insert(0, p)
        try:
            yield
        finally:
            for p in injected:
                with contextlib.suppress(ValueError):
                    sys.path.remove(p)

    # ------------------------------------------------------------------
    def load_module(
        self,
        tool_id: str,
        project_dir: Path,
        module_name: str,
        *,
        extra_paths: tuple[Path, ...] = (),
        preload: tuple[str, ...] = (),
    ) -> ModuleType:
        """加载项目内的一个模块。

        Parameters
        ----------
        tool_id:
            工具标识，用于模块归属记账。
        project_dir:
            项目根目录，将被临时加入 ``sys.path``。
        module_name:
            要导入的模块名，支持点号形式（如 ``video_extractor.ui``）。
        extra_paths:
            额外需要加入 ``sys.path`` 的目录（如使用 src 布局的项目）。
        preload:
            需要一并提前导入的模块名，用于规避运行期延迟导入带来的副本问题。
        """
        project_dir = project_dir.resolve()
        if not project_dir.is_dir():
            raise FileNotFoundError(f"项目目录不存在: {project_dir}")

        self._release_conflicts(tool_id, project_dir)

        with backend_locked(), self._project_on_path(project_dir, extra_paths):
            module = importlib.import_module(module_name)
            for name in preload:
                with contextlib.suppress(ImportError):
                    importlib.import_module(name)
        return module

    # ------------------------------------------------------------------
    def load_file(
        self,
        tool_id: str,
        project_dir: Path,
        file_name: str,
        *,
        alias: str | None = None,
        extra_paths: tuple[Path, ...] = (),
    ) -> ModuleType:
        """按文件路径加载模块，用于中文名等无法直接 import 的情形。

        例如 ``统计面积.py``、``应力面积统计.py``、``tif图像衬度分析工具.py``。
        """
        project_dir = project_dir.resolve()
        target = project_dir / file_name
        if not target.is_file():
            raise FileNotFoundError(f"模块文件不存在: {target}")

        self._release_conflicts(tool_id, project_dir)

        mod_name = alias or f"_temsuite_{tool_id}_{target.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, target)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法为 {target} 创建模块 spec")

        module = importlib.util.module_from_spec(spec)
        with backend_locked(), self._project_on_path(project_dir, extra_paths):
            # 先注册再执行，以支持模块内部的自引用
            sys.modules[mod_name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(mod_name, None)
                raise
        self._owner[mod_name] = tool_id
        return module


# 进程级共享的单例加载器
LOADER = ProjectLoader()
