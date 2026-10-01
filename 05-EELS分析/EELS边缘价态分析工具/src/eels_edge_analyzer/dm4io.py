"""DM3/DM4 读取、对象检查和空间标尺处理。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .models import DatasetInfo, LoadedDataset


def _dm_module() -> Any:
    try:
        from ncempy.io import dm
    except ImportError as exc:  # pragma: no cover - 取决于部署环境
        raise RuntimeError(
            "缺少 ncempy，无法读取 DM3/DM4。请安装 ncempy，或使用 TEM Suite 的统一 Python 环境。"
        ) from exc
    return dm


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    """以流式方式计算原始文件哈希，不把原始文件写回磁盘。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


class HashWatch:
    """带 stat 快路径的文件 SHA-256 监视器。

    以 (大小, mtime_ns) 作为快速判据：stat 未变化时直接复用上一次哈希，
    stat 变化时才重算完整哈希。长计算与导出前后都要校验原始文件，这个
    快路径避免对多 GB 文件做反复全量读取，同时不放过任何改动。
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._hash: str | None = None
        self._stat: tuple[int, int] | None = None

    def current(self) -> str:
        try:
            status = self.path.stat()
            signature: tuple[int, int] | None = (status.st_size, status.st_mtime_ns)
        except OSError:
            signature = None
        if signature is not None and self._hash is not None and signature == self._stat:
            return self._hash
        self._hash = file_sha256(self.path)
        self._stat = signature
        return self._hash


def _units_from_record(record: dict[str, Any], n_axes: int) -> tuple[str, ...]:
    raw = record.get("pixelUnit", record.get("pixelUnits", ()))
    if isinstance(raw, str):
        return tuple(raw for _ in range(n_axes))
    if raw is None:
        return tuple("" for _ in range(n_axes))
    values = tuple(str(item) for item in raw)
    return values + tuple("" for _ in range(max(0, n_axes - len(values))))


def _title_from_record(record: dict[str, Any], index: int) -> str:
    for key in ("filename", "title", "name"):
        value = record.get(key)
        if value:
            return str(value)
    return f"dataset-{index}"


def read_dataset(path: Path, index: int) -> LoadedDataset:
    """读取一个数据对象并标准化为 float32 数组与坐标元组。"""

    dm = _dm_module()
    record = dm.dmReader(str(path), dSetNum=index)
    data = np.asarray(record["data"], dtype=np.float32)
    coords_raw = record.get("coords", ())
    coords = tuple(np.asarray(axis, dtype=float) for axis in coords_raw)
    if len(coords) < data.ndim:
        coords = coords + tuple(np.arange(size, dtype=float) for size in data.shape[len(coords) :])
    return LoadedDataset(
        index=index,
        data=data,
        coords=coords,
        units=_units_from_record(record, data.ndim),
        title=_title_from_record(record, index),
    )


def all_tags(path: Path) -> dict[str, Any]:
    """读取完整标签树，供 survey/SI 注册使用。"""

    dm = _dm_module()
    with dm.fileDM(str(path)) as handle:
        return dict(handle.allTags)


# ncempy fileDM._DM2NPDataTypes 的副本（补上 RGB 缩略图的 23），
# 供头部路径报告 dtype 而不实际读取数据体。
_DM_DATA_TYPES: dict[int, str] = {
    1: "int16",
    2: "float32",
    3: "complex64",
    6: "uint8",
    7: "int32",
    9: "int8",
    10: "uint16",
    11: "uint32",
    12: "float64",
    13: "complex128",
    23: "uint8",
}


def _coords_from_calibration(scale: float, origin: float, size: int) -> np.ndarray | float:
    """按 ncempy dmReader 的公式由标定参数重建坐标轴。

    两条路径（全量读取与头部解析）必须产生完全相同的坐标，因此这里精确
    镜像 dmReader 的实现：origin 先乘 ``-scale`` 并四舍五入到 4 位小数，
    坐标整体再四舍五入到 4 位小数；长度为 1 的轴返回标量 0。
    """

    if size == 1:
        return 0
    shifted = round(-1.0 * float(origin) * float(scale), ndigits=4)
    values = np.linspace(0, float(scale) * (size - 1), size) + shifted
    return np.round(values, decimals=4)


def _header_shape(dataset_ndim: int, sizes: tuple[int, int, int, int]) -> tuple[int, ...]:
    """按 ncempy getDataset 的 reshape/squeeze 规则由头部尺寸得到数组形状。"""

    x_size, y_size, z_size, z_size2 = sizes
    if dataset_ndim <= 2:
        if dataset_ndim == 2:
            return (int(y_size), int(x_size))
        # 1D 谱先 reshape (y, x) 再被 squeeze 掉单一维度。
        return tuple(int(s) for s in (int(y_size), int(x_size)) if s != 1)
    if dataset_ndim == 3:
        return (int(z_size), int(y_size), int(x_size))
    return (int(z_size2), int(z_size), int(y_size), int(x_size))


def _header_dataset_info(handle: Any, logical_index: int, path: Path) -> DatasetInfo:
    """用已打开 fileDM 的头部属性构造 DatasetInfo，不读取数据体。

    ncempy 的 getDataset 在多对象文件中把逻辑编号 i 映射到物理对象 i+1
    （对象 0 通常是缩略图）。这里必须使用同一映射，inspect 给出的编号才能
    与 read_dataset 实际读取的对象一致。
    """

    physical = logical_index if handle.numObjects == 1 else logical_index + 1
    ndim = int(handle.dataShape[physical])
    sizes = (
        int(handle.xSize[physical]),
        int(handle.ySize[physical]),
        int(handle.zSize[physical]),
        int(handle.zSize2[physical]),
    )
    shape = _header_shape(ndim, sizes)

    start = sum(int(n) for n in handle.dataShape[:physical])
    scales = list(handle.scale[start : start + ndim])[::-1]
    origins = list(handle.origin[start : start + ndim])[::-1]
    units_raw = list(handle.scaleUnit[start : start + ndim])[::-1]

    coords = [
        _coords_from_calibration(scale, origin, size)
        for scale, origin, size in zip(scales, origins, shape, strict=True)
    ]
    units = tuple(str(unit) for unit in units_raw)
    units = units + tuple("" for _ in range(max(0, len(shape) - len(units))))

    energy: np.ndarray | None = None
    if len(shape) >= 3 and coords:
        candidate = np.atleast_1d(np.asarray(coords[0], dtype=float))
        if candidate.size == shape[0] and candidate.size > 1 and np.all(np.isfinite(candidate)):
            energy = candidate

    data_type = int(handle.dataType[physical])
    dtype_name = _DM_DATA_TYPES.get(data_type, f"dm-type-{data_type}")

    title_key = f".ImageList.{physical + 1}.ImageTags.Name"
    title = str(handle.allTags.get(title_key) or path.name)

    return DatasetInfo(
        index=logical_index,
        shape=shape,
        dtype=dtype_name,
        energy_min_ev=float(np.min(energy)) if energy is not None else None,
        energy_max_ev=float(np.max(energy)) if energy is not None else None,
        spatial_shape=tuple(int(v) for v in shape[1:])
        if len(shape) >= 3
        else tuple(int(v) for v in shape),
        role_hint=_role_hint(tuple(shape), energy),
        name=title,
    )


def _logical_dataset_count(handle: Any) -> int:
    """与 getDataset 的索引映射保持一致的逻辑对象数。"""

    if handle.numObjects == 1:
        return 1
    # 多对象文件中逻辑 0 对应物理 1，因此可寻址的逻辑对象数为 numObjects-1；
    # 再按物理属性列表长度兜底，防止标定标签缺失导致索引越界。
    return max(0, min(handle.numObjects - 1, len(handle.dataType) - 1))


def _inspect_via_header(path: Path, max_datasets: int | None) -> list[DatasetInfo]:
    """只解析头部枚举对象；任何解析失败都返回空列表交由调用方回退。"""

    try:
        dm = _dm_module()
        items: list[DatasetInfo] = []
        with dm.fileDM(str(path)) as handle:
            count = _logical_dataset_count(handle)
            if max_datasets is not None:
                count = min(count, max_datasets)
            for index in range(count):
                try:
                    items.append(_header_dataset_info(handle, index, path))
                except Exception:  # noqa: BLE001 - ncempy 对异构对象的异常类型不固定
                    break
        return items
    except Exception:  # noqa: BLE001 - 头部路径的任何失败都交由读取路径兜底
        return []


def _inspect_via_read(path: Path, max_datasets: int | None) -> list[DatasetInfo]:
    """旧的完整读取路径；仅在头部路径得不到结果时使用。"""

    items: list[DatasetInfo] = []
    limit = max_datasets if max_datasets is not None else 64
    for index in range(limit):
        try:
            dataset = read_dataset(path, index)
        except Exception:  # ncempy 对超出对象编号通常抛不同异常
            if index == 0:
                raise
            break
        energy: np.ndarray | None = None
        # 仅 Spectrum Image 的第一个维度按能量轴解释。二维 survey 图的
        # 第一个坐标是空间 y，不能误报为能量范围。
        if dataset.data.ndim >= 3 and dataset.coords and dataset.coords[0].size == dataset.data.shape[0]:
            candidate = dataset.coords[0]
            if np.all(np.isfinite(candidate)) and candidate.size > 1:
                energy = candidate
        role = _role_hint(dataset.data.shape, energy)
        items.append(
            DatasetInfo(
                index=index,
                shape=tuple(int(value) for value in dataset.data.shape),
                dtype=str(dataset.data.dtype),
                energy_min_ev=float(np.min(energy)) if energy is not None else None,
                energy_max_ev=float(np.max(energy)) if energy is not None else None,
                spatial_shape=tuple(int(value) for value in dataset.data.shape[1:])
                if dataset.data.ndim >= 3
                else tuple(int(value) for value in dataset.data.shape),
                role_hint=role,
                name=dataset.title,
            )
        )
    return items


def inspect_dm4(path: Path, max_datasets: int | None = None) -> list[DatasetInfo]:
    """枚举 DM3/DM4 可读取对象并给出低损/高损/Survey 的候选角色。

    优先只解析文件头部（ncempy 解析标签树时不读取数据体），多 GB 的
    Dual-EELS 文件也能瞬时检查；头部路径得不到结果时回退到完整读取路径，
    行为不劣于旧版本。
    """

    items = _inspect_via_header(path, max_datasets)
    if not items:
        items = _inspect_via_read(path, max_datasets)
    if not items:
        raise ValueError(f"无法从 {path} 读取任何数据对象。")
    return items


def format_dataset_table(
    items: list[DatasetInfo],
    survey: int | None,
    low: int | None,
    high: int | None,
) -> str:
    """把对象检查结果格式化为 CLI 与 GUI 共用的表格文本。"""

    lines = ["index | shape | energy range (eV) | role"]
    for item in items:
        energy = (
            f"{item.energy_min_ev:.3f}-{item.energy_max_ev:.3f}"
            if item.energy_min_ev is not None and item.energy_max_ev is not None
            else "n/a"
        )
        lines.append(f"{item.index:5d} | {item.shape!s:18s} | {energy:22s} | {item.role_hint}")
    lines.append("")
    lines.append(f"自动推荐：Survey={survey}, 低损={low}, 高损={high}")
    return "\n".join(lines)


def _role_hint(shape: tuple[int, ...], energy: np.ndarray | None) -> str:
    if len(shape) == 2:
        return "survey_image"
    if len(shape) < 3 or energy is None:
        return "unknown"
    lo, hi = float(np.min(energy)), float(np.max(energy))
    if lo <= 0.0 <= hi:
        return "low_loss_si"
    if hi > 100.0 and lo > -5.0:
        return "core_loss_si"
    return "spectrum_image"


def infer_dataset_indices(items: list[DatasetInfo]) -> tuple[int | None, int | None, int | None]:
    """按维度和能量范围推荐 survey、低损和高损对象，不静默掩盖歧义。"""

    survey = next((item.index for item in items if item.role_hint == "survey_image"), None)
    low = next((item.index for item in items if item.role_hint == "low_loss_si"), None)
    high_candidates = [item for item in items if item.role_hint == "core_loss_si"]
    high = high_candidates[-1].index if high_candidates else None
    return survey, low, high


def find_tag_ending(tags: dict[str, Any], ending: str) -> Any | None:
    target = ending.lower()
    for key, value in tags.items():
        if key.lower().endswith(target):
            return value
    return None


def survey_rectangle(tags: dict[str, Any]) -> tuple[int, int, int, int] | None:
    """提取 Gatan survey-image 中标注的 Spectrum Image 矩形。"""

    value = find_tag_ending(tags, "SI.Acquisition.Survey Image.Spectrum Image Rect")
    if value is None:
        return None
    array = np.asarray(value, dtype=float).ravel()
    if array.size != 4:
        return None
    top, left, bottom, right = np.rint(array).astype(int)
    if bottom <= top or right <= left:
        return None
    return int(top), int(left), int(bottom), int(right)


def block_average(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """把 survey 图区域无插值地块平均到 Spectrum Image 网格。"""

    ny, nx = shape
    y_edges = np.linspace(0, image.shape[0], ny + 1)
    x_edges = np.linspace(0, image.shape[1], nx + 1)
    result = np.empty(shape, dtype=np.float32)
    for row in range(ny):
        y0, y1 = round(y_edges[row]), round(y_edges[row + 1])
        for column in range(nx):
            x0, x1 = round(x_edges[column]), round(x_edges[column + 1])
            patch = image[y0:y1, x0:x1]
            result[row, column] = float(np.mean(patch)) if patch.size else np.nan
    return result


def register_survey_to_si(
    survey: LoadedDataset | None,
    tags: dict[str, Any],
    spatial_shape: tuple[int, int],
) -> np.ndarray | None:
    """将 survey 映射到高损/低损 SI 的空间网格；标签缺失时返回 None。"""

    if survey is None or survey.data.ndim != 2:
        return None
    rect = survey_rectangle(tags)
    if rect is None:
        return None
    top, left, bottom, right = rect
    if top < 0 or left < 0 or bottom > survey.data.shape[0] or right > survey.data.shape[1]:
        return None
    crop = survey.data[top:bottom, left:right]
    if crop.size == 0:
        return None
    return block_average(crop, spatial_shape)


def _step_nm_with_diagnostics(axis: np.ndarray, unit: str) -> dict[str, Any]:
    """把相邻坐标步长换算为 nm，并记录原始单位与是否使用了启发式。

    ``unit_kind`` 取值：``nm``/``um``/``m``（单位明确）、``assumed_um`` 或
    ``assumed_nm``（单位缺失或不认识，按步长启发式猜测）。
    """

    if axis.size < 2:
        return {"step_nm": 1.0, "unit_raw": unit, "unit_kind": "single_row", "raw_step": 0.0}
    step = float(np.median(np.abs(np.diff(axis))))
    normalized = unit.strip().lower().replace("μ", "u").replace("µ", "u")
    if normalized in {"nm", "nanometer", "nanometre"}:
        kind, converted = "nm", step
    elif normalized in {"um", "micrometer", "micrometre", "µm"}:
        kind, converted = "um", step * 1000.0
    elif normalized in {"m", "meter", "metre"}:
        kind, converted = "m", step * 1e9
    else:
        # Gatan EELS SI 坐标常以 µm 存储；小于 0.1 的步长通常属于这种情况。
        kind = "assumed_um" if 0 < step < 0.1 else "assumed_nm"
        converted = step * 1000.0 if kind == "assumed_um" else step
    return {"step_nm": converted, "unit_raw": unit, "unit_kind": kind, "raw_step": step}


def coordinate_step_nm(axis: np.ndarray, unit: str = "") -> float:
    """将空间坐标轴相邻步长转换为 nm；未知单位时采用保守启发式。"""

    return float(_step_nm_with_diagnostics(axis, unit)["step_nm"])


def spatial_steps_nm(dataset: LoadedDataset) -> tuple[float, float, dict[str, Any]]:
    """返回 Spectrum Image 的 (y_step_nm, x_step_nm) 与换算诊断。"""

    if dataset.data.ndim != 3:
        raise ValueError("v1 需要形状为 (energy, y, x) 的 Spectrum Image。")
    y_info = _step_nm_with_diagnostics(dataset.coords[1], dataset.units[1])
    x_info = _step_nm_with_diagnostics(dataset.coords[2], dataset.units[2])
    return float(y_info["step_nm"]), float(x_info["step_nm"]), {"y": y_info, "x": x_info}
