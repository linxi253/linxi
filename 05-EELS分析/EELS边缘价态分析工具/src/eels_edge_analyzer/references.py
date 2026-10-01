"""参考谱读取、统一预处理与可辨识性检查。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .dm4io import file_sha256, read_dataset
from .models import ReferenceSpec
from .processing import (
    PREEEDGE_BASELINE_WINDOW_EV,
    normalize_positive_area,
    savgol_by_ev,
)

CU_REFERENCE_FILENAMES: dict[int, tuple[str, ...]] = {
    0: ("Cu Metal Deconvolved Spectrum.dm4", "cu metal.dm4", "cu0.dm4"),
    1: ("Cu2O Deconvolved Spectrum.dm4", "cu2o.dm4", "cu1.dm4"),
    2: ("CuO Deconvolved Spectrum.dm4", "cuo.dm4", "cu2.dm4"),
}


def discover_cu_references(folder: Path) -> tuple[ReferenceSpec, ...]:
    """在论文参考谱目录中寻找 Cu0/Cu1/Cu2 三个常用文件。"""

    if not folder.is_dir():
        raise FileNotFoundError(f"参考谱文件夹不存在: {folder}")
    files = {path.name.lower(): path for path in folder.iterdir() if path.is_file()}
    specs: list[ReferenceSpec] = []
    for state, candidates in CU_REFERENCE_FILENAMES.items():
        found = next((files.get(item.lower()) for item in candidates if item.lower() in files), None)
        if found is None:
            # 允许用户改名后按关键字匹配，但要求唯一以防误选。
            token = {0: "metal", 1: "cu2o", 2: "cuo"}[state]
            matches = [
                path
                for name, path in files.items()
                if token in name and path.suffix.lower() in {".dm3", ".dm4", ".csv", ".txt"}
            ]
            if len(matches) == 1:
                found = matches[0]
        if found is None:
            raise FileNotFoundError(
                f"未在 {folder} 找到 Cu({state}) 参考谱。可使用标准文件名，"
                "或在 GUI 中分别选择 Cu0、Cu1、Cu2 文件。"
            )
        specs.append(ReferenceSpec(label=f"Cu{state}", oxidation_state=state, path=found))
    return tuple(specs)


def _read_text_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取两列文本/CSV 参考谱，兼容带或不带表头的简单格式。"""

    candidates: list[np.ndarray] = []
    for delimiter in (",", "\t", None):
        try:
            loaded = np.genfromtxt(path, delimiter=delimiter, comments="#", dtype=float)
        except (OSError, TypeError, ValueError):
            continue
        if loaded.ndim == 2 and loaded.shape[1] >= 2:
            finite = np.isfinite(loaded[:, 0]) & np.isfinite(loaded[:, 1])
            loaded = loaded[finite]
            if loaded.shape[0] >= 5:
                candidates.append(loaded[:, :2])
    if not candidates:
        raise ValueError(f"无法将 {path} 解析为两列能量-强度参考谱。")
    values = max(candidates, key=lambda item: item.shape[0])
    order = np.argsort(values[:, 0])
    return values[order, 0], values[order, 1]


def read_reference_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取 DM3/DM4 或 CSV/TXT 参考谱，并将多维对象沿空间轴平均。"""

    if path.suffix.lower() in {".csv", ".txt", ".dat", ".msa"}:
        return _read_text_reference(path)
    dataset = read_dataset(path, 0)
    if dataset.data.ndim < 1:
        raise ValueError(f"参考谱 {path} 没有强度数组。")
    energy = dataset.energy_ev
    if energy.size != dataset.data.shape[0]:
        raise ValueError(f"参考谱 {path} 的第一维不是能量轴。")
    if dataset.data.ndim == 1:
        intensity = dataset.data.astype(float)
    else:
        intensity = np.mean(dataset.data, axis=tuple(range(1, dataset.data.ndim))).astype(float)
    order = np.argsort(energy)
    return energy[order], intensity[order]


def prepare_references(
    specs: tuple[ReferenceSpec, ...],
    fit_energy_ev: np.ndarray,
    smoothing_ev: float,
    polyorder: int,
) -> tuple[np.ndarray, tuple[str, ...], list[dict[str, Any]]]:
    """以与样品相同的拟合网格、平滑尺度和面积归一化准备参考矩阵。"""

    ordered = tuple(sorted(specs, key=lambda item: item.oxidation_state))
    matrix: list[np.ndarray] = []
    details: list[dict[str, Any]] = []
    for spec in ordered:
        energy, intensity = read_reference_spectrum(spec.path)
        if energy.min() > fit_energy_ev.min() or energy.max() < fit_energy_ev.max():
            raise ValueError(
                f"参考谱 {spec.path.name} 不完全覆盖拟合能区 "
                f"{fit_energy_ev.min():.1f}–{fit_energy_ev.max():.1f} eV。"
            )
        # 先在原始参考谱的细能量网格上平滑，再插值至样品拟合网格，保持与
        # 已验证的 Cu-L 工作流一致。
        smoothed_full, window = savgol_by_ev(intensity, energy, smoothing_ev, polyorder)
        interpolated = np.interp(fit_energy_ev, energy, smoothed_full)
        # 预边基线窗口与样品谱共用同一常量：两处各写一个 3.0 会在某一边
        # 单独调整时静默失配，让参考与样品的扣基线口径不一致。
        preedge = (fit_energy_ev >= fit_energy_ev.min()) & (
            fit_energy_ev <= fit_energy_ev.min() + PREEEDGE_BASELINE_WINDOW_EV
        )
        baseline = float(np.mean(interpolated[preedge]))
        corrected = interpolated - baseline
        normalized = normalize_positive_area(corrected, fit_energy_ev)
        matrix.append(normalized)
        details.append(
            {
                "label": spec.label,
                "oxidation_state": spec.oxidation_state,
                "file": str(spec.path),
                "sha256": file_sha256(spec.path),
                "energy_start_ev": float(energy.min()),
                "energy_end_ev": float(energy.max()),
                "preedge_baseline": baseline,
                "savgol_window_points": window,
                "positive_area_before_normalization": float(
                    np.trapezoid(np.maximum(corrected, 0), fit_energy_ev)
                ),
            }
        )
    return np.column_stack(matrix), tuple(item.label for item in ordered), details


def reference_identifiability(matrix: np.ndarray) -> dict[str, Any]:
    """输出参考谱相关性、余弦相似度与条件数，提示难以独立区分的端元。"""

    if matrix.ndim != 2 or matrix.shape[1] < 2:
        return {"pearson_correlation": [], "cosine_similarity": [], "condition_number": None}
    correlations = np.corrcoef(matrix.T)
    normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=0, keepdims=True), 1e-12)
    cosine = normalized.T @ normalized
    singular = np.linalg.svd(matrix, compute_uv=False)
    return {
        "pearson_correlation": correlations.tolist(),
        "cosine_similarity": cosine.tolist(),
        "singular_values": singular.tolist(),
        "condition_number": float(singular[0] / max(singular[-1], 1e-12)),
    }
