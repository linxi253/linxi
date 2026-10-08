"""Scientifically explicit 2-D strain calculations.

All functions use physical Cartesian coordinates.  ``reference_positions`` are
material coordinates X and ``deformed_positions`` are current coordinates x.
The GUI may display images with y increasing downwards, but callers must apply
that display conversion before calling this module if they need physical y-up
signs for shear and rotation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree


class AnalysisError(ValueError):
    """Raised when input geometry cannot support a reliable strain result."""


@dataclass(frozen=True)
class StrainResult:
    """Per-element or per-site strain result in physical Cartesian coordinates."""

    locations: np.ndarray
    simplices: np.ndarray | None
    deformation_gradient: np.ndarray
    small_xx: np.ndarray
    small_yy: np.ndarray
    small_xy: np.ndarray
    green_xx: np.ndarray
    green_yy: np.ndarray
    green_xy: np.ndarray
    equivalent_small: np.ndarray
    equivalent_green: np.ndarray
    infinitesimal_rotation: np.ndarray
    polar_rotation: np.ndarray
    edge_mask: np.ndarray
    quality_mask: np.ndarray
    method: str
    # Indices into the input arrays that produced each result row.  For CST this
    # is None (rows are triangles); for local peak pairs it identifies the sites
    # represented by each result row.
    site_indices: np.ndarray | None = None
    # Per-site quality label and invalid reason.  These are populated by local
    # PPA, whose output deliberately keeps one row for every accepted atom.
    site_quality: np.ndarray | None = None
    invalid_reasons: np.ndarray | None = None


def _as_points(values: Iterable[Iterable[float]], name: str) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise AnalysisError(f"{name} must have shape (N, 2).")
    if len(points) < 3:
        raise AnalysisError("At least three points are required.")
    if not np.isfinite(points).all():
        raise AnalysisError(f"{name} contains NaN or infinity.")
    return points


# 参考晶格条件数的经验指导 (validate_reference_lattice 的 max_condition 参数)。
# 条件数恶化的来源是"基矢几何"而非"晶体对称性": 两矢量接近共线
# (夹角 → 0°/180°) 或长度悬殊。经验值:
#   general        30.0  —— 默认; 对应夹角约 >1.9° 且长度比 <15 的量级
#   high_symmetry  50.0  —— 允许更极端基矢选择的高对称晶系工作流 (慎用)
#   low_symmetry   15.0  —— 低对称/易畸变体系, 更严格地拒绝病态基矢
LATTICE_CONDITION_GUIDANCE = {
    "general": 30.0,
    "high_symmetry": 50.0,
    "low_symmetry": 15.0,
}


def validate_max_condition(value: float, *, label: str = "max_condition") -> float:
    """Validate a lattice condition-number limit and return it as ``float``.

    阈值必须为**有限正数**。``NaN``/``inf`` 会让 ``condition > max_condition``
    恒为 False，从而静默绕过所有病态基矢检查；``0``/负值则会把一切合法基矢都
    判为病态。两种情况都在任何计算发生前给出明确错误。
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{label} must be a finite positive number, got {value!r}.") from exc
    if not np.isfinite(number):
        raise AnalysisError(f"{label} must be a finite positive number, got {value!r}.")
    if number <= 0:
        raise AnalysisError(f"{label} must be a finite positive number, got {value!r}.")
    return number


def validate_reference_lattice(a_vec: Iterable[float], b_vec: Iterable[float], *, max_condition: float = 30.0) -> float:
    """Validate two non-collinear reference lattice vectors and return condition number.

    ``max_condition`` 可按 ``LATTICE_CONDITION_GUIDANCE`` 的指导按体系放宽/收紧;
    默认 30.0。条件数超限通常意味着两矢量接近共线或长度悬殊, 此时
    lattice-index 分配与局部应变的数值噪声都会被放大。
    阈值本身必须是有限正数（见 :func:`validate_max_condition`）。
    """
    limit = validate_max_condition(max_condition)
    lattice = np.column_stack((np.asarray(a_vec, dtype=float), np.asarray(b_vec, dtype=float)))
    if lattice.shape != (2, 2) or not np.isfinite(lattice).all():
        raise AnalysisError("Reference lattice vectors must be two finite 2-D vectors.")
    length_a, length_b = np.linalg.norm(lattice[:, 0]), np.linalg.norm(lattice[:, 1])
    if min(length_a, length_b) <= 1e-12:
        raise AnalysisError("Reference lattice vectors must have non-zero length.")
    condition = float(np.linalg.cond(lattice))
    if not np.isfinite(condition) or condition > limit:
        raise AnalysisError(
            f"Reference lattice is ill-conditioned (condition number {condition:.1f} > "
            f"{limit}); choose two well-separated directions with comparable "
            "lengths (included angle far from 0°/180°)."
        )
    return condition


def _equivalent_strain(exx: np.ndarray, eyy: np.ndarray, exy: np.ndarray, coefficient: float) -> np.ndarray:
    if coefficient <= 0 or not np.isfinite(coefficient):
        raise AnalysisError("Equivalent-strain coefficient must be a positive finite number.")
    radicand = coefficient * (exx * exx + eyy * eyy - exx * eyy + 3.0 * exy * exy)
    return np.sqrt(np.maximum(radicand, 0.0))


def _polar_angle(F: np.ndarray) -> np.ndarray:
    """Return the signed 2-D polar-decomposition rotation for each F."""
    angles = np.full(len(F), np.nan, dtype=np.float64)
    for i, matrix in enumerate(F):
        if not np.isfinite(matrix).all():
            continue
        u, _, vt = np.linalg.svd(matrix)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        angles[i] = np.arctan2(rotation[1, 0], rotation[0, 0])
    return angles


def _strain_from_F(
    F: np.ndarray,
    locations: np.ndarray,
    simplices: np.ndarray | None,
    edge_mask: np.ndarray,
    quality_mask: np.ndarray,
    method: str,
    equivalent_coefficient: float,
    site_indices: np.ndarray | None = None,
    site_quality: np.ndarray | None = None,
    invalid_reasons: np.ndarray | None = None,
) -> StrainResult:
    identity = np.eye(2)
    H = F - identity
    small = 0.5 * (H + np.swapaxes(H, 1, 2))
    green = 0.5 * (np.matmul(np.swapaxes(F, 1, 2), F) - identity)
    return StrainResult(
        locations=locations,
        simplices=simplices,
        deformation_gradient=F,
        small_xx=small[:, 0, 0],
        small_yy=small[:, 1, 1],
        small_xy=small[:, 0, 1],
        green_xx=green[:, 0, 0],
        green_yy=green[:, 1, 1],
        green_xy=green[:, 0, 1],
        equivalent_small=_equivalent_strain(small[:, 0, 0], small[:, 1, 1], small[:, 0, 1], equivalent_coefficient),
        equivalent_green=_equivalent_strain(green[:, 0, 0], green[:, 1, 1], green[:, 0, 1], equivalent_coefficient),
        infinitesimal_rotation=0.5 * (H[:, 1, 0] - H[:, 0, 1]),
        polar_rotation=_polar_angle(F),
        edge_mask=edge_mask,
        quality_mask=quality_mask,
        method=method,
        site_indices=site_indices,
        site_quality=site_quality,
        invalid_reasons=invalid_reasons,
    )


def compute_cst_strain(
    reference_positions: Iterable[Iterable[float]],
    deformed_positions: Iterable[Iterable[float]],
    *,
    max_edge_factor: float = 2.5,
    equivalent_coefficient: float = 4.0 / 9.0,
) -> StrainResult:
    """Compute constant-strain-triangle results with gradients in reference coordinates.

    Triangulation is deliberately performed on X, not x.  This makes finite
    Green-Lagrange strain objective under rigid rotation and prevents the
    systematic underestimation caused by differentiating in the deformed frame.
    """
    reference = _as_points(reference_positions, "reference_positions")
    deformed = _as_points(deformed_positions, "deformed_positions")
    if reference.shape != deformed.shape:
        raise AnalysisError("Reference and deformed positions must have the same shape.")
    if max_edge_factor <= 0:
        raise AnalysisError("max_edge_factor must be positive.")
    if len(np.unique(reference, axis=0)) != len(reference):
        raise AnalysisError("Reference positions contain duplicates; resolve duplicate lattice matches first.")
    if np.linalg.matrix_rank(reference - reference.mean(axis=0)) < 2:
        raise AnalysisError("Reference positions are collinear; 2-D strain is undefined.")

    try:
        triangulation = Delaunay(reference)
    except QhullError as exc:
        raise AnalysisError("Reference points cannot form a valid 2-D triangulation.") from exc

    simplices_all = triangulation.simplices
    tree = cKDTree(reference)
    distances, _ = tree.query(reference, k=2)
    spacing = float(np.median(distances[:, 1]))
    if not np.isfinite(spacing) or spacing <= 0:
        raise AnalysisError("Could not determine a positive reference spacing.")

    ref_triangles = reference[simplices_all]
    edge_lengths = np.stack(
        (
            np.linalg.norm(ref_triangles[:, 1] - ref_triangles[:, 0], axis=1),
            np.linalg.norm(ref_triangles[:, 2] - ref_triangles[:, 1], axis=1),
            np.linalg.norm(ref_triangles[:, 0] - ref_triangles[:, 2], axis=1),
        ),
        axis=1,
    )
    first_edge = ref_triangles[:, 1] - ref_triangles[:, 0]
    second_edge = ref_triangles[:, 2] - ref_triangles[:, 0]
    area2 = first_edge[:, 0] * second_edge[:, 1] - first_edge[:, 1] * second_edge[:, 0]
    valid = (np.max(edge_lengths, axis=1) <= max_edge_factor * spacing) & (np.abs(area2) > (0.05 * spacing) ** 2)
    if valid.sum() < 1:
        raise AnalysisError("No reliable triangles remain after geometry quality checks.")

    simplices = simplices_all[valid]
    ref_triangles = reference[simplices]
    def_triangles = deformed[simplices]
    # Edge matrices use the first triangle vertex as the material origin.
    DX = np.stack((ref_triangles[:, 1] - ref_triangles[:, 0], ref_triangles[:, 2] - ref_triangles[:, 0]), axis=2)
    dx = np.stack((def_triangles[:, 1] - def_triangles[:, 0], def_triangles[:, 2] - def_triangles[:, 0]), axis=2)
    determinants = np.linalg.det(DX)
    if np.any(np.abs(determinants) <= 1e-12):
        raise AnalysisError("Degenerate reference triangle encountered after validation.")
    F = np.matmul(dx, np.linalg.inv(DX))

    hull_vertices = set(triangulation.convex_hull.ravel().tolist())
    edge_mask = np.any(np.isin(simplices, list(hull_vertices)), axis=1)
    locations = ref_triangles.mean(axis=1)
    return _strain_from_F(
        F, locations, simplices, edge_mask, np.ones(len(simplices), dtype=bool), "lattice-cst", equivalent_coefficient
    )


def compute_local_peak_pair_strain(
    lattice_indices: Iterable[Iterable[int]],
    reference_positions: Iterable[Iterable[float]],
    deformed_positions: Iterable[Iterable[float]],
    *,
    equivalent_coefficient: float = 4.0 / 9.0,
    max_condition: float = LATTICE_CONDITION_GUIDANCE["general"],
) -> StrainResult:
    """Compute local deformation gradients while retaining every accepted atom.

    Four symmetric cardinal partners remain the highest-quality Peak Pairs
    estimate (grade A).  Border/defect sites fall back to a weighted local
    affine fit using available one-step lattice neighbours (grades B/C).  A
    geometrically underdetermined site stays in the output with NaN strain and
    an explicit invalid reason; it is never silently removed.

    ``max_condition`` is the same lattice condition-number limit used by
    :func:`validate_reference_lattice`; callers must pass the *same* value to
    both so a user-raised tolerance is not silently re-rejected here (default
    30.0, see ``LATTICE_CONDITION_GUIDANCE``).

    Examples
    --------
    Uniform tensile strain ``exx = 0.05`` along x is recovered exactly by the
    symmetric grade-A path::

        indices = [[0, 0], [1, 0], [2, 0], [0, 1], [0, 2],
                   [1, 1], [1, 2], [2, 1], [2, 2]]
        ref = np.array([[10.0 * n, 8.0 * m] for n, m in indices])
        deformed = ref * np.array([1.05, 1.0])
        result = compute_local_peak_pair_strain(indices, ref, deformed)
        assert np.allclose(result.small_xx[result.quality_mask], 0.05)
    """
    condition_limit = validate_max_condition(max_condition)
    indices = np.asarray(lattice_indices, dtype=int)
    reference = _as_points(reference_positions, "reference_positions")
    deformed = _as_points(deformed_positions, "deformed_positions")
    if indices.shape != reference.shape or deformed.shape != reference.shape:
        raise AnalysisError("Lattice indices, reference positions and deformed positions must all have shape (N, 2).")
    index_lookup: dict[tuple[int, int], int] = {}
    duplicate_indices: set[int] = set()
    for i, pair in enumerate(indices):
        key = (int(pair[0]), int(pair[1]))
        if key in index_lookup:
            duplicate_indices.update((i, index_lookup[key]))
        else:
            index_lookup[key] = i
    if duplicate_indices:
        raise AnalysisError("Duplicate lattice indices found; resolve peak-to-lattice matching before local PPA.")

    gradients = np.full((len(indices), 2, 2), np.nan, dtype=np.float64)
    quality_mask = np.zeros(len(indices), dtype=bool)
    site_quality = np.full(len(indices), "invalid", dtype=object)
    invalid_reasons = np.full(len(indices), "insufficient_noncollinear_neighbors", dtype=object)

    for i, (n, m) in enumerate(indices):
        neighbor_keys = ((n - 1, m), (n + 1, m), (n, m - 1), (n, m + 1))
        if all((int(a), int(b)) in index_lookup for a, b in neighbor_keys):
            minus_a, plus_a, minus_b, plus_b = (index_lookup[(int(a), int(b))] for a, b in neighbor_keys)
            ref_a = 0.5 * (reference[plus_a] - reference[minus_a])
            ref_b = 0.5 * (reference[plus_b] - reference[minus_b])
            def_a = 0.5 * (deformed[plus_a] - deformed[minus_a])
            def_b = 0.5 * (deformed[plus_b] - deformed[minus_b])
            DX = np.column_stack((ref_a, ref_b))
            if abs(np.linalg.det(DX)) > 1e-12 and np.linalg.cond(DX) <= condition_limit:
                gradients[i] = np.column_stack((def_a, def_b)) @ np.linalg.inv(DX)
                quality_mask[i] = True
                site_quality[i] = "A-symmetric"
                invalid_reasons[i] = ""
                continue

        # One lattice step around the centre includes cardinal and diagonal
        # neighbours.  Weighting by inverse reference distance gives cardinal
        # peaks priority while preserving a stable fit at image corners.
        neighbor_ids = []
        for dn in (-1, 0, 1):
            for dm in (-1, 0, 1):
                if dn == 0 and dm == 0:
                    continue
                neighbor = index_lookup.get((int(n + dn), int(m + dm)))
                if neighbor is not None:
                    neighbor_ids.append(neighbor)
        if len(neighbor_ids) < 2:
            continue
        ref_delta = reference[neighbor_ids] - reference[i]
        def_delta = deformed[neighbor_ids] - deformed[i]
        if np.linalg.matrix_rank(ref_delta) < 2:
            continue
        distances = np.linalg.norm(ref_delta, axis=1)
        if np.any(distances <= 1e-12):
            invalid_reasons[i] = "duplicate_reference_position"
            continue
        weights = 1.0 / distances
        weighted_ref = ref_delta * np.sqrt(weights)[:, None]
        if np.linalg.cond(weighted_ref) > condition_limit:
            invalid_reasons[i] = "ill_conditioned_neighbor_geometry"
            continue
        weighted_def = def_delta * np.sqrt(weights)[:, None]
        coef, *_ = np.linalg.lstsq(weighted_ref, weighted_def, rcond=None)
        gradients[i] = coef.T
        quality_mask[i] = True
        site_quality[i] = "B-local-fit" if len(neighbor_ids) >= 3 else "C-minimal-fit"
        invalid_reasons[i] = ""

    if not quality_mask.any():
        raise AnalysisError("No site has sufficient non-collinear neighbours for local strain.")
    return _strain_from_F(
        gradients, reference.copy(), None,
        np.zeros(len(indices), dtype=bool), quality_mask,
        "peak-pairs-local-all-points-v2", equivalent_coefficient,
        site_indices=np.arange(len(indices), dtype=int),
        site_quality=site_quality,
        invalid_reasons=invalid_reasons,
    )
