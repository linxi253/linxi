"""Shared display interpolation: pixel centers and measured support only."""
import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree
from scipy.interpolate import LinearNDInterpolator


def interpolate_fields(locations, fields, extent, grid_size=200, basis=None):
    """Linear interpolation never bridges invalid sites or wide unmeasured gaps.

    extent is (left, right, top, bottom) in display coordinates. All sites,
    including NaN sites, define the triangulation; no values are extrapolated.
    """
    if isinstance(grid_size, bool) or not isinstance(grid_size, (int, np.integer)) or not 2 <= grid_size <= 1000:
        raise ValueError('插值网格大小必须为 2–1000 的整数。')
    locations = np.asarray(locations, dtype=float)
    if locations.ndim != 2 or locations.shape[1] != 2 or not np.isfinite(locations).all():
        raise ValueError('插值位置必须为有限的 N×2 坐标。')
    left, right, top, bottom = extent
    if not np.isfinite(extent).all() or right <= left or bottom <= top:
        raise ValueError('插值范围无效。')
    gx = left + (np.arange(grid_size)+.5)*(right-left)/grid_size
    gy = top + (np.arange(grid_size)+.5)*(bottom-top)/grid_size
    xx, yy = np.meshgrid(gx, gy)
    query = np.column_stack((xx.ravel(), yy.ravel()))
    normalized = locations
    if basis is not None:
        basis = np.asarray(basis, dtype=float)
        if basis.shape != (2, 2) or not np.isfinite(basis).all() or np.linalg.cond(basis) > 1e6:
            raise ValueError('插值参考基矢无效。')
        inv = np.linalg.inv(basis).T
        normalized, query = locations @ inv, query @ inv
    grids = {key: np.full((grid_size, grid_size), np.nan) for key in fields}
    values_map = {key: np.asarray(value, dtype=float) for key, value in fields.items()}
    if any(value.shape != (len(locations),) for value in values_map.values()):
        raise ValueError('插值字段长度与位置数量不一致。')
    if len(locations) < 3:
        return grids
    try:
        triangulation = Delaunay(normalized)
    except QhullError:
        return grids
    triangles = normalized[triangulation.simplices]
    edges = np.max([np.linalg.norm(triangles[:, (k+1)%3]-triangles[:, k], axis=1) for k in range(3)], axis=0)
    spacing = np.median(cKDTree(normalized).query(normalized, k=2)[0][:, 1])
    support = edges <= 2.5*spacing
    containing = triangulation.find_simplex(query)
    for key, values in values_map.items():
        good = support & np.isfinite(values[triangulation.simplices]).all(axis=1)
        inside = containing >= 0
        inside[inside] &= good[containing[inside]]
        # Barycentric interpolation is nonnegative for nonnegative quantities.
        grids[key].ravel()[inside] = LinearNDInterpolator(triangulation, values)(query[inside])
    return grids
