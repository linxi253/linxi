"""Reciprocal local lattice topology, independent of absolute displacement.

No global affine fit or detrending is performed: each edge is measured against
the supplied reference basis. Ambiguous partners and inconsistent loops are
reported rather than silently changing or discarding accepted atoms.
"""
from collections import deque

import numpy as np
from scipy.spatial import cKDTree

OFFSETS = np.array([(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)], dtype=int)
REVERSE = np.array([1, 0, 3, 2, 7, 6, 5, 4])


def local_lattice_topology(points, origin, basis, *, tolerance=.4):
    """Return integer labels, reciprocal partners and unresolved-site mask.

    Partners are sought in normalized lattice coordinates within 0.4 steps of
    a nominal one-step edge. A 0.05-step nearest/runner-up ambiguity gap avoids
    assigning two nearly coincident accepted atoms the same physical identity.
    Disconnected components are anchored separately; their relative integer
    translation remains conditional on the supplied reference origin.
    """
    coords = (np.asarray(points, float)-origin) @ np.linalg.inv(basis).T
    count = len(coords)
    labels = np.rint(coords).astype(np.int64)
    partners = np.full((count, len(OFFSETS)), -1, dtype=int)
    unresolved = np.zeros(count, dtype=bool)
    if count < 2:
        return labels, partners, unresolved
    tree = cKDTree(coords)
    for direction, offset in enumerate(OFFSETS):
        distances, ids = tree.query(coords+offset, k=2, distance_upper_bound=tolerance)
        good = np.isfinite(distances[:, 0]) & (ids[:, 0] != np.arange(count))
        gap = np.full(count, np.inf)
        both = np.isfinite(distances).all(axis=1)
        gap[both] = distances[both, 1]-distances[both, 0]
        ambiguous = good & (gap < .05)
        unresolved |= ambiguous
        good &= ~ambiguous
        partners[good, direction] = ids[good, 0]
    for direction, reverse in enumerate(REVERSE):
        rows = np.flatnonzero(partners[:, direction] >= 0)
        ids = partners[rows, direction]
        partners[rows[partners[ids, reverse] != rows], direction] = -1

    visited = np.zeros(count, dtype=bool)
    # A component's closest reference-origin point supplies its integer anchor.
    order = np.argsort(np.linalg.norm(coords, axis=1), kind='stable')
    for anchor in order:
        if visited[anchor]:
            continue
        component, pending, inconsistent = [], deque([int(anchor)]), False
        visited[anchor] = True
        while pending:
            i = pending.popleft()
            component.append(i)
            for direction, j in enumerate(partners[i]):
                if j < 0:
                    continue
                candidate = labels[i]+OFFSETS[direction]
                if visited[j]:
                    inconsistent |= not np.array_equal(labels[j], candidate)
                else:
                    labels[j] = candidate
                    visited[j] = True
                    pending.append(int(j))
        if inconsistent or len(np.unique(labels[component], axis=0)) != len(component):
            unresolved[component] = True
            labels[component] = np.rint(coords[component]).astype(np.int64)
    partners[unresolved] = -1
    for direction in range(len(OFFSETS)):
        rows = np.flatnonzero(partners[:, direction] >= 0)
        partners[rows[unresolved[partners[rows, direction]]], direction] = -1
    return labels, partners, unresolved
