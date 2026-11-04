"""Cheap local geometry for every vertex, computed from points and edges only (no faces needed).

- `kn`: normal-variation curvature. Normals come from a PCA of each vertex's 1-ring; the curvature proxy at a
  vertex is the mean over its edges of (angle between the two end normals) / (edge length). On a smooth surface
  this approximates the normal curvature whatever the mesh resolution (about 1 on the unit sphere).
- `sv`: surface variation, lambda_min / sum(lambda), from the same PCA (resolution dependent, scale free).
- `aspect`: ratio of the two larger eigenvalues, a measure of how anisotropic the local sampling is.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.csgraph import dijkstra


def vertex_geometry(points: np.ndarray, graph: csr_matrix) -> dict[str, np.ndarray]:
    n = len(points)
    pattern = csr_matrix((np.ones(len(graph.indices)), graph.indices, graph.indptr), shape=graph.shape)
    a1 = pattern + identity(n, format="csr")
    deg = np.asarray(a1.sum(axis=1)).ravel()
    xx = (points[:, :, None] * points[:, None, :]).reshape(n, 9)
    mu = (a1 @ points) / deg[:, None]
    cov = (a1 @ xx) / deg[:, None] - 0.0
    cov = cov.reshape(n, 3, 3) - mu[:, :, None] * mu[:, None, :]
    cov = 0.5 * (cov + cov.transpose(0, 2, 1))
    w, v = np.linalg.eigh(cov)                            # ascending eigenvalues
    w = np.maximum(w, 0.0)
    normal = v[:, :, 0]
    sv = w[:, 0] / np.maximum(w.sum(axis=1), 1e-30)
    aspect = w[:, 1] / np.maximum(w[:, 2], 1e-30)
    coo = graph.tocoo()
    length = np.linalg.norm(points[coo.row] - points[coo.col], axis=1)
    ok = length > 1e-12
    row, col, length = coo.row[ok], coo.col[ok], length[ok]
    ang = np.arccos(np.clip(np.abs(np.sum(normal[row] * normal[col], axis=1)), 0.0, 1.0))
    num = np.bincount(row, weights=ang / length, minlength=n)
    cnt = np.bincount(row, minlength=n)
    kn = num / np.maximum(cnt, 1)
    return {"kn": kn, "sv": sv, "aspect": aspect, "degree": np.diff(graph.indptr).astype(float)}


def border_distance(graph: csr_matrix, patch_of: np.ndarray) -> np.ndarray:
    """Graph distance from every vertex to the nearest vertex that has a neighbour in another patch."""
    coo = graph.tocoo()
    cross = patch_of[coo.row] != patch_of[coo.col]
    border = np.unique(coo.row[cross])
    if len(border) == 0:
        return np.full(graph.shape[0], np.inf)
    return dijkstra(graph, directed=False, indices=border, min_only=True)
