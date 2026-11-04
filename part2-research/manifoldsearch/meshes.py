"""Surfaces as graphs: points in 3D joined by edges whose weight is their Euclidean length.

Design choices:
- A mesh is just `points` (N, 3) and `edges` (E, 2), each undirected edge once. Everything downstream
  works from that, so a new surface (or a real scan loaded from a file) only has to produce those two arrays.
- The grid generators match Part 1 exactly, so Part 2 results can be compared with the notebook.
- The irregular generators (random points on a sphere, a bumpy sphere) are there to ask whether the
  hierarchical search only works because the grid is tidy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import ConvexHull


@dataclass(frozen=True)
class Mesh:
    name: str
    points: np.ndarray   # (N, 3) float64
    edges: np.ndarray    # (E, 2) int64, u < v, each undirected edge once

    @property
    def n(self) -> int:
        return len(self.points)

    def csr(self) -> csr_matrix:
        """Symmetric sparse adjacency matrix, weights = Euclidean edge lengths."""
        u, v = self.edges[:, 0], self.edges[:, 1]
        w = np.linalg.norm(self.points[u] - self.points[v], axis=1)
        return csr_matrix((np.concatenate([w, w]), (np.concatenate([u, v]), np.concatenate([v, u]))), shape=(self.n, self.n))


def _unique_edges(pairs: np.ndarray) -> np.ndarray:
    pairs = np.sort(pairs, axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return np.unique(pairs, axis=0)


def grid_sphere(n_lat: int, n_lon: int) -> Mesh:
    """Unit sphere on a latitude-longitude grid (as in Part 1). Longitude wraps; the poles are not joined."""
    lat = np.linspace(0.0, np.pi, n_lat)
    lon = np.linspace(0.0, 2 * np.pi, n_lon, endpoint=False)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    pts = np.stack([np.sin(la) * np.cos(lo), np.sin(la) * np.sin(lo), np.cos(la)], axis=-1).reshape(-1, 3)
    vid = np.arange(n_lat * n_lon).reshape(n_lat, n_lon)
    pairs = [np.stack([vid[:, :], np.roll(vid, -1, axis=1)], axis=-1).reshape(-1, 2),
             np.stack([vid[:-1], vid[1:]], axis=-1).reshape(-1, 2)]
    return Mesh(f"grid_sphere_{n_lat}x{n_lon}", pts, _unique_edges(np.concatenate(pairs)))


def grid_torus(n_rows: int, n_cols: int, big_r: float = 2.0, small_r: float = 1.0) -> Mesh:
    """Torus with both directions wrapping (as in Part 1)."""
    th = np.linspace(0.0, 2 * np.pi, n_rows, endpoint=False)
    ph = np.linspace(0.0, 2 * np.pi, n_cols, endpoint=False)
    t, p = np.meshgrid(th, ph, indexing="ij")
    pts = np.stack([(big_r + small_r * np.cos(t)) * np.cos(p), (big_r + small_r * np.cos(t)) * np.sin(p), small_r * np.sin(t)], axis=-1).reshape(-1, 3)
    vid = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)
    pairs = [np.stack([vid, np.roll(vid, -1, axis=1)], axis=-1).reshape(-1, 2),
             np.stack([vid, np.roll(vid, -1, axis=0)], axis=-1).reshape(-1, 2)]
    return Mesh(f"grid_torus_{n_rows}x{n_cols}", pts, _unique_edges(np.concatenate(pairs)))


def bumpy_sphere(n_lat: int, n_lon: int, amplitude: float = 0.15, lobes: int = 5) -> Mesh:
    """The grid sphere with its radius varied by a smooth bump pattern: curvature that changes from place to place.
    No closed-form geodesic exists, so it is judged against exact graph distances."""
    base = grid_sphere(n_lat, n_lon)
    x, y, z = base.points.T
    radius = 1.0 + amplitude * np.sin(lobes * np.arctan2(y, x)) * np.sin(lobes * np.arccos(np.clip(z, -1, 1)))
    return Mesh(f"bumpy_sphere_{n_lat}x{n_lon}", base.points * radius[:, None], base.edges)


def delaunay_sphere(n: int, seed: int = 0) -> Mesh:
    """n random points on the unit sphere, triangulated by their convex hull: an irregular mesh
    (valence varies, spacing is uneven). A triangulated sphere has 3n - 6 edges."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    tri = ConvexHull(pts).simplices
    pairs = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]])
    return Mesh(f"delaunay_sphere_{n}", pts, _unique_edges(pairs))


def load_obj(path: str) -> Mesh:
    """A triangle mesh from a Wavefront .obj file (vertices and faces only)."""
    pts, pairs = [], []
    with open(path, encoding="utf8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                pts.append([float(c) for c in parts[1:4]])
            elif parts[0] == "f":
                idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                pairs += [(idx[i], idx[(i + 1) % len(idx)]) for i in range(len(idx))]
    return Mesh(path.rsplit("/", 1)[-1], np.array(pts), _unique_edges(np.array(pairs, dtype=np.int64)))
