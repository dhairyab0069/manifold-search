"""Per-mesh preprocessing context, the corridor search, and the per-query features.

Everything a query-time policy may use must be cheap: it is read off the patch graph (a few hundred nodes) and
off arrays that were computed once per mesh. The expensive things (exact distances, corridor distances) are only
used as labels, in dataset.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from ..hierarchical import HierarchicalIndex, build_index
from ..meshes import Mesh
from .geometry import border_distance, vertex_geometry

BASE_FEATURES = [
    "log_euclid", "euclid_over_diag", "euclid_over_patch_radius", "route_len_over_euclid", "route_len_over_diag",
    "n_route_patches", "n_route_over_sqrt_k",
    "rad_min", "rad_max", "rad_mean", "rad_min_over_euclid", "rad_mean_over_euclid",
    "pcurv_mean", "pcurv_max", "pcurv_min", "pcurv_peak",
    "route_dev_max", "route_dev_mean", "frac_ring1", "frac_ring2", "psize_mean",
    "kn_s", "kn_t", "kn_max", "kn_min", "sv_s", "sv_t", "aspect_s", "aspect_t",
    "border_s", "border_t", "border_min", "seed_s", "seed_t", "same_patch",
    "log_n", "log_k", "edge_over_diag", "mean_curv_mesh",
]
RING0_FEATURES = ["d0_over_euclid", "d0_over_route", "frac_ring0"]
MAX_RINGS = 3


@dataclass
class MeshContext:
    mesh: Mesh
    graph: csr_matrix
    index: HierarchicalIndex
    k: int
    vgeo: dict
    border: np.ndarray        # (N,) distance to the nearest patch border
    patch_size: np.ndarray    # (k,) vertices per patch
    patch_rad: np.ndarray     # (k,) max graph distance from the seed within the patch
    patch_curv_mean: np.ndarray
    patch_curv_max: np.ndarray
    pdist: np.ndarray         # (k, k) patch-graph distances
    ppred: np.ndarray         # (k, k) predecessors
    adj: csr_matrix
    diag: float
    mean_edge: float
    mean_curv: float
    scratch: np.ndarray       # (N,) int64 workspace for subgraph remapping


def build_context(mesh: Mesh, patches_per_sqrt: float = 1.0, seed: int = 0) -> MeshContext:
    graph, pts, n = mesh.csr(), mesh.points, mesh.n
    k = max(4, int(round(patches_per_sqrt * math.sqrt(n))))
    index = build_index(graph, pts, k, seed)
    k = index.k
    vgeo = vertex_geometry(pts, graph)
    po = index.patch_of
    size = np.bincount(po, minlength=k).astype(float)
    rad = np.zeros(k)
    np.maximum.at(rad, po, index.seed_dist)
    kn = vgeo["kn"]
    cmean = np.bincount(po, weights=kn, minlength=k) / np.maximum(size, 1)
    cmax = np.zeros(k)
    np.maximum.at(cmax, po, kn)
    pdist, ppred = dijkstra(index.patch_graph, directed=False, return_predecessors=True)
    diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    return MeshContext(mesh, graph, index, k, vgeo, border_distance(graph, po), size, rad, cmean, cmax, pdist, ppred,
                       index.patch_adj, diag, float(graph.data.mean()), float(kn.mean()), np.full(n, -1, dtype=np.int64))


def route_patches(ctx: MeshContext, ps: int, pt: int) -> list[int]:
    route, cur = [pt], pt
    while cur != ps:
        cur = int(ctx.ppred[ps, cur])
        if cur < 0:
            break
        route.append(cur)
    return route


def patch_mask(ctx: MeshContext, route: list[int], rings: int) -> np.ndarray:
    m = np.zeros(ctx.k, dtype=bool)
    m[route] = True
    for _ in range(rings):
        m |= (ctx.adj @ m.astype(np.int8)) > 0
    return m


def corridor_distance(ctx: MeshContext, s: int, t: int, route: list[int], rings: int) -> tuple[float, int]:
    """Same answer as HierarchicalIndex.query, with a faster subgraph extraction. Returns (distance, vertices searched)."""
    ids = np.flatnonzero(patch_mask(ctx, route, rings)[ctx.index.patch_of])
    m = len(ids)
    remap = ctx.scratch
    remap[ids] = np.arange(m)
    rows = ctx.graph[ids]
    cols = remap[rows.indices]
    keep = cols >= 0
    rid = np.repeat(np.arange(m), np.diff(rows.indptr))
    sub = csr_matrix((rows.data[keep], (rid[keep], cols[keep])), shape=(m, m))
    d = dijkstra(sub, directed=False, indices=int(remap[s]))[int(remap[t])]
    remap[ids] = -1
    return float(d), m


def query_features(ctx: MeshContext, s: int, t: int) -> tuple[np.ndarray, list[int]]:
    """Features of one query (base set) and its patch route."""
    pts, po = ctx.mesh.points, ctx.index.patch_of
    ps, pt = int(po[s]), int(po[t])
    route = route_patches(ctx, ps, pt)
    r = np.asarray(route)
    euclid = float(np.linalg.norm(pts[s] - pts[t])) + 1e-12
    route_len = float(ctx.pdist[ps, pt])
    rad = ctx.patch_rad[r]
    cm, cx = ctx.patch_curv_mean[r], ctx.patch_curv_max[r]
    seeds = pts[ctx.index.seeds[r]]
    line = pts[t] - pts[s]
    u = line / (np.linalg.norm(line) + 1e-12)
    rel = seeds - pts[s]
    dev = np.linalg.norm(rel - (rel @ u)[:, None] * u[None], axis=1) / euclid
    kn, sv, asp, brd = ctx.vgeo["kn"], ctx.vgeo["sv"], ctx.vgeo["aspect"], ctx.border
    rs, rt = max(ctx.patch_rad[ps], 1e-12), max(ctx.patch_rad[pt], 1e-12)
    mask1, mask2 = patch_mask(ctx, route, 1), patch_mask(ctx, route, 2)
    n = ctx.mesh.n
    f = [
        math.log(euclid), euclid / ctx.diag, euclid / max(rad.mean(), 1e-12), route_len / euclid, route_len / ctx.diag,
        len(route), len(route) / math.sqrt(ctx.k),
        rad.min(), rad.max(), rad.mean(), rad.min() / euclid, rad.mean() / euclid,
        cm.mean(), cm.max(), cm.min(), cx.max(),
        dev.max(), dev.mean(), ctx.patch_size[mask1].sum() / n, ctx.patch_size[mask2].sum() / n, ctx.patch_size[r].mean() / n,
        kn[s], kn[t], max(kn[s], kn[t]), min(kn[s], kn[t]), sv[s], sv[t], asp[s], asp[t],
        brd[s] / rs, brd[t] / rt, min(brd[s] / rs, brd[t] / rt), ctx.index.seed_dist[s] / rs, ctx.index.seed_dist[t] / rt,
        float(ps == pt),
        math.log(n), math.log(ctx.k), ctx.mean_edge / ctx.diag, ctx.mean_curv,
    ]
    return np.nan_to_num(np.asarray(f, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6), route


def ring0_features(ctx: MeshContext, euclid: float, route_len: float, d0: float, size0: int) -> np.ndarray:
    return np.array([d0 / euclid, d0 / max(route_len, 1e-12), size0 / ctx.mesh.n])
