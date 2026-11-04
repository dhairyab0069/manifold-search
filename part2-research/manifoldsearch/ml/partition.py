"""Direction 2: choosing the patch seeds.

Every partitioner here is reduced to a choice of k seed vertices; patches are then the graph Voronoi cells of those
seeds (so every patch is connected and the existing index code applies unchanged). Partitioners that produce
clusters (k-means, spectral, METIS) hand over the vertex nearest to each cluster's centre as the seed.

Objective of a partition (lower is better), for a fixed patch count k and a fixed set of validation queries:
    J = (mean stretch with 0 rings - 1) + balance_weight * CV(patch sizes)
CV is the coefficient of variation of the patch sizes: unbalanced patches make big corridors (more work) and tiny
patches (more route hops). Work (corridor fraction) is always reported next to J, so a partition that buys accuracy
by searching more is visible.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

from ..hierarchical import farthest_point_seeds
from ..meshes import Mesh
from .features import MAX_RINGS, build_context, corridor_distance, route_patches

VERTEX_FEATURES = ["log_kn", "log_sv", "log_edge", "log_aspect", "degree"]
BALANCE_WEIGHT = 0.05


# ---------------------------------------------------------------- baselines
def _snap(points: np.ndarray, centres: np.ndarray, k: int, rng) -> np.ndarray:
    """Nearest vertex to each centre; duplicates are replaced by random unused vertices."""
    _, idx = cKDTree(points).query(centres)
    seeds = list(dict.fromkeys(int(i) for i in idx))
    while len(seeds) < k:
        c = int(rng.integers(len(points)))
        if c not in seeds:
            seeds.append(c)
    return np.array(seeds[:k])


def seeds_random(mesh: Mesh, k: int, rng) -> np.ndarray:
    return rng.choice(mesh.n, size=k, replace=False)


def seeds_fps(mesh: Mesh, k: int, rng) -> np.ndarray:
    return farthest_point_seeds(mesh.points, k, rng)


def seeds_kmeans(mesh: Mesh, k: int, rng) -> np.ndarray:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=1, max_iter=40, random_state=int(rng.integers(1 << 30))).fit(mesh.points)
    return _snap(mesh.points, km.cluster_centers_, k, rng)


def seeds_spectral(mesh: Mesh, k: int, rng, dims: int = 8) -> np.ndarray:
    """Spectral embedding by the smallest eigenvectors of the (unweighted) normalised graph Laplacian, then k-means
    in the embedding; the seed of a cluster is the vertex nearest its centre in 3D space."""
    from sklearn.cluster import KMeans
    g = mesh.csr()
    a = csr_matrix((np.ones(len(g.indices)), g.indices, g.indptr), shape=g.shape)
    deg = np.asarray(a.sum(1)).ravel()
    dinv = diags(1.0 / np.sqrt(deg))
    m = dinv @ a @ dinv                                     # eigenvalues in [-1, 1]; the Laplacian's smallest are m's largest
    vals, vecs = eigsh(m, k=dims + 1, which="LA", tol=1e-3, v0=np.ones(mesh.n) / np.sqrt(mesh.n))
    emb = vecs[:, np.argsort(-vals)[1:dims + 1]]      # drop the trivial top eigenvector
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    km = KMeans(n_clusters=k, n_init=1, max_iter=30, random_state=int(rng.integers(1 << 30))).fit(emb)
    centres = np.stack([mesh.points[km.labels_ == c].mean(0) if (km.labels_ == c).any() else mesh.points[int(rng.integers(mesh.n))] for c in range(k)])
    return _snap(mesh.points, centres, k, rng)


def seeds_metis(mesh: Mesh, k: int, rng) -> np.ndarray:
    import pymetis
    g = mesh.csr()
    adj = [g.indices[g.indptr[i]:g.indptr[i + 1]].tolist() for i in range(mesh.n)]
    _, parts = pymetis.part_graph(k, adjacency=adj)
    parts = np.asarray(parts)
    centres = np.stack([mesh.points[parts == c].mean(0) if (parts == c).any() else mesh.points[int(rng.integers(mesh.n))] for c in range(k)])
    return _snap(mesh.points, centres, k, rng)


# ---------------------------------------------------------------- learned: weighted farthest-point seeds
def vertex_feature_matrix(ctx) -> np.ndarray:
    """Per-vertex geometry features, z-scored within the mesh so the scorer sees no absolute scale."""
    v = ctx.vgeo
    edge = np.asarray(ctx.graph.sum(axis=1)).ravel() / np.maximum(np.diff(ctx.graph.indptr), 1)
    cols = [np.log(v["kn"] + 1e-3), np.log(v["sv"] + 1e-6), np.log(edge + 1e-9), np.log(v["aspect"] + 1e-3), v["degree"]]
    f = np.stack(cols, axis=1)
    return (f - f.mean(0)) / (f.std(0) + 1e-9)


def seeds_weighted_fps(points: np.ndarray, phi: np.ndarray, theta: np.ndarray, k: int, rng) -> np.ndarray:
    """Farthest-point sampling where each vertex's distance to the chosen seeds is multiplied by w_v = exp(theta . phi_v).
    theta = 0 is plain farthest-point sampling; a positive weight on curvature puts more seeds where it is high."""
    w = np.exp(np.clip(phi @ theta, -3.0, 3.0))
    seeds = [int(rng.integers(len(points)))]
    nearest = np.linalg.norm(points - points[seeds[0]], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(nearest * w))
        seeds.append(nxt)
        nearest = np.minimum(nearest, np.linalg.norm(points - points[nxt], axis=1))
    return np.array(seeds)


# ---------------------------------------------------------------- objective
def make_queries(mesh: Mesh, n_sources: int, targets_per_source: int, seed: int) -> dict:
    graph = mesh.csr()
    rng = np.random.default_rng(seed)
    src = rng.integers(0, mesh.n, n_sources)
    tgt = rng.integers(0, mesh.n, (n_sources, targets_per_source))
    dist = dijkstra(graph, directed=False, indices=src)
    s = np.repeat(src, targets_per_source)
    t = tgt.ravel()
    ex = dist[np.repeat(np.arange(n_sources), targets_per_source), t]
    keep = ex > 1e-9
    return {"s": s[keep], "t": t[keep], "exact": ex[keep]}


def evaluate_partition(mesh: Mesh, seeds: np.ndarray, queries: dict, rings=(0, 1)) -> dict:
    ctx = build_context(mesh, seeds=np.asarray(seeds))
    po = ctx.index.patch_of
    sizes = np.bincount(po, minlength=ctx.k).astype(float)
    out = {"k": ctx.k, "size_cv": float(sizes.std() / sizes.mean()), "max_patch_frac": float(sizes.max() / mesh.n)}
    for r in rings:
        st, wk = [], []
        for s, t, ex in zip(queries["s"], queries["t"], queries["exact"]):
            route = route_patches(ctx, int(po[s]), int(po[t]))
            d, m = corridor_distance(ctx, int(s), int(t), route, r)
            st.append(d / ex); wk.append(m / mesh.n)
        st = np.array(st)
        out[f"stretch_r{r}"] = {"mean": float(st.mean()), "p95": float(np.percentile(st, 95)), "max": float(st.max())}
        out[f"work_r{r}"] = float(np.mean(wk))
    out["J"] = out["stretch_r0"]["mean"] - 1.0 + BALANCE_WEIGHT * out["size_cv"]
    return out


# ---------------------------------------------------------------- black-box optimiser
def evolution_strategy(f, dim: int, rng, pop: int = 16, elite: int = 4, gens: int = 25, sigma: float = 0.6, decay: float = 0.9):
    """A plain (mu, lambda) evolution strategy with isotropic noise, started at theta = 0 (= farthest-point sampling).
    f is minimised. Returns (best theta, best value, history of the best value per generation)."""
    mean = np.zeros(dim)
    best, best_v, hist = mean.copy(), f(mean), []
    hist.append(best_v)
    for _ in range(gens):
        cand = mean + sigma * rng.standard_normal((pop, dim))
        vals = np.array([f(c) for c in cand])
        order = np.argsort(vals)
        mean = cand[order[:elite]].mean(0)
        if vals[order[0]] < best_v:
            best, best_v = cand[order[0]].copy(), float(vals[order[0]])
        hist.append(best_v)
        sigma *= decay
    return best, best_v, hist
