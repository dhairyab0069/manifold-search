"""Label generation: for every query, the exact distance and the corridor distance after 0..3 rings.

Per mesh we draw `n_sources` random sources and `targets_per_source` random targets each. One exact single-source
Dijkstra (scipy, float64) per source gives the exact distance to all its targets, so the cost is n_sources full
searches rather than one per query (fine up to N ~ 500k). The corridor distance for ring r uses the same code as
the deployed search (route over the patch graph, then Dijkstra in the corridor), timed per query.

Labels (computed later, evaluation.py): for a tolerance tol, label = smallest r in 0..3 with stretch_r <= tol,
else 3. Stretch is non-increasing in r because corridors are nested.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time

import numpy as np
from scipy.sparse.csgraph import dijkstra

from .. import meshes
from .features import BASE_FEATURES, MAX_RINGS, RING0_FEATURES, MeshContext, build_context, corridor_distance, query_features, ring0_features

FAMILIES = {
    "grid_sphere": lambda n, seed: meshes.grid_sphere(n, 2 * n),
    "grid_torus": lambda n, seed: meshes.grid_torus(n, 2 * n),
    "bumpy_sphere": lambda n, seed: meshes.bumpy_sphere(n, 2 * n),
    "delaunay_sphere": lambda n, seed: meshes.delaunay_sphere(2 * n * n, seed),
}
TRAIN_FAMILIES = ("grid_sphere", "grid_torus")
TRAIN_MAX_SIZE = 128


def split_of(family: str, size: int) -> str:
    """Train meshes: grid sphere and torus up to size 128. Everything else is test: unseen families (bumpy sphere,
    random-point sphere) at any size, and the training families at the larger unseen sizes."""
    return "train" if family in TRAIN_FAMILIES and size <= TRAIN_MAX_SIZE else "test"


def mesh_list(cfg: dict) -> list[tuple[str, int]]:
    out = [(f, int(n)) for f in cfg["train_families"] for n in cfg["train_sizes"]]
    out += [(f, int(n)) for f in cfg["test_families"] for n in cfg["test_sizes"][f]]
    return out


_CTX: MeshContext | None = None


def _worker(job):
    sources, targets = job                       # (m,), (m, T)
    ctx = _CTX
    dist = dijkstra(ctx.graph, directed=False, indices=sources)
    rows = []
    for i, s in enumerate(sources):
        for t in targets[i]:
            s_, t_ = int(s), int(t)
            ex = float(dist[i, t_])
            if not ex > 1e-9:
                continue
            t0 = time.perf_counter()
            x, route = query_features(ctx, s_, t_)
            tf = time.perf_counter() - t0
            d, sz, tm = [], [], []
            for r in range(MAX_RINGS + 1):
                t0 = time.perf_counter()
                dd, m = corridor_distance(ctx, s_, t_, route, r)
                tm.append(time.perf_counter() - t0)
                d.append(dd); sz.append(m)
            rl = float(ctx.pdist[ctx.index.patch_of[s_], ctx.index.patch_of[t_]])
            eu = float(np.linalg.norm(ctx.mesh.points[s_] - ctx.mesh.points[t_])) + 1e-12
            rows.append((s_, t_, ex, x, ring0_features(ctx, eu, rl, d[0], sz[0]), d, sz, tm, tf))
    return rows


def generate_mesh(family: str, size: int, seed: int, queries: int, targets_per_source: int, procs: int,
                  patches_per_sqrt: float = 1.0) -> tuple[dict, dict]:
    global _CTX
    mesh = FAMILIES[family](size, seed)
    t0 = time.perf_counter()
    ctx = build_context(mesh, patches_per_sqrt, seed)
    t_pre = time.perf_counter() - t0
    _CTX = ctx
    rng = np.random.default_rng(1000 + seed)
    n_src = max(1, queries // targets_per_source)
    src = rng.integers(0, mesh.n, n_src)
    tgt = rng.integers(0, mesh.n, (n_src, targets_per_source))
    chunk = max(1, min(8, n_src // (4 * procs) or 1))
    jobs = [(src[i:i + chunk], tgt[i:i + chunk]) for i in range(0, n_src, chunk)]
    t0 = time.perf_counter()
    if procs > 1:
        with mp.get_context("fork").Pool(procs) as pool:
            res = [r for part in pool.imap(_worker, jobs) for r in part]
    else:
        res = [r for j in jobs for r in _worker(j)]
    t_gen = time.perf_counter() - t0
    data = {
        "s": np.array([r[0] for r in res]), "t": np.array([r[1] for r in res]), "exact": np.array([r[2] for r in res]),
        "X": np.stack([r[3] for r in res]), "X0": np.stack([r[4] for r in res]),
        "d": np.array([r[5] for r in res]), "size": np.array([r[6] for r in res]),
        "time": np.array([r[7] for r in res]), "tfeat": np.array([r[8] for r in res]),
    }
    meta = {"family": family, "size": size, "seed": seed, "mesh": mesh.name, "N": mesh.n, "k": ctx.k, "split": split_of(family, size),
            "queries": len(res), "preprocess_s": t_pre, "generate_s": t_gen, "procs": procs, "patches_per_sqrt": patches_per_sqrt}
    _CTX = None
    return data, meta


def save_mesh(outdir: str, data: dict, meta: dict) -> str:
    os.makedirs(outdir, exist_ok=True)
    stem = f"{meta['family']}_{meta['size']}_seed{meta['seed']}"
    np.savez_compressed(os.path.join(outdir, stem + ".npz"), **data)
    with open(os.path.join(outdir, stem + ".json"), "w") as f:
        json.dump(meta, f)
    return stem


def load_all(outdir: str, seed: int) -> list[tuple[dict, dict]]:
    out = []
    for fn in sorted(os.listdir(outdir)):
        if fn.endswith(f"_seed{seed}.json"):
            meta = json.load(open(os.path.join(outdir, fn)))
            z = np.load(os.path.join(outdir, fn[:-5] + ".npz"))
            out.append(({k: z[k] for k in z.files}, meta))
    return out


def labels(d: np.ndarray, exact: np.ndarray, tol: float) -> np.ndarray:
    """Smallest ring count whose stretch is within tol (3 if none is). d is (n, rings+1)."""
    ok = d / exact[:, None] <= tol + 1e-12
    first = np.where(ok.any(axis=1), ok.argmax(axis=1), d.shape[1] - 1)
    return first.astype(np.int64)
