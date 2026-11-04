"""Direction 3 experiment: learned A* heuristics against straight-line A*, ALT and Dijkstra on held-out meshes.

    python -m manifoldsearch.ml.run_heuristic experiments/configs/ml_heuristic.yaml --seed 0 --out results/ml/heuristic_seed0.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import numpy as np
import torch
import yaml
from scipy.sparse.csgraph import dijkstra

from .dataset import FAMILIES
from .geometry import vertex_geometry
from .models import default_device
from .heuristic import HeuristicNet, MeshGeo, alt_bounds, astar_with_h, landmarks_graph_fps, pair_features

_STATE: dict = {}


def build_geo(mesh, n_landmarks, rng):
    g = mesh.csr()
    ids, D = landmarks_graph_fps(g, n_landmarks, rng)
    return g, MeshGeo(mesh, vertex_geometry(mesh.points, g), D)


def training_data(cfg, seed, kind):
    rng = np.random.default_rng(seed)
    xs, es, ls, us, ys, diags = [], [], [], [], [], []
    for fam, size in cfg["train_meshes"]:
        mesh = FAMILIES[fam](size, seed)
        graph, geo = build_geo(mesh, cfg["landmarks"], rng)
        for t in rng.integers(0, mesh.n, cfg["train_targets"]):
            d = dijkstra(graph, directed=False, indices=int(t))
            vs = rng.integers(0, mesh.n, cfg["train_vertices_per_target"])
            keep = d[vs] > 1e-9
            vs = vs[keep]
            x, e, lb, ub = pair_features(geo, int(t), vs, kind)
            xs.append(x); es.append(e); ls.append(lb); us.append(ub); ys.append(d[vs]); diags.append(np.full(len(vs), geo.diag))
    return [np.concatenate(a) for a in (xs, es, ls, us, ys, diags)]


def method_h(geo8, geo16, nets, name, s, t, euclid):
    kind, _, arg = name.partition(":")
    if kind == "dijkstra":
        return np.zeros(len(euclid))
    if kind == "euclid":
        return euclid * float(arg or 1)
    lb8 = np.maximum(alt_bounds(geo8.D, t)[0], euclid)
    if kind == "alt8":
        return lb8 * float(arg or 1)
    if kind == "alt16":
        return np.maximum(alt_bounds(geo16.D, t)[0], euclid)
    return nets[name].predict(geo8, t)


def _run_query(job):
    torch.set_num_threads(1)
    mi, s, t = job
    st = _STATE[mi]
    graph, geo8, geo16, pts = st["graph"], st["geo8"], st["geo16"], st["points"]
    dall = dijkstra(graph, directed=False, indices=t)
    exact = float(dall[s])
    euclid = np.linalg.norm(pts - pts[t], axis=1)
    rows = {}
    for name in st["methods"]:
        h = method_h(geo8, geo16, st["nets"], name, s, t, euclid)
        cost, expanded = astar_with_h(graph, s, t, h)
        pos = dall > 1e-9
        rel_over = (h[pos] - dall[pos]) / dall[pos]
        rows[name] = (cost / exact, expanded, float((rel_over > 1e-6).mean()), float(rel_over.max()))
    return mi, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=16)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed = args.seed
    dev = default_device()
    nets, fit_s = {}, {}
    for kind in ("geo", "lm"):
        x, e, lb, ub, y, dg = training_data(cfg, seed, kind)
        for ow in cfg["over_weights"]:
            t0 = time.perf_counter()
            nets[f"learned_{kind}:{ow}"] = HeuristicNet(kind, ow, seed).fit(x, e, lb, ub, y, float(dg[0]), epochs=cfg["epochs"], device=dev)
            fit_s[f"learned_{kind}:{ow}"] = time.perf_counter() - t0
    methods = ["dijkstra", "euclid"] + [f"euclid:{w}" for w in cfg["weights"]] + ["alt8", "alt16"] + [f"alt8:{w}" for w in cfg["weights"]] + list(nets)
    rng = np.random.default_rng(10_000 + seed)
    jobs, meta = [], []
    for mi, (fam, size) in enumerate(cfg["test_meshes"]):
        mesh = FAMILIES[fam](size, seed)
        graph, geo16 = build_geo(mesh, 16, rng)
        geo8 = MeshGeo(mesh, {"kn": np.exp(geo16.logkn) - 1e-3}, geo16.D[:8])      # first 8 of the 16 farthest-point landmarks
        _STATE[mi] = {"graph": graph, "geo8": geo8, "geo16": geo16, "points": mesh.points, "nets": nets, "methods": methods}
        q = rng.integers(0, mesh.n, (cfg["queries"], 2))
        jobs += [(mi, int(s), int(t)) for s, t in q if s != t]
        meta.append({"mesh": mesh.name, "family": fam, "size": size, "N": mesh.n})
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(args.procs) as pool:
        res = pool.map(_run_query, jobs, chunksize=4)
    out = {"seed": seed, "fit_s": fit_s, "run_s": time.perf_counter() - t0, "over_weights": cfg["over_weights"], "results": []}
    for mi, m in enumerate(meta):
        rows = [r for i, r in res if i == mi]
        for name in methods:
            a = np.array([r[name] for r in rows])
            out["results"].append({**m, "method": name, "queries": len(rows), "expanded_frac": float((a[:, 1] / m["N"]).mean()),
                                   "subopt_mean": float(a[:, 0].mean()), "subopt_max": float(a[:, 0].max()),
                                   "frac_suboptimal": float((a[:, 0] > 1 + 1e-9).mean()), "over_frac": float(a[:, 2].mean()), "over_max_rel": float(a[:, 3].max())})
    json.dump(out, open(args.out, "w"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
