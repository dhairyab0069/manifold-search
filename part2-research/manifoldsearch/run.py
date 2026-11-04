"""Run one experiment from a YAML file and write one JSON line per measurement.

    python -m manifoldsearch.run experiments/configs/smoke.yaml --out results/smoke.jsonl

A config says which mesh families and sizes to build, how many patches to use (k = patches_per_sqrt * sqrt(N)),
how many random queries to average, and whether to take exact distances from the GPU batch solver.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import yaml

from . import meshes
from .baselines import astar, dijkstra, dijkstra_expanded
from .gpu import batched_sssp, best_device
from .hierarchical import build_index
from .metrics import sphere_geodesic, stretch_summary

FAMILIES = {
    "grid_sphere": lambda n, seed: meshes.grid_sphere(n, 2 * n),
    "grid_torus": lambda n, seed: meshes.grid_torus(n, 2 * n),
    "bumpy_sphere": lambda n, seed: meshes.bumpy_sphere(n, 2 * n),
    "delaunay_sphere": lambda n, seed: meshes.delaunay_sphere(2 * n * n, seed),   # about as many vertices as the grid
}


def measure(mesh, cfg: dict, seed: int, device: str) -> dict:
    graph, pts, n = mesh.csr(), mesh.points, mesh.n
    k = max(4, int(round(cfg.get("patches_per_sqrt", 1.0) * math.sqrt(n))))
    t0 = time.perf_counter()
    index = build_index(graph, pts, k, seed)
    t_pre = time.perf_counter() - t0

    rng = np.random.default_rng(seed + 1)
    q = rng.integers(0, n, size=(int(cfg.get("queries", 30)), 2))
    q = q[q[:, 0] != q[:, 1]]
    if cfg.get("gpu_truth"):
        exact = batched_sssp(graph, q[:, 0], device=device)[np.arange(len(q)), q[:, 1]]
    else:
        exact = np.array([dijkstra(graph, int(s), int(t)) for s, t in q])

    # the grid sphere has a whole row of coincident vertices at each pole (as in Part 1): distance 0, stretch undefined
    keep = exact > 1e-9
    q, exact = q[keep], exact[keep]
    t0 = time.perf_counter()
    res = [index.query(int(s), int(t), cfg.get("corridor_rings", 0)) for s, t in q]
    t_hier = (time.perf_counter() - t0) / len(q)
    approx = np.array([r[0] for r in res])
    row = {"mesh": mesh.name, "N": n, "k": k, "queries": len(q), "preprocess_s": t_pre, "hier_query_s": t_hier,
           "corridor_fraction": float(np.mean([r[1] for r in res]) / n), "stretch": stretch_summary(approx, exact)}
    if cfg.get("effort", False):                           # vertices expanded by Dijkstra and A* (slow in pure Python: small meshes only)
        row["dijkstra_expanded"] = float(np.mean([dijkstra_expanded(graph, int(s), int(t))[1] for s, t in q]))
        row["astar_expanded"] = float(np.mean([astar(graph, pts, int(s), int(t))[1] for s, t in q]))
    if mesh.name.startswith("grid_sphere") or mesh.name.startswith("delaunay_sphere"):
        geo = sphere_geodesic(pts[q[:, 0]], pts[q[:, 1]])
        row["exact_vs_geodesic"] = stretch_summary(exact, geo)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None, help="cpu, cuda or mps (default: the best available)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf8"))
    device = args.device or best_device()
    out = open(args.out, "a", encoding="utf8") if args.out else None
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for family in cfg["families"]:
        for n in cfg["sizes"]:
            row = measure(FAMILIES[family](n, cfg.get("seed", 0)), cfg, cfg.get("seed", 0), device)
            row["device"] = device
            line = json.dumps(row)
            print(line, flush=True)
            if out:
                out.write(line + "\n"); out.flush()


if __name__ == "__main__":
    main()
