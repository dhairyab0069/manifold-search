"""Direction 2 experiment: baselines against a learned seed scorer on held-out meshes.

    python -m manifoldsearch.ml.run_partition experiments/configs/ml_partition.yaml --seed 0 --out results/ml/partition_seed0.json
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import yaml

from .dataset import FAMILIES
from .features import build_context
from .partition import (BALANCE_WEIGHT, VERTEX_FEATURES, evaluate_partition, evolution_strategy, make_queries, seeds_fps, seeds_kmeans,
                        seeds_metis, seeds_random, seeds_spectral, seeds_weighted_fps, vertex_feature_matrix)


def k_for(n: int, c: float) -> int:
    return max(4, int(round(c * math.sqrt(n))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed = args.seed
    rng = np.random.default_rng(seed)

    # ---- training: optimise the scorer's weights on the training meshes (stretch with 0 rings + balance)
    train = []
    for fam, size in cfg["train_meshes"]:
        mesh = FAMILIES[fam](size, seed)
        ctx = build_context(mesh, 1.0, seed)
        q = make_queries(mesh, cfg["train_sources"], cfg["targets_per_source"], 5000 + seed)
        train.append((mesh, vertex_feature_matrix(ctx), q, k_for(mesh.n, 1.0)))

    def objective(theta):
        vals = []
        for i, (mesh, phi, q, k) in enumerate(train):
            seeds = seeds_weighted_fps(mesh.points, phi, theta, k, np.random.default_rng(777 + i))     # common random numbers
            vals.append(evaluate_partition(mesh, seeds, q, rings=(0,))["J"])
        return float(np.mean(vals))

    t0 = time.perf_counter()
    theta, best_v, hist = evolution_strategy(objective, len(VERTEX_FEATURES), rng, **cfg["es"])
    train_s = time.perf_counter() - t0
    out = {"seed": seed, "theta": dict(zip(VERTEX_FEATURES, map(float, theta))), "train_J_start": hist[0], "train_J_best": best_v,
           "train_history": hist, "train_s": train_s, "balance_weight": BALANCE_WEIGHT, "results": []}

    # ---- held-out evaluation
    methods = ["random", "fps", "kmeans", "spectral", "metis", "learned"]
    for fam, size in cfg["test_meshes"]:
        mesh = FAMILIES[fam](size, seed)
        q = make_queries(mesh, cfg["test_sources"], cfg["targets_per_source"], 9000 + seed)
        phi = vertex_feature_matrix(build_context(mesh, 1.0, seed))
        for c in cfg["patch_factors"]:
            k = k_for(mesh.n, c)
            for method in methods:
                reps = cfg["repeats"] if method in ("random", "fps", "kmeans", "learned") else 1
                rows = []
                for rep in range(reps):
                    r_ = np.random.default_rng(31 * seed + 7 * rep + 1)
                    try:
                        if method == "learned":
                            seeds = seeds_weighted_fps(mesh.points, phi, theta, k, r_)
                        elif method == "spectral" and mesh.n > cfg["spectral_max_n"]:
                            raise RuntimeError("skipped: too large")
                        else:
                            seeds = {"random": seeds_random, "fps": seeds_fps, "kmeans": seeds_kmeans, "spectral": seeds_spectral, "metis": seeds_metis}[method](mesh, k, r_)
                    except Exception as e:                     # missing pymetis, or too large for the dense spectral step
                        rows = None; err = str(e)[:80]; break
                    rows.append(evaluate_partition(mesh, seeds, q))
                if rows is None:
                    out["results"].append({"mesh": mesh.name, "family": fam, "size": size, "c": c, "method": method, "error": err}); continue
                agg = {key: float(np.mean([r[key] for r in rows])) for key in ("size_cv", "max_patch_frac", "work_r0", "work_r1", "J")}
                for r in (0, 1):
                    for stat in ("mean", "p95", "max"):
                        agg[f"stretch_r{r}_{stat}"] = float(np.mean([x[f"stretch_r{r}"][stat] for x in rows]))
                out["results"].append({"mesh": mesh.name, "family": fam, "size": size, "N": mesh.n, "k": rows[0]["k"], "c": c, "method": method, "repeats": reps, **agg})
        print("done", mesh.name, flush=True)
    json.dump(out, open(args.out, "w"))


if __name__ == "__main__":
    main()
