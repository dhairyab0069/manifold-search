"""Aggregate run_partition JSONs (one per seed) into a small summary and markdown tables.

    python -m manifoldsearch.ml.report_partition results/ml/partition_seed*.json --out results_summary
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

METHODS = ["random", "fps", "kmeans", "spectral", "metis", "learned"]
KEYS = ["stretch_r0_mean", "stretch_r0_p95", "stretch_r1_mean", "work_r0", "work_r1", "size_cv", "max_patch_frac", "J"]


def cell(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in KEYS}


def summarise(runs: list[dict]) -> dict:
    out = {"seeds": [r["seed"] for r in runs], "theta": {str(r["seed"]): r["theta"] for r in runs},
           "train_J": {str(r["seed"]): [r["train_J_start"], r["train_J_best"]] for r in runs}, "pooled": {}, "by_family": {}, "paired": {}}
    for scope, filt in (("c1_all", lambda r: r["c"] == 1.0), ("c0.5_all", lambda r: r["c"] == 0.5), ("c2_all", lambda r: r["c"] == 2.0),
                        ("c1_bumpy", lambda r: r["c"] == 1.0 and r["family"] == "bumpy_sphere"),
                        ("c1_delaunay", lambda r: r["c"] == 1.0 and r["family"] == "delaunay_sphere"),
                        ("c1_grid_larger", lambda r: r["c"] == 1.0 and r["family"] in ("grid_sphere", "grid_torus"))):
        per_method = {}
        for m in METHODS:
            seed_vals = []
            for run in runs:
                rows = [r for r in run["results"] if r["method"] == m and "error" not in r and filt(r)]
                if rows:
                    seed_vals.append(cell(rows))
            if seed_vals:
                per_method[m] = {k: {"mean": float(np.mean([v[k] for v in seed_vals])), "std": float(np.std([v[k] for v in seed_vals], ddof=1)) if len(seed_vals) > 1 else 0.0}
                                 for k in KEYS}
                per_method[m]["n_seeds"] = len(seed_vals)
        out["pooled"][scope] = per_method
    # paired: learned minus fps, per held-out (mesh, c) cell and per seed
    for key in ("J", "stretch_r0_mean", "work_r0", "stretch_r1_mean"):
        diffs, wins = [], 0
        for run in runs:
            by = defaultdict(dict)
            for r in run["results"]:
                if "error" not in r:
                    by[(r["mesh"], r["c"])][r["method"]] = r[key]
            d = [v["learned"] - v["fps"] for v in by.values() if "learned" in v and "fps" in v]
            diffs.append(float(np.mean(d))); wins += int(sum(x < 0 for x in d)); n = len(d)
        out["paired"][key] = {"mean_learned_minus_fps_per_seed": diffs, "cells_learned_lower": wins, "cells_total": n * len(runs)}
    out["skipped"] = sorted({(r["method"], r["error"]) for run in runs for r in run["results"] if "error" in r})
    return out


def markdown(s: dict, scope: str) -> str:
    lines = ["| method | mean stretch, 0 rings (sd) | p95, 0 rings | mean stretch, 1 ring | work, 0 rings | patch-size CV | J |", "|---|---|---|---|---|---|---|"]
    for m, e in s["pooled"][scope].items():
        f = lambda k, nd=4: f"{e[k]['mean']:.{nd}f}"
        lines.append(f"| {m} | {f('stretch_r0_mean')} ({e['stretch_r0_mean']['std']:.4f}) | {f('stretch_r0_p95')} | {f('stretch_r1_mean')} | {f('work_r0', 3)} | {f('size_cv', 3)} | {f('J')} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = summarise([json.load(open(p)) for p in sorted(a.runs)])
    os.makedirs(a.out, exist_ok=True)
    json.dump(s, open(os.path.join(a.out, "ml2_learned_partition.json"), "w"), indent=1)
    for scope in s["pooled"]:
        print(f"\n### {scope}\n"); print(markdown(s, scope))
    print(json.dumps(s["paired"], indent=1)); print(s["skipped"]); print(s["theta"]); print(s["train_J"])


if __name__ == "__main__":
    main()
