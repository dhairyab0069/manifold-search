"""Aggregate run_heuristic JSONs (one per seed) into a small summary and a markdown table.

    python -m manifoldsearch.ml.report_heuristic results/ml/heuristic_seed*.json --out results_summary
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

KEYS = ["expanded_frac", "subopt_mean", "subopt_max", "frac_suboptimal", "over_frac", "over_max_rel"]


def summarise(runs: list[dict]) -> dict:
    methods = [r["method"] for r in runs[0]["results"] if r["mesh"] == runs[0]["results"][0]["mesh"]]
    out = {"seeds": [r["seed"] for r in runs], "fit_s": runs[0]["fit_s"], "pooled": {}, "by_family": {}}
    for scope, filt in (("all", lambda r: True), ("bumpy_sphere", lambda r: r["family"] == "bumpy_sphere"),
                        ("delaunay_sphere", lambda r: r["family"] == "delaunay_sphere"),
                        ("grid_larger", lambda r: r["family"] in ("grid_sphere", "grid_torus"))):
        per = {}
        for m in methods:
            vals = []
            for run in runs:
                rows = [r for r in run["results"] if r["method"] == m and filt(r)]
                vals.append({k: float(np.mean([r[k] for r in rows])) if k != "subopt_max" and k != "over_max_rel" else float(np.max([r[k] for r in rows])) for k in KEYS})
            per[m] = {k: {"mean": float(np.mean([v[k] for v in vals])), "std": float(np.std([v[k] for v in vals], ddof=1)) if len(vals) > 1 else 0.0} for k in KEYS}
        out["pooled"][scope] = per
    return out


def markdown(s: dict, scope: str) -> str:
    P = s["pooled"][scope]
    base = P["dijkstra"]["expanded_frac"]["mean"]
    eu = P["euclid"]["expanded_frac"]["mean"]
    lines = ["| heuristic | expanded (% of N) | vs Dijkstra | vs straight-line A* | mean suboptimality (sd) | queries suboptimal | worst path | vertices overestimated | worst overestimate |", "|---|---|---|---|---|---|---|---|---|"]
    for m, e in P.items():
        x = e["expanded_frac"]["mean"]
        lines.append(f"| {m} | {100 * x:.2f} | {x / base:.3f} | {x / eu:.3f} | {e['subopt_mean']['mean']:.5f} ({e['subopt_mean']['std']:.5f}) | "
                     f"{100 * e['frac_suboptimal']['mean']:.1f}% | {e['subopt_max']['mean']:.3f} | {100 * e['over_frac']['mean']:.1f}% | {100 * e['over_max_rel']['mean']:.0f}% |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = summarise([json.load(open(p)) for p in sorted(a.runs)])
    os.makedirs(a.out, exist_ok=True)
    json.dump(s, open(os.path.join(a.out, "ml3_learned_heuristic.json"), "w"), indent=1)
    for scope in s["pooled"]:
        print(f"\n### {scope}\n"); print(markdown(s, scope))


if __name__ == "__main__":
    main()
