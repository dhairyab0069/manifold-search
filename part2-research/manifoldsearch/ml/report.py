"""Aggregate per-seed evaluation JSONs into one small summary (JSON, CSV) and the Pareto figures.

    python -m manifoldsearch.ml.report results/ml/eval_seed*.json --out results_summary --prefix ml1
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

METRICS = ["work", "time_ms", "mean", "p95", "max"]


def agg(vals):
    a = np.array(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0}


def summarise(evals: list[dict]) -> dict:
    out = {"seeds": [e["seed"] for e in evals], "groups": {}}
    for g in evals[0]["groups"]:
        pol = {}
        for key in evals[0]["groups"][g]["policies"]:
            rows = [e["groups"][g]["policies"][key] for e in evals]
            entry = {m: agg([r[m] for r in rows]) for m in METRICS}
            entry["ring_hist"] = np.sum([r["ring_hist"] for r in rows], axis=0).tolist()
            if "mix_equal_work" in rows[0]:
                for axis in ("work", "time"):
                    eq = [r[f"mix_equal_{axis}"] for r in rows]
                    ok = all(e is not None for e in eq)
                    if ok:
                        for m in ("mean", "p95", "max"):
                            # positive = learned policy has LOWER stretch than the best fixed-ring mix at equal cost
                            d = [e[m] - r[m] for e, r in zip(eq, rows)]
                            entry[f"adv_{axis}_{m}"] = {**agg(d), "wins": int(sum(x > 0 for x in d)), "n": len(d)}
                entry["under_rate"] = agg([r["under_rate"] for r in rows]); entry["accuracy"] = agg([r["accuracy_argmax"] for r in rows])
            pol[key] = entry
        out["groups"][g] = {"n_queries": evals[0]["groups"][g]["n_queries"], "meshes": evals[0]["groups"][g]["meshes"], "policies": pol}
    out["gbt_importance"] = evals[0].get("gbt_importance", {})
    out["label_hist_train"] = {str(e["seed"]): e["label_hist_train"] for e in evals}
    out["feature_time_ms"] = agg([e["feature_time_ms"] for e in evals])
    return out


def compact(summary: dict, keep_taus=(0.7, 0.9, 0.98)) -> dict:
    """A few tens of KB: means over seeds (and std for the headline stretch numbers), rounded; all taus for the pooled group only."""
    def r(x, nd=5):
        return round(x, nd)
    out = {"seeds": summary["seeds"], "feature_time_ms": summary["feature_time_ms"], "gbt_importance_seed0": summary["gbt_importance"], "groups": {}}
    for g, gd in summary["groups"].items():
        pol = {}
        for key, e in gd["policies"].items():
            if "_tau" in key and g != "all_test" and float(key.rsplit("_tau", 1)[1]) not in keep_taus:
                continue
            row = {m: r(e[m]["mean"]) for m in METRICS}
            row.update({f"{m}_std": r(e[m]["std"]) for m in ("mean", "p95")})
            for a in ("work", "time"):
                for m in ("mean", "p95"):
                    if f"adv_{a}_{m}" in e:
                        row[f"adv_{a}_{m}"] = r(e[f"adv_{a}_{m}"]["mean"]); row[f"adv_{a}_{m}_wins"] = e[f"adv_{a}_{m}"]["wins"]
            pol[key] = row
        out["groups"][g] = {"n_queries_per_seed": gd["n_queries"], "meshes": gd["meshes"], "policies": pol}
    return out


def write_csv(summary, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "policy"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")] + ["adv_work_mean_mean", "adv_work_p95_mean", "wins_work_p95"])
        for g, gd in summary["groups"].items():
            for key, e in gd["policies"].items():
                w.writerow([g, key] + [round(e[m][s], 5) for m in METRICS for s in ("mean", "std")] +
                           [round(e.get("adv_work_mean", {}).get("mean", float("nan")), 5), round(e.get("adv_work_p95", {}).get("mean", float("nan")), 5),
                            e.get("adv_work_p95", {}).get("wins", "")])


def plot(summary, path_prefix, group="all_test"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pol = summary["groups"][group]["policies"]
    for tol in ("1.02", "1.01"):
        fig, ax = plt.subplots(2, 2, figsize=(9, 6.5))
        for j, (xk, xl) in enumerate((("work", "fraction of vertices searched"), ("time_ms", "query time (ms)"))):
            for i, (yk, yl) in enumerate((("mean", "mean stretch"), ("p95", "p95 stretch"))):
                a = ax[i][j]
                fx = [(r, pol[f"fixed_{r}"]) for r in range(4)]
                a.plot([p[xk]["mean"] for _, p in fx], [p[yk]["mean"] for _, p in fx], "o-", color="0.45", label="fixed rings 0..3")
                for r, p in fx:
                    a.annotate(str(r), (p[xk]["mean"], p[yk]["mean"]), textcoords="offset points", xytext=(4, 4), fontsize=7, color="0.45")
                o = pol[f"oracle_{tol}"]
                a.plot(o[xk]["mean"], o[yk]["mean"], "*", color="k", ms=11, label="oracle")
                for (label, mk, col) in (("gbt_base", "gbt_base", "C0"), ("gbt_base+ring0", "gbt_base+ring0", "C1"), ("mlp_base", "mlp_base", "C2")):
                    pts = sorted([(e[xk]["mean"], e[yk]["mean"]) for k, e in pol.items() if k.startswith(f"{mk}_tol{tol}_")])
                    if pts:
                        a.plot(*zip(*pts), "o-", ms=3.5, color=col, label=label.replace("_", " "))
                a.set_xlabel(xl); a.set_ylabel(yl); a.set_xscale("log"); a.grid(alpha=0.3)
        ax[0][0].legend(fontsize=7)
        fig.suptitle(f"Adaptive corridor, held-out meshes, tolerance {tol} (mean of seeds)")
        fig.tight_layout()
        fig.savefig(f"{path_prefix}_pareto_tol{tol}.png", dpi=110)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("evals", nargs="+"); ap.add_argument("--out", required=True); ap.add_argument("--prefix", default="ml1")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    evals = [json.load(open(p)) for p in sorted(a.evals)]
    s = summarise(evals)
    os.makedirs(a.out, exist_ok=True)
    json.dump(compact(s), open(os.path.join(a.out, f"{a.prefix}_adaptive_corridor.json"), "w"), separators=(",", ":"))
    write_csv({**s, "groups": {"all_test": s["groups"]["all_test"]}}, os.path.join(a.out, f"{a.prefix}_adaptive_corridor_pooled.csv"))
    if not a.no_plot:
        plot(s, os.path.join(a.out, a.prefix))


if __name__ == "__main__":
    main()
