"""Train on the training meshes of one seed, evaluate on its held-out meshes, write one JSON.

    python -m manifoldsearch.ml.run_eval experiments/configs/ml_adaptive.yaml --data results/ml/data --seed 0 --out results/ml/eval_seed0.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import yaml

from .dataset import labels, load_all
from .evaluation import at_equal_cost, mix_curve, pick, policy_result, stretch_stats
from .features import BASE_FEATURES, RING0_FEATURES
from .models import GBTModel, MLPModel, policy_from_proba

GROUPS = {"bumpy_sphere": lambda m: m["family"] == "bumpy_sphere", "delaunay_sphere": lambda m: m["family"] == "delaunay_sphere",
          "grid_larger": lambda m: m["family"] in ("grid_sphere", "grid_torus"), "all_test": lambda m: True}


def stack(parts, key):
    return np.concatenate([p[0][key] for p in parts])


def group_arrays(parts):
    out = {k: stack(parts, k) for k in ("X", "X0", "d", "size", "time", "exact", "tfeat")}
    out["N"] = np.concatenate([np.full(len(p[0]["exact"]), p[1]["N"]) for p in parts])
    out["split_mesh"] = np.concatenate([np.full(len(p[0]["exact"]), i) for i, p in enumerate(parts)])
    return out


def fixed_and_oracle(g, tols, prefix):
    """Per-query cost normalisation: work is vertices searched / N of the query's own mesh."""
    res = {}
    n = g["N"]
    size_f = g["size"] / n[:, None]
    stretch = g["d"] / g["exact"][:, None]
    for r in range(g["d"].shape[1]):
        c = np.full(len(n), r)
        res[f"fixed_{r}"] = policy_result(c, g["d"], g["exact"], size_f, g["time"], 1.0)
    for tol in tols:
        c = labels(g["d"], g["exact"], tol)
        res[f"oracle_{tol}"] = policy_result(c, g["d"], g["exact"], size_f, g["time"], 1.0)
    return res, size_f, stretch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("--data", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    parts = load_all(args.data, args.seed)
    train = [p for p in parts if p[1]["split"] == "train"]
    test = [p for p in parts if p[1]["split"] == "test"]
    assert train and test and not ({p[1]["mesh"] for p in train} & {p[1]["mesh"] for p in test}), "train/test must be disjoint meshes"
    tr = group_arrays(train)
    tols, taus = cfg["tolerances"], cfg["taus"]
    out = {"seed": args.seed, "train_meshes": [p[1]["mesh"] for p in train], "test_meshes": [p[1]["mesh"] for p in test],
           "n_train_queries": int(len(tr["exact"])), "groups": {}}
    groups = {}
    for gname, pred in GROUPS.items():
        sel = [p for p in test if pred(p[1])]
        if sel:
            groups[gname] = group_arrays(sel)
            out["groups"][gname] = {"n_queries": int(len(groups[gname]["exact"])), "meshes": [p[1]["mesh"] for p in sel], "policies": {}}
    base, mixes, sizes = {}, {}, {}
    for gname, g in groups.items():
        res, size_f, stretch = fixed_and_oracle(g, tols, gname)
        out["groups"][gname]["policies"].update(res)
        base[gname], sizes[gname] = res, size_f
        mixes[gname] = {"work": mix_curve(size_f, stretch, args.seed), "time": mix_curve(g["time"] * 1e3, stretch, args.seed)}
    out["label_hist_train"] = {str(t): np.bincount(labels(tr["d"], tr["exact"], t), minlength=4).tolist() for t in tols}
    out["groups_label_hist"] = {gn: {str(t): np.bincount(labels(g["d"], g["exact"], t), minlength=4).tolist() for t in tols} for gn, g in groups.items()}
    importances = {}
    for tol in tols:
        ytr = labels(tr["d"], tr["exact"], tol)
        for fset in cfg["feature_sets"]:
            cascade = fset == "base+ring0"
            fx = (lambda a: np.hstack([a["X"], a["X0"]])) if cascade else (lambda a: a["X"])
            for mname in cfg["models"]:
                model = (GBTModel(args.seed, **cfg.get("gbt", {})) if mname == "gbt" else MLPModel(args.seed, **cfg.get("mlp", {})))
                t0 = time.perf_counter()
                model.fit(fx(tr), ytr)
                fit_s = time.perf_counter() - t0
                names = BASE_FEATURES + (RING0_FEATURES if cascade else [])
                if mname == "gbt":
                    imp = model.m.booster_.feature_importance("gain")
                    importances[f"{mname}_{fset}_{tol}"] = {names[i]: float(imp[i] / imp.sum()) for i in np.argsort(-imp)[:8]}
                # single-query latency (one row at a time), reported on the time axis for learned policies
                any_g = groups["all_test"]
                xs = fx(any_g)[:200]
                if mname == "gbt":
                    model.m.set_params(n_jobs=1)
                t0 = time.perf_counter()
                for i in range(len(xs)):
                    model.predict_proba(xs[i:i + 1])
                latency = (time.perf_counter() - t0) / len(xs)
                for gname, g in groups.items():
                    proba = model.predict_proba(fx(g))
                    yte = labels(g["d"], g["exact"], tol)
                    acc = float((proba.argmax(1) == yte).mean())
                    for tau in taus:
                        choice = policy_from_proba(proba, tau)
                        key = f"{mname}_{fset}_tol{tol}_tau{tau}"
                        r = policy_result(choice, g["d"], g["exact"], sizes[gname], g["time"], 1.0,
                                          overhead_s=float(g["tfeat"].mean()) + latency, cascade=cascade)
                        r["accuracy_argmax"] = acc
                        r["fit_s"] = fit_s; r["model_latency_ms"] = latency * 1e3
                        r["under_rate"] = float((choice < yte).mean())      # chose fewer rings than needed
                        r["over_rate"] = float((choice > yte).mean())
                        for axis in ("work", "time"):
                            eq = at_equal_cost(mixes[gname][axis], r[axis if axis == "work" else "time_ms"])
                            r[f"mix_equal_{axis}"] = eq
                        out["groups"][gname]["policies"][key] = r
    out["gbt_importance"] = importances
    out["feature_time_ms"] = float(tr["tfeat"].mean() * 1e3)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
