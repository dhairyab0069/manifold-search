"""Pareto evaluation of corridor policies: work and time against mean, p95 and max stretch.

A policy picks a ring count per query. The result of a query is the stretch of the corridor with that many rings.
Cost is the fraction of vertices searched (work) and the measured query time (feature extraction and model
latency included for learned policies). The honest comparison is against the best *mix of fixed rings* at equal
cost: sending a random fraction p of queries to ring r and the rest to ring r+1 costs the same as a learned policy
that spends that much, and needs no model.
"""
from __future__ import annotations

import numpy as np


def stretch_stats(s: np.ndarray) -> dict[str, float]:
    return {"mean": float(s.mean()), "p95": float(np.percentile(s, 95)), "max": float(s.max())}


def pick(arr: np.ndarray, choice: np.ndarray) -> np.ndarray:
    return arr[np.arange(len(choice)), choice]


def policy_result(choice: np.ndarray, d, exact, size, time, n_vertices, overhead_s=0.0, cascade=False) -> dict:
    """Result of a policy given its per-query ring choice. `cascade`: the policy needs the ring-0 corridor search
    first (its answer is a feature), so a choice of r > 0 pays for ring 0 as well as ring r."""
    s = pick(d, choice) / exact
    work = pick(size, choice).astype(float)
    t = pick(time, choice) + overhead_s
    if cascade:
        extra = choice > 0
        work = work + extra * size[:, 0]
        t = t + extra * time[:, 0]
    out = {"work": float(work.mean() / n_vertices), "time_ms": float(t.mean() * 1e3), **stretch_stats(s),
           "ring_hist": np.bincount(choice, minlength=d.shape[1]).tolist()}
    return out


def mix_curve(cost: np.ndarray, stretch: np.ndarray, seed: int = 0, steps: int = 41):
    """Costs and stretch stats of random mixtures of two adjacent fixed rings. cost, stretch: (n, rings+1)."""
    rng = np.random.default_rng(seed)
    u = rng.random(len(cost))
    pts = []
    for r in range(cost.shape[1] - 1):
        for p in np.linspace(0, 1, steps):
            choice = np.where(u < p, r, r + 1)
            s, c = pick(stretch, choice), pick(cost, choice)
            pts.append((float(c.mean()), *stretch_stats(s).values()))
    arr = np.array(sorted(pts))
    return arr                         # columns: cost, mean, p95, max


def at_equal_cost(curve: np.ndarray, cost: float) -> dict[str, float] | None:
    c = curve[:, 0]
    if cost < c.min() or cost > c.max():
        return None
    # lower envelope: cheapest-stretch at or below the cost; interpolate on the pareto frontier of each statistic
    out = {}
    for j, name in ((1, "mean"), (2, "p95"), (3, "max")):
        order = np.argsort(c)
        cc, vv = c[order], curve[order, j]
        best = np.minimum.accumulate(vv)             # best stretch achievable with at most this cost
        out[name] = float(np.interp(cost, cc, best))
    return out
