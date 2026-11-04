"""Direction 3: learned A* heuristics.

For a fixed target t, A* needs h(v) ~ d(v, t) for every vertex v it touches. Baselines: the straight line (admissible),
landmark ALT (admissible: max over landmarks l of |d(l,v) - d(l,t)|), and weighted versions of both (bounded by w).

Learned heuristics are built so the cheap parts are guaranteed and the network only decides how far to trust them:
- `geo`:  h = euclid * (1 + softplus(f(x)))            x: coordinates of v and t, local curvature, euclid.
          Never below the straight line (the clamp), no landmarks needed.
- `lm`:   h = lb + sigmoid(f(x)) * (ub - lb)            lb = max(euclid, ALT lower bound) <= d <= ub = min_l d(l,v) + d(l,t).
          The landmark distances enter through these two bounds, so the net transfers across meshes whatever
          landmarks were drawn; sigmoid = 0 is exactly ALT.
Training uses exact distances and an asymmetric loss: overestimates (h > d, which can make A* return a longer
path) cost `over_weight` times more than underestimates. Admissibility is not guaranteed for these; it is measured.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from ..baselines import _search

GEO_FEATURES = 11
LM_FEATURES = 15


def landmarks_graph_fps(graph: csr_matrix, n_landmarks: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Landmarks by farthest-point selection in graph distance. Returns (ids, D) with D[i] = distances from landmark i."""
    ids = [int(rng.integers(graph.shape[0]))]
    rows = [dijkstra(graph, directed=False, indices=ids[0])]
    nearest = rows[0].copy()
    while len(ids) < n_landmarks:
        nxt = int(np.argmax(nearest))
        ids.append(nxt)
        rows.append(dijkstra(graph, directed=False, indices=nxt))
        nearest = np.minimum(nearest, rows[-1])
    return np.array(ids), np.stack(rows)


def alt_bounds(D: np.ndarray, t: int, vs: np.ndarray | slice = slice(None)) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) bounds on d(v, t) from landmark distances (triangle inequality)."""
    dv, dt = D[:, vs], D[:, t:t + 1]
    return np.abs(dv - dt).max(axis=0), (dv + dt).min(axis=0)


def admissible_clamp(h: np.ndarray, euclid: np.ndarray, lower: np.ndarray | None = None) -> np.ndarray:
    """Never go below the best known lower bound (straight line, optionally ALT)."""
    h = np.maximum(h, euclid)
    return h if lower is None else np.maximum(h, lower)


class MeshGeo:
    """Per-mesh arrays the feature builder needs."""
    def __init__(self, mesh, vgeo, D):
        self.points = mesh.points
        self.centre = 0.5 * (mesh.points.max(0) + mesh.points.min(0))
        self.diag = float(np.linalg.norm(mesh.points.max(0) - mesh.points.min(0)))
        self.logkn = np.log(vgeo["kn"] + 1e-3)
        self.D = D


def pair_features(g: MeshGeo, t: int, vs: np.ndarray, kind: str):
    """Feature matrix for the vertices vs against target t, plus (euclid, lb, ub)."""
    p, pt = g.points[vs], g.points[t]
    euclid = np.linalg.norm(p - pt, axis=1)
    lb, ub = alt_bounds(g.D, t, vs)
    lb = np.maximum(lb, euclid)
    ub = np.maximum(ub, lb)
    base = [euclid[:, None] / g.diag, ((p - g.centre) / g.diag), ((pt - g.centre) / g.diag)[None].repeat(len(vs), 0),
            g.logkn[vs][:, None], np.full((len(vs), 1), g.logkn[t])]
    if kind == "geo":
        x = np.hstack(base)                                   # 1 + 3 + 3 + 1 + 1 = 9, plus the offset vector's norm and a bias-like term
        x = np.hstack([x, (np.linalg.norm(p - g.centre, axis=1) / g.diag)[:, None], (np.linalg.norm(pt - g.centre) / g.diag) * np.ones((len(vs), 1))])
    else:
        eps = 1e-6 * g.diag
        x = np.hstack(base + [lb[:, None] / (euclid[:, None] + eps), ub[:, None] / (euclid[:, None] + eps), ((ub - lb) / (ub + eps))[:, None],
                              (np.linalg.norm(p - g.centre, axis=1) / g.diag)[:, None], (np.linalg.norm(pt - g.centre) / g.diag) * np.ones((len(vs), 1)),
                              (lb / g.diag)[:, None]])
    return x.astype(np.float32), euclid, lb, ub


class HeuristicNet:
    """Small MLP; `kind` is "geo" or "lm". predict() returns h for the vertices vs against target t."""

    def __init__(self, kind: str, over_weight: float, seed: int = 0, hidden: int = 64):
        import torch
        self.torch, self.kind, self.over_weight, self.seed = torch, kind, over_weight, seed
        n_in = GEO_FEATURES if kind == "geo" else LM_FEATURES
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(hidden, 1))
        self.mu = self.sd = None

    def _h(self, x, euclid, lb, ub):
        torch = self.torch
        f = self.net(x).squeeze(-1)
        if self.kind == "geo":
            return euclid * (1.0 + torch.nn.functional.softplus(f))
        return lb + torch.sigmoid(f) * (ub - lb)

    def fit(self, x, euclid, lb, ub, dstar, diag, epochs=8, lr=2e-3, device="cpu", bs=2048):
        torch = self.torch
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        X, E, L, U, Y = t((x - self.mu) / self.sd), t(euclid), t(lb), t(ub), t(dstar)
        self.net.to(device)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        scale = Y + 0.05 * diag
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(epochs):
            perm = torch.randperm(len(X), generator=g).to(device)
            for i in range(0, len(X), bs):
                b = perm[i:i + bs]
                e = (self._h(X[b], E[b], L[b], U[b]) - Y[b]) / scale[b]
                loss = (self.over_weight * torch.relu(e) ** 2 + torch.relu(-e) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        self.net.cpu()
        return self

    def predict(self, g: MeshGeo, t: int, vs: np.ndarray | None = None) -> np.ndarray:
        torch = self.torch
        vs = np.arange(len(g.points)) if vs is None else vs
        x, e, lb, ub = pair_features(g, t, vs, self.kind)
        with torch.no_grad():
            tt = lambda a: torch.as_tensor(a, dtype=torch.float32)
            h = self._h(tt((x - self.mu) / self.sd), tt(e), tt(lb), tt(ub)).numpy().astype(np.float64)
        return h


def astar_with_h(graph: csr_matrix, s: int, t: int, h: np.ndarray) -> tuple[float, int]:
    hl = h.tolist()
    return _search(graph, s, t, lambda v: hl[v])
