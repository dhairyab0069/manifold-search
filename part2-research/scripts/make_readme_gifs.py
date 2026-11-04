"""Draw the animated figures in the top-level README from the real package.

    cd part2-research && python scripts/make_readme_gifs.py ../docs/gifs

Two GIFs: the patches on a turning mesh, and one query answered step by step (exact path, the patch route, then the
corridor at 0, 1 and 2 rings, with the measured stretch and share searched). The query uses the package's own
`build_index`; the script only re-runs its corridor step to get the path itself, and asserts the distance agrees.
"""
from __future__ import annotations

import io
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from PIL import Image
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, ".")
from manifoldsearch import build_index, delaunay_sphere  # noqa: E402

BG, INK, MUTED, GOLD, CYAN = "#0b1a2b", "#e8f1ff", "#6f86a3", "#ffd166", "#5ce1e6"
SIZE = 4.2   # inches; dpi below sets the pixel size


def rotation(az: float, el: float) -> np.ndarray:
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
    return rx @ rz


def patch_colours(k: int) -> np.ndarray:
    rng = np.random.default_rng(3)
    hues = (np.arange(k) * 0.381966) % 1.0                     # golden-ratio spacing: neighbours look different
    rng.shuffle(hues)
    hsv = np.stack([hues, np.full(k, 0.62), np.full(k, 0.95)], axis=1)
    return matplotlib.colors.hsv_to_rgb(hsv)


class Scene:
    def __init__(self, n: int = 2600, k: int = 46, seed: int = 4):
        self.mesh = delaunay_sphere(n, seed=seed)
        self.graph = self.mesh.csr()
        self.index = build_index(self.graph, self.mesh.points, k, seed=0)
        self.cols = patch_colours(self.index.k)
        self.edges = self.mesh.edges
        self.dot = 22
        self.pts = self.mesh.points

    def frame(self, az, el=0.35, vertex_rgba=None, edge_alpha=0.10, paths=(), marks=(), title="", lines=(), seeds=False):
        R = rotation(az, el)
        p = self.pts @ R.T                                     # x right, y up (z toward the viewer after the swap below)
        x, y, z = p[:, 0], p[:, 2], p[:, 1] * -1               # view down the -y axis
        front = z > -0.02
        fig = plt.figure(figsize=(SIZE, SIZE), dpi=110, facecolor=BG)
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(BG); ax.set_xlim(-1.12, 1.12); ax.set_ylim(-1.2, 1.12); ax.axis("off")
        e = self.edges
        keep = front[e[:, 0]] & front[e[:, 1]]
        seg = np.stack([np.stack([x[e[keep, 0]], y[e[keep, 0]]], 1), np.stack([x[e[keep, 1]], y[e[keep, 1]]], 1)], 1)
        ax.add_collection(LineCollection(seg, colors=MUTED, linewidths=0.35, alpha=edge_alpha))
        rgba = vertex_rgba if vertex_rgba is not None else np.c_[self.cols[self.index.patch_of], np.ones(len(x))]
        shade = 0.45 + 0.55 * np.clip(z, 0, 1)
        order = np.argsort(z)
        sel = order[front[order]]
        c = rgba[sel].copy(); c[:, :3] *= shade[sel, None]
        ax.scatter(x[sel], y[sel], s=self.dot, c=c, linewidths=0)
        if seeds:
            s = self.index.seeds; s = s[front[s]]
            ax.scatter(x[s], y[s], s=26, facecolors="none", edgecolors=INK, linewidths=0.8)
        for path, colour, lw, ls in paths:
            path = np.asarray(path)
            vis = front[path]
            for a, b in zip(path[:-1], path[1:]):
                if vis[list(path).index(a)] if False else (front[a] and front[b]):
                    ax.plot([x[a], x[b]], [y[a], y[b]], color=colour, lw=lw, ls=ls, solid_capstyle="round", zorder=4)
        for v, label, colour in marks:
            if front[v]:
                ax.scatter([x[v]], [y[v]], s=70, color=colour, zorder=6, edgecolors=BG, linewidths=1.2)
                ax.text(x[v], y[v] + 0.075, label, color=colour, fontsize=9, ha="center", fontweight="bold", zorder=7)
        if title:
            ax.text(0, 1.07, title, color=INK, fontsize=11, ha="center", va="top", fontweight="bold")
        for i, line in enumerate(lines):
            ax.text(-1.08, -1.04 - 0.0 - i * 0.075, line, color=MUTED if i else INK, fontsize=9, ha="left", va="center")
        buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor=BG); plt.close(fig); buf.seek(0)
        return Image.open(buf).convert("RGB")

    # ---- the query, step by step
    def query_parts(self, s: int, t: int):
        idx, g = self.index, self.graph
        d_all, pred = dijkstra(g, directed=False, indices=s, return_predecessors=True)
        exact = d_all[t]
        path = [t]
        while path[-1] != s:
            path.append(int(pred[path[-1]]))
        exact_path = path[::-1]
        ps, pt = int(idx.patch_of[s]), int(idx.patch_of[t])
        _, ppred = dijkstra(idx.patch_graph, directed=False, indices=ps, return_predecessors=True)
        route, cur = [pt], pt
        while cur != ps:
            cur = int(ppred[cur]); route.append(cur)
        return exact, exact_path, route[::-1]

    def corridor(self, route, rings, s, t):
        idx = self.index
        patches = np.zeros(idx.k, dtype=bool); patches[route] = True
        for _ in range(rings):
            patches |= (idx.patch_adj @ patches.astype(np.int8)) > 0
        mask = patches[idx.patch_of]
        ids = np.flatnonzero(mask); local = {int(v): i for i, v in enumerate(ids)}
        sub = self.graph[ids][:, ids]
        d, pred = dijkstra(sub, directed=False, indices=local[s], return_predecessors=True)
        path = [local[t]]
        while path[-1] != local[s]:
            path.append(int(pred[path[-1]]))
        return mask, [int(ids[i]) for i in path[::-1]], float(d[local[t]])


def save(frames, durations, path):
    quant = [f.quantize(colors=64, method=Image.MEDIANCUT, dither=Image.NONE) for f in frames]
    quant[0].save(path, save_all=True, append_images=quant[1:], duration=durations, loop=0, optimize=False, disposal=2)


def main(out: str) -> None:
    sc = Scene()
    # 1. the patches on a turning mesh
    frames = [sc.frame(az=a, title="The surface is cut into patches", seeds=True,
                       lines=[f"{sc.mesh.n:,} vertices, {sc.index.k} patches grown from seed points (circled)"]) for a in np.linspace(0, 2 * np.pi, 36, endpoint=False)]
    save(frames, [90] * len(frames), f"{out}/patches.gif")

    # 2. one query, step by step: pick endpoints about 100 degrees apart and face them
    rng = np.random.default_rng(11)
    P = sc.pts
    while True:
        s, t = (int(v) for v in rng.integers(0, sc.mesh.n, 2))
        ang = np.degrees(np.arccos(np.clip(P[s] @ P[t], -1, 1)))
        if 85 < ang < 105:
            break
    mid = P[s] + P[t]; mid /= np.linalg.norm(mid)
    az = -np.pi / 2 - np.arctan2(mid[1], mid[0])        # turn the midpoint to face the viewer
    el = float(np.arcsin(mid[2]))                        # and tilt it to the middle of the picture
    exact, exact_path, route = sc.query_parts(s, t)
    dim = np.c_[sc.cols[sc.index.patch_of], np.full(sc.mesh.n, 0.22)]
    marks = [(s, "A", CYAN), (t, "B", CYAN)]
    frames, durs = [], []
    def add(img, ms): frames.append(img); durs.append(ms)
    kw = dict(az=az, el=el)
    add(sc.frame(**kw, vertex_rgba=dim, marks=marks, title="From A to B", lines=["Two points on the surface"]), 900)
    for m in (0.3, 0.6, 1.0):
        k = max(2, int(len(exact_path) * m))
        add(sc.frame(**kw, vertex_rgba=dim, marks=marks, paths=[(exact_path[:k], INK, 2.0, "-")], title="The exact shortest path",
                     lines=[f"Searching every vertex finds it (length {exact:.3f})"]), 160 if m < 1 else 800)
    pm = np.zeros(sc.index.k, dtype=bool); pm[route] = True
    route_rgba = np.c_[sc.cols[sc.index.patch_of], np.where(pm[sc.index.patch_of], 1.0, 0.12)]
    add(sc.frame(**kw, vertex_rgba=route_rgba, marks=marks, title="First, a route over the patches", lines=[f"{len(route)} patches on the route; the rest are ignored"]), 1100)
    for r in (0, 1, 2):
        mask, hpath, d = sc.corridor(route, r, s, t)
        assert abs(d - sc.index.query(s, t, r)[0]) < 1e-9
        rgba = np.c_[sc.cols[sc.index.patch_of], np.where(mask, 1.0, 0.10)]
        add(sc.frame(**kw, vertex_rgba=rgba, marks=marks, paths=[(exact_path, INK, 1.2, (0, (2, 2))), (hpath, GOLD, 2.4, "-")],
                     title=f"Exact search inside the corridor: {r} ring{'s' if r != 1 else ''}",
                     lines=[f"stretch {d / exact:.3f}   |   searched {100 * mask.mean():.0f}% of the vertices", "white dashed: exact path    gold: corridor path"]), 1500)
    save(frames, durs, f"{out}/corridor.gif")
    print("ok", [f for f in ("patches.gif", "corridor.gif")])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../docs/gifs")
