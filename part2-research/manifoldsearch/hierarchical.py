"""The coarse-to-fine search from Part 1, rebuilt with arrays so it scales.

How it works (the same idea as the notebook):
 1. Preprocess once: pick k seed vertices spread over the surface, give every vertex to its nearest seed
    (a graph Voronoi partition), and build a small graph over the patches.
 2. Answer a query: route over the patch graph (tiny), then run an ordinary Dijkstra only inside the
    patches that route passes through, plus their neighbours (`corridor_rings`).

The answer is a real path in the mesh, so it is never shorter than the exact one; the question is how
much longer, and how much work it saves. If the corridor happens not to connect the endpoints, it is
widened until it does, so a query always gets an answer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as _dijkstra


def farthest_point_seeds(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k vertices spread out in 3D: start anywhere, then repeatedly add the vertex farthest from all chosen so far."""
    seeds = [int(rng.integers(len(points)))]
    nearest = np.linalg.norm(points - points[seeds[0]], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(nearest))
        seeds.append(nxt)
        nearest = np.minimum(nearest, np.linalg.norm(points - points[nxt], axis=1))
    return np.array(seeds)


@dataclass
class HierarchicalIndex:
    graph: csr_matrix
    seeds: np.ndarray          # (k,) vertex ids
    patch_of: np.ndarray       # (N,) patch number per vertex
    seed_dist: np.ndarray      # (N,) graph distance from each vertex to its seed
    patch_graph: csr_matrix    # (k, k) weights = best known seed-to-seed route through a border crossing
    patch_adj: csr_matrix      # (k, k) 0/1 adjacency (for widening the corridor)

    @property
    def k(self) -> int:
        return len(self.seeds)

    def query(self, source: int, target: int, corridor_rings: int = 0) -> tuple[float, int]:
        """Approximate distance and the number of vertices in the corridor that was searched."""
        ps, pt = int(self.patch_of[source]), int(self.patch_of[target])
        _, pred = _dijkstra(self.patch_graph, directed=False, indices=ps, return_predecessors=True)
        route, cur = [pt], pt
        while cur != ps:
            cur = int(pred[cur])
            if cur < 0:
                break                                  # patch graph not connected (cannot happen on a connected mesh)
            route.append(cur)
        patches = np.zeros(self.k, dtype=bool)
        patches[route] = True
        for _ in range(corridor_rings):
            patches |= (self.patch_adj @ patches.astype(np.int8)) > 0
        while True:
            mask = patches[self.patch_of]
            ids = np.flatnonzero(mask)
            local = {int(v): i for i, v in enumerate(ids)}
            sub = self.graph[ids][:, ids]
            d = _dijkstra(sub, directed=False, indices=local[source])[local[target]]
            if np.isfinite(d):
                return float(d), int(len(ids))
            grown = patches | ((self.patch_adj @ patches.astype(np.int8)) > 0)
            if grown.sum() == patches.sum():
                return float("inf"), int(len(ids))
            patches = grown


def build_index(graph: csr_matrix, points: np.ndarray, k: int, seed: int = 0) -> HierarchicalIndex:
    """Preprocess a mesh into k patches."""
    rng = np.random.default_rng(seed)
    k = max(2, min(k, graph.shape[0]))
    seeds = farthest_point_seeds(points, k, rng)
    dist, _, src = _dijkstra(graph, directed=False, indices=seeds, return_predecessors=True, min_only=True)
    order = {int(s): i for i, s in enumerate(seeds)}
    patch_of = np.fromiter((order[int(s)] for s in src), dtype=np.int64, count=graph.shape[0])

    coo = graph.tocoo()
    cross = patch_of[coo.row] != patch_of[coo.col]
    a, b = patch_of[coo.row[cross]], patch_of[coo.col[cross]]
    w = dist[coo.row[cross]] + coo.data[cross] + dist[coo.col[cross]]     # seed -> u -> v -> seed
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo * k + hi
    order_idx = np.lexsort((w, key))
    key, w, lo, hi = key[order_idx], w[order_idx], lo[order_idx], hi[order_idx]
    first = np.concatenate([[True], key[1:] != key[:-1]])                 # cheapest crossing per patch pair
    lo, hi, w = lo[first], hi[first], w[first]
    pg = csr_matrix((np.concatenate([w, w]), (np.concatenate([lo, hi]), np.concatenate([hi, lo]))), shape=(k, k))
    adj = csr_matrix((np.ones(2 * len(lo), dtype=np.int8), (np.concatenate([lo, hi]), np.concatenate([hi, lo]))), shape=(k, k))
    return HierarchicalIndex(graph, seeds, patch_of, dist, pg, adj)
