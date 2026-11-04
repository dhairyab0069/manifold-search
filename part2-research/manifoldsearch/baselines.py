"""The reference searches: exact Dijkstra (scipy, fast) and a counting A* and Dijkstra for search effort."""
from __future__ import annotations

import heapq

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as _dijkstra


def dijkstra(graph: csr_matrix, source: int, target: int | None = None):
    """Exact distances from `source` (scipy). With a target, returns just that distance."""
    d = _dijkstra(graph, directed=False, indices=source)
    return float(d[target]) if target is not None else d


def _search(graph: csr_matrix, source: int, target: int, h) -> tuple[float, int]:
    """Best-first search with heuristic h (h = 0 gives Dijkstra). Returns (distance, vertices expanded)."""
    indptr, indices, data = graph.indptr, graph.indices, graph.data
    best = {source: 0.0}
    heap = [(h(source), 0.0, source)]
    done: set[int] = set()
    while heap:
        _, g, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        if u == target:
            return g, len(done)
        for k in range(indptr[u], indptr[u + 1]):
            v, ng = int(indices[k]), g + float(data[k])
            if ng < best.get(v, np.inf):
                best[v] = ng
                heapq.heappush(heap, (ng + h(v), ng, v))
    return float("inf"), len(done)


def dijkstra_expanded(graph: csr_matrix, source: int, target: int) -> tuple[float, int]:
    return _search(graph, source, target, lambda v: 0.0)


def astar(graph: csr_matrix, points: np.ndarray, source: int, target: int) -> tuple[float, int]:
    """A* with the straight-line distance to the target: it never overestimates, because every edge is a straight line."""
    goal = points[target]
    return _search(graph, source, target, lambda v: float(np.linalg.norm(points[v] - goal)))
