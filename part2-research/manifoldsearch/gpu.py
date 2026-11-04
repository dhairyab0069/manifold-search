"""Exact distances for many sources at once, on a GPU (or the CPU, for testing).

Why: judging the hierarchical search needs exact answers for thousands of random queries on meshes with
millions of vertices. scipy answers one source at a time on one core. Here a batch of sources relaxes every
edge at the same time (Bellman-Ford, written as one scatter-min per round), until nothing improves; the
number of rounds is about the mesh's hop diameter, so it is worth it when there are many sources.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix


def batched_sssp(graph: csr_matrix, sources, device: str = "cpu", max_cells: int = 200_000_000) -> np.ndarray:
    """Distances from each source to every vertex: array (len(sources), N). Sources are processed in chunks so
    the (chunk x edges) work array stays under `max_cells` numbers."""
    coo = graph.tocoo()
    u = torch.as_tensor(coo.row, dtype=torch.long, device=device)
    v = torch.as_tensor(coo.col, dtype=torch.long, device=device)
    w = torch.as_tensor(coo.data, dtype=torch.float32, device=device)
    n, e = graph.shape[0], len(coo.data)
    sources = np.asarray(sources, dtype=np.int64)
    chunk = max(1, min(len(sources), max_cells // max(e, 1)))
    out = []
    for lo in range(0, len(sources), chunk):
        src = torch.as_tensor(sources[lo:lo + chunk], dtype=torch.long, device=device)
        b = len(src)
        dist = torch.full((b, n), float("inf"), dtype=torch.float32, device=device)
        dist[torch.arange(b, device=device), src] = 0.0
        index = v.unsqueeze(0).expand(b, -1)
        for _ in range(n):                                   # at most n - 1 rounds; stops as soon as nothing changes
            cand = dist[:, u] + w
            new = dist.clone()
            new.scatter_reduce_(1, index, cand, reduce="amin")
            if torch.equal(new, dist):
                break
            dist = new
        out.append(dist.cpu().numpy())
    return np.concatenate(out)


def best_device() -> str:
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
