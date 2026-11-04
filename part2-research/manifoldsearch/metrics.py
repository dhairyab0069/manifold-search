"""How good is an approximate distance? Stretch = approximate / exact (1.0 is perfect; it is never below 1,
because the approximate path is a real path in the mesh)."""
from __future__ import annotations

import numpy as np


def sphere_geodesic(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Great-circle distance between points on the unit sphere (row-wise)."""
    return np.arccos(np.clip(np.sum(p * q, axis=-1), -1.0, 1.0))


def stretch_summary(approx: np.ndarray, exact: np.ndarray) -> dict[str, float]:
    s = np.asarray(approx) / np.asarray(exact)
    return {"mean": float(s.mean()), "median": float(np.median(s)), "p95": float(np.percentile(s, 95)), "max": float(s.max())}
