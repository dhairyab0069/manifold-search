import numpy as np

from manifoldsearch import grid_torus
from manifoldsearch.baselines import dijkstra
from manifoldsearch.ml.heuristic import admissible_clamp, alt_bounds, landmarks_graph_fps


def test_clamp_never_goes_below_the_lower_bounds():
    h = np.array([0.1, 2.0, 0.5]); euclid = np.array([0.4, 1.0, 0.5]); alt = np.array([0.2, 0.3, 0.9])
    assert admissible_clamp(h, euclid).tolist() == [0.4, 2.0, 0.5]
    assert admissible_clamp(h, euclid, alt).tolist() == [0.4, 2.0, 0.9]


def test_alt_bounds_sandwich_the_exact_distance():
    mesh = grid_torus(8, 12)
    g = mesh.csr()
    ids, D = landmarks_graph_fps(g, 4, np.random.default_rng(0))
    for t in (3, 40):
        exact = dijkstra(g, t)
        lb, ub = alt_bounds(D, t)
        assert (lb <= exact + 1e-9).all() and (exact <= ub + 1e-9).all()
