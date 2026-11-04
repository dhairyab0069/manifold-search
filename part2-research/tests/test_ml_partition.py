import numpy as np

from manifoldsearch import bumpy_sphere
from manifoldsearch.hierarchical import farthest_point_seeds, index_from_seeds
from manifoldsearch.ml.partition import evolution_strategy, seeds_weighted_fps, vertex_feature_matrix
from manifoldsearch.ml.features import build_context

MESH = bumpy_sphere(16, 32)


def test_zero_weights_reduce_to_farthest_point_and_patches_cover_the_mesh():
    ctx = build_context(MESH, 1.0, 0)
    phi = vertex_feature_matrix(ctx)
    a = seeds_weighted_fps(MESH.points, phi, np.zeros(phi.shape[1]), 20, np.random.default_rng(5))
    b = farthest_point_seeds(MESH.points, 20, np.random.default_rng(5))
    assert np.array_equal(a, b)
    idx = index_from_seeds(MESH.csr(), a)
    assert idx.k == 20 and set(np.unique(idx.patch_of)) == set(range(20))


def test_evolution_strategy_improves_a_quadratic():
    best, val, hist = evolution_strategy(lambda x: float(((x - 1.0) ** 2).sum()), 3, np.random.default_rng(0), pop=12, elite=3, gens=20)
    assert val < hist[0] and val < 0.5
