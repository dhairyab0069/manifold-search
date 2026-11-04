import numpy as np

from manifoldsearch import bumpy_sphere
from manifoldsearch.ml.dataset import labels, split_of
from manifoldsearch.ml.evaluation import at_equal_cost, mix_curve
from manifoldsearch.ml.features import BASE_FEATURES, build_context, corridor_distance, query_features
from manifoldsearch.ml.models import policy_from_proba

MESH = bumpy_sphere(20, 40)


def test_features_are_deterministic_and_corridor_matches_the_index():
    a, b = build_context(MESH, 1.0, 3), build_context(MESH, 1.0, 3)
    for s, t in [(5, 700), (40, 333), (100, 101)]:
        xa, ra = query_features(a, s, t)
        xb, rb = query_features(b, s, t)
        assert len(xa) == len(BASE_FEATURES) and np.array_equal(xa, xb) and ra == rb
        for rings in (0, 1, 2):
            assert corridor_distance(a, s, t, ra, rings)[0] == a.index.query(s, t, rings)[0]


def test_labels_are_the_smallest_sufficient_ring():
    exact = np.array([1.0, 1.0, 1.0, 1.0])
    d = np.array([[1.0, 1.0, 1.0, 1.0], [1.05, 1.015, 1.0, 1.0], [1.3, 1.2, 1.1, 1.019], [1.3, 1.2, 1.1, 1.05]])
    assert labels(d, exact, 1.02).tolist() == [0, 1, 3, 3]
    assert labels(d, exact, 1.01).tolist() == [0, 2, 3, 3]


def test_train_test_split_is_by_mesh_and_family():
    assert split_of("grid_sphere", 128) == "train" and split_of("grid_torus", 32) == "train"
    assert split_of("grid_sphere", 256) == "test" and split_of("bumpy_sphere", 64) == "test" and split_of("delaunay_sphere", 32) == "test"


def test_policy_threshold_and_mix_curve_behave():
    proba = np.array([[0.9, 0.1, 0, 0], [0.2, 0.5, 0.3, 0.0]])
    assert policy_from_proba(proba, 0.8).tolist() == [0, 2] and policy_from_proba(proba, 0.99).tolist() == [1, 2]
    cost = np.tile(np.arange(4.0), (100, 1)); stretch = np.tile([1.1, 1.05, 1.02, 1.0], (100, 1))
    curve = mix_curve(cost, stretch)
    assert abs(at_equal_cost(curve, 0.5)["mean"] - 1.075) < 0.02
