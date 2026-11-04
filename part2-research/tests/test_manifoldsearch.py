import numpy as np
import pytest
import networkx as nx

from manifoldsearch import (astar, build_index, bumpy_sphere, delaunay_sphere, dijkstra, grid_sphere, grid_torus,
                            sphere_geodesic, stretch_summary)
from manifoldsearch.baselines import dijkstra_expanded
from manifoldsearch.gpu import batched_sssp
from manifoldsearch.hierarchical import farthest_point_seeds

MESHES = [grid_sphere(10, 20), grid_torus(10, 20), bumpy_sphere(10, 20), delaunay_sphere(150, seed=1)]


@pytest.mark.parametrize("mesh", MESHES, ids=lambda m: m.name)
def test_meshes_are_valid_graphs(mesh):
    assert mesh.points.shape == (mesh.n, 3) and mesh.edges.shape[1] == 2
    assert (mesh.edges[:, 0] < mesh.edges[:, 1]).all(), "each undirected edge is stored once, smaller id first"
    assert len(np.unique(mesh.edges, axis=0)) == len(mesh.edges)
    g = mesh.csr()
    assert (g != g.T).nnz == 0, "adjacency is symmetric"
    assert (g.data >= 0).all()
    assert nx.is_connected(nx.from_scipy_sparse_array(g))


def test_known_edge_counts():
    assert len(delaunay_sphere(200, seed=3).edges) == 3 * 200 - 6            # a triangulated sphere
    t = grid_torus(8, 12)
    assert len(t.edges) == 2 * t.n                                           # 4-regular
    s = grid_sphere(6, 10)
    assert len(s.edges) == 6 * 10 + 5 * 10                                   # rings plus meridian links


def test_dijkstra_matches_networkx():
    mesh = grid_torus(7, 9)
    g = mesh.csr()
    nxg = nx.from_scipy_sparse_array(g)
    for s, t in [(0, 40), (5, 33), (12, 50)]:
        assert dijkstra(g, s, t) == pytest.approx(nx.dijkstra_path_length(nxg, s, t, weight="weight"))


@pytest.mark.parametrize("mesh", MESHES, ids=lambda m: m.name)
def test_astar_and_counting_dijkstra_are_exact_and_astar_expands_no_more(mesh):
    g, rng = mesh.csr(), np.random.default_rng(0)
    for _ in range(8):
        s, t = rng.integers(0, mesh.n, 2)
        exact = dijkstra(g, int(s), int(t))
        d1, e1 = dijkstra_expanded(g, int(s), int(t))
        d2, e2 = astar(g, mesh.points, int(s), int(t))
        assert d1 == pytest.approx(exact) and d2 == pytest.approx(exact)
        assert e2 <= e1 + 1                                                  # the straight-line heuristic never hurts here


def test_farthest_point_seeds_are_distinct_and_spread():
    mesh = delaunay_sphere(300, seed=2)
    seeds = farthest_point_seeds(mesh.points, 12, np.random.default_rng(0))
    assert len(set(seeds.tolist())) == 12
    random_pick = np.random.default_rng(0).choice(mesh.n, 12, replace=False)
    def closest_pair(ids):
        p = mesh.points[ids]; d = np.linalg.norm(p[:, None] - p[None], axis=-1); return d[np.triu_indices(len(ids), 1)].min()
    assert closest_pair(seeds) > closest_pair(random_pick)


@pytest.mark.parametrize("mesh", MESHES, ids=lambda m: m.name)
def test_index_partitions_every_vertex_and_patch_graph_is_connected(mesh):
    idx = build_index(mesh.csr(), mesh.points, 14, seed=0)
    assert idx.patch_of.min() == 0 and idx.patch_of.max() == idx.k - 1
    assert len(np.unique(idx.patch_of)) == idx.k, "every patch owns at least its seed"
    assert (idx.patch_of[idx.seeds] == np.arange(idx.k)).all()
    assert nx.is_connected(nx.from_scipy_sparse_array(idx.patch_graph))
    assert idx.seed_dist[idx.seeds].max() == 0


@pytest.mark.parametrize("mesh", MESHES, ids=lambda m: m.name)
@pytest.mark.parametrize("rings", [0, 1])
def test_hierarchical_path_is_a_real_path_never_shorter_than_exact(mesh, rings):
    g = mesh.csr(); idx = build_index(g, mesh.points, 12, seed=0); rng = np.random.default_rng(1)
    for _ in range(12):
        s, t = (int(x) for x in rng.integers(0, mesh.n, 2))
        if s == t:
            continue
        approx, size = idx.query(s, t, rings)
        exact = dijkstra(g, s, t)
        assert np.isfinite(approx) and approx >= exact - 1e-9
        assert 0 < size <= mesh.n


def test_a_wider_corridor_is_never_worse():
    mesh = grid_sphere(14, 28); g = mesh.csr(); idx = build_index(g, mesh.points, 16, seed=0); rng = np.random.default_rng(4)
    for _ in range(15):
        s, t = (int(x) for x in rng.integers(0, mesh.n, 2))
        if s != t:
            assert idx.query(s, t, 1)[0] <= idx.query(s, t, 0)[0] + 1e-9


def test_one_patch_per_vertex_is_exact():
    mesh = grid_torus(5, 6); g = mesh.csr(); idx = build_index(g, mesh.points, mesh.n, seed=0)
    for s, t in [(0, 17), (3, 25)]:
        assert idx.query(s, t, 0)[0] == pytest.approx(dijkstra(g, s, t))


def test_graph_distance_on_the_sphere_is_at_least_the_great_circle():
    mesh = delaunay_sphere(400, seed=5); g = mesh.csr(); rng = np.random.default_rng(2)
    q = rng.integers(0, mesh.n, (20, 2)); q = q[q[:, 0] != q[:, 1]]
    exact = np.array([dijkstra(g, int(s), int(t)) for s, t in q])
    geo = sphere_geodesic(mesh.points[q[:, 0]], mesh.points[q[:, 1]])
    assert (exact >= geo - 1e-9).all() and (exact / geo).max() < 1.6        # a mesh path is longer, but not by much


def test_batched_gpu_solver_matches_scipy_on_cpu():
    mesh = bumpy_sphere(9, 18); g = mesh.csr(); sources = [0, 7, 40, 100]
    got = batched_sssp(g, sources, device="cpu")
    for row, s in zip(got, sources):
        assert row == pytest.approx(dijkstra(g, s), rel=1e-4, abs=1e-5)


def test_batched_solver_chunks_give_the_same_answer():
    mesh = grid_torus(6, 8); g = mesh.csr(); sources = list(range(0, 48, 5))
    whole = batched_sssp(g, sources, device="cpu")
    chunked = batched_sssp(g, sources, device="cpu", max_cells=3 * g.nnz)
    assert np.array_equal(whole, chunked)


def test_stretch_summary():
    s = stretch_summary(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 2.0]))
    assert s["max"] == 2.0 and s["median"] == 1.5 and s["mean"] == pytest.approx((1 + 2 + 1.5) / 3)


def test_runner_smoke(tmp_path):
    from manifoldsearch.run import FAMILIES, measure
    cfg = {"queries": 10, "patches_per_sqrt": 1.0, "corridor_rings": 0, "effort": True}
    for fam in FAMILIES:
        row = measure(FAMILIES[fam](8, 0), cfg, 0, "cpu")
        assert row["stretch"]["mean"] >= 1.0 - 1e-9 and 0 < row["corridor_fraction"] <= 1
        assert row["astar_expanded"] <= row["dijkstra_expanded"] + 1
