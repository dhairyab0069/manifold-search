<h1 align="center">Manifold search</h1>

<p align="center">
  Shortest paths over curved surfaces, and a coarse-to-fine search that answers them by looking at a<br>
  few percent of the mesh: how accurate is it, what decides its accuracy, and can machine learning<br>
  make it better?
</p>

<p align="center">
  <a href="https://dhairyab0069.github.io/manifold-io/"><b>Project page (live demo)</b></a>
  &nbsp;·&nbsp; <a href="part1-exploration/">Part 1: the course project</a>
  &nbsp;·&nbsp; <a href="part2-research/">Part 2: independent research</a>
  &nbsp;·&nbsp; <a href="docs/ml-1-adaptive-corridor.md">Direction 1 write-up</a>
</p>

---

## Contents

- [What it is](#what-it-is)
- [Where it is today](#where-it-is-today)
- [The idea in a minute](#the-idea-in-a-minute)
- [What the runs say so far](#what-the-runs-say-so-far)
- [Machine learning directions](#machine-learning-directions)
- [Run it](#run-it)
- [Roadmap](#roadmap)
- [Project layout](#project-layout)

## What it is

A mesh is a surface drawn as a graph: points in 3D, joined by edges whose weight is their length. Finding the
shortest path between two points is a solved problem (Dijkstra, A*), but it gets slow when the surface has
millions of points. This project studies a **two-level search**: first route over a small graph of surface
patches, then run an exact search only inside the patches that route passes through. The answer is always a real
path in the mesh, so it can only be longer than the true shortest path, never shorter. The questions are how much
longer, how little of the mesh it has to look at, and what controls the difference.

It began as a CS 5800 (Algorithms) course project, and it is now my own research in computational geometry.

## Where it is today

| | |
|---|---|
| **Part 1: Exploration** | The course notebook, finished: sphere and torus meshes, Dijkstra and A* baselines, the patch search, benchmarks. [`part1-exploration/`](part1-exploration/) |
| **Part 2: Independent research** | A tested Python package, GPU-computed exact distances for ground truth at scale, experiment configs, Slurm scripts for a GPU cluster, and the first machine learning direction. [`part2-research/`](part2-research/) |

First runs on the cluster and the first learned experiment are in. They are early: most scaling numbers come from
one seed, there is no comparison yet with exact surface-geodesic methods, and nothing has been tried on real scans.

## The idea in a minute

1. **Preprocess once.** Pick `k` seed points spread over the surface (farthest-point sampling, `k` about the square
   root of the number of vertices), give every vertex to its nearest seed (a graph Voronoi partition), and build a
   small graph over the patches.
2. **Answer a query.** Find the route over the patch graph (tiny), then run an ordinary Dijkstra only inside the
   patches on that route plus `rings` rings of neighbouring patches.
3. **Judge it.** Compare with the exact answer. *Stretch* is approximate distance divided by exact distance (1.0 is
   perfect). *Work* is the share of the mesh's vertices that were searched.

## What the runs say so far

Mesh families: a latitude-longitude sphere, a torus, a sphere with a bumpy radius (curvature that changes from
place to place), and random points on a sphere triangulated by their convex hull (irregular spacing). All numbers
are measured by the code in this repository; the raw rows are produced by `part2-research/manifoldsearch/run.py`.

- **Scaling** (2,000 to 2 million vertices, one seed, 100 queries per size): mean stretch stays between about 1.02
  and 1.11 on every family with no sign of getting worse, while the share of the mesh searched falls from about 12%
  to under 2%. Per-query time grows from 0.6 ms to about 18 ms, while the one-off preprocessing grows much faster
  (about two minutes at 2 million vertices).
- **Widening the corridor** is the lever that matters. On meshes the learned model never saw (3 seeds, about 240,000
  queries per seed):

  | Rings | Share of mesh searched | Mean stretch | 95th percentile |
  |---|---|---|---|
  | 0 | 3.6% | 1.056 | 1.172 |
  | 1 | 12.5% | 1.0097 | 1.061 |
  | 2 | 23.3% | 1.0037 | 1.020 |
  | 3 | 35.1% | 1.0018 | 1.001 |

- **The number of patches** barely changes accuracy; it mainly changes how much has to be searched.
- **Time versus an exact search** has not been measured yet, so there is no speed-up claim here.

## Machine learning directions

| # | Direction | Status |
|---|---|---|
| 1 | **Learn the corridor width per query.** Predict how many rings a query needs from cheap features, so easy queries stay narrow and hard ones widen. | Built and evaluated on held-out meshes with 3 seeds. A gradient-boosted-trees policy beats the best fixed mix of widths at equal work, modestly (mean stretch +0.0015, p95 about +0.013 at the best settings); a small neural network did not. The worst case got worse at moderate settings. See [the write-up](docs/ml-1-adaptive-corridor.md). |
| 2 | **Learn the partition.** Place patch borders where a path is least likely to be hurt by crossing them. | First experiments are running on the cluster. |
| 3 | **Learn a heuristic for A\*.** Predict the remaining distance, handling the fact that a heuristic that overestimates breaks exactness. | First experiments are running on the cluster. |

## Run it

```bash
cd part2-research
pip install -r requirements.txt
python -m pytest                                                   # the package's tests
python -m manifoldsearch.run experiments/configs/smoke.yaml --device cpu
```

If `pytest` fails while loading a plugin installed elsewhere on your machine, run it as
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest`.

**On a GPU cluster.** `part2-research/slurm/run_experiment.sbatch` runs any config on one GPU. It has no account line on
purpose: pass `--account=<account>` when you submit, and use an allocation you are allowed to use for the work.
`part2-research/slurm/submit.sh` prints the command and only submits with `--go`. The experiment configs are in
`part2-research/experiments/configs/` (`smoke`, `gpu_smoke`, `scaling`, `scaling_xl`, `patches`, `rings` and the
learned-corridor configs). Raw datasets stay on the cluster; small summaries are committed in
`part2-research/results_summary/`.

## Roadmap

- [x] **Part 1**, the course project: meshes, baselines, the patch search, benchmarks.
- [x] **Part 2 foundations:** a vectorised package, GPU batched exact distances, configs, tests, cluster scripts.
- [x] **First scaling, patch-count and corridor-width runs** (up to 2 million vertices).
- [x] **Direction 1:** learned corridor width, evaluated on unseen meshes.
- [ ] **Direction 2:** learned partition.
- [ ] **Direction 3:** learned A* heuristic.
- [ ] Time against exact Dijkstra, with several seeds at every size.
- [ ] Comparison with exact polyhedral geodesics, the heat method and fast marching.
- [ ] Real scanned meshes, surfaces with holes and boundaries.
- [ ] A bound on stretch in terms of patch size and curvature, checked against the measurements.

## Project layout

| Path | What it is |
|---|---|
| `part1-exploration/` | The CS 5800 notebook and its README |
| `part2-research/manifoldsearch/` | Meshes, baselines, the hierarchical search, the GPU solver, metrics, the experiment runner, and `ml/` for the learned corridor |
| `part2-research/experiments/configs/` | One YAML file per experiment |
| `part2-research/slurm/` | Cluster submission scripts |
| `part2-research/results_summary/` | Small committed result files and figures |
| `part2-research/tests/` | The package's tests |
| `docs/` | One write-up per machine learning direction |

Dhairya Bhatia, Northeastern University.
