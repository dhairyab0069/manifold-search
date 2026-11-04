# Part 2: Independent research (computational geometry)

Part 1 asked one question on two tidy surfaces. This part treats it as a research problem in computational geometry:
**how well does coarse-to-fine search approximate geodesic distance on a surface mesh, and what decides its accuracy?**

**Status: set up, not yet run at scale.** The code, tests and experiment configs below are done and tested on a laptop.
No cluster run has been made yet. Nothing here is a result yet, only a plan and the tools to get one.

## Questions

1. **Scaling.** How do speed and stretch (approximate / exact distance, never below 1) change from thousands to millions of vertices?
2. **Beyond tidy grids.** Does it hold on irregular meshes (random points on a sphere, uneven spacing), on a surface whose curvature
   changes from place to place (a bumpy sphere), and on real scanned meshes (`load_obj`)? Genus above 0 and surfaces with boundaries come next.
3. **What controls the error?** Patch count k, patch diameter, corridor width, and curvature. Is there a bound on stretch in terms
   of patch size, and does it match what is measured?
4. **Against the established methods.** Exact polyhedral geodesics, the heat method, fast marching, and landmark A* (ALT).
5. **Better partitions.** Farthest-point seeds (the baseline here) against spectral and learned partitions, and a learned A* heuristic.

## What is here

| Path | What it is |
|---|---|
| `manifoldsearch/meshes.py` | surfaces as graphs: grid sphere and torus (same as Part 1), bumpy sphere, random-point Delaunay sphere, `.obj` loader |
| `manifoldsearch/baselines.py` | exact Dijkstra (scipy), and counting Dijkstra and A* for search effort |
| `manifoldsearch/hierarchical.py` | the coarse-to-fine search, rebuilt with arrays: seeds, graph-Voronoi patches, patch graph, corridor search |
| `manifoldsearch/gpu.py` | exact distances for many sources at once on a GPU (batched Bellman-Ford), for ground truth at scale |
| `manifoldsearch/metrics.py` | stretch summary, sphere great-circle distance |
| `manifoldsearch/run.py` | runs a YAML config, one JSON line per measurement |
| `experiments/configs/` | `smoke` (seconds, CPU), `scaling`, `patches`, `rings` |
| `slurm/` | submit scripts for the cluster's GPU partition |
| `tests/` | 30 tests |

## Run it

```bash
cd part2-research
pip install -r requirements.txt
python -m pytest                                                      # 30 tests, a few seconds
python -m manifoldsearch.run experiments/configs/smoke.yaml --device cpu
```

If `pytest` fails while loading a plugin installed elsewhere on your machine, run it as
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest`.

## Running on the GPU cluster

`slurm/submit.sh` prints the `sbatch` command and only submits with `--go`. It needs `ACCOUNT=<slurm account>` and has no default,
on purpose: use an allocation you are allowed to use for this work. Start small (sizes up to 256), look at the logs, then scale up;
the largest sizes in `scaling.yaml` (about 8 million vertices) have not been tried.

## Honest limits

- The grid sphere has a row of coincident vertices at each pole (as in Part 1); queries between them have distance 0 and are skipped.
- The hierarchical answer is a real path in the mesh, so it is never shorter than the exact one; it can be longer.
- Distances here are graph distances along mesh edges, which overestimate the true surface geodesic (by up to about 1.6x on these
  meshes in the tests). The exact-polyhedral and heat-method comparisons (question 4) are what pin down the true geodesic.
