# Part 1: Exploration (CS 5800, Algorithms)

The course project, kept as it was handed in: `cs5800_project_manifold-search.ipynb`.

**The question.** Finding a shortest path on a graph is solved (Dijkstra, A*). But what if the graph is a surface
(a sphere, a torus) with hundreds of thousands of points? Can a *two-level* search, which first routes over a small
graph of surface patches and only then searches inside the patches on that route, answer queries faster, and how much
longer are its paths?

**What the notebook does.**
- builds sphere and torus grids as graphs (`coords` for 3D positions, `adj` for neighbours);
- implements BFS, Dijkstra (the exact baseline) and A* with a straight-line heuristic;
- builds the partition (patches grown from seeds by simultaneous BFS) and a patch graph, then the hierarchical search;
- benchmarks speed against flat Dijkstra on both surfaces, measures error against the true great-circle distance on the
  sphere, and the path overhead against exact Dijkstra on both;
- extra experiments: an A* comparison, correctness checks, a patch-count sweep, preprocessing cost, query-count stability;
- writes LaTeX tables for the report.

Run it in Jupyter (`numpy` and `matplotlib` only). It writes `images/` and `tables/` next to itself (git-ignored).

This part is finished. [Part 2](../part2-research/) takes the same idea much further.
