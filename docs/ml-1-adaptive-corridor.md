# Direction 1: learning the corridor width per query (adaptive rings)

## Question

The coarse-to-fine search routes over a graph of patches and then runs exact Dijkstra inside the corridor of patches along that route plus `rings` rings of neighbouring patches. Widening the corridor buys accuracy but costs work, and the first results showed that a fixed width is a blunt instrument: most queries are already exact at 0 or 1 ring and a few need 2 or 3. Can a model that sees only cheap, query-time information choose the number of rings per query, and does that beat the obvious alternative of mixing fixed widths at the same average cost?

## Method

*Queries and exact labels:* for every mesh, 2000 random sources with 10 random targets each (20,000 queries; pairs at distance 0 dropped, which only happens at the coincident poles of the grid sphere). One exact single-source Dijkstra (scipy, float64) per source gives the exact distance to its 10 targets. The corridor distance after 0, 1, 2 and 3 rings is computed with the deployed code path (patch route, then Dijkstra in the corridor; `manifoldsearch/ml/features.py: corridor_distance`, verified equal to `HierarchicalIndex.query` in the tests) and timed per query. Corridors are nested, so stretch never increases with rings.

*Labels:* for a tolerance `tol` the label is the smallest ring count in 0..3 whose stretch is at most `tol` (3 if none is). Tolerances 1.01 and 1.02. Label counts on the training meshes (seed 0, 0/1/2/3 rings): tol 1.02: 38,912 / 70,638 / 7,780 / 2,602; tol 1.01: 30,373 / 76,260 / 10,010 / 3,289.

*Features (39, all cheap at query time):* endpoint Euclid distance (raw, over mesh diagonal, over mean patch radius); patch-route length and its ratio to Euclid and to the diagonal; number of patches on the route; min/max/mean patch radius along the route (and ratios to Euclid); curvature along the route (mean/min/max/peak of per-patch mean and max normal-variation curvature); route deviation from the straight segment (max, mean, normalised by Euclid); vertex fraction of the corridor at 1 and 2 rings; mean patch size; endpoint curvature, surface variation and local anisotropy; endpoint border proximity (distance to the nearest patch border over the patch radius) and distance to own seed; same-patch flag; mesh scale features (log N, log k, mean edge over diagonal, mean mesh curvature). Curvature comes from a PCA of each vertex's 1-ring (`manifoldsearch/ml/geometry.py`), because meshes carry no faces here; angle defect was not used for that reason. Everything except the per-query route is precomputed once per mesh; the measured feature time is 0.21 ms per query.

A second feature set, `base+ring0`, adds three features from the ring-0 corridor search (ring-0 distance over Euclid and over the patch-route length, ring-0 corridor fraction). That search is the cheapest option and is needed by every policy anyway, but a policy that then picks r > 0 must pay for ring 0 and ring r, and the cost columns charge for that (no reuse).

*Models:* LightGBM multiclass (200 trees, 15 leaves, learning rate 0.05, min 200 samples per leaf, 0.8 subsampling) and a small MLP (2 hidden layers of 64, Adam, 30 epochs, standardised inputs). Both predict the class distribution; a policy picks the smallest r with P(label <= r) >= tau, so tau trades work for safety and each model gives a whole curve. tau in {0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99}. Hyperparameters were set once by hand and not tuned on test meshes.

*Split by mesh (never by query):* train on grid sphere and grid torus at sizes 32, 64, 128 (6 meshes, about 120,000 queries; N from 2k to 33k). Test on 12 meshes the model never saw: bumpy sphere and random-point (Delaunay) sphere at sizes 64, 128, 256, 512 (unseen families), and grid sphere and torus at 256 and 512 (unseen sizes; N up to 524k). A size-n mesh has about 2n^2 vertices (so 256 is N = 131k). Patch count k = sqrt(N). A test in `tests/test_ml.py` checks the split.

*Seeds:* 3 (0, 1, 2). A seed changes the farthest-point start (so the partition), the queries, the random-point meshes, and the model initialisation. The grid and bumpy meshes themselves are deterministic. Numbers are means over seeds with the seed standard deviation in brackets; each seed's models are trained only on that seed's training meshes.

*Evaluation:* work is the fraction of the mesh's vertices inside the searched corridor; time is the measured per-query time (route + subgraph + Dijkstra; for learned policies plus feature time plus single-row model latency, about 0.1 to 0.3 ms). Stretch is corridor distance over exact distance; mean, p95 and max over the pooled held-out queries. Compared against: fixed rings 0 to 3; the oracle (the minimal sufficient ring per query, which is the label, so it needs exact answers and is a bound, not a method); and the best mix of two adjacent fixed rings (a random fraction p of queries to ring r, the rest to r+1), interpolated at the learned policy's own work. "Gain vs mix" below is the mix's stretch minus the learned policy's stretch at equal work (positive: learned is better), with the number of seeds out of 3 in which it was positive.

## Results (held-out meshes, 3 seeds)

Figures: `part2-research/results_summary/ml1_pareto_tol1.02.png` and `ml1_pareto_tol1.01.png` (work and time against mean and p95 stretch; the grey fixed-ring line joins the four fixed points and is exactly the mix curve only for the mean). Data: `ml1_adaptive_corridor.json`, `ml1_adaptive_corridor_pooled.csv`.


**all_test** (239983 queries per seed)

| policy | work | ms | mean (sd) | p95 (sd) | max | mean gain vs mix | p95 gain vs mix |
|---|---|---|---|---|---|---|---|
| fixed_0 | 0.036 | 2.7 | 1.0561 (0.0009) | 1.1723 (0.0028) | 2.92 | - | - |
| fixed_1 | 0.125 | 7.8 | 1.0097 (0.0001) | 1.0610 (0.0012) | 1.60 | - | - |
| fixed_2 | 0.233 | 14.8 | 1.0037 (0.0001) | 1.0203 (0.0007) | 1.44 | - | - |
| fixed_3 | 0.351 | 23.5 | 1.0018 (0.0001) | 1.0007 (0.0002) | 1.35 | - | - |
| oracle_1.02 | 0.117 | 7.6 | 1.0056 (0.0001) | 1.0185 (0.0000) | 1.35 | - | - |
| oracle_1.01 | 0.134 | 9.0 | 1.0031 (0.0001) | 1.0093 (0.0000) | 1.35 | - | - |
| gbt_base_tol1.02_tau0.5 | 0.124 | 9.3 | 1.0154 (0.0014) | 1.0826 (0.0061) | 2.69 | -0.0045 (0/3) | -0.0138 (0/3) |
| gbt_base_tol1.02_tau0.7 | 0.144 | 10.8 | 1.0095 (0.0007) | 1.0563 (0.0034) | 2.59 | -0.0008 (0/3) | -0.0005 (2/3) |
| gbt_base_tol1.02_tau0.8 | 0.154 | 11.6 | 1.0074 (0.0003) | 1.0453 (0.0021) | 2.58 | +0.0006 (3/3) | +0.0073 (3/3) |
| gbt_base_tol1.02_tau0.9 | 0.171 | 12.8 | 1.0056 (0.0003) | 1.0342 (0.0020) | 2.04 | +0.0015 (3/3) | +0.0130 (3/3) |
| gbt_base_tol1.02_tau0.95 | 0.187 | 14.2 | 1.0045 (0.0004) | 1.0264 (0.0031) | 1.72 | +0.0017 (3/3) | +0.0148 (3/3) |
| gbt_base_tol1.02_tau0.98 | 0.211 | 16.1 | 1.0034 (0.0004) | 1.0172 (0.0038) | 1.50 | +0.0016 (3/3) | +0.0139 (3/3) |
| gbt_base_tol1.02_tau0.99 | 0.230 | 17.6 | 1.0028 (0.0003) | 1.0114 (0.0034) | 1.50 | +0.0012 (3/3) | +0.0112 (3/3) |
| gbt_base_tol1.01_tau0.7 | 0.153 | 11.5 | 1.0073 (0.0003) | 1.0447 (0.0015) | 2.59 | +0.0008 (3/3) | +0.0083 (3/3) |
| gbt_base_tol1.01_tau0.9 | 0.184 | 13.8 | 1.0046 (0.0003) | 1.0264 (0.0027) | 1.95 | +0.0018 (3/3) | +0.0159 (3/3) |
| gbt_base_tol1.01_tau0.98 | 0.229 | 17.5 | 1.0028 (0.0004) | 1.0108 (0.0039) | 1.50 | +0.0012 (3/3) | +0.0123 (3/3) |
| gbt_base+ring0_tol1.01_tau0.7 | 0.193 | 14.0 | 1.0052 (0.0006) | 1.0307 (0.0040) | 1.50 | +0.0007 (3/3) | +0.0081 (3/3) |
| gbt_base+ring0_tol1.01_tau0.9 | 0.218 | 15.8 | 1.0033 (0.0005) | 1.0171 (0.0037) | 1.50 | +0.0013 (3/3) | +0.0106 (3/3) |
| gbt_base+ring0_tol1.01_tau0.98 | 0.250 | 18.5 | 1.0023 (0.0003) | 1.0077 (0.0027) | 1.40 | +0.0012 (3/3) | +0.0097 (3/3) |
| gbt_base+ring0_tol1.02_tau0.7 | 0.183 | 13.1 | 1.0065 (0.0005) | 1.0382 (0.0031) | 1.54 | -0.0000 (1/3) | +0.0043 (3/3) |
| gbt_base+ring0_tol1.02_tau0.9 | 0.208 | 14.9 | 1.0041 (0.0005) | 1.0235 (0.0037) | 1.50 | +0.0010 (3/3) | +0.0090 (3/3) |
| gbt_base+ring0_tol1.02_tau0.98 | 0.238 | 17.2 | 1.0027 (0.0003) | 1.0117 (0.0029) | 1.40 | +0.0011 (3/3) | +0.0086 (3/3) |
| mlp_base_tol1.02_tau0.7 | 0.108 | 7.6 | 1.0185 (0.0060) | 1.0967 (0.0221) | 2.01 | -0.0000 (2/3) | +0.0034 (2/3) |
| mlp_base_tol1.02_tau0.9 | 0.125 | 8.6 | 1.0130 (0.0038) | 1.0758 (0.0152) | 1.91 | -0.0010 (1/3) | -0.0034 (2/3) |
| mlp_base_tol1.02_tau0.98 | 0.145 | 10.0 | 1.0092 (0.0023) | 1.0576 (0.0115) | 1.89 | -0.0006 (1/3) | -0.0023 (1/3) |
| mlp_base_tol1.01_tau0.7 | 0.118 | 8.9 | 1.0163 (0.0033) | 1.0916 (0.0088) | 2.59 | -0.0018 (0/3) | -0.0066 (1/3) |
| mlp_base_tol1.01_tau0.9 | 0.135 | 10.1 | 1.0115 (0.0025) | 1.0699 (0.0118) | 2.13 | -0.0024 (0/3) | -0.0118 (0/3) |
| mlp_base_tol1.01_tau0.98 | 0.155 | 11.4 | 1.0083 (0.0015) | 1.0518 (0.0096) | 1.95 | -0.0002 (1/3) | +0.0005 (1/3) |
| mlp_base+ring0_tol1.01_tau0.9 | 0.177 | 12.0 | 1.0083 (0.0014) | 1.0502 (0.0062) | 1.53 | -0.0015 (0/3) | -0.0051 (1/3) |

**bumpy_sphere** (79986 queries per seed)

| policy | work | ms | mean (sd) | p95 (sd) | max | mean gain vs mix | p95 gain vs mix |
|---|---|---|---|---|---|---|---|
| fixed_0 | 0.046 | 2.1 | 1.0754 (0.0021) | 1.2247 (0.0057) | 2.92 | - | - |
| fixed_1 | 0.155 | 5.5 | 1.0166 (0.0002) | 1.1088 (0.0013) | 1.60 | - | - |
| fixed_2 | 0.281 | 9.9 | 1.0071 (0.0002) | 1.0563 (0.0010) | 1.43 | - | - |
| fixed_3 | 0.412 | 15.0 | 1.0032 (0.0002) | 1.0133 (0.0016) | 1.35 | - | - |
| oracle_1.02 | 0.167 | 6.9 | 1.0056 (0.0002) | 1.0194 (0.0001) | 1.35 | - | - |
| oracle_1.01 | 0.180 | 7.4 | 1.0040 (0.0002) | 1.0133 (0.0016) | 1.35 | - | - |
| gbt_base_tol1.02_tau0.9 | 0.202 | 8.7 | 1.0115 (0.0002) | 1.0814 (0.0009) | 2.04 | +0.0016 (3/3) | +0.0117 (3/3) |
| gbt_base_tol1.02_tau0.98 | 0.247 | 10.7 | 1.0069 (0.0008) | 1.0511 (0.0061) | 1.50 | +0.0028 (3/3) | +0.0233 (3/3) |
| gbt_base_tol1.01_tau0.9 | 0.215 | 9.0 | 1.0099 (0.0005) | 1.0724 (0.0031) | 1.95 | +0.0022 (3/3) | +0.0160 (3/3) |
| gbt_base+ring0_tol1.01_tau0.98 | 0.322 | 13.8 | 1.0040 (0.0005) | 1.0240 (0.0063) | 1.40 | +0.0019 (3/3) | +0.0210 (3/3) |
| mlp_base_tol1.01_tau0.9 | 0.149 | 7.4 | 1.0245 (0.0038) | 1.1305 (0.0093) | 2.13 | -0.0044 (0/3) | -0.0097 (0/3) |

**delaunay_sphere** (79997 queries per seed)

| policy | work | ms | mean (sd) | p95 (sd) | max | mean gain vs mix | p95 gain vs mix |
|---|---|---|---|---|---|---|---|
| fixed_0 | 0.035 | 2.7 | 1.0234 (0.0003) | 1.0758 (0.0014) | 1.55 | - | - |
| fixed_1 | 0.127 | 8.5 | 1.0012 (0.0000) | 1.0068 (0.0001) | 1.19 | - | - |
| fixed_2 | 0.246 | 17.4 | 1.0002 (0.0000) | 1.0000 (0.0000) | 1.15 | - | - |
| fixed_3 | 0.378 | 29.0 | 1.0001 (0.0000) | 1.0000 (0.0000) | 1.12 | - | - |
| oracle_1.02 | 0.079 | 4.9 | 1.0053 (0.0000) | 1.0176 (0.0000) | 1.12 | - | - |
| oracle_1.01 | 0.103 | 7.1 | 1.0018 (0.0000) | 1.0085 (0.0000) | 1.12 | - | - |
| gbt_base_tol1.02_tau0.9 | 0.152 | 12.7 | 1.0006 (0.0001) | 1.0032 (0.0008) | 1.34 | +0.0004 (3/3) | +0.0023 (3/3) |
| gbt_base_tol1.02_tau0.98 | 0.191 | 16.7 | 1.0002 (0.0000) | 1.0002 (0.0003) | 1.12 | +0.0005 (3/3) | +0.0028 (3/3) |
| gbt_base_tol1.01_tau0.9 | 0.160 | 13.0 | 1.0004 (0.0001) | 1.0023 (0.0010) | 1.33 | +0.0005 (3/3) | +0.0028 (3/3) |
| gbt_base+ring0_tol1.01_tau0.98 | 0.200 | 16.7 | 1.0004 (0.0001) | 1.0018 (0.0013) | 1.12 | +0.0002 (3/3) | +0.0005 (3/3) |
| mlp_base_tol1.01_tau0.9 | 0.128 | 9.4 | 1.0019 (0.0013) | 1.0115 (0.0083) | 1.37 | -0.0004 (0/3) | -0.0026 (0/3) |

**grid_larger** (80000 queries per seed)

| policy | work | ms | mean (sd) | p95 (sd) | max | mean gain vs mix | p95 gain vs mix |
|---|---|---|---|---|---|---|---|
| fixed_0 | 0.028 | 3.4 | 1.0695 (0.0020) | 1.1668 (0.0048) | 1.73 | - | - |
| fixed_1 | 0.093 | 9.4 | 1.0112 (0.0005) | 1.0583 (0.0021) | 1.36 | - | - |
| fixed_2 | 0.171 | 17.2 | 1.0039 (0.0001) | 1.0256 (0.0013) | 1.36 | - | - |
| fixed_3 | 0.262 | 26.4 | 1.0022 (0.0001) | 1.0089 (0.0011) | 1.27 | - | - |
| oracle_1.02 | 0.106 | 11.1 | 1.0058 (0.0001) | 1.0190 (0.0001) | 1.27 | - | - |
| oracle_1.01 | 0.119 | 12.5 | 1.0035 (0.0001) | 1.0099 (0.0001) | 1.27 | - | - |
| gbt_base_tol1.02_tau0.9 | 0.157 | 17.2 | 1.0049 (0.0006) | 1.0295 (0.0032) | 1.58 | +0.0003 (3/3) | +0.0048 (3/3) |
| gbt_base_tol1.02_tau0.98 | 0.193 | 20.8 | 1.0031 (0.0004) | 1.0186 (0.0030) | 1.27 | +0.0004 (3/3) | +0.0039 (3/3) |
| gbt_base_tol1.01_tau0.9 | 0.176 | 19.2 | 1.0036 (0.0004) | 1.0218 (0.0031) | 1.52 | +0.0003 (3/3) | +0.0031 (3/3) |
| gbt_base+ring0_tol1.01_tau0.98 | 0.227 | 24.9 | 1.0025 (0.0003) | 1.0128 (0.0025) | 1.27 | +0.0004 (3/3) | +0.0037 (3/3) |
| mlp_base_tol1.01_tau0.9 | 0.128 | 13.5 | 1.0081 (0.0029) | 1.0447 (0.0129) | 1.49 | +0.0000 (2/3) | +0.0015 (2/3) |

["bumpy_sphere_128x256", "bumpy_sphere_256x512", "bumpy_sphere_512x1024", "bumpy_sphere_64x128", "delaunay_sphere_32768", "delaunay_sphere_131072", "delaunay_sphere_524288", "delaunay_sphere_8192", "grid_sphere_256x512", "grid_sphere_512x1024", "grid_torus_256x512", "grid_torus_512x1024"]


## What the data says

- *Fixed rings, held-out:* 0 / 1 / 2 / 3 rings give mean stretch 1.056 / 1.0097 / 1.0037 / 1.0018 at 3.6% / 12.5% / 23.3% / 35.1% of the vertices. One extra ring is the big step, as in the first results.
- *The oracle is far ahead:* the minimal sufficient ring per query reaches mean 1.0056 and p95 1.0185 (tolerance 1.02) at 11.7% work, better than fixed ring 1 in both stretch and p95 at slightly less work. No learned policy comes close to that.
- *Learned gradient-boosted trees beat the best fixed-ring mix at equal work, modestly:* with the base features, tolerance 1.02 and tau from 0.8 up, the gain in mean stretch is positive in 3/3 seeds (best +0.0017 at tau 0.95, about a quarter of the mix's excess stretch of 0.0062 at that work) and the gain in p95 is +0.007 to +0.015 (for example p95 1.0342 against about 1.047 for the mix at 17.1% work). At low tau (0.5, 0.7) the policy loses to the mix: it sends too many hard queries to a narrow corridor.
- *Gains are not uniform:* they are clearest on the bumpy sphere (mean +0.0016 to +0.0028, p95 +0.012 to +0.023 at tau 0.9 to 0.98, 3/3 seeds), small on the grid sphere and torus at the unseen larger sizes (mean about +0.0003, p95 about +0.004) and on the random-point sphere (mean about +0.0004 to +0.0005, p95 +0.002 to +0.003; almost everything there is already exact at ring 1), and the advantage does not transfer to wall-clock time: at equal measured time the mean gain is about zero and the sign of the p95 gain flips between seeds and groups (see the `adv_time_*` fields in the JSON). Query time here is dominated by the corridor Dijkstra and grows with mesh size, so time tracks work only roughly, and the time columns come from 16 worker processes sharing a node, which makes them noisy.
- *Worst case is worse:* at moderate tau the learned policy's max stretch (2.0 to 2.6 pooled) is higher than that of fixed ring 1 (1.60), because a few hard queries are sent to ring 0. Max stretch comes down to fixed-ring levels only at tau 0.95 and above (1.72 at 0.95, 1.50 at 0.98), where the work is already 19% to 21%, close to fixed ring 2. Max is a single query out of 240,000, so treat it as indicative.
- *Features:* the ring-0 features (`base+ring0`) raise argmax classification accuracy by about 0.05 (pooled, seed 1: 0.57 to 0.63) and lower the tail (max 1.50 against 2.04 at tau 0.9), but the extra cost of searching ring 0 first cancels it: the gain over the fixed-ring mix at equal work is not larger than for the base features (tolerance 1.02, tau 0.9: mean +0.0010 and p95 +0.0090, against +0.0015 and +0.0130), and on the grid meshes and the random-point sphere it is sometimes negative. The most important base features by gain are the patch-route length over the mesh diagonal and the route's deviation from the straight line; with ring 0 added, ring-0 distance over Euclid dominates (see `gbt_importance_seed0`).
- *MLP:* the small MLP does not beat the mix at equal work (gain negative or inconclusive across seeds, and with larger seed-to-seed spread, e.g. p95 sd 0.01 to 0.02 against 0.002 to 0.004 for the trees). At a given tau it searches less than the trees but ends at higher stretch, and on the equal-work comparison that is no better than the mix.
- *Tolerance 1.01 against 1.02:* tolerance 1.01 gives similar gains at somewhat higher work; both are in the tables.

Bottom line: the learned policy is a small, real but not dramatic improvement over mixing fixed widths at the same work (mean and p95, three seeds, gradient-boosted trees, work axis), it does not hold up in measured time, and it leaves most of the oracle's potential on the table. Argmax accuracy of the label prediction is only about 0.45 to 0.69 depending on model, feature set and group, which is why the policies need a high tau to avoid under-searching.

## Caveats

- One partition per mesh and seed (k = sqrt(N)); behaviour at other patch counts was not tested.
- Queries within a mesh share sources (10 targets each), so they are correlated; seeds are the only independent replicates, hence only 3 for the error bars.
- The pooled "all test" numbers weight every mesh equally (20,000 queries each), so the larger meshes dominate absolute time but not work.
- Time on the cluster comes from concurrent processes and different nodes (GPU nodes were used only for the MLP; everything else is CPU), and seed 0 and 2 generation shared a node. Compare time curves within this table, not against other machines.
- The learned policy needs a calibrated threshold; tau was not tuned per family, and picking tau to hit a work budget requires knowing the held-out work, which a deployment would estimate on validation meshes.
- Max stretch is a single worst query.

## Reproduce

```
# on the cluster (env as in slurm/run_ml.sbatch), per seed s
python -m manifoldsearch.ml.generate experiments/configs/ml_adaptive.yaml --seed s --out results/ml/data
python -m manifoldsearch.ml.run_eval experiments/configs/ml_adaptive.yaml --data results/ml/data --seed s --out results/ml/eval_seed$s.json
python -m manifoldsearch.ml.report results/ml/eval_seed*.json --out results_summary --prefix ml1
```

Generation took about 17 minutes per seed with 16 CPU workers; evaluation about 3 minutes per seed. Raw datasets (about 100 MB per seed) stay on the cluster in `part2-research/results/ml/`.
